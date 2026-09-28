//! A small decoder-only transformer baseline for apples-to-apples PSSA studies.
//!
//! The baseline deliberately uses the repository's own parameter containers,
//! RMSNorm convention, AdamW implementation, RNG, tokenizer, and dataset
//! windowing.  It is a one-layer causal self-attention model with fixed
//! sinusoidal positions; at a 2,048-token vocabulary its trainable parameter
//! count is 1,541,120, close to the default PSSA's 1,544,704.

use crate::dataset::Tokenizer;
use crate::linalg::{SimpleRng, dot_slice, rms_norm_slice, sigmoid};
use crate::pssa::{ParamMatrix, ParamVector};
const MAX_ALLOCATION_BYTES: usize = 1024 * 1024 * 1024;
const MAX_MODEL_DIM: usize = 4096;
const MAX_CHUNK_LEN: usize = 65_536;

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub d_vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub chunk_len: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub eps: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_vocab: 2048,
            d_model: 256,
            n_heads: 4,
            d_ff: 448,
            chunk_len: 64,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
        }
    }
}

impl TransformerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.d_vocab < 2
            || self.d_model == 0
            || self.n_heads == 0
            || self.d_ff == 0
            || self.chunk_len == 0
        {
            return Err("transformer dimensions and chunk length must be positive; vocabulary must contain at least two tokens".into());
        }
        if self.d_model % self.n_heads != 0 {
            return Err("transformer model width must be divisible by the number of heads".into());
        }
        if self.d_model > MAX_MODEL_DIM || self.d_ff > MAX_MODEL_DIM * 4 {
            return Err(format!(
                "transformer dimensions exceed the safe limit ({MAX_MODEL_DIM})"
            ));
        }
        if self.chunk_len > MAX_CHUNK_LEN {
            return Err(format!(
                "transformer chunk length must be at most {MAX_CHUNK_LEN}"
            ));
        }
        if !(self.lr.is_finite() && self.lr > 0.0)
            || !(self.beta1.is_finite() && (0.0..1.0).contains(&self.beta1))
            || !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2))
            || !(self.weight_decay.is_finite() && self.weight_decay >= 0.0)
            || !(self.eps.is_finite() && self.eps > 0.0)
        {
            return Err("transformer optimizer values must be finite and valid".into());
        }
        if allocation_bytes(self)? > MAX_ALLOCATION_BYTES {
            return Err(
                "transformer exceeds the 1 GiB allocation cap; reduce dimensions or --chunk".into(),
            );
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("overflow calculating {what}"))
}
fn checked_add(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("overflow calculating {what}"))
}

pub(crate) fn allocation_bytes(cfg: &TransformerConfig) -> Result<usize, String> {
    let l2 = checked_mul(cfg.chunk_len, cfg.chunk_len, "attention tape")?;
    let qkv = checked_mul(
        cfg.chunk_len,
        checked_mul(3, cfg.d_model, "qkv")?,
        "qkv tape",
    )?;
    let model = [
        checked_mul(cfg.d_vocab, cfg.d_model, "embedding")?,
        checked_mul(3 * cfg.d_model, cfg.d_model, "qkv weights")?,
        checked_mul(cfg.d_model, cfg.d_model, "attention output")?,
        checked_mul(cfg.d_ff, cfg.d_model, "feed-forward input")?,
        checked_mul(cfg.d_model, cfg.d_ff, "feed-forward output")?,
        checked_mul(cfg.d_vocab, cfg.d_model, "unembedding")?,
        checked_mul(cfg.d_model, 4, "norm parameters")?,
    ]
    .into_iter()
    .try_fold(0usize, |sum, x| {
        checked_add(sum, checked_mul(x, 16, "model bytes")?, "model bytes")
    })?;
    let tape_floats = [
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // x
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // norm1
        qkv,
        checked_mul(l2, cfg.n_heads, "attention probabilities")?,
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // context
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // attention output
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // residual1
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // norm2
        checked_mul(cfg.chunk_len, cfg.d_ff, "tape")?,    // ff pre
        checked_mul(cfg.chunk_len, cfg.d_ff, "tape")?,    // ff act
        checked_mul(cfg.chunk_len, cfg.d_model, "tape")?, // residual2
        checked_mul(cfg.chunk_len, cfg.d_vocab, "logits")?,
        checked_mul(cfg.chunk_len, cfg.d_vocab, "probs")?,
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g residual2
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g norm2
        checked_mul(cfg.chunk_len, cfg.d_ff, "backward")?,    // g ff
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g residual1
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g attn out
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g context
        qkv,                                                  // g qkv
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g norm1
        checked_mul(cfg.chunk_len, cfg.d_model, "backward")?, // g x
    ]
    .into_iter()
    .try_fold(0usize, |sum, x| {
        checked_add(sum, checked_mul(x, 4, "tape bytes")?, "tape bytes")
    })?;
    // Token/target IDs, RMS factors, per-token losses, and backward scratch.
    let ids = checked_mul(
        cfg.chunk_len,
        2 * std::mem::size_of::<usize>() + 12,
        "token tape",
    )?;
    let scratch = checked_mul(3 * cfg.d_model + cfg.chunk_len, 4, "backward scratch")?;
    checked_add(
        checked_add(model, tape_floats, "allocation")?,
        checked_add(ids, scratch, "scratch")?,
        "allocation",
    )
}

#[derive(Clone, Debug)]
struct TransformerTape {
    max_l: usize,
    len: usize,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    x: Vec<f32>,
    norm1: Vec<f32>,
    inv_rms1: Vec<f32>,
    qkv: Vec<f32>,
    attn_probs: Vec<f32>,
    context: Vec<f32>,
    attn_out: Vec<f32>,
    residual1: Vec<f32>,
    norm2: Vec<f32>,
    inv_rms2: Vec<f32>,
    ff_pre: Vec<f32>,
    ff_act: Vec<f32>,
    residual2: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    losses: Vec<f32>,
    g_residual2: Vec<f32>,
    g_norm2: Vec<f32>,
    g_ff_pre: Vec<f32>,
    g_residual1: Vec<f32>,
    g_attn_out: Vec<f32>,
    g_context: Vec<f32>,
    g_qkv: Vec<f32>,
    g_norm1: Vec<f32>,
    g_x: Vec<f32>,
}

impl TransformerTape {
    fn new(cfg: &TransformerConfig) -> Self {
        let l = cfg.chunk_len;
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let v = cfg.d_vocab;
        let heads = cfg.n_heads;
        Self {
            max_l: l,
            len: 0,
            input_ids: vec![0; l],
            target_ids: vec![0; l],
            x: vec![0.0; l * d],
            norm1: vec![0.0; l * d],
            inv_rms1: vec![0.0; l],
            qkv: vec![0.0; l * 3 * d],
            attn_probs: vec![0.0; l * l * heads],
            context: vec![0.0; l * d],
            attn_out: vec![0.0; l * d],
            residual1: vec![0.0; l * d],
            norm2: vec![0.0; l * d],
            inv_rms2: vec![0.0; l],
            ff_pre: vec![0.0; l * ff],
            ff_act: vec![0.0; l * ff],
            residual2: vec![0.0; l * d],
            logits: vec![0.0; l * v],
            probs: vec![0.0; l * v],
            losses: vec![0.0; l],
            g_residual2: vec![0.0; l * d],
            g_norm2: vec![0.0; l * d],
            g_ff_pre: vec![0.0; l * ff],
            g_residual1: vec![0.0; l * d],
            g_attn_out: vec![0.0; l * d],
            g_context: vec![0.0; l * d],
            g_qkv: vec![0.0; l * 3 * d],
            g_norm1: vec![0.0; l * d],
            g_x: vec![0.0; l * d],
        }
    }

    fn clear_backward(&mut self) {
        self.g_residual2.fill(0.0);
        self.g_norm2.fill(0.0);
        self.g_ff_pre.fill(0.0);
        self.g_residual1.fill(0.0);
        self.g_attn_out.fill(0.0);
        self.g_context.fill(0.0);
        self.g_qkv.fill(0.0);
        self.g_norm1.fill(0.0);
        self.g_x.fill(0.0);
    }
}

/// The trainable decoder-only baseline.
pub struct TransformerModel {
    pub cfg: TransformerConfig,
    pub step_counter: usize,
    pub rng: SimpleRng,
    pub vocabulary: Vec<String>,
    pub tokenizer_json: Option<String>,
    pub lr_schedule_total_updates: Option<usize>,
    pub token_embed: ParamMatrix,
    pub qkv: ParamMatrix,
    pub out_proj: ParamMatrix,
    pub ff1: ParamMatrix,
    pub ff2: ParamMatrix,
    pub unembed: ParamMatrix,
    pub norm1_gamma: ParamVector,
    pub norm1_beta: ParamVector,
    pub norm2_gamma: ParamVector,
    pub norm2_beta: ParamVector,
    tape: Option<TransformerTape>,
}

impl TransformerModel {
    pub fn new(cfg: TransformerConfig, seed: u64) -> Result<Self, String> {
        cfg.validate()?;
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let qkv = ParamMatrix::random_xavier(3 * d, d, &mut rng);
        Ok(Self {
            token_embed: ParamMatrix::random_xavier(cfg.d_vocab, d, &mut rng),
            qkv,
            out_proj: ParamMatrix::random_xavier(d, d, &mut rng),
            ff1: ParamMatrix::random_xavier(cfg.d_ff, d, &mut rng),
            ff2: ParamMatrix::random_xavier(d, cfg.d_ff, &mut rng),
            unembed: ParamMatrix::random_xavier(cfg.d_vocab, d, &mut rng),
            norm1_gamma: ParamVector::new(d, 1.0),
            norm1_beta: ParamVector::new(d, 0.0),
            norm2_gamma: ParamVector::new(d, 1.0),
            norm2_beta: ParamVector::new(d, 0.0),
            cfg: cfg.clone(),
            step_counter: 0,
            rng,
            vocabulary: Vec::new(),
            tokenizer_json: None,
            lr_schedule_total_updates: None,
            tape: Some(TransformerTape::new(&cfg)),
        })
    }

    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        let tok = match &self.tokenizer_json {
            Some(json) => Tokenizer::from_serialized(json)?,
            None => Tokenizer::from_vocabulary(&self.vocabulary)?,
        };
        if tok.vocab_size != self.cfg.d_vocab || tok.ordered_vocabulary()? != self.vocabulary {
            return Err("transformer checkpoint tokenizer/vocabulary mismatch".into());
        }
        Ok(tok)
    }

    /// Valid after forward_train_chunk, including a short final chunk.
    pub fn training_logits(&self) -> &[f32] {
        let tape = self.tape.as_ref().expect("training tape");
        &tape.logits[..tape.len * self.cfg.d_vocab]
    }

    pub fn parameter_count(&self) -> usize {
        [
            &self.token_embed,
            &self.qkv,
            &self.out_proj,
            &self.ff1,
            &self.ff2,
            &self.unembed,
        ]
        .into_iter()
        .map(|p| p.data.len())
        .sum::<usize>()
            + self.norm1_gamma.data.len()
            + self.norm1_beta.data.len()
            + self.norm2_gamma.data.len()
            + self.norm2_beta.data.len()
    }

    pub fn zero_gradients(&mut self) {
        for p in [
            &mut self.token_embed,
            &mut self.qkv,
            &mut self.out_proj,
            &mut self.ff1,
            &mut self.ff2,
            &mut self.unembed,
        ] {
            p.zero_grad();
        }
        for p in [
            &mut self.norm1_gamma,
            &mut self.norm1_beta,
            &mut self.norm2_gamma,
            &mut self.norm2_beta,
        ] {
            p.zero_grad();
        }
    }

    pub fn all_finite(&self) -> bool {
        let matrices = [
            &self.token_embed,
            &self.qkv,
            &self.out_proj,
            &self.ff1,
            &self.ff2,
            &self.unembed,
        ];
        matrices.iter().all(|p| {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|x| x.is_finite())
        }) && [
            &self.norm1_gamma,
            &self.norm1_beta,
            &self.norm2_gamma,
            &self.norm2_beta,
        ]
        .iter()
        .all(|p| {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|x| x.is_finite())
        })
    }

    pub fn apply_adamw(&mut self, lr: f32) -> Result<(), String> {
        self.step_counter = self
            .step_counter
            .checked_add(1)
            .ok_or_else(|| "optimizer step counter overflow".to_string())?;
        let step = self.step_counter;
        let b1 = self.cfg.beta1;
        let b2 = self.cfg.beta2;
        let wd = self.cfg.weight_decay;
        let eps = self.cfg.eps;
        for p in [
            &mut self.token_embed,
            &mut self.qkv,
            &mut self.out_proj,
            &mut self.ff1,
            &mut self.ff2,
            &mut self.unembed,
        ] {
            p.step_adamw(lr, b1, b2, wd, eps, step);
        }
        for p in [
            &mut self.norm1_gamma,
            &mut self.norm1_beta,
            &mut self.norm2_gamma,
            &mut self.norm2_beta,
        ] {
            p.step_adamw(lr, b1, b2, 0.0, eps, step);
        }
        Ok(())
    }

    #[inline]
    fn position(pos: usize, channel: usize, d: usize) -> f32 {
        let pair = (channel / 2) as f32;
        let angle = pos as f32 / 10_000.0_f32.powf(2.0 * pair / d as f32);
        if channel % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        }
    }

    fn rms_forward(input: &[f32], gamma: &ParamVector, beta: &ParamVector, out: &mut [f32]) -> f32 {
        let d = input.len();
        let inv = rms_norm_slice(input, out);
        for i in 0..d {
            out[i] = gamma.data[i] * out[i] + beta.data[i];
        }
        inv
    }

    fn project(p: &ParamMatrix, input: &[f32], out: &mut [f32]) {
        for r in 0..p.rows {
            out[r] = dot_slice(&p.data[r * p.cols..(r + 1) * p.cols], input);
        }
    }

    fn forward_into(
        &self,
        ids: &[usize],
        targets: Option<&[usize]>,
        tape: &mut TransformerTape,
    ) -> f32 {
        let l = ids.len();
        assert!(
            l > 0 && l <= tape.max_l,
            "transformer sequence length exceeds tape"
        );
        if let Some(targets) = targets {
            assert_eq!(targets.len(), l);
        }
        let d = self.cfg.d_model;
        let v = self.cfg.d_vocab;
        let heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        let logit_scale = 1.0 / (d as f32).sqrt();
        tape.len = l;
        tape.input_ids[..l].copy_from_slice(ids);
        for t in 0..l {
            assert!(ids[t] < v, "token ID outside transformer vocabulary");
            let xoff = t * d;
            let erow = &self.token_embed.data[ids[t] * d..(ids[t] + 1) * d];
            for i in 0..d {
                tape.x[xoff + i] = erow[i] + Self::position(t, i, d);
            }
            tape.inv_rms1[t] = Self::rms_forward(
                &tape.x[xoff..xoff + d],
                &self.norm1_gamma,
                &self.norm1_beta,
                &mut tape.norm1[xoff..xoff + d],
            );
            Self::project(
                &self.qkv,
                &tape.norm1[xoff..xoff + d],
                &mut tape.qkv[t * 3 * d..(t + 1) * 3 * d],
            );
        }
        tape.attn_probs.fill(0.0);
        tape.context.fill(0.0);
        for t in 0..l {
            for h in 0..heads {
                let poff = (t * heads + h) * tape.max_l;
                let qoff = t * 3 * d + h * hd;
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=t {
                    let koff = j * 3 * d + d + h * hd;
                    let score =
                        dot_slice(&tape.qkv[qoff..qoff + hd], &tape.qkv[koff..koff + hd]) * scale;
                    tape.attn_probs[poff + j] = score;
                    max_score = max_score.max(score);
                }
                let mut sum = 0.0;
                for j in 0..=t {
                    let p = (tape.attn_probs[poff + j] - max_score).exp();
                    tape.attn_probs[poff + j] = p;
                    sum += p;
                }
                for j in 0..=t {
                    tape.attn_probs[poff + j] /= sum;
                    let voff = j * 3 * d + 2 * d + h * hd;
                    let coff = t * d + h * hd;
                    let p = tape.attn_probs[poff + j];
                    for k in 0..hd {
                        tape.context[coff + k] += p * tape.qkv[voff + k];
                    }
                }
            }
            let xoff = t * d;
            Self::project(
                &self.out_proj,
                &tape.context[xoff..xoff + d],
                &mut tape.attn_out[xoff..xoff + d],
            );
            for i in 0..d {
                tape.residual1[xoff + i] = tape.x[xoff + i] + tape.attn_out[xoff + i];
            }
            tape.inv_rms2[t] = Self::rms_forward(
                &tape.residual1[xoff..xoff + d],
                &self.norm2_gamma,
                &self.norm2_beta,
                &mut tape.norm2[xoff..xoff + d],
            );
            Self::project(
                &self.ff1,
                &tape.norm2[xoff..xoff + d],
                &mut tape.ff_pre[t * self.cfg.d_ff..(t + 1) * self.cfg.d_ff],
            );
            for i in 0..self.cfg.d_ff {
                let z = tape.ff_pre[t * self.cfg.d_ff + i];
                tape.ff_act[t * self.cfg.d_ff + i] = z * sigmoid(z);
            }
            Self::project(
                &self.ff2,
                &tape.ff_act[t * self.cfg.d_ff..(t + 1) * self.cfg.d_ff],
                &mut tape.residual2[xoff..xoff + d],
            );
            for i in 0..d {
                tape.residual2[xoff + i] += tape.residual1[xoff + i];
            }
            for row in 0..v {
                tape.logits[t * v + row] = dot_slice(
                    &self.unembed.data[row * d..(row + 1) * d],
                    &tape.residual2[xoff..xoff + d],
                ) * logit_scale;
            }
        }
        if let Some(targets) = targets {
            let mut sum_loss = 0.0;
            for t in 0..l {
                let logits = &tape.logits[t * v..(t + 1) * v];
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter().map(|x| (*x - max).exp()).sum();
                let target = targets[t];
                assert!(target < v, "target ID outside transformer vocabulary");
                let loss = max + sum_exp.ln() - logits[target];
                tape.losses[t] = loss;
                sum_loss += loss;
                let probs = &mut tape.probs[t * v..(t + 1) * v];
                for i in 0..v {
                    probs[i] = (logits[i] - max).exp() / sum_exp;
                }
                tape.target_ids[t] = target;
            }
            sum_loss / l as f32
        } else {
            0.0
        }
    }

    pub fn forward_train_chunk(&mut self, ids: &[usize], targets: &[usize]) -> f32 {
        let mut tape = self.tape.take().expect("training tape");
        let loss = self.forward_into(ids, Some(targets), &mut tape);
        self.tape = Some(tape);
        loss
    }

    fn backward_rms(
        input: &[f32],
        inv: f32,
        gamma: &[f32],
        gy: &[f32],
        gamma_grad: &mut [f32],
        beta_grad: &mut [f32],
        gx: &mut [f32],
    ) {
        let d = input.len();
        let mut dot = 0.0;
        for i in 0..d {
            let scaled = gamma[i] * gy[i];
            dot += input[i] * scaled;
            gamma_grad[i] += gy[i] * input[i] * inv;
            beta_grad[i] += gy[i];
        }
        let correction = dot * inv * inv / d as f32;
        for i in 0..d {
            gx[i] = inv * (gamma[i] * gy[i] - input[i] * correction);
        }
    }

    pub fn backward_chunk(&mut self, len: usize, loss_scale: f32) {
        let mut tape = self.tape.take().expect("training tape");
        assert_eq!(len, tape.len);
        assert!(len > 0 && loss_scale.is_finite());
        tape.clear_backward();
        let d = self.cfg.d_model;
        let ff = self.cfg.d_ff;
        let v = self.cfg.d_vocab;
        let heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim();
        let attention_scale = 1.0 / (hd as f32).sqrt();
        let logit_scale = 1.0 / (d as f32).sqrt();

        // Local tape permits disjoint borrows of normalization data/gradients.
        let mut gx = vec![0.0; d];
        let mut gp = vec![0.0; len];
        // Output projection and the gradient entering each post-MLP residual.
        for t in 0..len {
            let xoff = t * d;
            let goff = t * v;
            for row in 0..v {
                let g = (tape.probs[goff + row]
                    - if row == tape.target_ids[t] { 1.0 } else { 0.0 })
                    * loss_scale
                    / len as f32;
                let woff = row * d;
                for i in 0..d {
                    self.unembed.grad[woff + i] += g * tape.residual2[xoff + i] * logit_scale;
                    tape.g_residual2[xoff + i] += self.unembed.data[woff + i] * g * logit_scale;
                }
            }
        }

        // Reverse the feed-forward block and its second RMSNorm.
        for t in (0..len).rev() {
            let xoff = t * d;
            let foff = t * ff;
            for row in 0..d {
                let g = tape.g_residual2[xoff + row];
                let woff = row * ff;
                for i in 0..ff {
                    self.ff2.grad[woff + i] += g * tape.ff_act[foff + i];
                    tape.g_ff_pre[foff + i] += self.ff2.data[woff + i] * g;
                }
                tape.g_residual1[xoff + row] += g;
            }
            for i in 0..ff {
                let z = tape.ff_pre[foff + i];
                let s = sigmoid(z);
                tape.g_ff_pre[foff + i] *= s + z * s * (1.0 - s);
            }
            for row in 0..ff {
                let g = tape.g_ff_pre[foff + row];
                let woff = row * d;
                for i in 0..d {
                    self.ff1.grad[woff + i] += g * tape.norm2[xoff + i];
                    tape.g_norm2[xoff + i] += self.ff1.data[woff + i] * g;
                }
            }
            Self::backward_rms(
                &tape.residual1[xoff..xoff + d],
                tape.inv_rms2[t],
                &self.norm2_gamma.data,
                &tape.g_norm2[xoff..xoff + d],
                &mut self.norm2_gamma.grad,
                &mut self.norm2_beta.grad,
                &mut gx,
            );
            for i in 0..d {
                tape.g_residual1[xoff + i] += gx[i];
                tape.g_attn_out[xoff + i] = tape.g_residual1[xoff + i];
                tape.g_x[xoff + i] += tape.g_residual1[xoff + i];
            }
        }

        // Output projection backpropagation gives gradients for each head's
        // context.  Attention gradients are then accumulated in reverse time
        // so future queries contribute to earlier keys and values.
        for t in 0..len {
            let xoff = t * d;
            for row in 0..d {
                let g = tape.g_attn_out[xoff + row];
                let woff = row * d;
                for i in 0..d {
                    self.out_proj.grad[woff + i] += g * tape.context[xoff + i];
                    tape.g_context[xoff + i] += self.out_proj.data[woff + i] * g;
                }
            }
        }
        let mut gq = vec![0.0; d];
        for t in (0..len).rev() {
            let xoff = t * d;
            let qkv_off = t * 3 * d;
            for h in 0..heads {
                let poff = (t * heads + h) * tape.max_l;
                let qoff = qkv_off + h * hd;
                gq.fill(0.0);
                let mut weighted = 0.0;
                for j in 0..=t {
                    let vj = j * 3 * d + 2 * d + h * hd;
                    gp[j] = dot_slice(
                        &tape.g_context[xoff + h * hd..xoff + (h + 1) * hd],
                        &tape.qkv[vj..vj + hd],
                    );
                    weighted += tape.attn_probs[poff + j] * gp[j];
                }
                for j in 0..=t {
                    let p = tape.attn_probs[poff + j];
                    let score_grad = p * (gp[j] - weighted);
                    let kv = j * 3 * d + d + h * hd;
                    let vv = j * 3 * d + 2 * d + h * hd;
                    for k in 0..hd {
                        let gc = tape.g_context[xoff + h * hd + k];
                        tape.g_qkv[vv + k] += p * gc;
                        gq[h * hd + k] += score_grad * tape.qkv[kv + k] * attention_scale;
                        tape.g_qkv[kv + k] += score_grad * tape.qkv[qoff + k] * attention_scale;
                    }
                }
                for k in 0..hd {
                    tape.g_qkv[qoff + k] += gq[h * hd + k];
                }
            }
            let noff = t * d;
            for row in 0..3 * d {
                let g = tape.g_qkv[qkv_off + row];
                let woff = row * d;
                for i in 0..d {
                    self.qkv.grad[woff + i] += g * tape.norm1[noff + i];
                    tape.g_norm1[noff + i] += self.qkv.data[woff + i] * g;
                }
            }
            Self::backward_rms(
                &tape.x[noff..noff + d],
                tape.inv_rms1[t],
                &self.norm1_gamma.data,
                &tape.g_norm1[noff..noff + d],
                &mut self.norm1_gamma.grad,
                &mut self.norm1_beta.grad,
                &mut gx,
            );
            let id = tape.input_ids[t];
            let eoff = id * d;
            for i in 0..d {
                tape.g_x[noff + i] += gx[i];
                self.token_embed.grad[eoff + i] += tape.g_x[noff + i];
            }
        }
        self.tape = Some(tape);
    }

    /// Returns logits for the final token in a context window.  Like the
    /// training path, attention is causal and is capped at the configured
    /// chunk length; the oldest prompt tokens are dropped when necessary.
    pub fn logits_for_context(&self, ids: &[usize], out: &mut [f32]) -> Result<(), String> {
        if ids.is_empty() {
            return Err("transformer context is empty".into());
        }
        if out.len() != self.cfg.d_vocab {
            return Err("transformer logits buffer has the wrong size".into());
        }
        let start = ids.len().saturating_sub(self.cfg.chunk_len);
        let context = &ids[start..];
        if context.iter().any(|&id| id >= self.cfg.d_vocab) {
            return Err("token ID outside transformer vocabulary".into());
        }
        let mut tape = TransformerTape::new(&self.cfg);
        self.forward_into(context, None, &mut tape);
        let off = (context.len() - 1) * self.cfg.d_vocab;
        out.copy_from_slice(&tape.logits[off..off + self.cfg.d_vocab]);
        Ok(())
    }
}
