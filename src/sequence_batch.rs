//! Independent sequence lanes with packed (not padded) dense token rows.
//!
//! Parameters and episodic memory are shared. Memory must stay unchanged between
//! forward and backward; insertions belong after backward. Each lane owns its SSM
//! carry/tape and reverse-time scratch, so scans run in parallel without atomics
//! or cross-sequence adjoints. Dense stages use the same CPU/GPU dispatch as L=1
//! chunk training, but see the sum of the active sequence lengths as GEMM rows.
use crate::{
    gpu_batch as stages,
    linalg::sigmoid,
    pssa::{ChunkActivationTape, PSSALayerV2},
};
use rayon::prelude::*;

pub struct Sequence<'a> {
    /// Stable lane index; omitted lanes keep their carry and do no work.
    pub lane: usize,
    pub inputs: &'a [usize],
    pub targets: &'a [usize],
    /// Reset only this lane, e.g. at a document boundary.
    pub reset: bool,
}

struct Lane {
    offset: usize,
    len: usize,
    carry: Vec<f32>,
    h: Vec<f32>,
    bar_a: Vec<f32>,
    y: Vec<f32>,
    gh: Vec<f32>,
    ga: Vec<f32>,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    gx: Vec<f32>,
}

/// Runtime-only workspace. Does not change model configuration or checkpoint
/// bytes. A forward must be followed by backward before another forward.
pub struct SequenceBatch {
    lanes: Vec<Lane>,
    shape: [usize; 7],
    tokens: usize,
    pending: bool,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    gx: Vec<f32>,
}

fn shape(m: &PSSALayerV2) -> [usize; 7] {
    [
        m.cfg.chunk_len,
        m.cfg.d_latent,
        m.cfg.d_state,
        m.cfg.d_vocab,
        m.cfg.d_mem_key,
        m.cfg.mem_capacity,
        m.adapters[0].rank,
    ]
}

impl SequenceBatch {
    pub fn new(m: &mut PSSALayerV2, batch_size: usize) -> Result<Self, String> {
        if batch_size == 0 {
            return Err(
                "batch size must be positive; use --batch-size 1 for serial training".into(),
            );
        }
        let [l, d, s, v, k, mem, rank] = shape(m);
        let rows = l
            .checked_mul(batch_size)
            .ok_or("batch tape size overflow; reduce --batch-size")?;
        // Conservative bound includes both the packed model tape and lane-local
        // recurrent tapes/scratch. Reuse the checked 1 GiB model allocation guard.
        let mut budget = m.cfg.clone();
        budget.chunk_len = l
            .checked_add(1)
            .and_then(|n| n.checked_mul(batch_size))
            .and_then(|n| n.checked_mul(2))
            .ok_or("batch allocation overflow; reduce --batch-size")?;
        crate::checkpoint::validate_model_config(&budget)
            .map_err(|e| format!("batch workspace: {e}; reduce --batch-size or --chunk"))?;
        m.tape = ChunkActivationTape::new(rows, v, d, s, k, mem, rank);
        m.bwd_g_zfinal.resize(rows * d, 0.0);
        m.bwd_g_zraw.resize(rows * d, 0.0);
        m.bwd_g_ad_down.resize(rows * rank, 0.0);
        m.bwd_g_xnorm.resize(rows * d, 0.0);
        m.bwd_g_ysm.resize(rows * d, 0.0);
        m.bwd_g_logits.resize(rows * v, 0.0);
        m.bwd_g_mlp.resize(rows * 2 * d, 0.0);
        let lanes = (0..batch_size)
            .map(|_| Lane {
                offset: 0,
                len: 0,
                carry: vec![0.0; d * s],
                h: vec![0.0; (l + 1) * d * s],
                bar_a: vec![0.0; l * d * s],
                y: vec![0.0; l * d],
                gh: vec![0.0; d * s],
                ga: vec![0.0; d * s],
                gd: vec![0.0; l * d],
                gb: vec![0.0; l * s],
                gc: vec![0.0; l * s],
                gx: vec![0.0; l * d],
            })
            .collect();
        Ok(Self {
            lanes,
            shape: shape(m),
            tokens: 0,
            pending: false,
            gd: vec![0.0; rows * d],
            gb: vec![0.0; rows * s],
            gc: vec![0.0; rows * s],
            gx: vec![0.0; rows * d],
        })
    }

    pub fn state(&self, lane: usize) -> &[f32] {
        &self.lanes[lane].carry
    }
    pub fn state_mut(&mut self, lane: usize) -> &mut [f32] {
        &mut self.lanes[lane].carry
    }
    pub fn reset_states(&mut self) {
        for lane in &mut self.lanes {
            lane.carry.fill(0.0);
        }
    }

    fn check_model(&self, m: &PSSALayerV2) -> Result<(), String> {
        if shape(m) != self.shape || m.tape.max_l < self.shape[0] * self.lanes.len() {
            return Err(
                "sequence batch workspace does not match model; create a new workspace".into(),
            );
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        m: &mut PSSALayerV2,
        sequences: &[Sequence<'_>],
    ) -> Result<f32, String> {
        self.check_model(m)?;
        if self.pending {
            return Err("finish batch backward before the next forward".into());
        }
        if sequences.is_empty() || sequences.len() > self.lanes.len() {
            return Err("sequence count must be between 1 and the configured batch size".into());
        }
        // Validate the whole batch before mutating any carry/tape.
        for (i, seq) in sequences.iter().enumerate() {
            if seq.lane >= self.lanes.len()
                || sequences[..i].iter().any(|other| other.lane == seq.lane)
            {
                return Err(
                    "each sequence needs a distinct lane within the configured batch size".into(),
                );
            }
            if seq.inputs.is_empty()
                || seq.inputs.len() > m.cfg.chunk_len
                || seq.inputs.len() != seq.targets.len()
            {
                return Err("each sequence needs matching nonempty input/target slices no longer than --chunk".into());
            }
            if seq
                .inputs
                .iter()
                .chain(seq.targets)
                .any(|&id| id >= m.cfg.d_vocab)
            {
                return Err("batch token ID is outside the model vocabulary".into());
            }
        }
        self.tokens = 0;
        for lane in &mut self.lanes {
            lane.len = 0;
        }
        for seq in sequences {
            let lane = &mut self.lanes[seq.lane];
            lane.offset = self.tokens;
            lane.len = seq.inputs.len();
            if seq.reset {
                lane.carry.fill(0.0);
            }
            let end = self.tokens + lane.len;
            m.tape.x_ids[self.tokens..end].copy_from_slice(seq.inputs);
            m.tape.target_ids[self.tokens..end].copy_from_slice(seq.targets);
            self.tokens = end;
        }
        stages::stage_embed_norm(m, self.tokens);
        stages::stage_projections(m, self.tokens);
        m.refresh_ssm_rates();
        self.lanes
            .par_iter_mut()
            .filter(|lane| lane.len > 0)
            .for_each(|lane| lane.forward(m));
        for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
            let start = lane.offset * m.cfg.d_latent;
            m.tape.y_ssm[start..start + lane.len * m.cfg.d_latent]
                .copy_from_slice(&lane.y[..lane.len * m.cfg.d_latent]);
        }
        stages::stage_memory(m, self.tokens);
        stages::stage_adapter(m, self.tokens);
        stages::stage_mlp(m, self.tokens);
        let loss = stages::stage_logits_loss(m, self.tokens);
        self.pending = true;
        Ok(loss)
    }

    /// Accumulate the token-mean gradient, multiplied by accumulation_scale.
    /// For a larger optimizer group pass batch_tokens / group_tokens. Carry is
    /// retained for the next forward, but TBPTT never differentiates across calls.
    pub fn backward(&mut self, m: &mut PSSALayerV2, accumulation_scale: f32) -> Result<(), String> {
        self.check_model(m)?;
        if !self.pending || !accumulation_scale.is_finite() {
            return Err(
                "batch backward needs a pending forward and a finite accumulation scale".into(),
            );
        }
        let n = self.tokens;
        stages::bwd_stage_logits(m, n, accumulation_scale / n as f32);
        stages::bwd_stage_mlp(m, n);
        stages::bwd_stage_adapter(m, n);
        stages::bwd_stage_adapter_down(m, n);
        stages::bwd_stage_memory(m, n);
        m.refresh_ssm_rates();
        self.lanes
            .par_iter_mut()
            .filter(|lane| lane.len > 0)
            .for_each(|lane| lane.backward(m));
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
            let start = lane.offset * d;
            let end = start + lane.len * d;
            self.gd[start..end].copy_from_slice(&lane.gd[..lane.len * d]);
            self.gb[lane.offset * s..(lane.offset + lane.len) * s]
                .copy_from_slice(&lane.gb[..lane.len * s]);
            self.gc[lane.offset * s..(lane.offset + lane.len) * s]
                .copy_from_slice(&lane.gc[..lane.len * s]);
            for (dst, src) in m.bwd_g_xnorm[start..end].iter_mut().zip(&lane.gx) {
                *dst += src;
            }
            for (dst, src) in m.a_mat.grad.iter_mut().zip(&lane.ga) {
                *dst += src;
            }
        }
        // Projection adjoints share weights across ALL sequences, like the head.
        for (g, w, rows) in [
            (&self.gd, &mut m.w_delta, d),
            (&self.gb, &mut m.w_b, s),
            (&self.gc, &mut m.w_c, s),
        ] {
            stages::dense_input_adjoint(g, &w.data, n, rows, d, &mut self.gx[..n * d]);
            stages::dense_weight_adjoint(g, &m.tape.x_norm, n, rows, d, &mut w.grad);
            for (dst, src) in m.bwd_g_xnorm[..n * d].iter_mut().zip(&self.gx) {
                *dst += src;
            }
        }
        for t in (0..n).rev() {
            let id = m.tape.x_ids[t];
            m.embed_row_marks[id] = m.step_counter + 1;
            let inv = m.tape.inv_rms[t];
            let e = &m.embed_w.data[id * d..(id + 1) * d];
            let gx = &m.bwd_g_xnorm[t * d..(t + 1) * d];
            let mut dot = 0.0;
            for i in 0..d {
                m.norm_beta.grad[i] += gx[i];
                m.norm_gamma.grad[i] += gx[i] * (e[i] * inv);
                dot += gx[i] * m.norm_gamma.data[i] * e[i];
            }
            for i in 0..d {
                m.embed_w.grad[id * d + i] +=
                    inv * (gx[i] * m.norm_gamma.data[i] - e[i] * (dot * inv * inv / d as f32));
            }
        }
        self.pending = false;
        Ok(())
    }
}

impl Lane {
    fn forward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        self.h[..hs].copy_from_slice(&self.carry);
        for t in 0..self.len {
            let row = self.offset + t;
            for i in 0..d {
                let delta = m.tape.delta[row * d + i];
                let mut y = 0.0;
                for j in 0..s {
                    let idx = i * s + j;
                    let a = (delta * m.ssm_rates[idx]).exp();
                    self.bar_a[t * hs + idx] = a;
                    let h = a * self.h[t * hs + idx]
                        + (delta * m.tape.b_proj[row * s + j]) * m.tape.x_norm[row * d + i];
                    self.h[(t + 1) * hs + idx] = h;
                    y += h * m.tape.c_proj[row * s + j];
                }
                self.y[t * d + i] = y;
            }
        }
        self.carry
            .copy_from_slice(&self.h[self.len * hs..(self.len + 1) * hs]);
    }

    fn backward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let scale = 1.0 / (s as f32).sqrt();
        self.gh.fill(0.0);
        self.ga.fill(0.0);
        self.gd[..self.len * d].fill(0.0);
        self.gb[..self.len * s].fill(0.0);
        self.gc[..self.len * s].fill(0.0);
        self.gx[..self.len * d].fill(0.0);
        for t in (0..self.len).rev() {
            let row = self.offset + t;
            for i in 0..d {
                let gy = m.bwd_g_zraw[row * d + i] * scale + m.bwd_g_ysm[row * d + i];
                let delta = m.tape.delta[row * d + i];
                let x = m.tape.x_norm[row * d + i];
                for j in 0..s {
                    let idx = i * s + j;
                    let a = self.bar_a[t * hs + idx];
                    let b = m.tape.b_proj[row * s + j];
                    let prev = self.h[t * hs + idx];
                    let gh = gy * m.tape.c_proj[row * s + j] + self.gh[idx];
                    self.gc[t * s + j] += gy * self.h[(t + 1) * hs + idx];
                    self.gh[idx] = gh * a;
                    self.ga[idx] += gh * (delta * a) * prev * m.ssm_rate_derivatives[idx];
                    self.gd[t * d + i] += gh * (m.ssm_rates[idx] * a * prev + b * x);
                    self.gb[t * s + j] += gh * (delta * x);
                    self.gx[t * d + i] += gh * (delta * b);
                }
                self.gd[t * d + i] *= sigmoid(m.tape.delta_raw[row * d + i]);
            }
        }
    }
}
