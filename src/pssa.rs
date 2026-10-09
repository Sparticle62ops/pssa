use crate::adapter::PlasticAdapterV2;
use crate::backend::Device;
use crate::linalg::{SimpleRng, dot_slice, sigmoid, softplus};
use crate::memory::HyperbolicEpisodicBankV2;
use std::time::Instant;

// =============================================================================
// PARAMETER TENSORS WITH INTEGRATED ADAMW MOMENTS
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct ParamVector {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamVector {
    pub fn new(len: usize, init_val: f32) -> Self {
        Self {
            data: vec![init_val; len],
            grad: vec![0.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    #[inline(always)]
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
        assert!(step > 0, "AdamW step must be positive");
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);

        for i in 0..self.data.len() {
            let g = self.grad[i];
            if weight_decay > 0.0 {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let total = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; total],
            grad: vec![0.0; total],
            m: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    pub fn random_xavier(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let total = rows * cols;
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(rng.gen_range_f32(-limit, limit));
        }
        Self {
            rows,
            cols,
            data,
            grad: vec![0.0; total],
            m: vec![0.0; total],
            v: vec![0.0; total],
        }
    }

    #[inline(always)]
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    #[inline(always)]
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(self.cols, x.len());
        assert_eq!(self.rows, out.len());
        for r in 0..self.rows {
            let row_slice = &self.data[r * self.cols..(r + 1) * self.cols];
            out[r] = dot_slice(row_slice, x);
        }
    }

    /// Reuse each weight row across a small prompt tile. Each dot has exactly
    /// the same reduction order as `matvec`; this is not a reassociated GEMM.
    fn matvec_batch(&self, inputs: &[f32], output: &mut [f32]) {
        assert!(self.cols > 0);
        assert_eq!(inputs.len() % self.cols, 0);
        assert_eq!(output.len(), inputs.len() / self.cols * self.rows);
        for r in 0..self.rows {
            let weights = &self.data[r * self.cols..(r + 1) * self.cols];
            for (t, input) in inputs.chunks_exact(self.cols).enumerate() {
                output[t * self.rows + r] = dot_slice(weights, input);
            }
        }
    }

    #[inline(always)]
    pub fn matvec_transpose(&self, g: &[f32], out: &mut [f32]) {
        assert_eq!(self.rows, g.len());
        assert_eq!(self.cols, out.len());
        out.fill(0.0);
        for r in 0..self.rows {
            let gr = g[r];
            let row_off = r * self.cols;
            for c in 0..self.cols {
                out[c] += gr * self.data[row_off + c];
            }
        }
    }

    pub fn step_adamw_row(
        &mut self,
        row: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
        assert!(row < self.rows);
        assert!(step > 0, "AdamW step must be positive");
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);
        let row_start = row * self.cols;
        let row_end = row_start + self.cols;

        for i in row_start..row_end {
            let g = self.grad[i];
            if weight_decay > 0.0 {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        eps: f32,
        step: usize,
    ) {
        assert!(step > 0, "AdamW step must be positive");
        let step_f = step as f32;
        let bias_corr1 = 1.0 - beta1.powf(step_f);
        let bias_corr2 = 1.0 - beta2.powf(step_f);

        for i in 0..self.data.len() {
            let g = self.grad[i];
            if weight_decay > 0.0 {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Result of an opt-in, globally clipped optimizer update. The norm is computed
/// once, before scaling, in f64 so large finite f32 gradients do not overflow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientClipOutcome {
    Applied { norm: f64 },
    Skipped { norm: f64 },
}

/// Borrowed optimizer state, in the same order as the historical global norm.
/// A non-unit scale is fused into Adam's existing pass, not a second clip walk.
pub(crate) struct AdamTensor<'a> {
    pub data: &'a mut [f32],
    pub grad: &'a mut [f32],
    pub m: &'a mut [f32],
    pub v: &'a mut [f32],
    pub weight_decay: f32,
}

impl AdamTensor<'_> {
    fn step(&mut self, cfg: &PSSAConfigV2, lr: f32, step: usize, scale: f64) {
        let bias1 = 1.0 - cfg.beta1.powf(step as f32);
        let bias2 = 1.0 - cfg.beta2.powf(step as f32);
        for i in 0..self.data.len() {
            let g = if scale < 1.0 {
                let g = (self.grad[i] as f64 * scale) as f32;
                self.grad[i] = g;
                g
            } else {
                self.grad[i]
            };
            if self.weight_decay > 0.0 {
                self.data[i] -= lr * self.weight_decay * self.data[i];
            }
            self.m[i] = cfg.beta1 * self.m[i] + (1.0 - cfg.beta1) * g;
            self.v[i] = cfg.beta2 * self.v[i] + (1.0 - cfg.beta2) * g * g;
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + cfg.eps);
        }
    }
}

// =============================================================================
// ENGINE CONFIGURATION
// =============================================================================

#[derive(Clone, Debug)]
pub struct PSSAConfigV2 {
    /// Number of independently owned continuous blocks (1..=32).
    pub depth: usize,
    pub d_vocab: usize,
    pub d_latent: usize,
    pub d_state: usize,
    pub d_mem_key: usize,
    pub mem_capacity: usize,
    pub chunk_len: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub eps: f32,
    pub tau_mem: f32,
    pub ema_alpha: f32,
}

impl PSSAConfigV2 {
    pub fn validate(&self) {
        assert!((1..=32).contains(&self.depth), "depth must be in 1..=32");
        assert!(
            self.d_vocab > 0
                && self.d_latent > 0
                && self.d_state > 0
                && self.d_mem_key > 0
                && self.mem_capacity > 0
                && self.chunk_len > 0,
            "all model dimensions, memory capacity, and chunk length must be positive"
        );
        assert!(
            self.lr.is_finite()
                && self.lr > 0.0
                && self.eps.is_finite()
                && self.eps > 0.0
                && self.tau_mem.is_finite()
                && self.tau_mem > 0.0,
            "lr, eps, and tau_mem must be finite and positive"
        );
        assert!(
            self.beta1.is_finite()
                && self.beta1 >= 0.0
                && self.beta1 < 1.0
                && self.beta2.is_finite()
                && self.beta2 >= 0.0
                && self.beta2 < 1.0,
            "Adam betas must be in [0, 1)"
        );
        assert!(
            self.weight_decay.is_finite()
                && self.weight_decay >= 0.0
                && self.ema_alpha.is_finite()
                && (0.0..=1.0).contains(&self.ema_alpha),
            "weight decay must be nonnegative and ema_alpha must be in [0, 1]"
        );
    }
}

impl Default for PSSAConfigV2 {
    fn default() -> Self {
        Self {
            depth: 1,
            d_vocab: 10_000,
            d_latent: 256,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            chunk_len: 64,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            tau_mem: 0.1,
            ema_alpha: 0.01,
        }
    }
}

// =============================================================================
// ZERO-ALLOCATION TBPTT ACTIVATION TAPE
// =============================================================================

#[derive(Clone, Debug)]
pub struct ChunkActivationTape {
    pub max_l: usize,
    pub x_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
    /// Raw continuous inputs, retained for the RMSNorm input adjoint.
    pub x_raw: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub inv_rms: Vec<f32>,
    pub delta_raw: Vec<f32>,
    pub delta: Vec<f32>,
    pub b_proj: Vec<f32>,
    pub c_proj: Vec<f32>,
    pub bar_a: Vec<f32>,
    pub bar_b: Vec<f32>,
    pub h_states: Vec<f32>,
    pub y_ssm: Vec<f32>,
    pub q_euc: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub q_poincare: Vec<f32>,
    pub mem_weights: Vec<f32>,
    pub m_val: Vec<f32>,
    pub g_mem: Vec<f32>,
    pub m_inj: Vec<f32>,
    pub m_proj: Vec<f32>,
    pub adapter_hidden: Vec<f32>,
    pub adapter_act: Vec<f32>,
    pub z_raw: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub mlp_act: Vec<f32>,
    pub z_final: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub losses: Vec<f32>,
}

impl ChunkActivationTape {
    pub fn new(
        max_l: usize,
        d_vocab: usize,
        d_latent: usize,
        d_state: usize,
        d_key: usize,
        mem_cap: usize,
        rank: usize,
    ) -> Self {
        Self::new_with_loops(max_l, d_vocab, d_latent, d_state, d_key, mem_cap, rank, 1)
    }

    /// Allocate one contiguous activation tape per Ouro iteration. The public
    /// tape shape remains unchanged for the historical one-loop path: token
    /// IDs, logits, probabilities, and losses are final-output storage, while
    /// the intermediate activation arrays are laid out as
    /// `[loop][token][feature]`.
    pub fn new_with_loops(
        max_l: usize,
        d_vocab: usize,
        d_latent: usize,
        d_state: usize,
        d_key: usize,
        mem_cap: usize,
        rank: usize,
        loops: usize,
    ) -> Self {
        assert!((1..=32).contains(&loops));
        // The staged batched path uses the zero-offset region as a working
        // buffer. Keep one extra runtime-only slot so loop zero survives the
        // later passes for reverse-mode; the historical scalar tape shape is
        // unchanged when loops == 1.
        let saved_slots = if loops > 1 { loops + 1 } else { 1 };
        let loop_l = max_l * saved_slots;
        Self {
            max_l,
            x_ids: vec![0; max_l],
            target_ids: vec![0; max_l],
            x_raw: vec![0.0; loop_l * d_latent],
            x_norm: vec![0.0; loop_l * d_latent],
            inv_rms: vec![0.0; loop_l],
            delta_raw: vec![0.0; loop_l * d_latent],
            delta: vec![0.0; loop_l * d_latent],
            b_proj: vec![0.0; loop_l * d_state],
            c_proj: vec![0.0; loop_l * d_state],
            bar_a: vec![0.0; loop_l * d_latent * d_state],
            bar_b: vec![0.0; loop_l * d_latent * d_state],
            // Extra loop carries are runtime-only, after the activation slots.
            h_states: vec![
                0.0f32;
                (saved_slots * (max_l + 1) + loops.saturating_sub(1))
                    * d_latent
                    * d_state
            ],
            y_ssm: vec![0.0; loop_l * d_latent],
            q_euc: vec![0.0; loop_l * d_key],
            q_norm: vec![0.0; loop_l],
            q_poincare: vec![0.0; loop_l * d_key],
            mem_weights: vec![0.0; loop_l * mem_cap],
            m_val: vec![0.0; loop_l * d_latent],
            g_mem: vec![0.0; loop_l * d_latent],
            m_inj: vec![0.0; loop_l * d_latent],
            m_proj: vec![0.0; loop_l * d_latent],
            adapter_hidden: vec![0.0; loop_l * rank],
            adapter_act: vec![0.0; loop_l * rank],
            z_raw: vec![0.0; loop_l * d_latent],
            mlp_hidden: vec![0.0; loop_l * 2 * d_latent],
            mlp_act: vec![0.0; loop_l * 2 * d_latent],
            z_final: vec![0.0; loop_l * d_latent],
            logits: vec![0.0; max_l * d_vocab],
            probs: vec![0.0; max_l * d_vocab],
            losses: vec![0.0; max_l],
        }
    }

    #[inline]
    pub fn loop_offset(&self, loop_index: usize, token: usize, width: usize) -> usize {
        loop_index * self.max_l * width + token * width
    }

    #[inline]
    pub fn loop_state_offset(&self, loop_index: usize, token: usize, width: usize) -> usize {
        loop_index * (self.max_l + 1) * width + token * width
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PSSAContinuousConfigV2 {
    pub d_latent: usize,
    pub d_state: usize,
    pub d_mem_key: usize,
    pub mem_capacity: usize,
    pub chunk_len: usize,
    pub tau_mem: f32,
    pub ema_alpha: f32,
}

impl From<&PSSAConfigV2> for PSSAContinuousConfigV2 {
    fn from(c: &PSSAConfigV2) -> Self {
        Self {
            d_latent: c.d_latent,
            d_state: c.d_state,
            d_mem_key: c.d_mem_key,
            mem_capacity: c.mem_capacity,
            chunk_len: c.chunk_len,
            tau_mem: c.tau_mem,
            ema_alpha: c.ema_alpha,
        }
    }
}

/// Vocabulary-free processing block: independently owned weights, carry, bank,
/// adapter slow state, optimizer moments and runtime scratch. The legacy tape
/// type retains endpoint slots for compatibility; only layer zero allocates them.
pub struct PSSAContinuousBlockV2 {
    pub cfg: PSSAContinuousConfigV2,
    pub norm_gamma: ParamVector,
    pub norm_beta: ParamVector,

    // 2. Multi-Channel Continuous SSM Recurrence
    pub a_mat: ParamMatrix,
    pub w_delta: ParamMatrix,
    pub w_b: ParamMatrix,
    pub w_c: ParamMatrix,
    pub h_persistent: Vec<f32>,
    /// Cached transforms, checked against public raw parameters at each use.
    pub ssm_raw_snapshot: Vec<f32>,
    pub ssm_rates: Vec<f32>,
    pub ssm_rate_derivatives: Vec<f32>,

    // 3. Diffeomorphic Poincaré Episodic Memory
    pub w_qx: ParamMatrix,
    pub w_qh: ParamMatrix,
    pub w_gate: ParamMatrix,
    pub w_proj: ParamMatrix,
    pub memory: HyperbolicEpisodicBankV2,

    // 4. Zero-Init Plastic Adapters
    pub adapters: Vec<PlasticAdapterV2>,
    /// Host-side effective adapter-up matrix, refreshed before GPU forward.
    /// Keeping it stable lets the device weight cache retain the combined
    /// fast and consolidated projection without caching a temporary Vec.
    pub adapter_up_effective: Vec<f32>,

    // 5. SiLU Feed-Forward Expansion Head
    pub mlp_w1: ParamMatrix,
    pub mlp_w2: ParamMatrix,

    // Preallocated Tape for Zero-Allocation Training
    pub tape: ChunkActivationTape,

    // Backward Temporary Adjoint Buffers
    pub grad_h_next: Vec<f32>,
    pub grad_z_final: Vec<f32>,
    pub grad_z_raw: Vec<f32>,
    pub grad_x_norm: Vec<f32>,
    pub buf_m_proj: Vec<f32>,
    pub buf_ad_out: Vec<f32>,
    pub buf_g_mlp_act: Vec<f32>,
    pub buf_g_mlp_hidden: Vec<f32>,
    pub buf_g_zraw_mlp: Vec<f32>,
    pub buf_g_ad_act: Vec<f32>,
    pub buf_g_ad_down: Vec<f32>,
    pub buf_g_m_proj_out: Vec<f32>,
    pub buf_g_m_val: Vec<f32>,
    /// Retrieval-query adjoints, reused per reverse-time step.
    pub g_query_pnc: Vec<f32>,
    pub g_query_euc: Vec<f32>,
    pub g_y_ssm: Vec<f32>,
    pub buf_g_delta: Vec<f32>,
    pub buf_g_b_proj: Vec<f32>,
    pub buf_g_c_proj: Vec<f32>,
    pub buf_g_h_prev: Vec<f32>,

    // Batched backward per-token adjoint storage (stage/twin dispatch path)
    pub bwd_g_zfinal: Vec<f32>,
    pub bwd_g_zraw: Vec<f32>,
    pub bwd_g_ad_down: Vec<f32>,
    pub bwd_g_xnorm: Vec<f32>,
    pub bwd_g_ysm: Vec<f32>,
    /// Packed per-token Euclidean query adjoints used by backward GEMMs.
    pub bwd_g_query_euc: Vec<f32>,
    /// Per-token Poincare query adjoints for parallel memory VJPs.
    pub bwd_g_query_pnc: Vec<f32>,
    pub bwd_g_logits: Vec<f32>,
    pub bwd_g_mlp: Vec<f32>,

    // Runtime-only affine scan workspaces. The scan arrays are padded to the
    // next power of two for the work-efficient Blelloch tree; they are not
    // checkpoint state. The per-token adjoints remain separate so the
    // reverse scan can be followed by parallel local derivative work.
    pub ssm_scan_a: Vec<f32>,
    pub ssm_scan_b: Vec<f32>,
    pub bwd_ssm_delta: Vec<f32>,
    pub bwd_ssm_b: Vec<f32>,
    pub bwd_ssm_c: Vec<f32>,
    pub bwd_ssm_a: Vec<f32>,

    // Inference Scratch Buffers
    pub inf_x_norm: Vec<f32>,
    pub inf_delta: Vec<f32>,
    pub inf_b: Vec<f32>,
    pub inf_c: Vec<f32>,
    pub inf_y_ssm: Vec<f32>,
    pub inf_q_euc: Vec<f32>,
    pub inf_q_pnc: Vec<f32>,
    pub inf_mem_weights: Vec<f32>,
    pub inf_m_val: Vec<f32>,
    pub inf_g_mem: Vec<f32>,
    pub inf_m_proj: Vec<f32>,
    pub inf_ad_act: Vec<f32>,
    pub inf_ad_out: Vec<f32>,
    pub inf_z_raw: Vec<f32>,
    pub inf_mlp_act: Vec<f32>,
    pub inf_mlp_out: Vec<f32>,
    pub inf_z_final: Vec<f32>,
}
impl PSSAContinuousBlockV2 {
    pub(crate) fn new_with_rng(
        cfg: PSSAContinuousConfigV2,
        rng: &mut SimpleRng,
        d_v: usize,
    ) -> Self {
        Self::new_with_rng_and_loops(cfg, rng, d_v, 1)
    }

    pub(crate) fn new_with_rng_and_loops(
        cfg: PSSAContinuousConfigV2,
        rng: &mut SimpleRng,
        d_v: usize,
        loops: usize,
    ) -> Self {
        let d_m = cfg.d_latent;
        let d_s = cfg.d_state;
        let d_k = cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let mem_cap = cfg.mem_capacity;
        let chunk_len = cfg.chunk_len;
        let scan_len = chunk_len.next_power_of_two();
        let state_width = d_m * d_s;
        let rank = 16;

        let norm_gamma = ParamVector::new(d_m, 1.0);
        let norm_beta = ParamVector::new(d_m, 0.0);

        // Continuous HiPPO timescale decay initialization
        let mut a_mat = ParamMatrix::zeros(d_m, d_s);
        let ln_min = 1.5f32.ln();
        let ln_max = 200.0f32.ln();
        for i in 0..d_m {
            for j in 0..d_s {
                let ratio = if d_s > 1 {
                    j as f32 / (d_s - 1) as f32
                } else {
                    0.0
                };
                let tau = (ln_min + ratio * (ln_max - ln_min)).exp();
                // Store an unconstrained raw rate. Physical A is -softplus(raw).
                let rate = 1.0 / tau;
                a_mat.data[i * d_s + j] = rate.exp_m1().ln();
            }
        }

        let w_delta = ParamMatrix::random_xavier(d_m, d_m, rng);
        let w_b = ParamMatrix::random_xavier(d_s, d_m, rng);
        let w_c = ParamMatrix::random_xavier(d_s, d_m, rng);
        let h_persistent = vec![0.0; d_m * d_s];

        let w_qx = ParamMatrix::random_xavier(d_k, d_m, rng);
        let w_qh = ParamMatrix::random_xavier(d_k, d_m, rng);
        let w_gate = ParamMatrix::random_xavier(d_m, d_m, rng);
        let w_proj = ParamMatrix::random_xavier(d_m, d_m, rng);
        let memory = HyperbolicEpisodicBankV2::new(mem_cap, d_k, d_m);

        let mut adapters = Vec::new();
        adapters.push(PlasticAdapterV2::new(d_m, rank, rng));
        let adapter_up_effective = vec![0.0; d_m * rank];

        let mlp_w1 = ParamMatrix::random_xavier(d_mlp, d_m, rng);
        let mlp_w2 = ParamMatrix::zeros(d_m, d_mlp);

        let tape = ChunkActivationTape::new_with_loops(
            chunk_len, d_v, d_m, d_s, d_k, mem_cap, rank, loops,
        );
        Self {
            cfg,
            norm_gamma,
            norm_beta,
            a_mat,
            w_delta,
            w_b,
            w_c,
            h_persistent,
            ssm_raw_snapshot: vec![f32::NAN; d_m * d_s],
            ssm_rates: vec![0.0; d_m * d_s],
            ssm_rate_derivatives: vec![0.0; d_m * d_s],
            w_qx,
            w_qh,
            w_gate,
            w_proj,
            memory,
            adapters,
            adapter_up_effective,
            mlp_w1,
            mlp_w2,
            tape,
            grad_h_next: vec![0.0; d_m * d_s],
            grad_z_final: vec![0.0; d_m],
            grad_z_raw: vec![0.0; d_m],
            grad_x_norm: vec![0.0; d_m],
            buf_m_proj: vec![0.0; d_m],
            buf_ad_out: vec![0.0; d_m],
            buf_g_mlp_act: vec![0.0; d_mlp],
            buf_g_mlp_hidden: vec![0.0; d_mlp],
            buf_g_zraw_mlp: vec![0.0; d_m],
            buf_g_ad_act: vec![0.0; rank],
            buf_g_ad_down: vec![0.0; rank],
            buf_g_m_proj_out: vec![0.0; d_m],
            buf_g_m_val: vec![0.0; d_m],
            g_query_pnc: vec![0.0; d_k],
            g_query_euc: vec![0.0; d_k],
            g_y_ssm: vec![0.0; d_m],
            buf_g_delta: vec![0.0; d_m],
            buf_g_b_proj: vec![0.0; d_s],
            buf_g_c_proj: vec![0.0; d_s],
            buf_g_h_prev: vec![0.0; d_m * d_s],
            bwd_g_zfinal: vec![0.0; chunk_len * d_m],
            bwd_g_zraw: vec![0.0; chunk_len * d_m],
            bwd_g_ad_down: vec![0.0; chunk_len * rank],
            bwd_g_xnorm: vec![0.0; chunk_len * d_m],
            bwd_g_ysm: vec![0.0; chunk_len * d_m],
            bwd_g_query_euc: vec![0.0; chunk_len * d_k],
            bwd_g_query_pnc: vec![0.0; chunk_len * d_k],
            bwd_g_logits: vec![0.0; chunk_len * d_v],
            bwd_g_mlp: vec![0.0; chunk_len * d_mlp],
            ssm_scan_a: vec![0.0; scan_len * state_width],
            ssm_scan_b: vec![0.0; scan_len * state_width],
            bwd_ssm_delta: vec![0.0; chunk_len * d_m],
            bwd_ssm_b: vec![0.0; chunk_len * d_s],
            bwd_ssm_c: vec![0.0; chunk_len * d_s],
            bwd_ssm_a: vec![0.0; chunk_len * state_width],
            inf_x_norm: vec![0.0; d_m],
            inf_delta: vec![0.0; d_m],
            inf_b: vec![0.0; d_s],
            inf_c: vec![0.0; d_s],
            inf_y_ssm: vec![0.0; d_m],
            inf_q_euc: vec![0.0; d_k],
            inf_q_pnc: vec![0.0; d_k],
            inf_mem_weights: vec![0.0; mem_cap],
            inf_m_val: vec![0.0; d_m],
            inf_g_mem: vec![0.0; d_m],
            inf_m_proj: vec![0.0; d_m],
            inf_ad_act: vec![0.0; rank],
            inf_ad_out: vec![0.0; d_m],
            inf_z_raw: vec![0.0; d_m],
            inf_mlp_act: vec![0.0; d_mlp],
            inf_mlp_out: vec![0.0; d_m],
            inf_z_final: vec![0.0; d_m],
        }
    }
    /// One transform evaluation per changed raw rate, not per timestep. Compare
    /// values rather than an optimizer version: callers may mutate public weights
    /// directly, including finite-difference probes and checkpoint decoding.
    pub(crate) fn refresh_adapter_up_effective(&mut self) {
        let fast = &self.adapters[0].up_proj.data;
        let consolidated = &self.adapters[0].consolidated_up;
        for ((effective, fast), consolidated) in self
            .adapter_up_effective
            .iter_mut()
            .zip(fast)
            .zip(consolidated)
        {
            *effective = *fast + *consolidated;
        }
    }

    pub(crate) fn refresh_ssm_rates(&mut self) {
        for i in 0..self.a_mat.data.len() {
            let raw = self.a_mat.data[i];
            if raw != self.ssm_raw_snapshot[i] {
                self.ssm_rates[i] = -softplus(raw);
                self.ssm_rate_derivatives[i] = -sigmoid(raw);
                self.ssm_raw_snapshot[i] = raw;
            }
        }
    }

    pub fn reset_recurrent_state(&mut self) {
        self.h_persistent.fill(0.0);
        let start = self.loop_carry_start();
        self.tape.h_states[start..].fill(0.0);
    }

    /// Each virtual pass has its own causal temporal carry, but shares every
    /// weight, optimizer moment and memory bank. Keeping these extra carries
    /// in runtime tape storage leaves the historical checkpoint state intact.
    fn loop_count(&self) -> usize {
        let slots = self.tape.x_raw.len() / (self.tape.max_l * self.cfg.d_latent);
        if slots > 1 { slots - 1 } else { 1 }
    }

    fn saved_loop_slot(&self, loop_index: usize) -> usize {
        assert!(loop_index < self.loop_count());
        if loop_index == 0 && self.loop_count() > 1 {
            self.loop_count()
        } else {
            loop_index
        }
    }

    pub(crate) fn loop_carry_start(&self) -> usize {
        let loops = self.loop_count();
        let saved_slots = if loops > 1 { loops + 1 } else { 1 };
        saved_slots * (self.tape.max_l + 1) * self.h_persistent.len()
    }

    fn swap_loop_carry(&mut self, loop_index: usize) {
        if loop_index != 0 {
            let start = self.loop_carry_start() + (loop_index - 1) * self.h_persistent.len();
            for (state, stored) in self
                .h_persistent
                .iter_mut()
                .zip(&mut self.tape.h_states[start..])
            {
                std::mem::swap(state, stored);
            }
        }
    }

    fn copy_loop_field_to_base(field: &mut [f32], max_l: usize, width: usize, loop_index: usize) {
        if loop_index == 0 {
            return;
        }
        let len = max_l * width;
        let offset = loop_index * len;
        let (base, rest) = field.split_at_mut(offset);
        base[..len].copy_from_slice(&rest[..len]);
    }

    fn copy_base_field_to_loop(field: &mut [f32], max_l: usize, width: usize, loop_index: usize) {
        if loop_index == 0 {
            return;
        }
        let len = max_l * width;
        let offset = loop_index * len;
        let (base, rest) = field.split_at_mut(offset);
        rest[..len].copy_from_slice(&base[..len]);
    }

    /// Move one loop's saved activations into the historical zero-offset tape
    /// used by the allocation-free batched stages.
    pub(crate) fn copy_loop_tape_to_base(&mut self, loop_index: usize) {
        let l = self.tape.max_l;
        let d = self.cfg.d_latent;
        let s = self.cfg.d_state;
        let k = self.cfg.d_mem_key;
        let mem = self.cfg.mem_capacity;
        let rank = self.adapters[0].rank;
        let saved_slot = self.saved_loop_slot(loop_index);
        Self::copy_loop_field_to_base(&mut self.tape.x_raw, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.x_norm, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.inv_rms, l, 1, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.delta_raw, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.delta, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.b_proj, l, s, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.c_proj, l, s, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.bar_a, l, d * s, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.bar_b, l, d * s, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.y_ssm, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.q_euc, l, k, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.q_norm, l, 1, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.q_poincare, l, k, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.mem_weights, l, mem, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.m_val, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.g_mem, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.m_inj, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.m_proj, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.adapter_hidden, l, rank, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.adapter_act, l, rank, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.z_raw, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.mlp_hidden, l, 2 * d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.mlp_act, l, 2 * d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.z_final, l, d, saved_slot);
        Self::copy_loop_field_to_base(&mut self.tape.h_states, l + 1, d * s, saved_slot);
    }

    /// Save the batched stages' zero-offset activations into one loop slot.
    pub(crate) fn copy_base_tape_to_loop(&mut self, loop_index: usize) {
        let l = self.tape.max_l;
        let d = self.cfg.d_latent;
        let s = self.cfg.d_state;
        let k = self.cfg.d_mem_key;
        let mem = self.cfg.mem_capacity;
        let rank = self.adapters[0].rank;
        let saved_slot = self.saved_loop_slot(loop_index);
        Self::copy_base_field_to_loop(&mut self.tape.x_raw, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.x_norm, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.inv_rms, l, 1, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.delta_raw, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.delta, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.b_proj, l, s, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.c_proj, l, s, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.bar_a, l, d * s, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.bar_b, l, d * s, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.y_ssm, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.q_euc, l, k, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.q_norm, l, 1, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.q_poincare, l, k, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.mem_weights, l, mem, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.m_val, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.g_mem, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.m_inj, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.m_proj, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.adapter_hidden, l, rank, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.adapter_act, l, rank, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.z_raw, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.mlp_hidden, l, 2 * d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.mlp_act, l, 2 * d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.z_final, l, d, saved_slot);
        Self::copy_base_field_to_loop(&mut self.tape.h_states, l + 1, d * s, saved_slot);
    }

    fn forward_inference_loop(&mut self, input: &[f32], output: &mut [f32], loop_index: usize) {
        self.swap_loop_carry(loop_index);
        self.forward_continuous_inference(input, output);
        self.swap_loop_carry(loop_index);
    }

    #[inline(always)]
    pub fn forward_continuous_inference(&mut self, x_features: &[f32], z_out: &mut [f32]) {
        assert_eq!(x_features.len(), self.cfg.d_latent);
        assert_eq!(z_out.len(), self.cfg.d_latent);
        self.refresh_ssm_rates();
        let d_m = self.cfg.d_latent;

        // 1. Raw continuous input & affine RMSNorm
        let e_t = x_features;
        let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();

        for i in 0..d_m {
            self.inf_x_norm[i] =
                self.norm_gamma.data[i] * (e_t[i] * inv_rms) + self.norm_beta.data[i];
        }

        // 2. Data-Dependent Projections
        self.w_delta.matvec(&self.inf_x_norm, &mut self.inf_delta);
        for i in 0..d_m {
            self.inf_delta[i] = softplus(self.inf_delta[i]);
        }

        self.w_b.matvec(&self.inf_x_norm, &mut self.inf_b);
        self.w_c.matvec(&self.inf_x_norm, &mut self.inf_c);

        self.forward_continuous_inference_projected(z_out);
    }

    /// Finish one inference row after its norm and data-dependent projections
    /// have been prepared. The recurrent and retrieval portion remains in
    /// token order so carries and final scratch values do not change.
    #[inline(always)]
    fn forward_continuous_inference_projected(&mut self, z_out: &mut [f32]) {
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();

        // 3. Multi-Channel SSM State Update (In-Place on Persistent State)
        for i in 0..d_m {
            let d_i = self.inf_delta[i];
            let mut y_i = 0.0f32;
            let row_off = i * d_s;
            for j in 0..d_s {
                let idx = row_off + j;
                let bar_a = (d_i * self.ssm_rates[idx]).exp();
                let bar_b = d_i * self.inf_b[j];
                let h_val = bar_a * self.h_persistent[idx] + bar_b * self.inf_x_norm[i];
                self.h_persistent[idx] = h_val;
                y_i += h_val * self.inf_c[j];
            }
            self.inf_y_ssm[i] = y_i;
        }

        // 4. Diffeomorphic Poincare memory retrieval.
        for r in 0..d_k {
            let row_x = &self.w_qx.data[r * d_m..(r + 1) * d_m];
            let row_h = &self.w_qh.data[r * d_m..(r + 1) * d_m];
            self.inf_q_euc[r] =
                dot_slice(row_x, &self.inf_x_norm) + dot_slice(row_h, &self.inf_y_ssm);
        }
        HyperbolicEpisodicBankV2::diffeomorphic_project(&self.inf_q_euc, &mut self.inf_q_pnc);
        self.memory.retrieve_soft_into(
            &self.inf_q_pnc,
            self.cfg.tau_mem,
            &mut self.inf_m_val,
            &mut self.inf_mem_weights,
        );
        self.w_gate.matvec(&self.inf_x_norm, &mut self.inf_g_mem);
        for i in 0..d_m {
            self.inf_g_mem[i] = sigmoid(self.inf_g_mem[i]);
        }
        self.w_proj.matvec(&self.inf_m_val, &mut self.inf_m_proj);

        // 5. Plastic Adapter.
        self.adapters[0]
            .down_proj
            .matvec(&self.inf_x_norm, &mut self.inf_ad_act[..rank]);
        for r in 0..rank {
            let h = self.inf_ad_act[r];
            self.inf_ad_act[r] = h * sigmoid(h);
        }
        self.adapters[0].total_up_matvec(&self.inf_ad_act[..rank], &mut self.inf_ad_out);

        // 6. Latent aggregation and SiLU MLP expansion.
        for i in 0..d_m {
            self.inf_z_raw[i] = (self.inf_y_ssm[i] * ssm_scale)
                + (self.inf_g_mem[i] * self.inf_m_proj[i])
                + self.inf_ad_out[i];
        }
        self.mlp_w1
            .matvec(&self.inf_z_raw, &mut self.inf_mlp_act[..d_mlp]);
        for i in 0..d_mlp {
            let h = self.inf_mlp_act[i];
            self.inf_mlp_act[i] = h * sigmoid(h);
        }
        self.mlp_w2
            .matvec(&self.inf_mlp_act[..d_mlp], &mut self.inf_mlp_out);
        for i in 0..d_m {
            self.inf_z_final[i] = self.inf_z_raw[i] + self.inf_mlp_out[i];
        }
        z_out.copy_from_slice(&self.inf_z_final);
    }

    /// Process raw continuous rows, detaching incoming carry at the chunk edge.
    pub fn forward_train_chunk(&mut self, inputs: &[f32], seq_len: usize) {
        self.forward_train_chunk_loop(inputs, seq_len, 0);
    }

    /// Forward one Ouro iteration into its preallocated activation slot.
    pub fn forward_train_chunk_loop(&mut self, inputs: &[f32], seq_len: usize, loop_index: usize) {
        assert!(seq_len > 0 && seq_len <= self.tape.max_l);
        assert_eq!(inputs.len(), seq_len * self.cfg.d_latent);
        let loop_l = loop_index * self.tape.max_l;
        let loop_state = loop_index * (self.tape.max_l + 1);
        assert!(loop_index < self.tape.x_raw.len() / (self.tape.max_l * self.cfg.d_latent));
        let x_raw_off = loop_l * self.cfg.d_latent;
        self.tape.x_raw[x_raw_off..x_raw_off + inputs.len()].copy_from_slice(inputs);
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let mem_cap = self.cfg.mem_capacity;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();

        self.swap_loop_carry(loop_index);
        self.tape.h_states[loop_state * d_m * d_s..(loop_state + 1) * d_m * d_s]
            .copy_from_slice(&self.h_persistent);

        self.refresh_ssm_rates();

        for t in 0..seq_len {
            // 1. Raw continuous input & affine RMSNorm
            let e_t = &self.tape.x_raw[(loop_l + t) * d_m..(loop_l + t + 1) * d_m];
            let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();
            self.tape.inv_rms[loop_l + t] = inv_rms;

            let xn_off = (loop_l + t) * d_m;
            for i in 0..d_m {
                self.tape.x_norm[xn_off + i] =
                    self.norm_gamma.data[i] * (e_t[i] * inv_rms) + self.norm_beta.data[i];
            }
            let x_n = &self.tape.x_norm[xn_off..xn_off + d_m];

            // 2. Data-Dependent Projections
            let del_off = (loop_l + t) * d_m;
            self.w_delta
                .matvec(x_n, &mut self.tape.delta_raw[del_off..del_off + d_m]);
            for i in 0..d_m {
                self.tape.delta[del_off + i] = softplus(self.tape.delta_raw[del_off + i]);
            }
            let delta = &self.tape.delta[del_off..del_off + d_m];

            let b_off = (loop_l + t) * d_s;
            self.w_b
                .matvec(x_n, &mut self.tape.b_proj[b_off..b_off + d_s]);
            let b_p = &self.tape.b_proj[b_off..b_off + d_s];

            let c_off = (loop_l + t) * d_s;
            self.w_c
                .matvec(x_n, &mut self.tape.c_proj[c_off..c_off + d_s]);
            let c_p = &self.tape.c_proj[c_off..c_off + d_s];

            // 3. Multi-Channel SSM Recurrent Scan
            let h_prev_off = (loop_state + t) * (d_m * d_s);
            let h_next_off = (loop_state + t + 1) * (d_m * d_s);
            let ssm_off = (loop_l + t) * (d_m * d_s);
            let y_off = (loop_l + t) * d_m;

            for i in 0..d_m {
                let d_i = delta[i];
                let mut y_i = 0.0f32;
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let bar_a = (d_i * self.ssm_rates[idx]).exp();
                    let bar_b = d_i * b_p[j];

                    self.tape.bar_a[ssm_off + idx] = bar_a;
                    self.tape.bar_b[ssm_off + idx] = bar_b;

                    let h_val = bar_a * self.tape.h_states[h_prev_off + idx] + bar_b * x_n[i];
                    self.tape.h_states[h_next_off + idx] = h_val;

                    y_i += h_val * c_p[j];
                }
                self.tape.y_ssm[y_off + i] = y_i;
            }
            let y_ssm = &self.tape.y_ssm[y_off..y_off + d_m];

            // 4. Diffeomorphic Poincaré Memory Retrieval
            let q_off = (loop_l + t) * d_k;
            for r in 0..d_k {
                let row_x = &self.w_qx.data[r * d_m..(r + 1) * d_m];
                let row_h = &self.w_qh.data[r * d_m..(r + 1) * d_m];
                self.tape.q_euc[q_off + r] = dot_slice(row_x, x_n) + dot_slice(row_h, y_ssm);
            }

            self.tape.q_norm[loop_l + t] = HyperbolicEpisodicBankV2::diffeomorphic_project(
                &self.tape.q_euc[q_off..q_off + d_k],
                &mut self.tape.q_poincare[q_off..q_off + d_k],
            );

            let m_off = (loop_l + t) * d_m;
            let mw_off = (loop_l + t) * mem_cap;
            self.memory.retrieve_soft_into(
                &self.tape.q_poincare[q_off..q_off + d_k],
                self.cfg.tau_mem,
                &mut self.tape.m_val[m_off..m_off + d_m],
                &mut self.tape.mem_weights[mw_off..mw_off + mem_cap],
            );

            self.w_gate
                .matvec(x_n, &mut self.tape.g_mem[m_off..m_off + d_m]);
            for i in 0..d_m {
                self.tape.g_mem[m_off + i] = sigmoid(self.tape.g_mem[m_off + i]);
            }

            self.w_proj
                .matvec(&self.tape.m_val[m_off..m_off + d_m], &mut self.buf_m_proj);
            for i in 0..d_m {
                self.tape.m_proj[m_off + i] = self.buf_m_proj[i];
                self.tape.m_inj[m_off + i] = self.tape.g_mem[m_off + i] * self.buf_m_proj[i];
            }

            // 5. Zero-Init Plastic Adapter
            let ad_off = (loop_l + t) * rank;
            self.adapters[0]
                .down_proj
                .matvec(x_n, &mut self.tape.adapter_hidden[ad_off..ad_off + rank]);
            for r in 0..rank {
                let h = self.tape.adapter_hidden[ad_off + r];
                self.tape.adapter_act[ad_off + r] = h * sigmoid(h);
            }
            self.adapters[0].total_up_matvec(
                &self.tape.adapter_act[ad_off..ad_off + rank],
                &mut self.buf_ad_out,
            );

            // 6. Latent Aggregation & SiLU MLP Expansion
            let z_off = (loop_l + t) * d_m;
            for i in 0..d_m {
                self.tape.z_raw[z_off + i] =
                    (y_ssm[i] * ssm_scale) + self.tape.m_inj[m_off + i] + self.buf_ad_out[i];
            }
            let z_raw = &self.tape.z_raw[z_off..z_off + d_m];

            let mlp_off = (loop_l + t) * d_mlp;
            self.mlp_w1
                .matvec(z_raw, &mut self.tape.mlp_hidden[mlp_off..mlp_off + d_mlp]);
            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                self.tape.mlp_act[mlp_off + i] = h * sigmoid(h);
            }
            self.mlp_w2.matvec(
                &self.tape.mlp_act[mlp_off..mlp_off + d_mlp],
                &mut self.tape.z_final[z_off..z_off + d_m],
            );
            for i in 0..d_m {
                self.tape.z_final[z_off + i] += z_raw[i];
            }
        }

        let last_h_off = (loop_state + seq_len) * (d_m * d_s);
        self.h_persistent
            .copy_from_slice(&self.tape.h_states[last_h_off..last_h_off + d_m * d_s]);
        self.swap_loop_carry(loop_index);
        if loop_index == 0 && self.loop_count() > 1 {
            self.copy_base_tape_to_loop(0);
        }
    }

    pub fn zero_gradients(&mut self) {
        self.norm_gamma.zero_grad();
        self.norm_beta.zero_grad();
        self.a_mat.zero_grad();
        self.w_delta.zero_grad();
        self.w_b.zero_grad();
        self.w_c.zero_grad();
        self.w_qx.zero_grad();
        self.w_qh.zero_grad();
        self.w_gate.zero_grad();
        self.w_proj.zero_grad();
        self.adapters[0].zero_grad();
        self.mlp_w1.zero_grad();
        self.mlp_w2.zero_grad();
    }

    /// Accumulate parameter gradients from arbitrary output adjoints and write
    /// the raw-input VJP. Incoming carry and stored memory are detached state.
    pub fn backward_chunk(
        &mut self,
        output_adjoints: &[f32],
        seq_len: usize,
        input_adjoints: &mut [f32],
    ) {
        self.backward_chunk_loop(output_adjoints, seq_len, input_adjoints, 0);
    }

    /// Backward one Ouro iteration into the shared parameter gradients.
    /// Temporal carry is detached at the chunk edge for each virtual pass;
    /// inter-pass feature adjoints are propagated by the outer residual loop.
    pub fn backward_chunk_loop(
        &mut self,
        output_adjoints: &[f32],
        seq_len: usize,
        input_adjoints: &mut [f32],
        loop_index: usize,
    ) {
        assert!(
            seq_len > 0 && seq_len <= self.cfg.chunk_len,
            "backward sequence length must be within tape capacity"
        );
        assert_eq!(output_adjoints.len(), seq_len * self.cfg.d_latent);
        assert_eq!(input_adjoints.len(), output_adjoints.len());
        let d_m = self.cfg.d_latent;
        let d_s = self.cfg.d_state;
        let d_k = self.cfg.d_mem_key;
        let d_mlp = d_m * 2;
        let rank = self.adapters[0].rank;
        let ssm_scale = 1.0 / (d_s as f32).sqrt();

        self.refresh_ssm_rates();
        self.grad_h_next.fill(0.0);
        if loop_index == 0 && self.loop_count() > 1 {
            self.copy_loop_tape_to_base(0);
        }
        let loop_l = loop_index * self.tape.max_l;
        let loop_state = loop_index * (self.tape.max_l + 1);
        assert!(loop_index < self.tape.x_raw.len() / (self.tape.max_l * d_m));

        // Reverse Time Loop across sequence chunk L
        for t in (0..seq_len).rev() {
            let z_off = (loop_l + t) * d_m;

            self.grad_z_final
                .copy_from_slice(&output_adjoints[t * d_m..(t + 1) * d_m]);

            // 2. SiLU MLP Backward
            let mlp_off = (loop_l + t) * d_mlp;
            self.mlp_w2
                .matvec_transpose(&self.grad_z_final, &mut self.buf_g_mlp_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_final[i];
                let row_off = i * d_mlp;
                for j in 0..d_mlp {
                    self.mlp_w2.grad[row_off + j] += gz_i * self.tape.mlp_act[mlp_off + j];
                }
            }

            for i in 0..d_mlp {
                let h = self.tape.mlp_hidden[mlp_off + i];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                self.buf_g_mlp_hidden[i] = self.buf_g_mlp_act[i] * silu_prime;
            }

            self.mlp_w1
                .matvec_transpose(&self.buf_g_mlp_hidden, &mut self.buf_g_zraw_mlp);

            for i in 0..d_mlp {
                let gh_i = self.buf_g_mlp_hidden[i];
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.mlp_w1.grad[row_off + j] += gh_i * self.tape.z_raw[z_off + j];
                }
            }

            for i in 0..d_m {
                self.grad_z_raw[i] = self.grad_z_final[i] + self.buf_g_zraw_mlp[i];
            }

            // 3. Adapter Backward
            let ad_off = (loop_l + t) * rank;
            self.adapters[0].total_up_matvec_transpose(&self.grad_z_raw, &mut self.buf_g_ad_act);

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let row_off = i * rank;
                for r in 0..rank {
                    self.adapters[0].up_proj.grad[row_off + r] +=
                        gz_i * self.tape.adapter_act[ad_off + r];
                }
            }

            for r in 0..rank {
                let h = self.tape.adapter_hidden[ad_off + r];
                let sig_h = sigmoid(h);
                let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
                self.buf_g_ad_down[r] = self.buf_g_ad_act[r] * silu_prime;
            }

            self.grad_x_norm.fill(0.0);
            for r in 0..rank {
                let gad_r = self.buf_g_ad_down[r];
                let row_off = r * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gad_r * self.adapters[0].down_proj.data[row_off + j];
                    self.adapters[0].down_proj.grad[row_off + j] +=
                        gad_r * self.tape.x_norm[(loop_l + t) * d_m + j];
                }
            }

            // 4. Memory Injection Backward
            let m_off = (loop_l + t) * d_m;
            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let g_mem = self.tape.g_mem[m_off + i];
                self.buf_g_m_proj_out[i] = gz_i * g_mem;

                let d_sig = g_mem * (1.0 - g_mem);
                let g_wgate_pre = gz_i * self.tape.m_proj[m_off + i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += g_wgate_pre * self.w_gate.data[row_off + j];
                    self.w_gate.grad[row_off + j] +=
                        g_wgate_pre * self.tape.x_norm[(loop_l + t) * d_m + j];
                }
            }

            self.w_proj
                .matvec_transpose(&self.buf_g_m_proj_out, &mut self.buf_g_m_val);
            for i in 0..d_m {
                let g_mp_i = self.buf_g_m_proj_out[i];
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.w_proj.grad[row_off + j] += g_mp_i * self.tape.m_val[m_off + j];
                }
            }

            // Retrieval is a full-bank softmax over -hyperbolic_distance/tau.
            // Bank keys/values are detached stored state; only the query path learns.
            self.g_query_pnc.fill(0.0);
            let q_off = (loop_l + t) * d_k;
            let q = &self.tape.q_poincare[q_off..q_off + d_k];
            let q_sq = HyperbolicEpisodicBankV2::squared_norm(q) as f64;
            for entry in 0..self.memory.count {
                let key_off = entry * d_k;
                let key = &self.memory.keys[key_off..key_off + d_k];
                let key_sq = self.memory.norm_sq[entry] as f64;
                let mut dot_g_value_minus_mean = 0.0f64;
                let value_off = entry * d_m;
                for j in 0..d_m {
                    dot_g_value_minus_mean += self.buf_g_m_val[j] as f64
                        * (self.memory.values[value_off + j] - self.tape.m_val[m_off + j]) as f64;
                }
                let g_score = self.tape.mem_weights[(loop_l + t) * self.cfg.mem_capacity + entry]
                    as f64
                    * dot_g_value_minus_mean;
                let mut sq = 0.0f64;
                for k in 0..d_k {
                    let diff = q[k] as f64 - key[k] as f64;
                    sq += diff * diff;
                }
                // The distance has a cusp at identical points.  We explicitly use
                // the symmetric subgradient zero there, avoiding division by zero.
                if sq > 0.0 {
                    let denom = (1.0 - q_sq) * (1.0 - key_sq);
                    assert!(denom > 0.0 && denom.is_finite());
                    let z = sq / denom;
                    let dd_dz = 1.0 / (z * (1.0 + z)).sqrt();
                    for k in 0..d_k {
                        let diff = q[k] as f64 - key[k] as f64;
                        let ddenom = -2.0 * q[k] as f64 * (1.0 - key_sq);
                        let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                        self.g_query_pnc[k] +=
                            (g_score * (-1.0 / self.cfg.tau_mem as f64) * dd_dz * dz) as f32;
                    }
                }
            }
            HyperbolicEpisodicBankV2::projection_adjoint(
                &self.tape.q_euc[q_off..q_off + d_k],
                &self.g_query_pnc,
                &mut self.g_query_euc,
            );
            self.g_y_ssm.fill(0.0);
            let xn = &self.tape.x_norm[(loop_l + t) * d_m..(loop_l + t + 1) * d_m];
            let y = &self.tape.y_ssm[(loop_l + t) * d_m..(loop_l + t + 1) * d_m];
            for r_i in 0..d_k {
                let gq = self.g_query_euc[r_i];
                let row = r_i * d_m;
                for j in 0..d_m {
                    self.w_qx.grad[row + j] += gq * xn[j];
                    self.w_qh.grad[row + j] += gq * y[j];
                    self.grad_x_norm[j] += gq * self.w_qx.data[row + j];
                    self.g_y_ssm[j] += gq * self.w_qh.data[row + j];
                }
            }

            // 5. Multi-Channel SSM Recurrence Backward & Temporal State Flow
            let ssm_off = (loop_l + t) * (d_m * d_s);
            let h_prev_off = (loop_state + t) * (d_m * d_s);
            let c_off = (loop_l + t) * d_s;
            let b_off = (loop_l + t) * d_s;
            let del_off = (loop_l + t) * d_m;

            self.buf_g_delta.fill(0.0);
            self.buf_g_b_proj.fill(0.0);
            self.buf_g_c_proj.fill(0.0);
            self.buf_g_h_prev.fill(0.0);

            for i in 0..d_m {
                let gz_i = self.grad_z_raw[i];
                let g_y_i = gz_i * ssm_scale + self.g_y_ssm[i];
                let d_i = self.tape.delta[del_off + i];
                let xn_i = self.tape.x_norm[(loop_l + t) * d_m + i];

                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let h_next = self.tape.h_states[(loop_state + t + 1) * (d_m * d_s) + idx];
                    let c_val = self.tape.c_proj[c_off + j];
                    let bar_a = self.tape.bar_a[ssm_off + idx];
                    let a_physical = self.ssm_rates[idx];
                    let b_val = self.tape.b_proj[b_off + j];

                    let g_h_total = g_y_i * c_val + self.grad_h_next[idx];

                    self.buf_g_c_proj[j] += g_y_i * h_next;
                    self.buf_g_h_prev[idx] += g_h_total * bar_a;

                    // dA/draw = -sigmoid(raw) for A=-softplus(raw).
                    self.a_mat.grad[idx] += g_h_total
                        * (d_i * bar_a)
                        * self.tape.h_states[h_prev_off + idx]
                        * self.ssm_rate_derivatives[idx];
                    self.buf_g_delta[i] += g_h_total
                        * (a_physical * bar_a * self.tape.h_states[h_prev_off + idx]
                            + b_val * xn_i);
                    self.buf_g_b_proj[j] += g_h_total * (d_i * xn_i);
                    self.grad_x_norm[i] += g_h_total * self.tape.bar_b[ssm_off + idx];
                }
            }

            self.grad_h_next.copy_from_slice(&self.buf_g_h_prev);

            for i in 0..d_m {
                let d_sig = sigmoid(self.tape.delta_raw[del_off + i]);
                let gd_i = self.buf_g_delta[i] * d_sig;
                let row_off = i * d_m;
                for j in 0..d_m {
                    self.grad_x_norm[j] += gd_i * self.w_delta.data[row_off + j];
                    self.w_delta.grad[row_off + j] +=
                        gd_i * self.tape.x_norm[(loop_l + t) * d_m + j];
                }
            }

            for j in 0..d_s {
                let gb_j = self.buf_g_b_proj[j];
                let gc_j = self.buf_g_c_proj[j];
                let row_off = j * d_m;
                for k in 0..d_m {
                    self.grad_x_norm[k] +=
                        gb_j * self.w_b.data[row_off + k] + gc_j * self.w_c.data[row_off + k];
                    self.w_b.grad[row_off + k] += gb_j * self.tape.x_norm[(loop_l + t) * d_m + k];
                    self.w_c.grad[row_off + k] += gc_j * self.tape.x_norm[(loop_l + t) * d_m + k];
                }
            }

            // 6. Affine RMSNorm backward to gamma, beta, and raw input rows
            let inv_rms = self.tape.inv_rms[loop_l + t];
            let e_t = &self.tape.x_raw[(loop_l + t) * d_m..(loop_l + t + 1) * d_m];

            let mut dot_gx_e = 0.0f32;
            for i in 0..d_m {
                let gx_i = self.grad_x_norm[i];
                self.norm_beta.grad[i] += gx_i;
                self.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);

                let g_unnorm = gx_i * self.norm_gamma.data[i];
                dot_gx_e += g_unnorm * e_t[i];
            }

            for i in 0..d_m {
                let g_unnorm = self.grad_x_norm[i] * self.norm_gamma.data[i];
                let g_e_i =
                    inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / (d_m as f32)));
                input_adjoints[t * d_m + i] = g_e_i;
            }
        }
    }

    pub fn apply_adamw(&mut self, lr: f32, beta1: f32, beta2: f32, wd: f32, eps: f32, step: usize) {
        self.norm_gamma.step_adamw(lr, beta1, beta2, 0.0, eps, step);
        self.norm_beta.step_adamw(lr, beta1, beta2, 0.0, eps, step);
        self.a_mat.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_delta.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_b.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_c.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_qx.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_qh.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_gate.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.w_proj.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.adapters[0].step_adamw(lr, beta1, beta2, wd, eps, step);
        self.mlp_w1.step_adamw(lr, beta1, beta2, wd, eps, step);
        self.mlp_w2.step_adamw(lr, beta1, beta2, wd, eps, step);
    }
    pub fn ema_consolidate_plasticity(&mut self) {
        self.adapters[0].consolidate(self.cfg.ema_alpha);
    }

    /// Replay one detached episodic value through the continuous block and
    /// update only the fast plastic adapter.  Main PSSA weights, optimizer
    /// moments, and the episodic bank remain untouched.  The adapter update is
    /// a local Hebbian-style VJP: the stored value is the target and the
    /// adapter activation supplies the presynaptic signal.
    pub(crate) fn dream_replay_value(&mut self, value: &[f32], output: &mut [f32]) {
        assert_eq!(value.len(), self.cfg.d_latent);
        assert_eq!(output.len(), self.cfg.d_latent);
        self.forward_continuous_inference(value, output);
        let target_norm = value
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let rate = 0.02 / (1.0 + target_norm.min(f32::MAX as f64) as f32);
        let rank = self.adapters[0].rank;
        for (i, &target) in value.iter().enumerate() {
            let error = (target - self.inf_ad_out[i]).clamp(-4.0, 4.0);
            let row = i * rank;
            for r in 0..rank {
                self.adapters[0].up_proj.data[row + r] += rate * error * self.inf_ad_act[r];
            }
        }
    }

    pub fn parameter_count(&self) -> usize {
        [
            &self.a_mat,
            &self.w_delta,
            &self.w_b,
            &self.w_c,
            &self.w_qx,
            &self.w_qh,
            &self.w_gate,
            &self.w_proj,
            &self.mlp_w1,
            &self.mlp_w2,
        ]
        .iter()
        .map(|p| p.data.len())
        .sum::<usize>()
            + self.norm_gamma.data.len()
            + self.norm_beta.data.len()
            + self
                .adapters
                .iter()
                .map(|a| a.down_proj.data.len() + a.up_proj.data.len())
                .sum::<usize>()
    }
}

/// Bounded, call-local prefill workspace. Only the first block's embedding
/// inputs are known ahead of time; deeper blocks and Ouro loops stay on the
/// token-major decode path. Nothing is published to runtime scratch ahead of
/// the corresponding token's cancellation check.
struct PrefillProjections {
    x_norm: Vec<f32>,
    delta: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

impl PrefillProjections {
    const TILE: usize = 32;

    fn new(d: usize, s: usize, tokens: usize) -> Self {
        Self {
            x_norm: vec![0.0; tokens * d],
            delta: vec![0.0; tokens * d],
            b: vec![0.0; tokens * s],
            c: vec![0.0; tokens * s],
        }
    }

    fn prepare(&mut self, model: &PSSALayerV2, ids: &[usize]) {
        let block = &model.block;
        let (d, s) = (model.cfg.d_latent, model.cfg.d_state);
        for (t, &id) in ids.iter().enumerate() {
            let input = &model.embed_w.data[id * d..(id + 1) * d];
            // Match forward_continuous_inference, including scalar RMS sum.
            let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / d as f32 + 1e-5).sqrt();
            for i in 0..d {
                self.x_norm[t * d + i] = block.norm_gamma.data[i] * (input[i] * inv_rms)
                    + block.norm_beta.data[i];
            }
        }
        let n = ids.len();
        let norm = &self.x_norm[..n * d];
        block.w_delta.matvec_batch(norm, &mut self.delta[..n * d]);
        block.w_b.matvec_batch(norm, &mut self.b[..n * s]);
        block.w_c.matvec_batch(norm, &mut self.c[..n * s]);
        for delta in &mut self.delta[..n * d] {
            *delta = softplus(*delta);
        }
    }

    fn publish(&self, block: &mut PSSAContinuousBlockV2, t: usize) {
        let (d, s) = (block.cfg.d_latent, block.cfg.d_state);
        block.inf_x_norm.copy_from_slice(&self.x_norm[t * d..(t + 1) * d]);
        block.inf_delta.copy_from_slice(&self.delta[t * d..(t + 1) * d]);
        block.inf_b.copy_from_slice(&self.b[t * s..(t + 1) * s]);
        block.inf_c.copy_from_slice(&self.c[t * s..(t + 1) * s]);
    }
}

#[derive(Clone, Debug)]
struct DreamSequence {
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
}

/// Shared vocabulary endpoints around independent continuous blocks. Layer zero
/// has no outer residual; subsequent blocks use h + F(h)/sqrt(depth).
/// Deref preserves the historical depth-one field API; new code should use
/// `block` explicitly when borrowing several fields at once.
pub struct PSSALayerV2 {
    pub cfg: PSSAConfigV2,
    pub step_counter: usize,
    pub device: Device,
    /// Runtime-only lazy worker handoff for allocation-free parallel scans.
    pub scan_executor: crate::scan_executor::ScanExecutor,
    pub rng: SimpleRng,
    pub vocabulary: Vec<String>,
    pub tokenizer_json: Option<String>,
    pub lr_schedule_total_updates: Option<usize>,
    pub lr_schedule_warmup_steps: Option<usize>,
    pub embed_w: ParamMatrix,
    pub unembed_w: ParamMatrix,
    pub embed_row_marks: Vec<usize>,
    pub block: PSSAContinuousBlockV2,
    pub extra_blocks: Vec<PSSAContinuousBlockV2>,
    pub residual_scales: Vec<f32>,
    pub continuous_inputs: Vec<f32>,
    pub output_adjoints: Vec<f32>,
    pub input_adjoints: Vec<f32>,
    pub residual_block_adjoints: Vec<f32>,
    pub residual_input_adjoints: Vec<f32>,
    /// Loss-normalized VJPs at embeddings, first block output, residual outputs.
    pub boundary_adjoints: Vec<Vec<f32>>,
    pub layer_activations: Vec<Vec<f32>>,
    pub inf_features: Vec<f32>,
    pub inf_block_out: Vec<f32>,
    /// Runtime-only supervised replay records. They intentionally do not enter
    /// checkpoints: dream controls and their ephemeral rehearsal cache must be
    /// supplied again when resuming a run.
    dream_sequences: Vec<DreamSequence>,
}

impl std::ops::Deref for PSSALayerV2 {
    type Target = PSSAContinuousBlockV2;
    fn deref(&self) -> &Self::Target {
        &self.block
    }
}
impl std::ops::DerefMut for PSSALayerV2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.block
    }
}

impl PSSAConfigV2 {
    /// Parameter-matched four-layer comparison against width 256 / depth one.
    /// Vocabulary is the actual tokenizer size, not just its requested ceiling.
    pub fn stacked_depth_test() -> Self {
        Self {
            depth: 4,
            d_latent: 166,
            d_vocab: 2048,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            ..Self::default()
        }
    }
}

impl PSSALayerV2 {
    pub const MAX_DEPTH: usize = 32;
    pub const MAX_LOOPS: usize = 32;

    pub fn new(cfg: PSSAConfigV2, seed: u64) -> Self {
        Self::new_with_device_and_loops(cfg, seed, Device::Cpu, 1)
    }
    pub fn new_with_depth(mut cfg: PSSAConfigV2, seed: u64, depth: usize) -> Self {
        cfg.depth = depth;
        Self::new(cfg, seed)
    }
    pub fn new_with_depth_and_loops(
        mut cfg: PSSAConfigV2,
        seed: u64,
        depth: usize,
        loops: usize,
    ) -> Self {
        cfg.depth = depth;
        Self::new_with_device_and_loops(cfg, seed, Device::Cpu, loops)
    }
    pub fn new_with_device(cfg: PSSAConfigV2, seed: u64, device: Device) -> Self {
        Self::new_with_device_and_loops(cfg, seed, device, 1)
    }
    fn new_with_device_and_loops(
        cfg: PSSAConfigV2,
        seed: u64,
        device: Device,
        loops: usize,
    ) -> Self {
        assert!((1..=Self::MAX_LOOPS).contains(&loops));
        cfg.validate();
        Self::validate_loops_config(&cfg, loops).expect("invalid model allocation");
        let mut rng = SimpleRng::new(seed);
        let (v, d, l, depth) = (cfg.d_vocab, cfg.d_latent, cfg.chunk_len, cfg.depth);
        // Preserve main's exact original embedding/block/head random draw order.
        // Extra blocks are initialized only after both shared endpoints.
        let embed_w = ParamMatrix::random_xavier(v, d, &mut rng);
        let block = if loops == 1 {
            PSSAContinuousBlockV2::new_with_rng((&cfg).into(), &mut rng, v)
        } else {
            PSSAContinuousBlockV2::new_with_rng_and_loops((&cfg).into(), &mut rng, v, loops)
        };
        let unembed_w = ParamMatrix::random_xavier(v, d, &mut rng);
        let extra_blocks = (1..depth)
            .map(|_| {
                PSSAContinuousBlockV2::new_with_rng_and_loops((&cfg).into(), &mut rng, 0, loops)
            })
            .collect();
        Self {
            cfg,
            step_counter: 0,
            device,
            scan_executor: crate::scan_executor::ScanExecutor::with_loops(loops),
            rng,
            vocabulary: Vec::new(),
            tokenizer_json: None,
            lr_schedule_total_updates: None,
            lr_schedule_warmup_steps: None,
            embed_w,
            unembed_w,
            embed_row_marks: vec![0; v],
            block,
            extra_blocks,
            residual_scales: vec![1.0 / (depth as f32).sqrt(); depth - 1],
            continuous_inputs: vec![0.0; l * d],
            output_adjoints: vec![0.0; l * d],
            input_adjoints: vec![0.0; l * d],
            residual_block_adjoints: vec![0.0; l * d],
            residual_input_adjoints: vec![0.0; l * d],
            boundary_adjoints: (0..=depth).map(|_| vec![0.0; l * d]).collect(),
            layer_activations: (1..depth).map(|_| vec![0.0; l * d]).collect(),
            inf_features: vec![0.0; d],
            inf_block_out: vec![0.0; d],
            dream_sequences: Vec::new(),
        }
    }
    pub fn depth(&self) -> usize {
        self.extra_blocks.len() + 1
    }

    /// Runtime-only Ouro count; it is deliberately absent from checkpoints.
    pub fn loops(&self) -> usize {
        self.scan_executor.loops
    }

    /// Check the extra runtime activation/carry storage against the existing
    /// allocation cap without adding anything to serialized configuration.
    pub fn validate_loops_config(cfg: &PSSAConfigV2, loops: usize) -> Result<(), String> {
        if !(1..=Self::MAX_LOOPS).contains(&loops) {
            return Err(
                "--loops must be between 1 and 32; use --loops 1 for the original model".into(),
            );
        }
        crate::checkpoint::validate_model_config(cfg)?;
        let base = crate::checkpoint::allocation_bytes(cfg).map_err(|e| e.to_string())? as u128;
        let (d, s, k, l, mem) = (
            cfg.d_latent as u128,
            cfg.d_state as u128,
            cfg.d_mem_key as u128,
            cfg.chunk_len as u128,
            cfg.mem_capacity as u128,
        );
        // All float activation arrays except the one shared vocabulary head.
        // Each extra pass also has an initial state row and a persistent carry.
        let per_pass = l * (15 * d + 2 * s + 2 * k + mem + 34 + 3 * d * s) + 2 * d * s;
        let bytes = base + 4 * cfg.depth as u128 * (loops - 1) as u128 * per_pass;
        if bytes > crate::checkpoint::MAX_LOAD_ALLOCATION_BYTES as u128 {
            return Err(format!(
                "Ouro runtime needs {bytes} bytes, over the model byte cap; reduce --loops, --chunk, or model dimensions"
            ));
        }
        Ok(())
    }

    fn resize_loop_tape(&mut self, loops: usize) {
        assert!((1..=Self::MAX_LOOPS).contains(&loops));
        let l = self.block.tape.max_l;
        let d = self.cfg.d_latent;
        let s = self.cfg.d_state;
        let k = self.cfg.d_mem_key;
        let mem = self.cfg.mem_capacity;
        let rank = self.block.adapters[0].rank;
        self.block.tape =
            ChunkActivationTape::new_with_loops(l, self.cfg.d_vocab, d, s, k, mem, rank, loops);
        for block in &mut self.extra_blocks {
            block.tape = ChunkActivationTape::new_with_loops(l, 0, d, s, k, mem, rank, loops);
        }
        self.scan_executor = crate::scan_executor::ScanExecutor::with_loops(loops);
    }

    /// Apply a runtime loop count after checkpoint loading without altering
    /// persistent model metadata or serialized bytes.
    pub fn set_loops(&mut self, loops: usize) -> Result<(), String> {
        if !(1..=Self::MAX_LOOPS).contains(&loops) {
            return Err(
                "--loops must be between 1 and 32; use --loops 1 for the original model".into(),
            );
        }
        if self.loops() != loops {
            let mut cfg = self.cfg.clone();
            cfg.chunk_len = self.block.tape.max_l;
            Self::validate_loops_config(&cfg, loops)?;
            self.resize_loop_tape(loops);
        }
        Ok(())
    }

    /// Grow all memory banks at a chunk boundary. No checkpoint format change:
    /// subsequent saves use the ordinary capacity field with larger arrays.
    /// Parameters, Adam moments, RNG, schedule and recurrent carries are retained.
    pub fn grow_memory(&mut self, capacity: usize) -> Result<(), String> {
        if self.device.is_gpu() {
            return Err("grow memory on the CPU checkpoint before selecting a GPU backend".into());
        }
        if capacity < self.cfg.mem_capacity {
            return Err("memory growth cannot shrink the bank".into());
        }
        if capacity == self.cfg.mem_capacity {
            return Ok(());
        }
        let mut cfg = self.cfg.clone();
        cfg.mem_capacity = capacity;
        cfg.chunk_len = self.block.tape.max_l;
        Self::validate_loops_config(&cfg, self.loops())?;
        for block in std::iter::once(&mut self.block).chain(&mut self.extra_blocks) {
            let rows = block.tape.mem_weights.len() / block.cfg.mem_capacity;
            block.memory.grow(capacity)?;
            block.tape.mem_weights = vec![0.0; rows * capacity];
            block.inf_mem_weights = vec![0.0; capacity];
            block.cfg.mem_capacity = capacity;
        }
        self.cfg.mem_capacity = capacity;
        Ok(())
    }

    /// Exact nearest-slot prototype is CPU-only; selection is runtime-only.
    pub fn set_memory_top_k(&mut self, top_k: Option<usize>) -> Result<(), String> {
        if top_k.is_some() && self.device.is_gpu() {
            return Err("memory top-k currently requires the CPU backend".into());
        }
        if top_k == Some(0) {
            return Err("memory top-k must be positive".into());
        }
        for block in std::iter::once(&mut self.block).chain(&mut self.extra_blocks) {
            block.memory.set_top_k(top_k)?;
        }
        Ok(())
    }

    pub fn reset_recurrent_state(&mut self) {
        self.block.reset_recurrent_state();
        for b in &mut self.extra_blocks {
            b.reset_recurrent_state();
        }
    }

    /// Number of detached recurrent carry scalars retained between chunks.
    /// Ouro passes have independent temporal carries even though their weights
    /// are shared, so packed sequence replay must preserve every pass.
    pub(crate) fn recurrent_state_len(&self) -> usize {
        self.depth() * self.loops() * self.cfg.d_latent * self.cfg.d_state
    }

    pub(crate) fn copy_recurrent_state_to(&self, out: &mut [f32]) {
        let hs = self.cfg.d_latent * self.cfg.d_state;
        assert_eq!(out.len(), self.recurrent_state_len());
        for (block, dst) in std::iter::once(&self.block)
            .chain(&self.extra_blocks)
            .zip(out.chunks_exact_mut(self.loops() * hs))
        {
            dst[..hs].copy_from_slice(&block.h_persistent);
            if self.loops() > 1 {
                let start = block.loop_carry_start();
                dst[hs..]
                    .copy_from_slice(&block.tape.h_states[start..start + (self.loops() - 1) * hs]);
            }
        }
    }

    pub(crate) fn copy_recurrent_state_from(&mut self, state: &[f32]) {
        let hs = self.cfg.d_latent * self.cfg.d_state;
        assert_eq!(state.len(), self.recurrent_state_len());
        let loops = self.loops();
        for (block, src) in std::iter::once(&mut self.block)
            .chain(&mut self.extra_blocks)
            .zip(state.chunks_exact(loops * hs))
        {
            block.h_persistent.copy_from_slice(&src[..hs]);
            if loops > 1 {
                let start = block.loop_carry_start();
                block.tape.h_states[start..start + (loops - 1) * hs].copy_from_slice(&src[hs..]);
            }
        }
    }

    fn forward_inference_features(
        &mut self,
        x_id: usize,
        projected: Option<(&PrefillProjections, usize)>,
    ) {
        assert!(x_id < self.cfg.d_vocab, "token ID must be in vocabulary");
        let d = self.cfg.d_latent;
        let loops = self.loops();
        if loops == 1 {
            if let Some((tile, row)) = projected {
                self.block.refresh_ssm_rates();
                tile.publish(&mut self.block, row);
                self.block.forward_continuous_inference_projected(&mut self.inf_features);
            } else {
                self.block.forward_continuous_inference(
                    &self.embed_w.data[x_id * d..(x_id + 1) * d],
                    &mut self.inf_features,
                );
            }
            for (b, &scale) in self.extra_blocks.iter_mut().zip(&self.residual_scales) {
                b.forward_continuous_inference(&self.inf_features, &mut self.inf_block_out);
                for i in 0..d {
                    self.inf_features[i] += scale * self.inf_block_out[i];
                }
            }
        } else {
            self.inf_features
                .copy_from_slice(&self.embed_w.data[x_id * d..(x_id + 1) * d]);
            let loop_scale = 1.0 / loops as f32;
            for loop_index in 0..loops {
                self.block.forward_inference_loop(
                    &self.inf_features,
                    &mut self.inf_block_out,
                    loop_index,
                );
                for i in 0..d {
                    self.inf_features[i] += loop_scale * self.inf_block_out[i];
                }
            }
            // Destructure the independent runtime buffers before borrowing a
            // block mutably; this keeps inference allocation-free while the
            // depth residual input remains in the unused prefix of the
            // training workspace.
            let PSSALayerV2 {
                extra_blocks,
                residual_scales,
                continuous_inputs,
                inf_features,
                inf_block_out,
                ..
            } = self;
            for (b, &scale) in extra_blocks.iter_mut().zip(residual_scales.iter()) {
                continuous_inputs[..d].copy_from_slice(inf_features);
                for loop_index in 0..loops {
                    b.forward_inference_loop(inf_features, inf_block_out, loop_index);
                    for i in 0..d {
                        inf_features[i] += loop_scale * inf_block_out[i];
                    }
                }
                for i in 0..d {
                    inf_features[i] = continuous_inputs[i] + scale * inf_features[i];
                }
            }
        }
    }

    fn inference_logits(&self, logits_out: &mut [f32]) {
        self.unembed_w.matvec(&self.inf_features, logits_out);
        let logit_scale = 1.0 / (self.cfg.d_latent as f32).sqrt();
        for logit in logits_out {
            *logit *= logit_scale;
        }
    }

    pub fn forward_inference(&mut self, x_id: usize, logits_out: &mut [f32]) {
        assert_eq!(logits_out.len(), self.cfg.d_vocab);
        self.forward_inference_features(x_id, None);
        self.inference_logits(logits_out);
    }

    /// Opt-in CPU prompt prefill, continuing the current recurrent state.
    /// Returns false on cancellation, checked before each token just like
    /// decode. Only the final token's head is computed; pass None for a zero
    /// generation budget. Logits are untouched on cancellation or invalid IDs.
    /// Single-loop models batch the first block's independent projections in
    /// 32-token tiles. Depth/loop/carry/retrieval ordering is never changed.
    pub fn prefill_inference<C: Fn() -> bool>(
        &mut self,
        token_ids: &[usize],
        logits_out: Option<&mut [f32]>,
        cancelled: C,
    ) -> Result<bool, String> {
        if token_ids.is_empty() {
            return Err("prefill requires at least one token".into());
        }
        if let Some(logits) = logits_out.as_ref() {
            assert_eq!(logits.len(), self.cfg.d_vocab);
        }
        let mut projections = None;
        for (index, &id) in token_ids.iter().enumerate() {
            if cancelled() {
                return Ok(false);
            }
            if id >= self.cfg.d_vocab {
                return Err(format!("prompt ID {id} outside model vocabulary"));
            }
            let row = index % PrefillProjections::TILE;
            if self.loops() == 1 && token_ids.len() > 1 && row == 0 {
                let end = (index + PrefillProjections::TILE).min(token_ids.len());
                // Do not validate a later ID before advancing earlier tokens.
                let valid = token_ids[index..end]
                    .iter()
                    .take_while(|&&id| id < self.cfg.d_vocab)
                    .count();
                let tile = projections.get_or_insert_with(|| PrefillProjections::new(
                    self.cfg.d_latent,
                    self.cfg.d_state,
                    token_ids.len().min(PrefillProjections::TILE),
                ));
                tile.prepare(self, &token_ids[index..index + valid]);
            }
            self.forward_inference_features(id, projections.as_ref().map(|tile| (tile, row)));
        }
        if let Some(logits) = logits_out {
            self.inference_logits(logits);
        }
        Ok(true)
    }

    /// Host-only forward used to turn an episodic latent seed into generation
    /// logits. It mirrors `forward_inference` without consuming an embedding
    /// row, so GPU training can fall back explicitly without changing the
    /// regular inference/training dispatch path.
    fn forward_dream_input(&mut self, input: &[f32], logits_out: &mut [f32]) {
        assert_eq!(input.len(), self.cfg.d_latent);
        assert_eq!(logits_out.len(), self.cfg.d_vocab);
        let d = self.cfg.d_latent;
        let loops = self.loops();
        if loops == 1 {
            self.block
                .forward_continuous_inference(input, &mut self.inf_features);
            for (b, &scale) in self.extra_blocks.iter_mut().zip(&self.residual_scales) {
                b.forward_continuous_inference(&self.inf_features, &mut self.inf_block_out);
                for i in 0..d {
                    self.inf_features[i] += scale * self.inf_block_out[i];
                }
            }
        } else {
            self.inf_features.copy_from_slice(input);
            let loop_scale = 1.0 / loops as f32;
            for loop_index in 0..loops {
                self.block.forward_inference_loop(
                    &self.inf_features,
                    &mut self.inf_block_out,
                    loop_index,
                );
                for i in 0..d {
                    self.inf_features[i] += loop_scale * self.inf_block_out[i];
                }
            }
            let PSSALayerV2 {
                extra_blocks,
                residual_scales,
                continuous_inputs,
                inf_features,
                inf_block_out,
                ..
            } = self;
            for (b, &scale) in extra_blocks.iter_mut().zip(residual_scales.iter()) {
                continuous_inputs[..d].copy_from_slice(inf_features);
                for loop_index in 0..loops {
                    b.forward_inference_loop(inf_features, inf_block_out, loop_index);
                    for i in 0..d {
                        inf_features[i] += loop_scale * inf_block_out[i];
                    }
                }
                for i in 0..d {
                    inf_features[i] = continuous_inputs[i] + scale * inf_features[i];
                }
            }
        }
        self.unembed_w.matvec(&self.inf_features, logits_out);
        let logit_scale = 1.0 / (d as f32).sqrt();
        for logit in logits_out {
            *logit *= logit_scale;
        }
    }

    /// Forward one weight-shared loop through the batched CPU/GPU stages.
    /// The stage implementation historically addresses the zero-offset tape;
    /// loop activations are copied into and out of that workspace so the
    /// optimized path can be reused without changing checkpoint layout.
    pub(crate) fn forward_train_chunk_batched_loop(&mut self, seq_len: usize, loop_index: usize) {
        assert!(self.loops() > 1);
        assert!(seq_len > 0 && seq_len <= self.cfg.chunk_len);
        let d = self.cfg.d_latent;
        let n = seq_len * d;
        assert!(loop_index < self.loops());

        self.block.swap_loop_carry(loop_index);
        self.block.tape.x_raw[..n].copy_from_slice(&self.continuous_inputs[..n]);
        let state_width = d * self.cfg.d_state;
        self.block.tape.h_states[..state_width].copy_from_slice(&self.block.h_persistent);

        crate::gpu_batch::stage_input_norm_block(&mut self.block, seq_len);
        crate::gpu_batch::stage_projections(self, seq_len);
        crate::gpu_batch::stage_ssm_scan(self, seq_len);
        crate::gpu_batch::stage_memory(self, seq_len);
        crate::gpu_batch::stage_adapter(self, seq_len);
        crate::gpu_batch::stage_mlp(self, seq_len);

        let last_h_off = seq_len * state_width;
        self.block
            .h_persistent
            .copy_from_slice(&self.block.tape.h_states[last_h_off..last_h_off + state_width]);
        self.block.copy_base_tape_to_loop(loop_index);
        self.block.swap_loop_carry(loop_index);
    }

    /// Backward one weight-shared loop through the batched stages. Each pass
    /// contributes to the shared parameter gradients, while the raw RMSNorm
    /// adjoint is returned for the outer residual chain.
    pub(crate) fn backward_train_chunk_batched_loop(&mut self, seq_len: usize, loop_index: usize) {
        assert!(self.loops() > 1);
        assert!(seq_len > 0 && seq_len <= self.cfg.chunk_len);
        let d = self.cfg.d_latent;
        let n = seq_len * d;
        assert!(loop_index < self.loops());

        self.block.copy_loop_tape_to_base(loop_index);
        self.block.bwd_g_zfinal[..n].copy_from_slice(&self.input_adjoints[..n]);
        crate::gpu_batch::bwd_stage_mlp(self, seq_len);
        crate::gpu_batch::bwd_stage_adapter(self, seq_len);
        crate::gpu_batch::bwd_stage_adapter_down(self, seq_len);
        crate::gpu_batch::bwd_stage_memory(self, seq_len);
        // The outer PSSA backward owns the embedding VJP. Every loop therefore
        // stops at its continuous input, including loop zero.
        crate::gpu_batch::bwd_stage_ssm_with_input(self, seq_len, false);
        crate::gpu_batch::bwd_stage_input_norm(
            &self.block,
            seq_len,
            &mut self.residual_input_adjoints[..n],
        );
    }

    pub fn forward_train_chunk(&mut self, token_ids: &[usize], target_ids: &[usize]) -> f32 {
        assert!(!token_ids.is_empty(), "training chunk must be nonempty");
        assert_eq!(
            token_ids.len(),
            target_ids.len(),
            "token and target counts must match"
        );
        assert!(
            token_ids.len() <= self.cfg.chunk_len,
            "training chunk length exceeds configured tape capacity"
        );
        assert!(
            token_ids
                .iter()
                .chain(target_ids)
                .all(|&id| id < self.cfg.d_vocab),
            "token IDs must be in vocabulary"
        );
        // SequenceBatch replays through this reference entry point. On a GPU,
        // use the same staged path as the single-lane trainer; the CPU path
        // remains the historical scalar implementation.
        if self.loops() == 1 && self.device.is_gpu() {
            return crate::gpu_batch::forward_train_chunk_batched(self, token_ids, target_ids);
        }
        let seq_len = token_ids.len();
        let (d, v) = (self.cfg.d_latent, self.cfg.d_vocab);
        let n = seq_len * d;
        let loops = self.loops();
        let gpu = self.device.gpu();
        self.block.tape.x_ids[..seq_len].copy_from_slice(token_ids);
        self.block.tape.target_ids[..seq_len].copy_from_slice(target_ids);
        for (t, &id) in token_ids.iter().enumerate() {
            self.continuous_inputs[t * d..(t + 1) * d]
                .copy_from_slice(&self.embed_w.data[id * d..(id + 1) * d]);
        }
        if loops == 1 {
            self.block
                .forward_train_chunk(&self.continuous_inputs[..n], seq_len);
            for layer in 0..self.extra_blocks.len() {
                let (prior, after) = self.layer_activations.split_at_mut(layer);
                let previous = if layer == 0 {
                    &self.block.tape.z_final[..n]
                } else {
                    &prior[layer - 1][..n]
                };
                let b = &mut self.extra_blocks[layer];
                b.forward_train_chunk(previous, seq_len);
                let scale = self.residual_scales[layer];
                for i in 0..n {
                    after[0][i] = previous[i] + scale * b.tape.z_final[i];
                }
            }
        } else {
            let loop_scale = 1.0 / loops as f32;
            for loop_index in 0..loops {
                self.forward_train_chunk_batched_loop(seq_len, loop_index);
                let off = loop_index * self.block.tape.max_l * d;
                for i in 0..n {
                    self.continuous_inputs[i] += loop_scale * self.block.tape.z_final[off + i];
                }
            }
            for layer in 0..self.extra_blocks.len() {
                // Preserve the input to this independently weighted depth block
                // while the shared loop reuses `continuous_inputs` as its work row.
                self.residual_input_adjoints[..n].copy_from_slice(&self.continuous_inputs[..n]);
                for loop_index in 0..loops {
                    let b = &mut self.extra_blocks[layer];
                    if let Some(gpu) = gpu.as_ref() {
                        b.swap_loop_carry(loop_index);
                        b.tape.x_raw[..n].copy_from_slice(&self.continuous_inputs[..n]);
                        let state_width = d * self.cfg.d_state;
                        b.tape.h_states[..state_width].copy_from_slice(&b.h_persistent);
                        crate::gpu_batch::forward_continuous_block_batched(b, seq_len, gpu);
                        let last_h_off = seq_len * state_width;
                        b.h_persistent.copy_from_slice(
                            &b.tape.h_states[last_h_off..last_h_off + state_width],
                        );
                        b.copy_base_tape_to_loop(loop_index);
                        b.swap_loop_carry(loop_index);
                    } else {
                        b.forward_train_chunk_loop(
                            &self.continuous_inputs[..n],
                            seq_len,
                            loop_index,
                        );
                    }
                    let off = loop_index * b.tape.max_l * d;
                    for i in 0..n {
                        self.continuous_inputs[i] += loop_scale * b.tape.z_final[off + i];
                    }
                }
                let scale = self.residual_scales[layer];
                for i in 0..n {
                    self.continuous_inputs[i] =
                        self.residual_input_adjoints[i] + scale * self.continuous_inputs[i];
                }
                self.layer_activations[layer][..n].copy_from_slice(&self.continuous_inputs[..n]);
            }
        }
        let tape = &mut self.block.tape;
        let final_z: &[f32] = if loops == 1 {
            self.layer_activations.last().unwrap_or(&tape.z_final)
        } else if let Some(last) = self.layer_activations.last() {
            &last[..n]
        } else {
            &self.continuous_inputs[..n]
        };
        if let Some(gpu) = gpu.as_ref() {
            gpu.dispatch_gemm_into(
                final_z,
                &self.unembed_w.data,
                seq_len,
                v,
                d,
                1,
                &mut tape.logits[..seq_len * v],
            )
            .expect("validated model GEMM dimensions");
        }
        let logit_scale = 1.0 / (d as f32).sqrt();
        let mut total_loss = 0.0f32;
        for t in 0..seq_len {
            let off = t * v;
            if gpu.is_none() {
                self.unembed_w
                    .matvec(&final_z[t * d..(t + 1) * d], &mut tape.logits[off..off + v]);
            }
            for i in 0..v {
                tape.logits[off + i] *= logit_scale;
            }
            let mut max_l = f32::NEG_INFINITY;
            for i in 0..v {
                if tape.logits[off + i] > max_l {
                    max_l = tape.logits[off + i];
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..v {
                let exp_l = (tape.logits[off + i] - max_l).exp();
                tape.probs[off + i] = exp_l;
                sum_exp += exp_l;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-8);
            for i in 0..v {
                tape.probs[off + i] *= inv_sum;
            }
            let loss = (max_l - tape.logits[off + target_ids[t]]) + sum_exp.ln();
            tape.losses[t] = loss;
            total_loss += loss;
        }
        total_loss / seq_len as f32
    }

    pub fn zero_gradients(&mut self) {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = self.device.clone() {
            if ctx.safeguarded_parameters_finite().is_some() {
                let d = self.cfg.d_latent;
                for (row, mark) in self.embed_row_marks.iter_mut().enumerate() {
                    if *mark != 0 {
                        self.embed_w.grad[row * d..(row + 1) * d].fill(0.0);
                        *mark = 0;
                    }
                }
                ctx.zero_safeguarded_gradients(&mut self.adam_tensors())
                    .expect("CUDA gradient clearing failed");
                return;
            }
        }
        self.embed_w.zero_grad();
        self.block.zero_gradients();
        for b in &mut self.extra_blocks {
            b.zero_gradients();
        }
        self.unembed_w.zero_grad();
    }

    pub fn backward_chunk(&mut self, seq_len: usize, accumulation_scale: f32) {
        assert!(
            seq_len > 0 && seq_len <= self.cfg.chunk_len,
            "backward sequence length must be within tape capacity"
        );
        assert!(accumulation_scale.is_finite());
        let (d, v) = (self.cfg.d_latent, self.cfg.d_vocab);
        let n = seq_len * d;
        let loops = self.loops();
        if loops == 1 && self.device.is_gpu() {
            return crate::gpu_batch::backward_chunk_batched(self, seq_len, accumulation_scale);
        }
        let gpu = if loops > 1 { self.device.gpu() } else { None };
        let scale_loss = accumulation_scale / seq_len as f32;
        let logit_scale = 1.0 / (d as f32).sqrt();
        let pending = self
            .step_counter
            .checked_add(1)
            .expect("optimizer step counter overflow");
        for t in 0..seq_len {
            self.embed_row_marks[self.block.tape.x_ids[t]] = pending;
        }
        self.output_adjoints[..n].fill(0.0);
        let final_z: &[f32] = if loops == 1 {
            self.layer_activations
                .last()
                .map_or(&self.block.tape.z_final[..], Vec::as_slice)
        } else if let Some(last) = self.layer_activations.last() {
            &last[..n]
        } else {
            &self.continuous_inputs[..n]
        };
        if let Some(gpu) = gpu.as_ref() {
            crate::gpu_batch::stacked_backward_logits(self, seq_len, scale_loss, Some(gpu));
            self.output_adjoints[..n].copy_from_slice(&self.block.bwd_g_zfinal[..n]);
        } else {
            // Keep the old reverse-time accumulation order for bit-exact depth one.
            for t in (0..seq_len).rev() {
                for i in 0..v {
                    let indicator = if i == self.block.tape.target_ids[t] {
                        1.0
                    } else {
                        0.0
                    };
                    let g =
                        (self.block.tape.probs[t * v + i] - indicator) * scale_loss * logit_scale;
                    for j in 0..d {
                        self.output_adjoints[t * d + j] += g * self.unembed_w.data[i * d + j];
                        self.unembed_w.grad[i * d + j] += g * final_z[t * d + j];
                    }
                }
            }
        }
        let depth = self.depth();
        self.boundary_adjoints[depth][..n].copy_from_slice(&self.output_adjoints[..n]);
        if loops == 1 {
            for layer in (0..self.extra_blocks.len()).rev() {
                let scale = self.residual_scales[layer];
                for i in 0..n {
                    self.residual_block_adjoints[i] = scale * self.output_adjoints[i];
                }
                self.extra_blocks[layer].backward_chunk(
                    &self.residual_block_adjoints[..n],
                    seq_len,
                    &mut self.residual_input_adjoints[..n],
                );
                for i in 0..n {
                    self.output_adjoints[i] += self.residual_input_adjoints[i];
                }
                self.boundary_adjoints[layer + 1][..n].copy_from_slice(&self.output_adjoints[..n]);
            }
            self.block.backward_chunk(
                &self.output_adjoints[..n],
                seq_len,
                &mut self.input_adjoints[..n],
            );
        } else {
            let loop_scale = 1.0 / loops as f32;
            for layer in (0..self.extra_blocks.len()).rev() {
                let depth_scale = self.residual_scales[layer];
                for i in 0..n {
                    self.residual_block_adjoints[i] = depth_scale * self.output_adjoints[i];
                }
                for loop_index in (0..loops).rev() {
                    for i in 0..n {
                        self.input_adjoints[i] = loop_scale * self.residual_block_adjoints[i];
                    }
                    let b = &mut self.extra_blocks[layer];
                    if let Some(gpu) = gpu.as_ref() {
                        b.copy_loop_tape_to_base(loop_index);
                        crate::gpu_batch::backward_continuous_block_batched(
                            b,
                            &self.input_adjoints[..n],
                            seq_len,
                            &mut self.residual_input_adjoints[..n],
                            gpu,
                        );
                    } else {
                        b.backward_chunk_loop(
                            &self.input_adjoints[..n],
                            seq_len,
                            &mut self.residual_input_adjoints[..n],
                            loop_index,
                        );
                    }
                    for i in 0..n {
                        self.residual_block_adjoints[i] += self.residual_input_adjoints[i];
                    }
                }
                for i in 0..n {
                    self.output_adjoints[i] += self.residual_block_adjoints[i];
                }
                self.boundary_adjoints[layer + 1][..n].copy_from_slice(&self.output_adjoints[..n]);
            }
            self.residual_block_adjoints[..n].copy_from_slice(&self.output_adjoints[..n]);
            for loop_index in (0..loops).rev() {
                for i in 0..n {
                    self.input_adjoints[i] = loop_scale * self.residual_block_adjoints[i];
                }
                self.backward_train_chunk_batched_loop(seq_len, loop_index);
                for i in 0..n {
                    self.residual_block_adjoints[i] += self.residual_input_adjoints[i];
                }
            }
            self.input_adjoints[..n].copy_from_slice(&self.residual_block_adjoints[..n]);
        }
        self.boundary_adjoints[0][..n].copy_from_slice(&self.input_adjoints[..n]);
        for t in (0..seq_len).rev() {
            let row = self.block.tape.x_ids[t] * d;
            for j in 0..d {
                self.embed_w.grad[row + j] += self.input_adjoints[t * d + j];
            }
        }
    }

    /// Visit exactly the gradients consumed by `apply_adamw`, including the
    /// shared embedding/head and every continuous block. Stored memory and the
    /// adapter's consolidated copy are detached, not Adam parameters.
    #[cfg(test)]
    fn for_each_adam_gradient(&mut self, mut visit: impl FnMut(&mut [f32])) {
        visit(&mut self.embed_w.grad);
        visit(&mut self.unembed_w.grad);
        for b in std::iter::once(&mut self.block).chain(&mut self.extra_blocks) {
            visit(&mut b.norm_gamma.grad);
            visit(&mut b.norm_beta.grad);
            for p in [
                &mut b.a_mat,
                &mut b.w_delta,
                &mut b.w_b,
                &mut b.w_c,
                &mut b.w_qx,
                &mut b.w_qh,
                &mut b.w_gate,
                &mut b.w_proj,
                &mut b.mlp_w1,
                &mut b.mlp_w2,
            ] {
                visit(&mut p.grad);
            }
            visit(&mut b.adapters[0].down_proj.grad);
            visit(&mut b.adapters[0].up_proj.grad);
        }
    }

    pub(crate) fn adam_tensors(&mut self) -> Vec<AdamTensor<'_>> {
        let wd = self.cfg.weight_decay;
        let mut tensors = Vec::with_capacity(2 + 14 * self.depth());
        fn matrix(p: &mut ParamMatrix, weight_decay: f32) -> AdamTensor<'_> {
            AdamTensor {
                data: &mut p.data,
                grad: &mut p.grad,
                m: &mut p.m,
                v: &mut p.v,
                weight_decay,
            }
        }
        tensors.push(matrix(&mut self.embed_w, wd));
        tensors.push(matrix(&mut self.unembed_w, wd));
        for b in std::iter::once(&mut self.block).chain(&mut self.extra_blocks) {
            for p in [&mut b.norm_gamma, &mut b.norm_beta] {
                tensors.push(AdamTensor {
                    data: &mut p.data,
                    grad: &mut p.grad,
                    m: &mut p.m,
                    v: &mut p.v,
                    weight_decay: 0.0,
                });
            }
            for p in [
                &mut b.a_mat,
                &mut b.w_delta,
                &mut b.w_b,
                &mut b.w_c,
                &mut b.w_qx,
                &mut b.w_qh,
                &mut b.w_gate,
                &mut b.w_proj,
                &mut b.mlp_w1,
                &mut b.mlp_w2,
            ] {
                tensors.push(matrix(p, wd));
            }
            let ad = &mut b.adapters[0];
            tensors.push(matrix(&mut ad.down_proj, wd));
            tensors.push(matrix(&mut ad.up_proj, wd));
        }
        tensors
    }

    /// Global f64 L2 norm, then min(1, max_norm / norm) before AdamW.
    /// CUDA reduces and scales on-device. The CPU fallback has one norm pass;
    /// scaling is fused into AdamW's required pass over optimizer state.
    /// Non-finite norms skip atomically, including the optimizer step counter.
    pub fn apply_adamw_with_grad_clip(&mut self, lr: f32, max_norm: f32) -> GradientClipOutcome {
        assert!(max_norm.is_finite() && max_norm > 0.0);
        let cfg = self.cfg.clone();
        let step = self.step_counter.checked_add(1);
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = self.device.clone() {
            let embedding_rows: Vec<_> = self
                .embed_row_marks
                .iter()
                .enumerate()
                .filter_map(|(row, &mark)| (Some(mark) == step).then_some(row))
                .collect();
            let norm = ctx
                .clip_adamw(
                    &mut self.adam_tensors(),
                    &cfg,
                    lr,
                    step,
                    max_norm,
                    &embedding_rows,
                )
                .expect("CUDA gradient clipping/AdamW failed; refusing a silent CPU fallback");
            if !norm.is_finite() {
                self.zero_gradients();
                return GradientClipOutcome::Skipped { norm };
            }
            self.step_counter = step.expect("optimizer step counter overflow");
            ctx.invalidate_weights();
            return GradientClipOutcome::Applied { norm };
        }
        let mut tensors = self.adam_tensors();
        let mut norm_sq = 0.0f64;
        for tensor in &tensors {
            for &g in tensor.grad.iter() {
                norm_sq += (g as f64) * (g as f64);
            }
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() {
            drop(tensors);
            self.zero_gradients();
            return GradientClipOutcome::Skipped { norm };
        }
        let step = step.expect("optimizer step counter overflow");
        let scale = if norm > max_norm as f64 {
            max_norm as f64 / norm
        } else {
            1.0
        };
        for tensor in &mut tensors {
            tensor.step(&cfg, lr, step, scale);
        }
        drop(tensors);
        self.step_counter = step;
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
        GradientClipOutcome::Applied { norm }
    }

    pub fn apply_adamw(&mut self, lr: f32) {
        self.step_counter = self
            .step_counter
            .checked_add(1)
            .expect("optimizer step counter overflow");
        let c = &self.cfg;
        let s = self.step_counter;
        self.embed_w
            .step_adamw(lr, c.beta1, c.beta2, c.weight_decay, c.eps, s);
        self.block
            .apply_adamw(lr, c.beta1, c.beta2, c.weight_decay, c.eps, s);
        for b in &mut self.extra_blocks {
            b.apply_adamw(lr, c.beta1, c.beta2, c.weight_decay, c.eps, s);
        }
        self.unembed_w
            .step_adamw(lr, c.beta1, c.beta2, c.weight_decay, c.eps, s);
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
    }
    pub fn backward_and_step_chunk(&mut self, seq_len: usize) {
        self.zero_gradients();
        self.backward_chunk(seq_len, 1.0);
        self.apply_adamw(self.cfg.lr);
    }
    pub fn ema_consolidate_plasticity(&mut self) {
        self.block.ema_consolidate_plasticity();
        for b in &mut self.extra_blocks {
            b.ema_consolidate_plasticity();
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = &self.device {
            for b in std::iter::once(&self.block).chain(&self.extra_blocks) {
                ctx.refresh_safeguarded_weight(&b.adapters[0].up_proj)
                    .expect("CUDA adapter EMA synchronization failed");
            }
        }
        // The effective adapter-up weights are cached by host pointer. EMA
        // changes the consolidated half in place, so force a fresh upload.
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
    }

    /// Consolidate the fast adapter after an offline replay and report the
    /// Euclidean norm of the transferred slow-copy delta.  The consolidation
    /// itself is shared with the historical EMA path so its coefficient
    /// transfer remains exact.
    fn dream_consolidate_plasticity(&mut self) -> f32 {
        let mut delta_sq = 0.0f64;
        for b in std::iter::once(&self.block).chain(&self.extra_blocks) {
            let alpha = b.cfg.ema_alpha as f64;
            for &fast in &b.adapters[0].up_proj.data {
                let delta = alpha * fast as f64;
                delta_sq += delta * delta;
            }
        }
        self.ema_consolidate_plasticity();
        delta_sq.sqrt() as f32
    }

    fn dream_replay_latent(&mut self, value: &[f32]) {
        for block in 0..self.depth() {
            if block == 0 {
                self.block.dream_replay_value(value, &mut self.inf_features);
            } else {
                self.extra_blocks[block - 1].dream_replay_value(value, &mut self.inf_features);
            }
        }
    }

    fn dream_replay_memory_entry(&mut self, block: usize, value: &[f32]) {
        if block == 0 {
            self.block.dream_replay_value(value, &mut self.inf_features);
        } else {
            self.extra_blocks[block - 1].dream_replay_value(value, &mut self.inf_features);
        }
    }

    fn dream_memory_entries(&self) -> Vec<(usize, usize)> {
        let mut entries = Vec::new();
        for (block, b) in std::iter::once(&self.block)
            .chain(&self.extra_blocks)
            .enumerate()
        {
            entries.extend((0..b.memory.count).map(|entry| (block, entry)));
        }
        entries
    }

    fn dream_entry_value(&self, block: usize, entry: usize) -> Vec<f32> {
        let b = if block == 0 {
            &self.block
        } else {
            &self.extra_blocks[block - 1]
        };
        let start = entry * b.cfg.d_latent;
        b.memory.values[start..start + b.cfg.d_latent].to_vec()
    }

    /// Remember one supervised sequence for the runtime-only rehearsal cache.
    /// The sequence is copied before the caller's training tape is reused. A
    /// bounded cache keeps the oldest replay records from growing without
    /// affecting checkpoint bytes.
    pub fn remember_dream_sequence(&mut self, input_ids: &[usize], target_ids: &[usize]) {
        assert!(!input_ids.is_empty());
        assert_eq!(input_ids.len(), target_ids.len());
        assert!(input_ids.len() <= self.cfg.chunk_len);
        assert!(
            input_ids
                .iter()
                .chain(target_ids)
                .all(|&id| id < self.cfg.d_vocab)
        );
        self.dream_sequences.push(DreamSequence {
            input_ids: input_ids.to_vec(),
            target_ids: target_ids.to_vec(),
        });
        let capacity = self.cfg.mem_capacity.max(1);
        if self.dream_sequences.len() > capacity {
            let excess = self.dream_sequences.len() - capacity;
            self.dream_sequences.drain(..excess);
        }
    }

    fn apply_dream_sgd(&mut self, lr: f32, target_ids: &[usize]) {
        assert!(lr.is_finite() && lr > 0.0);
        // Replay is a separate optimizer. The output rows for old-task labels
        // are the main weights most directly overwritten by a conflicting
        // task; keeping the fresh-task rows frozen makes this protection much
        // less likely to trade Task B accuracy for Task A retention. Adam
        // moments and the normal step counter remain untouched.
        let d = self.cfg.d_latent;
        let mut rows = vec![false; self.cfg.d_vocab];
        for &target in target_ids {
            assert!(target < self.cfg.d_vocab);
            rows[target] = true;
        }
        for (target, &selected) in rows.iter().enumerate() {
            if selected {
                let start = target * d;
                for i in start..start + d {
                    self.unembed_w.data[i] -= lr * self.unembed_w.grad[i];
                }
            }
        }
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
    }

    /// Rehearse one stored/generated sequence through the ordinary supervised
    /// loss. This uses a separate, small-LR SGD pass: the main weights change,
    /// but their Adam moments and the normal optimizer step counter do not.
    fn dream_gradient(
        &mut self,
        sequence: &DreamSequence,
        recurrent: &[f32],
    ) -> (f32, Vec<Vec<f32>>) {
        self.copy_recurrent_state_from(recurrent);
        let loss = self.forward_train_chunk(&sequence.input_ids, &sequence.target_ids);
        self.zero_gradients();
        self.backward_chunk(sequence.input_ids.len(), 1.0);
        let gradients = self
            .adam_tensors()
            .into_iter()
            .map(|tensor| tensor.grad.to_vec())
            .collect();
        self.zero_gradients();
        self.copy_recurrent_state_from(recurrent);
        (loss, gradients)
    }

    fn dream_loss(&mut self, sequence: &DreamSequence, recurrent: &[f32]) -> f32 {
        self.copy_recurrent_state_from(recurrent);
        let loss = self.forward_train_chunk(&sequence.input_ids, &sequence.target_ids);
        self.copy_recurrent_state_from(recurrent);
        loss
    }

    fn dream_parameter_snapshot(&mut self) -> Vec<Vec<f32>> {
        self.adam_tensors()
            .into_iter()
            .map(|tensor| tensor.data.to_vec())
            .collect()
    }

    fn dream_restore_parameters(&mut self, snapshot: &[Vec<f32>]) {
        for (tensor, saved) in self.adam_tensors().into_iter().zip(snapshot) {
            assert_eq!(tensor.data.len(), saved.len());
            tensor.data.copy_from_slice(saved);
        }
    }

    fn dream_gradient_snapshot(&mut self) -> Vec<Vec<f32>> {
        self.adam_tensors()
            .into_iter()
            .map(|tensor| tensor.grad.to_vec())
            .collect()
    }

    fn dream_restore_gradients(&mut self, snapshot: &[Vec<f32>]) {
        for (tensor, saved) in self.adam_tensors().into_iter().zip(snapshot) {
            assert_eq!(tensor.grad.len(), saved.len());
            tensor.grad.copy_from_slice(saved);
        }
    }

    /// Apply an old-task step after removing the component that would increase
    /// the current-task loss to first order. The exact current-task loss is
    /// checked by the caller as well: nonlinear effects are backtracked rather
    /// than being allowed to turn a projection into a regression.
    fn apply_dream_projected_sgd(
        &mut self,
        lr: f32,
        old_gradients: &[Vec<f32>],
        fresh_gradients: &[Vec<f32>],
        selected_output_targets: Option<&[usize]>,
    ) {
        assert_eq!(old_gradients.len(), fresh_gradients.len());
        let selected_rows = selected_output_targets.map(|targets| {
            let mut rows = vec![false; self.cfg.d_vocab];
            for &target in targets {
                assert!(target < rows.len());
                rows[target] = true;
            }
            rows
        });
        let mut dot = 0.0f64;
        let mut fresh_norm_sq = 0.0f64;
        for (tensor_index, (old, fresh)) in old_gradients.iter().zip(fresh_gradients).enumerate() {
            assert_eq!(old.len(), fresh.len());
            for (index, (&old, &fresh)) in old.iter().zip(fresh).enumerate() {
                let selected = match selected_rows.as_ref() {
                    None => true,
                    Some(rows) => tensor_index == 1 && rows[index / self.cfg.d_latent],
                };
                if selected {
                    dot += old as f64 * fresh as f64;
                    fresh_norm_sq += fresh as f64 * fresh as f64;
                }
            }
        }
        // Gradient descent on the old loss uses -old_grad. If old_grad and
        // fresh_grad have a negative dot product, its component along the
        // fresh gradient would raise fresh loss; remove exactly that component.
        let projection = if dot < 0.0 && fresh_norm_sq > 0.0 {
            dot / fresh_norm_sq
        } else {
            0.0
        };
        let d_latent = self.cfg.d_latent;
        for (tensor_index, (tensor, (old, fresh))) in self
            .adam_tensors()
            .into_iter()
            .zip(old_gradients.iter().zip(fresh_gradients))
            .enumerate()
        {
            for (index, ((weight, &old), &fresh)) in
                tensor.data.iter_mut().zip(old).zip(fresh).enumerate()
            {
                let selected = match selected_rows.as_ref() {
                    None => true,
                    Some(rows) => tensor_index == 1 && rows[index / d_latent],
                };
                if selected {
                    let projected = old as f64 - projection * fresh as f64;
                    *weight -= lr * projected as f32;
                }
            }
        }
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
    }

    fn rehearse_dream_sequence_guarded(
        &mut self,
        sequence: &DreamSequence,
        guard: &DreamSequence,
        lr: f32,
        steps: usize,
    ) -> f32 {
        const LOSS_TOLERANCE: f32 = 1e-6;
        let mut recurrent = vec![0.0; self.recurrent_state_len()];
        self.copy_recurrent_state_to(&mut recurrent);
        let mut last_loss = f32::NAN;
        for _ in 0..steps {
            let (baseline, fresh_gradients) = self.dream_gradient(guard, &recurrent);
            let (old_loss, old_gradients) = self.dream_gradient(sequence, &recurrent);
            last_loss = old_loss;
            let snapshot = self.dream_parameter_snapshot();
            let mut step_lr = lr;
            let mut accepted = false;
            // Projection is a first-order safeguard. Backtracking makes the
            // contract exact for this nonlinear recurrent model.
            for _ in 0..10 {
                self.dream_restore_parameters(&snapshot);
                self.apply_dream_projected_sgd(
                    step_lr,
                    &old_gradients,
                    &fresh_gradients,
                    Some(&sequence.target_ids),
                );
                let candidate = self.dream_loss(guard, &recurrent);
                if candidate.is_finite() && candidate <= baseline + LOSS_TOLERANCE {
                    accepted = true;
                    break;
                }
                step_lr *= 0.5;
            }
            if !accepted {
                self.dream_restore_parameters(&snapshot);
            }
            self.zero_gradients();
        }
        self.copy_recurrent_state_from(&recurrent);
        last_loss
    }

    fn rehearse_dream_sequence(
        &mut self,
        sequence: &DreamSequence,
        guard: Option<&DreamSequence>,
        lr: f32,
        steps: usize,
    ) -> f32 {
        if let Some(guard) = guard {
            return self.rehearse_dream_sequence_guarded(sequence, guard, lr, steps);
        }
        let mut recurrent = vec![0.0; self.recurrent_state_len()];
        self.copy_recurrent_state_to(&mut recurrent);
        let mut last_loss = f32::NAN;
        for _ in 0..steps {
            self.copy_recurrent_state_from(&recurrent);
            last_loss = self.forward_train_chunk(&sequence.input_ids, &sequence.target_ids);
            self.zero_gradients();
            self.backward_chunk(sequence.input_ids.len(), 1.0);
            self.apply_dream_sgd(lr, &sequence.target_ids);
            self.zero_gradients();
        }
        self.copy_recurrent_state_from(&recurrent);
        last_loss
    }

    /// Stage 1 API retained for callers that only want memory replay.
    pub fn dream_replay_memory(
        &mut self,
        replay: usize,
        rng: &mut SimpleRng,
    ) -> crate::dream::DreamSummary {
        self.dream_replay(crate::dream::DreamMode::Memory, replay, 0, 0.8, rng)
    }

    /// Run a host-only offline sleep phase with the default rehearsal policy.
    pub fn dream_replay(
        &mut self,
        mode: crate::dream::DreamMode,
        replay: usize,
        dream_len: usize,
        temperature: f32,
        rng: &mut SimpleRng,
    ) -> crate::dream::DreamSummary {
        self.dream_replay_with_options(
            mode,
            replay,
            dream_len,
            temperature,
            crate::dream::DEFAULT_REHEARSAL_LR,
            crate::dream::DEFAULT_REHEARSAL_STEPS,
            rng,
        )
    }

    /// Run replay and consolidate it into the main model. Memory values are
    /// sampled first; generate/both modes also produce supervised next-token
    /// sequences from each seed. Stored sequences and generated sequences take
    /// a separate small-LR SGD step on the main output rows associated with
    /// the replay targets. The normal Adam state and step counter are
    /// intentionally untouched, so dream updates cannot change the fresh-task
    /// LR schedule. Recurrent carry is restored around generation and
    /// rehearsal.
    pub fn dream_replay_with_options(
        &mut self,
        mode: crate::dream::DreamMode,
        replay: usize,
        dream_len: usize,
        temperature: f32,
        rehearsal_lr: f32,
        rehearsal_steps: usize,
        rng: &mut SimpleRng,
    ) -> crate::dream::DreamSummary {
        self.dream_replay_with_options_and_guard(
            mode,
            replay,
            dream_len,
            temperature,
            rehearsal_lr,
            rehearsal_steps,
            None,
            rng,
        )
    }

    /// Replay with an optional fresh-task guard. Guarded replay projects each
    /// old-task gradient away from the fresh-task gradient and then backtracks
    /// against the exact fresh-task cross-entropy. This is opt-in so callers
    /// that do not provide a fresh sequence retain the historical rehearsal
    /// behavior.
    pub fn dream_replay_with_options_and_guard(
        &mut self,
        mode: crate::dream::DreamMode,
        replay: usize,
        dream_len: usize,
        temperature: f32,
        rehearsal_lr: f32,
        rehearsal_steps: usize,
        fresh_task: Option<(&[usize], &[usize])>,
        rng: &mut SimpleRng,
    ) -> crate::dream::DreamSummary {
        let guard = fresh_task.map(|(input_ids, target_ids)| {
            assert!(!input_ids.is_empty());
            assert_eq!(input_ids.len(), target_ids.len());
            assert!(input_ids.len() <= self.cfg.chunk_len);
            assert!(
                input_ids
                    .iter()
                    .chain(target_ids)
                    .all(|&id| id < self.cfg.d_vocab)
            );
            DreamSequence {
                input_ids: input_ids.to_vec(),
                target_ids: target_ids.to_vec(),
            }
        });
        let started = Instant::now();
        assert!(rehearsal_lr.is_finite() && rehearsal_lr > 0.0);
        assert!(rehearsal_steps > 0);
        if replay == 0 {
            return crate::dream::DreamSummary {
                elapsed_seconds: started.elapsed().as_secs_f64(),
                ..Default::default()
            };
        }
        let saved_gradients = self.dream_gradient_snapshot();
        let saved_embed_row_marks = self.embed_row_marks.clone();
        if mode.includes_generation() {
            assert!(dream_len > 0);
            assert!(temperature.is_finite() && temperature > 0.0);
        }
        let mut entries = self.dream_memory_entries();
        let take = replay.min(entries.len());
        let sequence_take = if mode.includes_memory() {
            replay.min(self.dream_sequences.len())
        } else {
            0
        };
        if take == 0 && sequence_take == 0 {
            self.dream_restore_gradients(&saved_gradients);
            self.embed_row_marks.copy_from_slice(&saved_embed_row_marks);
            return crate::dream::DreamSummary {
                elapsed_seconds: started.elapsed().as_secs_f64(),
                ..Default::default()
            };
        }
        for i in 0..take {
            let remaining = entries.len() - i;
            let j = i + (rng.next_u32() as usize % remaining);
            entries.swap(i, j);
        }
        let values: Vec<Vec<f32>> = entries[..take]
            .iter()
            .map(|&(block, entry)| self.dream_entry_value(block, entry))
            .collect();

        let mut recurrent = vec![0.0; self.recurrent_state_len()];
        self.copy_recurrent_state_to(&mut recurrent);
        let mut generated = Vec::new();
        let mut generated_inputs = Vec::new();
        let mut generated_targets = Vec::new();
        if mode.includes_generation() {
            let mut logits = vec![0.0; self.cfg.d_vocab];
            for value in &values {
                // Each memory seed starts from the same live carry. This makes
                // K sequences independent while retaining deterministic model
                // state inside each sequence.
                self.copy_recurrent_state_from(&recurrent);
                self.forward_dream_input(value, &mut logits);
                let mut token = crate::dream::sample_token(&mut logits, temperature, rng);
                for _ in 0..dream_len {
                    let input = token;
                    self.forward_inference(input, &mut logits);
                    generated_inputs.push(input);
                    token = crate::dream::sample_token(&mut logits, temperature, rng);
                    generated_targets.push(token);
                    generated.push(self.inf_features.clone());
                }
            }
        }
        self.copy_recurrent_state_from(&recurrent);

        if mode.includes_memory() {
            for (&(block, _), value) in entries[..take].iter().zip(&values) {
                // A stored entry belongs to the block that captured it. Keep
                // memory mode's Stage 1 behavior for stacked models rather
                // than applying every block's adapter to the same value.
                self.dream_replay_memory_entry(block, value);
            }
        }
        for value in &generated {
            self.dream_replay_latent(value);
        }
        self.copy_recurrent_state_from(&recurrent);
        let consolidation_delta_norm = if mode.includes_memory() || !generated.is_empty() {
            self.dream_consolidate_plasticity()
        } else {
            0.0
        };

        // Training replay uses CPU workspaces even when the main run selected a
        // GPU. The caller synchronizes safeguarded weights before entering this
        // method; restore the backend and invalidate its cache afterward.
        let device = std::mem::replace(&mut self.device, Device::Cpu);
        let mut rehearsal_sequences = 0;
        let mut last_rehearsal_loss = None;
        let stored_sequences: Vec<DreamSequence> = self.dream_sequences[..sequence_take].to_vec();
        for sequence in stored_sequences {
            last_rehearsal_loss = Some(self.rehearse_dream_sequence(
                &sequence,
                guard.as_ref(),
                rehearsal_lr,
                rehearsal_steps,
            ));
            rehearsal_sequences += 1;
        }
        if mode.includes_generation() {
            for (inputs, targets) in generated_inputs
                .chunks(self.cfg.chunk_len)
                .zip(generated_targets.chunks(self.cfg.chunk_len))
            {
                if inputs.is_empty() {
                    continue;
                }
                let sequence = DreamSequence {
                    input_ids: inputs.to_vec(),
                    target_ids: targets.to_vec(),
                };
                last_rehearsal_loss = Some(self.rehearse_dream_sequence(
                    &sequence,
                    guard.as_ref(),
                    rehearsal_lr,
                    rehearsal_steps,
                ));
                rehearsal_sequences += 1;
            }
        }
        self.device = device;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = self.device.clone() {
            ctx.refresh_safeguarded_weights(&mut self.adam_tensors())
                .expect("CUDA dream weight synchronization failed");
        }
        if let Some(ctx) = self.device.gpu() {
            ctx.invalidate_weights();
        }
        self.dream_restore_gradients(&saved_gradients);
        self.embed_row_marks.copy_from_slice(&saved_embed_row_marks);
        crate::dream::DreamSummary {
            entries_replayed: take,
            generated_tokens: generated.len(),
            rehearsal_sequences,
            last_rehearsal_loss: last_rehearsal_loss.filter(|loss| loss.is_finite()),
            consolidation_delta_norm,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        }
    }

    pub fn insert_training_memory_at(&mut self, loss: f32, last: usize) {
        if !(loss > 3.5) {
            return;
        }
        let loops = self.loops();
        for b in std::iter::once(&mut self.block).chain(&mut self.extra_blocks) {
            let (k, d) = (b.cfg.d_mem_key, b.cfg.d_latent);
            let loop_l = if loops == 1 {
                0
            } else {
                (loops - 1) * b.tape.max_l
            };
            b.memory.insert_protected_with_device(
                &b.tape.q_poincare[(loop_l + last) * k..(loop_l + last + 1) * k],
                &b.tape.z_final[(loop_l + last) * d..(loop_l + last + 1) * d],
                loss,
                self.step_counter,
                &self.device,
            );
        }
    }
    pub fn insert_training_memory(&mut self, loss: f32, seq_len: usize) {
        assert!(seq_len > 0 && seq_len <= self.cfg.chunk_len);
        self.insert_training_memory_at(loss, seq_len - 1);
    }
    /// Trainable scalars only: memory, optimizer state, fixed scales and the
    /// adapter's non-independent consolidated copy are intentionally excluded.
    pub fn parameter_count(&self) -> usize {
        self.embed_w.data.len()
            + self.unembed_w.data.len()
            + self.block.parameter_count()
            + self
                .extra_blocks
                .iter()
                .map(PSSAContinuousBlockV2::parameter_count)
                .sum::<usize>()
    }
}

#[cfg(test)]
mod training_safeguards_tests {
    use super::*;

    fn model() -> PSSALayerV2 {
        PSSALayerV2::new_with_device(
            PSSAConfigV2 {
                depth: 2,
                d_vocab: 5,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 2,
                chunk_len: 2,
                ..Default::default()
            },
            42,
            Device::Cpu,
        )
    }

    fn gradients(m: &mut PSSALayerV2) -> Vec<f32> {
        let mut out = Vec::new();
        m.for_each_adam_gradient(|g| out.extend_from_slice(g));
        out
    }

    fn optimizer_state(m: &PSSALayerV2) -> Vec<f32> {
        let mut out = Vec::new();
        let mut matrix = |p: &ParamMatrix| {
            out.extend_from_slice(&p.data);
            out.extend_from_slice(&p.m);
            out.extend_from_slice(&p.v);
        };
        matrix(&m.embed_w);
        matrix(&m.unembed_w);
        for b in std::iter::once(&m.block).chain(&m.extra_blocks) {
            for p in [
                &b.a_mat,
                &b.w_delta,
                &b.w_b,
                &b.w_c,
                &b.w_qx,
                &b.w_qh,
                &b.w_gate,
                &b.w_proj,
                &b.mlp_w1,
                &b.mlp_w2,
                &b.adapters[0].down_proj,
                &b.adapters[0].up_proj,
            ] {
                matrix(p);
            }
        }
        for b in std::iter::once(&m.block).chain(&m.extra_blocks) {
            for p in [&b.norm_gamma, &b.norm_beta] {
                out.extend_from_slice(&p.data);
                out.extend_from_slice(&p.m);
                out.extend_from_slice(&p.v);
            }
        }
        out
    }

    fn historical_clip(m: &mut PSSALayerV2, lr: f32, max_norm: f32) -> GradientClipOutcome {
        let mut sum = 0.0f64;
        m.for_each_adam_gradient(|grad| {
            for &g in grad.iter() {
                sum += (g as f64) * (g as f64);
            }
        });
        let norm = sum.sqrt();
        if !norm.is_finite() {
            m.zero_gradients();
            return GradientClipOutcome::Skipped { norm };
        }
        if norm > max_norm as f64 {
            let scale = max_norm as f64 / norm;
            m.for_each_adam_gradient(|grad| {
                for g in grad {
                    *g = (*g as f64 * scale) as f32;
                }
            });
        }
        m.apply_adamw(lr);
        GradientClipOutcome::Applied { norm }
    }

    #[test]
    fn fused_cpu_clip_matches_historical_optimizer_exactly() {
        let mut old = model();
        let mut new = model();
        // Cover nonzero moments, decay (but not norm-vector decay), every block,
        // both signs, signed zero, huge finite values, and subnormal clip limits.
        for (step, cap) in [1.0, f32::MAX, 0.25, f32::from_bits(1), 2.0]
            .into_iter()
            .enumerate()
        {
            for m in [&mut old, &mut new] {
                let mut index = 0;
                m.for_each_adam_gradient(|grad| {
                    for g in grad {
                        *g = match index % 6 {
                            0 => -0.0,
                            1 => 0.125,
                            2 => -3.5,
                            3 => {
                                if step == 2 {
                                    1e21
                                } else {
                                    0.75
                                }
                            }
                            4 => f32::from_bits(1),
                            _ => -0.25,
                        };
                        index += 1;
                    }
                });
            }
            let expected = historical_clip(&mut old, 0.003, cap);
            assert_eq!(new.apply_adamw_with_grad_clip(0.003, cap), expected);
            assert_eq!(new.step_counter, old.step_counter);
            assert_eq!(optimizer_state(&new), optimizer_state(&old));
            // Compare bits too: below-threshold signed zeros must be untouched.
            let bits = |m: &mut PSSALayerV2| {
                gradients(m)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&mut new), bits(&mut old));
        }
    }

    #[test]
    fn nonfinite_clip_still_skips_at_step_counter_limit() {
        let mut m = model();
        m.step_counter = usize::MAX;
        m.embed_w.grad[0] = f32::NAN;
        let before = optimizer_state(&m);
        assert!(matches!(
            m.apply_adamw_with_grad_clip(1e-3, 1.0),
            GradientClipOutcome::Skipped { .. }
        ));
        assert_eq!(m.step_counter, usize::MAX);
        assert_eq!(optimizer_state(&m), before);
        assert!(gradients(&mut m).iter().all(|&g| g == 0.0));
    }

    #[test]
    fn stacked_batched_backward_marks_all_accumulated_embedding_rows() {
        let mut m = model();
        m.step_counter = 5;
        m.zero_gradients();
        // Exercise the same stacked scatter used by CUDA, on the CPU. The
        // second chunk represents another lane/microbatch before a shared step.
        for (inputs, targets, marks) in [
            ([0, 1], [1, 2], [6, 6, 0, 0, 0]),
            ([2, 0], [3, 1], [6, 6, 6, 0, 0]),
        ] {
            m.reset_recurrent_state();
            crate::gpu_batch::forward_train_chunk_batched(&mut m, &inputs, &targets);
            crate::gpu_batch::backward_chunk_batched(&mut m, 2, 0.5);
            assert_eq!(m.embed_row_marks, marks);
        }
        let d = m.cfg.d_latent;
        for row in 0..3 {
            assert!(
                m.embed_w.grad[row * d..(row + 1) * d]
                    .iter()
                    .any(|&g| g != 0.0)
            );
        }
        assert!(m.embed_w.grad[3 * d..].iter().all(|&g| g == 0.0));
    }

    #[cfg(feature = "cuda")]
    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
                "element {i}: {a} != {b}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU; exercises embedded PTX and resident optimizer state"]
    fn cuda_resident_clip_accumulation_skip_and_ema_match_cpu() {
        let ctx = crate::cuda::CudaContext::init().expect("CUDA GPU required");
        let mut old = model();
        let mut new = model();
        new.device = Device::Cuda(ctx.clone());
        ctx.begin_safeguarded_training(&new.adam_tensors()).unwrap();
        for iteration in 0..4 {
            for m in [&mut old, &mut new] {
                m.zero_gradients();
                m.for_each_adam_gradient(|g| g.fill(0.125));
                m.embed_row_marks.fill(m.step_counter + 1);
                m.unembed_w.grad.fill(0.0);
                if iteration == 3 {
                    m.block.a_mat.grad[0] = 1e21;
                }
            }
            let mut a = vec![0.5f32; 2 * old.cfg.d_vocab];
            if iteration == 1 {
                a[0] = f32::INFINITY;
            }
            let b = vec![0.25f32; 2 * old.cfg.d_latent];
            // Two TN dispatches must accumulate without any gradient readback.
            for _ in 0..2 {
                crate::backend::gemm_tn_cpu_accumulate_into(
                    &a,
                    &b,
                    2,
                    5,
                    4,
                    &mut old.unembed_w.grad,
                )
                .unwrap();
                ctx.try_gemm_tn_accumulate_into(&a, &b, 2, 5, 4, &mut new.unembed_w.grad)
                    .unwrap();
            }
            let cap = if iteration == 2 { f32::MAX } else { 1.0 };
            match (
                historical_clip(&mut old, 1e-3, cap),
                new.apply_adamw_with_grad_clip(1e-3, cap),
            ) {
                (
                    GradientClipOutcome::Applied { norm: a },
                    GradientClipOutcome::Applied { norm: b },
                ) => {
                    assert!((a - b).abs() <= 1e-12 * a.max(1.0));
                }
                (
                    GradientClipOutcome::Skipped { norm: a },
                    GradientClipOutcome::Skipped { norm: b },
                ) => {
                    assert!(!a.is_finite() && !b.is_finite());
                }
                outcome => panic!("clip outcomes differ: {outcome:?}"),
            }
            assert_eq!(new.step_counter, old.step_counter);
            if iteration == 0 {
                old.ema_consolidate_plasticity();
                new.ema_consolidate_plasticity();
            }
        }
        ctx.finish_safeguarded_training(&mut new.adam_tensors())
            .unwrap();
        assert_close(&optimizer_state(&new), &optimizer_state(&old));
        assert_close(&gradients(&mut new), &gradients(&mut old));
        // The fused finite check must include moments, not only weights/grads.
        new.block.a_mat.v[0] = f32::NAN;
        ctx.begin_safeguarded_training(&new.adam_tensors()).unwrap();
        new.zero_gradients();
        new.apply_adamw_with_grad_clip(1e-3, 1.0);
        assert_eq!(ctx.safeguarded_parameters_finite(), Some(false));
        ctx.finish_safeguarded_training(&mut new.adam_tensors())
            .unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU; checks real CLI scope, packed/replayed batches and handoff"]
    fn cuda_safeguarded_cli_training_matches_cpu() {
        use crate::cli::{CLIHandler, TrainingBackend, TrainingOptions};
        let words = (0..48)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let raw = format!("{words}\n\n{words}");
        for (depth, loops, batch_size) in [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2), (1, 2, 2)] {
            let options = TrainingOptions {
                backend: TrainingBackend::Cpu,
                latent: 4,
                state: 2,
                key: 2,
                memory: 2,
                chunk: 3,
                batch_size,
                accumulate: 2,
                epochs: 2,
                depth,
                loops,
                grad_clip: Some(1.0),
                memory_value_cap: Some(0.25),
                no_tui: true,
                ..Default::default()
            };
            let (mut old, _) = CLIHandler::train_corpus(&raw, &options).unwrap();
            let (mut new, _) = CLIHandler::train_corpus(
                &raw,
                &TrainingOptions {
                    backend: TrainingBackend::Cuda,
                    ..options
                },
            )
            .unwrap();
            assert_eq!(new.step_counter, old.step_counter);
            assert_close(&optimizer_state(&new), &optimizer_state(&old));
            assert_close(&gradients(&mut new), &gradients(&mut old));
            for (a, b) in std::iter::once(&new.block)
                .chain(&new.extra_blocks)
                .zip(std::iter::once(&old.block).chain(&old.extra_blocks))
            {
                assert!(a.memory.count > 0);
                assert_eq!(a.memory.count, b.memory.count);
                assert_close(&a.memory.values, &b.memory.values);
                assert_close(
                    &a.adapters[0].consolidated_up,
                    &b.adapters[0].consolidated_up,
                );
                for value in a.memory.values.chunks_exact(a.cfg.d_latent) {
                    assert!(
                        value.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() <= 0.25f64.powi(2)
                    );
                }
            }
        }
    }

    #[test]
    fn global_clip_covers_every_adam_parameter_and_preserves_small_gradients() {
        let mut m = model();
        m.for_each_adam_gradient(|g| g.fill(2.0));
        let original = gradients(&mut m);
        assert_eq!(original.len(), m.parameter_count());
        let expected_norm = 2.0 * (original.len() as f64).sqrt();
        let GradientClipOutcome::Applied { norm } = m.apply_adamw_with_grad_clip(1e-3, 1.0) else {
            panic!("finite gradients skipped");
        };
        assert_eq!(norm, expected_norm);
        let clipped = gradients(&mut m);
        let norm_after = clipped
            .iter()
            .map(|&g| (g as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm_after - 1.0).abs() < 1e-6);
        for g in clipped {
            assert!((g as f64 - 2.0 / expected_norm).abs() < 1e-6 / expected_norm);
        }
        m.zero_gradients();
        m.embed_w.grad[0] = 0.3;
        m.block.adapters[0].up_proj.grad[0] = 0.4;
        let before = gradients(&mut m);
        assert!(matches!(
            m.apply_adamw_with_grad_clip(1e-3, 1.0),
            GradientClipOutcome::Applied { .. }
        ));
        assert_eq!(gradients(&mut m), before); // no multiply, even by 1
        assert_eq!(m.step_counter, 2);
    }

    #[test]
    fn nonfinite_clip_skips_atomically_with_nonzero_adam_moments() {
        let mut m = model();
        m.for_each_adam_gradient(|g| g.fill(0.1));
        m.apply_adamw(1e-3);
        let before = optimizer_state(&m);
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            m.for_each_adam_gradient(|g| g.fill(0.2));
            m.extra_blocks[0].adapters[0].up_proj.grad[0] = bad;
            assert!(matches!(
                m.apply_adamw_with_grad_clip(1e-3, 1.0),
                GradientClipOutcome::Skipped { .. }
            ));
            assert_eq!(m.step_counter, 1);
            assert_eq!(optimizer_state(&m), before);
            assert!(gradients(&mut m).iter().all(|&g| g == 0.0));
        }
    }

    #[test]
    fn huge_finite_gradients_clip_before_f32_adam_second_moment_overflow() {
        let mut m = model();
        m.block.norm_gamma.grad[0] = 1e21;
        let GradientClipOutcome::Applied { norm } = m.apply_adamw_with_grad_clip(1e-3, 1.0) else {
            panic!("large finite gradient must not overflow the norm");
        };
        assert_eq!(norm, 1e21f32 as f64);
        assert_eq!(m.block.norm_gamma.grad[0], 1.0);
        assert!(optimizer_state(&m).iter().all(|x| x.is_finite()));
    }
}
