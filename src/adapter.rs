use crate::linalg::{SimpleRng, sigmoid};
use crate::pssa::ParamMatrix;

#[derive(Clone, Debug, PartialEq)]
pub struct PlasticAdapterV2 {
    pub down_proj: ParamMatrix, // [rank x d_latent]
    pub up_proj: ParamMatrix,   // fast coefficients [d_latent x rank]
    /// Slow coefficients are deliberately not Adam-updated.  Consolidation transfers
    /// mass from `up_proj` here while preserving their effective sum exactly.
    pub consolidated_up: Vec<f32>, // [d_latent x rank]
    pub rank: usize,
    pub d_latent: usize,
}

impl PlasticAdapterV2 {
    pub fn new(d_latent: usize, rank: usize, rng: &mut SimpleRng) -> Self {
        Self {
            down_proj: ParamMatrix::random_xavier(rank, d_latent, rng),
            up_proj: ParamMatrix::zeros(d_latent, rank),
            consolidated_up: vec![0.0; d_latent * rank],
            rank,
            d_latent,
        }
    }

    /// Computes `out = (U_fast + U_slow) * SiLU(D*x)` without allocation.
    #[inline(always)]
    pub fn forward_into(&self, x: &[f32], act_buf: &mut [f32], out: &mut [f32]) {
        self.down_proj.matvec(x, act_buf);
        for h in act_buf.iter_mut() {
            *h *= sigmoid(*h);
        }
        self.total_up_matvec(act_buf, out);
    }

    #[inline(always)]
    pub fn total_up_matvec(&self, act: &[f32], out: &mut [f32]) {
        assert_eq!(act.len(), self.rank);
        assert_eq!(out.len(), self.d_latent);
        for i in 0..self.d_latent {
            let off = i * self.rank;
            let mut total = 0.0;
            for r in 0..self.rank {
                total += (self.up_proj.data[off + r] + self.consolidated_up[off + r]) * act[r];
            }
            out[i] = total;
        }
    }

    #[inline(always)]
    pub fn total_up_matvec_transpose(&self, g: &[f32], out: &mut [f32]) {
        assert_eq!(g.len(), self.d_latent);
        assert_eq!(out.len(), self.rank);
        out.fill(0.0);
        for i in 0..self.d_latent {
            let off = i * self.rank;
            for r in 0..self.rank {
                out[r] += g[i] * (self.up_proj.data[off + r] + self.consolidated_up[off + r]);
            }
        }
    }

    /// Exact coefficient transfer: `(1-alpha) U_fast + (U_slow+alpha U_fast)`
    /// equals the prior effective up matrix. `alpha` must be in [0, 1].
    pub fn consolidate(&mut self, alpha: f32) {
        assert!(alpha.is_finite() && (0.0..=1.0).contains(&alpha));
        for i in 0..self.up_proj.data.len() {
            let fast = self.up_proj.data[i];
            self.consolidated_up[i] += alpha * fast;
            self.up_proj.data[i] = (1.0 - alpha) * fast;
        }
    }

    pub fn zero_grad(&mut self) {
        self.down_proj.zero_grad();
        self.up_proj.zero_grad();
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
        self.down_proj
            .step_adamw(lr, beta1, beta2, weight_decay, eps, step);
        self.up_proj
            .step_adamw(lr, beta1, beta2, weight_decay, eps, step);
    }
}
