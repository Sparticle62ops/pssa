use crate::defense::{RateLimiterGate, UpdateOutcome};
use crate::linalg::dot_slice;
use std::f32;

#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicEpisodicBankV2 {
    pub capacity: usize,
    pub count: usize,
    pub dim_key: usize,
    pub dim_val: usize,
    pub write_head: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub norm_sq: Vec<f32>,
    pub confidence: Vec<f32>,
    pub last_seen_step: Vec<usize>,
}

impl HyperbolicEpisodicBankV2 {
    pub fn new(capacity: usize, dim_key: usize, dim_val: usize) -> Self {
        assert!(
            capacity > 0 && dim_key > 0 && dim_val > 0,
            "memory dimensions and capacity must be positive"
        );
        Self {
            capacity,
            count: 0,
            dim_key,
            dim_val,
            write_head: 0,
            keys: vec![0.0; capacity * dim_key],
            values: vec![0.0; capacity * dim_val],
            norm_sq: vec![0.0; capacity],
            confidence: vec![1.0; capacity],
            last_seen_step: vec![0; capacity],
        }
    }

    #[inline(always)]
    pub fn diffeomorphic_project(q_euc: &[f32], out_pnc: &mut [f32]) -> f32 {
        assert!(!q_euc.is_empty() && q_euc.len() == out_pnc.len());
        let q_norm = dot_slice(q_euc, q_euc).sqrt();
        let scale = 1.0 / (1.0 + q_norm);
        for i in 0..q_euc.len() {
            out_pnc[i] = q_euc[i] * scale;
        }
        q_norm
    }

    /// Stable equivalent of acosh(1 + 2*s/denom): 2 asinh(sqrt(s/denom)).
    /// Inputs must lie in the open Poincare ball; malformed state is rejected.
    #[inline(always)]
    pub fn poincare_distance(u: &[f32], u_sq: f32, v: &[f32], v_sq: f32) -> f32 {
        assert_eq!(u.len(), v.len());
        assert!(
            u_sq.is_finite() && v_sq.is_finite() && u_sq < 1.0 && v_sq < 1.0,
            "Poincare points must be finite and inside the open ball"
        );
        let mut sq_dist = 0.0f64;
        for i in 0..u.len() {
            let d = (u[i] - v[i]) as f64;
            sq_dist += d * d;
        }
        let denom = (1.0f64 - u_sq as f64) * (1.0f64 - v_sq as f64);
        assert!(
            denom > 0.0 && denom.is_finite(),
            "invalid Poincare denominator"
        );
        (2.0 * (sq_dist / denom).sqrt().asinh()) as f32
    }

    pub fn insert(&mut self, key_pnc: &[f32], val: &[f32]) -> usize {
        assert_eq!(key_pnc.len(), self.dim_key);
        assert_eq!(val.len(), self.dim_val);
        let key_sq = dot_slice(key_pnc, key_pnc);
        assert!(
            key_sq.is_finite() && key_sq < 1.0,
            "memory key must be in open Poincare ball"
        );
        let idx = if self.count < self.capacity {
            let i = self.count;
            self.count += 1;
            i
        } else {
            let i = self.write_head;
            self.write_head = (self.write_head + 1) % self.capacity;
            i
        };
        let k_off = idx * self.dim_key;
        self.keys[k_off..k_off + self.dim_key].copy_from_slice(key_pnc);
        self.norm_sq[idx] = key_sq;
        let v_off = idx * self.dim_val;
        self.values[v_off..v_off + self.dim_val].copy_from_slice(val);
        self.confidence[idx] = 1.0;
        self.last_seen_step[idx] = 0;
        idx
    }

    pub fn insert_protected(
        &mut self,
        key_pnc: &[f32],
        val: &[f32],
        surprise: f32,
        current_step: usize,
    ) -> Option<usize> {
        if self.count < self.capacity {
            let idx = self.insert(key_pnc, val);
            self.last_seen_step[idx] = current_step;
            return Some(idx);
        }
        let idx = self.write_head;
        match RateLimiterGate::apply_refractory_overwrite(
            &mut self.confidence[idx],
            &mut self.last_seen_step[idx],
            current_step,
            surprise,
        ) {
            UpdateOutcome::Defended { .. } | UpdateOutcome::Stable { .. } => None,
            UpdateOutcome::Overwritten => {
                let inserted = self.insert(key_pnc, val);
                self.last_seen_step[inserted] = current_step;
                Some(inserted)
            }
        }
    }

    pub fn retrieve_soft_into(
        &self,
        q_pnc: &[f32],
        tau: f32,
        out_val: &mut [f32],
        out_weights: &mut [f32],
    ) -> f32 {
        assert_eq!(q_pnc.len(), self.dim_key);
        assert_eq!(out_val.len(), self.dim_val);
        assert!(out_weights.len() >= self.count);
        assert!(
            tau.is_finite() && tau > 0.0,
            "tau must be positive and finite"
        );
        let q_sq = dot_slice(q_pnc, q_pnc);
        assert!(
            q_sq.is_finite() && q_sq < 1.0,
            "query must be in open Poincare ball"
        );
        if self.count == 0 {
            out_val.fill(0.0);
            out_weights.fill(0.0);
            return 0.0;
        }
        let mut min_dist = f32::MAX;
        let mut max_score = f32::NEG_INFINITY;
        for idx in 0..self.count {
            let off = idx * self.dim_key;
            let dist = Self::poincare_distance(
                q_pnc,
                q_sq,
                &self.keys[off..off + self.dim_key],
                self.norm_sq[idx],
            );
            min_dist = min_dist.min(dist);
            let score = -dist / tau;
            out_weights[idx] = score;
            max_score = max_score.max(score);
        }
        let mut sum = 0.0;
        for w in &mut out_weights[..self.count] {
            *w = (*w - max_score).exp();
            sum += *w;
        }
        out_val.fill(0.0);
        for idx in 0..self.count {
            let w = out_weights[idx] / sum;
            out_weights[idx] = w;
            let off = idx * self.dim_val;
            for j in 0..self.dim_val {
                out_val[j] += w * self.values[off + j];
            }
        }
        min_dist
    }
}
