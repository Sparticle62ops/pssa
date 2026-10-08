use crate::backend::Device;
use crate::defense::{RateLimiterGate, UpdateOutcome};

#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicEpisodicBankV2 {
    pub capacity: usize,
    pub count: usize,
    pub dim_key: usize,
    pub dim_val: usize,
    pub write_head: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    /// Runtime-only Euclidean value norm limit; configure with `set_value_cap`.
    pub value_cap: Option<f32>,
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
            value_cap: None,
            norm_sq: vec![0.0; capacity],
            confidence: vec![1.0; capacity],
            last_seen_step: vec![0; capacity],
        }
    }

    /// Set an opt-in Euclidean norm cap for each stored value vector.
    /// Enabling or tightening it clamps existing values, including unused slots,
    /// without changing keys or their metadata. Call again after checkpoint load:
    /// the cap is runtime-only and is not part of the checkpoint format.
    /// Disabling it preserves current values and restores exact-copy writes.
    ///
    /// Panics if the cap is not finite and positive, or capped values are non-finite.
    pub fn set_value_cap(&mut self, value_cap: Option<f32>) {
        self.set_value_cap_with_device(value_cap, &Device::Cpu);
    }

    pub(crate) fn set_value_cap_with_device(&mut self, value_cap: Option<f32>, device: &Device) {
        if let Some(cap) = value_cap {
            assert!(
                cap.is_finite() && cap > 0.0,
                "memory value cap must be positive and finite"
            );
            Self::clamp_values_with_device(&mut self.values, self.dim_val, cap, device);
        }
        self.value_cap = value_cap;
    }

    fn clamp_values_with_device(values: &mut [f32], width: usize, cap: f32, device: &Device) {
        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(ctx) => ctx
                .cap_memory_values(values, width, cap)
                .expect("CUDA memory value cap failed; refusing a silent CPU fallback"),
            _ => {
                for value in values.chunks_exact_mut(width) {
                    Self::clamp_value(value, cap);
                }
            }
        }
    }

    fn clamp_value(value: &mut [f32], cap: f32) {
        let norm_sq = Self::squared_norm_f64(value);
        assert!(norm_sq.is_finite(), "capped memory value must be finite");
        let cap_sq = (cap as f64) * (cap as f64);
        if norm_sq <= cap_sq {
            return;
        }
        let scale = cap as f64 / norm_sq.sqrt();
        for x in value.iter_mut() {
            *x = (*x as f64 * scale) as f32;
        }
        // Nearest-f32 rounding can push the norm above the cap. Move components
        // one ulp toward zero until it fits; unlike multiplicative headroom, this
        // also works for subnormal caps/components, where a rescale may round away.
        while Self::squared_norm_f64(value) > cap_sq {
            for x in value.iter_mut() {
                if *x != 0.0 {
                    *x = f32::from_bits(x.to_bits() - 1);
                }
            }
        }
    }

    // Leave eight f32 ulps of radial headroom, including output rounding.
    // Beyond this radius the radial derivative is zero; tangential motion remains.
    const MAX_PROJECTED_RADIUS: f64 = 1.0 - 8.0 * f32::EPSILON as f64;

    /// Accumulate in f64 so finite f32 coordinates cannot overflow their norm.
    /// Preserve open-ball membership when an old, valid near-boundary key's
    /// f64 norm would round up to 1 in f32. Points actually on/outside the ball
    /// are never clamped inwards. This also supports legacy checkpoint norms.
    pub fn squared_norm(point: &[f32]) -> f32 {
        let norm = Self::squared_norm_f64(point);
        if norm < 1.0 && norm as f32 == 1.0 {
            f32::from_bits(1.0f32.to_bits() - 1)
        } else {
            norm as f32
        }
    }

    fn squared_norm_f64(point: &[f32]) -> f64 {
        point.iter().map(|&x| (x as f64) * (x as f64)).sum()
    }

    /// q / (1 + ||q||), radially saturated at a representable open-ball radius.
    /// The returned diagnostic norm saturates at f32::MAX; the adjoint computes
    /// its own f64 norm rather than relying on that potentially saturated value.
    #[inline]
    pub fn diffeomorphic_project(q_euc: &[f32], out_pnc: &mut [f32]) -> f32 {
        assert!(!q_euc.is_empty() && q_euc.len() == out_pnc.len());
        let r = Self::squared_norm_f64(q_euc).sqrt();
        assert!(r.is_finite(), "projection input must be finite");
        let scale = if r == 0.0 {
            1.0
        } else {
            (r / (1.0 + r)).min(Self::MAX_PROJECTED_RADIUS) / r
        };
        for (out, &q) in out_pnc.iter_mut().zip(q_euc) {
            *out = (q as f64 * scale) as f32;
        }
        r.min(f32::MAX as f64) as f32
    }

    /// VJP of the same radial map, including its saturated branch. Using unit
    /// directions avoids overflowing q·g or r*(1+r)^2 at finite extreme inputs.
    pub fn projection_adjoint(q_euc: &[f32], grad: &[f32], out: &mut [f32]) {
        assert_eq!(q_euc.len(), grad.len());
        assert_eq!(q_euc.len(), out.len());
        let r = Self::squared_norm_f64(q_euc).sqrt();
        assert!(r.is_finite(), "projection input must be finite");
        if r == 0.0 {
            out.copy_from_slice(grad);
            return;
        }
        let radius = r / (1.0 + r);
        let saturated = radius >= Self::MAX_PROJECTED_RADIUS;
        let scale = if saturated {
            Self::MAX_PROJECTED_RADIUS / r
        } else {
            1.0 / (1.0 + r)
        };
        let radial = if saturated { 1.0 } else { radius };
        let unit_dot_grad: f64 = q_euc
            .iter()
            .zip(grad)
            .map(|(&q, &g)| (q as f64 / r) * g as f64)
            .sum();
        for ((out, &q), &g) in out.iter_mut().zip(q_euc).zip(grad) {
            *out = (scale * (g as f64 - radial * (q as f64 / r) * unit_dot_grad)) as f32;
        }
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
            let d = u[i] as f64 - v[i] as f64;
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
        self.insert_with_device(key_pnc, val, &Device::Cpu)
    }

    fn insert_with_device(&mut self, key_pnc: &[f32], val: &[f32], device: &Device) -> usize {
        assert_eq!(key_pnc.len(), self.dim_key);
        assert_eq!(val.len(), self.dim_val);
        let key_sq = Self::squared_norm(key_pnc);
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
        if let Some(cap) = self.value_cap {
            Self::clamp_values_with_device(
                &mut self.values[v_off..v_off + self.dim_val],
                self.dim_val,
                cap,
                device,
            );
        }
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
        self.insert_protected_with_device(key_pnc, val, surprise, current_step, &Device::Cpu)
    }

    pub(crate) fn insert_protected_with_device(
        &mut self,
        key_pnc: &[f32],
        val: &[f32],
        surprise: f32,
        current_step: usize,
        device: &Device,
    ) -> Option<usize> {
        if self.count < self.capacity {
            let idx = self.insert_with_device(key_pnc, val, device);
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
                let inserted = self.insert_with_device(key_pnc, val, device);
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
        let q_sq = Self::squared_norm(q_pnc);
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
        for idx in 0..self.count {
            let off = idx * self.dim_key;
            let dist = Self::poincare_distance(
                q_pnc,
                q_sq,
                &self.keys[off..off + self.dim_key],
                self.norm_sq[idx],
            );
            min_dist = min_dist.min(dist);
            out_weights[idx] = dist;
        }
        let mut sum = 0.0;
        for w in &mut out_weights[..self.count] {
            // Subtract before dividing: even a subnormal tau leaves the nearest
            // entry at exp(0), rather than subtracting two -infinity scores.
            *w = ((min_dist - *w) / tau).exp();
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

    /// Control read for feature benchmarks: cosine similarity in the stored
    /// Euclidean coordinates. This is deliberately an opt-in helper; the
    /// production PSSA path continues to use `retrieve_soft_into` above.
    pub fn retrieve_soft_euclidean_into(
        &self,
        q: &[f32],
        tau: f32,
        out_val: &mut [f32],
        out_weights: &mut [f32],
    ) -> f32 {
        assert_eq!(q.len(), self.dim_key);
        assert_eq!(out_val.len(), self.dim_val);
        assert!(out_weights.len() >= self.count);
        assert!(tau.is_finite() && tau > 0.0);
        if self.count == 0 {
            out_val.fill(0.0);
            out_weights.fill(0.0);
            return 0.0;
        }
        let q_norm = q
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let mut max_distance = f32::NEG_INFINITY;
        for idx in 0..self.count {
            let off = idx * self.dim_key;
            let (dot, k_norm_sq) = q
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(d, n), (j, x)| {
                    let k = self.keys[off + j] as f64;
                    (d + *x as f64 * k, n + k * k)
                });
            let cosine = if q_norm > 0.0 && k_norm_sq > 0.0 {
                dot / (q_norm * k_norm_sq.sqrt())
            } else {
                0.0
            };
            let distance = (1.0 - cosine).clamp(0.0, 2.0) as f32;
            out_weights[idx] = -distance;
            max_distance = max_distance.max(-distance);
        }
        let mut sum = 0.0f32;
        for weight in &mut out_weights[..self.count] {
            *weight = ((*weight - max_distance) / tau).exp();
            sum += *weight;
        }
        out_val.fill(0.0);
        for idx in 0..self.count {
            let weight = out_weights[idx] / sum;
            out_weights[idx] = weight;
            let off = idx * self.dim_val;
            for j in 0..self.dim_val {
                out_val[j] += weight * self.values[off + j];
            }
        }
        -max_distance
    }
}

#[cfg(test)]
mod tests {
    use super::HyperbolicEpisodicBankV2 as Bank;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_capped(value: &[f32], cap: f32) {
        assert!(value.iter().all(|x| x.is_finite()));
        let norm_sq: f64 = value.iter().map(|&x| (x as f64) * (x as f64)).sum();
        assert!(norm_sq <= (cap as f64) * (cap as f64));
    }

    fn assert_metadata_eq(a: &Bank, b: &Bank) {
        assert_eq!(a.capacity, b.capacity);
        assert_eq!(a.count, b.count);
        assert_eq!(a.dim_key, b.dim_key);
        assert_eq!(a.dim_val, b.dim_val);
        assert_eq!(a.write_head, b.write_head);
        assert_eq!(bits(&a.keys), bits(&b.keys));
        assert_eq!(bits(&a.norm_sq), bits(&b.norm_sq));
        assert_eq!(bits(&a.confidence), bits(&b.confidence));
        assert_eq!(a.last_seen_step, b.last_seen_step);
    }

    #[test]
    fn value_cap_clamps_insert_and_ring_overwrite_without_changing_metadata() {
        let mut uncapped = Bank::new(2, 2, 2);
        let mut capped = uncapped.clone();
        capped.set_value_cap(Some(2.0));
        for (key, value) in [
            ([0.25, -0.125], [6.0, 8.0]),
            ([-0.5, 0.25], [0.0, -1.0]),
            ([0.125, 0.0], [30.0, -40.0]),
        ] {
            let idx = uncapped.insert(&key, &value);
            assert_eq!(capped.insert(&key, &value), idx);
            assert_metadata_eq(&capped, &uncapped);
            for stored in capped.values.chunks_exact(capped.dim_val) {
                assert_capped(stored, 2.0);
            }
        }
        assert_eq!(capped.write_head, 1);
        assert_eq!(&capped.values[2..4], &[0.0, -1.0]);
    }

    #[test]
    fn value_cap_clamps_protected_writes_but_not_defended_values() {
        let mut bank = Bank::new(1, 1, 2);
        bank.set_value_cap(Some(1.0));
        assert_eq!(bank.insert_protected(&[0.1], &[3.0, 4.0], 0.0, 1), Some(0));
        assert_capped(&bank.values, 1.0);
        let before = bank.clone();
        assert_eq!(bank.insert_protected(&[0.2], &[30.0, 40.0], 0.0, 2), None);
        assert_eq!(bits(&bank.values), bits(&before.values));
        assert_eq!(bits(&bank.keys), bits(&before.keys));
        assert_eq!(bits(&bank.norm_sq), bits(&before.norm_sq));
        assert_eq!(
            bank.insert_protected(&[0.2], &[-30.0, 40.0], 1e6, 1000),
            Some(0)
        );
        assert_capped(&bank.values, 1.0);
        assert!(bank.values[0] < 0.0);
        assert_eq!(bank.keys, vec![0.2]);
        assert_eq!(bank.norm_sq[0], Bank::squared_norm(&[0.2]));
        assert_eq!(bank.confidence, vec![1.0]);
        assert_eq!(bank.last_seen_step, vec![1000]);
    }

    #[test]
    fn enabling_value_cap_clamps_existing_and_loaded_unused_slots_only() {
        let mut bank = Bank::new(3, 2, 2);
        bank.insert(&[0.25, -0.5], &[3.0, 4.0]);
        bank.insert(&[-0.125, 0.25], &[0.125, -0.25]);
        // Checkpoints store all capacity slots, not just occupied slots.
        bank.values[4..6].copy_from_slice(&[10.0, -20.0]);
        bank.confidence[0] = 3.5;
        bank.last_seen_step[0] = 42;
        let before = bank.clone();
        bank.set_value_cap(Some(1.0));
        assert_eq!(bank.value_cap, Some(1.0));
        assert_metadata_eq(&bank, &before);
        for value in bank.values.chunks_exact(bank.dim_val) {
            assert_capped(value, 1.0);
        }
        assert_eq!(bits(&bank.values[2..4]), bits(&before.values[2..4]));
    }

    #[test]
    fn value_cap_can_be_tightened_and_disabled_without_restoring_values() {
        let mut bank = Bank::new(1, 1, 2);
        bank.insert(&[0.0], &[3.0, 4.0]);
        bank.set_value_cap(Some(4.0));
        assert_capped(&bank.values, 4.0);
        let capped = bits(&bank.values);
        bank.set_value_cap(Some(10.0));
        assert_eq!(bits(&bank.values), capped);
        bank.set_value_cap(Some(0.5));
        assert_capped(&bank.values, 0.5);
        let capped = bits(&bank.values);
        bank.set_value_cap(None);
        assert_eq!(bank.value_cap, None);
        assert_eq!(bits(&bank.values), capped);
        bank.insert(&[0.0], &[30.0, 40.0]);
        assert_eq!(bits(&bank.values), bits(&[30.0, 40.0]));
    }

    #[test]
    fn disabled_value_cap_preserves_exact_copy_bits() {
        let mut bank = Bank::new(1, 1, 8);
        assert_eq!(bank.value_cap, None);
        let value = [
            0.0,
            -0.0,
            f32::MAX,
            -f32::MAX,
            f32::from_bits(0x7fc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
        ];
        bank.insert(&[0.0], &value);
        assert_eq!(bits(&bank.values), bits(&value));
        bank.set_value_cap(None);
        assert_eq!(bits(&bank.values), bits(&value));
        let overwritten = value.map(|x| -x);
        bank.insert(&[0.1], &overwritten);
        assert_eq!(bits(&bank.values), bits(&overwritten));
    }

    #[test]
    fn enabled_value_cap_preserves_under_limit_bits() {
        let mut bank = Bank::new(1, 1, 3);
        bank.set_value_cap(Some(1.0));
        let value = [-0.0, f32::from_bits(1), 0.5];
        bank.insert(&[0.0], &value);
        assert_eq!(bits(&bank.values), bits(&value));
        bank.set_value_cap(Some(1.0));
        assert_eq!(bits(&bank.values), bits(&value));
    }

    #[test]
    fn value_cap_handles_extreme_finite_values_and_caps() {
        for cap in [1.0, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let mut bank = Bank::new(1, 1, 3);
            bank.set_value_cap(Some(cap));
            bank.insert(&[0.0], &[f32::MAX, -f32::MAX, f32::MAX]);
            assert_capped(&bank.values, cap);
        }
    }

    #[test]
    fn value_cap_corrects_normal_and_subnormal_rounding_overshoot() {
        // Direct nearest-f32 rounding of the normalized 3-4-5 vector overshoots.
        let rounded_norm_sq = (0.6f32 as f64).powi(2) + (0.8f32 as f64).powi(2);
        assert!(rounded_norm_sq > 1.0);
        let mut bank = Bank::new(1, 1, 2);
        bank.set_value_cap(Some(1.0));
        bank.insert(&[0.0], &[3.0, 4.0]);
        assert_capped(&bank.values, 1.0);
        assert!((bank.values[0] - 0.6).abs() <= f32::EPSILON);
        assert!((bank.values[1] - 0.8).abs() <= f32::EPSILON);

        let tiny = f32::from_bits(1);
        bank.set_value_cap(Some(tiny));
        bank.insert(&[0.0], &[tiny, -tiny]);
        assert_capped(&bank.values, tiny);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU; checks PTX cap rounding and protected-write semantics"]
    fn cuda_value_cap_matches_cpu_bits_and_metadata() {
        use crate::backend::Device;
        let ctx = crate::cuda::CudaContext::init().expect("CUDA GPU required");
        let device = Device::Cuda(ctx.clone());
        for cap in [1.0, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let mut old = Bank::new(3, 1, 3);
            old.insert(&[0.1], &[3.0, -4.0, 0.0]);
            old.values[3..6].copy_from_slice(&[f32::MAX, -f32::MAX, f32::MAX]);
            old.values[6..9].copy_from_slice(&[-0.0, f32::from_bits(1), 0.5]);
            let mut new = old.clone();
            old.set_value_cap(Some(cap));
            new.set_value_cap_with_device(Some(cap), &device);
            assert_eq!(bits(&new.values), bits(&old.values));
            assert_metadata_eq(&new, &old);
            for (step, surprise, value) in [
                (1, 0.0, [3.0, 4.0, 0.0]),
                (2, 0.0, [-0.0, f32::from_bits(1), 0.125]),
                (3, 0.0, [f32::NAN; 3]), // defended: must not cap/reject unused input
                (1000, 1e6, [f32::MAX, -f32::MAX, f32::MAX]),
            ] {
                let a = old.insert_protected(&[0.2], &value, surprise, step);
                let b = new.insert_protected_with_device(&[0.2], &value, surprise, step, &device);
                assert_eq!(a, b);
                assert_eq!(bits(&new.values), bits(&old.values));
                assert_metadata_eq(&new, &old);
                for value in new.values.chunks_exact(3) {
                    assert_capped(value, cap);
                }
            }
            old.set_value_cap(None);
            new.set_value_cap_with_device(None, &device);
            assert_eq!(new, old);
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(ctx.cap_memory_values(&mut [bad, 0.0], 2, 1.0).is_err());
        }
    }

    #[test]
    fn invalid_value_caps_are_rejected_without_changing_the_bank() {
        let mut bank = Bank::new(1, 1, 2);
        bank.insert(&[0.25], &[3.0, 4.0]);
        let before = bank.clone();
        for cap in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bank.set_value_cap(Some(cap));
            }));
            assert!(result.is_err());
            assert_eq!(bank, before);
        }
    }
}
