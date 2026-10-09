//! Runtime-only divergence detection; never serialized into checkpoints.
#[derive(Clone, Copy, Debug)]
pub struct LossGuardConfig {
    /// Maximum multiple of random-guess cross entropy, ln(vocabulary size).
    pub high_factor: f64,
    /// Maximum multiple of the preceding healthy running loss.
    pub jump_factor: f64,
    /// Consecutive suspect optimizer groups required before aborting.
    pub patience: usize,
}

impl Default for LossGuardConfig {
    fn default() -> Self {
        Self {
            high_factor: 4.0,
            jump_factor: 8.0,
            patience: 3,
        }
    }
}

impl LossGuardConfig {
    pub fn validate(self) -> Result<(), String> {
        for (flag, value) in [
            ("--loss-guard-high-factor", self.high_factor),
            ("--loss-guard-jump-factor", self.jump_factor),
        ] {
            if !value.is_finite() || value <= 1.0 {
                return Err(format!("{flag} must be finite and greater than 1"));
            }
        }
        if self.patience == 0 || self.patience > 1_000_000 {
            return Err("--loss-guard-patience must be between 1 and 1000000".into());
        }
        Ok(())
    }
}

pub struct LossGuard {
    config: LossGuardConfig,
    high_limit: f64,
    baseline: Option<f64>,
    consecutive: usize,
}

impl LossGuard {
    pub fn new(config: LossGuardConfig, vocab: usize) -> Result<Self, String> {
        config.validate()?;
        if vocab == 0 {
            return Err("loss guard requires a nonempty vocabulary".into());
        }
        Ok(Self {
            config,
            high_limit: config.high_factor * (vocab as f64).ln(),
            baseline: None,
            consecutive: 0,
        })
    }

    /// Observe one target-token-weighted group mean BEFORE Adam, including
    /// groups whose gradients will subsequently be skipped. A healthy EWMA
    /// (alpha=1/8) is frozen during a suspect streak, so a sustained plateau
    /// after a jump cannot raise its own reference and escape detection.
    pub fn observe(&mut self, loss: f64, update: usize) -> Result<(), String> {
        if !loss.is_finite() || loss < 0.0 {
            return Err(format!(
                "loss_guard=halted update_index={update} invalid loss={loss}; training aborted without checkpoint"
            ));
        }
        let jump_limit = self.baseline.map(|b| b.max(1e-6) * self.config.jump_factor);
        let suspect = loss > self.high_limit || jump_limit.is_some_and(|limit| loss > limit);
        if suspect {
            self.consecutive += 1;
            if self.consecutive >= self.config.patience {
                return Err(format!(
                    "loss_guard=halted update_index={update} loss={loss:.6e} high_limit={:.6e} jump_limit={} consecutive={} patience={}; training aborted without checkpoint",
                    self.high_limit,
                    jump_limit.map_or_else(|| "unavailable".into(), |x| format!("{x:.6e}")),
                    self.consecutive,
                    self.config.patience
                ));
            }
        } else {
            self.consecutive = 0;
            self.baseline = Some(self.baseline.map_or(loss, |b| b + (loss - b) / 8.0));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_blowup_from_first_step_halts_at_exact_patience() {
        let mut guard = LossGuard::new(LossGuardConfig::default(), 2048).unwrap();
        for step in 1..3 {
            guard.observe(28_000_000.0, step).unwrap();
        }
        let error = guard.observe(28_000_000.0, 3).unwrap_err();
        assert!(error.contains("update_index=3") && error.contains("without checkpoint"));
    }
    #[test]
    fn jump_plateau_cannot_contaminate_healthy_baseline() {
        let mut guard = LossGuard::new(
            LossGuardConfig {
                high_factor: 1000.0,
                jump_factor: 4.0,
                patience: 3,
            },
            100,
        )
        .unwrap();
        guard.observe(0.5, 1).unwrap();
        guard.observe(3.0, 2).unwrap();
        guard.observe(3.0, 3).unwrap();
        assert!(guard.observe(3.0, 4).is_err());
    }
    #[test]
    fn isolated_spikes_reset_and_threshold_is_strict() {
        let mut guard = LossGuard::new(
            LossGuardConfig {
                high_factor: 2.0,
                jump_factor: 4.0,
                patience: 2,
            },
            100,
        )
        .unwrap();
        for (i, loss) in [1.0, 20.0, 1.0, 4.0, 20.0, 1.0].into_iter().enumerate() {
            guard.observe(loss, i).unwrap();
        }
    }
    #[test]
    fn invalid_config_and_nonfinite_loss_are_rejected() {
        for value in [0.0, 1.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                LossGuardConfig {
                    high_factor: value,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
            assert!(
                LossGuardConfig {
                    jump_factor: value,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            LossGuardConfig {
                patience: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(LossGuard::new(Default::default(), 0).is_err());
        for loss in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(
                LossGuard::new(Default::default(), 2)
                    .unwrap()
                    .observe(loss, 1)
                    .is_err()
            );
        }
    }
}
