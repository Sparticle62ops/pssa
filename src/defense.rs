#[derive(Debug, PartialEq, Clone)]
pub enum UpdateOutcome {
    Stable {
        new_conf: f32,
        gain_applied: f32,
    },
    Defended {
        remaining_conf: f32,
        damage_absorbed: f32,
    },
    Overwritten,
}

pub struct RateLimiterGate;

impl RateLimiterGate {
    #[inline(always)]
    pub fn compute_refractory_gain(delta_tokens: usize, tau_refractory: f32) -> f32 {
        let dt = delta_tokens as f32;
        let linear_factor = 1.0 - (-dt / tau_refractory).exp();
        (linear_factor * linear_factor).max(0.005)
    }

    #[inline(always)]
    pub fn reinforce_stable(
        conf: &mut f32,
        last_seen_step: &mut usize,
        current_step: usize,
        eta: f32,
    ) -> UpdateOutcome {
        let delta_t = current_step.saturating_sub(*last_seen_step);
        let gain_mult = Self::compute_refractory_gain(delta_t, 30.0);
        let gain = eta * gain_mult.max(0.20);

        *conf = (*conf + gain).min(5.0);
        *last_seen_step = current_step;

        UpdateOutcome::Stable {
            new_conf: *conf,
            gain_applied: gain,
        }
    }

    #[inline(always)]
    pub fn apply_refractory_overwrite(
        conf: &mut f32,
        last_seen_step: &mut usize,
        current_step: usize,
        surprise: f32,
    ) -> UpdateOutcome {
        let delta_t = current_step.saturating_sub(*last_seen_step);
        let gain_mult = Self::compute_refractory_gain(delta_t, 60.0);
        let damage = 0.08 * surprise * gain_mult;

        if *conf > damage {
            *conf -= damage;
            *last_seen_step = current_step;
            UpdateOutcome::Defended {
                remaining_conf: *conf,
                damage_absorbed: damage,
            }
        } else {
            *conf = 1.0;
            *last_seen_step = current_step;
            UpdateOutcome::Overwritten
        }
    }
}
