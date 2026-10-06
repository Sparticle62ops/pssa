//! Failure-only activation diagnostics and opt-in CUDA wall-time tracing.
//! No normal-path tensor scans and no changes to CPU training arithmetic.
use crate::{backend::Device, pssa::PSSALayerV2};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;
use std::time::Instant;

/// Scan the active tape in forward dependency order, only after the cheap
/// scalar loss check failed. Packed lanes publish their SSM output in y_ssm;
/// their recurrent state tapes are lane-owned, not the model's h_states.
pub(crate) fn nonfinite_forward(m: &PSSALayerV2, tokens: usize) -> String {
    for (layer, block) in std::iter::once(&m.block).chain(&m.extra_blocks).enumerate() {
        let d = block.cfg.d_latent;
        let s = block.cfg.d_state;
        let k = block.cfg.d_mem_key;
        let cap = block.cfg.mem_capacity;
        let rank = block.adapters[0].rank;
        let tape = &block.tape;
        for (stage, name, data) in [
            ("embed_norm", "x_raw", &tape.x_raw[..tokens * d]),
            ("embed_norm", "inv_rms", &tape.inv_rms[..tokens]),
            ("embed_norm", "x_norm", &tape.x_norm[..tokens * d]),
            ("projections", "delta_raw", &tape.delta_raw[..tokens * d]),
            ("projections", "delta", &tape.delta[..tokens * d]),
            ("projections", "b_proj", &tape.b_proj[..tokens * s]),
            ("projections", "c_proj", &tape.c_proj[..tokens * s]),
            ("ssm", "y_ssm", &tape.y_ssm[..tokens * d]),
            ("memory", "q_euc", &tape.q_euc[..tokens * k]),
            ("memory", "q_poincare", &tape.q_poincare[..tokens * k]),
            ("memory", "q_norm", &tape.q_norm[..tokens]),
            ("memory", "mem_weights", &tape.mem_weights[..tokens * cap]),
            ("memory", "m_val", &tape.m_val[..tokens * d]),
            ("memory", "g_mem", &tape.g_mem[..tokens * d]),
            ("memory", "m_proj", &tape.m_proj[..tokens * d]),
            ("memory", "m_inj", &tape.m_inj[..tokens * d]),
            (
                "adapter",
                "adapter_hidden",
                &tape.adapter_hidden[..tokens * rank],
            ),
            ("mlp", "adapter_act", &tape.adapter_act[..tokens * rank]),
            ("mlp", "z_raw", &tape.z_raw[..tokens * d]),
            ("mlp", "mlp_hidden", &tape.mlp_hidden[..tokens * 2 * d]),
            ("mlp", "mlp_act", &tape.mlp_act[..tokens * 2 * d]),
            ("mlp", "z_final", &tape.z_final[..tokens * d]),
        ] {
            if let Some((index, value)) = data.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                return format!(
                    "stage={stage} layer={layer} tensor={name} index={index} value={value}"
                );
            }
        }
    }
    let v = m.cfg.d_vocab;
    for (name, data) in [
        ("logits", &m.tape.logits[..tokens * v]),
        ("probs", &m.tape.probs[..tokens * v]),
        ("losses", &m.tape.losses[..tokens]),
    ] {
        if let Some((index, value)) = data.iter().enumerate().find(|(_, x)| !x.is_finite()) {
            return format!("stage=logits_loss tensor={name} index={index} value={value}");
        }
    }
    "stage=logits_loss tensor=mean_loss (finite per-token losses overflowed their sum)".into()
}

pub(crate) fn loss_error(m: &PSSALayerV2, tokens: usize) -> String {
    format!(
        "non-finite loss; {}; training aborted without checkpoint",
        nonfinite_forward(m, tokens)
    )
}

/// Enable with PSSA_CUDA_PROFILE=1. Begin lines locate even a stuck first step;
/// end lines include host work, copies and the existing synchronization waits.
/// These are wall times, not device-event kernel timings.
pub(crate) struct StageTrace {
    started: Option<Instant>,
    stage: &'static str,
    rows: usize,
}

impl StageTrace {
    pub(crate) fn new(device: &Device, stage: &'static str, rows: usize) -> Self {
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            return Self::cuda(stage, rows);
        }
        let _ = device;
        Self {
            started: None,
            stage,
            rows,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(stage: &'static str, rows: usize) -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let enabled = *ENABLED
            .get_or_init(|| std::env::var_os("PSSA_CUDA_PROFILE").is_some_and(|v| v == "1"));
        let started = enabled.then(|| {
            eprintln!("cuda_profile stage={stage} rows={rows} phase=begin");
            Instant::now()
        });
        Self {
            started,
            stage,
            rows,
        }
    }
}

impl Drop for StageTrace {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            eprintln!(
                "cuda_profile stage={} rows={} phase=end wall_ms={:.3}",
                self.stage,
                self.rows,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pssa::PSSAConfigV2;

    #[test]
    fn first_nonfinite_stage_is_ordered_and_ignores_inactive_rows() {
        let mut m = PSSALayerV2::new(
            PSSAConfigV2 {
                d_latent: 4,
                d_state: 2,
                d_vocab: 5,
                d_mem_key: 2,
                mem_capacity: 3,
                chunk_len: 3,
                ..Default::default()
            },
            7,
        );
        m.tape.delta_raw[8] = f32::NAN; // inactive token: must not be reported
        m.tape.m_proj[5] = f32::INFINITY;
        m.tape.logits[0] = f32::NAN;
        assert!(nonfinite_forward(&m, 2).contains("stage=memory layer=0 tensor=m_proj index=5"));
        m.tape.y_ssm[1] = f32::NEG_INFINITY;
        assert!(nonfinite_forward(&m, 2).contains("stage=ssm layer=0 tensor=y_ssm index=1"));
        m.tape.delta[0] = f32::INFINITY;
        assert!(
            nonfinite_forward(&m, 2).contains("stage=projections layer=0 tensor=delta index=0")
        );
    }
}
