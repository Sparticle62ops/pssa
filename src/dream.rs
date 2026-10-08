//! Runtime-only offline sleep/dream replay summaries.
//!
//! Dream controls deliberately live outside the checkpoint model state.  A
//! checkpoint therefore remains byte-compatible whether or not a caller uses
//! the training-time replay phase.

use crate::linalg::SimpleRng;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DreamMode {
    #[default]
    Memory,
    Generate,
    Both,
}

impl DreamMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "memory" => Ok(Self::Memory),
            "generate" => Ok(Self::Generate),
            "both" => Ok(Self::Both),
            _ => Err(format!(
                "--dream-mode must be memory, generate, or both; got '{value}'"
            )),
        }
    }

    pub const fn includes_memory(self) -> bool {
        matches!(self, Self::Memory | Self::Both)
    }

    pub const fn includes_generation(self) -> bool {
        matches!(self, Self::Generate | Self::Both)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DreamSummary {
    pub entries_replayed: usize,
    pub generated_tokens: usize,
    pub consolidation_delta_norm: f32,
    pub elapsed_seconds: f64,
}

/// Sample a non-unknown vocabulary ID from a finite logit vector. Dream
/// generation deliberately keeps this sampler small and deterministic: it does
/// not share inference's top-k/top-p policy, so changing interactive generation
/// cannot change an offline training run.
/// The logit buffer is reused as sampling scratch and overwritten by the next
/// model forward pass.
pub(crate) fn sample_token(logits: &mut [f32], temperature: f32, rng: &mut SimpleRng) -> usize {
    assert!(!logits.is_empty());
    assert!(temperature.is_finite() && temperature > 0.0);
    let first = usize::from(logits.len() > 1);
    let max = logits[first..]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max.is_finite(),
        "dream generation received non-finite logits"
    );
    let mut total = 0.0f32;
    for logit in &mut logits[first..] {
        let weight = ((*logit - max) / temperature).exp();
        assert!(weight.is_finite());
        *logit = weight;
        total += weight;
    }
    assert!(total.is_finite() && total > 0.0);
    let draw = rng.gen_range_f32(0.0, total);
    let mut cumulative = 0.0;
    for (offset, &weight) in logits[first..].iter().enumerate() {
        cumulative += weight;
        if draw <= cumulative {
            return first + offset;
        }
    }
    logits.len() - 1
}

#[cfg(test)]
mod tests {
    use super::{DreamMode, DreamSummary};
    use crate::linalg::SimpleRng;
    use crate::pssa::{PSSAConfigV2, PSSALayerV2};

    #[test]
    fn sampler_preserves_seeded_draws_and_excludes_unknown_id() {
        // Captured from the allocating sampler before scratch-buffer reuse.
        let mut rng = SimpleRng::new(1234);
        let draws: Vec<_> = (0..16)
            .map(|_| {
                let mut logits = [100.0, -0.25, 0.0, 0.5, -0.5];
                super::sample_token(&mut logits, 0.8, &mut rng)
            })
            .collect();
        assert_eq!(draws, [3, 2, 2, 2, 4, 1, 3, 1, 2, 3, 2, 3, 2, 3, 3, 4]);
        assert_eq!(rng.state, 15862471842254482758);
        assert_eq!(super::sample_token(&mut [0.7], 0.8, &mut rng), 0);
    }

    fn model() -> PSSALayerV2 {
        PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: 5,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 4,
                chunk_len: 4,
                ..Default::default()
            },
            17,
        )
    }

    fn seeded_model() -> PSSALayerV2 {
        let mut model = model();
        model
            .block
            .memory
            .insert(&[0.1, -0.2], &[0.25, -0.4, 0.15, 0.3]);
        model
    }

    fn main_weight_bits(model: &PSSALayerV2) -> Vec<u32> {
        let mut bits = Vec::new();
        let mut matrix = |p: &crate::pssa::ParamMatrix| {
            bits.extend(p.data.iter().map(|x| x.to_bits()));
            bits.extend(p.grad.iter().map(|x| x.to_bits()));
            bits.extend(p.m.iter().map(|x| x.to_bits()));
            bits.extend(p.v.iter().map(|x| x.to_bits()));
        };
        matrix(&model.embed_w);
        matrix(&model.unembed_w);
        let b = &model.block;
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
        ] {
            matrix(p);
        }
        bits.extend(b.norm_gamma.data.iter().map(|x| x.to_bits()));
        bits.extend(b.norm_beta.data.iter().map(|x| x.to_bits()));
        bits.extend(b.h_persistent.iter().map(|x| x.to_bits()));
        bits.extend(b.memory.keys.iter().map(|x| x.to_bits()));
        bits.extend(b.memory.values.iter().map(|x| x.to_bits()));
        bits.extend(b.memory.norm_sq.iter().map(|x| x.to_bits()));
        bits.extend(b.memory.confidence.iter().map(|x| x.to_bits()));
        bits.extend(b.memory.last_seen_step.iter().copied().map(|x| x as u32));
        bits
    }

    #[test]
    fn dream_off_is_an_exact_no_op() {
        let mut model = seeded_model();
        let before = main_weight_bits(&model);
        let fast = model.block.adapters[0].up_proj.data.clone();
        let slow = model.block.adapters[0].consolidated_up.clone();
        let summary = model.dream_replay_memory(0, &mut SimpleRng::new(9));
        assert_eq!(summary.entries_replayed, 0);
        assert_eq!(main_weight_bits(&model), before);
        assert_eq!(model.block.adapters[0].up_proj.data, fast);
        assert_eq!(model.block.adapters[0].consolidated_up, slow);
    }

    #[test]
    fn empty_memory_dream_is_an_exact_no_op() {
        let mut model = model();
        model.block.adapters[0].up_proj.data[0] = 0.25;
        model.block.adapters[0].consolidated_up[0] = -0.5;
        let fast = model.block.adapters[0].up_proj.data.clone();
        let slow = model.block.adapters[0].consolidated_up.clone();
        let summary = model.dream_replay_memory(4, &mut SimpleRng::new(9));
        assert_eq!(summary.entries_replayed, 0);
        assert_eq!(model.block.adapters[0].up_proj.data, fast);
        assert_eq!(model.block.adapters[0].consolidated_up, slow);
    }

    #[test]
    fn memory_dream_changes_only_fast_and_consolidated_adapter_state() {
        let mut model = seeded_model();
        let before = main_weight_bits(&model);
        let fast = model.block.adapters[0].up_proj.data.clone();
        let slow = model.block.adapters[0].consolidated_up.clone();
        let summary = model.dream_replay_memory(1, &mut SimpleRng::new(9));
        assert_eq!(summary.entries_replayed, 1);
        assert!(summary.consolidation_delta_norm > 0.0);
        assert_eq!(main_weight_bits(&model), before);
        assert_ne!(model.block.adapters[0].up_proj.data, fast);
        assert_ne!(model.block.adapters[0].consolidated_up, slow);
    }

    #[test]
    fn memory_dream_is_deterministic_for_a_fixed_seed() {
        let mut a = seeded_model();
        let mut b = seeded_model();
        let sa = a.dream_replay_memory(1, &mut SimpleRng::new(1234));
        let sb = b.dream_replay_memory(1, &mut SimpleRng::new(1234));
        assert_eq!(sa.entries_replayed, sb.entries_replayed);
        assert_eq!(
            sa.consolidation_delta_norm.to_bits(),
            sb.consolidation_delta_norm.to_bits()
        );
        assert_eq!(
            a.block.adapters[0].up_proj.data,
            b.block.adapters[0].up_proj.data
        );
        assert_eq!(
            a.block.adapters[0].consolidated_up,
            b.block.adapters[0].consolidated_up
        );
    }

    #[test]
    fn generated_dream_replays_sampled_tokens_without_main_weight_updates() {
        let mut model = seeded_model();
        let before = main_weight_bits(&model);
        let carry = {
            let mut state = vec![0.0; model.recurrent_state_len()];
            model.copy_recurrent_state_to(&mut state);
            state
        };
        let fast = model.block.adapters[0].up_proj.data.clone();
        let slow = model.block.adapters[0].consolidated_up.clone();
        let summary = model.dream_replay(DreamMode::Generate, 1, 4, 0.8, &mut SimpleRng::new(77));
        assert_eq!(summary.entries_replayed, 1);
        assert_eq!(summary.generated_tokens, 4);
        assert!(summary.consolidation_delta_norm > 0.0);
        assert_eq!(main_weight_bits(&model), before);
        let mut restored = vec![0.0; model.recurrent_state_len()];
        model.copy_recurrent_state_to(&mut restored);
        assert_eq!(restored, carry);
        assert_ne!(model.block.adapters[0].up_proj.data, fast);
        assert_ne!(model.block.adapters[0].consolidated_up, slow);
    }

    #[test]
    fn generated_dream_is_deterministic_for_a_fixed_seed() {
        let mut a = seeded_model();
        let mut b = seeded_model();
        let sa = a.dream_replay(DreamMode::Generate, 1, 5, 0.8, &mut SimpleRng::new(88));
        let sb = b.dream_replay(DreamMode::Generate, 1, 5, 0.8, &mut SimpleRng::new(88));
        assert_eq!(sa.entries_replayed, sb.entries_replayed);
        assert_eq!(sa.generated_tokens, sb.generated_tokens);
        assert_eq!(
            sa.consolidation_delta_norm.to_bits(),
            sb.consolidation_delta_norm.to_bits()
        );
        assert_eq!(
            a.block.adapters[0].up_proj.data,
            b.block.adapters[0].up_proj.data
        );
        assert_eq!(
            a.block.adapters[0].consolidated_up,
            b.block.adapters[0].consolidated_up
        );
    }

    #[test]
    fn dream_mode_parser_rejects_unknown_sources() {
        assert_eq!(DreamMode::parse("memory"), Ok(DreamMode::Memory));
        assert_eq!(DreamMode::parse("generate"), Ok(DreamMode::Generate));
        assert_eq!(DreamMode::parse("both"), Ok(DreamMode::Both));
        assert!(DreamMode::parse("tokens").is_err());
    }

    #[test]
    fn summary_defaults_to_empty() {
        assert_eq!(DreamSummary::default().entries_replayed, 0);
    }
}
