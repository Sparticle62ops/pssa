//! Runtime-only offline sleep/dream replay summaries.
//!
//! Dream controls deliberately live outside the checkpoint model state.  A
//! checkpoint therefore remains byte-compatible whether or not a caller uses
//! the training-time replay phase.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DreamSummary {
    pub entries_replayed: usize,
    pub generated_tokens: usize,
    pub consolidation_delta_norm: f32,
    pub elapsed_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::DreamSummary;
    use crate::linalg::SimpleRng;
    use crate::pssa::{PSSAConfigV2, PSSALayerV2};

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
        assert_eq!(sa.consolidation_delta_norm.to_bits(), sb.consolidation_delta_norm.to_bits());
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
    fn summary_defaults_to_empty() {
        assert_eq!(DreamSummary::default().entries_replayed, 0);
    }
}
