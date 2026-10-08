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

/// Default learning rate for the separate, output-row SGD rehearsal optimizer.
/// Selected by the five-seed forgetting probe with projected fresh-task guards;
/// it does not advance Adam's moments or learning-rate schedule.
pub const DEFAULT_REHEARSAL_LR: f32 = 6e-3;
pub const DEFAULT_REHEARSAL_STEPS: usize = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DreamSummary {
    pub entries_replayed: usize,
    pub generated_tokens: usize,
    /// Number of supervised token sequences replayed through the main model.
    pub rehearsal_sequences: usize,
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

    fn parameter_bits(model: &mut PSSALayerV2) -> Vec<u32> {
        model
            .adam_tensors()
            .into_iter()
            .flat_map(|p| p.data.iter().map(|x| x.to_bits()))
            .collect()
    }

    fn optimizer_bits(model: &mut PSSALayerV2) -> Vec<u32> {
        let mut bits = Vec::new();
        for p in model.adam_tensors() {
            bits.extend(p.grad.iter().map(|x| x.to_bits()));
            bits.extend(p.m.iter().map(|x| x.to_bits()));
            bits.extend(p.v.iter().map(|x| x.to_bits()));
        }
        bits
    }

    fn carry(model: &PSSALayerV2) -> Vec<f32> {
        let mut state = vec![0.0; model.recurrent_state_len()];
        model.copy_recurrent_state_to(&mut state);
        state
    }

    fn assert_training_state_identical(a: &mut PSSALayerV2, b: &mut PSSALayerV2) {
        assert_eq!(parameter_bits(a), parameter_bits(b));
        assert_eq!(optimizer_bits(a), optimizer_bits(b));
        assert_eq!(a.step_counter, b.step_counter);
        assert_eq!(a.rng.state, b.rng.state);
        assert_eq!(a.embed_row_marks, b.embed_row_marks);
        assert_eq!(
            carry(a).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            carry(b).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for (a, b) in std::iter::once(&a.block)
            .chain(&a.extra_blocks)
            .zip(std::iter::once(&b.block).chain(&b.extra_blocks))
        {
            assert_eq!(a.memory, b.memory);
            assert_eq!(
                a.adapters[0]
                    .consolidated_up
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                b.adapters[0]
                    .consolidated_up
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
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
    fn guarded_replay_projects_and_backtracks_without_advancing_adam() {
        let mut model = seeded_model();
        let inputs = [1usize, 2, 1, 2];
        let targets = [2usize, 1, 2, 1];
        model.remember_dream_sequence(&inputs, &targets);
        model.reset_recurrent_state();
        let before_loss = model.forward_train_chunk(&inputs, &targets);
        model.zero_gradients();
        let before = main_weight_bits(&model);
        let before_m = model.unembed_w.m.clone();
        let before_v = model.unembed_w.v.clone();
        let step = model.step_counter;
        let summary = model.dream_replay_with_options_and_guard(
            DreamMode::Memory,
            1,
            0,
            0.8,
            1e-3,
            1,
            Some((&inputs, &targets)),
            &mut SimpleRng::new(91),
        );
        assert_eq!(summary.rehearsal_sequences, 1);
        assert_eq!(model.step_counter, step);
        assert_eq!(model.unembed_w.m, before_m);
        assert_eq!(model.unembed_w.v, before_v);
        assert_ne!(main_weight_bits(&model), before);
        model.reset_recurrent_state();
        let after_loss = model.forward_train_chunk(&inputs, &targets);
        assert!(after_loss <= before_loss + 1e-5);
        model.zero_gradients();
    }

    #[test]
    fn generated_dream_replays_sampled_tokens_into_main_weights() {
        let mut model = seeded_model();
        let before = main_weight_bits(&model);
        let optimizer = optimizer_bits(&mut model);
        let step = model.step_counter;
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
        assert_eq!(summary.rehearsal_sequences, 1);
        assert!(summary.consolidation_delta_norm > 0.0);
        assert_ne!(main_weight_bits(&model), before);
        assert_eq!(optimizer_bits(&mut model), optimizer);
        assert_eq!(model.step_counter, step);
        let mut restored = vec![0.0; model.recurrent_state_len()];
        model.copy_recurrent_state_to(&mut restored);
        assert_eq!(restored, carry);
        assert_ne!(model.block.adapters[0].up_proj.data, fast);
        assert_ne!(model.block.adapters[0].consolidated_up, slow);
    }

    #[test]
    fn generated_mode_does_not_rehearse_stored_sequences_without_memory_seeds() {
        let mut model = model();
        model.remember_dream_sequence(&[1, 2, 1, 2], &[2, 1, 2, 1]);
        let before = main_weight_bits(&model);
        let summary = model.dream_replay(DreamMode::Generate, 1, 4, 0.8, &mut SimpleRng::new(78));
        assert_eq!(summary.entries_replayed, 0);
        assert_eq!(summary.generated_tokens, 0);
        assert_eq!(summary.rehearsal_sequences, 0);
        assert_eq!(main_weight_bits(&model), before);
    }

    #[test]
    fn stored_sgd_preserves_carry_adam_and_gradients() {
        let mut model = seeded_model();
        model.remember_dream_sequence(&[1, 2, 1, 2], &[2, 1, 2, 1]);
        model.step_counter = 3;
        model.unembed_w.grad[0] = 0.125;
        model.unembed_w.m[0] = -0.25;
        model.unembed_w.v[1] = 0.5;
        let before = main_weight_bits(&model);
        let optimizer = optimizer_bits(&mut model);
        let marks = model.embed_row_marks.clone();
        let carry = {
            let mut state = vec![0.0; model.recurrent_state_len()];
            model.copy_recurrent_state_to(&mut state);
            state
        };
        let summary = model.dream_replay_with_options(
            DreamMode::Memory,
            1,
            0,
            0.8,
            6e-3,
            1,
            &mut SimpleRng::new(79),
        );
        assert_eq!(summary.rehearsal_sequences, 1);
        assert_ne!(main_weight_bits(&model), before);
        assert_eq!(optimizer_bits(&mut model), optimizer);
        assert_eq!(model.embed_row_marks, marks);
        assert_eq!(model.step_counter, 3);
        let mut restored = vec![0.0; model.recurrent_state_len()];
        model.copy_recurrent_state_to(&mut restored);
        assert_eq!(restored, carry);
    }

    #[test]
    fn stored_rehearsal_matches_supervised_output_row_sgd_exactly() {
        let cfg = PSSAConfigV2 {
            depth: 2,
            ..model().cfg
        };
        let mut replayed = PSSALayerV2::new(cfg.clone(), 17);
        let mut reference = PSSALayerV2::new(cfg, 17);
        let inputs = [1, 2, 1, 2];
        let targets = [2, 1, 2, 1];
        replayed.remember_dream_sequence(&inputs, &targets);
        // No latent entries: isolate real supervised rehearsal from the
        // historical adapter-only replay. Nonzero carry must be held fixed.
        let state = vec![0.125; replayed.recurrent_state_len()];
        replayed.copy_recurrent_state_from(&state);
        reference.copy_recurrent_state_from(&state);
        let before_loss = reference.forward_train_chunk(&inputs, &targets);
        let before_weights = reference.unembed_w.data.clone();
        let lr = 6e-3;
        for _ in 0..3 {
            reference.copy_recurrent_state_from(&state);
            reference.forward_train_chunk(&inputs, &targets);
            reference.zero_gradients();
            reference.backward_chunk(inputs.len(), 1.0);
            for target in [1, 2] {
                let start = target * reference.cfg.d_latent;
                for i in start..start + reference.cfg.d_latent {
                    reference.unembed_w.data[i] -= lr * reference.unembed_w.grad[i];
                }
            }
            reference.zero_gradients();
        }
        // CPU embedding row marks are scratch metadata for the sparse GPU
        // path; the replay API restores them to their entry state.
        reference.embed_row_marks.fill(0);
        reference.copy_recurrent_state_from(&state);
        let summary = replayed.dream_replay_with_options(
            DreamMode::Memory,
            1,
            0,
            0.8,
            lr,
            3,
            &mut SimpleRng::new(81),
        );
        assert_eq!(summary.entries_replayed, 0);
        assert_eq!(summary.generated_tokens, 0);
        assert_eq!(summary.rehearsal_sequences, 1);
        assert_ne!(replayed.unembed_w.data, before_weights);
        assert_training_state_identical(&mut replayed, &mut reference);
        let after_loss = replayed.forward_train_chunk(&inputs, &targets);
        assert!(after_loss < before_loss, "{before_loss} -> {after_loss}");
    }

    #[test]
    fn guarded_conflicting_replay_prevents_large_fresh_loss_regression() {
        let mut guarded = model();
        let mut unguarded = model();
        let inputs = [1, 2, 1, 2];
        let old_targets = [2, 1, 2, 1];
        let fresh_targets = [3, 4, 3, 4];
        for m in [&mut guarded, &mut unguarded] {
            m.remember_dream_sequence(&inputs, &old_targets);
            // Cover every optimizer tensor, including norm vectors, with
            // pending gradients and nonzero moments; replay must restore all.
            for p in m.adam_tensors() {
                p.grad.fill(-0.125);
                p.m.fill(0.25);
                p.v.fill(0.5);
            }
            m.step_counter = 3;
        }
        let state = carry(&guarded);
        let before_loss = guarded.forward_train_chunk(&inputs, &fresh_targets);
        guarded.copy_recurrent_state_from(&state);
        let optimizer = optimizer_bits(&mut guarded);
        let marks = guarded.embed_row_marks.clone();
        guarded.dream_replay_with_options_and_guard(
            DreamMode::Memory,
            1,
            0,
            0.8,
            1_000.0,
            2,
            Some((&inputs, &fresh_targets)),
            &mut SimpleRng::new(82),
        );
        assert_eq!(optimizer_bits(&mut guarded), optimizer);
        assert_eq!(guarded.step_counter, 3);
        assert_eq!(guarded.embed_row_marks, marks);
        assert_eq!(carry(&guarded), state);
        let after_loss = guarded.forward_train_chunk(&inputs, &fresh_targets);
        assert!(
            after_loss <= before_loss + 3e-6,
            "{before_loss} -> {after_loss}"
        );
        unguarded.dream_replay_with_options(
            DreamMode::Memory,
            1,
            0,
            0.8,
            1_000.0,
            2,
            &mut SimpleRng::new(82),
        );
        let unguarded_loss = unguarded.forward_train_chunk(&inputs, &fresh_targets);
        assert!(unguarded_loss > before_loss + 0.1);
    }

    #[test]
    fn zero_replay_preserves_the_entire_training_trajectory_and_rng() {
        let mut baseline = seeded_model();
        let mut disabled = seeded_model();
        let inputs = [1, 2, 1, 2];
        let targets = [2, 1, 2, 1];
        disabled.remember_dream_sequence(&inputs, &targets);
        let mut rng = SimpleRng::new(83);
        let initial_rng = rng.state;
        for _ in 0..4 {
            for m in [&mut baseline, &mut disabled] {
                m.forward_train_chunk(&inputs, &targets);
                m.zero_gradients();
                m.backward_chunk(inputs.len(), 1.0);
                m.apply_adamw(5e-3);
            }
            let summary = disabled.dream_replay_with_options_and_guard(
                DreamMode::Both,
                0,
                4,
                0.8,
                6e-3,
                1,
                Some((&inputs, &targets)),
                &mut rng,
            );
            assert_eq!(summary.rehearsal_sequences, 0);
            assert_eq!(summary.generated_tokens, 0);
            assert_eq!(summary.entries_replayed, 0);
            assert_eq!(rng.state, initial_rng);
            assert_training_state_identical(&mut baseline, &mut disabled);
        }
    }

    #[test]
    fn dream_off_cli_training_and_checkpoints_are_bit_identical() {
        use crate::cli::{CLIHandler, TrainingBackend, TrainingOptions};
        use crate::dataset::TokenizerKind;
        let raw = (0..40)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let raw = format!("{raw}\n\n{raw}");
        for (depth, loops, batch_size) in [(1, 1, 1), (2, 1, 2), (1, 2, 2)] {
            let options = TrainingOptions {
                backend: TrainingBackend::Cpu,
                tokenizer: TokenizerKind::Word,
                latent: 4,
                state: 2,
                key: 2,
                memory: 4,
                chunk: 4,
                accumulate: 2,
                epochs: 2,
                depth,
                loops,
                batch_size,
                no_tui: true,
                ..Default::default()
            };
            assert_eq!(options.dream_every, 0);
            let (mut baseline, _) = CLIHandler::train_corpus(&raw, &options).unwrap();
            let (mut disabled, _) = CLIHandler::train_corpus(
                &raw,
                &TrainingOptions {
                    dream_every: 0,
                    dream_mode: DreamMode::Both,
                    dream_replay: 2,
                    dream_len: 3,
                    dream_lr: 0.1,
                    dream_steps: 3,
                    ..options
                },
            )
            .unwrap();
            assert!(baseline.step_counter > 1);
            assert!(baseline.block.memory.count > 0);
            assert_training_state_identical(&mut baseline, &mut disabled);
            let path = std::env::temp_dir().join(format!(
                "pssa-dream-off-{}-{depth}-{loops}-{batch_size}.pssa",
                std::process::id()
            ));
            crate::checkpoint::save_model(&baseline, &path).unwrap();
            let expected = std::fs::read(&path).unwrap();
            crate::checkpoint::save_model(&disabled, &path).unwrap();
            let actual = std::fs::read(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            assert_eq!(actual, expected);
        }
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
