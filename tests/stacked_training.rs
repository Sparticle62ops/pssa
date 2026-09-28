use oxide_ai_pssa::{
    checkpoint::{self, CheckpointFormat},
    cli::{CLIHandler, TrainingOptions},
    dataset::{Tokenizer, TokenizerKind},
    evaluation::{self, EvaluationSlice},
    gpu_batch,
    pssa::{PSSAConfigV2, PSSAContinuousBlockV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};
use std::{fs, path::PathBuf, process::Command, sync::atomic::{AtomicUsize, Ordering}};

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-stacked-training-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().into()
    }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn model(depth: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(PSSAConfigV2 {
        depth, d_vocab: 9, d_latent: 5, d_state: 2, d_mem_key: 3,
        mem_capacity: 5, chunk_len: 4, tau_mem: 0.7, ..Default::default()
    }, 17);
    for (layer, b) in std::iter::once(&mut m.block).chain(&mut m.extra_blocks).enumerate() {
        for (i, x) in b.mlp_w2.data.iter_mut().enumerate() {
            *x = ((i % 9) as f32 - 4.0) * 0.023;
        }
        for (i, x) in b.adapters[0].up_proj.data.iter_mut().enumerate() {
            *x = ((i % 7) as f32 - 3.0) * 0.019;
        }
        for (i, x) in b.adapters[0].consolidated_up.iter_mut().enumerate() {
            *x = ((i % 5) as f32 - 2.0) * 0.011;
        }
        b.memory.insert(&[0.1, -0.1, 0.05], &[0.2, -0.3, 0.4, 0.1, 0.5]);
        b.memory.insert(&[-0.2, 0.1, 0.03], &[-0.4, 0.2, 0.1, -0.5, 0.3]);
        b.h_persistent.fill(0.02 * (layer + 1) as f32);
    }
    m
}
fn blocks(m: &PSSALayerV2) -> impl Iterator<Item = &PSSAContinuousBlockV2> {
    std::iter::once(&m.block).chain(&m.extra_blocks)
}
fn carries(m: &PSSALayerV2) -> Vec<f32> {
    blocks(m).flat_map(|b| b.h_persistent.iter().copied()).collect()
}
fn set_carries(m: &mut PSSALayerV2, values: &[f32]) {
    let hs = m.cfg.d_latent * m.cfg.d_state;
    for (b, value) in std::iter::once(&mut m.block).chain(&mut m.extra_blocks).zip(values.chunks_exact(hs)) {
        b.h_persistent.copy_from_slice(value);
    }
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && y.is_finite() && (x - y).abs() < 2e-6 + 2e-5 * x.abs().max(y.abs()),
            "coordinate {i}: {x} != {y}");
    }
}
fn gradients(a: &PSSALayerV2, b: &PSSALayerV2) {
    close(&a.embed_w.grad, &b.embed_w.grad);
    close(&a.unembed_w.grad, &b.unembed_w.grad);
    for (a, b) in blocks(a).zip(blocks(b)) {
        macro_rules! check { ($($field:ident),*) => { $(close(&a.$field.grad, &b.$field.grad);)* }; }
        check!(norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj, mlp_w1, mlp_w2);
        close(&a.adapters[0].down_proj.grad, &b.adapters[0].down_proj.grad);
        close(&a.adapters[0].up_proj.grad, &b.adapters[0].up_proj.grad);
    }
}

#[test]
fn stacked_public_batched_entrypoints_dispatch_the_complete_scalar_stack() {
    for depth in [2, 4] {
        let mut scalar = model(depth);
        let mut dispatched = model(depth);
        let a = scalar.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
        let b = gpu_batch::forward_train_chunk_batched(&mut dispatched, &[1, 2, 3], &[2, 3, 4]);
        assert_eq!(a, b);
        scalar.backward_chunk(3, 0.7);
        gpu_batch::backward_chunk_batched(&mut dispatched, 3, 0.7);
        gradients(&scalar, &dispatched);
        assert_eq!(carries(&scalar), carries(&dispatched));
        assert!(dispatched.extra_blocks.iter().all(|b| b.w_delta.grad.iter().any(|x| x.abs() > 1e-9)));
    }
}

#[test]
fn stacked_lane_replay_matches_independent_ragged_accumulation_and_deferred_writes() {
    for depth in [2, 4] {
        let mut separate = model(depth);
        let mut packed = model(depth);
        let incoming = carries(&packed);
        let mut batch = SequenceBatch::new(&mut packed, 3).unwrap();
        let mut states = vec![vec![0.0; incoming.len()]; 3];
        for (lane, state) in states.iter_mut().enumerate() {
            for (i, x) in state.iter_mut().enumerate() { *x = (lane + i % 5) as f32 * 0.01; }
            batch.state_mut(lane).copy_from_slice(state);
        }
        for round in 0..3 {
            let lane_order = if round == 2 { vec![2, 0] } else { vec![0, 1, 2] };
            let inputs: Vec<Vec<usize>> = lane_order.iter().map(|&lane| {
                (0..if round == 0 { 4 } else { lane + 1 }).map(|t| (t + lane + round) % 9).collect()
            }).collect();
            let targets: Vec<Vec<usize>> = inputs.iter().map(|x| x.iter().map(|&id| (id + 1) % 9).collect()).collect();
            let sequences: Vec<_> = lane_order.iter().enumerate().map(|(i, &lane)| Sequence {
                lane, inputs: &inputs[i], targets: &targets[i], reset: round == 1 && lane == 0,
            }).collect();
            let n: usize = inputs.iter().map(Vec::len).sum();
            let banks: Vec<_> = blocks(&packed).map(|b| b.memory.clone()).collect();
            let loss = batch.forward(&mut packed, &sequences).unwrap();
            assert_eq!(carries(&packed), incoming);
            let mut expected_loss = 0.0;
            let mut terminals = Vec::new();
            for seq in &sequences {
                set_carries(&mut separate, &states[seq.lane]);
                if seq.reset { separate.reset_recurrent_state(); }
                let l = seq.inputs.len();
                expected_loss += separate.forward_train_chunk(seq.inputs, seq.targets) * l as f32 / n as f32;
                separate.backward_chunk(l, 0.7 * l as f32 / n as f32);
                states[seq.lane] = carries(&separate);
                terminals.push(blocks(&separate).map(|b| (
                    b.tape.q_poincare[(l - 1) * 3..l * 3].to_vec(),
                    b.tape.z_final[(l - 1) * 5..l * 5].to_vec(),
                )).collect::<Vec<_>>());
            }
            assert!((loss - expected_loss).abs() < 2e-6);
            batch.backward(&mut packed, 0.7).unwrap();
            assert_eq!(carries(&packed), incoming);
            gradients(&separate, &packed);
            for (lane, state) in states.iter().enumerate() { close(batch.state(lane), state); }
            assert_eq!(blocks(&packed).map(|b| b.memory.clone()).collect::<Vec<_>>(), banks);
            // Force the usual loss threshold so terminal rows from EVERY layer
            // are checked, including rows beyond the scalar chunk capacity.
            let mut offset = 0;
            for (seq, values) in sequences.iter().zip(terminals) {
                offset += seq.inputs.len();
                packed.insert_training_memory_at(4.0, offset - 1);
                let step = separate.step_counter;
                for (b, (key, value)) in std::iter::once(&mut separate.block)
                    .chain(&mut separate.extra_blocks).zip(values)
                {
                    b.memory.insert_protected(&key, &value, 4.0, step);
                }
            }
            assert_eq!(blocks(&packed).map(|b| b.memory.clone()).collect::<Vec<_>>(),
                blocks(&separate).map(|b| b.memory.clone()).collect::<Vec<_>>());
        }
        separate.apply_adamw(1e-4);
        packed.apply_adamw(1e-4);
        close(&separate.embed_w.data, &packed.embed_w.data);
        for (a, b) in blocks(&separate).zip(blocks(&packed)) { close(&a.w_delta.data, &b.w_delta.data); }
    }
}

#[test]
fn stacked_bounded_evaluation_restores_every_carry_and_all_checkpoint_state() {
    let dir = TempDir::new();
    let tok = Tokenizer::from_vocabulary(&["<unk>", "a", "b", "c", "d", "e", "f", "g", "h"].map(str::to_string)).unwrap();
    let mut m = model(4);
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    m.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
    m.backward_chunk(3, 1.0);
    m.apply_adamw(1e-4);
    for (i, b) in std::iter::once(&mut m.block).chain(&mut m.extra_blocks).enumerate() {
        b.h_persistent.fill(2.0 + i as f32);
    }
    let before = carries(&m);
    let path = dir.file("before.pssa");
    checkpoint::save_model(&m, &path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let raw = "a b c d e\nf g h a b\nc d e";
    let slice = EvaluationSlice { skip_tokens: 2, max_tokens: Some(8) };
    let result = evaluation::evaluate_pssa(&mut m, &tok, raw, slice).unwrap();
    assert_eq!((result.tokens, result.encoded_tokens), (6, 8));
    assert_eq!(carries(&m), before);
    assert_eq!(evaluation::evaluate_pssa(&mut m, &tok, raw, slice).unwrap(), result);
    checkpoint::save_model(&m, &path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(evaluation::evaluate_pssa(&mut m, &tok, raw,
        EvaluationSlice { skip_tokens: 12, max_tokens: Some(2) }).is_err());
    assert_eq!(carries(&m), before);
    m.extra_blocks[2].mlp_w1.data.fill(f32::NAN);
    assert!(evaluation::evaluate_pssa(&mut m, &tok, raw, slice).unwrap_err().contains("non-finite"));
    assert_eq!(carries(&m), before);
}

#[test]
fn cli_depth_defaults_validation_resume_and_stacked_cpu_reporting() {
    let dir = TempDir::new();
    let corpus = dir.file("corpus.txt");
    fs::write(&corpus, "a b c d e f\nb c d\ne f a b").unwrap();
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let run = |out: &str, extra: &[&str]| Command::new(exe)
        .env("RAYON_NUM_THREADS", "2")
        .args(["train", &corpus, "-o", out, "-e", "1", "--tokenizer", "word",
            "--latent", "5", "--state", "2", "--key", "3", "--memory", "5", "--chunk", "2", "--accumulate", "2"])
        .args(extra).output().unwrap();
    let default = dir.file("default.pssa");
    let explicit = dir.file("one.pssa");
    for (out, extra) in [(&default, vec![]), (&explicit, vec!["--depth", "1"])] {
        let result = run(out, &extra);
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    }
    assert_eq!(fs::read(&default).unwrap(), fs::read(&explicit).unwrap());
    assert_eq!(checkpoint::load_checkpoint(&default).unwrap().model.depth(), 1);
    let invalid = dir.file("invalid.pssa");
    for depth in ["0", "33", "-1", "bad", "999999999999999999999999999999"] {
        let result = run(&invalid, &["--depth", depth]);
        assert!(!result.status.success());
        assert!(!std::path::Path::new(&invalid).exists());
        assert!(String::from_utf8_lossy(&result.stderr).contains("--depth"));
        assert!(!String::from_utf8_lossy(&result.stderr).contains("panicked"));
    }
    let stack = dir.file("stack.pssa");
    let result = run(&stack, &["--depth", "3", "--batch-size", "2"]);
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("backend=cpu (stacked depth 3"));
    assert!(stdout.contains("batch_backend=cpu-replay"));
    let loaded = checkpoint::load_checkpoint(&stack).unwrap();
    assert_eq!(loaded.format, CheckpointFormat::V8);
    assert_eq!(loaded.model.depth(), 3);
    let resumed = dir.file("resumed.pssa");
    let result = run(&resumed, &["--resume", &stack, "--batch-size", "2"]);
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    assert_eq!(checkpoint::load_checkpoint(&resumed).unwrap().model.depth(), 3);
    for depth in ["1", "4"] {
        let result = run(&invalid, &["--resume", &stack, "--depth", depth]);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("does not match resume checkpoint"));
        assert!(!std::path::Path::new(&invalid).exists());
    }
    let help = Command::new(exe).arg("help").output().unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("--resume") && help.contains("--depth"));
}

#[test]
fn stacked_batched_training_schedule_and_resume_match_uninterrupted_run() {
    let dir = TempDir::new();
    // >34 vocabulary items exercise high-loss per-layer training-memory writes.
    let line = (0..40).map(|i| format!("token{i}")).collect::<Vec<_>>().join(" ");
    let raw = format!("{line}\n{line}\n");
    let opts = TrainingOptions {
        epochs: 2, depth: 3, latent: 4, state: 2, key: 3, memory: 5,
        chunk: 4, batch_size: 2, accumulate: 3, tokenizer: TokenizerKind::Word,
        schedule_total_updates: Some(30), warmup_steps: 2, lr: 1e-4,
        ..Default::default()
    };
    let (full, _) = CLIHandler::train_corpus(&raw, &opts).unwrap();
    let (first, _) = CLIHandler::train_corpus(&raw, &TrainingOptions { epochs: 1, ..opts.clone() }).unwrap();
    assert!(blocks(&first).all(|b| b.memory.count > 0));
    let resume = dir.file("first.pssa");
    checkpoint::save_model(&first, &resume).unwrap();
    let (continued, _) = CLIHandler::train_corpus(&raw, &TrainingOptions {
        epochs: 1, resume: Some(resume), ..opts
    }).unwrap();
    assert_eq!(continued.step_counter, full.step_counter);
    let a = dir.file("full.pssa");
    let b = dir.file("continued.pssa");
    checkpoint::save_model(&full, &a).unwrap();
    checkpoint::save_model(&continued, &b).unwrap();
    assert_eq!(fs::read(a).unwrap(), fs::read(b).unwrap());
}
