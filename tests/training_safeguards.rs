use pssa::{
    backend::Device,
    checkpoint,
    cli::{CLIHandler, TrainingBackend, TrainingOptions},
    dataset::{Tokenizer, TokenizerKind},
    pssa::{GradientClipOutcome, PSSAConfigV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "pssa-safeguards-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn flags_off_three_link_resume_matches_unmodified_baseline_bytes() {
    let raw = "alpha beta gamma delta epsilon zeta eta theta iota\nalpha beta gamma\ndelta epsilon zeta\n";
    let opts = TrainingOptions {
        backend: TrainingBackend::Cpu,
        epochs: 3,
        latent: 7,
        state: 3,
        key: 4,
        memory: 4,
        chunk: 2,
        batch_size: 2,
        accumulate: 3,
        tokenizer: TokenizerKind::Word,
        schedule_total_updates: Some(20),
        warmup_steps: 2,
        lr: 1e-4,
        no_tui: true,
        ..Default::default()
    };
    let tmp = TempDir::new();
    let full_path = tmp.path("full.pssa");
    let chain_path = tmp.path("chain.pssa");
    let (full, _) = CLIHandler::train_corpus(raw, &opts).unwrap();
    checkpoint::save_model(&full, &full_path).unwrap();
    let full_bytes = fs::read(&full_path).unwrap();
    // Independent captures with these exact options: unmodified main 014f617
    // for AVX2/FMA, and pre-SIMD 4042886 for the legacy kernel. The current
    // main intentionally changed reduction rounding; keep strict bit checks
    // against the matching kernel, not a tolerance or a regenerated test value.
    // Includes serialized gradients, Adam moments, memory, RNG and schedule.
    let baseline_hash = full_bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    assert_eq!(full_bytes.len(), 15_157);
    #[cfg(target_arch = "x86_64")]
    let simd = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    let expected_hash = if simd {
        0x3e28ad0e5cd76c7d
    } else {
        0x446169f99aa899fe
    };
    assert_eq!(baseline_hash, expected_hash);
    for link in 0..3 {
        let (model, _) = CLIHandler::train_corpus(
            raw,
            &TrainingOptions {
                epochs: 1,
                resume: (link > 0).then(|| chain_path.clone()),
                ..opts.clone()
            },
        )
        .unwrap();
        checkpoint::save_model(&model, &chain_path).unwrap();
    }
    assert_eq!(fs::read(chain_path).unwrap(), full_bytes);
}

#[test]
fn cli_safeguards_are_opt_in_logged_and_not_checkpointed() {
    let tmp = TempDir::new();
    let corpus = tmp.path("corpus.txt");
    // Enough distinct words to trigger surprise writes (>3.5 nats).
    let text = (0..48)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&corpus, format!("{text}\n{text}\n")).unwrap();
    let exe = env!("CARGO_BIN_EXE_pssa");
    for (batch, depth, loops) in [(1, 1, 1), (2, 1, 1), (1, 2, 2), (2, 2, 2)] {
        let out = tmp.path(&format!("model-{batch}-{depth}-{loops}.pssa"));
        let result = Command::new(exe)
            .args([
                "train",
                &corpus,
                "-o",
                &out,
                "-e",
                "1",
                "--backend",
                "cpu",
                "--tokenizer",
                "word",
                "--latent",
                "7",
                "--state",
                "3",
                "--key",
                "4",
                "--memory",
                "4",
                "--chunk",
                "2",
                "--accumulate",
                "2",
                "--batch-size",
                &batch.to_string(),
                "--depth",
                &depth.to_string(),
                "--loops",
                &loops.to_string(),
                "--grad-clip",
                "1.0",
                "--memory-value-cap",
                "0.25",
                "--no-tui",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let log = String::from_utf8_lossy(&result.stdout);
        assert!(log.contains("grad_clip=1 memory_value_cap=0.25"));
        assert!(
            log.lines()
                .any(|l| l.contains("optimizer_updates=") && l.contains("grad_norm="))
        );
        assert!(log.contains("skipped_updates=0"));
        let m = checkpoint::load_checkpoint(&out).unwrap().model;
        assert!(m.step_counter > 0);
        for b in std::iter::once(&m.block).chain(&m.extra_blocks) {
            assert!(b.memory.count > 0);
            assert_eq!(b.memory.value_cap, None); // runtime flags never persisted
            for v in b.memory.values.chunks_exact(b.memory.dim_val) {
                let norm = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                assert!(norm <= 0.25, "{norm}");
            }
        }
    }
    let help = Command::new(exe)
        .args(["train", "--help"])
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("--grad-clip") && help.contains("--memory-value-cap"));
    let bad_out = tmp.path("bad.pssa");
    for flag in ["--grad-clip", "--memory-value-cap"] {
        for value in ["0", "-1", "NaN", "inf", "1e40", "garbage"] {
            let result = Command::new(exe)
                .args(["train", &corpus, "-o", &bad_out, flag, value])
                .output()
                .unwrap();
            assert!(!result.status.success());
            assert!(!PathBuf::from(&bad_out).exists());
        }
        assert!(
            !Command::new(exe)
                .args(["train-transformer", &corpus, "-o", &bad_out, flag, "1.0",])
                .output()
                .unwrap()
                .status
                .success()
        );
    }
}

#[test]
fn cli_nonfinite_gradients_skip_adam_and_abort_on_the_twenty_first_skip() {
    let tmp = TempDir::new();
    let corpus = tmp.path("skip.txt");
    let text = "alpha beta ".repeat(24);
    fs::write(&corpus, &text).unwrap();
    let tok = Tokenizer::from_corpus(&text, true).unwrap();
    let mut m = PSSALayerV2::new_with_device(
        PSSAConfigV2 {
            d_latent: 4,
            d_vocab: tok.vocab_size,
            d_state: 1,
            d_mem_key: 1,
            mem_capacity: 2,
            chunk_len: 1,
            ..Default::default()
        },
        42,
        Device::Cpu,
    );
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    for p in [
        &mut m.block.w_qx,
        &mut m.block.w_qh,
        &mut m.block.w_proj,
        &mut m.block.w_b,
        &mut m.block.w_c,
        &mut m.block.mlp_w2,
        &mut m.block.adapters[0].up_proj,
        &mut m.unembed_w,
    ] {
        p.data.fill(0.0);
    }
    // A finite retrieval with opposite near-limit values has a non-finite
    // value-difference adjoint (0 * inf), even though its projected output and
    // loss are finite. Leave the cap OFF to exercise the clipping skip path.
    m.block.memory.insert(&[0.0], &[-f32::MAX; 4]);
    m.block.memory.insert(&[0.5], &[f32::MAX; 4]);
    let resume = tmp.path("skip-seed.pssa");
    let out = tmp.path("must-not-save.pssa");
    checkpoint::save_model(&m, &resume).unwrap();
    let before = fs::read(&resume).unwrap();
    for batch in [1, 2] {
        let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
            .args([
                "train",
                &corpus,
                "--resume",
                &resume,
                "-o",
                &out,
                "-e",
                "1",
                "--backend",
                "cpu",
                "--accumulate",
                "1",
                "--batch-size",
                &batch.to_string(),
                "--grad-clip",
                "1.0",
                "--no-tui",
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        let warnings = String::from_utf8_lossy(&result.stderr);
        assert_eq!(
            warnings.matches("warning: non-finite grad_norm=").count(),
            21,
            "{warnings}"
        );
        assert_eq!(warnings.matches("global_step=0 ").count(), 21, "{warnings}");
        assert!(warnings.contains("more than 20 consecutive non-finite gradient updates"));
        assert!(warnings.contains("skipped_updates=21"));
        assert!(!PathBuf::from(&out).exists());
        assert_eq!(fs::read(&resume).unwrap(), before);
    }
}

/// Finite, frozen read/write gain of two, seeded at the scale observed in the
/// divergence diagnosis. Unclipped Adam overflows its second moments on the
/// first update; bounded values and gradients contain the same 13-update run.
#[test]
fn enormous_memory_feedback_is_finite_with_both_safeguards() {
    fn seed() -> PSSALayerV2 {
        let mut m = PSSALayerV2::new_with_device(
            PSSAConfigV2 {
                d_latent: 4,
                d_vocab: 4,
                d_state: 1,
                d_mem_key: 1,
                mem_capacity: 1,
                chunk_len: 1,
                ..Default::default()
            },
            42,
            Device::Cpu,
        );
        for p in [
            &mut m.block.w_b,
            &mut m.block.w_c,
            &mut m.block.w_qx,
            &mut m.block.w_qh,
            &mut m.block.w_gate,
            &mut m.block.w_proj,
            &mut m.block.mlp_w2,
            &mut m.block.adapters[0].up_proj,
            &mut m.unembed_w,
        ] {
            p.data.fill(0.0);
        }
        m.block.adapters[0].consolidated_up.fill(0.0);
        for i in 0..4 {
            m.block.w_proj.data[i * 4 + i] = 4.0;
            m.unembed_w.data[i] = 1.0;
        }
        // With the logit scale 1/sqrt(4), this produces a head gradient
        // of 2e21 (not exactly the asserted 1e21 boundary after rounding).
        m.block.memory.insert(&[0.0], &[2e21; 4]);
        m
    }
    fn finite_optimizer(m: &PSSALayerV2) -> bool {
        let b = &m.block;
        [
            &m.embed_w,
            &m.unembed_w,
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
            &b.adapters[0].up_proj,
        ]
        .into_iter()
        .all(|p| {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|x| x.is_finite())
        }) && [&b.norm_gamma, &b.norm_beta].into_iter().all(|p| {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|x| x.is_finite())
        })
    }
    // Thirty-two lanes exercise the same packed backward/write/update ordering
    // as the original diagnosis, but with only four latent coordinates.
    let sequences: Vec<_> = (0..32)
        .map(|lane| Sequence {
            lane,
            inputs: &[0],
            targets: &[1],
            reset: true,
        })
        .collect();
    let mut off = seed();
    let mut batch = SequenceBatch::new(&mut off, 32).unwrap();
    off.zero_gradients();
    let loss = batch.forward(&mut off, &sequences).unwrap();
    batch.backward(&mut off, 1.0).unwrap();
    assert!(loss.is_finite() && loss > 1e21);
    assert!(finite_optimizer(&off));
    let grad_max = off
        .unembed_w
        .grad
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f32::max);
    assert!(grad_max > 1e21);
    off.apply_adamw(1e-3);
    assert!(off.unembed_w.v.iter().any(|x| x.is_infinite()));
    println!("feedback flags=off loss={loss:e} grad_max={grad_max:e} adam_v=inf after_update=1");

    let mut on = seed();
    on.block.memory.set_value_cap(Some(512.0));
    let mut batch = SequenceBatch::new(&mut on, 32).unwrap();
    let mut loss_max = 0.0f32;
    let mut grad_norm_max = 0.0f64;
    for _ in 0..13 {
        on.zero_gradients();
        let loss = batch.forward(&mut on, &sequences).unwrap();
        assert!(loss.is_finite());
        loss_max = loss_max.max(loss);
        batch.backward(&mut on, 1.0).unwrap();
        for lane in 0..32 {
            on.insert_training_memory_at(on.tape.losses[lane], lane);
        }
        let GradientClipOutcome::Applied { norm } = on.apply_adamw_with_grad_clip(1e-3, 1.0) else {
            panic!("capped feedback must not produce a bad gradient norm");
        };
        grad_norm_max = grad_norm_max.max(norm);
        assert!(finite_optimizer(&on));
        assert!((0..32).all(|lane| batch.state(lane).iter().all(|x| x.is_finite())));
        let norm = on
            .block
            .memory
            .values
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(norm <= 512.0);
    }
    assert_eq!(on.step_counter, 13);
    println!(
        "feedback grad_clip=1 memory_value_cap=512 updates=13 loss_max={loss_max:e} pre_clip_grad_norm_max={grad_norm_max:e} finite=true"
    );
}

#[test]
fn resumed_nonfinite_loss_reports_update_stage_and_tensor_without_saving() {
    let tmp = TempDir::new();
    let text = "alpha beta\nalpha beta\n";
    let corpus = tmp.path("overflow.txt");
    fs::write(&corpus, text).unwrap();
    let tok = Tokenizer::from_corpus(text, true).unwrap();
    let mut m = PSSALayerV2::new(
        PSSAConfigV2 {
            d_latent: 4,
            d_vocab: tok.vocab_size,
            d_state: 1,
            d_mem_key: 1,
            mem_capacity: 2,
            chunk_len: 1,
            ..Default::default()
        },
        42,
    );
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    m.step_counter = 1814;
    for p in [
        &mut m.block.w_b,
        &mut m.block.w_c,
        &mut m.block.w_qx,
        &mut m.block.w_qh,
        &mut m.block.w_gate,
        &mut m.block.w_proj,
        &mut m.block.mlp_w1,
        &mut m.block.mlp_w2,
        &mut m.block.adapters[0].up_proj,
        &mut m.unembed_w,
    ] {
        p.data.fill(0.0);
    }
    for i in 0..4 {
        m.block.w_proj.data[i * 4 + i] = 1.0;
    }
    m.block.memory.insert(&[0.0], &[3.0; 4]);
    // All checkpoint tensors are finite. Memory injects 1.5 into z_final;
    // the head is the first stage to overflow during this update.
    m.unembed_w.data[0] = f32::MAX;
    let resume = tmp.path("overflow-seed.pssa");
    let out = tmp.path("must-not-save.pssa");
    checkpoint::save_model(&m, &resume).unwrap();
    let before = fs::read(&resume).unwrap();
    for batch in [1, 2] {
        let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
            .args([
                "train",
                &corpus,
                "--resume",
                &resume,
                "-o",
                &out,
                "-e",
                "1",
                "--backend",
                "cpu",
                "--batch-size",
                &batch.to_string(),
                "--no-tui",
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        for message in [
            "non-finite loss",
            "update_index=1",
            "global_step=1815",
            "stage=logits_loss",
            "tensor=logits",
            "index=0",
            "without checkpoint",
        ] {
            assert!(
                error.contains(message),
                "batch={batch}: missing {message}: {error}"
            );
        }
        assert!(!PathBuf::from(&out).exists());
        assert_eq!(fs::read(&resume).unwrap(), before);
    }
}

#[path = "support/cuda_memory_ptx.rs"]
mod cuda_memory_ptx;
#[path = "support/cuda_packed_ptx.rs"]
mod cuda_packed_ptx;
#[path = "support/cuda_packed_reduce_ptx.rs"]
mod cuda_packed_reduce_ptx;
#[path = "support/cuda_ssm_ptx.rs"]
mod cuda_ssm_ptx;
#[path = "../src/cuda/stage_bounds.rs"]
mod cuda_stage_bounds;
#[path = "support/cuda_stage_local_ptx.rs"]
mod cuda_stage_local_ptx;

#[test]
fn embedded_cuda_ptx_assembles_with_ptxas_when_available() {
    let ptxas = if let Some(path) = std::env::var_os("PTXAS") {
        if path.to_string_lossy().is_empty() {
            "ptxas".into()
        } else {
            path
        }
    } else {
        match Command::new("ptxas").arg("--version").output() {
            Ok(output) if output.status.success() => "ptxas".into(),
            Ok(output) => panic!(
                "ptxas found on PATH but --version failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipped CUDA PTX assembly test: ptxas not found via PTXAS or PATH");
                return;
            }
            Err(error) => panic!("could not run ptxas from PATH: {error}"),
        }
    };
    let version = Command::new(&ptxas)
        .arg("--version")
        .output()
        .expect("PTXAS must name an executable ptxas binary");
    assert!(
        version.status.success(),
        "PTXAS --version failed: {}",
        String::from_utf8_lossy(&version.stderr)
    );
    eprintln!(
        "testing embedded PTX with {}",
        String::from_utf8_lossy(&version.stdout)
    );

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda");
    let tmp = TempDir::new();
    for source in ["stages.ptx", "safeguards.ptx", "packed.ptx"] {
        for arch in ["sm_120", "sm_90", "sm_80"] {
            let output = tmp.path(&format!("{source}-{arch}.cubin"));
            let result = Command::new(&ptxas)
                .arg(format!("-arch={arch}"))
                .arg("-o")
                .arg(&output)
                .arg(root.join(source))
                .output()
                .expect("ptxas invocation should start");
            assert!(
                result.status.success(),
                "ptxas failed for {source} at {arch}:\n{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[test]
fn resume_caps_loaded_bank_before_first_forward_without_format_changes() {
    let raw = "alpha beta gamma delta epsilon\nalpha beta gamma\n";
    let opts = TrainingOptions {
        backend: TrainingBackend::Cpu,
        epochs: 1,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 2,
        accumulate: 1,
        tokenizer: TokenizerKind::Word,
        schedule_total_updates: Some(30),
        no_tui: true,
        ..Default::default()
    };
    let tmp = TempDir::new();
    let resume = tmp.path("resume.pssa");
    let (mut m, _) = CLIHandler::train_corpus(raw, &opts).unwrap();
    m.block.memory.insert(&[0.0, 0.0], &[1e21; 4]);
    checkpoint::save_model(&m, &resume).unwrap();
    let prior = m.step_counter;
    for batch_size in [1, 2] {
        let (continued, _) = CLIHandler::train_corpus(
            raw,
            &TrainingOptions {
                batch_size,
                resume: Some(resume.clone()),
                grad_clip: Some(1.0),
                memory_value_cap: Some(1.0),
                ..opts.clone()
            },
        )
        .unwrap();
        assert!(continued.step_counter > prior);
        assert_eq!(continued.block.memory.value_cap, Some(1.0));
        for v in continued.block.memory.values.chunks_exact(4) {
            assert!(v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() <= 1.0);
        }
        assert!(
            continued
                .embed_w
                .data
                .iter()
                .chain(&continued.embed_w.v)
                .all(|x| x.is_finite())
        );
    }
}
