use oxide_ai_pssa::cli::{self, CLIHandler};
use oxide_ai_pssa::dataset::{DatasetManager, Tokenizer};
use oxide_ai_pssa::inference::{InferenceConfig, PSSAInferenceEngine};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "oxide-runtime-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
fn cfg(vocab: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: vocab,
        d_latent: 4,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 2,
        chunk_len: 8,
        lr: 0.01,
        ..Default::default()
    }
}
#[test]
fn vocabulary_restore_is_exact_and_reordering_fails_match() {
    let ordered = vec!["<unk>".into(), "zeta".into(), "alpha".into()];
    let tok = Tokenizer::from_vocabulary(&ordered).unwrap();
    assert_eq!(tok.ordered_vocabulary().unwrap(), ordered);
    assert_eq!(tok.encode("alpha zeta", true), vec![2, 1]);
    assert!(Tokenizer::from_vocabulary(&["<unk>".into(), "x".into(), "x".into()]).is_err());
    let mut m = PSSALayerV2::new(cfg(3), 3);
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    let reordered =
        Tokenizer::from_vocabulary(&["<unk>".into(), "alpha".into(), "zeta".into()]).unwrap();
    assert!(PSSAInferenceEngine::try_new(&mut m, &reordered).is_err());
}
#[test]
fn local_loader_is_ordered_and_rejects_missing_empty_and_binary() {
    let d = temp("dataset");
    fs::create_dir(&d).unwrap();
    fs::write(d.join("z.txt"), "zulu").unwrap();
    fs::write(d.join("a.txt"), "alpha").unwrap();
    assert_eq!(
        DatasetManager::try_load_dataset(Some(d.to_str().unwrap())).unwrap(),
        "alpha\nzulu\n"
    );
    assert!(
        DatasetManager::try_load_dataset(Some(d.join("missing.txt").to_str().unwrap())).is_err()
    );
    fs::write(d.join("empty.txt"), "").unwrap();
    assert!(DatasetManager::try_load_dataset(Some(d.to_str().unwrap())).is_err());
    fs::remove_file(d.join("empty.txt")).unwrap();
    fs::write(d.join("binary.txt"), [0xff, 0xfe]).unwrap();
    assert!(DatasetManager::try_load_dataset(Some(d.to_str().unwrap())).is_err());
    fs::remove_dir_all(d).unwrap();
}
#[test]
fn schedule_has_positive_warmup_boundary_and_explicit_minimum() {
    assert!((cli::learning_rate_for_update(1.0, 1, 10, 2).unwrap() - 0.5).abs() < 1e-7);
    assert!((cli::learning_rate_for_update(1.0, 2, 10, 2).unwrap() - 1.0).abs() < 1e-7);
    assert!((cli::learning_rate_for_update(1.0, 10, 10, 2).unwrap() - 0.01).abs() < 1e-7);
    assert!(cli::learning_rate_for_update(1.0, 1, 1, 1).is_err());
}
#[test]
fn zero_temperature_is_deterministic_and_permits_repeats() {
    let tok = Tokenizer::from_vocabulary(&["<unk>".into(), "alpha".into(), "beta".into()]).unwrap();
    let mut m = PSSALayerV2::new(cfg(3), 1);
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    m.embed_w.data.fill(0.);
    m.a_mat.data.fill(0.);
    m.w_delta.data.fill(0.);
    m.w_b.data.fill(0.);
    m.w_c.data.fill(0.);
    m.w_qx.data.fill(0.);
    m.w_qh.data.fill(0.);
    m.w_gate.data.fill(0.);
    m.w_proj.data.fill(0.);
    m.mlp_w1.data.fill(0.);
    m.mlp_w2.data.fill(0.);
    m.unembed_w.data.fill(0.);
    let output = {
        let mut e = PSSAInferenceEngine::try_new(&mut m, &tok).unwrap();
        e.try_generate_chat_turn(
            "alpha",
            &InferenceConfig {
                temperature: 0.0,
                max_new_tokens: 3,
                ..Default::default()
            },
            |_| {},
        )
        .unwrap()
    };
    assert_eq!(output, "alpha alpha alpha");
}
#[test]
fn evaluate_uniform_logits_has_known_loss_and_oov_count() {
    let tok = Tokenizer::from_vocabulary(&["<unk>".into(), "alpha".into(), "beta".into()]).unwrap();
    let mut m = PSSALayerV2::new(cfg(3), 1);
    m.vocabulary = tok.ordered_vocabulary().unwrap();
    m.unembed_w.data.fill(0.);
    let (loss, tokens, _correct, oov) =
        CLIHandler::evaluate_corpus(&mut m, &tok, "alpha beta\nunknown alpha").unwrap();
    assert!((loss - (3_f64).ln()).abs() < 1e-5);
    assert_eq!(tokens, 2);
    assert_eq!(oov, 1);
}
#[test]
fn fixed_schedule_horizon_survives_resume() {
    let raw = "a b c d\na b c d\n";
    let first_opts = cli::TrainingOptions {
        tokenizer: oxide_ai_pssa::dataset::TokenizerKind::Word,
        epochs: 1,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 2,
        accumulate: 1,
        max_tokens: Some(8),
        schedule_total_updates: Some(20),
        ..Default::default()
    };
    let (first, _) = CLIHandler::train_corpus(raw, &first_opts).unwrap();
    let checkpoint = temp("fixed-schedule");
    CLIHandler::save_model_v2(&first, checkpoint.to_str().unwrap()).unwrap();
    let second_opts = cli::TrainingOptions {
        resume: Some(checkpoint.to_str().unwrap().to_string()),
        tokenizer: oxide_ai_pssa::dataset::TokenizerKind::Word,
        epochs: 1,
        accumulate: 1,
        max_tokens: Some(8),
        skip_tokens: 8,
        ..Default::default()
    };
    let (second, _) = CLIHandler::train_corpus(raw, &second_opts).unwrap();
    assert_eq!(second.lr_schedule_total_updates, Some(20));
    assert!(second.step_counter > first.step_counter);
    fs::remove_file(checkpoint).unwrap();
}
#[test]
fn fixed_warmup_resume_matches_uninterrupted_training_before_and_after_warmup() {
    let raw = "a b c d\na b c d\n";
    let opts = cli::TrainingOptions {
        tokenizer: oxide_ai_pssa::dataset::TokenizerKind::Word,
        epochs: 3,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 2,
        accumulate: 1,
        schedule_total_updates: Some(20),
        warmup_steps: 5,
        ..Default::default()
    };
    let (whole, _) = CLIHandler::train_corpus(raw, &opts).unwrap();
    let p = temp("warmup-segmented");
    let expected = temp("warmup-whole");
    let (first, _) = CLIHandler::train_corpus(raw, &cli::TrainingOptions {
        epochs: 1,
        ..opts.clone()
    }).unwrap();
    assert_eq!(first.step_counter, 4);
    assert_eq!(first.lr_schedule_warmup_steps, Some(5));
    CLIHandler::save_model_v2(&first, p.to_str().unwrap()).unwrap();
    for step in [8, 12] {
        let (resumed, _) = CLIHandler::train_corpus(raw, &cli::TrainingOptions {
            epochs: 1,
            resume: Some(p.to_str().unwrap().into()),
            // Omitted CLI flags must restore both parts of the stored schedule.
            schedule_total_updates: None,
            warmup_steps: 0,
            ..opts.clone()
        }).unwrap();
        assert_eq!(resumed.step_counter, step);
        assert_eq!(resumed.lr_schedule_warmup_steps, Some(5));
        CLIHandler::save_model_v2(&resumed, p.to_str().unwrap()).unwrap();
    }
    CLIHandler::save_model_v2(&whole, expected.to_str().unwrap()).unwrap();
    assert_eq!(fs::read(&p).unwrap(), fs::read(&expected).unwrap());
    fs::remove_file(p).unwrap();
    fs::remove_file(expected).unwrap();
}

#[test]
fn word_tokenizer_reports_unusable_corpus_and_windows_wrap_at_eof() {
    let bad = cli::TrainingOptions {
        tokenizer: oxide_ai_pssa::dataset::TokenizerKind::Word,
        epochs: 1,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 2,
        accumulate: 1,
        max_tokens: Some(4),
        ..Default::default()
    };
    assert!(CLIHandler::train_corpus("😀😀\n", &bad).is_err());

    let wrapped = cli::TrainingOptions {
        tokenizer: oxide_ai_pssa::dataset::TokenizerKind::Word,
        epochs: 1,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 2,
        accumulate: 1,
        max_tokens: Some(5),
        skip_tokens: 4,
        ..Default::default()
    };
    assert!(CLIHandler::train_corpus("a b c\nd e f\n", &wrapped).is_ok());
}
#[test]
fn cli_errors_do_not_train_or_write_and_bad_numeric_exits_nonzero() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let out = temp("should-not-exist");
    let missing = Command::new(exe)
        .args([
            "train",
            "/definitely/not/here",
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!missing.success());
    assert!(!out.exists());
    let bad = Command::new(exe)
        .args(["train", "science", "--epochs", "nope"])
        .status()
        .unwrap();
    assert!(!bad.success());
    let prompt = Command::new(exe)
        .args(["generate", "", "--model", "/definitely/not/here"])
        .status()
        .unwrap();
    assert!(!prompt.success());
    let temperature = Command::new(exe)
        .args([
            "generate",
            "alpha",
            "--temperature",
            "0",
            "--model",
            "/definitely/not/here",
        ])
        .output()
        .unwrap();
    assert!(!temperature.status.success());
    assert!(!String::from_utf8_lossy(&temperature.stderr).contains("unknown option"));
    let excessive = Command::new(exe)
        .args([
            "generate",
            "alpha",
            "--model",
            "/definitely/not/here",
            "--max-new-tokens",
            "100001",
        ])
        .output()
        .unwrap();
    assert!(!excessive.status.success());
    assert!(String::from_utf8_lossy(&excessive.stderr).contains("at most"));
}
#[test]
fn shipped_legacy_v5_chat_uses_checkpoint_vocabulary_with_or_without_data() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = manifest.join("data/model.pssa");
    let data = manifest.join("data/downloaded.txt");
    for with_data in [true, false] {
        let mut command = Command::new(exe);
        command
            .args(["chat", "--model", model.to_str().unwrap(), "--temperature", "0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if with_data {
            command.args(["--data", data.to_str().unwrap()]);
        }
        let mut child = command.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"quantum mechanics\n/exit\n")
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "with_data={with_data}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("interactive: /exit"));
    }
}

#[test]
fn process_train_save_then_generate_without_corpus() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let data = temp("reference.txt");
    let model = temp("reference.pssa");
    fs::write(&data, "alpha beta gamma delta epsilon\nalpha beta gamma delta epsilon\nalpha beta gamma delta epsilon\n").unwrap();
    let status = Command::new(exe)
        .args([
            "train",
            data.to_str().unwrap(),
            "--out",
            model.to_str().unwrap(),
            "--epochs",
            "50",
            "--latent",
            "16",
            "--state",
            "4",
            "--key",
            "8",
            "--memory",
            "8",
            "--chunk",
            "16",
            "--lr",
            "0.02",
            "--accumulate",
            "1",
            "--warmup-steps",
            "2",
            "--seed",
            "7",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(model.exists());
    fs::remove_file(&data).unwrap();
    let output = Command::new(exe)
        .args([
            "generate",
            "alpha",
            "--model",
            model.to_str().unwrap(),
            "--temp",
            "0",
            "--max-new-tokens",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.split_whitespace().count(), 4);
    assert!(text.contains("beta"), "completion={text}");
    fs::remove_file(model).unwrap();
}
