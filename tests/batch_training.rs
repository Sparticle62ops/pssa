use oxide_ai_pssa::{
    checkpoint,
    cli::{CLIHandler, TrainingOptions},
    dataset::TokenizerKind,
    training::{chunk_plan, sequence_plan},
};
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

fn path(label: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "oxide-batch-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
        .to_str()
        .unwrap()
        .into()
}

#[test]
fn document_lanes_preserve_all_transitions_carry_boundaries_and_refill() {
    let docs: Vec<Vec<usize>> = [9, 3, 3, 1, 0, 5]
        .iter()
        .map(|&len| (0..len).collect())
        .collect();
    for b in [1, 2, 3, 20] {
        let plan = sequence_plan(&docs, 2, b).unwrap();
        let mut actual: Vec<_> = plan
            .iter()
            .flatten()
            .map(|c| (c.doc, c.start, c.len))
            .collect();
        let expected = chunk_plan(&docs, 2);
        if b == 1 {
            assert_eq!(actual, expected);
        }
        actual.sort();
        assert_eq!(actual, expected);
        let mut previous = vec![None; b];
        for batch in &plan {
            assert!(!batch.is_empty() && batch.len() <= b);
            for (i, c) in batch.iter().enumerate() {
                assert!(
                    !batch[..i]
                        .iter()
                        .any(|other| other.doc == c.doc || other.lane == c.lane)
                );
                if c.start > 0 {
                    assert_eq!(previous[c.lane], Some((c.doc, c.start)));
                }
                previous[c.lane] = Some((c.doc, c.start + c.len));
                assert!(c.start + c.len < docs[c.doc].len());
            }
        }
    }
    // Imbalance matters: ceil(6 chunks / 2 lanes)=3 is NOT the 4 microbatches.
    let imbalanced = sequence_plan(&docs[..3], 2, 2).unwrap();
    assert_eq!(imbalanced.len(), 4);
    assert_eq!(imbalanced[1][1].doc, 2); // lane refill
    assert_eq!(imbalanced[3].len(), 1); // no padding/extra gradients
    assert_eq!(sequence_plan(&docs[..1], 2, 8).unwrap().len(), 4);
    assert!(sequence_plan(&docs, 0, 2).is_err());
    assert!(sequence_plan(&docs, 2, 0).is_err());
    assert!(sequence_plan(&[], 2, 2).unwrap().is_empty());
    assert!(sequence_plan(&docs[3..5], 2, 2).unwrap().is_empty());
}

#[test]
fn batched_schedule_resume_is_exact_and_old_horizonless_checkpoint_loads() {
    let raw = "alpha beta gamma delta epsilon zeta eta theta iota\nalpha beta gamma\ndelta epsilon zeta\n";
    let opts = TrainingOptions {
        epochs: 2,
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
        ..Default::default()
    };
    let (full, tok) = CLIHandler::train_corpus(raw, &opts).unwrap();
    let docs = CLIHandler::documents(raw, &tok, None, 0).unwrap();
    let updates = sequence_plan(&docs, 2, 2).unwrap().len().div_ceil(3);
    assert_eq!(full.step_counter, 2 * updates);
    let (first, _) = CLIHandler::train_corpus(
        raw,
        &TrainingOptions {
            epochs: 1,
            ..opts.clone()
        },
    )
    .unwrap();
    let resume = path("resume.pssa");
    checkpoint::save_model(&first, &resume).unwrap();
    let (resumed, _) = CLIHandler::train_corpus(
        raw,
        &TrainingOptions {
            epochs: 1,
            resume: Some(resume.clone()),
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(resumed.cfg.chunk_len, 2); // never serialized as B*L
    let a = path("full.pssa");
    let b = path("resumed.pssa");
    checkpoint::save_model(&full, &a).unwrap();
    checkpoint::save_model(&resumed, &b).unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    // Historical V7 without a horizon is still a valid batched warm start.
    let mut legacy = first;
    legacy.lr_schedule_total_updates = None;
    legacy.lr_schedule_warmup_steps = None;
    checkpoint::save_model(&legacy, &resume).unwrap();
    let (continued, _) = CLIHandler::train_corpus(
        raw,
        &TrainingOptions {
            epochs: 1,
            schedule_total_updates: None,
            resume: Some(resume.clone()),
            ..opts
        },
    )
    .unwrap();
    assert_eq!(continued.step_counter, 2 * updates);
    assert_eq!(continued.lr_schedule_total_updates, None);
    for p in [resume, a, b] {
        fs::remove_file(p).unwrap();
    }
}

#[test]
fn cli_batch_one_is_byte_identical_and_bad_sizes_do_not_write() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let corpus = path("corpus.txt");
    fs::write(
        &corpus,
        "alpha beta gamma delta epsilon\nalpha beta gamma\ndelta epsilon zeta\n",
    )
    .unwrap();
    let a = path("default.pssa");
    let b = path("one.pssa");
    let run = |out: &str, extra: &[&str]| {
        Command::new(exe)
            .args([
                "train",
                &corpus,
                "-o",
                out,
                "-e",
                "1",
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
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let default = run(&a, &[]);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    assert!(run(&b, &["--batch-size", "1"]).status.success());
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    let bad = path("bad.pssa");
    for size in ["0", "-1", "garbage", "18446744073709551615", "65537"] {
        let output = run(&bad, &["--batch-size", size]);
        assert!(!output.status.success());
        assert!(!std::path::Path::new(&bad).exists());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
    let batch = run(&bad, &["--batch-size", "2"]);
    assert!(
        batch.status.success(),
        "{}",
        String::from_utf8_lossy(&batch.stderr)
    );
    let stdout = String::from_utf8_lossy(&batch.stdout);
    assert!(stdout.contains("batch_size=2") && stdout.contains("sequence_plan_fnv1a64="));
    let help = Command::new(exe).arg("help").output().unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("--resume") && help.contains("--batch-size"));
    assert!(
        !Command::new(exe)
            .args(["train-transformer", &corpus, "--batch-size", "2"])
            .output()
            .unwrap()
            .status
            .success()
    );
    // Batched checkpoints still feed every existing PSSA inference command.
    let generated = Command::new(exe)
        .args([
            "generate",
            "-m",
            &bad,
            "-p",
            "alpha beta",
            "--temperature",
            "0",
            "--max-new-tokens",
            "2",
        ])
        .output()
        .unwrap();
    assert!(generated.status.success());
    let evaluated = Command::new(exe)
        .args(["evaluate", &corpus, "-m", &bad])
        .output()
        .unwrap();
    assert!(evaluated.status.success());
    let metrics: serde_json::Value = serde_json::from_slice(&evaluated.stdout).unwrap();
    assert!(metrics["cross_entropy"].as_f64().unwrap().is_finite());
    let mut chat = Command::new(exe)
        .args(["chat", "-m", &bad, "--temperature", "0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    chat.stdin
        .take()
        .unwrap()
        .write_all(b"alpha beta\n/exit\n")
        .unwrap();
    let chat = chat.wait_with_output().unwrap();
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
    assert!(String::from_utf8_lossy(&chat.stdout).contains("interactive: /exit"));
    for p in [corpus, a, b, bad] {
        fs::remove_file(p).unwrap();
    }
}

#[test]
fn transformer_api_rejects_unsupported_sequence_batches() {
    let result = oxide_ai_pssa::transformer_training::train_corpus(
        "alpha beta\n",
        &TrainingOptions {
            batch_size: 2,
            ..Default::default()
        },
        None,
    );
    assert!(matches!(result, Err(message) if message.contains("use batch size 1")));
}
