use pssa::cli::CLIHandler;

#[test]
fn option_like_missing_value_is_rejected_before_typed_parsing() {
    let error = CLIHandler::parse_and_execute(vec![
        "pssa".into(),
        "train".into(),
        "--epochs".into(),
        "--bogus".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "option '--epochs' requires a value");
}

#[test]
fn hf_flags_reject_invalid_or_ambiguous_sources_before_network_access() {
    for (args, expected) in [
        (vec!["--hf-dataset", "bad-name"], "expected an owner/name"),
        (
            vec!["--hf-dataset", "owner/name", "science"],
            "cannot be combined",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--data", "science"],
            "cannot be combined",
        ),
        (vec!["--hf-config", "default"], "require --hf-dataset"),
        (vec!["--hf-split", "train"], "require --hf-dataset"),
        (vec!["--hf-field", "text"], "require --hf-dataset"),
        (
            vec!["--hf-dataset", "owner/name", "--hf-split", ""],
            "must not be empty",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--hf-field", ""],
            "must not be empty",
        ),
        (
            vec!["--hf-dataset", "owner/name", "--hf-config", ""],
            "must not be empty",
        ),
        (vec!["--hf-dataset"], "requires a value"),
    ] {
        let mut command = vec!["pssa".to_string(), "train".to_string()];
        command.extend(args.iter().map(|s| s.to_string()));
        let error = CLIHandler::parse_and_execute(command).unwrap_err();
        assert!(error.contains(expected), "{args:?}: {error}");
    }
}

#[test]
fn cached_hf_training_emits_real_feed_and_matches_local_checkpoint() {
    use pssa::{dataset::Tokenizer, training::sequence_plan};
    use std::hash::{Hash, Hasher};
    use std::{fs, process::Command};

    let root = std::env::temp_dir().join(format!("pssa-hf-cli-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    // Small, offline cache fixture in /tmp. Keep this key in sync with the
    // versioned on-disk cache contract; no live dataset is needed by tests.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    "hf-row-lines-v2".hash(&mut hash);
    "fixture/corpus".hash(&mut hash);
    Some("tiny".to_string()).hash(&mut hash);
    "train".hash(&mut hash);
    "text".hash(&mut hash);
    let raw = "ONE  Two THREE Four FIVE Six SEVEN Eight NINE Ten\nSHORT  Row\nA Different SIZED Final SAMPLE With EXTRA Tokens\n";
    fs::write(root.join(format!("{:016x}.txt", hash.finish())), raw).unwrap();
    let local = root.join("local.txt");
    fs::write(&local, raw).unwrap();
    let run = |hf: bool| {
        let checkpoint = root.join(if hf { "hf.pssa" } else { "local.pssa" });
        let mut command = Command::new(env!("CARGO_BIN_EXE_pssa"));
        command.arg("train");
        if hf {
            command.args(["--hf-dataset", "fixture/corpus", "--hf-config", "tiny"]);
        } else {
            command.arg(&local);
        }
        let output = command
            .args([
                "--tokenizer",
                "word",
                "--latent",
                "8",
                "--state",
                "2",
                "--key",
                "2",
                "--memory",
                "4",
                "--chunk",
                "3",
                "--accumulate",
                "2",
                "--batch-size",
                "2",
                "--skip-tokens",
                "3",
                "--max-tokens",
                "30",
                "-e",
                "2",
                "--no-tui",
                "-o",
            ])
            .arg(&checkpoint)
            .env("PSSA_HF_CACHE", &root)
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8(output.stdout).unwrap();
        assert!(!log.contains('\x1b'), "--no-tui must stay plain");
        (log, fs::read(checkpoint).unwrap())
    };
    let (feed, hf_checkpoint) = run(true);
    let (plain, local_checkpoint) = run(false);
    assert_eq!(
        hf_checkpoint, local_checkpoint,
        "feed metadata must not affect training bits"
    );
    assert!(
        !root.join("local.txt.pssatok").exists(),
        "omitting --token-cache must not enable the persistent cache"
    );
    assert!(
        plain.contains("feed_schema=2"),
        "local training must emit the same real feed schema"
    );
    let local_feed = plain
        .lines()
        .filter(|line| line.contains("feed_schema=2"))
        .last()
        .unwrap();
    assert!(local_feed.contains(&format!("feed_dataset={}", local.display())));
    let tokenizer = Tokenizer::from_corpus(raw, true).unwrap();
    let docs = CLIHandler::documents(raw, &tokenizer, Some(30), 3).unwrap();
    let plan = sequence_plan(&docs, 3, 2).unwrap();
    let c = plan.iter().flatten().last().unwrap();
    let end = c.start + c.len;
    let ids = docs[c.doc][end.saturating_sub(16).max(c.start)..end]
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("%2C");
    let last = feed
        .lines()
        .filter(|line| line.contains("feed_dataset="))
        .last()
        .unwrap();
    let tokens: usize = plan.iter().flatten().map(|c| c.len).sum::<usize>() * 2;
    for expected in [
        "feed_schema=2".to_string(),
        "feed_dataset=fixture/corpus".to_string(),
        "feed_config=tiny".to_string(),
        "feed_text_kind=raw".to_string(),
        format!("feed_rows={}", docs.len()),
        format!("feed_tokens={tokens}"),
        format!("feed_epoch_tokens={}", tokens / 2),
        format!("feed_epoch_total={}", tokens / 2),
        "feed_epoch=2".to_string(),
        "feed_epochs=2".to_string(),
        format!("feed_batch={}", plan.len()),
        format!("feed_batches={}", plan.len()),
        format!("feed_step={}", plan.len().div_ceil(2) * 2),
        format!("feed_row={}", c.doc + 1),
        format!(
            "feed_start={}",
            docs[..c.doc].iter().map(Vec::len).sum::<usize>() + end.saturating_sub(16).max(c.start)
        ),
        format!(
            "feed_end={}",
            docs[..c.doc].iter().map(Vec::len).sum::<usize>() + end
        ),
        format!("feed_token_ids={ids}"),
    ] {
        assert!(
            last.split_whitespace().any(|field| field == expected),
            "missing {expected}: {last}"
        );
    }
    let decode = |value: &str| {
        let mut out = Vec::new();
        let mut bytes = value.bytes();
        while let Some(byte) = bytes.next() {
            if byte == b'%' {
                let hex = [bytes.next().unwrap(), bytes.next().unwrap()];
                out.push(u8::from_str_radix(std::str::from_utf8(&hex).unwrap(), 16).unwrap());
            } else {
                out.push(byte);
            }
        }
        String::from_utf8(out).unwrap()
    };
    let field = |line: &str, key: &str| {
        let value = line
            .split_whitespace()
            .find_map(|entry| entry.strip_prefix(&format!("{key}=")))
            .unwrap();
        decode(value)
    };
    let snippet = field(last, "feed_snippet");
    assert!(
        raw.lines().any(|line| line.contains(&snippet)),
        "must preserve actual source case/spacing: {snippet}"
    );
    let expected_ids = &docs[c.doc][end.saturating_sub(16).max(c.start)..end];
    assert_ne!(
        snippet,
        tokenizer.decode(expected_ids),
        "raw must not be normalized decode"
    );
    let pieces: Vec<String> = serde_json::from_str(&field(last, "feed_token_pieces")).unwrap();
    assert_eq!(
        pieces,
        expected_ids
            .iter()
            .map(|id| tokenizer.id_to_token[id].clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(field(local_feed, "feed_snippet"), snippet);
    assert!(
        !last.contains("feed_bytes="),
        "word byte consumption is not exact"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn negative_numeric_option_value_reaches_domain_validation() {
    let error = CLIHandler::parse_and_execute(vec![
        "pssa".into(),
        "generate".into(),
        "prompt".into(),
        "--temperature".into(),
        "-1".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "--temp must be >= 0");
}
