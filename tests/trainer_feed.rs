use pssa::{
    cli::CLIHandler,
    dataset::Tokenizer,
    training::{chunk_plan, sequence_plan},
};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "pssa-trainer-feed-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn decode(value: &str) -> String {
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
}

fn fields(line: &str) -> HashMap<&str, String> {
    line.split_whitespace()
        .filter_map(|entry| entry.split_once('='))
        .map(|(key, value)| (key, decode(value)))
        .collect()
}

#[test]
fn transformer_bpe_actual_window_cache_and_epochs_do_not_change_checkpoint_bits() {
    let fixture = Fixture::new();
    let raw = "Zé😀 AB CD EF GH IJ\nKLMNO PQ RS\n";
    let source = fixture.0.join("corpus.txt");
    let cache = fixture.0.join("corpus.pssatok");
    fs::write(&source, raw).unwrap();
    let tokenizer = Tokenizer::from_corpus_bpe(raw, 257).unwrap();
    let docs = CLIHandler::documents(raw, &tokenizer, Some(27), 3).unwrap();
    let plan = chunk_plan(&docs, 3);
    let epoch_tokens: usize = plan.iter().map(|chunk| chunk.2).sum();
    let epoch_bytes: usize = docs
        .iter()
        .flat_map(|doc| &doc[..doc.len() - 1])
        .map(|&id| tokenizer.token_bytes(id).unwrap().len())
        .sum();
    let updates = plan.len().div_ceil(2) * 2;
    let run = |label: &str, cached: bool, telemetry: bool| {
        let out = fixture.0.join(format!("{label}.trfm"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_pssa"));
        command
            .arg("train-transformer")
            .arg(&source)
            .args([
                "--tokenizer",
                "bpe",
                "--vocab-size",
                "257",
                "--chunk",
                "3",
                "--accumulate",
                "2",
                "--skip-tokens",
                "3",
                "--max-tokens",
                "27",
                "--epochs",
                "2",
                "--no-tui",
                "--out",
            ])
            .arg(&out)
            .env("RAYON_NUM_THREADS", "1");
        if cached {
            command.arg("--token-cache").arg(&cache);
        }
        if !telemetry {
            command.arg("--no-feed-telemetry");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8(output.stdout).unwrap();
        assert!(!log.contains('\x1b'));
        (log, fs::read(out).unwrap())
    };
    let (disabled, reference) = run("disabled", false, false);
    assert!(
        !disabled.contains("feed_schema="),
        "explicit opt-out must omit dataset windows"
    );
    assert!(disabled.contains("feed_telemetry=false checkpoint_target="));
    let (uncached, uncached_bits) = run("uncached", false, true);
    assert_eq!(
        uncached_bits, reference,
        "telemetry must not change model bits"
    );
    assert!(!cache.exists(), "cache stays opt-in");
    let (built, built_bits) = run("built", true, true);
    let cache_bits = fs::read(&cache).unwrap();
    let (reused, reused_bits) = run("reused", true, true);
    assert_eq!(built_bits, reference);
    assert_eq!(reused_bits, reference);
    assert_eq!(fs::read(&cache).unwrap(), cache_bits);
    assert!(built.contains("token_cache=built"));
    assert!(reused.contains("token_cache=reused"));
    for log in [&uncached, &built, &reused] {
        assert!(log.contains("feed_telemetry=true checkpoint_target="));
        let samples: Vec<_> = log
            .lines()
            .filter(|line| line.contains("feed_schema=2"))
            .map(fields)
            .collect();
        assert!(!samples.is_empty());
        assert!(
            samples.len() <= updates,
            "telemetry uses the existing progress throttle"
        );
        for sample in &samples {
            assert_eq!(sample["feed_dataset"], source.to_str().unwrap());
            assert_eq!(sample["feed_text_kind"], "raw");
            assert!(!sample.contains_key("feed_config"));
            assert!(!sample.contains_key("feed_field"));
            let ids: Vec<usize> = sample["feed_token_ids"]
                .split(',')
                .map(|id| id.parse().unwrap())
                .collect();
            assert!(ids.len() <= 16);
            let pieces: Vec<String> = serde_json::from_str(&sample["feed_token_pieces"]).unwrap();
            assert_eq!(
                pieces,
                ids.iter()
                    .map(|id| tokenizer.id_to_token[id].clone())
                    .collect::<Vec<_>>()
            );
            assert!(sample["feed_snippet"].chars().count() <= 256);
            assert!(
                raw.lines()
                    .any(|line| line.contains(&sample["feed_snippet"]))
            );
            let batch: usize = sample["feed_batch"].parse().unwrap();
            let (doc, start, len) = plan[batch - 1];
            let end = start + len;
            assert_eq!(ids, docs[doc][end.saturating_sub(16).max(start)..end]);
            assert_eq!(sample["feed_row"], (doc + 1).to_string());
            assert_eq!(sample["feed_rows"], docs.len().to_string());
            let stream_offset: usize = docs[..doc].iter().map(Vec::len).sum();
            assert_eq!(
                sample["feed_start"],
                (stream_offset + end.saturating_sub(16).max(start)).to_string()
            );
            assert_eq!(sample["feed_end"], (stream_offset + end).to_string());
            assert_eq!(sample["feed_skip_tokens"], "3");
            let source_tokens: Vec<usize> = raw
                .lines()
                .flat_map(|line| tokenizer.encode(line, true))
                .collect();
            let source_start: usize = sample["feed_source_start"].parse().unwrap();
            let source_end: usize = sample["feed_source_end"].parse().unwrap();
            assert_eq!(ids, source_tokens[source_start..source_end]);
            let source_row: usize = sample["feed_source_row"].parse().unwrap();
            assert!(
                raw.lines()
                    .nth(source_row - 1)
                    .unwrap()
                    .contains(&sample["feed_snippet"])
            );
            let consumed: usize = plan[..batch].iter().map(|chunk| chunk.2).sum();
            assert_eq!(sample["feed_epoch_tokens"], consumed.to_string());
            let epoch: usize = sample["feed_epoch"].parse().unwrap();
            assert_eq!(
                sample["feed_tokens"],
                ((epoch - 1) * epoch_tokens + consumed).to_string()
            );
        }
        let last = samples.last().unwrap();
        for (key, expected) in [
            ("feed_epoch", 2),
            ("feed_epochs", 2),
            ("feed_step", updates),
            ("feed_epoch_total", epoch_tokens),
            ("feed_epoch_tokens", epoch_tokens),
            ("feed_batch", plan.len()),
            ("feed_batches", plan.len()),
            ("feed_tokens", epoch_tokens * 2),
            ("feed_bytes", epoch_bytes * 2),
            ("feed_bytes_total", epoch_bytes * 2),
        ] {
            assert_eq!(last[key], expected.to_string(), "{key}");
        }
    }
}

fn run_pssa(
    fixture: &Fixture,
    source: &std::path::Path,
    label: &str,
    telemetry: bool,
    extra: &[&str],
) -> (String, Vec<u8>, Vec<u8>, Duration) {
    run_pssa_window(fixture, source, label, telemetry, extra, 31)
}

fn run_pssa_window(
    fixture: &Fixture,
    source: &std::path::Path,
    label: &str,
    telemetry: bool,
    extra: &[&str],
    max_tokens: usize,
) -> (String, Vec<u8>, Vec<u8>, Duration) {
    let max_tokens = max_tokens.to_string();
    let out = fixture.0.join(format!("{label}.pssa"));
    let curve = fixture.0.join(format!("{label}.csv"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_pssa"));
    command
        .arg("train")
        .arg(source)
        .args([
            "--backend",
            "cpu",
            "--memory",
            "4",
            "--accumulate",
            "2",
            "--epochs",
            "2",
            "--skip-tokens",
            "3",
            "--max-tokens",
            &max_tokens,
            "--dream-every",
            "0",
            "--no-tui",
            "--loss-every",
            "1",
            "--tokens-seen",
            "0",
            "--loss-csv",
        ])
        .arg(&curve)
        .arg("--out")
        .arg(&out)
        .args(extra)
        .env("RAYON_NUM_THREADS", "1");
    // Small parity cases keep their existing shape. Timing cases supply a
    // representative shape explicitly, without duplicate CLI options.
    for (flag, value) in [
        ("--latent", "7"),
        ("--state", "3"),
        ("--key", "4"),
        ("--chunk", "3"),
    ] {
        if !extra.contains(&flag) {
            command.args([flag, value]);
        }
    }
    if !telemetry {
        command.arg("--no-feed-telemetry");
    }
    let started = Instant::now();
    let result = command.output().unwrap();
    let elapsed = started.elapsed();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    // Throughput is wall-clock telemetry, not deterministic training state.
    let curve = fs::read_to_string(curve).unwrap();
    let loss_rows = curve
        .lines()
        .map(|row| row.rsplit_once(',').unwrap().0)
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes();
    (
        String::from_utf8(result.stdout).unwrap(),
        fs::read(out).unwrap(),
        loss_rows,
        elapsed,
    )
}

#[test]
fn pssa_dream_off_telemetry_preserves_checkpoint_and_loss_bits_for_lanes_depth_loops_and_resume() {
    let fixture = Fixture::new();
    let raw =
        "Discard FIRST\nAlpha BETA gamma delta epsilon\nSolo\nδELTA EPSILON! tail final ROW\n";
    let source = fixture.0.join("corpus.txt");
    fs::write(&source, raw).unwrap();
    for (kind, batch, depth, loops) in [
        ("word", "1", "1", "1"),
        ("word", "2", "2", "2"),
        ("bpe", "2", "1", "1"),
    ] {
        let args = [
            "--tokenizer",
            kind,
            "--vocab-size",
            "257",
            "--batch-size",
            batch,
            "--depth",
            depth,
            "--loops",
            loops,
        ];
        let label = format!("{kind}-{batch}-{depth}-{loops}");
        let disabled_label = format!("{label}-disabled");
        let (disabled, reference, curve, _) =
            run_pssa(&fixture, &source, &disabled_label, false, &args);
        assert!(
            !disabled.contains("feed_schema="),
            "explicit opt-out must omit dataset windows"
        );
        assert!(!disabled.contains("dream phase="));
        assert!(disabled.contains("feed_telemetry=false checkpoint_target="));
        let (enabled, actual, enabled_curve, elapsed) =
            run_pssa(&fixture, &source, &format!("{label}-enabled"), true, &args);
        assert_eq!(
            actual, reference,
            "weights, memory, RNG and Adam bits: {label}"
        );
        assert_eq!(
            enabled_curve, curve,
            "exact losses and consumed target counts: {label}"
        );
        assert!(!enabled.contains("dream phase="));
        assert!(enabled.contains("feed_telemetry=true checkpoint_target="));
        let tokenizer = if kind == "word" {
            Tokenizer::from_corpus(raw, true).unwrap()
        } else {
            Tokenizer::from_corpus_bpe(raw, 257).unwrap()
        };
        let docs = CLIHandler::documents(raw, &tokenizer, Some(31), 3).unwrap();
        let plan = sequence_plan(&docs, 3, batch.parse().unwrap()).unwrap();
        let epoch_tokens: usize = plan.iter().flatten().map(|c| c.len).sum();
        let updates = plan.len().div_ceil(2) * 2;
        let samples: Vec<_> = enabled
            .lines()
            .filter(|line| line.contains("feed_schema=2"))
            .map(fields)
            .collect();
        assert!(!samples.is_empty());
        assert!(samples.len() <= updates);
        assert!(
            samples.len() <= 2 + elapsed.as_secs() as usize / 5,
            "headless telemetry is first/final plus at most one sample per five seconds"
        );
        let source_ids: Vec<_> = raw
            .lines()
            .flat_map(|line| tokenizer.encode(line, true))
            .collect();
        for sample in &samples {
            let epoch: usize = sample["feed_epoch"].parse().unwrap();
            let completed: usize = sample["feed_batch"].parse().unwrap();
            let consumed: usize = plan[..completed].iter().flatten().map(|c| c.len).sum();
            let last = plan[completed - 1].last().unwrap();
            let start = (last.start + last.len).saturating_sub(16).max(last.start);
            let end = last.start + last.len;
            let ids: Vec<usize> = sample["feed_token_ids"]
                .split(',')
                .map(|id| id.parse().unwrap())
                .collect();
            assert_eq!(
                ids,
                docs[last.doc][start..end],
                "last consumed lane, excluding target"
            );
            let source_start: usize = sample["feed_source_start"].parse().unwrap();
            let source_end: usize = sample["feed_source_end"].parse().unwrap();
            assert_eq!(
                ids,
                source_ids[source_start..source_end],
                "cyclic source coordinates"
            );
            let pieces: Vec<String> = serde_json::from_str(&sample["feed_token_pieces"]).unwrap();
            assert_eq!(
                pieces,
                ids.iter()
                    .map(|id| tokenizer.id_to_token[id].clone())
                    .collect::<Vec<_>>()
            );
            assert_eq!(sample["feed_row"], (last.doc + 1).to_string());
            assert_eq!(sample["feed_rows"], docs.len().to_string());
            assert_eq!(sample["feed_epoch_tokens"], consumed.to_string());
            assert_eq!(
                sample["feed_tokens"],
                ((epoch - 1) * epoch_tokens + consumed).to_string()
            );
            assert_eq!(sample["feed_dataset"], source.to_str().unwrap());
            assert_eq!(sample["feed_text_kind"], "raw");
            assert!(sample["feed_snippet"].chars().count() <= 256);
            let source_row: usize = sample["feed_source_row"].parse().unwrap();
            assert!(
                raw.lines()
                    .nth(source_row - 1)
                    .unwrap()
                    .contains(&sample["feed_snippet"])
            );
            if kind == "word" {
                let bytes: usize = sample["feed_bytes"].parse().unwrap();
                let total: usize = sample["feed_bytes_total"].parse().unwrap();
                assert!(
                    bytes > 0 && bytes <= total,
                    "counts must come from raw input spans"
                );
            } else {
                let previous_bytes: usize = docs
                    .iter()
                    .flat_map(|doc| &doc[..doc.len() - 1])
                    .map(|&id| tokenizer.token_bytes(id).unwrap().len())
                    .sum();
                let current_bytes: usize = plan[..completed]
                    .iter()
                    .flatten()
                    .flat_map(|c| &docs[c.doc][c.start..c.start + c.len])
                    .map(|&id| tokenizer.token_bytes(id).unwrap().len())
                    .sum();
                assert_eq!(
                    sample["feed_bytes"],
                    ((epoch - 1) * previous_bytes + current_bytes).to_string()
                );
                assert_eq!(sample["feed_bytes_total"], (2 * previous_bytes).to_string());
            }
        }
        let last = samples.last().unwrap();
        assert_eq!(last["feed_epoch"], "2");
        assert_eq!(last["feed_batch"], plan.len().to_string());
        assert_eq!(last["feed_batches"], plan.len().to_string());
        assert_eq!(last["feed_tokens"], (2 * epoch_tokens).to_string());
        assert_eq!(last["feed_epoch_total"], epoch_tokens.to_string());
        assert_eq!(last["feed_epoch_tokens"], epoch_tokens.to_string());
        assert_eq!(last["feed_step"], updates.to_string());
        assert_eq!(last["feed_bytes"], last["feed_bytes_total"]);
        let resume = fixture.0.join(format!("{disabled_label}.pssa"));
        let mut resume_args = args.to_vec();
        resume_args.extend(["--resume", resume.to_str().unwrap()]);
        let (_, resumed_reference, resumed_curve, _) = run_pssa(
            &fixture,
            &source,
            &format!("{label}-resume-disabled"),
            false,
            &resume_args,
        );
        let (resumed_log, resumed_actual, resumed_enabled_curve, _) = run_pssa(
            &fixture,
            &source,
            &format!("{label}-resume-enabled"),
            true,
            &resume_args,
        );
        assert_eq!(
            resumed_actual, resumed_reference,
            "resumed model bits: {label}"
        );
        assert_eq!(
            resumed_enabled_curve, resumed_curve,
            "resumed loss bits: {label}"
        );
        let last = resumed_log
            .lines()
            .filter(|line| line.contains("feed_schema=2"))
            .map(fields)
            .last()
            .unwrap();
        assert_eq!(
            last["feed_step"],
            (2 * updates).to_string(),
            "global step survives resume"
        );
    }
}

#[test]
#[ignore = "paired CPU telemetry overhead measurement; run explicitly with --ignored --nocapture"]
fn telemetry_overhead_paired_cpu_measurement() {
    let fixture = Fixture::new();
    let raw = (0..256)
        .map(|row| {
            (0..96)
                .map(|word| format!("token{}", (word + row) % 96))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let source = fixture.0.join("corpus.txt");
    fs::write(&source, raw).unwrap();
    for kind in ["word", "bpe"] {
        let args = [
            "--tokenizer",
            kind,
            "--vocab-size",
            "257",
            "--batch-size",
            "2",
            "--latent",
            "64",
            "--state",
            "16",
            "--key",
            "16",
            "--chunk",
            "32",
        ];
        // Keep model and selected input identical, warm both paths, then reverse
        // order on alternate pairs to reduce startup and scheduling bias.
        let _ = run_pssa_window(
            &fixture,
            &source,
            &format!("{kind}-warm-disabled"),
            false,
            &args,
            6144,
        );
        let _ = run_pssa_window(
            &fixture,
            &source,
            &format!("{kind}-warm-enabled"),
            true,
            &args,
            6144,
        );
        let mut disabled = Vec::new();
        let mut enabled = Vec::new();
        for pair in 0..7 {
            let off = format!("{kind}-pair{pair}-disabled");
            let on = format!("{kind}-pair{pair}-enabled");
            let (a, b) = if pair % 2 == 0 {
                (
                    run_pssa_window(&fixture, &source, &off, false, &args, 6144),
                    run_pssa_window(&fixture, &source, &on, true, &args, 6144),
                )
            } else {
                let b = run_pssa_window(&fixture, &source, &on, true, &args, 6144);
                (
                    run_pssa_window(&fixture, &source, &off, false, &args, 6144),
                    b,
                )
            };
            assert_eq!(a.1, b.1);
            assert_eq!(a.2, b.2);
            let training_seconds = |log: &str| {
                log.lines()
                    .find_map(|line| line.strip_prefix("training_seconds="))
                    .and_then(|line| line.split_whitespace().next())
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            };
            println!(
                "telemetry_pair tokenizer={kind} pair={pair} disabled_seconds={:.6} enabled_seconds={:.6} disabled_training_seconds={:.3} enabled_training_seconds={:.3}",
                a.3.as_secs_f64(),
                b.3.as_secs_f64(),
                training_seconds(&a.0),
                training_seconds(&b.0)
            );
            disabled.push(a.3.as_secs_f64());
            enabled.push(b.3.as_secs_f64());
        }
        disabled.sort_by(f64::total_cmp);
        enabled.sort_by(f64::total_cmp);
        println!(
            "telemetry_overhead tokenizer={kind} pairs=7 warmed=true rayon_threads=1 latent=64 state=16 key=16 chunk=32 selected_tokens=6144 epochs=2 disabled_median_seconds={:.6} enabled_median_seconds={:.6} ratio={:.4} (subprocess wall; includes startup/tokenizer/checkpoint/CSV)",
            disabled[3],
            enabled[3],
            enabled[3] / disabled[3]
        );
    }
}
