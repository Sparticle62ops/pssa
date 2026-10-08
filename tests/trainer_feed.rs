use pssa::{cli::CLIHandler, dataset::Tokenizer, training::chunk_plan};
use std::{collections::HashMap, fs, path::PathBuf, process::Command};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("pssa-transformer-feed-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
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
    let run = |label: &str, cached: bool| {
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
    let (uncached, reference) = run("uncached", false);
    assert!(!cache.exists(), "cache stays opt-in");
    let (built, built_bits) = run("built", true);
    let cache_bits = fs::read(&cache).unwrap();
    let (reused, reused_bits) = run("reused", true);
    assert_eq!(built_bits, reference);
    assert_eq!(reused_bits, reference);
    assert_eq!(fs::read(&cache).unwrap(), cache_bits);
    assert!(built.contains("token_cache=built"));
    assert!(reused.contains("token_cache=reused"));
    for log in [&uncached, &built, &reused] {
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
