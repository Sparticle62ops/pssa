use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "oxide-comparison-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(args)
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap()
}
fn success(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "{args:?}\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}
fn rows(path: &str) -> Vec<Vec<String>> {
    let text = fs::read_to_string(path).unwrap();
    assert_eq!(
        text.lines().next().unwrap(),
        "tokens_seen,updates,loss,tokens_per_second"
    );
    text.lines()
        .skip(1)
        .map(|s| s.split(',').map(str::to_owned).collect())
        .collect()
}

#[test]
fn both_trainers_append_global_csv_and_score_strict_heldout_slices() {
    let dir = Scratch::new();
    let corpus = dir.file("corpus.txt");
    let heldout = dir.file("heldout.txt");
    fs::write(
        &corpus,
        "alpha beta gamma delta\nepsilon zeta eta theta\niota kappa lambda mu\n",
    )
    .unwrap();
    fs::write(&heldout, "iota kappa lambda mu\n").unwrap();
    let pssa = dir.file("pssa.pssa");
    let transformer = dir.file("transformer.trfm");
    for (train, eval, model, csv) in [
        ("train", "evaluate", &pssa, dir.file("pssa.csv")),
        (
            "train-transformer",
            "evaluate-transformer",
            &transformer,
            dir.file("transformer.csv"),
        ),
    ] {
        let mut args = vec![
            train,
            &corpus,
            "-o",
            model,
            "-e",
            "1",
            "--tokenizer",
            "word",
            "--chunk",
            "2",
            "--accumulate",
            "1",
            "--max-tokens",
            "4",
            "--total-updates",
            "8",
            "--loss-csv",
            &csv,
            "--loss-every",
            "2",
        ];
        if train == "train" {
            args.extend([
                "--latent", "8", "--state", "2", "--key", "2", "--memory", "2",
            ]);
        } else {
            args.extend(["--tokenizer-from", &pssa]);
        }
        success(&args);
        let first = rows(&csv);
        assert_eq!(first.len(), 2); // cadence + final partial interval
        assert_eq!(&first[0][..2], ["2", "1"]);
        assert_eq!(&first[1][..2], ["3", "2"]);
        success(&[
            train,
            &corpus,
            "-o",
            model,
            "--resume",
            model,
            "-e",
            "1",
            "--accumulate",
            "1",
            "--max-tokens",
            "4",
            "--skip-tokens",
            "4",
            "--loss-csv",
            &csv,
            "--loss-every",
            "2",
        ]);
        let all = rows(&csv);
        assert_eq!(&all.last().unwrap()[..2], ["6", "4"]);
        for row in all {
            assert_eq!(row.len(), 4);
            assert!(row[2].parse::<f64>().unwrap().is_finite());
            assert!(row[3].parse::<f64>().unwrap() >= 0.0);
        }
        let before = fs::read(model).unwrap();
        let sliced = success(&[
            eval,
            &corpus,
            "-m",
            model,
            "--skip-tokens",
            "8",
            "--max-tokens",
            "4",
        ]);
        let standalone = success(&[eval, &heldout, "-m", model]);
        assert_eq!(sliced, standalone);
        let metrics: serde_json::Value = serde_json::from_str(sliced.trim()).unwrap();
        assert_eq!(metrics["token_count"], 3);
        let ce = metrics["cross_entropy"].as_f64().unwrap();
        assert!((ce.exp() - metrics["perplexity"].as_f64().unwrap()).abs() < 1e-5);
        assert_eq!(before, fs::read(model).unwrap());
        for bad in [
            vec!["--skip-tokens", "12"],
            vec!["--skip-tokens", "10", "--max-tokens", "4"],
            vec!["--max-tokens", "0"],
        ] {
            let mut args = vec![eval, &corpus, "-m", model];
            args.extend(bad);
            let output = run(&args);
            assert_eq!(output.status.code(), Some(2));
            assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
        }
        let new_csv = dir.file(&format!("new-{train}.csv"));
        let output = run(&[
            train,
            &corpus,
            "-o",
            model,
            "--resume",
            model,
            "-e",
            "1",
            "--max-tokens",
            "4",
            "--loss-csv",
            &new_csv,
        ]);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("tokens-seen"));
        assert_eq!(before, fs::read(model).unwrap());
    }
    assert!(success(&["help"]).contains("--resume"));
}
