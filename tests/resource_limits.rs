//! Resource limits exercise the real CLI child, not the test process.
use oxide_ai_pssa::cli::CLIHandler;
use std::{fs, path::PathBuf, process::Command};

#[test]
fn invalid_limits_fail_before_loading_data_or_checkpoint() {
    for (flag, value) in [
        ("--threads", ""),
        ("--ram-mib", ""),
        ("--threads", "0"),
        ("--threads", "-1"),
        ("--threads", "65537"),
        ("--threads", "NaN"),
        ("--ram-mib", "0"),
        ("--ram-mib", "184467440737095516160"),
        ("--batch-size", "0"),
        ("--max-tokens", "0"),
    ] {
        let error = CLIHandler::parse_and_execute(
            [
                "oxide",
                "train",
                "missing corpus",
                "--resume",
                "missing checkpoint",
                flag,
                value,
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        )
        .unwrap_err();
        assert!(error.contains(flag), "{flag}={value}: {error}");
        assert!(!error.contains("cannot inspect"), "{error}");
    }
    for options in [
        vec!["--threads"],
        vec!["--ram-mib"],
        vec!["--threads", "2", "--threads", "3"],
    ] {
        let mut args: Vec<_> = ["oxide", "train", "missing corpus"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        args.extend(options.into_iter().map(str::to_owned));
        let error = CLIHandler::parse_and_execute(args).unwrap_err();
        assert!(
            error.contains("requires a value") || error.contains("more than once"),
            "{error}"
        );
    }
}

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn limited_cpu_training_preserves_plain_logs_batch_token_caps_and_resume_paths() {
    let root = Fixture(
        std::env::temp_dir().join(format!("pssa-limits-{} with spaces", std::process::id())),
    );
    fs::create_dir_all(&root.0).unwrap();
    let corpus = root.0.join("source file.txt");
    fs::write(&corpus, "one two three four five six seven eight\nnine ten eleven twelve thirteen fourteen fifteen\n".repeat(3)).unwrap();
    let first = root.0.join("default checkpoint.pssa");
    let limited = root.0.join("limited checkpoint.pssa");
    let resumed = root.0.join("resumed checkpoint.pssa");
    for (output, constrained, resume) in [
        (&first, false, None),
        (&limited, true, None),
        (&resumed, true, Some(&limited)),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"));
        command
            .arg("train")
            .arg(&corpus)
            .args([
                "--backend",
                "cpu",
                "--tokenizer",
                "word",
                "--latent",
                "4",
                "--state",
                "2",
                "--key",
                "2",
                "--memory",
                "4",
                "--epochs",
                "1",
                "--chunk",
                "4",
                "--accumulate",
                "1",
                "--batch-size",
                "2",
                "--max-tokens",
                "24",
                "--no-tui",
                "--out",
            ])
            .arg(output)
            .env("RAYON_NUM_THREADS", "1");
        if constrained {
            command.args(["--threads", "1"]);
            // Keep the real-kernel test usable on Linux images without util-linux;
            // malformed/missing prlimit is separately reported by the wrapper.
            if cfg!(target_os = "linux")
                && Command::new("prlimit").arg("--version").output().is_ok()
            {
                command.args(["--ram-mib", "2048"]);
            }
        }
        if let Some(path) = resume {
            command.arg("--resume").arg(path);
        }
        let result = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(result.status.success(), "{stdout}\n{stderr}");
        assert!(
            !stdout.contains('\x1b') && !stderr.contains('\x1b'),
            "plain logs changed"
        );
        assert!(stdout.contains("backend=cpu"), "{stdout}");
        assert!(stdout.contains("batch_size=2"), "{stdout}");
        let epoch = stdout
            .lines()
            .find(|line| line.starts_with("epoch ") && line.contains("tokens="))
            .unwrap();
        let tokens: usize = epoch
            .split_whitespace()
            .find_map(|field| field.strip_prefix("tokens="))
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            (1..=24).contains(&tokens),
            "corpus cap not honored: {epoch}"
        );
        assert!(output.is_file());
        if resume.is_some() {
            assert!(stdout.contains("resumed_from="));
        }
    }
    assert_eq!(
        fs::read(first).unwrap(),
        fs::read(limited).unwrap(),
        "an explicit one-thread pool / roomy RAM budget must not alter this CPU fixture's checkpoint"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn missing_prlimit_is_an_error_not_a_silently_ignored_ram_budget() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args([
            "train",
            "missing corpus",
            "--ram-mib",
            "512",
            "--backend",
            "cpu",
        ])
        .env("PATH", "/nonexistent-pssa-resource-limit-tools")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("cannot apply --ram-mib"), "{error}");
    assert!(error.contains("prlimit"), "{error}");
}
