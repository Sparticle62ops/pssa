//! Exercise the wizard's CLI contract against a real CPU trainer, including a
//! resume path with spaces. Model arithmetic and checkpoint tolerances are untouched.
use std::{fs, path::PathBuf, process::Command};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("pssa-setup-cli-{} with spaces", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        fs::write(
            path.join("source.txt"),
            "the small net learns the same short sentence.\n".repeat(4),
        )
        .unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn wizard_cli_contract_trains_and_resumes_with_plain_logs() {
    let fixture = Fixture::new();
    let first = fixture.0.join("first checkpoint.pssa");
    let second = fixture.0.join("resumed checkpoint.pssa");
    for (output, resume) in [(&first, None), (&second, Some(&first))] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"));
        command
            .args(["train", "--data"])
            .arg(format!("file:{}", fixture.0.join("source.txt").display()))
            .args([
                "--latent",
                "4",
                "--state",
                "2",
                "--vocab-size",
                "257",
                "--depth",
                "2",
                "--loops",
                "2",
                "--lr",
                "0.001",
                "--epochs",
                "1",
                "--max-tokens",
                "32",
                "--seed",
                "42",
                "--chunk",
                "4",
                "--accumulate",
                "2",
                "--backend",
                "cpu",
                "--out",
            ])
            .arg(output);
        if let Some(resume) = resume {
            // Piped/non-TTY output must stay plain even without --no-tui.
            command.arg("--resume").arg(resume);
        } else {
            command.arg("--no-tui");
        }
        let result = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(result.status.success(), "{stdout}\n{stderr}");
        assert!(stdout.contains("backend=cpu"), "{stdout}");
        assert!(stdout.contains("depth=2 loops=2"), "{stdout}");
        assert!(
            !stdout.contains('\x1b'),
            "non-TTY and --no-tui must stay plain"
        );
        assert!(
            !stderr.contains('\x1b'),
            "child errors must also stay plain"
        );
        assert!(output.is_file());
        if resume.is_some() {
            assert!(stdout.contains("resumed_from="));
        }
    }
}

#[test]
fn invalid_backend_is_rejected_before_dataset_loading() {
    let result = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(["train", "missing-file", "--backend", "tpu", "--no-tui"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--backend must be"));
}
