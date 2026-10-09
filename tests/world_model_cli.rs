use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pssa"))
}

#[test]
fn world_model_help_is_discoverable_and_does_not_run_training() {
    for args in [
        vec!["help"],
        vec!["world-model", "--help"],
        vec!["help", "world-model"],
    ] {
        let output = binary().args(args.clone()).output().unwrap();
        assert!(
            output.status.success(),
            "args={args:?} stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            text.contains("world-model"),
            "args={args:?} stdout={} stderr={}",
            text,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!text.contains("world_model=start"));
    }
    let output = binary().args(["world-model", "--help"]).output().unwrap();
    let help = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--seed",
        "--side",
        "--hidden",
        "--state",
        "--categories",
        "--train-episodes",
        "--heldout-episodes",
        "--epochs",
        "--horizon",
        "--rollout-horizon",
        "--lr",
        "--kl-weight",
        "--auxiliary-weight",
        "--grad-clip",
    ] {
        assert!(help.contains(flag), "missing {flag}");
    }
}

#[test]
fn world_model_rejects_invalid_and_text_training_options_before_start() {
    for args in [
        vec!["--side", "2"],
        vec!["--hidden", "49"],
        vec!["--state", "0"],
        vec!["--categories", "33"],
        vec!["--train-episodes", "0"],
        vec!["--heldout-episodes", "65"],
        vec!["--horizon", "25"],
        vec!["--horizon", "2", "--rollout-horizon", "3"],
        vec!["--epochs", "13"],
        vec!["--lr", "NaN"],
        vec!["--kl-weight", "inf"],
        vec!["--auxiliary-weight", "-1"],
        vec!["--grad-clip", "0"],
        vec!["--seed", "-1"],
        vec!["--epochs"],
        vec!["--epochs", "1", "--epochs", "2"],
        vec!["corpus.txt"],
        vec!["--resume", "missing.pssa"],
        vec!["--out", "bad.pssa"],
        vec!["--backend", "cuda"],
        vec!["--memory", "64"],
        vec!["--dream-every", "1"],
        vec![
            "--train-episodes",
            "128",
            "--epochs",
            "12",
            "--hidden",
            "48",
            "--horizon",
            "24",
        ],
    ] {
        let output = binary().arg("world-model").args(&args).output().unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("world_model=start"),
            "{args:?}"
        );
        assert!(!output.stderr.is_empty(), "{args:?}");
    }
    let output = binary()
        .args(["train", "--categories", "4"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown option"));
}

#[test]
fn world_model_cli_runs_both_models_without_checkpoint_and_pipes_to_tui() {
    let path = std::env::temp_dir().join(format!("pssa-world-cli-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("keep.pssa"), b"do not replace").unwrap();
    let output = binary()
        .current_dir(&path)
        .args([
            "world-model",
            "--side",
            "3",
            "--hidden",
            "6",
            "--state",
            "2",
            "--categories",
            "4",
            "--train-episodes",
            "4",
            "--heldout-episodes",
            "2",
            "--horizon",
            "4",
            "--rollout-horizon",
            "3",
            "--epochs",
            "2",
            "--seed",
            "73",
            "--lr",
            "0.003",
            "--kl-weight",
            "0.1",
            "--auxiliary-weight",
            "0.25",
            "--grad-clip",
            "5",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout.clone()).unwrap();
    assert_eq!(text.matches("world_model=epoch").count(), 2);
    assert!(text.contains("world_model=result variant=stochastic"));
    assert!(text.contains("world_model=result variant=autoregressive"));
    assert_eq!(text.matches("updates=8").count(), 4); // structured and human report for each model
    assert!(text.contains("one_step_nll=") && text.contains("rollout_nll="));
    assert!(text.contains("world_model=complete checkpoint=none"));
    assert!(text.contains("NOT parameter- or time-matched"));
    assert_eq!(fs::read(path.join("keep.pssa")).unwrap(), b"do not replace");
    assert_eq!(fs::read_dir(&path).unwrap().count(), 1);
    let mut tui = binary()
        .current_dir(&path)
        .arg("tui")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    tui.stdin.take().unwrap().write_all(&output.stdout).unwrap();
    let displayed = tui.wait_with_output().unwrap();
    assert!(
        displayed.status.success(),
        "{}",
        String::from_utf8_lossy(&displayed.stderr)
    );
    assert!(String::from_utf8_lossy(&displayed.stdout).contains("world_model=complete"));
    fs::remove_dir_all(path).unwrap();
}
