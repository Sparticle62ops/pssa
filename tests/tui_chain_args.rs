use pssa::tui;

#[test]
fn chain_requires_a_directory_value() {
    assert!(tui::run(&["--chain".into()]).is_err());
    assert!(tui::run(&["--chain".into(), "--no-tui".into()]).is_err());
    assert!(tui::run(&["--compare".into()]).is_err());
    assert!(tui::run(&["--compare".into(), "--chain".into()]).is_err());
}

#[test]
fn chats_directory_requires_a_value() {
    assert!(tui::run(&["--chats-dir".into()]).is_err());
    assert!(tui::run(&["--chats-dir".into(), "--chain".into()]).is_err());
}

#[test]
fn tui_rejects_repeated_options_and_mixed_chain_aliases() {
    for args in [
        vec!["--chain", "first", "--chain", "second"],
        vec!["-c", "first", "--chain", "second"],
        vec!["--chats-dir", "first", "--chats-dir", "second"],
        vec!["--compare", "first.log", "--compare", "second.log"],
    ] {
        let args: Vec<String> = args.into_iter().map(String::from).collect();
        assert!(tui::run(&args).is_err(), "accepted duplicate TUI args: {args:?}");
    }
}

#[cfg(unix)]
#[test]
fn comparison_fifo_is_rejected_without_opening_or_hanging() {
    use std::{process::{Command, Stdio}, time::{Duration, Instant}};
    let fifo = std::env::temp_dir().join(format!("pssa-comparison-fifo-{}", std::process::id()));
    assert!(Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let mut child = Command::new(env!("CARGO_BIN_EXE_pssa"))
        .args(["tui", "--compare"]).arg(&fifo)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        child.kill().unwrap();
        child.wait().unwrap();
        std::fs::remove_file(fifo).unwrap();
        panic!("comparison FIFO blocked the TUI");
    }
    let output = child.wait_with_output().unwrap();
    std::fs::remove_file(fifo).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("regular log file"));
}

#[test]
fn non_tty_tui_still_passes_plain_logs_without_creating_chats() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let dir = std::env::temp_dir().join(format!("pssa no chats {}", std::process::id()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_pssa"))
        .args(["tui", "--chain", "chain with spaces", "--chats-dir"])
        .arg(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\x1b[32mtraining 1/2 loss=1.0\x1b[0m\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "training 1/2 loss=1.0\n"
    );
    assert!(!dir.exists());
}
