use oxide_ai_pssa::tui;

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
fn non_tty_tui_still_passes_plain_logs_without_creating_chats() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let dir = std::env::temp_dir().join(format!("oxide no chats {}", std::process::id()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
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
