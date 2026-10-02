use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn chat_handles_info_and_updates_temperature() {
    let model = format!("{}/data/model.pssa", env!("CARGO_MANIFEST_DIR"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(["chat", "--model", &model, "--temp", "0.7"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start chat");
    child
        .stdin
        .take()
        .expect("chat stdin")
        .write_all(b"/info\n/temp 0\n/exit\n")
        .expect("send chat commands");
    let output = child.wait_with_output().expect("wait for chat");
    assert!(
        output.status.success(),
        "chat failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("model: {model}")), "{stdout}");
    assert!(stdout.contains("memory slots:"), "{stdout}");
    assert!(stdout.contains("adapters:"), "{stdout}");
    assert!(stdout.contains("temperature set to 0.0000"), "{stdout}");
}
