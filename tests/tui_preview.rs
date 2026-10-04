//! Exercise the real read-only preview subprocess and plain-log entry point.
use oxide_ai_pssa::{
    checkpoint,
    dataset::Tokenizer,
    pssa::{PSSAConfigV2, PSSALayerV2},
};
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn preview_worker_loads_checkpoint_with_spaces_and_never_writes_it() {
    let path = std::env::temp_dir().join(format!(
        "pssa preview {} checkpoint.pssa",
        std::process::id()
    ));
    let tokenizer =
        Tokenizer::from_vocabulary(&["<unk>".into(), "hello".into(), "world".into()]).unwrap();
    let mut model = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 3,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 2,
            chunk_len: 2,
            ..Default::default()
        },
        42,
    );
    model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
    checkpoint::save_model(&model, &path).unwrap();
    let before = fs::read(&path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(["tui", "--preview-worker"])
        .arg(&path)
        .arg("2")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    let after = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(before, after);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(value.get("error").is_none(), "{value}");
    assert!(!value["text"].as_str().unwrap().is_empty());
    let marks = value["confidence"].as_array().unwrap();
    assert_eq!(marks.len(), 32);
    assert_eq!(
        marks.last().unwrap()["end"].as_u64().unwrap() as usize,
        value["text"].as_str().unwrap().len()
    );
    assert!(
        marks
            .iter()
            .all(|mark| (0.0..=1.0).contains(&mark["probability"].as_f64().unwrap()))
    );

    let output = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(["tui", "--preview-worker"])
        .arg(&path)
        .arg("1")
        .output()
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(value["error"].as_str().unwrap().contains("No such file"));
}

#[test]
fn non_tty_tui_still_passes_plain_progress_without_starting_workers() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .arg("tui")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"training loss=3.0\nlast_checkpoint=path with spaces.pssa\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        output.stdout,
        b"training loss=3.0\nlast_checkpoint=path with spaces.pssa\n"
    );
    assert!(output.stderr.is_empty());
}
