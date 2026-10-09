use pssa::{
    backend::Device,
    checkpoint,
    dataset::Tokenizer,
    pssa::{PSSAConfigV2, PSSALayerV2},
};
use std::{fs, process::Command};

#[test]
fn finite_loss_blowup_aborts_without_overwriting_checkpoint_for_all_training_paths() {
    let dir = std::env::temp_dir().join(format!("pssa-loss-guard-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let text = "alpha beta ".repeat(12);
    let corpus = dir.join("corpus.txt");
    fs::write(&corpus, &text).unwrap();
    let tok = Tokenizer::from_corpus(&text, true).unwrap();
    for (batch, depth, loops) in [(1, 1, 1), (2, 1, 1), (1, 2, 2), (2, 2, 2)] {
        let mut model = PSSALayerV2::new_with_device(
            PSSAConfigV2 {
                d_latent: 4,
                d_state: 1,
                d_mem_key: 1,
                d_vocab: tok.vocab_size,
                mem_capacity: 2,
                chunk_len: 1,
                depth,
                ..Default::default()
            },
            42,
            Device::Cpu,
        );
        model.vocabulary = tok.ordered_vocabulary().unwrap();
        // An enormous but finite head produces high CE without needing non-finite tensors.
        for (i, value) in model.unembed_w.data.iter_mut().enumerate() {
            *value = if i < 4 { 1e7 } else { -1e7 };
        }
        let resume = dir.join(format!("seed-{batch}-{depth}-{loops}.pssa"));
        checkpoint::save_model(&model, &resume).unwrap();
        let before = fs::read(&resume).unwrap();
        for preexisting in [false, true] {
            let out = dir.join("out.pssa");
            if preexisting {
                fs::write(&out, b"keep previous good checkpoint").unwrap();
            }
            let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
                .arg("train")
                .arg(&corpus)
                .arg("--resume")
                .arg(&resume)
                .arg("--out")
                .arg(&out)
                .args([
                    "--epochs",
                    "1",
                    "--backend",
                    "cpu",
                    "--accumulate",
                    "1",
                    "--batch-size",
                    &batch.to_string(),
                    "--loops",
                    &loops.to_string(),
                    "--loss-guard-patience",
                    "1",
                    "--no-tui",
                ])
                .output()
                .unwrap();
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(!result.status.success(), "{error}");
            assert!(
                error.contains("loss_guard=halted") && error.contains("without checkpoint"),
                "{error}"
            );
            assert_eq!(fs::read(&resume).unwrap(), before);
            if preexisting {
                assert_eq!(fs::read(&out).unwrap(), b"keep previous good checkpoint");
                fs::remove_file(out).unwrap();
            } else {
                assert!(!out.exists());
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_validates_guard_flags_before_loading_data_and_documents_defaults() {
    for flag in ["--loss-guard-high-factor", "--loss-guard-jump-factor"] {
        for value in ["0", "1", "NaN", "inf", "bad"] {
            let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
                .args(["train", "missing-file", flag, value])
                .output()
                .unwrap();
            assert!(!result.status.success());
            assert!(String::from_utf8_lossy(&result.stderr).contains(flag));
        }
    }
    let result = Command::new(env!("CARGO_BIN_EXE_pssa"))
        .args(["train", "missing-file", "--loss-guard-patience", "0"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&result.stderr).contains("--loss-guard-patience"));
    let help = Command::new(env!("CARGO_BIN_EXE_pssa"))
        .args(["train", "--help"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&help.stdout).contains("--loss-guard-high-factor"));
}
