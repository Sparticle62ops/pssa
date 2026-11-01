//! Tests for the universal converter: architecture detection on synthetic
//! configs and an end-to-end conversion of a tiny synthetic safetensors file.
use pssa::convert::detect::detect;
use pssa::convert::{convert, ConvertOptions};
use serde_json::json;

#[test]
fn qwen3_dense_is_detected() {
    let config = json!({
        "model_type": "qwen3",
        "vocab_size": 151_936usize,
        "hidden_size": 2_048usize,
        "num_hidden_layers": 28usize,
        "tie_word_embeddings": true
    });
    let info = detect(&config).unwrap();
    assert_eq!(info.family.label(), "Qwen3 (dense)");
    assert_eq!(info.vocab_size, 151_936);
    assert_eq!(info.hidden_size, 2_048);
    assert!(info.tie_word_embeddings);
}

#[test]
fn llama_family_is_detected() {
    let config = json!({
        "model_type": "llama",
        "vocab_size": 128_256usize,
        "hidden_size": 4_096usize,
        "num_hidden_layers": 32usize,
        "tie_word_embeddings": false
    });
    let info = detect(&config).unwrap();
    assert_eq!(info.family.label(), "Llama");
    assert!(!info.tie_word_embeddings);
}

#[test]
fn hybrid_linear_attention_is_rejected_cleanly() {
    let config = json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
            "vocab_size": 248_320usize,
            "hidden_size": 2_560usize,
            "num_hidden_layers": 32usize
        }
    });
    let error = detect(&config).unwrap_err();
    assert!(error.contains("hybrid linear-attention"), "unexpected error: {error}");
}

#[test]
fn unknown_model_type_names_the_family() {
    let config = json!({
        "model_type": "deepseek_v3",
        "vocab_size": 129_280usize,
        "hidden_size": 7_168usize,
        "num_hidden_layers": 61usize
    });
    let error = detect(&config).unwrap_err();
    assert!(error.contains("deepseek_v3"), "unexpected error: {error}");
}

#[test]
fn missing_field_is_named() {
    let config = json!({ "model_type": "llama", "vocab_size": 32_000usize, "num_hidden_layers": 32usize });
    let error = detect(&config).unwrap_err();
    assert!(error.contains("hidden_size"), "unexpected error: {error}");
}

#[test]
fn nested_text_config_sizes_are_read() {
    let config = json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "vocab_size": 262_144usize,
            "hidden_size": 2_304usize,
            "num_hidden_layers": 34usize,
            "layer_types": ["sliding_attention", "full_attention"]
        }
    });
    let info = detect(&config).unwrap();
    assert_eq!(info.family.label(), "Gemma");
    assert_eq!(info.vocab_size, 262_144);
    assert_eq!(info.hidden_size, 2_304);
}

#[test]
fn moe_flag_is_reported() {
    let config = json!({
        "model_type": "mixtral",
        "vocab_size": 32_000usize,
        "hidden_size": 4_096usize,
        "num_hidden_layers": 32usize,
        "num_local_experts": 8usize
    });
    let info = detect(&config).unwrap();
    assert!(info.mixture_of_experts);
}

fn write_tiny_safetensors(path: &std::path::Path, embedding_name: &str) {
    let vocab = 8usize;
    let hidden = 16usize;
    let mut data = vec![0u8; vocab * hidden * 4];
    for (index, chunk) in data.chunks_exact_mut(4).enumerate() {
        let value = ((index % 31) as f32 - 15.0) / 15.0;
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    let header = format!(
        "{{\"{embedding_name}\":{{\"dtype\":\"F32\",\"shape\":[{vocab},{hidden}],\"data_offsets\":[0,{}]}}}}",
        data.len()
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn end_to_end_tiny_conversion() {
    let dir = std::env::temp_dir().join("pssa_convert_test");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tiny.safetensors");
    write_tiny_safetensors(&source, "model.embed_tokens.weight");
    let config = json!({
        "model_type": "llama",
        "vocab_size": 8usize,
        "hidden_size": 16usize,
        "num_hidden_layers": 1usize
    });
    let info = detect(&config).unwrap();
    let out = dir.join("tiny.v6");
    let report = convert(ConvertOptions {
        shards: vec![source],
        info,
        target_mode: "compact".into(),
        seed: 7,
        out_path: out.clone(),
    })
    .unwrap();
    assert_eq!(report.vocab_size, 8);
    assert_eq!(report.hidden_size, 16);
    assert_eq!(report.depth_used, 1);
    assert_eq!(report.embedding_tensor, "model.embed_tokens.weight");
    assert!(report.checkpoint_bytes > 0);
    assert!(out.exists());
    assert!(report.logit_rms_before.is_finite());
    assert!(report.logit_scale.is_finite() && report.logit_scale > 0.0);
}

#[test]
fn missing_embedding_fails_with_a_useful_message() {
    let dir = std::env::temp_dir().join("pssa_convert_test_noneg");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("noembed.safetensors");
    write_tiny_safetensors(&source, "some.random.weight");
    let config = json!({
        "model_type": "llama",
        "vocab_size": 8usize,
        "hidden_size": 16usize,
        "num_hidden_layers": 1usize
    });
    let info = detect(&config).unwrap();
    let error = convert(ConvertOptions {
        shards: vec![source],
        info,
        target_mode: "compact".into(),
        seed: 7,
        out_path: dir.join("never.v6"),
    })
    .unwrap_err();
    assert!(error.contains("no embedding tensor"), "unexpected error: {error}");
}

#[test]
fn size_target_solves_the_latent_width() {
    let dir = std::env::temp_dir().join("pssa_convert_test_size");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tiny.safetensors");
    write_tiny_safetensors(&source, "model.embed_tokens.weight");
    let config = json!({
        "model_type": "llama",
        "vocab_size": 8usize,
        "hidden_size": 16usize,
        "num_hidden_layers": 1usize
    });
    let info = detect(&config).unwrap();
    let report = convert(ConvertOptions {
        shards: vec![source],
        info,
        target_mode: "2k".into(),
        seed: 7,
        out_path: dir.join("tiny-2k.v6"),
    })
    .unwrap();
    assert_eq!(report.latent_dim, 9);
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("solved to latent 9")),
        "notes: {:?}",
        report.notes
    );
}
