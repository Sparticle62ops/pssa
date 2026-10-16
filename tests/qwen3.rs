use pssa::qwen3::{
    Qwen3Config, Qwen3ConversionOptions, Qwen3Model, Qwen3PssaTarget, Qwen3ResourceEstimate,
};

fn qwen3_1_7b_json() -> String {
    r#"{
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000,
        "hidden_act": "silu",
        "tie_word_embeddings": true
    }"#
    .to_owned()
}

#[test]
fn official_17b_shape_and_parameter_count_are_exact() {
    let config = Qwen3Config::from_json_str(&qwen3_1_7b_json()).unwrap();
    assert_eq!(config, Qwen3Config::qwen3_1_7b());
    assert_eq!(config.query_dim(), 2048);
    assert_eq!(config.key_value_dim(), 1024);
    assert_eq!(config.parameter_count().unwrap(), 1_720_574_976);
    assert_eq!(config.parameter_count().unwrap() * 2, 3_441_149_952);
}

#[test]
fn qwen06b_query_width_is_independent_of_residual_width() {
    let config = Qwen3Config::from_json_str(
        &qwen3_1_7b_json()
            .replace("2048", "1024")
            .replace("6144", "3072"),
    )
    .unwrap();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.query_dim(), 2048);
    assert_eq!(config.key_value_dim(), 1024);
}

#[test]
fn target_memory_values_follow_pssa_latent_width() {
    let config = Qwen3Config::qwen3_1_7b();
    let compact = Qwen3PssaTarget::compact_for(&config);
    let source_width = Qwen3PssaTarget::source_width_for(&config);
    let compact_estimate =
        Qwen3ResourceEstimate::for_config_with_target(&config, &compact).unwrap();
    let source_estimate =
        Qwen3ResourceEstimate::for_config_with_target(&config, &source_width).unwrap();

    assert_eq!(compact_estimate.memory_bank_bytes, 16_744_448);
    assert_eq!(source_estimate.memory_bank_bytes, 119_504_896);
    assert_eq!(compact_estimate.scan_len, 64);
    assert_eq!(compact_estimate.scan_workspace_bytes, 58_720_256);
    assert!(source_estimate.memory_bank_bytes > compact_estimate.memory_bank_bytes * 7);
    assert!(
        source_estimate.pssa_target_allocation_bytes
            > compact_estimate.pssa_target_allocation_bytes
    );
    assert!(compact_estimate.exceeds_one_gib());
}

#[test]
fn legacy_options_still_describe_an_explicit_compact_target() {
    let config = Qwen3Config::qwen3_1_7b();
    let options = Qwen3ConversionOptions::default();
    let estimate = Qwen3ResourceEstimate::for_config(&config, &options).unwrap();
    assert_eq!(estimate.memory_bank_bytes, 16_744_448);
    assert_eq!(estimate.pssa_target_allocation_bytes > 0, true);
}

#[test]
fn target_scan_length_overflow_is_reported_without_allocating() {
    let mut target = Qwen3PssaTarget::compact_for(&Qwen3Config::qwen3_1_7b());
    target.chunk_len = usize::MAX;
    let error = Qwen3ResourceEstimate::for_config_with_target(&Qwen3Config::qwen3_1_7b(), &target)
        .unwrap_err();
    assert!(error.contains("scan length"));
}

#[test]
fn full_reference_runner_is_a_non_allocating_17b_smoke_guard() {
    let error = Qwen3Model::random(Qwen3Config::qwen3_1_7b(), 7).unwrap_err();
    assert!(error.contains("reference runner is limited"));
}

#[test]
fn tiny_qwen_path_exercises_gqa_and_nonmatching_query_width() {
    let config = Qwen3Config {
        model_type: "qwen3".into(),
        vocab_size: 17,
        hidden_size: 6,
        intermediate_size: 12,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        use_qk_norm: true,
    };
    let model = Qwen3Model::random(config, 11).unwrap();
    let logits = model.forward_logits(&[1, 2, 1]).unwrap();
    assert_eq!(logits.len(), 17);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(model.generate_greedy(&[1], 3).unwrap().len(), 4);
}
