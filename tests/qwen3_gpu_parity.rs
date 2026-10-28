//! Cross-check for the device-path teacher forward: `forward_logits_device`
//! against the independent per-position CPU reference `forward_logits`.

use pssa::backend::Device;
use pssa::qwen3::{Qwen3Config, Qwen3Model};

fn small_config() -> Qwen3Config {
    Qwen3Config {
        model_type: "qwen3".into(),
        vocab_size: 997,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        max_position_embeddings: 40_960,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        use_qk_norm: true,
    }
}

#[test]
fn device_forward_matches_cpu_reference() {
    let config = small_config();
    let model = Qwen3Model::random(config, 77).expect("random model");
    let tokens = vec![11usize, 250, 3, 940, 42, 7];
    let n = tokens.len();
    let vocab = model.config.vocab_size;
    let mut device_logits = vec![0.0f32; n * vocab];
    let gpu = Device::try_gpu().ok().and_then(|d| d.gpu());
    model
        .forward_logits_device(gpu.as_ref(), &tokens, &mut device_logits)
        .expect("device forward");
    for position in 0..n {
        let reference = model
            .forward_logits(&tokens[..position + 1])
            .expect("cpu reference forward");
        let row = &device_logits[position * vocab..(position + 1) * vocab];
        for (i, (&device_value, &reference_value)) in row.iter().zip(&reference).enumerate() {
            let diff = (device_value - reference_value).abs();
            assert!(
                diff < 2e-3,
                "logit mismatch at position {position} vocab {i}: device {device_value} reference {reference_value} diff {diff}"
            );
        }
    }
}
