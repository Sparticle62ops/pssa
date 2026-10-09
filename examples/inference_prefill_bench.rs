//! CPU prefill and real chat time-to-first-token, with alternating off/on pairs.
//! /workspace/bin/csrun "cargo run --profile fast --example inference_prefill_bench"
//! No training. Construction, tokenizer validation, and warmup are not timed.
use pssa::dataset::Tokenizer;
use pssa::inference::{InferenceConfig, PSSAInferenceEngine};
use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ROUNDS: usize = 7;
const VOCAB: usize = 4096;
const DECODE_TOKENS: usize = 32;

fn model(seed: u64) -> PSSALayerV2 {
    let mut model = PSSALayerV2::new(
        PSSAConfigV2 {
            depth: 1,
            d_vocab: VOCAB,
            d_latent: 64,
            d_state: 8,
            d_mem_key: 16,
            mem_capacity: 32,
            chunk_len: 32,
            ..Default::default()
        },
        seed,
    );
    // Include a full, fixed bank, not just the empty-bank fast path.
    for entry in 0..32 {
        let key: Vec<_> = (0..16)
            .map(|i| 0.00025 * (entry + 1) as f32 * (i + 1) as f32)
            .collect();
        let value: Vec<_> = (0..64)
            .map(|i| 0.003 * (entry + 1) as f32 * ((i % 11) as f32 - 5.0))
            .collect();
        model.memory.insert(&key, &value);
    }
    model
}

fn prefill(model: &mut PSSALayerV2, ids: &[usize], logits: &mut [f32], enabled: bool) -> f64 {
    model.reset_recurrent_state();
    let start = Instant::now();
    if enabled {
        assert!(model.prefill_inference(black_box(ids), Some(black_box(&mut *logits)), || false).unwrap());
    } else {
        for &id in ids {
            model.forward_inference(black_box(id), black_box(&mut *logits));
        }
    }
    black_box(&*logits);
    start.elapsed().as_secs_f64()
}

fn decode(model: &mut PSSALayerV2, logits: &mut [f32]) -> f64 {
    // Greedy autoregressive continuation; head and argmax both included.
    let start = Instant::now();
    for _ in 0..DECODE_TOKENS {
        let id = (1..VOCAB)
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        model.forward_inference(black_box(id), black_box(&mut *logits));
    }
    black_box(&*logits);
    start.elapsed().as_secs_f64()
}

fn ttft(model: &mut PSSALayerV2, tokenizer: &Tokenizer, prompt: &str, enabled: bool) -> (f64, String) {
    let mut engine = PSSAInferenceEngine::new(model, tokenizer);
    engine.set_prefill_enabled(enabled);
    let cfg = InferenceConfig { temperature: 0.0, max_new_tokens: 1, ..Default::default() };
    let start = Instant::now();
    let mut first = None;
    let out = engine.try_generate_chat_turn_controlled(
        black_box(prompt), &cfg,
        |_, count| {
            if count == 1 { first = Some(start.elapsed().as_secs_f64()); }
        },
        || false,
    ).unwrap();
    (first.expect("first generated-token callback"), out)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn main() {
    let deadline = Instant::now();
    let vocabulary: Vec<_> = std::iter::once("<unk>".to_owned())
        .chain((1..VOCAB).map(|i| format!("token{i}")))
        .collect();
    let tokenizer = Tokenizer::from_vocabulary(&vocabulary).unwrap();
    println!("{}", serde_json::json!({
        "backend": "CPU", "arch": std::env::consts::ARCH,
        "settings": {"vocab": VOCAB, "latent": 64, "state": 8, "key": 16,
                     "memory_occupied": 32, "depth": 1, "loops": 1, "tile": 32},
        "rounds": ROUNDS, "warmups_per_mode": 1,
        "ttft_includes": "tokenize/reset/allocate/prefill/greedy sampling/first callback",
        "prefill_includes": "all prompt rows and final head; reset outside timer",
        "decode_includes": "32 autoregressive greedy argmax and single-token steps"
    }));
    for seed in [11, 23, 47] {
        for len in [64, 256, 1024] {
            assert!(deadline.elapsed() < Duration::from_secs(240), "four-minute measurement limit");
            let ids: Vec<_> = (0..len).map(|t| (t * 97 + 11) % (VOCAB - 1) + 1).collect();
            let prompt = ids.iter().map(|&id| vocabulary[id].as_str()).collect::<Vec<_>>().join(" ");
            assert_eq!(tokenizer.try_encode(&prompt, true).unwrap(), ids);
            let mut models = [model(seed), model(seed)];
            let mut logits = [vec![0.0; VOCAB], vec![0.0; VOCAB]];
            for mode in 0..2 {
                prefill(&mut models[mode], &ids, &mut logits[mode], mode == 1);
                decode(&mut models[mode], &mut logits[mode]);
                ttft(&mut models[mode], &tokenizer, &prompt, mode == 1);
            }
            let mut prefill_s = [Vec::new(), Vec::new()];
            let mut ttft_s = [Vec::new(), Vec::new()];
            let mut decode_s = [Vec::new(), Vec::new()];
            for round in 0..ROUNDS {
                assert!(deadline.elapsed() < Duration::from_secs(240), "four-minute measurement limit");
                let mut texts = [String::new(), String::new()];
                for mode in [round % 2, 1 - round % 2] {
                    prefill_s[mode].push(prefill(&mut models[mode], &ids, &mut logits[mode], mode == 1));
                }
                assert!(logits[0].iter().zip(&logits[1]).all(|(a, b)| a.to_bits() == b.to_bits()));
                for mode in [round % 2, 1 - round % 2] {
                    decode_s[mode].push(decode(&mut models[mode], &mut logits[mode]));
                    let (time, out) = ttft(&mut models[mode], &tokenizer, &prompt, mode == 1);
                    ttft_s[mode].push(time);
                    texts[mode] = out;
                }
                assert_eq!(texts[0], texts[1]);
                assert!(logits[0].iter().zip(&logits[1]).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            for mode in 0..2 {
                println!("{}", serde_json::json!({
                    "seed": seed, "prompt_tokens": len, "prefill_enabled": mode == 1,
                    "prefill_tok_s": len as f64 / median(&prefill_s[mode]),
                    "prefill_ms": median(&prefill_s[mode]) * 1e3,
                    "ttft_ms": median(&ttft_s[mode]) * 1e3,
                    "decode_tok_s": DECODE_TOKENS as f64 / median(&decode_s[mode]),
                    "prefill_samples_ms": prefill_s[mode].iter().map(|s| s * 1e3).collect::<Vec<_>>(),
                    "ttft_samples_ms": ttft_s[mode].iter().map(|s| s * 1e3).collect::<Vec<_>>(),
                    "decode_samples_ms": decode_s[mode].iter().map(|s| s * 1e3).collect::<Vec<_>>(),
                }));
            }
        }
    }
}
