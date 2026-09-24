//! Reproducible CPU-only timing probe for the live PSSA model paths.
//! Run with the same release environment used by the optimization report.

use oxide_ai_pssa::linalg::dot_slice;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::hint::black_box;
use std::time::{Duration, Instant};

const ROUNDS: usize = 7;
const DOT_TARGET: Duration = Duration::from_millis(35);
const TRAIN_PASSES_PER_ROUND: usize = 2;
const INFERENCE_TOKENS_PER_ROUND: usize = 32;

/// Kept exported so `cargo rustc --example perf_probe --release -- --emit=asm`
/// exposes the code generated for the actual `dot_slice` implementation.
///
/// # Safety
///
/// `a` and `b` must each point to at least `len` initialized, correctly
/// aligned `f32` values that stay valid and unaliased for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn perf_probe_dot_export(a: *const f32, b: *const f32, len: usize) -> f32 {
    // The probe supplies valid pointers. Keeping the call here prevents the dot
    // implementation from becoming an unnamed, fully inlined-only code path in
    // assembly evidence.
    unsafe {
        black_box(dot_slice(
            std::slice::from_raw_parts(a, len),
            std::slice::from_raw_parts(b, len),
        ))
    }
}

fn config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 4096,
        d_latent: 64,
        d_state: 8,
        d_mem_key: 16,
        mem_capacity: 32,
        chunk_len: 32,
        // The timed training passes are deliberately few and use a small LR so
        // their parameters stay finite without timing a divergent trajectory.
        lr: 1e-5,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.1,
    }
}

fn make_model() -> PSSALayerV2 {
    let mut model = PSSALayerV2::new(config(), 0x5eed_1234);
    let mut key = vec![0.0; model.cfg.d_mem_key];
    let mut value = vec![0.0; model.cfg.d_latent];
    for entry in 0..model.cfg.mem_capacity {
        for (i, x) in key.iter_mut().enumerate() {
            *x = 0.00025 * (entry as f32 + 1.0) * (i as f32 + 1.0);
        }
        for (i, x) in value.iter_mut().enumerate() {
            *x = 0.003 * (entry as f32 + 1.0) * ((i % 11) as f32 - 5.0);
        }
        // The deterministic key norm is safely inside the open Poincare ball.
        model.memory.insert(&key, &value);
    }
    model
}

fn token_fixture() -> (Vec<usize>, Vec<usize>) {
    let mut tokens = Vec::with_capacity(32);
    let mut targets = Vec::with_capacity(32);
    for i in 0..32 {
        tokens.push((i * 97 + 11) % 4096);
        targets.push((i * 193 + 29) % 4096);
    }
    (tokens, targets)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn spread_percent(values: &[f64], median: f64) -> f64 {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for &value in values {
        low = low.min(value);
        high = high.max(value);
    }
    if median == 0.0 {
        0.0
    } else {
        100.0 * (high - low) / median
    }
}

fn dot_samples(len: usize) -> Vec<f64> {
    let a: Vec<f32> = (0..len)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.013)
        .collect();
    let b: Vec<f32> = (0..len)
        .map(|i| ((i * 29 % 89) as f32 - 44.0) * 0.017)
        .collect();
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let start = Instant::now();
        let mut calls = 0usize;
        let mut sum = 0.0f32;
        while start.elapsed() < DOT_TARGET {
            sum += black_box(dot_slice(black_box(&a), black_box(&b)));
            calls += 1;
        }
        black_box(sum);
        samples.push(start.elapsed().as_secs_f64() * 1e9 / calls as f64);
    }
    samples
}

fn train_samples(tokens: &[usize], targets: &[usize]) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        // Each independent round starts from identical seeded weights and memory.
        let mut model = make_model();
        let start = Instant::now();
        let mut loss = 0.0f32;
        for _ in 0..TRAIN_PASSES_PER_ROUND {
            model.reset_recurrent_state();
            loss += model.forward_train_chunk(black_box(tokens), black_box(targets));
            model.zero_gradients();
            model.backward_chunk(32, 1.0);
            model.apply_adamw(model.cfg.lr);
        }
        black_box(loss);
        assert!(model.embed_w.data.iter().all(|x| x.is_finite()));
        samples.push(start.elapsed().as_secs_f64() * 1e3 / TRAIN_PASSES_PER_ROUND as f64);
    }
    samples
}

fn inference_samples(tokens: &[usize]) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        // As above, each round uses the same initialized model and fixed bank.
        let mut model = make_model();
        let mut logits = vec![0.0; model.cfg.d_vocab];
        model.reset_recurrent_state();
        let start = Instant::now();
        for i in 0..INFERENCE_TOKENS_PER_ROUND {
            model.forward_inference(black_box(tokens[i]), black_box(&mut logits));
            black_box(logits[0]);
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push(INFERENCE_TOKENS_PER_ROUND as f64 / elapsed);
    }
    samples
}

fn print_series(name: &str, unit: &str, mut samples: Vec<f64>, trailing_comma: bool) {
    let raw = samples.clone();
    let med = median(&mut samples);
    let comma = if trailing_comma { "," } else { "" };
    println!(
        "\"{name}\":{{\"unit\":\"{unit}\",\"rounds\":{ROUNDS},\"samples\":[{}],\"median\":{med:.6},\"spread_pct\":{:.6}}}{comma}",
        raw.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(","),
        spread_percent(&raw, med),
    );
}

fn main() {
    let (tokens, targets) = token_fixture();

    // Warmups are deliberately outside all timed samples.
    let mut warm_model = make_model();
    let mut warm_logits = vec![0.0; warm_model.cfg.d_vocab];
    warm_model.forward_inference(tokens[0], &mut warm_logits);
    warm_model.reset_recurrent_state();
    warm_model.forward_train_chunk(&tokens, &targets);
    warm_model.zero_gradients();
    warm_model.backward_chunk(32, 1.0);
    warm_model.apply_adamw(warm_model.cfg.lr);
    black_box(dot_slice(&[1.0; 32], &[2.0; 32]));

    // JSON-lines keeps baseline and after artifacts diffable without dependencies.
    println!("{{");
    print_series("dot_32_ns", "ns/call", dot_samples(32), true);
    print_series("dot_64_ns", "ns/call", dot_samples(64), true);
    print_series("dot_256_ns", "ns/call", dot_samples(256), true);
    print_series("dot_1024_ns", "ns/call", dot_samples(1024), true);
    print_series(
        "train_chunk_ms",
        "ms/forward_backward_adam_chunk",
        train_samples(&tokens, &targets),
        true,
    );
    print_series(
        "inference_tokens_per_s",
        "tokens/s",
        inference_samples(&tokens),
        false,
    );
    println!("}}");
}
