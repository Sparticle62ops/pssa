//! Cheap CPU timing probe for the extra work introduced by global clipping.
//! Run: cargo run --release --example grad_clip_probe -- 2000000 7
//! Alternates the old norm/scale/Adam passes with the fused-scale Adam path.
use pssa::{backend::Device, pssa::{PSSAConfigV2, PSSALayerV2}};
use std::{hint::black_box, time::Instant};

fn model(parameters: usize) -> PSSALayerV2 {
    PSSALayerV2::new_with_device(PSSAConfigV2 {
        d_vocab: parameters / 16,
        d_latent: 8,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 1,
        chunk_len: 1,
        ..Default::default()
    }, 42, Device::Cpu)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let parameters = args.get(1).map_or(2_000_000, |s| s.parse().unwrap());
    let repeats = args.get(2).map_or(7, |s| s.parse().unwrap());
    assert!(parameters >= 16 && repeats > 0);
    let mut before = model(parameters);
    let mut after = model(parameters);
    let mut old_ms = Vec::new();
    let mut new_ms = Vec::new();
    // All non-vocabulary gradients stay zero. The norm still covers every Adam
    // parameter, as in production, but we can express the historical reference
    // here without exposing an optimizer-internal visitor in the public API.
    for iteration in 0..repeats + 2 {
        before.embed_w.grad.fill(2.0);
        before.unembed_w.grad.fill(2.0);
        after.embed_w.grad.fill(2.0);
        after.unembed_w.grad.fill(2.0);
        let mut old = || {
            let started = Instant::now();
            let norm = before.embed_w.grad.iter().chain(&before.unembed_w.grad)
                .fold(0.0f64, |sum, &g| sum + (g as f64) * (g as f64)).sqrt();
            let scale = 1.0 / norm;
            for g in before.embed_w.grad.iter_mut().chain(&mut before.unembed_w.grad) {
                *g = (*g as f64 * scale) as f32;
            }
            before.apply_adamw(1e-3);
            black_box(&before);
            started.elapsed().as_secs_f64() * 1000.0
        };
        let mut new = || {
            let started = Instant::now();
            black_box(after.apply_adamw_with_grad_clip(1e-3, 1.0));
            black_box(&after);
            started.elapsed().as_secs_f64() * 1000.0
        };
        let (a, b) = if iteration % 2 == 0 { (old(), new()) } else {
            let b = new();
            (old(), b)
        };
        if iteration >= 2 { old_ms.push(a); new_ms.push(b); }
    }
    println!("CPU parameters={} warmups=2 repeats={repeats}", after.parameter_count());
    println!("old_ms={old_ms:?}\nnew_ms={new_ms:?}");
    old_ms.sort_by(f64::total_cmp);
    new_ms.sort_by(f64::total_cmp);
    println!("median before={:.3} ms/update after={:.3} ms/update", old_ms[repeats / 2], new_ms[repeats / 2]);
}
