//! Cheap CPU timing probe for the extra work introduced by global clipping.
//! Run: cargo run --release --example grad_clip_probe -- 2000000 7
//! Alternates the old norm/scale/Adam passes with the fused-scale Adam path.
use pssa::{
    backend::Device,
    pssa::{PSSAConfigV2, PSSALayerV2},
};
use std::{hint::black_box, time::Instant};

fn model(parameters: usize) -> PSSALayerV2 {
    PSSALayerV2::new_with_device(
        PSSAConfigV2 {
            d_vocab: parameters / 16,
            d_latent: 8,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 1,
            chunk_len: 1,
            ..Default::default()
        },
        42,
        Device::Cpu,
    )
}

fn gradients(m: &mut PSSALayerV2, mut visit: impl FnMut(&mut [f32])) {
    visit(&mut m.embed_w.grad);
    visit(&mut m.unembed_w.grad);
    for b in std::iter::once(&mut m.block).chain(&mut m.extra_blocks) {
        visit(&mut b.norm_gamma.grad);
        visit(&mut b.norm_beta.grad);
        for p in [
            &mut b.a_mat,
            &mut b.w_delta,
            &mut b.w_b,
            &mut b.w_c,
            &mut b.w_qx,
            &mut b.w_qh,
            &mut b.w_gate,
            &mut b.w_proj,
            &mut b.mlp_w1,
            &mut b.mlp_w2,
        ] {
            visit(&mut p.grad);
        }
        visit(&mut b.adapters[0].down_proj.grad);
        visit(&mut b.adapters[0].up_proj.grad);
    }
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
    // All non-vocabulary gradients stay zero. Both paths nevertheless visit
    // exactly every Adam gradient, in the historical global-norm order.
    for iteration in 0..repeats + 2 {
        before.embed_w.grad.fill(2.0);
        before.unembed_w.grad.fill(2.0);
        after.embed_w.grad.fill(2.0);
        after.unembed_w.grad.fill(2.0);
        let mut old = || {
            let started = Instant::now();
            let mut sum = 0.0f64;
            gradients(&mut before, |grad| {
                for &g in grad.iter() {
                    sum += (g as f64) * (g as f64);
                }
            });
            let scale = 1.0 / sum.sqrt();
            gradients(&mut before, |grad| {
                for g in grad {
                    *g = (*g as f64 * scale) as f32;
                }
            });
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
        let (a, b) = if iteration % 2 == 0 {
            (old(), new())
        } else {
            let b = new();
            (old(), b)
        };
        if iteration >= 2 {
            old_ms.push(a);
            new_ms.push(b);
        }
    }
    println!(
        "CPU parameters={} warmups=2 repeats={repeats}",
        after.parameter_count()
    );
    println!("old_ms={old_ms:?}\nnew_ms={new_ms:?}");
    old_ms.sort_by(f64::total_cmp);
    new_ms.sort_by(f64::total_cmp);
    println!(
        "median before={:.3} ms/update after={:.3} ms/update",
        old_ms[repeats / 2],
        new_ms[repeats / 2]
    );
}
