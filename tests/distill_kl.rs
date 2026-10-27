//! Finite-difference regressions for the distillation loss: CE + temperature
//! KL against precomputed teacher probabilities, verified through the public
//! batched forward/backward entry points on the CPU staged path.

use pssa::{
    gpu_batch::{backward_chunk_batched, forward_train_chunk_batched},
    pssa::{PSSAConfigV2, PSSALayerV2},
};

const SEQ: usize = 4;
const TOKENS: [usize; SEQ] = [1, 5, 9, 13];
const TARGETS: [usize; SEQ] = [2, 7, 11, 3];
const TEMP: f32 = 2.0;
const EPS: f32 = 5e-2;
const FD_TOL: f32 = 5e-2;

fn model(depth: usize) -> PSSALayerV2 {
    let cfg = PSSAConfigV2 {
        depth,
        d_vocab: 41,
        d_latent: 16,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 5,
        chunk_len: 8,
        tau_mem: 0.7,
        ..Default::default()
    };
    let mut m = PSSALayerV2::new(cfg, 12345);
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = ((i % 11) as f32 - 5.0) * 0.037;
    }
    for x in &mut m.w_b.data {
        *x *= 0.2;
    }
    for x in &mut m.w_c.data {
        *x *= 0.2;
    }
    m
}

/// Smooth deterministic teacher distribution peaked near each target.
fn teacher_probs(v: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; SEQ * v];
    for (s, &tgt) in TARGETS.iter().enumerate() {
        let off = s * v;
        for i in 0..v {
            t[off + i] = 1.0 / (1.0 + (i as f32 - tgt as f32).abs() * 0.5);
        }
        let sum: f32 = t[off..off + v].iter().sum();
        for x in t[off..off + v].iter_mut() {
            *x /= sum;
        }
    }
    t
}

/// Analytic d(mean loss)/d(unembed weight) at the probe indices, from one
/// forward/backward pass, and the central-difference estimate from two more.
/// Returns (analytic, numeric) pairs.
fn fd_pairs(depth: usize, temp: f32, ce_mix: f32, teacher: Option<&[f32]>) -> Vec<(usize, f32, f32)> {
    // Analytic run.
    let mut m = model(depth);
    if let Some(t) = teacher {
        m.set_distill_targets(t, temp, ce_mix, SEQ);
    }
    let _ = forward_train_chunk_batched(&mut m, &TOKENS, &TARGETS);
    backward_chunk_batched(&mut m, SEQ, 1.0);
    let grad: Vec<f32> = m.unembed_w.grad.clone();
    let v = m.cfg.d_vocab;
    let d = m.cfg.d_latent;

    let probes: Vec<usize> = [0usize, 7, 13 * d + 5, (v - 1) * d + (d - 1)].to_vec();
    probes
        .iter()
        .map(|&k| {
            let loss_at = |delta: f32| {
                let mut mm = model(depth);
                if let Some(t) = teacher {
                    mm.set_distill_targets(t, temp, ce_mix, SEQ);
                }
                mm.unembed_w.data[k] += delta;
                forward_train_chunk_batched(&mut mm, &TOKENS, &TARGETS)
            };
            let numeric = (loss_at(EPS) - loss_at(-EPS)) / (2.0 * EPS);
            (k, grad[k], numeric)
        })
        .collect()
}

fn assert_grad_matches(depth: usize, temp: f32, ce_mix: f32, teacher: Option<&[f32]>, label: &str) {
    for (k, analytic, numeric) in fd_pairs(depth, temp, ce_mix, teacher) {
        let diff = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-6);
        let rel = diff / denom;
        // f32 finite differences bottom out on tiny gradients: fall back to
        // an absolute noise floor when the probe is too small to resolve.
        assert!(
            rel < FD_TOL || diff < 1e-5,
            "{label}: unembed grad mismatch at weight {k}: analytic {analytic} numeric {numeric} rel {rel}"
        );
    }
}

#[test]
fn ce_only_regression_unchanged() {
    // Distillation off: the refactored path must reproduce plain CE gradients.
    assert_grad_matches(1, 0.0, 1.0, None, "ce-only depth 1");
    assert_grad_matches(3, 0.0, 1.0, None, "ce-only stacked");
}

#[test]
fn distill_mixed_gradient_matches_finite_difference() {
    let v = model(1).cfg.d_vocab;
    let teacher = teacher_probs(v);
    assert_grad_matches(1, TEMP, 0.5, Some(&teacher), "distill mixed depth 1");
    assert_grad_matches(3, TEMP, 0.5, Some(&teacher), "distill mixed stacked");
}

#[test]
fn self_distill_kl_is_zero_with_zero_gradient() {
    // Teacher = the student's own temperature softmax: KL collapses to zero
    // and the whole logits gradient with it.
    let depth = 1;
    let mut m = model(depth);
    let _ = forward_train_chunk_batched(&mut m, &TOKENS, &TARGETS);
    let v = m.cfg.d_vocab;
    let logits: Vec<f32> = m.block.tape.logits[..SEQ * v].to_vec();
    let mut teacher = vec![0.0f32; SEQ * v];
    for t in 0..SEQ {
        let off = t * v;
        let mut max_s = f32::NEG_INFINITY;
        for i in 0..v {
            let s = logits[off + i] / TEMP;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..v {
            sum += (logits[off + i] / TEMP - max_s).exp();
        }
        for i in 0..v {
            teacher[off + i] = (logits[off + i] / TEMP - max_s).exp() / sum;
        }
    }
    m.set_distill_targets(&teacher, TEMP, 0.0, SEQ);
    let loss = forward_train_chunk_batched(&mut m, &TOKENS, &TARGETS);
    assert!(loss.abs() < 1e-4, "self-distill KL should vanish, got {loss}");
    backward_chunk_batched(&mut m, SEQ, 1.0);
    let worst = m.unembed_w
        .grad
        .iter()
        .fold(0.0f32, |a: f32, b: &f32| a.max(*b));
    assert!(worst < 1e-4, "self-distill gradient should vanish, got {worst}");
}
