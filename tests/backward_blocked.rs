//! Backward-pass restructure twins: the blocked GEMM formulation of the logits
//! and MLP backward stages must reproduce the fused per-token loops exactly
//! enough for training, run with `gpu = None` so the CPU twins are exercised.

use oxide_ai_pssa::gpu_batch;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn cfg() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 11,
        d_latent: 8,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 4,
        chunk_len: 6,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.25,
    }
}

fn populate(m: &mut PSSALayerV2) {
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = (i % 11) as f32 * 0.013 - 0.04;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.019;
    }
    for (i, x) in m.adapters[0].consolidated_up.iter_mut().enumerate() {
        *x = ((i % 5) as f32 - 2.0) * 0.011;
    }
    for entry in 0..m.cfg.mem_capacity {
        let key: Vec<f32> = (0..m.cfg.d_mem_key)
            .map(|i| ((entry + 2 * i) % 7) as f32 * 0.04 - 0.1)
            .collect();
        let value: Vec<f32> = (0..m.cfg.d_latent)
            .map(|i| ((3 * entry + i) % 11) as f32 * 0.08 - 0.4)
            .collect();
        m.memory.insert(&key, &value);
    }
    for (i, x) in m.mlp_w1.data.iter_mut().enumerate() {
        *x = ((i % 13) as f32 - 6.0) * 0.017;
    }
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = ((i % 9) as f32 - 4.0) * 0.023;
    }
    for (i, x) in m.unembed_w.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.019;
    }
}

fn primed_model() -> (PSSALayerV2, usize) {
    let mut m = PSSALayerV2::new(cfg(), 7);
    populate(&mut m);
    let ids: Vec<usize> = vec![1, 4, 9, 2, 6];
    let targets: Vec<usize> = vec![4, 9, 2, 6, 3];
    let seq_len = ids.len();
    gpu_batch::forward_train_chunk_batched(&mut m, &ids, &targets);
    (m, seq_len)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|x| x.is_finite()),
        "nonfinite comparison input"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_model_grad_diff(a: &PSSALayerV2, b: &PSSALayerV2) -> f32 {
    let mut diff = 0.0f32;
    macro_rules! compare {
        ($field:ident) => {
            diff = diff.max(max_abs_diff(&a.$field.grad, &b.$field.grad));
        };
    }
    compare!(embed_w);
    compare!(norm_gamma);
    compare!(norm_beta);
    compare!(a_mat);
    compare!(w_delta);
    compare!(w_b);
    compare!(w_c);
    compare!(w_qx);
    compare!(w_qh);
    compare!(w_gate);
    compare!(w_proj);
    compare!(mlp_w1);
    compare!(mlp_w2);
    compare!(unembed_w);
    diff = diff.max(max_abs_diff(
        &a.adapters[0].down_proj.grad,
        &b.adapters[0].down_proj.grad,
    ));
    diff = diff.max(max_abs_diff(
        &a.adapters[0].up_proj.grad,
        &b.adapters[0].up_proj.grad,
    ));
    diff
}

#[test]
fn blocked_logits_backward_matches_scalar_twin() {
    let (mut scalar, seq_len) = primed_model();
    let (mut blocked, _) = primed_model();

    gpu_batch::bwd_stage_logits_scalar(&mut scalar, seq_len, 0.5);
    gpu_batch::bwd_stage_logits_blocked(&mut blocked, seq_len, 0.5, None);

    assert!(
        max_abs_diff(&scalar.bwd_g_zfinal, &blocked.bwd_g_zfinal) < 1e-5,
        "grad_z_final diverged"
    );
    assert!(
        max_abs_diff(&scalar.unembed_w.grad, &blocked.unembed_w.grad) < 1e-5,
        "unembed weight grad diverged"
    );
}

#[test]
fn full_batched_forward_and_backward_match_reference() {
    let cfg = cfg();
    let ids = [1, 4, 9, 2, 6];
    let targets = [4, 9, 2, 6, 3];
    let mut reference = PSSALayerV2::new(cfg.clone(), 7);
    let mut batched = PSSALayerV2::new(cfg, 7);
    populate(&mut reference);
    populate(&mut batched);
    // Retain carry and accumulated gradients over a full and a partial chunk.
    for len in [ids.len(), 3] {
        let reference_loss = reference.forward_train_chunk(&ids[..len], &targets[..len]);
        let batched_loss =
            gpu_batch::forward_train_chunk_batched(&mut batched, &ids[..len], &targets[..len]);
        assert!((reference_loss - batched_loss).abs() < 1e-5);
        assert!(max_abs_diff(&reference.h_persistent, &batched.h_persistent) < 1e-5);
        assert!(max_abs_diff(&reference.tape.mem_weights, &batched.tape.mem_weights) < 1e-5);
        reference.backward_chunk(len, 0.7);
        gpu_batch::backward_chunk_batched(&mut batched, len, 0.7);
        assert!(max_model_grad_diff(&reference, &batched) < 1e-5);
    }
    for grad in [
        &reference.w_qx.grad,
        &reference.w_qh.grad,
        &reference.adapters[0].down_proj.grad,
        &reference.mlp_w1.grad,
    ] {
        assert!(
            grad.iter().any(|x| x.abs() > 1e-9),
            "vacuous branch gradient"
        );
    }
}

#[test]
fn coarse_parallel_backward_matches_scalar_with_accumulated_gradients() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let mut config = cfg();
        config.d_latent = 96;
        config.d_vocab = 257;
        config.chunk_len = 64;
        let mut scalar = PSSALayerV2::new(config.clone(), 7);
        let mut parallel = PSSALayerV2::new(config, 7);
        populate(&mut scalar);
        populate(&mut parallel);
        let ids: Vec<usize> = (0..64).map(|t| t * 3 % 257).collect();
        let targets: Vec<usize> = (0..64).map(|t| (t * 3 + 1) % 257).collect();
        gpu_batch::forward_train_chunk_batched(&mut scalar, &ids, &targets);
        gpu_batch::forward_train_chunk_batched(&mut parallel, &ids, &targets);
        for _ in 0..2 {
            gpu_batch::bwd_stage_logits_scalar(&mut scalar, 64, 1.0 / 64.0);
            gpu_batch::bwd_stage_logits(&mut parallel, 64, 1.0 / 64.0);
            gpu_batch::bwd_stage_mlp_scalar(&mut scalar, 64);
            gpu_batch::bwd_stage_mlp(&mut parallel, 64);
            assert!(max_model_grad_diff(&scalar, &parallel) < 1e-5);
            assert!(max_abs_diff(&scalar.bwd_g_zraw, &parallel.bwd_g_zraw) < 1e-5);
        }
    });
}

#[test]
fn saturated_query_and_tiny_temperature_match_reference_without_nans() {
    for tau in [0.7, 1e-40] {
        let mut config = cfg();
        config.tau_mem = tau;
        let mut reference = PSSALayerV2::new(config.clone(), 7);
        let mut staged = PSSALayerV2::new(config, 7);
        for m in [&mut reference, &mut staged] {
            populate(m);
            for weight in &mut m.w_qx.data {
                *weight *= 1e8;
            }
        }
        let ids = [1, 4, 9, 2, 6];
        let targets = [4, 9, 2, 6, 3];
        reference.forward_train_chunk(&ids, &targets);
        gpu_batch::forward_train_chunk_batched(&mut staged, &ids, &targets);
        reference.backward_chunk(ids.len(), 1.0);
        gpu_batch::backward_chunk_batched(&mut staged, ids.len(), 1.0);
        assert!(max_model_grad_diff(&reference, &staged) < 1e-5);
    }
}

/// Opt-in timing probe, not a CI speed assertion. Run in release mode without
/// concurrent trainers/builds; this measures dense backward stages, not a run.
#[test]
#[ignore = "manual release-mode dense backward benchmark"]
fn dense_backward_timing_probe() {
    use std::time::Instant;
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut config = cfg();
            config.d_vocab = 2048;
            config.d_latent = 256;
            config.chunk_len = 64;
            let mut m = PSSALayerV2::new(config, 7);
            populate(&mut m);
            let ids: Vec<usize> = (0..64).map(|i| i * 7).collect();
            let targets: Vec<usize> = (0..64).map(|i| i * 7 + 1).collect();
            gpu_batch::forward_train_chunk_batched(&mut m, &ids, &targets);
            let mut samples = [Vec::new(), Vec::new()];
            for sample in 0..7 {
                for mode in 0..2 {
                    m.zero_gradients();
                    let start = Instant::now();
                    if mode == 0 {
                        gpu_batch::bwd_stage_logits_scalar(&mut m, 64, 1.0 / 64.0);
                        gpu_batch::bwd_stage_mlp_scalar(&mut m, 64);
                    } else {
                        gpu_batch::bwd_stage_logits_blocked(&mut m, 64, 1.0 / 64.0, None);
                        gpu_batch::bwd_stage_mlp_blocked(&mut m, 64, None);
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    std::hint::black_box(&m);
                    if sample > 0 { samples[mode].push(elapsed); }
                }
            }
            for values in &mut samples { values.sort_by(f64::total_cmp); }
            let scalar = samples[0][3];
            let blocked = samples[1][3];
            println!("dense backward L64 D256 V2048 threads={threads}: scalar={scalar:.3}ms blocked={blocked:.3}ms ratio={:.3}x", scalar / blocked);
        });
    }
}

#[test]
fn blocked_mlp_backward_matches_scalar_twin() {
    let (mut scalar, seq_len) = primed_model();
    let (mut blocked, _) = primed_model();

    gpu_batch::bwd_stage_logits_scalar(&mut scalar, seq_len, 0.5);
    gpu_batch::bwd_stage_logits_scalar(&mut blocked, seq_len, 0.5);
    gpu_batch::bwd_stage_mlp_scalar(&mut scalar, seq_len);
    gpu_batch::bwd_stage_mlp_blocked(&mut blocked, seq_len, None);

    assert!(
        max_abs_diff(&scalar.bwd_g_zraw, &blocked.bwd_g_zraw) < 1e-5,
        "grad_z_raw diverged"
    );
    assert!(
        max_abs_diff(&scalar.mlp_w1.grad, &blocked.mlp_w1.grad) < 1e-5,
        "mlp_w1 grad diverged"
    );
    assert!(
        max_abs_diff(&scalar.mlp_w2.grad, &blocked.mlp_w2.grad) < 1e-5,
        "mlp_w2 grad diverged"
    );
}
