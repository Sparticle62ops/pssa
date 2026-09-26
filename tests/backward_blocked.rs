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

fn primed_model() -> (PSSALayerV2, usize) {
    let mut m = PSSALayerV2::new(cfg(), 7);
    for (i, x) in m.mlp_w1.data.iter_mut().enumerate() {
        *x = ((i % 13) as f32 - 6.0) * 0.017;
    }
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = ((i % 9) as f32 - 4.0) * 0.023;
    }
    for (i, x) in m.unembed_w.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.019;
    }
    let ids: Vec<usize> = vec![1, 4, 9, 2, 6];
    let targets: Vec<usize> = vec![4, 9, 2, 6, 3];
    let seq_len = ids.len();
    gpu_batch::forward_train_chunk_batched(&mut m, &ids, &targets);
    (m, seq_len)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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
