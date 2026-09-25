//! CPU-twin verification: the batched stage dispatch path must reproduce the
//! reference forward/backward token-loop math to f32 roundoff.
use oxide_ai_pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn tiny_cfg() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 50,
        d_latent: 32,
        d_state: 8,
        d_mem_key: 8,
        mem_capacity: 16,
        chunk_len: 12,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.1,
    }
}

fn main() {
    let cfg = tiny_cfg();
    let mut ref_model = PSSALayerV2::new(cfg.clone(), 12345);
    let mut twin_model = PSSALayerV2::new(cfg.clone(), 12345);

    let seq_len = 12usize;
    let token_ids: Vec<usize> = (0..seq_len).map(|t| (t * 7 + 3) % cfg.d_vocab).collect();
    let target_ids: Vec<usize> = (0..seq_len).map(|t| (t * 11 + 5) % cfg.d_vocab).collect();

    // ---- Forward twin check ----
    let ref_loss = ref_model.forward_train_chunk(&token_ids, &target_ids);
    let twin_loss = forward_train_chunk_batched(&mut twin_model, &token_ids, &target_ids);

    let mut max_diff: f32 = (ref_loss - twin_loss).abs();
    for i in 0..ref_model.tape.probs.len() {
        max_diff = max_diff.max((ref_model.tape.probs[i] - twin_model.tape.probs[i]).abs());
    }
    for i in 0..ref_model.tape.z_final.len() {
        max_diff = max_diff.max((ref_model.tape.z_final[i] - twin_model.tape.z_final[i]).abs());
    }
    for i in 0..ref_model.tape.m_inj.len() {
        max_diff = max_diff.max((ref_model.tape.m_inj[i] - twin_model.tape.m_inj[i]).abs());
    }
    for i in 0..ref_model.tape.h_states.len() {
        max_diff = max_diff.max((ref_model.tape.h_states[i] - twin_model.tape.h_states[i]).abs());
    }
    println!("forward: ref_loss={:.8} twin_loss={:.8} max_tape_diff={:.3e}", ref_loss, twin_loss, max_diff);
    assert!(max_diff < 1e-5, "forward twin mismatch: {}", max_diff);

    // ---- Backward twin check ----
    let scale = 1.0f32;
    ref_model.backward_chunk(seq_len, scale);
    backward_chunk_batched(&mut twin_model, seq_len, scale);

    let mut bwd_diff: f32 = 0.0;
    macro_rules! cmp {
        ($field:ident, $name:expr) => {{
            let r = &ref_model.$field.grad;
            let t = &twin_model.$field.grad;
            for i in 0..r.len() {
                bwd_diff = bwd_diff.max((r[i] - t[i]).abs());
            }
            println!("  grad {:<14} max diff {:.3e}", $name, bwd_diff);
        }};
    }
    cmp!(embed_w, "embed_w");
    cmp!(norm_gamma, "norm_gamma");
    cmp!(norm_beta, "norm_beta");
    cmp!(a_mat, "a_mat");
    cmp!(w_delta, "w_delta");
    cmp!(w_b, "w_b");
    cmp!(w_c, "w_c");
    cmp!(w_qx, "w_qx");
    cmp!(w_qh, "w_qh");
    cmp!(w_gate, "w_gate");
    cmp!(w_proj, "w_proj");
    cmp!(mlp_w1, "mlp_w1");
    cmp!(mlp_w2, "mlp_w2");
    cmp!(unembed_w, "unembed_w");
    {
        let r = &ref_model.adapters[0];
        let t = &twin_model.adapters[0];
        for i in 0..r.down_proj.grad.len() {
            bwd_diff = bwd_diff.max((r.down_proj.grad[i] - t.down_proj.grad[i]).abs());
        }
        for i in 0..r.up_proj.grad.len() {
            bwd_diff = bwd_diff.max((r.up_proj.grad[i] - t.up_proj.grad[i]).abs());
        }
    }

    println!("backward: max_grad_diff={:.3e}", bwd_diff);
    assert!(bwd_diff < 1e-5, "backward twin mismatch: {}", bwd_diff);

    println!("TWIN CHECK PASSED: batched dispatch matches reference math to f32 roundoff");
}
