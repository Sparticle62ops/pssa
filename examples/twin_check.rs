//! Populated-memory CPU-twin verification. `--gpu` additionally requires strict
//! hardware execution of every dense forward-stage shape (no CPU fallback).
//! This is not a claim of end-to-end hardware backward/adjoint verification.
use oxide_ai_pssa::backend::{Device, GpuDispatch, gemm_cpu_reference};
use oxide_ai_pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn tiny_cfg() -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth: 1,
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

fn populate(m: &mut PSSALayerV2) {
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = (i % 11) as f32 * 0.013 - 0.04;
    }
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = ((i % 9) as f32 - 4.0) * 0.013;
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
}

fn difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparison length mismatch");
    assert!(
        a.iter().chain(b).all(|x| x.is_finite()),
        "nonfinite comparison input"
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn strict_forward_gemms(gpu: &GpuDispatch, m: &PSSALayerV2) -> Result<(), String> {
    let l = m.cfg.chunk_len;
    let d = m.cfg.d_latent;
    for (name, x, w, rows, cols) in [
        ("delta", &m.tape.x_norm, &m.w_delta.data, d, d),
        ("memory gate", &m.tape.x_norm, &m.w_gate.data, d, d),
        ("memory projection", &m.tape.m_val, &m.w_proj.data, d, d),
        ("MLP down", &m.tape.z_raw, &m.mlp_w1.data, 2 * d, d),
        ("MLP up", &m.tape.mlp_act, &m.mlp_w2.data, d, 2 * d),
        (
            "logits",
            &m.tape.z_final,
            &m.unembed_w.data,
            m.cfg.d_vocab,
            d,
        ),
    ] {
        let actual = gpu.try_dispatch_gemm(x, w, l, rows, cols, 1)?;
        let expected = gemm_cpu_reference(x, w, l, rows, cols, 1);
        let error = difference(&actual, &expected);
        let scale = expected.iter().map(|x| x.abs()).fold(1.0, f32::max);
        if error > 2e-5 * scale {
            return Err(format!("strict {name} hardware mismatch: {error}"));
        }
        println!("  strict GPU {name:<18} max diff {error:.3e}");
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let hardware = std::env::args().skip(1).any(|arg| arg == "--gpu");
    let gpu = if hardware {
        let device = Device::try_gpu()?;
        Some(
            device
                .gpu()
                .ok_or("GPU initialization returned a CPU device")?,
        )
    } else {
        None
    };
    let cfg = tiny_cfg();
    let mut reference = PSSALayerV2::new(cfg.clone(), 12345);
    let mut staged = PSSALayerV2::new(cfg.clone(), 12345);
    populate(&mut reference);
    populate(&mut staged);
    let len = cfg.chunk_len;
    let tokens: Vec<usize> = (0..len).map(|t| (t * 7 + 3) % cfg.d_vocab).collect();
    let targets: Vec<usize> = (0..len).map(|t| (t * 11 + 5) % cfg.d_vocab).collect();

    for round in 0..2 {
        // The second chunk retains nonzero carry and accumulated gradients.
        let a = reference.forward_train_chunk(&tokens, &targets);
        let b = forward_train_chunk_batched(&mut staged, &tokens, &targets);
        assert!(a.is_finite() && b.is_finite());
        let mut error = (a - b).abs();
        for (a, b) in [
            (&reference.tape.probs, &staged.tape.probs),
            (&reference.tape.z_final, &staged.tape.z_final),
            (&reference.tape.m_inj, &staged.tape.m_inj),
            (&reference.tape.mem_weights, &staged.tape.mem_weights),
            (&reference.tape.h_states, &staged.tape.h_states),
        ] {
            error = error.max(difference(a, b));
        }
        println!("round {round} forward: loss={a:.8} max_tape_diff={error:.3e}");
        assert!(error < 1e-5, "forward twin mismatch");
        if let Some(gpu) = &gpu {
            strict_forward_gemms(gpu, &reference)?;
        }
        reference.backward_chunk(len, 1.0);
        backward_chunk_batched(&mut staged, len, 1.0);
        let mut bwd_error = 0.0f32;
        macro_rules! compare {
            ($($field:ident),*) => { $(
                bwd_error = bwd_error.max(difference(&reference.$field.grad, &staged.$field.grad));
            )* };
        }
        compare!(
            embed_w, norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj,
            mlp_w1, mlp_w2, unembed_w
        );
        bwd_error = bwd_error.max(difference(
            &reference.adapters[0].down_proj.grad,
            &staged.adapters[0].down_proj.grad,
        ));
        bwd_error = bwd_error.max(difference(
            &reference.adapters[0].up_proj.grad,
            &staged.adapters[0].up_proj.grad,
        ));
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
        println!("round {round} backward: max_grad_diff={bwd_error:.3e}");
        assert!(bwd_error < 1e-5, "backward twin mismatch");
    }
    println!("CPU TWIN CHECK PASSED: populated-memory/carry/adapter/MLP forward and backward");
    if let Some(gpu) = gpu {
        println!(
            "STRICT {} FORWARD GEMM CHECK PASSED (not a hardware backward proof)",
            gpu.backend_label()
        );
    }
    Ok(())
}
