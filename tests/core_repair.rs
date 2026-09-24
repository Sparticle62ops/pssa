use oxide_ai_pssa::memory::HyperbolicEpisodicBankV2;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn cfg(latent: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 7,
        d_latent: latent,
        d_state: 2,
        d_mem_key: 3,
        mem_capacity: 3,
        chunk_len: 3,
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.25,
    }
}

fn model(latent: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(cfg(latent), 19);
    // Avoid zero-gradient initialization paths in the finite-difference fixture.
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = ((i % 11) as f32 - 5.0) * 0.031;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.019;
    }
    for (i, x) in m.adapters[0].consolidated_up.iter_mut().enumerate() {
        *x = ((i % 5) as f32 - 2.0) * 0.011;
    }
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = (i as f32 + 1.0) * 0.017;
    }
    let mut key = vec![0.0; 3];
    let mut val = vec![0.0; latent];
    for n in 0..3 {
        for k in 0..3 {
            key[k] = 0.04 * (n as f32 + 1.0) * (k as f32 + 1.0);
        }
        for (j, v) in val.iter_mut().enumerate() {
            *v = 0.03 * (n as f32 + 1.0) * (j as f32 + 1.0);
        }
        m.memory.insert(&key, &val);
    }
    m
}

#[derive(Copy, Clone, Debug)]
enum F {
    Embed,
    Gamma,
    Beta,
    A,
    Delta,
    B,
    C,
    Qx,
    Qh,
    Gate,
    Proj,
    Down,
    Up,
    Mlp1,
    Mlp2,
    Unembed,
}
fn val(m: &PSSALayerV2, f: F) -> f32 {
    match f {
        F::Embed => m.embed_w.data[0],
        F::Gamma => m.norm_gamma.data[0],
        F::Beta => m.norm_beta.data[0],
        F::A => m.a_mat.data[0],
        F::Delta => m.w_delta.data[0],
        F::B => m.w_b.data[0],
        F::C => m.w_c.data[0],
        F::Qx => m.w_qx.data[0],
        F::Qh => m.w_qh.data[0],
        F::Gate => m.w_gate.data[0],
        F::Proj => m.w_proj.data[0],
        F::Down => m.adapters[0].down_proj.data[0],
        F::Up => m.adapters[0].up_proj.data[0],
        F::Mlp1 => m.mlp_w1.data[0],
        F::Mlp2 => m.mlp_w2.data[0],
        F::Unembed => m.unembed_w.data[0],
    }
}
fn put(m: &mut PSSALayerV2, f: F, x: f32) {
    match f {
        F::Embed => m.embed_w.data[0] = x,
        F::Gamma => m.norm_gamma.data[0] = x,
        F::Beta => m.norm_beta.data[0] = x,
        F::A => m.a_mat.data[0] = x,
        F::Delta => m.w_delta.data[0] = x,
        F::B => m.w_b.data[0] = x,
        F::C => m.w_c.data[0] = x,
        F::Qx => m.w_qx.data[0] = x,
        F::Qh => m.w_qh.data[0] = x,
        F::Gate => m.w_gate.data[0] = x,
        F::Proj => m.w_proj.data[0] = x,
        F::Down => m.adapters[0].down_proj.data[0] = x,
        F::Up => m.adapters[0].up_proj.data[0] = x,
        F::Mlp1 => m.mlp_w1.data[0] = x,
        F::Mlp2 => m.mlp_w2.data[0] = x,
        F::Unembed => m.unembed_w.data[0] = x,
    }
}
fn grad(m: &PSSALayerV2, f: F) -> f32 {
    match f {
        F::Embed => m.embed_w.grad[0],
        F::Gamma => m.norm_gamma.grad[0],
        F::Beta => m.norm_beta.grad[0],
        F::A => m.a_mat.grad[0],
        F::Delta => m.w_delta.grad[0],
        F::B => m.w_b.grad[0],
        F::C => m.w_c.grad[0],
        F::Qx => m.w_qx.grad[0],
        F::Qh => m.w_qh.grad[0],
        F::Gate => m.w_gate.grad[0],
        F::Proj => m.w_proj.grad[0],
        F::Down => m.adapters[0].down_proj.grad[0],
        F::Up => m.adapters[0].up_proj.grad[0],
        F::Mlp1 => m.mlp_w1.grad[0],
        F::Mlp2 => m.mlp_w2.grad[0],
        F::Unembed => m.unembed_w.grad[0],
    }
}
fn loss(m: &mut PSSALayerV2, state: &[f32]) -> f32 {
    m.h_persistent.copy_from_slice(state);
    m.forward_train_chunk(&[0, 1, 2], &[1, 2, 3])
}

#[test]
fn central_differences_cover_every_trainable_family() {
    let mut m = model(5);
    let initial = m.h_persistent.clone();
    loss(&mut m, &initial);
    m.zero_gradients();
    m.backward_chunk(3, 1.0);
    let families = [
        F::Embed,
        F::Gamma,
        F::Beta,
        F::A,
        F::Delta,
        F::B,
        F::C,
        F::Qx,
        F::Qh,
        F::Gate,
        F::Proj,
        F::Down,
        F::Up,
        F::Mlp1,
        F::Mlp2,
        F::Unembed,
    ];
    let h = 1e-3;
    let mut max_err = 0.0f32;
    let mut max_ratio = 0.0f32;
    for f in families {
        let x = val(&m, f);
        put(&mut m, f, x + h);
        let plus = loss(&mut m, &initial);
        put(&mut m, f, x - h);
        let minus = loss(&mut m, &initial);
        put(&mut m, f, x);
        let numeric = (plus - minus) / (2.0 * h);
        let analytic = grad(&m, f);
        let err = (numeric - analytic).abs();
        let limit = 3e-3 + 0.18 * numeric.abs().max(analytic.abs());
        max_err = max_err.max(err);
        max_ratio = max_ratio.max(err / limit);
        assert!(
            err <= limit,
            "{f:?}: analytic={analytic} numeric={numeric} err={err} limit={limit}"
        );
    }
    println!("finite-difference max_abs_error={max_err:.8} max_limit_fraction={max_ratio:.6}");
}

#[test]
fn latent_five_tail_and_full_softmax_are_safe() {
    let mut m = model(5);
    let st = m.h_persistent.clone();
    loss(&mut m, &st);
    m.zero_gradients();
    m.backward_chunk(3, 1.0);
    assert!(m.unembed_w.grad.iter().all(|x| x.is_finite()));
    // An extreme target logit is still exact CE, rather than the former ~27.63 cap.
    m.unembed_w.data.fill(0.0);
    m.unembed_w.data[0] = 300.0;
    let z_state = m.h_persistent.clone();
    let l = loss(&mut m, &z_state);
    assert!(l.is_finite() && l > 100.0, "loss={l}");
}

#[test]
fn consolidation_preserves_logits_and_effective_sum() {
    let mut m = model(5);
    let mut before = vec![0.0; 7];
    let st = m.h_persistent.clone();
    m.h_persistent.copy_from_slice(&st);
    m.forward_inference(0, &mut before);
    let sum_before: f32 = m.adapters[0]
        .up_proj
        .data
        .iter()
        .zip(&m.adapters[0].consolidated_up)
        .map(|(a, b)| a + b)
        .sum();
    m.adapters[0].consolidate(0.0);
    m.adapters[0].consolidate(0.01);
    m.adapters[0].consolidate(1.0);
    let sum_after: f32 = m.adapters[0]
        .up_proj
        .data
        .iter()
        .zip(&m.adapters[0].consolidated_up)
        .map(|(a, b)| a + b)
        .sum();
    let mut after = vec![0.0; 7];
    m.h_persistent.copy_from_slice(&st);
    m.forward_inference(0, &mut after);
    assert!((sum_before - sum_after).abs() < 1e-6);
    for (a, b) in before.iter().zip(after) {
        assert!((a - b).abs() < 2e-6, "{a} {b}");
    }
}

#[test]
fn empty_zero_and_coincident_memory_are_finite() {
    let bank = HyperbolicEpisodicBankV2::new(2, 3, 2);
    let mut out = [9.0; 2];
    let mut w = [9.0; 2];
    bank.retrieve_soft_into(&[0.0; 3], 1.0, &mut out, &mut w);
    assert_eq!(out, [0.0; 2]);
    let mut bank = HyperbolicEpisodicBankV2::new(2, 3, 2);
    bank.insert(&[0.0; 3], &[1.0, 2.0]);
    bank.retrieve_soft_into(&[0.0; 3], 1.0, &mut out, &mut w);
    assert!(out.iter().chain(w.iter()).all(|x| x.is_finite()));
}

#[test]
fn inference_matches_training_tape_and_dense_embedding_adam() {
    let mut m = model(5);
    let state = m.h_persistent.clone();
    let mut logits = vec![0.0; 7];
    m.h_persistent.copy_from_slice(&state);
    m.forward_inference(0, &mut logits);
    m.h_persistent.copy_from_slice(&state);
    m.forward_train_chunk(&[0], &[1]);
    for (a, b) in logits.iter().zip(&m.tape.logits[..7]) {
        assert!((a - b).abs() < 2e-6);
    }
    m.zero_gradients();
    m.backward_chunk(1, 1.0);
    let untouched = 6 * 5;
    let prior = m.embed_w.data[untouched];
    m.apply_adamw(0.01);
    assert_ne!(m.embed_w.data[untouched], prior);
}

#[test]
fn tiny_non_target_probabilities_still_contribute_to_ce_gradient() {
    let mut m = model(5);
    let state = m.h_persistent.clone();
    // Obtain a nonzero latent, then set one target and one non-target output row
    // to make the latter probability tiny but representable.
    loss(&mut m, &state);
    let z = m.tape.z_final[..5].to_vec();
    let norm2: f32 = z.iter().map(|x| x * x).sum();
    assert!(norm2 > 0.0);
    m.unembed_w.data.fill(0.0);
    let amp = 12.0 * (5.0f32).sqrt() / norm2;
    for j in 0..5 {
        m.unembed_w.data[j] = -amp * z[j];
        m.unembed_w.data[5 + j] = amp * z[j];
    }
    m.h_persistent.copy_from_slice(&state);
    m.forward_train_chunk(&[0], &[1]);
    let p = m.tape.probs[0];
    assert!(p > 0.0 && p < 1e-5, "p={p}");
    m.zero_gradients();
    m.backward_chunk(1, 1.0);
    assert!(
        m.unembed_w.grad[..5].iter().any(|x| *x != 0.0),
        "tiny non-target gradient was pruned"
    );
}

#[test]
fn recurrent_model_memorizes_short_repeated_sequence() {
    let mut c = cfg(5);
    c.d_vocab = 4;
    c.chunk_len = 4;
    c.mem_capacity = 2;
    c.lr = 0.03;
    c.weight_decay = 0.0;
    let mut m = PSSALayerV2::new(c, 73);
    // The prescribed recurrent model itself is trained; no alternate n-gram path.
    let x = [0, 1, 2, 3];
    let y = [1, 2, 3, 0];
    for _ in 0..700 {
        m.reset_recurrent_state();
        m.forward_train_chunk(&x, &y);
        m.backward_and_step_chunk(4);
    }
    m.reset_recurrent_state();
    let loss = m.forward_train_chunk(&x, &y);
    let mut correct = 0;
    for t in 0..4 {
        let logits = &m.tape.logits[t * 4..(t + 1) * 4];
        let guess = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        correct += usize::from(guess == y[t]);
    }
    let accuracy = correct as f32 / 4.0;
    println!("memorization loss={loss:.6} accuracy={accuracy:.3}");
    assert!(
        loss < 0.1 && accuracy > 0.95,
        "loss={loss}, accuracy={accuracy}"
    );
}
