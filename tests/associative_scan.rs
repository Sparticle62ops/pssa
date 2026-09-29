//! Scalar-oracle regressions for the forward and reverse associative SSM scans.
//! The oracle deliberately uses PSSALayerV2, not the staged scan helpers.

use oxide_ai_pssa::{
    gpu_batch::{backward_chunk_batched, forward_train_chunk_batched},
    pssa::{PSSAConfigV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};

const GRAD_TOL: f32 = 1e-6;
const FORWARD_TOL: f32 = 2e-5;

fn model(latent: usize, chunk_len: usize) -> PSSALayerV2 {
    let cfg = PSSAConfigV2 {
        d_vocab: 41,
        d_latent: latent,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 5,
        chunk_len,
        tau_mem: 0.7,
        ..Default::default()
    };
    let mut m = PSSALayerV2::new(cfg, 12345);
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = ((i % 11) as f32 - 5.0) * 0.037;
    }
    // Keep long-chunk activations well-conditioned without removing any path.
    // The default slow SSM rates still retain substantial carry over 129 steps.
    for x in &mut m.w_b.data {
        *x *= 0.2;
    }
    for x in &mut m.w_c.data {
        *x *= 0.2;
    }
    for (i, x) in m.norm_gamma.data.iter_mut().enumerate() {
        *x = 0.9 + (i % 5) as f32 * 0.04;
    }
    for (i, x) in m.norm_beta.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.009;
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
    // Wrap the bank so a spurious write/reset cannot hide behind write_head=0.
    for entry in 0..m.cfg.mem_capacity + 2 {
        let key: Vec<_> = (0..m.cfg.d_mem_key)
            .map(|i| ((entry + 2 * i) % 7) as f32 * 0.04 - 0.1)
            .collect();
        let value: Vec<_> = (0..m.cfg.d_latent)
            .map(|i| ((3 * entry + i) % 11) as f32 * 0.08 - 0.4)
            .collect();
        m.memory.insert(&key, &value);
    }
    for entry in 0..m.cfg.mem_capacity {
        m.memory.confidence[entry] = 0.4 + entry as f32 * 0.07;
        m.memory.last_seen_step[entry] = 11 + entry * 3;
    }
    m.step_counter = 37;
    m
}

fn tokens(len: usize, round: usize) -> (Vec<usize>, Vec<usize>) {
    // Repeated adjacent and nonadjacent IDs exercise aliased embedding grads.
    let pattern = [3, 3, 7, 2, 3, 11, 7];
    let inputs = (0..len)
        .map(|t| pattern[(t + round) % pattern.len()])
        .collect();
    let targets = (0..len).map(|t| (t * 5 + round * 3 + 1) % 41).collect();
    (inputs, targets)
}

fn close(a: &[f32], b: &[f32], tolerance: f32, context: &str, field: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{context}: {field} shape");
    let mut max_error = 0.0f32;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let error = (x - y).abs();
        assert!(
            x.is_finite() && y.is_finite() && error < tolerance,
            "{context}: {field}[{i}] scalar={x:e}, scan={y:e}, abs error={error:e} (must be < {tolerance:e})"
        );
        max_error = max_error.max(error);
    }
    max_error
}

fn gradients(a: &PSSALayerV2, b: &PSSALayerV2, context: &str) -> f32 {
    let mut error = 0.0f32;
    macro_rules! check {
        ($($field:ident),* $(,)?) => {$(
            error = error.max(close(
                &a.$field.grad, &b.$field.grad, GRAD_TOL, context, stringify!($field),
            ));
        )*};
    }
    check!(
        embed_w, norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj,
        mlp_w1, mlp_w2, unembed_w,
    );
    for (name, x, y) in [
        (
            "adapter down",
            &a.adapters[0].down_proj.grad,
            &b.adapters[0].down_proj.grad,
        ),
        (
            "adapter up",
            &a.adapters[0].up_proj.grad,
            &b.adapters[0].up_proj.grad,
        ),
    ] {
        error = error.max(close(x, y, GRAD_TOL, context, name));
    }
    assert_eq!(
        a.embed_row_marks, b.embed_row_marks,
        "{context}: embedding marks"
    );
    error
}

fn nonvacuous_gradients(m: &PSSALayerV2, context: &str) {
    for (name, grad) in [
        ("SSM rates", &m.a_mat.grad),
        ("SSM delta", &m.w_delta.grad),
        ("SSM B", &m.w_b.grad),
        ("SSM C", &m.w_c.grad),
        ("memory query/input", &m.w_qx.grad),
        ("memory query/SSM", &m.w_qh.grad),
        ("memory gate", &m.w_gate.grad),
        ("memory projection", &m.w_proj.grad),
        ("adapter down", &m.adapters[0].down_proj.grad),
        ("adapter up", &m.adapters[0].up_proj.grad),
        ("MLP down", &m.mlp_w1.grad),
        ("MLP up", &m.mlp_w2.grad),
    ] {
        assert!(
            grad.iter().any(|x| x.abs() > 1e-9),
            "{context}: vacuous {name}"
        );
    }
}

// Packed lanes publish all these token rows but keep their recurrent tapes private.
fn forward_rows(a: &PSSALayerV2, b: &PSSALayerV2, offset: usize, len: usize, context: &str) {
    let d = a.cfg.d_latent;
    let k = a.cfg.d_mem_key;
    let s = a.cfg.d_state;
    macro_rules! check {
        ($field:ident, $width:expr) => {
            close(
                &a.tape.$field[..len * $width],
                &b.tape.$field[offset * $width..(offset + len) * $width],
                FORWARD_TOL,
                context,
                stringify!($field),
            );
        };
    }
    check!(x_norm, d);
    check!(delta, d);
    check!(b_proj, s);
    check!(c_proj, s);
    check!(y_ssm, d);
    check!(q_euc, k);
    check!(q_poincare, k);
    check!(mem_weights, a.cfg.mem_capacity);
    check!(m_val, d);
    check!(g_mem, d);
    check!(m_inj, d);
    check!(adapter_act, a.adapters[0].rank);
    check!(z_raw, d);
    check!(mlp_act, 2 * d);
    check!(z_final, d);
    check!(logits, a.cfg.d_vocab);
    check!(probs, a.cfg.d_vocab);
    check!(losses, 1);
}

#[test]
fn batched_scan_matches_scalar_across_lengths_threads_carry_and_accumulation() {
    let mut max_grad_error = 0.0f32;
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            // Odd latent width exercises scalar/SIMD tails as well as odd time lengths.
            for latent in [7, 32] {
                for capacity in [1, 2, 3, 7, 16, 17, 63, 64, 65, 129] {
                    let mut scalar = model(latent, capacity);
                    let mut scanned = model(latent, capacity);
                    let memory = scalar.memory.clone();
                    for (round, len) in [capacity, capacity.div_ceil(2), capacity].into_iter().enumerate() {
                        let context = format!("threads={threads}, latent={latent}, capacity={capacity}, round={round}, len={len}");
                        let (inputs, targets) = tokens(len, round);
                        let a = scalar.forward_train_chunk(&inputs, &targets);
                        let b = forward_train_chunk_batched(&mut scanned, &inputs, &targets);
                        close(&[a], &[b], FORWARD_TOL, &context, "loss");
                        forward_rows(&scalar, &scanned, 0, len, &context);
                        let hs = latent * scalar.cfg.d_state;
                        close(&scalar.tape.h_states[..(len + 1) * hs], &scanned.tape.h_states[..(len + 1) * hs], FORWARD_TOL, &context, "all recurrent states");
                        close(&scalar.tape.bar_a[..len * hs], &scanned.tape.bar_a[..len * hs], FORWARD_TOL, &context, "local A");
                        close(&scalar.tape.bar_b[..len * hs], &scanned.tape.bar_b[..len * hs], FORWARD_TOL, &context, "local B");
                        close(&scalar.h_persistent, &scanned.h_persistent, FORWARD_TOL, &context, "carry");
                        assert_eq!(scalar.memory, memory, "{context}: scalar forward changed memory/refractory state");
                        assert_eq!(scanned.memory, memory, "{context}: scan forward changed memory/refractory state");
                        let scalar_carry = scalar.h_persistent.clone();
                        let scanned_carry = scanned.h_persistent.clone();
                        // Do not zero gradients or carry between chunks. Shortening and
                        // regrowing also catches stale scan padding and reverse scratch.
                        let scale = if round == 1 { 0.625 } else { 1.0 };
                        scalar.backward_chunk(len, scale);
                        backward_chunk_batched(&mut scanned, len, scale);
                        max_grad_error = max_grad_error.max(gradients(&scalar, &scanned, &context));
                        nonvacuous_gradients(&scalar, &context);
                        assert_eq!(scalar.h_persistent, scalar_carry, "{context}: scalar backward changed carry");
                        assert_eq!(scanned.h_persistent, scanned_carry, "{context}: scan backward changed carry");
                        assert_eq!(scalar.memory, memory, "{context}: scalar backward changed memory/refractory state");
                        assert_eq!(scanned.memory, memory, "{context}: scan backward changed memory/refractory state");
                    }
                }
            }
        });
    }
    eprintln!("full scan/scalar twin max absolute gradient error: {max_grad_error:e}");
}

#[test]
fn reverse_scan_matches_scalar_on_identical_forward_tapes() {
    // Isolate the reverse scan from forward reassociation, so an error in one
    // direction cannot compensate for an error in the other.
    for threads in [1, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut scalar = model(7, 129);
            let mut scanned = model(7, 129);
            let memory = scalar.memory.clone();
            for (round, len) in [1, 2, 5, 64, 129, 3, 65].into_iter().enumerate() {
                let context = format!("reverse-only threads={threads}, round={round}, len={len}");
                let (inputs, targets) = tokens(len, round);
                scalar.forward_train_chunk(&inputs, &targets);
                scanned.forward_train_chunk(&inputs, &targets);
                assert_eq!(
                    scalar.tape.h_states, scanned.tape.h_states,
                    "{context}: oracle tapes"
                );
                assert_eq!(
                    scalar.memory, memory,
                    "{context}: scalar forward changed memory"
                );
                assert_eq!(
                    scanned.memory, memory,
                    "{context}: twin forward changed memory"
                );
                scalar.backward_chunk(len, 0.75);
                backward_chunk_batched(&mut scanned, len, 0.75);
                gradients(&scalar, &scanned, &context);
                nonvacuous_gradients(&scalar, &context);
                assert_eq!(
                    scalar.memory, memory,
                    "{context}: scalar backward changed memory"
                );
                assert_eq!(
                    scanned.memory, memory,
                    "{context}: reverse scan changed memory/refractory state"
                );
            }
        });
    }
}

#[test]
fn protected_memory_writes_remain_ordered_between_scanned_chunks() {
    let mut scalar = model(7, 17);
    let mut scanned = model(7, 17);
    for round in 0..4 {
        let context = format!("protected write round={round}");
        // Alternate a defended write with an actual overwrite, then consume
        // that updated bank on the next chunk. These operations cannot be
        // folded into the affine scan or moved ahead of backward.
        let slot = scalar.memory.write_head;
        let overwrite = round % 2 == 1;
        for m in [&mut scalar, &mut scanned] {
            m.step_counter = 100 * (round + 1);
            m.memory.confidence[slot] = if overwrite { 0.001 } else { 5.0 };
            m.memory.last_seen_step[slot] = 0;
        }
        let (inputs, targets) = tokens(17, round);
        scalar.forward_train_chunk(&inputs, &targets);
        forward_train_chunk_batched(&mut scanned, &inputs, &targets);
        scalar.backward_chunk(17, 1.0);
        backward_chunk_batched(&mut scanned, 17, 1.0);
        gradients(&scalar, &scanned, &context);
        let before = scanned.memory.clone();
        scalar.insert_training_memory(4.0, 17);
        scanned.insert_training_memory(4.0, 17);
        assert_eq!(scanned.memory.count, scalar.memory.count);
        assert_eq!(scanned.memory.write_head, scalar.memory.write_head);
        assert_eq!(scanned.memory.last_seen_step, scalar.memory.last_seen_step);
        assert_eq!(scanned.memory.confidence, scalar.memory.confidence);
        close(&scalar.memory.keys, &scanned.memory.keys, FORWARD_TOL, &context, "bank keys");
        close(&scalar.memory.values, &scanned.memory.values, FORWARD_TOL, &context, "bank values");
        close(&scalar.memory.norm_sq, &scanned.memory.norm_sq, FORWARD_TOL, &context, "bank norms");
        assert_eq!(scanned.memory.last_seen_step[slot], scanned.step_counter);
        if overwrite {
            assert_eq!(scanned.memory.write_head, (slot + 1) % scanned.cfg.mem_capacity);
            assert_ne!(scanned.memory.values, before.values);
        } else {
            assert_eq!(scanned.memory.write_head, slot);
            assert_eq!(scanned.memory.keys, before.keys);
            assert_eq!(scanned.memory.values, before.values);
            assert!(scanned.memory.confidence[slot] < before.confidence[slot]);
        }
    }
}

#[test]
fn packed_scans_match_scalar_with_ragged_reset_and_inactive_lanes() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut scalar = model(7, 65);
            let mut packed = model(7, 65);
            let memory = packed.memory.clone();
            let model_carry = packed.h_persistent.clone();
            let mut batch = SequenceBatch::new(&mut packed, 4).unwrap();
            let mut states = vec![model_carry.clone(); 4];
            for (lane, state) in states.iter_mut().enumerate() {
                for x in state.iter_mut() {
                    *x += lane as f32 * 0.031;
                }
                batch.state_mut(lane).copy_from_slice(state);
            }
            // Lane order differs from storage order; lane 3 is always inactive.
            // Reset a previously nonzero lane, omit it, then reuse it later.
            let rounds: &[&[(usize, usize, bool)]] = &[
                &[(2, 65, false), (0, 1, false), (1, 2, false)],
                &[(1, 3, true), (2, 17, false), (0, 33, false)],
                &[(2, 1, false), (0, 2, true)],
                &[(1, 65, false)],
            ];
            for (round, spec) in rounds.iter().enumerate() {
                let context = format!("packed threads={threads}, round={round}");
                let data: Vec<_> = spec
                    .iter()
                    .map(|&(lane, len, _)| tokens(len, round + lane))
                    .collect();
                let sequences: Vec<_> = spec
                    .iter()
                    .zip(&data)
                    .map(|(&(lane, _, reset), (inputs, targets))| Sequence {
                        lane,
                        inputs,
                        targets,
                        reset,
                    })
                    .collect();
                let total: usize = sequences.iter().map(|s| s.inputs.len()).sum();
                let before: Vec<_> = (0..4).map(|lane| batch.state(lane).to_vec()).collect();
                let actual_loss = batch.forward(&mut packed, &sequences).unwrap();
                assert_eq!(
                    packed.memory, memory,
                    "{context}: packed forward changed memory/refractory state"
                );
                assert_eq!(
                    packed.h_persistent, model_carry,
                    "{context}: packed forward changed model carry"
                );
                let mut expected_loss = 0.0;
                let mut offset = 0;
                for seq in &sequences {
                    scalar.h_persistent.copy_from_slice(&states[seq.lane]);
                    if seq.reset {
                        scalar.reset_recurrent_state();
                    }
                    let len = seq.inputs.len();
                    expected_loss += scalar.forward_train_chunk(seq.inputs, seq.targets)
                        * len as f32
                        / total as f32;
                    forward_rows(&scalar, &packed, offset, len, &context);
                    states[seq.lane].copy_from_slice(&scalar.h_persistent);
                    close(
                        &states[seq.lane],
                        batch.state(seq.lane),
                        FORWARD_TOL,
                        &context,
                        "lane carry",
                    );
                    scalar.backward_chunk(len, 0.75 * len as f32 / total as f32);
                    offset += len;
                }
                close(
                    &[expected_loss],
                    &[actual_loss],
                    FORWARD_TOL,
                    &context,
                    "token-weighted loss",
                );
                let after_forward: Vec<_> = (0..4).map(|lane| batch.state(lane).to_vec()).collect();
                batch.backward(&mut packed, 0.75).unwrap();
                gradients(&scalar, &packed, &context);
                nonvacuous_gradients(&scalar, &context);
                for lane in 0..4 {
                    assert_eq!(
                        batch.state(lane),
                        after_forward[lane],
                        "{context}: backward changed lane {lane} carry"
                    );
                    if !sequences.iter().any(|s| s.lane == lane) {
                        assert_eq!(
                            batch.state(lane),
                            before[lane],
                            "{context}: inactive lane {lane} advanced"
                        );
                    }
                }
                assert_eq!(
                    packed.h_persistent, model_carry,
                    "{context}: packed backward changed model carry"
                );
                assert_eq!(
                    packed.memory, memory,
                    "{context}: packed backward changed memory/refractory state"
                );
                assert_eq!(
                    scalar.memory, memory,
                    "{context}: scalar changed memory/refractory state"
                );
            }
        });
    }
}
