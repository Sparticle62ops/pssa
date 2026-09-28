use oxide_ai_pssa::{
    pssa::{PSSAConfigV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};

fn model(large: bool) -> PSSALayerV2 {
    let cfg = PSSAConfigV2 {
        d_vocab: if large { 257 } else { 11 },
        d_latent: if large { 64 } else { 7 },
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 4,
        chunk_len: if large { 17 } else { 6 },
        tau_mem: 0.7,
        ..Default::default()
    };
    let mut m = PSSALayerV2::new(cfg, 7);
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = ((i % 9) as f32 - 4.0) * 0.023;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = ((i % 7) as f32 - 3.0) * 0.019;
    }
    for (i, x) in m.adapters[0].consolidated_up.iter_mut().enumerate() {
        *x = ((i % 5) as f32 - 2.0) * 0.011;
    }
    for entry in 0..4 {
        let key: Vec<_> = (0..4)
            .map(|i| ((entry + 2 * i) % 7) as f32 * 0.04 - 0.1)
            .collect();
        let value: Vec<_> = (0..m.cfg.d_latent)
            .map(|i| ((3 * entry + i) % 11) as f32 * 0.08 - 0.4)
            .collect();
        m.memory.insert(&key, &value);
    }
    m
}

fn close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite() && y.is_finite() && (x - y).abs() < 2e-5 + 2e-4 * x.abs().max(y.abs()),
            "{label}[{i}]: {x} != {y}"
        );
    }
}

fn gradients(a: &PSSALayerV2, b: &PSSALayerV2) {
    macro_rules! check { ($($f:ident),*) => { $(close(&a.$f.grad, &b.$f.grad, stringify!($f));)* }; }
    check!(
        embed_w, norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj,
        mlp_w1, mlp_w2, unembed_w
    );
    close(
        &a.adapters[0].down_proj.grad,
        &b.adapters[0].down_proj.grad,
        "adapter down",
    );
    close(
        &a.adapters[0].up_proj.grad,
        &b.adapters[0].up_proj.grad,
        "adapter up",
    );
}

#[test]
fn independent_batch_matches_separate_runs_all_gradients_carry_ragged_and_accumulation() {
    for (large, count) in [(false, 1), (false, 3), (true, 4)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut separate = model(large);
            let mut packed = model(large);
            let mut batch = SequenceBatch::new(&mut packed, count).unwrap();
            let mut states = vec![vec![0.0; packed.cfg.d_latent * 3]; count];
            for (lane, state) in states.iter_mut().enumerate() {
                for (i, x) in state.iter_mut().enumerate() {
                    *x = ((i + lane) % 11) as f32 * 0.013 - 0.04;
                }
                batch.state_mut(lane).copy_from_slice(state);
            }
            // Full, ragged, then sparse/reordered reuse. Carries survive calls,
            // but all reference backwards detach at each chunk boundary.
            for round in 0..3 {
                let lanes: Vec<_> = if round == 2 {
                    (0..count).rev().step_by(2).collect()
                } else {
                    (0..count).collect()
                };
                let inputs: Vec<Vec<_>> = lanes
                    .iter()
                    .map(|&lane| {
                        let len = if round == 0 {
                            packed.cfg.chunk_len
                        } else {
                            1 + lane
                        };
                        (0..len)
                            .map(|t| (t * 3 + lane) % packed.cfg.d_vocab)
                            .collect()
                    })
                    .collect();
                let targets: Vec<Vec<_>> = inputs
                    .iter()
                    .map(|x| x.iter().map(|t| (t + 1) % packed.cfg.d_vocab).collect())
                    .collect();
                let sequences: Vec<_> = lanes
                    .iter()
                    .enumerate()
                    .map(|(i, &lane)| Sequence {
                        lane,
                        inputs: &inputs[i],
                        targets: &targets[i],
                        reset: round == 1 && lane == 0,
                    })
                    .collect();
                let total: usize = inputs.iter().map(Vec::len).sum();
                let loss = batch.forward(&mut packed, &sequences).unwrap();
                let mut expected_loss = 0.0;
                let mut offset = 0;
                for seq in &sequences {
                    separate.h_persistent.copy_from_slice(&states[seq.lane]);
                    if seq.reset {
                        separate.reset_recurrent_state();
                    }
                    let n = seq.inputs.len();
                    expected_loss += separate.forward_train_chunk(seq.inputs, seq.targets)
                        * n as f32
                        / total as f32;
                    close(
                        &separate.tape.probs[..n * separate.cfg.d_vocab],
                        &packed.tape.probs
                            [offset * separate.cfg.d_vocab..(offset + n) * separate.cfg.d_vocab],
                        "probabilities",
                    );
                    separate.backward_chunk(n, 0.7 * n as f32 / total as f32);
                    states[seq.lane].copy_from_slice(&separate.h_persistent);
                    offset += n;
                }
                assert!((loss - expected_loss).abs() < 2e-5);
                batch.backward(&mut packed, 0.7).unwrap();
                for (lane, state) in states.iter().enumerate() {
                    close(state, batch.state(lane), "lane carry");
                }
                gradients(&separate, &packed);
            }
            for g in [
                &packed.w_qx.grad,
                &packed.w_qh.grad,
                &packed.mlp_w1.grad,
                &packed.adapters[0].down_proj.grad,
            ] {
                assert!(g.iter().any(|x| x.abs() > 1e-9), "vacuous gradient");
            }
            separate.apply_adamw(1e-4);
            packed.apply_adamw(1e-4);
            close(
                &separate.embed_w.data,
                &packed.embed_w.data,
                "Adam embeddings",
            );
            close(&separate.a_mat.data, &packed.a_mat.data, "Adam rates");
        });
    }
}

#[test]
fn batch_boundaries_do_not_couple_other_lanes() {
    let mut a = model(false);
    let mut b = model(false);
    let mut ba = SequenceBatch::new(&mut a, 2).unwrap();
    let mut bb = SequenceBatch::new(&mut b, 2).unwrap();
    bb.state_mut(0).fill(3.0);
    let seqs = [
        Sequence {
            lane: 0,
            inputs: &[1, 2, 3],
            targets: &[2, 3, 4],
            reset: false,
        },
        Sequence {
            lane: 1,
            inputs: &[5],
            targets: &[6],
            reset: false,
        },
    ];
    ba.forward(&mut a, &seqs).unwrap();
    bb.forward(&mut b, &seqs).unwrap();
    assert_ne!(ba.state(0), bb.state(0));
    assert_eq!(ba.state(1), bb.state(1));
    assert_eq!(&a.tape.probs[33..44], &b.tape.probs[33..44]);
    ba.backward(&mut a, 1.0).unwrap();
    bb.backward(&mut b, 1.0).unwrap();
    close(
        &a.bwd_g_xnorm[21..28],
        &b.bwd_g_xnorm[21..28],
        "isolated input adjoint",
    );
}

#[test]
fn invalid_batch_requests_are_errors_and_do_not_advance_carry() {
    let mut m = model(false);
    assert!(SequenceBatch::new(&mut m, 0).is_err());
    assert!(SequenceBatch::new(&mut m, usize::MAX).is_err());
    let mut batch = SequenceBatch::new(&mut m, 2).unwrap();
    assert!(batch.backward(&mut m, 1.0).is_err());
    assert!(batch.forward(&mut m, &[]).is_err());
    for (lane, x, y) in [
        (0, vec![], vec![]),
        (0, vec![1], vec![2, 3]),
        (0, vec![11], vec![0]),
        (2, vec![1], vec![2]),
        (0, vec![1; 7], vec![2; 7]),
    ] {
        assert!(
            batch
                .forward(
                    &mut m,
                    &[Sequence {
                        lane,
                        inputs: &x,
                        targets: &y,
                        reset: false
                    }]
                )
                .is_err()
        );
        assert!(batch.state(0).iter().all(|&x| x == 0.0));
    }
    let seq = [Sequence {
        lane: 0,
        inputs: &[1],
        targets: &[2],
        reset: false,
    }];
    batch.forward(&mut m, &seq).unwrap();
    assert!(batch.forward(&mut m, &seq).is_err());
    assert!(batch.backward(&mut m, f32::NAN).is_err());
    batch.backward(&mut m, 1.0).unwrap();
    assert!(batch.backward(&mut m, 1.0).is_err());
}
