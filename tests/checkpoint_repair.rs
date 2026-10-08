use pssa::checkpoint::{self, CheckpointFormat};
use pssa::pssa::{PSSAConfigV2, PSSALayerV2, ParamMatrix, ParamVector};
use std::{fs, path::PathBuf};

fn path(s: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pssa-checkpoint-{s}-{}", std::process::id()))
}
fn fill(x: &mut [f32], v: f32) {
    for (i, n) in x.iter_mut().enumerate() {
        *n = v + i as f32 * 0.001;
    }
}
fn pm(p: &mut ParamMatrix, n: f32) {
    fill(&mut p.data, n * 0.001);
    fill(&mut p.grad, n + 1.);
    fill(&mut p.m, n + 2.);
    fill(&mut p.v, n + 3.);
}
fn pv(p: &mut ParamVector, n: f32) {
    fill(&mut p.data, n * 0.001);
    fill(&mut p.grad, n + 1.);
    fill(&mut p.m, n + 2.);
    fill(&mut p.v, n + 3.);
}
fn model() -> PSSALayerV2 {
    let mut x = PSSALayerV2::new(
        PSSAConfigV2 {
            depth: 1,
            d_vocab: 5,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 2,
            lr: 0.002,
            beta1: 0.8,
            beta2: 0.9,
            weight_decay: 0.03,
            eps: 1e-6,
            tau_mem: 0.2,
            ema_alpha: 0.2,
        },
        19,
    );
    x.vocabulary = vec![
        "<unk>".into(),
        "alpha".into(),
        "beta".into(),
        "gamma".into(),
        "delta".into(),
    ];
    for i in 0..12 {
        pm(persistent_matrix_mut(&mut x, i), i as f32 + 0.1);
    }
    pv(&mut x.norm_gamma, 20.);
    pv(&mut x.norm_beta, 30.);
    pm(&mut x.adapters[0].down_proj, 40.);
    pm(&mut x.adapters[0].up_proj, 50.);
    fill(&mut x.adapters[0].consolidated_up, 0.06);
    fill(&mut x.h_persistent, 0.07);
    x.step_counter = 11;
    x.rng.state = 0x1234_5678_9abc_def0;
    for (k, v) in [
        ([0.1, 0.2], [1., 2., 3., 4.]),
        ([0.2, 0.1], [2., 3., 4., 5.]),
        ([0.3, 0.1], [3., 4., 5., 6.]),
        ([0.1, 0.3], [4., 5., 6., 7.]),
    ] {
        x.memory.insert(&k, &v);
    }
    x.memory.confidence.copy_from_slice(&[1.5, 2., 2.5]);
    x.memory.last_seen_step.copy_from_slice(&[1, 2, 3]);
    x.embed_row_marks.copy_from_slice(&[5, 4, 3, 2, 1]);
    x
}
fn matrix(a: &ParamMatrix, b: &ParamMatrix) {
    assert_eq!(a.data, b.data);
    assert_eq!(a.grad, b.grad);
    assert_eq!(a.m, b.m);
    assert_eq!(a.v, b.v)
}
fn vector(a: &ParamVector, b: &ParamVector) {
    assert_eq!(a.data, b.data);
    assert_eq!(a.grad, b.grad);
    assert_eq!(a.m, b.m);
    assert_eq!(a.v, b.v)
}
fn state(a: &PSSALayerV2, b: &PSSALayerV2) {
    assert_eq!(a.cfg.d_vocab, b.cfg.d_vocab);
    assert_eq!(a.cfg.d_latent, b.cfg.d_latent);
    assert_eq!(a.cfg.d_state, b.cfg.d_state);
    assert_eq!(a.cfg.d_mem_key, b.cfg.d_mem_key);
    assert_eq!(a.cfg.mem_capacity, b.cfg.mem_capacity);
    assert_eq!(a.cfg.chunk_len, b.cfg.chunk_len);
    assert_eq!(a.cfg.lr, b.cfg.lr);
    assert_eq!(a.cfg.beta1, b.cfg.beta1);
    assert_eq!(a.cfg.beta2, b.cfg.beta2);
    assert_eq!(a.cfg.weight_decay, b.cfg.weight_decay);
    assert_eq!(a.cfg.eps, b.cfg.eps);
    assert_eq!(a.cfg.tau_mem, b.cfg.tau_mem);
    assert_eq!(a.cfg.ema_alpha, b.cfg.ema_alpha);
    assert_eq!(a.step_counter, b.step_counter);
    assert_eq!(a.rng.state, b.rng.state);
    assert_eq!(a.vocabulary, b.vocabulary);
    assert_eq!(a.tokenizer_json, b.tokenizer_json);
    assert_eq!(a.lr_schedule_total_updates, b.lr_schedule_total_updates);
    assert_eq!(a.lr_schedule_warmup_steps, b.lr_schedule_warmup_steps);
    for (x, y) in [
        (&a.embed_w, &b.embed_w),
        (&a.a_mat, &b.a_mat),
        (&a.w_delta, &b.w_delta),
        (&a.w_b, &b.w_b),
        (&a.w_c, &b.w_c),
        (&a.w_qx, &b.w_qx),
        (&a.w_qh, &b.w_qh),
        (&a.w_gate, &b.w_gate),
        (&a.w_proj, &b.w_proj),
        (&a.mlp_w1, &b.mlp_w1),
        (&a.mlp_w2, &b.mlp_w2),
        (&a.unembed_w, &b.unembed_w),
        (&a.adapters[0].down_proj, &b.adapters[0].down_proj),
        (&a.adapters[0].up_proj, &b.adapters[0].up_proj),
    ] {
        matrix(x, y)
    }
    vector(&a.norm_gamma, &b.norm_gamma);
    vector(&a.norm_beta, &b.norm_beta);
    assert_eq!(a.adapters[0].consolidated_up, b.adapters[0].consolidated_up);
    assert_eq!(a.h_persistent, b.h_persistent);
    assert_eq!(
        (
            a.memory.count,
            a.memory.write_head,
            &a.memory.keys,
            &a.memory.values,
            &a.memory.norm_sq,
            &a.memory.confidence,
            &a.memory.last_seen_step,
            &a.embed_row_marks
        ),
        (
            b.memory.count,
            b.memory.write_head,
            &b.memory.keys,
            &b.memory.values,
            &b.memory.norm_sq,
            &b.memory.confidence,
            &b.memory.last_seen_step,
            &b.embed_row_marks
        )
    );
}
#[test]
fn v6_exact_state_logits_and_next_update() {
    let mut a = model();
    let p = path("state");
    checkpoint::save_model_v6(&a, &p).unwrap();
    let mut b = checkpoint::load_checkpoint(&p).unwrap();
    assert_eq!(b.format, CheckpointFormat::V6);
    state(&a, &b.model);
    let (mut la, mut lb) = (vec![0.; 5], vec![0.; 5]);
    a.forward_inference(1, &mut la);
    b.model.forward_inference(1, &mut lb);
    for (x, y) in la.iter().zip(lb) {
        assert!((x - y).abs() <= 1e-6)
    }
    a.forward_train_chunk(&[1, 2], &[2, 3]);
    b.model.forward_train_chunk(&[1, 2], &[2, 3]);
    a.backward_chunk(2, 1.);
    b.model.backward_chunk(2, 1.);
    a.apply_adamw(0.002);
    b.model.apply_adamw(0.002);
    state(&a, &b.model);
    fs::remove_file(p).unwrap();
}
fn fnv(x: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in x {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    }
    h
}
fn checksum(x: &mut [u8]) {
    let h = fnv(&x[22..]);
    x[14..22].copy_from_slice(&h.to_le_bytes())
}
#[test]
fn v7_exact_state_for_word_checkpoint() {
    let a = model();
    let p = path("v7-state");
    checkpoint::save_model(&a, &p).unwrap();
    let b = checkpoint::load_checkpoint(&p).unwrap();
    assert_eq!(b.format, CheckpointFormat::V7);
    assert!(b.model.tokenizer_json.is_none());
    assert_eq!(b.model.lr_schedule_total_updates, None);
    assert_eq!(b.model.lr_schedule_warmup_steps, None);
    state(&a, &b.model);
    fs::remove_file(p).unwrap();
}

#[test]
fn v7_persists_fixed_schedule_horizon() {
    let mut a = model();
    a.lr_schedule_total_updates = Some(1234);
    let p = path("v7-schedule-horizon");
    checkpoint::save_model(&a, &p).unwrap();
    let b = checkpoint::load_checkpoint(&p).unwrap();
    assert_eq!(b.format, CheckpointFormat::V7);
    assert_eq!(b.model.lr_schedule_total_updates, Some(1234));
    assert_eq!(b.model.lr_schedule_warmup_steps, None);
    fs::remove_file(p).unwrap();
}

#[test]
fn v7_schedule_warmup_roundtrips_and_v6_refuses_metadata_loss() {
    let p = path("warmup-roundtrip");
    let mut a = model();
    for warmup in [0, 1, 10, 99] {
        a.lr_schedule_total_updates = Some(100);
        a.lr_schedule_warmup_steps = Some(warmup);
        checkpoint::save_model(&a, &p).unwrap();
        let b = checkpoint::load_checkpoint(&p).unwrap();
        state(&a, &b.model);
        let good = fs::read(&p).unwrap();
        assert!(
            checkpoint::save_model_v6(&a, &p)
                .unwrap_err()
                .to_string()
                .contains("schedule")
        );
        assert_eq!(fs::read(&p).unwrap(), good);
    }
    a.lr_schedule_warmup_steps = None;
    assert!(checkpoint::save_model_v6(&a, &p).is_err());
    a.lr_schedule_total_updates = None;
    a.lr_schedule_warmup_steps = Some(0);
    assert!(checkpoint::save_model_v6(&a, &p).is_err());
    fs::remove_file(p).unwrap();
}

#[test]
fn invalid_schedule_metadata_cannot_replace_a_checkpoint() {
    let p = path("invalid-schedule-save");
    checkpoint::save_model(&model(), &p).unwrap();
    let good = fs::read(&p).unwrap();
    for (horizon, warmup) in [
        (Some(0), None),
        (Some(10), None), // model clock is already at 11
        (Some(10), Some(0)),
        (Some(0), Some(0)),
        (Some(10), Some(10)),
        (Some(10), Some(11)),
        (None, Some(0)),
        (None, Some(5)),
    ] {
        let mut a = model();
        a.lr_schedule_total_updates = horizon;
        a.lr_schedule_warmup_steps = warmup;
        assert!(
            checkpoint::save_model(&a, &p).is_err(),
            "{horizon:?}/{warmup:?}"
        );
        assert_eq!(fs::read(&p).unwrap(), good);
    }
    fs::remove_file(p).unwrap();
}

#[test]
fn old_v7_tails_load_and_invalid_new_tails_are_rejected() {
    let p = path("v7-tail-compatibility");
    checkpoint::save_model(&model(), &p).unwrap();
    let no_tail = fs::read(&p).unwrap();
    for (words, expected) in [
        (vec![], Some((None, None))),
        (vec![0], Some((None, None))),
        (vec![10], None), // horizon may equal, but not precede, the clock
        (vec![11], Some((Some(11), None))),
        (vec![100], Some((Some(100), None))),
        (vec![100, 0], Some((Some(100), Some(0)))),
        (vec![100, 10], Some((Some(100), Some(10)))),
        (vec![0, 0], None),
        (vec![10, 10], None),
        (vec![10, 11], None),
        (vec![100, 10, 1], None),
    ] {
        let mut bytes = no_tail.clone();
        for word in &words {
            bytes.extend_from_slice(&(*word as u64).to_le_bytes());
        }
        let len = (bytes.len() - 22) as u64;
        bytes[6..14].copy_from_slice(&len.to_le_bytes());
        checksum(&mut bytes);
        fs::write(&p, bytes).unwrap();
        let result = checkpoint::load_checkpoint(&p);
        if let Some((horizon, warmup)) = expected {
            let loaded = result.unwrap().model;
            assert_eq!(loaded.lr_schedule_total_updates, horizon);
            assert_eq!(loaded.lr_schedule_warmup_steps, warmup);
        } else {
            assert!(result.is_err(), "accepted tail {words:?}");
        }
    }
    for extra_len in [1, 7, 9, 15, 17] {
        let mut bytes = no_tail.clone();
        bytes.resize(bytes.len() + extra_len, 1);
        let len = (bytes.len() - 22) as u64;
        bytes[6..14].copy_from_slice(&len.to_le_bytes());
        checksum(&mut bytes);
        fs::write(&p, bytes).unwrap();
        assert!(checkpoint::load_checkpoint(&p).is_err());
    }
    fs::remove_file(p).unwrap();
}

#[test]
fn malformed_v6_is_rejected() {
    let m = model();
    let p = path("good");
    checkpoint::save_model_v6(&m, &p).unwrap();
    let good = fs::read(&p).unwrap();
    let cases = vec![
        {
            let mut x = good.clone();
            x[30] ^= 1;
            x
        },
        good[..21].to_vec(),
        good[..22].to_vec(),
        good[..100].to_vec(),
        good[..good.len() - 1].to_vec(),
        {
            let mut x = good.clone();
            x[22..30].copy_from_slice(&0u64.to_le_bytes());
            checksum(&mut x);
            x
        },
        {
            let mut x = good.clone();
            x[22..30].copy_from_slice(&u64::MAX.to_le_bytes());
            checksum(&mut x);
            x
        },
        {
            let mut x = good.clone();
            x.push(1);
            x
        },
    ];
    for (i, x) in cases.into_iter().enumerate() {
        let q = path(&format!("bad-{i}"));
        fs::write(&q, x).unwrap();
        assert!(checkpoint::load_checkpoint(&q).is_err());
        let _ = fs::remove_file(q);
    }
    let offset = 22 + 48 + 28 + 8 + 8;
    let mut x = good.clone();
    x[offset..offset + 8].copy_from_slice(&1u64.to_le_bytes());
    checksum(&mut x);
    let q = path("vocab");
    fs::write(&q, x).unwrap();
    assert!(checkpoint::load_checkpoint(&q).is_err());
    let _ = fs::remove_file(q);
    let tensor = 22 + 48 + 28 + 8 + 8 + 8 + m.vocabulary.iter().map(|s| 8 + s.len()).sum::<usize>();
    let mut x = good.clone();
    x[tensor + 8..tensor + 12].copy_from_slice(&f32::NAN.to_le_bytes());
    checksum(&mut x);
    let q = path("nan");
    fs::write(&q, x).unwrap();
    assert!(checkpoint::load_checkpoint(&q).is_err());
    let _ = fs::remove_file(q);
    let mut x = m;
    x.vocabulary[0] = "bad".into();
    assert!(checkpoint::save_model_v6(&x, path("bad-save")).is_err());
    x = model();
    x.embed_w.v[0] = -1.;
    assert!(checkpoint::save_model_v6(&x, path("bad-v")).is_err());
    fs::remove_file(p).unwrap();
}
fn old(m: &PSSALayerV2) -> Vec<u8> {
    fn u(b: &mut Vec<u8>, n: usize) {
        b.extend_from_slice(&(n as u32).to_le_bytes())
    }
    fn f(b: &mut Vec<u8>, x: &[f32]) {
        u(b, x.len());
        for &v in x {
            b.extend_from_slice(&v.to_le_bytes())
        }
    }
    let mut b = Vec::new();
    b.extend_from_slice(b"PSSA");
    b.extend_from_slice(&5u16.to_le_bytes());
    for n in [
        m.cfg.d_vocab,
        m.cfg.d_latent,
        m.cfg.d_state,
        m.cfg.d_mem_key,
        m.cfg.mem_capacity,
        m.cfg.chunk_len,
        m.memory.count,
        1,
    ] {
        u(&mut b, n)
    }
    f(&mut b, &m.embed_w.data);
    f(&mut b, &m.norm_gamma.data);
    f(&mut b, &m.norm_beta.data);
    let a: Vec<f32> = m.a_mat.data.iter().map(|x| -(x.exp()).ln_1p()).collect();
    f(&mut b, &a);
    for p in [
        &m.w_delta,
        &m.w_b,
        &m.w_c,
        &m.w_qx,
        &m.w_qh,
        &m.w_gate,
        &m.w_proj,
        &m.mlp_w1,
        &m.mlp_w2,
        &m.unembed_w,
    ] {
        f(&mut b, &p.data)
    }
    f(&mut b, &m.memory.keys[..m.memory.count * m.cfg.d_mem_key]);
    f(&mut b, &m.memory.values[..m.memory.count * m.cfg.d_latent]);
    u(&mut b, 16);
    f(&mut b, &m.adapters[0].down_proj.data);
    f(&mut b, &m.adapters[0].up_proj.data);
    b
}
#[test]
fn legacy_v5_conversion_and_artifacts() {
    let mut m = model();
    m.vocabulary.clear();
    m.h_persistent.fill(0.);
    m.adapters[0].consolidated_up.fill(0.);
    m.memory.count = 1;
    m.memory.write_head = 0;
    let p = path("legacy");
    fs::write(&p, old(&m)).unwrap();
    let mut b = checkpoint::load_checkpoint(&p).unwrap();
    assert_eq!(b.format, CheckpointFormat::LegacyV5InferenceOnly);
    assert!(b.model.vocabulary.is_empty());
    assert_eq!(b.model.step_counter, 0);
    let (mut la, mut lb) = (vec![0.; 5], vec![0.; 5]);
    m.forward_inference(1, &mut la);
    b.model.forward_inference(1, &mut lb);
    for (x, y) in la.iter().zip(lb) {
        assert!((x - y).abs() < 2e-6)
    }
    fs::remove_file(p).unwrap();
    for file in ["data/model.pssa", "data/model_v2.pssa"] {
        let r = checkpoint::load_checkpoint(file);
        assert!(r.is_ok(), "{file}: {:?}", r.err().map(|e| e.to_string()));
        assert_eq!(r.unwrap().format, CheckpointFormat::LegacyV5InferenceOnly)
    }
}
#[test]
fn malformed_legacy_count_and_rank_are_rejected() {
    let mut m = model();
    m.vocabulary.clear();
    m.memory.count = 1;
    m.memory.write_head = 0;
    let good = old(&m);
    let mut x = good.clone();
    x[30..34].copy_from_slice(&4u32.to_le_bytes());
    let p = path("legacy-count");
    fs::write(&p, x).unwrap();
    assert!(checkpoint::load_checkpoint(&p).is_err());
    let _ = fs::remove_file(p);
    let mut x = good;
    let rank = x.len() - (4 + 64 * 4) * 2 - 4;
    x[rank..rank + 4].copy_from_slice(&15u32.to_le_bytes());
    let p = path("legacy-rank");
    fs::write(&p, x).unwrap();
    assert!(checkpoint::load_checkpoint(&p).is_err());
    let _ = fs::remove_file(p);
}

#[test]
fn legacy_partial_memory_resaves_and_overwrites_oldest_slot_first() {
    let p = path("legacy-partial-ring");
    let mut m = model();
    m.memory.count = 1;
    m.memory.write_head = 0;
    fs::write(&p, old(&m)).unwrap();
    let mut loaded = checkpoint::load_checkpoint(&p).unwrap().model;
    assert_eq!(loaded.memory.count, 1);
    assert_eq!(loaded.memory.write_head, 0);
    checkpoint::save_model_v6(&loaded, &p).unwrap();
    let copy = checkpoint::load_checkpoint(&p).unwrap().model;
    state(&loaded, &copy);
    checkpoint::save_model(&loaded, &p).unwrap();
    let copy = checkpoint::load_checkpoint(&p).unwrap().model;
    state(&loaded, &copy);
    assert_eq!(loaded.memory.insert(&[0.1, 0.1], &[10.; 4]), 1);
    assert_eq!(loaded.memory.insert(&[0.2, 0.1], &[20.; 4]), 2);
    assert_eq!(loaded.memory.write_head, 0);
    assert_eq!(loaded.memory.insert(&[0.3, 0.1], &[30.; 4]), 0);
    assert_eq!(&loaded.memory.values[..4], &[30.; 4]);
    fs::remove_file(p).unwrap();
}

#[test]
fn valid_large_tape_checkpoints_are_not_rejected_by_a_payload_ratio() {
    let p = path("large-tape");
    let mut cfg = model().cfg;
    cfg.chunk_len = 4096;
    let m = PSSALayerV2::new(cfg, 7);
    for version in [5, 6, 7] {
        match version {
            5 => fs::write(&p, old(&m)).unwrap(),
            6 => checkpoint::save_model_v6(&m, &p).unwrap(),
            _ => checkpoint::save_model(&m, &p).unwrap(),
        }
        let loaded = checkpoint::load_checkpoint(&p).unwrap().model;
        assert_eq!(loaded.cfg.chunk_len, 4096);
        assert_eq!(loaded.embed_w.data, m.embed_w.data);
    }
    fs::remove_file(p).unwrap();
}

#[test]
fn legacy_inside_ball_keys_survive_norm_rounding_and_all_checkpoint_versions() {
    let p = path("legacy-near-boundary");
    let mut cfg = model().cfg;
    cfg.d_mem_key = 3;
    let mut m = PSSALayerV2::new(cfg, 7);
    // Produced by the old f32 projection; its true squared norm is below 1,
    // but converting that f64 norm back to f32 rounds it to exactly 1.
    let key = [
        -0.5546702146530151f32,
        0.7364687323570251,
        0.38723990321159363,
    ];
    let norm: f64 = key.iter().map(|&x| (x as f64).powi(2)).sum();
    assert!(norm < 1.0);
    assert_eq!(norm as f32, 1.0);
    m.memory.keys[..3].copy_from_slice(&key);
    m.memory.norm_sq[0] = f32::from_bits(1.0f32.to_bits() - 1);
    m.memory.values[..4].fill(2.0);
    m.memory.count = 1;
    for version in [5, 6, 7] {
        match version {
            5 => fs::write(&p, old(&m)).unwrap(),
            6 => checkpoint::save_model_v6(&m, &p).unwrap(),
            _ => checkpoint::save_model(&m, &p).unwrap(),
        }
        let loaded = checkpoint::load_checkpoint(&p).unwrap().model;
        assert_eq!(&loaded.memory.keys[..3], &key);
        assert!(loaded.memory.norm_sq[0] < 1.0);
        let mut out = [0.0; 4];
        loaded
            .memory
            .retrieve_soft_into(&key, 0.7, &mut out, &mut [0.0; 3]);
        assert_eq!(out, [2.0; 4]);
    }
    fs::remove_file(p).unwrap();
}

fn persistent_matrix_mut(m: &mut PSSALayerV2, index: usize) -> &mut ParamMatrix {
    match index {
        0 => &mut m.embed_w,
        1 => &mut m.a_mat,
        2 => &mut m.w_delta,
        3 => &mut m.w_b,
        4 => &mut m.w_c,
        5 => &mut m.w_qx,
        6 => &mut m.w_qh,
        7 => &mut m.w_gate,
        8 => &mut m.w_proj,
        9 => &mut m.mlp_w1,
        10 => &mut m.mlp_w2,
        11 => &mut m.unembed_w,
        12 => &mut m.adapters[0].down_proj,
        13 => &mut m.adapters[0].up_proj,
        _ => panic!("persistent matrix index out of range: {index}"),
    }
}

fn assert_bad_save_preserves(m: &PSSALayerV2, p: &PathBuf, good: &[u8], case: &str) {
    assert!(checkpoint::save_model(m, p).is_err(), "V7 accepted {case}");
    assert_eq!(
        fs::read(p).unwrap(),
        good,
        "V7 replaced old checkpoint for {case}"
    );
    assert!(
        checkpoint::save_model_v6(m, p).is_err(),
        "V6 accepted {case}"
    );
    assert_eq!(
        fs::read(p).unwrap(),
        good,
        "V6 replaced old checkpoint for {case}"
    );
}

#[test]
fn malformed_persistent_tensor_shapes_preserve_old_checkpoint() {
    let p = path("bad-persistent-shapes");
    checkpoint::save_model(&model(), &p).unwrap();
    let good = fs::read(&p).unwrap();
    for tensor in 0..14 {
        for corruption in 0..6 {
            let mut m = model();
            let mat = persistent_matrix_mut(&mut m, tensor);
            match corruption {
                // Internally consistent, but not the configuration's tensor.
                0 => *mat = ParamMatrix::zeros(mat.rows + 1, mat.cols),
                // Correct element count, wrong declared row/column dimensions.
                1 => {
                    mat.rows *= mat.cols;
                    mat.cols = 1;
                }
                2 => {
                    mat.data.pop();
                }
                3 => {
                    mat.grad.pop();
                }
                4 => {
                    mat.m.pop();
                }
                5 => {
                    mat.v.pop();
                }
                _ => unreachable!(),
            }
            assert_bad_save_preserves(&m, &p, &good, &format!("matrix {tensor}/{corruption}"));
        }
    }
    for norm in 0..2 {
        for corruption in 0..5 {
            let mut m = model();
            let n = if norm == 0 {
                &mut m.norm_gamma
            } else {
                &mut m.norm_beta
            };
            match corruption {
                0 => *n = ParamVector::new(n.data.len() + 1, 1.0),
                1 => {
                    n.data.pop();
                }
                2 => {
                    n.grad.pop();
                }
                3 => {
                    n.m.pop();
                }
                4 => {
                    n.v.pop();
                }
                _ => unreachable!(),
            }
            assert_bad_save_preserves(&m, &p, &good, &format!("norm {norm}/{corruption}"));
        }
    }
    assert!(checkpoint::load_checkpoint(&p).is_ok());
    fs::remove_file(p).unwrap();
}

#[test]
fn malformed_memory_adapter_and_metadata_preserve_old_checkpoint_without_panics() {
    let p = path("bad-memory-metadata");
    checkpoint::save_model(&model(), &p).unwrap();
    let good = fs::read(&p).unwrap();
    let cases: &[fn(&mut PSSALayerV2)] = &[
        |m| {
            m.memory.keys.pop();
        },
        |m| m.memory.keys.push(0.0),
        |m| {
            m.memory.values.pop();
        },
        |m| m.memory.values.push(0.0),
        |m| {
            m.memory.norm_sq.pop();
        },
        |m| m.memory.norm_sq.push(0.0),
        |m| {
            m.memory.confidence.pop();
        },
        |m| m.memory.confidence.push(1.0),
        |m| {
            m.memory.last_seen_step.pop();
        },
        |m| m.memory.last_seen_step.push(0),
        |m| m.memory.capacity += 1,
        |m| m.memory.dim_key += 1,
        |m| m.memory.dim_val += 1,
        |m| m.memory.count = m.memory.capacity + 1,
        |m| m.memory.write_head = m.memory.capacity,
        |m| {
            m.memory.count = 1;
            m.memory.write_head = 1;
        },
        |m| m.memory.keys[0] = f32::NAN,
        |m| m.memory.values[0] = f32::INFINITY,
        |m| m.memory.norm_sq[0] = -1.0,
        |m| m.memory.norm_sq[0] = 1.0,
        |m| m.memory.norm_sq[0] += 0.1,
        |m| m.memory.confidence[0] = -1.0,
        |m| m.memory.last_seen_step[0] = m.step_counter + 1,
        |m| {
            m.memory.count = 0;
            m.memory.write_head = 0;
            m.memory.norm_sq[2] = -1.0;
        },
        |m| {
            m.memory.count = 0;
            m.memory.write_head = 0;
            m.memory.confidence.clear();
        },
        |m| m.adapters.clear(),
        |m| {
            let adapter = m.adapters[0].clone();
            m.adapters.push(adapter);
        },
        |m| m.adapters[0].rank += 1,
        |m| m.adapters[0].d_latent += 1,
        |m| {
            m.adapters[0].consolidated_up.pop();
        },
        |m| m.adapters[0].consolidated_up.push(0.0),
        |m| m.adapters[0].consolidated_up[0] = f32::NAN,
        |m| {
            m.h_persistent.pop();
        },
        |m| m.h_persistent.push(0.0),
        |m| m.h_persistent[0] = f32::INFINITY,
        |m| {
            m.embed_row_marks.pop();
        },
        |m| m.embed_row_marks.push(0),
        |m| m.step_counter = usize::MAX,
    ];
    for (i, corrupt) in cases.iter().enumerate() {
        let mut m = model();
        corrupt(&mut m);
        assert_bad_save_preserves(&m, &p, &good, &format!("metadata {i}"));
    }
    assert!(checkpoint::load_checkpoint(&p).is_ok());
    fs::remove_file(p).unwrap();
}

fn vector_bytes<T>(v: &Vec<T>) -> usize {
    // Constructors should allocate exactly their numeric backing storage.
    assert_eq!(v.len(), v.capacity());
    v.len() * std::mem::size_of::<T>()
}

fn matrix_bytes(p: &ParamMatrix) -> usize {
    [&p.data, &p.grad, &p.m, &p.v]
        .into_iter()
        .map(vector_bytes)
        .sum()
}

fn actual_numeric_storage(m: &PSSALayerV2) -> usize {
    // Classify every accessible model/tape buffer here and in allocation_bytes.
    // The private dream cache is runtime-only and intentionally excluded.
    let PSSALayerV2 {
        cfg: _,
        step_counter: _,
        device: _,
        scan_executor: _, // Runtime scheduler, no numeric/checkpoint storage.
        rng: _,
        vocabulary: _,
        tokenizer_json: _,
        lr_schedule_total_updates: _,
        lr_schedule_warmup_steps: _,
        embed_w,
        unembed_w,
        embed_row_marks,
        block,
        extra_blocks,
        residual_scales,
        continuous_inputs,
        output_adjoints,
        input_adjoints,
        residual_block_adjoints,
        residual_input_adjoints,
        boundary_adjoints,
        layer_activations,
        inf_features,
        inf_block_out,
        ..
    } = m;
    matrix_bytes(embed_w)
        + matrix_bytes(unembed_w)
        + vector_bytes(embed_row_marks)
        + actual_block_storage(block)
        + extra_blocks.iter().map(actual_block_storage).sum::<usize>()
        + [
            residual_scales,
            continuous_inputs,
            output_adjoints,
            input_adjoints,
            residual_block_adjoints,
            residual_input_adjoints,
            inf_features,
            inf_block_out,
        ]
        .into_iter()
        .map(vector_bytes)
        .sum::<usize>()
        + boundary_adjoints
            .iter()
            .chain(layer_activations)
            .map(vector_bytes)
            .sum::<usize>()
}

fn actual_block_storage(b: &pssa::pssa::PSSAContinuousBlockV2) -> usize {
    let pssa::pssa::PSSAContinuousBlockV2 {
        cfg: _,
        norm_gamma,
        norm_beta,
        a_mat,
        w_delta,
        w_b,
        w_c,
        h_persistent,
        ssm_raw_snapshot,
        ssm_rates,
        ssm_rate_derivatives,
        w_qx,
        w_qh,
        w_gate,
        w_proj,
        memory,
        adapters,
        adapter_up_effective,
        mlp_w1,
        mlp_w2,
        tape,
        grad_h_next,
        grad_z_final,
        grad_z_raw,
        grad_x_norm,
        buf_m_proj,
        buf_ad_out,
        buf_g_mlp_act,
        buf_g_mlp_hidden,
        buf_g_zraw_mlp,
        buf_g_ad_act,
        buf_g_ad_down,
        buf_g_m_proj_out,
        buf_g_m_val,
        g_query_pnc,
        g_query_euc,
        g_y_ssm,
        buf_g_delta,
        buf_g_b_proj,
        buf_g_c_proj,
        buf_g_h_prev,
        bwd_g_zfinal,
        bwd_g_zraw,
        bwd_g_ad_down,
        bwd_g_xnorm,
        bwd_g_ysm,
        bwd_g_query_euc,
        bwd_g_logits,
        bwd_g_mlp,
        inf_x_norm,
        inf_delta,
        inf_b,
        inf_c,
        inf_y_ssm,
        inf_q_euc,
        inf_q_pnc,
        inf_mem_weights,
        inf_m_val,
        inf_g_mem,
        inf_m_proj,
        inf_ad_act,
        inf_ad_out,
        inf_z_raw,
        inf_mlp_act,
        inf_mlp_out,
        inf_z_final,
        ssm_scan_a,
        ssm_scan_b,
        bwd_ssm_delta,
        bwd_ssm_b,
        bwd_ssm_c,
        bwd_g_query_pnc,
        bwd_ssm_a,
    } = b;
    let params: usize = [
        a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj, mlp_w1, mlp_w2,
    ]
    .into_iter()
    .map(matrix_bytes)
    .sum();
    let norms: usize = [norm_gamma, norm_beta]
        .into_iter()
        .map(|p| {
            [&p.data, &p.grad, &p.m, &p.v]
                .into_iter()
                .map(vector_bytes)
                .sum::<usize>()
        })
        .sum();
    let adapter_bytes: usize = adapters
        .iter()
        .map(|a| {
            matrix_bytes(&a.down_proj) + matrix_bytes(&a.up_proj) + vector_bytes(&a.consolidated_up)
        })
        .sum();
    let float_bytes: usize = [
        h_persistent,
        adapter_up_effective,
        ssm_raw_snapshot,
        ssm_rates,
        ssm_rate_derivatives,
        grad_h_next,
        grad_z_final,
        grad_z_raw,
        grad_x_norm,
        buf_m_proj,
        buf_ad_out,
        buf_g_mlp_act,
        buf_g_mlp_hidden,
        buf_g_zraw_mlp,
        buf_g_ad_act,
        buf_g_ad_down,
        buf_g_m_proj_out,
        buf_g_m_val,
        g_query_pnc,
        g_query_euc,
        g_y_ssm,
        buf_g_delta,
        buf_g_b_proj,
        buf_g_c_proj,
        buf_g_h_prev,
        bwd_g_zfinal,
        bwd_g_zraw,
        bwd_g_ad_down,
        bwd_g_xnorm,
        bwd_g_ysm,
        bwd_g_query_euc,
        bwd_g_logits,
        bwd_g_mlp,
        inf_x_norm,
        inf_delta,
        inf_b,
        inf_c,
        inf_y_ssm,
        inf_q_euc,
        inf_q_pnc,
        inf_mem_weights,
        inf_m_val,
        inf_g_mem,
        inf_m_proj,
        inf_ad_act,
        inf_ad_out,
        inf_z_raw,
        inf_mlp_act,
        inf_mlp_out,
        inf_z_final,
        ssm_scan_a,
        ssm_scan_b,
        bwd_ssm_delta,
        bwd_ssm_b,
        bwd_ssm_c,
        bwd_ssm_a,
    ]
    .into_iter()
    .map(vector_bytes)
    .sum::<usize>()
        + vector_bytes(bwd_g_query_pnc);
    let pssa::memory::HyperbolicEpisodicBankV2 {
        capacity: _,
        count: _,
        dim_key: _,
        dim_val: _,
        write_head: _,
        keys,
        values,
        value_cap: _, // runtime-only; not an allocated/serialized buffer
        norm_sq,
        confidence,
        last_seen_step,
    } = memory;
    let memory_bytes = [keys, values, norm_sq, confidence]
        .into_iter()
        .map(vector_bytes)
        .sum::<usize>()
        + vector_bytes(last_seen_step);
    let pssa::pssa::ChunkActivationTape {
        max_l: _,
        x_ids,
        target_ids,
        x_raw,
        x_norm,
        inv_rms,
        delta_raw,
        delta,
        b_proj,
        c_proj,
        bar_a,
        bar_b,
        h_states,
        y_ssm,
        q_euc,
        q_norm,
        q_poincare,
        mem_weights,
        m_val,
        g_mem,
        m_inj,
        m_proj,
        adapter_hidden,
        adapter_act,
        z_raw,
        mlp_hidden,
        mlp_act,
        z_final,
        logits,
        probs,
        losses,
    } = tape;
    let tape_bytes = [
        x_raw,
        x_norm,
        inv_rms,
        delta_raw,
        delta,
        b_proj,
        c_proj,
        bar_a,
        bar_b,
        h_states,
        y_ssm,
        q_euc,
        q_norm,
        q_poincare,
        mem_weights,
        m_val,
        g_mem,
        m_inj,
        m_proj,
        adapter_hidden,
        adapter_act,
        z_raw,
        mlp_hidden,
        mlp_act,
        z_final,
        logits,
        probs,
        losses,
    ]
    .into_iter()
    .map(vector_bytes)
    .sum::<usize>()
        + vector_bytes(x_ids)
        + vector_bytes(target_ids);
    params + norms + adapter_bytes + float_bytes + memory_bytes + tape_bytes
}

#[test]
fn large_projected_memory_keys_use_the_banks_norm_policy_on_save_and_load() {
    use pssa::memory::HyperbolicEpisodicBankV2;
    let p = path("high-dimensional-memory");
    let mut m = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 2,
            d_latent: 1,
            d_state: 1,
            d_mem_key: 10_000,
            mem_capacity: 1,
            chunk_len: 1,
            ..Default::default()
        },
        42,
    );
    let mut key = vec![0.0; m.cfg.d_mem_key];
    HyperbolicEpisodicBankV2::diffeomorphic_project(&vec![1e20; m.cfg.d_mem_key], &mut key);
    assert!(HyperbolicEpisodicBankV2::squared_norm(&key) < 1.0);
    m.memory.insert(&key, &[1.0]);
    checkpoint::save_model(&m, &p).unwrap();
    let copy = checkpoint::load_checkpoint(&p).unwrap().model;
    state(&m, &copy);
    fs::remove_file(p).unwrap();
}

#[test]
fn checked_allocation_accounting_matches_actual_model_vectors() {
    for (v, m, s, k, cap, chunk) in [
        (5, 4, 2, 2, 3, 2),
        (7, 3, 5, 4, 2, 7),
        (257, 256, 1, 1, 1, 64),
    ] {
        let cfg = PSSAConfigV2 {
            d_vocab: v,
            d_latent: m,
            d_state: s,
            d_mem_key: k,
            mem_capacity: cap,
            chunk_len: chunk,
            ..Default::default()
        };
        for depth in [1, 4] {
            let cfg = PSSAConfigV2 {
                depth,
                ..cfg.clone()
            };
            let expected = checkpoint::allocation_bytes(&cfg).unwrap();
            let actual = PSSALayerV2::new(cfg, 42);
            assert_eq!(
                expected,
                actual_numeric_storage(&actual),
                "{:?}",
                actual.cfg
            );
        }
    }
}

#[test]
fn allocation_guard_rejects_omitted_backward_buffer_case_and_overflows() {
    let cfg = PSSAConfigV2 {
        d_vocab: 257,
        d_latent: 256,
        d_state: 1,
        d_mem_key: 1,
        mem_capacity: 1,
        chunk_len: 50_000,
        ..Default::default()
    };
    // The old guard accepted about 946 MiB despite >1 GiB of actual storage.
    assert!(
        checkpoint::allocation_bytes(&cfg)
            .unwrap_err()
            .to_string()
            .contains("byte cap")
    );
    assert!(checkpoint::validate_model_config(&cfg).is_err());
    for field in 0..6 {
        let mut c = model().cfg;
        match field {
            0 => c.d_vocab = usize::MAX,
            1 => c.d_latent = usize::MAX,
            2 => c.d_state = usize::MAX,
            3 => c.d_mem_key = usize::MAX,
            4 => c.mem_capacity = usize::MAX,
            5 => c.chunk_len = usize::MAX,
            _ => unreachable!(),
        }
        assert!(checkpoint::allocation_bytes(&c).is_err(), "field {field}");
        assert!(
            checkpoint::validate_model_config(&c).is_err(),
            "field {field}"
        );
    }
}
