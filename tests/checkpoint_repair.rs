use oxide_ai_pssa::checkpoint::{self, CheckpointFormat};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2, ParamMatrix, ParamVector};
use std::{fs, path::PathBuf};

fn path(s: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oxide-checkpoint-{s}-{}", std::process::id()))
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
    for (i, p) in [
        &mut x.embed_w,
        &mut x.a_mat,
        &mut x.w_delta,
        &mut x.w_b,
        &mut x.w_c,
        &mut x.w_qx,
        &mut x.w_qh,
        &mut x.w_gate,
        &mut x.w_proj,
        &mut x.mlp_w1,
        &mut x.mlp_w2,
        &mut x.unembed_w,
    ]
    .into_iter()
    .enumerate()
    {
        pm(p, i as f32 + 0.1)
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
    state(&a, &b.model);
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
