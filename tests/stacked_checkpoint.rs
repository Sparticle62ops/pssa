use oxide_ai_pssa::checkpoint::{self, CheckpointFormat};
use oxide_ai_pssa::dataset::Tokenizer;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSAContinuousBlockV2, PSSALayerV2, ParamMatrix};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
struct File(PathBuf);
impl File {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "oxide-stacked-checkpoint-{}-{}.pssa",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for File {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn config(depth: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth,
        d_vocab: 5,
        d_latent: 4,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 3,
        chunk_len: 3,
        ..Default::default()
    }
}
fn fill(xs: &mut [f32], base: f32) {
    for (i, x) in xs.iter_mut().enumerate() {
        *x = base + i as f32 * 0.00001;
    }
}
fn populate(p: &mut ParamMatrix, base: f32) {
    fill(&mut p.data, base);
    fill(&mut p.grad, base * 0.1);
    fill(&mut p.m, base * 0.2);
    fill(&mut p.v, base * 0.3);
}
fn model(depth: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(config(depth), 17);
    m.step_counter = 11;
    m.rng.state = 0x1234_5678_9abc_def0;
    m.lr_schedule_total_updates = Some(100);
    m.lr_schedule_warmup_steps = Some(7);
    m.vocabulary = ["<unk>", "alpha", "beta", "gamma", "delta"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    populate(&mut m.embed_w, 0.02);
    populate(&mut m.unembed_w, 0.03);
    m.embed_row_marks.copy_from_slice(&[3, 4, 5, 6, 7]);
    for (i, b) in std::iter::once(&mut m.block)
        .chain(&mut m.extra_blocks)
        .enumerate()
    {
        let base = 0.01 * (i + 1) as f32;
        for p in [
            &mut b.a_mat,
            &mut b.w_delta,
            &mut b.w_b,
            &mut b.w_c,
            &mut b.w_qx,
            &mut b.w_qh,
            &mut b.w_gate,
            &mut b.w_proj,
            &mut b.mlp_w1,
            &mut b.mlp_w2,
        ] {
            populate(p, base);
        }
        for p in [&mut b.norm_gamma, &mut b.norm_beta] {
            fill(&mut p.data, base + 0.2);
            fill(&mut p.grad, base * 0.1);
            fill(&mut p.m, base * 0.2);
            fill(&mut p.v, base * 0.3);
        }
        let ad = &mut b.adapters[0];
        populate(&mut ad.down_proj, base);
        populate(&mut ad.up_proj, base * 0.5);
        fill(&mut ad.consolidated_up, base * 0.2);
        fill(&mut b.h_persistent, base * 0.3);
        // Full bank and nonzero replacement head, independently populated.
        for j in 0..4 {
            b.memory
                .insert(&[base, 0.01 * (j + 1) as f32], &[base + j as f32 * 0.01; 4]);
        }
        b.memory.confidence.copy_from_slice(&[1.5, 2.0, 2.5]);
        b.memory.last_seen_step.copy_from_slice(&[1, 2, 3]);
    }
    m
}
fn block_state(a: &PSSAContinuousBlockV2, b: &PSSAContinuousBlockV2) {
    for (a, b) in [
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
        (&a.adapters[0].down_proj, &b.adapters[0].down_proj),
        (&a.adapters[0].up_proj, &b.adapters[0].up_proj),
    ] {
        assert_eq!(a, b);
    }
    assert_eq!(a.norm_gamma, b.norm_gamma);
    assert_eq!(a.norm_beta, b.norm_beta);
    assert_eq!(a.adapters[0].consolidated_up, b.adapters[0].consolidated_up);
    assert_eq!(a.h_persistent, b.h_persistent);
    assert_eq!(a.memory.count, b.memory.count);
    assert_eq!(a.memory.write_head, b.memory.write_head);
    assert_eq!(a.memory.keys, b.memory.keys);
    assert_eq!(a.memory.values, b.memory.values);
    assert_eq!(a.memory.norm_sq, b.memory.norm_sq);
    assert_eq!(a.memory.confidence, b.memory.confidence);
    assert_eq!(a.memory.last_seen_step, b.memory.last_seen_step);
}
fn state(a: &PSSALayerV2, b: &PSSALayerV2) {
    assert_eq!(format!("{:?}", a.cfg), format!("{:?}", b.cfg));
    assert_eq!(a.depth(), b.depth());
    assert_eq!(a.residual_scales, b.residual_scales);
    assert_eq!(a.step_counter, b.step_counter);
    assert_eq!(a.rng.state, b.rng.state);
    assert_eq!(a.vocabulary, b.vocabulary);
    assert_eq!(a.tokenizer_json, b.tokenizer_json);
    assert_eq!(a.lr_schedule_total_updates, b.lr_schedule_total_updates);
    assert_eq!(a.lr_schedule_warmup_steps, b.lr_schedule_warmup_steps);
    assert_eq!(a.embed_w, b.embed_w);
    assert_eq!(a.unembed_w, b.unembed_w);
    assert_eq!(a.embed_row_marks, b.embed_row_marks);
    for (a, b) in std::iter::once(&a.block)
        .chain(&a.extra_blocks)
        .zip(std::iter::once(&b.block).chain(&b.extra_blocks))
    {
        block_state(a, b);
    }
}

#[test]
fn depth_four_roundtrips_all_state_and_next_optimizer_update() {
    let p = File::new();
    let q = File::new();
    let mut a = model(4);
    checkpoint::save_model(&a, &p.0).unwrap();
    let loaded = checkpoint::load_checkpoint(&p.0).unwrap();
    assert_eq!(loaded.format, CheckpointFormat::V8);
    let mut b = loaded.model;
    state(&a, &b);
    assert_eq!(b.residual_scales, vec![0.5; 3]);
    checkpoint::save_model(&b, &q.0).unwrap();
    assert_eq!(fs::read(&p.0).unwrap(), fs::read(&q.0).unwrap());

    let (mut la, mut lb) = (vec![0.0; 5], vec![0.0; 5]);
    a.forward_inference(1, &mut la);
    b.forward_inference(1, &mut lb);
    assert_eq!(la, lb);
    state(&a, &b);
    a.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
    b.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
    assert_eq!(a.tape.logits, b.tape.logits);
    a.backward_chunk(3, 0.75);
    b.backward_chunk(3, 0.75);
    state(&a, &b);
    a.apply_adamw(0.0005);
    b.apply_adamw(0.0005);
    assert_eq!(a.step_counter, 12);
    state(&a, &b);
}

fn checksum(bytes: &mut [u8]) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in &bytes[22..] {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    bytes[14..22].copy_from_slice(&h.to_le_bytes());
}
fn set_u64(bytes: &mut [u8], offset: usize, n: u64) {
    bytes[offset..offset + 8].copy_from_slice(&n.to_le_bytes());
}
fn word(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}
fn error_for(bytes: &[u8]) -> String {
    let p = File::new();
    fs::write(&p.0, bytes).unwrap();
    match checkpoint::load_checkpoint(&p.0) {
        Ok(_) => panic!("corrupt checkpoint unexpectedly loaded"),
        Err(e) => e.to_string(),
    }
}
// Offsets in the documented V8 prefix (the legacy config is six u64 + seven f32).
const DEPTH: usize = 22 + 6 * 8 + 7 * 4;
const SCALE_COUNT: usize = DEPTH + 8;
const SCALES: usize = SCALE_COUNT + 8;
const STEP: usize = SCALES + 3 * 4;
fn tensors_offset(bytes: &[u8]) -> usize {
    let mut at = STEP + 16;
    let count = word(bytes, at);
    at += 8;
    for _ in 0..count {
        at += 8 + word(bytes, at);
    }
    at
}
fn skip_floats(bytes: &[u8], at: &mut usize) {
    *at += 8 + 4 * word(bytes, *at);
}
fn block_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut at = tensors_offset(bytes);
    for _ in 0..8 {
        skip_floats(bytes, &mut at);
    } // two shared matrices
    let mut out = Vec::new();
    for _ in 0..4 {
        out.push(at);
        for _ in 0..48 {
            skip_floats(bytes, &mut at);
        } // 10 matrices + 2 vectors
        skip_floats(bytes, &mut at); // recurrent carry
        at += 16; // memory count/head
        for _ in 0..4 {
            skip_floats(bytes, &mut at);
        }
        at += 8 + 8 * word(bytes, at); // memory timestamps
        for _ in 0..9 {
            skip_floats(bytes, &mut at);
        } // adapter matrices + slow
    }
    out
}

#[test]
fn v8_rejects_corrupt_depth_scales_shapes_state_and_schedule() {
    let p = File::new();
    checkpoint::save_model(&model(4), &p.0).unwrap();
    let good = fs::read(&p.0).unwrap();
    for n in [0, 1, 33, u64::MAX] {
        let mut bad = good.clone();
        set_u64(&mut bad, DEPTH, n);
        checksum(&mut bad);
        assert!(error_for(&bad).contains("depth"));
    }
    let mut bad = good.clone();
    set_u64(&mut bad, SCALE_COUNT, 4);
    checksum(&mut bad);
    assert!(error_for(&bad).contains("residual scales length"));
    for x in [0.0f32, -0.5, 1.0, f32::NAN, f32::INFINITY] {
        let mut bad = good.clone();
        bad[SCALES + 4..SCALES + 8].copy_from_slice(&x.to_le_bytes());
        checksum(&mut bad);
        assert!(error_for(&bad).contains("residual scale"));
    }
    let mut bad = good.clone();
    set_u64(&mut bad, STEP, usize::MAX as u64);
    checksum(&mut bad);
    assert!(error_for(&bad).contains("exhausted"));
    for at in std::iter::once(tensors_offset(&good)).chain(block_offsets(&good)) {
        let mut bad = good.clone();
        set_u64(&mut bad, at, 1);
        checksum(&mut bad);
        assert!(error_for(&bad).contains("length"));
    }
    for at in block_offsets(&good) {
        // Each matrix has four identically sized arrays; corrupt a second moment.
        let v_word = at + 3 * (8 + 4 * word(&good, at)) + 8;
        let mut bad = good.clone();
        bad[v_word..v_word + 4].copy_from_slice(&(-1.0f32).to_le_bytes());
        checksum(&mut bad);
        assert!(error_for(&bad).contains("negative a_mat_raw.v"));
        let mut carry = at;
        for _ in 0..48 {
            skip_floats(&good, &mut carry);
        }
        let mut bad = good.clone();
        bad[carry + 8..carry + 12].copy_from_slice(&f32::NAN.to_le_bytes());
        checksum(&mut bad);
        assert!(error_for(&bad).contains("h_persistent"));
    }
    for (horizon, warmup) in [(0, 0), (10, 1), (100, 100), (100, 101)] {
        let mut bad = good.clone();
        let n = bad.len();
        set_u64(&mut bad, n - 16, horizon);
        set_u64(&mut bad, n - 8, warmup);
        checksum(&mut bad);
        assert!(error_for(&bad).contains("schedule"));
    }
    let mut bad = good.clone();
    bad.extend_from_slice(&0u64.to_le_bytes());
    let payload_len = (bad.len() - 22) as u64;
    set_u64(&mut bad, 6, payload_len);
    checksum(&mut bad);
    assert!(error_for(&bad).contains("metadata tail"));
    let mut bad = good;
    bad[SCALES] ^= 1;
    assert!(error_for(&bad).contains("checksum"));
}

#[test]
fn v8_validates_backing_before_construction_and_allocation_including_endpoint_tapes() {
    for depth in [0, 33, usize::MAX] {
        assert!(checkpoint::validate_model_config(&config(depth)).is_err());
    }
    let mut cfg = config(4);
    cfg.d_vocab = 10_000;
    cfg.chunk_len = 100_000;
    assert!(
        checkpoint::allocation_bytes(&cfg)
            .unwrap_err()
            .to_string()
            .contains("byte cap")
    );
    cfg = config(32);
    cfg.d_latent = 1024;
    assert!(checkpoint::validate_model_config(&cfg).is_err());
    let p = File::new();
    checkpoint::save_model(&model(4), &p.0).unwrap();
    let mut bytes = fs::read(&p.0).unwrap();
    // Keep the configuration/header but remove all backing tensors. Increase
    // chunk length: valid tape size is not a payload*128 heuristic.
    let tensors = tensors_offset(&bytes);
    bytes.truncate(tensors);
    set_u64(&mut bytes, 22 + 5 * 8, 10_000);
    let payload_len = (bytes.len() - 22) as u64;
    set_u64(&mut bytes, 6, payload_len);
    checksum(&mut bytes);
    assert!(error_for(&bytes).contains("truncated persistent tensors"));
}

#[test]
fn malformed_public_stacked_storage_cannot_replace_checkpoint_or_panic() {
    let p = File::new();
    checkpoint::save_model(&model(4), &p.0).unwrap();
    let good = fs::read(&p.0).unwrap();
    let corruptions: Vec<Box<dyn Fn(&mut PSSALayerV2)>> = vec![
        Box::new(|m| m.cfg.depth = 3),
        Box::new(|m| {
            m.extra_blocks.pop();
        }),
        Box::new(|m| m.residual_scales.clear()),
        Box::new(|m| m.residual_scales[0] = 0.75),
        Box::new(|m| m.extra_blocks[2].cfg.d_latent += 1),
        Box::new(|m| m.extra_blocks[1].cfg.tau_mem *= 2.0),
        Box::new(|m| m.extra_blocks[0].w_qx.rows += 1),
        Box::new(|m| m.extra_blocks[1].w_b.grad.clear()),
        Box::new(|m| m.extra_blocks[2].norm_beta.m.clear()),
        Box::new(|m| m.extra_blocks[0].adapters.clear()),
        Box::new(|m| m.extra_blocks[1].adapters[0].consolidated_up.clear()),
        Box::new(|m| m.extra_blocks[2].h_persistent.clear()),
        Box::new(|m| m.extra_blocks[0].memory.keys.clear()),
        Box::new(|m| m.extra_blocks[1].memory.confidence.clear()),
        Box::new(|m| m.extra_blocks[2].memory.last_seen_step.clear()),
        Box::new(|m| m.extra_blocks[0].memory.values[0] = f32::INFINITY),
        Box::new(|m| m.extra_blocks[1].memory.norm_sq[0] = 0.9),
        Box::new(|m| m.extra_blocks[2].memory.last_seen_step[0] = 12),
        Box::new(|m| m.extra_blocks[0].a_mat.v[0] = -1.0),
        Box::new(|m| m.extra_blocks[1].a_mat.grad[0] = f32::NAN),
        Box::new(|m| m.embed_w.cols += 1),
        Box::new(|m| m.unembed_w.grad.clear()),
        Box::new(|m| m.embed_row_marks.clear()),
        Box::new(|m| m.step_counter = usize::MAX),
        Box::new(|m| m.lr_schedule_total_updates = Some(10)),
        Box::new(|m| m.lr_schedule_warmup_steps = Some(100)),
        Box::new(|m| m.lr_schedule_total_updates = None),
    ];
    for (i, corrupt) in corruptions.into_iter().enumerate() {
        let mut m = model(4);
        corrupt(&mut m);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checkpoint::save_model(&m, &p.0)
        }));
        assert!(result.is_ok(), "corruption {i} panicked");
        assert!(result.unwrap().is_err(), "corruption {i} was accepted");
        assert_eq!(
            fs::read(&p.0).unwrap(),
            good,
            "corruption {i} replaced file"
        );
    }
    let mut m = model(4);
    m.lr_schedule_total_updates = None;
    m.lr_schedule_warmup_steps = None;
    assert!(
        checkpoint::save_model_v6(&m, &p.0)
            .unwrap_err()
            .to_string()
            .contains("stacked")
    );
    assert_eq!(fs::read(&p.0).unwrap(), good);
}

// Independent legacy encoder: this pins the pre-stacking field order and
// proves that adding cfg.depth and block ownership does not alter V6/V7 bytes.
struct LegacyWriter(Vec<u8>);
impl LegacyWriter {
    fn usize(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_le_bytes());
    }
    fn floats(&mut self, xs: &[f32]) {
        self.usize(xs.len());
        for x in xs {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn usizes(&mut self, xs: &[usize]) {
        self.usize(xs.len());
        for &x in xs {
            self.usize(x);
        }
    }
    fn matrix(&mut self, p: &ParamMatrix) {
        for xs in [&p.data, &p.grad, &p.m, &p.v] {
            self.floats(xs);
        }
    }
}
fn legacy_payload(m: &PSSALayerV2) -> Vec<u8> {
    let c = &m.cfg;
    let mut w = LegacyWriter(Vec::new());
    for n in [
        c.d_vocab,
        c.d_latent,
        c.d_state,
        c.d_mem_key,
        c.mem_capacity,
        c.chunk_len,
    ] {
        w.usize(n);
    }
    for x in [
        c.lr,
        c.beta1,
        c.beta2,
        c.weight_decay,
        c.eps,
        c.tau_mem,
        c.ema_alpha,
    ] {
        w.0.extend_from_slice(&x.to_le_bytes());
    }
    w.usize(m.step_counter);
    w.0.extend_from_slice(&m.rng.state.to_le_bytes());
    w.usize(m.vocabulary.len());
    for token in &m.vocabulary {
        w.usize(token.len());
        w.0.extend_from_slice(token.as_bytes());
    }
    for p in [
        &m.embed_w,
        &m.a_mat,
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
        w.matrix(p);
    }
    for p in [&m.norm_gamma, &m.norm_beta] {
        for xs in [&p.data, &p.grad, &p.m, &p.v] {
            w.floats(xs);
        }
    }
    w.floats(&m.h_persistent);
    w.usize(m.memory.count);
    w.usize(m.memory.write_head);
    for xs in [
        &m.memory.keys,
        &m.memory.values,
        &m.memory.norm_sq,
        &m.memory.confidence,
    ] {
        w.floats(xs);
    }
    w.usizes(&m.memory.last_seen_step);
    w.matrix(&m.adapters[0].down_proj);
    w.matrix(&m.adapters[0].up_proj);
    w.floats(&m.adapters[0].consolidated_up);
    w.usizes(&m.embed_row_marks);
    w.0
}
#[test]
fn depth_one_v6_v7_writer_bytes_and_schedule_tails_are_unchanged() {
    let p = File::new();
    let mut m = model(1);
    m.lr_schedule_total_updates = None;
    m.lr_schedule_warmup_steps = None;
    let legacy = legacy_payload(&m);
    checkpoint::save_model_v6(&m, &p.0).unwrap();
    let bytes = fs::read(&p.0).unwrap();
    assert_eq!(&bytes[4..6], &6u16.to_le_bytes());
    assert_eq!(&bytes[22..], legacy);
    let loaded = checkpoint::load_checkpoint(&p.0).unwrap();
    assert_eq!(loaded.model.cfg.depth, 1);
    state(&m, &loaded.model);
    for (horizon, warmup) in [
        (None, None),
        (Some(100), None),
        (Some(100), Some(0)),
        (Some(100), Some(7)),
    ] {
        m.lr_schedule_total_updates = horizon;
        m.lr_schedule_warmup_steps = warmup;
        checkpoint::save_model(&m, &p.0).unwrap();
        let bytes = fs::read(&p.0).unwrap();
        let mut expected = legacy.clone();
        expected.extend_from_slice(&0u64.to_le_bytes()); // no tokenizer JSON
        if let Some(n) = horizon {
            expected.extend_from_slice(&(n as u64).to_le_bytes());
        }
        if let Some(n) = warmup {
            expected.extend_from_slice(&(n as u64).to_le_bytes());
        }
        assert_eq!(&bytes[4..6], &7u16.to_le_bytes());
        assert_eq!(&bytes[22..], expected);
        let loaded = checkpoint::load_checkpoint(&p.0).unwrap();
        assert_eq!(loaded.format, CheckpointFormat::V7);
        state(&m, &loaded.model);
    }
}

fn block_storage(b: &PSSAContinuousBlockV2) -> usize {
    let mut f32s = 0;
    for p in [
        &b.a_mat,
        &b.w_delta,
        &b.w_b,
        &b.w_c,
        &b.w_qx,
        &b.w_qh,
        &b.w_gate,
        &b.w_proj,
        &b.mlp_w1,
        &b.mlp_w2,
        &b.adapters[0].down_proj,
        &b.adapters[0].up_proj,
    ] {
        f32s += p.data.len() + p.grad.len() + p.m.len() + p.v.len();
    }
    for p in [&b.norm_gamma, &b.norm_beta] {
        f32s += p.data.len() + p.grad.len() + p.m.len() + p.v.len();
    }
    let t = &b.tape;
    for xs in [
        &b.h_persistent,
        &b.ssm_raw_snapshot,
        &b.ssm_rates,
        &b.ssm_rate_derivatives,
        &b.adapters[0].consolidated_up,
        &b.memory.keys,
        &b.memory.values,
        &b.memory.norm_sq,
        &b.memory.confidence,
        &t.x_raw,
        &t.x_norm,
        &t.inv_rms,
        &t.delta_raw,
        &t.delta,
        &t.b_proj,
        &t.c_proj,
        &t.bar_a,
        &t.bar_b,
        &t.h_states,
        &t.y_ssm,
        &t.q_euc,
        &t.q_norm,
        &t.q_poincare,
        &t.mem_weights,
        &t.m_val,
        &t.g_mem,
        &t.m_inj,
        &t.m_proj,
        &t.adapter_hidden,
        &t.adapter_act,
        &t.z_raw,
        &t.mlp_hidden,
        &t.mlp_act,
        &t.z_final,
        &t.logits,
        &t.probs,
        &t.losses,
        &b.grad_h_next,
        &b.grad_z_final,
        &b.grad_z_raw,
        &b.grad_x_norm,
        &b.buf_m_proj,
        &b.buf_ad_out,
        &b.buf_g_mlp_act,
        &b.buf_g_mlp_hidden,
        &b.buf_g_zraw_mlp,
        &b.buf_g_ad_act,
        &b.buf_g_ad_down,
        &b.buf_g_m_proj_out,
        &b.buf_g_m_val,
        &b.g_query_pnc,
        &b.g_query_euc,
        &b.g_y_ssm,
        &b.buf_g_delta,
        &b.buf_g_b_proj,
        &b.buf_g_c_proj,
        &b.buf_g_h_prev,
        &b.bwd_g_zfinal,
        &b.bwd_g_zraw,
        &b.bwd_g_ad_down,
        &b.bwd_g_xnorm,
        &b.bwd_g_ysm,
        &b.bwd_g_logits,
        &b.bwd_g_mlp,
        &b.ssm_scan_a,
        &b.ssm_scan_b,
        &b.bwd_ssm_delta,
        &b.bwd_ssm_b,
        &b.bwd_ssm_c,
        &b.bwd_ssm_a,
        &b.inf_x_norm,
        &b.inf_delta,
        &b.inf_b,
        &b.inf_c,
        &b.inf_y_ssm,
        &b.inf_q_euc,
        &b.inf_q_pnc,
        &b.inf_mem_weights,
        &b.inf_m_val,
        &b.inf_g_mem,
        &b.inf_m_proj,
        &b.inf_ad_act,
        &b.inf_ad_out,
        &b.inf_z_raw,
        &b.inf_mlp_act,
        &b.inf_mlp_out,
        &b.inf_z_final,
    ] {
        f32s += xs.len();
    }
    f32s * std::mem::size_of::<f32>()
        + (b.memory.last_seen_step.len() + t.x_ids.len() + t.target_ids.len())
            * std::mem::size_of::<usize>()
}
#[test]
fn stacked_allocation_accounting_matches_actual_numeric_storage() {
    for depth in [1, 2, 4, 32] {
        let cfg = config(depth);
        let expected = checkpoint::allocation_bytes(&cfg).unwrap();
        let m = PSSALayerV2::new(cfg, 17);
        let mut f32s = 0;
        for p in [&m.embed_w, &m.unembed_w] {
            f32s += p.data.len() + p.grad.len() + p.m.len() + p.v.len();
        }
        for xs in [
            &m.residual_scales,
            &m.continuous_inputs,
            &m.output_adjoints,
            &m.input_adjoints,
            &m.residual_block_adjoints,
            &m.residual_input_adjoints,
            &m.inf_features,
            &m.inf_block_out,
        ]
        .into_iter()
        .chain(m.boundary_adjoints.iter())
        .chain(m.layer_activations.iter())
        {
            f32s += xs.len();
        }
        let blocks: usize = std::iter::once(&m.block)
            .chain(&m.extra_blocks)
            .map(block_storage)
            .sum();
        let actual = blocks
            + f32s * std::mem::size_of::<f32>()
            + m.embed_row_marks.len() * std::mem::size_of::<usize>();
        assert_eq!(expected, actual, "depth {depth}");
    }
}

#[test]
fn v8_preserves_bpe_and_all_schedule_tail_variants() {
    let tokenizer = Tokenizer::from_corpus_bpe("alpha beta gamma\nalpha beta\n", 280).unwrap();
    let mut cfg = config(4);
    cfg.d_vocab = tokenizer.vocab_size;
    let mut m = PSSALayerV2::new(cfg, 17);
    m.vocabulary = tokenizer.ordered_vocabulary().unwrap();
    m.tokenizer_json = tokenizer.serialized_metadata();
    let p = File::new();
    for (horizon, warmup) in [
        (None, None),
        (Some(100), None),
        (Some(100), Some(0)),
        (Some(100), Some(7)),
    ] {
        m.lr_schedule_total_updates = horizon;
        m.lr_schedule_warmup_steps = warmup;
        checkpoint::save_model(&m, &p.0).unwrap();
        let loaded = checkpoint::load_checkpoint(&p.0).unwrap();
        assert_eq!(loaded.format, CheckpointFormat::V8);
        state(&m, &loaded.model);
    }
    let good = fs::read(&p.0).unwrap();
    m.vocabulary.swap(1, 2);
    assert!(checkpoint::save_model(&m, &p.0).is_err());
    assert_eq!(fs::read(&p.0).unwrap(), good);
}
