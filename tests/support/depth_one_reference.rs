//! Deterministic depth-one fixture recipe, also compiled against unmodified
//! main 85d9d337875d58c709d5e32703d2f43825185f8e to generate the golden V7 files.
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

pub fn model() -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 7,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 3,
            tau_mem: 0.8,
            lr: 0.0003,
            ..Default::default()
        },
        192,
    );
    m.vocabulary = ["<unk>", "one", "two", "three", "four", "five", "six"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    m.lr_schedule_total_updates = Some(100);
    m.lr_schedule_warmup_steps = Some(4);
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = (i as f32 * 0.23).sin() * 0.07;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = (i as f32 * 0.17).cos() * 0.03;
    }
    for (i, x) in m.adapters[0].consolidated_up.iter_mut().enumerate() {
        *x = (i as f32 * 0.31).sin() * 0.02;
    }
    m.memory.insert(&[0.1, -0.2], &[0.3, -0.2, 0.1, 0.5]);
    m.memory.insert(&[-0.2, 0.15], &[-0.1, 0.4, -0.5, 0.2]);
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = i as f32 * 0.005 - 0.01;
    }
    m
}

pub fn advance(m: &mut PSSALayerV2) -> Vec<u32> {
    let mut observed = Vec::new();
    for _ in 0..2 {
        m.zero_gradients();
        observed.push(m.forward_train_chunk(&[1, 2, 1], &[2, 1, 3]).to_bits());
        observed.extend(m.tape.logits.iter().map(|v| v.to_bits()));
        m.backward_chunk(3, 0.75);
        observed.push(m.forward_train_chunk(&[4], &[5]).to_bits());
        m.backward_chunk(1, 0.25);
        m.apply_adamw(0.0003);
        m.ema_consolidate_plasticity();
    }
    let mut logits = vec![0.0; 7];
    for id in [1, 3, 2] {
        m.forward_inference(id, &mut logits);
        observed.extend(logits.iter().map(|v| v.to_bits()));
    }
    observed
}
