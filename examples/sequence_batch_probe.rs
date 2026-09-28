//! Fixed-work CPU training benchmark: eight independent L64 sequences/update.
//! No tokenizer, I/O or model construction is timed. Compare medians, not CI limits.
use oxide_ai_pssa::{gpu_batch, pssa::{PSSAConfigV2, PSSALayerV2}};
use std::{hint::black_box, time::Instant};

fn model() -> PSSALayerV2 {
    let cfg = PSSAConfigV2 { d_vocab: 2048, mem_capacity: 32, lr: 1e-5, ..Default::default() };
    let mut m = PSSALayerV2::new(cfg, 42);
    for entry in 0..32 {
        let key: Vec<_> = (0..32).map(|i| ((entry + i) % 7) as f32 * 0.01).collect();
        let value: Vec<_> = (0..256).map(|i| ((entry + i) % 11) as f32 * 0.01).collect();
        m.memory.insert(&key, &value);
    }
    // Exercise all backward branches rather than the zero-initialized heads.
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() { *x = (i % 7) as f32 * 0.001; }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() { *x = (i % 5) as f32 * 0.001; }
    m
}

fn main() {
    let inputs: Vec<Vec<_>> = (0..8).map(|s| (0..64).map(|t| (s * 97 + t * 13) % 2048).collect()).collect();
    let targets: Vec<Vec<_>> = inputs.iter().map(|x| x.iter().map(|t| (t + 13) % 2048).collect()).collect();
    let mut samples = Vec::new();
    for round in 0..6 {
        let mut m = model();
        let start = Instant::now();
        m.zero_gradients();
        for (x, y) in inputs.iter().zip(&targets) {
            m.reset_recurrent_state();
            black_box(gpu_batch::forward_train_chunk_batched(&mut m, x, y));
            gpu_batch::backward_chunk_batched(&mut m, x.len(), 1.0 / 8.0);
        }
        m.apply_adamw(m.cfg.lr);
        let tps = 512.0 / start.elapsed().as_secs_f64();
        assert!(m.embed_w.data.iter().all(|v| v.is_finite()));
        black_box(&m);
        if round > 0 { samples.push(tps); }
    }
    println!("mode=separate D=256 S=16 V=2048 K=32 memory=32 L=64 sequences=8 threads={} samples={samples:?}", rayon::current_num_threads());
    samples.sort_by(f64::total_cmp);
    println!("median_tokens_per_second={:.3}", samples[2]);
}
