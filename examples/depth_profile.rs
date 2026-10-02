use oxide_ai_pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::time::Instant;

fn model(depth: usize) -> PSSALayerV2 {
    PSSALayerV2::new(
        PSSAConfigV2 {
            depth,
            d_vocab: 2048,
            d_latent: 256,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            chunk_len: 64,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            tau_mem: 0.1,
            ema_alpha: 0.01,
        },
        42,
    )
}

fn run(depth: usize, staged: bool) {
    let mut m = model(depth);
    let tokens: Vec<usize> = (0..64).map(|i| (i * 37 + 11) % 2048).collect();
    let targets: Vec<usize> = (0..64).map(|i| (i * 71 + 19) % 2048).collect();
    for _ in 0..2 {
        if staged {
            forward_train_chunk_batched(&mut m, &tokens, &targets);
        } else {
            m.forward_train_chunk(&tokens, &targets);
        }
        m.zero_gradients();
        if staged {
            backward_chunk_batched(&mut m, 64, 1.0);
        } else {
            m.backward_chunk(64, 1.0);
        }
        m.apply_adamw(m.cfg.lr);
    }
    let mut forward = 0.0;
    let mut backward = 0.0;
    let mut update = 0.0;
    let rounds = 8;
    for _ in 0..rounds {
        m.reset_recurrent_state();
        let t = Instant::now();
        let loss = if staged {
            forward_train_chunk_batched(&mut m, &tokens, &targets)
        } else {
            m.forward_train_chunk(&tokens, &targets)
        };
        forward += t.elapsed().as_secs_f64();
        m.zero_gradients();
        let t = Instant::now();
        if staged {
            backward_chunk_batched(&mut m, 64, 1.0);
        } else {
            m.backward_chunk(64, 1.0);
        }
        backward += t.elapsed().as_secs_f64();
        let t = Instant::now();
        m.apply_adamw(m.cfg.lr);
        update += t.elapsed().as_secs_f64();
        assert!(loss.is_finite());
    }
    let total = forward + backward + update;
    println!(
        "depth={depth} staged={staged} forward_ms={:.3} backward_ms={:.3} update_ms={:.3} total_ms={:.3} tok_s={:.3}",
        forward * 1000.0 / rounds as f64,
        backward * 1000.0 / rounds as f64,
        update * 1000.0 / rounds as f64,
        total * 1000.0 / rounds as f64,
        rounds as f64 * 64.0 / total
    );
}

fn main() {
    run(1, false);
    run(1, true);
    run(2, false);
    run(4, false);
}
