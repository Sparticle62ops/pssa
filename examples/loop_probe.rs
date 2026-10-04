use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
fn main() {
    for loops in [1, 2, 4] {
        let cfg = PSSAConfigV2 {
            d_vocab: 2048,
            d_latent: 256,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            chunk_len: 64,
            ..Default::default()
        };
        let mut m = PSSALayerV2::new_with_depth_and_loops(cfg, 42, 1, loops);
        let ids: Vec<usize> = (0..64).map(|i| i % 2048).collect();
        let targets: Vec<usize> = (0..64).map(|i| (i + 1) % 2048).collect();
        let start = std::time::Instant::now();
        let loss = m.forward_train_chunk(&ids, &targets);
        let forward = start.elapsed().as_secs_f64();
        let z = if loops == 1 {
            &m.block.tape.z_final[..64 * 256]
        } else {
            &m.continuous_inputs[..64 * 256]
        };
        for loop_index in 0..loops {
            let off = loop_index * m.block.tape.max_l * 256;
            let row = &m.block.tape.z_final[off..off + 64 * 256];
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32).sqrt();
            println!("loops={loops} pass={loop_index} z_rms={rms}");
        }
        let max = z.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let rms = (z.iter().map(|x| x * x).sum::<f32>() / z.len() as f32).sqrt();
        println!("loops={loops} loss={loss} z_rms={rms} z_max={max}");
        m.zero_gradients();
        let start = std::time::Instant::now();
        m.backward_chunk(64, 1.0);
        println!(
            "forward_ms={} backward_ms={}",
            forward * 1000.0,
            start.elapsed().as_secs_f64() * 1000.0
        );
        let gmax = m
            .embed_w
            .grad
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        let brms = (m.embed_w.grad.iter().map(|x| x * x).sum::<f32>()
            / m.embed_w.grad.len() as f32)
            .sqrt();
        println!("loops={loops} grad_embed_rms={brms} grad_embed_max={gmax}");
    }
}
