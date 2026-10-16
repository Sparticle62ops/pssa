//! Benchmark a converted Qwen3->PSSA checkpoint and probe 4B-port scaling.
//!
//! No training. Construction, warmup, and the first token are not timed.
//!
//! cargo run --release --example qwen3_bench -- <checkpoint.v6> [context-tokens]
//! cargo run --release --features cuda --example qwen3_bench -- <checkpoint.v6>
use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::hint::black_box;
use std::time::Instant;

fn rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1).map(|v| v.to_owned()))
        })
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0) / 1024
}

fn bench_model(model: &mut PSSALayerV2, prefill_len: usize, decode_len: usize) -> (f64, f64, u64) {
    let vocab = model.cfg.d_vocab;
    let mut logits = vec![0.0f32; vocab];
    let ids: Vec<usize> = (0..prefill_len).map(|i| (i * 7 + 3) % vocab).collect();

    // Prefill: repeated chunk-length passes on fresh recurrent state, so the
    // number is per-token wall time the converted model would see live.
    model.reset_recurrent_state();
    let start = Instant::now();
    let mut passes = 0;
    while passes * model.cfg.chunk_len < prefill_len {
        let end = ((passes + 1) * model.cfg.chunk_len).min(prefill_len);
        model
            .prefill_inference(&ids[passes * model.cfg.chunk_len..end], None, || false)
            .expect("prefill failed");
        passes += 1;
    }
    black_box(&logits);
    let prefill_secs = start.elapsed().as_secs_f64();

    // Decode: greedy autoregressive continuation, head and argmax included.
    let start = Instant::now();
    for _ in 0..decode_len {
        model.forward_inference(black_box(ids[0]), black_box(&mut logits));
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        black_box(best);
    }
    let decode_secs = start.elapsed().as_secs_f64();

    (
        prefill_len as f64 / prefill_secs,
        decode_len as f64 / decode_secs,
        rss_mib(),
    )
}

fn scaling_config(chunk_len: usize, mem_capacity: usize, depth: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth,
        d_vocab: 4096,
        d_latent: 256,
        d_state: 16,
        d_mem_key: 32,
        mem_capacity,
        chunk_len,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Option<pssa::backend::Device> {
    match pssa::cuda::CudaContext::init() {
        Ok(ctx) => Some(pssa::backend::Device::Cuda(ctx)),
        Err(error) => {
            println!("cuda init failed: {error}");
            None
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> Option<pssa::backend::Device> {
    None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("rss_mib_at_start {}", rss_mib());

    if let Some(path) = args.first() {
        println!("=== converted checkpoint {path} ===");
        let load_start = Instant::now();
        let mut model = pssa::checkpoint::load_model_v6(path)
            .unwrap_or_else(|error| panic!("load failed: {error}"));
        println!(
            "loaded in {:.1?}: latent {} state {} key {} capacity {} chunk {} depth {} vocab {}",
            load_start.elapsed(),
            model.cfg.d_latent,
            model.cfg.d_state,
            model.cfg.d_mem_key,
            model.cfg.mem_capacity,
            model.cfg.chunk_len,
            model.cfg.depth,
            model.cfg.d_vocab
        );
        let chunk = model.cfg.chunk_len;
        let (prefill_tps, decode_tps, rss) = bench_model(&mut model, chunk, 32);
        println!(
            "cpu prefill {:.1} tok/s, decode {:.1} tok/s, rss {} MiB",
            prefill_tps, decode_tps, rss
        );
        // Bank behavior sanity: feed a second pass and confirm retrieval runs.
        let ids: Vec<usize> = (0..model.cfg.chunk_len).map(|i| (i * 11 + 5) % model.cfg.d_vocab).collect();
        model.reset_recurrent_state();
        model
            .prefill_inference(&ids, None, || false)
            .expect("second prefill failed");
        println!("bank second pass ok");
    }

    println!("=== 4B-port scaling probes (fresh seeded models, latent 256, vocab 4096) ===");
    println!("chunk mem depth device prefill_tok_per_s decode_tok_per_s rss_mib");
    #[allow(unused_mut, unused_variables)]
    let cuda_ctx = cuda_device();
    for chunk in [64usize, 128, 256] {
        for mem in [512usize, 2048] {
            for depth in [1usize, 4] {
                let cfg = scaling_config(chunk, mem, depth);
                let mut model = PSSALayerV2::new(cfg.clone(), 42);
                let (p, d, rss) = bench_model(&mut model, chunk, 16);
                println!("{} {} {} cpu {:.1} {:.1} {}", chunk, mem, depth, p, d, rss);
                #[cfg(feature = "cuda")]
                if let Some(ctx) = &cuda_ctx {
                    let mut gpu = PSSALayerV2::new_with_device(cfg, 42, ctx.clone());
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bench_model(&mut gpu, chunk, 16)
                    })) {
                        Ok((p, d, rss)) => {
                            println!("{} {} {} cuda {:.1} {:.1} {}", chunk, mem, depth, p, d, rss)
                        }
                        Err(_) => println!("{} {} {} cuda FAILED", chunk, mem, depth),
                    }
                }
            }
        }
    }
}
