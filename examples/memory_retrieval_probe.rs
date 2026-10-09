//! Full-occupancy CPU read microbenchmark (not end-to-end training throughput).
//! Run via csrun: cargo run --profile fast --example memory_retrieval_probe
use pssa::{linalg::SimpleRng, memory::HyperbolicEpisodicBankV2 as Bank};
use std::{hint::black_box, time::Instant};
fn main() {
    for capacity in [64, 256, 1024, 4096] {
        let mut rng = SimpleRng::new(17);
        let mut bank = Bank::new(capacity, 16, 64);
        let mut key = [0.0; 16];
        let mut projected = key;
        for _ in 0..capacity {
            for x in &mut key {
                *x = rng.gen_range_f32(-0.5, 0.5);
            }
            Bank::diffeomorphic_project(&key, &mut projected);
            bank.insert(&projected, &[0.5; 64]);
        }
        let (mut output, mut weights) = ([0.0; 64], vec![0.0; capacity]);
        for k in [None, Some(4)] {
            bank.set_top_k(k).unwrap();
            for _ in 0..20 {
                bank.retrieve_soft_into(&projected, 1.0, &mut output, &mut weights);
            }
            let start = Instant::now();
            for _ in 0..1000 {
                bank.retrieve_soft_into(black_box(&projected), 1.0, &mut output, &mut weights);
                black_box(&output);
            }
            println!(
                "capacity={capacity} top_k={k:?} reads=1000 ns_per_read={:.1}",
                start.elapsed().as_nanos() as f64 / 1000.0
            );
        }
    }
}
