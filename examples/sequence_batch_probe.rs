//! Fixed-work CPU training benchmark: eight independent L64 sequences/update.
//! No tokenizer, I/O or model/workspace construction is timed. Compare medians,
//! not CI limits. Run with --batch-size 1 (serial baseline) or up to 8 lanes.
use oxide_ai_pssa::{
    gpu_batch,
    pssa::{PSSAConfigV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};
use std::{hint::black_box, process::ExitCode, time::Instant};

const SEQUENCES: usize = 8;
const LENGTH: usize = 64;

fn model() -> PSSALayerV2 {
    let cfg = PSSAConfigV2 {
        d_vocab: 2048,
        mem_capacity: 32,
        lr: 1e-5,
        ..Default::default()
    };
    // The constructor chooses CPU; do not auto-detect GPU for this baseline.
    let mut m = PSSALayerV2::new(cfg, 42);
    for entry in 0..32 {
        let key: Vec<_> = (0..32).map(|i| ((entry + i) % 7) as f32 * 0.01).collect();
        let value: Vec<_> = (0..256).map(|i| ((entry + i) % 11) as f32 * 0.01).collect();
        m.memory.insert(&key, &value);
    }
    // Exercise all backward branches rather than the zero-initialized heads.
    for (i, x) in m.mlp_w2.data.iter_mut().enumerate() {
        *x = (i % 7) as f32 * 0.001;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = (i % 5) as f32 * 0.001;
    }
    m
}

fn parse_batch_size(args: &[String]) -> Result<usize, String> {
    match args {
        [] => Ok(1),
        [flag, value] if flag == "--batch-size" => value
            .parse::<usize>()
            .ok()
            .filter(|&n| (1..=SEQUENCES).contains(&n))
            .ok_or_else(|| format!("--batch-size must be an integer from 1 to {SEQUENCES}")),
        _ => Err(format!(
            "usage: sequence_batch_probe [--batch-size <1..{SEQUENCES}>]"
        )),
    }
}

fn run(batch_size: usize) {
    let inputs: Vec<Vec<_>> = (0..SEQUENCES)
        .map(|s| (0..LENGTH).map(|t| (s * 97 + t * 13) % 2048).collect())
        .collect();
    let targets: Vec<Vec<_>> = inputs
        .iter()
        .map(|x| x.iter().map(|t| (t + 13) % 2048).collect())
        .collect();
    let sequences: Vec<_> = inputs
        .iter()
        .zip(&targets)
        .enumerate()
        .map(|(i, (x, y))| Sequence {
            lane: i % batch_size,
            inputs: x,
            targets: y,
            reset: true,
        })
        .collect();
    let mut samples = Vec::new();
    for round in 0..6 {
        let mut m = model();
        let mut batch = (batch_size > 1)
            .then(|| SequenceBatch::new(&mut m, batch_size).expect("valid benchmark workspace"));
        let start = Instant::now();
        m.zero_gradients();
        for microbatch in sequences.chunks(batch_size) {
            if let Some(batch) = &mut batch {
                black_box(batch.forward(&mut m, microbatch).unwrap());
                batch
                    .backward(&mut m, microbatch.len() as f32 / SEQUENCES as f32)
                    .unwrap();
            } else {
                let seq = &microbatch[0];
                m.reset_recurrent_state();
                black_box(gpu_batch::forward_train_chunk_batched(
                    &mut m,
                    seq.inputs,
                    seq.targets,
                ));
                gpu_batch::backward_chunk_batched(&mut m, LENGTH, 1.0 / SEQUENCES as f32);
            }
        }
        m.apply_adamw(m.cfg.lr);
        let tps = (SEQUENCES * LENGTH) as f64 / start.elapsed().as_secs_f64();
        assert!(m.embed_w.data.iter().all(|v| v.is_finite()));
        black_box(&m);
        if round > 0 {
            samples.push(tps);
        }
    }
    let mode = if batch_size == 1 {
        "separate"
    } else {
        "packed"
    };
    println!(
        "mode={mode} backend=cpu batch_size={batch_size} D=256 S=16 V=2048 K=32 memory=32 L={LENGTH} sequences={SEQUENCES} tokens_per_update={} microbatches_per_update={} threads={} samples={samples:?}",
        SEQUENCES * LENGTH,
        SEQUENCES.div_ceil(batch_size),
        rayon::current_num_threads()
    );
    samples.sort_by(f64::total_cmp);
    println!("median_tokens_per_second={:.3}", samples[samples.len() / 2]);
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!("usage: sequence_batch_probe [--batch-size <1..{SEQUENCES}>] (default: 1)");
        return ExitCode::SUCCESS;
    }
    match parse_batch_size(&args) {
        Ok(batch_size) => {
            run(batch_size);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_explicit_batch_sizes() {
        assert_eq!(parse_batch_size(&[]).unwrap(), 1);
        for n in 1..=SEQUENCES {
            assert_eq!(
                parse_batch_size(&["--batch-size".into(), n.to_string()]).unwrap(),
                n
            );
        }
    }

    #[test]
    fn invalid_arguments_are_errors() {
        for args in [
            vec!["--batch-size"],
            vec!["--batch-size", "0"],
            vec!["--batch-size", "9"],
            vec!["--batch-size", "-1"],
            vec!["--batch-size", "many"],
            vec!["--batch-size", "18446744073709551616"],
            vec!["--unknown", "2"],
            vec!["--batch-size", "2", "extra"],
        ] {
            assert!(
                parse_batch_size(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err()
            );
        }
    }
}
