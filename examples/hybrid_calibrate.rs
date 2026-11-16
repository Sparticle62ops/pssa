//! Hybrid-read calibration: same-seed PSSA models train on a synthetic
//! associative-recall corpus that repeats keys with re-bound values, so the
//! newest binding is the only correct answer. Arms differ only in the hybrid
//! read toggle (off / lam) and whether lam is length-scaled: scaled arms pass
//! lam^(1/chunk_len) to the read, so a slot decays to lam after one full
//! 32-token chunk instead of per retrieval step. Reports final train loss and
//! held-out loss. Memory-research tool; run locally only.
use pssa::backend::Device;
use pssa::pssa::{PSSAConfigV2, PSSALayerV2};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const KEYS: u64 = 8;
const VAL_BASE: u64 = 32;
const VAL_RANGE: u64 = 16;
const FILLER_BASE: u64 = 48;

/// One 32-token chunk: eight key/value pairs (keys repeat; each rewrite of a
/// key re-binds it to a new value), then eight queries answered by the most
/// recent binding. Returns (inputs, targets) with targets shifted by one.
fn make_chunk(rng: &mut Rng, query_bias_recent: bool) -> (Vec<usize>, Vec<usize>) {
    let mut stream: Vec<u64> = Vec::with_capacity(32);
    let mut version = [0u64; KEYS as usize];
    let mut latest = [0u64; KEYS as usize];
    for _ in 0..8 {
        let k = rng.below(KEYS);
        version[k as usize] += 1;
        let v = VAL_BASE + (k * 5 + version[k as usize]) % VAL_RANGE;
        latest[k as usize] = v;
        stream.push(k);
        stream.push(v);
    }
    for _ in 0..8 {
        let k = if query_bias_recent && rng.below(2) == 0 {
            // query the key rewritten most recently
            let mut best = 0u64;
            let mut best_v = 0u64;
            for (i, &v) in version.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i as u64;
                }
            }
            best
        } else {
            rng.below(KEYS)
        };
        stream.push(k);
        stream.push(latest[k as usize]);
    }
    stream.push(FILLER_BASE + rng.below(VAL_RANGE));
    let inputs: Vec<usize> = stream[..32].iter().map(|&t| t as usize).collect();
    let targets: Vec<usize> = stream[1..33].iter().map(|&t| t as usize).collect();
    (inputs, targets)
}

fn train_arm(
    seed: u64,
    lam: Option<f32>,
    scaled: bool,
    train_chunks: usize,
    eval_chunks: usize,
) -> (f32, f32) {
    let mut model = PSSALayerV2::new_with_device(
        PSSAConfigV2 {
            depth: 2,
            d_vocab: 64,
            d_latent: 64,
            d_state: 16,
            d_mem_key: 64,
            mem_capacity: 256,
            chunk_len: 32,
            tau_mem: 0.1,
            ..Default::default()
        },
        42,
        Device::Cpu,
    );
    // Length-scaled arms translate "decay to lam over one full chunk" into a
    // per-step decay rate, since ages still bump once per retrieval step.
    let lam_eff = lam.map(|l| if scaled { l.powf(1.0 / 32.0) } else { l });
    if let Some(lam_eff) = lam_eff {
        model
            .set_hybrid_read(true, lam_eff)
            .expect("cpu model accepts the hybrid read");
    }
    let mut rng = Rng(seed);
    let mut recent_loss = 0.0f32;
    for step in 0..train_chunks {
        model.reset_recurrent_state();
        let (inputs, targets) = make_chunk(&mut rng, true);
        let loss = model.forward_train_chunk(&inputs, &targets);
        assert!(loss.is_finite(), "non-finite training loss at chunk {step}");
        model.backward_and_step_chunk(inputs.len());
        model.insert_training_memory(loss, inputs.len());
        if step >= train_chunks - 50 {
            recent_loss += loss;
        }
    }
    recent_loss /= 50.0;
    let mut eval_rng = Rng(seed ^ 0xc0ffee);
    let mut eval_loss = 0.0f32;
    for _ in 0..eval_chunks {
        model.reset_recurrent_state();
        let (inputs, targets) = make_chunk(&mut eval_rng, true);
        eval_loss += model.forward_train_chunk(&inputs, &targets);
    }
    eval_loss /= eval_chunks as f32;
    (recent_loss, eval_loss)
}

fn main() {
    const SEEDS: [u64; 4] = [42, 43, 44, 45];
    let arms: [(&str, Option<f32>, bool); 6] = [
        ("hybrid off (baseline)", None, false),
        ("hybrid lam=1.0", Some(1.0), false),
        ("lam=0.99 raw (per step)", Some(0.99), false),
        ("lam=0.99 chunk-scaled", Some(0.99), true),
        ("lam=0.95 chunk-scaled", Some(0.95), true),
        ("lam=0.9 chunk-scaled", Some(0.9), true),
    ];
    let mut rows = Vec::new();
    for (name, lam, scaled) in arms {
        let mut train_losses = Vec::new();
        let mut eval_losses = Vec::new();
        for &seed in &SEEDS {
            let (train_loss, eval_loss) = train_arm(seed, lam, scaled, 300, 200);
            train_losses.push(train_loss);
            eval_losses.push(eval_loss);
            println!("{name:<24} seed {seed}: train(last50) {train_loss:.4}  heldout {eval_loss:.4}");
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let spread = |v: &[f32]| {
            let m = mean(v);
            (v.iter().fold(0.0f32, |acc, &x| acc + (x - m) * (x - m)) / v.len() as f32).sqrt()
        };
        println!(
            "{name:<24} MEAN train(last50) {:.4}±{:.4}  heldout {:.4}±{:.4}",
            mean(&train_losses),
            spread(&train_losses),
            mean(&eval_losses),
            spread(&eval_losses)
        );
        rows.push((
            name.to_string(),
            lam,
            scaled,
            mean(&train_losses),
            spread(&train_losses),
            mean(&eval_losses),
            spread(&eval_losses),
        ));
    }
    let baseline = rows[0].5;
    let mut json = String::from(
        "{\"schema\":3,\"kind\":\"hybrid_read_calibration\",\"seeds\":[42,43,44,45],\"arms\":[",
    );
    for (i, (name, lam, scaled, train_mean, train_sd, eval_mean, eval_sd)) in
        rows.iter().enumerate()
    {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"arm\":\"{name}\",\"lam\":{},\"scaled\":{},\"train_loss_last50_mean\":{train_mean:.6},\"train_loss_last50_sd\":{train_sd:.6},\"heldout_loss_mean\":{eval_mean:.6},\"heldout_loss_sd\":{eval_sd:.6},\"heldout_delta_vs_baseline\":{:.6}}}",
            lam.map(|l| format!("{l}")).unwrap_or_else(|| "null".into()),
            scaled,
            eval_mean - baseline
        ));
    }
    json.push_str("]}\n");
    std::fs::write("/workspace/nightly-build/hybrid_calibrate_results.json", json)
        .expect("write results json");
    println!("wrote /workspace/nightly-build/hybrid_calibrate_results.json");
}
