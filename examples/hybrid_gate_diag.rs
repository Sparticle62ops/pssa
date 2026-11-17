//! Hybrid-read diagnostics: why doesn't the hybrid read beat the plain soft
//! read on the re-binding recall task? Three probes on same-seed trained
//! models. (1) Write-gate spread: the distribution of hybrid_gate scores over
//! live slots after training; a collapsed gate means the hybrid read reduces
//! to the soft read with extra machinery. (2) Read-temperature sweep at eval
//! time only: tau_mem is mutated after training, so training dynamics are
//! unchanged. (3) Sharp read: top_k=1 exact nearest-slot read at eval time.
//! Memory-research tool; run locally only.
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

fn train_model(seed: u64, train_chunks: usize) -> PSSALayerV2 {
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
    model
        .set_hybrid_read(true, 1.0)
        .expect("cpu model accepts the hybrid read");
    let mut rng = Rng(seed);
    for _ in 0..train_chunks {
        model.reset_recurrent_state();
        let (inputs, targets) = make_chunk(&mut rng, true);
        let loss = model.forward_train_chunk(&inputs, &targets);
        assert!(loss.is_finite(), "non-finite training loss");
        model.backward_and_step_chunk(inputs.len());
        model.insert_training_memory(loss, inputs.len());
    }
    model
}

fn eval_pass(model: &mut PSSALayerV2, seed: u64, eval_chunks: usize) -> f32 {
    model.reset_recurrent_state();
    let mut eval_rng = Rng(seed ^ 0xc0ffee);
    let mut eval_loss = 0.0f32;
    for _ in 0..eval_chunks {
        model.reset_recurrent_state();
        let (inputs, targets) = make_chunk(&mut eval_rng, true);
        eval_loss += model.forward_train_chunk(&inputs, &targets);
    }
    eval_loss / eval_chunks as f32
}

fn bank(model: &mut PSSALayerV2) -> &mut pssa::memory::HyperbolicEpisodicBankV2 {
    &mut model.block.memory
}

fn set_top_k_all(model: &mut PSSALayerV2, k: Option<usize>) {
    bank(model).set_top_k(k).expect("set top_k on main block");
    for b in &mut model.extra_blocks {
        b.memory.set_top_k(k).expect("set top_k on extra block");
    }
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    let idx = ((n as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(n - 1)]
}

fn main() {
    const SEEDS: [u64; 4] = [42, 43, 44, 45];
    let taus = [0.05f32, 0.1, 0.2, 0.5];
    let mut rows = Vec::new();

    for &seed in &SEEDS {
        let mut model = train_model(seed, 300);

        // (1) Write-gate and age spread over live slots, after training.
        let n = bank(&mut model).count;
        let mut gates: Vec<f32> = bank(&mut model).hybrid_gate[..n].to_vec();
        gates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ages: Vec<f64> = bank(&mut model).hybrid_age[..n]
            .iter()
            .map(|&a| a as f64)
            .collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let sd = |v: &[f32]| {
            let m = mean(v);
            (v.iter().fold(0.0f32, |acc, &x| acc + (x - m) * (x - m)) / v.len() as f32).sqrt()
        };
        let gate_mean = mean(&gates);
        let gate_sd = sd(&gates);
        let age_mean = ages.iter().sum::<f64>() / ages.len() as f64;
        println!(
            "seed {seed}: slots {n}  gate mean {gate_mean:.4} sd {gate_sd:.4} \
             min {:.4} p25 {:.4} p50 {:.4} p75 {:.4} max {:.4}  age mean {age_mean:.1}",
            gates[0],
            quantile(&gates, 0.25),
            quantile(&gates, 0.50),
            quantile(&gates, 0.75),
            gates[gates.len() - 1],
        );
        rows.push((
            seed,
            gate_mean,
            gate_sd,
            age_mean,
        ));

        // (2) Eval-time tau sweep on the frozen trained model. Baseline tau
        // first so the reference is measured on this exact model.
        let baseline_tau = 0.1f32;
        let mut tau_rows = Vec::new();
        for &tau in std::iter::once(&baseline_tau).chain(taus.iter()) {
            model.cfg.tau_mem = tau;
            for b in &mut model.extra_blocks {
                b.cfg.tau_mem = tau;
            }
            let loss = eval_pass(&mut model, seed, 200);
            tau_rows.push((tau, loss));
            println!("seed {seed} tau {tau:.2}: heldout {loss:.4}");
        }
        model.cfg.tau_mem = baseline_tau;
        for b in &mut model.extra_blocks {
            b.cfg.tau_mem = baseline_tau;
        }

        // (3) Sharp read: top_k=1 at eval time, then restore the soft read.
        set_top_k_all(&mut model, Some(1));
        let sharp_loss = eval_pass(&mut model, seed, 200);
        set_top_k_all(&mut model, None);
        println!("seed {seed} top_k=1: heldout {sharp_loss:.4}");

        rows.last_mut().unwrap();
        // stash tau rows and sharp loss on the last row via side table below
        TAU_RESULTS.with(|cell| cell.borrow_mut().push((seed, tau_rows, sharp_loss)));
    }

    // Aggregate across seeds.
    let mut out = String::from(
        "{\"schema\":1,\"kind\":\"hybrid_gate_diag\",\"seeds\":[42,43,44,45],\"runs\":[",
    );
    for (i, (seed, gm, gsd, am)) in rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let (_, taus_r, sharp) = TAU_RESULTS.with(|cell| cell.borrow()[i].clone());
        let mut tau_json = String::from("[");
        for (j, (tau, loss)) in taus_r.iter().enumerate() {
            if j > 0 {
                tau_json.push(',');
            }
            tau_json.push_str(&format!("{{\"tau\":{tau},\"heldout\":{loss:.6}}}"));
        }
        tau_json.push(']');
        out.push_str(&format!(
            "{{\"seed\":{seed},\"gate_mean\":{gm:.6},\"gate_sd\":{gsd:.6},\"age_mean\":{am:.2},\"tau_sweep\":{tau_json},\"topk1_heldout\":{sharp:.6}}}"
        ));
    }
    out.push_str("]}\n");
    std::fs::write("/workspace/nightly-build/hybrid_gate_diag_results.json", out)
        .expect("write results json");
    println!("wrote /workspace/nightly-build/hybrid_gate_diag_results.json");
}

thread_local! {
    static TAU_RESULTS: std::cell::RefCell<Vec<(u64, Vec<(f32, f32)>, f32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
