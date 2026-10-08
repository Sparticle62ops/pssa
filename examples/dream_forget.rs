//! Harder forgetting probe: task B reuses task A's input tokens with
//! conflicting targets, so learning B overwrites A. Compares task-A loss
//! after B with and without dream replay over several seeds.
use pssa::{
    dream::DreamMode,
    linalg::SimpleRng,
    pssa::{PSSAConfigV2, PSSALayerV2},
};

const A_IN: [usize; 8] = [1, 2, 1, 2, 1, 2, 1, 2];
const A_TG: [usize; 8] = [2, 1, 2, 1, 2, 1, 2, 1];
const B_IN: [usize; 8] = [1, 2, 1, 2, 1, 2, 1, 2];
const B_TG: [usize; 8] = [3, 4, 3, 4, 3, 4, 3, 4];

fn new_model(seed: u64) -> PSSALayerV2 {
    PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 8,
            d_latent: 8,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 16,
            chunk_len: 8,
            ..Default::default()
        },
        seed,
    )
}
fn update(m: &mut PSSALayerV2, i: &[usize], t: &[usize], lr: f32) {
    m.reset_recurrent_state();
    m.forward_train_chunk(i, t);
    m.zero_gradients();
    m.backward_chunk(i.len(), 1.0);
    m.apply_adamw(lr);
}
fn loss(m: &mut PSSALayerV2, i: &[usize], t: &[usize]) -> f32 {
    m.reset_recurrent_state();
    m.forward_train_chunk(i, t)
}

fn run(
    seed: u64,
    mode: Option<DreamMode>,
    a_steps: usize,
    b_steps: usize,
    replay: usize,
    rehearsal_lr: f32,
    rehearsal_steps: usize,
) -> (f32, f32, f32) {
    let mut m = new_model(seed);
    for _ in 0..a_steps {
        update(&mut m, &A_IN, &A_TG, 5e-3);
    }
    let a0 = loss(&mut m, &A_IN, &A_TG);
    let d = m.cfg.d_latent;
    let keys = [
        [0.5f32, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
        [0.25, -0.75],
        [-0.75, 0.25],
        [0.8, 0.1],
        [0.1, 0.8],
    ];
    for p in 0..8 {
        let v = m.block.tape.z_final[p * d..(p + 1) * d].to_vec();
        m.block.memory.insert(&keys[p], &v);
    }
    // Keep the supervised Task-A sequence in the runtime-only rehearsal cache;
    // the episodic values above remain the seed source for adapter replay.
    m.remember_dream_sequence(&A_IN, &A_TG);
    let mut rng = SimpleRng::new(0xD0EA_2026 ^ seed);
    for _ in 0..b_steps {
        update(&mut m, &B_IN, &B_TG, 5e-3);
        if let Some(md) = mode {
            m.dream_replay_with_options_and_guard(
                md,
                replay,
                8,
                0.8,
                rehearsal_lr,
                rehearsal_steps,
                Some((&B_IN, &B_TG)),
                &mut rng,
            );
        }
    }
    (a0, loss(&mut m, &A_IN, &A_TG), loss(&mut m, &B_IN, &B_TG))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rehearsal_lr = args
        .get(1)
        .map_or(6e-3, |s| s.parse().expect("rehearsal LR"));
    let rehearsal_steps = args
        .get(2)
        .map_or(1, |s| s.parse().expect("rehearsal steps"));
    println!("rehearsal_lr={rehearsal_lr} rehearsal_steps={rehearsal_steps}");
    let seeds = [7u64, 11, 23, 42, 99];
    for (label, mode) in [
        ("off", None),
        ("memory", Some(DreamMode::Memory)),
        ("both", Some(DreamMode::Both)),
    ] {
        let (mut fa, mut fb, mut f0) = (0.0, 0.0, 0.0);
        for &s in &seeds {
            let (a0, a1, b1) = run(s, mode, 40, 40, 8, rehearsal_lr, rehearsal_steps);
            println!(
                "seed={s} dream={label} a_after_a={a0:.4} a_after_b={a1:.4} b_after_b={b1:.4} forget={:.4}",
                a1 - a0
            );
            fa += a1;
            fb += b1;
            f0 += a0;
        }
        let n = seeds.len() as f32;
        println!(
            "MEAN dream={label} a_after_a={:.4} a_after_b={:.4} b_after_b={:.4} forget={:.4}",
            f0 / n,
            fa / n,
            fb / n,
            (fa - f0) / n
        );
    }
}
