//! Tiny sequential-task probe for the offline dream replay phase.
//!
//! Run with `cargo run --release --example dream_probe`.  Task A and task B
//! use disjoint two-token cycles; the reported task-A loss after B is a small
//! forgetting measure.  The dream run seeds episodic memory from A, then
//! replays one A entry after each B update without an Adam step.

use pssa::{
    dream::DreamMode,
    linalg::SimpleRng,
    pssa::{PSSAConfigV2, PSSALayerV2},
};

const TASK_A_INPUTS: [usize; 8] = [1, 2, 1, 2, 1, 2, 1, 2];
const TASK_A_TARGETS: [usize; 8] = [2, 1, 2, 1, 2, 1, 2, 1];
const TASK_B_INPUTS: [usize; 8] = [3, 4, 3, 4, 3, 4, 3, 4];
const TASK_B_TARGETS: [usize; 8] = [4, 3, 4, 3, 4, 3, 4, 3];

fn new_model() -> PSSALayerV2 {
    PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 8,
            d_latent: 8,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 8,
            chunk_len: 8,
            ..Default::default()
        },
        7,
    )
}

fn update(model: &mut PSSALayerV2, inputs: &[usize], targets: &[usize]) {
    model.reset_recurrent_state();
    model.forward_train_chunk(inputs, targets);
    model.zero_gradients();
    model.backward_chunk(inputs.len(), 1.0);
    model.apply_adamw(2e-3);
}

fn loss(model: &mut PSSALayerV2, inputs: &[usize], targets: &[usize]) -> f32 {
    model.reset_recurrent_state();
    model.forward_train_chunk(inputs, targets)
}

fn run(dream: bool) -> (f32, f32) {
    let mut model = new_model();
    for _ in 0..12 {
        update(&mut model, &TASK_A_INPUTS, &TASK_A_TARGETS);
    }
    let task_a_loss_after_a = loss(&mut model, &TASK_A_INPUTS, &TASK_A_TARGETS);

    // Keep one detached A representation in the episodic bank while B is
    // trained. This makes the probe independent of the surprise-write gate.
    let a_value = model.block.tape.z_final[7 * model.cfg.d_latent..8 * model.cfg.d_latent].to_vec();
    model.block.memory.insert(&[0.25, -0.25], &a_value);

    let mut rng = SimpleRng::new(0xD0EA_2026);
    for _ in 0..12 {
        update(&mut model, &TASK_B_INPUTS, &TASK_B_TARGETS);
        if dream {
            let summary = model.dream_replay(DreamMode::Memory, 1, 0, 0.8, &mut rng);
            debug_assert_eq!(summary.entries_replayed, 1);
        }
    }
    (
        task_a_loss_after_a,
        loss(&mut model, &TASK_A_INPUTS, &TASK_A_TARGETS),
    )
}

fn main() {
    let (plain_before_b, plain_after_b) = run(false);
    let (dream_before_b, dream_after_b) = run(true);
    println!(
        "dream=false task_a_loss_after_a={plain_before_b:.6} task_a_loss_after_b={plain_after_b:.6} forgetting_delta={:.6}",
        plain_after_b - plain_before_b
    );
    println!(
        "dream=true  task_a_loss_after_a={dream_before_b:.6} task_a_loss_after_b={dream_after_b:.6} forgetting_delta={:.6}",
        dream_after_b - dream_before_b
    );
}
