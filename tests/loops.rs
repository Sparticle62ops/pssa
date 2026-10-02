//! Regression coverage for the runtime-only weight-shared Ouro passes.
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

const IDS: [usize; 3] = [1, 3, 5];
const TARGETS: [usize; 3] = [3, 5, 0];

fn config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth: 1,
        d_vocab: 7,
        d_latent: 4,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 3,
        chunk_len: IDS.len(),
        weight_decay: 0.0,
        ..Default::default()
    }
}

fn fixture_with_loops(loops: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new_with_depth_and_loops(config(), 0xfeed, 1, loops);
    for (i, value) in m.block.mlp_w2.data.iter_mut().enumerate() {
        *value = (i as f32 + 1.0) * 0.013;
    }
    m.block.memory.insert(&[0.2, -0.1], &[0.7, -0.4, 0.2, 0.9]);
    m.block
        .memory
        .insert(&[-0.3, 0.15], &[-0.2, 0.6, -0.8, 0.1]);
    m
}

fn fixture() -> PSSALayerV2 {
    fixture_with_loops(2)
}

fn loss(m: &mut PSSALayerV2) -> f64 {
    m.reset_recurrent_state();
    m.forward_train_chunk(&IDS, &TARGETS) as f64
}

#[test]
fn first_shared_pass_preserves_the_single_pass_math() {
    let mut one = fixture_with_loops(1);
    let mut two = fixture();
    one.forward_train_chunk(&IDS, &TARGETS);
    two.forward_train_chunk(&IDS, &TARGETS);
    let n = IDS.len() * config().d_latent;
    let loop0 = two.loops() * two.block.tape.max_l * config().d_latent;
    assert_eq!(
        &two.block.tape.z_final[loop0..loop0 + n],
        &one.block.tape.z_final[..n]
    );
    let state_width = config().d_latent * config().d_state;
    let loop0_states = two.loops() * (two.block.tape.max_l + 1) * state_width;
    assert_eq!(
        &two.block.tape.h_states[loop0_states..loop0_states + (IDS.len() + 1) * state_width],
        &one.block.tape.h_states[..(IDS.len() + 1) * state_width]
    );
}

#[test]
fn shared_loop_backward_matches_finite_differences() {
    // (label, analytic gradient, perturbation)
    let mut analytic = fixture();
    loss(&mut analytic);
    analytic.zero_gradients();
    analytic.backward_chunk(IDS.len(), 1.0);

    let cases: &[(&str, fn(&PSSALayerV2) -> f32, fn(&mut PSSALayerV2, f32))] = &[
        (
            "embedding",
            |m| m.embed_w.grad[0],
            |m, delta| m.embed_w.data[0] += delta,
        ),
        (
            "ssm projection",
            |m| m.block.w_delta.grad[0],
            |m, delta| m.block.w_delta.data[0] += delta,
        ),
        (
            "memory query",
            |m| m.block.w_qx.grad[0],
            |m, delta| m.block.w_qx.data[0] += delta,
        ),
        (
            "mlp input",
            |m| m.block.mlp_w1.grad[0],
            |m, delta| m.block.mlp_w1.data[0] += delta,
        ),
        (
            "unembedding",
            |m| m.unembed_w.grad[0],
            |m, delta| m.unembed_w.data[0] += delta,
        ),
    ];

    for &(label, get_gradient, perturb) in cases {
        let analytic_gradient = get_gradient(&analytic);
        let epsilon = 1e-3f32;
        let mut hi = fixture();
        let mut lo = fixture();
        perturb(&mut hi, epsilon);
        perturb(&mut lo, -epsilon);
        let numeric = (loss(&mut hi) - loss(&mut lo)) / (2.0 * epsilon as f64);
        let error = (analytic_gradient as f64 - numeric).abs();
        let tolerance = 2e-4 + 0.02 * (analytic_gradient as f64).abs().max(numeric.abs());
        assert!(
            error <= tolerance,
            "{label}: analytic={analytic_gradient:.8} numeric={numeric:.8} error={error:.8} tolerance={tolerance:.8}"
        );
    }
}
