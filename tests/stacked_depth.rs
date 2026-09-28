//! End-to-end finite differences exercise every trainable coordinate of small
//! depth-two/four models, including populated detached memory and nonzero carry.
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSAContinuousBlockV2, PSSALayerV2};

const IDS: [usize; 3] = [1, 3, 5];
const TARGETS: [usize; 3] = [3, 5, 7];
const STEP: f32 = 0.002;
const ABS: f64 = 4e-6;
const REL: f64 = 0.015;

fn config(depth: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth,
        d_vocab: 9,
        d_latent: 3,
        d_state: 2,
        d_mem_key: 2,
        mem_capacity: 4,
        chunk_len: 3,
        tau_mem: 0.75,
        weight_decay: 0.0,
        ..Default::default()
    }
}
fn block(m: &PSSALayerV2, l: usize) -> &PSSAContinuousBlockV2 {
    if l == 0 {
        &m.block
    } else {
        &m.extra_blocks[l - 1]
    }
}
fn block_mut(m: &mut PSSALayerV2, l: usize) -> &mut PSSAContinuousBlockV2 {
    if l == 0 {
        &mut m.block
    } else {
        &mut m.extra_blocks[l - 1]
    }
}
fn pattern(xs: &mut [f32], base: f32, layer: usize) {
    for (i, x) in xs.iter_mut().enumerate() {
        *x = base + 0.013 * ((i % 7) as f32 - 3.0) + 0.004 * layer as f32;
    }
}
fn fixture(depth: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(config(depth), 0x5a17);
    pattern(&mut m.embed_w.data, 0.17, 0);
    pattern(&mut m.unembed_w.data, -0.11, 1);
    // Increase head contrast, not its common softmax-invariant offset.
    for x in &mut m.unembed_w.data {
        *x *= 4.0;
    }
    for l in 0..depth {
        let b = block_mut(&mut m, l);
        for (family, base) in [
            0.98, 0.025, -0.58, 0.13, 0.17, -0.15, 0.055, -0.047, 0.16, -0.22, 0.07, 0.065, 0.075,
            0.055,
        ]
        .into_iter()
        .enumerate()
        {
            pattern(data_mut(b, family), base, l);
        }
        pattern(&mut b.adapters[0].consolidated_up, -0.018, l);
        pattern(&mut b.h_persistent, 0.19, l);
        b.memory.insert(&[0.22, -0.14], &[1.55, -0.95, 0.68]);
        b.memory.insert(&[-0.17, 0.19], &[-1.03, 1.23, -0.78]);
    }
    m
}
fn family(b: &PSSAContinuousBlockV2, f: usize) -> (&[f32], &[f32]) {
    match f {
        0 => (&b.norm_gamma.data, &b.norm_gamma.grad),
        1 => (&b.norm_beta.data, &b.norm_beta.grad),
        _ => {
            let p = match f {
                2 => &b.a_mat,
                3 => &b.w_delta,
                4 => &b.w_b,
                5 => &b.w_c,
                6 => &b.w_qx,
                7 => &b.w_qh,
                8 => &b.w_gate,
                9 => &b.w_proj,
                10 => &b.adapters[0].down_proj,
                11 => &b.adapters[0].up_proj,
                12 => &b.mlp_w1,
                13 => &b.mlp_w2,
                _ => unreachable!(),
            };
            (&p.data, &p.grad)
        }
    }
}
fn data_mut(b: &mut PSSAContinuousBlockV2, f: usize) -> &mut [f32] {
    match f {
        0 => &mut b.norm_gamma.data,
        1 => &mut b.norm_beta.data,
        2 => &mut b.a_mat.data,
        3 => &mut b.w_delta.data,
        4 => &mut b.w_b.data,
        5 => &mut b.w_c.data,
        6 => &mut b.w_qx.data,
        7 => &mut b.w_qh.data,
        8 => &mut b.w_gate.data,
        9 => &mut b.w_proj.data,
        10 => &mut b.adapters[0].down_proj.data,
        11 => &mut b.adapters[0].up_proj.data,
        12 => &mut b.mlp_w1.data,
        13 => &mut b.mlp_w2.data,
        _ => unreachable!(),
    }
}
fn ce(m: &mut PSSALayerV2) -> f64 {
    m.forward_train_chunk(&IDS, &TARGETS);
    ce_logits(&m.tape.logits, m.cfg.d_vocab)
}
fn ce_logits(logits: &[f32], v: usize) -> f64 {
    TARGETS
        .iter()
        .enumerate()
        .map(|(t, &target)| {
            let row = &logits[t * v..(t + 1) * v];
            let max = row
                .iter()
                .map(|&x| x as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            max - row[target] as f64
                + row
                    .iter()
                    .map(|&x| (x as f64 - max).exp())
                    .sum::<f64>()
                    .ln()
        })
        .sum::<f64>()
        / TARGETS.len() as f64
}
fn assert_fd(a: f32, n: f64, label: &str) {
    let error = (a as f64 - n).abs();
    let limit = ABS + REL * (a as f64).abs().max(n.abs());
    assert!(
        error <= limit,
        "{label}: analytic={a:.9} numeric={n:.9} error={error:.9} limit={limit:.9}"
    );
}

#[test]
fn every_coordinate_of_every_layer_and_endpoint_matches_finite_differences() {
    for depth in [2, 4] {
        let mut analytic = fixture(depth);
        ce(&mut analytic);
        analytic.zero_gradients();
        analytic.backward_chunk(IDS.len(), 1.0);
        let mut checked = 0;
        for layer in 0..depth {
            for f in 0..14 {
                let (data, grad) = family(block(&analytic, layer), f);
                assert!(
                    grad.iter().any(|g| g.abs() > 3e-5),
                    "vacuous family: depth={depth} layer={layer} family={f}"
                );
                for i in 0..data.len() {
                    let mut hi = fixture(depth);
                    let mut lo = fixture(depth);
                    data_mut(block_mut(&mut hi, layer), f)[i] = data[i] + STEP;
                    data_mut(block_mut(&mut lo, layer), f)[i] = data[i] - STEP;
                    let n = (ce(&mut hi) - ce(&mut lo)) / (2.0 * STEP) as f64;
                    assert_fd(
                        grad[i],
                        n,
                        &format!("depth{depth}/layer{layer}/family{f}[{i}]"),
                    );
                    checked += 1;
                }
            }
        }
        for endpoint in 0..2 {
            let p = if endpoint == 0 {
                &analytic.embed_w
            } else {
                &analytic.unembed_w
            };
            assert!(p.grad.iter().any(|g| g.abs() > 3e-5));
            for i in 0..p.data.len() {
                let mut hi = fixture(depth);
                let mut lo = fixture(depth);
                if endpoint == 0 {
                    hi.embed_w.data[i] += STEP;
                    lo.embed_w.data[i] -= STEP;
                } else {
                    hi.unembed_w.data[i] += STEP;
                    lo.unembed_w.data[i] -= STEP;
                }
                assert_fd(
                    p.grad[i],
                    (ce(&mut hi) - ce(&mut lo)) / (2.0 * STEP) as f64,
                    &format!("depth{depth}/endpoint{endpoint}[{i}]"),
                );
                checked += 1;
            }
        }
        assert_eq!(checked, analytic.parameter_count());
        println!("depth={depth}: checked all {checked} trainable coordinates");
    }
}

// Continue from an arbitrary continuous boundary, independently composing the
// remaining blocks. This probes the raw-input VJP AND each outer identity edge.
fn boundary_ce(depth: usize, boundary: usize, input: &[f32]) -> f64 {
    let mut m = fixture(depth);
    let mut features = input.to_vec();
    for l in boundary..depth {
        let b = block_mut(&mut m, l);
        b.forward_train_chunk(&features, IDS.len());
        if l == 0 {
            features.copy_from_slice(&b.tape.z_final);
        } else {
            for (x, &branch) in features.iter_mut().zip(&b.tape.z_final) {
                *x += branch / (depth as f32).sqrt();
            }
        }
    }
    let v = m.cfg.d_vocab;
    let d = m.cfg.d_latent;
    let mut logits = vec![0.0; IDS.len() * v];
    for t in 0..IDS.len() {
        m.unembed_w.matvec(
            &features[t * d..(t + 1) * d],
            &mut logits[t * v..(t + 1) * v],
        );
    }
    for x in &mut logits {
        *x *= 1.0 / (d as f32).sqrt();
    }
    ce_logits(&logits, v)
}

#[test]
fn cross_layer_boundary_adjoints_match_arbitrary_continuous_perturbations() {
    let depth = 4;
    let mut m = fixture(depth);
    ce(&mut m);
    m.zero_gradients();
    m.backward_chunk(IDS.len(), 1.0);
    for boundary in 0..=depth {
        let x = match boundary {
            0 => &m.continuous_inputs,
            1 => &m.block.tape.z_final,
            _ => &m.layer_activations[boundary - 2],
        };
        assert!(m.boundary_adjoints[boundary].iter().any(|g| g.abs() > 3e-5));
        for i in 0..x.len() {
            let mut hi = x.clone();
            let mut lo = x.clone();
            hi[i] += STEP;
            lo[i] -= STEP;
            let n = (boundary_ce(depth, boundary, &hi) - boundary_ce(depth, boundary, &lo))
                / (2.0 * STEP) as f64;
            assert_fd(
                m.boundary_adjoints[boundary][i],
                n,
                &format!("boundary{boundary}[{i}]"),
            );
        }
    }
}

#[test]
fn parameter_matched_depth_four_config_has_the_required_count() {
    let baseline = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 2048,
            ..Default::default()
        },
        42,
    );
    let stack = PSSALayerV2::new(PSSAConfigV2::stacked_depth_test(), 42);
    assert_eq!(baseline.parameter_count(), 1_544_704);
    assert_eq!(stack.parameter_count(), 1_548_448);
    assert_eq!(stack.depth(), 4);
    assert_eq!(stack.residual_scales, [0.5, 0.5, 0.5]);
    for b in std::iter::once(&stack.block).chain(&stack.extra_blocks) {
        assert_eq!(b.memory.capacity, 512);
        assert_eq!(b.cfg.d_latent, 166);
        assert_eq!(b.cfg.d_state, 16);
        assert_eq!(b.cfg.d_mem_key, 32);
    }
    // Added blocks do not allocate vocabulary buffers or vocabulary parameters.
    assert!(
        stack
            .extra_blocks
            .iter()
            .all(|b| b.tape.logits.is_empty() && b.tape.probs.is_empty())
    );
}

#[test]
fn stacked_token_inference_matches_chunk_forward_and_layer_carries() {
    for depth in [1, 2, 4] {
        let mut chunk = fixture(depth);
        let mut tokens = fixture(depth);
        chunk.forward_train_chunk(&IDS, &TARGETS);
        let mut logits = vec![0.0; chunk.cfg.d_vocab];
        for (t, &id) in IDS.iter().enumerate() {
            tokens.forward_inference(id, &mut logits);
            assert_eq!(
                logits,
                chunk.tape.logits[t * logits.len()..(t + 1) * logits.len()]
            );
        }
        for l in 0..depth {
            assert_eq!(
                block(&tokens, l).h_persistent,
                block(&chunk, l).h_persistent
            );
        }
    }
}
