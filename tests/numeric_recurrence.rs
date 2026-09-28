//! Regressions for representable softplus tails in the reference and staged
//! CPU recurrences. GPU GEMM backends share these host-side recurrence stages.
use oxide_ai_pssa::gpu_batch;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn model() -> PSSALayerV2 {
    let mut m = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 2,
            d_latent: 1,
            d_state: 1,
            d_mem_key: 1,
            mem_capacity: 1,
            chunk_len: 2,
            ..PSSAConfigV2::default()
        },
        19,
    );
    // Make the normalized input exactly one, independent of RMS roundoff.
    m.norm_gamma.data.fill(0.0);
    m.norm_beta.data.fill(1.0);
    m.w_b.data.fill(1.0);
    m.w_c.data.fill(1.0);
    m.a_mat.data.fill(0.0);
    m.w_qx.data.fill(0.0);
    m.w_qh.data.fill(0.0);
    m.unembed_w.data.copy_from_slice(&[0.0, 1.0]);
    m
}

fn forward(m: &mut PSSALayerV2, staged: bool) -> f32 {
    if staged {
        gpu_batch::forward_train_chunk_batched(m, &[0, 0], &[1, 1])
    } else {
        m.forward_train_chunk(&[0, 0], &[1, 1])
    }
}

fn backward(m: &mut PSSALayerV2, staged: bool) {
    if staged {
        gpu_batch::backward_chunk_batched(m, 2, 1.0);
    } else {
        m.backward_chunk(2, 1.0);
    }
}

#[test]
fn negative_delta_tail_drives_state_and_matches_its_backward_derivative() {
    for raw_delta in [-17.0f32, -20.0] {
        for staged in [false, true] {
            let mut m = model();
            m.w_delta.data.fill(raw_delta);
            // Amplify the tiny (but representable) state in the output only,
            // so the loss finite difference is well above float32 resolution.
            m.unembed_w.data[1] = 1.0e7;
            let loss = forward(&mut m, staged);
            assert!(loss.is_finite());
            let expected_delta = (raw_delta as f64).exp().ln_1p();
            for &delta in &m.tape.delta[..2] {
                assert!(delta > 0.0);
                assert!((delta as f64 / expected_delta - 1.0).abs() < 2.0e-7);
            }
            assert!(m.tape.bar_b[..2].iter().all(|&x| x > 0.0));
            assert!((m.h_persistent[0] as f64 / (2.0 * expected_delta) - 1.0).abs() < 2.0e-6);
            let training_state = m.h_persistent[0];
            backward(&mut m, staged);
            let analytic = m.w_delta.grad[0];
            assert!(analytic.is_finite() && analytic != 0.0);
            assert!(m.w_b.grad[0].is_finite() && m.w_b.grad[0] != 0.0);

            let step = 0.01;
            m.reset_recurrent_state();
            m.w_delta.data[0] = raw_delta + step;
            let plus = forward(&mut m, staged);
            m.reset_recurrent_state();
            m.w_delta.data[0] = raw_delta - step;
            let minus = forward(&mut m, staged);
            let numeric = (plus - minus) / (2.0 * step);
            assert!(
                (numeric - analytic).abs() <= 0.01 * analytic.abs(),
                "raw_delta={raw_delta}, staged={staged}: analytic={analytic}, numeric={numeric}"
            );

            m.reset_recurrent_state();
            m.w_delta.data[0] = raw_delta;
            let mut logits = [0.0; 2];
            m.forward_inference(0, &mut logits);
            m.forward_inference(0, &mut logits);
            assert_eq!(m.h_persistent[0], training_state);
        }
    }
}

#[test]
fn negative_rate_tail_preserves_decay_in_both_forward_and_backward_paths() {
    for raw_rate in [-17.0f32, -20.0] {
        for staged in [false, true] {
            let mut m = model();
            m.a_mat.data[0] = raw_rate;
            m.w_delta.data[0] = 1.0e7;
            m.w_b.data[0] = 0.0;
            m.h_persistent[0] = 1.0;
            forward(&mut m, staged);
            let rate = (raw_rate as f64).exp().ln_1p();
            let expected_decay = (-1.0e7 * rate).exp();
            for &decay in &m.tape.bar_a[..2] {
                assert!(decay < 1.0 && decay > 0.0);
                assert!((decay as f64 - expected_decay).abs() < 1.0e-7);
            }
            let training_state = m.h_persistent[0];
            assert!((training_state as f64 - expected_decay.powi(2)).abs() < 2.0e-7);
            backward(&mut m, staged);
            let analytic = m.a_mat.grad[0];
            assert!(analytic.is_finite() && analytic != 0.0);

            let step = 0.01;
            m.h_persistent[0] = 1.0;
            m.a_mat.data[0] = raw_rate + step;
            let plus = forward(&mut m, staged);
            m.h_persistent[0] = 1.0;
            m.a_mat.data[0] = raw_rate - step;
            let minus = forward(&mut m, staged);
            let numeric = (plus - minus) / (2.0 * step);
            assert!(
                (numeric - analytic).abs() <= 0.01 * analytic.abs(),
                "raw_rate={raw_rate}, staged={staged}: analytic={analytic}, numeric={numeric}"
            );

            m.h_persistent[0] = 1.0;
            m.a_mat.data[0] = raw_rate;
            let mut logits = [0.0; 2];
            m.forward_inference(0, &mut logits);
            m.forward_inference(0, &mut logits);
            assert_eq!(m.h_persistent[0], training_state);
        }
    }
}
