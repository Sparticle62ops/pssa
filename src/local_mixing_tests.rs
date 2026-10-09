//! CPU local-mixing ablation contract and all-coordinate finite differences.
use crate::pssa::{PSSAConfigV2, PSSALayerV2};

const IDS: [usize; 6] = [1, 2, 1, 3, 4, 2];
const TARGETS: [usize; 6] = [2, 1, 3, 4, 2, 0];

fn model(conv: bool, skip: bool, depth: usize, loops: usize) -> PSSALayerV2 {
    let mut m = PSSALayerV2::new_with_depth_and_loops(PSSAConfigV2 {
        d_vocab: 5, d_latent: 3, d_state: 2, d_mem_key: 2,
        mem_capacity: 3, chunk_len: 6, tau_mem: 0.7, ..Default::default()
    }, 42, depth, loops);
    m.set_local_conv(conv).unwrap();
    m.set_ssm_skip(skip).unwrap();
    for b in std::iter::once(&mut m.block).chain(&mut m.extra_blocks) {
        for (i, v) in b.mlp_w2.data.iter_mut().enumerate() { *v = (i as f32 * 0.7).sin() * 0.08; }
        for (i, v) in b.adapters[0].up_proj.data.iter_mut().enumerate() { *v = (i as f32).cos() * 0.03; }
        b.h_persistent.fill(0.02);
        b.memory.insert(&[0.12, -0.08], &[0.04, -0.08, 0.1]);
        b.memory.insert(&[-0.09, 0.03], &[-0.05, 0.1, 0.03]);
        if let Some(k) = b.local_conv_kernel.as_mut() {
            for i in 0..3 { k.data[i * 4..i * 4 + 4].copy_from_slice(&[0.8, 0.16, -0.12, 0.09]); }
            b.local_conv_history.fill(0.04);
            b.local_conv_loop_history.fill(-0.02);
        }
        if let Some(d) = b.ssm_skip.as_mut() { d.data.copy_from_slice(&[0.25, -0.12, 0.18]); }
    }
    m
}

fn loss(m: &mut PSSALayerV2) -> f64 {
    m.forward_train_chunk(&IDS, &TARGETS);
    m.tape.logits[..IDS.len() * 5].chunks_exact(5).zip(TARGETS)
        .map(|(z, target)| {
            let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            max + z.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln() - z[target] as f64
        }).sum::<f64>() / IDS.len() as f64
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x-y).abs() < 2e-5 * (1.0 + x.abs().max(y.abs())), "{i}: {x} vs {y}");
    }
}

#[test]
fn local_mixing_all_parameter_gradients() {
    for (conv, skip, depth, loops) in [(true,false,1,1), (false,true,1,1), (true,true,1,1), (true,true,2,2)] {
        let mut analytic = model(conv, skip, depth, loops);
        loss(&mut analytic);
        analytic.backward_chunk(6, 1.0);
        let gradients: Vec<Vec<f32>> = analytic.adam_tensors().iter().map(|p| p.grad.to_vec()).collect();
        let mut checked = 0;
        for (family, grad) in gradients.iter().enumerate() {
            for (index, &g) in grad.iter().enumerate() {
                let h = 0.002;
                let mut plus = model(conv, skip, depth, loops);
                let mut minus = model(conv, skip, depth, loops);
                plus.adam_tensors()[family].data[index] += h;
                minus.adam_tensors()[family].data[index] -= h;
                let numeric = (loss(&mut plus) - loss(&mut minus)) / (2.0*h as f64);
                let error = (numeric - g as f64).abs();
                assert!(error < 0.00015 + 0.025 * numeric.abs().max(g.abs() as f64),
                    "conv={conv} skip={skip} depth={depth} loops={loops} family={family} i={index} analytic={g} numeric={numeric}");
                checked += 1;
            }
        }
        assert_eq!(checked, analytic.parameter_count());
        if conv { assert!(analytic.local_conv_kernel.as_ref().unwrap().grad.iter().any(|g| g.abs() > 1e-4)); }
        if skip { assert!(analytic.ssm_skip.as_ref().unwrap().grad.iter().any(|g| g.abs() > 1e-4)); }
    }
}

#[test]
fn local_mixing_continuous_input_gradients() {
    let x: Vec<f32> = (0..18).map(|i| (i as f32 * 0.31 + 0.2).sin()).collect();
    let adj: Vec<f32> = (0..18).map(|i| (i as f32 * 0.77).cos() * 0.2).collect();
    for (conv, skip) in [(true,false),(false,true),(true,true)] {
        let mut m = model(conv, skip, 1, 1);
        m.block.forward_train_chunk(&x, 6);
        let mut gx = vec![0.; 18];
        m.block.backward_chunk(&adj, 6, &mut gx);
        let objective = |input: &[f32]| {
            let mut m = model(conv, skip, 1, 1);
            m.block.forward_train_chunk(input, 6);
            m.tape.z_final[..18].iter().zip(&adj).map(|(&a,&b)| a as f64*b as f64).sum::<f64>()
        };
        for i in 0..18 {
            let mut xp = x.clone(); let mut xm = x.clone();
            xp[i] += 0.001; xm[i] -= 0.001;
            let numeric = (objective(&xp)-objective(&xm))/0.002;
            assert!((gx[i] as f64-numeric).abs() < 0.0005 + numeric.abs()*0.02,
                "conv={conv} skip={skip} input={i} analytic={} numeric={numeric}", gx[i]);
        }
    }
}

#[test]
fn local_mixing_forward_formula_and_causal_taps() {
    let mut m = model(true, true, 1, 1);
    m.memory.count = 0;
    m.adapters[0].up_proj.data.fill(0.0);
    m.mlp_w2.data.fill(0.0);
    m.reset_recurrent_state();
    let x: Vec<f32> = (0..18).map(|i| (i as f32 * 0.4 + 0.2).sin()).collect();
    m.block.forward_train_chunk(&x, 6);
    for t in 0..6 {
        for i in 0..3 {
            let mut expected = 0.;
            for lag in 0..4 {
                if t >= lag { expected += m.local_conv_kernel.as_ref().unwrap().data[4*i+lag] * m.tape.x_norm[(t-lag)*3+i]; }
            }
            assert_eq!(m.tape.ssm_input[t*3+i], expected);
            let expected_z = m.tape.y_ssm[t*3+i] / 2f32.sqrt()
                + m.ssm_skip.as_ref().unwrap().data[i] * m.tape.x_norm[t*3+i];
            assert!((m.tape.z_final[t*3+i] - expected_z).abs() < 1e-6);
        }
        let mut b = vec![0.; 2]; let mut c = vec![0.; 2]; let mut delta = vec![0.; 3];
        m.w_b.matvec(&m.tape.ssm_input[t*3..t*3+3], &mut b);
        m.w_c.matvec(&m.tape.ssm_input[t*3..t*3+3], &mut c);
        m.w_delta.matvec(&m.tape.ssm_input[t*3..t*3+3], &mut delta);
        assert_eq!(&m.tape.b_proj[t*2..t*2+2], &b);
        assert_eq!(&m.tape.c_proj[t*2..t*2+2], &c);
        assert_eq!(&m.tape.delta_raw[t*3..t*3+3], &delta);
    }
}

#[test]
fn local_mixing_chunks_inference_causality_and_loop_carries() {
    for (depth, loops) in [(1,1), (2,1), (1,2), (2,2)] {
        let mut full = model(true,true,depth,loops);
        let mut chunked = model(true,true,depth,loops);
        let mut inference = model(true,true,depth,loops);
        full.forward_train_chunk(&IDS, &TARGETS);
        let logits = full.tape.logits[..30].to_vec();
        let mut offset = 0;
        for n in [1,2,3] {
            chunked.forward_train_chunk(&IDS[offset..offset+n], &TARGETS[offset..offset+n]);
            close(&logits[offset*5..(offset+n)*5], &chunked.tape.logits[..n*5]);
            offset += n;
        }
        for (t, id) in IDS.into_iter().enumerate() {
            let mut out = vec![0.; 5]; inference.forward_inference(id, &mut out);
            close(&logits[t*5..t*5+5], &out);
        }
        let mut causal = model(true,true,depth,loops);
        causal.forward_train_chunk(&[1,2,1,0,0,0], &TARGETS);
        close(&logits[..15], &causal.tape.logits[..15]);
        let mut carry = vec![0.; full.recurrent_state_len()];
        full.copy_recurrent_state_to(&mut carry);
        full.reset_recurrent_state();
        assert!(full.local_conv_history.iter().all(|x| *x == 0.));
        full.copy_recurrent_state_from(&carry);
        let mut restored = vec![0.; carry.len()]; full.copy_recurrent_state_to(&mut restored);
        assert_eq!(carry, restored);
    }
}

#[test]
fn local_mixing_default_off_and_optimizer() {
    let cfg = PSSAConfigV2 { d_vocab: 5, d_latent: 3, d_state: 2, d_mem_key: 2,
        mem_capacity: 3, chunk_len: 6, ..Default::default() };
    let mut base = PSSALayerV2::new(cfg.clone(), 42);
    let mut enabled = PSSALayerV2::new(cfg, 42);
    assert!(!base.local_mixing_enabled());
    assert!(base.tape.ssm_input.is_empty());
    enabled.set_local_conv(true).unwrap(); enabled.set_ssm_skip(true).unwrap();
    assert_eq!(enabled.parameter_count(), base.parameter_count()+15);
    assert_eq!(loss(&mut base), loss(&mut enabled)); // identity convolution, zero D
    enabled.backward_chunk(6, 1.0);
    let conv = enabled.local_conv_kernel.as_ref().unwrap().data.clone();
    let skip = enabled.ssm_skip.as_ref().unwrap().data.clone();
    enabled.apply_adamw_with_grad_clip(0.001, 1.0);
    assert_ne!(conv, enabled.local_conv_kernel.as_ref().unwrap().data);
    assert_ne!(skip, enabled.ssm_skip.as_ref().unwrap().data);
    enabled.zero_gradients();
    assert!(enabled.local_conv_kernel.as_ref().unwrap().grad.iter().all(|x| *x == 0.));
    assert!(enabled.ssm_skip.as_ref().unwrap().grad.iter().all(|x| *x == 0.));
}

#[test]
fn local_mixing_evaluation_preserves_all_carries() {
    let mut m = model(true, true, 2, 2);
    m.forward_train_chunk(&IDS, &TARGETS);
    let mut before = vec![0.; m.recurrent_state_len()];
    m.copy_recurrent_state_to(&mut before);
    let vocab: Vec<String> = ["<unk>", "a", "b", "c", "d"].into_iter().map(str::to_owned).collect();
    let tok = crate::dataset::Tokenizer::from_vocabulary(&vocab).unwrap();
    let result = crate::evaluation::evaluate_pssa(&mut m, &tok, "a b c d a b", Default::default()).unwrap();
    assert!(result.loss.is_finite());
    let mut after = vec![0.; before.len()];
    m.copy_recurrent_state_to(&mut after);
    assert_eq!(before, after);
    m.set_loops(3).unwrap();
    m.reset_recurrent_state();
    assert!(m.forward_train_chunk(&IDS, &TARGETS).is_finite());
}

#[test]
fn local_mixing_dispatch_and_unsupported_paths_are_explicit() {
    let mut scalar = model(true,true,1,1);
    let mut staged = model(true,true,1,1);
    assert_eq!(scalar.forward_train_chunk(&IDS,&TARGETS), crate::gpu_batch::forward_train_chunk_batched(&mut staged,&IDS,&TARGETS));
    scalar.backward_chunk(6,1.0); crate::gpu_batch::backward_chunk_batched(&mut staged,6,1.0);
    for (a,b) in scalar.adam_tensors().iter().zip(staged.adam_tensors()) { assert_eq!(a.grad,b.grad); }
    assert!(crate::sequence_batch::SequenceBatch::new(&mut scalar,2).is_err());
    let path = std::env::temp_dir().join(format!("pssa-arch2-reject-{}.pssa", std::process::id()));
    assert!(crate::checkpoint::save_model(&scalar,&path).unwrap_err().to_string().contains("in-memory"));
    assert!(!path.exists());
}
