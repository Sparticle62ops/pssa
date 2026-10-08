//! Optional WebGPU stage parity. The sandbox often has no adapter; in that
//! case these tests report a skip instead of turning a platform limitation
//! into a source failure.

use pssa::backend::{Device, WgpuContext};
use pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use pssa::sequence_batch::{Sequence, SequenceBatch};

fn webgpu() -> Option<WgpuContext> {
    let allow_software = std::env::var("PSSA_WGPU_ALLOW_SOFTWARE").as_deref() == Ok("1");
    match WgpuContext::init_with_software_policy(allow_software) {
        Ok(ctx) => Some(ctx),
        Err(error) if !allow_software && error.contains("software GPU adapter refused") => {
            eprintln!(
                "WebGPU parity skipped: software adapter (CPU/llvmpipe/lavapipe/swiftshader) is disabled; set PSSA_WGPU_ALLOW_SOFTWARE=1 to override ({error})"
            );
            None
        }
        Err(error) if error.contains("WebGPU operation failed") => {
            panic!("WebGPU shader/pipeline validation failed: {error}");
        }
        Err(error) => {
            eprintln!("WebGPU parity skipped: no adapter ({error})");
            None
        }
    }
}

fn model() -> PSSALayerV2 {
    let cfg = PSSAConfigV2 {
        d_vocab: 17,
        d_latent: 7,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 5,
        chunk_len: 5,
        tau_mem: 0.7,
        ..Default::default()
    };
    let mut model = PSSALayerV2::new(cfg, 0x5eed);
    model.memory.insert(&[0.1, -0.1, 0.05, 0.02], &[0.03; 7]);
    model.memory.insert(&[-0.2, 0.07, 0.04, -0.03], &[-0.02; 7]);
    for (i, value) in model.h_persistent.iter_mut().enumerate() {
        *value = (i as f32 - 4.0) * 0.01;
    }
    model
}

fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs() / 1e-6f32.max(x.abs()).max(y.abs())
        })
        .fold(0.0, f32::max)
}

fn assert_close(name: &str, a: &[f32], b: &[f32]) {
    let error = relative_error(a, b);
    assert!(error < 1e-3, "{name}: relative error {error:e}");
}

fn compare_forward(cpu: &PSSALayerV2, gpu: &PSSALayerV2) {
    let d = cpu.cfg.d_latent;
    let s = cpu.cfg.d_state;
    let k = cpu.cfg.d_mem_key;
    let v = cpu.cfg.d_vocab;
    let c = cpu.cfg.mem_capacity;
    for (name, a, b) in [
        (
            "x_norm",
            &cpu.tape.x_norm[..5 * d],
            &gpu.tape.x_norm[..5 * d],
        ),
        ("delta", &cpu.tape.delta[..5 * d], &gpu.tape.delta[..5 * d]),
        (
            "b_proj",
            &cpu.tape.b_proj[..5 * s],
            &gpu.tape.b_proj[..5 * s],
        ),
        (
            "c_proj",
            &cpu.tape.c_proj[..5 * s],
            &gpu.tape.c_proj[..5 * s],
        ),
        (
            "h_states",
            &cpu.tape.h_states[..6 * d * s],
            &gpu.tape.h_states[..6 * d * s],
        ),
        (
            "bar_a",
            &cpu.tape.bar_a[..5 * d * s],
            &gpu.tape.bar_a[..5 * d * s],
        ),
        (
            "bar_b",
            &cpu.tape.bar_b[..5 * d * s],
            &gpu.tape.bar_b[..5 * d * s],
        ),
        ("y_ssm", &cpu.tape.y_ssm[..5 * d], &gpu.tape.y_ssm[..5 * d]),
        ("q_euc", &cpu.tape.q_euc[..5 * k], &gpu.tape.q_euc[..5 * k]),
        (
            "q_poincare",
            &cpu.tape.q_poincare[..5 * k],
            &gpu.tape.q_poincare[..5 * k],
        ),
        ("q_norm", &cpu.tape.q_norm[..5], &gpu.tape.q_norm[..5]),
        (
            "mem_weights",
            &cpu.tape.mem_weights[..5 * c],
            &gpu.tape.mem_weights[..5 * c],
        ),
        ("m_val", &cpu.tape.m_val[..5 * d], &gpu.tape.m_val[..5 * d]),
        ("g_mem", &cpu.tape.g_mem[..5 * d], &gpu.tape.g_mem[..5 * d]),
        (
            "m_proj",
            &cpu.tape.m_proj[..5 * d],
            &gpu.tape.m_proj[..5 * d],
        ),
        ("m_inj", &cpu.tape.m_inj[..5 * d], &gpu.tape.m_inj[..5 * d]),
        (
            "adapter_act",
            &cpu.tape.adapter_act[..5 * cpu.adapters[0].rank],
            &gpu.tape.adapter_act[..5 * gpu.adapters[0].rank],
        ),
        ("z_raw", &cpu.tape.z_raw[..5 * d], &gpu.tape.z_raw[..5 * d]),
        (
            "mlp_act",
            &cpu.tape.mlp_act[..5 * 2 * d],
            &gpu.tape.mlp_act[..5 * 2 * d],
        ),
        (
            "z_final",
            &cpu.tape.z_final[..5 * d],
            &gpu.tape.z_final[..5 * d],
        ),
        (
            "logits",
            &cpu.tape.logits[..5 * v],
            &gpu.tape.logits[..5 * v],
        ),
        ("probs", &cpu.tape.probs[..5 * v], &gpu.tape.probs[..5 * v]),
        ("losses", &cpu.tape.losses[..5], &gpu.tape.losses[..5]),
    ] {
        assert_close(name, a, b);
    }
    assert_close("carry", &cpu.h_persistent, &gpu.h_persistent);
}

fn compare_grads(cpu: &PSSALayerV2, gpu: &PSSALayerV2) {
    macro_rules! matrices {
        ($($field:ident),+ $(,)?) => {$({
            assert_close(stringify!($field), &cpu.$field.grad, &gpu.$field.grad);
        })+};
    }
    matrices!(
        embed_w, norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj,
        mlp_w1, mlp_w2, unembed_w,
    );
    assert_close(
        "adapter_down",
        &cpu.adapters[0].down_proj.grad,
        &gpu.adapters[0].down_proj.grad,
    );
    assert_close(
        "adapter_up",
        &cpu.adapters[0].up_proj.grad,
        &gpu.adapters[0].up_proj.grad,
    );
}

#[test]
fn wgpu_small_forward_and_all_gradients_match_cpu() {
    let Some(ctx) = webgpu() else { return };
    let mut cpu = model();
    let mut gpu = model();
    gpu.device = Device::Gpu(ctx);
    let inputs = [1, 4, 3, 4, 2];
    let targets = [4, 3, 2, 1, 0];
    let cpu_loss = cpu.forward_train_chunk(&inputs, &targets);
    let gpu_loss = forward_train_chunk_batched(&mut gpu, &inputs, &targets);
    assert_close("loss", &[cpu_loss], &[gpu_loss]);
    compare_forward(&cpu, &gpu);
    cpu.backward_chunk(inputs.len(), 0.75);
    backward_chunk_batched(&mut gpu, inputs.len(), 0.75);
    compare_grads(&cpu, &gpu);
}

#[test]
fn wgpu_packed_sequence_batch_matches_cpu() {
    let Some(ctx) = webgpu() else { return };
    let mut cpu = model();
    let mut gpu = model();
    let mut cpu_batch = SequenceBatch::new(&mut cpu, 2).expect("CPU batch workspace");
    let mut gpu_batch = SequenceBatch::new(&mut gpu, 2).expect("WGSL batch workspace");
    let sequences = [
        Sequence {
            lane: 1,
            inputs: &[1, 4, 3, 2],
            targets: &[4, 3, 2, 1],
            reset: true,
        },
        Sequence {
            lane: 0,
            inputs: &[5, 6, 7],
            targets: &[6, 7, 8],
            reset: true,
        },
    ];
    let cpu_loss = cpu_batch
        .forward(&mut cpu, &sequences)
        .expect("CPU packed forward");
    gpu.device = Device::Gpu(ctx);
    let gpu_loss = gpu_batch
        .forward(&mut gpu, &sequences)
        .expect("WGSL packed forward");
    assert_close("packed loss", &[cpu_loss], &[gpu_loss]);
    cpu_batch
        .backward(&mut cpu, 0.75)
        .expect("CPU packed backward");
    gpu_batch
        .backward(&mut gpu, 0.75)
        .expect("WGSL packed backward");
    compare_grads(&cpu, &gpu);
    for lane in 0..2 {
        assert_close("packed carry", cpu_batch.state(lane), gpu_batch.state(lane));
    }
}

#[test]
fn wgpu_training_shape_smoke_uses_ssm_and_memory_kernels() {
    let Some(ctx) = webgpu() else { return };
    let gpu = pssa::backend::GpuDispatch::Wgpu(ctx);
    let (len, dm, ds, dk, dv, cap) = (2, 3584, 16, 32, 3584, 512);
    let (vocab, depth, loops, batch) = (2048, 1, 1, 2);
    assert_eq!((depth, loops), (1, 1));
    let delta = vec![0.1; len * dm];
    let raw = vec![0.0; len * dm];
    let b = vec![0.01; len * ds];
    let x = vec![0.02; len * dm];
    let rates = vec![-0.1; dm * ds];
    let deriv = vec![1.0; dm * ds];
    let c = vec![0.03; len * ds];
    let initial = vec![0.0; dm * ds];
    let mut bar_a = vec![0.0; len * dm * ds];
    let mut bar_b = vec![0.0; len * dm * ds];
    let mut states = vec![0.0; (len + 1) * dm * ds];
    let mut y = vec![0.0; len * dm];
    gpu.ssm_forward(
        &delta,
        &raw,
        &b,
        &x,
        &rates,
        &deriv,
        &c,
        &initial,
        len,
        dm,
        ds,
        &mut bar_a,
        &mut bar_b,
        &mut states,
        &mut y,
    )
    .expect("training-sized WGSL SSM dispatch");
    assert!(y.iter().all(|value| value.is_finite()));

    // Strict backward dispatch catches binding errors that production's CPU
    // fallback could otherwise hide in the model-level parity comparisons.
    let mut gd = vec![0.0; len * dm];
    let mut gb = vec![0.0; len * ds];
    let mut gc = vec![0.0; len * ds];
    let mut ga = vec![0.0; len * dm * ds];
    let mut gx = vec![0.0; len * dm];
    gpu.ssm_backward(
        &delta, &raw, &b, &c, &rates, &deriv, &x, &states, &bar_a, &bar_b, &y, &y, len, dm, ds,
        1.0, &mut gd, &mut gb, &mut gc, &mut ga, &mut gx,
    )
    .expect("training-sized WGSL SSM backward dispatch");
    assert!(
        gd.iter()
            .chain(&gb)
            .chain(&gc)
            .chain(&ga)
            .chain(&gx)
            .all(|x| x.is_finite())
    );

    let wqx = vec![0.0; dk * dm];
    let wqh = vec![0.0; dk * dm];
    let wgate = vec![0.0; dm * dm];
    let wproj = vec![0.0; dm * dv];
    let keys = vec![0.0; cap * dk];
    let norms = vec![0.0; cap];
    let values = vec![0.0; cap * dv];
    let mut qe = vec![0.0; len * dk];
    let mut qp = vec![0.0; len * dk];
    let mut qn = vec![0.0; len];
    let mut weights = vec![0.0; len * cap];
    let mut mv = vec![0.0; len * dv];
    let mut gate = vec![0.0; len * dm];
    let mut mp = vec![0.0; len * dm];
    let mut inj = vec![0.0; len * dm];
    gpu.memory_forward(
        &x,
        &y,
        &wqx,
        &wqh,
        &wgate,
        &wproj,
        &keys,
        &norms,
        &values,
        len,
        dm,
        dk,
        dv,
        cap,
        0,
        0.7,
        &mut qe,
        &mut qp,
        &mut qn,
        &mut weights,
        &mut mv,
        &mut gate,
        &mut mp,
        &mut inj,
    )
    .expect("training-sized WGSL memory dispatch");
    assert!(inj.iter().all(|value| value.is_finite()));

    // Exercise the same latent/vocabulary GEMM row folding used by a small
    // training batch: M=1, batch=2 becomes two device rows, with the real
    // 2048-wide vocabulary head.
    let logits_x = vec![0.01; batch * dm];
    let logits_w = vec![0.0; vocab * dm];
    let logits = gpu
        .try_dispatch_gemm(&logits_x, &logits_w, 1, vocab, dm, batch)
        .expect("training-sized WGSL vocabulary GEMM");
    assert_eq!(logits.len(), batch * vocab);
    assert!(logits.iter().all(|value| value.is_finite()));
}
