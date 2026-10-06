//! End-to-end CUDA/WebGPU training smoke check for depth and Ouro loops.
//!
//! The sandbox normally has no GPU, so this example treats an unavailable
//! device as a successful, clearly reported skip. On a GPU host it compares
//! CPU and GPU gradients after several optimizer updates and then times one
//! approximately 50M-parameter update.

use std::time::Instant;

use pssa::backend::Device;
use pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use pssa::pssa::{PSSAConfigV2, PSSAContinuousBlockV2, PSSALayerV2};
use pssa::sequence_batch::{Sequence, SequenceBatch};

const SEED: u64 = 0x4750_5543_4845_434b;
const SMALL_STEPS: usize = 3;
const GRAD_REL_TOLERANCE: f32 = 1e-3;

fn small_config(depth: usize) -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth,
        d_vocab: 64,
        d_latent: 32,
        d_state: 4,
        d_mem_key: 8,
        mem_capacity: 16,
        chunk_len: 16,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.1,
    }
}

fn tokens(cfg: &PSSAConfigV2, step: usize) -> (Vec<usize>, Vec<usize>) {
    let inputs = (0..cfg.chunk_len)
        .map(|t| (t * 17 + step * 5 + 3) % cfg.d_vocab)
        .collect();
    let targets = (0..cfg.chunk_len)
        .map(|t| (t * 23 + step * 7 + 11) % cfg.d_vocab)
        .collect();
    (inputs, targets)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

fn gradient_error(cpu: &[f32], gpu: &[f32]) -> (f32, f32) {
    assert_eq!(cpu.len(), gpu.len(), "gradient length mismatch");
    let abs = cpu
        .iter()
        .zip(gpu)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    let scale = max_abs(cpu).max(max_abs(gpu)).max(1e-6);
    (abs, abs / scale)
}

fn report_block_gradients(
    prefix: &str,
    cpu: &PSSAContinuousBlockV2,
    gpu: &PSSAContinuousBlockV2,
) -> (f32, f32) {
    let pairs: [(&str, &[f32], &[f32]); 14] = [
        ("norm_gamma", &cpu.norm_gamma.grad, &gpu.norm_gamma.grad),
        ("norm_beta", &cpu.norm_beta.grad, &gpu.norm_beta.grad),
        ("a_mat", &cpu.a_mat.grad, &gpu.a_mat.grad),
        ("w_delta", &cpu.w_delta.grad, &gpu.w_delta.grad),
        ("w_b", &cpu.w_b.grad, &gpu.w_b.grad),
        ("w_c", &cpu.w_c.grad, &gpu.w_c.grad),
        ("w_qx", &cpu.w_qx.grad, &gpu.w_qx.grad),
        ("w_qh", &cpu.w_qh.grad, &gpu.w_qh.grad),
        ("w_gate", &cpu.w_gate.grad, &gpu.w_gate.grad),
        ("w_proj", &cpu.w_proj.grad, &gpu.w_proj.grad),
        ("mlp_w1", &cpu.mlp_w1.grad, &gpu.mlp_w1.grad),
        ("mlp_w2", &cpu.mlp_w2.grad, &gpu.mlp_w2.grad),
        (
            "adapter_down",
            &cpu.adapters[0].down_proj.grad,
            &gpu.adapters[0].down_proj.grad,
        ),
        (
            "adapter_up",
            &cpu.adapters[0].up_proj.grad,
            &gpu.adapters[0].up_proj.grad,
        ),
    ];
    let mut worst_abs: f32 = 0.0;
    let mut worst_rel: f32 = 0.0;
    for (name, cpu_grad, gpu_grad) in pairs {
        let (abs, relative) = gradient_error(cpu_grad, gpu_grad);
        println!("    grad {prefix}.{name:<14} max_abs={abs:.3e} relative={relative:.3e}");
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(relative);
    }
    (worst_abs, worst_rel)
}

fn report_gradients(cpu: &PSSALayerV2, gpu: &PSSALayerV2, step: usize) -> Result<(), String> {
    println!("  step {step} gradient comparison:");
    let mut worst_abs: f32 = 0.0;
    let mut worst_rel: f32 = 0.0;
    for (name, cpu_grad, gpu_grad) in [
        ("embed_w", &cpu.embed_w.grad[..], &gpu.embed_w.grad[..]),
        (
            "unembed_w",
            &cpu.unembed_w.grad[..],
            &gpu.unembed_w.grad[..],
        ),
    ] {
        let (abs, relative) = gradient_error(cpu_grad, gpu_grad);
        println!("    grad {name:<23} max_abs={abs:.3e} relative={relative:.3e}");
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(relative);
    }
    let (abs, relative) = report_block_gradients("block", &cpu.block, &gpu.block);
    worst_abs = worst_abs.max(abs);
    worst_rel = worst_rel.max(relative);
    for (index, (cpu_block, gpu_block)) in
        cpu.extra_blocks.iter().zip(&gpu.extra_blocks).enumerate()
    {
        let (abs, relative) =
            report_block_gradients(&format!("block[{index}]"), cpu_block, gpu_block);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(relative);
    }
    println!("    gradient summary: max_abs={worst_abs:.3e} max_relative={worst_rel:.3e}");
    if worst_rel > GRAD_REL_TOLERANCE {
        return Err(format!(
            "GPU gradient mismatch at step {step}: relative error {worst_rel:.3e} exceeds {GRAD_REL_TOLERANCE:.3e}"
        ));
    }
    Ok(())
}

fn compare_loss(label: &str, cpu_loss: f32, gpu_loss: f32) -> Result<(), String> {
    let abs = (cpu_loss - gpu_loss).abs();
    let relative = abs / cpu_loss.abs().max(gpu_loss.abs()).max(1e-6);
    println!(
        "    {label} loss cpu={cpu_loss:.6} gpu={gpu_loss:.6} max_abs={abs:.3e} relative={relative:.3e}"
    );
    if relative > GRAD_REL_TOLERANCE {
        return Err(format!(
            "GPU packed loss mismatch: relative error {relative:.3e} exceeds {GRAD_REL_TOLERANCE:.3e}"
        ));
    }
    Ok(())
}

fn seed_memory(model: &mut PSSALayerV2) {
    let entries = model.cfg.mem_capacity.min(4);
    for entry in 0..entries {
        let key = (0..model.cfg.d_mem_key)
            .map(|i| ((entry + 2 * i) % 7) as f32 * 0.04 - 0.1)
            .collect::<Vec<_>>();
        let value = (0..model.cfg.d_latent)
            .map(|i| ((3 * entry + i) % 11) as f32 * 0.02 - 0.1)
            .collect::<Vec<_>>();
        model.memory.insert(&key, &value);
    }
}

fn packed_tokens(
    cfg: &PSSAConfigV2,
    step: usize,
    lane: usize,
    len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let inputs = (0..len)
        .map(|t| (t * 17 + step * 5 + lane * 29 + 3) % cfg.d_vocab)
        .collect();
    let targets = (0..len)
        .map(|t| (t * 23 + step * 7 + lane * 31 + 11) % cfg.d_vocab)
        .collect();
    (inputs, targets)
}

fn run_packed_config(
    device: &Device,
    label: &str,
    cfg: PSSAConfigV2,
    batch_size: usize,
) -> Result<(), String> {
    println!("checking packed path: {label} (batch_size={batch_size})");
    let mut cpu = PSSALayerV2::new(cfg.clone(), SEED);
    let mut gpu = PSSALayerV2::new(cfg.clone(), SEED);
    seed_memory(&mut cpu);
    seed_memory(&mut gpu);
    gpu.device = device.clone();
    let mut cpu_batch = SequenceBatch::new(&mut cpu, batch_size)?;
    let mut gpu_batch = SequenceBatch::new(&mut gpu, batch_size)?;
    let mut cpu_seconds = 0.0;
    let mut gpu_seconds = 0.0;

    // Unequal lane lengths exercise packed offsets as well as independent
    // carries. The first call resets both lanes; later calls retain them.
    for step in 0..2 {
        let lane_len = (cfg.chunk_len * 3 / 4).max(1);
        let (cpu_inputs0, cpu_targets0) = packed_tokens(&cfg, step, 0, cfg.chunk_len);
        let (cpu_inputs1, cpu_targets1) = packed_tokens(&cfg, step, 1, lane_len);
        let (gpu_inputs0, gpu_targets0) = (cpu_inputs0.clone(), cpu_targets0.clone());
        let (gpu_inputs1, gpu_targets1) = (cpu_inputs1.clone(), cpu_targets1.clone());
        let cpu_sequences = [
            Sequence {
                lane: 0,
                inputs: &cpu_inputs0,
                targets: &cpu_targets0,
                reset: step == 0,
            },
            Sequence {
                lane: 1,
                inputs: &cpu_inputs1,
                targets: &cpu_targets1,
                reset: step == 0,
            },
        ];
        let gpu_sequences = [
            Sequence {
                lane: 0,
                inputs: &gpu_inputs0,
                targets: &gpu_targets0,
                reset: step == 0,
            },
            Sequence {
                lane: 1,
                inputs: &gpu_inputs1,
                targets: &gpu_targets1,
                reset: step == 0,
            },
        ];

        cpu.zero_gradients();
        let started = Instant::now();
        let cpu_loss = cpu_batch.forward(&mut cpu, &cpu_sequences)?;
        cpu_batch.backward(&mut cpu, 1.0)?;
        cpu_seconds += started.elapsed().as_secs_f64();

        gpu.zero_gradients();
        let started = Instant::now();
        let gpu_loss = gpu_batch.forward(&mut gpu, &gpu_sequences)?;
        gpu_batch.backward(&mut gpu, 1.0)?;
        gpu_seconds += started.elapsed().as_secs_f64();

        compare_loss("packed", cpu_loss, gpu_loss)?;
        report_gradients(&cpu, &gpu, step)?;
        for lane in 0..batch_size {
            let (abs, relative) = gradient_error(cpu_batch.state(lane), gpu_batch.state(lane));
            println!("    carry lane={lane} max_abs={abs:.3e} relative={relative:.3e}");
            if relative > GRAD_REL_TOLERANCE {
                return Err(format!(
                    "GPU packed carry mismatch on lane {lane}: relative error {relative:.3e}"
                ));
            }
        }
    }
    let tokens = (2 * (cfg.chunk_len + (cfg.chunk_len * 3 / 4).max(1))) as f64;
    println!(
        "  packed {label}: cpu_step_seconds={cpu_seconds:.3} gpu_step_seconds={gpu_seconds:.3} cpu_tok/s={:.1} gpu_tok/s={:.1}",
        tokens / cpu_seconds.max(f64::MIN_POSITIVE),
        tokens / gpu_seconds.max(f64::MIN_POSITIVE),
    );
    Ok(())
}

fn run_small_config(device: &Device, depth: usize, loops: usize) -> Result<(), String> {
    let cfg = small_config(depth);
    let mut cpu = PSSALayerV2::new_with_depth_and_loops(cfg.clone(), SEED, depth, loops);
    let mut gpu = PSSALayerV2::new_with_depth_and_loops(cfg.clone(), SEED, depth, loops);
    gpu.device = device.clone();

    let mut cpu_seconds = 0.0;
    let mut gpu_seconds = 0.0;
    let mut cpu_loss = 0.0;
    let mut gpu_loss = 0.0;
    for step in 0..SMALL_STEPS {
        let (inputs, targets) = tokens(&cfg, step);
        let cpu_started = Instant::now();
        cpu.zero_gradients();
        cpu_loss = forward_train_chunk_batched(&mut cpu, &inputs, &targets);
        backward_chunk_batched(&mut cpu, cfg.chunk_len, 1.0);
        cpu_seconds += cpu_started.elapsed().as_secs_f64();

        let gpu_started = Instant::now();
        gpu.zero_gradients();
        gpu_loss = forward_train_chunk_batched(&mut gpu, &inputs, &targets);
        backward_chunk_batched(&mut gpu, cfg.chunk_len, 1.0);
        gpu_seconds += gpu_started.elapsed().as_secs_f64();
        report_gradients(&cpu, &gpu, step)?;

        let cpu_started = Instant::now();
        cpu.apply_adamw(cfg.lr);
        cpu_seconds += cpu_started.elapsed().as_secs_f64();
        let gpu_started = Instant::now();
        gpu.apply_adamw(cfg.lr);
        gpu_seconds += gpu_started.elapsed().as_secs_f64();
    }
    let tokens = (SMALL_STEPS * cfg.chunk_len) as f64;
    println!(
        "  depth={depth} loops={loops} backend={} params={} cpu_loss={cpu_loss:.6} gpu_loss={gpu_loss:.6} cpu_tok/s={:.1} gpu_tok/s={:.1}",
        gpu.device
            .gpu()
            .map(|context| context.backend_label())
            .unwrap_or_else(|| "cpu".to_string()),
        gpu.parameter_count(),
        tokens / cpu_seconds.max(f64::MIN_POSITIVE),
        tokens / gpu_seconds.max(f64::MIN_POSITIVE),
    );
    Ok(())
}

fn run_large_timing(device: &Device) -> Result<(), String> {
    let cfg = PSSAConfigV2 {
        d_vocab: 2048,
        d_latent: 2048,
        d_state: 16,
        d_mem_key: 32,
        mem_capacity: 16,
        chunk_len: 8,
        lr: 1e-3,
        ..PSSAConfigV2::default()
    };
    let mut model = PSSALayerV2::new(cfg.clone(), SEED);
    model.device = device.clone();
    let (inputs, targets) = tokens(&cfg, 0);

    // Warm the CUDA/WebGPU workspaces and resident-weight cache before timing
    // the representative update itself.
    model.zero_gradients();
    forward_train_chunk_batched(&mut model, &inputs, &targets);
    backward_chunk_batched(&mut model, cfg.chunk_len, 1.0);
    model.apply_adamw(cfg.lr);

    let started = Instant::now();
    model.zero_gradients();
    let loss = forward_train_chunk_batched(&mut model, &inputs, &targets);
    backward_chunk_batched(&mut model, cfg.chunk_len, 1.0);
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "50M timing: params={} latent={} vocab={} loss={loss:.6} tok/s={:.1} seconds={seconds:.3}",
        model.parameter_count(),
        cfg.d_latent,
        cfg.d_vocab,
        cfg.chunk_len as f64 / seconds.max(f64::MIN_POSITIVE),
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let device = match Device::try_gpu() {
        Ok(device) => device,
        Err(error) => {
            println!("No GPU present; skipping gpu_train_check ({error})");
            return Ok(());
        }
    };
    let context = device
        .gpu()
        .ok_or("GPU initialization returned a CPU device")?;
    println!("GPU training check: {}", context.backend_label());

    for depth in [1, 2] {
        for loops in [1, 2] {
            println!("checking depth={depth} loops={loops}");
            run_small_config(&device, depth, loops)?;
        }
    }
    run_packed_config(&device, "small", small_config(1), 2)?;
    // These are the molab training widths (depth=1, loops=1, batch=32).
    // At most two tokens per lane keep the CPU reference bounded while still exercising
    // every packed forward/backward gradient at the production parameter
    // shapes; the CLI's chunk length remains independent of this diagnostic.
    run_packed_config(
        &device,
        "molab-training-shape",
        PSSAConfigV2 {
            d_vocab: 2048,
            d_latent: 3584,
            d_state: 16,
            d_mem_key: 32,
            mem_capacity: 512,
            chunk_len: 2,
            ..PSSAConfigV2::default()
        },
        32,
    )?;
    run_large_timing(&device)
}
