//! Qwen3-1.7B -> PSSA compact student distillation trainer.
//!
//! Reads a framed binary stream of teacher targets from a file or stdin
//! (produced by teacher_server.py: token ids, next-token targets, and teacher
//! probabilities at the distillation temperature) and trains the student
//! with the CE + temperature-KL loss in gpu_batch::distill_*.
//!
//! Frame layout (little-endian):
//!   u32 magic 0x4B4C4431 ("KLD1")
//!   u32 n_tokens
//!   u32 d_vocab
//!   u64 chunk_index
//!   u32[n] token_ids
//!   u32[n] target_ids
//!   f32[n * d_vocab] teacher_probs (row-major [token][vocab])

use std::io::{BufReader, Read};

use pssa::checkpoint::load_checkpoint;
use pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use pssa::pssa::{GradientClipOutcome, PSSALayerV2};

const FRAME_MAGIC: u32 = 0x4B4C4431;

struct Frame {
    chunk_index: u64,
    tokens: Vec<usize>,
    targets: Vec<usize>,
    teacher: Vec<f32>,
}

fn read_u16le(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32le(r: &mut impl Read, count: usize) -> std::io::Result<Vec<f32>> {
    let mut raw = vec![0u8; count * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_frame(r: &mut impl Read, expected_vocab: usize, max_chunk: usize) -> Option<Frame> {
    let magic = match read_u32le(r) {
        Ok(m) => m,
        Err(_) => return None, // clean EOF between frames
    };
    if magic != FRAME_MAGIC {
        eprintln!("distill_train: bad frame magic 0x{magic:08x}; stream is corrupt");
        std::process::exit(2);
    }
    let n = match read_u32le(r) {
        Ok(n) => n as usize,
        Err(_) => return None,
    };
    let vocab = read_u32le(r).ok()? as usize;
    let chunk_index = read_u64le(r).ok()?;
    if vocab != expected_vocab {
        eprintln!(
            "distill_train: frame vocab {vocab} != student vocab {expected_vocab}; teacher and student must share the tokenizer"
        );
        std::process::exit(2);
    }
    if n == 0 || n > max_chunk {
        eprintln!("distill_train: frame token count {n} outside 1..={max_chunk}");
        std::process::exit(2);
    }
    let mut raw_ids = vec![0u8; n * 4];
    if r.read_exact(&mut raw_ids).is_err() {
        return None;
    }
    let tokens: Vec<usize> = raw_ids
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let mut raw_ids = vec![0u8; n * 4];
    if r.read_exact(&mut raw_ids).is_err() {
        return None;
    }
    let targets: Vec<usize> = raw_ids
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let teacher = match read_f32le(r, n * vocab) {
        Ok(t) => t,
        Err(_) => return None,
    };
    Some(Frame {
        chunk_index,
        tokens,
        targets,
        teacher,
    })
}

struct Args {
    student: String,
    teacher_input: String,
    steps: usize,
    lr: f32,
    warmup: usize,
    grad_clip: Option<f32>,
    save_every: usize,
    out: String,
    temp: f32,
    ce_mix: f32,
}

fn parse_args() -> Args {
    let mut a = Args {
        student: String::new(),
        teacher_input: String::new(),
        steps: 0,
        lr: 1e-3,
        warmup: 0,
        grad_clip: Some(1.0),
        save_every: 1000,
        out: String::new(),
        temp: 2.0,
        ce_mix: 0.5,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let mut next = |i: &mut usize, key: &str| {
            *i += 1;
            argv.get(*i)
                .map(|s| s.clone())
                .unwrap_or_else(|| panic!("{key} needs a value"))
        };
        match argv[i].as_str() {
            "--student" => a.student = next(&mut i, "--student"),
            "--teacher-input" => a.teacher_input = next(&mut i, "--teacher-input"),
            "--steps" => a.steps = next(&mut i, "--steps").parse().expect("--steps usize"),
            "--lr" => a.lr = next(&mut i, "--lr").parse().expect("--lr f32"),
            "--warmup" => a.warmup = next(&mut i, "--warmup").parse().expect("--warmup usize"),
            "--grad-clip" => {
                let v: f32 = next(&mut i, "--grad-clip").parse().expect("--grad-clip f32");
                a.grad_clip = if v > 0.0 { Some(v) } else { None };
            }
            "--save-every" => {
                a.save_every = next(&mut i, "--save-every").parse().expect("--save-every usize")
            }
            "--out" => a.out = next(&mut i, "--out"),
            "--temp" => a.temp = next(&mut i, "--temp").parse().expect("--temp f32"),
            "--ce-mix" => a.ce_mix = next(&mut i, "--ce-mix").parse().expect("--ce-mix f32"),
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    if a.student.is_empty() || a.out.is_empty() {
        panic!("--student and --out are required");
    }
    a
}

fn main() {
    let args = parse_args();
    let loaded = match load_checkpoint(&args.student) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("distill_train: cannot load student '{}': {e}", args.student);
            std::process::exit(1);
        }
    };
    let mut model: PSSALayerV2 = loaded.model;
    eprintln!(
        "distill_train: student loaded (device {}, vocab {}, chunk {})",
        match model.device {
            pssa::backend::Device::Cpu => "cpu".to_string(),
            pssa::backend::Device::Gpu(_) => "gpu".to_string(),
            #[cfg(feature = "cuda")]
            pssa::backend::Device::Cuda(_) => "cuda".to_string(),
        }, model.cfg.d_vocab, model.cfg.chunk_len
    );
    let d_vocab = model.cfg.d_vocab;

    let mut reader: Box<dyn Read> = if args.teacher_input == "-" || args.teacher_input.is_empty() {
        Box::new(std::io::stdin().lock())
    } else {
        let f = std::fs::File::open(&args.teacher_input)
            .unwrap_or_else(|e| panic!("cannot open teacher stream '{}': {e}", args.teacher_input));
        Box::new(BufReader::with_capacity(1 << 20, f))
    };

    let started = std::time::Instant::now();
    let mut step = 0usize;
    let mut skipped_updates = 0usize;
    let mut loss_ema = 0.0f64;
    while args.steps == 0 || step < args.steps {
        let frame = match read_frame(&mut reader, d_vocab, model.cfg.chunk_len) {
            Some(f) => f,
            None => {
                eprintln!("distill_train: teacher stream ended after {step} updates");
                break;
            }
        };
        let n = frame.tokens.len();
        for (&t, &g) in frame.tokens.iter().zip(frame.targets.iter()) {
            assert!(t < d_vocab && g < d_vocab, "frame token id out of vocab range");
        }
        model.set_distill_targets(&frame.teacher, args.temp, args.ce_mix, n);
        let loss = forward_train_chunk_batched(&mut model, &frame.tokens, &frame.targets);
        if !loss.is_finite() {
            eprintln!(
                "distill_train: non-finite loss {} at chunk {}; aborting without further updates",
                loss, frame.chunk_index
            );
            std::process::exit(3);
        }
        backward_chunk_batched(&mut model, n, 1.0);
        // PSSA memory writes happen after backward, in training order.
        model.insert_training_memory(loss, n);
        let lr = if args.warmup > 0 {
            args.lr * ((step + 1) as f32 / args.warmup as f32).min(1.0)
        } else {
            args.lr
        };
        let outcome = match args.grad_clip {
            Some(max_norm) => model.apply_adamw_with_grad_clip(lr, max_norm),
            None => {
                model.apply_adamw(lr);
                GradientClipOutcome::Applied { norm: 0.0 }
            }
        };
        let grad_norm = match outcome {
            GradientClipOutcome::Applied { norm } => norm,
            GradientClipOutcome::Skipped { norm } => {
                skipped_updates += 1;
                norm
            }
        };
        step += 1;
        loss_ema = if loss_ema == 0.0 {
            loss as f64
        } else {
            0.98 * loss_ema + 0.02 * loss as f64
        };
        if step % 25 == 0 || step == 1 {
            let rate = step as f64 / started.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "distill_train: step {step} chunk {} loss {loss:.4} ema {loss_ema:.4} lr {lr:.3e} grad_norm {grad_norm:.3} skipped {skipped_updates} rate {rate:.2} upd/s",
                frame.chunk_index
            );
        }
        if args.save_every > 0 && step % args.save_every == 0 {
            if let Err(e) = pssa::checkpoint::save_model_v6(&model, &args.out) {
                eprintln!("distill_train: save failed at step {step}: {e}");
                std::process::exit(4);
            }
            eprintln!("distill_train: checkpoint saved to {} at step {step}", args.out);
        }
    }
    if let Err(e) = pssa::checkpoint::save_model_v6(&model, &args.out) {
        eprintln!("distill_train: final save failed: {e}");
        std::process::exit(4);
    }
    eprintln!(
        "distill_train: done, {step} updates, final ema {loss_ema:.4}, saved {}",
        args.out
    );
}
