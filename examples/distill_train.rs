//! Qwen3-1.7B -> PSSA compact student distillation trainer. Pure Rust.
//!
//! Two teacher sources:
//!   1. `--corpus` (preferred): an in-process Qwen3 teacher. The repo's own
//!      device forward (`Qwen3Model::forward_logits_device`) runs the teacher
//!      on the same GPU as the student, tokenizing the corpus with the shared
//!      Qwen tokenizer. No python, no pipe.
//!   2. `--teacher-input`: a framed binary stream on stdin or a file
//!      (u32 magic 0x4B4C4431, u32 n, u32 vocab, u64 chunk_index,
//!      u32[n] token ids, u32[n] target ids, f32[n*vocab] teacher probs),
//!      for externally produced teacher targets.
//!
//! Loss: ce_mix * CE + (1 - ce_mix) * KL at temperature T over the full
//! student vocabulary (see gpu_batch::distill_* and tests/distill_kl.rs).

use std::io::{BufReader, Read};

use pssa::checkpoint::load_checkpoint;
use pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use pssa::pssa::{GradientClipOutcome, PSSALayerV2};
use pssa::qwen3::{Qwen3Config, Qwen3Model};
use pssa::teacher_frames::{InProcessTeacher, CorpusReader, read_frame};

enum TeacherSource {
    Stream(Box<dyn Read>),
    InProcess(Box<InProcessTeacher>),
}

impl TeacherSource {
    fn next_chunk(
        &mut self,
        probs_out: &mut [f32],
        expected_vocab: usize,
        max_chunk: usize,
    ) -> Option<(Vec<usize>, Vec<usize>, u64)> {
        match self {
            Self::Stream(reader) => {
                let (tokens, targets, chunk_index, teacher) =
                    read_frame(reader, expected_vocab, max_chunk)?;
                probs_out.copy_from_slice(&teacher);
                Some((tokens, targets, chunk_index))
            }
            Self::InProcess(teacher) => teacher.next_chunk(probs_out),
        }
    }
}

struct Args {
    student: String,
    teacher_input: String,
    corpus: String,
    teacher: String,
    tokenizer: String,
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
        corpus: String::new(),
        teacher: String::new(),
        tokenizer: String::new(),
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
            "--corpus" => a.corpus = next(&mut i, "--corpus"),
            "--teacher" => a.teacher = next(&mut i, "--teacher"),
            "--tokenizer" => a.tokenizer = next(&mut i, "--tokenizer"),
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
    if !a.corpus.is_empty() && a.teacher.is_empty() {
        panic!("--corpus needs --teacher (the Qwen3 checkpoint directory)");
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
    let device_label = match model.device {
        pssa::backend::Device::Cpu => "cpu".to_string(),
        pssa::backend::Device::Gpu(_) => "gpu".to_string(),
        #[cfg(feature = "cuda")]
        pssa::backend::Device::Cuda(_) => "cuda".to_string(),
    };
    eprintln!(
        "distill_train: student loaded (device {device_label}, vocab {}, chunk {}, depth {}, loops {})",
        model.cfg.d_vocab,
        model.cfg.chunk_len,
        model.depth(),
        model.loops()
    );
    let d_vocab = model.cfg.d_vocab;

    // Backend selection mirrors the CLI trainer: auto picks CUDA (when built
    // with the cuda feature) or WebGPU, and falls back to CPU only when no
    // accelerator exists. The staged batched path reads model.device.
    match pssa::backend::Device::try_gpu() {
        Ok(gpu_device) => {
            let label = gpu_device
                .gpu()
                .map(|g| g.backend_label())
                .unwrap_or_else(|| "cpu".to_string());
            eprintln!("distill_train: backend={label}");
            model.device = gpu_device;
        }
        Err(e) => {
            eprintln!("distill_train: backend=cpu ({e}); no accelerator available");
        }
    }

    let mut source = if args.corpus.is_empty() {
        let reader: Box<dyn Read> =
            if args.teacher_input == "-" || args.teacher_input.is_empty() {
                Box::new(std::io::stdin().lock())
            } else {
                let f = std::fs::File::open(&args.teacher_input).unwrap_or_else(|e| {
                    panic!("cannot open teacher stream '{}': {e}", args.teacher_input)
                });
                Box::new(BufReader::with_capacity(1 << 20, f))
            };
        TeacherSource::Stream(reader)
    } else {
        let config_path = format!("{}/config.json", args.teacher);
        let config = match Qwen3Config::from_json_file(&config_path) {
            Ok(c) => c,
            Err(_) => {
                eprintln!(
                    "distill_train: no usable config.json in '{}', falling back to the Qwen3-1.7B shape contract",
                    args.teacher
                );
                Qwen3Config::qwen3_1_7b()
            }
        };
        let teacher_model = match Qwen3Model::load_safetensors(&args.teacher, config) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("distill_train: cannot load teacher '{}': {e}", args.teacher);
                std::process::exit(1);
            }
        };
        if teacher_model.config.vocab_size != d_vocab {
            eprintln!(
                "distill_train: teacher vocab {} != student vocab {d_vocab}",
                teacher_model.config.vocab_size
            );
            std::process::exit(2);
        }
        let tokenizer_path = if args.tokenizer.is_empty() {
            format!("{}/tokenizer.json", args.teacher)
        } else {
            args.tokenizer.clone()
        };
        let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("distill_train: cannot load tokenizer '{tokenizer_path}': {e}");
                std::process::exit(1);
            }
        };
        let gpu = model.device.gpu();
        match &gpu {
            Some(_) => eprintln!("distill_train: teacher forward on the device path"),
            None => eprintln!(
                "distill_train: no GPU device, teacher forward runs on the CPU path (slow; fine for a smoke test)"
            ),
        }
        let files: Vec<String> = args
            .corpus
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        let teacher = InProcessTeacher {
            logits: vec![0.0f32; model.cfg.chunk_len * d_vocab],
            model: teacher_model,
            tokenizer,
            gpu,
            reader: CorpusReader::new(&files),
            carry: Vec::new(),
            pos: 0,
            emitted: 0,
            chunk: model.cfg.chunk_len,
            temp: args.temp,
            vocab: d_vocab,
        };
        eprintln!(
            "distill_train: in-process teacher ready (temp {}, chunk {})",
            args.temp, model.cfg.chunk_len
        );
        TeacherSource::InProcess(Box::new(teacher))
    };

    let mut probs = vec![0.0f32; model.cfg.chunk_len * d_vocab];
    let started = std::time::Instant::now();
    let mut step = 0usize;
    let mut skipped_updates = 0usize;
    let mut loss_ema = 0.0f64;
    while args.steps == 0 || step < args.steps {
        let (tokens, targets, chunk_index) =
            match source.next_chunk(&mut probs, d_vocab, model.cfg.chunk_len) {
                Some(c) => c,
                None => {
                    eprintln!("distill_train: teacher stream ended after {step} updates");
                    break;
                }
            };
        let n = tokens.len();
        for (&t, &g) in tokens.iter().zip(targets.iter()) {
            assert!(t < d_vocab && g < d_vocab, "token id out of vocab range");
        }
        if chunk_index == 0 {
            // Teacher self-CE on the raw token stream: -ln p_teacher(target) at
            // the streaming softmax temperature. A sane Qwen teacher lands in the
            // single digits; a garbage number indicts the loader/tokenizer/forward
            // path rather than the converted student.
            let mut teacher_ce = 0.0f64;
            for t in 0..n {
                let row = &probs[t * d_vocab..(t + 1) * d_vocab];
                teacher_ce += (-row[targets[t]].max(1e-30).ln()) as f64;
            }
            teacher_ce /= n as f64;
            eprintln!("distill_train: teacher_ce {teacher_ce:.4} (temp {})", args.temp);
        }
        model.set_distill_targets(&probs, args.temp, args.ce_mix, n);
        let phase_t = std::time::Instant::now();
        let loss = forward_train_chunk_batched(&mut model, &tokens, &targets);
        let phase_fwd = phase_t.elapsed();
        if !loss.is_finite() {
            eprintln!(
                "distill_train: non-finite loss {loss} at chunk {chunk_index}; aborting without further updates"
            );
            std::process::exit(3);
        }
        let phase_b0 = std::time::Instant::now();
        backward_chunk_batched(&mut model, n, 1.0);
        let phase_bwd = phase_b0.elapsed();
        let phase_o0 = std::time::Instant::now();
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
        let phase_opt = phase_o0.elapsed();
        if step == 0 || step % 25 == 0 {
            eprintln!(
                "distill_train: phase ms fwd {:.0} bwd {:.0} opt {:.0}",
                phase_fwd.as_secs_f64() * 1e3,
                phase_bwd.as_secs_f64() * 1e3,
                phase_opt.as_secs_f64() * 1e3
            );
        }
        step += 1;
        loss_ema = if loss_ema == 0.0 {
            loss as f64
        } else {
            0.98 * loss_ema + 0.02 * loss as f64
        };
        if step % 25 == 0 || step == 1 {
            let rate = step as f64 / started.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "distill_train: step {step} chunk {chunk_index} loss {loss:.4} ema {loss_ema:.4} lr {lr:.3e} grad_norm {grad_norm:.3} skipped {skipped_updates} rate {rate:.2} upd/s"
            );
        }
        if args.save_every > 0 && step % args.save_every == 0 {
            if let Err(e) = pssa::checkpoint::save_model(&model, &args.out) {
                eprintln!("distill_train: save failed at step {step}: {e}");
                std::process::exit(4);
            }
            eprintln!("distill_train: checkpoint saved to {} at step {step}", args.out);
        }
    }
    if let Err(e) = pssa::checkpoint::save_model(&model, &args.out) {
        eprintln!("distill_train: final save failed: {e}");
        std::process::exit(4);
    }
    eprintln!(
        "distill_train: done, {step} updates, final ema {loss_ema:.4}, saved {}",
        args.out
    );
}
