//! Pre-compute Qwen3 teacher frames for PSSA distillation. Pure Rust.
//!
//! Writes the framed teacher-probability stream (u32 magic 0x4B4C4431, u32 n,
//! u32 vocab, u64 chunk_index, u32[n] token ids, u32[n] target ids,
//! f32[n*vocab] probs) that `distill_train --teacher-input` consumes, so
//! teacher compute can run on separate machines or GPUs in parallel and be
//! reused across training epochs. Shards the corpus with --skip-chunks and
//! --chunks so several workers can cover a corpus without overlapping.
//!
//! Storage note: each frame is n*vocab f32 values (~39 MiB at n=64,
//! vocab=151936), so pre-computed frames only pay off for pilot-scale runs
//! and epoch reuse. At corpus scale the in-process teacher
//! (`distill_train --corpus`) is the storage-free path.

use std::io::{BufWriter, Write};

use pssa::qwen3::{Qwen3Config, Qwen3Model};
use pssa::teacher_frames::{CorpusReader, InProcessTeacher, write_frame};

struct Args {
    teacher: String,
    tokenizer: String,
    corpus: String,
    out: String,
    skip_chunks: u64,
    chunks: u64,
    temp: f32,
}

fn parse_args() -> Args {
    let mut a = Args {
        teacher: String::new(),
        tokenizer: String::new(),
        corpus: String::new(),
        out: String::new(),
        skip_chunks: 0,
        chunks: u64::MAX,
        temp: 2.0,
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
            "--teacher" => a.teacher = next(&mut i, "--teacher"),
            "--tokenizer" => a.tokenizer = next(&mut i, "--tokenizer"),
            "--corpus" => a.corpus = next(&mut i, "--corpus"),
            "--out" => a.out = next(&mut i, "--out"),
            "--skip-chunks" => {
                a.skip_chunks = next(&mut i, "--skip-chunks").parse().expect("--skip-chunks u64")
            }
            "--chunks" => a.chunks = next(&mut i, "--chunks").parse().expect("--chunks u64"),
            "--temp" => a.temp = next(&mut i, "--temp").parse().expect("--temp f32"),
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    if a.teacher.is_empty() || a.corpus.is_empty() || a.out.is_empty() {
        panic!("--teacher, --corpus and --out are required");
    }
    a
}

fn main() {
    let args = parse_args();
    let config_path = format!("{}/config.json", args.teacher);
    let config = match Qwen3Config::from_json_file(&config_path) {
        Ok(c) => c,
        Err(_) => {
            eprintln!(
                "teacher_dump: no usable config.json in '{}', falling back to the Qwen3-1.7B shape contract",
                args.teacher
            );
            Qwen3Config::qwen3_1_7b()
        }
    };
    let model = match Qwen3Model::load_safetensors(&args.teacher, config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("teacher_dump: cannot load teacher '{}': {e}", args.teacher);
            std::process::exit(1);
        }
    };
    let vocab = model.config.vocab_size;
    let chunk_len = 64; // frame token count; the trainer validates against its own chunk_len
    let tokenizer_path = if args.tokenizer.is_empty() {
        format!("{}/tokenizer.json", args.teacher)
    } else {
        args.tokenizer.clone()
    };
    let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("teacher_dump: cannot load tokenizer '{tokenizer_path}': {e}");
            std::process::exit(1);
        }
    };
    let device = pssa::backend::Device::try_gpu().ok();
    let gpu = device.as_ref().and_then(|d| d.gpu());
    match &gpu {
        Some(_) => eprintln!("teacher_dump: teacher forward on the device path"),
        None => eprintln!("teacher_dump: no GPU device, forward runs on the CPU path (slow)"),
    }
    let files: Vec<String> = args
        .corpus
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    let mut teacher = InProcessTeacher {
        logits: vec![0.0f32; chunk_len * vocab],
        model,
        tokenizer,
        gpu,
        reader: CorpusReader::new(&files),
        carry: Vec::new(),
        pos: 0,
        emitted: 0,
        chunk: chunk_len,
        temp: args.temp,
        vocab,
    };
    let mut probs = vec![0.0f32; chunk_len * vocab];
    let mut out = BufWriter::with_capacity(1 << 24, std::fs::File::create(&args.out).unwrap_or_else(
        |e| panic!("cannot create '{}': {e}", args.out),
    ));
    // Skip without allocating the probabilities twice: run the chunk through
    // the same buffer and drop the frame.
    for _ in 0..args.skip_chunks {
        if teacher.next_chunk(&mut probs).is_none() {
            eprintln!("teacher_dump: corpus ended before --skip-chunks {}", args.skip_chunks);
            std::process::exit(1);
        }
    }
    let mut written: u64 = 0;
    let started = std::time::Instant::now();
    while written < args.chunks {
        let (tokens, targets, chunk_index) = match teacher.next_chunk(&mut probs) {
            Some(x) => x,
            None => break,
        };
        if let Err(e) = write_frame(&mut out, &tokens, &targets, args.skip_chunks + chunk_index, &probs) {
            eprintln!("teacher_dump: write failed: {e}");
            std::process::exit(1);
        }
        written += 1;
        if written % 500 == 0 {
            let rate = written as f64 / started.elapsed().as_secs_f64();
            eprintln!("teacher_dump: {written} chunks ({rate:.2}/s)");
        }
    }
    out.flush().expect("flush frames");
    eprintln!(
        "teacher_dump: wrote {written} frames to '{}' (skip {}, temp {})",
        args.out, args.skip_chunks, args.temp
    );
}
