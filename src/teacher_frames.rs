//! Shared teacher-side plumbing for Qwen3 -> PSSA distillation.
//!
//! `teacher_dump` (examples/teacher_dump.rs) pre-computes teacher frames on
//! a machine without a trainer: it tokenizes the corpus, runs the Qwen3
//! device forward per chunk, and writes the framed binary stream the
//! distill trainer reads through `--teacher-input` (stdin, a file, or
//! concatenated shard files). The frame protocol: u32 magic 0x4B4C4431,
//! u32 n, u32 vocab, u64 chunk_index, u32[n] token ids, u32[n] target ids,
//! f32[n*vocab] teacher probabilities.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use crate::backend::GpuDispatch;
use crate::qwen3::Qwen3Model;
use tokenizers::Tokenizer;

pub const FRAME_MAGIC: u32 = 0x4B4C4431;

pub fn read_u32le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_u64le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub fn read_f32le(r: &mut impl Read, count: usize) -> std::io::Result<Vec<f32>> {
    let mut raw = vec![0u8; count * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn write_frame(
    w: &mut impl Write,
    tokens: &[usize],
    targets: &[usize],
    chunk_index: u64,
    probs: &[f32],
) -> std::io::Result<()> {
    debug_assert_eq!(tokens.len(), targets.len());
    w.write_all(&FRAME_MAGIC.to_le_bytes())?;
    w.write_all(&(tokens.len() as u32).to_le_bytes())?;
    let vocab = if probs.is_empty() || tokens.is_empty() {
        0
    } else {
        probs.len() / tokens.len()
    };
    w.write_all(&(vocab as u32).to_le_bytes())?;
    w.write_all(&chunk_index.to_le_bytes())?;
    for id in tokens {
        w.write_all(&(*id as u32).to_le_bytes())?;
    }
    for id in targets {
        w.write_all(&(*id as u32).to_le_bytes())?;
    }
    for p in probs {
        w.write_all(&p.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_frame(
    r: &mut impl Read,
    expected_vocab: usize,
    max_chunk: usize,
) -> Option<(Vec<usize>, Vec<usize>, u64, Vec<f32>)> {
    let magic = match read_u32le(r) {
        Ok(m) => m,
        Err(_) => return None,
    };
    if magic != FRAME_MAGIC {
        eprintln!("teacher_frames: bad frame magic 0x{magic:08x}; stream is corrupt");
        std::process::exit(2);
    }
    let n = read_u32le(r).ok()? as usize;
    let vocab = read_u32le(r).ok()? as usize;
    let chunk_index = read_u64le(r).ok()?;
    if vocab != expected_vocab {
        eprintln!(
            "teacher_frames: frame vocab {vocab} != expected vocab {expected_vocab}; teacher and student must share the tokenizer"
        );
        std::process::exit(2);
    }
    if n == 0 || n > max_chunk {
        eprintln!("teacher_frames: frame token count {n} outside 1..={max_chunk}");
        std::process::exit(2);
    }
    let mut raw_ids = vec![0u8; n * 4];
    r.read_exact(&mut raw_ids).ok()?;
    let tokens: Vec<usize> = raw_ids
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let mut raw_ids = vec![0u8; n * 4];
    r.read_exact(&mut raw_ids).ok()?;
    let targets: Vec<usize> = raw_ids
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let teacher = read_f32le(r, n * vocab).ok()?;
    Some((tokens, targets, chunk_index, teacher))
}

/// Block-wise corpus reader: hands out large text blocks while keeping any
/// trailing partial UTF-8 sequence for the next block, so multibyte
/// characters never split at a block boundary.
pub struct CorpusReader {
    files: Vec<PathBuf>,
    next_file: usize,
    file: Option<BufReader<std::fs::File>>,
    pending: Vec<u8>,
}

impl CorpusReader {
    pub fn new(paths: &[String]) -> Self {
        Self {
            files: paths.iter().map(PathBuf::from).collect(),
            next_file: 0,
            file: None,
            pending: Vec::new(),
        }
    }

    pub fn next_block(&mut self) -> Option<String> {
        loop {
            if self.file.is_none() {
                if self.next_file >= self.files.len() {
                    return None;
                }
                let file = std::fs::File::open(&self.files[self.next_file]).ok()?;
                self.file = Some(BufReader::with_capacity(1 << 20, file));
                self.next_file += 1;
                continue;
            }
            let mut buf = vec![0u8; 1 << 24];
            let read = match self.file.as_mut().expect("checked above").read(&mut buf) {
                Ok(0) => {
                    self.file = None;
                    continue;
                }
                Ok(read) => read,
                Err(_) => {
                    self.file = None;
                    continue;
                }
            };
            buf.truncate(read);
            let mut bytes = std::mem::take(&mut self.pending);
            bytes.extend_from_slice(&buf);
            let valid = match std::str::from_utf8(&bytes) {
                Ok(_) => bytes.len(),
                Err(e) => {
                    let mut keep_from = e.valid_up_to();
                    if let Some(invalid_len) = e.error_len() {
                        keep_from += invalid_len;
                    }
                    let keep = bytes.split_off(keep_from);
                    self.pending = keep;
                    e.valid_up_to()
                }
            };
            return Some(String::from_utf8_lossy(&bytes[..valid]).into_owned());
        }
    }
}

/// Temperature softmax over `[token][vocab]` rows, matching the frame stream.
pub fn softmax_rows(logits: &[f32], probs: &mut [f32], n: usize, vocab: usize, temp: f32) {
    for t in 0..n {
        let off = t * vocab;
        let mut max_v = f32::NEG_INFINITY;
        for i in 0..vocab {
            let v = logits[off + i] / temp;
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..vocab {
            let e = (logits[off + i] / temp - max_v).exp();
            probs[off + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-30);
        for p in &mut probs[off..off + vocab] {
            *p *= inv;
        }
    }
}

/// In-process Qwen3 teacher: tokenizes the corpus with the shared tokenizer,
/// runs the device forward per chunk, and writes temperature-softmax teacher
/// probabilities straight into the caller's buffer.
pub struct InProcessTeacher {
    pub model: Qwen3Model,
    pub tokenizer: Tokenizer,
    pub gpu: Option<GpuDispatch>,
    pub logits: Vec<f32>,
    pub reader: CorpusReader,
    pub carry: Vec<u32>,
    pub pos: usize,
    pub emitted: u64,
    pub chunk: usize,
    pub temp: f32,
    pub vocab: usize,
}

impl InProcessTeacher {
    pub fn next_chunk(&mut self, probs_out: &mut [f32]) -> Option<(Vec<usize>, Vec<usize>, u64)> {
        loop {
            if self.carry.len() >= self.pos + self.chunk + 1 {
                let n = self.chunk;
                let tokens: Vec<usize> = self.carry[self.pos..self.pos + n]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                let targets: Vec<usize> = self.carry[self.pos + 1..self.pos + 1 + n]
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                if let Err(e) = self
                    .model
                    .forward_logits_device(self.gpu.as_ref(), &tokens, &mut self.logits)
                {
                    eprintln!("teacher: forward failed: {e}");
                    std::process::exit(3);
                }
                softmax_rows(&self.logits, probs_out, n, self.vocab, self.temp);
                self.pos += n;
                if self.pos > (1 << 20) {
                    self.carry.drain(..self.pos);
                    self.pos = 0;
                }
                self.emitted += 1;
                return Some((tokens, targets, self.emitted - 1));
            }
            let block = self.reader.next_block()?;
            match self.tokenizer.encode(block.as_str(), false) {
                Ok(encoding) => self.carry.extend_from_slice(encoding.get_ids()),
                Err(e) => {
                    eprintln!("teacher: tokenizer failed on a corpus block: {e}");
                    std::process::exit(3);
                }
            }
        }
    }
}
