//! Fixed-suite checkpoint evaluation, isolated from both the trainer and redraws.
//!
//! Integration: call `run_worker(args)` at the START of `tui::run`, returning its
//! `Some(result)` before parsing ordinary flags or starting terminal/stdin I/O.
//! The dashboard owns one `Eval`, calls `watch(chain, latest)` and `poll()` each
//! tick, and routes eval-tab keys to `key`. `p` edits the prompt-file path (empty
//! restores the embedded suite), `r` reloads it, `a` pauses, Up/Down choose a
//! checkpoint, PgUp/PgDn choose a prompt. Global shortcuts remain the shell's.
//!
//! JSONL is append-only, beside each evaluated checkpoint. Identity includes
//! path, size, modification time and canonical suite hash. Changed suites never
//! share a graph. Scores are teacher-forced *reference-only* mean NLL in nats,
//! conditioned on the prompt, not scores of generated answers. Tokenization is
//! per checkpoint, so cross-tokenizer NLL comparisons are not apples-to-apples.
//! The embedded prompts are small built-in-science smoke probes, not a held-out
//! benchmark. Reference strings are encoded separately; preserve their leading
//! space for BPE continuations. OOV probes remain visible but are not charted.
//! PSSA uses its checkpoint memory, fresh recurrent carry, and runtime loops=1.
//! No training, optimizer, memory-write, GPU, or checkpoint-save path is called.
use super::{AMBER, NORMAL_GREEN, SECOND_ACCENT, accent, charts, panel, panel_area};
use crate::{
    checkpoint,
    cli::CLIHandler,
    dataset::{Tokenizer, TokenizerKind},
    inference::{InferenceConfig, PSSAInferenceEngine},
    linalg::SimpleRng,
    pssa::{PSSAConfigV2, PSSALayerV2},
    transformer::{TransformerConfig, TransformerModel},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, UNIX_EPOCH},
};

const DEFAULT_PROMPTS: &str = include_str!("../../assets/eval_prompts.json");
const WORKER_ARG: &str = "--eval-worker";
const HISTORY_FILE: &str = "auto-eval.jsonl";
const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_ALLOCATION: usize = 128 * 1024 * 1024;
const MAX_JSON: usize = 64 * 1024;
const MAX_HISTORY: u64 = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 256;
const MAX_DISCOVERY: usize = 2048;
const SCAN_INTERVAL: Duration = Duration::from_secs(5);
const WORKER_TIMEOUT: Duration = Duration::from_secs(120);
const JOB_GAP: Duration = Duration::from_secs(2);
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn text(value: &Value, name: &str, max: usize) -> Result<String, String> {
    let value = value[name]
        .as_str()
        .ok_or_else(|| format!("missing string '{name}'"))?;
    if value.len() > max
        || value
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err(format!(
            "'{name}' is too long or contains terminal control characters"
        ));
    }
    Ok(value.to_owned())
}

fn clean(value: &str, max: usize) -> String {
    value
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .take(max)
        .collect()
}

#[derive(Clone, Debug)]
struct Prompt {
    id: String,
    prompt: String,
    reference: String,
}

#[derive(Clone, Debug)]
struct Suite {
    prompts: Vec<Prompt>,
    max_new_tokens: usize,
    id: String,
}

impl Suite {
    fn parse(raw: &str) -> Result<Self, String> {
        if raw.len() > MAX_JSON {
            return Err("prompt JSON exceeds 64 KiB".into());
        }
        let value: Value =
            serde_json::from_str(raw).map_err(|e| format!("invalid prompt JSON: {e}"))?;
        if value["version"].as_u64() != Some(1) {
            return Err("prompt JSON requires version 1".into());
        }
        let max_new_tokens = match value.get("max_new_tokens") {
            None => 24,
            Some(n) => n.as_u64().ok_or("max_new_tokens must be an integer")?,
        };
        if !(1..=32).contains(&max_new_tokens) {
            return Err("max_new_tokens must be 1..32".into());
        }
        let entries = value["prompts"]
            .as_array()
            .ok_or("prompts must be an array")?;
        if !(1..=8).contains(&entries.len()) {
            return Err("use 1..8 fixed prompts".into());
        }
        let mut ids = HashSet::new();
        let mut prompts = Vec::new();
        for entry in entries {
            let id = text(entry, "id", 64)?;
            let prompt = text(entry, "prompt", 512)?;
            let reference = text(entry, "reference", 512)?;
            if id.trim().is_empty() || prompt.trim().is_empty() || reference.trim().is_empty() {
                return Err("id, prompt and reference must not be empty".into());
            }
            if !ids.insert(id.clone()) {
                return Err(format!("duplicate prompt id '{id}'"));
            }
            prompts.push(Prompt {
                id,
                prompt,
                reference,
            });
        }
        let mut suite = Self {
            prompts,
            max_new_tokens: max_new_tokens as usize,
            id: String::new(),
        };
        // Stable, non-security FNV identity; JSON whitespace/key order do not matter.
        let hash = suite
            .json()
            .to_string()
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        suite.id = format!("v1-{hash:016x}");
        Ok(suite)
    }

    fn json(&self) -> Value {
        json!({"version": 1, "max_new_tokens": self.max_new_tokens, "prompts": self.prompts.iter().map(|p| {
            json!({"id": p.id, "prompt": p.prompt, "reference": p.reference})
        }).collect::<Vec<_>>()})
    }

    fn load(path: Option<&Path>) -> Result<Self, String> {
        match path {
            None => Self::parse(DEFAULT_PROMPTS),
            Some(path) => Self::parse(&read_small(path, MAX_JSON)?),
        }
    }
}

fn read_small(path: &Path, max: usize) -> Result<String, String> {
    // Check before open too: opening a named pipe could otherwise hang an editor.
    if !fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .is_file()
    {
        return Err("expected a regular file".into());
    }
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > max {
        return Err(format!("file exceeds {} KiB limit", max / 1024));
    }
    String::from_utf8(bytes).map_err(|_| "file is not UTF-8".into())
}

#[derive(Clone, Debug)]
struct Sample {
    id: String,
    prompt: String,
    reference: String,
    answer: String,
    nll: Option<f64>,
    tokens: usize,
    note: String,
}

impl Sample {
    fn json(&self) -> Value {
        let ppl = self.nll.map(f64::exp).filter(|x| x.is_finite());
        json!({"id": self.id, "prompt": self.prompt, "reference": self.reference,
            "answer": self.answer, "reference_nll": self.nll, "reference_perplexity": ppl,
            "reference_tokens": self.tokens, "note": self.note})
    }

    fn parse(value: &Value) -> Result<Self, String> {
        let nll = if value["reference_nll"].is_null() {
            None
        } else {
            Some(
                value["reference_nll"]
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or("invalid reference NLL")?,
            )
        };
        let tokens = value["reference_tokens"]
            .as_u64()
            .filter(|n| *n <= 64)
            .ok_or("invalid reference token count")? as usize;
        if nll.is_some() && tokens == 0 {
            return Err("a score needs reference tokens".into());
        }
        Ok(Self {
            id: text(value, "id", 64)?,
            prompt: text(value, "prompt", 512)?,
            reference: text(value, "reference", 512)?,
            answer: text(value, "answer", 8192)?,
            nll,
            tokens,
            note: text(value, "note", 2048)?,
        })
    }
}

#[derive(Clone, Debug)]
struct Record {
    checkpoint: String,
    fingerprint: String,
    number: u64,
    suite: String,
    model: String,
    step: Option<u64>,
    samples: Vec<Sample>,
    note: String,
}

impl Record {
    fn key(&self) -> String {
        format!("{}\0{}\0{}", self.checkpoint, self.fingerprint, self.suite)
    }

    fn mean_nll(&self) -> Option<f64> {
        // A changing subset of in-vocabulary prompts is not a fair trend line.
        if self.samples.is_empty() || self.samples.iter().any(|s| s.nll.is_none()) {
            return None;
        }
        let tokens: usize = self.samples.iter().map(|s| s.tokens).sum();
        if tokens == 0 {
            return None;
        }
        let mean = self
            .samples
            .iter()
            .map(|s| s.nll.unwrap_or(0.0) * (s.tokens as f64 / tokens as f64))
            .sum::<f64>();
        mean.is_finite().then_some(mean)
    }

    fn json(&self) -> Value {
        json!({"version": 1, "checkpoint": self.checkpoint, "fingerprint": self.fingerprint,
            "checkpoint_number": self.number, "suite": self.suite, "model": self.model,
            "step": self.step, "loops": 1, "status": if self.mean_nll().is_some() { "ok" } else { "skipped" },
            "mean_reference_nll": self.mean_nll(), "samples": self.samples.iter().map(Sample::json).collect::<Vec<_>>(),
            "note": self.note})
    }

    fn parse(raw: &str) -> Result<Self, String> {
        if raw.len() > MAX_JSON {
            return Err("record exceeds 64 KiB".into());
        }
        let value: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if value["version"].as_u64() != Some(1) {
            return Err("unsupported eval record".into());
        }
        let entries = value["samples"]
            .as_array()
            .filter(|v| v.len() <= 8)
            .ok_or("invalid samples")?;
        Ok(Self {
            checkpoint: text(&value, "checkpoint", 4096)?,
            fingerprint: text(&value, "fingerprint", 128)?,
            number: value["checkpoint_number"]
                .as_u64()
                .ok_or("missing checkpoint number")?,
            suite: text(&value, "suite", 64)?,
            model: text(&value, "model", 32)?,
            step: value["step"].as_u64(),
            samples: entries
                .iter()
                .map(Sample::parse)
                .collect::<Result<_, _>>()?,
            note: text(&value, "note", 2048)?,
        })
    }
}

fn append_record(path: &Path, record: &Record) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.file_type().is_file() => {
            return Err("eval history must be a regular file, not a symlink or pipe".into());
        }
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.to_string()),
        _ => {}
    }
    let mut raw = record.json().to_string();
    if raw.len() + 1 > MAX_JSON {
        return Err("eval record exceeds persistence limit".into());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("cannot persist {}: {e}", path.display()))?;
    // Recover a torn final append without concatenating a new JSON record to it.
    if file.metadata().map_err(|e| e.to_string())?.len() > 0 {
        file.seek(SeekFrom::End(-1)).map_err(|e| e.to_string())?;
        let mut last = [0];
        file.read_exact(&mut last).map_err(|e| e.to_string())?;
        if last[0] != b'\n' {
            raw.insert(0, '\n');
        }
    }
    raw.push('\n');
    file.write_all(raw.as_bytes())
        .and_then(|_| file.flush())
        .map_err(|e| e.to_string())
}

fn load_history(path: &Path) -> Result<(Vec<Record>, usize), String> {
    if fs::metadata(path).is_ok_and(|meta| !meta.is_file()) {
        return Err("eval history must be a regular file".into());
    }
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let start = file
        .metadata()
        .map_err(|e| e.to_string())?
        .len()
        .saturating_sub(MAX_HISTORY);
    file.seek(SeekFrom::Start(start))
        .map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file.take(MAX_HISTORY));
    if start > 0 {
        reader.skip_until(b'\n').map_err(|e| e.to_string())?;
    }
    let mut records = Vec::new();
    let mut invalid = 0;
    loop {
        let mut raw = Vec::new();
        let n = reader
            .by_ref()
            .take(MAX_JSON as u64 + 1)
            .read_until(b'\n', &mut raw)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        if n > MAX_JSON {
            if raw.last() != Some(&b'\n') {
                reader.skip_until(b'\n').map_err(|e| e.to_string())?;
            }
            invalid += 1;
            continue;
        }
        match std::str::from_utf8(&raw)
            .ok()
            .and_then(|s| Record::parse(s).ok())
        {
            Some(record) => {
                records.push(record);
                if records.len() > MAX_DISCOVERY {
                    records.remove(0);
                }
            }
            None => invalid += 1,
        }
    }
    Ok((records, invalid))
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Result<Self, String> {
        for _ in 0..10 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("pssa-eval-{}-{id}", std::process::id()));
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("eval scratch directory: {e}")),
            }
        }
        Err("cannot create eval scratch directory".into())
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Read only enough of the public checkpoint container to bound allocation.
/// The official loader still validates checksums, shapes, tokenizer and payload.
fn preflight(path: &Path) -> Result<&'static str, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len > MAX_FILE {
        return Err("checkpoint exceeds auto-eval 64 MiB file cap".into());
    }
    let mut head = [0u8; 106]; // 22-byte header + PSSA config (76) + V8 depth (8)
    file.read_exact(&mut head)
        .map_err(|_| "checkpoint is truncated")?;
    let version = u16::from_le_bytes([head[4], head[5]]);
    let mut reader = checkpoint::Reader::new(&head[22..]);
    let mut size = || {
        reader
            .usize("auto-eval dimension")
            .map_err(|e| e.to_string())
    };
    let vocab = u64::from_le_bytes(head[22..30].try_into().unwrap());
    if !(2..=65_536).contains(&vocab) {
        return Err("auto-eval vocabulary must contain 2..65536 tokens".into());
    }
    let (kind, allocation) = match (&head[..4], version) {
        (b"PSSA", 6..=8) => {
            let mut cfg = PSSAConfigV2 {
                d_vocab: size()?,
                d_latent: size()?,
                d_state: size()?,
                d_mem_key: size()?,
                mem_capacity: size()?,
                chunk_len: size()?,
                ..Default::default()
            };
            if version == 8 {
                cfg.depth = usize::try_from(u64::from_le_bytes(head[98..106].try_into().unwrap()))
                    .map_err(|_| "depth overflow")?;
            }
            checkpoint::validate_model_config(&cfg)?;
            (
                "PSSA",
                checkpoint::allocation_bytes(&cfg).map_err(|e| e.to_string())?,
            )
        }
        (b"TRFM", 1) => {
            let cfg = TransformerConfig {
                d_vocab: size()?,
                d_model: size()?,
                n_heads: size()?,
                d_ff: size()?,
                chunk_len: size()?,
                ..Default::default()
            };
            cfg.validate()?;
            ("transformer", crate::transformer::allocation_bytes(&cfg)?)
        }
        _ => {
            return Err(
                "unsupported checkpoint; auto-eval accepts PSSA V6/V7/V8 or TRFM V1".into(),
            );
        }
    };
    // Transformer inference can transiently own a second tape. Budget below
    // is per allocation, so peak model/tapes remain below ~256 MiB plus loader.
    if allocation > MAX_ALLOCATION {
        return Err("checkpoint exceeds auto-eval 128 MiB model/tape cap".into());
    }
    Ok(kind)
}

fn fingerprint(path: &Path) -> Result<String, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() {
        return Err("not a regular checkpoint file".into());
    }
    let nanos = meta
        .modified()
        .map_err(|e| e.to_string())?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(format!("{}:{nanos}", meta.len()))
}

fn checkpoint_number(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let start = stem.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    stem[start..].parse().ok()
}

fn blank_record(path: &Path, suite: &Suite) -> Record {
    Record {
        checkpoint: path.to_string_lossy().into_owned(),
        fingerprint: fingerprint(path).unwrap_or_default(),
        number: checkpoint_number(path).unwrap_or(0),
        suite: suite.id.clone(),
        model: "unknown".into(),
        step: None,
        samples: Vec::new(),
        note: String::new(),
    }
}

// These forwards use the same inference APIs as chat/generate; reference NLL
// includes every vocabulary logit (unlike generation, which excludes <unk>).
fn reference_nll(logits: &[f32], target: usize) -> Result<f64, String> {
    if target >= logits.len() || logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite or invalid reference logits".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    Ok(logits
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
        - (logits[target] as f64 - max))
}

fn throttle(start: Instant, enabled: bool) {
    if enabled {
        // Approximate 10% of one CPU for short forwards, plus a minimum pause.
        // A forward is indivisible; priority and the wall-time watchdog bound
        // long operations instead of pretending this is a hard CPU quota.
        thread::sleep(
            start
                .elapsed()
                .saturating_mul(9)
                .max(Duration::from_millis(5)),
        );
    }
}

enum Model {
    Pssa(Box<PSSALayerV2>),
    Transformer(Box<TransformerModel>),
}
impl Model {
    fn reset(&mut self) {
        if let Self::Pssa(model) = self {
            model.reset_recurrent_state();
        }
    }
    fn forward(&mut self, context: &[usize], logits: &mut [f32], slow: bool) -> Result<(), String> {
        let start = Instant::now();
        match self {
            Self::Pssa(model) => {
                model.forward_inference(*context.last().ok_or("empty context")?, logits)
            }
            Self::Transformer(model) => model.logits_for_context(context, logits)?,
        }
        throttle(start, slow);
        Ok(())
    }

    fn sample(&mut self, tok: &Tokenizer, prompt: &Prompt, suite: &Suite, slow: bool) -> Sample {
        let mut sample = Sample {
            id: prompt.id.clone(),
            prompt: prompt.prompt.clone(),
            reference: prompt.reference.clone(),
            answer: String::new(),
            nll: None,
            tokens: 0,
            note: String::new(),
        };
        let result = (|| -> Result<(), String> {
            let ids = tok.try_encode(&prompt.prompt, true)?;
            let reference = tok.try_encode(&prompt.reference, true)?;
            if ids.is_empty() || ids.len() > 64 || reference.is_empty() || reference.len() > 64 {
                return Err("prompt/reference must each encode to 1..64 tokens".into());
            }
            if ids.iter().all(|&id| id == 0) {
                return Err("prompt is entirely out of vocabulary".into());
            }
            let mut logits = vec![0.0; tok.vocab_size];
            let mut context = Vec::new();
            self.reset();
            if matches!(self, Self::Transformer(_)) {
                context = ids.clone();
                self.forward(&context, &mut logits, slow)?;
            } else {
                for &id in &ids {
                    context.push(id);
                    self.forward(&context, &mut logits, slow)?;
                }
            }
            let mut nll = 0.0;
            for (i, &id) in reference.iter().enumerate() {
                nll += reference_nll(&logits, id)?;
                if i + 1 < reference.len() {
                    context.push(id);
                    self.forward(&context, &mut logits, slow)?;
                }
            }
            // Do not let <unk> make a tiny word-level checkpoint look good.
            if ids.contains(&0) || reference.contains(&0) {
                sample.note =
                    "Reference score skipped: prompt/reference has out-of-vocabulary words.".into();
            } else {
                sample.nll = Some(nll / reference.len() as f64);
                sample.tokens = reference.len();
            }
            let cfg = InferenceConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                repetition_penalty: 1.0,
                max_new_tokens: suite.max_new_tokens,
            };
            sample.answer = match self {
                Self::Pssa(model) => {
                    let last = std::cell::Cell::new(Instant::now());
                    PSSAInferenceEngine::try_new(model, tok)?.try_generate_chat_turn_controlled(
                        &prompt.prompt,
                        &cfg,
                        |_, _| {},
                        || {
                            throttle(last.get(), slow);
                            last.set(Instant::now());
                            false
                        },
                    )?
                }
                Self::Transformer(_) => {
                    context = ids;
                    let mut answer = Vec::new();
                    let mut rng = SimpleRng::new(1337);
                    let mut probs = vec![0.0; tok.vocab_size];
                    let mut candidates = Vec::new();
                    let mut sentences = 0;
                    for _ in 0..suite.max_new_tokens {
                        self.forward(&context, &mut logits, slow)?;
                        let id = PSSAInferenceEngine::sample(
                            &mut rng,
                            &cfg,
                            &context,
                            &mut logits,
                            &mut probs,
                            &mut candidates,
                        )?;
                        context.push(id);
                        answer.push(id);
                        // Match the normal transformer generation stop policy.
                        if tok.kind() == TokenizerKind::Word
                            && matches!(tok.id_to_token[&id].as_str(), "." | "?" | "!")
                        {
                            sentences += 1;
                            if sentences >= 2 {
                                break;
                            }
                        }
                    }
                    tok.decode(&answer)
                }
            };
            sample.answer = clean(&sample.answer, 1024);
            Ok(())
        })();
        if let Err(e) = result {
            sample.note = clean(&e, 512);
            sample.nll = None;
            sample.tokens = 0;
        }
        sample
    }
}

fn evaluate_checkpoint(path: &Path, suite: &Suite, scratch: &Path, slow: bool) -> Record {
    let mut record = blank_record(path, suite);
    let result = (|| -> Result<(), String> {
        // A capped private snapshot avoids a TOCTOU between the dimension guard
        // and the official loader while the trainer atomically replaces a path.
        let original = File::open(path).map_err(|e| e.to_string())?;
        let meta = original.metadata().map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() > MAX_FILE {
            return Err("checkpoint exceeds auto-eval 64 MiB file cap or is not regular".into());
        }
        let snapshot = scratch.join("snapshot.checkpoint");
        let mut out = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&snapshot)
            .map_err(|e| e.to_string())?;
        let copied =
            std::io::copy(&mut original.take(MAX_FILE + 1), &mut out).map_err(|e| e.to_string())?;
        drop(out);
        if copied > MAX_FILE {
            return Err("checkpoint grew beyond auto-eval file cap".into());
        }
        if fingerprint(path)? != record.fingerprint {
            return Err("checkpoint changed during snapshot; waiting for next version".into());
        }
        record.model = preflight(&snapshot)?.into();
        let (mut model, tok, step) = if record.model == "PSSA" {
            let (model, tok) = CLIHandler::load_for_inference(
                snapshot.to_str().ok_or("non-UTF-8 scratch path")?,
                None,
            )?;
            let step = model.step_counter;
            (Model::Pssa(Box::new(model)), tok, step)
        } else {
            let model = crate::transformer_checkpoint::load_checkpoint(&snapshot)
                .map_err(|e| e.to_string())?;
            let tok = model.tokenizer()?;
            let step = model.step_counter;
            (Model::Transformer(Box::new(model)), tok, step)
        };
        if tok.id_to_token.values().any(|token| token.len() > 1024) {
            return Err("auto-eval skips vocabulary entries larger than 1024 bytes".into());
        }
        record.step = Some(step as u64);
        if checkpoint_number(path).is_none() {
            record.number = step as u64;
        }
        for prompt in &suite.prompts {
            record.samples.push(model.sample(&tok, prompt, suite, slow));
        }
        record.note = if record.mean_nll().is_some() {
            "Greedy answers; reference-only NLL; loops=1."
        } else {
            "Some prompts skipped; no aggregate until the whole fixed suite is scoreable."
        }
        .into();
        Ok(())
    })();
    if let Err(error) = result {
        record.note = clean(&format!("Skipped: {error}"), 512);
    }
    record
}

/// Hidden dispatch hook. Ordinary TUI arguments return None, even on a pipe.
/// Exact worker invocation: `EXE tui --eval-worker CHECKPOINT SUITE_JSON RESULT`.
/// The parent writes the frozen suite and supplies a private scratch directory.
pub(super) fn run_worker(args: &[String]) -> Option<Result<(), String>> {
    if args.first().map(String::as_str) != Some(WORKER_ARG) {
        return None;
    }
    Some((|| {
        if args.len() != 4 {
            return Err("invalid auto-eval worker invocation".into());
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .map_err(|e| e.to_string())?;
        let suite = Suite::load(Some(Path::new(&args[2])))?;
        let result = Path::new(&args[3]);
        let scratch = result.parent().ok_or("worker result needs a directory")?;
        let record = evaluate_checkpoint(Path::new(&args[1]), &suite, scratch, true);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(result)
            .map_err(|e| e.to_string())?;
        file.write_all(record.json().to_string().as_bytes())
            .map_err(|e| e.to_string())
    })())
}

struct Job {
    child: Child,
    scratch: TempDir,
    record: Record,
    started: Instant,
}
impl Job {
    fn start(path: &Path, suite: &Suite) -> Result<Self, String> {
        let scratch = TempDir::new()?;
        let prompts = scratch.0.join("prompts.json");
        fs::write(&prompts, suite.json().to_string()).map_err(|e| e.to_string())?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("nice");
            command.arg("-n").arg("19").arg(&exe);
            command
        };
        #[cfg(not(unix))]
        let mut command = Command::new(exe);
        let child = command
            .arg("tui")
            .arg(WORKER_ARG)
            .arg(path)
            .arg(prompts)
            .arg(scratch.0.join("result.json"))
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start low-priority eval worker: {e}"))?;
        Ok(Self {
            child,
            scratch,
            record: blank_record(path, suite),
            started: Instant::now(),
        })
    }

    fn finish(&mut self) -> Option<Record> {
        let result = match self.child.try_wait() {
            Ok(None) if self.started.elapsed() < WORKER_TIMEOUT => return None,
            Ok(None) => Err("Skipped: evaluation exceeded 120 second wall-time limit".into()),
            Ok(Some(status)) if status.success() => {
                read_small(&self.scratch.0.join("result.json"), MAX_JSON)
                    .and_then(|raw| Record::parse(&raw))
            }
            Ok(Some(status)) => Err(format!("Skipped: evaluation worker exited {status}")),
            Err(e) => Err(format!("Skipped: evaluation worker: {e}")),
        };
        Some(match result {
            Ok(record) if record.key() == self.record.key() => record,
            Ok(_) => {
                self.record.note =
                    "Skipped: checkpoint changed before evaluation; waiting for next scan".into();
                self.record.clone()
            }
            Err(e) => {
                self.record.note = clean(&e, 512);
                self.record.clone()
            }
        })
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Default)]
struct Snapshot {
    records: Vec<Record>,
    note: String,
    suite: String,
    busy: bool,
}
enum Control {
    Watch(PathBuf, Option<PathBuf>),
    Configure(Option<PathBuf>),
    Enabled(bool),
    Stop,
}

fn discover(chain: &Path, latest: Option<&Path>) -> (Vec<PathBuf>, bool) {
    let mut paths = Vec::new();
    let mut capped = false;
    if let Ok(entries) = fs::read_dir(chain) {
        for (i, entry) in entries.enumerate() {
            if i >= MAX_DISCOVERY {
                capped = true;
                break;
            }
            if let Ok(entry) = entry {
                let path = entry.path();
                if matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("pssa" | "trfm")
                ) {
                    paths.push(path);
                }
            }
        }
    }
    if let Some(path) = latest {
        paths.push(path.to_path_buf());
    }
    // Canonical identities prevent chain + progress-log aliases scoring twice.
    let mut paths: Vec<_> = paths
        .into_iter()
        .filter_map(|p| fs::canonicalize(p).ok())
        .collect();
    paths.sort_by_key(|p| (checkpoint_number(p).unwrap_or(u64::MAX), p.clone()));
    paths.dedup();
    (paths, capped)
}

fn controller(
    rx: mpsc::Receiver<Control>,
    tx: mpsc::SyncSender<Snapshot>,
    initial: Option<PathBuf>,
) {
    let mut suite = Suite::load(initial.as_deref());
    let mut chain = PathBuf::new();
    let mut latest = None;
    let mut enabled = true;
    let mut job: Option<Job> = None;
    let mut records = Vec::<Record>::new();
    let mut seen = HashSet::new();
    let mut observed = HashMap::<PathBuf, String>::new();
    let mut history_dirs = HashSet::new();
    let mut scan = Instant::now() - SCAN_INTERVAL;
    let mut next_job = Instant::now();
    let mut note = "Watching for completed checkpoints.".to_owned();
    let mut dirty = true;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Control::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Ok(Control::Enabled(value)) => {
                enabled = value;
                if !enabled {
                    job = None;
                }
                note = if value {
                    "Auto evaluation resumed."
                } else {
                    "Paused; training is unaffected."
                }
                .into();
                dirty = true;
            }
            Ok(Control::Configure(path)) => {
                job = None;
                suite = Suite::load(path.as_deref());
                records.clear();
                seen.clear();
                observed.clear();
                history_dirs.clear();
                note = "Prompt suite reloaded; each suite has its own history.".into();
                scan = Instant::now() - SCAN_INTERVAL;
                dirty = true;
            }
            Ok(Control::Watch(dir, checkpoint)) => {
                if chain != dir {
                    job = None;
                    records.clear();
                    seen.clear();
                    observed.clear();
                    history_dirs.clear();
                    note = "Watching new checkpoint directory.".into();
                }
                chain = dir;
                latest = checkpoint;
                scan = Instant::now() - SCAN_INTERVAL;
                dirty = true;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if let Some(record) = job.as_mut().and_then(Job::finish) {
            job = None; // kill/reap BEFORE starting another model or removing scratch
            let history = Path::new(&record.checkpoint)
                .parent()
                .unwrap_or(Path::new("."))
                .join(HISTORY_FILE);
            note = match append_record(&history, &record) {
                Ok(()) => format!("Checkpoint {}: {}", record.number, record.note),
                Err(e) => format!("Result in memory only: {e}"),
            };
            seen.insert(record.key());
            records.push(record);
            if records.len() > MAX_RECORDS {
                records.remove(0);
            }
            next_job = Instant::now() + JOB_GAP;
            dirty = true;
        }
        if let Ok(suite) = &suite
            && !chain.as_os_str().is_empty()
            && scan.elapsed() >= SCAN_INTERVAL
        {
            scan = Instant::now();
            let (paths, capped) = discover(&chain, latest.as_deref());
            if capped {
                note =
                    "Directory scan capped at 2048 entries; use a smaller chain directory.".into();
                dirty = true;
            }
            let mut stable = Vec::new();
            for path in paths {
                if let Some(dir) = path.parent()
                    && history_dirs.insert(dir.to_path_buf())
                {
                    match load_history(&dir.join(HISTORY_FILE)) {
                        Ok((loaded, invalid)) => {
                            for record in loaded.into_iter().filter(|r| r.suite == suite.id) {
                                if seen.insert(record.key()) {
                                    records.push(record);
                                }
                            }
                            if records.len() > MAX_RECORDS {
                                records.drain(..records.len() - MAX_RECORDS);
                            }
                            if invalid > 0 {
                                note =
                                    format!("Ignored {invalid} malformed/partial JSONL records.");
                            }
                        }
                        Err(e) => note = e,
                    }
                    dirty = true;
                }
                if let Ok(mark) = fingerprint(&path) {
                    if observed.get(&path) == Some(&mark) {
                        stable.push(path.clone());
                    }
                    observed.insert(path, mark);
                }
            }
            if enabled && job.is_none() && Instant::now() >= next_job {
                if let Some(path) = stable
                    .into_iter()
                    .find(|p| !seen.contains(&blank_record(p, suite).key()))
                {
                    match Job::start(&path, suite) {
                        Ok(value) => {
                            note = format!(
                                "Evaluating {} (one low-priority CPU worker)",
                                path.file_name().unwrap_or_default().to_string_lossy()
                            );
                            job = Some(value);
                        }
                        Err(e) => {
                            let mut record = blank_record(&path, suite);
                            record.note = clean(&format!("Skipped: {e}"), 512);
                            let saved = append_record(
                                &path.parent().unwrap_or(Path::new(".")).join(HISTORY_FILE),
                                &record,
                            );
                            note = saved.err().unwrap_or_else(|| record.note.clone());
                            seen.insert(record.key());
                            records.push(record);
                            next_job = Instant::now() + JOB_GAP;
                        }
                    }
                    dirty = true;
                }
            }
            // Bound memory even when a trainer replaces one checkpoint forever.
            if observed.len() > MAX_DISCOVERY + 1 {
                observed.clear();
            }
            if seen.len() > MAX_DISCOVERY * 2 {
                seen = records.iter().map(Record::key).collect();
            }
        }
        if dirty {
            records.sort_by_key(|r| (r.number, r.checkpoint.clone()));
            if records.len() > MAX_RECORDS {
                records.drain(..records.len() - MAX_RECORDS);
            }
            let snapshot = Snapshot {
                records: records.clone(),
                note: suite
                    .as_ref()
                    .err()
                    .map(|e| format!("Prompt configuration: {e}"))
                    .unwrap_or_else(|| note.clone()),
                suite: suite.as_ref().map(|s| s.id.clone()).unwrap_or_default(),
                busy: job.is_some(),
            };
            if tx.try_send(snapshot).is_ok() {
                dirty = false;
            }
        }
    }
    // Dropping Job kills and reaps its subprocess; no orphan continues after quit.
}

pub(super) struct Eval {
    tx: mpsc::Sender<Control>,
    rx: mpsc::Receiver<Snapshot>,
    controller: Option<thread::JoinHandle<()>>,
    watched: Option<(PathBuf, Option<PathBuf>)>,
    prompt_file: Option<PathBuf>,
    edit: Option<String>,
    snapshot: Snapshot,
    enabled: bool,
    selected: usize,
    prompt: usize,
    run_context: String,
}
impl Default for Eval {
    fn default() -> Self {
        Self::new()
    }
}
impl Eval {
    pub(super) fn new() -> Self {
        let prompt_file = crate::env_var_os("PSSA_EVAL_PROMPTS", "OXIDE_EVAL_PROMPTS")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let initial = prompt_file.clone();
        let (tx, commands) = mpsc::channel();
        let (updates, rx) = mpsc::sync_channel(1);
        let controller = thread::spawn(move || controller(commands, updates, initial));
        Self {
            tx,
            rx,
            controller: Some(controller),
            watched: None,
            prompt_file,
            edit: None,
            snapshot: Snapshot {
                note: "Auto eval watches completed .pssa / .trfm checkpoints.".into(),
                ..Default::default()
            },
            enabled: true,
            selected: 0,
            prompt: 0,
            run_context: "No training run connected; watching configured checkpoint directory.".into(),
        }
    }

    pub(super) fn set_run_context(&mut self, context: String) {
        self.run_context = context;
    }

    /// Cheap, change-only channel send; scans and all file I/O run elsewhere.
    pub(super) fn watch(&mut self, chain_dir: &Path, latest: Option<&Path>) {
        let watch = (chain_dir.to_path_buf(), latest.map(Path::to_path_buf));
        if self.watched.as_ref() != Some(&watch) {
            let _ = self
                .tx
                .send(Control::Watch(watch.0.clone(), watch.1.clone()));
            self.watched = Some(watch);
        }
    }
    pub(super) fn poll(&mut self) {
        while let Ok(snapshot) = self.rx.try_recv() {
            let follow = self.selected + 1 >= self.snapshot.records.len();
            self.snapshot = snapshot;
            if follow {
                self.selected = self.snapshot.records.len().saturating_sub(1);
            }
            self.selected = self
                .selected
                .min(self.snapshot.records.len().saturating_sub(1));
        }
    }
    pub(super) fn editing(&self) -> bool {
        self.edit.is_some()
    }
    pub(super) fn prompt_file(&self) -> Option<&Path> {
        self.prompt_file.as_deref()
    }
    /// Loaded/validated asynchronously; errors are shown without replacing files.
    pub(super) fn set_prompt_file(&mut self, path: Option<PathBuf>) {
        self.prompt_file = path;
        let _ = self.tx.send(Control::Configure(self.prompt_file.clone()));
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        if let Some(edit) = &mut self.edit {
            match key.code {
                KeyCode::Esc => self.edit = None,
                KeyCode::Enter => {
                    let value = self.edit.take().unwrap();
                    self.set_prompt_file((!value.is_empty()).then(|| PathBuf::from(value)));
                }
                KeyCode::Backspace => {
                    edit.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => edit.clear(),
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    if edit.len() + c.len_utf8() <= 4096 {
                        edit.push(c);
                    }
                }
                _ => {}
            }
            return;
        }
        if key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        {
            return;
        }
        match key.code {
            KeyCode::Char('p') => {
                self.edit = Some(
                    self.prompt_file
                        .as_deref()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                )
            }
            KeyCode::Char('r') => self.set_prompt_file(self.prompt_file.clone()),
            KeyCode::Char('a') => {
                self.enabled = !self.enabled;
                let _ = self.tx.send(Control::Enabled(self.enabled));
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected =
                    (self.selected + 1).min(self.snapshot.records.len().saturating_sub(1))
            }
            KeyCode::PageUp => self.prompt = self.prompt.saturating_sub(1),
            KeyCode::PageDown => self.prompt = (self.prompt + 1).min(7),
            _ => {}
        }
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        if area.width < 3 || area.height < 3 {
            return;
        }
        let narrow = area.width < 80;
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(if area.height < 20 {
                    2
                } else if narrow {
                    5
                } else {
                    4
                }),
                // Keep eight data rows after chart chrome on the 120x40 shell,
                // without stealing the selected answer or prompt controls at 80x24.
                Constraint::Length(if area.height >= 26 { 13 } else { 8 }),
                Constraint::Min(3),
                Constraint::Length(if area.height < 20 { 1 } else { 2 }),
            ])
            .split(area);
        let status = if !self.enabled {
            "PAUSED"
        } else if self.snapshot.busy {
            "SCORING"
        } else {
            "WATCHING"
        };
        let source = self
            .edit
            .as_deref()
            .map(|s| format!("Prompt file ▶ {s}  [Enter apply / Esc cancel]"))
            .unwrap_or_else(|| {
                format!(
                    "Prompts: {}",
                    self.prompt_file
                        .as_deref()
                        .map(|p| clean(&p.to_string_lossy(), 256))
                        .unwrap_or_else(|| "embedded assets/eval_prompts.json".into())
                )
            });
        if parts[0].height < 4 {
            // A two-row bordered panel has no content; keep prompt editing usable.
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(format!(
                        "AUTO EVAL / {status} / {}",
                        clean(&self.snapshot.note, 512)
                    )),
                    Line::from(source),
                ])
                .style(accent()),
                parts[0],
            );
        } else {
            let status_area = panel_area(f, parts[0]);
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(source),
                    Line::from(clean(&self.snapshot.note, 512)),
                ])
                .style(accent())
                .wrap(Wrap { trim: false })
                .block(panel(&format!(" AUTO EVAL / {status} "))),
                status_area,
            );
        }
        self.draw_chart(f, parts[1]);
        let selected = self.snapshot.records.get(self.selected);
        let previous = self
            .selected
            .checked_sub(1)
            .and_then(|i| self.snapshot.records.get(i));
        if narrow {
            self.draw_answer(f, parts[2], selected, "selected answer");
        } else {
            let answers = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(parts[2]);
            self.draw_answer(f, answers[0], previous, "previous answer");
            self.draw_answer(f, answers[1], selected, "selected answer");
        }
        let hint = if self.editing() {
            "Prompt JSON: version=1, max_new_tokens=1..32, prompts=[{id,prompt,reference}]"
        } else if narrow {
            "a pause  p prompts  r reload\n↑↓ checkpoint  PgUp/Dn prompt"
        } else {
            "a pause  p prompt file  r reload  ↑↓ checkpoint  PgUp/PgDn prompt\nCPU only / 1 worker / loops=1 / NLL is tokenizer-specific, not answer correctness"
        };
        f.render_widget(
            Paragraph::new(hint).style(Style::new().fg(SECOND_ACCENT)),
            parts[3],
        );
    }

    fn draw_chart(&self, f: &mut Frame, area: Rect) {
        // History can contain several fingerprints of a replaced checkpoint.
        // Plot its latest score once, not a vertical segment through stale
        // revisions. Keep the full history available to the answer browser.
        let latest: std::collections::BTreeMap<_, _> = self
            .snapshot
            .records
            .iter()
            .filter(|r| r.suite == self.snapshot.suite)
            .map(|r| (r.number, r.mean_nll().unwrap_or(f64::NAN)))
            .collect();
        let points: Vec<_> = latest.into_iter().map(|(n, y)| (n as f64, y)).collect();
        if !points.iter().any(|p| p.1.is_finite()) {
            let chart_area = panel_area(f, area);
            f.render_widget(Paragraph::new(format!("Reference NLL ↓ better\n{}\n{}\nUnknown words / invalid or oversized checkpoints are skipped.", self.run_context, self.snapshot.note))
                .style(accent()).wrap(Wrap { trim: true }).block(panel(" quality / checkpoint ")), chart_area);
            return;
        }
        charts::draw(
            f,
            area,
            charts::Plot {
                title: " quality / checkpoint ",
                caption: "Lower = less surprise on fixed reference answers, not correctness",
                x: "checkpoint",
                integer_x: true,
                y: "reference loss",
                x_bounds: charts::domain(&points),
                y_bounds: charts::bounds(points.iter().map(|p| p.1)),
            },
            &[charts::Series::line(
                "reference loss",
                &points,
                NORMAL_GREEN,
            )],
        );
    }

    fn draw_answer(&self, f: &mut Frame, area: Rect, record: Option<&Record>, label: &str) {
        let Some(record) = record else {
            let answer_area = panel_area(f, area);
            f.render_widget(Paragraph::new(format!("No checkpoint answer yet.\n{}\n{}\nResults are saved as auto-eval.jsonl next to checkpoints.", self.run_context, self.snapshot.note))
                .style(Style::new().fg(SECOND_ACCENT)).wrap(Wrap { trim: false }).block(panel(label)), answer_area);
            return;
        };
        let mut lines = vec![Line::from(format!(
            "#{} {}  step {}",
            record.number,
            record.model,
            record
                .step
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into())
        ))];
        if let Some(sample) = record
            .samples
            .get(self.prompt.min(record.samples.len().saturating_sub(1)))
        {
            lines.push(Line::from(format!("{}: {}", sample.id, sample.prompt)));
            lines.push(Line::styled(
                format!("Reference: {}", sample.reference),
                Style::new().fg(SECOND_ACCENT),
            ));
            lines.push(Line::from(match sample.nll {
                Some(nll) => format!(
                    "NLL {}  perplexity {}  n={}",
                    charts::number(nll),
                    if nll.exp().is_finite() {
                        charts::number(nll.exp())
                    } else {
                        "overflow".into()
                    },
                    sample.tokens
                ),
                None => "Reference score skipped".into(),
            }));
            lines.push(Line::from(format!("Answer: {}", sample.answer)));
            if !sample.note.is_empty() {
                lines.push(Line::styled(sample.note.clone(), Style::new().fg(AMBER)));
            }
        } else {
            lines.push(Line::styled(record.note.clone(), Style::new().fg(AMBER)));
        }
        let answer_area = panel_area(f, area);
        f.render_widget(
            Paragraph::new(lines)
                .style(accent())
                .wrap(Wrap { trim: false })
                .block(panel(label)),
            answer_area,
        );
    }
}
impl Drop for Eval {
    fn drop(&mut self) {
        let _ = self.tx.send(Control::Stop);
        if let Some(thread) = self.controller.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn suite() -> Suite {
        Suite::parse(
            r#"{"version":1,"max_new_tokens":2,"prompts":[
            {"id":"tiny","prompt":"hello world","reference":" world hello"}] }"#,
        )
        .unwrap()
    }

    fn record(number: u64) -> Record {
        let suite = suite();
        let prompt = &suite.prompts[0];
        Record {
            checkpoint: format!("/tmp/chain with spaces/ck{number}.pssa"),
            fingerprint: format!("10:{number}"),
            number,
            suite: suite.id,
            model: "PSSA".into(),
            step: Some(number),
            samples: vec![Sample {
                id: prompt.id.clone(),
                prompt: prompt.prompt.clone(),
                reference: prompt.reference.clone(),
                answer: format!("answer-{number}"),
                nll: Some(3.0 / number.max(1) as f64),
                tokens: 2,
                note: String::new(),
            }],
            note: "fixed suite".into(),
        }
    }

    fn screen() -> (Eval, mpsc::Receiver<Control>) {
        // Rendering and key tests never start the controller or any subprocess.
        let (tx, controls) = mpsc::channel();
        let (_, rx) = mpsc::sync_channel(1);
        (
            Eval {
                tx,
                rx,
                controller: None,
                watched: None,
                prompt_file: None,
                edit: None,
                snapshot: Snapshot {
                    records: vec![record(1), record(2)],
                    note: "2 checkpoints scored".into(),
                    suite: suite().id,
                    busy: false,
                },
                enabled: true,
                selected: 1,
                prompt: 0,
                run_context: "First checkpoint at run end (step 500); current step 120".into(),
            },
            controls,
        )
    }

    fn render(eval: &Eval, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| eval.draw(f, f.area())).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn fixed_suite_parsing_has_stable_identity_and_strict_bounds() {
        assert_eq!(Suite::parse(DEFAULT_PROMPTS).unwrap().prompts.len(), 3);
        let original = suite();
        let formatted = serde_json::to_string_pretty(&original.json()).unwrap();
        assert_eq!(Suite::parse(&formatted).unwrap().id, original.id);
        let mut changed = original.json();
        changed["prompts"][0]["reference"] = json!("hello");
        assert_ne!(Suite::parse(&changed.to_string()).unwrap().id, original.id);
        for invalid in ["{}", "[", r#"{"version":1,"prompts":[]}"#] {
            assert!(Suite::parse(invalid).is_err());
        }
        for value in [json!(0), json!(33), json!(-1), json!("4"), Value::Null] {
            let mut invalid = original.json();
            invalid["max_new_tokens"] = value;
            assert!(Suite::parse(&invalid.to_string()).is_err());
        }
        let mut invalid = original.json();
        invalid["prompts"][0]["prompt"] = json!("a".repeat(513));
        assert!(Suite::parse(&invalid.to_string()).is_err());
        invalid["prompts"][0]["prompt"] = json!("\u{1b}[31m");
        assert!(Suite::parse(&invalid.to_string()).is_err());
        let mut duplicate = original.json();
        duplicate["prompts"]
            .as_array_mut()
            .unwrap()
            .push(original.json()["prompts"][0].clone());
        assert!(Suite::parse(&duplicate.to_string()).is_err());
    }

    #[test]
    fn prompt_file_with_spaces_loads_and_missing_or_oversized_files_fail() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("my eval prompts.json");
        fs::write(&path, suite().json().to_string()).unwrap();
        assert_eq!(Suite::load(Some(&path)).unwrap().id, suite().id);
        fs::write(&path, vec![b' '; MAX_JSON + 1]).unwrap();
        assert!(Suite::load(Some(&path)).is_err());
        assert!(Suite::load(Some(&dir.0.join("missing.json"))).is_err());
    }

    #[test]
    fn persistence_round_trips_answers_and_metrics_and_recovers_torn_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join(HISTORY_FILE);
        append_record(&path, &record(1)).unwrap();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"torn\":").unwrap();
        drop(file);
        append_record(&path, &record(2)).unwrap();
        let (loaded, invalid) = load_history(&path).unwrap();
        assert_eq!(invalid, 1);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].key(), record(1).key());
        assert_eq!(loaded[1].samples[0].answer, "answer-2");
        assert_eq!(loaded[1].mean_nll(), record(2).mean_nll());
        let raw = read_small(&path, MAX_JSON).unwrap();
        assert!(raw.contains("reference_perplexity"));
        assert!(raw.contains("checkpoint_number"));
        assert!(!raw.contains("NaN"));
    }

    #[test]
    fn oversized_and_invalid_history_lines_do_not_hide_later_valid_results() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join(HISTORY_FILE);
        fs::write(&path, format!("{}\nnot json\n", "x".repeat(MAX_JSON + 1))).unwrap();
        append_record(&path, &record(3)).unwrap();
        let (records, invalid) = load_history(&path).unwrap();
        assert_eq!(invalid, 2);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].number, 3);
        let mut bad = record(3).json();
        bad["samples"][0]["reference_nll"] = json!(-1);
        assert!(Record::parse(&bad.to_string()).is_err());
    }

    #[test]
    fn reference_score_is_stable_and_partial_suites_are_not_charted() {
        // Adding the large maximum before subtracting the target would erase
        // ln(3) for otherwise valid finite checkpoints with very large logits.
        for logit in [0.0, 1000.0, 1e20, f32::MAX, -f32::MAX] {
            let nll = reference_nll(&[logit; 3], 1).unwrap();
            assert!((nll - 3.0f64.ln()).abs() < 1e-10, "logit={logit}");
        }
        assert_eq!(reference_nll(&[f32::MAX, -f32::MAX], 0).unwrap(), 0.0);
        assert!(reference_nll(&[f32::NAN], 0).is_err());
        assert!(reference_nll(&[0.0], 2).is_err());
        let mut partial = record(1);
        let mut skipped = partial.samples[0].clone();
        skipped.nll = None;
        skipped.tokens = 0;
        partial.samples.push(skipped);
        assert_eq!(partial.mean_nll(), None);
        partial.samples[0].nll = Some(1000.0);
        assert!(partial.json()["samples"][0]["reference_perplexity"].is_null());
    }

    fn pssa_checkpoint(path: &Path, depth: usize) {
        let mut model = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: 3,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 2,
                chunk_len: 2,
                depth,
                ..Default::default()
            },
            7,
        );
        model.vocabulary = ["<unk>", "hello", "world"].map(str::to_owned).to_vec();
        model.unembed_w.data.fill(0.0);
        checkpoint::save_model(&model, path).unwrap();
    }

    #[test]
    fn tiny_pssa_and_stacked_checkpoints_score_reference_and_keep_bytes_unchanged() {
        let dir = TempDir::new().unwrap();
        for depth in [1, 2] {
            let path = dir.0.join(format!("ck{depth}.pssa"));
            pssa_checkpoint(&path, depth);
            let original = fs::read(&path).unwrap();
            let scratch = TempDir::new().unwrap();
            let result = evaluate_checkpoint(&path, &suite(), &scratch.0, false);
            assert_eq!(result.model, "PSSA");
            assert_eq!(result.samples.len(), 1, "{}", result.note);
            assert_eq!(result.samples[0].answer, "hello hello");
            assert_eq!(result.samples[0].tokens, 2); // not prompt transitions
            assert!((result.samples[0].nll.unwrap() - 3.0f64.ln()).abs() < 1e-10);
            assert_eq!(fs::read(&path).unwrap(), original);
        }
    }

    #[test]
    fn tiny_transformer_checkpoint_uses_the_same_reference_metric() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("ck12.trfm");
        let mut model = TransformerModel::new(
            TransformerConfig {
                d_vocab: 3,
                d_model: 4,
                n_heads: 1,
                d_ff: 8,
                chunk_len: 4,
                ..Default::default()
            },
            7,
        )
        .unwrap();
        model.vocabulary = ["<unk>", "hello", "world"].map(str::to_owned).to_vec();
        model.unembed.data.fill(0.0);
        crate::transformer_checkpoint::save_model(&model, &path).unwrap();
        let original = fs::read(&path).unwrap();
        let scratch = TempDir::new().unwrap();
        let result = evaluate_checkpoint(&path, &suite(), &scratch.0, false);
        assert_eq!(result.number, 12);
        assert_eq!(result.model, "transformer");
        assert_eq!(result.samples.len(), 1, "{}", result.note);
        assert_eq!(result.samples[0].answer, "hello hello");
        assert!((result.mean_nll().unwrap() - 3.0f64.ln()).abs() < 1e-10);
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn transformer_answers_match_normal_greedy_generation_sentence_stop() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("ck13.trfm");
        let mut model = TransformerModel::new(
            TransformerConfig {
                d_vocab: 3,
                d_model: 4,
                n_heads: 1,
                d_ff: 8,
                chunk_len: 4,
                ..Default::default()
            },
            7,
        )
        .unwrap();
        model.vocabulary = ["<unk>", ".", "hello"].map(str::to_owned).to_vec();
        model.unembed.data.fill(0.0);
        crate::transformer_checkpoint::save_model(&model, &path).unwrap();
        let suite = Suite::parse(
            r#"{"version":1,"max_new_tokens":8,"prompts":[
            {"id":"stop","prompt":"hello","reference":" hello"}]}"#,
        )
        .unwrap();
        let expected = crate::transformer_inference::generate(
            path.to_str().unwrap(),
            "hello",
            &InferenceConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                repetition_penalty: 1.0,
                max_new_tokens: suite.max_new_tokens,
            },
        )
        .unwrap();
        assert_eq!(expected, "..");
        let scratch = TempDir::new().unwrap();
        let result = evaluate_checkpoint(&path, &suite, &scratch.0, false);
        assert_eq!(result.samples.len(), 1, "{}", result.note);
        assert_eq!(result.samples[0].answer, expected);
        assert!((result.mean_nll().unwrap() - 3.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn unknown_reference_and_invalid_checkpoints_are_persistable_skips() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("ck1.pssa");
        pssa_checkpoint(&path, 1);
        let mut prompts = suite();
        prompts.prompts[0].reference = "unknown".into();
        let scratch = TempDir::new().unwrap();
        let result = evaluate_checkpoint(&path, &prompts, &scratch.0, false);
        assert!(result.mean_nll().is_none());
        assert!(result.samples[0].note.contains("out-of-vocabulary"));
        assert_eq!(result.samples[0].answer, "hello hello");
        fs::write(&path, b"not a checkpoint").unwrap();
        let scratch = TempDir::new().unwrap();
        let skipped = evaluate_checkpoint(&path, &suite(), &scratch.0, false);
        assert!(skipped.samples.is_empty());
        assert!(skipped.note.starts_with("Skipped:"));
        append_record(&dir.0.join(HISTORY_FILE), &skipped).unwrap();
        assert_eq!(load_history(&dir.0.join(HISTORY_FILE)).unwrap().0.len(), 1);
    }

    #[test]
    fn checksum_corruption_is_a_read_only_persistable_skip() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("ck4.pssa");
        pssa_checkpoint(&path, 1);
        let mut damaged = fs::read(&path).unwrap();
        *damaged.last_mut().unwrap() ^= 1;
        fs::write(&path, &damaged).unwrap();
        assert_eq!(preflight(&path).unwrap(), "PSSA");
        let scratch = TempDir::new().unwrap();
        let skipped = evaluate_checkpoint(&path, &suite(), &scratch.0, false);
        assert!(skipped.samples.is_empty());
        assert!(skipped.note.contains("checksum"), "{}", skipped.note);
        assert_eq!(fs::read(&path).unwrap(), damaged);
        let history = dir.0.join(HISTORY_FILE);
        append_record(&history, &skipped).unwrap();
        let (loaded, invalid) = load_history(&history).unwrap();
        assert_eq!(invalid, 0);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].number, 4);
        assert!(loaded[0].mean_nll().is_none());
    }

    #[test]
    fn allocation_guard_rejects_large_tapes_and_sparse_oversized_files_before_loading() {
        let dir = TempDir::new().unwrap();
        let path = dir.0.join("huge.trfm");
        let file = File::create(&path).unwrap();
        file.set_len(MAX_FILE + 1).unwrap();
        assert!(preflight(&path).unwrap_err().contains("64 MiB"));
        let mut head = vec![0; 106];
        head[..4].copy_from_slice(b"TRFM");
        head[4..6].copy_from_slice(&1u16.to_le_bytes());
        for (index, value) in [3u64, 4, 1, 8, 8192].iter().enumerate() {
            head[22 + index * 8..30 + index * 8].copy_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, head).unwrap();
        assert!(preflight(&path).unwrap_err().contains("128 MiB"));
    }

    #[test]
    fn discovery_is_numeric_and_deduplicates_progress_path_aliases() {
        let dir = TempDir::new().unwrap();
        for name in ["ck10.pssa", "ck2.trfm", "ignore.txt", "ck1.pssa.tmp"] {
            fs::write(dir.0.join(name), b"fixture").unwrap();
        }
        let (paths, capped) = discover(&dir.0, Some(&dir.0.join("ck2.trfm")));
        assert!(!capped);
        assert_eq!(paths.len(), 2);
        assert_eq!(checkpoint_number(&paths[0]), Some(2));
        assert_eq!(checkpoint_number(&paths[1]), Some(10));
        assert_eq!(
            checkpoint_number(Path::new("a path/checkpoint-004.pssa")),
            Some(4)
        );
        assert_eq!(checkpoint_number(Path::new("model.pssa")), None);
    }

    #[test]
    fn hidden_worker_hook_does_not_intercept_ordinary_tui_arguments() {
        assert!(run_worker(&[]).is_none());
        assert!(run_worker(&["--chain".into(), "a directory".into()]).is_none());
        assert!(run_worker(&[WORKER_ARG.into()]).unwrap().is_err());
    }

    #[test]
    fn watch_sends_only_changes_and_editor_preserves_global_shortcut_characters() {
        let (mut eval, controls) = screen();
        eval.watch(Path::new("chain with spaces"), Some(Path::new("ck1.pssa")));
        eval.watch(Path::new("chain with spaces"), Some(Path::new("ck1.pssa")));
        assert!(matches!(controls.try_recv().unwrap(), Control::Watch(_, _)));
        assert!(controls.try_recv().is_err());
        eval.key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE));
        assert!(eval.editing());
        for c in "q? a.json".chars() {
            eval.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        eval.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(eval.prompt_file(), Some(Path::new("q? a.json")));
        assert!(matches!(
            controls.try_recv().unwrap(),
            Control::Configure(Some(_))
        ));
        eval.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        assert!(!eval.enabled);
        assert!(matches!(
            controls.try_recv().unwrap(),
            Control::Enabled(false)
        ));
    }

    #[test]
    fn wide_render_has_chart_and_adjacent_checkpoint_answers_with_closed_borders() {
        let (eval, _) = screen();
        let text = render(&eval, 110, 30);
        for expected in [
            "AUTO EVAL",
            "quality / checkpoint",
            "previous answer",
            "selected answer",
            "answer-1",
            "answer-2",
            "Reference:",
            "NLL",
        ] {
            assert!(text.contains(expected), "missing {expected}:\n{text}");
        }
        let mut terminal = Terminal::new(TestBackend::new(110, 30)).unwrap();
        terminal.draw(|f| eval.draw(f, f.area())).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(0, 0)].symbol(), "┌");
        assert_eq!(buffer[(109, 0)].symbol(), "┐");
        assert_eq!(buffer[(0, 3)].symbol(), "└");
        assert_eq!(buffer[(109, 3)].symbol(), "┘");
        assert!(buffer.content().iter().any(|cell| cell.fg == NORMAL_GREEN));
    }

    #[test]
    fn quality_chart_has_braille_axes_at_both_sizes() {
        let (mut eval, _) = screen();
        eval.snapshot.records.push(record(3));
        eval.edit = Some("custom-prompts.json".into());
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| eval.draw(f, f.area())).unwrap();
            charts::assert_named_plot(
                terminal.backend().buffer(),
                "quality / checkpoint",
                &[
                    "quality / checkpoint",
                    "checkpoint",
                    "reference loss",
                    "Lower = less surprise",
                    "1",
                    "2",
                    "3",
                ],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "quality / checkpoint",
                if h == 40 { 8 } else { 3 },
            );
            terminal
                .draw(|f| eval.draw(f, super::super::feature_area(f.area())))
                .unwrap();
            charts::assert_named_plot(
                terminal.backend().buffer(),
                "quality / checkpoint",
                &["checkpoint", "reference loss", "Lower = less surprise"],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "quality / checkpoint",
                if h == 40 { 8 } else { 3 },
            );
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(text.contains("AUTO EVAL / WATCHING"));
            assert!(text.contains("Prompt file ▶ custom-prompts.json"));
        }
    }

    #[test]
    fn quality_chart_uses_latest_revision_without_a_checkpoint_one_spike() {
        let (mut eval, _) = screen();
        let mut stale = record(1);
        stale.fingerprint = "old-revision".into();
        stale.samples[0].nll = Some(6.0);
        let mut other_suite = record(2);
        other_suite.suite = "other-suite".into();
        other_suite.samples[0].nll = Some(100.0);
        for (w, h) in [(120, 40), (80, 24)] {
            for missing_latest in [false, true] {
                let mut current = vec![record(1), record(2), record(3)];
                if missing_latest {
                    // A skipped new revision is a gap, not its old valid score.
                    current[0].samples[0].nll = None;
                }
                let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
                let area = Rect::new(2, 3, w - 4, if h == 40 { 13 } else { 8 });
                eval.snapshot.records = current.clone();
                terminal.draw(|f| eval.draw_chart(f, area)).unwrap();
                let expected = terminal.backend().buffer().clone();
                eval.snapshot.records.insert(0, stale.clone());
                eval.snapshot.records.push(other_suite.clone());
                terminal.draw(|f| eval.draw_chart(f, area)).unwrap();
                assert_eq!(
                    terminal.backend().buffer(),
                    &expected,
                    "stale revisions or another suite changed the chart at {w}x{h}"
                );
                assert_eq!(eval.snapshot.records.len(), 5, "history must be preserved");
            }
        }
    }

    #[test]
    fn narrow_and_tiny_renders_keep_selected_answer_and_do_not_panic() {
        let (mut eval, _) = screen();
        let text = render(&eval, 60, 28);
        assert!(text.contains("AUTO EVAL"));
        assert!(text.contains("answer-2"));
        assert!(!text.contains("previous answer"));
        for (width, height) in [(79, 20), (40, 15), (20, 8), (3, 3), (1, 1)] {
            render(&eval, width, height);
        }
        eval.snapshot.records.clear();
        let empty = render(&eval, 60, 28);
        assert!(empty.contains("Waiting for a fully scoreable"));
        assert!(empty.contains("No checkpoint answer yet"));
    }

    #[cfg(unix)]
    #[test]
    fn history_cannot_follow_a_symlink_into_a_checkpoint() {
        let dir = TempDir::new().unwrap();
        let checkpoint = dir.0.join("ck1.pssa");
        fs::write(&checkpoint, b"read-only checkpoint fixture").unwrap();
        let history = dir.0.join(HISTORY_FILE);
        std::os::unix::fs::symlink(&checkpoint, &history).unwrap();
        assert!(append_record(&history, &record(1)).is_err());
        assert_eq!(
            fs::read(&checkpoint).unwrap(),
            b"read-only checkpoint fixture"
        );
    }

    #[cfg(unix)]
    #[test]
    fn dropping_the_only_job_kills_and_reaps_its_child() {
        let scratch = TempDir::new().unwrap();
        let child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        let path = scratch.0.clone();
        let job = Job {
            child,
            scratch,
            record: record(1),
            started: Instant::now(),
        };
        drop(job);
        assert!(!path.exists());
        #[cfg(target_os = "linux")]
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
        #[cfg(not(target_os = "linux"))]
        let _ = pid;
    }
}
