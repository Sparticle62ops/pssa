mod backend_choice;
pub use backend_choice::TrainingBackend;
pub mod resource_limits;

use crate::backend::{Device, gemm_cpu_reference};
use crate::checkpoint::{self, CheckpointFormat};
use crate::dataset::{
    DatasetManager, HuggingFaceDatasetOptions, Tokenizer, TokenizerKind, clean_wikitext,
};
use crate::dream::DreamMode;
use crate::inference::{InferenceConfig, PSSAInferenceEngine};
use crate::linalg::SimpleRng;
use crate::pssa::{GradientClipOutcome, PSSAConfigV2, PSSALayerV2};
use crate::token_cache;
use crate::ui;
use std::collections::{HashMap, HashSet};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Keep command-line generation requests bounded before inference preallocates
/// its token and byte buffers.
const MAX_GENERATION_TOKENS: usize = 100_000;

/// Opt-in clipping tolerates isolated bad gradients, not sustained divergence.
const MAX_CONSECUTIVE_SKIPPED_UPDATES: usize = 20;

fn dream_phase_start_line(update: usize, mode: DreamMode) -> String {
    format!("dream phase=start update={update} mode={}", mode.as_str())
}

fn dream_phase_end_line(
    update: usize,
    mode: DreamMode,
    summary: &crate::dream::DreamSummary,
) -> String {
    let dream_loss = summary
        .last_rehearsal_loss
        .filter(|loss| loss.is_finite())
        .map_or_else(|| "unavailable".to_owned(), |loss| format!("{loss:.6e}"));
    format!(
        "dream phase=end update={update} mode={} entries_replayed={} generated_tokens={} rehearsal_sequences={} consolidation_delta_norm={:.6e} time={:.3}s dream_loss={dream_loss}",
        mode.as_str(),
        summary.entries_replayed,
        summary.generated_tokens,
        summary.rehearsal_sequences,
        summary.consolidation_delta_norm,
        summary.elapsed_seconds,
    )
}

#[derive(Default)]
struct SkippedUpdates {
    total: usize,
    consecutive: usize,
}
impl SkippedUpdates {
    fn record(&mut self, norm: f64, step: usize) -> Result<(), String> {
        self.total += 1;
        self.consecutive += 1;
        eprintln!(
            "warning: non-finite grad_norm={norm:.6e}; skipped optimizer update at global_step={step} skipped_updates={} consecutive_skips={}/{} (gradients cleared; Adam unchanged)",
            self.total, self.consecutive, MAX_CONSECUTIVE_SKIPPED_UPDATES
        );
        if self.consecutive > MAX_CONSECUTIVE_SKIPPED_UPDATES {
            Err(format!(
                "more than {MAX_CONSECUTIVE_SKIPPED_UPDATES} consecutive non-finite gradient updates; training aborted without checkpoint (skipped_updates={})",
                self.total
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingOptions {
    /// Runtime-only selection; automatic GPU/CPU fallback remains the default.
    pub backend: TrainingBackend,
    pub epochs: usize,
    pub latent: usize,
    /// Continuous PSSA blocks, sharing one embedding and output head.
    pub depth: usize,
    /// Runtime-only repeated passes through each shared continuous block.
    pub loops: usize,
    /// Whether the runtime loop count was explicitly supplied by the CLI.
    pub loops_explicit: bool,
    pub state: usize,
    pub key: usize,
    pub memory: usize,
    pub chunk: usize,
    pub lr: f32,
    pub accumulate: usize,
    /// Runtime-only global L2 gradient limit; None preserves historical Adam.
    pub grad_clip: Option<f32>,
    /// Runtime-only per-vector episodic memory L2 limit; repeat on resume.
    pub memory_value_cap: Option<f32>,
    /// Run an offline memory replay after every N optimizer updates; zero is off.
    pub dream_every: usize,
    /// Maximum number of occupied episodic entries replayed per dream.
    pub dream_replay: usize,
    /// Offline replay source; memory is the Stage 1-compatible default.
    pub dream_mode: DreamMode,
    /// Generated tokens per memory seed in generate/both modes.
    pub dream_len: usize,
    /// Learning rate for the separate main-weight SGD rehearsal pass.
    pub dream_lr: f32,
    /// Number of SGD rehearsal passes per stored/generated sequence.
    pub dream_steps: usize,
    /// Independent document lanes per PSSA microbatch (runtime-only).
    pub batch_size: usize,
    pub warmup_steps: usize,
    /// Fixed total optimizer-update horizon shared by all resumed links.
    /// `None` preserves the legacy per-link schedule behavior.
    pub schedule_total_updates: Option<usize>,
    pub seed: u64,
    /// A global cap across documents; documents are never individually reset to this cap.
    pub max_tokens: Option<usize>,
    /// Optional streamed Hugging Face dataset source. These fields are
    /// runtime-only and are deliberately not written into checkpoints.
    pub hf_dataset: Option<String>,
    pub hf_config: Option<String>,
    pub hf_split: String,
    pub hf_field: String,
    pub tokenizer: TokenizerKind,
    pub vocab_size: usize,
    /// Continue training from an existing checkpoint instead of fresh initialization.
    pub resume: Option<String>,
    /// Skip this many encoded tokens from the front of the corpus before training.
    pub skip_tokens: usize,
    /// Optional persistent token cache. Runtime-only; never checkpointed.
    pub token_cache: Option<String>,
    /// Local source used for the cache's size/mtime fingerprint.
    pub token_cache_source: Option<String>,
    /// Optional append-only, target-token-weighted training curve.
    pub loss_csv: Option<String>,
    pub loss_every: usize,
    /// Explicit target-token offset when starting a new CSV from a checkpoint.
    pub tokens_seen: Option<usize>,
    /// Disable cursor control while retaining rate-limited plain progress logs.
    pub no_tui: bool,
    /// Runtime-only destination shown by the progress display; never checkpointed.
    pub checkpoint_path: Option<String>,
}
impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::Auto,
            epochs: 4,
            latent: 256,
            depth: 1,
            loops: 1,
            loops_explicit: false,
            state: 16,
            key: 32,
            memory: 512,
            chunk: 64,
            lr: 1e-3,
            accumulate: 8,
            grad_clip: None,
            memory_value_cap: None,
            dream_every: 0,
            dream_replay: 32,
            dream_mode: DreamMode::Memory,
            dream_len: 64,
            dream_lr: crate::dream::DEFAULT_REHEARSAL_LR,
            dream_steps: crate::dream::DEFAULT_REHEARSAL_STEPS,
            batch_size: 1,
            warmup_steps: 0,
            schedule_total_updates: None,
            seed: 42,
            max_tokens: None,
            hf_dataset: None,
            hf_config: None,
            hf_split: "train".into(),
            hf_field: "text".into(),
            skip_tokens: 0,
            token_cache: None,
            token_cache_source: None,
            tokenizer: TokenizerKind::Bpe,
            vocab_size: 2048,
            resume: None,
            loss_csv: None,
            loss_every: 10_000,
            tokens_seen: None,
            no_tui: false,
            checkpoint_path: None,
        }
    }
}

/// Linear warm-up followed by cosine decay. Update and total are one-based.
pub fn learning_rate_for_update(
    base: f32,
    update: usize,
    total: usize,
    warmup: usize,
) -> Result<f32, String> {
    if !(base.is_finite() && base > 0.0) || total == 0 || update == 0 || update > total {
        return Err("invalid learning-rate schedule inputs".into());
    }
    if warmup >= total && warmup != 0 {
        return Err("warmup-steps must be less than total optimizer updates".into());
    }
    let min = base * 0.01;
    if warmup > 0 && update <= warmup {
        return Ok(base * update as f32 / warmup as f32);
    }
    let progress = if warmup == 0 {
        (update - 1) as f32 / total.saturating_sub(1).max(1) as f32
    } else {
        (update - warmup) as f32 / (total - warmup) as f32
    };
    Ok(min + 0.5 * (base - min) * (1.0 + (std::f32::consts::PI * progress).cos()))
}

#[derive(Debug)]
struct Parsed {
    flags: HashMap<String, String>,
    positional: Vec<String>,
}
impl Parsed {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Self, String> {
        let allowed: HashSet<&str> = allowed.iter().copied().collect();
        let mut flags = HashMap::new();
        let mut positional = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg.starts_with('-') {
                if !allowed.contains(arg.as_str()) {
                    return Err(format!("unknown option '{arg}'"));
                }
                // Boolean switches are deliberately explicit rather than
                // accepting arbitrary valueless options.  Keep the existing
                // value-taking parser for every other flag.
                if arg == "--no-tui" {
                    if flags.insert(arg.clone(), "true".into()).is_some() {
                        return Err(format!("option '{arg}' was specified more than once"));
                    }
                    i += 1;
                    continue;
                }
                let missing_value = match args.get(i + 1) {
                    None => true,
                    Some(value) if !value.starts_with('-') => false,
                    Some(value) if allowed.contains(value.as_str()) => true,
                    // Negative numeric values are legitimate values for
                    // flags such as --temperature; leave their eventual
                    // domain validation to the typed option parser.
                    Some(value) => value.parse::<f64>().is_err(),
                };
                if missing_value {
                    return Err(format!("option '{arg}' requires a value"));
                }
                if flags.insert(arg.clone(), args[i + 1].clone()).is_some() {
                    return Err(format!("option '{arg}' was specified more than once"));
                }
                i += 2;
            } else {
                positional.push(arg.clone());
                i += 1;
            }
        }
        Ok(Self { flags, positional })
    }
    fn string(&self, long: &str, short: &str) -> Option<&str> {
        self.flags
            .get(long)
            .or_else(|| self.flags.get(short))
            .map(String::as_str)
    }
    fn first(&self, names: &[&str]) -> Option<&str> {
        names
            .iter()
            .find_map(|name| self.flags.get(*name).map(String::as_str))
    }
    fn reject_duplicate_aliases(&self, names: &[&str], label: &str) -> Result<(), String> {
        let count = names
            .iter()
            .filter(|name| self.flags.contains_key(**name))
            .count();
        if count > 1 {
            Err(format!(
                "{label} was specified more than once (use one spelling)"
            ))
        } else {
            Ok(())
        }
    }
    fn required_usize(&self, long: &str, short: &str, default: usize) -> Result<usize, String> {
        self.string(long, short).map_or(Ok(default), |x| {
            x.parse()
                .map_err(|_| format!("{long} must be a positive integer"))
        })
    }
    fn usize_nonzero(&self, long: &str, short: &str, default: usize) -> Result<usize, String> {
        let n = self.required_usize(long, short, default)?;
        if n == 0 {
            Err(format!("{long} must be positive"))
        } else {
            Ok(n)
        }
    }
    fn loops(&self) -> Result<usize, String> {
        let loops = self.string("--loops", "").map_or(Ok(1), |value| {
            value
                .parse::<usize>()
                .map_err(|_| "--loops must be an integer between 1 and 32".to_string())
        })?;
        CLIHandler::validate_loops(loops)?;
        Ok(loops)
    }
    fn optional_positive_f32(&self, flag: &str) -> Result<Option<f32>, String> {
        self.string(flag, "")
            .map(|_| {
                let value = self.f32(flag, "", 0.0)?;
                if value > 0.0 {
                    Ok(value)
                } else {
                    Err(format!("{flag} must be finite and positive"))
                }
            })
            .transpose()
    }
    fn f32(&self, long: &str, short: &str, default: f32) -> Result<f32, String> {
        self.f32_first(&[long, short], long, default)
    }
    fn f32_first(&self, names: &[&str], label: &str, default: f32) -> Result<f32, String> {
        let n = self.first(names).map_or(Ok(default), |x| {
            x.parse().map_err(|_| format!("{label} must be a number"))
        })?;
        if n.is_finite() {
            Ok(n)
        } else {
            Err(format!("{label} must be finite"))
        }
    }
}

pub struct CLIHandler;

#[cfg(test)]
pub(crate) fn parse_train_options_for_test(args: &[String]) -> Result<TrainingOptions, String> {
    let allowed = [
        "--data",
        "-d",
        "--out",
        "-o",
        "--epochs",
        "-e",
        "--latent",
        "--state",
        "--key",
        "--memory",
        "--chunk",
        "--lr",
        "--accumulate",
        "--warmup-steps",
        "--total-updates",
        "--seed",
        "--max-tokens",
        "--tokenizer",
        "--vocab-size",
        "--resume",
        "--skip-tokens",
        "--loss-csv",
        "--loss-every",
        "--tokens-seen",
        "--no-tui",
        "--threads",
        "--ram-mib",
        "--batch-size",
        "--depth",
        "--loops",
        "--backend",
        "--grad-clip",
        "--memory-value-cap",
        "--token-cache",
        "--dream-every",
        "--dream-replay",
        "--dream-mode",
        "--dream-len",
        "--dream-lr",
        "--dream-steps",
    ];
    let parsed = Parsed::parse(args, &allowed)?;
    CLIHandler::common_options(&parsed)
}

impl CLIHandler {
    fn validate_loops(loops: usize) -> Result<(), String> {
        if !(1..=PSSALayerV2::MAX_LOOPS).contains(&loops) {
            return Err(
                "--loops must be between 1 and 32; use --loops 1 for the original model".into(),
            );
        }
        Ok(())
    }

    fn default_data() -> String {
        if std::path::Path::new("data/downloaded.txt").exists() {
            "data/downloaded.txt".into()
        } else {
            "science".into()
        }
    }
    fn local_cache_source(source: &str) -> Option<String> {
        if source == "science"
            || source.contains(',')
            || source.starts_with("hf:")
            || source.starts_with("http://")
            || source.starts_with("https://")
        {
            return None;
        }
        let path = source
            .strip_prefix("file:")
            .or_else(|| source.strip_prefix("dir:"))
            .or_else(|| source.strip_prefix("local:"))
            .unwrap_or(source);
        let path = Path::new(path);
        path.is_file().then(|| path.to_string_lossy().into_owned())
    }
    fn run_clean_wikitext(input_path: &str, output_path: &str) -> Result<(), String> {
        let input = std::fs::File::open(input_path).map_err(|e| {
            format!("cannot open input '{input_path}': {e}; provide a readable UTF-8 file")
        })?;
        // create_new also rejects symlinks and hard links to the input, without
        // a check-then-truncate race. Cleaning is intentionally not in-place.
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_path)
            .map_err(|e| {
                format!("cannot create output '{output_path}': {e}; use --out with a new file in an existing writable directory")
            })?;
        let mut writer = BufWriter::new(output);
        let result =
            clean_wikitext(BufReader::new(input), &mut writer).and_then(|()| writer.flush());
        drop(writer);
        if let Err(e) = result {
            let cleanup = match std::fs::remove_file(output_path) {
                Ok(()) => String::new(),
                Err(error) => format!(
                    "; cannot remove partial output: {error}; remove '{output_path}' before retrying"
                ),
            };
            return Err(format!(
                "cannot clean '{input_path}' into '{output_path}': {e}; check input UTF-8 and output disk space/permissions{cleanup}"
            ));
        }
        Ok(())
    }

    /// Historical name retained for callers; current output is V7.
    pub fn save_model_v2(model: &PSSALayerV2, path: &str) -> io::Result<()> {
        checkpoint::save_model(model, path).map_err(|e| io::Error::other(e.to_string()))
    }
    pub fn load_model_v2(path: &str) -> Result<PSSALayerV2, String> {
        checkpoint::load_checkpoint(path)
            .map(|x| x.model)
            .map_err(|e| e.to_string())
    }

    fn common_options(parsed: &Parsed) -> Result<TrainingOptions, String> {
        let tokenizer = match parsed.string("--tokenizer", "").unwrap_or("bpe") {
            "bpe" => TokenizerKind::Bpe,
            "word" => TokenizerKind::Word,
            x => return Err(format!("--tokenizer must be bpe or word, got '{x}'")),
        };
        let x = TrainingOptions {
            backend: TrainingBackend::parse(parsed.string("--backend", "").unwrap_or("auto"))?,
            epochs: parsed.usize_nonzero("--epochs", "-e", 4)?,
            latent: parsed.usize_nonzero("--latent", "", 256)?,
            depth: parsed.usize_nonzero("--depth", "", 1)?,
            loops: parsed.loops()?,
            loops_explicit: parsed.flags.contains_key("--loops"),
            state: parsed.usize_nonzero("--state", "", 16)?,
            key: parsed.usize_nonzero("--key", "", 32)?,
            memory: parsed.usize_nonzero("--memory", "", 512)?,
            chunk: parsed.usize_nonzero("--chunk", "", 64)?,
            lr: parsed.f32("--lr", "", 1e-3)?,
            accumulate: parsed.usize_nonzero("--accumulate", "", 8)?,
            grad_clip: parsed.optional_positive_f32("--grad-clip")?,
            memory_value_cap: parsed.optional_positive_f32("--memory-value-cap")?,
            dream_every: parsed.required_usize("--dream-every", "", 0)?,
            dream_replay: parsed.usize_nonzero("--dream-replay", "", 32)?,
            dream_mode: DreamMode::parse(parsed.string("--dream-mode", "").unwrap_or("memory"))?,
            dream_len: parsed.usize_nonzero("--dream-len", "", 64)?,
            dream_lr: parsed.f32("--dream-lr", "", crate::dream::DEFAULT_REHEARSAL_LR)?,
            dream_steps: parsed.usize_nonzero(
                "--dream-steps",
                "",
                crate::dream::DEFAULT_REHEARSAL_STEPS,
            )?,
            batch_size: parsed.usize_nonzero("--batch-size", "", 1)?,
            warmup_steps: parsed.required_usize("--warmup-steps", "", 0)?,
            schedule_total_updates: parsed
                .string("--total-updates", "")
                .map(|x| {
                    x.parse::<usize>()
                        .map_err(|_| "--total-updates must be a positive integer".to_string())
                })
                .transpose()?,
            seed: parsed.string("--seed", "").map_or(Ok(42), |x| {
                x.parse()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())
            })?,
            max_tokens: parsed
                .string("--max-tokens", "")
                .map(|x| {
                    x.parse::<usize>()
                        .map_err(|_| "--max-tokens must be a positive integer".to_string())
                })
                .transpose()?,
            hf_dataset: parsed.string("--hf-dataset", "").map(str::to_string),
            hf_config: parsed.string("--hf-config", "").map(str::to_string),
            hf_split: parsed
                .string("--hf-split", "")
                .unwrap_or("train")
                .to_string(),
            hf_field: parsed
                .string("--hf-field", "")
                .unwrap_or("text")
                .to_string(),
            tokenizer,
            vocab_size: parsed.usize_nonzero("--vocab-size", "", 2048)?,
            resume: parsed.string("--resume", "").map(str::to_string),
            skip_tokens: parsed.required_usize("--skip-tokens", "", 0)?,
            token_cache: parsed.string("--token-cache", "").map(str::to_string),
            token_cache_source: None,
            loss_csv: parsed.string("--loss-csv", "").map(str::to_string),
            loss_every: parsed.usize_nonzero("--loss-every", "", 10_000)?,
            tokens_seen: parsed
                .string("--tokens-seen", "")
                .map(|s| {
                    s.parse()
                        .map_err(|_| "--tokens-seen must be an unsigned integer".to_string())
                })
                .transpose()?,
            no_tui: parsed.flags.contains_key("--no-tui"),
            checkpoint_path: None,
        };
        if x.loss_csv.is_none()
            && (parsed.flags.contains_key("--loss-every") || x.tokens_seen.is_some())
        {
            return Err("--loss-every and --tokens-seen require --loss-csv PATH".into());
        }
        if let Some(dataset) = x.hf_dataset.as_deref() {
            DatasetManager::validate_huggingface_dataset_name(dataset)?;
            if x.hf_split.trim().is_empty() {
                return Err("--hf-split must not be empty".into());
            }
            if x.hf_field.trim().is_empty() {
                return Err("--hf-field must not be empty".into());
            }
            if x.hf_config
                .as_deref()
                .is_some_and(|config| config.trim().is_empty())
            {
                return Err("--hf-config must not be empty".into());
            }
        } else if x.hf_config.is_some()
            || parsed.flags.contains_key("--hf-split")
            || parsed.flags.contains_key("--hf-field")
        {
            return Err("--hf-config, --hf-split, and --hf-field require --hf-dataset".into());
        }
        if !(x.lr > 0.0) {
            return Err("--lr must be positive".into());
        }
        if !x.dream_lr.is_finite() || x.dream_lr <= 0.0 {
            return Err("--dream-lr must be finite and positive".into());
        }
        if x.max_tokens == Some(0) {
            return Err("--max-tokens must be positive".into());
        }
        if x.schedule_total_updates == Some(0) {
            return Err("--total-updates must be positive".into());
        }
        if x.tokenizer == TokenizerKind::Bpe && x.vocab_size < 257 {
            return Err("--vocab-size must be at least 257 for byte-level BPE".into());
        }
        for (name, value, maximum) in [
            ("--epochs", x.epochs, 1_000_000usize),
            ("--latent", x.latent, 4_096),
            ("--depth", x.depth, 32),
            ("--state", x.state, 4_096),
            ("--key", x.key, 4_096),
            ("--memory", x.memory, 1_000_000),
            ("--chunk", x.chunk, 65_536),
            ("--accumulate", x.accumulate, 1_000_000),
            ("--dream-every", x.dream_every, 1_000_000),
            ("--dream-replay", x.dream_replay, 1_000_000),
            ("--dream-len", x.dream_len, 100_000),
            ("--dream-steps", x.dream_steps, 1_000_000),
            ("--batch-size", x.batch_size, 65_536),
        ] {
            if value > maximum {
                return Err(format!("{name} must be at most {maximum}"));
            }
        }
        Ok(x)
    }

    fn options(parsed: &Parsed) -> Result<TrainingOptions, String> {
        let mut x = Self::common_options(parsed)?;
        if let Some(resume) = x.resume.as_deref() {
            let loaded = checkpoint::load_checkpoint(resume)
                .map_err(|e| format!("cannot inspect resume checkpoint '{resume}': {e}"))?;
            let checks = [
                ("--latent", "--latent", x.latent, loaded.model.cfg.d_latent),
                ("--depth", "--depth", x.depth, loaded.model.depth()),
                ("--state", "--state", x.state, loaded.model.cfg.d_state),
                ("--key", "--key", x.key, loaded.model.cfg.d_mem_key),
                (
                    "--memory",
                    "--memory",
                    x.memory,
                    loaded.model.cfg.mem_capacity,
                ),
                ("--chunk", "--chunk", x.chunk, loaded.model.cfg.chunk_len),
            ];
            for (label, flag, requested, actual) in checks {
                if parsed.flags.contains_key(flag) && requested != actual {
                    return Err(format!(
                        "{label}={requested} does not match resume checkpoint value {actual}"
                    ));
                }
            }
            // Omitted depth inherits the checkpoint; an explicit mismatch above
            // is never interpreted as a request to expand or truncate a stack.
            x.depth = loaded.model.depth();
            // A resume keeps the checkpoint's optimizer schedule unless the
            // caller explicitly supplies a new learning rate.  Previously the
            // parser's 1e-3 default silently replaced a custom checkpoint LR.
            if !parsed.flags.contains_key("--lr") {
                x.lr = loaded.model.cfg.lr;
            }
            if let (Some(requested), Some(stored)) = (
                x.schedule_total_updates,
                loaded.model.lr_schedule_total_updates,
            ) && requested != stored
            {
                return Err(format!(
                    "--total-updates={requested} does not match resume checkpoint horizon {stored}"
                ));
            }
        }
        Ok(x)
    }

    pub fn documents(
        raw: &str,
        tokenizer: &Tokenizer,
        limit: Option<usize>,
        skip: usize,
    ) -> Result<Vec<Vec<usize>>, String> {
        Ok(token_cache::documents(raw, tokenizer, limit, skip, None, None)?.docs)
    }

    /// Select a training window and optionally load/build its persistent token
    /// cache. The source path is used only for the cache fingerprint; raw text
    /// remains the single source of training behavior.
    pub fn documents_with_cache(
        raw: &str,
        tokenizer: &Tokenizer,
        limit: Option<usize>,
        skip: usize,
        cache_path: Option<&Path>,
        source_path: Option<&Path>,
    ) -> Result<Vec<Vec<usize>>, String> {
        let result = token_cache::documents(raw, tokenizer, limit, skip, cache_path, source_path)?;
        if cache_path.is_some() {
            println!(
                "token_cache={} token_window_seconds={:.3}",
                result.status.label(),
                result.elapsed.as_secs_f64()
            );
        }
        Ok(result.docs)
    }
    fn memory_occupancy(model: &PSSALayerV2) -> Option<(usize, usize)> {
        let banks = std::iter::once(&model.block).chain(model.extra_blocks.iter());
        let mut used = 0usize;
        let mut capacity = 0usize;
        for block in banks {
            used = used.checked_add(block.memory.count)?;
            capacity = capacity.checked_add(block.memory.capacity)?;
        }
        (capacity > 0).then_some((used, capacity))
    }

    fn finite(model: &PSSALayerV2) -> bool {
        fn matrix_finite(p: &crate::pssa::ParamMatrix) -> bool {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|v| v.is_finite())
        }
        fn vector_finite(p: &crate::pssa::ParamVector) -> bool {
            p.data
                .iter()
                .chain(&p.grad)
                .chain(&p.m)
                .chain(&p.v)
                .all(|v| v.is_finite())
        }
        #[cfg(feature = "cuda")]
        let device_finite = match &model.device {
            crate::backend::Device::Cuda(ctx) => ctx.safeguarded_parameters_finite(),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let device_finite = None::<bool>;
        if device_finite == Some(false) {
            return false;
        }
        // The CUDA Adam kernel checks data/grad/m/v in its existing pass. Host
        // moments/gradients are deliberately stale until training handoff.
        let matrix_finite = |p| device_finite == Some(true) || matrix_finite(p);
        let vector_finite = |p| device_finite == Some(true) || vector_finite(p);
        matrix_finite(&model.embed_w)
            && matrix_finite(&model.unembed_w)
            && std::iter::once(&model.block)
                .chain(&model.extra_blocks)
                .all(|block| {
                    [
                        &block.a_mat,
                        &block.w_delta,
                        &block.w_b,
                        &block.w_c,
                        &block.w_qx,
                        &block.w_qh,
                        &block.w_gate,
                        &block.w_proj,
                        &block.mlp_w1,
                        &block.mlp_w2,
                    ]
                    .iter()
                    .all(|p| matrix_finite(p))
                        && vector_finite(&block.norm_gamma)
                        && vector_finite(&block.norm_beta)
                        && block.adapters.iter().all(|ad| {
                            matrix_finite(&ad.down_proj)
                                && matrix_finite(&ad.up_proj)
                                && ad.consolidated_up.iter().all(|v| v.is_finite())
                        })
                        && block.h_persistent.iter().all(|v| v.is_finite())
                        && block.memory.keys.iter().all(|v| v.is_finite())
                        && block.memory.values.iter().all(|v| v.is_finite())
                        && block.memory.norm_sq.iter().all(|v| v.is_finite())
                        && block.memory.confidence.iter().all(|v| v.is_finite())
                })
    }

    pub fn train_corpus(
        raw: &str,
        options: &TrainingOptions,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        let _plain_output = ui::plain_output(options.no_tui);
        if options.epochs == 0
            || options.latent == 0
            || options.state == 0
            || options.key == 0
            || options.memory == 0
            || options.chunk == 0
            || options.accumulate == 0
            || options.batch_size == 0
        {
            return Err(
                "epochs, latent, state, key, memory, chunk, accumulate, and batch-size must be positive"
                    .into(),
            );
        }
        if !(1..=32).contains(&options.depth) {
            return Err(
                "--depth must be between 1 and 32; use --depth 1 for the original model".into(),
            );
        }
        Self::validate_loops(options.loops)?;
        for (flag, value) in [
            ("--grad-clip", options.grad_clip),
            ("--memory-value-cap", options.memory_value_cap),
        ] {
            if value.is_some_and(|x| !x.is_finite() || x <= 0.0) {
                return Err(format!("{flag} must be finite and positive"));
            }
        }
        if options.batch_size > 65_536 {
            return Err("--batch-size must be at most 65536; use fewer document lanes".into());
        }
        if options.dream_replay == 0 {
            return Err("--dream-replay must be positive".into());
        }
        if options.dream_len == 0 {
            return Err("--dream-len must be positive".into());
        }
        if !options.dream_lr.is_finite() || options.dream_lr <= 0.0 {
            return Err("--dream-lr must be finite and positive".into());
        }
        if options.dream_steps == 0 {
            return Err("--dream-steps must be positive".into());
        }
        if !options.lr.is_finite() || options.lr <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        if options.max_tokens == Some(0) {
            return Err("max_tokens must be positive when supplied".into());
        }
        let (mut model, tokenizer) = match options.resume.as_deref() {
            Some(path) => {
                let loaded = checkpoint::load_checkpoint(path)
                    .map_err(|e| format!("cannot resume from '{path}': {e}"))?;
                let model = loaded.model;
                if options.depth != model.depth() {
                    return Err(format!(
                        "--depth={} does not match resume checkpoint value {}; use the checkpoint depth or start a fresh run",
                        options.depth,
                        model.depth()
                    ));
                }
                if model.vocabulary.is_empty() {
                    return Err("resume checkpoint lacks vocabulary provenance".into());
                }
                let tokenizer = match &model.tokenizer_json {
                    Some(json) => Tokenizer::from_serialized(json)?,
                    None => Tokenizer::from_vocabulary(&model.vocabulary)?,
                };
                if tokenizer.vocab_size != model.cfg.d_vocab
                    || tokenizer.ordered_vocabulary()? != model.vocabulary
                {
                    return Err("resume checkpoint tokenizer/vocabulary mismatch".into());
                }
                println!(
                    "resumed_from={path} vocab={} d_latent={} depth={} loops={} prior_steps={}",
                    model.cfg.d_vocab,
                    model.cfg.d_latent,
                    model.depth(),
                    options.loops,
                    model.step_counter
                );
                if options.loops == 1 && !options.loops_explicit {
                    println!(
                        "note: --loops not given, running 1 pass (loops are not checkpointed)"
                    );
                }
                ui::success(&format!(
                    "resumed {} at {} prior optimizer steps",
                    ui::bold(path),
                    ui::thousands(model.step_counter as usize)
                ));
                (model, tokenizer)
            }
            None => {
                let tokenizer = match options.tokenizer {
                    TokenizerKind::Word => Tokenizer::from_corpus(raw, true)?,
                    TokenizerKind::Bpe => Tokenizer::from_corpus_bpe(raw, options.vocab_size)?,
                };
                let cfg = PSSAConfigV2 {
                    d_vocab: tokenizer.vocab_size,
                    d_latent: options.latent,
                    depth: options.depth,
                    d_state: options.state,
                    d_mem_key: options.key,
                    mem_capacity: options.memory,
                    chunk_len: options.chunk,
                    lr: options.lr,
                    ..Default::default()
                };
                checkpoint::validate_model_config(&cfg)?;
                PSSALayerV2::validate_loops_config(&cfg, options.loops)?;
                let mut model = PSSALayerV2::new_with_depth_and_loops(
                    cfg,
                    options.seed,
                    options.depth,
                    options.loops,
                );
                model.vocabulary = tokenizer.ordered_vocabulary()?;
                model.tokenizer_json = tokenizer.serialized_metadata();
                (model, tokenizer)
            }
        };
        if options.resume.is_some() {
            model.set_loops(options.loops)?;
        }
        // Runtime-only safeguards: never serialized, and no extra math/logging
        // on the historical flags-off path. Cap loaded values before any read.
        if options.grad_clip.is_some() || options.memory_value_cap.is_some() {
            println!(
                "grad_clip={} memory_value_cap={} max_consecutive_skipped_updates={MAX_CONSECUTIVE_SKIPPED_UPDATES} (runtime-only; repeat flags on resume)",
                options
                    .grad_clip
                    .map_or_else(|| "off".into(), |x| x.to_string()),
                options
                    .memory_value_cap
                    .map_or_else(|| "off".into(), |x| x.to_string())
            );
        }
        // Keep an explicit resume override in the checkpoint's persisted
        // configuration so a later link does not silently revert to the old LR.
        model.cfg.lr = options.lr;
        // Every model shape uses the same device selection. Dense forward and
        // backward stages dispatch through cuBLAS when available. CUDA's
        // resident scan/memory kernels are used by the depth-one packed path;
        // host tape bookkeeping and unsupported elementwise stages remain
        // explicit CPU work.
        match options.backend.device() {
            Err(e) if options.backend != TrainingBackend::Auto => {
                return Err(format!("requested training backend unavailable: {e}"));
            }
            Ok(gpu_device) => {
                let label = gpu_device
                    .gpu()
                    .map(|g| g.backend_label())
                    .unwrap_or_else(|| "cpu".to_string());
                let explicit_cpu = gpu_device.gpu().is_none();
                model.device = gpu_device;
                if explicit_cpu && (model.depth() > 1 || model.loops() > 1) {
                    println!(
                        "backend=cpu ({}; explicitly selected)",
                        if model.depth() > 1 {
                            format!("stacked depth {}", model.depth())
                        } else {
                            format!("Ouro loops {}", model.loops())
                        }
                    );
                    if options.batch_size > 1 {
                        println!(
                            "batch_backend=cpu-replay (independent lanes; no packed GEMM acceleration)"
                        );
                    }
                } else {
                    println!("backend={label}");
                }
            }
            Err(e) => {
                if model.depth() > 1 || model.loops() > 1 {
                    println!(
                        "backend=cpu ({}; GPU unavailable: {e})",
                        if model.depth() > 1 {
                            format!("stacked depth {}", model.depth())
                        } else {
                            format!("Ouro loops {}", model.loops())
                        }
                    );
                    if options.batch_size > 1 {
                        println!(
                            "batch_backend=cpu-replay (independent lanes; no packed GEMM acceleration)"
                        );
                    }
                } else {
                    println!("backend=cpu ({e})");
                }
            }
        }
        if options.memory_value_cap.is_some() {
            for block in std::iter::once(&mut model.block).chain(&mut model.extra_blocks) {
                block
                    .memory
                    .set_value_cap_with_device(options.memory_value_cap, &model.device);
            }
        }
        if options.dream_every > 0 && model.device.is_gpu() {
            eprintln!(
                "warning: dream replay is host-only; using an explicit CPU fallback and synchronizing host weights for backend {}",
                model
                    .device
                    .gpu()
                    .map(|g| g.backend_label())
                    .unwrap_or_else(|| "gpu".into())
            );
        }
        let cache_path = options.token_cache.as_deref().map(Path::new);
        let source_path = options.token_cache_source.as_deref().map(Path::new);
        let docs = Self::documents_with_cache(
            raw,
            &tokenizer,
            options.max_tokens,
            options.skip_tokens,
            cache_path,
            source_path,
        )?;
        // Feed metadata is derived from the actual selected training window,
        // never estimated from row lengths. Local training keeps its old path.
        let mut feed_rows_consumed = 0usize;
        let chunk_len = if options.resume.is_some() {
            // A checkpoint owns its tape capacity.  Using a fresh CLI default
            // here could make the plan longer than that tape and panic in the
            // batched backward pass.
            model.cfg.chunk_len
        } else {
            options.chunk
        };
        let plan = crate::training::sequence_plan(&docs, chunk_len, options.batch_size)?;
        let mut sequence_batch = if options.batch_size > 1 {
            Some(crate::sequence_batch::SequenceBatch::new(
                &mut model,
                options.batch_size.min(docs.len()).max(1),
            )?)
        } else {
            None
        };
        let schedule = crate::training::Schedule::new_with_warmup(
            plan.len(),
            model.step_counter,
            model.lr_schedule_total_updates,
            model.lr_schedule_warmup_steps,
            options,
        )?;
        model.lr_schedule_total_updates = schedule.fixed_horizon;
        model.lr_schedule_warmup_steps = schedule.fixed_horizon.map(|_| schedule.warmup);
        let total_updates = schedule.updates;
        let schedule_total = schedule.total;
        let first_lr = schedule.lr(1)?;
        let last_lr = schedule.lr(total_updates)?;
        println!(
            "model=pssa parameters={} vocab={} depth={} loops={}",
            model.parameter_count(),
            model.cfg.d_vocab,
            model.depth(),
            model.loops()
        );
        crate::training::report_stream(&docs, chunk_len, options.accumulate);
        if options.batch_size > 1 {
            crate::training::report_sequence_plan(&plan, options.batch_size);
        }
        ui::banner("train", "plastic state-space architecture");
        ui::field(
            "corpus",
            &format!("{} tokens", ui::thousands(docs.iter().map(Vec::len).sum())),
        );
        ui::field("vocabulary", &ui::thousands(model.cfg.d_vocab));
        ui::field(
            "width",
            &format!(
                "latent {} / state {} / depth {}",
                model.cfg.d_latent,
                model.cfg.d_state,
                model.depth()
            ),
        );
        ui::field(
            "memory",
            &format!(
                "{} slots, key width {}",
                model.cfg.mem_capacity, model.cfg.d_mem_key
            ),
        );
        ui::field(
            "schedule",
            &format!(
                "{} epoch(s), {} updates, lr first={:.8} last={:.8} (base {:.8}, horizon {})",
                options.epochs,
                ui::thousands(total_updates),
                first_lr,
                last_lr,
                options.lr,
                ui::thousands(schedule_total)
            ),
        );
        println!();

        let mut curve = options
            .loss_csv
            .as_deref()
            .map(|path| {
                crate::loss_csv::LossCsv::open(
                    path,
                    options.loss_every,
                    model.step_counter,
                    options.tokens_seen,
                )
            })
            .transpose()?;
        // Sequence values are borrowed views into `docs`; reuse the descriptor
        // vector so packed training does not allocate once per microbatch.
        let mut sequence_views = Vec::with_capacity(options.batch_size);
        #[cfg(feature = "cuda")]
        let cuda_optimizer = match model.device.clone() {
            crate::backend::Device::Cuda(ctx) if options.grad_clip.is_some() => {
                // Validate checkpoint/initial host state once, before its moments
                // and dense gradients become exclusively device-owned.
                if !Self::finite(&model) {
                    return Err("non-finite initial parameters".into());
                }
                ctx.begin_safeguarded_training(&model.adam_tensors())?;
                Some(ctx)
            }
            _ => None,
        };
        let started = Instant::now();
        let mut update = 0;
        let mut skipped = SkippedUpdates::default();
        let mut dream_rng =
            (options.dream_every > 0).then(|| SimpleRng::new(options.seed ^ 0xd0e5_5eed_5eed_0001));
        // The last fresh chunk is an ephemeral guard for projected replay. It
        // is populated only when dreaming is enabled, so the default path does
        // not allocate or copy any additional training data.
        let mut dream_fresh_task: Option<(Vec<usize>, Vec<usize>)> = None;
        let mut tokens_seen = 0usize;
        let mut progress = ui::Progress::new_with_tui("training", total_updates, !options.no_tui);
        if let Some(path) = options.checkpoint_path.as_deref() {
            progress.set_checkpoint_path(path);
        }
        if let Some(path) = options.resume.as_deref() {
            progress.set_last_checkpoint(path);
        }
        progress.set_prior_updates(model.step_counter.saturating_sub(update));
        println!(
            "progress_schema=2 updates_total={} prior_updates={} checkpoint_target={}",
            total_updates,
            model.step_counter,
            options.checkpoint_path.as_deref().unwrap_or("-")
        );
        println!(
            "last_checkpoint={}",
            options.resume.as_deref().unwrap_or("-")
        );
        for epoch in 0..options.epochs {
            model.reset_recurrent_state();
            if let Some(batch) = &mut sequence_batch {
                batch.reset_states();
            }
            let mut loss_sum = 0.0f64;
            let mut token_sum = 0usize;
            for group in plan.chunks(options.accumulate) {
                let prior_loss = loss_sum;
                let total_tokens: usize = group.iter().flatten().map(|x| x.len).sum();
                if total_tokens == 0 {
                    continue;
                }
                {
                    let _trace = crate::training_diagnostics::StageTrace::new(
                        &model.device,
                        "update.zero_gradients",
                        total_tokens,
                    );
                    model.zero_gradients();
                }
                for microbatch in group {
                    let batch_tokens: usize = microbatch.iter().map(|c| c.len).sum();
                    if options.dream_every > 0
                        && let Some(c) = microbatch.last()
                    {
                        dream_fresh_task = Some((
                            docs[c.doc][c.start..c.start + c.len].to_vec(),
                            docs[c.doc][c.start + 1..c.start + 1 + c.len].to_vec(),
                        ));
                    }
                    let loss = if let Some(batch) = &mut sequence_batch {
                        sequence_views.clear();
                        for c in microbatch {
                            sequence_views.push(crate::sequence_batch::Sequence {
                                lane: c.lane,
                                inputs: &docs[c.doc][c.start..c.start + c.len],
                                targets: &docs[c.doc][c.start + 1..c.start + 1 + c.len],
                                reset: c.start == 0,
                            });
                        }
                        batch
                            .forward(&mut model, &sequence_views)
                            .map_err(|error| {
                                format!(
                                    "update_index={} global_step={}: {error}",
                                    update + 1,
                                    model.step_counter + 1
                                )
                            })?
                    } else {
                        let c = microbatch[0];
                        if c.start == 0 {
                            model.reset_recurrent_state();
                        }
                        let inputs = &docs[c.doc][c.start..c.start + c.len];
                        let targets = &docs[c.doc][c.start + 1..c.start + 1 + c.len];
                        if model.loops() > 1 {
                            // gpu_batch only implements one pass through each block.
                            model.forward_train_chunk(inputs, targets)
                        } else {
                            // Preserve the historical single-lane math and write order.
                            crate::gpu_batch::forward_train_chunk_batched(
                                &mut model, inputs, targets,
                            )
                        }
                    };
                    if !loss.is_finite() {
                        return Err(format!(
                            "update_index={} global_step={}: {}",
                            update + 1,
                            model.step_counter + 1,
                            crate::training_diagnostics::loss_error(&model, batch_tokens),
                        ));
                    }
                    let scale = batch_tokens as f32 / total_tokens as f32;
                    if let Some(batch) = &mut sequence_batch {
                        batch.backward(&mut model, scale).map_err(|error| {
                            format!(
                                "update_index={} global_step={}: {error}",
                                update + 1,
                                model.step_counter + 1
                            )
                        })?;
                        for c in microbatch {
                            if let Some((index, value)) = batch
                                .state(c.lane)
                                .iter()
                                .enumerate()
                                .find(|(_, x)| !x.is_finite())
                            {
                                return Err(format!(
                                    "non-finite batch carry; update_index={} global_step={} stage=ssm tensor=carry lane={} index={index} value={value}; training aborted without checkpoint",
                                    update + 1,
                                    model.step_counter + 1,
                                    c.lane,
                                ));
                            }
                        }
                    } else if model.loops() > 1 {
                        model.backward_chunk(batch_tokens, scale);
                    } else {
                        crate::gpu_batch::backward_chunk_batched(&mut model, batch_tokens, scale);
                    }
                    // All retrieval adjoints see the same bank as forward. Writes
                    // happen only now, in deterministic lane order, using each
                    // chunk's own mean loss and terminal token (not the batch mean).
                    let _trace = crate::training_diagnostics::StageTrace::new(
                        &model.device,
                        "update.memory_writes",
                        batch_tokens,
                    );
                    let mut offset = 0;
                    for c in microbatch {
                        let loss = model.tape.losses[offset..offset + c.len]
                            .iter()
                            .sum::<f32>()
                            / c.len as f32;
                        model.insert_training_memory_at(loss, offset + c.len - 1);
                        if options.dream_every > 0 && loss > 3.5 {
                            let inputs = model.block.tape.x_ids[offset..offset + c.len].to_vec();
                            let targets =
                                model.block.tape.target_ids[offset..offset + c.len].to_vec();
                            model.remember_dream_sequence(&inputs, &targets);
                        }
                        loss_sum += loss as f64 * c.len as f64;
                        token_sum += c.len;
                        offset += c.len;
                    }
                }
                let learning_rate = schedule.lr(update + 1)?;
                let optimizer_trace = crate::training_diagnostics::StageTrace::new(
                    &model.device,
                    "update.optimizer",
                    total_tokens,
                );
                if let Some(max_norm) = options.grad_clip {
                    match model.apply_adamw_with_grad_clip(learning_rate, max_norm) {
                        GradientClipOutcome::Applied { norm } => {
                            skipped.consecutive = 0;
                            progress.set_gradient_metrics(norm, skipped.total);
                        }
                        GradientClipOutcome::Skipped { norm } => {
                            skipped.record(norm, model.step_counter)?;
                            // Consume this group, but neither Adam nor the LR
                            // schedule advances. The next group gets clean grads.
                            continue;
                        }
                    }
                } else {
                    model.apply_adamw(learning_rate);
                }
                drop(optimizer_trace);
                update += 1;
                if options.dream_every > 0
                    && update % options.dream_every == 0
                    && let Some(rng) = dream_rng.as_mut()
                {
                    println!("{}", dream_phase_start_line(update, options.dream_mode));
                    #[cfg(feature = "cuda")]
                    if let Some(ctx) = cuda_optimizer.as_ref() {
                        ctx.sync_safeguarded_weights(&mut model.adam_tensors())?;
                    }
                    let fresh_task = dream_fresh_task
                        .as_ref()
                        .map(|(inputs, targets)| (inputs.as_slice(), targets.as_slice()));
                    let summary = model.dream_replay_with_options_and_guard(
                        options.dream_mode,
                        options.dream_replay,
                        options.dream_len,
                        0.8,
                        options.dream_lr,
                        options.dream_steps,
                        fresh_task,
                        rng,
                    );
                    println!("{}", dream_phase_end_line(update, options.dream_mode, &summary));
                }
                let finite = {
                    let _trace = crate::training_diagnostics::StageTrace::new(
                        &model.device,
                        "update.parameters_finite",
                        total_tokens,
                    );
                    Self::finite(&model)
                };
                if !finite {
                    return Err(format!(
                        "non-finite parameters; update_index={update} global_step={} stage=optimizer; training aborted without checkpoint",
                        model.step_counter
                    ));
                }
                tokens_seen += total_tokens;
                if let Some(curve) = &mut curve {
                    curve.record(total_tokens, model.step_counter, loss_sum - prior_loss)?;
                }
                if let Some(dataset) = options.hf_dataset.as_deref() {
                    feed_rows_consumed += group
                        .iter()
                        .flatten()
                        .filter(|c| c.start + c.len + 1 == docs[c.doc].len())
                        .count();
                    // Decode only a bounded slice, only when the existing logger
                    // is due (5s piped / 200ms interactive). No extra tokenization
                    // or work in the forward/backward path.
                    if progress.should_emit(update) {
                        let c = group.iter().flatten().last().expect("nonempty update");
                        let end = c.start + c.len;
                        let ids = &docs[c.doc][end.saturating_sub(16).max(c.start)..end];
                        let snippet: String = tokenizer.decode(ids).chars().take(96).collect();
                        let token_ids = ids
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(",");
                        progress.set_feed(
                            dataset,
                            options.hf_config.as_deref(),
                            &options.hf_split,
                            &options.hf_field,
                            feed_rows_consumed,
                            tokens_seen,
                            c.doc + 1,
                            &snippet,
                            &token_ids,
                        );
                    }
                }
                let update_loss = (loss_sum - prior_loss) / total_tokens.max(1) as f64;
                progress.update_with_metrics(
                    update,
                    total_tokens,
                    update_loss,
                    Some(learning_rate),
                    Self::memory_occupancy(&model),
                );
            }
            progress.finish();
            model.ema_consolidate_plasticity();
            println!(
                "epoch {}/{} loss={:.6} tokens={} updates={}",
                epoch + 1,
                options.epochs,
                loss_sum / token_sum.max(1) as f64,
                token_sum,
                update
            );
        }
        progress.finish();
        #[cfg(feature = "cuda")]
        if let Some(ctx) = cuda_optimizer {
            // Checkpoint/public API state is complete only at this handoff;
            // there are no full gradient/moment readbacks inside the loop.
            ctx.finish_safeguarded_training(&mut model.adam_tensors())?;
        }
        if let Some(curve) = &mut curve {
            curve.finish()?;
        }
        let wall = started.elapsed().as_secs_f64();
        println!(
            "training_seconds={:.3} optimizer_updates={update}",
            started.elapsed().as_secs_f32()
        );
        if options.grad_clip.is_some() {
            println!(
                "skipped_updates={} consecutive_skips={}",
                skipped.total, skipped.consecutive
            );
        }
        ui::section("summary");
        ui::field("wall time", &ui::duration(wall));
        ui::field("tokens", &ui::thousands(tokens_seen));
        ui::field(
            "throughput",
            &format!(
                "{:.0} tokens/second",
                if wall > 0.0 {
                    tokens_seen as f64 / wall
                } else {
                    0.0
                }
            ),
        );
        ui::field("updates", &ui::thousands(update));
        println!();
        Ok((model, tokenizer))
    }

    pub fn run_training(data: &str, options: &TrainingOptions, out: &str) -> Result<(), String> {
        let _plain_output = ui::plain_output(options.no_tui);
        let raw = if let Some(dataset) = options.hf_dataset.as_deref() {
            let mut request = HuggingFaceDatasetOptions::new(dataset);
            request.config = options.hf_config.clone();
            request.split = options.hf_split.clone();
            request.field = options.hf_field.clone();
            DatasetManager::load_huggingface_dataset(&request)?
        } else {
            DatasetManager::try_load_dataset(Some(data))?
        };
        let mut run_options = options.clone();
        run_options.checkpoint_path = Some(out.to_string());
        // Token caching is explicitly opt-in. In particular, do not infer a
        // cache path for local files: omitting --token-cache must retain the
        // historical lazy serial tokenization path.
        if run_options.token_cache.is_some() && run_options.token_cache_source.is_none() {
            run_options.token_cache_source = Self::local_cache_source(data);
        }
        let (model, _) = Self::train_corpus(&raw, &run_options)?;
        Self::save_model_v2(&model, out)
            .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
        ui::checkpoint_saved(out);
        ui::success(&format!("checkpoint written to {}", ui::bold(out)));
        println!();
        Ok(())
    }

    fn word_tokenizer_for_model(
        model: &PSSALayerV2,
        data: Option<&str>,
        label: &str,
    ) -> Result<Tokenizer, String> {
        if model.vocabulary.is_empty() {
            if label == "legacy V5" {
                return Err(
                    "legacy V5 checkpoint carries no vocabulary; train a fresh checkpoint with a tokenizer"
                        .into(),
                );
            }
            return Err(format!("{label} checkpoint lacks vocabulary provenance"));
        }
        let tokenizer = Tokenizer::from_vocabulary(&model.vocabulary)?;
        if let Some(source) = data {
            let raw = DatasetManager::try_load_dataset(Some(source))?;
            let external = Tokenizer::from_corpus(&raw, true)?;
            if external.ordered_vocabulary()? != model.vocabulary && label != "legacy V5" {
                return Err(format!(
                    "--data tokenizer/order does not match {label} checkpoint"
                ));
            }
        }
        Ok(tokenizer)
    }
    pub(crate) fn load_for_inference(
        model_path: &str,
        data: Option<&str>,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        let loaded = checkpoint::load_checkpoint(model_path)
            .map_err(|e| format!("cannot load model '{model_path}': {e}"))?;
        let mut model = loaded.model;
        match loaded.format {
            CheckpointFormat::V7 | CheckpointFormat::V8 => match &model.tokenizer_json {
                Some(json) => {
                    if data.is_some() {
                        return Err("--data is legacy word provenance only; V7/V8 BPE checkpoints restore their embedded tokenizer and never retrain it".into());
                    }
                    let tokenizer = Tokenizer::from_serialized(json)?;
                    if tokenizer.ordered_vocabulary()? != model.vocabulary
                        || tokenizer.vocab_size != model.cfg.d_vocab
                    {
                        return Err(
                            "checkpoint tokenizer metadata/model vocabulary mismatch".into()
                        );
                    }
                    Ok((model, tokenizer))
                }
                None => {
                    let tokenizer = Self::word_tokenizer_for_model(&model, data, "V7/V8 word")?;
                    Ok((model, tokenizer))
                }
            },
            CheckpointFormat::V6 => {
                let tokenizer = Self::word_tokenizer_for_model(&model, data, "V6")?;
                Ok((model, tokenizer))
            }
            CheckpointFormat::LegacyV5InferenceOnly => {
                eprintln!(
                    "warning: legacy V5 checkpoint is inference-only; optimizer and tokenizer provenance are unavailable"
                );
                // The checked-in V5 artifact predates vocabulary serialization but
                // was trained from the built-in science corpus. Restore that
                // ordered vocabulary when its size identifies the artifact; other
                // vocabulary-less V5 checkpoints still receive the explicit
                // fresh-checkpoint error from word_tokenizer_for_model.
                if model.vocabulary.is_empty() {
                    let legacy =
                        Tokenizer::from_corpus(DatasetManager::SCIENCE_REFERENCE_CORPUS, true)?;
                    if legacy.vocab_size == model.cfg.d_vocab {
                        model.vocabulary = legacy.ordered_vocabulary()?;
                    }
                }
                let tokenizer = Self::word_tokenizer_for_model(&model, data, "legacy V5")?;
                Ok((model, tokenizer))
            }
        }
    }

    pub fn run_generate(
        prompt: &str,
        model_path: &str,
        data: Option<&str>,
        temperature: f32,
        max_new: usize,
    ) -> Result<String, String> {
        Self::run_generate_with_loops(prompt, model_path, data, temperature, max_new, 1)
    }

    /// Generate with a runtime-only Ouro override; checkpoints do not store it.
    pub fn run_generate_with_loops(
        prompt: &str,
        model_path: &str,
        data: Option<&str>,
        temperature: f32,
        max_new: usize,
        loops: usize,
    ) -> Result<String, String> {
        Self::validate_loops(loops)?;
        if max_new > MAX_GENERATION_TOKENS {
            return Err(format!(
                "max_new_tokens must be at most {MAX_GENERATION_TOKENS}"
            ));
        }
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
        model.set_loops(loops)?;
        let cfg = InferenceConfig {
            temperature,
            max_new_tokens: max_new,
            top_k: if temperature == 0.0 { 1 } else { 24 },
            repetition_penalty: if temperature == 0.0 { 1.0 } else { 1.25 },
            ..Default::default()
        };
        PSSAInferenceEngine::try_new(&mut model, &tokenizer)?.try_generate_chat_turn(
            prompt,
            &cfg,
            |_| {},
        )
    }

    pub fn evaluate_corpus(
        model: &mut PSSALayerV2,
        tokenizer: &Tokenizer,
        raw: &str,
    ) -> Result<(f64, usize, usize, usize), String> {
        let metrics = crate::evaluation::evaluate_pssa(
            model,
            tokenizer,
            raw,
            crate::evaluation::EvaluationSlice::default(),
        )?;
        Ok((metrics.loss, metrics.tokens, metrics.correct, metrics.oov))
    }
    fn run_evaluate(
        model_path: &str,
        data: &str,
        slice: crate::evaluation::EvaluationSlice,
        loops: usize,
    ) -> Result<(), String> {
        Self::validate_loops(loops)?;
        // V5 checkpoints contain no tokenizer provenance, so the evaluation
        // corpus also has to seed their legacy word tokenizer.  Newer formats
        // restore their tokenizer independently of the held-out corpus.
        let format = checkpoint::load_checkpoint(model_path)
            .map_err(|e| format!("cannot inspect model '{model_path}': {e}"))?
            .format;
        let provenance = (format == CheckpointFormat::LegacyV5InferenceOnly).then_some(data);
        let (mut model, tokenizer) = Self::load_for_inference(model_path, provenance)?;
        model.set_loops(loops)?;
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let metrics = crate::evaluation::evaluate_pssa(&mut model, &tokenizer, &raw, slice)?;
        println!("{}", metrics.json());
        Ok(())
    }
    fn run_chat(
        model_path: &str,
        data: Option<&str>,
        temp: f32,
        loops: usize,
    ) -> Result<(), String> {
        Self::validate_loops(loops)?;
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
        model.set_loops(loops)?;
        let memory_slots = std::iter::once(&model.block)
            .chain(model.extra_blocks.iter())
            .map(|block| block.memory.capacity)
            .sum::<usize>();
        let adapter_count = std::iter::once(&model.block)
            .chain(model.extra_blocks.iter())
            .map(|block| block.adapters.len())
            .sum::<usize>();
        let effective_loops = model.loops();
        let mut engine = PSSAInferenceEngine::try_new(&mut model, &tokenizer)?;
        let mut temp = temp;
        println!("interactive: /exit  /info  /temp <value>");
        loop {
            print!("user> ");
            io::stdout().flush().map_err(|e| e.to_string())?;
            let mut line = String::new();
            if io::stdin()
                .read_line(&mut line)
                .map_err(|e| e.to_string())?
                == 0
            {
                break;
            }
            let p = line.trim();
            if p == "/exit" || p == "quit" {
                break;
            }
            if p == "/info" {
                println!("model: {model_path}");
                println!("memory slots: {memory_slots}");
                println!("adapters: {adapter_count}");
                println!("loops: {effective_loops}");
                continue;
            }
            let mut words = p.split_whitespace();
            if words.next() == Some("/temp") {
                match (words.next(), words.next()) {
                    (Some(value), None) => match value.parse::<f32>() {
                        Ok(next) if next.is_finite() && next >= 0.0 => {
                            temp = next;
                            println!("temperature set to {temp:.4}");
                        }
                        _ => eprintln!("error: /temp value must be a finite number >= 0"),
                    },
                    _ => eprintln!("error: usage: /temp <value>"),
                }
                continue;
            }
            if p.is_empty() {
                continue;
            }
            let cfg = InferenceConfig {
                temperature: temp,
                ..Default::default()
            };
            match engine.try_generate_chat_turn(p, &cfg, |_| {}) {
                Ok(output) => println!("{output}"),
                Err(error) => eprintln!("error: {error}"),
            }
        }
        Ok(())
    }
    pub fn run_benchmark() -> Result<(), String> {
        let raw = "the patient scientist observes the bright moon .\nthe patient scientist observes the bright moon .\nthe patient scientist observes the bright moon .\n";
        let opts = TrainingOptions {
            epochs: 120,
            latent: 16,
            state: 4,
            key: 8,
            memory: 8,
            chunk: 8,
            lr: 0.02,
            accumulate: 1,
            batch_size: 1,
            warmup_steps: 4,
            schedule_total_updates: None,
            seed: 7,
            max_tokens: None,
            skip_tokens: 0,
            tokenizer: TokenizerKind::Word,
            vocab_size: 2048,
            resume: None,
            ..TrainingOptions::default()
        };
        let (mut m, tok) = Self::train_corpus(raw, &opts)?;
        let (ce, _, _, _) = Self::evaluate_corpus(&mut m, &tok, raw)?;
        if !(ce < 0.8) {
            return Err(format!(
                "benchmark learned-model fidelity insufficient: CE={ce}"
            ));
        }
        let out = PSSAInferenceEngine::try_new(&mut m, &tok)?.try_generate_chat_turn(
            "the patient scientist",
            &InferenceConfig {
                temperature: 0.0,
                max_new_tokens: 5,
                ..Default::default()
            },
            |_| {},
        )?;
        if !out.contains("observes") {
            return Err(format!(
                "benchmark completion lacks learned reference token: {out}"
            ));
        }
        println!("benchmark_pass ce={ce:.6} completion={out}");
        Ok(())
    }
    fn run_throughput(
        model_path: &str,
        data: &str,
        slice: crate::evaluation::EvaluationSlice,
    ) -> Result<(), String> {
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let started = Instant::now();
        let (tokens, elapsed) = if model_path.ends_with(".trfm") {
            let model = crate::transformer_checkpoint::load_checkpoint(model_path)
                .map_err(|e| format!("cannot load transformer model '{model_path}': {e}"))?;
            let tokenizer = model.tokenizer()?;
            let mut model = model;
            let metrics =
                crate::evaluation::evaluate_transformer(&mut model, &tokenizer, &raw, slice)?;
            (metrics.tokens, started.elapsed().as_secs_f64())
        } else {
            let format = checkpoint::load_checkpoint(model_path)
                .map_err(|e| format!("cannot inspect model '{model_path}': {e}"))?
                .format;
            let provenance = (format == CheckpointFormat::LegacyV5InferenceOnly).then_some(data);
            let (mut model, tokenizer) = Self::load_for_inference(model_path, provenance)?;
            let metrics = crate::evaluation::evaluate_pssa(&mut model, &tokenizer, &raw, slice)?;
            (metrics.tokens, started.elapsed().as_secs_f64())
        };
        let rate = if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "throughput_tokens={} elapsed_seconds={elapsed:.6} tokens_per_second={rate:.3} model={model_path}",
            tokens
        );
        Ok(())
    }

    /// Everything the project can do, on one screen, with the state of the
    /// working directory next to it. This is what `pssa` alone prints.
    pub fn print_home() {
        ui::clear_screen();
        ui::logo();
        println!(
            "   {}  {}",
            ui::dim("plastic state-space architecture"),
            ui::dim(concat!("v", env!("CARGO_PKG_VERSION")))
        );
        println!();

        ui::panel_top("commands");
        for (name, blurb) in [
            ("train", "fit a PSSA checkpoint on a text corpus"),
            ("train-transformer", "fit the CPU decoder-only baseline"),
            ("generate", "continue a prompt with a trained checkpoint"),
            (
                "generate-transformer",
                "continue a prompt with the baseline",
            ),
            ("chat", "interactive prompt loop against a checkpoint"),
            ("score", "score a checkpoint on held-out text (JSON)"),
            (
                "score-transformer",
                "score the baseline on held-out text (JSON)",
            ),
            ("status", "checkpoints and corpora in this directory"),
            ("download", "pull a Hugging Face dataset to a local file"),
            (
                "clean-wikitext",
                "stream-clean raw WikiText into a new file",
            ),
            ("benchmark", "end-to-end smoke test on the built-in corpus"),
            ("throughput", "measure frozen-model tokens/sec on a corpus"),
            (
                "tui",
                "live dashboard for a piped training run (train ... | pssa tui)",
            ),
            (
                "gpu-probe",
                "check whether a WebGPU compute device is usable",
            ),
            (
                "compare",
                "replay a matched transformer against a PSSA chain",
            ),
        ] {
            ui::panel_row(&format!(
                "{}{}",
                ui::cyan(&format!("{name:<16}")),
                ui::dim(blurb)
            ));
        }
        ui::panel_bottom();
        println!();

        Self::workspace_panel();
        println!();
        println!(
            "  {} {}",
            ui::dim("try"),
            ui::bold("pssa train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1")
        );
        println!("  {}", ui::dim("pssa help for every flag"));
        println!();
    }

    /// Checkpoints and corpora found nearby, newest first.
    fn workspace_panel() {
        let mut checkpoints: Vec<(String, u64)> = Vec::new();
        let mut corpora: Vec<(String, u64)> = Vec::new();
        let mut seen = HashSet::new();
        for dir in [".", "data", "chain", "data/chain"] {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let identity = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if !seen.insert(identity) {
                    continue;
                }
                let Some(name) = path.to_str() else { continue };
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let label = name.trim_start_matches("./").to_string();
                if name.ends_with(".pssa") || name.ends_with(".trfm") {
                    checkpoints.push((label, size));
                } else if name.ends_with(".txt") && size > 4096 {
                    corpora.push((label, size));
                }
            }
        }
        checkpoints.sort();
        corpora.sort();

        ui::panel_top("workspace");
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        ui::panel_field("device", &format!("cpu, {threads} threads"));
        if checkpoints.is_empty() {
            ui::panel_field("checkpoints", &ui::dim("none yet, run pssa train"));
        } else {
            let shown = checkpoints.len().min(4);
            for (i, (name, size)) in checkpoints.iter().take(shown).enumerate() {
                let label = if i == 0 { "checkpoints" } else { "" };
                ui::panel_field(label, &format!("{}  {}", name, ui::dim(&ui::bytes(*size))));
            }
            if checkpoints.len() > shown {
                ui::panel_field(
                    "",
                    &ui::dim(&format!("+{} more", checkpoints.len() - shown)),
                );
            }
        }
        if corpora.is_empty() {
            ui::panel_field("corpora", &ui::dim("none found"));
        } else {
            for (i, (name, size)) in corpora.iter().take(3).enumerate() {
                let label = if i == 0 { "corpora" } else { "" };
                ui::panel_field(label, &format!("{}  {}", name, ui::dim(&ui::bytes(*size))));
            }
        }
        ui::panel_bottom();
    }

    /// `pssa status`: the workspace panel on its own, plus what each
    /// checkpoint actually contains.
    fn run_status() -> Result<(), String> {
        println!();
        Self::workspace_panel();
        let mut described = 0usize;
        let mut seen = HashSet::new();
        for dir in ["data", "chain", "data/chain", "."] {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut paths: Vec<String> = entries
                .flatten()
                .filter_map(|e| e.path().to_str().map(|s| s.to_string()))
                .filter(|s| s.ends_with(".pssa") || s.ends_with(".trfm"))
                .collect();
            paths.sort();
            for path in paths {
                let identity = std::fs::canonicalize(&path)
                    .unwrap_or_else(|_| std::path::PathBuf::from(&path));
                if !seen.insert(identity) {
                    continue;
                }
                if described == 0 {
                    println!();
                    ui::panel_top("checkpoint detail");
                }
                if described >= 6 {
                    break;
                }
                described += 1;
                ui::panel_row(&ui::bold(path.trim_start_matches("./")));
                if path.ends_with(".trfm") {
                    match crate::transformer_checkpoint::load_checkpoint(&path) {
                        Ok(m) => {
                            ui::panel_field(
                                "  shape",
                                &format!(
                                    "vocab {} / width {} / feed-forward {}",
                                    ui::thousands(m.cfg.d_vocab),
                                    m.cfg.d_model,
                                    m.cfg.d_ff
                                ),
                            );
                            ui::panel_field(
                                "  trained",
                                &format!("{} optimizer steps", ui::thousands(m.step_counter)),
                            );
                        }
                        Err(e) => ui::panel_field("  unreadable", &ui::dim(&e.to_string())),
                    }
                } else {
                    match checkpoint::load_checkpoint(&path) {
                        Ok(loaded) => {
                            let m = loaded.model;
                            ui::panel_field(
                                "  shape",
                                &format!(
                                    "vocab {} / latent {} / state {}",
                                    ui::thousands(m.cfg.d_vocab),
                                    m.cfg.d_latent,
                                    m.cfg.d_state
                                ),
                            );
                            ui::panel_field(
                                "  trained",
                                &format!("{} optimizer steps", ui::thousands(m.step_counter)),
                            );
                        }
                        Err(e) => ui::panel_field("  unreadable", &ui::dim(&e.to_string())),
                    }
                }
            }
        }
        if described > 0 {
            ui::panel_bottom();
        }
        println!();
        Ok(())
    }

    pub fn print_help() {
        let bin = "pssa";
        println!();
        println!(
            "  {}  {}",
            ui::bold(&ui::cyan(bin)),
            ui::dim(concat!(
                "plastic state-space architecture, v",
                env!("CARGO_PKG_VERSION")
            ))
        );
        println!("  {}", ui::dim(&"\u{2500}".repeat(62)));
        println!();
        println!("  {}", ui::bold("USAGE"));
        println!("    {bin} <command> [options]");
        println!();
        println!("  {}", ui::bold("COMMANDS"));
        for (name, blurb) in [
            ("train", "fit a checkpoint on a text corpus"),
            ("train-transformer", "fit the CPU decoder-only baseline"),
            ("generate", "continue a prompt with a trained checkpoint"),
            (
                "generate-transformer",
                "continue a prompt with the baseline",
            ),
            ("score", "score a checkpoint on held-out text (JSON)"),
            (
                "score-transformer",
                "score the baseline on held-out text (JSON)",
            ),
            ("chat", "interactive prompt loop against a checkpoint"),
            ("status", "checkpoints and corpora in this directory"),
            ("download", "pull a Hugging Face dataset to a local file"),
            (
                "clean-wikitext",
                "stream-clean raw WikiText into a new file",
            ),
            ("benchmark", "end-to-end smoke test on the built-in corpus"),
            ("throughput", "measure frozen-model tokens/sec on a corpus"),
            (
                "tui",
                "live dashboard for a piped training run (train ... | pssa tui)",
            ),
            (
                "gpu-probe",
                "check whether a WebGPU compute device is usable",
            ),
            (
                "compare",
                "replay a matched transformer against a PSSA chain",
            ),
            ("help", "show this message"),
        ] {
            println!("    {:<22}{}", ui::cyan(name), ui::dim(blurb));
        }
        println!();
        println!("    evaluate / evaluate-transformer / repl  hidden compatibility aliases");
        println!("  {}", ui::bold("TRAIN"));
        println!("    {bin} train [source] [-d|--data source] [-o|--out path]");
        println!(
            "    {:<48}{}",
            "  --tokenizer bpe|word",
            ui::dim("default bpe")
        );
        println!(
            "    {:<48}{}",
            "  --vocab-size n",
            ui::dim("byte-level BPE ceiling, default 2048")
        );
        println!(
            "    {:<48}{}",
            "  -e|--epochs n",
            ui::dim("passes over the selected slice")
        );
        println!(
            "    {:<48}{}",
            "  --latent n --state n --depth n --loops n",
            ui::dim(
                "width, recurrent state size, blocks, shared Ouro passes (1..32; default 1; repeat on resume; not checkpointed)"
            )
        );
        println!(
            "    {:<48}{}",
            "  --key n --memory n",
            ui::dim("episodic key width and bank capacity")
        );
        println!(
            "    {:<48}{}",
            "  --chunk n --accumulate n",
            ui::dim("chunk length and microbatches per update")
        );
        println!(
            "    {:<48}{}",
            "  --batch-size n",
            ui::dim("independent document lanes (default 1)")
        );
        println!(
            "    {:<48}{}",
            "  --grad-clip max_norm",
            ui::dim("opt-in global L2 limit; skip bad norms, abort after 20 consecutive skips")
        );
        println!(
            "    {:<48}{}",
            "  --memory-value-cap c",
            ui::dim("opt-in memory value L2 limit (also caps loaded banks; repeat on resume)")
        );
        println!(
            "    {:<48}{}",
            "  --dream-every n --dream-replay k",
            ui::dim(
                "offline replay every n updates; k memory seeds/entries (default 32; off by default)"
            )
        );
        println!(
            "    {:<48}{}",
            "  --dream-mode memory|generate|both --dream-len n",
            ui::dim(
                "memory, generated, or both; generated length defaults to 64 at temperature 0.8"
            )
        );
        println!(
            "    {:<48}{}",
            "  --lr f --warmup-steps n --total-updates n",
            ui::dim("optimizer schedule; fixed whole-run horizon for fresh chains")
        );
        println!(
            "    {:<48}{}",
            "  --seed n",
            ui::dim("deterministic initialisation")
        );
        println!(
            "    {:<48}{}",
            "  --max-tokens n",
            ui::dim("global cap on training tokens")
        );
        println!(
            "    {:<48}{}",
            "  --hf-dataset OWNER/NAME",
            ui::dim("stream Hugging Face rows; split train and field text by default")
        );
        println!(
            "    {:<48}{}",
            "  --hf-config CONFIG --hf-split SPLIT --hf-field FIELD",
            ui::dim("optional config, split, and dotted text field")
        );
        println!(
            "    {:<48}{}",
            "  --skip-tokens n",
            ui::dim("drop this many tokens from the front first")
        );
        println!(
            "    {:<48}{}",
            "  --token-cache path",
            ui::dim(
                "opt-in persistent binary token cache; local files provide a source fingerprint"
            )
        );
        println!(
            "    {:<48}{}",
            "  --resume path",
            ui::dim("continue from an existing checkpoint")
        );
        println!();
        println!(
            "    --loss-csv PATH --loss-every N  append training curves every N target tokens"
        );
        println!("    --tokens-seen N  offset for a new CSV on resume (existing CSV restores it)");
        println!("    --no-tui         disable cursor updates; keep plain progress logs");
        println!();
        println!("  {}", ui::bold("GENERATE"));
        println!(
            "    {bin} generate <prompt> [-m|--model path] [-t|--temp|--temperature f] [--max-new-tokens n] [--loops n]"
        );
        println!();
        println!("  {}", ui::bold("EXAMPLES"));
        println!(
            "    {}",
            ui::dim("# train a fresh checkpoint on the first 200k tokens")
        );
        println!(
            "    Example: {bin} train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1"
        );
        println!(
            "    Example: {bin} train --hf-dataset Salesforce/wikitext --hf-config wikitext-103-v1 --hf-field text -e 1"
        );
        println!();
        println!(
            "    {}",
            ui::dim("# continue that run on the next slice of the same corpus")
        );
        println!("    {bin} train data/downloaded.txt -o data/ck02.pssa \\");
        println!("      --resume data/model.pssa --max-tokens 200000 --skip-tokens 200000 -e 1");
        println!();
        println!("    {}", ui::dim("# sample from the result"));
        println!("    {bin} generate \"The sun is\" -m data/ck02.pssa --max-new-tokens 64");
        println!();
        println!("  {}", ui::bold("NOTES"));
        println!(
            "    {}",
            ui::dim("--skip-tokens plus --max-tokens is how a chain of runs walks a whole corpus")
        );
        println!(
            "    {}",
            ui::dim("instead of retraining the same prefix every link.")
        );
        println!(
            "    {}",
            ui::dim("--data on generate is a legacy-word provenance check only; V7 BPE")
        );
        println!(
            "    {}",
            ui::dim("checkpoints restore their embedded tokenizer metadata.")
        );
        println!();
    }
    fn canonical_command(command: &str) -> &str {
        match command {
            // Canonical developer-facing names. Historical spellings remain
            // accepted as hidden aliases for scripts and notebooks.
            "repl" => "chat",
            "evaluate" => "score",
            "evaluate-transformer" => "score-transformer",
            other => other,
        }
    }

    fn print_command_help(command: &str) -> Result<(), String> {
        let bin = "pssa";
        let command = Self::canonical_command(command);
        match command {
            "train-transformer" => {
                println!("Usage: {bin} train-transformer [SOURCE] [OPTIONS]");
                println!("Decoder-only baseline: 1 layer, width 256, 4 heads, FFN 448; CPU only.");
                println!(
                    "Same data/tokenizer/window/update schedule as train; context resets each chunk."
                );
                println!("  -d, --data SOURCE   -o, --out PATH (default: data/model.trfm)");
                println!("  -e, --epochs N (4)  --chunk N (64)  --accumulate N (8)");
                println!("  --tokenizer bpe|word (bpe)  --vocab-size N (2048 ceiling)");
                println!(
                    "  --tokenizer-from PSSA_CHECKPOINT  import EXACT tokenizer for comparisons"
                );
                println!("  --max-tokens N  --skip-tokens N (wraps at EOF)");
                println!(
                    "  --lr F (0.001)  --warmup-steps N (0)  --total-updates N  --seed N (42)"
                );
                println!(
                    "  --resume PATH  restore transformer weights, moments, tokenizer and horizon"
                );
                println!(
                    "  --loss-csv PATH --loss-every N (10000)  append target-token loss curve"
                );
                println!("  --tokens-seen N  required when starting a new curve on resume");
                println!("  --threads N     opt-in Rayon pool size; default unchanged");
                println!("  --ram-mib N     Linux prlimit address-space budget, not RSS/VRAM");
                println!("  --no-tui         disable cursor updates; keep plain progress logs");
                println!(
                    "Omit --tokenizer-from on resume; identical chunk/accumulation flags give identical updates."
                );
                println!();
                println!(
                    "Example: {bin} train-transformer data/downloaded.txt -o data/model.trfm --max-tokens 200000 -e 1"
                );
            }
            "train" => {
                println!("Usage: {bin} train [SOURCE] [OPTIONS]");
                println!();
                println!(
                    "Fit a checkpoint on a text corpus. SOURCE may be a local file, directory, URL, hf:REPO, or science."
                );
                println!();
                println!("Options:");
                println!(
                    "  -d, --data <SOURCE>           dataset source (also accepted as SOURCE)"
                );
                println!(
                    "  -o, --out <PATH>              checkpoint output (default: data/model.pssa)"
                );
                println!("  -e, --epochs <N>              positive number of passes (default: 4)");
                println!("      --tokenizer <bpe|word>    tokenizer family (default: bpe)");
                println!("      --vocab-size <N>          BPE vocabulary ceiling (default: 2048)");
                println!("      --latent <N>              latent width (default: 256)");
                println!("      --depth <N>               continuous blocks, 1..32 (default: 1)");
                println!(
                    "      --loops <N>               shared Ouro passes, 1..32 (default: 1; repeat on resume; not checkpointed)"
                );
                println!("      --state <N>               recurrent state width (default: 16)");
                println!(
                    "      --backend <NAME>          auto (default), cpu, webgpu, cuda (feature required)"
                );
                println!(
                    "      --threads <N>             opt-in Rayon pool size (default unchanged)"
                );
                println!(
                    "      --ram-mib <N>             Linux prlimit address-space budget, NOT RSS/VRAM"
                );
                println!("      --key <N>                 memory key width (default: 32)");
                println!("      --memory <N>              memory capacity (default: 512)");
                println!("      --chunk <N>               training chunk length (default: 64)");
                println!("      --batch-size <N>          independent document lanes (default: 1)");
                println!(
                    "      --accumulate <N>          microbatches per optimizer update (default: 8)"
                );
                println!(
                    "      --grad-clip <MAX_NORM>     opt-in global gradient L2 limit, finite >0; bad norms skip Adam (abort on 21st consecutive skip)"
                );
                println!(
                    "      --memory-value-cap <C>    opt-in memory value L2 limit, finite >0; caps loaded banks and writes"
                );
                println!(
                    "      --dream-every <N>         memory replay cadence in optimizer updates (default: off)"
                );
                println!(
                    "      --dream-replay <K>        occupied episodic entries/seeds per dream (default: 32)"
                );
                println!(
                    "      --dream-mode <M>          memory, generate, or both (default: memory)"
                );
                println!(
                    "      --dream-len <N>           generated tokens per seed (default: 64; temp 0.8)"
                );
                println!(
                    "      --dream-lr <F>            main-weight rehearsal SGD rate (default: 0.006)"
                );
                println!(
                    "      --dream-steps <N>         rehearsal passes per sequence (default: 1)"
                );
                println!(
                    "                                dream controls are runtime-only; repeat flags on resume"
                );
                println!("      --lr <F>                  base learning rate (default: 0.001)");
                println!("      --warmup-steps <N>        linear warm-up updates (default: 0)");
                println!(
                    "      --total-updates <N>       fixed whole-run schedule horizon (fresh run)"
                );
                println!("      --seed <N>                initialization seed (default: 42)");
                println!("      --max-tokens <N>          global token cap");
                println!(
                    "      --hf-dataset <OWNER/NAME> stream Hugging Face rows (default split train, field text)"
                );
                println!("      --hf-config <CONFIG>      optional dataset configuration");
                println!("      --hf-split <SPLIT>        dataset split (default train)");
                println!(
                    "      --hf-field <FIELD>        text field, including dotted paths (default text)"
                );
                println!("      --skip-tokens <N>         offset into the corpus; wraps at EOF");
                println!(
                    "      --token-cache <PATH>      opt-in persistent binary cache (local source fingerprinted)"
                );
                println!(
                    "      --resume <PATH>            continue optimizer/model state from a checkpoint"
                );
                println!(
                    "      --loss-csv <PATH>          append target-token training loss curve"
                );
                println!(
                    "      --loss-every <N>           token cadence (default: 10000), at update boundaries"
                );
                println!(
                    "      --tokens-seen <N>          offset for a new CSV on resume; otherwise restored"
                );
                println!(
                    "      --no-tui                   disable cursor updates; keep plain progress logs"
                );
                println!();
                println!("Example:");
                println!(
                    "  {bin} train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1"
                );
                println!(
                    "  {bin} train --hf-dataset Salesforce/wikitext --hf-config wikitext-103-v1 --hf-field text -e 1"
                );
                println!(
                    "  {bin} train data/downloaded.txt -o data/ck02.pssa --resume data/ck01.pssa --skip-tokens 200000 --max-tokens 200000 -e 1"
                );
            }
            "generate" | "generate-transformer" => {
                let model = if command == "generate" {
                    "data/model.pssa"
                } else {
                    "data/model.trfm"
                };
                println!("Usage: {bin} {command} [PROMPT] [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -p, --prompt <TEXT>           prompt (also accepted as PROMPT)");
                println!("  -m, --model <PATH>            checkpoint (default: {model})");
                println!(
                    "  -d, --data <SOURCE>           legacy PSSA word-tokenizer provenance; transformer checkpoints reject it"
                );
                println!(
                    "  -t, --temp, --temperature <F> sampling temperature; 0 is greedy (default: 0.70)"
                );
                println!(
                    "      --max-new-tokens <N>      generation cap (default: 64, max: {MAX_GENERATION_TOKENS})"
                );
                if command == "generate" {
                    println!(
                        "      --loops <N>               shared Ouro passes, 1..32 (repeat; not checkpointed)"
                    );
                }
                println!();
                println!("Example:");
                println!("  {bin} {command} -m {model} -p \"The sun is\"");
                println!(
                    "  {bin} {command} \"quantum mechanics\" --temperature 0 --max-new-tokens 32"
                );
            }
            "chat" | "repl" => {
                println!("Usage: {bin} chat [DATA] [OPTIONS]");
                println!();
                println!(
                    "Start an interactive prompt loop. DATA is an optional legacy word-tokenizer provenance source."
                );
                println!("Options:");
                println!("  -m, --model <PATH>            checkpoint (default: data/model.pssa)");
                println!("  -d, --data <SOURCE>           legacy tokenizer provenance source");
                println!("  -t, --temp, --temperature <F> sampling temperature (default: 0.70)");
                println!(
                    "      --loops <N>               shared Ouro passes, 1..32 (repeat; not checkpointed)"
                );
                println!();
                println!("Example: {bin} chat -m data/model.pssa --temperature 0.7");
                println!("Commands: /exit (or quit), /info, /temp <value> (finite and >= 0)");
            }
            "score" | "score-transformer" => {
                println!("Usage: {bin} {command} [DATA] [-m|--model PATH] [-d|--data SOURCE]");
                println!();
                println!("Evaluate a checkpoint and print one JSON metrics object.");
                println!(
                    "  --skip-tokens N --max-tokens N  strict held-out slice, never wraps at EOF"
                );
                if command == "score" {
                    println!(
                        "  --loops N                      shared Ouro passes, 1..32 (repeat; not checkpointed)"
                    );
                }
                println!("Use the embedded tokenizer; no training or checkpoint writes.");
                let model = if command == "score" {
                    "data/model.pssa"
                } else {
                    "data/model.trfm"
                };
                println!("Example: {bin} {command} data/heldout.txt --model {model}");
            }
            "clean-wikitext" => {
                println!("Usage: {bin} clean-wikitext INPUT -o OUTPUT");
                println!();
                println!("Stream-clean a local UTF-8 WikiText raw dump, one line at a time.");
                println!(
                    "  -o, --out <PATH>  required new output file; existing files are never overwritten"
                );
                println!(
                    "Removes headings and <unk>, joins @-@ / @.@ / @,@, normalizes punctuation spacing,"
                );
                println!(
                    "and collapses blank lines. Output uses LF endings. Input is never modified."
                );
                println!(
                    "Run once before a fresh training chain; do not switch corpora mid-resume."
                );
                println!(
                    "Example: {bin} clean-wikitext wiki.train.raw --out data/wikitext-clean.txt"
                );
            }
            "download" => {
                println!("Usage: {bin} download REPOSITORY [-o|--out PATH]");
                println!();
                println!("Download the train split from Hugging Face as plain text.");
                println!("Example: {bin} download wikimedia/wikipedia --out data/downloaded.txt");
            }
            "compare" => {
                println!("Usage: {bin} compare DATA --chain-dir DIR --out NEW_DIR [OPTIONS]");
                println!(
                    "Replay an existing PSSA chain with the token/update-matched transformer; score both on unseen tokens."
                );
                println!(
                    "--links 64 --window 200000 --batch-size 8 --accumulate 1 --eval-tokens 200000"
                );
                println!(
                    "--eval-skip-tokens N --link-plan JSON --loss-every 10000 --seed 42 --warmup-steps 0"
                );
                println!("Use the ORIGINAL corpus and training settings. See docs/COMPARISON.md.");
            }
            "benchmark" => {
                println!("Usage: {bin} benchmark [-f|--feature NAME] [-o|--out PATH]");
                println!("Features: continual, retention, geometry, refractory, ridge, or all.");
                println!("Without options, runs the historical verification smoke test.");
                println!("Feature runs require --feature; --out selects their output directory.");
                println!(
                    "Example: {bin} benchmark --feature geometry --out __agent__/feature_results"
                );
            }
            "status" => {
                println!("Usage: {bin} status");
                println!();
                println!(
                    "List nearby corpora and checkpoints, including model shapes and optimizer steps."
                );
                println!("Example: {bin} status");
            }
            "gpu-probe" => {
                println!("Usage: {bin} gpu-probe");
                println!();
                println!(
                    "Probe the available GPU backend and compare one GEMM against the CPU reference."
                );
                println!("Example: {bin} gpu-probe");
            }
            "throughput" => {
                println!(
                    "Usage: {bin} throughput [DATA] [-d|--data SOURCE] [-m|--model PATH] [--skip-tokens N] [--max-tokens N]"
                );
                println!();
                println!(
                    "Measure frozen-model scoring throughput. DATA defaults to the normal training source."
                );
                println!(
                    "Options: -d, --data SOURCE; -m, --model PATH; --skip-tokens N; --max-tokens N"
                );
                println!(
                    "Example: {bin} throughput data/heldout.txt --model data/model.pssa --max-tokens 10000"
                );
            }
            "help" => {
                println!("Usage: {bin} help [COMMAND]");
                println!();
                println!("Show the complete command list or one command's options and example.");
                println!("Example: {bin} help train");
            }
            "tui" => {
                println!("Usage: {bin} tui [-c|--chain DIR] [--compare LOG] [--chats-dir DIR]");
                println!();
                println!(
                    "Open the dashboard and local inference chat; piped output is passed through plainly."
                );
                println!(
                    "Example: {bin} train data/downloaded.txt --no-tui | {bin} tui --compare transformer.log"
                );
            }
            _ => return Err(format!("unknown command '{command}'; run {bin} help")),
        }
        Ok(())
    }

    pub fn parse_and_execute(args: Vec<String>) -> Result<(), String> {
        if args.len() < 2 {
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                return crate::tui::run(&[]);
            }
            Self::print_home();
            return Ok(());
        }
        let command = Self::canonical_command(args[1].as_str());
        if command == "help" {
            if args.len() == 2 {
                Self::print_help();
                return Ok(());
            }
            if args.len() == 3 {
                if matches!(args[2].as_str(), "--help" | "-h") {
                    return Self::print_command_help("help");
                }
                return Self::print_command_help(&args[2]);
            }
            if args.len() == 4 && matches!(args[3].as_str(), "--help" | "-h") {
                return Self::print_command_help(&args[2]);
            }
            return Err("help accepts one command name, optionally followed by --help".into());
        }
        if matches!(command, "--help" | "-h") {
            Self::print_help();
            return Ok(());
        }
        if args[2..]
            .iter()
            .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
        {
            return Self::print_command_help(command);
        }
        match command {
            "help" | "--help" | "-h" => {
                Self::print_help();
                Ok(())
            }
            "train" | "train-transformer" => {
                let baseline = command == "train-transformer";
                let mut allowed = vec![
                    "--data",
                    "-d",
                    "--hf-dataset",
                    "--hf-config",
                    "--hf-split",
                    "--hf-field",
                    "--out",
                    "-o",
                    "--epochs",
                    "-e",
                    "--latent",
                    "--state",
                    "--key",
                    "--memory",
                    "--chunk",
                    "--lr",
                    "--accumulate",
                    "--warmup-steps",
                    "--total-updates",
                    "--seed",
                    "--max-tokens",
                    "--tokenizer",
                    "--vocab-size",
                    "--resume",
                    "--skip-tokens",
                    "--loss-csv",
                    "--loss-every",
                    "--tokens-seen",
                    "--no-tui",
                    "--threads",
                    "--ram-mib",
                ];
                if baseline {
                    allowed.retain(|x| !["--latent", "--state", "--key", "--memory"].contains(x));
                    allowed.push("--tokenizer-from");
                } else {
                    allowed.extend([
                        "--batch-size",
                        "--depth",
                        "--loops",
                        "--backend",
                        "--grad-clip",
                        "--memory-value-cap",
                        "--token-cache",
                        "--dream-every",
                        "--dream-replay",
                        "--dream-mode",
                        "--dream-len",
                        "--dream-lr",
                        "--dream-steps",
                    ]);
                }
                let p = Parsed::parse(&args[2..], &allowed)?;
                for flag in ["--threads", "--ram-mib"] {
                    if p.string(flag, "") == Some("") {
                        return Err(format!("{flag} must be a positive integer"));
                    }
                }
                let limits = resource_limits::ResourceLimits::from_inputs([
                    p.string("--threads", "").unwrap_or(""),
                    p.string("--ram-mib", "").unwrap_or(""),
                    p.string("--batch-size", "").unwrap_or(""),
                    p.string("--max-tokens", "").unwrap_or(""),
                ])?;
                if baseline && p.string("--hf-dataset", "").is_some() {
                    return Err("--hf-dataset is supported by train, not train-transformer".into());
                }
                for (names, label) in [
                    (&["--data", "-d"][..], "--data"),
                    (&["--out", "-o"][..], "--out"),
                    (&["--epochs", "-e"][..], "--epochs"),
                ] {
                    p.reject_duplicate_aliases(names, label)?;
                }
                if p.positional.len() > 1 {
                    return Err("train accepts at most one positional source".into());
                }
                if p.positional.len() == 1 && p.string("--data", "-d").is_some() {
                    return Err(
                        "train source was specified both positionally and with --data".into(),
                    );
                }
                let explicit_data = p.string("--data", "-d").is_some() || !p.positional.is_empty();
                if p.string("--hf-dataset", "").is_some() && explicit_data {
                    return Err("--hf-dataset cannot be combined with SOURCE or --data; choose one dataset source".into());
                }
                let data = p
                    .string("--data", "-d")
                    .map(str::to_string)
                    .or_else(|| p.positional.first().cloned())
                    .unwrap_or_else(Self::default_data);
                let default_out = if baseline {
                    "data/model.trfm"
                } else {
                    "data/model.pssa"
                };
                let out = p.string("--out", "-o").unwrap_or(default_out);
                // Validate flags, then apply the process budget before any model
                // or resume allocation. Without limits the old path is unchanged.
                if limits.ram_mib.is_some() {
                    Self::common_options(&p)?;
                    limits.apply_process_budget(&args)?;
                }
                if baseline {
                    let mut opts = Self::common_options(&p)?;
                    if let Some(path) = &opts.resume {
                        if p.string("--tokenizer-from", "").is_some() {
                            return Err(
                                "--resume restores its own tokenizer; omit --tokenizer-from".into(),
                            );
                        }
                        let model =
                            crate::transformer_checkpoint::load_checkpoint(path).map_err(|e| {
                                format!("cannot inspect resume checkpoint '{path}': {e}")
                            })?;
                        if p.flags.contains_key("--chunk") && opts.chunk != model.cfg.chunk_len {
                            return Err(format!(
                                "--chunk={} does not match resume checkpoint value {}",
                                opts.chunk, model.cfg.chunk_len
                            ));
                        }
                        if !p.flags.contains_key("--lr") {
                            opts.lr = model.cfg.lr;
                        }
                        if let (Some(requested), Some(stored)) =
                            (opts.schedule_total_updates, model.lr_schedule_total_updates)
                            && requested != stored
                        {
                            return Err(format!(
                                "--total-updates={requested} does not match resume checkpoint horizon {stored}"
                            ));
                        }
                    }
                    limits.run(|| {
                        crate::transformer_training::run_training(
                            &data,
                            &opts,
                            out,
                            p.string("--tokenizer-from", ""),
                        )
                    })
                } else {
                    limits.run(|| Self::run_training(&data, &Self::options(&p)?, out))
                }
            }
            "generate" | "generate-transformer" => {
                let mut allowed = vec![
                    "--prompt",
                    "-p",
                    "--model",
                    "-m",
                    "--data",
                    "-d",
                    "--temp",
                    "--temperature",
                    "-t",
                    "--max-new-tokens",
                ];
                if command == "generate" {
                    allowed.push("--loops");
                }
                let p = Parsed::parse(&args[2..], &allowed)?;
                let loops = p.loops()?;
                for (names, label) in [
                    (&["--prompt", "-p"][..], "--prompt"),
                    (&["--model", "-m"][..], "--model"),
                    (&["--data", "-d"][..], "--data"),
                    (&["--temp", "--temperature", "-t"][..], "--temperature"),
                ] {
                    p.reject_duplicate_aliases(names, label)?;
                }
                if p.positional.len() > 1 {
                    return Err("generate accepts one positional prompt".into());
                }
                if p.positional.len() == 1 && p.string("--prompt", "-p").is_some() {
                    return Err(
                        "generate prompt was specified both positionally and with --prompt".into(),
                    );
                }
                let prompt = p
                    .string("--prompt", "-p")
                    .or_else(|| p.positional.first().map(String::as_str))
                    .ok_or("generate requires a prompt")?;
                let temp =
                    p.f32_first(&["--temp", "--temperature", "-t"], "--temperature", 0.70)?;
                if temp < 0.0 {
                    return Err("--temp must be >= 0".into());
                }
                let max = p.usize_nonzero("--max-new-tokens", "", 64)?;
                if max > MAX_GENERATION_TOKENS {
                    return Err(format!(
                        "--max-new-tokens must be at most {MAX_GENERATION_TOKENS}"
                    ));
                }
                let text = if command == "generate-transformer" {
                    if p.string("--data", "-d").is_some() {
                        return Err(
                            "transformer checkpoints embed their tokenizer; omit --data".into()
                        );
                    }
                    crate::transformer_inference::generate(
                        p.string("--model", "-m").unwrap_or("data/model.trfm"),
                        prompt,
                        &InferenceConfig {
                            temperature: temp,
                            max_new_tokens: max,
                            top_k: if temp == 0.0 { 1 } else { 24 },
                            repetition_penalty: if temp == 0.0 { 1.0 } else { 1.25 },
                            ..Default::default()
                        },
                    )?
                } else {
                    Self::run_generate_with_loops(
                        prompt,
                        p.string("--model", "-m").unwrap_or("data/model.pssa"),
                        p.string("--data", "-d"),
                        temp,
                        max,
                        loops,
                    )?
                };
                println!("{text}");
                Ok(())
            }
            "score" | "score-transformer" => {
                let mut allowed = vec![
                    "--model",
                    "-m",
                    "--data",
                    "-d",
                    "--skip-tokens",
                    "--max-tokens",
                ];
                if command == "score" {
                    allowed.push("--loops");
                }
                let p = Parsed::parse(&args[2..], &allowed)?;
                let loops = p.loops()?;
                p.reject_duplicate_aliases(&["--model", "-m"], "--model")?;
                p.reject_duplicate_aliases(&["--data", "-d"], "--data")?;
                if p.positional.len() > 1 {
                    return Err("score accepts one positional data source".into());
                }
                if p.positional.len() == 1 && p.string("--data", "-d").is_some() {
                    return Err("score data was specified both positionally and with --data".into());
                }
                let data = p
                    .string("--data", "-d")
                    .map(str::to_owned)
                    .or_else(|| p.positional.first().cloned())
                    .unwrap_or_else(Self::default_data);
                let slice = crate::evaluation::EvaluationSlice {
                    skip_tokens: p.required_usize("--skip-tokens", "", 0)?,
                    max_tokens: p
                        .string("--max-tokens", "")
                        .map(|_| p.usize_nonzero("--max-tokens", "", 1))
                        .transpose()?,
                };
                if command == "score-transformer" {
                    crate::transformer_inference::evaluate_slice(
                        p.string("--model", "-m").unwrap_or("data/model.trfm"),
                        &data,
                        slice,
                    )
                } else {
                    Self::run_evaluate(
                        p.string("--model", "-m").unwrap_or("data/model.pssa"),
                        &data,
                        slice,
                        loops,
                    )
                }
            }
            "chat" | "repl" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
                        "--model",
                        "-m",
                        "--data",
                        "-d",
                        "--temp",
                        "--temperature",
                        "-t",
                        "--loops",
                    ],
                )?;
                let loops = p.loops()?;
                p.reject_duplicate_aliases(&["--model", "-m"], "--model")?;
                p.reject_duplicate_aliases(&["--data", "-d"], "--data")?;
                p.reject_duplicate_aliases(&["--temp", "--temperature", "-t"], "--temperature")?;
                if p.positional.len() > 1 {
                    return Err("chat accepts one positional data source".into());
                }
                if p.positional.len() == 1 && p.string("--data", "-d").is_some() {
                    return Err("chat data was specified both positionally and with --data".into());
                }
                let t = p.f32_first(&["--temp", "--temperature", "-t"], "--temperature", 0.70)?;
                if t < 0.0 {
                    return Err("--temperature must be >= 0".into());
                }
                Self::run_chat(
                    p.string("--model", "-m").unwrap_or("data/model.pssa"),
                    p.string("--data", "-d")
                        .or_else(|| p.positional.first().map(String::as_str)),
                    t,
                    loops,
                )
            }
            "clean-wikitext" => {
                let p = Parsed::parse(&args[2..], &["--out", "-o"])?;
                p.reject_duplicate_aliases(&["--out", "-o"], "--out")?;
                if p.positional.len() != 1 {
                    return Err("clean-wikitext requires one input file; use clean-wikitext INPUT -o OUTPUT".into());
                }
                let output = p.string("--out", "-o").ok_or(
                    "clean-wikitext requires --out OUTPUT (or -o OUTPUT); choose a new output file",
                )?;
                Self::run_clean_wikitext(&p.positional[0], output)
            }
            "download" => {
                let p = Parsed::parse(&args[2..], &["--out", "-o"])?;
                p.reject_duplicate_aliases(&["--out", "-o"], "--out")?;
                if p.positional.len() != 1 {
                    return Err("download requires one Hugging Face repository".into());
                }
                let text = DatasetManager::download_huggingface_dataset(&p.positional[0])?;
                std::fs::write(
                    p.string("--out", "-o").unwrap_or("data/downloaded.txt"),
                    text,
                )
                .map_err(|e| e.to_string())
            }
            "status" => {
                if args.len() != 2 {
                    return Err("status takes no options".into());
                }
                Self::run_status()
            }
            "gpu-probe" => {
                if args.len() != 2 {
                    return Err("gpu-probe takes no options".into());
                }
                run_gpu_probe()
            }
            "compare" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
                        "--chain-dir",
                        "--out",
                        "-o",
                        "--links",
                        "--window",
                        "--batch-size",
                        "--accumulate",
                        "--link-plan",
                        "--eval-skip-tokens",
                        "--eval-tokens",
                        "--loss-every",
                        "--seed",
                        "--warmup-steps",
                    ],
                )?;
                if p.positional.len() != 1 {
                    return Err("compare requires exactly one original corpus path".into());
                }
                p.reject_duplicate_aliases(&["--out", "-o"], "output")?;
                let opts = crate::comparison::ComparisonOptions {
                    chain_dir: p
                        .string("--chain-dir", "")
                        .ok_or("compare requires --chain-dir")?
                        .into(),
                    out_dir: p
                        .string("--out", "-o")
                        .ok_or("compare requires --out NEW_DIR")?
                        .into(),
                    links: p.usize_nonzero("--links", "", 64)?,
                    window: p.usize_nonzero("--window", "", 200000)?,
                    batch_size: p.usize_nonzero("--batch-size", "", 8)?,
                    accumulate: p.usize_nonzero("--accumulate", "", 1)?,
                    link_plan: p.string("--link-plan", "").map(str::to_owned),
                    eval_skip_tokens: p
                        .string("--eval-skip-tokens", "")
                        .map(|s| {
                            s.parse()
                                .map_err(|_| "--eval-skip-tokens must be an unsigned integer")
                        })
                        .transpose()?,
                    eval_tokens: p.usize_nonzero("--eval-tokens", "", 200000)?,
                    loss_every: p.usize_nonzero("--loss-every", "", 10000)?,
                    seed: p
                        .string("--seed", "")
                        .unwrap_or("42")
                        .parse()
                        .map_err(|_| "--seed must be an unsigned integer")?,
                    legacy_warmup: p.required_usize("--warmup-steps", "", 0)?,
                };
                crate::comparison::run(&p.positional[0], &opts)
            }
            "benchmark" => {
                if args.len() == 2 {
                    Self::run_benchmark()
                } else {
                    let p = Parsed::parse(&args[2..], &["--feature", "-f", "--out", "-o"])?;
                    p.reject_duplicate_aliases(&["--feature", "-f"], "--feature")?;
                    p.reject_duplicate_aliases(&["--out", "-o"], "--out")?;
                    if !p.positional.is_empty() {
                        return Err("benchmark accepts only --feature and --out".into());
                    }
                    let feature = p
                        .string("--feature", "-f")
                        .ok_or("benchmark requires --feature when options are supplied")?;
                    crate::feature_benchmark::run(feature, p.string("--out", "-o"))
                }
            }
            "throughput" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
                        "--model",
                        "-m",
                        "--data",
                        "-d",
                        "--skip-tokens",
                        "--max-tokens",
                    ],
                )?;
                p.reject_duplicate_aliases(&["--model", "-m"], "--model")?;
                p.reject_duplicate_aliases(&["--data", "-d"], "--data")?;
                if p.positional.len() > 1 {
                    return Err("throughput accepts one positional data source".into());
                }
                if p.positional.len() == 1 && p.string("--data", "-d").is_some() {
                    return Err(
                        "throughput data was specified both positionally and with --data".into(),
                    );
                }
                let data = p
                    .string("--data", "-d")
                    .map(str::to_owned)
                    .or_else(|| p.positional.first().cloned())
                    .unwrap_or_else(Self::default_data);
                let slice = crate::evaluation::EvaluationSlice {
                    skip_tokens: p.required_usize("--skip-tokens", "", 0)?,
                    max_tokens: p
                        .string("--max-tokens", "")
                        .map(|_| p.usize_nonzero("--max-tokens", "", 1))
                        .transpose()?,
                };
                Self::run_throughput(
                    p.string("--model", "-m").unwrap_or("data/model.pssa"),
                    &data,
                    slice,
                )
            }
            "tui" => crate::tui::run(&args[2..]),
            _ => Err(format!("unknown command '{}'; run pssa help", args[1])),
        }
    }
}
fn probe_max_abs_diff(actual: &[f32], expected: &[f32]) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "GPU probe output length {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    if actual.iter().chain(expected).any(|x| !x.is_finite()) {
        return Err("GPU probe comparison contains nonfinite values".into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max))
}

#[cfg(test)]
mod training_safeguards_tests {
    use super::*;

    #[test]
    fn opt_in_flags_require_positive_finite_f32() {
        let defaults = CLIHandler::common_options(&Parsed::parse(&[], &[]).unwrap()).unwrap();
        assert_eq!(defaults.grad_clip, None);
        assert_eq!(defaults.memory_value_cap, None);
        for flag in ["--grad-clip", "--memory-value-cap"] {
            for bad in ["0", "-1", "NaN", "inf", "1e40", "1e-50", "garbage"] {
                let parsed = Parsed::parse(&[flag.into(), bad.into()], &[flag]).unwrap();
                assert!(CLIHandler::common_options(&parsed).is_err(), "{flag}={bad}");
            }
            let parsed = Parsed::parse(&[flag.into(), "1.0".into()], &[flag]).unwrap();
            let options = CLIHandler::common_options(&parsed).unwrap();
            assert_eq!(
                if flag == "--grad-clip" {
                    options.grad_clip
                } else {
                    options.memory_value_cap
                },
                Some(1.0)
            );
        }
    }

    #[test]
    fn dream_rehearsal_flags_parse_and_reject_invalid_values() {
        let parsed = Parsed::parse(
            &[
                "--dream-lr".into(),
                "0.002".into(),
                "--dream-steps".into(),
                "3".into(),
            ],
            &["--dream-lr", "--dream-steps"],
        )
        .unwrap();
        let options = CLIHandler::options(&parsed).unwrap();
        assert_eq!(options.dream_lr, 0.002);
        assert_eq!(options.dream_steps, 3);
        for (flag, bad) in [
            ("--dream-lr", "0"),
            ("--dream-lr", "NaN"),
            ("--dream-steps", "0"),
        ] {
            let parsed =
                Parsed::parse(&[flag.into(), bad.into()], &["--dream-lr", "--dream-steps"])
                    .unwrap();
            assert!(CLIHandler::options(&parsed).is_err(), "{flag}={bad}");
        }
    }

    #[test]
    fn dream_phase_events_report_finite_loss_or_explicit_unavailable() {
        assert_eq!(
            dream_phase_start_line(7, DreamMode::Both),
            "dream phase=start update=7 mode=both"
        );

        let finite = crate::dream::DreamSummary {
            last_rehearsal_loss: Some(0.125),
            ..Default::default()
        };
        let line = dream_phase_end_line(7, DreamMode::Both, &finite);
        assert!(line.contains("dream phase=end update=7 mode=both"));
        assert!(line.contains("dream_loss=1.250000e-1"));
        assert!(!line.contains("loss=nan"));

        for loss in [None, Some(f32::NAN), Some(f32::INFINITY)] {
            let summary = crate::dream::DreamSummary {
                last_rehearsal_loss: loss,
                ..Default::default()
            };
            let line = dream_phase_end_line(8, DreamMode::Memory, &summary);
            assert!(line.contains("dream_loss=unavailable"), "{line}");
            assert!(!line.contains("loss=nan"), "{line}");
        }
    }

    #[test]
    fn skip_limit_allows_twenty_and_resets_after_success() {
        let mut skipped = SkippedUpdates::default();
        for _ in 0..MAX_CONSECUTIVE_SKIPPED_UPDATES {
            skipped.record(f64::INFINITY, 7).unwrap();
        }
        skipped.consecutive = 0; // the successful-update branch
        for _ in 0..MAX_CONSECUTIVE_SKIPPED_UPDATES {
            skipped.record(f64::NAN, 8).unwrap();
        }
        assert!(skipped.record(f64::NAN, 8).is_err());
        assert_eq!(skipped.total, 41);
    }
}

#[cfg(test)]
mod probe_tests {
    use super::probe_max_abs_diff;

    #[test]
    fn comparison_rejects_short_or_nonfinite_gpu_results() {
        assert!(probe_max_abs_diff(&[], &[0.0]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(probe_max_abs_diff(&[bad], &[0.0]).is_err());
            assert!(probe_max_abs_diff(&[0.0], &[bad]).is_err());
        }
        assert_eq!(
            probe_max_abs_diff(&[1.0, -2.0], &[1.25, -2.0]).unwrap(),
            0.25
        );
    }
}

/// Bring up a GPU compute device, run a strict GEMM dispatch, and check the
/// result against the CPU reference implementation without fallback.
pub fn run_gpu_probe() -> Result<(), String> {
    println!("=== pssa gpu-probe ===");
    let device = match Device::try_gpu() {
        Ok(d) => {
            println!("adapter: GPU compute device acquired");
            d
        }
        Err(e) => {
            println!("adapter: unavailable ({})", e);
            println!("result: no GPU on this machine; training stays on CPU");
            return Ok(());
        }
    };

    let ctx = match device.gpu() {
        Some(ctx) => {
            println!("backend: {}", ctx.backend_label());
            ctx
        }
        None => {
            println!("result: CPU device returned; nothing to probe");
            return Ok(());
        }
    };

    let (batch, m, n, k) = (2usize, 64usize, 96usize, 128usize);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let x: Vec<f32> = (0..batch * m * k).map(|_| next()).collect();
    let w: Vec<f32> = (0..n * k).map(|_| next()).collect();

    let t_gpu = Instant::now();
    let y_gpu = ctx
        .try_dispatch_gemm(&x, &w, m, n, k, batch)
        .map_err(|e| format!("GPU probe dispatch failed (no CPU fallback): {e}"))?;
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    let t_cpu = Instant::now();
    let y_cpu = gemm_cpu_reference(&x, &w, m, n, k, batch);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let max_abs = probe_max_abs_diff(&y_gpu, &y_cpu)?;

    println!("shape: batch={} M={} N={} K={}", batch, m, n, k);
    println!("gpu:   {:.3} ms", gpu_ms);
    println!("cpu:   {:.3} ms", cpu_ms);
    println!("max_abs_diff: {:.3e}", max_abs);
    if max_abs < 1e-3 {
        println!("result: PASS, GPU kernel matches CPU reference");
    } else {
        println!("result: FAIL, GPU kernel diverges from CPU reference");
        return Err(format!(
            "GPU GEMM differs from CPU reference by {max_abs:.3e}"
        ));
    }
    println!("note: this verifies GPU GEMM, not end-to-end training or hardware backward parity");
    Ok(())
}
