use crate::backend::{Device, gemm_cpu_reference};
use crate::checkpoint::{self, CheckpointFormat};
use crate::dataset::{DatasetManager, Tokenizer, TokenizerKind};
use crate::inference::{InferenceConfig, PSSAInferenceEngine};
use crate::pssa::{PSSAConfigV2, PSSALayerV2};
use crate::ui;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::time::Instant;

/// Keep command-line generation requests bounded before inference preallocates
/// its token and byte buffers.
const MAX_GENERATION_TOKENS: usize = 100_000;

#[derive(Clone, Debug)]
pub struct TrainingOptions {
    pub epochs: usize,
    pub latent: usize,
    pub state: usize,
    pub key: usize,
    pub memory: usize,
    pub chunk: usize,
    pub lr: f32,
    pub accumulate: usize,
    pub warmup_steps: usize,
    /// Fixed total optimizer-update horizon shared by all resumed links.
    /// `None` preserves the legacy per-link schedule behavior.
    pub schedule_total_updates: Option<usize>,
    pub seed: u64,
    /// A global cap across documents; documents are never individually reset to this cap.
    pub max_tokens: Option<usize>,
    pub tokenizer: TokenizerKind,
    pub vocab_size: usize,
    /// Continue training from an existing checkpoint instead of fresh initialization.
    pub resume: Option<String>,
    /// Skip this many encoded tokens from the front of the corpus before training.
    pub skip_tokens: usize,
}
impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 4,
            latent: 256,
            state: 16,
            key: 32,
            memory: 512,
            chunk: 64,
            lr: 1e-3,
            accumulate: 8,
            warmup_steps: 0,
            schedule_total_updates: None,
            seed: 42,
            max_tokens: None,
            skip_tokens: 0,
            tokenizer: TokenizerKind::Bpe,
            vocab_size: 2048,
            resume: None,
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
                if i + 1 >= args.len()
                    || (args[i + 1].starts_with('-') && allowed.contains(args[i + 1].as_str()))
                {
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
        let count = names.iter().filter(|name| self.flags.contains_key(**name)).count();
        if count > 1 {
            Err(format!("{label} was specified more than once (use one spelling)"))
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
impl CLIHandler {
    fn default_data() -> String {
        if std::path::Path::new("data/downloaded.txt").exists() {
            "data/downloaded.txt".into()
        } else {
            "science".into()
        }
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
            epochs: parsed.usize_nonzero("--epochs", "-e", 4)?,
            latent: parsed.usize_nonzero("--latent", "", 256)?,
            state: parsed.usize_nonzero("--state", "", 16)?,
            key: parsed.usize_nonzero("--key", "", 32)?,
            memory: parsed.usize_nonzero("--memory", "", 512)?,
            chunk: parsed.usize_nonzero("--chunk", "", 64)?,
            lr: parsed.f32("--lr", "", 1e-3)?,
            accumulate: parsed.usize_nonzero("--accumulate", "", 8)?,
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
            tokenizer,
            vocab_size: parsed.usize_nonzero("--vocab-size", "", 2048)?,
            resume: parsed.string("--resume", "").map(str::to_string),
            skip_tokens: parsed.required_usize("--skip-tokens", "", 0)?,
        };
        if !(x.lr > 0.0) {
            return Err("--lr must be positive".into());
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
            ("--state", x.state, 4_096),
            ("--key", x.key, 4_096),
            ("--memory", x.memory, 1_000_000),
            ("--chunk", x.chunk, 65_536),
            ("--accumulate", x.accumulate, 1_000_000),
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
                ("--state", "--state", x.state, loaded.model.cfg.d_state),
                ("--key", "--key", x.key, loaded.model.cfg.d_mem_key),
                ("--memory", "--memory", x.memory, loaded.model.cfg.mem_capacity),
                ("--chunk", "--chunk", x.chunk, loaded.model.cfg.chunk_len),
            ];
            for (label, flag, requested, actual) in checks {
                if parsed.flags.contains_key(flag) && requested != actual {
                    return Err(format!(
                        "{label}={requested} does not match resume checkpoint value {actual}"
                    ));
                }
            }
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
        // Tokenize once, then walk the nonempty documents cyclically.  A
        // chained Kaggle window may cross EOF; returning a second segment from
        // the beginning is preferable to silently training on fewer tokens.
        let encoded: Vec<Vec<usize>> = raw
            .lines()
            .map(|line| tokenizer.try_encode(line, true))
            .collect::<Result<_, _>>()?;
        let nonempty: Vec<&[usize]> = encoded
            .iter()
            .map(Vec::as_slice)
            .filter(|ids| !ids.is_empty())
            .collect();
        let total = nonempty.iter().try_fold(0usize, |sum, ids| {
            sum.checked_add(ids.len())
                .ok_or_else(|| "dataset token count overflow".to_string())
        })?;
        if total < 2 {
            return Err("dataset has no token transitions".into());
        }

        let mut remaining = limit.unwrap_or(total.saturating_sub(skip % total));
        if remaining == 0 {
            return Err("dataset has no token transitions in the selected window".into());
        }
        let mut offset = skip % total;
        let mut doc_index = 0;
        while offset >= nonempty[doc_index].len() {
            offset -= nonempty[doc_index].len();
            doc_index = (doc_index + 1) % nonempty.len();
        }

        let mut docs = Vec::new();
        while remaining > 0 {
            let ids = nonempty[doc_index];
            let take = (ids.len() - offset).min(remaining);
            if take >= 2 {
                docs.push(ids[offset..offset + take].to_vec());
            }
            remaining -= take;
            doc_index = (doc_index + 1) % nonempty.len();
            offset = 0;
            if limit.is_none() && doc_index == 0 {
                break;
            }
        }
        if docs.is_empty() {
            Err("dataset has no token transitions in the selected window".into())
        } else {
            Ok(docs)
        }
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
        [
            &model.embed_w,
            &model.a_mat,
            &model.w_delta,
            &model.w_b,
            &model.w_c,
            &model.w_qx,
            &model.w_qh,
            &model.w_gate,
            &model.w_proj,
            &model.mlp_w1,
            &model.mlp_w2,
            &model.unembed_w,
        ]
        .iter()
        .all(|p| matrix_finite(p))
            && vector_finite(&model.norm_gamma)
            && vector_finite(&model.norm_beta)
            && model
                .adapters
                .iter()
                .all(|ad| matrix_finite(&ad.down_proj) && matrix_finite(&ad.up_proj))
            && model.h_persistent.iter().all(|v| v.is_finite())
            && model.memory.keys.iter().all(|v| v.is_finite())
            && model.memory.values.iter().all(|v| v.is_finite())
            && model.memory.norm_sq.iter().all(|v| v.is_finite())
            && model.memory.confidence.iter().all(|v| v.is_finite())
            && model.adapters.iter().all(|ad| {
                ad.consolidated_up.iter().all(|v| v.is_finite())
            })
    }

    pub fn train_corpus(
        raw: &str,
        options: &TrainingOptions,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        if options.epochs == 0
            || options.latent == 0
            || options.state == 0
            || options.key == 0
            || options.memory == 0
            || options.chunk == 0
            || options.accumulate == 0
        {
            return Err(
                "epochs, latent, state, key, memory, chunk, and accumulate must be positive"
                    .into(),
            );
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
                    "resumed_from={path} vocab={} d_latent={} prior_steps={}",
                    model.cfg.d_vocab, model.cfg.d_latent, model.step_counter
                );
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
                    d_state: options.state,
                    d_mem_key: options.key,
                    mem_capacity: options.memory,
                    chunk_len: options.chunk,
                    lr: options.lr,
                    ..Default::default()
                };
                checkpoint::validate_model_config(&cfg)?;
                let mut model = PSSALayerV2::new(cfg, options.seed);
                model.vocabulary = tokenizer.ordered_vocabulary()?;
                model.tokenizer_json = tokenizer.serialized_metadata();
                (model, tokenizer)
            }
        };
        // Keep an explicit resume override in the checkpoint's persisted
        // configuration so a later link does not silently revert to the old LR.
        model.cfg.lr = options.lr;
        // Attach the GPU dispatch path when a WebGPU compute adapter is
        // available; the stage math is cpu-twin verified and falls back to CPU.
        match Device::try_gpu() {
            Ok(gpu_device) => {
                let label = gpu_device
                    .gpu()
                    .map(|g| g.backend_label())
                    .unwrap_or_else(|| "cpu".to_string());
                model.device = gpu_device;
                println!("backend={label}");
            }
            Err(e) => println!("backend=cpu ({e})"),
        }
        let docs = Self::documents(raw, &tokenizer, options.max_tokens, options.skip_tokens)?;
        let chunk_len = if options.resume.is_some() {
            // A checkpoint owns its tape capacity.  Using a fresh CLI default
            // here could make the plan longer than that tape and panic in the
            // batched backward pass.
            model.cfg.chunk_len
        } else {
            options.chunk
        };
        let plan = crate::training::chunk_plan(&docs, chunk_len);
        let schedule = crate::training::Schedule::new(
            plan.len(),
            model.step_counter,
            model.lr_schedule_total_updates,
            options,
        )?;
        model.lr_schedule_total_updates = schedule.fixed_horizon;
        let total_updates = schedule.updates;
        let schedule_total = schedule.total;
        let first_lr = schedule.lr(1)?;
        let last_lr = schedule.lr(total_updates)?;
        println!(
            "model=pssa parameters={} vocab={}",
            model.parameter_count(), model.cfg.d_vocab
        );
        crate::training::report_stream(&docs, chunk_len, options.accumulate);
        ui::banner("train", "plastic state-space architecture");
        ui::field(
            "corpus",
            &format!("{} tokens", ui::thousands(docs.iter().map(Vec::len).sum())),
        );
        ui::field("vocabulary", &ui::thousands(model.cfg.d_vocab));
        ui::field(
            "width",
            &format!(
                "latent {} / state {}",
                model.cfg.d_latent, model.cfg.d_state
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

        let started = Instant::now();
        let mut update = 0;
        let mut tokens_seen = 0usize;
        let mut progress = ui::Progress::new("training", total_updates);
        for epoch in 0..options.epochs {
            model.reset_recurrent_state();
            let mut loss_sum = 0.0f64;
            let mut token_sum = 0usize;
            for group in plan.chunks(options.accumulate) {
                let total_tokens: usize = group.iter().map(|x| x.2).sum();
                if total_tokens == 0 {
                    continue;
                }
                model.zero_gradients();
                for &(doc_id, start, len) in group {
                    if start == 0 {
                        model.reset_recurrent_state();
                    }
                    let input = &docs[doc_id][start..start + len];
                    let target = &docs[doc_id][start + 1..start + 1 + len];
                    let loss = crate::gpu_batch::forward_train_chunk_batched(&mut model, input, target);
                    if !loss.is_finite() {
                        return Err("non-finite loss; training aborted without checkpoint".into());
                    }
                    crate::gpu_batch::backward_chunk_batched(&mut model, len, len as f32 / total_tokens as f32);
                    if loss > 3.5 {
                        let last = len - 1;
                        let q = &model.tape.q_poincare
                            [last * model.cfg.d_mem_key..(last + 1) * model.cfg.d_mem_key];
                        let v = &model.tape.z_final
                            [last * model.cfg.d_latent..(last + 1) * model.cfg.d_latent];
                        model
                            .memory
                            .insert_protected(q, v, loss, model.step_counter);
                    }
                    loss_sum += loss as f64 * len as f64;
                    token_sum += len;
                }
                update += 1;
                model.apply_adamw(schedule.lr(update)?);
                if !Self::finite(&model) {
                    return Err("non-finite parameters; training aborted without checkpoint".into());
                }
                tokens_seen += total_tokens;
                progress.update(update, total_tokens, loss_sum / token_sum.max(1) as f64);
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
        let wall = started.elapsed().as_secs_f64();
        println!(
            "training_seconds={:.3} optimizer_updates={update}",
            started.elapsed().as_secs_f32()
        );
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
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let (model, _) = Self::train_corpus(&raw, options)?;
        Self::save_model_v2(&model, out)
            .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
        println!("saved_checkpoint={out}");
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
            return Err(format!("{label} checkpoint lacks vocabulary provenance"));
        }
        let tokenizer = Tokenizer::from_vocabulary(&model.vocabulary)?;
        if let Some(source) = data {
            let raw = DatasetManager::try_load_dataset(Some(source))?;
            let external = Tokenizer::from_corpus(&raw, true)?;
            if external.ordered_vocabulary()? != model.vocabulary {
                return Err(format!(
                    "--data tokenizer/order does not match {label} checkpoint"
                ));
            }
        }
        Ok(tokenizer)
    }
    fn load_for_inference(
        model_path: &str,
        data: Option<&str>,
    ) -> Result<(PSSALayerV2, Tokenizer), String> {
        let loaded = checkpoint::load_checkpoint(model_path)
            .map_err(|e| format!("cannot load model '{model_path}': {e}"))?;
        let model = loaded.model;
        match loaded.format {
            CheckpointFormat::V7 => match &model.tokenizer_json {
                Some(json) => {
                    if data.is_some() {
                        return Err("--data is legacy word provenance only; V7 BPE checkpoints restore their embedded tokenizer and never retrain it".into());
                    }
                    let tokenizer = Tokenizer::from_serialized(json)?;
                    if tokenizer.ordered_vocabulary()? != model.vocabulary
                        || tokenizer.vocab_size != model.cfg.d_vocab
                    {
                        return Err("V7 tokenizer metadata/model vocabulary mismatch".into());
                    }
                    Ok((model, tokenizer))
                }
                None => {
                    let tokenizer = Self::word_tokenizer_for_model(&model, data, "V7 word")?;
                    Ok((model, tokenizer))
                }
            },
            CheckpointFormat::V6 => {
                let tokenizer = Self::word_tokenizer_for_model(&model, data, "V6")?;
                Ok((model, tokenizer))
            }
            CheckpointFormat::LegacyV5InferenceOnly => {
                let source = data.ok_or("legacy V5 checkpoint requires explicit --data; tokenizer provenance is unavailable")?;
                eprintln!(
                    "warning: legacy V5 checkpoint is inference-only; optimizer and tokenizer provenance are unavailable"
                );
                let raw = DatasetManager::try_load_dataset(Some(source))?;
                let tokenizer = Tokenizer::from_corpus(&raw, true)?;
                if tokenizer.vocab_size != model.cfg.d_vocab {
                    return Err(format!(
                        "legacy corpus vocabulary size {} does not match checkpoint {}",
                        tokenizer.vocab_size, model.cfg.d_vocab
                    ));
                }
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
        if max_new > MAX_GENERATION_TOKENS {
            return Err(format!(
                "max_new_tokens must be at most {MAX_GENERATION_TOKENS}"
            ));
        }
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
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
        let docs = Self::documents(raw, tokenizer, None, 0)?;
        let mut loss = 0.0f64;
        let mut tokens = 0usize;
        let mut correct = 0usize;
        let mut oov = 0usize;
        for doc in &docs {
            oov += doc.iter().filter(|&&x| x == 0).count();
            model.reset_recurrent_state();
            let mut start = 0;
            while start + 1 < doc.len() {
                let len = model.cfg.chunk_len.min(doc.len() - 1 - start);
                let l = model.forward_train_chunk(
                    &doc[start..start + len],
                    &doc[start + 1..start + 1 + len],
                );
                if !l.is_finite() {
                    return Err("non-finite evaluation loss".into());
                }
                loss += l as f64 * len as f64;
                for t in 0..len {
                    let logits =
                        &model.tape.logits[t * model.cfg.d_vocab..(t + 1) * model.cfg.d_vocab];
                    let guess = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .map(|x| x.0)
                        .unwrap_or(0);
                    correct += usize::from(guess == doc[start + 1 + t]);
                }
                tokens += len;
                start += len;
            }
        }
        Ok((loss / tokens.max(1) as f64, tokens, correct, oov))
    }
    fn run_evaluate(model_path: &str, data: &str) -> Result<(), String> {
        // V5 checkpoints contain no tokenizer provenance, so the evaluation
        // corpus also has to seed their legacy word tokenizer.  Newer formats
        // restore their tokenizer independently of the held-out corpus.
        let format = checkpoint::load_checkpoint(model_path)
            .map_err(|e| format!("cannot inspect model '{model_path}': {e}"))?
            .format;
        let provenance = (format == CheckpointFormat::LegacyV5InferenceOnly).then_some(data);
        let (mut model, tokenizer) = Self::load_for_inference(model_path, provenance)?;
        let raw = DatasetManager::try_load_dataset(Some(data))?;
        let (ce, tokens, correct, oov) = Self::evaluate_corpus(&mut model, &tokenizer, &raw)?;
        let encoded = docs_token_count(&raw, &tokenizer)?;
        let ppl = ce.exp();
        let (ppl_json, overflow) = if ppl.is_finite() {
            (format!("{ppl:.8}"), false)
        } else {
            ("null".into(), true)
        };
        println!(
            "{{\"cross_entropy\":{ce:.8},\"perplexity\":{ppl_json},\"perplexity_overflow\":{overflow},\"oov_rate\":{:.8},\"token_count\":{},\"next_token_accuracy\":{:.8}}}",
            oov as f64 / encoded.max(1) as f64,
            tokens,
            correct as f64 / tokens.max(1) as f64
        );
        Ok(())
    }
    fn run_chat(model_path: &str, data: Option<&str>, temp: f32) -> Result<(), String> {
        let (mut model, tokenizer) = Self::load_for_inference(model_path, data)?;
        println!("interactive: /exit");
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
            if p.is_empty() {
                continue;
            }
            let cfg = InferenceConfig {
                temperature: temp,
                ..Default::default()
            };
            println!(
                "{}",
                PSSAInferenceEngine::try_new(&mut model, &tokenizer)?.try_generate_chat_turn(
                    p,
                    &cfg,
                    |_| {}
                )?
            );
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
            warmup_steps: 4,
            schedule_total_updates: None,
            seed: 7,
            max_tokens: None,
            skip_tokens: 0,
            tokenizer: TokenizerKind::Word,
            vocab_size: 2048,
            resume: None,
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
    /// Everything the project can do, on one screen, with the state of the
    /// working directory next to it. This is what `oxide` alone prints.
    pub fn print_home() {
        ui::clear_screen();
        ui::logo();
        println!(
            "   {}  {}",
            ui::dim("plastic state-space architecture"),
            ui::dim("v0.4.0")
        );
        println!();

        ui::panel_top("commands");
        for (name, blurb) in [
            ("train", "fit a checkpoint on a text corpus"),
            ("generate", "continue a prompt with a trained checkpoint"),
            ("chat", "interactive prompt loop against a checkpoint"),
            ("evaluate", "cross entropy, perplexity and accuracy as JSON"),
            ("status", "checkpoints and corpora in this directory"),
            ("download", "pull a Hugging Face dataset to a local file"),
            ("benchmark", "end-to-end smoke test on the built-in corpus"),
            ("tui", "live dashboard for a piped training run (train ... | oxide tui)"),
            (
                "gpu-probe",
                "check whether a WebGPU compute device is usable",
            ),
        ] {
            ui::panel_row(&format!(
                "{}{}",
                ui::cyan(&format!("{name:<12}")),
                ui::dim(blurb)
            ));
        }
        ui::panel_bottom();
        println!();

        println!("  train-transformer: same-size decoder-only baseline (help train-transformer)");
        Self::workspace_panel();
        println!();
        println!(
            "  {} {}",
            ui::dim("try"),
            ui::bold("oxide train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1")
        );
        println!("  {}", ui::dim("oxide help for every flag"));
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
                if name.ends_with(".pssa") {
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
            ui::panel_field("checkpoints", &ui::dim("none yet, run oxide train"));
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

    /// `oxide status`: the workspace panel on its own, plus what each
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
                .filter(|s| s.ends_with(".pssa"))
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
                match checkpoint::load_checkpoint(&path) {
                    Ok(loaded) => {
                        let m = loaded.model;
                        ui::panel_row(&ui::bold(path.trim_start_matches("./")));
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
                            &format!("{} optimizer steps", ui::thousands(m.step_counter as usize)),
                        );
                    }
                    Err(e) => {
                        ui::panel_row(&ui::bold(path.trim_start_matches("./")));
                        ui::panel_field("  unreadable", &ui::dim(&e.to_string()));
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
        let bin = "oxide_ai_pssa";
        println!();
        println!(
            "  {}  {}",
            ui::bold(&ui::cyan(bin)),
            ui::dim("plastic state-space architecture, v0.4.0")
        );
        println!("  {}", ui::dim(&"\u{2500}".repeat(62)));
        println!();
        println!("  {}", ui::bold("USAGE"));
        println!("    {bin} <command> [options]");
        println!();
        println!("  {}", ui::bold("COMMANDS"));
        for (name, blurb) in [
            ("train", "fit a checkpoint on a text corpus"),
            ("generate", "continue a prompt with a trained checkpoint"),
            ("evaluate", "cross entropy, perplexity and accuracy as JSON"),
            ("chat", "interactive prompt loop against a checkpoint"),
            ("download", "pull a Hugging Face dataset to a local file"),
            ("benchmark", "end-to-end smoke test on the built-in corpus"),
            ("tui", "live dashboard for a piped training run (train ... | oxide tui)"),
            (
                "gpu-probe",
                "check whether a WebGPU compute device is usable",
            ),
            ("help", "show this message"),
        ] {
            println!("    {:<12}{}", ui::cyan(name), ui::dim(blurb));
        }
        println!();
        println!("    train-transformer  same-size decoder-only baseline (help train-transformer)");
        println!("    generate-transformer / evaluate-transformer  sample / score the baseline");
        println!("  {}", ui::bold("TRAIN"));
        println!("    {bin} train [source] [-d|--data source] [-o|--out path]");
        println!(
            "    {:<30}{}",
            "  --tokenizer bpe|word",
            ui::dim("default bpe")
        );
        println!(
            "    {:<30}{}",
            "  --vocab-size n",
            ui::dim("byte-level BPE ceiling, default 2048")
        );
        println!(
            "    {:<30}{}",
            "  -e|--epochs n",
            ui::dim("passes over the selected slice")
        );
        println!(
            "    {:<30}{}",
            "  --latent n --state n",
            ui::dim("model width and recurrent state size")
        );
        println!(
            "    {:<30}{}",
            "  --key n --memory n",
            ui::dim("episodic key width and bank capacity")
        );
        println!(
            "    {:<30}{}",
            "  --chunk n --accumulate n",
            ui::dim("sequence chunk and gradient accumulation")
        );
        println!(
            "    {:<30}{}",
            "  --lr f --warmup-steps n --total-updates n",
            ui::dim("optimizer schedule; fixed whole-run horizon for fresh chains")
        );
        println!(
            "    {:<30}{}",
            "  --seed n",
            ui::dim("deterministic initialisation")
        );
        println!(
            "    {:<30}{}",
            "  --max-tokens n",
            ui::dim("global cap on training tokens")
        );
        println!(
            "    {:<30}{}",
            "  --skip-tokens n",
            ui::dim("drop this many tokens from the front first")
        );
        println!(
            "    {:<30}{}",
            "  --resume path",
            ui::dim("continue from an existing checkpoint")
        );
        println!();
        println!("  {}", ui::bold("GENERATE"));
        println!(
            "    {bin} generate <prompt> [-m|--model path] [-t|--temp|--temperature f] [--max-new-tokens n]"
        );
        println!();
        println!("  {}", ui::bold("EXAMPLES"));
        println!(
            "    {}",
            ui::dim("# train a fresh checkpoint on the first 200k tokens")
        );
        println!("    {bin} train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1");
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
    fn print_command_help(command: &str) -> Result<(), String> {
        let bin = "oxide_ai_pssa";
        match command {
            "train-transformer" => {
                println!("Usage: {bin} train-transformer [SOURCE] [OPTIONS]");
                println!("Decoder-only baseline: 1 layer, width 256, 4 heads, FFN 448; CPU only.");
                println!("Same data/tokenizer/window/update schedule as train; context resets each chunk.");
                println!("  -d, --data SOURCE   -o, --out PATH (default: data/model.trfm)");
                println!("  -e, --epochs N (4)  --chunk N (64)  --accumulate N (8)");
                println!("  --tokenizer bpe|word (bpe)  --vocab-size N (2048 ceiling)");
                println!("  --tokenizer-from PSSA_CHECKPOINT  import EXACT tokenizer for comparisons");
                println!("  --max-tokens N  --skip-tokens N (wraps at EOF)");
                println!("  --lr F (0.001)  --warmup-steps N (0)  --total-updates N  --seed N (42)");
                println!("  --resume PATH  restore transformer weights, moments, tokenizer and horizon");
                println!("Omit --tokenizer-from on resume; identical chunk/accumulation flags give identical updates.");
            }
            "train" => {
                println!("Usage: {bin} train [SOURCE] [OPTIONS]");
                println!();
                println!("Fit a checkpoint on a text corpus. SOURCE may be a local file, directory, URL, hf:REPO, or science.");
                println!();
                println!("Options:");
                println!("  -d, --data <SOURCE>           dataset source (also accepted as SOURCE)");
                println!("  -o, --out <PATH>              checkpoint output (default: data/model.pssa)");
                println!("  -e, --epochs <N>              positive number of passes (default: 4)");
                println!("      --tokenizer <bpe|word>    tokenizer family (default: bpe)");
                println!("      --vocab-size <N>          BPE vocabulary ceiling (default: 2048)");
                println!("      --latent <N>              latent width (default: 256)");
                println!("      --state <N>               recurrent state width (default: 16)");
                println!("      --key <N>                 memory key width (default: 32)");
                println!("      --memory <N>              memory capacity (default: 512)");
                println!("      --chunk <N>               training chunk length (default: 64)");
                println!("      --accumulate <N>          chunks per optimizer update (default: 8)");
                println!("      --lr <F>                  base learning rate (default: 0.001)");
                println!("      --warmup-steps <N>        linear warm-up updates (default: 0)");
                println!("      --total-updates <N>       fixed whole-run schedule horizon (fresh run)");
                println!("      --seed <N>                initialization seed (default: 42)");
                println!("      --max-tokens <N>          global token cap");
                println!("      --skip-tokens <N>         offset into the corpus; wraps at EOF");
                println!("      --resume <PATH>            continue optimizer/model state from a checkpoint");
                println!();
                println!("Examples:");
                println!("  {bin} train data/downloaded.txt -o data/model.pssa --max-tokens 200000 -e 1");
                println!("  {bin} train data/downloaded.txt -o data/ck02.pssa --resume data/ck01.pssa --skip-tokens 200000 --max-tokens 200000 -e 1");
            }
            "generate" | "generate-transformer" => {
                let model = if command == "generate" { "data/model.pssa" } else { "data/model.trfm" };
                println!("Usage: {bin} {command} [PROMPT] [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -p, --prompt <TEXT>           prompt (also accepted as PROMPT)");
                println!("  -m, --model <PATH>            checkpoint (default: {model})");
                println!("  -t, --temp, --temperature <F> sampling temperature; 0 is greedy (default: 0.70)");
                println!("      --max-new-tokens <N>      generation cap (default: 64, max: {MAX_GENERATION_TOKENS})");
                println!();
                println!("Examples:");
                println!("  {bin} {command} -m {model} -p \"The sun is\"");
                println!("  {bin} {command} \"quantum mechanics\" --temperature 0 --max-new-tokens 32");
            }
            "chat" | "repl" => {
                println!("Usage: {bin} chat [DATA] [OPTIONS]");
                println!();
                println!("Start an interactive prompt loop. DATA is an optional legacy word-tokenizer provenance source.");
                println!("Options:");
                println!("  -m, --model <PATH>            checkpoint (default: data/model.pssa)");
                println!("  -d, --data <SOURCE>           legacy tokenizer provenance source");
                println!("  -t, --temp, --temperature <F> sampling temperature (default: 0.70)");
                println!();
                println!("Example: {bin} chat -m data/model.pssa --temperature 0.7");
            }
            "evaluate" | "evaluate-transformer" => {
                println!("Usage: {bin} {command} [DATA] [-m|--model PATH] [-d|--data SOURCE]");
                println!();
                println!("Evaluate a checkpoint and print one JSON metrics object.");
                let model = if command == "evaluate" { "data/model.pssa" } else { "data/model.trfm" };
                println!("Example: {bin} {command} data/heldout.txt --model {model}");
            }
            "download" => {
                println!("Usage: {bin} download REPOSITORY [-o|--out PATH]");
                println!();
                println!("Download the train split from Hugging Face as plain text.");
                println!("Example: {bin} download wikimedia/wikipedia --out data/downloaded.txt");
            }
            "status" | "benchmark" | "gpu-probe" => {
                println!("Usage: {bin} {command}");
                println!();
                println!("This command takes no options.");
            }
            "tui" => {
                println!("Usage: {bin} tui");
                println!();
                println!("Render a dashboard for a piped training run: train ... | {bin} tui");
            }
            _ => return Err(format!("unknown command '{command}'; run {bin} help")),
        }
        Ok(())
    }

    pub fn parse_and_execute(args: Vec<String>) -> Result<(), String> {
        if args.len() < 2 {
            Self::print_home();
            return Ok(());
        }
        let command = args[1].as_str();
        if command == "help" {
            if args.len() == 2 {
                Self::print_help();
                return Ok(());
            }
            if args.len() == 3 {
                return Self::print_command_help(&args[2]);
            }
            return Err("help accepts at most one command name".into());
        }
        if matches!(command, "--help" | "-h") {
            Self::print_help();
            return Ok(());
        }
        if args[2..].iter().any(|arg| matches!(arg.as_str(), "--help" | "-h")) {
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
                    "--data", "-d", "--out", "-o", "--epochs", "-e",
                    "--latent", "--state", "--key", "--memory", "--chunk", "--lr",
                    "--accumulate", "--warmup-steps", "--total-updates", "--seed",
                    "--max-tokens", "--tokenizer", "--vocab-size", "--resume", "--skip-tokens",
                ];
                if baseline {
                    allowed.retain(|x| !["--latent", "--state", "--key", "--memory"].contains(x));
                    allowed.push("--tokenizer-from");
                }
                let p = Parsed::parse(&args[2..], &allowed)?;
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
                    return Err("train source was specified both positionally and with --data".into());
                }
                let data = p
                    .string("--data", "-d")
                    .map(str::to_string)
                    .or_else(|| p.positional.first().cloned())
                    .unwrap_or_else(Self::default_data);
                let default_out = if baseline { "data/model.trfm" } else { "data/model.pssa" };
                let out = p.string("--out", "-o").unwrap_or(default_out);
                if baseline {
                    let mut opts = Self::common_options(&p)?;
                    if let Some(path) = &opts.resume {
                        if p.string("--tokenizer-from", "").is_some() {
                            return Err("--resume restores its own tokenizer; omit --tokenizer-from".into());
                        }
                        let model = crate::transformer_checkpoint::load_checkpoint(path)
                            .map_err(|e| format!("cannot inspect resume checkpoint '{path}': {e}"))?;
                        if p.flags.contains_key("--chunk") && opts.chunk != model.cfg.chunk_len {
                            return Err(format!(
                                "--chunk={} does not match resume checkpoint value {}",
                                opts.chunk, model.cfg.chunk_len
                            ));
                        }
                        if !p.flags.contains_key("--lr") {
                            opts.lr = model.cfg.lr;
                        }
                        if let (Some(requested), Some(stored)) = (
                            opts.schedule_total_updates, model.lr_schedule_total_updates,
                        ) && requested != stored {
                            return Err(format!("--total-updates={requested} does not match resume checkpoint horizon {stored}"));
                        }
                    }
                    crate::transformer_training::run_training(
                        &data, &opts, out, p.string("--tokenizer-from", ""),
                    )
                } else {
                    Self::run_training(&data, &Self::options(&p)?, out)
                }
            }
            "generate" | "generate-transformer" => {
                let p = Parsed::parse(
                    &args[2..],
                    &[
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
                    ],
                )?;
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
                    return Err("generate prompt was specified both positionally and with --prompt".into());
                }
                let prompt = p
                    .string("--prompt", "-p")
                    .or_else(|| p.positional.first().map(String::as_str))
                    .ok_or("generate requires a prompt")?;
                let temp = p.f32_first(&["--temp", "--temperature", "-t"], "--temperature", 0.70)?;
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
                        return Err("transformer checkpoints embed their tokenizer; omit --data".into());
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
                    Self::run_generate(
                        prompt, p.string("--model", "-m").unwrap_or("data/model.pssa"),
                        p.string("--data", "-d"), temp, max,
                    )?
                };
                println!("{text}");
                Ok(())
            }
            "evaluate" | "evaluate-transformer" => {
                let p = Parsed::parse(&args[2..], &["--model", "-m", "--data", "-d"])?;
                p.reject_duplicate_aliases(&["--model", "-m"], "--model")?;
                p.reject_duplicate_aliases(&["--data", "-d"], "--data")?;
                if p.positional.len() > 1 {
                    return Err("evaluate accepts one positional data source".into());
                }
                if p.positional.len() == 1 && p.string("--data", "-d").is_some() {
                    return Err("evaluate data was specified both positionally and with --data".into());
                }
                let data = p
                    .string("--data", "-d")
                    .or_else(|| p.positional.first().map(String::as_str))
                    .ok_or("evaluate requires a data source (use DATA or --data DATA)")?;
                if command == "evaluate-transformer" {
                    crate::transformer_inference::evaluate(
                        p.string("--model", "-m").unwrap_or("data/model.trfm"), data,
                    )
                } else {
                    Self::run_evaluate(
                        p.string("--model", "-m").unwrap_or("data/model.pssa"), data,
                    )
                }
            }
            "chat" | "repl" => {
                let p = Parsed::parse(
                    &args[2..],
                    &["--model", "-m", "--data", "-d", "--temp", "--temperature", "-t"],
                )?;
                p.reject_duplicate_aliases(&["--model", "-m"], "--model")?;
                p.reject_duplicate_aliases(&["--data", "-d"], "--data")?;
                p.reject_duplicate_aliases(
                    &["--temp", "--temperature", "-t"],
                    "--temperature",
                )?;
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
                )
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
            "benchmark" => {
                if args.len() != 2 {
                    return Err("benchmark takes no options".into());
                }
                Self::run_benchmark()
            }
            "tui" => crate::tui::run(&args[2..]),
            _ => Err(format!("unknown command '{}'; run oxide help", args[1])),
        }
    }
}
fn docs_token_count(raw: &str, tokenizer: &Tokenizer) -> Result<usize, String> {
    raw.lines().try_fold(0usize, |n, line| {
        let ids = tokenizer.try_encode(line, true)?;
        n.checked_add(ids.len())
            .ok_or_else(|| "dataset token count overflow".to_string())
    })
}

/// Bring up the WebGPU compute device, run the embedded tiled GEMM kernel on it,
/// and check the result against the CPU reference implementation.
pub fn run_gpu_probe() -> Result<(), String> {
    println!("=== oxide gpu-probe ===");
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
    let y_gpu = ctx.dispatch_gemm(&x, &w, m, n, k, batch);
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    let t_cpu = Instant::now();
    let y_cpu = gemm_cpu_reference(&x, &w, m, n, k, batch);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let mut max_abs = 0.0f32;
    for (a, b) in y_gpu.iter().zip(y_cpu.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    println!("shape: batch={} M={} N={} K={}", batch, m, n, k);
    println!("gpu:   {:.3} ms", gpu_ms);
    println!("cpu:   {:.3} ms", cpu_ms);
    println!("max_abs_diff: {:.3e}", max_abs);
    if max_abs < 1e-3 {
        println!("result: PASS, GPU kernel matches CPU reference");
    } else {
        println!("result: FAIL, GPU kernel diverges from CPU reference");
        return Err(format!("GPU GEMM differs from CPU reference by {max_abs:.3e}"));
    }
    println!("note: layer math is still CPU-dispatched; this proves the device path only");
    Ok(())
}
