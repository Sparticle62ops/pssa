//! Replay an existing PSSA checkpoint chain's token exposure and optimizer schedule.
//! Checkpoints are evidence for tokenizer, shape, LR and update clock, not corpus
//! provenance. The caller must supply the original corpus and window/lane plan.
use crate::checkpoint;
use crate::cli::{CLIHandler, TrainingOptions};
use crate::dataset::DatasetManager;
use crate::evaluation::{self, EvaluationSlice};
use crate::training::{Schedule, sequence_plan};
use crate::transformer::{TransformerConfig, TransformerModel};
use crate::{transformer_checkpoint, transformer_training};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub struct ComparisonOptions {
    pub chain_dir: String,
    pub out_dir: String,
    pub links: usize,
    pub window: usize,
    pub batch_size: usize,
    pub accumulate: usize,
    pub link_plan: Option<String>,
    pub eval_skip_tokens: Option<usize>,
    pub eval_tokens: usize,
    pub loss_every: usize,
    pub seed: u64,
    pub legacy_warmup: usize,
}

#[derive(Clone, Debug)]
struct Window {
    skip: usize,
    size: usize,
    batch: usize,
    accumulate: usize,
}

fn windows(opts: &ComparisonOptions) -> Result<Vec<Window>, String> {
    if opts.links == 0 || opts.window == 0 || opts.eval_tokens == 0 || opts.loss_every == 0
        || opts.batch_size == 0 || opts.batch_size > 65_536 || opts.accumulate == 0
    {
        return Err("comparison counts must be positive and batch-size at most 65536".into());
    }
    if let Some(path) = &opts.link_plan {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read link plan '{path}': {e}"))?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid link plan JSON: {e}"))?;
        let rows = value.as_array().ok_or("link plan must be a JSON array")?;
        if rows.len() != opts.links {
            return Err(format!("link plan must contain exactly {} entries", opts.links));
        }
        rows.iter().map(|row| {
            let object = row.as_object().ok_or("each link plan entry must be an object")?;
            let keys = ["skip_tokens", "max_tokens", "batch_size", "accumulate"];
            if object.len() != keys.len() || object.keys().any(|k| !keys.contains(&k.as_str())) {
                return Err("each link plan entry requires exactly skip_tokens, max_tokens, batch_size, accumulate".into());
            }
            let number = |key: &str| -> Result<usize, String> {
                object[key].as_u64().and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| format!("link plan {key} must be an unsigned integer"))
            };
            let w = Window { skip: number("skip_tokens")?, size: number("max_tokens")?, batch: number("batch_size")?, accumulate: number("accumulate")? };
            if w.size == 0 || w.batch == 0 || w.batch > 65_536 || w.accumulate == 0 || w.accumulate > 1_000_000 {
                return Err("link plan max_tokens/accumulate/batch_size must be positive; batch_size <= 65536, accumulate <= 1000000".into());
            }
            Ok(w)
        }).collect()
    } else {
        (0..opts.links).map(|i| Ok(Window {
            skip: i.checked_mul(opts.window).ok_or("chain token offset overflow")?,
            size: opts.window, batch: opts.batch_size, accumulate: opts.accumulate,
        })).collect()
    }
}

/// Training may wrap, but none of its selected input tokens may intersect the
/// evaluation slice. Deliberately conservative: even an unscored boundary token
/// makes a slice unsuitable for calling it held out.
fn check_holdout(windows: &[Window], corpus: usize, slice: EvaluationSlice) -> Result<(), String> {
    let end = slice.skip_tokens.checked_add(slice.max_tokens.ok_or("comparison needs an evaluation token count")?)
        .ok_or("held-out token range overflow")?;
    if end > corpus || slice.skip_tokens >= end {
        return Err(format!("held-out slice {}..{end} exceeds corpus length {corpus}; choose --eval-skip-tokens/--eval-tokens within the original corpus", slice.skip_tokens));
    }
    for (i, w) in windows.iter().enumerate() {
        let start = w.skip % corpus;
        let first = w.size.min(corpus - start);
        let overlaps = |a: usize, b: usize| a < end && slice.skip_tokens < b;
        if w.size >= corpus || overlaps(start, start + first) || (w.size > first && overlaps(0, w.size - first)) {
            return Err(format!("held-out slice overlaps training link {}; choose an unseen slice (training wraps at EOF)", i + 1));
        }
    }
    Ok(())
}

fn checkpoint_path(dir: &str, link: usize) -> PathBuf {
    Path::new(dir).join(format!("ck{link:02}.pssa"))
}
fn path_string(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| "comparison paths must be valid UTF-8".into())
}
fn save_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, format!("{text}\n")).map_err(|e| format!("cannot write '{}': {e}", path.display()))
}
fn fnv(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3));
    format!("{hash:016x}")
}

struct Link {
    options: TrainingOptions,
    end_steps: usize,
    targets: usize,
    stored_horizon: Option<usize>,
    stored_warmup: Option<usize>,
}

pub fn run(data: &str, opts: &ComparisonOptions) -> Result<(), String> {
    let windows = windows(opts)?;
    let out = Path::new(&opts.out_dir);
    if out.exists() {
        return Err(format!("comparison output '{}' already exists; choose a new --out directory", out.display()));
    }
    let source_path = checkpoint_path(&opts.chain_dir, 1);
    let (first, tokenizer) = CLIHandler::load_for_inference(path_string(&source_path)?, None)?;
    let raw = DatasetManager::try_load_dataset(Some(data))?;
    let encoded = raw.lines().map(|line| tokenizer.try_encode(line, true)).collect::<Result<Vec<_>, _>>()?;
    let corpus_tokens = encoded.iter().try_fold(0usize, |n, doc| n.checked_add(doc.len()).ok_or("corpus token count overflow"))?;
    if corpus_tokens < 2 { return Err("comparison corpus has no token transitions".into()); }
    let budget = windows.iter().try_fold(0usize, |n, w| n.checked_add(w.size).ok_or("chain token budget overflow"))?;
    let slice = EvaluationSlice { skip_tokens: opts.eval_skip_tokens.unwrap_or(budget), max_tokens: Some(opts.eval_tokens) };
    check_holdout(&windows, corpus_tokens, slice)?;
    // Also reject transition-free slices before creating any output or training.
    evaluation::documents(&raw, &tokenizer, slice)?;
    let mut links = Vec::new();
    let mut audit = Vec::new();
    let mut prior_steps = 0;
    let mut prior_horizon = None;
    let mut prior_warmup = None;
    let mut targets_seen = 0usize;
    for (index, w) in windows.iter().enumerate() {
        let path = checkpoint_path(&opts.chain_dir, index + 1);
        let source = checkpoint::load_checkpoint(&path)
            .map_err(|e| format!("cannot inspect chain link '{}': {e}; supply every ck01..ck{:02}.pssa", path.display(), opts.links))?.model;
        if source.vocabulary != first.vocabulary || source.tokenizer_json != first.tokenizer_json
            || source.cfg.d_vocab != first.cfg.d_vocab || source.cfg.chunk_len != first.cfg.chunk_len
            || source.cfg.d_latent != first.cfg.d_latent || source.cfg.d_state != first.cfg.d_state
            || source.cfg.d_mem_key != first.cfg.d_mem_key || source.cfg.mem_capacity != first.cfg.mem_capacity
        {
            return Err(format!("chain link {} changes tokenizer or shape; provide one consistent chain", index + 1));
        }
        let docs = crate::training::documents_from_encoded(&encoded, Some(w.size), w.skip)?;
        let plan = sequence_plan(&docs, first.cfg.chunk_len, w.batch)?;
        let targets = docs.iter().map(|d| d.len() - 1).sum::<usize>();
        let options = TrainingOptions {
            epochs: 1, chunk: first.cfg.chunk_len, lr: source.cfg.lr,
            batch_size: w.batch, accumulate: w.accumulate,
            warmup_steps: source.lr_schedule_warmup_steps.unwrap_or(opts.legacy_warmup),
            schedule_total_updates: source.lr_schedule_total_updates,
            max_tokens: Some(w.size), skip_tokens: w.skip, seed: opts.seed,
            loss_every: opts.loss_every,
            ..Default::default()
        };
        let schedule = Schedule::new_with_warmup(plan.len(), prior_steps, prior_horizon, prior_warmup, &options)?;
        let expected = prior_steps.checked_add(schedule.updates).ok_or("chain update overflow")?;
        if source.step_counter != expected {
            return Err(format!("chain link {} has {} updates, but supplied corpus/window/batch/accumulate plan predicts {expected}; use the original corpus/settings or --link-plan for mixed historical links", index + 1, source.step_counter));
        }
        if source.lr_schedule_total_updates != schedule.fixed_horizon
            || source.lr_schedule_warmup_steps.is_some_and(|n| n != schedule.warmup)
        {
            return Err(format!("chain link {} schedule metadata is inconsistent with its predecessor", index + 1));
        }
        targets_seen = targets_seen.checked_add(targets).ok_or("chain target count overflow")?;
        audit.push(json!({
            "link": index + 1, "pssa_checkpoint": path, "skip_tokens": w.skip, "max_tokens": w.size,
            "batch_size": w.batch, "accumulate": w.accumulate, "targets": targets,
            "tokens_seen": targets_seen, "updates": expected, "base_lr": source.cfg.lr,
            "horizon": schedule.total, "fixed_horizon": source.lr_schedule_total_updates,
            "warmup_steps": schedule.warmup, "first_lr": schedule.lr(1)?, "last_lr": schedule.lr(schedule.updates)?,
        }));
        println!("comparison_preflight link={} targets={targets} tokens_seen={targets_seen} updates={expected}", index + 1);
        prior_steps = expected;
        prior_horizon = source.lr_schedule_total_updates;
        prior_warmup = source.lr_schedule_warmup_steps;
        links.push(Link { options, end_steps: expected, targets, stored_horizon: prior_horizon, stored_warmup: prior_warmup });
    }
    let mut baseline = TransformerModel::new(TransformerConfig {
        d_vocab: tokenizer.vocab_size, chunk_len: first.cfg.chunk_len, lr: links[0].options.lr,
        ..Default::default()
    }, opts.seed)?;
    baseline.vocabulary = tokenizer.ordered_vocabulary()?;
    baseline.tokenizer_json = tokenizer.serialized_metadata();
    let pssa_parameters = first.parameter_count();
    let transformer_parameters = baseline.parameter_count();
    drop(first);
    std::fs::create_dir(out).map_err(|e| format!("cannot create comparison output '{}': {e}; use a new directory under an existing writable parent", out.display()))?;
    let manifest = json!({
        "schema": 1, "data": data, "corpus_fnv1a64": fnv(raw.as_bytes()), "corpus_encoded_tokens": corpus_tokens,
        "tokenizer_source": source_path, "seed": opts.seed, "chunk": baseline.cfg.chunk_len,
        "encoded_training_budget": budget, "tokens_seen": targets_seen, "updates": prior_steps,
        "eval_skip_tokens": slice.skip_tokens, "eval_tokens": opts.eval_tokens,
        "pssa_parameters": pssa_parameters, "transformer_parameters": transformer_parameters,
        "links": audit,
        "provenance_note": "Caller supplies original corpus and lane plan; old PSSA checkpoints do not store corpus fingerprints or tokens_seen. FNV detects accidental drift, not cryptographic identity."
    });
    save_json(&out.join("manifest.json"), &manifest)?;
    let curve = out.join("transformer.csv");
    let mut cumulative_targets = 0;
    for (i, mut link) in links.into_iter().enumerate() {
        link.options.loss_csv = Some(path_string(&curve)?.to_owned());
        link.options.tokens_seen = Some(cumulative_targets);
        let docs = crate::training::documents_from_encoded(&encoded, link.options.max_tokens, link.options.skip_tokens)?;
        println!("comparison_train link={}", i + 1);
        transformer_training::train_documents(&mut baseline, &docs, &link.options)?;
        if baseline.step_counter != link.end_steps { return Err("internal comparison update mismatch".into()); }
        // Preserve the source's legacy metadata absence too: an old checkpoint
        // without warmup must not accidentally gain a reinterpreted schedule.
        baseline.lr_schedule_total_updates = link.stored_horizon;
        baseline.lr_schedule_warmup_steps = link.stored_warmup;
        cumulative_targets += link.targets;
        let checkpoint = out.join(format!("ck{:02}.trfm", i + 1));
        transformer_checkpoint::save_model(&baseline, &checkpoint).map_err(|e| e.to_string())?;
        println!("saved_checkpoint={}", checkpoint.display());
    }
    let final_source = checkpoint_path(&opts.chain_dir, opts.links);
    let (mut pssa, _) = CLIHandler::load_for_inference(path_string(&final_source)?, None)?;
    let pssa_metrics = evaluation::evaluate_pssa(&mut pssa, &tokenizer, &raw, slice)?;
    let transformer_metrics = evaluation::evaluate_transformer(&mut baseline, &tokenizer, &raw, slice)?;
    let parse_metrics = |m: &evaluation::Metrics| serde_json::from_str::<Value>(&m.json()).map_err(|e| e.to_string());
    let results = json!({
        "tokens_seen": cumulative_targets, "updates": baseline.step_counter,
        "eval_skip_tokens": slice.skip_tokens, "eval_tokens": opts.eval_tokens,
        "pssa": parse_metrics(&pssa_metrics)?, "transformer": parse_metrics(&transformer_metrics)?,
        "pssa_parameters": pssa_parameters, "transformer_parameters": transformer_parameters,
    });
    save_json(&out.join("results.json"), &results)?;
    println!("comparison_model=pssa {}", pssa_metrics.json());
    println!("comparison_model=transformer {}", transformer_metrics.json());
    println!("comparison_results={}", out.join("results.json").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heldout_checks_wrap_and_boundary_overlap() {
        let w = |skip, size| Window { skip, size, batch: 8, accumulate: 1 };
        let slice = EvaluationSlice { skip_tokens: 10, max_tokens: Some(4) };
        assert!(check_holdout(&[w(0, 10)], 20, slice).is_ok());
        assert!(check_holdout(&[w(0, 11)], 20, slice).is_err());
        assert!(check_holdout(&[w(15, 10)], 20, slice).is_ok());
        assert!(check_holdout(&[w(15, 16)], 20, slice).is_err());
        assert!(check_holdout(&[w(0, 20)], 20, slice).is_err());
        assert!(check_holdout(&[w(20, 10)], 20, slice).is_ok());
        assert!(check_holdout(&[w(0, 2)], 12, slice).is_err());
    }
}
