//! Minimal single-PSSA card ported from round one's frozen eval-card.
//! The external paired runner owns training, seeds, aggregation and failures.
use crate::checkpoint::{self, CheckpointFormat};
use crate::dataset::Tokenizer;
use crate::evaluation::{self, EvaluationSlice};
use crate::pssa::PSSALayerV2;
use serde_json::{Value, json};
use std::time::Instant;

fn fingerprint(bytes: &[u8]) -> String {
    format!("{:016x}", checkpoint::fnv1a64(bytes))
}

fn tokenizer_identity(tokenizer: &Tokenizer) -> Result<String, String> {
    let metadata = tokenizer.serialized_metadata()
        .map(|text| serde_json::from_str::<Value>(&text).map_err(|e| e.to_string()))
        .transpose()?;
    serde_json::to_string(&json!({
        "kind": format!("{:?}", tokenizer.kind()),
        "vocabulary": tokenizer.ordered_vocabulary()?,
        "metadata": metadata,
    })).map_err(|e| e.to_string())
}

/// Score a configured CPU model without training or memory writes. Prototype
/// callers must enable runtime-only flags BEFORE calling this function.
/// Timing excludes tokenization/I/O; recurrent carries are restored on errors.
pub fn score_model(
    model: &mut PSSALayerV2,
    tokenizer: &Tokenizer,
    raw: &str,
    slice: EvaluationSlice,
) -> Result<Value, String> {
    if model.device.is_gpu() {
        return Err("eval-card requires a CPU model".into());
    }
    if slice.max_tokens.is_none() {
        return Err("eval-card requires an explicit token window".into());
    }
    if tokenizer.ordered_vocabulary()? != model.vocabulary {
        return Err("eval-card tokenizer/vocabulary mismatch".into());
    }
    let docs = evaluation::documents(raw, tokenizer, slice)?;
    let identity = tokenizer_identity(tokenizer)?;
    let mut stream = Vec::new();
    for n in docs.iter().flat_map(|doc| std::iter::once(doc.len()).chain(doc.iter().copied())) {
        stream.extend_from_slice(&(n as u64).to_le_bytes());
    }
    let metrics = evaluation::evaluate_pssa_documents(model, tokenizer, &docs)?;
    let mut seconds = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let observed = evaluation::evaluate_pssa_documents(model, tokenizer, &docs)?;
        let elapsed = start.elapsed().as_secs_f64();
        if observed != metrics {
            return Err("frozen evaluation changed between passes".into());
        }
        seconds.push(elapsed);
    }
    seconds.sort_by(f64::total_cmp);
    let median = seconds[1];
    Ok(json!({
        "schema": 1, "kind": "pssa_eval_card",
        "identity": {
            "corpus_fnv1a64": fingerprint(raw.as_bytes()),
            "tokenizer_fnv1a64": fingerprint(identity.as_bytes()),
            "token_stream_fnv1a64": fingerprint(&stream),
            "skip_tokens": slice.skip_tokens, "max_tokens": slice.max_tokens,
            "targets": metrics.tokens, "encoded_tokens": metrics.encoded_tokens,
            "documents": docs.len(), "oov_tokens": metrics.oov,
        },
        "metrics": {
            "cross_entropy": metrics.loss, "perplexity": metrics.loss.exp(),
            "next_token_accuracy": metrics.correct as f64 / metrics.tokens as f64,
            "targets_per_second": if median > 0.0 { Some(metrics.tokens as f64 / median) } else { None },
        },
        "timing": { "seconds_sorted": seconds, "median_seconds": median },
        "model": {
            "parameters": model.parameter_count(), "optimizer_updates": model.step_counter,
            "width": model.cfg.d_latent, "state": model.cfg.d_state, "depth": model.depth(),
            "memory_slots_occupied": std::iter::once(&model.block).chain(&model.extra_blocks)
                .map(|b| b.memory.count).collect::<Vec<_>>(),
        },
        "protocol": {
            "backend": "cpu", "loops": model.loops(), "chunk": model.cfg.chunk_len,
            "rayon_threads": rayon::current_num_threads(), "architecture": std::env::consts::ARCH,
            "warmup_passes": 1, "measured_passes": 3,
            "scope": "prepared-document teacher-forced scoring; not generation or training throughput",
            "context": "carry across chunks within a line; reset between lines; memory read-only",
            "heldout_verified": false,
        },
    }))
}

/// Default (no experimental runtime flags) checkpoint scoring entry point.
pub fn checkpoint_card(
    data: &str, path: &str, slice: EvaluationSlice, loops: usize,
) -> Result<Value, String> {
    let loaded = checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
    if loaded.format == CheckpointFormat::LegacyV5InferenceOnly {
        return Err("eval-card requires an embedded tokenizer (not legacy V5)".into());
    }
    let mut model = loaded.model;
    model.set_loops(loops)?;
    let tokenizer = match &model.tokenizer_json {
        Some(text) => Tokenizer::from_serialized(text)?,
        None => Tokenizer::from_vocabulary(&model.vocabulary)?,
    };
    // Local immutable files only: no implicit downloads or source mixtures.
    let raw = std::fs::read_to_string(data).map_err(|e| e.to_string())?;
    let mut card = score_model(&mut model, &tokenizer, &raw, slice)?;
    card["checkpoint"] = json!({
        "path": path, "fnv1a64": fingerprint(&std::fs::read(path).map_err(|e| e.to_string())?),
    });
    Ok(card)
}

#[cfg(test)]
mod tests;
