//! Baseline generation uses the same seeded sampler and decoding policy as PSSA.
use crate::cli::CLIHandler;
use crate::dataset::{DatasetManager, TokenizerKind};
use crate::inference::{InferenceConfig, PSSAInferenceEngine};
use crate::linalg::SimpleRng;
use crate::training::chunk_plan;
use crate::transformer_checkpoint;

pub fn generate(path: &str, prompt: &str, cfg: &InferenceConfig) -> Result<String, String> {
    PSSAInferenceEngine::validate(cfg)?;
    if cfg.max_new_tokens > 100_000 {
        return Err("max_new_tokens must be at most 100000".into());
    }
    let model = transformer_checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
    let tok = model.tokenizer()?;
    let mut ids = tok.try_encode(prompt, true)?;
    if ids.is_empty() {
        return Err("prompt is empty after tokenization".into());
    }
    if ids.iter().all(|&id| id == 0) {
        return Err("prompt contains no known vocabulary tokens".into());
    }
    let prompt_len = ids.len();
    let mut logits = vec![0.0; tok.vocab_size];
    let mut probs = vec![0.0; tok.vocab_size];
    let mut candidates = Vec::with_capacity(tok.vocab_size);
    let mut rng = SimpleRng::new(1337);
    let mut sentences = 0;
    for _ in 0..cfg.max_new_tokens {
        model.logits_for_context(&ids, &mut logits)?;
        let id = PSSAInferenceEngine::sample(
            &mut rng,
            cfg,
            &ids,
            &mut logits,
            &mut probs,
            &mut candidates,
        )?;
        ids.push(id);
        if tok.kind() == TokenizerKind::Word
            && matches!(tok.id_to_token[&id].as_str(), "." | "?" | "!")
        {
            sentences += 1;
            if sentences >= 2 {
                break;
            }
        }
    }
    if tok.kind() == TokenizerKind::Bpe {
        let bytes: Vec<u8> = ids[prompt_len..]
            .iter()
            .flat_map(|&id| tok.token_bytes(id).unwrap_or(&[]).iter().copied())
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        Ok(tok.decode(&ids[prompt_len..]))
    }
}

pub fn evaluate(path: &str, data: &str) -> Result<(), String> {
    let mut model = transformer_checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
    let tok = model.tokenizer()?;
    let raw = DatasetManager::try_load_dataset(Some(data))?;
    let docs = CLIHandler::documents(&raw, &tok, None, 0)?;
    let mut loss = 0.0f64;
    let mut tokens = 0;
    let mut correct = 0;
    let oov = docs.iter().flatten().filter(|&&id| id == 0).count();
    let encoded = raw.lines().try_fold(0usize, |n, line| {
        tok.try_encode(line, true).map(|ids| n + ids.len())
    })?;
    for (doc, start, len) in chunk_plan(&docs, model.cfg.chunk_len) {
        let targets = &docs[doc][start + 1..start + 1 + len];
        let ce = model.forward_train_chunk(&docs[doc][start..start + len], targets);
        if !ce.is_finite() {
            return Err("non-finite evaluation loss".into());
        }
        loss += ce as f64 * len as f64;
        tokens += len;
        for (logits, target) in model.training_logits().chunks(tok.vocab_size).zip(targets) {
            let guess = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|x| x.0)
                .unwrap_or(0);
            correct += usize::from(guess == *target);
        }
    }
    let ce = loss / tokens.max(1) as f64;
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
