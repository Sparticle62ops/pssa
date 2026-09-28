//! Baseline generation uses the same seeded sampler and decoding policy as PSSA.
use crate::dataset::{DatasetManager, TokenizerKind};
use crate::evaluation::{self, EvaluationSlice};
use crate::inference::{InferenceConfig, PSSAInferenceEngine};
use crate::linalg::SimpleRng;
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
    evaluate_slice(path, data, EvaluationSlice::default())
}

pub fn evaluate_slice(path: &str, data: &str, slice: EvaluationSlice) -> Result<(), String> {
    let mut model = transformer_checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
    let tok = model.tokenizer()?;
    let raw = DatasetManager::try_load_dataset(Some(data))?;
    println!(
        "{}",
        evaluation::evaluate_transformer(&mut model, &tok, &raw, slice)?.json()
    );
    Ok(())
}
