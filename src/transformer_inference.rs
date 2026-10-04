//! Baseline generation uses the same seeded sampler and decoding policy as PSSA.
use crate::dataset::{DatasetManager, TokenizerKind};
use crate::evaluation::{self, EvaluationSlice};
use crate::inference::{InferenceConfig, PSSAInferenceEngine, unknown_prompt_error};
use crate::linalg::SimpleRng;
use crate::transformer_checkpoint;

pub fn generate(path: &str, prompt: &str, cfg: &InferenceConfig) -> Result<String, String> {
    generate_impl(path, prompt, cfg, None, &|| false)
}

/// TUI-only streaming/cancellation boundary; the CLI retains its original
/// decoding and seeded sampling behavior without per-token decoding work.
pub(crate) fn generate_controlled(
    path: &str,
    prompt: &str,
    cfg: &InferenceConfig,
    callback: &mut dyn FnMut(&str, usize),
    cancelled: &dyn Fn() -> bool,
) -> Result<String, String> {
    generate_impl(path, prompt, cfg, Some(callback), cancelled)
}

fn generate_impl(
    path: &str,
    prompt: &str,
    cfg: &InferenceConfig,
    mut callback: Option<&mut dyn FnMut(&str, usize)>,
    cancelled: &dyn Fn() -> bool,
) -> Result<String, String> {
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
        return Err(unknown_prompt_error(&tok, prompt));
    }
    let prompt_len = ids.len();
    let mut logits = vec![0.0; tok.vocab_size];
    let mut probs = vec![0.0; tok.vocab_size];
    let mut candidates = Vec::with_capacity(tok.vocab_size);
    let mut rng = SimpleRng::new(1337);
    let mut sentences = 0;
    for _ in 0..cfg.max_new_tokens {
        if cancelled() { break; }
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
        if let Some(callback) = &mut callback {
            let text = if tok.kind() == TokenizerKind::Bpe {
                let bytes: Vec<u8> = ids[prompt_len..].iter()
                    .flat_map(|&id| tok.token_bytes(id).unwrap_or(&[]).iter().copied()).collect();
                String::from_utf8_lossy(&bytes).into_owned()
            } else { tok.decode(&ids[prompt_len..]) };
            callback(&text, ids.len() - prompt_len);
        }
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

#[cfg(test)]
mod tui_tests {
    use super::*;
    #[test]
    fn transformer_chat_streams_same_seeded_answer_and_can_stop() {
        use crate::{dataset::Tokenizer, transformer::{TransformerConfig, TransformerModel}};
        let tok = Tokenizer::from_corpus("hello world user assistant test", true).unwrap();
        let mut model = TransformerModel::new(TransformerConfig {
            d_vocab: tok.vocab_size, d_model: 4, n_heads: 1, d_ff: 8, chunk_len: 4,
            ..Default::default()
        }, 42).unwrap();
        model.vocabulary = tok.ordered_vocabulary().unwrap();
        let path = std::env::temp_dir().join(format!("pssa-trfm-chat-{}.trfm", std::process::id()));
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
        let _cleanup = Cleanup(path.clone());
        transformer_checkpoint::save_model(&model, &path).unwrap();
        let cfg = InferenceConfig { max_new_tokens: 4, ..Default::default() };
        let expected = generate(path.to_str().unwrap(), "hello", &cfg).unwrap();
        let mut updates = Vec::new();
        let actual = generate_controlled(path.to_str().unwrap(), "hello", &cfg,
            &mut |text, count| updates.push((text.to_owned(), count)), &|| false).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(updates.last().unwrap(), &(expected, 4));
        let stopped = std::cell::Cell::new(false);
        let mut count = 0;
        generate_controlled(path.to_str().unwrap(), "hello", &cfg,
            &mut |_, n| { count = n; stopped.set(true); }, &|| stopped.get()).unwrap();
        assert_eq!(count, 1);
    }
}
