use crate::dataset::{Tokenizer, TokenizerKind};
use crate::linalg::SimpleRng;
use crate::pssa::PSSALayerV2;

pub struct InferenceConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
}
impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.70,
            top_p: 0.85,
            top_k: 24,
            repetition_penalty: 1.25,
            max_new_tokens: 64,
        }
    }
}

pub struct PSSAInferenceEngine<'a> {
    model: &'a mut PSSALayerV2,
    tokenizer: &'a Tokenizer,
    rng: SimpleRng,
}
impl<'a> PSSAInferenceEngine<'a> {
    pub fn try_new(model: &'a mut PSSALayerV2, tokenizer: &'a Tokenizer) -> Result<Self, String> {
        if tokenizer.vocab_size < 2 || model.cfg.d_vocab < 2 {
            return Err("generation requires a vocabulary with at least two tokens".into());
        }
        if tokenizer.vocab_size != model.cfg.d_vocab {
            return Err(format!(
                "tokenizer/model vocabulary size mismatch: {} != {}",
                tokenizer.vocab_size, model.cfg.d_vocab
            ));
        }
        if !model.vocabulary.is_empty() && model.vocabulary != tokenizer.ordered_vocabulary()? {
            return Err("tokenizer vocabulary/order does not match checkpoint".into());
        }
        match (model.tokenizer_json.as_ref(), tokenizer.kind()) {
            (Some(json), TokenizerKind::Bpe)
                if tokenizer.serialized_metadata().as_deref() == Some(json) => {}
            (Some(_), _) => {
                return Err(
                    "checkpoint BPE metadata does not exactly match inference tokenizer".into(),
                );
            }
            (None, TokenizerKind::Word) => {}
            (None, TokenizerKind::Bpe) => {
                return Err("BPE tokenizer requires serialized checkpoint metadata".into());
            }
        }
        Ok(Self {
            model,
            tokenizer,
            rng: SimpleRng::new(1337),
        })
    }
    pub fn new(model: &'a mut PSSALayerV2, tokenizer: &'a Tokenizer) -> Self {
        Self::try_new(model, tokenizer).expect("invalid inference model/tokenizer")
    }
    fn validate(cfg: &InferenceConfig) -> Result<(), String> {
        if !cfg.temperature.is_finite() || cfg.temperature < 0.0 {
            return Err("temperature must be finite and >= 0".into());
        }
        if !(cfg.top_p.is_finite() && cfg.top_p > 0.0 && cfg.top_p <= 1.0) {
            return Err("top-p must be finite in (0, 1]".into());
        }
        if cfg.top_k == 0 {
            return Err("top-k must be positive".into());
        }
        if !(cfg.repetition_penalty.is_finite() && cfg.repetition_penalty >= 1.0) {
            return Err("repetition penalty must be finite and >= 1".into());
        }
        Ok(())
    }
    fn sample(
        &mut self,
        cfg: &InferenceConfig,
        generated_ids: &[usize],
        logits: &mut [f32],
        probs: &mut [f32],
        candidates: &mut Vec<(usize, f32)>,
    ) -> Result<usize, String> {
        let d_v = self.model.cfg.d_vocab;
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("model emitted non-finite logits".into());
        }
        if cfg.repetition_penalty > 1.0 {
            for &id in &generated_ids[generated_ids.len().saturating_sub(64)..] {
                if logits[id] > 0.0 {
                    logits[id] /= cfg.repetition_penalty;
                } else {
                    logits[id] *= cfg.repetition_penalty;
                }
            }
        }
        if cfg.temperature == 0.0 {
            return (1..d_v)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
                .ok_or_else(|| "no valid generation candidates".into());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for i in 0..d_v {
            probs[i] = ((logits[i] / cfg.temperature) - max / cfg.temperature).exp();
            sum += probs[i];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err("invalid sampling probability mass".into());
        }
        candidates.clear();
        for i in 1..d_v {
            if probs[i].is_finite() {
                candidates.push((i, probs[i] / sum));
            }
        }
        if candidates.is_empty() {
            return Err("no finite generation candidates".into());
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let k = candidates.len().min(cfg.top_k);
        let mut cutoff = k;
        let mut cumulative = 0.0;
        for (i, &(_, p)) in candidates[..k].iter().enumerate() {
            cumulative += p;
            if cumulative >= cfg.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let filtered = &candidates[..cutoff.max(1)];
        let mass: f32 = filtered.iter().map(|x| x.1).sum();
        let draw = self.rng.gen_range_f32(0.0, mass);
        let mut running = 0.0;
        let mut id = filtered[filtered.len() - 1].0;
        for &(candidate, p) in filtered {
            running += p;
            if draw <= running {
                id = candidate;
                break;
            }
        }
        Ok(id)
    }
    /// Generates autoregressively. BPE callbacks receive decoded UTF-8 segments,
    /// never internal ByteLevel labels; an incomplete final UTF-8 suffix is held.
    pub fn try_generate_chat_turn<F>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        Self::validate(cfg)?;
        let prompt_ids = self.tokenizer.try_encode(prompt, true)?;
        if prompt_ids.is_empty() {
            return Err("prompt is empty after tokenization".into());
        }
        if prompt_ids.iter().all(|&id| id == 0) {
            return Err("prompt contains no known vocabulary tokens".into());
        }
        let d_v = self.model.cfg.d_vocab;
        let mut logits = vec![0.0f32; d_v];
        let mut probs = vec![0.0f32; d_v];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(d_v);
        let mut generated_ids = Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
        generated_ids.extend_from_slice(&prompt_ids);
        self.model.reset_recurrent_state();
        for &id in &prompt_ids {
            if id >= d_v {
                return Err(format!("prompt ID {id} outside model vocabulary"));
            }
            self.model.forward_inference(id, &mut logits);
        }
        match self.tokenizer.kind() {
            TokenizerKind::Word => {
                let mut out = String::with_capacity(cfg.max_new_tokens.saturating_mul(8));
                let mut sentence_count = 0;
                for step in 0..cfg.max_new_tokens {
                    if step > 0 {
                        self.model.forward_inference(
                            *generated_ids.last().expect("generated token"),
                            &mut logits,
                        );
                    }
                    let selected = self.sample(
                        cfg,
                        &generated_ids,
                        &mut logits,
                        &mut probs,
                        &mut candidates,
                    )?;
                    generated_ids.push(selected);
                    let token = self
                        .tokenizer
                        .id_to_token
                        .get(&selected)
                        .ok_or_else(|| format!("missing tokenizer token ID {selected}"))?;
                    if matches!(token.as_str(), "." | "," | "?" | "!") {
                        out.push_str(token);
                    } else {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(token);
                    }
                    callback(token);
                    if matches!(token.as_str(), "." | "?" | "!") {
                        sentence_count += 1;
                        if sentence_count >= 2 {
                            break;
                        }
                    }
                }
                Ok(out)
            }
            TokenizerKind::Bpe => {
                let max_token_bytes = (0..self.tokenizer.vocab_size)
                    .filter_map(|id| self.tokenizer.token_bytes(id))
                    .map(|x| x.len())
                    .max()
                    .unwrap_or(1);
                let raw_capacity = max_token_bytes.saturating_mul(cfg.max_new_tokens);
                let mut raw = Vec::with_capacity(raw_capacity);
                // Invalid raw bytes can each expand to U+FFFD (three bytes).
                let mut out = String::with_capacity(raw_capacity.saturating_mul(3));
                let mut emitted = 0usize;
                for step in 0..cfg.max_new_tokens {
                    if step > 0 {
                        self.model.forward_inference(
                            *generated_ids.last().expect("generated token"),
                            &mut logits,
                        );
                    }
                    let selected = self.sample(
                        cfg,
                        &generated_ids,
                        &mut logits,
                        &mut probs,
                        &mut candidates,
                    )?;
                    generated_ids.push(selected);
                    raw.extend_from_slice(
                        self.tokenizer
                            .token_bytes(selected)
                            .ok_or_else(|| format!("missing BPE bytes for token {selected}"))?,
                    );
                    let begin = out.len();
                    loop {
                        match std::str::from_utf8(&raw[emitted..]) {
                            Ok(valid) => {
                                out.push_str(valid);
                                emitted = raw.len();
                                break;
                            }
                            Err(error) => {
                                let good = error.valid_up_to();
                                if good > 0 {
                                    let end = emitted + good;
                                    out.push_str(
                                        std::str::from_utf8(&raw[emitted..end])
                                            .expect("validated UTF-8 prefix"),
                                    );
                                    emitted = end;
                                }
                                match error.error_len() {
                                    Some(bad) => {
                                        out.push('\u{FFFD}');
                                        emitted += bad;
                                    }
                                    None => break, // retain incomplete UTF-8 until the next token
                                }
                            }
                        }
                    }
                    if out.len() > begin {
                        callback(&out[begin..]);
                    }
                }
                Ok(out)
            }
        }
    }
    pub fn generate_chat_turn<F>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        callback: F,
    ) -> String
    where
        F: FnMut(&str),
    {
        self.try_generate_chat_turn(prompt, cfg, callback)
            .unwrap_or_default()
    }
}
