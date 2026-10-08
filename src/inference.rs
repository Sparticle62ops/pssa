use crate::dataset::{Tokenizer, TokenizerKind};
use crate::linalg::SimpleRng;
use crate::pssa::PSSALayerV2;

pub(crate) fn unknown_prompt_error(tokenizer: &Tokenizer, prompt: &str) -> String {
    let unknown = tokenizer.unknown_words(prompt, true);
    let listed = if unknown.is_empty() {
        "the entered words".to_string()
    } else {
        unknown.join(", ")
    };
    format!(
        "none of these words are in this model's vocabulary (word-level tokenizer, {} words learned from the training text): {listed}. try words from the training data, or use a BPE-tokenized checkpoint",
        tokenizer.vocab_size.saturating_sub(1)
    )
}

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
    pub(crate) fn validate(cfg: &InferenceConfig) -> Result<(), String> {
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
    pub(crate) fn sample(
        rng: &mut SimpleRng,
        cfg: &InferenceConfig,
        generated_ids: &[usize],
        logits: &mut [f32],
        probs: &mut [f32],
        candidates: &mut Vec<(usize, f32)>,
    ) -> Result<usize, String> {
        let d_v = logits.len();
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
        // ID 0 is <unk> and is deliberately excluded from candidates.  Do
        // not let its (irrelevant) logit become the numerical reference for
        // the softmax: a very large <unk> logit would otherwise underflow all
        // valid candidates to zero and report a spurious sampling failure.
        let max = logits[1..]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        probs[0] = 0.0;
        for i in 1..d_v {
            // Subtract before dividing.  With a tiny positive temperature,
            // dividing each finite logit first can produce `inf - inf` and
            // turn an otherwise valid distribution into NaNs.
            probs[i] = ((logits[i] - max) / cfg.temperature).exp();
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
        let draw = rng.gen_range_f32(0.0, mass);
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
    /// never internal ByteLevel labels; incomplete UTF-8 is held until the next
    /// token and any incomplete final suffix is emitted as a replacement character.
    pub fn try_generate_chat_turn<F>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        self.try_generate_chat_turn_impl(
            prompt,
            cfg,
            false,
            |_, delta, _, _| {
                if let Some(delta) = delta {
                    callback(delta);
                }
            },
            || false,
            |_, _, _, _| {},
        )
    }

    /// Generates with cooperative cancellation checked before each prompt token
    /// and each generation step. Cancellation returns the partial reply as `Ok`.
    ///
    /// The callback receives the cumulative decoded reply and number of generated
    /// tokens after every token, even when an incomplete BPE UTF-8 sequence leaves
    /// the reply unchanged. A final incomplete sequence is replaced with U+FFFD;
    /// that final update invokes the callback again with the same token count.
    pub fn try_generate_chat_turn_controlled<F, C>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
        cancelled: C,
    ) -> Result<String, String>
    where
        F: FnMut(&str, usize),
        C: Fn() -> bool,
    {
        self.try_generate_chat_turn_impl(
            prompt,
            cfg,
            false,
            |out, _, count, _| callback(out, count),
            cancelled,
            |_, _, _, _| {},
        )
    }

    /// Observe raw model softmax confidence (before temperature, repetition
    /// penalty, top-k/top-p and exclusion of ID 0). This does not alter sampling.
    /// UTF-8 flushing repeats the final token count/probability, not a new token.
    pub fn try_generate_chat_turn_scored<F, C>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
        cancelled: C,
    ) -> Result<String, String>
    where
        F: FnMut(&str, usize, f32),
        C: Fn() -> bool,
    {
        self.try_generate_chat_turn_impl(
            prompt, cfg, true,
            |out, _, count, probability| callback(out, count, probability),
            cancelled,
            |_, _, _, _| {},
        )
    }

    /// Controlled generation with an optional read-only view of the existing
    /// retrieval scratch buffers. No extra forward pass or sampling is done.
    /// The observer receives (generated count, query/input ID, selected ID, model)
    /// once per generated token, not for prefill or the final BPE text flush.
    /// Weights belong to the input that predicted the selected token; with loops
    /// enabled, scratch buffers hold the final pass of each layer.
    #[cfg(test)]
    pub(crate) fn try_generate_chat_turn_observed<F, C, O>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
        cancelled: C,
        observer: O,
    ) -> Result<String, String>
    where
        F: FnMut(&str, usize),
        C: Fn() -> bool,
        O: FnMut(usize, usize, usize, &PSSALayerV2),
    {
        self.try_generate_chat_turn_impl(
            prompt,
            cfg,
            false,
            |out, _, count, _| callback(out, count),
            cancelled,
            observer,
        )
    }

    /// Combine confidence coloring and retrieval telemetry in the same forward
    /// pass. Both callbacks only observe; sampling and legacy APIs are unchanged.
    pub(crate) fn try_generate_chat_turn_scored_observed<F, C, O>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        mut callback: F,
        cancelled: C,
        observer: O,
    ) -> Result<String, String>
    where
        F: FnMut(&str, usize, f32),
        C: Fn() -> bool,
        O: FnMut(usize, usize, usize, &PSSALayerV2),
    {
        self.try_generate_chat_turn_impl(
            prompt,
            cfg,
            true,
            |out, _, count, probability| callback(out, count, probability),
            cancelled,
            observer,
        )
    }

    // `delta` preserves legacy word-token / decoded-BPE-segment callbacks, while
    // `out` and `count` expose progress even when a token has no decoded text yet.
    fn try_generate_chat_turn_impl<F, C, O>(
        &mut self,
        prompt: &str,
        cfg: &InferenceConfig,
        scored: bool,
        mut callback: F,
        cancelled: C,
        mut observer: O,
    ) -> Result<String, String>
    where
        F: FnMut(&str, Option<&str>, usize, f32),
        C: Fn() -> bool,
        O: FnMut(usize, usize, usize, &PSSALayerV2),
    {
        Self::validate(cfg)?;
        let prompt_ids = self.tokenizer.try_encode(prompt, true)?;
        if prompt_ids.is_empty() {
            return Err("prompt is empty after tokenization".into());
        }
        if prompt_ids.iter().all(|&id| id == 0) {
            return Err(unknown_prompt_error(self.tokenizer, prompt));
        }
        let d_v = self.model.cfg.d_vocab;
        let mut logits = vec![0.0f32; d_v];
        let mut probs = vec![0.0f32; d_v];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(d_v);
        let mut confidence = if scored { vec![0.0; d_v] } else { Vec::new() };
        let mut last_probability = 0.0;
        let mut generated_ids = Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
        generated_ids.extend_from_slice(&prompt_ids);
        self.model.reset_recurrent_state();
        for &id in &prompt_ids {
            if cancelled() {
                return Ok(String::new());
            }
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
                    if cancelled() {
                        break;
                    }
                    if step > 0 {
                        self.model.forward_inference(
                            *generated_ids.last().expect("generated token"),
                            &mut logits,
                        );
                    }
                    raw_confidence(&logits, &mut confidence);
                    let selected = Self::sample(
                        &mut self.rng,
                        cfg,
                        &generated_ids,
                        &mut logits,
                        &mut probs,
                        &mut candidates,
                    )?;
                    last_probability = confidence.get(selected).copied().unwrap_or(0.0);
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
                    observer(
                        step + 1,
                        generated_ids[generated_ids.len() - 2],
                        selected,
                        self.model,
                    );
                    callback(&out, Some(token), step + 1, last_probability);
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
                    if cancelled() {
                        break;
                    }
                    if step > 0 {
                        self.model.forward_inference(
                            *generated_ids.last().expect("generated token"),
                            &mut logits,
                        );
                    }
                    raw_confidence(&logits, &mut confidence);
                    let selected = Self::sample(
                        &mut self.rng,
                        cfg,
                        &generated_ids,
                        &mut logits,
                        &mut probs,
                        &mut candidates,
                    )?;
                    last_probability = confidence.get(selected).copied().unwrap_or(0.0);
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
                    let delta = (out.len() > begin).then_some(&out[begin..]);
                    observer(
                        step + 1,
                        generated_ids[generated_ids.len() - 2],
                        selected,
                        self.model,
                    );
                    callback(&out, delta, step + 1, last_probability);
                }
                // A byte-level token stream may end in the middle of a UTF-8
                // sequence.  Preserve that output as the standard replacement
                // character instead of silently dropping the final bytes.
                if emitted < raw.len() {
                    let tail = String::from_utf8_lossy(&raw[emitted..]);
                    out.push_str(&tail);
                    callback(&out, Some(&tail), generated_ids.len() - prompt_ids.len(), last_probability);
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

// Empty output makes legacy generation a no-op here, retaining its cost and RNG.
fn raw_confidence(logits: &[f32], out: &mut [f32]) {
    if out.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (p, &logit) in out.iter_mut().zip(logits) {
        *p = (logit - max).exp();
        sum += *p;
    }
    for p in out {
        *p /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceConfig, PSSAInferenceEngine, unknown_prompt_error};
    use crate::dataset::Tokenizer;
    use crate::pssa::{PSSAConfigV2, PSSALayerV2};
    use std::cell::Cell;

    fn tiny_model(tokenizer: &Tokenizer) -> PSSALayerV2 {
        let mut model = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: tokenizer.vocab_size,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 2,
                chunk_len: 2,
                ..Default::default()
            },
            5,
        );
        model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
        model.tokenizer_json = tokenizer.serialized_metadata();
        // Equal finite logits make greedy sampling select ID 1 deterministically.
        model.unembed_w.data.fill(0.0);
        model
    }

    fn greedy(max_new_tokens: usize) -> InferenceConfig {
        InferenceConfig {
            temperature: 0.0,
            max_new_tokens,
            ..Default::default()
        }
    }

    fn word_tokenizer(first: &str) -> Tokenizer {
        Tokenizer::from_vocabulary(&["<unk>".into(), first.into(), "prompt".into()]).unwrap()
    }

    fn bpe_tokenizer(first_byte: u8) -> Tokenizer {
        let tokenizer = Tokenizer::from_corpus_bpe("a", 257).unwrap();
        let target = (1..tokenizer.vocab_size)
            .find(|&id| tokenizer.token_bytes(id) == Some(&[first_byte][..]))
            .unwrap();
        // Keep the full byte alphabet but put the byte under test at greedy ID 1.
        let mut metadata: serde_json::Value =
            serde_json::from_str(&tokenizer.serialized_metadata().unwrap()).unwrap();
        let vocab = metadata["model"]["vocab"].as_object_mut().unwrap();
        vocab.insert(tokenizer.id_to_token[&1].clone(), target.into());
        vocab.insert(tokenizer.id_to_token[&target].clone(), 1.into());
        Tokenizer::from_serialized(&metadata.to_string()).unwrap()
    }

    #[test]
    fn confidence_observation_preserves_sampling_and_reports_raw_probability() {
        for temperature in [0.0, 0.7] {
            let tokenizer = word_tokenizer("word");
            let mut plain_model = tiny_model(&tokenizer);
            let mut scored_model = tiny_model(&tokenizer);
            let cfg = InferenceConfig { temperature, top_k: 1, max_new_tokens: 3, ..Default::default() };
            let plain = PSSAInferenceEngine::new(&mut plain_model, &tokenizer)
                .try_generate_chat_turn_controlled("prompt", &cfg, |_, _| {}, || false).unwrap();
            let mut scores = Vec::new();
            let scored = PSSAInferenceEngine::new(&mut scored_model, &tokenizer)
                .try_generate_chat_turn_scored("prompt", &cfg, |text, count, p| scores.push((text.to_owned(), count, p)), || false).unwrap();
            assert_eq!(plain, scored);
            assert_eq!(scores.len(), 3);
            for (_, _, p) in scores { assert_eq!(p, 1.0 / 3.0); } // includes unk, before top-k=1
            assert_eq!(plain_model.inf_features, scored_model.inf_features);
        }
        let tokenizer = bpe_tokenizer(0xc3);
        let mut model = tiny_model(&tokenizer);
        let mut scores = Vec::new();
        PSSAInferenceEngine::new(&mut model, &tokenizer).try_generate_chat_turn_scored(
            "a", &greedy(2), |text, count, p| scores.push((text.to_owned(), count, p)), || false).unwrap();
        assert_eq!(scores.iter().map(|(_, n, _)| *n).collect::<Vec<_>>(), [1, 2, 2]);
        assert_eq!(scores[1].2, scores[2].2);
        assert_eq!(scores[2].0, "��");
    }

    #[test]
    fn controlled_words_are_cumulative_and_legacy_callbacks_remain_raw() {
        for (token, expected, snapshots) in [
            (
                "word",
                "word word word",
                vec!["word", "word word", "word word word"],
            ),
            (".", "..", vec![".", ".."]),
        ] {
            let tokenizer = word_tokenizer(token);
            let mut model = tiny_model(&tokenizer);
            let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
            let mut legacy = Vec::new();
            let legacy_out = engine
                .try_generate_chat_turn("prompt prompt", &greedy(3), |s| legacy.push(s.to_owned()))
                .unwrap();
            let mut progress = Vec::new();
            let controlled_out = engine
                .try_generate_chat_turn_controlled(
                    "prompt prompt",
                    &greedy(3),
                    |s, count| progress.push((s.to_owned(), count)),
                    || false,
                )
                .unwrap();
            assert_eq!(legacy_out, expected);
            assert_eq!(controlled_out, legacy_out);
            assert_eq!(legacy, vec![token; snapshots.len()]);
            assert_eq!(
                progress,
                snapshots
                    .into_iter()
                    .enumerate()
                    .map(|(index, text)| (text.to_owned(), index + 1))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn controlled_bpe_preserves_legacy_decoded_deltas_and_counts_incomplete_tokens() {
        for (byte, expected, deltas, snapshots) in [
            (
                b' ',
                "   ",
                vec![" ", " ", " "],
                vec![(" ", 1), ("  ", 2), ("   ", 3)],
            ),
            // Each leading byte is incomplete until the next byte proves it invalid.
            // Final flushing emits a replacement without inventing a fourth token.
            (
                0xc3,
                "���",
                vec!["�", "�", "�"],
                vec![("", 1), ("�", 2), ("��", 3), ("���", 3)],
            ),
        ] {
            let tokenizer = bpe_tokenizer(byte);
            let mut model = tiny_model(&tokenizer);
            let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
            let mut legacy = Vec::new();
            let legacy_out = engine
                .try_generate_chat_turn("a", &greedy(3), |s| legacy.push(s.to_owned()))
                .unwrap();
            let mut progress = Vec::new();
            let controlled_out = engine
                .try_generate_chat_turn_controlled(
                    "a",
                    &greedy(3),
                    |s, count| progress.push((s.to_owned(), count)),
                    || false,
                )
                .unwrap();
            assert_eq!(legacy_out, expected);
            assert_eq!(legacy, deltas);
            assert_eq!(controlled_out, legacy_out);
            assert_eq!(
                progress,
                snapshots
                    .into_iter()
                    .map(|(text, count)| (text.to_owned(), count))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn cancellation_returns_partial_words_and_resets_on_next_turn() {
        let tokenizer = word_tokenizer("word");
        let mut model = tiny_model(&tokenizer);
        let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
        let count = Cell::new(0);
        let out = engine
            .try_generate_chat_turn_controlled(
                "prompt",
                &greedy(10),
                |_, generated| count.set(generated),
                || count.get() >= 2,
            )
            .unwrap();
        assert_eq!(out, "word word");
        assert_eq!(count.get(), 2);
        assert_eq!(
            engine
                .try_generate_chat_turn("prompt", &greedy(3), |_| {})
                .unwrap(),
            "word word word"
        );
    }

    #[test]
    fn cancellation_after_incomplete_bpe_token_flushes_partial_reply() {
        let tokenizer = bpe_tokenizer(0xc3);
        let mut model = tiny_model(&tokenizer);
        let count = Cell::new(0);
        let mut progress = Vec::new();
        let out = PSSAInferenceEngine::new(&mut model, &tokenizer)
            .try_generate_chat_turn_controlled(
                "a",
                &greedy(10),
                |s, generated| {
                    count.set(generated);
                    progress.push((s.to_owned(), generated));
                },
                || count.get() >= 1,
            )
            .unwrap();
        assert_eq!(out, "�");
        assert_eq!(progress, vec![(String::new(), 1), ("�".into(), 1)]);
    }

    #[test]
    fn cancellation_during_prefill_stops_before_next_prompt_token() {
        for tokenizer in [word_tokenizer("word"), bpe_tokenizer(b' ')] {
            let prompt = "prompt prompt prompt";
            let ids = tokenizer.try_encode(prompt, true).unwrap();
            assert!(ids.len() > 2);
            let mut model = tiny_model(&tokenizer);
            let mut expected = tiny_model(&tokenizer);
            expected.reset_recurrent_state();
            expected.forward_inference(ids[0], &mut vec![0.0; tokenizer.vocab_size]);
            let checks = Cell::new(0);
            let out = PSSAInferenceEngine::new(&mut model, &tokenizer)
                .try_generate_chat_turn_controlled(
                    prompt,
                    &greedy(10),
                    |_, _| panic!("prefill must not emit generated-token callbacks"),
                    || {
                        checks.set(checks.get() + 1);
                        checks.get() == 2
                    },
                )
                .unwrap();
            assert_eq!(out, "");
            assert_eq!(checks.get(), 2);
            assert_eq!(model.inf_features, expected.inf_features);
            assert_eq!(model.inf_z_final, expected.inf_z_final);
        }
    }

    #[test]
    fn cancellation_before_generation_and_zero_budget_emit_no_callbacks() {
        for tokenizer in [word_tokenizer("word"), bpe_tokenizer(b' ')] {
            for max_new_tokens in [0, 3] {
                let mut model = tiny_model(&tokenizer);
                let out = PSSAInferenceEngine::new(&mut model, &tokenizer)
                    .try_generate_chat_turn_controlled(
                        "prompt",
                        &greedy(max_new_tokens),
                        |_, _| panic!("no generated tokens expected"),
                        || max_new_tokens > 0,
                    )
                    .unwrap();
                assert_eq!(out, "");
            }
        }
    }

    #[test]
    fn controlled_generation_preserves_validation_errors() {
        let tokenizer = word_tokenizer("word");
        let mut model = tiny_model(&tokenizer);
        let mut engine = PSSAInferenceEngine::new(&mut model, &tokenizer);
        for prompt in ["", "unknown", "prompt"] {
            let mut cfg = greedy(1);
            if prompt == "prompt" {
                cfg.temperature = f32::NAN;
            }
            let legacy = engine
                .try_generate_chat_turn(prompt, &cfg, |_| {})
                .unwrap_err();
            let controlled = engine
                .try_generate_chat_turn_controlled(prompt, &cfg, |_, _| {}, || false)
                .unwrap_err();
            assert_eq!(controlled, legacy);
        }
    }

    #[test]
    fn memory_observer_preserves_sampling_output_and_exact_retrieval_state() {
        fn populated(tokenizer: &Tokenizer, depth: usize, loops: usize) -> PSSALayerV2 {
            let mut model = PSSALayerV2::new_with_depth_and_loops(
                PSSAConfigV2 {
                    d_vocab: tokenizer.vocab_size,
                    d_latent: 4,
                    d_state: 2,
                    d_mem_key: 2,
                    mem_capacity: 3,
                    chunk_len: 2,
                    ..Default::default()
                },
                23,
                depth,
                loops,
            );
            model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
            model.tokenizer_json = tokenizer.serialized_metadata();
            for block in std::iter::once(&mut model.block).chain(&mut model.extra_blocks) {
                block.memory.insert(&[0.1, 0.2], &[0.3, -0.5, 0.7, 0.1]);
                block.memory.insert(&[-0.3, 0.4], &[-0.1, 0.4, 0.6, -0.2]);
                // The unused tail is not a valid occupied-slot strength.
                block.inf_mem_weights[2] = -123.0;
            }
            model
        }
        fn runtime_bits(model: &PSSALayerV2) -> Vec<u32> {
            let mut carry = vec![0.0; model.recurrent_state_len()];
            model.copy_recurrent_state_to(&mut carry);
            let mut bits: Vec<_> = carry.iter().map(|v| v.to_bits()).collect();
            bits.extend(model.inf_features.iter().map(|v| v.to_bits()));
            for block in std::iter::once(&model.block).chain(&model.extra_blocks) {
                for values in [
                    &block.inf_mem_weights,
                    &block.inf_q_pnc,
                    &block.inf_m_val,
                    &block.inf_z_final,
                ] {
                    bits.extend(values.iter().map(|v| v.to_bits()));
                }
            }
            bits
        }
        for tokenizer in [word_tokenizer("word"), bpe_tokenizer(0xc3)] {
            for (depth, loops) in [(1, 1), (2, 3)] {
                let cfg = InferenceConfig {
                    max_new_tokens: 5,
                    ..Default::default()
                };
                let mut baseline = populated(&tokenizer, depth, loops);
                let mut observed = populated(&tokenizer, depth, loops);
                let mut replay = populated(&tokenizer, depth, loops);
                let banks: Vec<_> = std::iter::once(&observed.block)
                    .chain(&observed.extra_blocks)
                    .map(|b| b.memory.clone())
                    .collect();
                let prompt_ids = tokenizer.try_encode("prompt prompt", true).unwrap();
                let mut logits = vec![0.0; tokenizer.vocab_size];
                replay.reset_recurrent_state();
                for &id in &prompt_ids {
                    replay.forward_inference(id, &mut logits);
                }
                let mut baseline_progress = Vec::new();
                let mut baseline_engine = PSSAInferenceEngine::new(&mut baseline, &tokenizer);
                let expected = baseline_engine
                    .try_generate_chat_turn_controlled(
                        "prompt prompt",
                        &cfg,
                        |text, count| baseline_progress.push((text.to_owned(), count)),
                        || false,
                    )
                    .unwrap();
                let rng = baseline_engine.rng.state;
                let mut observed_progress = Vec::new();
                let mut seen = 0;
                let mut last_id = *prompt_ids.last().unwrap();
                let mut observed_engine = PSSAInferenceEngine::new(&mut observed, &tokenizer);
                let actual = observed_engine
                    .try_generate_chat_turn_observed(
                        "prompt prompt",
                        &cfg,
                        |text, count| observed_progress.push((text.to_owned(), count)),
                        || false,
                        |count, query_id, selected_id, model| {
                            seen += 1;
                            assert_eq!(count, seen);
                            assert_eq!(query_id, last_id);
                            if count > 1 {
                                replay.forward_inference(query_id, &mut logits);
                            }
                            assert_eq!(runtime_bits(model), runtime_bits(&replay));
                            for (block, bank) in std::iter::once(&model.block)
                                .chain(&model.extra_blocks)
                                .zip(&banks)
                            {
                                assert_eq!(&block.memory, bank);
                            }
                            last_id = selected_id;
                        },
                    )
                    .unwrap();
                assert_eq!(
                    observed_engine.rng.state, rng,
                    "observation must not consume RNG"
                );
                assert_eq!(actual, expected);
                assert_eq!(observed_progress, baseline_progress);
                assert_eq!(seen, baseline_progress.last().unwrap().1);
                assert_eq!(runtime_bits(&observed), runtime_bits(&baseline));
                assert_eq!(observed.step_counter, baseline.step_counter);
            }
        }
    }

    #[test]
    fn memory_observer_skips_prefill_and_bpe_flush_and_respects_cancellation() {
        let tokenizer = bpe_tokenizer(0xc3);
        let mut model = tiny_model(&tokenizer);
        let count = Cell::new(0);
        let mut observed = Vec::new();
        let mut progress = Vec::new();
        let out = PSSAInferenceEngine::new(&mut model, &tokenizer)
            .try_generate_chat_turn_observed(
                "a",
                &greedy(10),
                |text, tokens| {
                    count.set(tokens);
                    progress.push((text.to_owned(), tokens));
                },
                || count.get() == 1,
                |tokens, _, selected, model| {
                    observed.push((tokens, selected));
                    assert_eq!(model.memory.count, 0);
                },
            )
            .unwrap();
        assert_eq!(out, "�");
        assert_eq!(observed, [(1, 1)]);
        assert_eq!(progress, [(String::new(), 1), ("�".into(), 1)]);
        for max_new_tokens in [0, 3] {
            let out = PSSAInferenceEngine::new(&mut model, &tokenizer)
                .try_generate_chat_turn_observed(
                    "a",
                    &greedy(max_new_tokens),
                    |_, _| panic!("no output"),
                    || max_new_tokens > 0,
                    |_, _, _, _| panic!("no observation without a generated token"),
                )
                .unwrap();
            assert_eq!(out, "");
        }
    }

    #[test]
    fn unknown_prompt_error_explains_word_vocabulary() {
        let tokenizer =
            Tokenizer::from_vocabulary(&["<unk>".into(), "alpha".into(), "beta".into()]).unwrap();
        let error = unknown_prompt_error(&tokenizer, "hello saturn hello");
        assert!(error.contains("word-level tokenizer, 2 words learned from the training text"));
        assert!(error.contains(": hello, saturn."));
        assert!(error.contains("try words from the training data"));
    }
}
