//! Frozen-model held-out scoring with strict, non-wrapping token windows.
//!
//! Lines are independent documents, as in training. Offsets count encoded
//! tokens, not next-token predictions; neither context nor targets outside the
//! selected window are used. Unlike training windows, evaluation never wraps
//! at EOF and rejects an explicit window that extends beyond the corpus.

use crate::dataset::{TokenChunkIterator, Tokenizer};
use crate::pssa::PSSALayerV2;
use crate::transformer::TransformerModel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationSlice {
    pub skip_tokens: usize,
    pub max_tokens: Option<usize>,
}

/// Select a contiguous window of the line-encoded corpus without joining lines.
///
/// Nonempty one-token fragments are retained for encoded-token/OOV accounting,
/// but never create a prediction across a document or window boundary. At least
/// one selected document must contain a next-token transition.
pub fn documents(
    raw: &str,
    tokenizer: &Tokenizer,
    slice: EvaluationSlice,
) -> Result<Vec<Vec<usize>>, String> {
    if slice.max_tokens == Some(0) {
        return Err(
            "evaluation --max-tokens must be positive; omit it to score through EOF".into(),
        );
    }
    let requested_end = slice
        .max_tokens
        .map(|limit| {
            slice.skip_tokens.checked_add(limit).ok_or_else(|| {
                "evaluation token window overflow; reduce --skip-tokens or --max-tokens".to_string()
            })
        })
        .transpose()?;
    let mut total = 0usize;
    let mut selected = Vec::new();
    for line in raw.lines() {
        let ids = tokenizer.try_encode(line, true)?;
        let line_start = total;
        total = total
            .checked_add(ids.len())
            .ok_or("evaluation corpus token count overflow")?;
        let start = slice.skip_tokens.max(line_start);
        let end = requested_end.unwrap_or(usize::MAX).min(total);
        if start < end {
            selected.push(ids[start - line_start..end - line_start].to_vec());
        }
    }
    if slice.skip_tokens > total {
        return Err(format!(
            "evaluation --skip-tokens={} exceeds corpus length {total}; choose an offset within the corpus (evaluation does not wrap)",
            slice.skip_tokens
        ));
    }
    if let Some(end) = requested_end {
        if end > total {
            return Err(format!(
                "evaluation token window ends at {end}, beyond corpus length {total}; reduce --max-tokens or omit it to score through EOF (evaluation does not wrap)"
            ));
        }
    }
    if !selected.iter().any(|doc| doc.len() >= 2) {
        return Err(
            "evaluation slice has no within-document token transitions; select at least two tokens from the same line".into(),
        );
    }
    Ok(selected)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metrics {
    /// Mean next-token cross entropy in nats (token-weighted across chunks).
    pub loss: f64,
    /// Number of scored next-token transitions, not selected encoded tokens.
    pub tokens: usize,
    pub correct: usize,
    /// Unknown-token count across all selected encoded tokens.
    pub oov: usize,
    /// Selected encoded tokens, including fragments without a transition.
    pub encoded_tokens: usize,
    /// Effective runtime pass count used by the evaluator.
    pub loops: usize,
}

impl Metrics {
    fn for_documents(docs: &[Vec<usize>], loops: usize) -> Self {
        Self {
            loss: 0.0,
            tokens: 0,
            correct: 0,
            oov: docs.iter().flatten().filter(|&&id| id == 0).count(),
            encoded_tokens: docs.iter().map(Vec::len).sum(),
            loops,
        }
    }

    fn add_chunk(
        &mut self,
        loss: f32,
        logits: &[f32],
        targets: &[usize],
        vocab: usize,
    ) -> Result<(), String> {
        if !loss.is_finite() || logits.iter().any(|value| !value.is_finite()) {
            return Err("non-finite evaluation loss or logits".into());
        }
        self.loss += loss as f64 * targets.len() as f64;
        self.tokens += targets.len();
        for (row, &target) in logits.chunks_exact(vocab).zip(targets) {
            // The lowest token ID wins a tie, matching both legacy evaluators.
            let guess = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(id, _)| id)
                .unwrap_or(0);
            self.correct += usize::from(guess == target);
        }
        Ok(())
    }

    /// Legacy machine-readable metric fields. Overflowed perplexity is JSON
    /// null, never infinity. Non-finite manually constructed losses are also
    /// represented as null; the evaluators themselves reject non-finite scores.
    pub fn json(&self) -> String {
        let ce = if self.loss.is_finite() {
            format!("{:.8}", self.loss)
        } else {
            "null".into()
        };
        let ppl = self.loss.exp();
        let overflow = !self.loss.is_finite() || !ppl.is_finite();
        let ppl_json = if overflow {
            "null".into()
        } else {
            format!("{ppl:.8}")
        };
        format!(
            "{{\"cross_entropy\":{ce},\"perplexity\":{ppl_json},\"perplexity_overflow\":{overflow},\"oov_rate\":{:.8},\"token_count\":{},\"next_token_accuracy\":{:.8},\"loops\":{}}}",
            self.oov as f64 / self.encoded_tokens.max(1) as f64,
            self.tokens,
            self.correct as f64 / self.tokens.max(1) as f64,
            self.loops,
        )
    }
}

fn validate_model_input(
    tokenizer: &Tokenizer,
    docs: &[Vec<usize>],
    vocab: usize,
    chunk_len: usize,
) -> Result<(), String> {
    if tokenizer.vocab_size != vocab || docs.iter().flatten().any(|&id| id >= vocab) {
        return Err(
            "evaluation tokenizer/model vocabulary mismatch; use the checkpoint tokenizer".into(),
        );
    }
    if chunk_len == 0 {
        return Err("evaluation model chunk length must be positive".into());
    }
    Ok(())
}

/// Score with fresh recurrent state for each selected document. Trained memory
/// remains available for retrieval, but is never written. Parameters, gradients,
/// optimizer state, RNG and incoming recurrent carry are unchanged; only runtime
/// caches and the forward tape are overwritten. Carry is retained across chunks
/// within each document, just as in the legacy whole-corpus evaluator.
pub fn evaluate_pssa(
    model: &mut PSSALayerV2,
    tokenizer: &Tokenizer,
    raw: &str,
    slice: EvaluationSlice,
) -> Result<Metrics, String> {
    let docs = documents(raw, tokenizer, slice)?;
    let vocab = model.cfg.d_vocab;
    validate_model_input(tokenizer, &docs, vocab, model.cfg.chunk_len)?;
    let mut metrics = Metrics::for_documents(&docs, model.loops());
    let mut incoming_carries = vec![0.0; model.recurrent_state_len()];
    model.copy_recurrent_state_to(&mut incoming_carries);
    let result = (|| {
        for doc in &docs {
            model.reset_recurrent_state();
            for (input, targets) in TokenChunkIterator::new(doc, model.cfg.chunk_len) {
                // Forward only: no backward, optimizer, consolidation or memory-write path.
                let loss = model.forward_train_chunk(input, targets);
                metrics.add_chunk(
                    loss,
                    &model.tape.logits[..input.len() * vocab],
                    targets,
                    vocab,
                )?;
            }
        }
        metrics.loss /= metrics.tokens as f64;
        Ok(metrics)
    })();
    // Restore every layer even when scoring returns a non-finite-logit error.
    model.copy_recurrent_state_from(&incoming_carries);
    result
}

/// Score the baseline using its existing chunk-local context policy. Forward
/// tape/scratch are overwritten, but no gradients or optimizer updates occur.
pub fn evaluate_transformer(
    model: &mut TransformerModel,
    tokenizer: &Tokenizer,
    raw: &str,
    slice: EvaluationSlice,
) -> Result<Metrics, String> {
    let docs = documents(raw, tokenizer, slice)?;
    let vocab = model.cfg.d_vocab;
    validate_model_input(tokenizer, &docs, vocab, model.cfg.chunk_len)?;
    let mut metrics = Metrics::for_documents(&docs, 1);
    for doc in &docs {
        for (input, targets) in TokenChunkIterator::new(doc, model.cfg.chunk_len) {
            let loss = model.forward_train_chunk(input, targets);
            metrics.add_chunk(loss, model.training_logits(), targets, vocab)?;
        }
    }
    metrics.loss /= metrics.tokens as f64;
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pssa::{PSSAConfigV2, ParamMatrix, ParamVector};
    use crate::transformer::TransformerConfig;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_vocabulary(&["<unk>", "a", "b", "c", "d", "e", "f"].map(str::to_string))
            .unwrap()
    }

    fn slice(skip_tokens: usize, max_tokens: usize) -> EvaluationSlice {
        EvaluationSlice {
            skip_tokens,
            max_tokens: Some(max_tokens),
        }
    }

    #[test]
    fn slices_preserve_partial_lines_singletons_and_exact_boundaries() {
        let tok = tokenizer();
        let raw = "a b c\n\nunknown\nd e f a\nb c";
        assert_eq!(
            documents(raw, &tok, slice(1, 6)).unwrap(),
            vec![vec![2, 3], vec![0], vec![4, 5, 6]],
        );
        assert_eq!(
            documents(raw, &tok, slice(4, 4)).unwrap(),
            vec![vec![4, 5, 6, 1]]
        );
        assert_eq!(
            documents(raw, &tok, slice(4, 3)).unwrap(),
            vec![vec![4, 5, 6]]
        );
        assert_eq!(
            documents(
                raw,
                &tok,
                EvaluationSlice {
                    skip_tokens: 8,
                    max_tokens: None
                }
            )
            .unwrap(),
            vec![vec![2, 3]],
        );
        assert_eq!(
            documents(raw, &tok, EvaluationSlice::default()).unwrap(),
            vec![vec![1, 2, 3], vec![0], vec![4, 5, 6, 1], vec![2, 3]],
        );
    }

    #[test]
    fn slices_reject_out_of_range_overflow_and_no_transitions_without_wrapping() {
        let tok = tokenizer();
        let raw = "a b\nc d";
        for window in [
            slice(5, 2),
            slice(4, 2),
            slice(0, 5),
            slice(3, 2),
            slice(0, 0),
            slice(usize::MAX, 2),
            slice(1, usize::MAX),
            EvaluationSlice {
                skip_tokens: 5,
                max_tokens: None,
            },
            EvaluationSlice {
                skip_tokens: 4,
                max_tokens: None,
            },
            // Two selected tokens, but they belong to different documents.
            slice(1, 2),
            slice(0, 1),
        ] {
            assert!(documents(raw, &tok, window).is_err(), "accepted {window:?}");
        }
        for raw in ["", "\n\n", "a", "a\nb\nc"] {
            assert!(documents(raw, &tok, EvaluationSlice::default()).is_err());
        }
        assert!(
            documents("a b\nc d", &tok, slice(3, 2))
                .unwrap_err()
                .contains("does not wrap")
        );
        assert!(
            documents("a b", &tok, slice(1, usize::MAX))
                .unwrap_err()
                .contains("overflow")
        );
    }

    #[test]
    fn bpe_offsets_count_encoded_tokens_and_do_not_encode_newlines() {
        let raw = "Alpha beta gamma.\n\nDelta epsilon zeta.";
        let tok = Tokenizer::from_corpus_bpe(raw, 270).unwrap();
        let first = tok.try_encode("Alpha beta gamma.", true).unwrap();
        let second = tok.try_encode("Delta epsilon zeta.", true).unwrap();
        let count = first.len() + second.len() - 2;
        assert_eq!(
            documents(raw, &tok, slice(1, count)).unwrap(),
            vec![first[1..].to_vec(), second[..second.len() - 1].to_vec()],
        );
    }

    fn pssa() -> PSSALayerV2 {
        let mut model = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: 7,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 3,
                chunk_len: 2,
                ..Default::default()
            },
            11,
        );
        model.memory.insert(&[0.1, -0.1], &[0.2, 0.3, -0.4, 0.5]);
        model.memory.insert(&[-0.15, 0.2], &[-0.3, 0.7, 0.6, 0.1]);
        // Nonzero gradients/moments make the frozen-state check non-vacuous.
        model.forward_train_chunk(&[1, 2], &[2, 3]);
        model.backward_chunk(2, 1.0);
        model.apply_adamw(model.cfg.lr);
        model.h_persistent.fill(9.0);
        model
    }

    fn transformer() -> TransformerModel {
        let mut model = TransformerModel::new(
            TransformerConfig {
                d_vocab: 7,
                d_model: 4,
                n_heads: 2,
                d_ff: 5,
                chunk_len: 2,
                ..Default::default()
            },
            11,
        )
        .unwrap();
        model.forward_train_chunk(&[1, 2], &[2, 3]);
        model.backward_chunk(2, 1.0);
        model.apply_adamw(model.cfg.lr).unwrap();
        model
    }

    type Parameters = (Vec<ParamMatrix>, Vec<ParamVector>, usize, u64);

    fn pssa_parameters(model: &PSSALayerV2) -> Parameters {
        (
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
            .into_iter()
            .cloned()
            .collect(),
            vec![model.norm_gamma.clone(), model.norm_beta.clone()],
            model.step_counter,
            model.rng.state,
        )
    }

    fn transformer_parameters(model: &TransformerModel) -> Parameters {
        (
            [
                &model.token_embed,
                &model.qkv,
                &model.out_proj,
                &model.ff1,
                &model.ff2,
                &model.unembed,
            ]
            .into_iter()
            .cloned()
            .collect(),
            vec![
                model.norm1_gamma.clone(),
                model.norm1_beta.clone(),
                model.norm2_gamma.clone(),
                model.norm2_beta.clone(),
            ],
            model.step_counter,
            model.rng.state,
        )
    }

    fn correct(logits: &[f32], targets: &[usize]) -> usize {
        logits
            .chunks_exact(7)
            .zip(targets)
            .filter(|(row, target)| {
                let mut best = 0;
                for id in 1..row.len() {
                    if row[id] > row[best] {
                        best = id;
                    }
                }
                best == **target
            })
            .count()
    }

    const RAW: &str = "f e d\na b c d e\nunknown\nf a b c\ne d c";

    #[test]
    fn pssa_selected_score_matches_manual_chunks_and_changes_no_trained_state() {
        let tok = tokenizer();
        let mut model = pssa();
        let before = pssa_parameters(&model);
        let memory = model.memory.clone();
        let adapters = model.adapters.clone();
        let carry = model.h_persistent.clone();
        assert!(model.embed_w.grad.iter().any(|&x| x != 0.0));
        assert!(model.embed_w.m.iter().any(|&x| x != 0.0));
        let metrics = evaluate_pssa(&mut model, &tok, RAW, slice(4, 8)).unwrap();
        assert_eq!(pssa_parameters(&model), before);
        assert_eq!(model.memory, memory);
        assert_eq!(model.adapters, adapters);
        assert_eq!(model.h_persistent, carry);
        assert_eq!(
            (metrics.tokens, metrics.encoded_tokens, metrics.oov),
            (5, 8, 1)
        );

        // Explicitly score the three selected chunks. No selector or chunk-plan
        // helper is used for the reference, and the second document resets carry.
        let mut manual = pssa();
        manual.reset_recurrent_state();
        let mut sum = manual.forward_train_chunk(&[2, 3], &[3, 4]) as f64 * 2.0;
        let mut hits = correct(&manual.tape.logits[..14], &[3, 4]);
        sum += manual.forward_train_chunk(&[4], &[5]) as f64;
        hits += correct(&manual.tape.logits[..7], &[5]);
        manual.reset_recurrent_state();
        sum += manual.forward_train_chunk(&[6, 1], &[1, 2]) as f64 * 2.0;
        hits += correct(&manual.tape.logits[..14], &[1, 2]);
        assert_eq!(metrics.loss, sum / 5.0);
        assert_eq!(metrics.correct, hits);
        assert!(metrics.loss.is_finite());
        assert!(metrics.loss.exp().is_finite());
        assert_eq!(
            evaluate_pssa(&mut model, &tok, RAW, slice(4, 8)).unwrap(),
            metrics
        );

        // Frozen does not mean disabled: retained trained memory affects scores.
        let mut no_memory = pssa();
        no_memory.memory.count = 0;
        let without = evaluate_pssa(&mut no_memory, &tok, RAW, slice(4, 8)).unwrap();
        assert!((metrics.loss - without.loss).abs() > 1e-7);
    }

    #[test]
    fn transformer_selected_score_matches_manual_chunks_and_changes_no_trained_state() {
        let tok = tokenizer();
        let mut model = transformer();
        let before = transformer_parameters(&model);
        assert!(model.token_embed.grad.iter().any(|&x| x != 0.0));
        assert!(model.token_embed.m.iter().any(|&x| x != 0.0));
        let metrics = evaluate_transformer(&mut model, &tok, RAW, slice(4, 8)).unwrap();
        assert_eq!(transformer_parameters(&model), before);
        assert_eq!(
            (metrics.tokens, metrics.encoded_tokens, metrics.oov),
            (5, 8, 1)
        );
        let mut manual = transformer();
        let mut sum = manual.forward_train_chunk(&[2, 3], &[3, 4]) as f64 * 2.0;
        let mut hits = correct(manual.training_logits(), &[3, 4]);
        sum += manual.forward_train_chunk(&[4], &[5]) as f64;
        hits += correct(manual.training_logits(), &[5]);
        sum += manual.forward_train_chunk(&[6, 1], &[1, 2]) as f64 * 2.0;
        hits += correct(manual.training_logits(), &[1, 2]);
        assert_eq!(metrics.loss, sum / 5.0);
        assert_eq!(metrics.correct, hits);
        assert!(metrics.loss.is_finite());
        assert!(metrics.loss.exp().is_finite());
        assert_eq!(
            evaluate_transformer(&mut model, &tok, RAW, slice(4, 8)).unwrap(),
            metrics
        );
    }

    #[test]
    fn default_evaluation_matches_explicit_whole_corpus_window_for_both_models() {
        let tok = tokenizer();
        let mut pssa = pssa();
        let mut transformer = transformer();
        assert_eq!(
            evaluate_pssa(&mut pssa, &tok, RAW, EvaluationSlice::default()).unwrap(),
            evaluate_pssa(&mut pssa, &tok, RAW, slice(0, 16)).unwrap(),
        );
        assert_eq!(
            evaluate_transformer(&mut transformer, &tok, RAW, EvaluationSlice::default()).unwrap(),
            evaluate_transformer(&mut transformer, &tok, RAW, slice(0, 16)).unwrap(),
        );
    }

    #[test]
    fn json_uses_selected_token_denominator_and_legacy_fields_with_finite_perplexity() {
        let tok = tokenizer();
        let mut pssa = pssa();
        let mut transformer = transformer();
        pssa.unembed_w.data.fill(0.0);
        transformer.unembed.data.fill(0.0);
        // Unknown tokens outside the window must not affect OOV accounting.
        let raw = "outside\na unknown b\nignored";
        for metrics in [
            evaluate_pssa(&mut pssa, &tok, raw, slice(1, 3)).unwrap(),
            evaluate_transformer(&mut transformer, &tok, raw, slice(1, 3)).unwrap(),
        ] {
            assert_eq!(
                (
                    metrics.encoded_tokens,
                    metrics.tokens,
                    metrics.correct,
                    metrics.oov
                ),
                (3, 2, 1, 1)
            );
            let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
            assert_eq!(json.as_object().unwrap().len(), 7);
            assert_eq!(json["loops"], 1);
            assert!((json["cross_entropy"].as_f64().unwrap() - 7f64.ln()).abs() < 1e-6);
            assert!((json["perplexity"].as_f64().unwrap() - 7.0).abs() < 1e-5);
            assert_eq!(json["perplexity_overflow"], false);
            assert!((json["oov_rate"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-8);
            assert_eq!(json["token_count"], 2);
            assert_eq!(json["next_token_accuracy"], 0.5);
        }
        // Singleton fragments remain part of the selected OOV denominator.
        let metrics =
            evaluate_pssa(&mut pssa, &tok, "unknown\na b", EvaluationSlice::default()).unwrap();
        assert_eq!(
            (metrics.encoded_tokens, metrics.tokens, metrics.oov),
            (3, 1, 1)
        );
    }

    #[test]
    fn json_contains_effective_loop_count() {
        let metrics = Metrics {
            loss: 0.0,
            tokens: 0,
            correct: 0,
            oov: 0,
            encoded_tokens: 0,
            loops: 3,
        };
        let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
        assert_eq!(json["loops"], 3);
    }

    #[test]
    fn json_encodes_perplexity_overflow_as_null_not_infinity() {
        let mut metrics = Metrics {
            loss: 1000.0,
            tokens: 2,
            correct: 1,
            oov: 1,
            encoded_tokens: 3,
            loops: 1,
        };
        let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
        assert_eq!(json["cross_entropy"], 1000.0);
        assert_eq!(json["perplexity_overflow"], true);
        assert!(json["perplexity"].is_null());
        for loss in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            metrics.loss = loss;
            let json: serde_json::Value = serde_json::from_str(&metrics.json()).unwrap();
            assert!(json["cross_entropy"].is_null());
            assert!(json["perplexity"].is_null());
            assert_eq!(json["perplexity_overflow"], true);
        }
    }

    #[test]
    fn invalid_model_tokenizer_and_nonfinite_scores_return_errors() {
        let tok = Tokenizer::from_vocabulary(&["<unk>", "a"].map(str::to_string)).unwrap();
        let mut pssa = pssa();
        let mut transformer = transformer();
        assert!(
            evaluate_pssa(&mut pssa, &tok, "a a", EvaluationSlice::default())
                .unwrap_err()
                .contains("vocabulary mismatch")
        );
        assert!(
            evaluate_transformer(&mut transformer, &tok, "a a", EvaluationSlice::default())
                .unwrap_err()
                .contains("vocabulary mismatch")
        );
        let tok = tokenizer();
        let carry = pssa.h_persistent.clone();
        let memory = pssa.memory.clone();
        pssa.unembed_w.data[0] = f32::NAN;
        transformer.unembed.data[0] = f32::NAN;
        assert!(
            evaluate_pssa(&mut pssa, &tok, "a b", EvaluationSlice::default())
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(
            evaluate_transformer(&mut transformer, &tok, "a b", EvaluationSlice::default())
                .unwrap_err()
                .contains("non-finite")
        );
        assert_eq!(pssa.h_persistent, carry);
        assert_eq!(pssa.memory, memory);
    }
}
