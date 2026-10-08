//! Shared data plan and optimizer schedule for PSSA and the transformer.
use crate::cli::{TrainingOptions, learning_rate_for_update};

pub fn chunk_plan(docs: &[Vec<usize>], chunk_len: usize) -> Vec<(usize, usize, usize)> {
    assert!(chunk_len > 0);
    let mut plan = Vec::new();
    for (doc_id, doc) in docs.iter().enumerate() {
        let mut start = 0;
        while start + 1 < doc.len() {
            let len = chunk_len.min(doc.len() - 1 - start);
            plan.push((doc_id, start, len));
            start += len;
        }
    }
    plan
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceChunk {
    pub lane: usize,
    pub doc: usize,
    pub start: usize,
    pub len: usize,
}

/// Advance one TBPTT chunk per document lane. Finished lanes take the next
/// document on the next microbatch; consecutive chunks never run concurrently.
/// A one-document corpus therefore cannot exploit sequence batching.
pub fn sequence_plan(
    docs: &[Vec<usize>],
    chunk: usize,
    batch_size: usize,
) -> Result<Vec<Vec<SequenceChunk>>, String> {
    if chunk == 0 || batch_size == 0 {
        return Err("chunk and batch size must be positive".into());
    }
    let mut lanes = vec![None; batch_size.min(docs.len())];
    let mut next_doc = 0;
    let mut plan = Vec::new();
    loop {
        let mut batch = Vec::new();
        for (lane, cursor) in lanes.iter_mut().enumerate() {
            if cursor.is_none() {
                while next_doc < docs.len() && docs[next_doc].len() < 2 {
                    next_doc += 1;
                }
                if next_doc < docs.len() {
                    *cursor = Some((next_doc, 0));
                    next_doc += 1;
                }
            }
            if let Some((doc, start)) = *cursor {
                let len = chunk.min(docs[doc].len() - 1 - start);
                batch.push(SequenceChunk {
                    lane,
                    doc,
                    start,
                    len,
                });
                *cursor = if start + len + 1 < docs[doc].len() {
                    Some((doc, start + len))
                } else {
                    None
                };
            }
        }
        if batch.is_empty() {
            break;
        }
        plan.push(batch);
    }
    Ok(plan)
}

/// Additional grouping fingerprint: leave the legacy stream line unchanged.
pub fn report_sequence_plan(plan: &[Vec<SequenceChunk>], batch_size: usize) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for n in std::iter::once(batch_size).chain(plan.iter().flat_map(|batch| {
        std::iter::once(batch.len())
            .chain(batch.iter().flat_map(|c| [c.lane, c.doc, c.start, c.len]))
    })) {
        for b in (n as u64).to_le_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!(
        "sequence_plan_fnv1a64={hash:016x} batch_size={batch_size} microbatches={} memory_writes=after_batch",
        plan.len()
    );
}

/// An audit fingerprint of the actual selected IDs, document boundaries and
/// update grouping. This is a reproducibility check, not a security digest.
pub fn report_stream(docs: &[Vec<usize>], chunk: usize, accumulate: usize) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for n in [chunk, accumulate].into_iter().chain(
        docs.iter()
            .flat_map(|d| std::iter::once(d.len()).chain(d.iter().copied())),
    ) {
        for b in (n as u64).to_le_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("token_stream_fnv1a64={hash:016x} chunk={chunk} accumulate={accumulate}");
}

/// Source views and counters are selected-document-sized. No corpus is
/// re-tokenized for telemetry, and only `sample` (behind Progress::should_emit)
/// allocates per-event strings. This never owns or mutates the training plan.
struct FeedDocument<'a> {
    text: Option<&'a str>,
    token_start: usize,
    source_token_start: usize,
    source_row: Option<usize>,
    byte_start: Option<usize>,
    next_byte: Option<usize>,
    stream_start: usize,
}

pub(crate) struct DatasetFeed<'a> {
    tokenizer: &'a crate::dataset::Tokenizer,
    docs: &'a [Vec<usize>],
    sources: Vec<FeedDocument<'a>>,
    opts: &'a TrainingOptions,
    epoch_total: usize,
    bytes_total: Option<usize>,
    tokens: usize,
    epoch_tokens: usize,
    bytes: Option<usize>,
    epoch: usize,
    batch: usize,
    batches: usize,
    latest: Option<(usize, usize, usize)>,
}

impl<'a> DatasetFeed<'a> {
    pub(crate) fn new(
        raw: &'a str,
        tokenizer: &'a crate::dataset::Tokenizer,
        window: &'a crate::token_cache::WindowResult,
        opts: &'a TrainingOptions,
        batches: usize,
    ) -> Self {
        let mut stream_start = 0;
        let mut sources: Vec<_> = window
            .selections
            .iter()
            .zip(&window.docs)
            .map(|(selection, doc)| {
                let source = FeedDocument {
                    text: None,
                    token_start: selection.token_start,
                    source_token_start: selection.source_token_start,
                    source_row: None,
                    byte_start: selection.byte_start,
                    next_byte: selection.byte_start,
                    stream_start,
                };
                stream_start += doc.len();
                source
            })
            .collect();
        // Locate only selected source rows, including duplicates on cyclic
        // wrap. Cached and uncached windows share the same nonempty-row index.
        // The scan creates no full-corpus map and does not encode any text.
        let mut selected: Vec<_> = (0..sources.len()).collect();
        selected.sort_unstable_by_key(|&i| window.selections[i].source_doc);
        let mut wanted = selected.into_iter().peekable();
        for (row, (source_row, text)) in raw
            .lines()
            .enumerate()
            .filter(|(_, text)| tokenizer.source_has_tokens(text))
            .enumerate()
        {
            while let Some(&doc) = wanted.peek() {
                if window.selections[doc].source_doc != row {
                    break;
                }
                sources[doc].text = Some(text);
                sources[doc].source_row = Some(source_row + 1);
                wanted.next();
            }
            if wanted.peek().is_none() {
                break;
            }
        }
        let mut epoch_bytes = Some(0usize);
        for (source, doc) in sources.iter_mut().zip(&window.docs) {
            // Claim BPE raw offsets/bytes only after verifying the cached IDs
            // against the supplied source, without a second tokenization pass.
            let cursor = source.byte_start.and_then(|start| {
                doc.iter().try_fold(start, |start, &id| {
                    let bytes = tokenizer.token_bytes(id)?;
                    let end = start.checked_add(bytes.len())?;
                    (source.text?.as_bytes().get(start..end)? == bytes).then_some(end)
                })
            });
            if cursor.is_none() {
                source.byte_start = None;
                source.next_byte = None;
                epoch_bytes = None;
            } else {
                let input_bytes = doc[..doc.len() - 1].iter().try_fold(0usize, |sum, &id| {
                    sum.checked_add(tokenizer.token_bytes(id)?.len())
                });
                epoch_bytes = epoch_bytes.and_then(|sum| sum.checked_add(input_bytes?));
            }
        }
        let bytes_total = epoch_bytes.and_then(|bytes| bytes.checked_mul(opts.epochs));
        Self {
            tokenizer,
            docs: &window.docs,
            sources,
            opts,
            epoch_total: window.docs.iter().map(|doc| doc.len() - 1).sum(),
            bytes_total,
            tokens: 0,
            epoch_tokens: 0,
            bytes: bytes_total.map(|_| 0),
            epoch: 0,
            batch: 0,
            batches,
            latest: None,
        }
    }

    pub(crate) fn begin_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.epoch_tokens = 0;
        self.batch = 0;
        self.latest = None;
        for source in &mut self.sources {
            source.next_byte = source.byte_start;
        }
    }

    /// Called in existing chunk completion order, even if a gradient safeguard
    /// subsequently skips Adam. Inputs were still consumed in that case.
    pub(crate) fn consume(&mut self, doc: usize, start: usize, len: usize) {
        self.tokens += len;
        self.epoch_tokens += len;
        self.latest = Some((doc, start, start + len));
        let source = &mut self.sources[doc];
        if source.next_byte.is_some() {
            let byte_count: usize = self.docs[doc][start..start + len]
                .iter()
                .map(|&id| self.tokenizer.token_bytes(id).unwrap().len())
                .sum();
            self.bytes = self.bytes.and_then(|bytes| bytes.checked_add(byte_count));
            source.next_byte = source
                .next_byte
                .and_then(|offset| offset.checked_add(byte_count));
        }
    }

    pub(crate) fn finish_batch(&mut self) {
        self.batch += 1;
    }

    pub(crate) fn inputs_complete(&self) -> bool {
        self.epoch == self.opts.epochs && self.epoch_tokens == self.epoch_total
    }

    pub(crate) fn sample(&self, step: usize) -> crate::ui::FeedSample {
        let (doc, chunk_start, end) = self.latest.expect("a completed input chunk");
        let start = end.saturating_sub(16).max(chunk_start);
        let ids = &self.docs[doc][start..end];
        let source = &self.sources[doc];
        let raw = source.text.and_then(|text| {
            let range = match self.tokenizer.kind() {
                crate::dataset::TokenizerKind::Word => crate::dataset::Tokenizer::word_source_span(
                    text,
                    source.token_start + start,
                    source.token_start + end,
                )?,
                crate::dataset::TokenizerKind::Bpe => {
                    let byte_end = source.next_byte?;
                    let count: usize = ids
                        .iter()
                        .map(|&id| self.tokenizer.token_bytes(id).unwrap().len())
                        .sum();
                    let mut byte_start = byte_end.checked_sub(count)?;
                    let mut byte_end = byte_end;
                    // A byte token can bisect UTF-8. Include its containing
                    // source character, never replacement-character decoding.
                    while byte_start > 0 && !text.is_char_boundary(byte_start) {
                        byte_start -= 1;
                    }
                    while byte_end < text.len() && !text.is_char_boundary(byte_end) {
                        byte_end += 1;
                    }
                    byte_start..byte_end
                }
            };
            text.get(range)
        });
        let (snippet, text_kind) = match raw {
            Some(text) => (text.chars().take(256).collect(), "raw"),
            None => (
                self.tokenizer.decode(ids).chars().take(256).collect(),
                "decoded",
            ),
        };
        let token_ids = ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        // A word vocabulary entry can be arbitrarily long. Bound each label
        // before JSON/percent encoding so one input cannot flood the log or
        // exceed the reader's field limit. IDs remain exact; ellipsis is explicit.
        let pieces: Vec<_> = ids
            .iter()
            .map(|id| {
                let mut chars = self.tokenizer.id_to_token[id].chars();
                let mut piece: String = chars.by_ref().take(32).collect();
                if chars.next().is_some() {
                    piece.push('…');
                }
                piece
            })
            .collect();
        let hf = self.opts.hf_dataset.is_some();
        crate::ui::FeedSample {
            dataset: self
                .opts
                .hf_dataset
                .as_deref()
                .or(self.opts.dataset_source.as_deref())
                .unwrap_or("in-memory corpus")
                .to_string(),
            config: hf.then(|| self.opts.hf_config.clone()).flatten(),
            split: hf.then(|| self.opts.hf_split.clone()),
            field: hf.then(|| self.opts.hf_field.clone()),
            snippet,
            token_ids,
            token_pieces: serde_json::to_string(&pieces).expect("string array is serializable"),
            tokens: self.tokens,
            epoch_tokens: self.epoch_tokens,
            epoch_total: self.epoch_total,
            epoch: self.epoch,
            epochs: self.opts.epochs,
            step,
            batch: self.batch,
            batches: self.batches,
            row: doc + 1,
            rows: self.docs.len(),
            start: source.stream_start + start,
            end: source.stream_start + end,
            source_row: source.source_row,
            source_start: source.source_token_start + start,
            source_end: source.source_token_start + end,
            skip_tokens: self.opts.skip_tokens,
            bytes: self.bytes,
            bytes_total: self.bytes_total,
            text_kind,
        }
    }
}

#[cfg(test)]
mod feed_tests {
    use super::*;
    use crate::dataset::Tokenizer;
    use crate::token_cache;

    #[test]
    fn selected_word_preview_is_raw_not_lowercase_decode_and_excludes_target() {
        let raw = "#\nUP:PER,  MiXeD\tİSTANBUL! LAST\nSecond\tROW ends\n";
        let tokenizer = Tokenizer::from_corpus(raw, true).unwrap();
        let opts = TrainingOptions {
            epochs: 2,
            dataset_source: Some("local:fixture.txt".into()),
            skip_tokens: 1,
            ..Default::default()
        };
        // Skip UP:PER, retain five tokens, discard a one-token next-row tail.
        let window = token_cache::documents(raw, &tokenizer, Some(6), 1, None, None).unwrap();
        assert_eq!(window.docs.len(), 1);
        let mut feed = DatasetFeed::new(raw, &tokenizer, &window, &opts, 1);
        for epoch in 1..=2 {
            feed.begin_epoch(epoch);
            feed.consume(0, 0, 4);
            feed.finish_batch();
            let sample = feed.sample(10 + epoch);
            assert_eq!(sample.dataset, "local:fixture.txt");
            assert_eq!(sample.snippet, ",  MiXeD\tİSTANBUL!");
            assert_eq!(sample.text_kind, "raw");
            assert_eq!(
                (sample.tokens, sample.epoch_tokens, sample.epoch_total),
                (4 * epoch, 4, 4)
            );
            assert_eq!(
                (sample.epoch, sample.epochs, sample.batch, sample.batches),
                (epoch, 2, 1, 1)
            );
            assert_eq!(
                (sample.row, sample.rows, sample.start, sample.end),
                (1, 1, 0, 4)
            );
            assert_eq!(sample.step, 10 + epoch);
            assert_eq!(sample.source_row, Some(2));
            assert_eq!(
                (sample.source_start, sample.source_end, sample.skip_tokens),
                (1, 5, 1)
            );
            assert_eq!(
                sample.bytes, None,
                "normalized word IDs cannot establish exact source byte consumption"
            );
            assert_eq!(sample.bytes_total, None);
            let pieces: Vec<String> = serde_json::from_str(&sample.token_pieces).unwrap();
            let ids: Vec<usize> = sample
                .token_ids
                .split(',')
                .map(|id| id.parse().unwrap())
                .collect();
            assert_eq!(ids, window.docs[0][..4]);
            assert_eq!(
                pieces,
                ids.iter()
                    .map(|id| tokenizer.id_to_token[id].clone())
                    .collect::<Vec<_>>()
            );
            assert!(!pieces.iter().any(|piece| piece == "last"));
        }
    }

    #[test]
    fn byte_bpe_preview_covers_split_utf8_without_lossy_decoding() {
        let raw = "é😀\n";
        let tokenizer = Tokenizer::from_corpus_bpe(raw, 257).unwrap();
        let opts = TrainingOptions {
            epochs: 2,
            ..Default::default()
        };
        let window = token_cache::documents(raw, &tokenizer, Some(4), 1, None, None).unwrap();
        assert_eq!(window.docs[0].len(), 4);
        let mut feed = DatasetFeed::new(raw, &tokenizer, &window, &opts, 2);
        for epoch in 1..=2 {
            feed.begin_epoch(epoch);
            feed.consume(0, 0, 2);
            feed.finish_batch();
            let sample = feed.sample(epoch * 2 - 1);
            assert_eq!(sample.snippet, "é😀");
            assert_eq!(sample.text_kind, "raw");
            assert_eq!(sample.bytes, Some((epoch - 1) * 3 + 2));
            assert_eq!(sample.bytes_total, Some(6));
            let pieces: Vec<String> = serde_json::from_str(&sample.token_pieces).unwrap();
            assert_eq!(
                pieces,
                window.docs[0][..2]
                    .iter()
                    .map(|id| tokenizer.id_to_token[id].clone())
                    .collect::<Vec<_>>()
            );
            feed.consume(0, 2, 1);
            feed.finish_batch();
            let sample = feed.sample(epoch * 2);
            assert_eq!(sample.snippet, "😀");
            assert_eq!(sample.bytes, Some(epoch * 3));
            assert_eq!((sample.start, sample.end), (2, 3));
        }
    }

    #[test]
    fn cache_and_lazy_word_bpe_windows_share_exact_samples_after_skips_wraps_and_epochs() {
        let raw = "#\nAlpha BETA gamma\n\nδelta  EPSILON!\nSolo\nfinal\tROW here\n";
        for tokenizer in [
            Tokenizer::from_corpus(raw, true).unwrap(),
            Tokenizer::from_corpus_bpe(raw, 280).unwrap(),
        ] {
            let total: usize = raw
                .lines()
                .map(|line| tokenizer.encode(line, true).len())
                .sum();
            let path = std::env::temp_dir().join(format!(
                "pssa-feed-{}-{:?}.pssatok",
                std::process::id(),
                tokenizer.kind()
            ));
            let _ = std::fs::remove_file(&path);
            let opts = TrainingOptions {
                epochs: 2,
                hf_dataset: Some("fixture/corpus".into()),
                hf_config: Some("tiny".into()),
                ..Default::default()
            };
            for (skip, limit) in [
                (1, Some(total + 5)),
                (total + 2, Some(total + 3)),
                (total - 1, Some(9)),
                (2, None),
            ] {
                let lazy =
                    token_cache::documents(raw, &tokenizer, limit, skip, None, None).unwrap();
                let cached =
                    token_cache::documents(raw, &tokenizer, limit, skip, Some(&path), None)
                        .unwrap();
                let reused =
                    token_cache::documents(raw, &tokenizer, limit, skip, Some(&path), None)
                        .unwrap();
                assert_eq!(reused.status, token_cache::CacheStatus::Reused);
                assert_eq!(lazy.docs, cached.docs);
                assert_eq!(lazy.docs, reused.docs);
                let plan = sequence_plan(&lazy.docs, 3, 2).unwrap();
                let mut feeds: Vec<_> = [&lazy, &cached, &reused]
                    .into_iter()
                    .map(|window| DatasetFeed::new(raw, &tokenizer, window, &opts, plan.len()))
                    .collect();
                let epoch_tokens: usize = plan.iter().flatten().map(|c| c.len).sum();
                for epoch in 1..=2 {
                    for feed in &mut feeds {
                        feed.begin_epoch(epoch);
                    }
                    for (batch, chunks) in plan.iter().enumerate() {
                        for feed in &mut feeds {
                            for c in chunks {
                                feed.consume(c.doc, c.start, c.len);
                            }
                            feed.finish_batch();
                        }
                        let samples: Vec<_> =
                            feeds.iter().map(|feed| feed.sample(batch + 1)).collect();
                        assert_eq!(samples[0], samples[1], "skip={skip} limit={limit:?}");
                        assert_eq!(samples[0], samples[2]);
                        let sample = &samples[0];
                        let c = chunks.last().unwrap();
                        let end = c.start + c.len;
                        let expected = &lazy.docs[c.doc][end.saturating_sub(16).max(c.start)..end];
                        let actual: Vec<usize> = sample
                            .token_ids
                            .split(',')
                            .map(|id| id.parse().unwrap())
                            .collect();
                        assert_eq!(actual, expected);
                        assert_eq!(sample.text_kind, "raw");
                        assert_eq!(sample.dataset, "fixture/corpus");
                        assert_eq!(sample.config.as_deref(), Some("tiny"));
                        assert_eq!(sample.row, c.doc + 1);
                        assert_eq!(sample.rows, lazy.docs.len());
                        assert_eq!(sample.batch, batch + 1);
                        assert_eq!(sample.epoch_total, epoch_tokens);
                        let selection = lazy.selections[c.doc];
                        assert_eq!(sample.source_start, selection.source_token_start + c.start);
                        assert_eq!(sample.source_end, selection.source_token_start + end);
                        let source_line = raw.lines().nth(sample.source_row.unwrap() - 1).unwrap();
                        assert!(source_line.contains(&sample.snippet));
                    }
                    let last = feeds[0].sample(plan.len());
                    assert_eq!(last.epoch_tokens, epoch_tokens);
                    assert_eq!(last.tokens, epoch * epoch_tokens);
                }
            }
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn preview_tail_is_at_most_sixteen_actual_inputs_and_mismatch_is_decoded() {
        let raw = (0..40)
            .map(|i| format!("WORD{i}"))
            .collect::<Vec<_>>()
            .join("  ");
        for tokenizer in [
            Tokenizer::from_corpus(&raw, true).unwrap(),
            Tokenizer::from_corpus_bpe(&raw, 257).unwrap(),
        ] {
            let opts = TrainingOptions::default();
            let window = token_cache::documents(&raw, &tokenizer, None, 0, None, None).unwrap();
            let mut feed = DatasetFeed::new(&raw, &tokenizer, &window, &opts, 1);
            feed.begin_epoch(1);
            feed.consume(0, 0, 32);
            feed.finish_batch();
            let sample = feed.sample(1);
            assert_eq!((sample.start, sample.end), (16, 32));
            let ids: Vec<usize> = sample
                .token_ids
                .split(',')
                .map(|id| id.parse().unwrap())
                .collect();
            assert_eq!(ids, window.docs[0][16..32]);
            assert_eq!(ids.len(), 16);
            assert_eq!(sample.text_kind, "raw");
            if tokenizer.kind() == crate::dataset::TokenizerKind::Word {
                assert_eq!(
                    sample.snippet,
                    (16..32)
                        .map(|i| format!("WORD{i}"))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
            } else {
                assert_eq!(sample.snippet, raw[16..32]);
                let changed = raw.replace('W', "X");
                let mut feed = DatasetFeed::new(&changed, &tokenizer, &window, &opts, 1);
                feed.begin_epoch(1);
                feed.consume(0, 0, 32);
                feed.finish_batch();
                let sample = feed.sample(1);
                assert_eq!(sample.text_kind, "decoded");
                assert_eq!(sample.bytes, None);
                assert_eq!(sample.bytes_total, None);
            }
        }
    }

    #[test]
    fn sixteen_oversized_unicode_pieces_stay_bounded_without_changing_ids() {
        let raw = std::iter::repeat_n("界".repeat(400), 17)
            .collect::<Vec<_>>()
            .join(" ");
        let tokenizer = Tokenizer::from_corpus(&raw, true).unwrap();
        let opts = TrainingOptions::default();
        let window = token_cache::documents(&raw, &tokenizer, None, 0, None, None).unwrap();
        let mut feed = DatasetFeed::new(&raw, &tokenizer, &window, &opts, 1);
        feed.begin_epoch(1);
        feed.consume(0, 0, 16);
        feed.finish_batch();
        let sample = feed.sample(1);
        let pieces: Vec<String> = serde_json::from_str(&sample.token_pieces).unwrap();
        assert_eq!(pieces.len(), 16);
        assert!(
            pieces
                .iter()
                .all(|piece| piece == &format!("{}…", "界".repeat(32)))
        );
        assert!(
            sample.token_pieces.len() < 4096,
            "reader can accept every emitted label array"
        );
        assert_eq!(sample.token_ids.split(',').count(), 16);
        assert_eq!(sample.snippet.chars().count(), 256);
        assert_eq!(
            window.docs[0],
            tokenizer.encode(&raw, true),
            "telemetry cannot mutate inputs"
        );
    }

    #[test]
    fn preview_is_bounded_and_fallback_is_explicit() {
        let raw = format!("{} tail final", "A".repeat(400));
        let tokenizer = Tokenizer::from_corpus(&raw, true).unwrap();
        let opts = TrainingOptions::default();
        let window = token_cache::documents(&raw, &tokenizer, None, 0, None, None).unwrap();
        let mut feed = DatasetFeed::new(&raw, &tokenizer, &window, &opts, 1);
        feed.begin_epoch(1);
        feed.consume(0, 0, 2);
        feed.finish_batch();
        let sample = feed.sample(1);
        assert_eq!(sample.snippet.chars().count(), 256);
        let pieces: Vec<String> = serde_json::from_str(&sample.token_pieces).unwrap();
        assert_eq!(pieces[0], format!("{}…", "a".repeat(32)));
        assert!(sample.token_pieces.len() < 4096);
        assert_eq!(sample.dataset, "in-memory corpus");
        assert_eq!(sample.text_kind, "raw");
        feed.sources[0].text = None;
        let sample = feed.sample(1);
        assert_eq!(sample.text_kind, "decoded");
        assert_eq!(sample.snippet.chars().count(), 256);
    }
}

pub struct Schedule {
    pub updates: usize,
    pub prior_steps: usize,
    pub total: usize,
    pub warmup: usize,
    pub fixed_horizon: Option<usize>,
    pub base_lr: f32,
}

impl Schedule {
    pub fn new(
        chunks: usize,
        prior_steps: usize,
        stored: Option<usize>,
        opts: &TrainingOptions,
    ) -> Result<Self, String> {
        Self::new_with_warmup(chunks, prior_steps, stored, None, opts)
    }

    /// Restore the complete fixed schedule at the global optimizer step. Older
    /// checkpoints lack warmup metadata and retain their no-rewarmup behavior.
    pub fn new_with_warmup(
        chunks: usize,
        prior_steps: usize,
        stored: Option<usize>,
        stored_warmup: Option<usize>,
        opts: &TrainingOptions,
    ) -> Result<Self, String> {
        if stored_warmup.is_some() && stored.is_none() {
            return Err("stored schedule warmup requires a fixed horizon".into());
        }
        if opts.epochs == 0 || opts.accumulate == 0 {
            return Err("epochs and accumulate must be positive".into());
        }
        if let (Some(requested), Some(stored)) = (opts.schedule_total_updates, stored)
            && requested != stored
        {
            return Err(format!(
                "--total-updates={requested} does not match resume checkpoint horizon {stored}"
            ));
        }
        let updates = chunks
            .div_ceil(opts.accumulate)
            .checked_mul(opts.epochs)
            .ok_or("training update count overflow")?;
        if updates == 0 {
            return Err("dataset has no training chunks".into());
        }
        let end = prior_steps
            .checked_add(updates)
            .ok_or("training update count overflow")?;
        let fixed_horizon = stored.or(opts.schedule_total_updates);
        let total = fixed_horizon.unwrap_or(end);
        if total < end {
            return Err(format!(
                "learning-rate schedule horizon ({total}) must reach link end ({end}); set --total-updates on the fresh run to the whole-chain update count"
            ));
        }
        // Warmup also defines the cosine phase after the warmup interval.
        // Evaluating it at the global step continues, rather than restarts, it.
        // With no stored definition, preserve legacy resume behavior.
        let warmup = stored_warmup.unwrap_or_else(|| {
            if prior_steps > 0 {
                0
            } else {
                opts.warmup_steps
            }
        });
        if warmup >= total && warmup != 0 {
            return Err(format!(
                "--warmup-steps ({warmup}) must be less than total optimizer updates ({total})"
            ));
        }
        if let Some(horizon) = fixed_horizon {
            println!(
                "lr_schedule=fixed horizon={horizon} from_step={prior_steps} to_step={end} warmup={warmup}{}",
                if prior_steps > 0 {
                    " (restored at global step, no warmup restart)"
                } else {
                    ""
                }
            );
        } else if prior_steps > 0 {
            println!(
                "lr_schedule=continued from_step={prior_steps} to_step={total} (legacy per-link horizon, no restart, no re-warmup)"
            );
        } else {
            println!(
                "lr_schedule=per-run horizon={total} from_step=0 to_step={end} warmup={warmup}"
            );
        }
        let schedule = Self {
            updates,
            prior_steps,
            total,
            warmup,
            fixed_horizon,
            base_lr: opts.lr,
        };
        schedule.lr(1)?;
        Ok(schedule)
    }

    pub fn lr(&self, link_update: usize) -> Result<f32, String> {
        learning_rate_for_update(
            self.base_lr,
            self.prior_steps
                .checked_add(link_update)
                .ok_or("optimizer step overflow")?,
            self.total,
            self.warmup,
        )
    }
}
