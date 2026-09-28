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
        std::iter::once(batch.len()).chain(batch.iter().flat_map(|c| [c.lane, c.doc, c.start, c.len]))
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
            if prior_steps > 0 { 0 } else { opts.warmup_steps }
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
