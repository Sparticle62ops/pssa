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
