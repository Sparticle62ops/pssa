//! Matched-benchmark orchestration only. All forward/backward/optimizer math
//! stays in the unchanged transformer implementation. Existing CLI trainers are
//! not routed through this adapter and retain their original behavior.
use crate::{
    cli::TrainingOptions,
    training::{Schedule, sequence_plan},
    transformer::TransformerModel,
};

/// Same cyclic document window policy as CLIHandler::documents, using the
/// harness's already-tokenized corpus instead of re-tokenizing every chain link.
pub(crate) fn documents_from_encoded(
    encoded: &[Vec<usize>],
    limit: Option<usize>,
    skip: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let nonempty: Vec<&[usize]> = encoded
        .iter()
        .map(Vec::as_slice)
        .filter(|ids| !ids.is_empty())
        .collect();
    let total = nonempty.iter().try_fold(0usize, |sum, ids| {
        sum.checked_add(ids.len())
            .ok_or("dataset token count overflow")
    })?;
    if total < 2 {
        return Err("dataset has no token transitions".into());
    }
    let mut remaining = limit.unwrap_or(total.saturating_sub(skip % total));
    if remaining == 0 {
        return Err("dataset has no token transitions in the selected window".into());
    }
    let mut offset = skip % total;
    let mut doc_index = 0;
    while offset >= nonempty[doc_index].len() {
        offset -= nonempty[doc_index].len();
        doc_index = (doc_index + 1) % nonempty.len();
    }
    let mut docs = Vec::new();
    while remaining > 0 {
        let ids = nonempty[doc_index];
        let take = (ids.len() - offset).min(remaining);
        if take >= 2 {
            docs.push(ids[offset..offset + take].to_vec());
        }
        remaining -= take;
        doc_index = (doc_index + 1) % nonempty.len();
        offset = 0;
        if limit.is_none() && doc_index == 0 {
            break;
        }
    }
    if docs.is_empty() {
        Err("dataset has no token transitions in the selected window".into())
    } else {
        Ok(docs)
    }
}

/// Execute the shared PSSA lane plan serially on the CPU baseline, accumulating
/// the same target-weighted chunks before each update. No hardware parallelism
/// or parameter-count equivalence is implied by this matched-exposure replay.
pub(crate) fn train_documents(
    model: &mut TransformerModel,
    docs: &[Vec<usize>],
    opts: &TrainingOptions,
) -> Result<(), String> {
    let plan = sequence_plan(docs, model.cfg.chunk_len, opts.batch_size)?;
    let schedule = Schedule::new_with_warmup(
        plan.len(),
        model.step_counter,
        model.lr_schedule_total_updates,
        model.lr_schedule_warmup_steps,
        opts,
    )?;
    model.cfg.lr = opts.lr;
    model.lr_schedule_total_updates = schedule.fixed_horizon;
    model.lr_schedule_warmup_steps = schedule.fixed_horizon.map(|_| schedule.warmup);
    let mut curve = opts
        .loss_csv
        .as_deref()
        .map(|p| {
            crate::loss_csv::LossCsv::open(p, opts.loss_every, model.step_counter, opts.tokens_seen)
        })
        .transpose()?;
    let mut progress =
        crate::ui::Progress::new_with_tui("matched transformer", schedule.updates, false);
    progress.set_prior_updates(model.step_counter);
    println!(
        "progress_schema=2 updates_total={} prior_updates={}",
        schedule.updates, model.step_counter
    );
    crate::training::report_sequence_plan(&plan, opts.batch_size);
    let start = std::time::Instant::now();
    let mut update = 0;
    for _ in 0..opts.epochs {
        let mut loss_sum = 0.0f64;
        for group in plan.chunks(opts.accumulate) {
            let prior_loss = loss_sum;
            let targets: usize = group.iter().flatten().map(|c| c.len).sum();
            model.zero_gradients();
            for chunk in group.iter().flatten() {
                let doc = &docs[chunk.doc];
                let begin = chunk.start;
                let len = chunk.len;
                let loss = model.forward_train_chunk(
                    &doc[begin..begin + len],
                    &doc[begin + 1..begin + len + 1],
                );
                if !loss.is_finite() {
                    return Err("non-finite loss; comparison stopped without checkpoint".into());
                }
                model.backward_chunk(len, len as f32 / targets as f32);
                loss_sum += loss as f64 * len as f64;
            }
            update += 1;
            let lr = schedule.lr(update)?;
            model.apply_adamw(lr)?;
            if !model.all_finite() {
                return Err("non-finite parameters; comparison stopped without checkpoint".into());
            }
            let weighted_loss = loss_sum - prior_loss;
            if let Some(curve) = &mut curve {
                curve.record(targets, model.step_counter, weighted_loss)?;
            }
            progress.update_with_metrics(
                update,
                targets,
                weighted_loss / targets as f64,
                Some(lr),
                None,
            );
        }
    }
    progress.finish();
    if let Some(curve) = &mut curve {
        curve.finish()?;
    }
    println!(
        "training_seconds={:.3} optimizer_updates={update}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_windows_match_existing_cli_including_wraps_and_singletons() {
        let raw = "alpha beta gamma\n\ndelta\nepsilon zeta eta theta\n";
        let tok = crate::dataset::Tokenizer::from_corpus(raw, true).unwrap();
        let encoded = raw
            .lines()
            .map(|s| tok.try_encode(s, true).unwrap())
            .collect::<Vec<_>>();
        for skip in 0..20 {
            for limit in [None, Some(0), Some(1), Some(2), Some(7), Some(19)] {
                assert_eq!(
                    documents_from_encoded(&encoded, limit, skip),
                    crate::cli::CLIHandler::documents(raw, &tok, limit, skip)
                );
            }
        }
    }
    #[test]
    fn batch_one_replay_is_bit_identical_to_existing_baseline_trainer() {
        let raw = "alpha beta gamma\ndelta epsilon zeta eta\n";
        let opts = TrainingOptions {
            tokenizer: crate::dataset::TokenizerKind::Word,
            epochs: 1,
            chunk: 2,
            accumulate: 2,
            max_tokens: Some(7),
            batch_size: 1,
            no_tui: true,
            ..Default::default()
        };
        let (expected, tok) = crate::transformer_training::train_corpus(raw, &opts, None).unwrap();
        let docs = crate::cli::CLIHandler::documents(raw, &tok, opts.max_tokens, opts.skip_tokens)
            .unwrap();
        let mut actual = TransformerModel::new(
            crate::transformer::TransformerConfig {
                d_vocab: tok.vocab_size,
                chunk_len: opts.chunk,
                lr: opts.lr,
                ..Default::default()
            },
            opts.seed,
        )
        .unwrap();
        actual.vocabulary = tok.ordered_vocabulary().unwrap();
        actual.tokenizer_json = tok.serialized_metadata();
        train_documents(&mut actual, &docs, &opts).unwrap();
        let dir = std::env::temp_dir().join(format!("oxide-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        crate::transformer_checkpoint::save_model(&actual, dir.join("a.trfm")).unwrap();
        crate::transformer_checkpoint::save_model(&expected, dir.join("b.trfm")).unwrap();
        assert_eq!(
            std::fs::read(dir.join("a.trfm")).unwrap(),
            std::fs::read(dir.join("b.trfm")).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
