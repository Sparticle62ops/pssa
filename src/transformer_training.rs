//! Baseline runtime: shares PSSA's tokenizer/window planner, schedule and loss
//! reporting. Attention resets at each chunk; PSSA retains its recurrent carry.
use crate::checkpoint;
use crate::cli::{CLIHandler, TrainingOptions};
use crate::dataset::{DatasetManager, Tokenizer, TokenizerKind};
use crate::training::{DatasetFeed, Schedule, chunk_plan, report_stream};
use crate::transformer::{TransformerConfig, TransformerModel};
use crate::transformer_checkpoint;
use crate::ui;
use std::time::Instant;

pub fn train_corpus(
    raw: &str,
    opts: &TrainingOptions,
    tokenizer_from: Option<&str>,
) -> Result<(TransformerModel, Tokenizer), String> {
    let _plain_output = ui::plain_output(opts.no_tui);
    if opts.batch_size != 1 {
        return Err(
            "--batch-size is supported by PSSA train only; use batch size 1 for the transformer"
                .into(),
        );
    }
    if opts.chunk == 0 || opts.max_tokens == Some(0) || opts.epochs == 0 || opts.accumulate == 0 {
        return Err("chunk, max-tokens, epochs and accumulate must be positive".into());
    }
    if !(opts.lr.is_finite() && opts.lr > 0.0) {
        return Err("learning rate must be finite and positive".into());
    }
    if opts.resume.is_some() && tokenizer_from.is_some() {
        return Err("--resume restores its own tokenizer; omit --tokenizer-from".into());
    }
    let (mut model, tokenizer) = if let Some(path) = &opts.resume {
        let model = transformer_checkpoint::load_checkpoint(path)
            .map_err(|e| format!("cannot resume from '{path}': {e}"))?;
        let tok = model.tokenizer()?;
        println!(
            "resumed_from={path} vocab={} d_latent={} prior_steps={}",
            model.cfg.d_vocab, model.cfg.d_model, model.step_counter
        );
        (model, tok)
    } else {
        let tok = if let Some(path) = tokenizer_from {
            // Only tokenizer identity is imported: weights, optimizer state and
            // schedule of the PSSA checkpoint are never used by the baseline.
            let source = checkpoint::load_checkpoint(path)
                .map_err(|e| format!("cannot load tokenizer source '{path}': {e}"))?
                .model;
            let tok = match &source.tokenizer_json {
                Some(json) => Tokenizer::from_serialized(json)?,
                None => Tokenizer::from_vocabulary(&source.vocabulary)?,
            };
            if tok.vocab_size != source.cfg.d_vocab
                || tok.ordered_vocabulary()? != source.vocabulary
            {
                return Err("tokenizer source vocabulary mismatch".into());
            }
            tok
        } else {
            match opts.tokenizer {
                TokenizerKind::Word => Tokenizer::from_corpus(raw, true)?,
                TokenizerKind::Bpe => Tokenizer::from_corpus_bpe(raw, opts.vocab_size)?,
            }
        };
        let cfg = TransformerConfig {
            d_vocab: tok.vocab_size,
            chunk_len: opts.chunk,
            lr: opts.lr,
            ..Default::default()
        };
        let mut model = TransformerModel::new(cfg, opts.seed)?;
        model.vocabulary = tok.ordered_vocabulary()?;
        model.tokenizer_json = tok.serialized_metadata();
        (model, tok)
    };
    model.cfg.lr = opts.lr;
    let window = CLIHandler::training_window(
        raw,
        &tokenizer,
        opts.max_tokens,
        opts.skip_tokens,
        opts.token_cache.as_deref().map(std::path::Path::new),
        opts.token_cache_source.as_deref().map(std::path::Path::new),
    )?;
    let docs = &window.docs;
    let plan = chunk_plan(docs, model.cfg.chunk_len);
    let mut feed = opts
        .feed_telemetry
        .then(|| DatasetFeed::new(raw, &tokenizer, &window, opts, plan.len()));
    let schedule = Schedule::new_with_warmup(
        plan.len(),
        model.step_counter,
        model.lr_schedule_total_updates,
        model.lr_schedule_warmup_steps,
        opts,
    )?;
    model.lr_schedule_total_updates = schedule.fixed_horizon;
    model.lr_schedule_warmup_steps = schedule.fixed_horizon.map(|_| schedule.warmup);
    println!("backend=cpu (transformer reference baseline)");
    println!(
        "model=transformer parameters={} vocab={}",
        model.parameter_count(),
        model.cfg.d_vocab
    );
    report_stream(docs, model.cfg.chunk_len, opts.accumulate);
    ui::banner("train-transformer", "decoder-only transformer baseline");
    ui::field(
        "corpus",
        &format!("{} tokens", ui::thousands(docs.iter().map(Vec::len).sum())),
    );
    ui::field("vocabulary", &ui::thousands(model.cfg.d_vocab));
    ui::field(
        "width",
        &format!(
            "{} / heads {} / feed-forward {} / layers 1",
            model.cfg.d_model, model.cfg.n_heads, model.cfg.d_ff
        ),
    );
    ui::field(
        "schedule",
        &format!(
            "{} epoch(s), {} updates, lr first={:.8} last={:.8} (base {:.8}, horizon {})",
            opts.epochs,
            ui::thousands(schedule.updates),
            schedule.lr(1)?,
            schedule.lr(schedule.updates)?,
            opts.lr,
            ui::thousands(schedule.total)
        ),
    );
    println!();
    let mut curve = opts
        .loss_csv
        .as_deref()
        .map(|path| {
            crate::loss_csv::LossCsv::open(
                path,
                opts.loss_every,
                model.step_counter,
                opts.tokens_seen,
            )
        })
        .transpose()?;
    let started = Instant::now();
    let mut update = 0;
    let mut tokens_seen = 0;
    let mut progress = ui::Progress::new_with_tui("training", schedule.updates, !opts.no_tui);
    if let Some(path) = opts.checkpoint_path.as_deref() {
        progress.set_checkpoint_path(path);
    }
    if let Some(path) = opts.resume.as_deref() {
        progress.set_last_checkpoint(path);
    }
    progress.set_prior_updates(model.step_counter.saturating_sub(update));
    println!(
        "progress_schema=2 updates_total={} prior_updates={} checkpoint_target={}",
        schedule.updates,
        model.step_counter,
        opts.checkpoint_path.as_deref().unwrap_or("-")
    );
    println!("last_checkpoint={}", opts.resume.as_deref().unwrap_or("-"));
    for epoch in 0..opts.epochs {
        if let Some(feed) = &mut feed {
            feed.begin_epoch(epoch + 1);
        }
        let mut loss_sum = 0.0f64;
        let mut token_sum = 0usize;
        for group in plan.chunks(opts.accumulate) {
            let prior_loss = loss_sum;
            let total_tokens: usize = group.iter().map(|x| x.2).sum();
            model.zero_gradients();
            for &(doc, start, len) in group {
                let loss = model.forward_train_chunk(
                    &docs[doc][start..start + len],
                    &docs[doc][start + 1..start + 1 + len],
                );
                if !loss.is_finite() {
                    return Err("non-finite loss; training aborted without checkpoint".into());
                }
                model.backward_chunk(len, len as f32 / total_tokens as f32);
                loss_sum += loss as f64 * len as f64;
                token_sum += len;
                if let Some(feed) = &mut feed {
                    feed.consume(doc, start, len);
                    feed.finish_batch();
                }
            }
            update += 1;
            let learning_rate = schedule.lr(update)?;
            model.apply_adamw(learning_rate)?;
            if !model.all_finite() {
                return Err("non-finite parameters; training aborted without checkpoint".into());
            }
            tokens_seen += total_tokens;
            if let Some(curve) = &mut curve {
                curve.record(total_tokens, model.step_counter, loss_sum - prior_loss)?;
            }
            let update_loss = (loss_sum - prior_loss) / total_tokens.max(1) as f64;
            if let Some(feed) = &mut feed {
                progress.update_with_feed(
                    update,
                    total_tokens,
                    update_loss,
                    Some(learning_rate),
                    None,
                    || feed.sample(model.step_counter),
                );
            } else {
                progress.update_with_metrics(
                    update,
                    total_tokens,
                    update_loss,
                    Some(learning_rate),
                    None,
                );
            }
        }
        progress.finish();
        println!(
            "epoch {}/{} loss={:.6} tokens={} updates={}",
            epoch + 1,
            opts.epochs,
            loss_sum / token_sum.max(1) as f64,
            token_sum,
            update
        );
    }
    progress.finish();
    if let Some(curve) = &mut curve {
        curve.finish()?;
    }
    let wall = started.elapsed().as_secs_f64();
    println!(
        "training_seconds={:.3} optimizer_updates={update}",
        started.elapsed().as_secs_f32()
    );
    ui::section("summary");
    ui::field("wall time", &ui::duration(wall));
    ui::field("tokens", &ui::thousands(tokens_seen));
    ui::field(
        "throughput",
        &format!(
            "{:.0} tokens/second",
            if wall > 0.0 {
                tokens_seen as f64 / wall
            } else {
                0.0
            }
        ),
    );
    ui::field("updates", &ui::thousands(update));
    println!();
    Ok((model, tokenizer))
}

pub fn run_training(
    data: &str,
    opts: &TrainingOptions,
    out: &str,
    tokenizer_from: Option<&str>,
) -> Result<(), String> {
    let raw = DatasetManager::try_load_dataset(Some(data))?;
    let mut run_options = opts.clone();
    run_options.checkpoint_path = Some(out.to_string());
    run_options.dataset_source = Some(data.to_string());
    if run_options.token_cache.is_some() && run_options.token_cache_source.is_none() {
        run_options.token_cache_source = CLIHandler::local_cache_source(data);
    }
    let (model, _) = train_corpus(&raw, &run_options, tokenizer_from)?;
    transformer_checkpoint::save_model(&model, out)
        .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
    ui::checkpoint_saved(out);
    ui::success(&format!("checkpoint written to {}", ui::bold(out)));
    Ok(())
}
