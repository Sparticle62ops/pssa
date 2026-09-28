//! Baseline runtime: shares PSSA's tokenizer/window planner, schedule and loss
//! reporting. Attention resets at each chunk; PSSA retains its recurrent carry.
use crate::checkpoint;
use crate::cli::{CLIHandler, TrainingOptions};
use crate::dataset::{DatasetManager, Tokenizer, TokenizerKind};
use crate::training::{Schedule, chunk_plan, report_stream};
use crate::transformer::{TransformerConfig, TransformerModel};
use crate::transformer_checkpoint;
use crate::ui;
use std::time::Instant;

pub fn train_corpus(
    raw: &str,
    opts: &TrainingOptions,
    tokenizer_from: Option<&str>,
) -> Result<(TransformerModel, Tokenizer), String> {
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
    let docs = CLIHandler::documents(raw, &tokenizer, opts.max_tokens, opts.skip_tokens)?;
    let plan = chunk_plan(&docs, model.cfg.chunk_len);
    let schedule = Schedule::new(
        plan.len(),
        model.step_counter,
        model.lr_schedule_total_updates,
        opts,
    )?;
    model.lr_schedule_total_updates = schedule.fixed_horizon;
    println!("backend=cpu (transformer reference baseline)");
    println!(
        "model=transformer parameters={} vocab={}",
        model.parameter_count(),
        model.cfg.d_vocab
    );
    report_stream(&docs, model.cfg.chunk_len, opts.accumulate);
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
    let started = Instant::now();
    let mut update = 0;
    let mut tokens_seen = 0;
    let mut progress = ui::Progress::new("training", schedule.updates);
    for epoch in 0..opts.epochs {
        let mut loss_sum = 0.0f64;
        let mut token_sum = 0usize;
        for group in plan.chunks(opts.accumulate) {
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
            }
            update += 1;
            model.apply_adamw(schedule.lr(update)?)?;
            if !model.all_finite() {
                return Err("non-finite parameters; training aborted without checkpoint".into());
            }
            tokens_seen += total_tokens;
            progress.update(update, total_tokens, loss_sum / token_sum.max(1) as f64);
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
    let (model, _) = train_corpus(&raw, opts, tokenizer_from)?;
    transformer_checkpoint::save_model(&model, out)
        .map_err(|e| format!("cannot save checkpoint '{out}': {e}"))?;
    println!("saved_checkpoint={out}");
    ui::success(&format!("checkpoint written to {}", ui::bold(out)));
    Ok(())
}
