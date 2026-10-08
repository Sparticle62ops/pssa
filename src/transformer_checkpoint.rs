//! Separate `TRFM` v1 container: never changes the PSSA V5/V6/V7 wire formats.
//! Uses the same checked tensor readers, checksum and atomic-write machinery.
//! Stores configuration, clock, RNG, tokenizer, schedule horizon, weights,
//! gradients and Adam moments, with an optional u64 warmup tail. Older v1
//! checkpoints without the tail retain an unknown warmup definition. Checkpoints are between chunks; the dataset
//! cursor is explicit `--skip-tokens`, just as for PSSA.
use crate::checkpoint::{self as wire, Reader, Writer, invalid};
use crate::transformer::{TransformerConfig, TransformerModel};
use std::path::Path;

fn validate_warmup(model: &TransformerModel) -> wire::Result<()> {
    if let Some(warmup) = model.lr_schedule_warmup_steps {
        if !model
            .lr_schedule_total_updates
            .is_some_and(|horizon| warmup < horizon)
        {
            return Err(invalid(
                "transformer schedule warmup requires a larger fixed horizon",
            ));
        }
    }
    Ok(())
}

pub fn save_model(model: &TransformerModel, path: impl AsRef<Path>) -> wire::Result<()> {
    validate_warmup(model)?;
    model.cfg.validate().map_err(invalid)?;
    model.tokenizer().map_err(invalid)?;
    if model.step_counter == usize::MAX
        || model
            .lr_schedule_total_updates
            .is_some_and(|n| n == 0 || n < model.step_counter)
    {
        return Err(invalid("invalid transformer step/schedule horizon"));
    }
    let c = &model.cfg;
    let mut w = Writer::new();
    for n in [c.d_vocab, c.d_model, c.n_heads, c.d_ff, c.chunk_len] {
        w.usize(n, "transformer dimension")?;
    }
    for x in [c.lr, c.beta1, c.beta2, c.weight_decay, c.eps] {
        w.f32(x);
    }
    w.usize(model.step_counter, "step")?;
    w.u64(model.rng.state);
    w.usize(
        model.lr_schedule_total_updates.unwrap_or(0),
        "schedule horizon",
    )?;
    wire::write_vocab(&mut w, &model.vocabulary, c.d_vocab)?;
    let json = model.tokenizer_json.as_deref().unwrap_or("");
    w.usize(json.len(), "tokenizer JSON")?;
    w.bytes.extend_from_slice(json.as_bytes());
    for (p, rows, cols) in [
        (&model.token_embed, c.d_vocab, c.d_model),
        (&model.qkv, 3 * c.d_model, c.d_model),
        (&model.out_proj, c.d_model, c.d_model),
        (&model.ff1, c.d_ff, c.d_model),
        (&model.ff2, c.d_model, c.d_ff),
        (&model.unembed, c.d_vocab, c.d_model),
    ] {
        if p.rows != rows || p.cols != cols {
            return Err(invalid("transformer matrix/config shape mismatch"));
        }
        wire::write_matrix(&mut w, p, "transformer matrix")?;
    }
    for p in [
        &model.norm1_gamma,
        &model.norm1_beta,
        &model.norm2_gamma,
        &model.norm2_beta,
    ] {
        if p.data.len() != c.d_model {
            return Err(invalid("transformer norm/config shape mismatch"));
        }
        wire::write_vector(&mut w, p, "transformer norm")?;
    }
    if let Some(warmup) = model.lr_schedule_warmup_steps {
        w.usize(warmup, "schedule warmup")?;
    }
    let mut bytes = Vec::with_capacity(22 + w.bytes.len());
    bytes.extend_from_slice(b"TRFM");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(w.bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&wire::fnv1a64(&w.bytes).to_le_bytes());
    bytes.extend_from_slice(&w.bytes);
    wire::atomic_write(path.as_ref(), &bytes)
}

pub fn load_checkpoint(path: impl AsRef<Path>) -> wire::Result<TransformerModel> {
    let bytes = wire::read_file_capped(path.as_ref())?;
    if bytes.len() < 6 || &bytes[..4] != b"TRFM" || bytes[4..6] != 1u16.to_le_bytes() {
        return Err(invalid(
            "expected a TRFM v1 checkpoint; use train for PSSA checkpoints",
        ));
    }
    let payload = wire::checked_payload(&bytes, "TRFM v1")?;
    let mut r = Reader::new(payload);
    let cfg = TransformerConfig {
        d_vocab: r.usize("vocabulary")?,
        d_model: r.usize("width")?,
        n_heads: r.usize("heads")?,
        d_ff: r.usize("feed-forward width")?,
        chunk_len: r.usize("chunk length")?,
        lr: r.f32("learning rate")?,
        beta1: r.f32("beta1")?,
        beta2: r.f32("beta2")?,
        weight_decay: r.f32("weight decay")?,
        eps: r.f32("optimizer epsilon")?,
    };
    cfg.validate().map_err(invalid)?;
    let step = r.usize("step")?;
    let rng = r.u64("RNG state")?;
    let horizon = r.usize("schedule horizon")?;
    if step == usize::MAX || (horizon != 0 && horizon < step) {
        return Err(invalid("invalid transformer step/schedule horizon"));
    }
    let vocab = wire::read_vocab(&mut r, cfg.d_vocab)?;
    let json_len = r.usize("tokenizer JSON length")?;
    if json_len > 16 * 1024 * 1024 {
        return Err(invalid("tokenizer JSON exceeds 16 MiB cap"));
    }
    let json = if json_len == 0 {
        None
    } else {
        Some(
            std::str::from_utf8(r.take(json_len, "tokenizer JSON")?)
                .map_err(|_| invalid("tokenizer JSON is not UTF-8"))?
                .to_string(),
        )
    };
    // Every parameter must have data, grad, m and v bytes present before model
    // allocation. Shape declarations cannot turn tiny files into huge models.
    let count = 2usize
        .checked_mul(
            cfg.d_vocab
                .checked_mul(cfg.d_model)
                .ok_or_else(|| invalid("transformer parameter element count overflow"))?,
        )
        .and_then(|n| n.checked_add(4usize.checked_mul(cfg.d_model.checked_mul(cfg.d_model)?)?))
        .and_then(|n| n.checked_add(2usize.checked_mul(cfg.d_model.checked_mul(cfg.d_ff)?)?))
        .and_then(|n| n.checked_add(4usize.checked_mul(cfg.d_model)?))
        .ok_or_else(|| invalid("transformer parameter element count overflow"))?;
    let required = count
        .checked_mul(16)
        .and_then(|n| n.checked_add(10 * 4 * 8))
        .ok_or_else(|| invalid("transformer parameter byte count overflow"))?;
    if r.remaining() < required {
        return Err(invalid("truncated transformer parameter tensors"));
    }
    let mut model = TransformerModel::new(cfg, 1).map_err(invalid)?;
    model.step_counter = step;
    model.rng.state = rng;
    model.lr_schedule_total_updates = (horizon != 0).then_some(horizon);
    model.vocabulary = vocab;
    model.tokenizer_json = json;
    model.tokenizer().map_err(invalid)?;
    for p in [
        &mut model.token_embed,
        &mut model.qkv,
        &mut model.out_proj,
        &mut model.ff1,
        &mut model.ff2,
        &mut model.unembed,
    ] {
        wire::read_matrix(&mut r, p, "transformer matrix")?;
    }
    for p in [
        &mut model.norm1_gamma,
        &mut model.norm1_beta,
        &mut model.norm2_gamma,
        &mut model.norm2_beta,
    ] {
        wire::read_vector(&mut r, p, "transformer norm")?;
    }
    if r.remaining() == 8 {
        model.lr_schedule_warmup_steps = Some(r.usize("schedule warmup")?);
    }
    r.done()?;
    validate_warmup(&model)?;
    Ok(model)
}
