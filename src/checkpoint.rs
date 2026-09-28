//! Versioned PSSA checkpoint container.
//!
//! **V7 wire format (all integer and float words little-endian):**
//! `b"PSSA" | version:u16(7) | payload_len:u64 | fnv1a64(payload):u64 |
//! payload`. V7 writes the exact complete V6 payload, followed by
//! `tokenizer_json_len:u64 | tokenizer_json:utf8[byte_len]`, and optionally a
//! `lr_schedule_total_updates:u64` tail when a fixed schedule horizon is set,
//! followed optionally by `lr_schedule_warmup_steps:u64`. Older horizon-only
//! checkpoints retain an unknown (`None`) warmup definition.
//! The optional JSON is zero-length only for the legacy word tokenizer. Shapes
//! are derived from the preceding configuration and every declared length is
//! checked. The checksum is accidental-corruption detection only, not
//! authentication. V7 checkpoints without the optional schedule tail remain
//! readable and use the legacy schedule behavior.
//!
//! Scratch and activation tape buffers are intentionally not serialized. V7 and
//! V6 checkpoints are supported between chunks (after backward or an optimizer
//! update) and resume at the next forward call; they do not resume an in-flight
//! backward pass or an external dataset cursor.

use crate::linalg::SimpleRng;
use crate::pssa::{PSSAConfigV2, PSSALayerV2, ParamMatrix, ParamVector};
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FORMAT_VERSION: u16 = 7;
pub const V6_FORMAT_VERSION: u16 = 6;
const HEADER_LEN: usize = 22;
const MAX_TOKENIZER_JSON_BYTES: usize = 16 * 1024 * 1024;
const HEADER_V5_LEN: usize = 38;
/// Limits model data plus tape/scratch allocation, before constructing a model.
pub const MAX_LOAD_ALLOCATION_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    Invalid(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "checkpoint I/O error: {e}"),
            Self::Invalid(e) => write!(f, "invalid checkpoint: {e}"),
        }
    }
}
impl std::error::Error for CheckpointError {}
impl From<io::Error> for CheckpointError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub(crate) type Result<T> = std::result::Result<T, CheckpointError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    V7,
    V6,
    /// V5 lacks optimizer, recurrent, vocabulary, and memory metadata state.
    LegacyV5InferenceOnly,
}

pub struct LoadedCheckpoint {
    pub model: PSSALayerV2,
    pub format: CheckpointFormat,
}

pub(crate) fn invalid(msg: impl Into<String>) -> CheckpointError {
    CheckpointError::Invalid(msg.into())
}

pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(format!("overflow calculating {what}")))
}
fn checked_add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid(format!("overflow calculating {what}")))
}
fn add_mul(total: &mut usize, a: usize, b: usize, what: &str) -> Result<()> {
    *total = checked_add(*total, checked_mul(a, b, what)?, what)?;
    Ok(())
}

fn validate_config(cfg: &PSSAConfigV2) -> Result<()> {
    if cfg.d_vocab == 0
        || cfg.d_latent == 0
        || cfg.d_state == 0
        || cfg.d_mem_key == 0
        || cfg.mem_capacity == 0
        || cfg.chunk_len == 0
    {
        return Err(invalid(
            "all dimensions, memory capacity, and chunk length must be positive",
        ));
    }
    if !cfg.lr.is_finite()
        || cfg.lr <= 0.0
        || !cfg.eps.is_finite()
        || cfg.eps <= 0.0
        || !cfg.tau_mem.is_finite()
        || cfg.tau_mem <= 0.0
    {
        return Err(invalid("lr, eps, and tau_mem must be finite and positive"));
    }
    if !cfg.beta1.is_finite()
        || !(0.0..1.0).contains(&cfg.beta1)
        || !cfg.beta2.is_finite()
        || !(0.0..1.0).contains(&cfg.beta2)
    {
        return Err(invalid("Adam betas must be in [0, 1)"));
    }
    if !cfg.weight_decay.is_finite()
        || cfg.weight_decay < 0.0
        || !cfg.ema_alpha.is_finite()
        || !(0.0..=1.0).contains(&cfg.ema_alpha)
    {
        return Err(invalid("weight decay/ema alpha invalid"));
    }
    allocation_bytes(cfg).map(|_| ())
}

/// Validate a fresh model configuration before its large activation and
/// optimizer buffers are allocated.  Checkpoint loading uses the same guard.
pub fn validate_model_config(cfg: &PSSAConfigV2) -> std::result::Result<(), String> {
    validate_config(cfg).map_err(|e| e.to_string())
}

/// Require enough bytes for the actual persisted tensors before allocating.
/// Tape/chunk size is deliberately absent: a valid long tape need not have a
/// proportionally large checkpoint. validate_config enforces the allocation cap.
fn ensure_backed_by_file(
    c: &PSSAConfigV2,
    available_bytes: usize,
    legacy_count: Option<usize>,
) -> Result<()> {
    let mut parameters = 0;
    for (rows, cols) in [
        (c.d_vocab, c.d_latent), (c.d_latent, c.d_state),
        (c.d_latent, c.d_latent), (c.d_state, c.d_latent), (c.d_state, c.d_latent),
        (c.d_mem_key, c.d_latent), (c.d_mem_key, c.d_latent),
        (c.d_latent, c.d_latent), (c.d_latent, c.d_latent),
        (checked_mul(2, c.d_latent, "MLP width")?, c.d_latent),
        (c.d_latent, checked_mul(2, c.d_latent, "MLP width")?),
        (c.d_vocab, c.d_latent), (16, c.d_latent), (c.d_latent, 16),
        (c.d_latent, 1), (c.d_latent, 1),
    ] {
        add_mul(&mut parameters, rows, cols, "serialized parameters")?;
    }
    let mut required;
    if let Some(count) = legacy_count {
        // Data only: 16 parameter arrays, two memory arrays, and adapter rank.
        required = checked_mul(parameters, 4, "legacy parameter bytes")?;
        add_mul(&mut required, 19, 4, "legacy tensor lengths/rank")?;
        for width in [c.d_mem_key, c.d_latent] {
            add_mul(&mut required, checked_mul(count, width, "legacy memory")?, 4, "legacy memory bytes")?;
        }
    } else {
        required = checked_mul(parameters, 16, "parameter/Adam bytes")?;
        // Four arrays per parameter; eight other arrays plus count and head.
        add_mul(&mut required, 16 * 4 + 10, 8, "tensor lengths/memory metadata")?;
        for (rows, cols) in [
            (c.d_latent, c.d_state), (c.mem_capacity, c.d_mem_key),
            (c.mem_capacity, c.d_latent), (c.mem_capacity, 2), (c.d_latent, 16),
        ] {
            add_mul(&mut required, checked_mul(rows, cols, "persistent storage")?, 4, "persistent bytes")?;
        }
        add_mul(&mut required, c.mem_capacity, 8, "memory timestamps")?;
        add_mul(&mut required, c.d_vocab, 8, "embedding marks")?;
    }
    if available_bytes < required {
        return Err(invalid(format!("truncated persistent tensors: need {required} bytes, have {available_bytes}")));
    }
    Ok(())
}

/// Exact numeric vector storage allocated by `PSSALayerV2::new`, including
/// parameters, optimizer state, memory, tape and both execution paths' scratch.
/// Excludes Vec/adapter headers, allocator bookkeeping, backend resources and
/// optional tokenizer strings. Returns an error on overflow or above the cap.
/// Keep the named fields in sync with the constructor and the actual-storage test.
pub fn allocation_bytes(c: &PSSAConfigV2) -> Result<usize> {
    let (v, m, s, k, cap, l) = (
        c.d_vocab,
        c.d_latent,
        c.d_state,
        c.d_mem_key,
        c.mem_capacity,
        c.chunk_len,
    );
    let two_m = checked_mul(m, 2, "twice d_latent")?;
    let ms = checked_mul(m, s, "latent * state")?;
    let lm = checked_mul(l, m, "chunk * latent")?;
    let ls = checked_mul(l, s, "chunk * state")?;
    let lk = checked_mul(l, k, "chunk * key")?;
    let lv = checked_mul(l, v, "chunk * vocab")?;
    let lr = checked_mul(l, 16, "chunk * rank")?;
    let lms = checked_mul(l, ms, "chunk * latent * state")?;
    let l_two_m = checked_mul(l, two_m, "chunk * MLP width")?;
    let mut f32_count = 0usize;
    let mut usize_count = 0usize;
    // Every parameter owns data, grad, first moment and second moment.
    for (name, rows, cols) in [
        ("embed_w", v, m),
        ("a_mat", m, s),
        ("w_delta", m, m),
        ("w_b", s, m),
        ("w_c", s, m),
        ("w_qx", k, m),
        ("w_qh", k, m),
        ("w_gate", m, m),
        ("w_proj", m, m),
        ("mlp_w1", two_m, m),
        ("mlp_w2", m, two_m),
        ("unembed_w", v, m),
        ("adapter.down", 16, m),
        ("adapter.up", m, 16),
        ("norm_gamma", m, 1),
        ("norm_beta", m, 1),
    ] {
        add_mul(&mut f32_count, checked_mul(rows, cols, name)?, 4, name)?;
    }
    for (name, n) in [
        (
            "adapter.consolidated_up",
            checked_mul(m, 16, "slow adapter")?,
        ),
        ("h_persistent", ms),
        ("ssm_raw_snapshot", ms),
        ("ssm_rates", ms),
        ("ssm_rate_derivatives", ms),
        ("memory.keys", checked_mul(cap, k, "memory keys")?),
        ("memory.values", checked_mul(cap, m, "memory values")?),
        ("memory.norm_sq", cap),
        ("memory.confidence", cap),
        ("tape.x_norm", lm),
        ("tape.inv_rms", l),
        ("tape.delta_raw", lm),
        ("tape.delta", lm),
        ("tape.b_proj", ls),
        ("tape.c_proj", ls),
        ("tape.bar_a", lms),
        ("tape.bar_b", lms),
        (
            "tape.h_states",
            checked_mul(checked_add(l, 1, "chunk + 1")?, ms, "h_states")?,
        ),
        ("tape.y_ssm", lm),
        ("tape.q_euc", lk),
        ("tape.q_norm", l),
        ("tape.q_poincare", lk),
        ("tape.mem_weights", checked_mul(l, cap, "memory weights")?),
        ("tape.m_val", lm),
        ("tape.g_mem", lm),
        ("tape.m_inj", lm),
        ("tape.m_proj", lm),
        ("tape.adapter_hidden", lr),
        ("tape.adapter_act", lr),
        ("tape.z_raw", lm),
        ("tape.mlp_hidden", l_two_m),
        ("tape.mlp_act", l_two_m),
        ("tape.z_final", lm),
        ("tape.logits", lv),
        ("tape.probs", lv),
        ("tape.losses", l),
        ("grad_h_next", ms),
        ("grad_z_final", m),
        ("grad_z_raw", m),
        ("grad_x_norm", m),
        ("buf_m_proj", m),
        ("buf_ad_out", m),
        ("buf_g_mlp_act", two_m),
        ("buf_g_mlp_hidden", two_m),
        ("buf_g_zraw_mlp", m),
        ("buf_g_ad_act", 16),
        ("buf_g_ad_down", 16),
        ("buf_g_m_proj_out", m),
        ("buf_g_m_val", m),
        ("g_query_pnc", k),
        ("g_query_euc", k),
        ("g_y_ssm", m),
        ("buf_g_delta", m),
        ("buf_g_b_proj", s),
        ("buf_g_c_proj", s),
        ("buf_g_h_prev", ms),
        ("bwd_g_zfinal", lm),
        ("bwd_g_zraw", lm),
        ("bwd_g_ad_down", lr),
        ("bwd_g_xnorm", lm),
        ("bwd_g_ysm", lm),
        ("bwd_g_logits", lv),
        ("bwd_g_mlp", l_two_m),
        ("inf_x_norm", m),
        ("inf_delta", m),
        ("inf_b", s),
        ("inf_c", s),
        ("inf_y_ssm", m),
        ("inf_q_euc", k),
        ("inf_q_pnc", k),
        ("inf_mem_weights", cap),
        ("inf_m_val", m),
        ("inf_g_mem", m),
        ("inf_m_proj", m),
        ("inf_ad_act", 16),
        ("inf_ad_out", m),
        ("inf_z_raw", m),
        ("inf_mlp_act", two_m),
        ("inf_mlp_out", m),
        ("inf_z_final", m),
    ] {
        f32_count = checked_add(f32_count, n, name)?;
    }
    for (name, n) in [
        ("memory.last_seen_step", cap),
        ("tape.x_ids", l),
        ("tape.target_ids", l),
        ("embed_row_marks", v),
    ] {
        usize_count = checked_add(usize_count, n, name)?;
    }
    let bytes = checked_add(
        checked_mul(f32_count, std::mem::size_of::<f32>(), "f32 allocation")?,
        checked_mul(
            usize_count,
            std::mem::size_of::<usize>(),
            "usize allocation",
        )?,
        "allocation",
    )?;
    if bytes > MAX_LOAD_ALLOCATION_BYTES {
        return Err(invalid(format!(
            "declared model needs {bytes} bytes, over {MAX_LOAD_ALLOCATION_BYTES} byte cap"
        )));
    }
    Ok(bytes)
}

pub(crate) struct Writer {
    pub(crate) bytes: Vec<u8>,
}
impl Writer {
    pub(crate) fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    pub(crate) fn u64(&mut self, n: u64) {
        self.bytes.extend_from_slice(&n.to_le_bytes());
    }
    pub(crate) fn f32(&mut self, n: f32) {
        self.bytes.extend_from_slice(&n.to_le_bytes());
    }
    pub(crate) fn usize(&mut self, n: usize, what: &str) -> Result<()> {
        self.u64(u64::try_from(n).map_err(|_| invalid(format!("{what} does not fit u64")))?);
        Ok(())
    }
    fn floats(&mut self, xs: &[f32], what: &str) -> Result<()> {
        if !xs.iter().all(|x| x.is_finite()) {
            return Err(invalid(format!("non-finite {what} cannot be saved")));
        }
        self.usize(xs.len(), what)?;
        for &x in xs {
            self.f32(x);
        }
        Ok(())
    }
    fn usizes(&mut self, xs: &[usize], what: &str) -> Result<()> {
        self.usize(xs.len(), what)?;
        for &x in xs {
            self.usize(x, what)?;
        }
        Ok(())
    }
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    off: usize,
}
impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }
    pub(crate) fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .off
            .checked_add(n)
            .ok_or_else(|| invalid(format!("overflow reading {what}")))?;
        if end > self.bytes.len() {
            return Err(invalid(format!("truncated while reading {what}")));
        }
        let out = &self.bytes[self.off..end];
        self.off = end;
        Ok(out)
    }
    fn u16(&mut self, what: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2, what)?.try_into().expect("sized"),
        ))
    }
    pub(crate) fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, what)?.try_into().expect("sized"),
        ))
    }
    pub(crate) fn usize(&mut self, what: &str) -> Result<usize> {
        usize::try_from(self.u64(what)?)
            .map_err(|_| invalid(format!("{what} exceeds platform usize")))
    }
    pub(crate) fn f32(&mut self, what: &str) -> Result<f32> {
        let x = f32::from_le_bytes(self.take(4, what)?.try_into().expect("sized"));
        if !x.is_finite() {
            return Err(invalid(format!("non-finite {what}")));
        }
        Ok(x)
    }
    fn floats_into(&mut self, out: &mut [f32], what: &str, nonnegative: bool) -> Result<()> {
        let n = self.usize(&format!("{what} length"))?;
        if n != out.len() {
            return Err(invalid(format!(
                "{what} length {n}, expected {}",
                out.len()
            )));
        }
        self.float_words_into(out, what, nonnegative)
    }
    fn float_words_into(&mut self, out: &mut [f32], what: &str, nonnegative: bool) -> Result<()> {
        let words = self.take(checked_mul(out.len(), 4, what)?, what)?;
        for (slot, word) in out.iter_mut().zip(words.chunks_exact(4)) {
            let x = f32::from_le_bytes(word.try_into().expect("sized"));
            if !x.is_finite() {
                return Err(invalid(format!("non-finite {what}")));
            }
            if nonnegative && x < 0.0 {
                return Err(invalid(format!("negative {what}")));
            }
            *slot = x;
        }
        Ok(())
    }
    fn usizes_into(&mut self, out: &mut [usize], what: &str) -> Result<()> {
        let n = self.usize(&format!("{what} length"))?;
        if n != out.len() {
            return Err(invalid(format!(
                "{what} length {n}, expected {}",
                out.len()
            )));
        }
        let words = self.take(checked_mul(n, 8, what)?, what)?;
        for (slot, word) in out.iter_mut().zip(words.chunks_exact(8)) {
            *slot = usize::try_from(u64::from_le_bytes(word.try_into().expect("sized")))
                .map_err(|_| invalid(format!("{what} exceeds platform usize")))?;
        }
        Ok(())
    }
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.off)
    }
    pub(crate) fn done(&self) -> Result<()> {
        if self.off == self.bytes.len() {
            Ok(())
        } else {
            Err(invalid("trailing bytes"))
        }
    }
}

pub(crate) fn write_matrix(w: &mut Writer, p: &ParamMatrix, name: &str) -> Result<()> {
    let expected = checked_mul(p.rows, p.cols, name)?;
    if p.data.len() != expected
        || p.grad.len() != expected
        || p.m.len() != expected
        || p.v.len() != expected
    {
        return Err(invalid(format!("{name} parameter shape mismatch")));
    }
    if p.v.iter().any(|&x| x < 0.0) {
        return Err(invalid(format!("negative {name}.v")));
    }
    w.floats(&p.data, name)?;
    w.floats(&p.grad, &format!("{name}.grad"))?;
    w.floats(&p.m, &format!("{name}.m"))?;
    w.floats(&p.v, &format!("{name}.v"))?;
    Ok(())
}
pub(crate) fn write_vector(w: &mut Writer, p: &ParamVector, name: &str) -> Result<()> {
    if p.data.len() != p.grad.len() || p.data.len() != p.m.len() || p.data.len() != p.v.len() {
        return Err(invalid(format!("{name} parameter shape mismatch")));
    }
    if p.v.iter().any(|&x| x < 0.0) {
        return Err(invalid(format!("negative {name}.v")));
    }
    w.floats(&p.data, name)?;
    w.floats(&p.grad, &format!("{name}.grad"))?;
    w.floats(&p.m, &format!("{name}.m"))?;
    w.floats(&p.v, &format!("{name}.v"))?;
    Ok(())
}
pub(crate) fn read_matrix(r: &mut Reader<'_>, p: &mut ParamMatrix, name: &str) -> Result<()> {
    r.floats_into(&mut p.data, name, false)?;
    r.floats_into(&mut p.grad, &format!("{name}.grad"), false)?;
    r.floats_into(&mut p.m, &format!("{name}.m"), false)?;
    r.floats_into(&mut p.v, &format!("{name}.v"), true)?;
    Ok(())
}
pub(crate) fn read_vector(r: &mut Reader<'_>, p: &mut ParamVector, name: &str) -> Result<()> {
    r.floats_into(&mut p.data, name, false)?;
    r.floats_into(&mut p.grad, &format!("{name}.grad"), false)?;
    r.floats_into(&mut p.m, &format!("{name}.m"), false)?;
    r.floats_into(&mut p.v, &format!("{name}.v"), true)?;
    Ok(())
}

fn config_to_payload(w: &mut Writer, c: &PSSAConfigV2) -> Result<()> {
    for (n, name) in [
        (c.d_vocab, "d_vocab"),
        (c.d_latent, "d_latent"),
        (c.d_state, "d_state"),
        (c.d_mem_key, "d_mem_key"),
        (c.mem_capacity, "mem_capacity"),
        (c.chunk_len, "chunk_len"),
    ] {
        w.usize(n, name)?;
    }
    for x in [
        c.lr,
        c.beta1,
        c.beta2,
        c.weight_decay,
        c.eps,
        c.tau_mem,
        c.ema_alpha,
    ] {
        w.f32(x);
    }
    Ok(())
}
fn config_from_payload(r: &mut Reader<'_>) -> Result<PSSAConfigV2> {
    let c = PSSAConfigV2 {
        d_vocab: r.usize("d_vocab")?,
        d_latent: r.usize("d_latent")?,
        d_state: r.usize("d_state")?,
        d_mem_key: r.usize("d_mem_key")?,
        mem_capacity: r.usize("mem_capacity")?,
        chunk_len: r.usize("chunk_len")?,
        lr: r.f32("lr")?,
        beta1: r.f32("beta1")?,
        beta2: r.f32("beta2")?,
        weight_decay: r.f32("weight_decay")?,
        eps: r.f32("eps")?,
        tau_mem: r.f32("tau_mem")?,
        ema_alpha: r.f32("ema_alpha")?,
    };
    validate_config(&c)?;
    Ok(c)
}

fn validate_vocab(vocab: &[String], d_vocab: usize) -> Result<()> {
    if vocab.is_empty() {
        return Ok(());
    }
    if vocab.len() != d_vocab {
        return Err(invalid(format!(
            "vocabulary has {}, expected {d_vocab}",
            vocab.len()
        )));
    }
    if vocab[0] != "<unk>" {
        return Err(invalid("vocabulary index 0 must be <unk>"));
    }
    let mut seen = HashSet::with_capacity(vocab.len());
    if vocab.iter().any(|s| !seen.insert(s)) {
        return Err(invalid("vocabulary tokens must be unique"));
    }
    Ok(())
}
pub(crate) fn write_vocab(w: &mut Writer, vocab: &[String], d_vocab: usize) -> Result<()> {
    validate_vocab(vocab, d_vocab)?;
    w.usize(vocab.len(), "vocabulary count")?;
    for token in vocab {
        w.usize(token.len(), "vocabulary token length")?;
        w.bytes.extend_from_slice(token.as_bytes());
    }
    Ok(())
}
pub(crate) fn read_vocab(r: &mut Reader<'_>, d_vocab: usize) -> Result<Vec<String>> {
    let n = r.usize("vocabulary count")?;
    if n != 0 && n != d_vocab {
        return Err(invalid("vocabulary count must be zero or d_vocab"));
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let len = r.usize("vocabulary token length")?;
        if len > r.bytes.len().saturating_sub(r.off) {
            return Err(invalid("truncated vocabulary token"));
        }
        let s = std::str::from_utf8(r.take(len, "vocabulary token")?)
            .map_err(|_| invalid("vocabulary token is not UTF-8"))?;
        v.push(s.to_string());
    }
    validate_vocab(&v, d_vocab)?;
    Ok(v)
}

fn validate_tokenizer_metadata(model: &PSSALayerV2) -> Result<()> {
    match &model.tokenizer_json {
        None => Ok(()),
        Some(json) => {
            if json.len() > MAX_TOKENIZER_JSON_BYTES {
                return Err(invalid("tokenizer JSON exceeds 16 MiB cap"));
            }
            let tokenizer = crate::dataset::Tokenizer::from_serialized(json)
                .map_err(|e| invalid(format!("invalid tokenizer JSON: {e}")))?;
            if tokenizer.ordered_vocabulary().map_err(invalid)? != model.vocabulary {
                return Err(invalid(
                    "tokenizer JSON vocabulary/order differs from model vocabulary",
                ));
            }
            Ok(())
        }
    }
}

fn validate_matrix_shape(p: &ParamMatrix, rows: usize, cols: usize, name: &str) -> Result<()> {
    let expected = checked_mul(rows, cols, name)?;
    if p.rows != rows
        || p.cols != cols
        || p.data.len() != expected
        || p.grad.len() != expected
        || p.m.len() != expected
        || p.v.len() != expected
    {
        return Err(invalid(format!(
            "{name} must have configuration-derived shape [{rows}, {cols}]"
        )));
    }
    Ok(())
}

fn validate_persistent_shapes(model: &PSSALayerV2) -> Result<()> {
    let c = &model.cfg;
    validate_config(c)?;
    if model.step_counter == usize::MAX {
        return Err(invalid("step_counter is exhausted"));
    }
    let mlp = checked_mul(c.d_latent, 2, "MLP width")?;
    for (p, rows, cols, name) in [
        (&model.embed_w, c.d_vocab, c.d_latent, "embed_w"),
        (&model.a_mat, c.d_latent, c.d_state, "a_mat_raw"),
        (&model.w_delta, c.d_latent, c.d_latent, "w_delta"),
        (&model.w_b, c.d_state, c.d_latent, "w_b"),
        (&model.w_c, c.d_state, c.d_latent, "w_c"),
        (&model.w_qx, c.d_mem_key, c.d_latent, "w_qx"),
        (&model.w_qh, c.d_mem_key, c.d_latent, "w_qh"),
        (&model.w_gate, c.d_latent, c.d_latent, "w_gate"),
        (&model.w_proj, c.d_latent, c.d_latent, "w_proj"),
        (&model.mlp_w1, mlp, c.d_latent, "mlp_w1"),
        (&model.mlp_w2, c.d_latent, mlp, "mlp_w2"),
        (&model.unembed_w, c.d_vocab, c.d_latent, "unembed_w"),
    ] {
        validate_matrix_shape(p, rows, cols, name)?;
    }
    for (p, name) in [
        (&model.norm_gamma, "norm_gamma"),
        (&model.norm_beta, "norm_beta"),
    ] {
        if p.data.len() != c.d_latent
            || p.grad.len() != c.d_latent
            || p.m.len() != c.d_latent
            || p.v.len() != c.d_latent
        {
            return Err(invalid(format!(
                "{name} must have configuration-derived length {}",
                c.d_latent
            )));
        }
    }
    if model.adapters.len() != 1
        || model.adapters[0].rank != 16
        || model.adapters[0].d_latent != c.d_latent
    {
        return Err(invalid(
            "V6/V7 requires exactly one rank-16 adapter matching d_latent",
        ));
    }
    let ad = &model.adapters[0];
    validate_matrix_shape(&ad.down_proj, 16, c.d_latent, "adapter.down")?;
    validate_matrix_shape(&ad.up_proj, c.d_latent, 16, "adapter.up")?;
    for (actual, expected, name) in [
        (
            ad.consolidated_up.len(),
            checked_mul(c.d_latent, 16, "slow adapter")?,
            "adapter.consolidated_up",
        ),
        (
            model.h_persistent.len(),
            checked_mul(c.d_latent, c.d_state, "h_persistent")?,
            "h_persistent",
        ),
        (model.embed_row_marks.len(), c.d_vocab, "embed_row_marks"),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{name} length {actual}, expected {expected}"
            )));
        }
    }
    validate_memory(model)
}

/// The common V6 payload, retained byte-for-byte by V7 before its metadata tail.
fn payload_for_v6(model: &PSSALayerV2) -> Result<Vec<u8>> {
    validate_persistent_shapes(model)?;
    validate_vocab(&model.vocabulary, model.cfg.d_vocab)?;
    let mem = &model.memory;
    let mut w = Writer::new();
    config_to_payload(&mut w, &model.cfg)?;
    w.usize(model.step_counter, "step_counter")?;
    w.u64(model.rng.state);
    write_vocab(&mut w, &model.vocabulary, model.cfg.d_vocab)?;
    for (p, n) in [
        (&model.embed_w, "embed_w"),
        (&model.a_mat, "a_mat_raw"),
        (&model.w_delta, "w_delta"),
        (&model.w_b, "w_b"),
        (&model.w_c, "w_c"),
        (&model.w_qx, "w_qx"),
        (&model.w_qh, "w_qh"),
        (&model.w_gate, "w_gate"),
        (&model.w_proj, "w_proj"),
        (&model.mlp_w1, "mlp_w1"),
        (&model.mlp_w2, "mlp_w2"),
        (&model.unembed_w, "unembed_w"),
    ] {
        write_matrix(&mut w, p, n)?;
    }
    write_vector(&mut w, &model.norm_gamma, "norm_gamma")?;
    write_vector(&mut w, &model.norm_beta, "norm_beta")?;
    w.floats(&model.h_persistent, "h_persistent")?;
    w.usize(mem.count, "memory count")?;
    w.usize(mem.write_head, "memory write_head")?;
    w.floats(&mem.keys, "memory keys")?;
    w.floats(&mem.values, "memory values")?;
    w.floats(&mem.norm_sq, "memory norm_sq")?;
    w.floats(&mem.confidence, "memory confidence")?;
    w.usizes(&mem.last_seen_step, "memory last_seen_step")?;
    let ad = &model.adapters[0];
    write_matrix(&mut w, &ad.down_proj, "adapter.down")?;
    write_matrix(&mut w, &ad.up_proj, "adapter.up")?;
    w.floats(&ad.consolidated_up, "adapter.consolidated_up")?;
    w.usizes(&model.embed_row_marks, "embed_row_marks")?;
    Ok(w.bytes)
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stem = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp: PathBuf = parent.join(format!(".{stem}.{}.{}.tmp", std::process::id(), nonce));
    let result = (|| -> Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn container_bytes(version: u16, payload: Vec<u8>) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(b"PSSA");
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(
        &(u64::try_from(payload.len()).map_err(|_| invalid("payload too large"))?).to_le_bytes(),
    );
    bytes.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

fn validate_schedule(model: &PSSALayerV2) -> Result<()> {
    match (
        model.lr_schedule_total_updates,
        model.lr_schedule_warmup_steps,
    ) {
        (Some(0), _) => Err(invalid("learning-rate schedule horizon must be positive")),
        (Some(horizon), _) if horizon < model.step_counter => Err(invalid(
            "learning-rate schedule horizon precedes optimizer step",
        )),
        (Some(horizon), Some(warmup)) if warmup >= horizon => Err(invalid(
            "learning-rate schedule warmup must be less than its horizon",
        )),
        (None, Some(_)) => Err(invalid("learning-rate schedule warmup requires a horizon")),
        _ => Ok(()),
    }
}

/// Writes V7, preserving all V6 state plus tokenizer and schedule metadata.
pub fn save_model(model: &PSSALayerV2, path: impl AsRef<Path>) -> Result<()> {
    validate_schedule(model)?;
    validate_tokenizer_metadata(model)?;
    let mut payload = payload_for_v6(model)?;
    if let Some(json) = &model.tokenizer_json {
        payload.extend_from_slice(
            &(u64::try_from(json.len()).map_err(|_| invalid("tokenizer JSON too large"))?)
                .to_le_bytes(),
        );
        payload.extend_from_slice(json.as_bytes());
    } else {
        payload.extend_from_slice(&0u64.to_le_bytes());
    }
    if let Some(total_updates) = model.lr_schedule_total_updates {
        payload.extend_from_slice(
            &u64::try_from(total_updates)
                .map_err(|_| invalid("learning-rate schedule horizon too large"))?
                .to_le_bytes(),
        );
        if let Some(warmup) = model.lr_schedule_warmup_steps {
            payload.extend_from_slice(
                &u64::try_from(warmup)
                    .map_err(|_| invalid("learning-rate schedule warmup too large"))?
                    .to_le_bytes(),
            );
        }
    }
    atomic_write(path.as_ref(), &container_bytes(FORMAT_VERSION, payload)?)
}

/// Compatibility writer for explicitly requested V6 word checkpoints. It cannot
/// serialize BPE or schedule metadata; use `save_model` for new checkpoints.
pub fn save_model_v6(model: &PSSALayerV2, path: impl AsRef<Path>) -> Result<()> {
    if model.tokenizer_json.is_some() {
        return Err(invalid(
            "V6 cannot serialize byte-level tokenizer metadata; use V7",
        ));
    }
    if model.lr_schedule_total_updates.is_some() || model.lr_schedule_warmup_steps.is_some() {
        return Err(invalid(
            "V6 cannot serialize learning-rate schedule metadata; use V7",
        ));
    }
    atomic_write(
        path.as_ref(),
        &container_bytes(V6_FORMAT_VERSION, payload_for_v6(model)?)?,
    )
}

pub(crate) fn read_file_capped(path: &Path) -> Result<Vec<u8>> {
    let len = usize::try_from(fs::metadata(path)?.len()).map_err(|_| invalid("file too large"))?;
    if len
        > MAX_LOAD_ALLOCATION_BYTES
            .checked_add(HEADER_LEN)
            .ok_or_else(|| invalid("load cap overflow"))?
    {
        return Err(invalid("checkpoint file exceeds load cap"));
    }
    Ok(fs::read(path)?)
}

fn validate_memory(model: &PSSALayerV2) -> Result<()> {
    let m = &model.memory;
    if m.capacity != model.cfg.mem_capacity
        || m.dim_key != model.cfg.d_mem_key
        || m.dim_val != model.cfg.d_latent
        || m.capacity == 0
    {
        return Err(invalid("memory dimensions do not match configuration"));
    }
    // Validate every backing vector before indexing, even for an empty bank.
    for (actual, expected, name) in [
        (
            m.keys.len(),
            checked_mul(m.capacity, m.dim_key, "memory keys")?,
            "memory keys",
        ),
        (
            m.values.len(),
            checked_mul(m.capacity, m.dim_val, "memory values")?,
            "memory values",
        ),
        (m.norm_sq.len(), m.capacity, "memory norm_sq"),
        (m.confidence.len(), m.capacity, "memory confidence"),
        (m.last_seen_step.len(), m.capacity, "memory last_seen_step"),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{name} length {actual}, expected {expected}"
            )));
        }
    }
    if !m.keys.iter().chain(&m.values).all(|x| x.is_finite()) {
        return Err(invalid("non-finite memory keys/values"));
    }
    if m.norm_sq.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(invalid("memory norm_sq must be finite and nonnegative"));
    }
    if m.count > m.capacity
        || m.write_head >= m.capacity
        || (m.count < m.capacity && m.write_head != 0)
    {
        return Err(invalid("memory count/write head invalid"));
    }
    for i in 0..m.capacity {
        if !m.confidence[i].is_finite() || m.confidence[i] < 0.0 {
            return Err(invalid("memory confidence invalid"));
        }
        if i < m.count {
            let key = &m.keys[i * m.dim_key..(i + 1) * m.dim_key];
            let sq = key_dot(key);
            let stored = m.norm_sq[i];
            if !stored.is_finite()
                || stored < 0.0
                || stored >= 1.0
                || !sq.is_finite()
                || sq >= 1.0
                || (stored - sq).abs() > 2e-5 * (1.0 + sq.abs())
            {
                return Err(invalid("memory norm_sq/key metadata invalid"));
            }
            if m.last_seen_step[i] > model.step_counter {
                return Err(invalid("memory timestamp exceeds checkpoint step"));
            }
        }
    }
    Ok(())
}
fn key_dot(xs: &[f32]) -> f32 {
    // Match the bank's robust norm policy. A separate f32 accumulation can
    // round a valid high-dimensional projected key outside the open ball.
    crate::memory::HyperbolicEpisodicBankV2::squared_norm(xs)
}

pub(crate) fn checked_payload<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [u8]> {
    if bytes.len() < HEADER_LEN {
        return Err(invalid(format!("truncated {label} header")));
    }
    let payload_len = usize::try_from(u64::from_le_bytes(bytes[6..14].try_into().expect("sized")))
        .map_err(|_| invalid("payload length exceeds platform usize"))?;
    if payload_len > MAX_LOAD_ALLOCATION_BYTES
        || bytes.len()
            != HEADER_LEN
                .checked_add(payload_len)
                .ok_or_else(|| invalid("payload length overflow"))?
    {
        return Err(invalid("payload length does not match file"));
    }
    let payload = &bytes[HEADER_LEN..];
    if fnv1a64(payload) != u64::from_le_bytes(bytes[14..22].try_into().expect("sized")) {
        return Err(invalid("payload checksum mismatch"));
    }
    Ok(payload)
}

fn load_payload(payload: &[u8], is_v7: bool) -> Result<LoadedCheckpoint> {
    let mut r = Reader::new(payload);
    let cfg = config_from_payload(&mut r)?;
    let step = r.usize("step_counter")?;
    if step == usize::MAX {
        return Err(invalid("step_counter is exhausted"));
    }
    let rng_state = r.u64("rng state")?;
    let vocabulary = read_vocab(&mut r, cfg.d_vocab)?;
    ensure_backed_by_file(&cfg, r.remaining(), None)?;
    let mut model = PSSALayerV2::new(cfg, 1);
    model.step_counter = step;
    model.rng = SimpleRng::new(rng_state);
    model.rng.state = rng_state;
    for (p, n) in [
        (&mut model.embed_w, "embed_w"),
        (&mut model.a_mat, "a_mat_raw"),
        (&mut model.w_delta, "w_delta"),
        (&mut model.w_b, "w_b"),
        (&mut model.w_c, "w_c"),
        (&mut model.w_qx, "w_qx"),
        (&mut model.w_qh, "w_qh"),
        (&mut model.w_gate, "w_gate"),
        (&mut model.w_proj, "w_proj"),
        (&mut model.mlp_w1, "mlp_w1"),
        (&mut model.mlp_w2, "mlp_w2"),
        (&mut model.unembed_w, "unembed_w"),
    ] {
        read_matrix(&mut r, p, n)?;
    }
    read_vector(&mut r, &mut model.norm_gamma, "norm_gamma")?;
    read_vector(&mut r, &mut model.norm_beta, "norm_beta")?;
    r.floats_into(&mut model.h_persistent, "h_persistent", false)?;
    let count = r.usize("memory count")?;
    let head = r.usize("memory write_head")?;
    if count > model.memory.capacity || head >= model.memory.capacity {
        return Err(invalid("memory count/write head invalid"));
    }
    model.memory.count = count;
    model.memory.write_head = head;
    r.floats_into(&mut model.memory.keys, "memory keys", false)?;
    r.floats_into(&mut model.memory.values, "memory values", false)?;
    r.floats_into(&mut model.memory.norm_sq, "memory norm_sq", true)?;
    r.floats_into(&mut model.memory.confidence, "memory confidence", true)?;
    r.usizes_into(&mut model.memory.last_seen_step, "memory last_seen_step")?;
    read_matrix(&mut r, &mut model.adapters[0].down_proj, "adapter.down")?;
    read_matrix(&mut r, &mut model.adapters[0].up_proj, "adapter.up")?;
    r.floats_into(
        &mut model.adapters[0].consolidated_up,
        "adapter.consolidated_up",
        false,
    )?;
    r.usizes_into(&mut model.embed_row_marks, "embed_row_marks")?;
    model.vocabulary = vocabulary;
    if is_v7 {
        let json_len = r.usize("tokenizer JSON length")?;
        if json_len > MAX_TOKENIZER_JSON_BYTES {
            return Err(invalid("tokenizer JSON exceeds 16 MiB cap"));
        }
        let json = if json_len == 0 {
            None
        } else {
            let text = std::str::from_utf8(r.take(json_len, "tokenizer JSON")?)
                .map_err(|_| invalid("tokenizer JSON is not UTF-8"))?;
            Some(text.to_string())
        };
        model.tokenizer_json = json;
        validate_tokenizer_metadata(&model)?;
        if !matches!(r.remaining(), 0 | 8 | 16) {
            return Err(invalid("invalid V7 metadata tail"));
        }
        if r.remaining() >= 8 {
            let total_updates = r.usize("learning-rate schedule horizon")?;
            // Preserve the old reader's zero-horizon sentinel for horizon-only tails.
            model.lr_schedule_total_updates = (total_updates != 0).then_some(total_updates);
            if r.remaining() == 8 {
                model.lr_schedule_warmup_steps = Some(r.usize("learning-rate schedule warmup")?);
            }
        }
        validate_schedule(&model)?;
    }
    r.done()?;
    validate_memory(&model)?;
    Ok(LoadedCheckpoint {
        model,
        format: if is_v7 {
            CheckpointFormat::V7
        } else {
            CheckpointFormat::V6
        },
    })
}

fn load_v6(bytes: &[u8]) -> Result<LoadedCheckpoint> {
    load_payload(checked_payload(bytes, "V6")?, false)
}

fn load_v7(bytes: &[u8]) -> Result<LoadedCheckpoint> {
    load_payload(checked_payload(bytes, "V7")?, true)
}

fn legacy_u32(r: &mut Reader<'_>, what: &str) -> Result<usize> {
    usize::try_from(u32::from_le_bytes(
        r.take(4, what)?.try_into().expect("sized"),
    ))
    .map_err(|_| invalid(format!("{what} too large")))
}
fn legacy_slice_into(r: &mut Reader<'_>, out: &mut [f32], what: &str) -> Result<()> {
    let n = legacy_u32(r, &format!("{what} length"))?;
    if n != out.len() {
        return Err(invalid(format!(
            "legacy {what} length {n}, expected {}",
            out.len()
        )));
    }
    r.float_words_into(out, what, false)
}
fn inverse_softplus(y: f32) -> f32 {
    if y > 20.0 {
        y + (-(-y).exp()).ln_1p()
    } else {
        y.exp_m1().ln()
    }
}
fn load_v5(bytes: &[u8]) -> Result<LoadedCheckpoint> {
    if bytes.len() < HEADER_V5_LEN {
        return Err(invalid("truncated legacy V5 header"));
    }
    let mut r = Reader::new(bytes);
    if r.take(4, "magic")? != b"PSSA" {
        return Err(invalid("bad magic"));
    }
    if r.u16("version")? != 5 {
        return Err(invalid("legacy loader called for non-V5"));
    }
    let cfg = PSSAConfigV2 {
        d_vocab: legacy_u32(&mut r, "d_vocab")?,
        d_latent: legacy_u32(&mut r, "d_latent")?,
        d_state: legacy_u32(&mut r, "d_state")?,
        d_mem_key: legacy_u32(&mut r, "d_mem_key")?,
        mem_capacity: legacy_u32(&mut r, "mem_capacity")?,
        chunk_len: legacy_u32(&mut r, "chunk_len")?,
        ..Default::default()
    };
    validate_config(&cfg)?;
    let mem_count = legacy_u32(&mut r, "mem_count")?;
    let adapter_count = legacy_u32(&mut r, "adapter_count")?;
    if mem_count > cfg.mem_capacity {
        return Err(invalid("legacy mem_count exceeds capacity"));
    }
    if adapter_count != 1 {
        return Err(invalid("legacy requires exactly one adapter"));
    }
    ensure_backed_by_file(&cfg, r.remaining(), Some(mem_count))?;
    let mut model = PSSALayerV2::new(cfg, 42);
    legacy_slice_into(&mut r, &mut model.embed_w.data, "embed_w")?;
    legacy_slice_into(&mut r, &mut model.norm_gamma.data, "norm_gamma")?;
    legacy_slice_into(&mut r, &mut model.norm_beta.data, "norm_beta")?;
    legacy_slice_into(&mut r, &mut model.a_mat.data, "a_mat physical")?;
    let mut bad = 0usize;
    for &a in &model.a_mat.data {
        if !a.is_finite() || a >= 0.0 {
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(invalid(format!(
            "legacy physical A contains {bad} nonfinite or nonnegative entries; refusing unsafe conversion"
        )));
    }
    for a in &mut model.a_mat.data {
        *a = inverse_softplus(-*a);
    }
    if model.a_mat.data.iter().any(|x| !x.is_finite()) {
        return Err(invalid(
            "legacy physical A cannot be represented as finite raw rates",
        ));
    }
    for (slot, name) in [
        (&mut model.w_delta, "w_delta"),
        (&mut model.w_b, "w_b"),
        (&mut model.w_c, "w_c"),
        (&mut model.w_qx, "w_qx"),
        (&mut model.w_qh, "w_qh"),
        (&mut model.w_gate, "w_gate"),
        (&mut model.w_proj, "w_proj"),
        (&mut model.mlp_w1, "mlp_w1"),
        (&mut model.mlp_w2, "mlp_w2"),
        (&mut model.unembed_w, "unembed_w"),
    ] {
        legacy_slice_into(&mut r, &mut slot.data, name)?;
    }
    let k_len = checked_mul(mem_count, model.cfg.d_mem_key, "legacy keys")?;
    let v_len = checked_mul(mem_count, model.cfg.d_latent, "legacy values")?;
    legacy_slice_into(&mut r, &mut model.memory.keys[..k_len], "memory keys")?;
    legacy_slice_into(&mut r, &mut model.memory.values[..v_len], "memory values")?;
    model.memory.count = mem_count;
    // V5 does not record a head. A partially filled bank must start at zero;
    // a full imported bank likewise starts by replacing its first stored slot.
    model.memory.write_head = 0;
    for i in 0..mem_count {
        let sq =
            key_dot(&model.memory.keys[i * model.cfg.d_mem_key..(i + 1) * model.cfg.d_mem_key]);
        if !sq.is_finite() || sq >= 1.0 {
            return Err(invalid(format!(
                "legacy memory key {i} is outside the open Poincare ball"
            )));
        }
        model.memory.norm_sq[i] = sq;
    }
    let rank = legacy_u32(&mut r, "adapter rank")?;
    if rank != 16 {
        return Err(invalid(format!("legacy adapter rank {rank}; expected 16")));
    }
    legacy_slice_into(
        &mut r,
        &mut model.adapters[0].down_proj.data,
        "adapter down",
    )?;
    legacy_slice_into(&mut r, &mut model.adapters[0].up_proj.data, "adapter up")?;
    r.done()?;
    Ok(LoadedCheckpoint {
        model,
        format: CheckpointFormat::LegacyV5InferenceOnly,
    })
}

/// Loads V6 checkpoints with complete train-resume state, or checked legacy V5
/// checkpoints marked inference-only because V5 did not contain recoverable state.
pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<LoadedCheckpoint> {
    let bytes = read_file_capped(path.as_ref())?;
    if bytes.len() < 6 || &bytes[..4] != b"PSSA" {
        return Err(invalid("invalid PSSA magic/header"));
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().expect("sized"));
    match version {
        FORMAT_VERSION => load_v7(&bytes),
        V6_FORMAT_VERSION => load_v6(&bytes),
        5 => load_v5(&bytes),
        _ => Err(invalid(format!("unsupported checkpoint version {version}"))),
    }
}
pub fn load_model_v6(path: impl AsRef<Path>) -> Result<PSSALayerV2> {
    Ok(load_checkpoint(path)?.model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "manual release-mode checkpoint decoder timing probe"]
    fn benchmark_checkpoint_decoder_reuse() {
        use std::{hint::black_box, time::Instant};
        let source = ParamMatrix::zeros(2048, 256);
        let mut w = Writer::new();
        write_matrix(&mut w, &source, "matrix").unwrap();
        let mut target = ParamMatrix::zeros(2048, 256);
        let mut timings = [Vec::new(), Vec::new()];
        for round in 0..6 {
            // Alternate order to avoid systematically favoring warm caches.
            for mode in [round % 2, 1 - round % 2] {
                let start = Instant::now();
                for _ in 0..16 {
                    let mut r = Reader::new(black_box(&w.bytes));
                    if mode == 0 {
                        // Prior decoder: allocate each array, then replace it.
                        for (out, nonnegative) in [
                            (&mut target.data, false),
                            (&mut target.grad, false),
                            (&mut target.m, false),
                            (&mut target.v, true),
                        ] {
                            let n = r.usize("length").unwrap();
                            assert_eq!(n, out.len());
                            let mut decoded = Vec::with_capacity(n);
                            for _ in 0..n {
                                let x = r.f32("matrix").unwrap();
                                assert!(!nonnegative || x >= 0.0);
                                decoded.push(x);
                            }
                            *out = decoded;
                        }
                    } else {
                        read_matrix(&mut r, &mut target, "matrix").unwrap();
                    }
                    black_box(&target);
                }
                timings[mode].push(start.elapsed().as_secs_f64());
            }
        }
        for samples in &mut timings {
            samples.sort_by(f64::total_cmp);
        }
        eprintln!(
            "checkpoint decoder (16 x 8 MiB, median of 6): allocating={:.4}s reuse={:.4}s ratio={:.2}x",
            timings[0][3],
            timings[1][3],
            timings[0][3] / timings[1][3]
        );
        assert_eq!(source, target);
    }

    #[test]
    fn parameter_decode_reuses_existing_storage() {
        let mut source = ParamMatrix::zeros(3, 2);
        source.data.fill(1.5);
        source.grad.fill(-0.5);
        source.m.fill(0.25);
        source.v.fill(0.75);
        let mut w = Writer::new();
        write_matrix(&mut w, &source, "matrix").unwrap();
        let mut target = ParamMatrix::zeros(3, 2);
        let pointers = [
            target.data.as_ptr(),
            target.grad.as_ptr(),
            target.m.as_ptr(),
            target.v.as_ptr(),
        ];
        read_matrix(&mut Reader::new(&w.bytes), &mut target, "matrix").unwrap();
        assert_eq!(
            pointers,
            [
                target.data.as_ptr(),
                target.grad.as_ptr(),
                target.m.as_ptr(),
                target.v.as_ptr()
            ]
        );
        assert_eq!(source, target);

        let source = ParamVector::new(7, 0.5);
        let mut w = Writer::new();
        write_vector(&mut w, &source, "vector").unwrap();
        let mut target = ParamVector::new(7, 0.0);
        let pointers = [
            target.data.as_ptr(),
            target.grad.as_ptr(),
            target.m.as_ptr(),
            target.v.as_ptr(),
        ];
        read_vector(&mut Reader::new(&w.bytes), &mut target, "vector").unwrap();
        assert_eq!(
            pointers,
            [
                target.data.as_ptr(),
                target.grad.as_ptr(),
                target.m.as_ptr(),
                target.v.as_ptr()
            ]
        );
        assert_eq!(source, target);
    }
}
