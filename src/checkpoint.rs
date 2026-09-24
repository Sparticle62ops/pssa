//! Versioned PSSA checkpoint container.
//!
//! **V7 wire format (all integer and float words little-endian):**
//! `b"PSSA" | version:u16(7) | payload_len:u64 | fnv1a64(payload):u64 |
//! payload`. V7 writes the exact complete V6 payload, followed by
//! `tokenizer_json_len:u64 | tokenizer_json:utf8[byte_len]`. The optional JSON is
//! zero-length only for the legacy word tokenizer. Shapes are derived from the
//! preceding configuration and every declared length is checked. The checksum is
//! accidental-corruption detection only, not authentication.
//!
//! Scratch and activation tape buffers are intentionally not serialized. V7 and
//! V6 checkpoints are supported between chunks (after backward or an optimizer
//! update) and resume at the next forward call; they do not resume an in-flight
//! backward pass or an external dataset cursor.

use crate::adapter::PlasticAdapterV2;
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

type Result<T> = std::result::Result<T, CheckpointError>;

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

fn invalid(msg: impl Into<String>) -> CheckpointError {
    CheckpointError::Invalid(msg.into())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
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

/// Reject a dimension declaration whose tape/scratch allocation is implausibly
/// larger than the bytes available to populate even its persisted tensors. This
/// prevents a valid-looking tiny header from causing a near-cap allocation.
fn ensure_backed_by_file(cfg: &PSSAConfigV2, available_bytes: usize) -> Result<()> {
    let allocated = allocation_bytes(cfg)?;
    let backed = available_bytes.saturating_mul(128);
    if allocated > backed {
        return Err(invalid(format!(
            "declared allocation {allocated} bytes is disproportionate to {available_bytes} bytes of checkpoint data"
        )));
    }
    Ok(())
}

/// Conservative byte count of all persistent parameters, tape and scratch vectors
/// allocated by `PSSALayerV2::new`, excluding Vec headers.
fn allocation_bytes(c: &PSSAConfigV2) -> Result<usize> {
    let (v, m, s, k, cap, l) = (
        c.d_vocab,
        c.d_latent,
        c.d_state,
        c.d_mem_key,
        c.mem_capacity,
        c.chunk_len,
    );
    let two_m = checked_mul(m, 2, "twice d_latent")?;
    let l_plus_one = checked_add(l, 1, "chunk_len + 1")?;
    let mut f32_count = 0usize;
    let mut usize_count = 0usize;
    // Parameter matrices (four vectors each), plus the slow adapter coefficients.
    for (rows, cols) in [
        (v, m),
        (m, s),
        (m, m),
        (s, m),
        (s, m),
        (k, m),
        (k, m),
        (m, m),
        (m, m),
        (two_m, m),
        (m, two_m),
        (v, m),
        (16, m),
        (m, 16),
    ] {
        add_mul(
            &mut f32_count,
            checked_mul(rows, cols, "parameter shape")?,
            4,
            "parameter vectors",
        )?;
    }
    add_mul(
        &mut f32_count,
        checked_mul(m, 16, "slow adapter")?,
        1,
        "slow adapter",
    )?;
    add_mul(&mut f32_count, m, 8, "norm parameter vectors")?;
    add_mul(
        &mut f32_count,
        checked_mul(m, s, "recurrent state")?,
        1,
        "recurrent state",
    )?;
    add_mul(
        &mut f32_count,
        checked_mul(cap, k, "memory keys")?,
        1,
        "memory keys",
    )?;
    add_mul(
        &mut f32_count,
        checked_mul(cap, m, "memory values")?,
        1,
        "memory values",
    )?;
    add_mul(&mut f32_count, cap, 2, "memory metadata")?;
    add_mul(&mut usize_count, cap, 1, "memory timestamps")?;
    // Activation tape f32 arrays.
    for n in [
        checked_mul(l, m, "tape")?,
        l,
        checked_mul(l, m, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, s, "tape")?,
        checked_mul(l, s, "tape")?,
        checked_mul(checked_mul(l, m, "tape")?, s, "tape")?,
        checked_mul(checked_mul(l, m, "tape")?, s, "tape")?,
        checked_mul(l_plus_one, checked_mul(m, s, "tape")?, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, k, "tape")?,
        l,
        checked_mul(l, k, "tape")?,
        checked_mul(l, cap, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, 16, "tape")?,
        checked_mul(l, 16, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, two_m, "tape")?,
        checked_mul(l, two_m, "tape")?,
        checked_mul(l, m, "tape")?,
        checked_mul(l, v, "tape")?,
        checked_mul(l, v, "tape")?,
        l,
    ] {
        add_mul(&mut f32_count, n, 1, "tape")?;
    }
    add_mul(&mut usize_count, l, 2, "tape token ids")?;
    // Backward and inference scratch: an intentionally conservative 40 latent/state/key vectors.
    for n in [
        checked_mul(m, s, "scratch")?,
        m,
        m,
        m,
        m,
        m,
        m,
        two_m,
        two_m,
        m,
        16,
        16,
        m,
        m,
        m,
        k,
        k,
        m,
        m,
        s,
        s,
        checked_mul(m, s, "scratch")?,
        m,
        m,
        s,
        s,
        m,
        k,
        k,
        cap,
        m,
        m,
        m,
        16,
        m,
        m,
        two_m,
        m,
    ] {
        add_mul(&mut f32_count, n, 1, "scratch")?;
    }
    add_mul(&mut usize_count, v, 1, "embed marks")?;
    let bytes = checked_add(
        checked_mul(f32_count, 4, "f32 allocation")?,
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

struct Writer {
    bytes: Vec<u8>,
}
impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn u64(&mut self, n: u64) {
        self.bytes.extend_from_slice(&n.to_le_bytes());
    }
    fn f32(&mut self, n: f32) {
        self.bytes.extend_from_slice(&n.to_le_bytes());
    }
    fn usize(&mut self, n: usize, what: &str) -> Result<()> {
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

struct Reader<'a> {
    bytes: &'a [u8],
    off: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
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
    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, what)?.try_into().expect("sized"),
        ))
    }
    fn usize(&mut self, what: &str) -> Result<usize> {
        usize::try_from(self.u64(what)?)
            .map_err(|_| invalid(format!("{what} exceeds platform usize")))
    }
    fn f32(&mut self, what: &str) -> Result<f32> {
        let x = f32::from_le_bytes(self.take(4, what)?.try_into().expect("sized"));
        if !x.is_finite() {
            return Err(invalid(format!("non-finite {what}")));
        }
        Ok(x)
    }
    fn floats(&mut self, expected: usize, what: &str, nonnegative: bool) -> Result<Vec<f32>> {
        let n = self.usize(&format!("{what} length"))?;
        if n != expected {
            return Err(invalid(format!("{what} length {n}, expected {expected}")));
        }
        let byte_len = checked_mul(n, 4, what)?;
        if self
            .off
            .checked_add(byte_len)
            .map_or(true, |e| e > self.bytes.len())
        {
            return Err(invalid(format!("truncated while reading {what}")));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let x = self.f32(what)?;
            if nonnegative && x < 0.0 {
                return Err(invalid(format!("negative {what}")));
            }
            out.push(x);
        }
        Ok(out)
    }
    fn usizes(&mut self, expected: usize, what: &str) -> Result<Vec<usize>> {
        let n = self.usize(&format!("{what} length"))?;
        if n != expected {
            return Err(invalid(format!("{what} length {n}, expected {expected}")));
        }
        let mut o = Vec::with_capacity(n);
        for _ in 0..n {
            o.push(self.usize(what)?);
        }
        Ok(o)
    }
    fn done(&self) -> Result<()> {
        if self.off == self.bytes.len() {
            Ok(())
        } else {
            Err(invalid("trailing bytes"))
        }
    }
}

fn write_matrix(w: &mut Writer, p: &ParamMatrix, name: &str) -> Result<()> {
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
fn write_vector(w: &mut Writer, p: &ParamVector, name: &str) -> Result<()> {
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
fn read_matrix(r: &mut Reader<'_>, p: &mut ParamMatrix, name: &str) -> Result<()> {
    p.data = r.floats(p.data.len(), name, false)?;
    p.grad = r.floats(p.grad.len(), &format!("{name}.grad"), false)?;
    p.m = r.floats(p.m.len(), &format!("{name}.m"), false)?;
    p.v = r.floats(p.v.len(), &format!("{name}.v"), true)?;
    Ok(())
}
fn read_vector(r: &mut Reader<'_>, p: &mut ParamVector, name: &str) -> Result<()> {
    p.data = r.floats(p.data.len(), name, false)?;
    p.grad = r.floats(p.grad.len(), &format!("{name}.grad"), false)?;
    p.m = r.floats(p.m.len(), &format!("{name}.m"), false)?;
    p.v = r.floats(p.v.len(), &format!("{name}.v"), true)?;
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
fn write_vocab(w: &mut Writer, vocab: &[String], d_vocab: usize) -> Result<()> {
    validate_vocab(vocab, d_vocab)?;
    w.usize(vocab.len(), "vocabulary count")?;
    for token in vocab {
        w.usize(token.len(), "vocabulary token length")?;
        w.bytes.extend_from_slice(token.as_bytes());
    }
    Ok(())
}
fn read_vocab(r: &mut Reader<'_>, d_vocab: usize) -> Result<Vec<String>> {
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

/// The common V6 payload, retained byte-for-byte by V7 before its metadata tail.
fn payload_for_v6(model: &PSSALayerV2) -> Result<Vec<u8>> {
    validate_config(&model.cfg)?;
    allocation_bytes(&model.cfg)?;
    validate_vocab(&model.vocabulary, model.cfg.d_vocab)?;
    if model.adapters.len() != 1
        || model.adapters[0].rank != 16
        || model.adapters[0].d_latent != model.cfg.d_latent
    {
        return Err(invalid("V6 requires exactly one rank-16 adapter"));
    }
    if model.h_persistent.len()
        != checked_mul(model.cfg.d_latent, model.cfg.d_state, "h_persistent")?
    {
        return Err(invalid("h_persistent shape mismatch"));
    }
    let mem = &model.memory;
    if mem.capacity != model.cfg.mem_capacity
        || mem.dim_key != model.cfg.d_mem_key
        || mem.dim_val != model.cfg.d_latent
        || mem.count > mem.capacity
        || mem.write_head >= mem.capacity
    {
        return Err(invalid("memory dimensions or metadata invalid"));
    }
    validate_memory(model)?;
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
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
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

/// Writes the current V7 format. It preserves all V6 state plus tokenizer metadata.
pub fn save_model(model: &PSSALayerV2, path: impl AsRef<Path>) -> Result<()> {
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
    atomic_write(path.as_ref(), &container_bytes(FORMAT_VERSION, payload)?)
}

/// Compatibility writer for explicitly requested V6 word checkpoints. It cannot
/// serialize BPE metadata; use `save_model` for all new checkpoints.
pub fn save_model_v6(model: &PSSALayerV2, path: impl AsRef<Path>) -> Result<()> {
    if model.tokenizer_json.is_some() {
        return Err(invalid(
            "V6 cannot serialize byte-level tokenizer metadata; use V7",
        ));
    }
    atomic_write(
        path.as_ref(),
        &container_bytes(V6_FORMAT_VERSION, payload_for_v6(model)?)?,
    )
}

fn read_file_capped(path: &Path) -> Result<Vec<u8>> {
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
    xs.iter().map(|x| x * x).sum()
}

fn checked_payload<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [u8]> {
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
    ensure_backed_by_file(&cfg, payload.len())?;
    let step = r.usize("step_counter")?;
    let rng_state = r.u64("rng state")?;
    let vocabulary = read_vocab(&mut r, cfg.d_vocab)?;
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
    model.h_persistent = r.floats(model.h_persistent.len(), "h_persistent", false)?;
    let count = r.usize("memory count")?;
    let head = r.usize("memory write_head")?;
    if count > model.memory.capacity || head >= model.memory.capacity {
        return Err(invalid("memory count/write head invalid"));
    }
    model.memory.count = count;
    model.memory.write_head = head;
    model.memory.keys = r.floats(model.memory.keys.len(), "memory keys", false)?;
    model.memory.values = r.floats(model.memory.values.len(), "memory values", false)?;
    model.memory.norm_sq = r.floats(model.memory.norm_sq.len(), "memory norm_sq", true)?;
    model.memory.confidence = r.floats(model.memory.confidence.len(), "memory confidence", true)?;
    model.memory.last_seen_step =
        r.usizes(model.memory.last_seen_step.len(), "memory last_seen_step")?;
    read_matrix(&mut r, &mut model.adapters[0].down_proj, "adapter.down")?;
    read_matrix(&mut r, &mut model.adapters[0].up_proj, "adapter.up")?;
    model.adapters[0].consolidated_up = r.floats(
        model.adapters[0].consolidated_up.len(),
        "adapter.consolidated_up",
        false,
    )?;
    model.embed_row_marks = r.usizes(model.embed_row_marks.len(), "embed_row_marks")?;
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
fn legacy_slice(r: &mut Reader<'_>, expected: usize, what: &str) -> Result<Vec<f32>> {
    let n = legacy_u32(r, &format!("{what} length"))?;
    if n != expected {
        return Err(invalid(format!(
            "legacy {what} length {n}, expected {expected}"
        )));
    }
    r.floats_legacy(n, what)
}
impl Reader<'_> {
    fn floats_legacy(&mut self, n: usize, what: &str) -> Result<Vec<f32>> {
        let byte_len = checked_mul(n, 4, what)?;
        if self
            .off
            .checked_add(byte_len)
            .map_or(true, |e| e > self.bytes.len())
        {
            return Err(invalid(format!("truncated legacy {what}")));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.f32(what)?);
        }
        Ok(v)
    }
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
    ensure_backed_by_file(&cfg, bytes.len())?;
    let mem_count = legacy_u32(&mut r, "mem_count")?;
    let adapter_count = legacy_u32(&mut r, "adapter_count")?;
    if mem_count > cfg.mem_capacity {
        return Err(invalid("legacy mem_count exceeds capacity"));
    }
    if adapter_count != 1 {
        return Err(invalid("legacy requires exactly one adapter"));
    }
    let mut model = PSSALayerV2::new(cfg, 42);
    model.embed_w.data = legacy_slice(&mut r, model.embed_w.data.len(), "embed_w")?;
    model.norm_gamma.data = legacy_slice(&mut r, model.norm_gamma.data.len(), "norm_gamma")?;
    model.norm_beta.data = legacy_slice(&mut r, model.norm_beta.data.len(), "norm_beta")?;
    let physical = legacy_slice(&mut r, model.a_mat.data.len(), "a_mat physical")?;
    let mut bad = 0usize;
    for &a in &physical {
        if !a.is_finite() || a >= 0.0 {
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(invalid(format!(
            "legacy physical A contains {bad} nonfinite or nonnegative entries; refusing unsafe conversion"
        )));
    }
    model.a_mat.data = physical.into_iter().map(|a| inverse_softplus(-a)).collect();
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
        slot.data = legacy_slice(&mut r, slot.data.len(), name)?;
    }
    let k_len = checked_mul(mem_count, model.cfg.d_mem_key, "legacy keys")?;
    let v_len = checked_mul(mem_count, model.cfg.d_latent, "legacy values")?;
    let keys = legacy_slice(&mut r, k_len, "memory keys")?;
    let vals = legacy_slice(&mut r, v_len, "memory values")?;
    model.memory.keys[..k_len].copy_from_slice(&keys);
    model.memory.values[..v_len].copy_from_slice(&vals);
    model.memory.count = mem_count;
    model.memory.write_head = mem_count % model.memory.capacity;
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
    model.adapters[0] = PlasticAdapterV2::new(model.cfg.d_latent, 16, &mut model.rng);
    model.adapters[0].down_proj.data = legacy_slice(
        &mut r,
        model.adapters[0].down_proj.data.len(),
        "adapter down",
    )?;
    model.adapters[0].up_proj.data =
        legacy_slice(&mut r, model.adapters[0].up_proj.data.len(), "adapter up")?;
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
