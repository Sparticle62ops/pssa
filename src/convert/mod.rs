//! Universal checkpoint converter: HuggingFace safetensors -> PSSA v6.
//!
//! The converter transplants the token table (embedding matrix) of any dense
//! decoder checkpoint into a freshly seeded PSSA student: every embedding row
//! is pushed through a fixed seeded projection into the target latent width,
//! rescaled to unit RMS, the unembedding is tied to the projected table, and
//! the logit scale is calibrated on the emitted student (see
//! docs/qwen-distill.md for the math and why). Everything outside the token
//! table keeps its seeded initialization, exactly like the Qwen3-specific
//! `qwen3_convert` path.
//!
//! Families are detected from `config.json` (see `detect`); hybrid
//! linear-attention architectures (Qwen3.5/3.6/3.8) are rejected with a clear
//! reason rather than silently misread.

pub mod detect;
pub mod maps;

use crate::linalg::SimpleRng;
use crate::pssa::PSSALayerV2;
use crate::qwen3::Qwen3PssaTarget;
use detect::{ArchInfo, Family};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ------------------------------------------------------------------ report

#[derive(Clone, Debug)]
pub struct ConversionReport {
    pub family_label: String,
    pub source_model_type: String,
    pub embedding_tensor: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub depth_requested: usize,
    pub depth_used: usize,
    pub latent_dim: usize,
    pub checkpoint_path: PathBuf,
    pub checkpoint_bytes: u64,
    pub logit_rms_before: f32,
    pub logit_scale: f32,
    pub notes: Vec<String>,
}

impl ConversionReport {
    /// Human-readable summary printed by the CLI and surfaced in the TUI.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "source: {} (model_type '{}', {} layers, hidden {}, vocab {})",
            self.family_label, self.source_model_type, self.depth_requested, self.hidden_size, self.vocab_size
        ));
        lines.push(format!(
            "token table: '{}' ({} rows x {} cols)",
            self.embedding_tensor, self.vocab_size, self.hidden_size
        ));
        lines.push(format!(
            "target: latent {} state 16 key 32 capacity 512 chunk 64 depth {}",
            self.latent_dim, self.depth_used
        ));
        lines.push(format!(
            "wrote {} ({} bytes); logit RMS before calibration {:.3}, unembed scale {:.3e}",
            self.checkpoint_path.display(),
            self.checkpoint_bytes,
            self.logit_rms_before,
            self.logit_scale
        ));
        for note in &self.notes {
            lines.push(format!("note: {note}"));
        }
        lines.join("\n")
    }
}

// ------------------------------------------------------------ safetensors

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SfDType {
    F32,
    Bf16,
    F16,
}

impl SfDType {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "F32" => Ok(SfDType::F32),
            "BF16" => Ok(SfDType::Bf16),
            "F16" => Ok(SfDType::F16),
            other => Err(format!(
                "unsupported tensor dtype '{other}' (reader supports F32, BF16, F16)"
            )),
        }
    }

    fn element_size(self) -> u64 {
        match self {
            SfDType::F32 => 4,
            SfDType::Bf16 | SfDType::F16 => 2,
        }
    }
}

#[derive(Clone)]
struct TensorInfo {
    name: String,
    dtype: SfDType,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

struct SafetensorsFile {
    path: PathBuf,
    file: File,
    data_start: u64,
    tensors: Vec<TensorInfo>,
}

fn open_safetensors(path: &Path) -> Result<SafetensorsFile, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|error| format!("{}: truncated safetensors header: {error}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    if header_len > 64 * 1024 * 1024 {
        return Err(format!(
            "{}: safetensors header is {} bytes, above the 64 MiB sanity cap",
            path.display(),
            header_len
        ));
    }
    let mut header_json = vec![0u8; header_len as usize];
    file.read_exact(&mut header_json)
        .map_err(|error| format!("{}: truncated safetensors header: {error}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&header_json)
        .map_err(|error| format!("{}: malformed safetensors header: {error}", path.display()))?;
    let object = header
        .as_object()
        .ok_or_else(|| format!("{}: safetensors header is not an object", path.display()))?;

    let mut tensors = Vec::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = SfDType::parse(
            entry
                .get("dtype")
                .and_then(|value| value.as_str())
                .ok_or_else(|| format!("{}: tensor '{name}' has no dtype", path.display()))?,
        )?;
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|values| values.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .ok_or_else(|| format!("{}: tensor '{name}' has no shape", path.display()))?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("{}: tensor '{name}' has no data_offsets", path.display()))?;
        if offsets.len() != 2 {
            return Err(format!("{}: tensor '{name}' has malformed data_offsets", path.display()));
        }
        let start = offsets[0].as_u64().unwrap_or(0);
        let end = offsets[1].as_u64().unwrap_or(0);
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(SafetensorsFile {
        path: path.to_path_buf(),
        file,
        data_start: 8 + header_len,
        tensors,
    })
}

impl SafetensorsFile {
    fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Read one row of a 2-D tensor, converted to f32. Rows are contiguous in
    /// safetensors, so a row is one seek + one read: a 150k-vocab table never
    /// has to be resident all at once.
    fn read_row(&mut self, tensor: &TensorInfo, row: usize, out: &mut Vec<f32>) -> Result<(), String> {
        let cols = *tensor.shape.last().ok_or("empty tensor shape")?;
        let row_bytes = cols as u64 * tensor.dtype.element_size();
        let offset = self.data_start + tensor.start + row as u64 * row_bytes;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|error| format!("{}: seek failed: {error}", self.path.display()))?;
        let mut raw = vec![0u8; row_bytes as usize];
        self.file
            .read_exact(&mut raw)
            .map_err(|error| format!("{}: read failed at row {row}: {error}", self.path.display()))?;
        out.clear();
        out.reserve(cols);
        match tensor.dtype {
            SfDType::F32 => {
                for chunk in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            SfDType::Bf16 => {
                for pair in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([pair[0], pair[1]]);
                    out.push(f32::from_bits((bits as u32) << 16));
                }
            }
            SfDType::F16 => {
                for pair in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([pair[0], pair[1]]);
                    out.push(f16_bits_to_f32(bits));
                }
            }
        }
        Ok(())
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;
    let out_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                // Subnormal: renormalize into an F32 exponent.
                let mut shifts = 0u32;
                let mut mantissa = mantissa;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    shifts += 1;
                }
                mantissa &= 0x03FF;
                sign | ((113 - shifts) << 23) | (mantissa << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(out_bits)
}

// -------------------------------------------------------------- discovery

fn shard_paths(source: &Path) -> Result<Vec<PathBuf>, String> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if source.is_dir() {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(source)
            .map_err(|error| format!("{}: {error}", source.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!(
                "{}: no *.safetensors files found; point at the model directory or a shard file",
                source.display()
            ));
        }
        return Ok(shards);
    }
    Err(format!("{}: no such file or directory", source.display()))
}

/// Locate the input-embedding tensor across the shards: family candidates
/// first, then a generic scan so unusual checkpoints still convert.
fn find_embedding(shards: &[SafetensorsFile], family: Family) -> Result<(usize, String), String> {
    for candidate in maps::embedding_candidates(family) {
        for (index, shard) in shards.iter().enumerate() {
            if let Some(tensor) = shard.find(candidate) {
                return Ok((index, tensor.name.clone()));
            }
        }
    }
    for (index, shard) in shards.iter().enumerate() {
        for tensor in &shard.tensors {
            let looks_like_embedding = (tensor.name.contains("embed_tokens.weight")
                || tensor.name.ends_with("wte.weight")
                || tensor.name == "tok_embeddings.weight")
                && !tensor.name.contains("lm_head");
            if looks_like_embedding {
                return Ok((index, tensor.name.clone()));
            }
        }
    }
    let preview: Vec<String> = shards[0]
        .tensors
        .iter()
        .take(8)
        .map(|tensor| tensor.name.clone())
        .collect();
    Err(format!(
        "no embedding tensor found (looked for {:?}, then any '*embed*' table). First tensors in {}: {:?}",
        maps::embedding_candidates(family),
        shards[0].path.display(),
        preview
    ))
}

// -------------------------------------------------------------- conversion

/// Detect the architecture and discover the shards for `source` without
/// running the conversion. Shared by the CLI and the TUI convert panel.
pub fn plan_from_paths(
    source: &Path,
    config: &serde_json::Value,
) -> Result<(ArchInfo, Vec<PathBuf>), String> {
    let shards = shard_paths(source)?;
    let info = detect::detect(config)?;
    Ok((info, shards))
}

pub struct ConvertOptions {
    pub shards: Vec<PathBuf>,
    pub info: ArchInfo,
    pub target_mode: String,
    pub seed: u64,
    pub out_path: PathBuf,
}

pub fn convert(options: ConvertOptions) -> Result<ConversionReport, String> {
    let ConvertOptions {
        shards,
        info,
        target_mode,
        seed,
        out_path,
    } = options;

    let mut files: Vec<SafetensorsFile> = Vec::with_capacity(shards.len());
    for shard in &shards {
        files.push(open_safetensors(shard)?);
    }

    let (shard_index, embedding_name) = find_embedding(&files, info.family)?;
    let mut notes = vec![maps::family_note(info.family).to_owned()];
    if info.mixture_of_experts {
        notes.push("mixture-of-experts routing detected; only the token table transfers".to_owned());
    }

    // Target shape: compact (latent 256) or source-width. Depth follows the
    // source stack but the student shape tops out at 32 layers; clamp with a
    // visible note instead of failing deep stacks outright.
    let depth_requested = info.num_hidden_layers;
    let depth_used = depth_requested.min(32);
    if depth_used != depth_requested {
        notes.push(format!(
            "depth clamped {depth_requested} -> {depth_used} (student shape supports up to 32 layers)"
        ));
    }
    let latent_dim = match target_mode.as_str() {
        "compact" => 256,
        "source-width" => info.hidden_size,
        other => return Err(format!("unknown --target '{other}'; expected compact or source-width")),
    };
    let target = Qwen3PssaTarget {
        vocab_size: info.vocab_size,
        depth: depth_used,
        latent_dim,
        memory_key_dim: 32,
        memory_capacity: 512,
        chunk_len: 64,
        state_dim: 16,
    };
    target.validate()?;

    let mut layer = PSSALayerV2::new(target.pssa_config(), seed);
    let hidden = info.hidden_size;
    let vocab = info.vocab_size;
    let latent = target.latent_dim;

    // Fixed seeded projection, row-major [latent][hidden], matching the
    // Qwen3 emitter's layout and scale.
    let mut projection_rng = SimpleRng::new(seed ^ 0x5a17_c0de_1234_5678);
    let mut projection = vec![0.0f32; latent * hidden];
    for value in projection.iter_mut() {
        *value = projection_rng.gen_range_f32(-1.0, 1.0);
    }
    let projection_scale = 1.0 / (hidden as f32).sqrt();
    for value in projection.iter_mut() {
        *value *= projection_scale;
    }

    let mut projected = vec![0.0f32; vocab * latent];
    {
        let shard = &mut files[shard_index];
        let tensor = shard
            .tensors
            .iter()
            .find(|tensor| tensor.name == embedding_name)
            .ok_or_else(|| format!("embedding tensor '{embedding_name}' disappeared mid-conversion"))?
            .clone();
        if tensor.shape.len() != 2 || tensor.shape[0] != vocab || tensor.shape[1] != hidden {
            return Err(format!(
                "embedding tensor '{}' has shape {:?}; expected [{vocab}, {hidden}] from config.json",
                tensor.name, tensor.shape
            ));
        }
        let mut row_source = Vec::with_capacity(hidden);
        for token in 0..vocab {
            shard.read_row(&tensor, token, &mut row_source)?;
            let out_row = &mut projected[token * latent..(token + 1) * latent];
            for (j, slot) in out_row.iter_mut().enumerate() {
                let proj_row = &projection[j * hidden..(j + 1) * hidden];
                let mut acc = 0.0f32;
                for (source, weight) in row_source.iter().zip(proj_row.iter()) {
                    acc += source * weight;
                }
                *slot = acc;
            }
            let mut sum_squares = 0.0f32;
            for value in out_row.iter() {
                sum_squares += value * value;
            }
            let rms = (sum_squares / latent as f32).sqrt();
            if rms > 0.0 {
                for value in out_row.iter_mut() {
                    *value /= rms;
                }
            }
        }
    }

    layer.embed_w.data.copy_from_slice(&projected);
    for buffer in [&mut layer.embed_w.grad, &mut layer.embed_w.m, &mut layer.embed_w.v] {
        buffer.iter_mut().for_each(|value| *value = 0.0);
    }
    layer.unembed_w = layer.embed_w.clone();

    // ---- logit-scale calibration ------------------------------------
    // Identical to the Qwen3 emitter: probe the fresh student with random
    // chunks and rescale the unembedding so cold-start logits sit near unit
    // RMS instead of exploding (docs/qwen-distill.md).
    let probe_chunks = 8usize;
    let seq = layer.cfg.chunk_len;
    let d_v = layer.cfg.d_vocab;
    let h_saved = layer.block.h_persistent.clone();
    let mut probe_rng = SimpleRng::new(seed ^ 0x0ca1_b7ae_9e0b_e7a1);
    let mut sum_sq = 0.0f64;
    for _ in 0..probe_chunks {
        let tokens: Vec<usize> = (0..seq)
            .map(|_| ((probe_rng.gen_range_f32(0.0, 1.0) * d_v as f32) as usize).min(d_v - 1))
            .collect();
        let _loss = crate::gpu_batch::forward_train_chunk_batched(&mut layer, &tokens, &tokens);
        for slot in &layer.block.tape.logits[..seq * d_v] {
            sum_sq += (*slot as f64) * (*slot as f64);
        }
        layer.block.h_persistent.copy_from_slice(&h_saved);
    }
    let logit_rms_before = (sum_sq / ((probe_chunks * seq * d_v) as f64)).sqrt() as f32;
    let logit_scale = if logit_rms_before.is_finite() && logit_rms_before > 0.0 {
        let scale = 1.0 / logit_rms_before;
        for value in layer.unembed_w.data.iter_mut() {
            *value *= scale;
        }
        for buffer in [&mut layer.unembed_w.grad, &mut layer.unembed_w.m, &mut layer.unembed_w.v] {
            buffer.iter_mut().for_each(|value| *value = 0.0);
        }
        scale
    } else {
        1.0
    };

    crate::checkpoint::save_model(&layer, &out_path)
        .map_err(|error| format!("checkpoint save failed: {error}"))?;
    let checkpoint_bytes = std::fs::metadata(&out_path)
        .map_err(|error| format!("checkpoint metadata read failed: {error}"))?
        .len();

    Ok(ConversionReport {
        family_label: info.family.label().to_owned(),
        source_model_type: info.model_type.clone(),
        embedding_tensor: embedding_name,
        vocab_size: vocab,
        hidden_size: hidden,
        depth_requested,
        depth_used,
        latent_dim,
        checkpoint_path: out_path,
        checkpoint_bytes,
        logit_rms_before,
        logit_scale,
        notes,
    })
}

// -------------------------------------------------------------------- CLI

pub const USAGE: &str = "convert <model-or-dir> <config.json> <out.v6> [--target compact|source-width] [--seed N]";

/// Entry point for `pssa convert ...` and the TUI convert action. Prints a
/// human-readable conversion report to stdout.
pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: pssa {USAGE}");
        println!();
        println!("Convert a HuggingFace LLM checkpoint into a PSSA v6 checkpoint.");
        println!("Source may be a single .safetensors file or a model directory with shards.");
        println!("Supported today: Qwen3 (dense), Llama, Mistral, Qwen2, Mixtral, Gemma, Phi-3, GPT-2 style.");
        println!("Newer Qwen generations (3.5/3.6/3.8) use a hybrid scan the converter cannot read yet.");
        return Ok(());
    }

    let mut positional: Vec<String> = Vec::new();
    let mut target_mode = "compact".to_owned();
    let mut seed = 1234u64;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--target" => {
                index += 1;
                target_mode = args
                    .get(index)
                    .cloned()
                    .ok_or("--target needs a value (compact or source-width)")?;
            }
            "--seed" => {
                index += 1;
                let raw = args.get(index).cloned().ok_or("--seed needs a number")?;
                seed = raw.parse().map_err(|_| "--seed needs an integer")?;
            }
            other => positional.push(other.to_owned()),
        }
        index += 1;
    }
    if positional.len() != 3 {
        return Err(format!("usage: pssa {USAGE}"));
    }

    let source = PathBuf::from(&positional[0]);
    let shards = shard_paths(&source)?;
    let config_text = std::fs::read_to_string(&positional[1])
        .map_err(|error| format!("{}: {error}", positional[1]))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|error| format!("{}: malformed JSON: {error}", positional[1]))?;
    let info = detect::detect(&config).map_err(|error| format!("{}: {error}", positional[1]))?;
    println!(
        "detected {} (model_type '{}', {} layers, hidden {}, vocab {})",
        info.family.label(),
        info.model_type,
        info.num_hidden_layers,
        info.hidden_size,
        info.vocab_size
    );

    let report = convert(ConvertOptions {
        shards,
        info,
        target_mode,
        seed,
        out_path: PathBuf::from(&positional[2]),
    })?;
    println!("{}", report.summary());
    Ok(())
}
