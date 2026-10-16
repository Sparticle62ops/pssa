//! Qwen3 configuration, sizing, conversion preflight, and a small
//! inference-only reference path.
//!
//! The existing `TransformerModel` is intentionally a one-layer training
//! baseline and cannot represent a Qwen checkpoint.  This module keeps the
//! Qwen-specific shape contract separate: configuration parsing and resource
//! accounting work with a real Hugging Face `config.json`, while the reference
//! runner is deliberately small enough for an offline smoke test. It is not a
//! replacement for a full safetensors loader or a training implementation.

use crate::checkpoint::allocation_bytes_uncapped;
use crate::linalg::SimpleRng;
use crate::pssa::{PSSAConfigV2, PSSALayerV2};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const SAFETENSORS_HEADER_CAP: usize = 64 * 1024 * 1024;
const SAFETENSORS_READ_CHUNK: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub use_qk_norm: bool,
}

impl Qwen3Config {
    /// The public Qwen3-1.7B shape contract from Hugging Face.
    pub fn qwen3_1_7b() -> Self {
        Self {
            model_type: "qwen3".into(),
            vocab_size: 151_936,
            hidden_size: 2_048,
            intermediate_size: 6_144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 40_960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
            use_qk_norm: true,
        }
    }

    /// Qwen3 calls the smaller public checkpoint 0.6B; it is the usual
    /// stand-in for the earlier "0.5B" smoke path.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1_024,
            intermediate_size: 3_072,
            ..Self::qwen3_1_7b()
        }
    }

    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let root: Value = serde_json::from_str(json)
            .map_err(|err| format!("invalid Qwen3 config JSON: {err}"))?;
        let object = root
            .as_object()
            .ok_or_else(|| "Qwen3 config must be a JSON object".to_string())?;
        let get_usize = |name: &str| -> Result<usize, String> {
            object
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("Qwen3 config field {name} must be a positive integer"))
        };
        let get_f32 = |name: &str| -> Result<f32, String> {
            object
                .get(name)
                .and_then(Value::as_f64)
                .map(|x| x as f32)
                .ok_or_else(|| format!("Qwen3 config field {name} must be a number"))
        };
        let model_type = object
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| "Qwen3 config is missing model_type".to_string())?
            .to_string();
        let hidden_act = object
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu")
            .to_string();
        let tie_word_embeddings = object
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Qwen3's released configs do not serialize this implementation detail;
        // it is true for the architecture.  Honor an explicit field in future
        // configs so an incompatible variant fails during validation.
        let use_qk_norm = object
            .get("use_qk_norm")
            .and_then(Value::as_bool)
            .unwrap_or(model_type == "qwen3");
        let num_attention_heads = get_usize("num_attention_heads")?;
        let head_dim = object
            .get("head_dim")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or_else(|| {
                object
                    .get("hidden_size")
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(0)
                    / num_attention_heads.max(1)
            });
        let config = Self {
            model_type,
            vocab_size: get_usize("vocab_size")?,
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: object
                .get("num_key_value_heads")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(num_attention_heads),
            head_dim,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            rms_norm_eps: get_f32("rms_norm_eps")?,
            rope_theta: get_f32("rope_theta")?,
            hidden_act,
            tie_word_embeddings,
            use_qk_norm,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|err| format!("read Qwen3 config {}: {err}", path.display()))?;
        Self::from_json_str(&json)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "qwen3" {
            return Err(format!(
                "expected Qwen3 model_type, got {}",
                self.model_type
            ));
        }
        if self.vocab_size < 2
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim < 2
            || self.max_position_embeddings == 0
        {
            return Err(
                "Qwen3 dimensions must be positive and vocabulary must contain at least two tokens"
                    .into(),
            );
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "Qwen3 query projection dimension overflow".to_string())?;
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "Qwen3 key/value projection dimension overflow".to_string())?;
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err("Qwen3 attention heads must be divisible by key/value heads".into());
        }
        if self.head_dim % 2 != 0 {
            return Err("Qwen3 head_dim must be even for rotary embeddings".into());
        }
        if self.hidden_act != "silu" {
            return Err(format!(
                "Qwen3 converter requires SwiGLU with hidden_act=silu, got {}",
                self.hidden_act
            ));
        }
        if !self.tie_word_embeddings {
            return Err("Qwen3 converter requires tied input/output embeddings".into());
        }
        if !self.use_qk_norm {
            return Err("Qwen3 converter requires per-head Q/K RMSNorm".into());
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0)
            || !(self.rope_theta.is_finite() && self.rope_theta > 0.0)
        {
            return Err("Qwen3 RMSNorm epsilon and RoPE theta must be finite and positive".into());
        }
        Ok(())
    }

    pub fn query_dim(&self) -> usize {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .expect("validated Qwen3 query dimension")
    }

    pub fn key_value_dim(&self) -> usize {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .expect("validated Qwen3 key/value dimension")
    }

    /// Exact parameter count for the dense Qwen3 tensor families represented by
    /// this module. The count is useful before allocating a real checkpoint.
    pub fn parameter_count(&self) -> Result<usize, String> {
        self.validate()?;
        let embedding = self
            .vocab_size
            .checked_mul(self.hidden_size)
            .ok_or_else(|| "Qwen3 embedding size overflow".to_string())?;
        let attention = self
            .hidden_size
            .checked_mul(self.query_dim())
            .and_then(|n| {
                self.hidden_size
                    .checked_mul(self.key_value_dim())
                    .and_then(|kv| kv.checked_mul(2).and_then(|kv| n.checked_add(kv)))
            })
            .and_then(|n| n.checked_add(self.hidden_size.checked_mul(self.hidden_size)?))
            .ok_or_else(|| "Qwen3 attention size overflow".to_string())?;
        let mlp = 3usize
            .checked_mul(
                self.hidden_size
                    .checked_mul(self.intermediate_size)
                    .ok_or_else(|| "Qwen3 MLP size overflow".to_string())?,
            )
            .ok_or_else(|| "Qwen3 MLP size overflow".to_string())?;
        let per_layer_norms = self
            .hidden_size
            .checked_mul(2)
            .and_then(|n| {
                self.head_dim
                    .checked_mul(2)
                    .and_then(|norms| n.checked_add(norms))
            })
            .ok_or_else(|| "Qwen3 norm size overflow".to_string())?;
        let layers = self
            .num_hidden_layers
            .checked_mul(
                attention
                    .checked_add(mlp)
                    .and_then(|n| n.checked_add(per_layer_norms))
                    .ok_or_else(|| "Qwen3 layer size overflow".to_string())?,
            )
            .and_then(|n| n.checked_add(self.hidden_size))
            .ok_or_else(|| "Qwen3 parameter count overflow".to_string())?;
        embedding
            .checked_add(layers)
            .ok_or_else(|| "Qwen3 parameter count overflow".to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ConversionOptions {
    pub memory_capacity: usize,
    pub memory_key_dim: usize,
    pub memory_value_dim: usize,
    pub chunk_len: usize,
    pub state_dim: usize,
}

impl Default for Qwen3ConversionOptions {
    fn default() -> Self {
        Self {
            memory_capacity: 512,
            memory_key_dim: 32,
            memory_value_dim: 256,
            chunk_len: 64,
            state_dim: 16,
        }
    }
}

/// The dimensions of the PSSA model produced by a conversion.
///
/// Qwen's hidden width and a PSSA block's latent width are independent.  A
/// conversion may therefore use a compact PSSA target (the default latent
/// width is 256), or deliberately keep the source width for an apples-to-
/// apples architectural experiment.  Keeping this target explicit prevents
/// the memory-bank value width from being mistaken for Qwen's KV width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3PssaTarget {
    pub vocab_size: usize,
    pub depth: usize,
    pub latent_dim: usize,
    pub memory_key_dim: usize,
    pub memory_capacity: usize,
    pub chunk_len: usize,
    pub state_dim: usize,
}

impl Qwen3PssaTarget {
    pub fn compact_for(config: &Qwen3Config) -> Self {
        Self {
            vocab_size: config.vocab_size,
            depth: config.num_hidden_layers,
            latent_dim: 256,
            memory_key_dim: 32,
            memory_capacity: 512,
            chunk_len: 64,
            state_dim: 16,
        }
    }

    pub fn source_width_for(config: &Qwen3Config) -> Self {
        Self {
            vocab_size: config.vocab_size,
            depth: config.num_hidden_layers,
            latent_dim: config.hidden_size,
            memory_key_dim: 32,
            memory_capacity: 512,
            chunk_len: 64,
            state_dim: 16,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size < 2
            || !(1..=32).contains(&self.depth)
            || self.latent_dim == 0
            || self.memory_key_dim == 0
            || self.memory_capacity == 0
            || self.chunk_len == 0
            || self.state_dim == 0
        {
            return Err(
                "Qwen3 PSSA target dimensions, capacity, and vocabulary must be positive; depth must be 1..=32"
                    .into(),
            );
        }
        self.chunk_len
            .checked_next_power_of_two()
            .ok_or_else(|| "Qwen3 PSSA target scan length overflows usize".to_string())?;
        Ok(())
    }

    pub fn pssa_config(&self) -> PSSAConfigV2 {
        PSSAConfigV2 {
            depth: self.depth,
            d_vocab: self.vocab_size,
            d_latent: self.latent_dim,
            d_state: self.state_dim,
            d_mem_key: self.memory_key_dim,
            mem_capacity: self.memory_capacity,
            chunk_len: self.chunk_len,
            ..Default::default()
        }
    }
}

impl Qwen3ConversionOptions {
    fn pssa_target(&self, config: &Qwen3Config) -> Qwen3PssaTarget {
        Qwen3PssaTarget {
            vocab_size: config.vocab_size,
            depth: config.num_hidden_layers,
            latent_dim: self.memory_value_dim,
            memory_key_dim: self.memory_key_dim,
            memory_capacity: self.memory_capacity,
            chunk_len: self.chunk_len,
            state_dim: self.state_dim,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ResourceEstimate {
    pub parameter_count: usize,
    pub bf16_weight_bytes: usize,
    pub f32_weight_bytes: usize,
    pub memory_bank_bytes: usize,
    pub scan_len: usize,
    pub scan_workspace_bytes: usize,
    /// Full numeric allocation of the PSSA target, including optimizer state
    /// and its training tape. This is the value checked against the 1 GiB
    /// checkpoint/model allocation cap.
    pub pssa_target_allocation_bytes: usize,
    pub estimated_cpu_conversion_bytes: usize,
    pub estimated_vram_bytes: usize,
}

impl Qwen3ResourceEstimate {
    pub fn for_config(
        config: &Qwen3Config,
        options: &Qwen3ConversionOptions,
    ) -> Result<Self, String> {
        config.validate()?;
        if options.memory_capacity == 0
            || options.memory_key_dim == 0
            || options.memory_value_dim == 0
            || options.chunk_len == 0
            || options.state_dim == 0
        {
            return Err("Qwen3 conversion resource dimensions must be positive".into());
        }
        Self::for_config_with_target(config, &options.pssa_target(config))
    }

    /// Estimate a conversion while keeping source-model and target-model
    /// dimensions separate. In particular, `memory_bank_bytes` uses the
    /// target PSSA latent width because PSSA memory values are `d_latent`.
    pub fn for_config_with_target(
        config: &Qwen3Config,
        target: &Qwen3PssaTarget,
    ) -> Result<Self, String> {
        config.validate()?;
        target.validate()?;
        let parameter_count = config.parameter_count()?;
        let bf16_weight_bytes = parameter_count
            .checked_mul(2)
            .ok_or_else(|| "Qwen3 BF16 weight size overflow".to_string())?;
        let f32_weight_bytes = parameter_count
            .checked_mul(F32_BYTES)
            .ok_or_else(|| "Qwen3 f32 weight size overflow".to_string())?;
        let metadata_bytes = target
            .memory_capacity
            .checked_mul(2 * F32_BYTES + std::mem::size_of::<usize>())
            .ok_or_else(|| "Qwen3 memory metadata size overflow".to_string())?;
        let bank_bytes = target
            .memory_capacity
            .checked_mul(
                target
                    .memory_key_dim
                    .checked_add(target.latent_dim)
                    .ok_or_else(|| "Qwen3 memory width overflow".to_string())?,
            )
            .and_then(|n| n.checked_mul(F32_BYTES))
            .and_then(|n| n.checked_add(metadata_bytes))
            .ok_or_else(|| "Qwen3 memory bank size overflow".to_string())?;
        let memory_bank_bytes = target
            .depth
            .checked_mul(bank_bytes)
            .ok_or_else(|| "Qwen3 stacked memory bank size overflow".to_string())?;
        let scan_len = target
            .chunk_len
            .checked_next_power_of_two()
            .ok_or_else(|| "Qwen3 scan length overflow; reduce chunk length".to_string())?;
        let scan_workspace_bytes = target
            .depth
            .checked_mul(scan_len)
            .and_then(|n| n.checked_mul(target.latent_dim))
            .and_then(|n| n.checked_mul(target.state_dim))
            .and_then(|n| n.checked_mul(2 * F32_BYTES))
            .ok_or_else(|| "Qwen3 scan workspace size overflow".to_string())?;
        let pssa_target_allocation_bytes = allocation_bytes_uncapped(&target.pssa_config())
            .map_err(|err| format!("Qwen3 PSSA target sizing failed: {err}"))?;
        // Conversion keeps BF16 source shards, materializes f32 target tensors,
        // and allocates the complete PSSA target. The estimates intentionally
        // err high for preflight decisions; they are not process RSS measurements.
        let estimated_cpu_conversion_bytes = bf16_weight_bytes
            .checked_add(f32_weight_bytes)
            .and_then(|n| n.checked_add(pssa_target_allocation_bytes))
            .ok_or_else(|| "Qwen3 CPU conversion size overflow".to_string())?;
        let estimated_vram_bytes = f32_weight_bytes
            .checked_add(pssa_target_allocation_bytes)
            .ok_or_else(|| "Qwen3 VRAM conversion size overflow".to_string())?;
        Ok(Self {
            parameter_count,
            bf16_weight_bytes,
            f32_weight_bytes,
            memory_bank_bytes,
            scan_len,
            scan_workspace_bytes,
            pssa_target_allocation_bytes,
            estimated_cpu_conversion_bytes,
            estimated_vram_bytes,
        })
    }

    pub fn exceeds_one_gib(&self) -> bool {
        self.estimated_cpu_conversion_bytes > 1024 * 1024 * 1024
    }
}

#[derive(Clone, Debug)]
struct SafeTensorSpec {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// A small, streaming reader for the subset of safetensors used by Qwen3.
///
/// The header is kept in memory, but tensor payloads are read in bounded
/// chunks.  This matters for the 1.7B checkpoint: reading the whole BF16 file
/// before materializing f32 weights would add another several GiB to the peak
/// CPU conversion footprint.
struct SafeTensorReader {
    file: File,
    data_start: u64,
    tensors: HashMap<String, SafeTensorSpec>,
}

impl SafeTensorReader {
    fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|err| format!("open safetensors {}: {err}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|err| format!("stat safetensors {}: {err}", path.display()))?
            .len();
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|err| format!("read safetensors header length: {err}"))?;
        let header_len = u64::from_le_bytes(length_bytes);
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| "safetensors header length exceeds platform usize".to_string())?;
        if header_len_usize > SAFETENSORS_HEADER_CAP {
            return Err(format!(
                "safetensors header is {header_len_usize} bytes, over {SAFETENSORS_HEADER_CAP} byte cap"
            ));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| "safetensors data offset overflows u64".to_string())?;
        if data_start > file_len {
            return Err("safetensors header extends past end of file".into());
        }
        let mut header = vec![0u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|err| format!("read safetensors header: {err}"))?;
        let object: serde_json::Map<String, Value> = serde_json::from_slice(&header)
            .map_err(|err| format!("invalid safetensors header JSON: {err}"))?;
        let data_len = file_len - data_start;
        let mut tensors = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let tensor = value
                .as_object()
                .ok_or_else(|| format!("safetensors entry {name} is not an object"))?;
            let dtype = tensor
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("safetensors tensor {name} is missing dtype"))?
                .to_owned();
            let shape = tensor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} is missing shape"))?
                .iter()
                .map(|dim| {
                    dim.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| format!("safetensors tensor {name} has an invalid shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} is missing data_offsets"))?;
            if offsets.len() != 2 {
                return Err(format!(
                    "safetensors tensor {name} has invalid data_offsets"
                ));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| format!("safetensors tensor {name} has invalid start offset"))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| format!("safetensors tensor {name} has invalid end offset"))?;
            if start > end || end > data_len {
                return Err(format!(
                    "safetensors tensor {name} data range {start}..{end} exceeds payload {data_len}"
                ));
            }
            let element_count = shape.iter().try_fold(1usize, |count, &dim| {
                count
                    .checked_mul(dim)
                    .ok_or_else(|| format!("safetensors tensor {name} shape overflows usize"))
            })?;
            let item_size = safetensors_item_size(&dtype)?;
            let expected_bytes = element_count
                .checked_mul(item_size)
                .ok_or_else(|| format!("safetensors tensor {name} byte size overflows usize"))?;
            if end - start != u64::try_from(expected_bytes).unwrap_or(u64::MAX) {
                return Err(format!(
                    "safetensors tensor {name} byte range does not match dtype and shape"
                ));
            }
            if tensors
                .insert(
                    name.clone(),
                    SafeTensorSpec {
                        dtype,
                        shape,
                        start,
                        end,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate safetensors tensor {name}"));
            }
        }
        Ok(Self {
            file,
            data_start,
            tensors,
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn stream_values<F>(
        &mut self,
        name: &str,
        expected_shape: &[usize],
        mut visit: F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, f32) -> Result<(), String>,
    {
        let spec = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("safetensors tensor {name} is missing"))?
            .clone();
        if spec.shape != expected_shape {
            return Err(format!(
                "safetensors tensor {name} has shape {:?}, expected {:?}",
                spec.shape, expected_shape
            ));
        }
        let item_size = safetensors_item_size(&spec.dtype)?;
        let element_count = expected_shape.iter().try_fold(1usize, |count, &dim| {
            count
                .checked_mul(dim)
                .ok_or_else(|| format!("safetensors tensor {name} shape overflows usize"))
        })?;
        let byte_count = element_count
            .checked_mul(item_size)
            .ok_or_else(|| format!("safetensors tensor {name} byte size overflows usize"))?;
        debug_assert_eq!(u64::try_from(byte_count).ok(), Some(spec.end - spec.start));
        let offset = self
            .data_start
            .checked_add(spec.start)
            .ok_or_else(|| format!("safetensors tensor {name} file offset overflows u64"))?;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|err| format!("seek safetensors tensor {name}: {err}"))?;
        let chunk_bytes = SAFETENSORS_READ_CHUNK
            .max(item_size)
            .checked_sub(SAFETENSORS_READ_CHUNK.max(item_size) % item_size)
            .unwrap_or(item_size);
        let mut buffer = vec![0u8; chunk_bytes.max(item_size)];
        let mut byte_offset = 0usize;
        while byte_offset < byte_count {
            let amount = (byte_count - byte_offset).min(buffer.len());
            self.file
                .read_exact(&mut buffer[..amount])
                .map_err(|err| format!("read safetensors tensor {name}: {err}"))?;
            for (within_chunk, bytes) in buffer[..amount].chunks_exact(item_size).enumerate() {
                let index = byte_offset / item_size + within_chunk;
                visit(index, decode_safetensors_value(&spec.dtype, bytes)?)?;
            }
            byte_offset += amount;
        }
        Ok(())
    }

    fn read_values(&mut self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>, String> {
        let count = expected_shape.iter().try_fold(1usize, |count, &dim| {
            count
                .checked_mul(dim)
                .ok_or_else(|| format!("safetensors tensor {name} shape overflows usize"))
        })?;
        let mut values = vec![0.0; count];
        self.stream_values(name, expected_shape, |index, value| {
            values[index] = value;
            Ok(())
        })?;
        Ok(values)
    }

    fn verify_values(
        &mut self,
        name: &str,
        expected_shape: &[usize],
        expected: &[f32],
    ) -> Result<(), String> {
        self.stream_values(name, expected_shape, |index, value| {
            if value.to_bits() != expected[index].to_bits() {
                return Err(format!(
                    "safetensors tensor {name} differs from tied embeddings at element {index}"
                ));
            }
            Ok(())
        })
    }
}

fn safetensors_item_size(dtype: &str) -> Result<usize, String> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        "F64" => Ok(8),
        other => Err(format!(
            "Qwen3 safetensors loader supports BF16, F16, F32, and F64; got {other}"
        )),
    }
}

fn decode_safetensors_value(dtype: &str, bytes: &[u8]) -> Result<f32, String> {
    Ok(match dtype {
        "BF16" => f32::from_bits(
            u32::from(u16::from_le_bytes(
                bytes.try_into().expect("BF16 item has two bytes"),
            )) << 16,
        ),
        "F16" => f16_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("F16 item has two bytes"),
        )),
        "F32" => f32::from_le_bytes(bytes.try_into().expect("F32 item has four bytes")),
        "F64" => f64::from_le_bytes(bytes.try_into().expect("F64 item has eight bytes")) as f32,
        _ => return Err(format!("unsupported safetensors dtype {dtype}")),
    })
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = fraction;
                let mut exponent = 127 - 15 + 1;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent as u32 + 127 - 15) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn read_weight_matrix(
    reader: &mut SafeTensorReader,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightMatrix, String> {
    Ok(WeightMatrix {
        rows,
        cols,
        data: reader.read_values(name, &[rows, cols])?,
    })
}

#[derive(Clone, Debug)]
struct WeightMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl WeightMatrix {
    fn random(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let scale = (2.0 / cols.max(1) as f32).sqrt() * 0.1;
        Self {
            rows,
            cols,
            data: (0..rows * cols)
                .map(|_| rng.gen_range_f32(-scale, scale))
                .collect(),
        }
    }

    fn matvec(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(output.len(), self.rows);
        for (row, out) in output.iter_mut().enumerate() {
            *out = self.data[row * self.cols..(row + 1) * self.cols]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
    }
}

#[derive(Clone, Debug)]
struct Qwen3LayerWeights {
    input_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q_proj: WeightMatrix,
    k_proj: WeightMatrix,
    v_proj: WeightMatrix,
    o_proj: WeightMatrix,
    post_attention_norm: Vec<f32>,
    gate_proj: WeightMatrix,
    up_proj: WeightMatrix,
    down_proj: WeightMatrix,
}

/// A deterministic, inference-only Qwen3 reference implementation.
///
/// It intentionally refuses a full-size allocation.  The real 1.7B model is
/// covered by `Qwen3Config` and `Qwen3ResourceEstimate`; this runner is for
/// exercising RMSNorm + GQA + RoPE + Q/K norm + SwiGLU + tied embeddings in
/// ordinary tests without downloading or training a multi-gigabyte model.
#[derive(Clone, Debug)]
pub struct Qwen3Model {
    pub config: Qwen3Config,
    token_embedding: WeightMatrix,
    layers: Vec<Qwen3LayerWeights>,
    final_norm: Vec<f32>,
}

impl Qwen3Model {
    const MAX_REFERENCE_PARAMETERS: usize = 10_000_000;

    pub fn random(config: Qwen3Config, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let parameters = config.parameter_count()?;
        if parameters > Self::MAX_REFERENCE_PARAMETERS {
            return Err(format!(
                "Qwen3 reference runner is limited to {} parameters; use resource estimation for full checkpoints",
                Self::MAX_REFERENCE_PARAMETERS
            ));
        }
        let mut rng = SimpleRng::new(seed);
        let q_dim = config.query_dim();
        let kv_dim = config.key_value_dim();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen3LayerWeights {
                input_norm: vec![1.0; config.hidden_size],
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                q_proj: WeightMatrix::random(q_dim, config.hidden_size, &mut rng),
                k_proj: WeightMatrix::random(kv_dim, config.hidden_size, &mut rng),
                v_proj: WeightMatrix::random(kv_dim, config.hidden_size, &mut rng),
                o_proj: WeightMatrix::random(config.hidden_size, q_dim, &mut rng),
                post_attention_norm: vec![1.0; config.hidden_size],
                gate_proj: WeightMatrix::random(
                    config.intermediate_size,
                    config.hidden_size,
                    &mut rng,
                ),
                up_proj: WeightMatrix::random(
                    config.intermediate_size,
                    config.hidden_size,
                    &mut rng,
                ),
                down_proj: WeightMatrix::random(
                    config.hidden_size,
                    config.intermediate_size,
                    &mut rng,
                ),
            });
        }
        Ok(Self {
            token_embedding: WeightMatrix::random(config.vocab_size, config.hidden_size, &mut rng),
            final_norm: vec![1.0; config.hidden_size],
            config,
            layers,
        })
    }

    /// Load a Qwen3 checkpoint from a Hugging Face safetensors shard.
    ///
    /// Qwen3 stores query projections for all query heads and key/value
    /// projections for only the KV heads.  The matrices are retained at those
    /// native widths; `forward_layer` performs the usual repeated-KV-head
    /// mapping instead of expanding the source tensors during conversion.
    /// Input/output embeddings remain one allocation when the config requests
    /// tied embeddings, even if a redundant `lm_head.weight` tensor is present.
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: Qwen3Config,
    ) -> Result<Self, String> {
        config.validate()?;
        let mut reader = SafeTensorReader::open(path)?;
        let q_dim = config.query_dim();
        let kv_dim = config.key_value_dim();
        let token_embedding = read_weight_matrix(
            &mut reader,
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer_index}");
            layers.push(Qwen3LayerWeights {
                input_norm: reader.read_values(
                    &format!("{prefix}.input_layernorm.weight"),
                    &[config.hidden_size],
                )?,
                q_norm: reader.read_values(
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                )?,
                k_norm: reader.read_values(
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                )?,
                q_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    q_dim,
                    config.hidden_size,
                )?,
                k_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.hidden_size,
                )?,
                v_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.hidden_size,
                )?,
                o_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    config.hidden_size,
                    q_dim,
                )?,
                post_attention_norm: reader.read_values(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[config.hidden_size],
                )?,
                gate_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    config.intermediate_size,
                    config.hidden_size,
                )?,
                up_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    config.intermediate_size,
                    config.hidden_size,
                )?,
                down_proj: read_weight_matrix(
                    &mut reader,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    config.hidden_size,
                    config.intermediate_size,
                )?,
            });
        }
        let final_norm = reader.read_values("model.norm.weight", &[config.hidden_size])?;
        if reader.has_tensor("lm_head.weight") {
            if config.tie_word_embeddings {
                reader.verify_values(
                    "lm_head.weight",
                    &[config.vocab_size, config.hidden_size],
                    &token_embedding.data,
                )?;
            } else {
                return Err("Qwen3 converter requires tied input/output embeddings".into());
            }
        }
        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
        })
    }

    /// Alias retained for callers that name the source format rather than the
    /// constructor operation.
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        config: Qwen3Config,
    ) -> Result<Self, String> {
        Self::from_safetensors(path, config)
    }

    pub fn forward_logits(&self, tokens: &[usize]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("Qwen3 forward requires at least one token".into());
        }
        if tokens.len() > self.config.max_position_embeddings {
            return Err(format!(
                "Qwen3 sequence length {} exceeds max position {}",
                tokens.len(),
                self.config.max_position_embeddings
            ));
        }
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| token >= self.config.vocab_size)
        {
            return Err(format!("Qwen3 token {token} is outside vocabulary"));
        }
        let hidden_size = self.config.hidden_size;
        let mut hidden = vec![0.0; tokens.len() * hidden_size];
        for (position, &token) in tokens.iter().enumerate() {
            hidden[position * hidden_size..(position + 1) * hidden_size].copy_from_slice(
                &self.token_embedding.data[token * hidden_size..(token + 1) * hidden_size],
            );
        }
        for layer in &self.layers {
            hidden = self.forward_layer(layer, &hidden, tokens.len());
        }
        let last = &hidden[(tokens.len() - 1) * hidden_size..];
        let normalized = rms_norm(last, &self.final_norm, self.config.rms_norm_eps);
        let mut logits = vec![0.0; self.config.vocab_size];
        for (token, logit) in logits.iter_mut().enumerate() {
            *logit = self.token_embedding.data[token * hidden_size..(token + 1) * hidden_size]
                .iter()
                .zip(&normalized)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        if logits.iter().all(|value| value.is_finite()) {
            Ok(logits)
        } else {
            Err("Qwen3 forward produced a non-finite logit".into())
        }
    }

    pub fn generate_greedy(&self, prompt: &[usize], steps: usize) -> Result<Vec<usize>, String> {
        if prompt.is_empty() {
            return Err("Qwen3 generation requires a non-empty prompt".into());
        }
        let mut output = prompt.to_vec();
        for _ in 0..steps {
            let logits = self.forward_logits(&output)?;
            let next = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(token, _)| token)
                .ok_or_else(|| "Qwen3 vocabulary is empty".to_string())?;
            output.push(next);
        }
        Ok(output)
    }

    fn forward_layer(
        &self,
        layer: &Qwen3LayerWeights,
        input: &[f32],
        sequence_len: usize,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.query_dim();
        let kv_dim = self.config.key_value_dim();
        let head_dim = self.config.head_dim;
        let mut queries = vec![0.0; sequence_len * q_dim];
        let mut keys = vec![0.0; sequence_len * kv_dim];
        let mut values = vec![0.0; sequence_len * kv_dim];
        for position in 0..sequence_len {
            let x = &input[position * hidden_size..(position + 1) * hidden_size];
            let normalized = rms_norm(x, &layer.input_norm, self.config.rms_norm_eps);
            layer.q_proj.matvec(
                &normalized,
                &mut queries[position * q_dim..(position + 1) * q_dim],
            );
            layer.k_proj.matvec(
                &normalized,
                &mut keys[position * kv_dim..(position + 1) * kv_dim],
            );
            layer.v_proj.matvec(
                &normalized,
                &mut values[position * kv_dim..(position + 1) * kv_dim],
            );
            for head in 0..self.config.num_attention_heads {
                let start = position * q_dim + head * head_dim;
                rms_norm_in_place(
                    &mut queries[start..start + head_dim],
                    &layer.q_norm,
                    self.config.rms_norm_eps,
                );
                apply_rope(
                    &mut queries[start..start + head_dim],
                    position,
                    head_dim,
                    self.config.rope_theta,
                );
            }
            for head in 0..self.config.num_key_value_heads {
                let start = position * kv_dim + head * head_dim;
                rms_norm_in_place(
                    &mut keys[start..start + head_dim],
                    &layer.k_norm,
                    self.config.rms_norm_eps,
                );
                apply_rope(
                    &mut keys[start..start + head_dim],
                    position,
                    head_dim,
                    self.config.rope_theta,
                );
            }
        }

        let mut output = vec![0.0; sequence_len * hidden_size];
        let group_size = self.config.num_attention_heads / self.config.num_key_value_heads;
        let scale = (head_dim as f32).sqrt().recip();
        for position in 0..sequence_len {
            let mut attended = vec![0.0; q_dim];
            for q_head in 0..self.config.num_attention_heads {
                let q = &queries[position * q_dim + q_head * head_dim
                    ..position * q_dim + (q_head + 1) * head_dim];
                let kv_head = q_head / group_size;
                let mut scores = vec![0.0; position + 1];
                for previous in 0..=position {
                    let k = &keys[previous * kv_dim + kv_head * head_dim
                        ..previous * kv_dim + (kv_head + 1) * head_dim];
                    scores[previous] = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                softmax_in_place(&mut scores);
                let context = &mut attended[q_head * head_dim..(q_head + 1) * head_dim];
                for previous in 0..=position {
                    let v = &values[previous * kv_dim + kv_head * head_dim
                        ..previous * kv_dim + (kv_head + 1) * head_dim];
                    for (out, &value) in context.iter_mut().zip(v) {
                        *out += scores[previous] * value;
                    }
                }
            }
            let mut attention_output = vec![0.0; hidden_size];
            layer.o_proj.matvec(&attended, &mut attention_output);
            let input_row = &input[position * hidden_size..(position + 1) * hidden_size];
            let mut residual = input_row.to_vec();
            for (value, update) in residual.iter_mut().zip(&attention_output) {
                *value += update;
            }
            let normalized = rms_norm(
                &residual,
                &layer.post_attention_norm,
                self.config.rms_norm_eps,
            );
            let mut gate = vec![0.0; self.config.intermediate_size];
            let mut up = vec![0.0; self.config.intermediate_size];
            layer.gate_proj.matvec(&normalized, &mut gate);
            layer.up_proj.matvec(&normalized, &mut up);
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value = silu(*gate_value) * up_value;
            }
            let mut mlp_output = vec![0.0; hidden_size];
            layer.down_proj.matvec(&gate, &mut mlp_output);
            let output_row = &mut output[position * hidden_size..(position + 1) * hidden_size];
            for (((out, old), attention), update) in output_row
                .iter_mut()
                .zip(input_row)
                .zip(&attention_output)
                .zip(mlp_output)
            {
                *out = *old + *attention + update;
            }
        }
        output
    }
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut output = input.to_vec();
    rms_norm_in_place(&mut output, weight, eps);
    output
}

fn rms_norm_in_place(input: &mut [f32], weight: &[f32], eps: f32) {
    let mean_square = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for (value, multiplier) in input.iter_mut().zip(weight) {
        *value *= scale * multiplier;
    }
}

fn apply_rope(input: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    // Qwen3 follows the Hugging Face split-half rotary layout: the first half
    // of a head rotates against the second half. This is intentionally not the
    // adjacent-pair layout used by several older decoder implementations.
    let original = input.to_vec();
    let half = head_dim / 2;
    for pair in 0..half {
        let frequency = theta.powf(-((2 * pair) as f32) / head_dim as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let first = original[pair];
        let second = original[pair + half];
        input[pair] = first * cos - second * sin;
        input[pair + half] = first * sin + second * cos;
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    let reciprocal = total.recip();
    for value in values {
        *value *= reciprocal;
    }
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Summary of a completed Qwen3-to-PSSA emission, for logs and task notes.
#[derive(Clone, Debug)]
pub struct Qwen3EmissionReport {
    pub target: Qwen3PssaTarget,
    pub estimate: Qwen3ResourceEstimate,
    pub checkpoint_path: std::path::PathBuf,
    pub checkpoint_bytes: u64,
}

/// Emit a PSSA v6 checkpoint laid out for the requested target.
///
/// PSSA's scan, memory-bank, and adapter weights have no direct Qwen3
/// counterpart, so they keep their seeded initialization. What transfers is
/// the token table: every Qwen3 embedding row is pushed through a fixed
/// seeded projection into the target latent width and rescaled to unit RMS,
/// and the PSSA unembedding starts from the same projected table, so the
/// tied input/output relationship survives the width change.
pub fn emit_pssa_checkpoint(
    model: &Qwen3Model,
    target: &Qwen3PssaTarget,
    out_path: impl AsRef<Path>,
    seed: u64,
) -> Result<Qwen3EmissionReport, String> {
    target.validate()?;
    if model.config.vocab_size != target.vocab_size {
        return Err(format!(
            "target vocab {} does not match the source checkpoint's {}",
            target.vocab_size, model.config.vocab_size
        ));
    }
    let estimate = Qwen3ResourceEstimate::for_config_with_target(&model.config, target)?;
    let mut layer = PSSALayerV2::new(target.pssa_config(), seed);

    let hidden = model.config.hidden_size;
    let vocab = target.vocab_size;
    let latent = target.latent_dim;
    // Fixed seeded projection, row-major [latent][hidden], so each dot
    // product below reads one contiguous source slice per output slot.
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
    for token in 0..vocab {
        let source_row = &model.token_embedding.data[token * hidden..(token + 1) * hidden];
        let out_row = &mut projected[token * latent..(token + 1) * latent];
        for (j, slot) in out_row.iter_mut().enumerate() {
            let proj_row = &projection[j * hidden..(j + 1) * hidden];
            let mut acc = 0.0f32;
            for (source, weight) in source_row.iter().zip(proj_row.iter()) {
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

    layer.embed_w.data.copy_from_slice(&projected);
    for buffer in [
        &mut layer.embed_w.grad,
        &mut layer.embed_w.m,
        &mut layer.embed_w.v,
    ] {
        buffer.iter_mut().for_each(|value| *value = 0.0);
    }
    // Qwen3 ties its input and output embeddings; mirror that in the
    // projected latent space so converted logits keep the same geometry.
    layer.unembed_w = layer.embed_w.clone();

    let out_path = out_path.as_ref();
    // V6/V7 cannot serialize stacked (depth > 1) models; save_model picks
    // V7 for depth one and V8 for stacks automatically.
    crate::checkpoint::save_model(&layer, out_path)
        .map_err(|error| format!("checkpoint save failed: {error}"))?;
    let checkpoint_bytes = std::fs::metadata(out_path)
        .map_err(|error| format!("checkpoint metadata read failed: {error}"))?
        .len();
    Ok(Qwen3EmissionReport {
        target: target.clone(),
        estimate,
        checkpoint_path: out_path.to_path_buf(),
        checkpoint_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_uses_qwen_split_half_layout() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 4, 1.0);
        let (sin, cos) = 1.0f32.sin_cos();
        let expected = [
            cos - 3.0 * sin,
            2.0 * cos - 4.0 * sin,
            sin + 3.0 * cos,
            2.0 * sin + 4.0 * cos,
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn tied_logits_do_not_apply_an_extra_hidden_width_scale() {
        let config = Qwen3Config {
            model_type: "qwen3".into(),
            vocab_size: 2,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
            use_qk_norm: true,
        };
        let eps = config.rms_norm_eps;
        let mut model = Qwen3Model::random(config, 1).unwrap();
        model
            .token_embedding
            .data
            .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        for layer in &mut model.layers {
            for matrix in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ] {
                matrix.data.fill(0.0);
            }
        }
        let logits = model.forward_logits(&[0]).unwrap();
        let expected = 1.0 / (0.5 + eps).sqrt();
        assert!((logits[0] - expected).abs() < 1e-5);
        assert!(
            logits[0] > 1.2,
            "unexpected scaled tied logit: {}",
            logits[0]
        );
    }
}
