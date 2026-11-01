//! Architecture detection for the universal converter.
//!
//! Reads a HuggingFace `config.json` and decides (1) which weight-naming
//! family the checkpoint belongs to and (2) whether the converter can read
//! it at all. The converter transplants the token table (embeddings), so a
//! family only needs an identifiable embedding tensor; hybrid and exotic
//! architectures are rejected with a message that names what was found.

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Qwen3Dense,
    Llama,
    Mistral,
    Qwen2,
    Mixtral,
    Gemma,
    Phi3,
    Gpt2,
}

impl Family {
    pub fn label(self) -> &'static str {
        match self {
            Family::Qwen3Dense => "Qwen3 (dense)",
            Family::Llama => "Llama",
            Family::Mistral => "Mistral",
            Family::Qwen2 => "Qwen2",
            Family::Mixtral => "Mixtral (MoE)",
            Family::Gemma => "Gemma",
            Family::Phi3 => "Phi-3",
            Family::Gpt2 => "GPT-2 style",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArchInfo {
    pub family: Family,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub tie_word_embeddings: bool,
    pub mixture_of_experts: bool,
}

fn field_usize(obj: &Value, name: &str) -> Result<usize, String> {
    obj.get(name)
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .ok_or_else(|| format!("config.json is missing a usable integer '{name}'"))
}

fn field_bool(obj: &Value, name: &str) -> bool {
    obj.get(name).and_then(|value| value.as_bool()).unwrap_or(false)
}

/// Decide the family and pull the sizes the converter needs. Multimodal
/// wrappers (Qwen3.5 and later) nest the text stack under `text_config`;
/// those are detected and rejected here so the user sees a clear reason.
pub fn detect(config: &Value) -> Result<ArchInfo, String> {
    let text = config.get("text_config").filter(|value| value.is_object()).unwrap_or(config);

    if let Some(layer_types) = text.get("layer_types").and_then(|value| value.as_array()) {
        let hybrid = layer_types
            .iter()
            .any(|value| value.as_str() == Some("linear_attention"));
        if hybrid {
            return Err(
                "hybrid linear-attention architecture (Qwen3.5/3.6/3.8 family): this converter \
                 transplants token tables only and cannot read the scan/state layers these \
                 checkpoints use. Use a classic dense checkpoint (Qwen3, Llama, Mistral, Qwen2, \
                 Gemma, Phi-3) or distill from a dense teacher."
                    .into(),
            );
        }
    }

    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .or_else(|| text.get("model_type").and_then(|value| value.as_str()))
        .unwrap_or("")
        .to_owned();

    let family = match model_type.as_str() {
        "qwen3" | "qwen3_moe" => Family::Qwen3Dense,
        "llama" => Family::Llama,
        "mistral" => Family::Mistral,
        "qwen2" => Family::Qwen2,
        "mixtral" => Family::Mixtral,
        "gemma" | "gemma2" | "gemma3" | "gemma3_text" => Family::Gemma,
        "phi3" | "phi" => Family::Phi3,
        "gpt2" | "gpt_neo" | "gpt_neox" => Family::Gpt2,
        other => {
            return Err(format!(
                "unsupported model_type '{other}'. Supported today: qwen3 (dense), llama, \
                 mistral, qwen2, mixtral, gemma, phi3, gpt2-style. Newer Qwen generations \
                 (3.5/3.6/3.8) are hybrid linear-attention models and are not loadable yet."
            ));
        }
    };

    let mixture_of_experts = text.get("num_local_experts").is_some()
        || text.get("num_experts").is_some()
        || model_type.contains("moe");

    Ok(ArchInfo {
        family,
        model_type,
        vocab_size: field_usize(text, "vocab_size")?,
        hidden_size: field_usize(text, "hidden_size")?,
        num_hidden_layers: field_usize(text, "num_hidden_layers")?,
        tie_word_embeddings: field_bool(text, "tie_word_embeddings"),
        mixture_of_experts,
    })
}
