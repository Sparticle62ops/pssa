//! Weight-name mapping tables for the universal converter.
//!
//! The converter transplants the token table (see docs/qwen-distill.md for
//! the math), so the only tensor it needs from the source checkpoint is the
//! input embedding. Families name it differently; these tables list the
//! candidates in priority order, plus a per-family note rendered in the
//! conversion report.

use super::detect::Family;

/// Candidate tensor names for the input embedding, in priority order.
pub fn embedding_candidates(family: Family) -> Vec<&'static str> {
    match family {
        Family::Gpt2 => vec!["transformer.wte.weight", "wte.weight"],
        _ => vec![
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
        ],
    }
}

/// One-line description of what a conversion transfers for this family.
pub fn family_note(family: Family) -> &'static str {
    match family {
        Family::Qwen3Dense => {
            "Qwen3 dense checkpoint: token table transplants; scan, memory bank and adapter keep seeded initialization; logit scale calibrated on the emitted student."
        }
        Family::Mixtral => {
            "Mixture-of-experts checkpoint: only the token table transfers; expert routing weights are not represented in the PSSA student."
        }
        Family::Gemma => {
            "Gemma checkpoint: token table transplants (Gemma's embedding scaling is neutralized by the per-row unit-RMS rescale)."
        }
        _ => {
            "Dense decoder checkpoint: token table transplants; every other PSSA component keeps seeded initialization; logit scale calibrated on the emitted student."
        }
    }
}
