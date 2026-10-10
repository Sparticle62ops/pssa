use super::*;
use crate::pssa::PSSAConfigV2;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture() -> (PSSALayerV2, Tokenizer) {
    let vocabulary = ["<unk>", "a", "b", "c", "d"].map(str::to_owned).to_vec();
    let tokenizer = Tokenizer::from_vocabulary(&vocabulary).unwrap();
    let mut model = PSSALayerV2::new_with_depth_and_loops(PSSAConfigV2 {
        d_vocab: 5, d_latent: 8, d_state: 2, d_mem_key: 2,
        mem_capacity: 2, chunk_len: 2, ..Default::default()
    }, 11, 2, 3);
    model.vocabulary = vocabulary;
    model.unembed_w.data.fill(0.0);
    (model, tokenizer)
}

#[test]
fn prepared_card_has_known_ce_exact_targets_and_restores_all_loop_carries() {
    let (mut model, tokenizer) = fixture();
    let before: Vec<_> = (0..model.recurrent_state_len()).map(|i| i as f32 + 0.5).collect();
    model.copy_recurrent_state_from(&before);
    let raw = "a b c d\nb c d a\nc d a b\n";
    let slice = EvaluationSlice { skip_tokens: 4, max_tokens: Some(8) };
    let first = score_model(&mut model, &tokenizer, raw, slice).unwrap();
    let second = score_model(&mut model, &tokenizer, raw, slice).unwrap();
    let mut after = vec![0.0; before.len()];
    model.copy_recurrent_state_to(&mut after);
    assert_eq!(after, before);
    assert_eq!(first["identity"], second["identity"]);
    assert_eq!(first["identity"]["targets"], 6);
    assert_eq!(first["identity"]["encoded_tokens"], 8);
    assert_eq!(first["protocol"]["loops"], 3);
    assert_eq!(first["model"]["optimizer_updates"], 0);
    assert_eq!(first["metrics"]["cross_entropy"], second["metrics"]["cross_entropy"]);
    assert!((first["metrics"]["cross_entropy"].as_f64().unwrap() - 5.0f64.ln()).abs() < 1e-6);
    assert!((first["metrics"]["perplexity"].as_f64().unwrap() - 5.0).abs() < 1e-5);
    assert_eq!(first["timing"]["seconds_sorted"].as_array().unwrap().len(), 3);
    model.unembed_w.data[0] = f32::NAN;
    assert!(score_model(&mut model, &tokenizer, raw, slice).is_err());
    model.copy_recurrent_state_to(&mut after);
    assert_eq!(after, before);
}

#[test]
fn invalid_windows_and_mismatched_tokenizer_are_rejected() {
    let (mut model, tokenizer) = fixture();
    for (skip_tokens, max_tokens) in [(0, None), (0, Some(0)), (3, Some(9)), (usize::MAX, Some(2)), (0, Some(1))] {
        assert!(score_model(&mut model, &tokenizer, "a b c d", EvaluationSlice { skip_tokens, max_tokens }).is_err());
    }
    model.vocabulary.swap(1, 2);
    assert!(score_model(&mut model, &tokenizer, "a b c d", EvaluationSlice { skip_tokens: 0, max_tokens: Some(4) })
        .unwrap_err().contains("mismatch"));
}

#[test]
fn checkpoint_cli_is_read_only_and_scores_explicit_window() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("pssa-paired-card-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&dir).unwrap();
    let data = dir.join("data.txt");
    let path = dir.join("model.pssa");
    let (model, _) = fixture();
    checkpoint::save_model(&model, &path).unwrap();
    std::fs::write(&data, "a b c d\na b c d\n").unwrap();
    let before = std::fs::read(&path).unwrap();
    crate::cli::CLIHandler::parse_and_execute([
        "pssa", "eval-card", data.to_str().unwrap(), "--model", path.to_str().unwrap(),
        "--skip-tokens", "0", "--max-tokens", "8", "--loops", "3",
    ].map(str::to_owned).to_vec()).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn canonical_bpe_identity_survives_reload() {
    let tokenizer = Tokenizer::from_corpus_bpe("alpha beta gamma alpha beta", 270).unwrap();
    let loaded = Tokenizer::from_serialized(&tokenizer.serialized_metadata().unwrap()).unwrap();
    assert_eq!(tokenizer_identity(&tokenizer).unwrap(), tokenizer_identity(&loaded).unwrap());
}
