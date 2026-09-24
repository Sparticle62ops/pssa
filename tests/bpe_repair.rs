use oxide_ai_pssa::checkpoint::{self, CheckpointFormat};
use oxide_ai_pssa::cli::{CLIHandler, TrainingOptions};
use oxide_ai_pssa::dataset::{Tokenizer, TokenizerKind};
use oxide_ai_pssa::inference::{InferenceConfig, PSSAInferenceEngine};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "oxide-bpe-{name}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
fn checksum(bytes: &mut [u8]) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in &bytes[22..] {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    bytes[14..22].copy_from_slice(&h.to_le_bytes());
}

#[test]
fn bpe_preserves_unseen_unicode_bytes_and_serializes_exactly() {
    let corpus = "Hello, WORLD!\r\nbyte pairs learn from this tiny corpus.\n";
    let tokenizer = Tokenizer::from_corpus_bpe(corpus, 320).unwrap();
    let probes = [
        "Hello, WORLD!\r\n",
        "unseen_nonce_٩ 😀\n",
        "mixed CASE\tpunctuation?!",
    ];
    for probe in probes {
        let ids = tokenizer.try_encode(probe, true).unwrap();
        assert!(!ids.is_empty());
        assert!(
            ids.iter().all(|&id| id != 0),
            "unexpected <unk> for {probe:?}"
        );
        assert_eq!(tokenizer.decode(&ids), probe);
    }
    let json = tokenizer.serialized_metadata().unwrap();
    let restored = Tokenizer::from_serialized(&json).unwrap();
    assert_eq!(
        restored.ordered_vocabulary().unwrap(),
        tokenizer.ordered_vocabulary().unwrap()
    );
    assert_eq!(
        restored.try_encode(probes[1], true).unwrap(),
        tokenizer.try_encode(probes[1], true).unwrap()
    );
    let second = Tokenizer::from_corpus_bpe(corpus, 320).unwrap();
    assert_eq!(
        second.try_encode(probes[2], true).unwrap(),
        tokenizer.try_encode(probes[2], true).unwrap()
    );
}

#[test]
fn v7_bpe_roundtrip_rejects_metadata_mismatch_and_v6_fixture_loads() {
    let corpus = "alpha beta gamma\nalpha beta gamma\n";
    let tokenizer = Tokenizer::from_corpus_bpe(corpus, 300).unwrap();
    let mut model = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: tokenizer.vocab_size,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 2,
            chunk_len: 2,
            lr: 0.01,
            ..Default::default()
        },
        8,
    );
    model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
    model.tokenizer_json = tokenizer.serialized_metadata();
    let p = temp("v7.pssa");
    checkpoint::save_model(&model, &p).unwrap();
    let loaded = checkpoint::load_checkpoint(&p).unwrap();
    assert_eq!(loaded.format, CheckpointFormat::V7);
    assert_eq!(loaded.model.vocabulary, model.vocabulary);
    assert_eq!(loaded.model.tokenizer_json, model.tokenizer_json);
    let mut wrong = model;
    wrong.vocabulary.swap(1, 2);
    assert!(checkpoint::save_model(&wrong, temp("wrong-save")).is_err());
    let mut corrupt = fs::read(&p).unwrap();
    corrupt.pop();
    let q = temp("truncated.pssa");
    fs::write(&q, corrupt).unwrap();
    assert!(checkpoint::load_checkpoint(&q).is_err());
    let fixture = checkpoint::load_checkpoint("tests/fixtures/v6-word-pre-bpe.pssa").unwrap();
    assert_eq!(fixture.format, CheckpointFormat::V6);
    assert!(fixture.model.tokenizer_json.is_none());
    fs::remove_file(p).unwrap();
    fs::remove_file(q).unwrap();
}

#[test]
fn bpe_generation_streams_display_text_not_bytelevel_labels() {
    let tokenizer = Tokenizer::from_corpus_bpe("hello world\nhello world\n", 300).unwrap();
    let mut model = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: tokenizer.vocab_size,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 2,
            chunk_len: 2,
            lr: 0.01,
            ..Default::default()
        },
        5,
    );
    model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
    model.tokenizer_json = tokenizer.serialized_metadata();
    model.embed_w.data.fill(0.);
    model.a_mat.data.fill(0.);
    model.w_delta.data.fill(0.);
    model.w_b.data.fill(0.);
    model.w_c.data.fill(0.);
    model.w_qx.data.fill(0.);
    model.w_qh.data.fill(0.);
    model.w_gate.data.fill(0.);
    model.w_proj.data.fill(0.);
    model.mlp_w1.data.fill(0.);
    model.mlp_w2.data.fill(0.);
    model.unembed_w.data.fill(0.);
    let mut segments = Vec::new();
    let out = PSSAInferenceEngine::try_new(&mut model, &tokenizer)
        .unwrap()
        .try_generate_chat_turn(
            "hello",
            &InferenceConfig {
                temperature: 0.0,
                max_new_tokens: 3,
                ..Default::default()
            },
            |s| segments.push(s.to_owned()),
        )
        .unwrap();
    assert!(!out.contains('Ġ'));
    assert!(!segments.join("").contains('Ġ'));
}

#[test]
fn fresh_bpe_cli_checkpoint_generates_without_corpus() {
    let raw = "alpha beta gamma delta\nalpha beta gamma delta\n";
    let opts = TrainingOptions {
        epochs: 1,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 8,
        lr: 0.01,
        accumulate: 1,
        warmup_steps: 0,
        seed: 3,
        max_tokens: None,
        tokenizer: TokenizerKind::Bpe,
        vocab_size: 300,
        resume: None,
        skip_tokens: 0,
    };
    let (model, _) = CLIHandler::train_corpus(raw, &opts).unwrap();
    let p = temp("fresh.pssa");
    CLIHandler::save_model_v2(&model, p.to_str().unwrap()).unwrap();
    let out = CLIHandler::run_generate("alpha 😀", p.to_str().unwrap(), None, 0.0, 2).unwrap();
    assert!(!out.contains('Ġ'));
    fs::remove_file(p).unwrap();
}

#[test]
fn v7_rejects_policy_json_and_ordered_vocab_tampering() {
    let tokenizer = Tokenizer::from_corpus_bpe("alpha beta gamma\n", 300).unwrap();
    let mut model = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: tokenizer.vocab_size,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 2,
            chunk_len: 2,
            lr: 0.01,
            ..Default::default()
        },
        4,
    );
    model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
    model.tokenizer_json = tokenizer.serialized_metadata();
    let p = temp("tamper-good.pssa");
    checkpoint::save_model(&model, &p).unwrap();
    let good = fs::read(&p).unwrap();
    let marker = b"\"type\":\"BPE\"";
    let pos = good
        .windows(marker.len())
        .position(|w| w == marker)
        .unwrap();
    let mut wrong_policy = good.clone();
    wrong_policy[pos + 8] = b'X';
    checksum(&mut wrong_policy);
    let q = temp("tamper-policy.pssa");
    fs::write(&q, wrong_policy).unwrap();
    assert!(checkpoint::load_checkpoint(&q).is_err());
    let (from, to) = model
        .vocabulary
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(i, from)| {
            model
                .vocabulary
                .iter()
                .skip(i + 1)
                .find(|to| to.len() == from.len() && *to != from)
                .map(|to| (from, to))
        })
        .expect("BPE alphabet has equal-width tokens");
    let mut needle = (from.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(from.as_bytes());
    let offset = good
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    let mut wrong_vocab = good;
    wrong_vocab[offset + 8..offset + 8 + from.len()].copy_from_slice(to.as_bytes());
    checksum(&mut wrong_vocab);
    let r = temp("tamper-vocab.pssa");
    fs::write(&r, wrong_vocab).unwrap();
    assert!(checkpoint::load_checkpoint(&r).is_err());
    fs::remove_file(p).unwrap();
    fs::remove_file(q).unwrap();
    fs::remove_file(r).unwrap();
}
