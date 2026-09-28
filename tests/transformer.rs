use oxide_ai_pssa::checkpoint;
use oxide_ai_pssa::cli::{CLIHandler, TrainingOptions};
use oxide_ai_pssa::dataset::Tokenizer;
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2, ParamMatrix, ParamVector};
use oxide_ai_pssa::training::{Schedule, chunk_plan};
use oxide_ai_pssa::transformer::{TransformerConfig, TransformerModel};
use oxide_ai_pssa::transformer_checkpoint as ck;
use std::{fs, path::PathBuf, process::Command};

fn tiny() -> TransformerModel {
    let mut m = TransformerModel::new(
        TransformerConfig {
            d_vocab: 7,
            d_model: 4,
            n_heads: 2,
            d_ff: 5,
            chunk_len: 4,
            lr: 0.02,
            ..Default::default()
        },
        19,
    )
    .unwrap();
    m.vocabulary = ["<unk>", "a", "b", "c", "d", "e", "f"]
        .map(str::to_string)
        .to_vec();
    m
}
fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oxide-transformer-{name}-{}", std::process::id()))
}
fn matrices(m: &TransformerModel) -> [&ParamMatrix; 6] {
    [
        &m.token_embed,
        &m.qkv,
        &m.out_proj,
        &m.ff1,
        &m.ff2,
        &m.unembed,
    ]
}
fn matrix(m: &mut TransformerModel, i: usize) -> &mut ParamMatrix {
    match i {
        0 => &mut m.token_embed,
        1 => &mut m.qkv,
        2 => &mut m.out_proj,
        3 => &mut m.ff1,
        4 => &mut m.ff2,
        5 => &mut m.unembed,
        _ => unreachable!(),
    }
}
fn vector(m: &mut TransformerModel, i: usize) -> &mut ParamVector {
    match i {
        0 => &mut m.norm1_gamma,
        1 => &mut m.norm1_beta,
        2 => &mut m.norm2_gamma,
        3 => &mut m.norm2_beta,
        _ => unreachable!(),
    }
}
fn loss(m: &mut TransformerModel) -> f32 {
    // Repeated embedding IDs test gradient scatter-add; len < chunk tests strides.
    m.forward_train_chunk(&[1, 2, 1], &[2, 1, 3])
}
fn step(m: &mut TransformerModel, lr: f32) {
    m.zero_gradients();
    loss(m);
    m.backward_chunk(3, 1.0);
    m.apply_adamw(lr).unwrap();
}

#[test]
fn default_sizes_are_close_and_count_only_trainable_scalars() {
    let pssa = PSSALayerV2::new(
        PSSAConfigV2 {
            d_vocab: 2048,
            ..Default::default()
        },
        42,
    );
    let transformer = TransformerModel::new(TransformerConfig::default(), 42).unwrap();
    assert_eq!(pssa.parameter_count(), 1_544_704);
    assert_eq!(transformer.parameter_count(), 1_541_120);
    let relative = pssa
        .parameter_count()
        .abs_diff(transformer.parameter_count()) as f64
        / pssa.parameter_count() as f64;
    assert!(relative < 0.003);
}

#[test]
fn every_transformer_parameter_passes_central_differences() {
    let mut m = tiny();
    loss(&mut m);
    m.zero_gradients();
    m.backward_chunk(3, 1.0);
    let h = 0.002;
    let mut checked = 0;
    for family in 0..10 {
        let (data, grads) = if family < 6 {
            let p = matrix(&mut m, family);
            (p.data.clone(), p.grad.clone())
        } else {
            let p = vector(&mut m, family - 6);
            (p.data.clone(), p.grad.clone())
        };
        assert!(
            grads.iter().any(|x| x.abs() > 1e-6),
            "vacuous family {family}"
        );
        for (i, (&x, &analytic)) in data.iter().zip(&grads).enumerate() {
            let set = |m: &mut TransformerModel, x| {
                if family < 6 {
                    matrix(m, family).data[i] = x;
                } else {
                    vector(m, family - 6).data[i] = x;
                }
            };
            set(&mut m, x + h);
            let plus = loss(&mut m);
            set(&mut m, x - h);
            let minus = loss(&mut m);
            set(&mut m, x);
            let numeric = (plus - minus) / (2.0 * h);
            let tolerance = 1.5e-4 + 0.02 * numeric.abs().max(analytic.abs());
            assert!(
                (numeric - analytic).abs() < tolerance,
                "family {family} element {i}: numeric={numeric}, analytic={analytic}, tolerance={tolerance}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, m.parameter_count());
}

#[test]
fn causal_mask_short_chunks_and_inference_agree() {
    let mut m = tiny();
    let v = m.cfg.d_vocab;
    m.forward_train_chunk(&[1, 2, 3, 4], &[2, 3, 4, 5]);
    let first = m.training_logits().to_vec();
    m.forward_train_chunk(&[1, 2, 6, 5], &[6, 6, 6, 6]);
    assert_eq!(&first[..2 * v], &m.training_logits()[..2 * v]);
    m.forward_train_chunk(&[1, 2], &[2, 3]);
    assert_eq!(&first[..2 * v], m.training_logits());
    let mut out = vec![0.; v];
    m.logits_for_context(&[1, 2], &mut out).unwrap();
    assert_eq!(&out, &first[v..2 * v]);
    let mut truncated = vec![0.; v];
    m.logits_for_context(&[6, 1, 2, 3, 4], &mut out).unwrap();
    m.logits_for_context(&[1, 2, 3, 4], &mut truncated).unwrap();
    assert_eq!(out, truncated);
    // Earlier tokens really affect the final prediction (not independent MLPs).
    m.logits_for_context(&[5, 2], &mut out).unwrap();
    m.logits_for_context(&[1, 2], &mut truncated).unwrap();
    assert!(
        out.iter()
            .zip(&truncated)
            .any(|(a, b)| (a - b).abs() > 1e-5)
    );
}

#[test]
fn token_weighted_accumulation_matches_sum_and_model_learns() {
    let mut a = tiny();
    let mut b = tiny();
    a.zero_gradients();
    a.forward_train_chunk(&[1, 2, 1], &[2, 1, 3]);
    a.backward_chunk(3, 0.75);
    b.forward_train_chunk(&[1, 2, 1], &[2, 1, 3]);
    b.backward_chunk(3, 1.0);
    let before: Vec<_> = matrices(&b).iter().map(|p| p.grad.clone()).collect();
    b.zero_gradients();
    b.forward_train_chunk(&[4], &[5]);
    b.backward_chunk(1, 1.0);
    a.forward_train_chunk(&[4], &[5]);
    a.backward_chunk(1, 0.25);
    for (family, (pa, pb)) in matrices(&a).into_iter().zip(matrices(&b)).enumerate() {
        for (i, (&g1, &g2)) in pa.grad.iter().zip(&pb.grad).enumerate() {
            assert!((g1 - 0.75 * before[family][i] - 0.25 * g2).abs() < 2e-6);
        }
    }
    let initial = loss(&mut a);
    for _ in 0..180 {
        step(&mut a, 0.025);
    }
    let final_loss = loss(&mut a);
    assert!(
        final_loss < initial * 0.4,
        "initial={initial} final={final_loss}"
    );
    assert!(a.all_finite());
}

#[test]
fn checkpoint_preserves_all_state_and_next_update_exactly() {
    let p = path("roundtrip.trfm");
    let q = path("roundtrip-next.trfm");
    let mut a = tiny();
    a.lr_schedule_total_updates = Some(20);
    step(&mut a, 0.02);
    ck::save_model(&a, &p).unwrap();
    let mut b = ck::load_checkpoint(&p).unwrap();
    assert_eq!(b.cfg.lr, 0.02);
    assert_eq!(b.cfg.chunk_len, 4);
    assert_eq!(b.lr_schedule_total_updates, Some(20));
    assert_eq!(b.rng.state, a.rng.state);
    assert_eq!(
        b.tokenizer().unwrap().ordered_vocabulary().unwrap(),
        a.vocabulary
    );
    ck::save_model(&b, &q).unwrap();
    assert_eq!(fs::read(&p).unwrap(), fs::read(&q).unwrap());
    step(&mut a, 0.01);
    step(&mut b, 0.01);
    ck::save_model(&a, &p).unwrap();
    ck::save_model(&b, &q).unwrap();
    assert_eq!(fs::read(&p).unwrap(), fs::read(&q).unwrap());
    fs::remove_file(p).unwrap();
    fs::remove_file(q).unwrap();
}

fn checksum(bytes: &mut [u8]) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in &bytes[22..] {
        hash = (hash ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    bytes[14..22].copy_from_slice(&hash.to_le_bytes());
}
#[test]
fn malformed_checkpoints_and_unsafe_dimensions_are_rejected() {
    let p = path("malformed.trfm");
    let mut m = tiny();
    ck::save_model(&m, &p).unwrap();
    let good = fs::read(&p).unwrap();
    let mut cases = vec![
        good[..5].to_vec(),
        good[..22].to_vec(),
        good[..good.len() - 1].to_vec(),
    ];
    let mut corrupt = good.clone();
    corrupt[30] ^= 1;
    cases.push(corrupt);
    let mut giant = good.clone();
    giant[22..30].copy_from_slice(&u64::MAX.to_le_bytes());
    checksum(&mut giant);
    cases.push(giant);
    let mut zero_heads = good.clone();
    zero_heads[38..46].fill(0);
    checksum(&mut zero_heads);
    cases.push(zero_heads);
    let mut nan = good.clone();
    let len = nan.len();
    nan[len - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    checksum(&mut nan);
    cases.push(nan);
    let mut negative_v = good.clone();
    let len = negative_v.len();
    negative_v[len - 4..].copy_from_slice(&(-1f32).to_le_bytes());
    checksum(&mut negative_v);
    cases.push(negative_v);
    for bytes in cases {
        fs::write(&p, bytes).unwrap();
        assert!(ck::load_checkpoint(&p).is_err());
    }
    m.qkv.v[0] = -1.;
    assert!(ck::save_model(&m, &p).is_err());
    for cfg in [
        TransformerConfig {
            d_model: usize::MAX,
            ..Default::default()
        },
        TransformerConfig {
            d_vocab: usize::MAX,
            ..Default::default()
        },
        TransformerConfig {
            n_heads: 0,
            ..Default::default()
        },
        TransformerConfig {
            chunk_len: 65_536,
            ..Default::default()
        },
    ] {
        assert!(TransformerModel::new(cfg, 42).is_err());
    }
    fs::remove_file(p).unwrap();
}

#[test]
fn shared_windows_preserve_document_boundaries_eof_wrap_and_targets() {
    let raw = "a b c d\ne\nf a b c d\n";
    let tok = Tokenizer::from_corpus(raw, true).unwrap();
    let docs = CLIHandler::documents(raw, &tok, Some(14), 8).unwrap();
    let expected = ["c d", "a b c d", "f a b c d", "a b"];
    assert_eq!(
        docs,
        expected.map(|s| tok.try_encode(s, true).unwrap()).to_vec()
    );
    assert_eq!(
        docs,
        CLIHandler::documents(raw, &tok, Some(14), 18).unwrap()
    );
    let plan = chunk_plan(&docs, 3);
    assert_eq!(
        plan,
        vec![(0, 0, 1), (1, 0, 3), (2, 0, 3), (2, 3, 1), (3, 0, 1)]
    );
}

#[test]
fn shared_fixed_schedule_continues_and_refuses_drift() {
    let opts = TrainingOptions {
        epochs: 1,
        accumulate: 2,
        schedule_total_updates: Some(10),
        ..Default::default()
    };
    let first = Schedule::new(6, 0, None, &opts).unwrap();
    let resumed = Schedule::new(
        6,
        3,
        Some(10),
        &TrainingOptions {
            schedule_total_updates: None,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(first.lr(4).unwrap(), resumed.lr(1).unwrap());
    assert_eq!(resumed.fixed_horizon, Some(10));
    assert!(Schedule::new(6, 9, Some(10), &opts).is_err());
    assert!(Schedule::new(6, 3, Some(11), &opts).is_err());
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args(args)
        .env("NO_COLOR", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap()
}
fn success(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "args={args:?}\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn cli_same_bpe_stream_logs_resume_lr_and_legacy_surface() {
    let corpus = path("corpus.txt");
    let pssa = path("comparison.pssa");
    let trfm = path("comparison.trfm");
    let next = path("comparison-next.trfm");
    fs::write(
        &corpus,
        "a scientist studies the moon.\na patient observes the stars.\n",
    )
    .unwrap();
    let cp = corpus.to_str().unwrap();
    let pp = pssa.to_str().unwrap();
    let tp = trfm.to_str().unwrap();
    let np = next.to_str().unwrap();
    let p_log = success(&[
        "train",
        cp,
        "-o",
        pp,
        "--latent",
        "8",
        "--state",
        "2",
        "--key",
        "2",
        "--memory",
        "2",
        "--vocab-size",
        "257",
        "--chunk",
        "4",
        "--accumulate",
        "2",
        "--max-tokens",
        "7",
        "--skip-tokens",
        "51",
        "--total-updates",
        "20",
        "-e",
        "1",
    ]);
    let t_log = success(&[
        "train-transformer",
        cp,
        "-o",
        tp,
        "--tokenizer-from",
        pp,
        "--chunk",
        "4",
        "--accumulate",
        "2",
        "--max-tokens",
        "7",
        "--skip-tokens",
        "51",
        "--total-updates",
        "20",
        "--lr",
        "0.003",
        "-e",
        "1",
    ]);
    let fingerprint = |s: &str| {
        s.lines()
            .find(|s| s.starts_with("token_stream_fnv1a64="))
            .unwrap()
            .to_string()
    };
    assert_eq!(fingerprint(&p_log), fingerprint(&t_log));
    for s in [&p_log, &t_log] {
        assert!(s.contains("parameters="));
        let epoch = s
            .lines()
            .find(|s| s.starts_with("epoch 1/1 loss="))
            .unwrap();
        assert!(epoch.ends_with("tokens=6 updates=1"), "{epoch}");
    }
    let p_model = checkpoint::load_checkpoint(&pssa).unwrap().model;
    let t_model = ck::load_checkpoint(&trfm).unwrap();
    assert_eq!(p_model.vocabulary, t_model.vocabulary);
    assert_eq!(p_model.tokenizer_json, t_model.tokenizer_json);
    let log = success(&[
        "train-transformer",
        cp,
        "-o",
        np,
        "--resume",
        tp,
        "--max-tokens",
        "7",
        "--skip-tokens",
        "58",
        "--accumulate",
        "2",
        "-e",
        "1",
    ]);
    assert!(log.contains("horizon=20 from_step=1 to_step=2"));
    let resumed = ck::load_checkpoint(&next).unwrap();
    assert_eq!(resumed.cfg.lr, 0.003);
    assert_eq!(resumed.cfg.chunk_len, 4);
    assert_eq!(resumed.step_counter, 2);
    for tail in [
        vec!["--chunk", "8"],
        vec!["--total-updates", "21"],
        vec!["--tokenizer-from", pp],
    ] {
        let mut args = vec!["train-transformer", cp, "--resume", tp];
        args.extend(tail);
        let out = run(&args);
        assert_eq!(out.status.code(), Some(2));
    }
    assert!(success(&["help"]).contains("--resume"));
    assert!(success(&["train-transformer", "--help"]).contains("--tokenizer-from"));
    success(&[
        "generate",
        "-m",
        pp,
        "-p",
        "the moon",
        "--temperature",
        "0",
        "--max-new-tokens",
        "2",
    ]);
    let generated = success(&[
        "generate-transformer",
        "-m",
        np,
        "-p",
        "the moon",
        "--temperature",
        "0",
        "--max-new-tokens",
        "2",
    ]);
    assert_eq!(
        generated,
        success(&[
            "generate-transformer",
            "-m",
            np,
            "-p",
            "the moon",
            "--temperature",
            "0",
            "--max-new-tokens",
            "2"
        ])
    );
    for command in ["evaluate", "evaluate-transformer"] {
        let path = if command == "evaluate" { pp } else { np };
        let log = success(&[command, cp, "-m", path]);
        let metrics: serde_json::Value = serde_json::from_str(log.trim()).unwrap();
        assert!(metrics["cross_entropy"].as_f64().unwrap().is_finite());
        assert!(metrics["token_count"].as_u64().unwrap() > 0);
    }
    for p in [corpus, pssa, trfm, next] {
        fs::remove_file(p).unwrap();
    }
}
