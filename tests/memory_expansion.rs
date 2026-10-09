use pssa::{
    checkpoint,
    memory::HyperbolicEpisodicBankV2 as Bank,
    pssa::{PSSAConfigV2, PSSALayerV2},
};
use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

#[test]
fn growth_preserves_eviction_order_and_protection_metadata() {
    let mut bank = Bank::new(3, 1, 1);
    for i in 0..5 {
        bank.insert(&[i as f32 / 10.0], &[i as f32]);
    }
    bank.confidence = vec![1.1, 1.2, 1.3];
    bank.last_seen_step = vec![3, 4, 2];
    bank.set_top_k(Some(2)).unwrap();
    bank.grow(5).unwrap();
    assert_eq!(&bank.values[..3], &[2.0, 3.0, 4.0]);
    assert_eq!(&bank.confidence[..3], &[1.3, 1.1, 1.2]);
    assert_eq!(&bank.last_seen_step[..3], &[2, 3, 4]);
    assert_eq!(bank.top_k(), Some(2));
    for i in 5..8 {
        bank.insert(&[i as f32 / 10.0], &[i as f32]);
    }
    assert_eq!(bank.values, [7.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(bank.write_head, 1);
    let before = bank.clone();
    bank.grow(5).unwrap();
    assert_eq!(bank, before);
    assert!(bank.grow(4).is_err());
    assert_eq!(bank, before);
}

#[test]
fn growth_handles_empty_and_partial_banks() {
    for count in [0, 2] {
        let mut b = Bank::new(3, 1, 1);
        for i in 0..count {
            b.insert(&[0.1], &[i as f32]);
        }
        b.grow(6).unwrap();
        assert_eq!(b.count, count);
        assert_eq!(b.insert(&[0.2], &[9.0]), count);
        assert_eq!(b.values[count], 9.0);
    }
}

#[test]
fn nearest_read_masks_other_slots_and_full_k_is_exact() {
    let mut bank = Bank::new(5, 1, 1);
    for (key, value) in [(0.0, 2.0), (0.1, 4.0), (0.6, 100.0)] {
        bank.insert(&[key], &[value]);
    }
    let mut out = [0.0];
    let mut weights = [0.0; 5];
    bank.retrieve_soft_into(&[0.0], 1.0, &mut out, &mut weights);
    let full = (out, weights);
    for k in [3, 9] {
        bank.set_top_k(Some(k)).unwrap();
        bank.retrieve_soft_into(&[0.0], 1.0, &mut out, &mut weights);
        assert_eq!((out, weights), full);
    }
    bank.set_top_k(Some(1)).unwrap();
    bank.retrieve_soft_into(&[0.0], 1.0, &mut out, &mut weights);
    assert_eq!(out, [2.0]);
    assert_eq!(weights, [1.0, 0.0, 0.0, 0.0, 0.0]);
    bank.set_top_k(Some(2)).unwrap();
    bank.retrieve_soft_into(&[0.0], 1.0, &mut out, &mut weights);
    assert!(out[0] > 2.0 && out[0] < 4.0);
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert_eq!(weights[2], 0.0);
    assert!(bank.set_top_k(Some(0)).is_err());
}

#[test]
fn nearest_ties_tiny_temperature_and_empty_are_stable() {
    let mut b = Bank::new(4, 1, 1);
    b.set_top_k(Some(1)).unwrap();
    let (mut out, mut w) = ([9.0], [9.0; 4]);
    b.retrieve_soft_into(&[0.0], f32::from_bits(1), &mut out, &mut w);
    assert_eq!(out, [0.0]);
    assert_eq!(w, [0.0; 4]);
    b.insert(&[0.2], &[1.0]);
    b.insert(&[-0.2], &[2.0]);
    b.retrieve_soft_into(&[0.0], f32::from_bits(1), &mut out, &mut w);
    assert_eq!(out, [1.0]);
    assert_eq!(w, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn top_k_query_gradient_matches_fixed_selection_finite_difference() {
    fn model(delta: f32) -> PSSALayerV2 {
        let cfg = PSSAConfigV2 {
            d_vocab: 5,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 4,
            chunk_len: 1,
            ..Default::default()
        };
        let mut m = PSSALayerV2::new(cfg, 19);
        for (key, val) in [
            ([0.0, 0.0], [1.0, -0.5, 0.2, 0.1]),
            ([0.1, 0.1], [-0.5, 1.0, -0.2, 0.7]),
            ([0.8, 0.3], [4.0; 4]),
            ([-0.8, -0.3], [-4.0; 4]),
        ] {
            m.memory.insert(&key, &val);
        }
        m.set_memory_top_k(Some(2)).unwrap();
        m.w_qx.data[0] += delta;
        m
    }
    let mut m = model(0.0);
    m.forward_train_chunk(&[1], &[2]);
    m.backward_chunk(1, 1.0);
    let analytic = m.w_qx.grad[0];
    let h = 0.002;
    let numeric = (model(h).forward_train_chunk(&[1], &[2])
        - model(-h).forward_train_chunk(&[1], &[2]))
        / (2.0 * h);
    assert!(analytic.abs() > 1e-6, "vacuous gradient: {analytic}");
    assert!(
        (analytic - numeric).abs() < 2e-4 + numeric.abs() * 0.03,
        "analytic={analytic} numeric={numeric}"
    );
}

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "pssa-memory-expansion-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn stacked_loop_growth_preserves_state_and_roundtrips_existing_format() {
    let tmp = Temp::new();
    for depth in [1, 2] {
        let cfg = PSSAConfigV2 {
            depth,
            d_vocab: 5,
            d_latent: 4,
            d_state: 2,
            d_mem_key: 2,
            mem_capacity: 3,
            chunk_len: 3,
            ..Default::default()
        };
        let mut model = PSSALayerV2::new_with_depth_and_loops(cfg, 7, depth, 2);
        model.vocabulary = ["<unk>", "a", "b", "c", "d"].map(str::to_string).to_vec();
        model.forward_train_chunk(&[1, 2, 3], &[2, 3, 4]);
        model.backward_and_step_chunk(3);
        let embeds = model.embed_w.clone();
        let step = model.step_counter;
        let rng = model.rng.state;
        let carry = model.h_persistent.clone();
        model.grow_memory(9).unwrap();
        assert_eq!(model.embed_w, embeds);
        assert_eq!(model.step_counter, step);
        assert_eq!(model.rng.state, rng);
        assert_eq!(model.h_persistent, carry);
        for block in std::iter::once(&model.block).chain(&model.extra_blocks) {
            assert_eq!(block.tape.mem_weights.len(), 3 * 3 * 9); // working + two saved loop slots
            assert_eq!(block.inf_mem_weights.len(), 9);
            assert_eq!(block.memory.capacity, 9);
        }
        model.set_memory_top_k(Some(2)).unwrap();
        let path = tmp.0.join(format!("depth-{depth}.pssa"));
        checkpoint::save_model(&model, &path).unwrap();
        let mut loaded = checkpoint::load_checkpoint(path).unwrap().model;
        assert_eq!(loaded.cfg.mem_capacity, 9);
        assert_eq!(loaded.memory.top_k(), None); // runtime-only; bytes unchanged
        loaded.set_loops(2).unwrap();
        loaded.set_memory_top_k(Some(2)).unwrap();
        assert!(
            loaded
                .forward_train_chunk(&[1, 2, 3], &[2, 3, 4])
                .is_finite()
        );
        loaded.backward_and_step_chunk(3);
        assert_eq!(loaded.step_counter, step + 1);
        assert!(loaded.grow_memory(usize::MAX).is_err());
    }
}

#[test]
fn cli_growth_and_score_top_k_work_and_invalid_requests_do_not_save() {
    let tmp = Temp::new();
    let corpus = tmp.0.join("data.txt");
    fs::write(&corpus, "a b c d a b c d a b c d a b c d").unwrap();
    let first = tmp.0.join("first.pssa");
    let grown = tmp.0.join("grown.pssa");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_pssa"))
            .args(args)
            .output()
            .unwrap()
    };
    let base = [
        "train",
        corpus.to_str().unwrap(),
        "--backend",
        "cpu",
        "--tokenizer",
        "word",
        "--latent",
        "4",
        "--state",
        "2",
        "--key",
        "2",
        "--chunk",
        "3",
        "--memory",
        "3",
        "--epochs",
        "1",
        "--accumulate",
        "1",
        "--no-tui",
    ];
    let mut args = base.to_vec();
    args.extend(["-o", first.to_str().unwrap()]);
    let output = run(&args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let original_checkpoint = fs::read(&first).unwrap();
    let mut args = base.to_vec();
    args.extend([
        "--resume",
        first.to_str().unwrap(),
        "--grow-memory",
        "8",
        "--memory-top-k",
        "2",
        "-o",
        grown.to_str().unwrap(),
    ]);
    let output = run(&args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(log.contains("memory_growth=3->8"));
    assert!(log.contains("memory_retrieval=top-2"));
    assert_eq!(fs::read(&first).unwrap(), original_checkpoint);
    assert_eq!(
        checkpoint::load_checkpoint(&grown)
            .unwrap()
            .model
            .cfg
            .mem_capacity,
        8
    );
    let output = run(&[
        "score",
        corpus.to_str().unwrap(),
        "-m",
        grown.to_str().unwrap(),
        "--memory-top-k",
        "2",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("cross_entropy"));
    for invalid in [
        vec!["--grow-memory", "8"],
        vec!["--memory-top-k", "0"],
        vec!["--memory-top-k", "2", "--backend", "auto"],
        vec!["--resume", first.to_str().unwrap(), "--grow-memory", "2"],
        vec!["--resume", first.to_str().unwrap(), "--memory", "8"],
    ] {
        let bad = tmp.0.join("bad.pssa");
        let mut args = vec![
            "train",
            corpus.to_str().unwrap(),
            "--epochs",
            "1",
            "--no-tui",
            "-o",
            bad.to_str().unwrap(),
        ];
        args.extend(invalid);
        assert!(!run(&args).status.success());
        assert!(!bad.exists());
    }
}
