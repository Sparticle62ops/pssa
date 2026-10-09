//! Independent goldens from unmodified main, not the implementation under test.
//! Initialization/legacy V7 files come from 85d9d33. SIMD goldens come from
//! 014f617, which intentionally changed eight-wide dots/RMSNorm rounding.
//! The recipe includes populated memory, nonzero adapter slow/fast + MLP weights,
//! nonzero carry, unequal-length accumulation, Adam, consolidation and inference.
#[path = "support/depth_one_reference.rs"]
mod reference;
use pssa::checkpoint::{CheckpointFormat, load_checkpoint, save_model};

const INITIAL: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_initial.pssa");
const TRAINED: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_trained.pssa");
const OBSERVED: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_observed.bin");
// Generated with the unchanged recipe against exported main 014f617, release
// opt-level 3 / fat LTO / one codegen unit, on x86_64 with AVX2/FMA.
// SHA-256 trained: 69549428d1285cac528c5870559282ea93030cdfed7956721d16109cac16cf91
// SHA-256 observed: f59b3a5872bed2afaa84cce668bf9d6a57569993db662ebb72aa131fa6703fbe
const SIMD_TRAINED: &[u8] = include_bytes!("fixtures/depth_one_main014f617_avx2_trained.pssa");
const SIMD_OBSERVED: &[u8] = include_bytes!("fixtures/depth_one_main014f617_avx2_observed.bin");

fn current_kernel_goldens() -> (&'static [u8], &'static [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return (SIMD_TRAINED, SIMD_OBSERVED);
    }
    (TRAINED, OBSERVED)
}

#[test]
fn depth_one_is_bit_exact_with_main_and_loads_unchanged_v7_checkpoints() {
    let (expected_trained, expected_observed) = current_kernel_goldens();
    let dir = std::env::temp_dir().join(format!("pssa-main-depth-one-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let new = dir.join("new.pssa");
    let old = dir.join("old.pssa");
    let mut fresh = reference::model();
    assert_eq!(fresh.depth(), 1);
    save_model(&fresh, &new).unwrap();
    assert_eq!(
        std::fs::read(&new).unwrap(),
        INITIAL,
        "depth-one initialization and V7 bytes must remain unchanged"
    );
    std::fs::write(&old, INITIAL).unwrap();
    let loaded = load_checkpoint(&old).unwrap();
    assert_eq!(loaded.format, CheckpointFormat::V7);
    assert_eq!(loaded.model.cfg.depth, 1);
    let mut restored = loaded.model;
    for model in [&mut fresh, &mut restored] {
        let observed: Vec<_> = reference::advance(model)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            observed, expected_observed,
            "losses and training/inference logits differ from main"
        );
        save_model(model, &new).unwrap();
        assert_eq!(
            std::fs::read(&new).unwrap(),
            expected_trained,
            "all parameters, gradients, moments, carry, memory and optimizer state must match main"
        );
    }
    std::fs::write(&old, expected_trained).unwrap();
    let mut trained = load_checkpoint(&old).unwrap().model;
    assert_eq!(
        reference::advance(&mut fresh),
        reference::advance(&mut trained)
    );
    save_model(&fresh, &new).unwrap();
    save_model(&trained, &old).unwrap();
    assert_eq!(std::fs::read(&new).unwrap(), std::fs::read(&old).unwrap());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pre_simd_v7_checkpoints_remain_byte_exact_on_roundtrip_and_continuation() {
    let dir = std::env::temp_dir().join(format!("pssa-legacy-depth-one-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let old = dir.join("legacy.pssa");
    let saved = dir.join("roundtrip.pssa");
    for bytes in [INITIAL, TRAINED] {
        std::fs::write(&old, bytes).unwrap();
        let loaded = load_checkpoint(&old).unwrap();
        assert_eq!(loaded.format, CheckpointFormat::V7);
        assert_eq!(loaded.model.depth(), 1);
        let mut original = loaded.model;
        save_model(&original, &saved).unwrap();
        assert_eq!(std::fs::read(&saved).unwrap(), bytes);
        let mut restored = load_checkpoint(&saved).unwrap().model;
        assert_eq!(
            reference::advance(&mut original),
            reference::advance(&mut restored)
        );
        save_model(&original, &old).unwrap();
        save_model(&restored, &saved).unwrap();
        assert_eq!(std::fs::read(&old).unwrap(), std::fs::read(&saved).unwrap());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
