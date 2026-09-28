//! This is not two constructors exercising the same new implementation: the
//! expected bytes and scalar f32 bits were produced by UNMODIFIED main 85d9d33.
//! The recipe includes populated memory, nonzero adapter slow/fast + MLP weights,
//! nonzero carry, unequal-length accumulation, Adam, consolidation and inference.
#[path = "support/depth_one_reference.rs"]
mod reference;
use oxide_ai_pssa::checkpoint::{CheckpointFormat, load_checkpoint, save_model};

const INITIAL: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_initial.pssa");
const TRAINED: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_trained.pssa");
const OBSERVED: &[u8] = include_bytes!("fixtures/depth_one_main85d9d33_observed.bin");

#[test]
fn depth_one_is_bit_exact_with_main_and_loads_unchanged_v7_checkpoints() {
    let dir = std::env::temp_dir().join(format!("oxide-main-depth-one-{}", std::process::id()));
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
            observed, OBSERVED,
            "losses and training/inference logits differ from main"
        );
        save_model(model, &new).unwrap();
        assert_eq!(
            std::fs::read(&new).unwrap(),
            TRAINED,
            "all parameters, gradients, moments, carry, memory and optimizer state must match main"
        );
    }
    std::fs::write(&old, TRAINED).unwrap();
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
