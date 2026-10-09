//! Bounded CPU-only comparison; no checkpoint files or default-training changes.
use pssa::world_model::{WorldModelConfig, run_world_model};

fn main() -> Result<(), String> {
    for seed in [73, 101, 211] {
        let config = WorldModelConfig {
            seed,
            ..WorldModelConfig::default()
        };
        println!("{}", run_world_model(&config)?.report());
    }
    Ok(())
}
