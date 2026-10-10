pub mod adapter;
pub mod backend;
pub mod checkpoint;
pub mod cli;
pub mod comparison;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dataset;
pub mod defense;
pub mod diagnostics;
pub mod eval_harness;
pub mod dream;
pub mod evaluation;
pub mod feature_benchmark;
pub mod gpu_batch;
pub mod inference;
pub mod linalg;
#[cfg(test)]
mod local_mixing_tests;
pub mod loss_csv;
pub mod loss_guard;
pub mod memory;
pub mod pssa;
pub mod scan_executor;
pub mod sequence_batch;
pub mod training;
mod training_diagnostics;
pub mod transformer;
pub mod transformer_checkpoint;
pub mod transformer_inference;
pub mod transformer_training;
mod token_cache;
pub mod tui;
pub mod ui;
pub mod world_model;

/// Fall back to a legacy environment variable only when the current one is
/// unset. An empty or non-Unicode current value still takes precedence.
pub(crate) fn env_var_os(name: &str, legacy: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).or_else(|| std::env::var_os(legacy))
}
