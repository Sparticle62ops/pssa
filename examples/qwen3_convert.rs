//! Convert a Qwen3 safetensors checkpoint into a PSSA v6 checkpoint.
//!
//! cargo run --release --example qwen3_convert -- <model.safetensors> <config.json> <out.v6>
//!     [--target compact|source-width] [--seed 1234]
use pssa::qwen3::{Qwen3Config, Qwen3Model, Qwen3PssaTarget};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut target_mode = "compact".to_owned();
    let mut seed = 1234u64;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--target" => {
                index += 1;
                target_mode = args.get(index).cloned().expect("--target needs a value");
            }
            "--seed" => {
                index += 1;
                seed = args.get(index).cloned().expect("--seed needs a value").parse().expect("--seed needs a number");
            }
            other => positional.push(other.to_owned()),
        }
        index += 1;
    }
    if positional.len() != 3 {
        eprintln!("usage: qwen3_convert <model.safetensors> <config.json> <out.v6> [--target compact|source-width] [--seed N]");
        std::process::exit(2);
    }
    let safetensors_path = PathBuf::from(&positional[0]);
    let config_path = PathBuf::from(&positional[1]);
    let out_path = PathBuf::from(&positional[2]);

    let config = Qwen3Config::from_json_file(&config_path).expect("read Qwen3 config.json");
    println!(
        "loading source ({} layers, hidden {}) from {}",
        config.num_hidden_layers,
        config.hidden_size,
        safetensors_path.display()
    );
    let model = Qwen3Model::from_safetensors(&safetensors_path, config).expect("load Qwen3 checkpoint");
    let target = match target_mode.as_str() {
        "compact" => Qwen3PssaTarget::compact_for(&model.config),
        "source-width" => Qwen3PssaTarget::source_width_for(&model.config),
        other => panic!("unknown --target {other}; expected compact or source-width"),
    };
    println!(
        "target: latent {} state {} key {} capacity {} chunk {} depth {}",
        target.latent_dim,
        target.state_dim,
        target.memory_key_dim,
        target.memory_capacity,
        target.chunk_len,
        target.depth
    );
    let report = pssa::qwen3::emit_pssa_checkpoint(&model, &target, &out_path, seed).expect("emit PSSA checkpoint");
    println!(
        "wrote {} ({} bytes); estimate: {} MiB target allocation, {} MiB memory bank, scan len {}; logit RMS before calibration {:.3}, unembed scale {:.3e}",
        report.checkpoint_path.display(),
        report.checkpoint_bytes,
        report.estimate.pssa_target_allocation_bytes / (1024 * 1024),
        report.estimate.memory_bank_bytes / (1024 * 1024),
        report.estimate.scan_len,
        report.logit_rms_before,
        report.logit_scale
    );
}
