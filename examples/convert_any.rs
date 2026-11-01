//! Convert any supported HuggingFace LLM checkpoint into a PSSA v6 checkpoint.
//!
//! cargo run --release --example convert_any -- <model.safetensors|dir> <config.json> <out.v6>
//!     [--target compact|source-width] [--seed 1234]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = pssa::convert::run_cli(&args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
