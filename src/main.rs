use oxide_ai_pssa::cli::CLIHandler;

fn main() {
    if let Err(error) = CLIHandler::parse_and_execute(std::env::args().collect()) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
