pub struct DiagnosticsSuite;

impl DiagnosticsSuite {
    pub fn print_banner(msg: &str) {
        println!("{}", "=".repeat(70));
        println!(" {:^68} ", msg);
        println!("{}", "=".repeat(70));
    }
}
