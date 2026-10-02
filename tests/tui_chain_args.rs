use oxide_ai_pssa::tui;

#[test]
fn chain_requires_a_directory_value() {
    assert!(tui::run(&["--chain".into()]).is_err());
    assert!(tui::run(&["--chain".into(), "--no-tui".into()]).is_err());
}
