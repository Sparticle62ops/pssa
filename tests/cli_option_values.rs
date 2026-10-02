use oxide_ai_pssa::cli::CLIHandler;

#[test]
fn option_like_missing_value_is_rejected_before_typed_parsing() {
    let error = CLIHandler::parse_and_execute(vec![
        "oxide_ai_pssa".into(),
        "train".into(),
        "--epochs".into(),
        "--bogus".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "option '--epochs' requires a value");
}

#[test]
fn negative_numeric_option_value_reaches_domain_validation() {
    let error = CLIHandler::parse_and_execute(vec![
        "oxide_ai_pssa".into(),
        "generate".into(),
        "prompt".into(),
        "--temperature".into(),
        "-1".into(),
    ])
    .unwrap_err();

    assert_eq!(error, "--temp must be >= 0");
}
