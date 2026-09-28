use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "oxide-wikitext-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn assert_error(output: Output, message: &str) {
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains(message), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(output.stdout.is_empty());
}

#[test]
fn cli_cleans_a_file_with_both_output_spellings_and_preserves_input() {
    let dir = Workspace::new();
    let raw = " = Robert Boulter = \r\nRobert Boulter is an English film , television and theatre actor .\r\n \t\n\n= = Career = =\n\nHe had a guest @-@ starring role on the television series The Bill in 2000 .\nDu Fu ( 杜甫 ) <unk> .\n52 @.@ 9 million , 500 @,@ 000 copies .";
    let expected = "Robert Boulter is an English film, television and theatre actor.\n\nHe had a guest-starring role on the television series The Bill in 2000.\nDu Fu (杜甫).\n52.9 million, 500,000 copies.\n";
    fs::write(dir.0.join("raw.txt"), raw).unwrap();
    for (args, filename) in [
        (
            ["clean-wikitext", "raw.txt", "-o", "short.txt"],
            "short.txt",
        ),
        (
            ["clean-wikitext", "--out", "long.txt", "raw.txt"],
            "long.txt",
        ),
    ] {
        let output = dir.run(&args);
        assert!(output.status.success(), "{:?}", output);
        assert_eq!(fs::read_to_string(dir.0.join(filename)).unwrap(), expected);
        assert_eq!(fs::read_to_string(dir.0.join("raw.txt")).unwrap(), raw);
    }
}

#[test]
fn cli_validates_arguments_before_creating_files() {
    let dir = Workspace::new();
    for (args, message) in [
        (vec!["clean-wikitext"], "requires one input file"),
        (vec!["clean-wikitext", "raw.txt"], "requires --out"),
        (vec!["clean-wikitext", "raw.txt", "-o"], "requires a value"),
        (
            vec!["clean-wikitext", "raw.txt", "extra.txt", "-o", "out.txt"],
            "requires one input file",
        ),
        (
            vec!["clean-wikitext", "raw.txt", "--unknown", "out.txt"],
            "unknown option",
        ),
        (
            vec![
                "clean-wikitext",
                "raw.txt",
                "-o",
                "out.txt",
                "--out",
                "out2.txt",
            ],
            "specified more than once",
        ),
    ] {
        assert_error(dir.run(&args), message);
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
    }
}

#[test]
fn cli_never_overwrites_existing_files_including_input_aliases() {
    let dir = Workspace::new();
    let raw = " He had a guest @-@ starring role . \n";
    fs::write(dir.0.join("raw.txt"), raw).unwrap();
    fs::write(dir.0.join("existing.txt"), "keep this output").unwrap();
    for path in ["raw.txt", "./raw.txt", "existing.txt"] {
        assert_error(
            dir.run(&["clean-wikitext", "raw.txt", "-o", path]),
            "use --out with a new file",
        );
    }
    assert_eq!(fs::read_to_string(dir.0.join("raw.txt")).unwrap(), raw);
    assert_eq!(
        fs::read_to_string(dir.0.join("existing.txt")).unwrap(),
        "keep this output"
    );

    #[cfg(unix)]
    {
        fs::hard_link(dir.0.join("raw.txt"), dir.0.join("hard.txt")).unwrap();
        std::os::unix::fs::symlink("raw.txt", dir.0.join("sym.txt")).unwrap();
        std::os::unix::fs::symlink("missing.txt", dir.0.join("dangling.txt")).unwrap();
        for path in ["hard.txt", "sym.txt", "dangling.txt"] {
            assert_error(
                dir.run(&["clean-wikitext", "raw.txt", "-o", path]),
                "use --out with a new file",
            );
        }
        assert_eq!(fs::read_to_string(dir.0.join("raw.txt")).unwrap(), raw);
        assert!(!dir.0.join("missing.txt").exists());
    }
}

#[test]
fn cli_reports_io_and_utf8_errors_and_removes_partial_output() {
    let dir = Workspace::new();
    assert_error(
        dir.run(&["clean-wikitext", "missing.txt", "-o", "out.txt"]),
        "provide a readable UTF-8 file",
    );
    assert!(!dir.0.join("out.txt").exists());

    fs::write(dir.0.join("invalid.txt"), b"Robert Boulter .\n\xff\n").unwrap();
    assert_error(
        dir.run(&["clean-wikitext", "invalid.txt", "-o", "out.txt"]),
        "check input UTF-8",
    );
    assert!(!dir.0.join("out.txt").exists());
    assert_eq!(
        fs::read(dir.0.join("invalid.txt")).unwrap(),
        b"Robert Boulter .\n\xff\n"
    );

    assert_error(
        dir.run(&["clean-wikitext", "invalid.txt", "-o", "missing/out.txt"]),
        "existing writable directory",
    );
}

#[test]
fn cli_accepts_empty_input_and_heading_only_input() {
    let dir = Workspace::new();
    for (input, output, text) in [
        ("empty.txt", "empty-clean.txt", ""),
        ("heading.txt", "heading-clean.txt", " = Robert Boulter = \n"),
    ] {
        fs::write(dir.0.join(input), text).unwrap();
        assert!(
            dir.run(&["clean-wikitext", input, "-o", output])
                .status
                .success()
        );
        assert!(fs::read(dir.0.join(output)).unwrap().is_empty());
    }
}

#[test]
fn help_documents_cleaning_and_keeps_the_kaggle_resume_contract() {
    let dir = Workspace::new();
    let help = dir.run(&["help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(text.contains("--resume"));
    assert!(text.contains("clean-wikitext"));
    for args in [["help", "clean-wikitext"], ["clean-wikitext", "--help"]] {
        let help = dir.run(&args);
        assert!(help.status.success());
        let text = String::from_utf8(help.stdout).unwrap();
        for expected in [
            "clean-wikitext INPUT -o OUTPUT",
            "--out",
            "UTF-8",
            "never overwritten",
        ] {
            assert!(text.contains(expected), "{text}");
        }
    }
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
}
