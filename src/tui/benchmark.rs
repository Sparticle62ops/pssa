//! UI wrapper around the existing, audited token/update-matched `compare` CLI.
use super::{
    accent, panel,
    process::{Job, clean},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{collections::VecDeque, io::Read, path::PathBuf};

const LABELS: [&str; 7] = [
    "original corpus",
    "PSSA chain directory",
    "new output directory",
    "links",
    "tokens per link",
    "batch size",
    "accumulate",
];
pub(super) struct Benchmark {
    fields: [String; 7],
    selected: usize,
    editing: bool,
    job: Option<Job>,
    logs: VecDeque<String>,
    result: Option<serde_json::Value>,
    note: String,
    pub notification: Option<(bool, String)>,
}
impl Benchmark {
    pub fn new(chain: PathBuf) -> Self {
        Self { fields: ["data/downloaded.txt".into(), chain.display().to_string(), "comparison".into(), "64".into(), "200000".into(), "8".into(), "1".into()], selected: 0, editing: false, job: None, logs: VecDeque::new(), result: None,
            note: "Supply the ORIGINAL corpus and chain settings, then b runs the matched benchmark. Output must not exist.".into(), notification: None }
    }
    fn args(&self) -> Result<Vec<String>, String> {
        if !std::path::Path::new(&self.fields[0]).is_file() {
            return Err("Original corpus must be a local file.".into());
        }
        if !std::path::Path::new(&self.fields[1]).is_dir() {
            return Err("PSSA chain must be an existing directory with ck01.pssa …".into());
        }
        let out = std::path::Path::new(&self.fields[2]);
        if self.fields[2].trim().is_empty() || out.exists() {
            return Err(
                "Choose a NEW output directory; existing experiments are never overwritten.".into(),
            );
        }
        for value in &self.fields[3..] {
            if value.parse::<usize>().ok().filter(|n| *n > 0).is_none() {
                return Err("Counts must be positive integers.".into());
            }
        }
        let mut args = vec!["compare".into(), self.fields[0].clone()];
        for (flag, value) in [
            "--chain-dir",
            "--out",
            "--links",
            "--window",
            "--batch-size",
            "--accumulate",
        ]
        .iter()
        .zip(&self.fields[1..])
        {
            args.extend([flag.to_string(), value.clone()]);
        }
        args.extend([
            "--eval-tokens".into(),
            "256".into(),
            "--seed".into(),
            "42".into(),
        ]);
        Ok(args)
    }
    pub fn editing(&self) -> bool {
        self.editing
    }
    pub fn start(&mut self) {
        if self.job.is_some() {
            self.note = "Benchmark already running; Esc cancels the child.".into();
            return;
        }
        match self.args().and_then(|args| Job::start(&args)) {
            Ok(job) => {
                self.job = Some(job);
                self.logs.clear();
                self.result = None;
                self.note = "Matching targets, updates, tokenizer and schedule; CPU baseline may take time. Esc cancels.".into();
            }
            Err(e) => self.note = e,
        }
    }
    pub fn poll(&mut self) {
        let Some(job) = &mut self.job else {
            return;
        };
        let done = match job.poll() {
            Ok((lines, done)) => {
                for line in lines {
                    self.logs.push_back(line);
                    if self.logs.len() > 100 {
                        self.logs.pop_front();
                    }
                }
                done
            }
            Err(e) => {
                self.note = format!("Benchmark process error: {e}");
                Some(false)
            }
        };
        if let Some(ok) = done {
            self.job = None;
            if ok {
                let path = PathBuf::from(&self.fields[2]).join("results.json");
                let result = (|| {
                    let mut text = String::new();
                    std::fs::File::open(path)
                        .map_err(|e| e.to_string())?
                        .take(64 * 1024)
                        .read_to_string(&mut text)
                        .map_err(|e| e.to_string())?;
                    serde_json::from_str(&text).map_err(|e| format!("Invalid result card: {e}"))
                })();
                match result {
                    Ok(value) => {
                        self.result = Some(value);
                        self.note = "Matched benchmark complete. Manifest, curves and results saved in output directory.".into();
                    }
                    Err(e) => {
                        self.note = e;
                        self.notification = Some((false, self.note.clone()));
                        return;
                    }
                }
            } else {
                self.note = "Benchmark failed; see log below. Preflight rejects overlap or mismatched chain settings.".into();
            }
            self.notification = Some((ok, self.note.clone()));
        }
    }
    pub fn key(&mut self, key: KeyEvent) {
        if self.editing {
            match key.code {
                KeyCode::Enter | KeyCode::Esc => self.editing = false,
                KeyCode::Backspace => {
                    self.fields[self.selected].pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.fields[self.selected].clear()
                }
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key.modifiers.contains(KeyModifiers::CONTROL)
                        && self.fields[self.selected].len() < 4096 =>
                {
                    self.fields[self.selected].push(c)
                }
                _ => {}
            }
        } else {
            match key.code {
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => self.selected = (self.selected + 1).min(6),
                KeyCode::Enter if self.job.is_none() => self.editing = true,
                KeyCode::Char('b') => self.start(),
                KeyCode::Esc if self.job.is_some() => {
                    self.job = None;
                    self.note = "Benchmark cancelled; partial files retained. Choose a new output for another run.".into();
                }
                _ => {}
            }
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines = vec![
            Line::styled("MATCHED / PSSA vs transformer", accent()),
            Line::from(
                "Replays the PSSA chain; trains ONLY the baseline; scores both on the next unseen 256 tokens.",
            ),
            Line::from("↑/↓ choose • Enter edit • Ctrl+U clear • b run • Esc stop • Tab tabs"),
        ];
        for (i, label) in LABELS.iter().enumerate() {
            lines.push(Line::from(format!(
                "{} {label}: {}{}",
                if i == self.selected { ">" } else { " " },
                clean(&self.fields[i]),
                if i == self.selected && self.editing {
                    " ▌"
                } else {
                    ""
                }
            )));
        }
        lines.push(Line::from(self.note.as_str()));
        if let Some(v) = &self.result {
            lines.push(Line::styled(
                "RESULT / held-out (not training loss)",
                accent(),
            ));
            for model in ["pssa", "transformer"] {
                lines.push(Line::from(format!(
                    "{model}: ppl {} • accuracy {} • parameters {}",
                    v[model]["perplexity"],
                    v[model]["next_token_accuracy"],
                    v[format!("{model}_parameters")]
                )));
            }
            lines.push(Line::from(format!(
                "Matched targets {} / updates {} / seed 42",
                v["tokens_seen"], v["updates"]
            )));
            lines.push(Line::from(
                "Parameter counts and hardware can differ; this matches exposure, not compute.",
            ));
        }
        for line in self
            .logs
            .iter()
            .rev()
            .take(area.height.saturating_sub(lines.len() as u16 + 2) as usize)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            lines.push(Line::from(line.as_str()));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" benchmark / b run ")),
            area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn benchmark_validation_and_paths_are_not_shell_commands() {
        let dir = std::env::temp_dir().join(format!("oxide-bench-ui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus with spaces.txt");
        std::fs::write(&corpus, "test").unwrap();
        let mut b = Benchmark::new(dir.clone());
        b.fields[0] = corpus.display().to_string();
        b.fields[2] = dir.join("new result").display().to_string();
        let args = b.args().unwrap();
        assert_eq!(args[1], corpus.display().to_string());
        assert!(args.contains(&"--eval-tokens".into()));
        b.fields[3] = "0".into();
        assert!(b.args().is_err());
        b.fields[3] = "1".into();
        b.fields[2] = dir.display().to_string();
        assert!(b.args().is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn benchmark_result_card_renders_wide_and_narrow() {
        let mut b = Benchmark::new("chain".into());
        b.result = Some(
            serde_json::json!({"pssa":{"perplexity":12},"transformer":{"perplexity":15},"updates":8}),
        );
        for (w, h) in [(120, 32), (79, 24), (24, 8), (1, 1)] {
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            t.draw(|f| b.draw(f, f.area())).unwrap();
            if w == 120 {
                let text: String = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("ppl 12"));
                assert!(text.contains("held-out"));
            }
        }
    }
}
