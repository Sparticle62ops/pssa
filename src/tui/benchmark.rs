//! UI wrapper around the existing, audited token/update-matched `compare` CLI.
use super::{
    accent, panel, panel_area,
    process::{Job, clean},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
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
    launched_output: Option<PathBuf>,
    logs: VecDeque<String>,
    result: Option<serde_json::Value>,
    note: String,
    pub notification: Option<(bool, String)>,
}
impl Benchmark {
    pub fn new(chain: PathBuf) -> Self {
        Self { fields: ["data/downloaded.txt".into(), chain.display().to_string(), "comparison".into(), "64".into(), "200000".into(), "8".into(), "1".into()], selected: 0, editing: false, job: None, launched_output: None, logs: VecDeque::new(), result: None,
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
    pub(super) fn busy(&self) -> bool {
        self.job.is_some()
    }
    pub fn start(&mut self) {
        self.start_with(Job::start);
    }
    fn start_with(&mut self, start: impl FnOnce(&[String]) -> Result<Job, String>) {
        if self.job.is_some() {
            self.note = "Benchmark already running; Esc cancels the child.".into();
            return;
        }
        match self.args().and_then(|args| start(&args)) {
            Ok(job) => {
                self.editing = false;
                self.launched_output = Some(PathBuf::from(&self.fields[2]));
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
            let output = self
                .launched_output
                .take()
                .expect("running benchmark has output snapshot");
            if ok {
                let path = output.join("results.json");
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
                    self.launched_output = None;
                    self.note = "Benchmark cancelled; partial files retained. Choose a new output for another run.".into();
                }
                _ => {}
            }
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        if let Some(result) = self
            .result
            .as_ref()
            .filter(|_| area.width >= 70 && area.height >= 15)
        {
            // Reserve eight data rows after shadows/borders/axes on a tall
            // screen; keep the selected input and run controls usable when short.
            let parts = Layout::vertical([
                Constraint::Min(0),
                Constraint::Length(if area.height >= 26 { 13 } else { 9 }),
            ])
            .split(area);
            // Paint the upper panel first; its shadow must not erase the plot title.
            self.draw_card(f, parts[0]);
            let pssa = result["pssa"]["perplexity"]
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.0);
            let transformer = result["transformer"]["perplexity"]
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.0);
            // Independent braille lollipops: models are categories, not time.
            let a: Vec<_> = pssa
                .into_iter()
                .flat_map(|v| [(0.0, 1.0), (v, 1.0)])
                .collect();
            let b: Vec<_> = transformer
                .into_iter()
                .flat_map(|v| [(0.0, 2.0), (v, 2.0)])
                .collect();
            let max = super::charts::bounds(pssa.into_iter().chain(transformer))[1].max(1.0);
            let label = |v: Option<f64>| v.map(super::charts::number).unwrap_or_else(|| "—".into());
            super::charts::draw_labeled(
                f,
                parts[1],
                super::charts::Plot {
                    title: &format!(
                        " held-out / PSSA ppl {} / transformer ppl {} ",
                        label(pssa),
                        label(transformer)
                    ),
                    caption: "Lower = less surprise on unseen text; matched training exposure",
                    x: "perplexity",
                    integer_x: false,
                    y: "model",
                    x_bounds: [0.0, max],
                    y_bounds: [0.0, 3.0],
                },
                &[
                    super::charts::Series::line("PSSA", &a, super::NORMAL_GREEN),
                    super::charts::Series::line("transformer", &b, super::SECOND_ACCENT),
                ],
                Some(vec![
                    "".into(),
                    "PSSA".into(),
                    "transformer".into(),
                    "".into(),
                ]),
            );
        } else {
            self.draw_card(f, area);
        }
    }

    fn draw_card(&self, f: &mut ratatui::Frame, area: Rect) {
        let area = panel_area(f, area);
        // A taller chart must not hide the matched-result details. On a wide,
        // short card, arrange inputs in two columns instead of dropping rows.
        let paired_fields = self.result.is_some() && area.width >= 100 && area.height < 20;
        let mut lines = vec![
            Line::styled(
                if paired_fields {
                    "MATCHED / PSSA vs transformer / held-out (not training loss)"
                } else {
                    "MATCHED / PSSA vs transformer"
                },
                accent(),
            ),
            Line::from(
                "Replays the PSSA chain; trains ONLY the baseline; scores both on the next unseen 256 tokens.",
            ),
            Line::from("↑/↓ choose • Enter edit • Ctrl+U clear • b run • Esc stop • Tab tabs"),
        ];
        let fields: Vec<_> = LABELS
            .iter()
            .enumerate()
            .map(|(i, label)| {
                format!(
                    "{} {label}: {}{}",
                    if i == self.selected { "▶" } else { " " },
                    clean(&self.fields[i]),
                    if i == self.selected && self.editing {
                        " ▌"
                    } else {
                        ""
                    }
                )
            })
            .collect();
        let field_rows = fields.len().div_ceil(2);
        let selected_line = 3 + if paired_fields {
            let column_width = usize::from(area.width.saturating_sub(4)) / 2;
            for i in 0..field_rows {
                lines.push(Line::from(format!(
                    "{:<column_width$}  {}",
                    fields[i],
                    fields.get(i + field_rows).map_or("", String::as_str),
                )));
            }
            self.selected % field_rows
        } else {
            lines.extend(fields.into_iter().map(Line::from));
            self.selected
        };
        lines.push(Line::from(self.note.as_str()));
        if let Some(v) = &self.result {
            if !paired_fields {
                lines.push(Line::styled(
                    "RESULT / held-out (not training loss)",
                    accent(),
                ));
            }
            for model in ["pssa", "transformer"] {
                lines.push(Line::from(format!(
                    "{model}: ppl {} • accuracy {} • parameters {}",
                    v[model]["perplexity"]
                        .as_f64()
                        .map(super::charts::number)
                        .unwrap_or_else(|| "—".into()),
                    v[model]["next_token_accuracy"]
                        .as_f64()
                        .map(|v| format!("{:.0}%", v * 100.0))
                        .unwrap_or_else(|| "—".into()),
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
        let block = panel(" benchmark / b run ");
        let inner = block.inner(area);
        let selected_row = Paragraph::new(lines[..selected_line].to_vec())
            .wrap(Wrap { trim: false })
            .line_count(inner.width);
        let scroll = selected_row
            .saturating_sub(inner.height.saturating_sub(1) as usize)
            .min(u16::MAX as usize) as u16;
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((scroll, 0))
                .block(block),
            area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn benchmark_validation_and_paths_are_not_shell_commands() {
        let dir = std::env::temp_dir().join(format!("pssa-bench-ui-{}", std::process::id()));
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
    #[cfg(unix)]
    #[test]
    fn palette_launch_finishes_editing_and_keeps_launched_output() {
        let dir = std::env::temp_dir().join(format!("pssa-bench-palette-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, "fixture").unwrap();
        let output = dir.join("new output");
        let mut b = Benchmark::new(dir.clone());
        b.fields[0] = corpus.display().to_string();
        b.fields[2] = output.display().to_string();
        b.selected = 2;
        b.editing = true;
        b.start_with(|_| Job::spawn(std::process::Command::new("true")));
        assert!(b.job.is_some());
        assert!(!b.editing, "palette launch must leave edit mode");
        b.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        b.key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(!b.editing);
        assert_eq!(b.fields[2], output.display().to_string());
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("results.json"), r#"{"pssa":{"perplexity":12}}"#).unwrap();
        // Even a later programmatic configuration change cannot redirect result loading.
        b.fields[2] = dir.join("not the launched output").display().to_string();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while b.job.is_some() && std::time::Instant::now() < deadline {
            b.poll();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(b.result.as_ref().unwrap()["pssa"]["perplexity"], 12);
        assert!(b.notification.as_ref().unwrap().0);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn selected_benchmark_field_stays_visible_below_80_columns() {
        let mut b = Benchmark::new("chain".into());
        b.selected = LABELS.len() - 1;
        b.editing = true;
        for (w, h) in [(79, 12), (40, 10), (24, 8)] {
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            t.draw(|f| b.draw(f, f.area())).unwrap();
            let text: String = t
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(
                text.contains("▶ accumulate"),
                "selected field hidden at {w}x{h}"
            );
        }
    }
    #[test]
    fn benchmark_comparison_has_braille_and_clear_axes_at_both_sizes() {
        let mut b = Benchmark::new("chain".into());
        b.result = Some(
            serde_json::json!({"pssa":{"perplexity":12.087},"transformer":{"perplexity":26.087}}),
        );
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| b.draw(f, f.area())).unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "held-out / PSSA",
                &[
                    "model",
                    "perplexity",
                    "PSSA",
                    "transformer",
                    "Lower = less surprise",
                    "0",
                    "13.4",
                    "26.8",
                ],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "held-out / PSSA",
                if h == 40 { 8 } else { 4 },
            );
            terminal
                .draw(|f| b.draw(f, super::super::feature_area(f.area())))
                .unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "held-out / PSSA",
                &[
                    "model",
                    "perplexity",
                    "PSSA",
                    "transformer",
                    "Lower = less surprise",
                ],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "held-out / PSSA",
                if h == 40 { 8 } else { 4 },
            );
            if h == 40 {
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                for label in LABELS.into_iter().chain([
                    "Enter edit",
                    "not training loss",
                    "accuracy",
                    "Matched targets",
                    "matches exposure, not compute",
                ]) {
                    assert!(
                        text.contains(label),
                        "hidden benchmark information: {label}"
                    );
                }
            }
            // The shorter card must still scroll to every selected input.
            for (index, label) in LABELS.iter().enumerate() {
                b.selected = index;
                terminal
                    .draw(|f| b.draw(f, super::super::feature_area(f.area())))
                    .unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(
                    text.contains(&format!("▶ {label}")),
                    "hidden input at {w}x{h}: {label}"
                );
            }
            b.selected = 0;
        }
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
