//! Convert screen: turn a dense Hugging Face checkpoint into a PSSA v6 file.

use crate::convert::{self, ConvertOptions};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Instant;

const FIELD_LABELS: [&str; 5] = [
    "source (HF checkpoint dir or .safetensors)",
    "config (config.json path)",
    "output (.pssa path to write)",
    "target (compact, source-width, or size like 5b)",
    "seed (integer, default 1234)",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Detected,
    Running,
    Done,
    Failed,
}

pub(super) struct Panel {
    fields: [String; 5],
    focus: usize,
    phase: Phase,
    summary: Vec<String>,
    note: Option<(String, bool)>,
    info: Option<convert::detect::ArchInfo>,
    shards: Vec<PathBuf>,
    job: Option<Receiver<Result<String, String>>>,
    started: Option<Instant>,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            fields: [
                String::new(),
                String::new(),
                String::new(),
                "compact".to_owned(),
                "1234".to_owned(),
            ],
            focus: 0,
            phase: Phase::Idle,
            summary: Vec::new(),
            note: None,
            info: None,
            shards: Vec::new(),
            job: None,
            started: None,
        }
    }
}

impl Panel {
    /// Move a finished background conversion into the phase machine.
    pub(super) fn poll(&mut self) {
        if self.phase != Phase::Running {
            return;
        }
        let Some(job) = &self.job else {
            self.fail("conversion worker stopped without a result");
            return;
        };
        match job.try_recv() {
            Ok(Ok(report)) => {
                self.phase = Phase::Done;
                self.summary = report.lines().map(str::to_owned).collect();
                self.note = None;
                self.job = None;
                self.started = None;
            }
            Ok(Err(error)) => self.fail(&error),
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.fail("conversion worker stopped without a result");
            }
        }
    }

    fn fail(&mut self, message: &str) {
        self.phase = Phase::Failed;
        self.note = Some((message.to_owned(), true));
        self.job = None;
        self.started = None;
    }

    pub(super) fn key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        use crossterm::event::KeyModifiers;
        match key.code {
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.fields[self.focus].clear();
                self.invalidate();
            }
            KeyCode::Char(c) if !c.is_control() => {
                self.fields[self.focus].push(c);
                self.invalidate();
            }
            KeyCode::Backspace => {
                self.fields[self.focus].pop();
                self.invalidate();
            }
            KeyCode::Up => self.focus = (self.focus + FIELD_LABELS.len() - 1) % FIELD_LABELS.len(),
            KeyCode::Down => self.focus = (self.focus + 1) % FIELD_LABELS.len(),
            KeyCode::Enter => match self.phase {
                Phase::Idle | Phase::Failed | Phase::Done => self.detect(),
                Phase::Detected => self.start(),
                Phase::Running => {
                    self.note = Some(("conversion already running".to_owned(), false));
                }
            },
            _ => {}
        }
    }

    /// Editing a field after a detection invalidates the stale summary.
    fn invalidate(&mut self) {
        if self.phase == Phase::Detected {
            self.phase = Phase::Idle;
            self.summary.clear();
            self.info = None;
            self.shards.clear();
        }
    }

    fn target(&self) -> Result<String, String> {
        match self.fields[3].trim() {
            "" => Ok("compact".to_owned()),
            "compact" | "source-width" => Ok(self.fields[3].trim().to_owned()),
            other => {
                if crate::convert::parse_target_size(other).is_some() {
                    Ok(self.fields[3].trim().to_owned())
                } else {
                    Err(format!(
                        "target must be compact, source-width, or a size like 5b, got '{other}'"
                    ))
                }
            }
        }
    }

    fn seed(&self) -> Result<u64, String> {
        match self.fields[4].trim() {
            "" => Ok(1234),
            raw => raw
                .parse::<u64>()
                .map_err(|_| format!("seed must be an integer, got '{raw}'")),
        }
    }

    /// Read the config, detect the family and discover the shards.
    fn detect(&mut self) {
        self.note = None;
        self.summary.clear();
        self.info = None;
        self.shards.clear();
        if let Err(error) = self.target().and_then(|_| self.seed()) {
            self.note = Some((error, true));
            return;
        }
        if self.fields[0].trim().is_empty() {
            self.note = Some(("source path is empty".to_owned(), true));
            return;
        }
        let config_path = self.fields[1].trim().to_owned();
        if config_path.is_empty() {
            self.note = Some(("config path is empty".to_owned(), true));
            return;
        }
        let config_text = match std::fs::read_to_string(&config_path) {
            Ok(text) => text,
            Err(error) => {
                self.note = Some((format!("{config_path}: {error}"), true));
                return;
            }
        };
        let config: serde_json::Value = match serde_json::from_str(&config_text) {
            Ok(value) => value,
            Err(error) => {
                self.note = Some((format!("{config_path}: malformed JSON: {error}"), true));
                return;
            }
        };
        let source = PathBuf::from(self.fields[0].trim());
        match convert::plan_from_paths(&source, &config) {
            Ok((info, shards)) => {
                self.summary = vec![
                    format!(
                        "detected {} (model_type '{}')",
                        info.family.label(),
                        info.model_type
                    ),
                    format!(
                        "layers {}, hidden {}, vocab {}",
                        info.num_hidden_layers, info.hidden_size, info.vocab_size
                    ),
                    format!("{} shard(s) found", shards.len()),
                ];
                self.info = Some(info);
                self.shards = shards;
                self.phase = Phase::Detected;
            }
            Err(error) => {
                // Covers missing shards and hybrid families alike: the message
                // explains which architectures the converter cannot read yet.
                self.note = Some((error, true));
                self.phase = Phase::Idle;
            }
        }
    }

    /// Run the conversion in the background and report the result via poll.
    fn start(&mut self) {
        let Some(info) = self.info.take() else {
            self.phase = Phase::Idle;
            return;
        };
        let output = self.fields[2].trim().to_owned();
        if output.is_empty() {
            self.note = Some(("output path is empty".to_owned(), true));
            return;
        }
        let target = match self.target() {
            Ok(target) => target,
            Err(error) => {
                self.note = Some((error, true));
                return;
            }
        };
        let seed = match self.seed() {
            Ok(seed) => seed,
            Err(error) => {
                self.note = Some((error, true));
                return;
            }
        };
        let shards = std::mem::take(&mut self.shards);
        let out_path = PathBuf::from(output);
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = convert::convert(ConvertOptions {
                shards,
                info,
                target_mode: target,
                seed,
                out_path,
            })
            .map(|report| report.summary());
            let _ = sender.send(result);
        });
        self.job = Some(receiver);
        self.phase = Phase::Running;
        self.started = Some(Instant::now());
        self.note = None;
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for (index, label) in FIELD_LABELS.iter().enumerate() {
            let focused = index == self.focus;
            let value = &self.fields[index];
            let cursor = if focused && self.phase != Phase::Running {
                "\u{258f}"
            } else {
                ""
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{label}: "),
                    if focused {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
                Span::styled(
                    format!("{value}{cursor}"),
                    if focused {
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    },
                ),
            ]));
        }
        lines.push(Line::from(String::new()));
        match self.phase {
            Phase::Idle => {
                lines.push(Line::from(
                    "Enter: detect the checkpoint family. Up/Down switch fields, Ctrl+U clears the focused one.",
                ));
                lines.push(Line::from(
                    "Hybrid linear-attention models (Qwen3.5/3.6/3.8) are rejected here with the reason.",
                ));
            }
            Phase::Detected => {
                for line in &self.summary {
                    lines.push(Line::from(line.clone()));
                }
                lines.push(Line::from("Enter: run the conversion now."));
            }
            Phase::Running => {
                let elapsed = self.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
                lines.push(Line::from(format!(
                    "converting\u{2026} {elapsed}s, {} shard(s)",
                    self.shards.len()
                )));
                lines.push(Line::from(
                    "working in the background; the report appears here when it finishes.",
                ));
            }
            Phase::Done => {
                for line in &self.summary {
                    lines.push(Line::from(line.clone()));
                }
                lines.push(Line::from("Enter: detect another checkpoint."));
            }
            Phase::Failed => {
                lines.push(Line::from("Enter: fix the fields and detect again."));
            }
        }
        if let Some((text, is_error)) = &self.note {
            lines.push(Line::from(Span::styled(
                text.clone(),
                if *is_error {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Green)
                },
            )));
        }
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " convert ",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(
            Paragraph::new(lines).block(block).wrap(Wrap { trim: false }),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn press(panel: &mut Panel, code: KeyCode) {
        panel.key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    #[test]
    fn typing_and_field_navigation() {
        let mut panel = Panel::default();
        press(&mut panel, KeyCode::Char('a'));
        assert_eq!(panel.fields[0], "a");
        press(&mut panel, KeyCode::Down);
        press(&mut panel, KeyCode::Char('b'));
        assert_eq!(panel.fields[1], "b");
        press(&mut panel, KeyCode::Backspace);
        assert_eq!(panel.fields[1], "");
        press(&mut panel, KeyCode::Up);
        assert_eq!(panel.focus, 0);
        panel.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert_eq!(panel.fields[0], "");
    }

    #[test]
    fn detect_rejects_empty_fields_with_an_error() {
        let mut panel = Panel::default();
        press(&mut panel, KeyCode::Enter);
        assert_eq!(panel.phase, Phase::Idle);
        let (_, is_error) = panel.note.expect("error note after empty detect");
        assert!(is_error);
    }

    #[test]
    fn detect_reads_a_checkpoint_directory_and_reports_the_family() {
        let dir = std::env::temp_dir().join(format!("pssa-convert-panel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type": "qwen3", "num_hidden_layers": 4, "hidden_size": 64, "vocab_size": 128}"#,
        )
        .unwrap();
        let mut panel = Panel::default();
        panel.fields[0] = dir.display().to_string();
        panel.fields[1] = dir.join("config.json").display().to_string();
        press(&mut panel, KeyCode::Enter);
        assert_eq!(panel.phase, Phase::Detected);
        assert!(panel.summary.iter().any(|line| line.contains("1 shard")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unsupported_families_are_rejected_with_a_message() {
        let dir = std::env::temp_dir().join(format!("pssa-convert-panel-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type": "totally-made-up-arch"}"#,
        )
        .unwrap();
        let mut panel = Panel::default();
        panel.fields[0] = dir.display().to_string();
        panel.fields[1] = dir.join("config.json").display().to_string();
        press(&mut panel, KeyCode::Enter);
        assert_eq!(panel.phase, Phase::Idle);
        let (_, is_error) = panel.note.expect("error note for unsupported family");
        assert!(is_error);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_panel_renders_with_a_test_backend() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let panel = Panel::default();
        terminal
            .draw(|f| panel.draw(f, f.area()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol().to_owned())
            .collect();
        assert!(text.contains("convert"));
        assert!(text.contains("source (HF checkpoint dir"));
        assert!(text.contains("config (config.json path)"));
        assert!(text.contains("target (compact, source-width, or size like 5b)"));
        assert!(text.contains("Enter: detect the checkpoint family"));
    }
}
