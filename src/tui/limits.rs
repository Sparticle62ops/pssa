//! Optional limits for the next training child, not the dashboard process.
use super::{AMBER, BRIGHT_RED, SECOND_ACCENT, accent, panel, panel_area};
use crate::cli::resource_limits::ResourceLimits;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Paragraph, Wrap},
};

const LABELS: [&str; 4] = [
    "Rayon threads",
    "RAM budget (MiB)",
    "Max batch lanes",
    "Max corpus tokens",
];

pub(super) struct Limits {
    values: [String; 4],
    selected: usize,
    edit: Option<String>,
    applied: ResourceLimits,
    message: String,
    error: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            values: std::array::from_fn(|_| String::new()),
            selected: 0,
            edit: None,
            applied: ResourceLimits::default(),
            message: "Blank = unchanged defaults. Apply affects the next training run only.".into(),
            error: false,
        }
    }
}

impl Limits {
    pub(super) fn editing(&self) -> bool {
        self.edit.is_some()
    }

    pub(super) fn applied(&self) -> ResourceLimits {
        self.applied
    }

    pub(super) fn set_limits(&mut self, limits: ResourceLimits) {
        self.applied = limits;
        self.values = [
            limits.threads,
            limits.ram_mib,
            limits.batch_size,
            limits.max_tokens,
        ]
        .map(|n| n.map(|n| n.to_string()).unwrap_or_default());
        self.edit = None;
    }

    /// Returns validated settings only when the user explicitly presses Apply.
    pub(super) fn key(&mut self, key: KeyEvent) -> Option<ResourceLimits> {
        if let Some(edit) = &mut self.edit {
            match key.code {
                KeyCode::Esc => self.edit = None,
                KeyCode::Enter => self.values[self.selected] = self.edit.take().unwrap(),
                KeyCode::Backspace => {
                    edit.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => edit.clear(),
                KeyCode::Char(c)
                    if c.is_ascii_digit()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    if edit.len() < 24 {
                        edit.push(c);
                    }
                }
                _ => {}
            }
            return None;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(LABELS.len()),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = LABELS.len(),
            KeyCode::Char('d') => {
                self.values = std::array::from_fn(|_| String::new());
                self.message = "Defaults restored in draft; select Apply to use them.".into();
                self.error = false;
            }
            KeyCode::Enter | KeyCode::Char(' ') if self.selected < LABELS.len() => {
                self.edit = Some(self.values[self.selected].clone());
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                match ResourceLimits::from_inputs(self.values.each_ref().map(String::as_str)) {
                    Ok(limits) => {
                        self.applied = limits;
                        self.message =
                            "Applied to the next training run. Active jobs are unchanged.".into();
                        self.error = false;
                        return Some(limits);
                    }
                    Err(error) => {
                        self.message = error;
                        self.error = true;
                    }
                }
            }
            _ => {}
        }
        None
    }

    pub(super) fn draw(&mut self, f: &mut Frame, area: Rect) {
        if area.is_empty() {
            return;
        }
        let parts = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);
        f.render_widget(
            Paragraph::new("resource limits / opt in").style(accent()),
            parts[0],
        );
        let available = parts[1].height.saturating_sub(2) as usize;
        let first = self.selected.saturating_sub(available.saturating_sub(1));
        let mut rows = Vec::new();
        for index in first..=LABELS.len() {
            let text = if index == LABELS.len() {
                "[ APPLY TO NEXT TRAINING RUN ]".into()
            } else {
                let value = if self.selected == index {
                    self.edit.as_ref().unwrap_or(&self.values[index])
                } else {
                    &self.values[index]
                };
                format!(
                    "{}: {}{}",
                    LABELS[index],
                    if value.is_empty() { "default" } else { value },
                    if self.selected == index && self.edit.is_some() {
                        "▏"
                    } else {
                        ""
                    }
                )
            };
            rows.push(Line::styled(
                format!("{} {text}", if self.selected == index { "▶" } else { " " }),
                if self.selected == index {
                    accent().add_modifier(Modifier::REVERSED)
                } else {
                    accent()
                },
            ));
        }
        let values_area = panel_area(f, parts[1]);
        f.render_widget(
            Paragraph::new(rows).block(panel(" next run / blank keeps defaults ")),
            values_area,
        );
        let mut args = Vec::new();
        self.applied.append_args(&mut args);
        let semantics_area = panel_area(f, parts[2]);
        f.render_widget(Paragraph::new(vec![
            Line::styled("Threads: local Rayon pool. Changing reduction order may change the last digits.", Style::new().fg(AMBER)),
            Line::from("RAM: Linux prlimit hard address-space (RLIMIT_AS) budget; NOT RSS or VRAM. Too low may abort the child; GPU mappings also count. Other OSes reject it."),
            Line::from("Batch lanes use --batch-size (default 1); not a VRAM cap. Tokens cap the corpus, not each epoch or inference reply."),
            Line::from("No CPU throttling, affinity, or GPU VRAM quota. In-process chat is not RAM-limited."),
            Line::styled(format!("Applied CLI: {}", if args.is_empty() { "(no extra flags)".into() } else { args.join(" ") }), Style::new().fg(SECOND_ACCENT)),
        ]).wrap(Wrap { trim: false }).block(panel(" enforcement / semantics ")), semantics_area);
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    self.message.clone(),
                    Style::new().fg(if self.error { BRIGHT_RED } else { AMBER }),
                ),
                Line::from(if self.editing() {
                    "Digits / Enter save / Esc cancel / Ctrl+U clear"
                } else {
                    "Up/Down select / Enter edit or apply / d defaults"
                }),
            ])
            .wrap(Wrap { trim: false }),
            parts[3],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn press(limits: &mut Limits, code: KeyCode) -> Option<ResourceLimits> {
        limits.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn edits_validate_and_apply_explicitly_without_changing_active_jobs() {
        let mut limits = Limits::default();
        press(&mut limits, KeyCode::Enter);
        press(&mut limits, KeyCode::Char('2'));
        press(&mut limits, KeyCode::Enter);
        assert_eq!(limits.applied(), ResourceLimits::default());
        press(&mut limits, KeyCode::End);
        assert_eq!(press(&mut limits, KeyCode::Enter).unwrap().threads, Some(2));
        limits.values[0] = "0".into();
        assert!(press(&mut limits, KeyCode::Enter).is_none());
        assert!(limits.error);
        assert_eq!(limits.applied().threads, Some(2));
        press(&mut limits, KeyCode::Char('d'));
        assert_eq!(limits.applied().threads, Some(2));
        assert_eq!(
            press(&mut limits, KeyCode::Enter),
            Some(ResourceLimits::default())
        );
    }

    #[test]
    fn editor_cancel_and_clear_keep_limits_opt_in() {
        let mut limits = Limits::default();
        press(&mut limits, KeyCode::Enter);
        press(&mut limits, KeyCode::Char('4'));
        press(&mut limits, KeyCode::Esc);
        assert_eq!(limits.values[0], "");
        press(&mut limits, KeyCode::Enter);
        press(&mut limits, KeyCode::Char('4'));
        limits.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        press(&mut limits, KeyCode::Enter);
        assert_eq!(limits.values[0], "");
    }

    #[test]
    fn test_backend_limits_wide_narrow_and_tiny() {
        for (width, height) in [(120, 30), (79, 24), (60, 24), (30, 12), (1, 1), (0, 0)] {
            let mut limits = Limits::default();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| limits.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 60 {
                assert!(text.contains("▶ Rayon threads"));
                for expected in [
                    "resource limits",
                    "Rayon threads",
                    "RAM budget",
                    "Max batch",
                    "Max corpus",
                    "RLIMIT_AS",
                    "not a VRAM cap",
                ] {
                    assert!(text.contains(expected), "missing {expected} at {width}");
                }
            }
            limits.selected = LABELS.len();
            terminal.draw(|f| limits.draw(f, f.area())).unwrap();
            if width >= 30 {
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("APPLY TO NEXT"));
            }
        }
    }
}
