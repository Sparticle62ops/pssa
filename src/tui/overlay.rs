//! Command palette. Global shortcuts and the single help overlay live in keybindings.
use super::{PANEL_BG, TABS, accent, panel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Action {
    Tab(usize),
    Benchmark,
    Quit,
}
#[derive(Default)]
pub(super) struct Overlay {
    pub open: bool,
    query: String,
    selected: usize,
}
impl Overlay {
    pub fn toggle(&mut self) {
        self.open = !self.open;
        self.query.clear();
        self.selected = 0;
    }
    fn choices(&self) -> Vec<(String, Action)> {
        let mut all: Vec<_> = TABS
            .iter()
            .enumerate()
            .map(|(i, name)| (format!("Open {name}"), Action::Tab(i)))
            .collect();
        all.push((
            "Run matched benchmark (configured corpus/chain)".into(),
            Action::Benchmark,
        ));
        all.push(("Quit".into(), Action::Quit));
        let query = self.query.to_lowercase();
        all.into_iter()
            .filter(|(s, _)| s.to_lowercase().contains(&query))
            .collect()
    }
    /// Called only for palette-local actions resolved by the shared registry.
    pub fn key(&mut self, key: KeyEvent) -> Option<Action> {
        if !self.open {
            return None;
        }
        match key.code {
            KeyCode::Esc => self.open = false,
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.choices().len().saturating_sub(1))
            }
            KeyCode::Enter => {
                let selected = self
                    .choices()
                    .into_iter()
                    .nth(self.selected)
                    .map(|(_, action)| action);
                if selected.is_some() {
                    self.open = false;
                }
                return selected;
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.selected = 0;
            }
            KeyCode::Char(c)
                if !c.is_control()
                    && !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && self.query.len() < 128 =>
            {
                self.query.push(c);
                self.selected = 0;
            }
            _ => {}
        }
        None
    }
    pub fn draw(&self, f: &mut ratatui::Frame) {
        if !self.open {
            return;
        }
        let full = f.area();
        let width = full.width.min(100);
        let height = full.height.min(35);
        let area = Rect::new(
            full.x + (full.width - width) / 2,
            full.y + (full.height - height) / 2,
            width,
            height,
        );
        f.render_widget(Clear, area);
        let mut lines = vec![
            Line::styled(format!("Search: {}▌", self.query), accent()),
            Line::from("↑/↓ choose • Enter execute • Esc close"),
        ];
        let choices = self.choices();
        if choices.is_empty() {
            lines.push(Line::from("No matching commands."));
        }
        let visible = area.height.saturating_sub(4) as usize;
        for (i, (label, _)) in choices
            .iter()
            .enumerate()
            .skip(self.selected.saturating_sub(visible.saturating_sub(1)))
            .take(visible)
        {
            lines.push(Line::from(format!(
                "{} {label}",
                if i == self.selected { "▶" } else { " " }
            )));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .style(accent().bg(PANEL_BG))
                .block(panel(" command palette / Ctrl+K ")),
            area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn palette_reaches_every_tab_once_in_shell_order() {
        let mut o = Overlay::default();
        for (tab, name) in TABS.iter().enumerate() {
            o.toggle();
            for c in format!("Open {name}").chars() {
                o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
            }
            assert_eq!(o.choices().len(), 1, "{name}");
            assert_eq!(
                o.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
                Some(Action::Tab(tab))
            );
            assert!(!o.open);
        }
    }
    #[test]
    fn palette_does_not_own_help_or_global_shortcuts() {
        use super::super::keybindings::{self, Action as KeyAction, Context};
        let mut o = Overlay::default();
        o.toggle();
        for c in ['?', 'q'] {
            let key = KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE);
            assert_eq!(
                keybindings::action(key, Context::Palette),
                Some(KeyAction::Palette)
            );
            o.key(key);
        }
        assert_eq!(o.query, "?q");
        for (code, expected) in [
            (KeyCode::Tab, KeyAction::NextTab),
            (KeyCode::F(1), KeyAction::ToggleHelp),
        ] {
            assert_eq!(
                keybindings::action(KeyEvent::new(code, KeyModifiers::NONE), Context::Palette),
                Some(expected)
            );
        }
        o.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!o.open);
    }
    #[test]
    fn palette_renders_tiny_narrow_wide() {
        for (w, h) in [(120, 40), (79, 24), (20, 6), (1, 1), (0, 0)] {
            let mut o = Overlay::default();
            o.toggle();
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            t.draw(|f| o.draw(f)).unwrap();
            if w == 120 {
                let s: String = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(s.contains("command palette"));
                assert!(s.contains("Open setup"));
                assert!(s.contains("Open HF login"));
            }
        }
    }
}
