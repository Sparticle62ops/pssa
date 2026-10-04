//! Global navigation without stealing printable characters from text entry.
use super::{TABS, accent, panel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};

pub(super) enum Action {
    Tab(usize),
    Benchmark,
    Quit,
}
#[derive(Default)]
pub(super) struct Overlay {
    mode: u8,
    query: String,
    selected: usize,
    scroll: u16,
    scroll_max: u16,
}
const HELP: &str = "GLOBAL\nTab next tab • Ctrl+C quit • Ctrl+K command palette • F1 help\n? help / q quit outside text entry • Esc quit on monitor/chain/model/feed\n←/→ previous/next tab outside monitor and text entry\nMONITOR\ng next graph • 1 loss • 2 perplexity • 3 speed • 4 learning rate\n5 comparison • 6 all • 7 memory • +/- or = zoom • ←/→ pan • 0 reset\nINFERENCE\nEnter send/command • Esc stop • PgUp/PgDn scroll • End follow\nBackspace edit • Ctrl+U clear • /help command reference\n/model PATH • /new • /chats • /open ID • /rename NAME • /delete ID\n/temp F • /top-p F • /top-k N • /max-tokens N • /repetition-penalty F\n/system TEXT • /attach PATH • /copy • /speech • /stop\nA/B: /ab a PATH • /ab b PATH • /ab on • /ab off (see /help)\nHF LOGIN\nEnter login • Backspace edit • Esc clear • Ctrl+L logout\nKAGGLE\nFolder path: Enter validate • Backspace edit • Ctrl+U clear\nLaunch preview: y confirm • n/Esc cancel • Esc detach log follow\nMEMORY INSPECTOR\nr refresh saved snapshot • PgUp/PgDn scroll\nRUNS\n↑/↓ select • r refresh • Enter monitor • c chat • s score • Esc stop\nScore path: Enter run • Backspace edit • Ctrl+U clear • Esc cancel\nBENCHMARK\n↑/↓ choose field • Enter edit • b run matched comparison • Esc stop\nEditing: Enter/Esc finish • Backspace edit • Ctrl+U clear\nPALETTE / HELP\nPalette: type to filter • ↑/↓ select • Enter execute • Backspace edit\nEsc close overlay • help ↑/↓ or PgUp/PgDn scroll";
impl Overlay {
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
    pub fn key(&mut self, key: KeyEvent, text_entry: bool) -> (bool, Option<Action>) {
        if key.code == KeyCode::Char('k') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.mode = if self.mode == 1 { 0 } else { 1 };
            self.query.clear();
            self.selected = 0;
            return (true, None);
        }
        if key.code == KeyCode::F(1)
            || (key.code == KeyCode::Char('?') && !text_entry && self.mode == 0)
        {
            self.mode = if self.mode == 2 { 0 } else { 2 };
            self.scroll = 0;
            return (true, None);
        }
        if self.mode == 0 {
            return (false, None);
        }
        match key.code {
            KeyCode::Esc => self.mode = 0,
            KeyCode::Up if self.mode == 1 => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down if self.mode == 1 => {
                self.selected = (self.selected + 1).min(self.choices().len().saturating_sub(1))
            }
            KeyCode::Enter if self.mode == 1 => {
                let selected = self
                    .choices()
                    .into_iter()
                    .nth(self.selected)
                    .map(|(_, action)| action);
                if selected.is_some() {
                    self.mode = 0;
                }
                return (true, selected);
            }
            KeyCode::Backspace if self.mode == 1 => {
                self.query.pop();
                self.selected = 0;
            }
            KeyCode::Char(c)
                if self.mode == 1
                    && !c.is_control()
                    && !key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.query.len() < 128 =>
            {
                self.query.push(c);
                self.selected = 0;
            }
            KeyCode::Down | KeyCode::PageDown if self.mode == 2 => {
                self.scroll = self.scroll.saturating_add(4).min(self.scroll_max)
            }
            KeyCode::Up | KeyCode::PageUp if self.mode == 2 => {
                self.scroll = self.scroll.saturating_sub(4)
            }
            _ => {}
        }
        (true, None)
    }
    pub fn draw(&mut self, f: &mut ratatui::Frame) {
        if self.mode == 0 {
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
        if self.mode == 2 {
            self.scroll_max = Paragraph::new(HELP).wrap(Wrap { trim: false })
                .line_count(area.width.saturating_sub(2))
                .saturating_sub(area.height.saturating_sub(2) as usize)
                .min(u16::MAX as usize) as u16;
            self.scroll = self.scroll.min(self.scroll_max);
        }
        let lines = if self.mode == 2 {
            HELP.lines().map(Line::from).collect()
        } else {
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
                    if i == self.selected { ">" } else { " " }
                )));
            }
            lines
        };
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((if self.mode == 2 { self.scroll } else { 0 }, 0))
                .block(panel(if self.mode == 2 {
                    " help / PgUp PgDn / Esc "
                } else {
                    " command palette / Ctrl+K "
                })),
            area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn palette_filters_routes_and_preserves_question_marks_in_text() {
        let mut o = Overlay::default();
        assert!(
            !o.key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE), true)
                .0
        );
        assert!(
            o.key(
                KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL),
                true
            )
            .0
        );
        for c in "memory".chars() {
            o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE), true);
        }
        let (_, action) = o.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), true);
        assert!(matches!(action, Some(Action::Tab(7))));
        assert_eq!(o.mode, 0);
    }
    #[test]
    fn help_can_scroll_to_final_keys_after_narrow_wrapping() {
        let mut o = Overlay { mode: 2, ..Default::default() };
        let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(24,8)).unwrap();
        t.draw(|f|o.draw(f)).unwrap();
        assert!(o.scroll_max as usize > HELP.lines().count());
        for _ in 0..100 { o.key(KeyEvent::new(KeyCode::PageDown,KeyModifiers::NONE),false); }
        t.draw(|f|o.draw(f)).unwrap();
        let s: String = t.backend().buffer().content.iter().map(|c|c.symbol()).collect();
        assert!(s.contains("scroll")); assert_eq!(o.scroll,o.scroll_max);
    }
    #[test]
    fn overlays_render_tiny_narrow_wide() {
        for mode in [1, 2] {
            for (w, h) in [(120, 40), (79, 24), (20, 6), (1, 1)] {
                let mut o = Overlay {
                    mode,
                    ..Default::default()
                };
                let mut t =
                    ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
                t.draw(|f| o.draw(f)).unwrap();
                if w == 120 {
                    let s: String = t
                        .backend()
                        .buffer()
                        .content
                        .iter()
                        .map(|c| c.symbol())
                        .collect();
                    assert!(s.contains(if mode == 1 {
                        "command palette"
                    } else {
                        "GLOBAL"
                    }));
                }
            }
        }
    }
}
