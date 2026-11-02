//! Command palette. Global shortcuts and the single help overlay live in keybindings.
use super::{nav, panel_bg, SCREENS, accent, panel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};

/// One token against one field: 900+ for a substring (earlier is better,
/// word starts get a bonus), 500+ for a subsequence of the letters, None when
/// the token does not appear at all.
fn fuzzy_score(field: &str, token: &str) -> Option<i64> {
    if token.is_empty() {
        return Some(0);
    }
    let field = field.to_lowercase();
    let field_chars: Vec<char> = field.chars().collect();
    if let Some(position) = field.find(token) {
        let word_start = position == 0
            || field_chars
                .get(position.saturating_sub(1))
                .is_some_and(|previous| !previous.is_alphanumeric());
        let mut score = 900 - i64::try_from(position).unwrap_or(i64::MAX);
        if word_start {
            score += 150;
        }
        return Some(score);
    }
    // Subsequence: every token letter appears in the field, in order.
    let token_chars: Vec<char> = token.chars().collect();
    let mut cursor = 0usize;
    let mut gaps = 0i64;
    let mut positions = Vec::with_capacity(token_chars.len());
    for wanted in &token_chars {
        let found = field_chars[cursor..]
            .iter()
            .position(|candidate| candidate == wanted)?;
        gaps += found as i64;
        cursor += found + 1;
        positions.push(cursor - 1);
    }
    let word_start = positions.first().is_some_and(|first| {
        *first == 0
            || field_chars
                .get(first.saturating_sub(1))
                .is_some_and(|previous| !previous.is_alphanumeric())
    });
    let mut score = 500 - gaps * 2;
    if word_start {
        score += 100;
    }
    Some(score)
}

/// Total palette rank for one entry: every token must match at least one
/// field; title hits count three times a blurb hit.
fn entry_score(title: &str, category: &str, blurb: &str, tokens: &[String]) -> Option<i64> {
    let mut total = 0i64;
    for token in tokens {
        let title_hit = fuzzy_score(title, token).map(|score| score * 3);
        let category_hit = fuzzy_score(category, token).map(|score| score * 2);
        let blurb_hit = fuzzy_score(blurb, token);
        let best = title_hit
            .into_iter()
            .chain(category_hit)
            .chain(blurb_hit)
            .max()?;
        total += best;
    }
    Some(total)
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
        // Each entry carries its category and blurb, so the palette doubles
        // as a browsable table of what every screen does.
        let mut all: Vec<_> = SCREENS
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let detail = nav::CATEGORIES
                    .iter()
                    .find_map(|category| {
                        category
                            .sections
                            .iter()
                            .find(|section| section.tab == i)
                            .map(|section| format!(" \u{2014} {} \u{b7} {}", category.name, section.blurb))
                    })
                    .unwrap_or_default();
                (format!("Open {name}{detail}"), Action::Tab(i))
            })
            .collect();
        all.push((
            "Run matched benchmark (configured corpus/chain)".into(),
            Action::Benchmark,
        ));
        all.push(("Quit".into(), Action::Quit));
        let tokens: Vec<String> = self
            .query
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect();
        if tokens.is_empty() {
            return all;
        }
        // Lenient search: every token must hit somewhere (title, category or
        // blurb), as a substring or just its letters in order. Entries rank by
        // how well they match, so exact hits still land on top.
        let mut scored: Vec<(i64, usize, (String, Action))> = all
            .into_iter()
            .enumerate()
            .filter_map(|(order, (label, action))| {
                let (title, rest) = match label.split_once(" \u{2014} ") {
                    Some(split) => split,
                    None => (label.as_str(), ""),
                };
                let (category, blurb) = match rest.split_once(" \u{b7} ") {
                    Some(split) => split,
                    None => (rest, ""),
                };
                entry_score(title, category, blurb, &tokens)
                    .map(|score| (score, order, (label, action)))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().map(|(_, _, entry)| entry).collect()
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
                .style(accent().bg(panel_bg()))
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
        for (tab, name) in SCREENS.iter().enumerate() {
            o.toggle();
            for c in format!("Open {name}").chars() {
                o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
            }
            assert!(
                !o.choices().is_empty(),
                "{name}: the exact screen name must still match"
            );
            assert_eq!(
                o.choices().first().map(|(_, action)| action.clone()),
                Some(Action::Tab(tab)),
                "{name}: the exact screen name must rank first"
            );
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
    #[test]
    fn palette_search_is_lenient_and_ranked() {
        use super::super::keybindings;
        // Missing letters still find the screen, ranked first.
        let mut o = Overlay::default();
        o.toggle();
        for c in "hflog".chars() {
            o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert_eq!(
            o.choices().first().map(|(_, action)| action.clone()),
            Some(Action::Tab(keybindings::HF_TAB))
        );
        let mut o = Overlay::default();
        o.toggle();
        for c in "kagl".chars() {
            o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert_eq!(
            o.choices().first().map(|(_, action)| action.clone()),
            Some(Action::Tab(keybindings::KAGGLE_TAB))
        );
        // Several tokens must all hit somewhere.
        let mut o = Overlay::default();
        o.toggle();
        for c in "settings backup".chars() {
            o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert!(o
            .choices()
            .iter()
            .any(|(label, _)| label.contains("backup")));
        // An empty query lists everything.
        let mut o = Overlay::default();
        o.toggle();
        assert_eq!(o.choices().len(), SCREENS.len() + 2);
        // Gibberish matches nothing instead of everything.
        let mut o = Overlay::default();
        o.toggle();
        for c in "zzzz".chars() {
            o.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        assert!(o.choices().is_empty());
    }

}
