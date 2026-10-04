//! Read-only community links and the README's exact SOL donation address.
use super::{accent, network, panel};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::sync::mpsc;

pub(super) const SOL: &str = "4XPZ9uAa2BMoth6msoHRxTWL4mUrMfq3LGrxbAGja96h";
const LINKS: [(&str, &str); 3] = [
    (
        "GitHub repository",
        "https://github.com/Sparticle62ops/pssa",
    ),
    ("Discord / PSlabs", "https://discord.gg/9sqfKeqWYF"),
    (
        "Issues / bug reports",
        "https://github.com/Sparticle62ops/pssa/issues",
    ),
];

pub(super) struct Support {
    selected: usize,
    status: String,
    pending: Option<mpsc::Receiver<Result<String, String>>>,
}
impl Default for Support {
    fn default() -> Self {
        Self {
            selected: 0,
            status: "Nothing is uploaded. Donations are optional.".into(),
            pending: None,
        }
    }
}
impl Support {
    pub(super) fn poll(&mut self) {
        let result = match self.pending.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(value)) => value,
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                Err("Desktop helper stopped; use displayed text".into())
            }
            _ => return,
        };
        self.pending = None;
        self.status = result.unwrap_or_else(|e| e);
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(LINKS.len() - 1),
            KeyCode::Char('c') if self.pending.is_none() => self.start(true),
            KeyCode::Enter | KeyCode::Char('o') if self.pending.is_none() => self.start(false),
            _ => {}
        }
    }
    fn start(&mut self, copy: bool) {
        let (tx, rx) = mpsc::sync_channel(1);
        self.pending = Some(rx);
        self.status = if copy {
            "Copying SOL address…"
        } else {
            "Opening selected link…"
        }
        .into();
        let url = LINKS[self.selected].1;
        std::thread::spawn(move || {
            let result = if copy {
                network::copy_text(SOL).map(|_| "SOL address copied to clipboard".into())
            } else {
                network::open_browser(url).map(|_| "Link opened in browser".into())
            };
            let _ = tx.send(result);
        });
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::styled("SUPPORT / PSSA COMMUNITY", accent()),
            Line::from("Up/Down choose / Enter or o open / c copy SOL"),
            Line::from(""),
        ];
        for (i, (label, url)) in LINKS.iter().enumerate() {
            lines.push(Line::styled(
                format!("{} {label}", if i == self.selected { "▶" } else { " " }),
                accent(),
            ));
            lines.push(Line::from(*url));
        }
        lines.extend([
            Line::from(""),
            Line::styled("DONATE / Solana (SOL)", accent()),
        ]);
        // Preserve every address character even when no clipboard is available
        // and the address is longer than the terminal's entire width.
        let width = usize::from(area.width.saturating_sub(2).max(1));
        for part in SOL.as_bytes().chunks(width) {
            lines.push(Line::from(std::str::from_utf8(part).unwrap()));
        }
        lines.push(Line::from(self.status.as_str()));
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" support ")),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    #[test]
    fn donation_matches_readme_character_for_character() {
        assert_eq!(SOL, "4XPZ9uAa2BMoth6msoHRxTWL4mUrMfq3LGrxbAGja96h");
        assert!(include_str!("../../README.md").contains(SOL));
        assert!(include_str!("../../README.md").contains(LINKS[1].1));
    }
    #[test]
    fn support_is_readable_wide_narrow_and_tiny_without_clipboard() {
        for (width, height) in [(110, 30), (79, 28), (48, 30), (24, 40), (1, 1), (0, 0)] {
            let mut t = Terminal::new(TestBackend::new(width, height)).unwrap();
            let page = Support::default();
            t.draw(|f| page.draw(f, f.area())).unwrap();
            let text: String = t
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 48 {
                assert!(text.contains(SOL));
                assert!(text.contains("Discord"));
                assert!(text.contains("c copy SOL"));
            }
            assert!(page.pending.is_none());
        }
    }
    #[test]
    fn unavailable_clipboard_reports_text_fallback() {
        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(Err("No clipboard; copy displayed address".into()))
            .unwrap();
        let mut page = Support {
            pending: Some(rx),
            ..Support::default()
        };
        page.poll();
        assert!(page.status.contains("displayed address"));
        assert!(page.pending.is_none());
    }
}
