//! Shared TUI widgets: real buttons instead of bare text hints.

use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, BorderType, Borders, Paragraph},
};

/// One button in a row: the label and whether Enter currently presses it.
pub(super) type Button<'a> = (&'a str, bool);

/// Draw a row of buttons, left aligned inside `area` with two spaces between
/// them. A focused button is filled with the accent color so the keyboard
/// focus and the clickable target are obvious; the rest stay outlined.
/// Returns each button's rect so screens can hit-test clicks against it.
pub(super) fn draw_button_row(f: &mut Frame, area: Rect, buttons: &[Button<'_>]) -> Vec<Rect> {
    let mut rects = Vec::with_capacity(buttons.len());
    if area.height < 3 {
        return rects;
    }
    let mut x = area.x;
    let right = area.x.saturating_add(area.width);
    let bottom = area.y.saturating_add(area.height);
    for (label, focused) in buttons {
        let width = (label.chars().count() as u16).saturating_add(4);
        if width <= 4 || x.saturating_add(width) > right {
            break;
        }
        let rect = Rect::new(x, area.y, width, 3);
        draw_button(f, rect, label, *focused);
        rects.push(rect);
        x = x.saturating_add(width).saturating_add(2);
    }
    rects
}

fn draw_button(f: &mut Frame, area: Rect, label: &str, focused: bool) {
    if focused {
        f.render_widget(
            Block::default().style(Style::default().bg(Color::Green)),
            area,
        );
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(if focused {
            Style::default()
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        });
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width > 0 && inner.height > 0 {
        let style = if focused {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        f.render_widget(
            Paragraph::new(Span::styled(label.to_owned(), style)).alignment(Alignment::Center),
            inner,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn buttons_render_as_bordered_boxes_and_fill_the_focused_one() {
        let mut t = Terminal::new(TestBackend::new(40, 8)).unwrap();
        t.draw(|f| {
            draw_button_row(f, f.area(), &[("run", true), ("stop", false)]);
        })
        .unwrap();
        let text: String = t
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol().to_owned())
            .collect();
        assert!(text.contains('╭'), "rounded border missing: {text}");
        assert!(text.contains("run"));
        assert!(text.contains("stop"));
        // The focused button fills its interior with the accent background.
        assert!(t.backend().buffer().content().iter().any(|c| {
            c.bg == Color::Green && c.symbol() == "r"
        }));
    }

    #[test]
    fn buttons_stop_at_the_area_edge_without_panic() {
        let mut t = Terminal::new(TestBackend::new(9, 3)).unwrap();
        t.draw(|f| {
            draw_button_row(f, f.area(), &[("first", true), ("second", false), ("third", false)]);
        })
        .unwrap();
        let text: String = t
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol().to_owned())
            .collect();
        assert!(text.contains("first"));
        assert!(!text.contains("second"));
    }

    #[test]
    fn tiny_areas_draw_nothing() {
        let mut t = Terminal::new(TestBackend::new(4, 2)).unwrap();
        t.draw(|f| {
            draw_button_row(f, f.area(), &[("run", true)]);
        })
        .unwrap();
    }
}
