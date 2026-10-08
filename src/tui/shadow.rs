//! Offset drop shadows for the TUI's retro window panels.
use ratatui::{Frame, layout::Rect, style::{Color, Style}};

const MIN_WIDTH: u16 = 80;
const MIN_HEIGHT: u16 = 24;

pub(super) fn enabled(frame_area: Rect) -> bool {
    frame_area.width >= MIN_WIDTH
        && frame_area.height >= MIN_HEIGHT
        && std::env::var_os("NO_COLOR").is_none()
}

/// Derive the shadow from the panel background instead of choosing a color at
/// each call site. This keeps the effect tied to the active retro palette.
pub(super) fn color(background: Color) -> Color {
    match background {
        Color::Rgb(red, green, blue) => Color::Rgb(red / 2, green / 2, blue / 2),
        Color::Gray => Color::DarkGray,
        Color::White => Color::Gray,
        Color::DarkGray | Color::Black => Color::Black,
        _ => Color::Black,
    }
}

pub(super) fn paint(frame: &mut Frame, panel: Rect, frame_area: Rect, shade: Color) {
    if !enabled(frame_area) || panel.is_empty() {
        return;
    }

    let right = panel.x.saturating_add(panel.width);
    fill(
        frame,
        clip(Rect::new(right, panel.y, 1, panel.height), frame_area),
        shade,
    );

    let bottom = panel.y.saturating_add(panel.height);
    // Include the corner where the two one-cell strips meet.
    fill(
        frame,
        clip(
            Rect::new(panel.x, bottom, panel.width.saturating_add(1), 1),
            frame_area,
        ),
        shade,
    );
}

fn clip(rect: Rect, bounds: Rect) -> Option<Rect> {
    let left = u32::from(rect.x).max(u32::from(bounds.x));
    let top = u32::from(rect.y).max(u32::from(bounds.y));
    let right = (u32::from(rect.x) + u32::from(rect.width))
        .min(u32::from(bounds.x) + u32::from(bounds.width));
    let bottom = (u32::from(rect.y) + u32::from(rect.height))
        .min(u32::from(bounds.y) + u32::from(bounds.height));
    (right > left && bottom > top).then(|| {
        Rect::new(
            left as u16,
            top as u16,
            (right - left) as u16,
            (bottom - top) as u16,
        )
    })
}

fn fill(frame: &mut Frame, area: Option<Rect>, shade: Color) {
    let Some(area) = area else {
        return;
    };
    let buffer = frame.buffer_mut();
    for y in area.y..area.bottom() {
        for x in area.x..area.right() {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_symbol(" ").set_style(Style::default().bg(shade));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer, widgets::{Block, Borders, Paragraph}};

    fn render(width: u16, height: u16, panel: Rect) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| {
            let frame_area = frame.area();
            paint(frame, panel, frame_area, color(Color::Rgb(7, 18, 16)));
        }).unwrap();
        terminal.backend().buffer().clone()
    }

    #[test]
    fn shadow_cells_appear_to_the_right_and_below_a_panel() {
        let panel = Rect::new(4, 5, 10, 6);
        let buffer = render(120, 40, panel);
        let shade = color(Color::Rgb(7, 18, 16));
        for y in panel.y..panel.bottom() {
            assert_eq!(buffer[(panel.right(), y)].bg, shade);
        }
        for x in panel.x..=panel.right() {
            assert_eq!(buffer[(x, panel.bottom())].bg, shade);
        }
    }

    #[test]
    fn narrow_terminals_keep_the_buffer_unchanged() {
        let panel = Rect::new(4, 5, 10, 6);
        let buffer = render(79, 24, panel);
        assert!(buffer.content().iter().all(|cell| cell.bg == Color::Reset));
    }

    #[test]
    fn shadow_is_clipped_at_frame_edges_without_panicking() {
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| {
            let frame_area = frame.area();
            paint(frame, Rect::new(117, 37, 2, 3), frame_area, Color::DarkGray);
        }).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(*buffer.area(), Rect::new(0, 0, 120, 40));
        assert!(buffer.content().iter().any(|cell| cell.bg == Color::DarkGray));
    }

    #[test]
    fn panel_content_is_not_overwritten_by_its_shadow() {
        let panel = Rect::new(3, 3, 16, 8);
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| {
            let frame_area = frame.area();
            paint(frame, panel, frame_area, Color::DarkGray);
            frame.render_widget(
                Paragraph::new("content").block(Block::default().borders(Borders::ALL)),
                panel,
            );
        }).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(panel.x + 2, panel.y + 1)].symbol(), "o");
        assert_eq!(buffer[(panel.right(), panel.y)].bg, Color::DarkGray);
    }
}
