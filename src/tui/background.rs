//! A sparse, static honeycomb with a bounded, UI-only mouse ripple.
use ratatui::{Frame, layout::Rect, style::Color};
use std::time::Duration;

const REST: (u8, u8, u8) = (13, 31, 22);
const MAX_PRESSED: usize = 64;
const NEIGHBOURS: [(i32, i32); 7] = [(0, 0), (0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)];
type Hex = (i32, i32);

#[derive(Default)]
pub(super) struct HexBackground {
    cursor: Option<Hex>,
    pressed: Vec<(Hex, f32)>,
}

fn center((q, r): Hex) -> (i32, i32) {
    (6 * q, 4 * r + 2 * q)
}

fn hit(x: u16, y: u16) -> Hex {
    let q = i32::from(x) / 6;
    let mut nearest = (0, 0);
    let mut distance = i64::MAX;
    for q in q - 1..=q + 1 {
        let r = (i32::from(y) - 2 * q).div_euclid(4);
        for r in r - 1..=r + 1 {
            let (cx, cy) = center((q, r));
            // Sample the column's centre, not its left edge. Doubled integer
            // coordinates avoid a half-cell tie that breaks shared hex edges.
            let dx = 2 * (i64::from(x) - i64::from(cx)) + 1;
            let dy = 2 * (i64::from(y) - i64::from(cy));
            let d = dx * dx + 4 * dy * dy;
            if d < distance {
                distance = d;
                nearest = (q, r);
            }
        }
    }
    nearest
}

impl HexBackground {
    pub(super) fn mouse(&mut self, column: u16, row: u16) {
        self.cursor = Some(hit(column, row));
    }

    pub(super) fn leave(&mut self) {
        self.cursor = None;
    }

    pub(super) fn advance(&mut self, elapsed: Duration) {
        let dt = elapsed.as_secs_f32().min(0.1);
        if let Some((q, r)) = self.cursor {
            for (dq, dr) in NEIGHBOURS {
                let hex = (q + dq, r + dr);
                if !self.pressed.iter().any(|(h, _)| *h == hex) {
                    // Fast pointer sweeps never grow work or memory without bound.
                    if self.pressed.len() == MAX_PRESSED {
                        let oldest = self
                            .pressed
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
                            .unwrap()
                            .0;
                        self.pressed.swap_remove(oldest);
                    }
                    self.pressed.push((hex, 0.0));
                }
            }
        }
        for (hex, pressure) in &mut self.pressed {
            let target = self.cursor.map_or(0.0, |(q, r)| {
                if *hex == (q, r) {
                    1.0
                } else if NEIGHBOURS.iter().any(|(dq, dr)| *hex == (q + dq, r + dr)) {
                    0.55
                } else {
                    0.0
                }
            });
            // Exponential easing is independent of mouse event frequency.
            let tau = if target > *pressure { 0.08 } else { 0.22 };
            *pressure += (target - *pressure) * (1.0 - (-dt / tau).exp());
        }
        self.pressed.retain(|(_, p)| *p > 0.005);
    }

    pub(super) fn content_area(area: Rect) -> Rect {
        if area.width < 80 || area.height < 16 {
            return area;
        }
        let (x, y) = if area.width >= 100 && area.height >= 28 {
            (4, 2)
        } else {
            (2, 1)
        };
        Rect::new(
            area.x + x,
            area.y + y,
            area.width - 2 * x,
            area.height - 2 * y,
        )
    }

    pub(super) fn draw(&self, frame: &mut Frame) {
        let area = frame.area();
        let content = Self::content_area(area);
        if content == area {
            return;
        }
        // Only gutters are visible: never put decoration behind text, chart
        // labels, or panel borders. Work is proportional to the screen perimeter.
        let buffer = frame.buffer_mut();
        for y in area.y..area.bottom() {
            let ranges = if y >= content.y && y < content.bottom() {
                [area.x..content.x, content.right()..area.right()]
            } else {
                [area.x..area.right(), 0..0]
            };
            for x in ranges.into_iter().flatten() {
                let hex = hit(x, y);
                let (cx, cy) = center(hex);
                let (dx, dy) = (i32::from(x) - cx, i32::from(y) - cy);
                let symbol = match (dx, dy) {
                    (-2..=1, -2 | 2) => "─",
                    (-3, -1) | (-4, 0) | (3, 1) | (2, 2) => "╱",
                    (2, -1) | (3, 0) | (-4, 1) | (-3, 2) => "╲",
                    _ => continue,
                };
                let pressure = self
                    .pressed
                    .iter()
                    .find(|(h, _)| *h == hex)
                    .map_or(0.0, |(_, p)| *p);
                let shade = |value: u8| (f32::from(value) * (1.0 - 0.60 * pressure)).round() as u8;
                buffer[(x, y)].set_symbol(symbol).set_fg(Color::Rgb(
                    shade(REST.0),
                    shade(REST.1),
                    shade(REST.2),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer};

    fn render(background: &HexBackground, width: u16, height: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| background.draw(frame)).unwrap();
        terminal.backend().buffer().clone()
    }

    #[test]
    fn test_backend_static_without_mouse_and_narrow_fallback() {
        let mut background = HexBackground::default();
        let before = render(&background, 100, 30);
        assert!(before.content().iter().any(|c| c.symbol() == "─"));
        let top: String = (3..9).map(|x| before[(x, 0)].symbol()).collect();
        assert_eq!(top, "╲────╱", "shared hex edges must join without gaps");
        for _ in 0..300 {
            background.advance(Duration::from_millis(34));
        }
        assert_eq!(before, render(&background, 100, 30));
        for (w, h) in [(0, 0), (1, 1), (29, 8), (79, 30), (100, 9)] {
            assert!(
                render(&background, w, h)
                    .content()
                    .iter()
                    .all(|c| c.symbol() == " ")
            );
        }
        let content = HexBackground::content_area(before.area);
        for y in content.y..content.bottom() {
            for x in content.x..content.right() {
                assert_eq!(before[(x, y)].symbol(), " ");
            }
        }
    }

    #[test]
    fn test_backend_hover_darks_center_and_six_neighbours_then_eases_back() {
        let mut background = HexBackground::default();
        let rest = render(&background, 100, 30);
        background.mouse(12, 0);
        for _ in 0..10 {
            background.advance(Duration::from_millis(34));
        }
        assert_eq!(background.pressed.len(), 7);
        let pressed = render(&background, 100, 30);
        let changed = rest
            .content()
            .iter()
            .zip(pressed.content())
            .filter(|(a, b)| a.fg != b.fg)
            .count();
        assert!(changed > 0);
        assert!(changed < 50, "hover must stay local");
        for (a, b) in rest.content().iter().zip(pressed.content()) {
            assert_eq!(a.symbol(), b.symbol());
            if let (Color::Rgb(_, ga, _), Color::Rgb(_, gb, _)) = (a.fg, b.fg) {
                assert!(gb <= ga);
            }
        }
        background.leave();
        background.advance(Duration::from_millis(34));
        let easing = render(&background, 100, 30);
        assert_ne!(easing, pressed);
        assert_ne!(easing, rest);
        for _ in 0..60 {
            background.advance(Duration::from_millis(34));
        }
        assert_eq!(rest, render(&background, 100, 30));
    }

    #[test]
    fn hit_test_and_pointer_sweeps_are_bounded() {
        assert_eq!(hit(12, 8), (2, 1));
        let mut background = HexBackground::default();
        for x in 0..1000 {
            background.mouse(x, x);
            background.advance(Duration::from_millis(1));
            assert!(background.pressed.len() <= MAX_PRESSED);
        }
        background.mouse(u16::MAX, u16::MAX);
        background.advance(Duration::from_millis(34));
        render(&background, 80, 24);
    }
}
