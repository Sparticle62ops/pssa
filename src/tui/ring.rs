//! Shared-network recurrence: one electric lap for each selected pass.
//! The caller supplies its UI clock, so idle monitor and setup stay in sync.
use super::{draw_neuron_frame, neuron_frame_at, neuron_green, panel, panel_area};
use ratatui::{
    Frame,
    layout::Rect,
    style::Color,
    symbols::Marker,
    widgets::canvas::{Canvas, Line, Points},
};
use std::{f64::consts::TAU, time::Duration};

const MAX_LOOPS: usize = 32;
const NODES: usize = 40;
const LAP: Duration = Duration::from_millis(2_400);
const ELECTRIC: Color = Color::Rgb(0x86, 0xff, 0xa6);

type Point = (f64, f64);

#[derive(Clone, Copy, Debug)]
struct RingFrame {
    loops: usize,
    pass: usize,
    phase: f64,
}

fn frame_at(loops: usize, elapsed: Duration) -> RingFrame {
    let loops = loops.clamp(1, MAX_LOOPS);
    // Integer division keeps the counter and pulse in agreement, including
    // exact pass boundaries and very long-running monitor sessions.
    let lap = elapsed.as_nanos() / LAP.as_nanos();
    RingFrame {
        loops,
        pass: (lap % loops as u128) as usize + 1,
        phase: (elapsed.as_nanos() % LAP.as_nanos()) as f64 / LAP.as_nanos() as f64,
    }
}

fn oval(aspect: f64) -> [Point; NODES] {
    let rx = 0.87 * aspect;
    // Keep the long axis horizontal even in a tall, narrow terminal.
    let ry = 0.73_f64.min(rx / 1.8);
    std::array::from_fn(|index| {
        let angle = TAU * index as f64 / NODES as f64;
        (rx * angle.cos(), ry * angle.sin())
    })
}

fn connection_point(nodes: &[Point; NODES], phase: f64) -> Point {
    let position = phase.rem_euclid(1.0) * NODES as f64;
    let index = position.floor() as usize % NODES;
    let t = position.fract();
    let (ax, ay) = nodes[index];
    let (bx, by) = nodes[(index + 1) % NODES];
    (ax + (bx - ax) * t, ay + (by - ay) * t)
}

/// Draw a recurring oval for 2..=32 loops; one pass keeps the Phase 5 neuron.
/// `elapsed` is UI animation time, not a trainer/model progress measurement.
pub(super) fn draw(f: &mut Frame, area: Rect, loops: usize, elapsed: Duration) {
    let area = area.intersection(f.area());
    if area.is_empty() {
        return;
    }
    let frame = frame_at(loops, elapsed);
    // Put the selected count first so even narrow panels retain that value.
    let title = format!(
        " loops {} / configuration diagram ",
        frame.loops
    );
    if frame.loops == 1 {
        draw_neuron_frame(f, area, neuron_frame_at(elapsed), &title);
        return;
    }
    let area = panel_area(f, area);
    let inner = panel(&title).inner(area);
    if inner.width < 2 || inner.height < 2 {
        f.render_widget(panel(&title), area);
        return;
    }
    let pixel_width = f64::from(inner.width) * 2.0 - 1.0;
    let pixel_height = f64::from(inner.height) * 4.0 - 1.0;
    let aspect = pixel_width / pixel_height;
    let nodes = oval(aspect);
    let inner_nodes: [Point; NODES] = std::array::from_fn(|index| {
        let (x, y) = connection_point(&nodes, (index as f64 + 0.5) / NODES as f64);
        (x * 0.73, y * 0.73)
    });
    let head = connection_point(&nodes, frame.phase);
    let canvas = Canvas::default()
        .block(panel(&title))
        .marker(Marker::Braille)
        .background_color(Color::Black)
        .x_bounds([-aspect, aspect])
        .y_bounds([-1.0, 1.0])
        .paint(|ctx| {
            // One shared network is revisited, not one network per loop.
            // Two staggered rings and short cross-links keep it an oval mesh.
            for index in 0..NODES {
                let next = (index + 1) % NODES;
                for (a, b, intensity) in [
                    (nodes[index], nodes[next], 0.40),
                    (inner_nodes[index], inner_nodes[next], 0.26),
                    (nodes[index], inner_nodes[index], 0.23),
                ] {
                    ctx.draw(&Line::new(a.0, a.1, b.0, b.1, neuron_green(intensity)));
                }
                if index % 2 == 0 {
                    let (a, b) = (inner_nodes[index], nodes[next]);
                    ctx.draw(&Line::new(a.0, a.1, b.0, b.1, neuron_green(0.18)));
                }
            }
            ctx.draw(&Points {
                coords: &inner_nodes,
                color: neuron_green(0.58),
            });
            ctx.draw(&Points {
                coords: &nodes,
                color: neuron_green(0.78),
            });
            ctx.layer();
            // Follow the actual polygon connections, including the wrap seam.
            // A short fading tail makes the direction legible at ~30 fps.
            for step in (0..12).rev() {
                let phase = frame.phase - step as f64 / (NODES as f64 * 5.0);
                let a = connection_point(&nodes, phase - 1.0 / (NODES as f64 * 5.0));
                let b = connection_point(&nodes, phase);
                let intensity = 1.0 - step as f64 / 15.0;
                ctx.draw(&Line::new(a.0, a.1, b.0, b.1, neuron_green(intensity)));
            }
            ctx.layer();
            // Pixel-sized forks flicker about the moving junction; the radius
            // is bounded independently of the terminal's aspect ratio.
            let tick = (elapsed.as_millis() / 68) % 8;
            let dx = (2.0 * aspect / pixel_width) * 2.2;
            let dy = (2.0 / pixel_height) * 2.2;
            for branch in 0..3 {
                let angle = tick as f64 * 0.83 + branch as f64 * TAU / 3.0;
                let end = (head.0 + dx * angle.cos(), head.1 + dy * angle.sin());
                ctx.draw(&Line::new(head.0, head.1, end.0, end.1, ELECTRIC));
            }
            ctx.draw(&Points {
                coords: &[head],
                color: ELECTRIC,
            });
        });
    f.render_widget(canvas, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer};

    fn render(loops: usize, elapsed: Duration, width: u16, height: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|f| draw(f, f.area(), loops, elapsed))
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn text(buffer: &Buffer) -> String {
        buffer.content().iter().map(|cell| cell.symbol()).collect()
    }

    fn braille_count(buffer: &Buffer) -> usize {
        buffer
            .content()
            .iter()
            .filter(|cell| {
                cell.symbol()
                    .chars()
                    .next()
                    .is_some_and(|ch| ('\u{2801}'..='\u{28ff}').contains(&ch))
            })
            .count()
    }

    #[test]
    fn test_backend_selected_count_and_pass_cover_every_loop_setting() {
        for loops in 1..=MAX_LOOPS {
            let elapsed = LAP * (loops - 1) as u32;
            let buffer = render(loops, elapsed, 64, 16);
            assert!(text(&buffer).contains(&format!("loops {loops} / configuration diagram")));
            assert!(braille_count(&buffer) > 0);
            let restarted = render(loops, elapsed + LAP, 64, 16);
            assert!(text(&restarted).contains(&format!("loops {loops} / configuration diagram")));
        }
    }

    #[test]
    fn one_lap_per_pass_follows_connected_oval_and_has_no_seam_jump() {
        let nodes = oval(2.4);
        for loops in 2..=MAX_LOOPS {
            for pass in 0..loops {
                for quarter in 0..4 {
                    let elapsed = LAP * pass as u32 + LAP * quarter / 4;
                    let frame = frame_at(loops, elapsed);
                    assert_eq!(frame.pass, pass + 1);
                    assert_eq!(frame.phase, f64::from(quarter) / 4.0);
                    let point = connection_point(&nodes, frame.phase);
                    assert_eq!(point, nodes[quarter as usize * NODES / 4]);
                }
            }
        }
        let before = connection_point(&nodes, frame_at(32, LAP - Duration::from_nanos(1)).phase);
        let after = connection_point(&nodes, frame_at(32, LAP).phase);
        assert!((before.0 - after.0).hypot(before.1 - after.1) < 1e-7);
        for aspect in [0.1, 0.5, 1.0, 2.4, 12.0] {
            let nodes = oval(aspect);
            assert!(nodes[0].0 / nodes[NODES / 4].1 >= 1.8 - 1e-12);
            assert!(
                nodes
                    .iter()
                    .all(|&(x, y)| x.abs() < aspect && y.abs() < 1.0)
            );
            for index in 0..NODES {
                let a = nodes[index];
                let b = nodes[(index + 1) % NODES];
                let halfway = connection_point(&nodes, (index as f64 + 0.5) / NODES as f64);
                assert!((halfway.0 - (a.0 + b.0) / 2.0).abs() < 1e-12);
                assert!((halfway.1 - (a.1 + b.1) / 2.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_backend_electricity_moves_over_a_braille_network() {
        let first = render(4, Duration::ZERO, 64, 16);
        let quarter = render(4, LAP / 4, 64, 16);
        let half = render(4, LAP / 2, 64, 16);
        assert_ne!(first, quarter);
        assert_ne!(quarter, half);
        for buffer in [&first, &quarter, &half] {
            assert!(
                braille_count(buffer) > 40,
                "a network, not just a moving dot"
            );
            assert!(buffer.content().iter().any(|cell| cell.fg == ELECTRIC));
            assert!(
                buffer.content().iter().any(|cell| {
                    matches!(cell.fg, Color::Rgb(r, g, b) if r > 0 && r < 0x39 && g > b && b > r)
                }),
                "dim green connections behind the bright pulse"
            );
            assert_eq!(buffer[(0, 0)].symbol(), "┌");
            assert_eq!(buffer[(63, 15)].symbol(), "┘");
        }
    }

    #[test]
    fn test_backend_single_loop_reuses_the_original_neuron_and_counts_are_clamped() {
        let mut original = Terminal::new(TestBackend::new(52, 12)).unwrap();
        for elapsed in [
            Duration::ZERO,
            Duration::from_millis(2_500),
            Duration::from_millis(6_800),
        ] {
            original
                .draw(|f| {
                    draw_neuron_frame(
                        f,
                        f.area(),
                        neuron_frame_at(elapsed),
                        " loops 1 / configuration diagram ",
                    );
                })
                .unwrap();
            assert_eq!(&render(1, elapsed, 52, 12), original.backend().buffer());
            assert_eq!(render(0, elapsed, 52, 12), render(1, elapsed, 52, 12));
            assert_eq!(
                render(usize::MAX, elapsed, 52, 12),
                render(32, elapsed, 52, 12)
            );
        }
    }

    #[test]
    fn test_backend_small_and_offset_areas_do_not_escape_the_panel_shadow_included() {
        for (width, height) in [(0, 0), (1, 1), (2, 2), (3, 8), (9, 3), (12, 5), (8, 30)] {
            for loops in [1, 2, 32] {
                for elapsed in [Duration::ZERO, LAP / 3, Duration::MAX] {
                    let buffer = render(loops, elapsed, width, height);
                    assert_eq!(
                        buffer.content().len(),
                        usize::from(width) * usize::from(height)
                    );
                }
            }
        }
        for area in [
            Rect::new(7, 4, 52, 14),
            Rect::new(72, 20, 30, 20),
            Rect::new(90, 30, 5, 5),
        ] {
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
            terminal
                .draw(|f| {
                    for cell in &mut f.buffer_mut().content {
                        cell.set_symbol(".");
                    }
                    draw(f, area, 32, LAP / 3);
                })
                .unwrap();
            let buffer = terminal.backend().buffer();
            let visible = area.intersection(buffer.area);
            for y in 0..24 {
                for x in 0..80 {
                    let shadow = (x == area.right()
                        && y >= area.y
                        && y < area.bottom())
                        || (y == area.bottom() && x >= area.x && x <= area.right());
                    if (x < visible.x
                        || x >= visible.right()
                        || y < visible.y
                        || y >= visible.bottom())
                        && !shadow
                    {
                        assert_eq!(buffer[(x, y)].symbol(), ".", "outside panel at {x},{y}");
                    }
                }
            }
        }
    }
}
