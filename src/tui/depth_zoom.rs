//! A zoom into the Phase 5 neuron, opening into perspective network layers.
//! No independent clock/thread: setup and idle monitor use the caller's clock.
use super::{NeuronFrame, draw_neuron_frame, neuron_green, panel, panel_area, smoothstep};
use ratatui::{
    Frame,
    layout::Rect,
    style::Color,
    symbols::Marker,
    widgets::{
        Clear,
        canvas::{Canvas, Points},
    },
};
use std::time::{Duration, Instant};

const MAX_DEPTH: usize = 32;
const SPRING: f64 = 9.0;
const INTRO: Duration = Duration::from_millis(1_200);
const SHADES: usize = 8;
// A single bounded raster of the real renderer supplies every network layer.
// This avoids generating 32 separate 80-node topologies on every UI frame.
const SOURCE_WIDTH: u16 = 66;
const SOURCE_HEIGHT: u16 = 20;
const BRAILLE_DOTS: [(u16, u16); 8] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (0, 3),
    (1, 3),
];
type Point = (f64, f64);
type Raster = [Vec<Point>; SHADES];

pub(super) struct DepthZoom {
    depth: usize,
    from_depth: f64,
    from_velocity: f64,
    changed_at: Instant,
    started_at: Instant,
}

impl Default for DepthZoom {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            depth: 1,
            from_depth: 1.0,
            from_velocity: 0.0,
            changed_at: now,
            started_at: now,
        }
    }
}

impl DepthZoom {
    /// Retarget from the *currently displayed* depth and velocity, rather than
    /// the previous selection. Repeated selections do not restart the motion.
    pub(super) fn set_depth(&mut self, depth: usize, now: Instant) {
        let depth = depth.clamp(1, MAX_DEPTH);
        if depth == self.depth {
            return;
        }
        let (position, velocity) = self.motion_at(now);
        self.from_depth = position;
        self.from_velocity = velocity;
        self.depth = depth;
        self.changed_at = now;
    }

    fn motion_at(&self, now: Instant) -> (f64, f64) {
        let elapsed = now.saturating_duration_since(self.changed_at).as_secs_f64();
        // An exact settled endpoint avoids a permanent almost-visible layer.
        if elapsed >= 3.0 {
            return (self.depth as f64, 0.0);
        }
        // Analytic critically damped spring: no per-frame integration, and
        // both position and velocity remain continuous during a reversal.
        let offset = self.from_depth - self.depth as f64;
        let tangent = self.from_velocity + SPRING * offset;
        let decay = (-SPRING * elapsed).exp();
        let position = self.depth as f64 + (offset + tangent * elapsed) * decay;
        let velocity = (self.from_velocity - SPRING * tangent * elapsed) * decay;
        let bounded = position.clamp(1.0, MAX_DEPTH as f64);
        (bounded, if bounded == position { velocity } else { 0.0 })
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect, now: Instant) {
        let area = area.intersection(f.area());
        if area.is_empty() {
            return;
        }
        let (depth, velocity) = self.motion_at(now);
        let intro = smoothstep(
            now.saturating_duration_since(self.started_at).as_secs_f64() / INTRO.as_secs_f64(),
        );
        let direction = if velocity > 0.02 || intro < 1.0 {
            "zoom IN"
        } else if velocity < -0.02 {
            "zoom OUT"
        } else {
            "stack"
        };
        let title = format!(
            " depth {} / {} layers / {direction} / diagram ",
            self.depth, self.depth
        );
        let inner = panel(&title).inner(area);
        if inner.width < 2 || inner.height < 2 {
            let panel_rect = panel_area(f, area);
            f.render_widget(panel(&title), panel_rect);
            return;
        }

        // Reuse the illustrative configuration renderer; this is not measured
        // model connectivity. Its temporary image stays inside this panel and is fully
        // replaced by the final canvas before the terminal frame is flushed.
        let source = Rect::new(
            area.x,
            area.y,
            area.width.min(SOURCE_WIDTH),
            area.height.min(SOURCE_HEIGHT),
        );
        f.render_widget(Clear, source);
        draw_neuron_frame(
            f,
            source,
            NeuronFrame {
                cycle: 0,
                phase: 0.0,
                growth: intro,
                scale: 1.0,
                network_alpha: 1.0,
                collapsed: false,
            },
            "",
        );
        let raster = neuron_raster(f, panel("").inner(source));
        let layers = layers_at(depth, intro);
        let canvas = Canvas::default()
            .block(panel(&title))
            .marker(Marker::Braille)
            .background_color(Color::Black)
            .x_bounds([-1.0, 1.0])
            .y_bounds([-1.0, 1.0])
            .paint(|ctx| {
                let mut projected =
                    Vec::with_capacity(raster.iter().map(Vec::len).max().unwrap_or(0));
                // Back-to-front compositing, on one braille grid. Saving a
                // whole canvas per layer would multiply terminal-sized work.
                for layer in layers.iter().rev() {
                    for (shade, pixels) in raster.iter().enumerate() {
                        projected.clear();
                        projected.extend(pixels.iter().map(|&point| layer.project(point)));
                        ctx.draw(&Points {
                            coords: &projected,
                            color: neuron_green((shade + 1) as f64 / SHADES as f64 * layer.alpha),
                        });
                    }
                }
            });
        // Canvas only paints occupied braille cells; explicitly erase the
        // temporary source (including its border) before compositing the stack.
        f.render_widget(Clear, area);
        let panel_rect = panel_area(f, area);
        f.render_widget(canvas, panel_rect);
    }
}

/// Read the real neuron's braille dots, preserving eight green intensity bands.
/// Bounds/resolution are capped before calling the original renderer.
fn neuron_raster(f: &mut Frame, inner: Rect) -> Raster {
    let mut raster: Raster = std::array::from_fn(|_| Vec::new());
    let width = f64::from(inner.width) * 2.0 - 1.0;
    let height = f64::from(inner.height) * 4.0 - 1.0;
    let buffer = f.buffer_mut();
    for y in inner.y..inner.bottom() {
        for x in inner.x..inner.right() {
            let cell = &buffer[(x, y)];
            let Some(ch @ '\u{2800}'..='\u{28ff}') = cell.symbol().chars().next() else {
                continue;
            };
            let Color::Rgb(_, green, _) = cell.fg else {
                continue;
            };
            if green == 0 {
                continue;
            }
            let shade = ((usize::from(green) * SHADES).div_ceil(0xe0)).clamp(1, SHADES) - 1;
            let bits = ch as u32 - 0x2800;
            for (bit, (dx, dy)) in BRAILLE_DOTS.into_iter().enumerate() {
                if bits & (1 << bit) != 0 {
                    raster[shade].push((
                        f64::from((x - inner.x) * 2 + dx) * 2.0 / width - 1.0,
                        1.0 - f64::from((y - inner.y) * 4 + dy) * 2.0 / height,
                    ));
                }
            }
        }
    }
    raster
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Layer {
    scale_x: f64,
    scale_y: f64,
    shear: f64,
    x: f64,
    y: f64,
    alpha: f64,
}

impl Layer {
    fn project(self, (x, y): Point) -> Point {
        (
            self.x + x * self.scale_x + y * self.shear,
            self.y + y * self.scale_y,
        )
    }
}

fn layers_at(depth: f64, intro: f64) -> Vec<Layer> {
    let depth = depth.clamp(1.0, MAX_DEPTH as f64);
    let spread = 1.0 - 1.0 / depth;
    let zoom = 0.18 + 0.82 * intro.clamp(0.0, 1.0);
    (0..depth.ceil() as usize)
        .map(|index| {
            let back = index as f64 / (depth - 1.0).max(1.0) * spread;
            let perspective = 1.0 - 0.27 * back;
            Layer {
                // Magnify the foreground as depth increases. Vertical
                // foreshortening tilts the net into a stack, not a zoom out.
                scale_x: (0.82 + 0.10 * spread) * perspective * zoom,
                scale_y: (0.82 - 0.53 * spread) * perspective * zoom,
                shear: 0.07 * spread * perspective * zoom,
                x: -0.10 * back * zoom,
                y: (0.55 * spread - 1.12 * back) * zoom,
                // A new layer starts transparent at the previous integer
                // depth; reversing selection fades it out along the same path.
                alpha: smoothstep(depth - index as f64) * (1.0 - 0.55 * back),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer};

    fn render(zoom: &DepthZoom, now: Instant, width: u16, height: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| zoom.draw(f, f.area(), now)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn text(buffer: &Buffer) -> String {
        buffer.content().iter().map(|cell| cell.symbol()).collect()
    }

    fn braille_pixels(buffer: &Buffer) -> u32 {
        buffer
            .content()
            .iter()
            .filter_map(|cell| cell.symbol().chars().next())
            .filter(|ch| ('\u{2800}'..='\u{28ff}').contains(ch))
            .map(|ch| (ch as u32 - 0x2800).count_ones())
            .sum()
    }

    #[test]
    fn test_backend_zoom_grows_from_a_neuron_into_selected_stacked_networks() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at;
        let seed = render(&zoom, start, 64, 20);
        let growing = render(&zoom, start + INTRO / 2, 64, 20);
        let network = render(&zoom, start + INTRO, 64, 20);
        assert!(text(&seed).contains("depth 1 / 1 layers / zoom IN"));
        assert!(braille_pixels(&seed) > 0);
        assert!(braille_pixels(&growing) > braille_pixels(&seed));
        assert!(braille_pixels(&network) > braille_pixels(&growing));
        assert_ne!(seed, growing);
        assert_ne!(growing, network);
        zoom.set_depth(8, start + INTRO);
        let moving = render(&zoom, start + INTRO + Duration::from_millis(240), 64, 20);
        let stack = render(&zoom, start + INTRO + Duration::from_secs(3), 64, 20);
        assert!(text(&moving).contains("depth 8 / 8 layers / zoom IN"));
        assert!(text(&stack).contains("depth 8 / 8 layers / stack"));
        assert_ne!(moving, stack);
        assert_ne!(network, stack);
        assert!(braille_pixels(&stack) > 100);
        assert!(
            stack
                .content()
                .iter()
                .any(|cell| cell.fg == neuron_green(1.0))
        );
    }

    #[test]
    fn test_backend_every_selected_depth_has_exactly_one_network_layer_per_unit() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at + INTRO;
        let mut previous_scale = 0.0;
        for depth in 1..=MAX_DEPTH {
            let now = start + Duration::from_secs(depth as u64 * 4);
            zoom.set_depth(depth, now);
            let settled = now + Duration::from_secs(3);
            let (position, velocity) = zoom.motion_at(settled);
            assert_eq!(position, depth as f64);
            assert_eq!(velocity, 0.0);
            let layers = layers_at(position, 1.0);
            assert_eq!(layers.len(), depth);
            assert!(layers.iter().all(|layer| layer.alpha > 0.0));
            assert!(layers.windows(2).all(|pair| pair[0].y > pair[1].y));
            assert!(
                layers[0].scale_x >= previous_scale,
                "the foreground zooms IN"
            );
            previous_scale = layers[0].scale_x;
            for layer in layers {
                for point in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                    let (x, y) = layer.project(point);
                    assert!(x.is_finite() && y.is_finite());
                    assert!(
                        x.abs() < 1.0 && y.abs() < 1.0,
                        "the whole stack keeps a margin"
                    );
                }
            }
            let buffer = render(&zoom, settled, 64, 20);
            assert!(text(&buffer).contains(&format!("depth {depth} / {depth} layers")));
            assert!(braille_pixels(&buffer) > 100);
            assert_eq!(buffer[(0, 0)].symbol(), "┌");
            assert_eq!(buffer[(63, 19)].symbol(), "┘");
        }
        let previous = layers_at(12.0, 1.0);
        let birth = layers_at(12.0 + 1e-6, 1.0);
        assert_eq!(birth.len(), 13);
        assert!(
            birth[12].alpha < 1e-10,
            "new layers fade in rather than pop"
        );
        assert!((birth[0].scale_x - previous[0].scale_x).abs() < 1e-7);
    }

    #[test]
    fn test_backend_retargeting_in_flight_preserves_position_velocity_and_picture() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at + INTRO;
        zoom.set_depth(32, start);
        let turn = start + Duration::from_millis(190);
        let before_motion = zoom.motion_at(turn);
        let before = render(&zoom, turn, 64, 20);
        zoom.set_depth(2, turn);
        let after_motion = zoom.motion_at(turn);
        let after = render(&zoom, turn, 64, 20);
        assert!((before_motion.0 - after_motion.0).abs() < 1e-12);
        assert!((before_motion.1 - after_motion.1).abs() < 1e-12);
        assert!(
            text(&after).contains("depth 2 / 2 layers"),
            "selection updates immediately"
        );
        for y in 1..19 {
            for x in 1..63 {
                assert_eq!(before[(x, y)], after[(x, y)], "no camera snap on retarget");
            }
        }
        let reversing = render(&zoom, turn + Duration::from_millis(400), 64, 20);
        assert!(text(&reversing).contains("zoom OUT"));
        assert_ne!(after, reversing);
        let settled = zoom.motion_at(turn + Duration::from_secs(3));
        assert_eq!(settled, (2.0, 0.0));

        let repeated = turn + Duration::from_millis(100);
        let later = turn + Duration::from_millis(650);
        let predicted = zoom.motion_at(later);
        zoom.set_depth(2, repeated);
        assert_eq!(
            zoom.changed_at, turn,
            "same selection cannot restart easing"
        );
        assert_eq!(zoom.motion_at(later), predicted);
    }

    #[test]
    fn rapid_reversals_and_invalid_depths_remain_bounded_and_settle() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at;
        for (index, target) in [32, 1, 31, 2, 30, 1, usize::MAX, 0].into_iter().enumerate() {
            let now = start + Duration::from_millis(index as u64 * 95);
            let before = zoom.motion_at(now);
            zoom.set_depth(target, now);
            let after = zoom.motion_at(now);
            assert!((before.0 - after.0).abs() < 1e-12);
            assert!((before.1 - after.1).abs() < 1e-12);
            for tick in 0..=10 {
                let (position, velocity) = zoom.motion_at(now + Duration::from_millis(tick * 9));
                assert!((1.0..=32.0).contains(&position));
                assert!(velocity.is_finite());
                assert!((1..=32).contains(&layers_at(position, 1.0).len()));
            }
        }
        assert_eq!(zoom.depth, 1);
        assert_eq!(zoom.motion_at(start + Duration::from_secs(4)), (1.0, 0.0));
        zoom.set_depth(usize::MAX, start + Duration::from_secs(4));
        assert_eq!(zoom.depth, 32);
        assert_eq!(zoom.motion_at(start + Duration::from_secs(7)), (32.0, 0.0));
    }

    #[test]
    fn test_backend_projection_source_is_the_actual_phase_five_renderer() {
        let mut terminal = Terminal::new(TestBackend::new(SOURCE_WIDTH, SOURCE_HEIGHT)).unwrap();
        for growth in [0.0, 1.0] {
            terminal
                .draw(|f| {
                    let area = f.area();
                    draw_neuron_frame(
                        f,
                        area,
                        NeuronFrame {
                            cycle: 0,
                            phase: 0.0,
                            growth,
                            scale: 1.0,
                            network_alpha: 1.0,
                            collapsed: false,
                        },
                        "",
                    );
                    let raster = neuron_raster(f, panel("").inner(area));
                    let count: usize = raster.iter().map(Vec::len).sum();
                    assert!(
                        raster
                            .iter()
                            .flatten()
                            .all(|&(x, y)| x.abs() <= 1.0 && y.abs() <= 1.0)
                    );
                    if growth == 0.0 {
                        assert_eq!(count, 5, "preserve the Phase 5 filled seed pixels");
                        assert_eq!(raster[SHADES - 1].len(), 5);
                    } else {
                        assert!(count > 100, "sample actual branches as well as the seed");
                        assert!(raster[..SHADES - 1].iter().any(|shade| !shade.is_empty()));
                        assert!(
                            count
                                <= usize::from(SOURCE_WIDTH - 2)
                                    * usize::from(SOURCE_HEIGHT - 2)
                                    * 8
                        );
                    }
                })
                .unwrap();
        }
    }

    #[test]
    fn test_backend_large_panel_does_not_leak_the_temporary_source_image() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at;
        zoom.set_depth(32, start);
        for now in [start, start + INTRO / 2, start + Duration::from_secs(3)] {
            let buffer = render(&zoom, now, 96, 32);
            for y in 1..31 {
                for x in 1..95 {
                    let ch = buffer[(x, y)].symbol().chars().next().unwrap();
                    assert!(
                        ch == ' ' || ('\u{2800}'..='\u{28ff}').contains(&ch),
                        "temporary renderer border leaked at {x},{y}: {ch}"
                    );
                }
            }
            if now == start {
                assert!(
                    braille_pixels(&buffer) <= 5,
                    "only one initial neuron, not a second source image"
                );
            }
        }
    }

    #[test]
    fn test_backend_small_clipped_and_offset_areas_stay_inside_the_panel_shadow_included() {
        let mut zoom = DepthZoom::default();
        let start = zoom.started_at;
        zoom.set_depth(32, start);
        for (width, height) in [(0, 0), (1, 1), (2, 2), (3, 8), (9, 3), (12, 5), (8, 30)] {
            for now in [start, start + INTRO / 2, start + Duration::from_secs(3)] {
                let buffer = render(&zoom, now, width, height);
                assert_eq!(
                    buffer.content().len(),
                    usize::from(width) * usize::from(height)
                );
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
                    zoom.draw(f, area, start + Duration::from_secs(3));
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
