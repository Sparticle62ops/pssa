//! Shared CRT plots: connected braille strokes, honest gaps, and readable scales.
use super::{NORMAL_GREEN, PANEL_BG, SECOND_ACCENT, accent, panel, panel_area};
use ratatui::{
    Frame,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Paragraph, Widget,
        canvas::{Canvas, Painter, Points, Shape},
    },
};

pub(super) struct Series<'a> {
    pub name: &'a str,
    pub points: &'a [(f64, f64)],
    pub color: Color,
    pub scatter: bool,
}

impl<'a> Series<'a> {
    pub fn line(name: &'a str, points: &'a [(f64, f64)], color: Color) -> Self {
        Self {
            name,
            points,
            color,
            scatter: false,
        }
    }
}

pub(super) struct Plot<'a> {
    pub title: &'a str,
    pub caption: &'a str,
    pub x: &'a str,
    pub y: &'a str,
    pub x_bounds: [f64; 2],
    pub y_bounds: [f64; 2],
    pub integer_x: bool,
}

/// Short labels, not false precision. Small learning rates retain their scale.
pub(super) fn number(value: f64) -> String {
    if !value.is_finite() {
        return "—".into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(0.01..1_000_000.0).contains(&magnitude) {
        return format!("{value:.1e}").replace(".0e", "e");
    }
    let digits = if magnitude >= 100.0 {
        0
    } else if magnitude >= 1.0 {
        1
    } else {
        2
    };
    let text = format!("{value:.digits$}");
    let text = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        &text
    };
    if text == "-0" {
        "0".into()
    } else {
        text.into()
    }
}

/// Fit the visible observations, with 5% breathing room on each side.
/// Tick rounding is separate: it must not turn a small change into a flat trace.
pub(super) fn bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.filter(|v| v.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    if !low.is_finite() {
        return [0.0, 1.0];
    }
    let pad = if low == high {
        (low.abs() * 0.05).max(1e-9)
    } else {
        ((high - low) * 0.05).max(f64::EPSILON * low.abs().max(high.abs()))
    };
    [low - pad, high + pad]
}

pub(super) fn domain(points: &[(f64, f64)]) -> [f64; 2] {
    let (low, high) = points
        .iter()
        .map(|p| p.0)
        .filter(|x| x.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), x| {
            (low.min(x), high.max(x))
        });
    if !low.is_finite() {
        [0.0, 2.0]
    } else if low == high {
        [low - 1.0, high + 1.0]
    } else {
        [low, high]
    }
}

fn ticks(bounds: [f64; 2]) -> Vec<String> {
    let step = (bounds[1] - bounds[0]) / 2.0;
    let magnitude = bounds[0].abs().max(bounds[1].abs());
    // Choose precision once for the entire scale, including zero/endpoints.
    let scientific = step < 0.01 || (magnitude >= 1_000_000.0 && step >= magnitude * 0.01);
    let digits = if step < 1.0 {
        (1.0 - step.log10().floor()).clamp(1.0, 8.0) as usize
    } else {
        usize::from(magnitude < 100.0)
    };
    (0..3)
        .map(|i| {
            let value = bounds[0] + step * i as f64;
            if scientific {
                format!("{value:.2e}")
            } else {
                // Keep nearby large step counts distinct rather than using 1e6.
                format!("{value:.digits$}")
            }
        })
        .collect()
}

/// Project values onto the available row intervals and round consistently.
/// Labels use whole rows; the curve retains Canvas's finer braille projection.
fn tick_row(value: f64, bounds: [f64; 2], height: u16) -> u16 {
    let fraction = (bounds[1] - value) / (bounds[1] - bounds[0]);
    (fraction * f64::from(height - 1)).round() as u16
}

/// Prefer three to five nice-valued ticks with evenly spaced rows. Short plots
/// may need fewer ticks, but never a mixture of adjacent and separated labels.
/// Tick selection must not expand the fitted bounds or introduce unequal
/// value steps.
fn y_ticks(bounds: [f64; 2], height: u16) -> Vec<(f64, String)> {
    let range = bounds[1] - bounds[0];
    let power = 10.0_f64.powf((range / 4.0).log10().floor());
    let mut best = Vec::new();
    let mut best_step = power;
    let mut best_score = (false, false, false, 0, 0);
    // Start with familiar 1/2/2.5/5 steps. The remaining steps fill awkward
    // narrow ranges without expanding the bounds.
    for multiplier in [1.0, 2.0, 2.5, 5.0, 10.0, 1.5, 3.0, 4.0, 6.0, 8.0] {
        let step = power * multiplier;
        let first = (bounds[0] / step).ceil();
        let last = (bounds[1] / step).floor();
        let count = (last - first + 1.0).max(0.0) as usize;
        if count < 3 {
            continue;
        }
        // Thin even dense candidates by a uniform stride, never by individual
        // row collisions: the latter silently introduces unequal value steps.
        // Try each starting offset so rounding at an edge cannot exclude an
        // evenly spaced sequence of nice values elsewhere inside the bounds.
        for stride in count.div_ceil(5)..count {
            for offset in 0..stride {
                let len = (count - 1 - offset) / stride + 1;
                if !(2..=5).contains(&len) {
                    continue;
                }
                let values: Vec<_> = (offset..count)
                    .step_by(stride)
                    .map(|i| (first + i as f64) * step)
                    .collect();
                let (min_gap, max_gap) = values.windows(2).fold((u16::MAX, 0), |(min, max), p| {
                    let gap = tick_row(p[0], bounds, height) - tick_row(p[1], bounds, height);
                    (min.min(gap), max.max(gap))
                });
                if min_gap == 0 || max_gap - min_gap > 1 || (min_gap == 1 && max_gap > 1) {
                    continue;
                }
                let span = tick_row(values[0], bounds, height)
                    - tick_row(*values.last().unwrap(), bounds, height);
                // Prefer exact gaps and familiar unthinned scales over more
                // labels. Allow one-row rounding differences only when no
                // scale with >=3 evenly spaced ticks fits.
                let score = (
                    values.len() >= 3,
                    min_gap == max_gap,
                    stride == 1,
                    values.len(),
                    span,
                );
                if score > best_score {
                    best_score = score;
                    best = values;
                    best_step = step;
                }
            }
        }
    }
    let magnitude = bounds[0].abs().max(bounds[1].abs());
    let scientific =
        magnitude < 0.01 || (magnitude >= 1_000_000.0 && best_step >= magnitude * 0.01);
    let scientific_digits = (magnitude / best_step).log10().ceil().clamp(2.0, 15.0) as usize;
    let digits = (0..=15)
        .find(|&d| {
            let scaled = best_step * 10.0_f64.powi(d);
            scaled >= 1.0 && (scaled - scaled.round()).abs() < 1e-6
        })
        .unwrap_or(15) as usize;
    best.into_iter()
        .map(|value| {
            let value = if value == 0.0 { 0.0 } else { value };
            let label = if scientific {
                format!("{value:.scientific_digits$e}")
            } else {
                format!("{value:.digits$}")
            };
            (value, label)
        })
        .collect()
}

pub(super) fn draw(f: &mut Frame, area: Rect, plot: Plot<'_>, series: &[Series<'_>]) {
    draw_labeled(f, area, plot, series, None);
}

/// Return the drawable plot rectangle after the panel border, y-axis gutter,
/// and two rows reserved for x labels. Keeping this calculation shared makes
/// overlay layers land on exactly the same cells as the primary plot.
pub(super) fn graph_rect(area: Rect, plot: &Plot<'_>) -> Option<Rect> {
    let inner = panel("").inner(area);
    let label_width = y_ticks(plot.y_bounds, inner.height.saturating_sub(2))
        .iter()
        .map(|(_, s)| s.len())
        .max()
        .unwrap_or(0) as u16;
    graph_rect_for_label_width(area, label_width)
}

fn graph_rect_for_label_width(area: Rect, label_width: u16) -> Option<Rect> {
    let inner = panel("").inner(area);
    if inner.width < 16 || inner.height < 5 {
        return None;
    }
    let gutter = label_width.min(inner.width / 3) + 1;
    Some(Rect::new(
        inner.x + gutter,
        inner.y,
        inner.width.saturating_sub(gutter),
        inner.height - 2,
    ))
}

/// Paint a continuous HalfBlock layer over a normal plot without clearing its
/// existing glyphs. This is used for geometry that must read as a solid wall
/// while the measurements below it remain small, separate braille dots.
pub(super) fn draw_solid_overlay(
    f: &mut Frame,
    area: Rect,
    plot: Plot<'_>,
    points: &[(f64, f64)],
    color: Color,
) {
    let area = panel_area(f, area);
    let Some(graph) = graph_rect(area, &plot) else {
        return;
    };
    f.render_widget(
        Canvas::default()
            .background_color(PANEL_BG)
            .marker(Marker::HalfBlock)
            .x_bounds(plot.x_bounds)
            .y_bounds(plot.y_bounds)
            .paint(|ctx| {
                ctx.draw(&Points { coords: points, color });
                for pair in points.windows(2) {
                    ctx.draw(&ThinLine {
                        from: pair[0],
                        to: pair[1],
                        color,
                    });
                }
                ctx.layer();
            }),
        graph,
    );
}

/// Optional category labels for comparisons (never connect unrelated models).
pub(super) fn draw_labeled(
    f: &mut Frame,
    area: Rect,
    plot: Plot<'_>,
    series: &[Series<'_>],
    y_labels: Option<Vec<String>>,
) {
    let area = panel_area(f, area);
    // Ratatui's Chart axis titles are drawn over the first/last data rows.
    // Keep units and any extra legend entries on the border instead.
    let heading = if plot.title.contains(plot.y) {
        plot.title.to_owned()
    } else {
        format!("{} / {} ", plot.title.trim_end(), plot.y)
    };
    let mut title = Line::styled(heading, accent().add_modifier(Modifier::BOLD));
    for s in series {
        let legend = format!(" │ {} ", s.name);
        if !title.to_string().contains(s.name)
            && title.width() + legend.len() <= area.width.saturating_sub(2) as usize
        {
            title
                .spans
                .push(Span::styled(legend, Style::new().fg(s.color)));
        }
    }
    let block = panel("")
        .title(title)
        .title_bottom(Line::styled(plot.caption, Style::new().fg(SECOND_ACCENT)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 16 || inner.height < 5 {
        f.render_widget(
            Paragraph::new(format!("{} / {}\nEnlarge for plot", plot.y, plot.x)).style(accent()),
            inner,
        );
        return;
    }
    let y_labels = y_labels.map_or_else(
        || y_ticks(plot.y_bounds, inner.height - 2),
        |labels| {
            let intervals = labels.len().saturating_sub(1).max(1) as f64;
            labels
                .into_iter()
                .enumerate()
                .map(|(i, label)| {
                    (
                        plot.y_bounds[0]
                            + (plot.y_bounds[1] - plot.y_bounds[0]) * i as f64 / intervals,
                        label,
                    )
                })
                .collect()
        },
    );
    // Reserve separate rows for the axis stroke and its labels, never data.
    let label_width = y_labels.iter().map(|(_, s)| s.len()).max().unwrap_or(0) as u16;
    let Some(graph) = graph_rect_for_label_width(area, label_width) else {
        return;
    };
    let gutter = graph.x - inner.x;
    let axis_x = graph.x - 1;
    let axis_y = graph.bottom();
    let buffer = f.buffer_mut();
    for y in graph.y..graph.bottom() {
        buffer[(axis_x, y)].set_symbol("│").set_style(accent());
    }
    for x in graph.x..graph.right() {
        buffer[(x, axis_y)].set_symbol("─").set_style(accent());
    }
    buffer[(axis_x, axis_y)].set_symbol("└").set_style(accent());
    for (value, label) in &y_labels {
        Line::styled(label.as_str(), accent())
            .right_aligned()
            .render(
                Rect::new(
                    inner.x,
                    graph.y + tick_row(*value, plot.y_bounds, graph.height),
                    gutter - 1,
                    1,
                ),
                buffer,
            );
    }
    let x_labels: Vec<String> = if plot.integer_x {
        (0..3)
            .map(|i| {
                let value =
                    plot.x_bounds[0] + (plot.x_bounds[1] - plot.x_bounds[0]) * i as f64 / 2.0;
                format!("{:.0}", value.round())
            })
            .collect()
    } else {
        ticks(plot.x_bounds)
    };
    draw_x_labels(buffer, graph, plot.x, &x_labels);
    f.render_widget(
        Canvas::default()
            .background_color(PANEL_BG)
            .marker(Marker::Braille)
            .x_bounds(plot.x_bounds)
            .y_bounds(plot.y_bounds)
            .paint(|ctx| {
                // One shared braille layer: a separate layer per series would
                // replace whole cells and cut holes in earlier series wherever
                // two lines cross. Every stroke keeps all of its dots; a shared
                // cell takes the color of the series listed last.
                for s in series {
                    // Split at missing observations, never bridge a gap.
                    for run in s.points.split(|(x, y)| !x.is_finite() || !y.is_finite()) {
                        ctx.draw(&Points {
                            coords: run,
                            color: s.color,
                        });
                        if !s.scatter {
                            for pair in run.windows(2) {
                                ctx.draw(&ThinLine {
                                    from: pair[0],
                                    to: pair[1],
                                    color: s.color,
                                });
                            }
                        }
                    }
                    // Sparse observations should read as measurements joined
                    // by slopes, not a continuous stream (e.g. checkpoints).
                    if !s.scatter
                        && s.points
                            .iter()
                            .filter(|p| p.0.is_finite() && p.1.is_finite())
                            .count()
                            <= 8
                    {
                        for &point in s
                            .points
                            .iter()
                            .filter(|p| p.0.is_finite() && p.1.is_finite())
                        {
                            ctx.draw(&PointMarker {
                                point,
                                color: s.color,
                                x_bounds: plot.x_bounds,
                                y_bounds: plot.y_bounds,
                            });
                        }
                    }
                }
            }),
        graph,
    );
}

/// Put the axis name in a free gap on the tick-label row. At small widths,
/// prefer the name and endpoints to an overlapping middle tick.
fn draw_x_labels(buffer: &mut Buffer, graph: Rect, name: &str, labels: &[String]) {
    let widths: Vec<_> = labels.iter().map(|s| s.len() as u16).collect();
    let mut positions = vec![
        (graph.x, 0),
        (
            graph.x + ((graph.width - 1) / 2).saturating_sub(widths[1] / 2),
            1,
        ),
        (graph.right().saturating_sub(widths[2]), 2),
    ];
    let gap = |positions: &[(u16, usize)]| {
        positions
            .windows(2)
            .map(|pair| {
                let start = pair[0].0 + widths[pair[0].1] + 1;
                (start, pair[1].0.saturating_sub(start + 1))
            })
            .max_by_key(|(_, width)| *width)
            .unwrap_or((graph.x, 0))
    };
    if labels[1] == labels[0]
        || labels[1] == labels[2]
        || gap(&positions).1 < name.len() as u16
    {
        positions.remove(1);
    }
    let (start, width) = gap(&positions);
    let row = graph.bottom() + 1;
    if width >= name.len() as u16 {
        Line::styled(name, accent())
            .centered()
            .render(Rect::new(start, row, width, 1), buffer);
        for (x, i) in positions {
            buffer.set_stringn(x, row, &labels[i], widths[i] as usize, accent());
        }
    } else {
        Line::styled(name, accent())
            .centered()
            .render(Rect::new(graph.x, row, graph.width, 1), buffer);
    }
}

/// A connected, one-sub-pixel Bresenham stroke in braille dot coordinates.
/// Never hold the previous sample or thicken the stroke within a text cell.
struct ThinLine {
    from: (f64, f64),
    to: (f64, f64),
    color: Color,
}

impl Shape for ThinLine {
    fn draw(&self, painter: &mut Painter) {
        let Some((x1, y1)) = painter.get_point(self.from.0, self.from.1) else {
            return;
        };
        let Some((x2, y2)) = painter.get_point(self.to.0, self.to.1) else {
            return;
        };
        let (mut x, mut y) = (x1 as i32, y1 as i32);
        let (end_x, end_y) = (x2 as i32, y2 as i32);
        let dx = (end_x - x).abs();
        let dy = -(end_y - y).abs();
        let sx = if x < end_x { 1 } else { -1 };
        let sy = if y < end_y { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            painter.paint(x as usize, y as usize, self.color);
            if x == end_x && y == end_y {
                break;
            }
            let twice_error = 2 * error;
            if twice_error >= dy {
                error += dy;
                x += sx;
            }
            if twice_error <= dx {
                error += dx;
                y += sy;
            }
        }
    }
}

/// A small cross only at sparse observations, clipped even at canvas edges.
struct PointMarker {
    point: (f64, f64),
    color: Color,
    x_bounds: [f64; 2],
    y_bounds: [f64; 2],
}

impl Shape for PointMarker {
    fn draw(&self, painter: &mut Painter) {
        let Some((x, y)) = painter.get_point(self.point.0, self.point.1) else {
            return;
        };
        let Some((max_x, max_y)) = painter.get_point(self.x_bounds[1], self.y_bounds[0]) else {
            return;
        };
        for (dx, dy) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (x, y) = (x as i32 + dx, y as i32 + dy);
            if x >= 0 && y >= 0 && x <= max_x as i32 && y <= max_y as i32 {
                painter.paint(x as usize, y as usize, self.color);
            }
        }
    }
}

/// A one-cell-high Canvas still has four vertical and two horizontal dots/cell.
pub(super) fn sparkline(values: &[f64], width: usize) -> String {
    let width = width.min(u16::MAX as usize) as u16;
    if width == 0 {
        return String::new();
    }
    if !values.iter().any(|v| v.is_finite()) {
        return "·".repeat(width as usize);
    }
    let points = values
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64, *y))
        .collect::<Vec<_>>();
    // Unlike a full chart there are no numeric ticks to round outward. Use
    // all four dot rows so a small change does not collapse into a flat mark.
    let (low, high) = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let y_bounds = if low == high {
        let pad = (low.abs() * 0.1).max(1e-9);
        [low - pad, high + pad]
    } else {
        [low, high]
    };
    let x_bounds = [0.0, (values.len().saturating_sub(1) as f64).max(1.0)];
    let area = Rect::new(0, 0, width, 1);
    let mut buffer = Buffer::empty(area);
    Canvas::default()
        .marker(Marker::Braille)
        .x_bounds(x_bounds)
        .y_bounds(y_bounds)
        .paint(|ctx| {
            for run in points.split(|p| !p.1.is_finite()) {
                ctx.draw(&Points {
                    coords: run,
                    color: NORMAL_GREEN,
                });
                for pair in run.windows(2) {
                    ctx.draw(&ThinLine {
                        from: pair[0],
                        to: pair[1],
                        color: NORMAL_GREEN,
                    });
                }
            }
        })
        .render(area, &mut buffer);
    buffer
        .content()
        .iter()
        .map(|c| if c.symbol() == " " { "·" } else { c.symbol() })
        .collect()
}

/// Thin braille meter with eight sub-cell levels; no solid block wall.
pub(super) fn bar(fraction: f64, width: usize) -> String {
    let fraction = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dots = (fraction * width as f64 * 8.0).round() as usize;
    let glyphs = ['·', '⡀', '⡄', '⡆', '⡇', '⣇', '⣧', '⣷', '⣿'];
    (0..width)
        .map(|i| glyphs[dots.saturating_sub(i * 8).min(8)])
        .collect()
}

#[cfg(test)]
pub(super) fn assert_plot(buffer: &Buffer, area: Rect, labels: &[&str]) {
    let rows: Vec<String> = (area.y..area.bottom())
        .map(|y| {
            (area.x..area.right())
                .map(|x| buffer[(x, y)].symbol())
                .collect()
        })
        .collect();
    let text = rows.join("\n");
    for label in labels {
        assert!(text.contains(label), "missing {label}:\n{text}");
    }
    assert!(
        text.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)),
        "no braille in plot:\n{text}"
    );
    assert!(
        !text.chars().any(|c| ('\u{2580}'..='\u{259f}').contains(&c)),
        "block glyph in plot:\n{text}"
    );
}

#[cfg(test)]
pub(super) fn assert_plot_with_halfblocks(buffer: &Buffer, area: Rect, labels: &[&str]) {
    let rows: Vec<String> = (area.y..area.bottom())
        .map(|y| {
            (area.x..area.right())
                .map(|x| buffer[(x, y)].symbol())
                .collect()
        })
        .collect();
    let text = rows.join("\n");
    for label in labels {
        assert!(text.contains(label), "missing {label}:\n{text}");
    }
    assert!(
        text.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)),
        "no braille in plot:\n{text}"
    );
    assert!(
        text.chars().any(|c| ('\u{2580}'..='\u{259f}').contains(&c)),
        "no half-block boundary in plot:\n{text}"
    );
}

#[cfg(test)]
pub(super) fn assert_named_plot(buffer: &Buffer, title: &str, labels: &[&str]) {
    for y in buffer.area.y..buffer.area.bottom() {
        for x in buffer.area.x..buffer.area.right() {
            if buffer[(x, y)].symbol() != "┌" {
                continue;
            }
            let Some(right) =
                (x + 1..buffer.area.right()).find(|&xx| buffer[(xx, y)].symbol() == "┐")
            else {
                continue;
            };
            let heading: String = (x..=right).map(|xx| buffer[(xx, y)].symbol()).collect();
            if !heading.contains(title) {
                continue;
            }
            let bottom = (y + 1..buffer.area.bottom())
                .find(|&yy| buffer[(x, yy)].symbol() == "└")
                .expect("closed chart border");
            assert_plot(
                buffer,
                Rect::new(x, y, right - x + 1, bottom - y + 1),
                labels,
            );
            return;
        }
    }
    let text = (buffer.area.y..buffer.area.bottom())
        .map(|y| {
            (buffer.area.x..buffer.area.right())
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    panic!("missing plot {title}:\n{text}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn connected_lines_and_readable_axes_at_both_sizes() {
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            // Two endpoints must paint across the plot, not just two dots.
            terminal
                .draw(|f| {
                    draw(
                        f,
                        Rect::new(0, 0, w, 9),
                        Plot {
                            title: " trend ",
                            caption: "Lower is better",
                            integer_x: true,
                            x: "training step",
                            y: "loss",
                            x_bounds: [0.0, 100.0],
                            y_bounds: [0.0, 4.0],
                        },
                        &[Series::line(
                            "loss",
                            &[(0.0, 3.0), (100.0, 1.0)],
                            NORMAL_GREEN,
                        )],
                    )
                })
                .unwrap();
            let b = terminal.backend().buffer();
            assert_plot(
                b,
                Rect::new(0, 0, w, 9),
                &["loss", "training step", "Lower is better", "50", "100", "4"],
            );
            let cells = b
                .content()
                .iter()
                .filter(|c| {
                    c.symbol()
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                })
                .count();
            assert!(
                cells > w as usize / 2,
                "line should span most plot columns, got {cells}"
            );
        }
    }

    #[test]
    fn crossing_series_do_not_erase_each_others_dots() {
        let loss: Vec<(f64, f64)> = (0..=200)
            .map(|i| (i as f64, 5.0 + (i as f64 / 15.0).sin()))
            .collect();
        let avg: Vec<(f64, f64)> = (0..=200).map(|i| (i as f64, 5.0)).collect();
        let plot = || Plot {
            title: " graph ",
            caption: "Lower is better",
            integer_x: true,
            x: "training step",
            y: "loss",
            x_bounds: [0.0, 200.0],
            y_bounds: [3.8, 6.2],
        };
        let area = Rect::new(0, 0, 100, 16);
        let render = |series: &[Series<'_>]| {
            let mut terminal = Terminal::new(TestBackend::new(100, 16)).unwrap();
            terminal.draw(|f| draw(f, area, plot(), series)).unwrap();
            terminal.backend().buffer().clone()
        };
        let alone = render(&[Series::line("loss", &loss, NORMAL_GREEN)]);
        let both = render(&[
            Series::line("loss", &loss, NORMAL_GREEN),
            Series::line("moving avg", &avg, SECOND_ACCENT),
        ]);
        let graph = graph_rect(&alone, area);
        let missing: Vec<_> = dots(&alone, graph)
            .difference(&dots(&both, graph))
            .copied()
            .collect();
        assert!(missing.is_empty(), "loss dots erased where lines cross: {missing:?}");
        let avg_alone = render(&[
            Series::line("loss", &[], NORMAL_GREEN),
            Series::line("moving avg", &avg, SECOND_ACCENT),
        ]);
        assert!(dots(&avg_alone, graph).is_subset(&dots(&both, graph)));
    }

    fn graph_rect(buffer: &Buffer, area: Rect) -> Rect {
        let axis_y = area.bottom() - 3;
        let axis_x = (area.x + 1..area.right() - 1)
            .find(|&x| buffer[(x, axis_y)].symbol() == "└")
            .expect("plot axis");
        Rect::new(
            axis_x + 1,
            area.y + 1,
            area.right() - axis_x - 2,
            area.height - 4,
        )
    }

    fn dots(buffer: &Buffer, graph: Rect) -> std::collections::BTreeSet<(u16, u16)> {
        let mut dots = std::collections::BTreeSet::new();
        for x in 0..graph.width * 2 {
            for y in 0..graph.height * 4 {
                let ch = buffer[(graph.x + x / 2, graph.y + y / 4)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap();
                let mask = [[1, 2, 4, 64], [8, 16, 32, 128]][x as usize % 2][y as usize % 4];
                if ('\u{2800}'..='\u{28ff}').contains(&ch) && (ch as u32 - 0x2800) & mask != 0 {
                    dots.insert((x, y));
                }
            }
        }
        dots
    }

    #[test]
    fn fitted_ticks_match_data_rows_at_both_sizes() {
        for (w, h) in [(120, 40), (80, 24), (48, 18)] {
            for height in [7, 8, 9, 12] {
                for data in [
                    [3.9, 6.0],
                    [40.0, 403.0],
                    [6.3006, 6.3011],
                    [-0.7, -0.3],
                    [0.28, 0.72],
                    [0.0001, 0.0004],
                    [912.0, 1343.0],
                    [-351.263608, -350.91351],
                    [16.686176, 16.689383],
                    [3.0, 3.0],
                    [0.0, 0.0],
                ] {
                    let y_bounds = bounds(data.into_iter());
                    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
                    // Offset the panel too: labels must use plot-relative rows.
                    let area = Rect::new(2, 3, w - 4, height);
                    terminal
                        .draw(|f| {
                            draw(
                                f,
                                area,
                                Plot {
                                    title: " trend ",
                                    caption: "Lower is better",
                                    x: "training step",
                                    y: "loss",
                                    integer_x: true,
                                    x_bounds: [1.0, 150.0],
                                    y_bounds,
                                },
                                &[Series::line(
                                    "measured",
                                    &[(1.0, data[1]), (150.0, data[0])],
                                    NORMAL_GREEN,
                                )],
                            )
                        })
                        .unwrap();
                    let b = terminal.backend().buffer();
                    let graph = graph_rect(b, area);
                    let mut labels = Vec::new();
                    for y in graph.y..graph.bottom() {
                        let label = (area.x + 1..graph.x - 1)
                            .map(|x| b[(x, y)].symbol())
                            .collect::<String>();
                        if !label.trim().is_empty() {
                            let value = label.trim().parse::<f64>().expect("numeric y tick");
                            let fraction = (y_bounds[1] - value) / (y_bounds[1] - y_bounds[0]);
                            let expected_row =
                                (fraction * f64::from(graph.height - 1)).round() as u16;
                            assert_eq!(
                                y,
                                graph.y + expected_row,
                                "misplaced {label} in {y_bounds:?}"
                            );
                            let dot = (fraction * f64::from(graph.height * 4 - 1)) as u16;
                            assert!(expected_row.abs_diff(dot / 4) <= 1, "tick far from curve");
                            assert!(!label.ends_with(' '), "ticks must share a right edge");
                            labels.push((y, value));
                        }
                        for x in graph.x..graph.right() {
                            let symbol = b[(x, y)].symbol();
                            assert!(
                                symbol == " "
                                    || symbol
                                        .chars()
                                        .all(|c| ('\u{2800}'..='\u{28ff}').contains(&c)),
                                "text over data: {symbol}"
                            );
                        }
                    }
                    assert!(
                        (2..=5).contains(&labels.len()),
                        "{y_bounds:?}, height {height}: {labels:?}"
                    );
                    assert!(
                        labels
                            .windows(2)
                            .all(|p| p[0].0 < p[1].0 && p[0].1 > p[1].1)
                    );
                    for y in area.y + 1..area.bottom() - 1 {
                        for x in area.x + 1..area.right() - 1 {
                            assert_eq!(b[(x, y)].bg, PANEL_BG, "plot must match its panel");
                        }
                    }
                    assert_eq!(b[(area.right() - 1, area.bottom() - 1)].symbol(), "┘");
                    assert_plot(
                        b,
                        area,
                        &["loss", "measured", "training step", "Lower is better"],
                    );
                }
            }
        }
    }

    #[test]
    fn rendered_y_ticks_have_even_row_and_value_gaps_at_both_sizes() {
        // Extrema from the quick recording's 150 synthetic monitor samples.
        let monitor_loss = [3.853054, 5.737391];
        for (w, h) in [(120, 40), (80, 24)] {
            // Include the nine-row monitor/timeline plots and every smaller or
            // larger plot that fits in each terminal, with an offset panel.
            for height in 7..=h - 3 {
                for (name, y_bounds) in [
                    ("monitor loss", bounds(monitor_loss.into_iter())),
                    (
                        "monitor perplexity",
                        bounds(monitor_loss.into_iter().map(f64::exp)),
                    ),
                    ("checkpoint timeline", bounds([3.9, 5.3].into_iter())),
                    ("auto-eval", bounds([0.77, 2.57].into_iter())),
                    ("sweep", bounds([6.330799, 6.341673].into_iter())),
                    ("negative", bounds([-351.263608, -350.91351].into_iter())),
                    ("flat", bounds([0.0, 0.0].into_iter())),
                    ("empty", bounds(std::iter::empty())),
                    ("endpoints", [0.0, 4.0]),
                ] {
                    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
                    let area = Rect::new(2, 3, w - 4, height);
                    terminal
                        .draw(|f| {
                            draw(
                                f,
                                area,
                                Plot {
                                    title: name,
                                    caption: "Lower is better",
                                    x: "step",
                                    y: "value",
                                    integer_x: true,
                                    x_bounds: [1.0, 150.0],
                                    y_bounds,
                                },
                                &[Series::line(
                                    "measured",
                                    &[(1.0, y_bounds[1]), (150.0, y_bounds[0])],
                                    NORMAL_GREEN,
                                )],
                            )
                        })
                        .unwrap();
                    let b = terminal.backend().buffer();
                    let graph = graph_rect(b, area);
                    let labels: Vec<_> = (graph.y..graph.bottom())
                        .filter_map(|y| {
                            let text: String = (area.x + 1..graph.x - 1)
                                .map(|x| b[(x, y)].symbol())
                                .collect();
                            (!text.trim().is_empty())
                                .then(|| (y, text.trim().parse::<f64>().unwrap()))
                        })
                        .collect();
                    let context =
                        format!("{name}, {w}x{h}, plot height {}: {labels:?}", graph.height);
                    assert!((2..=5).contains(&labels.len()), "{context}");
                    let gaps: Vec<_> = labels.windows(2).map(|p| p[1].0 - p[0].0).collect();
                    let min = *gaps.iter().min().unwrap();
                    let max = *gaps.iter().max().unwrap();
                    assert!(min > 0, "tick rows must strictly increase: {context}");
                    assert!(max - min <= 1, "uneven tick rows: {context}");
                    assert!(min != 1 || max == 1, "mixed adjacent tick rows: {context}");
                    if name == "endpoints" {
                        assert_eq!(
                            labels.last().copied(),
                            Some((graph.bottom() - 1, 0.0)),
                            "bottom endpoint must use the last plot row: {context}"
                        );
                    }
                    let step = labels[0].1 - labels[1].1;
                    assert!(step > 0.0, "{context}");
                    for pair in labels.windows(2) {
                        assert!(
                            ((pair[0].1 - pair[1].1) - step).abs() <= step * 1e-8,
                            "unequal value steps: {context}"
                        );
                    }
                    let row_step = step * f64::from(graph.height - 1) / (y_bounds[1] - y_bounds[0]);
                    if (row_step - row_step.round()).abs() < 1e-8
                        || (graph.height == 9
                            && matches!(
                                name,
                                "monitor loss" | "monitor perplexity" | "checkpoint timeline"
                            ))
                        || (graph.height == 15 && name == "sweep")
                    {
                        assert_eq!(
                            min, max,
                            "rounding does not require unequal gaps: {context}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn strokes_are_one_dot_thick_and_connected_in_all_directions() {
        for (width, height) in [(3, 5), (20, 2)] {
            for (from, to) in [
                ((0.0, 0.0), (1.0, 1.0)),
                ((1.0, 1.0), (0.0, 0.0)),
                ((0.0, 1.0), (1.0, 0.0)),
                ((1.0, 0.0), (0.0, 1.0)),
                ((0.0, 0.0), (0.0, 1.0)),
                ((0.0, 1.0), (0.0, 0.0)),
                ((0.0, 0.0), (1.0, 0.0)),
                ((1.0, 0.0), (0.0, 0.0)),
                ((0.0, 0.0), (0.0, 0.0)),
            ] {
                let area = Rect::new(0, 0, width, height);
                let mut b = Buffer::empty(area);
                Canvas::default()
                    .marker(Marker::Braille)
                    .x_bounds([0.0, 1.0])
                    .y_bounds([0.0, 1.0])
                    .paint(|ctx| {
                        ctx.draw(&ThinLine {
                            from,
                            to,
                            color: NORMAL_GREEN,
                        })
                    })
                    .render(area, &mut b);
                let pixels = dots(&b, area);
                let project = |(x, y): (f64, f64)| {
                    (
                        (x * f64::from(width * 2 - 1)) as u16,
                        ((1.0 - y) * f64::from(height * 4 - 1)) as u16,
                    )
                };
                let (a, z) = (project(from), project(to));
                assert!(pixels.contains(&a) && pixels.contains(&z));
                let (dx, dy) = (a.0.abs_diff(z.0), a.1.abs_diff(z.1));
                assert_eq!(
                    pixels.len(),
                    usize::from(dx.max(dy)) + 1,
                    "thick stroke {from:?} -> {to:?}"
                );
                let mut ordered: Vec<_> = pixels.into_iter().collect();
                if dy > dx {
                    ordered.sort_by_key(|p| (p.1, p.0));
                }
                assert!(
                    ordered
                        .windows(2)
                        .all(|p| p[0].0.abs_diff(p[1].0) <= 1 && p[0].1.abs_diff(p[1].1) <= 1),
                    "disconnected stroke"
                );
            }
        }
        assert!(
            sparkline(&[4.0, 4.0], 20)
                .chars()
                .all(|c| (c as u32 - 0x2800).count_ones() == 2)
        );
    }

    #[test]
    fn linear_ramp_is_monotone_without_held_samples_at_both_sizes() {
        for (w, h, height) in [(120, 40, 12), (80, 24, 8)] {
            for descending in [false, true] {
                let values = if descending { [6.0, 3.9] } else { [3.9, 6.0] };
                let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
                let area = Rect::new(2, 3, w - 4, height);
                terminal
                    .draw(|f| {
                        draw(
                            f,
                            area,
                            Plot {
                                title: " ramp ",
                                caption: "Lower is better",
                                x: "training step",
                                y: "loss",
                                integer_x: true,
                                x_bounds: [0.0, 100.0],
                                y_bounds: bounds(values.into_iter()),
                            },
                            &[Series::line(
                                "loss",
                                &[(0.0, values[0]), (100.0, values[1])],
                                NORMAL_GREEN,
                            )],
                        )
                    })
                    .unwrap();
                let b = terminal.backend().buffer();
                let graph = graph_rect(b, area);
                assert_eq!(graph.height, if h == 40 { 8 } else { 4 });
                let pixels = dots(b, graph);
                let mut ys = Vec::new();
                // Exclude the endpoint marker radius, not the connecting line.
                for x in 2..graph.width * 2 - 2 {
                    let column: Vec<_> = pixels.iter().filter(|p| p.0 == x).map(|p| p.1).collect();
                    assert_eq!(
                        column.len(),
                        1,
                        "expected one-dot line at x={x}: {column:?}"
                    );
                    ys.push(column[0]);
                }
                assert!(ys.windows(2).all(|p| if descending {
                    p[0] <= p[1]
                } else {
                    p[0] >= p[1]
                }));
                let mut run = 1;
                let span = ys.first().unwrap().abs_diff(*ys.last().unwrap());
                let expected_step = (graph.width * 2 - 1).div_ceil(span);
                for pair in ys.windows(2) {
                    run = if pair[0] == pair[1] { run + 1 } else { 1 };
                    assert!(
                        run <= expected_step,
                        "plateau of {run} dots exceeds {expected_step}"
                    );
                }
            }
        }
    }

    #[test]
    fn three_checkpoints_have_markers_and_sloped_segments_not_a_spike() {
        let points = [(1.0, 6.0), (2.0, 1.8), (3.0, 0.77)];
        let y_bounds = bounds(points.iter().map(|p| p.1));
        for (w, h, height) in [(120, 40, 13), (80, 24, 8)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            let area = Rect::new(2, 3, w - 4, height);
            terminal
                .draw(|f| {
                    draw(
                        f,
                        area,
                        Plot {
                            title: " quality / checkpoint ",
                            caption: "Lower = less surprise",
                            x: "checkpoint",
                            y: "reference loss",
                            integer_x: true,
                            x_bounds: domain(&points),
                            y_bounds,
                        },
                        &[Series::line("reference loss", &points, NORMAL_GREEN)],
                    )
                })
                .unwrap();
            let b = terminal.backend().buffer();
            let graph = graph_rect(b, area);
            let pixels = dots(b, graph);
            let projected: Vec<_> = points
                .iter()
                .map(|&(x, y)| {
                    (
                        ((x - 1.0) * f64::from(graph.width * 2 - 1) / 2.0) as u16,
                        ((y_bounds[1] - y) * f64::from(graph.height * 4 - 1)
                            / (y_bounds[1] - y_bounds[0])) as u16,
                    )
                })
                .collect();
            for &(x, y) in &projected {
                for (dx, dy) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (x, y) = (i32::from(x) + dx, i32::from(y) + dy);
                    if x >= 0
                        && x < i32::from(graph.width * 2)
                        && y >= 0
                        && y < i32::from(graph.height * 4)
                    {
                        assert!(
                            pixels.contains(&(x as u16, y as u16)),
                            "missing point marker at {x},{y}"
                        );
                    }
                }
            }
            for pair in projected.windows(2) {
                let (a, z) = (pair[0], pair[1]);
                for x in a.0 + 2..z.0 - 1 {
                    let ys: Vec<_> = pixels.iter().filter(|p| p.0 == x).map(|p| p.1).collect();
                    assert_eq!(ys.len(), 1, "thick/disconnected segment at {x}: {ys:?}");
                    let expected = f64::from(a.1)
                        + f64::from(z.1 - a.1) * f64::from(x - a.0) / f64::from(z.0 - a.0);
                    assert!(
                        (f64::from(ys[0]) - expected).abs() <= 0.5 + 1e-9,
                        "held sample at {x}: {ys:?} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn two_checkpoints_do_not_repeat_a_rounded_middle_x_tick() {
        let points = [(1.0, 2.0), (2.0, 1.0)];
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            let area = Rect::new(2, 3, w - 4, if h == 40 { 12 } else { 8 });
            terminal
                .draw(|f| {
                    draw(
                        f,
                        area,
                        Plot {
                            title: " two checkpoints ",
                            caption: "Lower is better",
                            x: "checkpoint",
                            y: "loss",
                            integer_x: true,
                            x_bounds: domain(&points),
                            y_bounds: bounds(points.iter().map(|p| p.1)),
                        },
                        &[Series::line("loss", &points, NORMAL_GREEN)],
                    )
                })
                .unwrap();
            let buffer = terminal.backend().buffer();
            let graph = graph_rect(buffer, area);
            let row: String = (graph.x..graph.right())
                .map(|x| buffer[(x, graph.bottom() + 1)].symbol())
                .collect();
            let labels: Vec<_> = row.split_whitespace().collect();
            assert_eq!(labels, ["1", "checkpoint", "2"]);
        }
    }

    #[test]
    fn missing_samples_leave_a_visible_gap_between_connected_runs() {
        let points = [
            (0.0, 3.0),
            (20.0, 3.0),
            (50.0, f64::NAN),
            (80.0, 1.0),
            (100.0, 1.0),
        ];
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal
                .draw(|f| {
                    draw(
                        f,
                        Rect::new(0, 0, w, 9),
                        Plot {
                            title: " observations ",
                            caption: "Gaps mean no measurement",
                            integer_x: true,
                            x: "training step",
                            y: "loss",
                            x_bounds: [0.0, 100.0],
                            y_bounds: [0.0, 4.0],
                        },
                        &[Series::line("loss", &points, NORMAL_GREEN)],
                    )
                })
                .unwrap();
            let buffer = terminal.backend().buffer();
            assert_plot(
                buffer,
                Rect::new(0, 0, w, 9),
                &["training step", "loss", "Gaps mean no measurement"],
            );
            for x in w / 2 - 3..w / 2 + 3 {
                for y in 0..9 {
                    assert!(
                        !buffer[(x, y)]
                            .symbol()
                            .chars()
                            .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)),
                        "missing sample must not be bridged"
                    );
                }
            }
        }
    }

    #[test]
    fn formatting_flat_empty_and_missing_samples() {
        assert_eq!(number(331.303), "331");
        assert_eq!(number(26.087), "26.1");
        assert_eq!(number(0.00025), "2.5e-4");
        assert_eq!(ticks([2.0, 6.0]), ["2.0", "4.0", "6.0"]);
        assert_eq!(ticks([0.0, 0.0004]), ["0.00e0", "2.00e-4", "4.00e-4"]);
        assert_eq!(
            ticks([1_000_000.0, 1_000_200.0]),
            ["1000000", "1000100", "1000200"]
        );
        assert_eq!(bounds(std::iter::empty()), [0.0, 1.0]);
        let fitted = bounds([3.9, f64::NAN, 6.0, f64::INFINITY].into_iter());
        assert!((fitted[0] - 3.795).abs() < 1e-12);
        assert!((fitted[1] - 6.105).abs() < 1e-12);
        assert_eq!(domain(&[(1.0, 6.0), (2.0, 4.0)]), [1.0, 2.0]);
        assert_eq!(
            domain(&[(3.0, 1.0), (1.0, 6.0), (2.0, 4.0)]),
            [1.0, 3.0]
        );
        let flat_ticks = ticks(bounds([3.0, 3.0].into_iter()));
        assert!(flat_ticks.windows(2).all(|pair| pair[0] != pair[1]));
        for v in [0.0, 3.0, 0.00025] {
            let [lo, hi] = bounds([v, v].into_iter());
            assert!(lo < v && hi > v);
        }
        let line = sparkline(&[4.0, 1.0], 20);
        assert_eq!(line.chars().count(), 20);
        assert_ne!(
            line.chars().next().unwrap() as u32 & 0x09,
            0,
            "high endpoint uses the top dots"
        );
        assert_ne!(
            line.chars().last().unwrap() as u32 & 0xc0,
            0,
            "low endpoint uses the bottom dots"
        );
        assert!(
            line.chars()
                .filter(|c| ('\u{2801}'..='\u{28ff}').contains(c))
                .count()
                >= 18
        );
        let gaps = sparkline(&[f64::NAN, 2.0, f64::INFINITY], 12);
        assert!(gaps.starts_with('·') && gaps.ends_with('·'));
        assert_eq!(
            gaps.chars()
                .filter(|c| ('\u{2801}'..='\u{28ff}').contains(c))
                .count(),
            1
        );
        assert_eq!(sparkline(&[], 12), "·".repeat(12));
        assert_eq!(bar(0.5, 4), "⣿⣿··");
    }
}
