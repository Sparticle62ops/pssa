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

/// Round outwards to a useful domain; include room above/below a flat trace.
pub(super) fn bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.filter(|v| v.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    if !low.is_finite() {
        return [0.0, 1.0];
    }
    let pad = ((high - low) * 0.06).max(high.abs() * 0.01).max(1e-9);
    let rough = (high - low + 2.0 * pad) / 2.0;
    let power = 10.0_f64.powf(rough.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .into_iter()
        .find(|n| *n * power >= rough)
        .unwrap_or(10.0)
        * power;
    [
        ((low - pad) / step).floor() * step,
        ((high + pad) / step).ceil() * step,
    ]
}

pub(super) fn domain(points: &[(f64, f64)]) -> [f64; 2] {
    let first = points.first().map_or(0.0, |p| p.0);
    let last = points.last().map_or(first + 2.0, |p| p.0);
    [first, last.max(first + 2.0)]
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

pub(super) fn draw(f: &mut Frame, area: Rect, plot: Plot<'_>, series: &[Series<'_>]) {
    draw_labeled(f, area, plot, series, None);
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
        let legend = format!(" | {} ", s.name);
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
    let y_labels = y_labels.unwrap_or_else(|| ticks(plot.y_bounds));
    let label_width = y_labels.iter().map(|s| s.len()).max().unwrap_or(0) as u16;
    let gutter = label_width.min(inner.width / 3) + 1;
    // Reserve separate rows for the axis stroke and its labels, never data.
    let graph = Rect::new(
        inner.x + gutter,
        inner.y,
        inner.width - gutter,
        inner.height - 2,
    );
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
    for (i, label) in y_labels.iter().enumerate() {
        // Anchor the endpoints to the actual data rows; rounding can differ
        // by at most one terminal row at the midpoint of an even height.
        let offset = i as u16 * (graph.height - 1) / y_labels.len().saturating_sub(1).max(1) as u16;
        Line::styled(label.as_str(), accent())
            .right_aligned()
            .render(
                Rect::new(inner.x, graph.bottom() - 1 - offset, gutter - 1, 1),
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
                for s in series {
                    // Split at missing observations, never bridge a gap.
                    for run in s.points.split(|(x, y)| !x.is_finite() || !y.is_finite()) {
                        ctx.draw(&Points {
                            coords: run,
                            color: s.color,
                        });
                        if !s.scatter {
                            for pair in run.windows(2) {
                                ctx.draw(&DenseLine {
                                    from: pair[0],
                                    to: pair[1],
                                    color: s.color,
                                });
                            }
                        }
                    }
                    ctx.layer();
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
    if gap(&positions).1 < name.len() as u16 {
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

/// A two-dot-weight braille stroke. Fill each column's vertical interval to
/// the previous column, rather than leaving a diagonal chain of single dots.
/// Scatter points remain points, and callers split missing observations first.
struct DenseLine {
    from: (f64, f64),
    to: (f64, f64),
    color: Color,
}

impl Shape for DenseLine {
    fn draw(&self, painter: &mut Painter) {
        let Some((mut x1, mut y1)) = painter.get_point(self.from.0, self.from.1) else {
            return;
        };
        let Some((mut x2, mut y2)) = painter.get_point(self.to.0, self.to.1) else {
            return;
        };
        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }
        let mut previous = y1;
        for x in x1..=x2 {
            let y = if x1 == x2 {
                y2
            } else {
                (y1 as f64 + (y2 as f64 - y1 as f64) * (x - x1) as f64 / (x2 - x1) as f64).round()
                    as usize
            };
            for dot_y in previous.min(y)..=previous.max(y) {
                painter.paint(x, dot_y, self.color);
                // Pair within the same cell, including at the canvas edges.
                painter.paint(x, dot_y ^ 1, self.color);
            }
            previous = y;
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
                    ctx.draw(&DenseLine {
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
    fn loss_ticks_are_even_and_labels_stay_outside_data() {
        for (w, h) in [(120, 40), (80, 24), (48, 18)] {
            for height in [8, 9, 12] {
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
                                y_bounds: [2.0, 6.0],
                            },
                            &[Series::line(
                                "measured",
                                &[(1.0, 6.0), (150.0, 2.0)],
                                NORMAL_GREEN,
                            )],
                        )
                    })
                    .unwrap();
                let b = terminal.backend().buffer();
                let row = |y| {
                    (area.x..area.right())
                        .map(|x| b[(x, y)].symbol())
                        .collect::<String>()
                };
                let top = area.y + 1;
                let bottom = area.bottom() - 4;
                let middle = bottom - (bottom - top) / 2;
                for (y, label) in [(top, "6.0"), (middle, "4.0"), (bottom, "2.0")] {
                    assert_eq!(
                        (area.x + 1..area.x + 4)
                            .map(|x| b[(x, y)].symbol())
                            .collect::<String>(),
                        label
                    );
                }
                assert!((middle - top).abs_diff(bottom - middle) <= 1);
                assert!(row(area.y).contains("loss"));
                assert!(row(area.y).contains("measured"));
                let labels = row(area.bottom() - 2);
                let tokens: Vec<_> = labels
                    .split(|c: char| c.is_whitespace() || c == '│')
                    .filter(|s| !s.is_empty())
                    .collect();
                for label in ["1", "76", "150", "training", "step"] {
                    assert!(tokens.contains(&label), "missing {label}: {labels}");
                }
                assert!(!labels.contains('.'), "fractional training step: {labels}");
                for y in top..=bottom {
                    for x in area.x + 5..area.right() - 1 {
                        let cell = &b[(x, y)];
                        assert!(
                            cell.symbol() == " "
                                || cell
                                    .symbol()
                                    .chars()
                                    .all(|c| ('\u{2800}'..='\u{28ff}').contains(&c)),
                            "text over data: {}",
                            row(y)
                        );
                    }
                }
                // The high endpoint occupies the first data cell, not a title.
                assert!(
                    b[(area.x + 5, top)]
                        .symbol()
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                );
                for y in top..area.bottom() - 1 {
                    for x in area.x + 1..area.right() - 1 {
                        assert_eq!(b[(x, y)].bg, PANEL_BG, "plot must match its panel");
                    }
                }
            }
        }
    }

    #[test]
    fn dense_strokes_fill_vertical_gaps_and_give_flat_traces_weight() {
        for values in [&[4.0, 4.0][..], &[4.0, 1.0], &[1.0, 4.0]] {
            let line = sparkline(values, 20);
            assert!(
                line.chars().all(|c| (c as u32 - 0x2800).count_ones() >= 4),
                "faint trace: {line}"
            );
        }
        // Exercise steep and vertical strokes, both directions, at dot resolution.
        for (from, to) in [
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, 1.0)),
        ] {
            let area = Rect::new(0, 0, 3, 5);
            let mut b = Buffer::empty(area);
            Canvas::default()
                .marker(Marker::Braille)
                .x_bounds([0.0, 1.0])
                .y_bounds([0.0, 1.0])
                .paint(|ctx| {
                    ctx.draw(&DenseLine {
                        from,
                        to,
                        color: NORMAL_GREEN,
                    })
                })
                .render(area, &mut b);
            let mut previous: Option<Vec<usize>> = None;
            for x in 0..area.width * 2 {
                let mut dots = Vec::new();
                for y in 0..area.height * 4 {
                    let ch = b[(x / 2, y / 4)].symbol().chars().next().unwrap();
                    let mask = [[1, 2, 4, 64], [8, 16, 32, 128]][x as usize % 2][y as usize % 4];
                    if ch != ' ' && (ch as u32 - 0x2800) & mask != 0 {
                        dots.push(y as usize);
                    }
                }
                if dots.is_empty() {
                    continue;
                }
                assert!(dots.len() >= 2);
                assert!(
                    dots.windows(2).all(|pair| pair[1] == pair[0] + 1),
                    "gap in stroke: {dots:?}"
                );
                if let Some(prior) = &previous {
                    assert!(
                        dots.iter().any(|y| prior.contains(y)),
                        "disconnected columns"
                    );
                }
                previous = Some(dots);
            }
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
