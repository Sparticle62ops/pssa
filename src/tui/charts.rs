//! Shared CRT plots: connected braille strokes, honest gaps, and readable scales.
use super::{NORMAL_GREEN, SECOND_ACCENT, accent, panel, panel_area};
use ratatui::{
    Frame,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    text::Line,
    widgets::{
        Axis, Chart, Dataset, GraphType, Paragraph, Widget,
        canvas::{Canvas, Line as Stroke, Points},
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
    (0..3)
        .map(|i| {
            let value = bounds[0] + step * i as f64;
            if step < 0.01 && value != 0.0 {
                format!("{value:.2e}").replace(".00e", "e")
            } else if step < 1.0 {
                format!("{value:.2}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_owned()
            } else if value.abs() >= 1_000_000.0 && step < value.abs() * 0.01 {
                // Nearby large training steps must not all become "1e6".
                format!("{value:.0}")
            } else {
                number(value)
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
    let block =
        panel(plot.title).title_bottom(Line::styled(plot.caption, Style::new().fg(SECOND_ACCENT)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 16 || inner.height < 5 {
        f.render_widget(
            Paragraph::new(format!("{} / {}\nEnlarge for plot", plot.y, plot.x)).style(accent()),
            inner,
        );
        return;
    }
    // Split at missing observations rather than drawing an invented bridge.
    let datasets = series
        .iter()
        .flat_map(|s| {
            s.points
                .split(|(x, y)| !x.is_finite() || !y.is_finite())
                .filter(|run| !run.is_empty())
                .map(move |run| {
                    Dataset::default()
                        .name(s.name)
                        .marker(Marker::Braille)
                        .graph_type(if s.scatter {
                            GraphType::Scatter
                        } else {
                            GraphType::Line
                        })
                        .style(Style::new().fg(s.color))
                        .data(run)
                })
        })
        .collect::<Vec<_>>();
    f.render_widget(
        Chart::new(datasets)
            // Legends otherwise hide much of a short plot; captions identify traces.
            .legend_position(None)
            .x_axis(
                Axis::default()
                    .title(plot.x)
                    .bounds(plot.x_bounds)
                    .labels(ticks(plot.x_bounds))
                    .style(accent()),
            )
            .y_axis(
                Axis::default()
                    .title(plot.y)
                    .bounds(plot.y_bounds)
                    .labels(y_labels.unwrap_or_else(|| ticks(plot.y_bounds)))
                    .style(accent()),
            ),
        inner,
    );
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
                    ctx.draw(&Stroke::new(
                        pair[0].0,
                        pair[0].1,
                        pair[1].0,
                        pair[1].1,
                        NORMAL_GREEN,
                    ));
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
