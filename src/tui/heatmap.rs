//! Display-only raw-softmax confidence, never a sampling control.
use super::{AMBER, NORMAL_GREEN, SECOND_ACCENT};
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

const LOW: Color = Color::Rgb(0xe0, 0x62, 0x62);

#[derive(Clone, Debug)]
pub(super) struct TokenMark {
    pub end: usize,
    pub probability: f32,
}

pub(super) fn to_json(marks: &[TokenMark]) -> serde_json::Value {
    serde_json::Value::Array(
        marks
            .iter()
            .map(|m| {
                serde_json::json!({
                    "end": m.end, "probability": m.probability
                })
            })
            .collect(),
    )
}

pub(super) fn from_json(value: &serde_json::Value) -> Result<Vec<TokenMark>, String> {
    let entries = value
        .as_array()
        .filter(|a| a.len() <= 4096)
        .ok_or("invalid confidence")?;
    entries
        .iter()
        .map(|m| {
            Ok(TokenMark {
                end: m["end"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("invalid token end")?,
                probability: m["probability"].as_f64().ok_or("invalid probability")? as f32,
            })
        })
        .collect()
}

pub(super) fn record(marks: &mut Vec<TokenMark>, text: &str, count: usize, probability: f32) {
    if count == 0 || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return;
    }
    let mark = TokenMark {
        end: text.len(),
        probability,
    };
    if count == marks.len() + 1 {
        marks.push(mark);
    } else if count == marks.len() {
        // BPE final UTF-8 flush updates the last token, not its count.
        marks[count - 1] = mark;
    }
}

pub(super) fn valid(text: &str, marks: &[TokenMark]) -> bool {
    let mut previous = 0;
    marks.len() <= 4096
        && marks.iter().all(|mark| {
            let ok = mark.end >= previous
                && text.is_char_boundary(mark.end)
                && mark.probability.is_finite()
                && (0.0..=1.0).contains(&mark.probability);
            previous = mark.end;
            ok
        })
        && marks.last().is_none_or(|m| m.end == text.len())
}

fn color(probability: f32) -> Color {
    if probability < 0.1 {
        LOW
    } else if probability < 0.5 {
        AMBER
    } else {
        NORMAL_GREEN
    }
}

pub(super) fn legend(enabled: bool) -> Line<'static> {
    if !enabled {
        return Line::from("F6 heatmap off / raw model confidence");
    }
    Line::from(vec![
        Span::raw("F6 heatmap: "),
        Span::styled("<10% ", Style::new().fg(LOW)),
        Span::styled("10-50% ", Style::new().fg(AMBER)),
        Span::styled(">=50% ", Style::new().fg(NORMAL_GREEN)),
        Span::styled("n/a", Style::new().fg(SECOND_ACCENT)),
    ])
}

pub(super) fn clean(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

pub(super) fn lines(text: &str, marks: &[TokenMark], enabled: bool) -> Vec<Line<'static>> {
    let mut lines = vec![Line::default()];
    let mut append = |text: &str, style: Style| {
        for (index, part) in clean(text).split('\n').enumerate() {
            if index > 0 {
                lines.push(Line::default());
            }
            lines
                .last_mut()
                .unwrap()
                .spans
                .push(Span::styled(part.to_owned(), style));
        }
    };
    if !enabled || marks.is_empty() || !valid(text, marks) {
        append(
            text,
            if enabled {
                Style::new().fg(SECOND_ACCENT)
            } else {
                Style::default()
            },
        );
        return lines;
    }
    let mut start = 0;
    let mut probability = 1.0_f32;
    for mark in marks {
        // UTF-8 codepoints split over BPE tokens cannot be colored by byte.
        // Combine those tokens using their minimum confidence; never split UTF-8.
        probability = probability.min(mark.probability);
        if mark.end > start {
            append(&text[start..mark.end], Style::new().fg(color(probability)));
            start = mark.end;
            probability = 1.0;
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{
        Terminal,
        backend::TestBackend,
        widgets::{Paragraph, Wrap},
    };

    #[test]
    fn render_confidence_legend_and_multiline_unicode_at_all_widths() {
        for width in [120, 79, 40, 20] {
            let mut terminal = Terminal::new(TestBackend::new(width, 12)).unwrap();
            let text = "low medium high\n猫";
            let marks = vec![
                TokenMark {
                    end: 4,
                    probability: 0.01,
                },
                TokenMark {
                    end: 11,
                    probability: 0.3,
                },
                TokenMark {
                    end: text.len(),
                    probability: 0.9,
                },
            ];
            terminal
                .draw(|f| {
                    let mut content = vec![legend(true)];
                    content.extend(lines(text, &marks, true));
                    f.render_widget(Paragraph::new(content).wrap(Wrap { trim: false }), f.area());
                })
                .unwrap();
            let b = terminal.backend().buffer();
            let output: String = b.content().iter().map(|c| c.symbol()).collect();
            assert!(output.contains("low medium high"));
            assert!(output.contains("猫"));
            for (word, expected) in [("l", LOW), ("m", AMBER), ("h", NORMAL_GREEN)] {
                assert!(
                    b.content()
                        .iter()
                        .any(|c| c.symbol() == word && c.fg == expected)
                );
            }
        }
    }

    #[test]
    fn bpe_flush_validation_unknown_and_control_sequences() {
        let mut marks = Vec::new();
        record(&mut marks, "", 1, 0.01);
        record(&mut marks, "é", 2, 0.9);
        record(&mut marks, "é�", 2, 0.9);
        assert_eq!(marks.len(), 2);
        assert!(valid("é�", &marks));
        assert_eq!(lines("é�", &marks, true)[0].spans[0].style.fg, Some(LOW));
        assert!(!valid("a", &marks));
        assert!(!valid(
            "é",
            &[TokenMark {
                end: 1,
                probability: 0.1
            }]
        ));
        assert_eq!(
            lines("old chat", &[], true)[0].spans[0].style.fg,
            Some(SECOND_ACCENT)
        );
        assert_eq!(clean("a\x1bb"), "ab");
    }
}
