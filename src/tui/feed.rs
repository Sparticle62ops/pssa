//! Bounded, recorded dataset windows. Nothing moves or advances between events.
use super::{RunState, accent, panel, panel_area, parse_log_value};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::time::Instant;

#[derive(Default)]
pub(super) struct FeedState {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub field: String,
    pub rows: u64,
    pub tokens: u64,
    pub row: u64,
    pub snippet: String,
    pub token_ids: Vec<u64>,
    pub token_pieces: Vec<String>,
    pub epoch_tokens: Option<u64>,
    pub epoch_total: Option<u64>,
    pub epoch: Option<u64>,
    pub epochs: Option<u64>,
    pub step: Option<u64>,
    pub batch: Option<u64>,
    pub batches: Option<u64>,
    pub start: Option<u64>,
    pub end: Option<u64>,
    pub bytes: Option<u64>,
    pub bytes_total: Option<u64>,
    pub text_kind: String,
    pub updated_at: Option<Instant>,
}

// Match whole field names: feed_epoch_tokens must never become feed_tokens.
fn number(line: &str, key: &str) -> Option<u64> {
    line.split_whitespace().find_map(|part| {
        let (name, value) = part.split_once('=')?;
        (name == key).then(|| value.parse().ok()).flatten()
    })
}

impl FeedState {
    pub(super) fn parse(line: &str) -> Option<Self> {
        let dataset = parse_log_value(line, "feed_dataset")?;
        let ids = parse_log_value(line, "feed_token_ids").unwrap_or_default();
        // Reject a damaged list instead of shifting the id/piece pairing.
        let token_ids = if ids.is_empty() {
            Vec::new()
        } else {
            let ids: Vec<u64> = ids
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            if ids.len() > 16 {
                return None;
            }
            ids
        };
        let token_pieces = match parse_log_value(line, "feed_token_pieces") {
            Some(json) => {
                let pieces: Vec<String> = serde_json::from_str(&json).ok()?;
                if pieces.len() != token_ids.len() || pieces.len() > 16 {
                    return None;
                }
                pieces
                    .into_iter()
                    .map(|piece| crate::ui::terminal_text(&piece).chars().take(256).collect())
                    .collect()
            }
            None => Vec::new(),
        };
        let start = number(line, "feed_start");
        let end = number(line, "feed_end");
        if start.zip(end).is_some_and(|(start, end)| start > end) {
            return None;
        }
        Some(Self {
            dataset,
            config: parse_log_value(line, "feed_config").unwrap_or_default(),
            split: parse_log_value(line, "feed_split").unwrap_or_default(),
            field: parse_log_value(line, "feed_field").unwrap_or_default(),
            rows: number(line, "feed_rows").unwrap_or(0),
            tokens: number(line, "feed_tokens").unwrap_or(0),
            row: number(line, "feed_row").unwrap_or(0),
            snippet: parse_log_value(line, "feed_snippet")
                .unwrap_or_default()
                .chars()
                .take(256)
                .collect(),
            token_ids,
            token_pieces,
            epoch_tokens: number(line, "feed_epoch_tokens"),
            epoch_total: number(line, "feed_epoch_total"),
            epoch: number(line, "feed_epoch"),
            epochs: number(line, "feed_epochs"),
            step: number(line, "feed_step"),
            batch: number(line, "feed_batch"),
            batches: number(line, "feed_batches"),
            start,
            end,
            bytes: number(line, "feed_bytes"),
            bytes_total: number(line, "feed_bytes_total"),
            text_kind: parse_log_value(line, "feed_text_kind").unwrap_or_default(),
            updated_at: Some(Instant::now()),
        })
    }

    fn position(&self) -> String {
        match (self.epoch_tokens, self.epoch_total) {
            (Some(done), Some(total)) if total > 0 => format!(
                "epoch {}/{}: {done}/{total} input tokens ({:.2}%) / {} tokens consumed",
                value(self.epoch),
                value(self.epochs),
                done as f64 * 100.0 / total as f64,
                self.tokens
            ),
            _ => format!(
                "{} rows consumed / {} tokens / selected row {} (legacy log)",
                self.rows, self.tokens, self.row
            ),
        }
    }
}

fn value(value: Option<u64>) -> String {
    value.map_or_else(|| "unrecorded".into(), |n| n.to_string())
}

pub(super) fn draw(f: &mut Frame, area: Rect, state: &RunState) {
    let area = panel_area(f, area);
    let Some(feed) = &state.feed else {
        let mut lines = vec![
            Line::styled("Waiting for dataset telemetry", accent()),
            Line::from(if state.training_active {
                "Training connected; first sampled window arrives with the first progress event."
            } else {
                "Start training in Setup, or pipe a real training log:"
            }),
            Line::from("pssa train --data CORPUS --no-tui | pssa tui"),
            Line::from("Local, HF and transformer runs report actual input windows."),
        ];
        if state.corpus.is_some() {
            lines.push(Line::from(format!(
                "Selected corpus: {}",
                state.corpus.as_deref().unwrap()
            )));
            lines.push(Line::from(
                "Older logs have no window/token pieces; they cannot be recovered.",
            ));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" dataset feed / no window recorded ")),
            area,
        );
        return;
    };
    let mut lines = vec![
        Line::styled(&feed.dataset, accent()),
        Line::from(feed.position()),
    ];
    if area.height > 7 {
        if !feed.split.is_empty() || !feed.field.is_empty() {
            lines.push(Line::from(format!(
                "config {} / split {} / field {}",
                feed.config, feed.split, feed.field
            )));
        }
        lines.push(Line::from(format!(
            "step {} / batch {}/{} / selected document {}/{} / tokens {}..{}",
            value(feed.step),
            value(feed.batch),
            value(feed.batches),
            feed.row,
            feed.rows,
            value(feed.start),
            value(feed.end)
        )));
        lines.push(Line::from(match (feed.bytes, feed.bytes_total) {
            (Some(bytes), Some(total)) => format!("Source bytes consumed: {bytes}/{total} (UTF-8)"),
            _ => {
                "Source bytes consumed: unavailable (no exact source offsets in this token stream)"
                    .into()
            }
        }));
    }
    let text_label = match feed.text_kind.as_str() {
        "raw" => "Source text window (truncated)",
        "decoded" => "Input window decoded from actual IDs (normalized; not original source bytes)",
        _ => "Recorded snippet (legacy log; original source fidelity unrecorded)",
    };
    lines.push(Line::styled(text_label, accent()));
    lines.push(Line::from(feed.snippet.clone()));
    lines.push(Line::styled(
        "Token pieces → IDs (bounded preview, in input order)",
        accent(),
    ));
    if feed.token_pieces.is_empty() {
        lines.push(Line::from(format!(
            "IDs: {} / pieces not recorded by this older log",
            feed.token_ids
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    } else {
        // Compact rows retain all sixteen id/piece pairs on a normal terminal.
        for pairs in feed
            .token_ids
            .iter()
            .zip(&feed.token_pieces)
            .collect::<Vec<_>>()
            .chunks(4)
        {
            lines.push(Line::from(
                pairs
                    .iter()
                    .map(|(id, piece)| format!("{piece:?} → {id}"))
                    .collect::<Vec<_>>()
                    .join("  |  "),
            ));
        }
    }
    lines.push(Line::from(format!(
        "Latest recorded window{}; positions do not advance between events.",
        feed.updated_at.map_or(String::new(), |at| format!(
            " / {:.1}s ago",
            at.elapsed().as_secs_f64()
        ))
    )));
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" dataset feed / real text → tokens ")),
        area,
    );
}

pub(super) fn draw_compact(f: &mut Frame, area: Rect, state: &RunState) {
    let area = panel_area(f, area);
    let lines = match &state.feed {
        Some(feed) => vec![
            Line::styled(
                format!("step {} / doc {}", value(feed.step), feed.row),
                accent(),
            ),
            Line::from(feed.snippet.clone()),
            Line::from(format!(
                "IDs: {}",
                feed.token_ids
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )),
            Line::from(format!("{} input tokens consumed", feed.tokens)),
        ],
        None => vec![
            Line::from("No input window recorded"),
            Line::from(format!("step {}", value(state.current_step()))),
            Line::from("Feed tab shows dataset telemetry when emitted."),
        ],
    };
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" actual input / Feed tab ")),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::encode_log_value;
    use ratatui::{Terminal, backend::TestBackend};

    fn event() -> String {
        format!(
            "feed_schema=2 feed_dataset={} feed_tokens=64 feed_rows=3 feed_row=2 feed_epoch=2 feed_epochs=4 feed_epoch_tokens=16 feed_epoch_total=32 feed_step=8 feed_batch=2 feed_batches=4 feed_start=12 feed_end=14 feed_text_kind=raw feed_bytes=27 feed_bytes_total=54 feed_snippet={} feed_token_ids=7%2C9 feed_token_pieces={}",
            encode_log_value("/tmp/corpus with spaces.txt"),
            encode_log_value("héllo 世界"),
            encode_log_value(r#"["héllo","世界"]"#)
        )
    }
    #[test]
    fn parses_exact_fields_and_pairs_unicode_pieces_with_ids() {
        let feed = FeedState::parse(&event()).unwrap();
        assert_eq!(feed.dataset, "/tmp/corpus with spaces.txt");
        assert_eq!(feed.token_ids, [7, 9]);
        assert_eq!(feed.token_pieces, ["héllo", "世界"]);
        assert_eq!(feed.tokens, 64);
        assert_eq!(feed.epoch_tokens, Some(16));
        assert_eq!(
            (feed.start, feed.end, feed.bytes),
            (Some(12), Some(14), Some(27))
        );
        assert!(feed.position().contains("50.00%"));
    }
    #[test]
    fn rejects_malformed_or_misaligned_events_and_sanitizes_controls() {
        assert!(FeedState::parse(&event().replace("7%2C9", "7%2Cbad")).is_none());
        assert!(FeedState::parse(&event().replace("feed_end=14", "feed_end=10")).is_none());
        assert!(
            FeedState::parse("feed_dataset=x feed_token_ids=1 feed_token_pieces=%5B%5D").is_none()
        );
        assert!(FeedState::parse("feed_dataset=%FF").is_none());
        let line = format!(
            "feed_dataset=x feed_token_ids=1 feed_token_pieces={}",
            encode_log_value(r#"["\u001b[31m\n"]"#)
        );
        let feed = FeedState::parse(&line).unwrap();
        assert!(!feed.token_pieces[0].contains('\u{1b}'));
        assert!(!feed.token_pieces[0].contains('\n'));
    }
    #[test]
    fn renders_real_window_progress_and_tokens_at_wide_and_narrow_sizes() {
        let mut state = RunState::default();
        state.ingest(&event());
        for (w, h) in [
            (120, 40),
            (80, 24),
            (60, 20),
            (30, 10),
            (10, 5),
            (1, 1),
            (0, 0),
        ] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, f.area(), &state)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if w >= 60 {
                for expected in [
                    "corpus with spaces.txt",
                    "50.00%",
                    "héllo",
                    "世界",
                    "→ 7",
                    "→ 9",
                    "27/54",
                    "step 8",
                ] {
                    assert!(
                        text.contains(expected),
                        "{w}x{h} missing {expected}: {text}"
                    );
                }
            }
        }
    }
    #[test]
    fn absent_bytes_and_legacy_pieces_are_not_invented() {
        let mut state = RunState::default();
        state.ingest("feed_dataset=old feed_tokens=5 feed_token_ids=1%2C2 feed_snippet=hello");
        let mut terminal = Terminal::new(TestBackend::new(120, 24)).unwrap();
        terminal.draw(|f| draw(f, f.area(), &state)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("Source bytes consumed: unavailable"));
        assert!(text.contains("pieces not recorded"));
        assert!(text.contains("legacy log"));
        assert!(!text.contains("0.00%"));
    }
}
