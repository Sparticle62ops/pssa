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
    pub rows: Option<u64>,
    pub tokens: Option<u64>,
    pub row: Option<u64>,
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
    pub source_row: Option<u64>,
    pub source_start: Option<u64>,
    pub source_end: Option<u64>,
    pub skip_tokens: Option<u64>,
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
        let schema = line
            .split_whitespace()
            .find_map(|part| part.strip_prefix("feed_schema="))
            .map(str::parse::<u64>)
            .transpose()
            .ok()?;
        if schema.is_some_and(|schema| schema != 2) {
            return None;
        }
        if schema == Some(2) {
            // Optional counters may be absent, but a present damaged value
            // must not masquerade as an honest unavailable measurement.
            for key in [
                "feed_tokens",
                "feed_rows",
                "feed_row",
                "feed_epoch_tokens",
                "feed_epoch_total",
                "feed_epoch",
                "feed_epochs",
                "feed_step",
                "feed_batch",
                "feed_batches",
                "feed_start",
                "feed_end",
                "feed_source_row",
                "feed_source_start",
                "feed_source_end",
                "feed_skip_tokens",
                "feed_bytes",
                "feed_bytes_total",
            ] {
                let mut fields = line.split_whitespace().filter_map(|part| {
                    let (name, value) = part.split_once('=')?;
                    (name == key).then_some(value)
                });
                if let Some(value) = fields.next()
                    && (value.parse::<u64>().is_err() || fields.next().is_some())
                {
                    return None;
                }
            }
        }
        let ids = parse_log_value(line, "feed_token_ids");
        if schema == Some(2) && ids.is_none() {
            return None;
        }
        let ids = ids.unwrap_or_default();
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
            None if schema == Some(2) => return None,
            None => Vec::new(),
        };
        let start = number(line, "feed_start");
        let end = number(line, "feed_end");
        if start.zip(end).is_some_and(|(start, end)| start > end) {
            return None;
        }
        let snippet = parse_log_value(line, "feed_snippet");
        let tokens = number(line, "feed_tokens");
        let row = number(line, "feed_row");
        let rows = number(line, "feed_rows");
        let epoch_tokens = number(line, "feed_epoch_tokens");
        let epoch_total = number(line, "feed_epoch_total");
        let epoch = number(line, "feed_epoch");
        let epochs = number(line, "feed_epochs");
        let step = number(line, "feed_step");
        let batch = number(line, "feed_batch");
        let batches = number(line, "feed_batches");
        let text_kind = parse_log_value(line, "feed_text_kind").unwrap_or_default();
        if schema == Some(2) {
            let (tokens, row, rows, done, total, epoch, epochs, batch, batches, start, end) = (
                tokens?,
                row?,
                rows?,
                epoch_tokens?,
                epoch_total?,
                epoch?,
                epochs?,
                batch?,
                batches?,
                start?,
                end?,
            );
            if snippet.is_none()
                || step.is_none()
                || token_ids.is_empty()
                || row == 0
                || row > rows
                || total == 0
                || done > total
                || done > tokens
                || epoch == 0
                || epoch > epochs
                || batch == 0
                || batch > batches
                || end - start != token_ids.len() as u64
                || !matches!(text_kind.as_str(), "raw" | "decoded")
            {
                return None;
            }
        }
        let source_start = number(line, "feed_source_start");
        let source_end = number(line, "feed_source_end");
        if source_start.is_some() != source_end.is_some()
            || source_start
                .zip(source_end)
                .is_some_and(|(start, end)| end < start || end - start != token_ids.len() as u64)
        {
            return None;
        }
        let bytes = number(line, "feed_bytes");
        let bytes_total = number(line, "feed_bytes_total");
        if bytes.is_some() != bytes_total.is_some()
            || bytes
                .zip(bytes_total)
                .is_some_and(|(done, total)| done > total)
        {
            return None;
        }
        Some(Self {
            dataset,
            config: parse_log_value(line, "feed_config").unwrap_or_default(),
            split: parse_log_value(line, "feed_split").unwrap_or_default(),
            field: parse_log_value(line, "feed_field").unwrap_or_default(),
            rows,
            tokens,
            row,
            snippet: snippet.unwrap_or_default().chars().take(256).collect(),
            token_ids,
            token_pieces,
            epoch_tokens,
            epoch_total,
            epoch,
            epochs,
            step,
            batch,
            batches,
            start,
            end,
            source_row: number(line, "feed_source_row"),
            source_start,
            source_end,
            skip_tokens: number(line, "feed_skip_tokens"),
            bytes,
            bytes_total,
            text_kind,
            updated_at: Some(Instant::now()),
        })
    }

    fn position(&self) -> String {
        match (self.epoch_tokens, self.epoch_total) {
            (Some(done), Some(total)) if total > 0 => format!(
                "epoch {}/{}: {done}/{total} inputs ({:.2}%) / consumed {}",
                value(self.epoch),
                value(self.epochs),
                done as f64 * 100.0 / total as f64,
                value(self.tokens)
            ),
            _ => format!(
                "{} rows consumed / {} tokens / selected row {} (legacy log)",
                value(self.rows),
                value(self.tokens),
                value(self.row)
            ),
        }
    }
}

pub(super) fn value(value: Option<u64>) -> String {
    value.map_or_else(|| "unrecorded".into(), |n| n.to_string())
}

pub(super) fn draw(f: &mut Frame, area: Rect, state: &RunState) {
    let area = panel_area(f, area);
    let Some(feed) = &state.feed else {
        let mut lines = vec![
            Line::styled("Waiting for dataset telemetry", accent()),
            Line::from(if !state.metric_series.is_empty() {
                "Progress received without a valid window; producer/log may predate dataset telemetry."
            } else if state.training_active {
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
                if feed.config.is_empty() {
                    "unrecorded"
                } else {
                    &feed.config
                },
                if feed.split.is_empty() {
                    "unrecorded"
                } else {
                    &feed.split
                },
                if feed.field.is_empty() {
                    "unrecorded"
                } else {
                    &feed.field
                }
            )));
        }
        lines.push(Line::from(format!(
            "step {} / batch {}/{} / selected doc {}/{} / tokens {}..{}",
            value(feed.step),
            value(feed.batch),
            value(feed.batches),
            value(feed.row),
            value(feed.rows),
            value(feed.start),
            value(feed.end)
        )));
        if feed.source_start.is_some() {
            lines.push(Line::from(format!(
                "Dataset row {} / source tokens {}..{} / skip {} (offset wraps at EOF)",
                value(feed.source_row),
                value(feed.source_start),
                value(feed.source_end),
                value(feed.skip_tokens)
            )));
        }
        lines.push(Line::from(match (feed.bytes, feed.bytes_total) {
            (Some(bytes), Some(total)) => format!(
                "Source bytes consumed: {bytes}/{total} (UTF-8 input spans; excludes row separators/target-only tokens)"
            ),
            _ => {
                "Source bytes consumed: unavailable (no exact source offsets in this token stream)"
                    .into()
            }
        }));
    }
    let text_label = match feed.text_kind.as_str() {
        "raw" => "Source text window (truncated; UTF-8 boundary context)",
        "decoded" => "Input window decoded from actual IDs (normalized; not original source bytes)",
        _ => "Recorded snippet (legacy log; original source fidelity unrecorded)",
    };
    lines.push(Line::styled(text_label, accent()));
    lines.push(Line::from(feed.snippet.clone()));
    lines.push(Line::styled(
        "Token pieces → IDs (input order; … = truncated)",
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
        "Latest recorded window{}.",
        feed.updated_at.map_or(String::new(), |at| format!(
            " / {:.1}s ago",
            at.elapsed().as_secs_f64()
        ))
    )));
    lines.push(Line::from("Positions do not advance between events."));
    let block = panel(" dataset feed / real text → tokens / PgUp/Dn scroll ");
    let inner = block.inner(area);
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let max_scroll = paragraph
        .line_count(inner.width)
        .saturating_sub(inner.height as usize)
        .min(u16::MAX as usize) as u16;
    state
        .feed_scroll
        .set(state.feed_scroll.get().min(max_scroll));
    f.render_widget(
        paragraph.scroll((state.feed_scroll.get(), 0)).block(block),
        area,
    );
}

pub(super) fn draw_compact(f: &mut Frame, area: Rect, state: &RunState) {
    let area = panel_area(f, area);
    let lines = match &state.feed {
        Some(feed) => vec![
            Line::styled(
                format!("step {} / doc {}", value(feed.step), value(feed.row)),
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
            Line::from(format!("{} input tokens consumed", value(feed.tokens))),
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

    fn rendered_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        if buffer.area.width == 0 {
            return String::new();
        }
        let mut text = String::new();
        for row in buffer.content().chunks(buffer.area.width as usize) {
            let mut continuation = 0;
            for cell in row {
                if continuation > 0 {
                    continuation -= 1;
                    continue;
                }
                text.push_str(cell.symbol());
                continuation = Line::from(cell.symbol()).width().saturating_sub(1);
            }
            text.push('\n');
        }
        text
    }

    fn event() -> String {
        format!(
            "feed_schema=2 feed_dataset={} feed_tokens=64 feed_rows=3 feed_row=2 feed_epoch=2 feed_epochs=4 feed_epoch_tokens=16 feed_epoch_total=32 feed_step=8 feed_batch=2 feed_batches=4 feed_start=12 feed_end=14 feed_source_row=4 feed_source_start=112 feed_source_end=114 feed_skip_tokens=100 feed_text_kind=raw feed_bytes=27 feed_bytes_total=54 feed_snippet={} feed_token_ids=7%2C9 feed_token_pieces={}",
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
        assert_eq!(feed.tokens, Some(64));
        assert_eq!(feed.epoch_tokens, Some(16));
        assert_eq!(feed.source_row, Some(4));
        assert_eq!(
            (feed.source_start, feed.source_end, feed.skip_tokens),
            (Some(112), Some(114), Some(100))
        );
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
    fn damaged_new_schema_never_replaces_the_last_valid_window() {
        let mut state = RunState::default();
        state.ingest(&event());
        for (field, damage) in [
            ("feed_schema=2", "feed_schema=bad"),
            ("feed_schema=2", "feed_schema=3"),
            ("feed_schema=2", "feed_schema=%ZZ"),
            ("feed_schema=2", "feed_schema="),
            ("feed_epoch_total=32", "feed_epoch_total=0"),
            ("feed_epoch_tokens=16", "feed_epoch_tokens=33"),
            ("feed_row=2", "feed_row=0"),
            ("feed_batch=2", "feed_batch=5"),
            ("feed_end=14", "feed_end=15"),
            ("feed_source_end=114", "feed_source_end=111"),
            ("feed_bytes=27", "feed_bytes=55"),
            (
                "feed_bytes=27 feed_bytes_total=54",
                "feed_bytes=bad feed_bytes_total=bad",
            ),
            ("feed_source_row=4", "feed_source_row=bad"),
            ("feed_skip_tokens=100", "feed_skip_tokens=bad"),
            ("feed_bytes=27", "feed_bytes=27 feed_bytes=28"),
            ("feed_token_ids=7%2C9", "feed_token_ids=%ZZ"),
            ("feed_token_pieces=", "damaged_token_pieces="),
            ("feed_snippet=", "damaged_snippet="),
        ] {
            let damaged = event().replace(field, damage);
            assert!(FeedState::parse(&damaged).is_none(), "accepted {damaged}");
            state.ingest(&damaged);
            assert_eq!(state.feed.as_ref().unwrap().snippet, "héllo 世界");
        }
    }

    #[test]
    fn missing_legacy_counters_are_unrecorded_not_zero() {
        let mut state = RunState::default();
        state.ingest("feed_dataset=legacy feed_snippet=hello feed_token_ids=1");
        let feed = state.feed.as_ref().unwrap();
        assert_eq!((feed.tokens, feed.row, feed.rows), (None, None, None));
        assert!(feed.position().contains("unrecorded tokens"));
        assert!(!feed.position().contains("0 tokens"));
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
            let text = rendered_text(&terminal);
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
    fn full_feed_shell_scrolls_to_every_pair_without_inventing_positions() {
        use super::super::keybindings::{self, Action, Context};
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let ids = (1..=16)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let pieces: Vec<_> = (1..=16)
            .map(|n| format!("a_long_real_vocabulary_piece_{n}"))
            .collect();
        let line = event()
            .replace("feed_start=12 feed_end=14", "feed_start=0 feed_end=16")
            .replace("feed_source_end=114", "feed_source_end=128")
            .replace("7%2C9", &encode_log_value(&ids))
            .replace(
                &encode_log_value(r#"["héllo","世界"]"#),
                &encode_log_value(&serde_json::to_string(&pieces).unwrap()),
            );
        let mut state = RunState::default();
        state.ingest(&line);
        assert_eq!(Context::for_tab(3, false), Context::Feed);
        for (key, expected) in [
            (KeyCode::Down, Action::FeedScroll(1)),
            (KeyCode::PageDown, Action::FeedScroll(8)),
            (KeyCode::Home, Action::FeedTop),
            (KeyCode::End, Action::FeedBottom),
        ] {
            assert_eq!(
                keybindings::action(KeyEvent::new(key, KeyModifiers::NONE), Context::Feed),
                Some(expected)
            );
        }
        for (w, h) in [(80, 24), (60, 20), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            state.feed_scroll.set(0);
            terminal.draw(|f| super::super::draw(f, &state, 3)).unwrap();
            let top = rendered_text(&terminal);
            assert!(top.contains("50.00%"), "{w}x{h}: {top}");
            state.feed_scroll.set(u16::MAX);
            terminal.draw(|f| super::super::draw(f, &state, 3)).unwrap();
            let bottom = rendered_text(&terminal);
            assert!(bottom.contains("→ 16"), "{w}x{h}: {bottom}");
            assert!(bottom.contains("do not advance between events"));
            assert_eq!(
                state.feed.as_ref().unwrap().token_ids,
                (1..=16).collect::<Vec<_>>()
            );
        }
        state.ingest("progress_schema=2 prior_updates=0 updates_total=10");
        assert_eq!(state.feed_scroll.get(), 0);
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
