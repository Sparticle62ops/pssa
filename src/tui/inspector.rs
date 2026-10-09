//! Read-only plastic memory telemetry. Never infer a write from a feed sample.
use super::{RunState, accent, panel, panel_area, parse_kv, process::clean};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant, SystemTime},
};

struct Slot {
    layer: usize,
    index: usize,
    fingerprint: u64,
    step: usize,
    confidence: f32,
}
struct Snapshot {
    path: PathBuf,
    used: usize,
    capacity: usize,
    slots: Vec<Slot>,
}
fn snapshot(path: PathBuf) -> Result<Snapshot, String> {
    let metadata = std::fs::metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > 64 * 1024 * 1024 {
        return Err("Background snapshot limit is 64 MiB per regular checkpoint; live occupancy remains available.".into());
    }
    let model = crate::checkpoint::load_checkpoint(&path)
        .map_err(|e| e.to_string())?
        .model;
    let mut result = Snapshot {
        path,
        used: 0,
        capacity: 0,
        slots: Vec::new(),
    };
    for (layer, block) in std::iter::once(&model.block)
        .chain(&model.extra_blocks)
        .enumerate()
    {
        let bank = &block.memory;
        result.used += bank.count;
        result.capacity += bank.capacity;
        for i in 0..bank.count.min(4096usize.saturating_sub(result.slots.len())) {
            let mut hash = 0xcbf29ce484222325u64;
            for x in bank.keys[i * bank.dim_key..(i + 1) * bank.dim_key]
                .iter()
                .chain(&bank.values[i * bank.dim_val..(i + 1) * bank.dim_val])
            {
                for byte in x.to_bits().to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                }
            }
            result.slots.push(Slot {
                layer,
                index: i,
                fingerprint: hash,
                step: bank.last_seen_step[i],
                confidence: bank.confidence[i],
            });
        }
    }
    Ok(result)
}
#[derive(Default)]
pub(super) struct Inspector {
    history: VecDeque<u64>,
    last_occupancy: Option<(u64, u64)>,
    events: VecDeque<String>,
    explicit_evictions: Option<u64>,
    evictions_at: Option<(u64, Instant)>,
    eviction_rate: Option<f64>,
    snapshot: Option<Snapshot>,
    pending: Option<mpsc::Receiver<Result<Snapshot, String>>>,
    requested: Option<(PathBuf, Option<SystemTime>)>,
    checked: Option<Instant>,
    snapshot_at: Option<Instant>,
    changed: usize,
    note: String,
    scroll: u16,
}
impl Inspector {
    pub fn ingest(&mut self, line: &str) {
        if line.contains("progress_schema=") {
            *self = Self::default();
        }
        if let Some(total) = parse_kv::<u64>(line, "memory_evictions=") {
            let now = Instant::now();
            self.eviction_rate = self
                .evictions_at
                .filter(|(before, _)| total >= *before)
                .map(|(before, at)| {
                    (total - before) as f64 / now.duration_since(at).as_secs_f64().max(0.001)
                });
            self.explicit_evictions = Some(total);
            self.evictions_at = Some((total, now));
        }
        if let Some(text) = super::parse_log_value(line, "memory_write_snippet") {
            self.events
                .push_front(clean(&text).chars().take(200).collect());
            self.events.truncate(32);
        }
    }
    pub fn poll(&mut self, state: &RunState, visible: bool) {
        if let Some(pair) = state.memory_used.zip(state.memory_capacity)
            && self.last_occupancy != Some(pair)
        {
            self.last_occupancy = Some(pair);
            self.history.push_back(pair.0);
            if self.history.len() > 120 {
                self.history.pop_front();
            }
        }
        if let Some(result) = self.pending.as_ref().and_then(|r| r.try_recv().ok()) {
            self.pending = None;
            match result {
                Ok(new) => {
                    self.changed = self.snapshot.as_ref().map_or(0, |old| {
                        let slots: std::collections::HashMap<_, _> = old
                            .slots
                            .iter()
                            .map(|s| ((s.layer, s.index), s.fingerprint))
                            .collect();
                        new.slots
                            .iter()
                            .filter(|s| {
                                slots
                                    .get(&(s.layer, s.index))
                                    .is_some_and(|hash| *hash != s.fingerprint)
                            })
                            .count()
                    });
                    self.note = if let Some(at) = self.snapshot_at {
                        format!(
                            "{} occupied slots changed / {:.1}s between snapshots (lower bound, not exact evictions)",
                            self.changed,
                            at.elapsed().as_secs_f64()
                        )
                    } else {
                        "First checkpoint snapshot; no overwrite interval yet.".into()
                    };
                    if self.last_occupancy.is_none() {
                        self.history.push_back(new.used as u64);
                        if self.history.len() > 120 {
                            self.history.pop_front();
                        }
                    }
                    self.snapshot = Some(new);
                    self.snapshot_at = Some(Instant::now());
                }
                Err(e) => {
                    self.note = format!("Checkpoint unavailable: {e}");
                    self.requested = None;
                }
            }
        }
        if !visible
            || self.pending.is_some()
            || self
                .checked
                .is_some_and(|t| t.elapsed() < Duration::from_secs(5))
        {
            return;
        }
        self.checked = Some(Instant::now());
        let Some(path) = state
            .last_checkpoint
            .as_ref()
            .or(state.selected_checkpoint.as_ref())
            .or(state.resumed_from.as_ref())
            .filter(|s| s.ends_with(".pssa"))
            .map(PathBuf::from)
        else {
            return;
        };
        let stamp = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if self.requested.as_ref() == Some(&(path.clone(), stamp)) {
            return;
        }
        self.requested = Some((path.clone(), stamp));
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.note = "Reading latest saved checkpoint in background…".into();
        std::thread::spawn(move || {
            let _ = tx.send(snapshot(path));
        });
    }
    pub fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') => {
                self.requested = None;
                self.checked = None;
            }
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(8),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(8),
            _ => {}
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect, state: &RunState) {
        let occupancy = self.last_occupancy.or_else(|| {
            self.snapshot
                .as_ref()
                .map(|s| (s.used as u64, s.capacity as u64))
        });
        let (used, capacity) = occupancy.unwrap_or((0, 0));
        let title = occupancy.map_or_else(
            || " plastic memory / occupancy unrecorded ".into(),
            |(used, capacity)| format!(" plastic memory / {used}/{capacity} slots "),
        );
        // Budget for eight actual history rows on the full-height shell, but
        // retain room for occupancy details and refresh/scroll hints when short.
        let chunks = Layout::vertical([
            Constraint::Length(if area.height >= 26 { 13 } else { 8 }),
            Constraint::Min(0),
        ])
        .split(area);
        let history: Vec<_> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v as f64))
            .collect();
        super::charts::draw(
            f,
            chunks[0],
            super::charts::Plot {
                title: &title,
                caption: "Used memory slots over recent observations",
                x: "sample",
                integer_x: true,
                y: "used slots",
                x_bounds: super::charts::domain(&history),
                y_bounds: [0.0, capacity.max(1) as f64],
            },
            &[super::charts::Series::line(
                "used slots",
                &history,
                super::NORMAL_GREEN,
            )],
        );
        let content = if area.width >= 80 {
            let columns =
                Layout::horizontal([Constraint::Percentage(35), Constraint::Percentage(65)])
                    .split(chunks[1]);
            let occupancy = RunState {
                memory_used: occupancy.map(|pair| pair.0),
                memory_capacity: occupancy.map(|pair| pair.1),
                ..Default::default()
            };
            super::draw_memory_graph(f, columns[0], &occupancy);
            columns[1]
        } else {
            chunks[1]
        };
        let mut lines = vec![
            Line::styled("MEMORY / store → protect → replace", accent()),
            Line::from(if capacity == 0 {
                if state
                    .checkpoint_target
                    .as_deref()
                    .or(state.last_checkpoint.as_deref())
                    .or(state.selected_checkpoint.as_deref())
                    .is_some_and(|path| path.ends_with(".trfm"))
                {
                    "Transformer runs have no PSSA plastic-memory bank.".into()
                } else {
                    format!("Occupancy unrecorded. {}", state.checkpoint_context())
                }
            } else {
                format!(
                    "Occupancy {:.1}% (live progress when available)",
                    used as f64 * 100.0 / capacity as f64
                )
            }),
            Line::from(match (self.explicit_evictions, self.eviction_rate) {
                (Some(total), Some(rate)) => {
                    format!("Reported evictions {total} / {rate:.2}/s observed (arrival-time rate)")
                }
                (Some(total), _) => format!("Reported evictions {total} / rate pending"),
                _ => "Exact eviction rate: not recorded by this trainer.".into(),
            }),
            Line::from(if self.note.is_empty() {
                state.checkpoint_context()
            } else {
                self.note.clone()
            }),
            Line::from("r refresh • PgUp/PgDn scroll • Tab tabs"),
            Line::from(""),
            Line::styled(
                "Confirmed write snippets (only explicit memory_write_snippet fields)",
                accent(),
            ),
        ];
        if self.events.is_empty() {
            lines.push(Line::from(
                "Not recorded. Checkpoints store vectors, not source token IDs/text.",
            ));
        }
        for event in &self.events {
            lines.push(Line::from(event.as_str()));
        }
        if let Some(feed) = &state.feed {
            lines.push(Line::from(format!(
                "Feed context only, NOT a confirmed write: {}",
                clean(&feed.snippet)
            )));
        }
        if let Some(snap) = &self.snapshot {
            lines.push(Line::from(format!(
                "Snapshot: {} / up to 4096 slots",
                clean(&snap.path.display().to_string())
            )));
            lines.push(Line::styled(
                "layer  slot  last-seen step  confidence",
                accent(),
            ));
            for s in &snap.slots {
                lines.push(Line::from(format!(
                    "{:>5} {:>5} {:>15} {:>11.3}",
                    s.layer + 1,
                    s.index,
                    s.step,
                    s.confidence
                )));
            }
        }
        let content_area = panel_area(f, content);
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.scroll, 0))
                .block(panel(" memory inspector / read-only ")),
            content_area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_snapshot_reads_real_slots_without_modifying_bytes() {
        let path = std::env::temp_dir().join(format!("pssa-inspector-{}.pssa", std::process::id()));
        let mut model = crate::pssa::PSSALayerV2::new(
            crate::pssa::PSSAConfigV2 {
                d_vocab: 3,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 2,
                chunk_len: 2,
                ..Default::default()
            },
            7,
        );
        model.vocabulary = vec!["<unk>".into(), "hello".into(), "world".into()];
        model
            .block
            .memory
            .insert(&[0.1, 0.2], &[1.0, 2.0, 3.0, 4.0]);
        model.block.memory.last_seen_step[0] = 12;
        model.step_counter = 12;
        crate::checkpoint::save_model(&model, &path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let snap = snapshot(path.clone()).unwrap();
        assert_eq!((snap.used, snap.capacity), (1, 2));
        assert_eq!(snap.slots.len(), 1);
        assert_eq!(snap.slots[0].step, 12);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn checkpoint_observation_history_retains_the_latest_after_its_bound() {
        let mut inspector = Inspector::default();
        for used in 0..125 {
            let (tx, rx) = mpsc::channel();
            tx.send(Ok(Snapshot {
                path: PathBuf::from("fixture.pssa"),
                used,
                capacity: 128,
                slots: Vec::new(),
            }))
            .unwrap();
            inspector.pending = Some(rx);
            inspector.poll(&RunState::default(), false);
        }
        assert_eq!(inspector.history.len(), 120);
        assert_eq!(inspector.history.front(), Some(&5));
        assert_eq!(inspector.history.back(), Some(&124));
    }

    #[test]
    fn occupancy_and_explicit_writes_are_distinct_and_bounded() {
        let mut inspector = Inspector::default();
        let mut state = RunState::default();
        state.ingest("memory_occupancy=3/8 feed_snippet=not-a-write");
        inspector.poll(&state, false);
        assert_eq!(inspector.last_occupancy, Some((3, 8)));
        assert!(inspector.events.is_empty());
        for _ in 0..40 {
            inspector.ingest("memory_evictions=2 memory_write_snippet=real%20write");
        }
        assert_eq!(inspector.events.len(), 32);
        assert_eq!(inspector.events[0], "real write");
    }
    #[test]
    fn checkpoint_only_transformer_memory_is_not_applicable_not_missing_occupancy() {
        let state = RunState {
            selected_checkpoint: Some("selected.trfm".into()),
            ..Default::default()
        };
        for (width, height) in [(120, 40), (80, 24)] {
            let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(
                width, height,
            ))
            .unwrap();
            terminal
                .draw(|f| Inspector::default().draw(f, f.area(), &state))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("no PSSA plastic-memory bank"), "{text}");
            assert!(!text.contains("Occupancy unrecorded"));
        }
    }

    #[test]
    fn occupancy_history_uses_braille_and_labels_at_both_sizes() {
        let mut inspector = Inspector::default();
        inspector.history = [2, 4, 3, 6].into_iter().collect();
        inspector.last_occupancy = Some((6, 8));
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal
                .draw(|f| inspector.draw(f, f.area(), &RunState::default()))
                .unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "plastic memory",
                &[
                    "used slots",
                    "sample",
                    "Used memory slots",
                    "6/8 slots",
                    "0",
                ],
            );
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "plastic memory",
                if h == 40 {
                    &["0", "2", "4", "6", "8"]
                } else {
                    &["0.0", "2.5", "5.0", "7.5"]
                },
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "plastic memory",
                if h == 40 { 8 } else { 3 },
            );
            terminal
                .draw(|f| {
                    inspector.draw(
                        f,
                        super::super::feature_area(f.area()),
                        &RunState::default(),
                    )
                })
                .unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "plastic memory",
                &["used slots", "sample", "Used memory slots"],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "plastic memory",
                if h == 40 { 8 } else { 3 },
            );
        }
    }

    #[test]
    fn test_backend_inspector_has_truthful_empty_and_narrow_states() {
        for (w, h) in [(120, 30), (79, 24), (24, 8), (1, 1)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal
                .draw(|f| Inspector::default().draw(f, f.area(), &RunState::default()))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            if w >= 79 {
                assert!(text.contains("not recorded"));
                assert!(text.contains("read-only"));
            }
        }
    }
}
