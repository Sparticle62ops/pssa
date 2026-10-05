//! Read-only chat telemetry: copied retrieval weights, never a second retrieval.
//! Only the latest full snapshot is retained; history stores 64 token summaries.
use super::{AMBER, BRIGHT_RED, SECOND_ACCENT, accent, charts::bar, panel, panel_area};
use crate::{dataset::Tokenizer, pssa::PSSALayerV2};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};
use std::{collections::VecDeque, sync::Arc};

const HISTORY_LIMIT: usize = 64;
const RANKED_LIMIT: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerStrengths {
    pub capacity: usize,
    // Exactly inf_mem_weights[..memory.count], in original slot order.
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SlotStrength {
    layer: usize,
    slot: usize,
    weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MemorySnapshot {
    pub generated_tokens: usize,
    pub query_token_id: usize,
    pub generated_token_id: usize,
    pub query_token: String,
    pub generated_token: String,
    pub loops: usize,
    pub layers: Vec<LayerStrengths>,
    strongest: Vec<SlotStrength>,
}

impl MemorySnapshot {
    pub(super) fn capture(
        generated_tokens: usize,
        query_token_id: usize,
        generated_token_id: usize,
        tokenizer: &Tokenizer,
        model: &PSSALayerV2,
    ) -> Self {
        let layers: Vec<_> = std::iter::once(&model.block)
            .chain(&model.extra_blocks)
            .map(|block| LayerStrengths {
                capacity: block.memory.capacity,
                weights: block.inf_mem_weights[..block.memory.count].to_vec(),
            })
            .collect();
        // Keep a small ranked display index. All raw occupied-slot weights remain
        // available in `layers`; no normalization or aggregation changes them.
        let mut strongest: Vec<SlotStrength> = Vec::with_capacity(RANKED_LIMIT);
        for (layer, values) in layers.iter().enumerate() {
            for (slot, &weight) in values.weights.iter().enumerate() {
                let rank = strongest.partition_point(|v| v.weight.total_cmp(&weight).is_ge());
                if rank < RANKED_LIMIT {
                    if strongest.len() == RANKED_LIMIT {
                        strongest.pop();
                    }
                    strongest.insert(
                        rank,
                        SlotStrength {
                            layer,
                            slot,
                            weight,
                        },
                    );
                }
            }
        }
        Self {
            generated_tokens,
            query_token_id,
            generated_token_id,
            query_token: token_label(tokenizer, query_token_id),
            generated_token: token_label(tokenizer, generated_token_id),
            loops: model.loops(),
            layers,
            strongest,
        }
    }

    fn occupancy(&self) -> (usize, usize) {
        self.layers.iter().fold((0, 0), |(used, total), layer| {
            (used + layer.weights.len(), total + layer.capacity)
        })
    }
}

fn visible(text: &str, limit: usize) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(limit)
        .collect()
}

fn token_label(tokenizer: &Tokenizer, id: usize) -> String {
    if let Some(bytes) = tokenizer.token_bytes(id) {
        visible(&String::from_utf8_lossy(bytes), 32)
    } else {
        visible(
            tokenizer.id_to_token.get(&id).map_or("?", String::as_str),
            32,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TokenSummary {
    number: usize,
    token_id: usize,
    token: String,
    strongest: Option<SlotStrength>,
}

#[derive(Clone, Default)]
pub(super) struct MemoryView {
    model: String,
    latest: Option<Arc<MemorySnapshot>>,
    history: VecDeque<TokenSummary>,
}

impl MemoryView {
    pub(super) fn new(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            ..Self::default()
        }
    }

    pub(super) fn record(&mut self, snapshot: MemorySnapshot) {
        if self.history.len() == HISTORY_LIMIT {
            self.history.pop_front();
        }
        self.history.push_back(TokenSummary {
            number: snapshot.generated_tokens,
            token_id: snapshot.generated_token_id,
            token: snapshot.generated_token.clone(),
            strongest: snapshot.strongest.first().copied(),
        });
        self.latest = Some(Arc::new(snapshot));
    }

    pub(super) fn latest(&self) -> Option<&MemorySnapshot> {
        self.latest.as_deref()
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        if area.is_empty() {
            return;
        }
        let area = panel_area(f, area);
        let block = panel(" live memory / read-only ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.is_empty() {
            return;
        }
        let Some(snapshot) = self.latest() else {
            let reason = if self.model.is_empty() {
                "No model selected."
            } else if std::path::Path::new(&self.model).extension().is_some_and(|e| e.eq_ignore_ascii_case("trfm")) {
                "Transformer: no PSSA memory slots."
            } else {
                "No token observations yet."
            };
            f.render_widget(
                Paragraph::new(vec![
                    Line::styled(reason, accent()),
                    Line::from("Open inference, /model PATH, then send a message."),
                    Line::from("Real PSSA retrieval strengths appear here per generated token."),
                    Line::from("No synthetic slots. No checkpoint or model writes."),
                ])
                .wrap(Wrap { trim: false }),
                inner,
            );
            return;
        };
        let (used, capacity) = snapshot.occupancy();
        if inner.width < 34 || inner.height < 10 {
            let mut lines = vec![
                Line::styled(
                    format!(
                        "token #{} / {used}/{capacity} slots",
                        snapshot.generated_tokens
                    ),
                    accent(),
                ),
                Line::from(format!(
                    "out {}: {}",
                    snapshot.generated_token_id, snapshot.generated_token
                )),
                Line::from("strength 0..1 / slots used/capacity"),
            ];
            if used == 0 {
                lines.push(Line::from("No occupied slots."));
            } else {
                lines.extend(
                    snapshot
                        .strongest
                        .iter()
                        .take(inner.height.saturating_sub(4) as usize)
                        .map(|slot| strength_line(*slot, inner.width)),
                );
            }
            lines.push(Line::styled(
                "read-only / latest retrieval",
                Style::new().fg(SECOND_ACCENT),
            ));
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        let parts = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(2),
        ])
        .split(inner);
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    format!(
                        "TOKEN #{}   {used}/{capacity} slots used/capacity   {} layer(s)",
                        snapshot.generated_tokens,
                        snapshot.layers.len()
                    ),
                    accent(),
                ),
                Line::from(format!(
                    "predicted [{}] {}",
                    snapshot.generated_token_id, snapshot.generated_token
                )),
                Line::styled(
                    format!(
                        "query/input [{}] {}",
                        snapshot.query_token_id, snapshot.query_token
                    ),
                    Style::new().fg(SECOND_ACCENT),
                ),
            ]),
            parts[0],
        );
        let wide = area.width >= 80;
        let body = if wide {
            Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(parts[1])
        } else {
            Layout::vertical([Constraint::Min(4), Constraint::Length(5)]).split(parts[1])
        };
        self.draw_strengths(f, body[0], snapshot);
        self.draw_history(f, body[1]);
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    format!(
                        "Retrieval strength 0..1 (0..100%); ticks 0 / 50 / 100; final loop pass {}/{}.",
                        snapshot.loops, snapshot.loops
                    ),
                    Style::new().fg(SECOND_ACCENT),
                ),
                Line::from(
                    "Checkpoint bank is read-only in chat; no token/slot provenance stored.",
                ),
            ]),
            parts[2],
        );
    }

    fn draw_strengths(&self, f: &mut Frame, area: Rect, snapshot: &MemorySnapshot) {
        // Nested panels have no shadow: their rects abut the enclosing border.
        let block = panel(" strongest slots / per layer ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.is_empty() {
            return;
        }
        let (used, _) = snapshot.occupancy();
        if used == 0 {
            f.render_widget(Paragraph::new("No occupied slots.\nThis checkpoint has an empty memory bank; chat does not insert entries.")
                .wrap(Wrap { trim: false }), inner);
            return;
        }
        let shown = (inner.height.saturating_sub(1) as usize).min(snapshot.strongest.len());
        let mut lines: Vec<_> = snapshot.strongest[..shown]
            .iter()
            .map(|slot| strength_line(*slot, inner.width))
            .collect();
        lines.push(Line::styled(
            format!("{shown}/{used} slots shown; L=layer, S=slot (0-based)"),
            Style::new().fg(SECOND_ACCENT),
        ));
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_history(&self, f: &mut Frame, area: Rect) {
        // Like the strengths panel, keep all painting inside this nested rect.
        let block = panel(" recent tokens / strongest slot ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let lines: Vec<_> = self
            .history
            .iter()
            .rev()
            .take(inner.height as usize)
            .map(|token| {
                let slot = token.strongest.map_or_else(
                    || "no slots".into(),
                    |slot| {
                        format!(
                            "L{} S{} {:.1}%",
                            slot.layer + 1,
                            slot.slot,
                            slot.weight * 100.0
                        )
                    },
                );
                Line::from(format!(
                    "#{} [{}] {slot} {}",
                    token.number, token.token_id, token.token
                ))
            })
            .collect();
        f.render_widget(Paragraph::new(lines), inner);
    }
}

fn strength_line(slot: SlotStrength, width: u16) -> Line<'static> {
    let label = format!("L{} S{} ", slot.layer + 1, slot.slot);
    let valid = slot.weight.is_finite() && (0.0..=1.0).contains(&slot.weight);
    let value = if valid {
        format!(" {:5.1}%", slot.weight * 100.0)
    } else {
        " invalid".into()
    };
    let bar_width = usize::from(width)
        .saturating_sub(label.len() + value.len())
        .min(40);
    let bar = strength_bar(slot.weight, bar_width);
    Line::from(vec![
        Span::styled(label, Style::new().fg(SECOND_ACCENT)),
        Span::styled(
            bar,
            if valid {
                accent()
            } else {
                Style::new().fg(BRIGHT_RED)
            },
        ),
        Span::styled(
            value,
            if valid {
                accent()
            } else {
                Style::new().fg(AMBER)
            },
        ),
    ])
}

fn strength_bar(weight: f32, width: usize) -> String {
    // Out-of-range telemetry is not a real strength; retain the invalid label
    // without drawing a plausible-looking clamped value.
    let fraction = if weight.is_finite() && (0.0..=1.0).contains(&weight) {
        f64::from(weight)
    } else {
        f64::NAN
    };
    bar(fraction, width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inference::{InferenceConfig, PSSAInferenceEngine},
        pssa::PSSAConfigV2,
    };
    use ratatui::{Terminal, backend::TestBackend};

    fn fixture() -> (Tokenizer, PSSALayerV2) {
        let tokenizer =
            Tokenizer::from_vocabulary(&["<unk>".into(), "hello".into(), "world".into()]).unwrap();
        let mut model = PSSALayerV2::new_with_depth_and_loops(
            PSSAConfigV2 {
                d_vocab: 3,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 3,
                chunk_len: 2,
                ..Default::default()
            },
            7,
            2,
            2,
        );
        model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
        for block in std::iter::once(&mut model.block).chain(&mut model.extra_blocks) {
            block.memory.insert(&[0.1, 0.2], &[0.4, 0.3, -0.2, 0.1]);
            block.memory.insert(&[-0.3, 0.4], &[0.7, -0.4, 0.3, 0.6]);
            // Controlled render fixture: captured bytes are copied, never recomputed.
            block.inf_mem_weights.copy_from_slice(&[0.25, 0.75, 999.0]);
        }
        (tokenizer, model)
    }

    fn view() -> MemoryView {
        let (tokenizer, model) = fixture();
        let mut view = MemoryView::new("path with spaces/model.pssa");
        for count in 1..=4 {
            view.record(MemorySnapshot::capture(count, 2, 1, &tokenizer, &model));
        }
        view
    }

    fn text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn snapshots_copy_actual_occupied_weights_bit_for_bit_without_mutating_banks() {
        let (tokenizer, mut model) = fixture();
        let banks: Vec<_> = std::iter::once(&model.block)
            .chain(&model.extra_blocks)
            .map(|block| block.memory.clone())
            .collect();
        let mut observed = Vec::new();
        let cfg = InferenceConfig {
            temperature: 0.0,
            max_new_tokens: 4,
            ..Default::default()
        };
        PSSAInferenceEngine::new(&mut model, &tokenizer)
            .try_generate_chat_turn_observed(
                "hello",
                &cfg,
                |_, _| {},
                || false,
                |count, query, selected, model| {
                    let before: Vec<_> = std::iter::once(&model.block)
                        .chain(&model.extra_blocks)
                        .map(|b| {
                            b.inf_mem_weights
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let snapshot =
                        MemorySnapshot::capture(count, query, selected, &tokenizer, model);
                    assert_eq!(snapshot.loops, 2);
                    assert_eq!(snapshot.layers.len(), 2);
                    for ((layer, block), bits) in snapshot
                        .layers
                        .iter()
                        .zip(std::iter::once(&model.block).chain(&model.extra_blocks))
                        .zip(&before)
                    {
                        assert_eq!(layer.capacity, 3);
                        assert_eq!(layer.weights.len(), 2);
                        assert_eq!(
                            layer
                                .weights
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            bits[..2]
                        );
                        assert_eq!(
                            block
                                .inf_mem_weights
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            *bits
                        );
                    }
                    observed.push(snapshot);
                },
            )
            .unwrap();
        assert_eq!(observed.len(), 4);
        for (block, bank) in std::iter::once(&model.block)
            .chain(&model.extra_blocks)
            .zip(&banks)
        {
            assert_eq!(&block.memory, bank);
        }
        // Snapshots are owned; later scratch writes cannot rewrite past telemetry.
        let original = observed[0].clone();
        model.block.inf_mem_weights.fill(0.0);
        assert_eq!(observed[0], original);
    }

    #[test]
    fn history_is_bounded_and_latest_owns_all_raw_slot_values() {
        let (tokenizer, model) = fixture();
        let mut view = MemoryView::new("model.pssa");
        for count in 1..=HISTORY_LIMIT + 9 {
            view.record(MemorySnapshot::capture(count, 2, 1, &tokenizer, &model));
        }
        assert_eq!(view.history.len(), HISTORY_LIMIT);
        assert_eq!(view.history.front().unwrap().number, 10);
        assert_eq!(view.history.back().unwrap().number, HISTORY_LIMIT + 9);
        let latest = view.latest().unwrap();
        assert_eq!(latest.layers[0].weights, [0.25, 0.75]);
        assert_eq!(latest.layers[1].weights, [0.25, 0.75]);
        assert_eq!(latest.strongest.len(), 4);
        assert_eq!(latest.strongest[0].slot, 1);
        assert_eq!(latest.generated_tokens, HISTORY_LIMIT + 9);
        assert!(Arc::ptr_eq(
            view.latest.as_ref().unwrap(),
            view.clone().latest.as_ref().unwrap()
        ));
    }

    #[test]
    fn live_memory_renders_real_strength_bars_wide_and_below_eighty_columns() {
        let view = view();
        for (width, height) in [(120, 40), (120, 32), (80, 24), (79, 24), (52, 24), (40, 16)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| view.draw(f, f.area())).unwrap();
            let rendered = text(&terminal);
            for expected in [
                "TOKEN #4",
                "4/6 slots",
                "predicted [1] hello",
                "L1 S1",
                "75.0%",
                "⣿",
            ] {
                assert!(
                    rendered.contains(expected),
                    "{width}x{height}: missing {expected}: {rendered}"
                );
            }
            if height >= 24 {
                assert!(rendered.contains("#4 [1]"), "history at {width}");
            }
            assert!(
                !rendered
                    .chars()
                    .any(|c| ('\u{2580}'..='\u{259f}').contains(&c))
            );
            if width >= 80 {
                for label in [
                    "used/capacity",
                    "Retrieval strength 0..1 (0..100%)",
                    "ticks 0 / 50 / 100",
                    "final loop pass 2/2",
                ] {
                    assert!(
                        rendered.contains(label),
                        "{width}x{height}: missing {label}"
                    );
                }
                assert!(terminal.backend().buffer().content().iter().any(|cell| {
                    cell.fg == super::super::NORMAL_GREEN
                        && cell
                            .symbol()
                            .chars()
                            .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                }));
            }
            assert!(
                !rendered.contains("999"),
                "unused buffer tail is not a slot"
            );
            let buffer = terminal.backend().buffer();
            assert_eq!(buffer[(0, 0)].symbol(), "┌");
            assert_eq!(buffer[(width - 1, 0)].symbol(), "┐");
            assert_eq!(buffer[(0, height - 1)].symbol(), "└");
            assert_eq!(buffer[(width - 1, height - 1)].symbol(), "┘");
        }
    }

    #[test]
    fn nested_panels_keep_the_enclosing_right_border_closed() {
        let view = view();
        for (width, height) in [(120, 40), (80, 24)] {
            for inset in [0, 1] {
                let area = Rect::new(inset, inset, width - 2 * inset, height - 2 * inset);
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| view.draw(f, area)).unwrap();
                let buffer = terminal.backend().buffer();
                for y in area.y..area.bottom() {
                    let expected = if y == area.y {
                        "┐"
                    } else if y == area.bottom() - 1 {
                        "┘"
                    } else {
                        "│"
                    };
                    assert_eq!(
                        buffer[(area.right() - 1, y)].symbol(),
                        expected,
                        "{width}x{height}, inset {inset}, right border row {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_selected_empty_bank_and_tiny_views_are_honest() {
        let mut selected = MemoryView::new("empty.pssa");
        let (tokenizer, mut model) = fixture();
        model.memory.count = 0;
        model.extra_blocks[0].memory.count = 0;
        selected.record(MemorySnapshot::capture(1, 2, 1, &tokenizer, &model));
        for (view, expected) in [
            (MemoryView::default(), "No model selected."),
            (MemoryView::new("model.pssa"), "No token observations yet."),
            (selected, "No occupied slots."),
        ] {
            for (width, height) in [(120, 28), (79, 24), (40, 16), (25, 8), (1, 1), (0, 0)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| view.draw(f, f.area())).unwrap();
                if width >= 40 {
                    assert!(text(&terminal).contains(expected), "{width}: {expected}");
                }
            }
        }
    }

    #[test]
    fn bars_have_fixed_width_and_invalid_values_do_not_invent_strength() {
        assert_eq!(strength_bar(0.0, 4), "····");
        assert_eq!(strength_bar(0.75, 4), "⣿⣿⣿·");
        assert_eq!(strength_bar(0.5 / 8.0, 2), "⡀·");
        assert_eq!(strength_bar(1.0, 4), "⣿⣿⣿⣿");
        assert_eq!(strength_bar(f32::NAN, 4), "····");
        for invalid in [f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert_eq!(strength_bar(invalid, 4), "····");
        }
        for weight in [0.0, 0.0625, 0.25, 0.75, 1.0, f32::NAN] {
            for width in [0, 1, 4, 40] {
                assert_eq!(strength_bar(weight, width).chars().count(), width);
            }
        }
        let line = strength_line(
            SlotStrength {
                layer: 0,
                slot: 0,
                weight: f32::NAN,
            },
            30,
        );
        assert!(
            line.spans
                .iter()
                .any(|span| span.content.contains("invalid"))
        );
    }
}
