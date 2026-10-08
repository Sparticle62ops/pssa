//! Equations are editable content, not another implementation of the model.
use super::{RunState, SECOND_ACCENT, accent, panel, panel_area, parse_field, parse_kv};
use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::Line,
    widgets::{Paragraph, Wrap},
};

const EQUATIONS: &str = include_str!("../../assets/math.md");

#[derive(Default)]
pub(super) struct Values {
    model: Option<String>,
    latent: Option<u64>,
    state: Option<u64>,
    vocab: Option<u64>,
    depth: Option<u64>,
    loops: Option<u64>,
    key: Option<u64>,
    slots: Option<u64>,
    parameters: Option<u64>,
}
impl Values {
    pub(super) fn ingest(&mut self, line: &str) {
        if line.starts_with("model=") {
            self.model = parse_kv(line, "model=");
        }
        for (label, field) in [
            ("parameters=", &mut self.parameters),
            ("vocab=", &mut self.vocab),
            ("depth=", &mut self.depth),
            ("loops=", &mut self.loops),
            ("d_latent=", &mut self.latent),
        ] {
            if let Some(n) = parse_kv(line, label) {
                *field = Some(n);
            }
        }
        if let Some(width) = parse_field(line, "width") {
            self.latent = parse_kv(&width, "latent ").or(self.latent);
            self.state = parse_kv(&width, "state ").or(self.state);
            self.depth = parse_kv(&width, "depth ").or(self.depth);
        }
        if let Some(memory) = parse_field(line, "memory") {
            self.slots = memory
                .split_whitespace()
                .next()
                .and_then(|s| s.replace(',', "").parse().ok())
                .or(self.slots);
            self.key = parse_kv(&memory, "key width ").or(self.key);
        }
        if let Some(vocab) = parse_field(line, "vocabulary") {
            self.vocab = vocab.replace(',', "").parse().ok().or(self.vocab);
        }
    }

    fn dense_macs(&self) -> Option<u128> {
        let (d, s, k, v, depth, loops) = (
            self.latent? as u128,
            self.state? as u128,
            self.key? as u128,
            self.vocab? as u128,
            self.depth? as u128,
            self.loops? as u128,
        );
        // Wdelta, Wgate, Wproj: 3d²; MLP: 4d²; WB/WC: 2ds;
        // Wqx/Wqh: 2dk; rank-16 adapter: 32d; single final readout: vd.
        let block = 7u128
            .checked_mul(d)?
            .checked_mul(d)?
            .checked_add(2u128.checked_mul(d)?.checked_mul(s)?)?
            .checked_add(2u128.checked_mul(d)?.checked_mul(k)?)?
            .checked_add(32u128.checked_mul(d)?)?;
        depth
            .checked_mul(loops)?
            .checked_mul(block)?
            .checked_add(v.checked_mul(d)?)
    }
}

#[derive(Default)]
pub(super) struct Math {
    pub scroll: std::cell::Cell<u16>,
}
impl Math {
    pub(super) fn draw(&self, f: &mut Frame, area: Rect, state: &RunState) {
        let area = panel_area(f, area);
        let v = &state.math;
        let n = |value: Option<u64>| value.map_or_else(|| "n/a".into(), |n| n.to_string());
        let mut lines = vec![
            Line::styled("PSSA / executable equations", accent()),
            Line::from(format!(
                "LIVE latent {} / state {} / vocab {}",
                n(v.latent),
                n(v.state),
                n(v.vocab)
            )),
            Line::from(format!(
                "Depth {} independent blocks / loops {} shared passes",
                n(v.depth),
                n(v.loops)
            )),
            Line::from(format!(
                "Memory {} slots/block / key {} / occupied {}",
                n(v.slots),
                n(v.key),
                n(state.memory_used)
            )),
            Line::from(format!(
                "Trainable parameters {} / lr {}",
                n(v.parameters),
                state
                    .learning_rate
                    .map_or_else(|| "n/a".into(), |lr| format!("{lr:.6e}"))
            )),
            Line::from(format!(
                "Dense MAC/token {} (estimate, forward only)",
                v.dense_macs()
                    .map_or_else(|| "n/a".into(), |n| n.to_string())
            )),
            Line::from("Missing live fields stay n/a; open a training log/run."),
            Line::styled(
                "Up/Down or PgUp/PgDn scroll / Home top / End bottom",
                accent(),
            ),
        ];
        if v.model.as_deref() == Some("transformer") {
            lines = vec![
                Line::styled("Transformer run / logged configuration", accent()),
                Line::from(format!(
                    "Width: {}",
                    state.width.as_deref().unwrap_or("unrecorded")
                )),
                Line::from(format!(
                    "Vocabulary {} / trainable parameters {}",
                    n(v.vocab),
                    n(v.parameters)
                )),
                Line::from(format!(
                    "Learning rate {}",
                    state
                        .learning_rate
                        .map_or("unrecorded".into(), |lr| format!("{lr:.6e}"))
                )),
                Line::from("PSSA memory, recurrence and dense-MAC estimate: not applicable."),
                Line::from(
                    "Equations below are PSSA reference documentation, NOT this model's telemetry.",
                ),
            ];
        }
        for line in EQUATIONS.lines() {
            if let Some(title) = line.strip_prefix("# ") {
                lines.push(Line::styled(format!("▌ {title}"), accent()));
            } else if line.starts_with("Source:") {
                lines.push(Line::styled(line, Style::new().fg(SECOND_ACCENT)));
                lines.push(Line::styled(
                    "╌────────────────────",
                    Style::new().fg(SECOND_ACCENT),
                ));
            } else {
                lines.push(Line::from(line));
            }
        }
        let block = panel(" math / read-only ");
        let inner = block.inner(area);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let max = paragraph
            .line_count(inner.width)
            .saturating_sub(inner.height as usize)
            .min(u16::MAX as usize) as u16;
        self.scroll.set(self.scroll.get().min(max));
        f.render_widget(paragraph.scroll((self.scroll.get(), 0)).block(block), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn live_values_survive_log_rollover_and_cost_matches_dense_shapes() {
        let mut state = RunState::default();
        for line in [
            "model=pssa parameters=123456 vocab=256 depth=2 loops=3",
            "  width  latent 8 / state 4 / depth 2",
            "  memory  10 slots, key width 2",
        ] {
            state.ingest(line);
        }
        assert_eq!(
            state.math.dense_macs(),
            Some(6 * (7 * 64 + 2 * 8 * 4 + 2 * 8 * 2 + 32 * 8) + 256 * 8)
        );
        for _ in 0..450 {
            state.ingest("progress");
        }
        assert_eq!(state.math.parameters, Some(123456));
        for (w, h) in [(120, 38), (79, 24), (40, 16), (1, 1)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            let math = Math::default();
            terminal.draw(|f| math.draw(f, f.area(), &state)).unwrap();
            if w >= 40 {
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("LIVE latent 8"));
                assert!(text.contains("123456"));
                math.scroll.set(u16::MAX);
                terminal.draw(|f| math.draw(f, f.area(), &state)).unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("1964-1994"));
            }
        }
    }

    #[test]
    fn empty_math_shows_na_and_equations_not_fabricated_live_values() {
        let mut t = Terminal::new(TestBackend::new(79, 24)).unwrap();
        t.draw(|f| Math::default().draw(f, f.area(), &RunState::default()))
            .unwrap();
        let text: String = t
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("LIVE latent n/a"));
        assert!(text.contains("softplus"));
    }
}
