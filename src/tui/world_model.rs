//! Separate measured boxes-world telemetry; never feeds language CE charts.
use super::{RunState, accent, charts, panel, panel_area, parse_kv};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::time::Instant;

#[derive(Default)]
pub(super) struct Monitor {
    pub status: String,
    pub epoch: usize,
    pub epochs: usize,
    config: String,
    objectives: [Vec<f64>; 2],
    results: [Option<String>; 2],
}

impl Monitor {
    pub fn header(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(format!(
                "boxes-world / {} / epoch {}/{}",
                self.status, self.epoch, self.epochs
            )),
            Line::from("CPU / no checkpoint / not language CE"),
            Line::from(format!(
                "stochastic objective {} / epoch →",
                charts::sparkline(&self.objectives[0], 12)
            )),
        ]
    }

    pub fn draw(&self, f: &mut Frame, area: Rect) {
        let mut lines = self.header();
        lines.push(Line::from(self.config.clone()));
        for (index, name) in ["stochastic", "autoregressive"].iter().enumerate() {
            let objective = self.objectives[index]
                .last()
                .map_or("unreported".into(), |x| format!("{x:.4}"));
            lines.push(Line::from(format!(
                "{name} objective {objective} {}",
                charts::sparkline(&self.objectives[index], 12)
            )));
            lines.push(Line::from(
                self.results[index]
                    .clone()
                    .unwrap_or_else(|| "held-out evaluation pending".into()),
            ));
        }
        lines.extend([
            Line::from("NLL = nats/field; open-loop sees initial frame + actions only."),
            Line::from("Same data/updates/widths; NOT parameter- or time-matched."),
            Line::from("Objectives differ; not a full Dreamer agent or planning result."),
            Line::from("No actor/critic, diffusion or memory writes. Full results: train.log."),
        ]);
        let area = panel_area(f, area);
        f.render_widget(
            Paragraph::new(lines)
                .style(accent())
                .wrap(Wrap { trim: false })
                .block(panel(" boxes-world / measured comparison ")),
            area,
        );
    }
}

/// Claim only explicit world events (and their attached human-readable report).
/// A subsequent regular training banner restores the ordinary monitor.
pub(super) fn ingest(state: &mut RunState, line: &str) -> bool {
    if line.contains("progress_schema=") || line.starts_with("model=") {
        state.world_model = None;
        return false;
    }
    let event = line
        .strip_prefix("world_model=")
        .and_then(|s| s.split_whitespace().next());
    if event == Some("start") {
        // A new experiment has no text checkpoint or language-loss history.
        let chain_dir = state.chain_dir.clone();
        let comparison_series = std::mem::take(&mut state.comparison_series);
        let comparison_label = state.comparison_label.take();
        *state = RunState {
            chain_dir,
            comparison_series,
            comparison_label,
            ..RunState::default()
        };
        state.training_active = true;
        state.run_started_at = Some(Instant::now());
        state.last_progress_at = state.run_started_at;
        state.world_model = Some(Monitor {
            status: "running".into(),
            epochs: parse_kv::<usize>(line, "epochs=").unwrap_or(0).min(12),
            config: format!(
                "seed {} / side {} / train {} / held-out {}",
                value(line, "seed="),
                value(line, "side="),
                value(line, "train_episodes="),
                value(line, "heldout_episodes=")
            ),
            ..Monitor::default()
        });
    }
    let Some(monitor) = &mut state.world_model else {
        return false;
    };
    state.raw_lines.push(line.to_owned());
    if state.raw_lines.len() > 400 {
        state.raw_lines.remove(0);
    }
    match event {
        Some("epoch") => {
            let epoch = parse_kv::<usize>(line, "epoch=").unwrap_or(0);
            if epoch > monitor.epoch && epoch <= monitor.epochs {
                monitor.epoch = epoch;
                state.last_progress_at = Some(Instant::now());
                for (index, key) in ["stochastic_objective=", "autoregressive_objective="]
                    .iter()
                    .enumerate()
                {
                    if let Some(x) =
                        parse_kv::<f64>(line, key).filter(|x| x.is_finite() && *x >= 0.0)
                    {
                        monitor.objectives[index].push(x);
                    }
                }
            }
        }
        Some("result") => {
            let variant = parse_kv::<String>(line, "variant=");
            let index = match variant.as_deref() {
                Some("stochastic") => Some(0),
                Some("autoregressive") => Some(1),
                _ => None,
            };
            if let Some(index) = index {
                monitor.results[index] = Some(format!(
                    "next NLL {} / open-loop NLL {} / field {} / rollout field {} / params {} / ms {}",
                    metric(line, "one_step_nll="),
                    metric(line, "rollout_nll="),
                    metric(line, "field_accuracy="),
                    metric(line, "rollout_field_accuracy="),
                    value(line, "parameters="),
                    metric(line, "train_ms=")
                ));
            }
        }
        Some("complete") => {
            monitor.status = "complete".into();
            state.training_active = false;
        }
        Some("failed") => {
            monitor.status = "failed".into();
            state.training_active = false;
            state.record_problem("world-model failed; no checkpoint written");
        }
        _ => {}
    }
    true
}

fn value(line: &str, key: &str) -> String {
    parse_kv::<u64>(line, key).map_or("?".into(), |x| x.to_string())
}
fn metric(line: &str, key: &str) -> String {
    parse_kv::<f64>(line, key)
        .filter(|x| x.is_finite() && *x >= 0.0)
        .map_or("?".into(), |x| format!("{x:.3}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    const START: &str = "world_model=start seed=73 side=4 epochs=3 train_episodes=32 heldout_episodes=12 checkpoint=none";
    #[test]
    fn world_events_do_not_pollute_text_loss_or_claim_a_checkpoint() {
        let mut state = RunState {
            live_loss: Some(5.0),
            loss_series: vec![5.0],
            checkpoint_target: Some("old.pssa".into()),
            ..RunState::default()
        };
        state.ingest(START);
        state.ingest("world_model=epoch epoch=1 epochs=3 stochastic_objective=2.9 autoregressive_objective=3.0");
        state.ingest("world_model=result variant=stochastic parameters=6578 train_ms=50 one_step_nll=2.4 rollout_nll=2.5 field_accuracy=0.1 rollout_field_accuracy=0.2");
        state.ingest("world_model=complete checkpoint=none");
        assert!(!state.training_active);
        assert!(
            state.live_loss.is_none()
                && state.loss_series.is_empty()
                && state.metric_series.is_empty()
        );
        assert!(state.checkpoint_target.is_none() && !state.checkpoint_saved_current);
        assert!(state.checkpoint_context().contains("no checkpoint"));
        let m = state.world_model.as_ref().unwrap();
        assert_eq!(m.status, "complete");
        assert_eq!(m.objectives[0], vec![2.9]);
        assert!(m.results[0].as_ref().unwrap().contains("next NLL 2.400"));
        assert!(
            m.results[0]
                .as_ref()
                .unwrap()
                .contains("field 0.100 / rollout field 0.200")
        );
        state.ingest("progress_schema=2 prior_updates=0");
        assert!(state.world_model.is_none() && state.training_active);
    }

    #[test]
    fn malformed_duplicate_events_are_bounded_and_failures_are_visible() {
        let mut state = RunState::default();
        state.ingest(START);
        for _ in 0..500 {
            state.ingest(
                "world_model=epoch epoch=1 stochastic_objective=NaN autoregressive_objective=inf",
            );
        }
        state.ingest(
            "world_model=epoch epoch=999 stochastic_objective=2 autoregressive_objective=2",
        );
        state.ingest("world_model=result variant=unknown one_step_nll=2");
        assert!(state.raw_lines.len() <= 400);
        let m = state.world_model.as_ref().unwrap();
        assert_eq!(m.epoch, 1);
        assert!(m.objectives.iter().all(Vec::is_empty));
        assert!(m.results.iter().all(Option::is_none));
        state.ingest("world_model=failed checkpoint=none");
        assert!(!state.training_active);
        assert!(
            state
                .problem
                .as_deref()
                .unwrap()
                .contains("world-model failed")
        );
    }

    #[test]
    fn measured_world_monitor_renders_on_normal_and_narrow_terminals() {
        let mut state = RunState::default();
        state.ingest(START);
        state.ingest(
            "world_model=epoch epoch=1 stochastic_objective=2.9 autoregressive_objective=3.0",
        );
        state.ingest(
            "world_model=epoch epoch=2 stochastic_objective=2.6 autoregressive_objective=2.5",
        );
        for (width, height) in [(80, 24), (44, 18), (120, 32), (16, 6)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|f| super::super::draw_monitor(f, f.area(), &state))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 44 {
                assert!(text.contains("boxes-world"));
                assert!(text.contains("no checkpoint"));
                assert!(text.chars().any(|c| ('\u{2800}'..='\u{28ff}').contains(&c)));
            }
        }
    }
}
