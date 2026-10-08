//! Presentation-only, transition-triggered alerts; never emit logs or credentials.
use super::{BRIGHT_RED, HealthLevel, NORMAL_GREEN, RunState};
use ratatui::{
    layout::Rect,
    style::Style,
    widgets::{Clear, Paragraph},
};
#[derive(Default)]
pub(super) struct Alerts {
    pub bell: bool,
    message: Option<(bool, String)>,
    active: bool,
    last_problem: Option<String>,
}
impl Alerts {
    pub fn notify(&mut self, ok: bool, message: impl Into<String>) {
        self.message = Some((
            ok,
            super::process::clean(&message.into())
                .chars()
                .take(512)
                .collect(),
        ));
        self.bell = true;
    }
    pub fn ingest(&mut self, line: &str) {
        if line.contains("progress_schema=") {
            self.active = true;
            self.last_problem = None;
            self.message = None;
        }
        if line.contains("tokens_per_second=") {
            self.active = true;
        }
        if line.trim_start().starts_with("error:") || line.trim_start().starts_with("Error:") {
            self.notify(false, "Training error; inspect the monitor log.");
            self.active = false;
        } else if line.contains("training_seconds=") && self.active {
            self.notify(
                true,
                "Training computation finished; waiting for checkpoint save confirmation.",
            );
            self.active = false;
        }
        if line.trim_start().starts_with("saved_checkpoint=") {
            self.notify(true, "Checkpoint saved (confirmed by trainer event).");
        }
    }
    pub fn observe(&mut self, state: &RunState) {
        if state.training_active {
            self.active = true;
        }
        let health = state.health_status();
        if health.level == HealthLevel::Problem {
            if health.reason != self.last_problem {
                self.notify(
                    false,
                    format!(
                        "Training alert: {}",
                        health.reason.as_deref().unwrap_or("error")
                    ),
                );
            }
            self.last_problem = health.reason;
        } else {
            self.last_problem = None;
        }
    }
    pub fn eof(&mut self, state: &RunState) {
        if self.active {
            // The final progress update precedes checkpoint writes. Only a
            // completion summary proves the producer finished successfully.
            let done = state.training_seconds.is_some() && state.problem.is_none();
            self.notify(done,if done {"Training stream finished."} else {"Training stream ended without a completion summary; check the producer exit status."});
            self.active = false;
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame) {
        if let Some((ok, message)) = &self.message {
            let a = f.area();
            if a.is_empty() {
                return;
            }
            let line = Rect::new(a.x, a.y + a.height - 1, a.width, 1);
            f.render_widget(Clear, line);
            f.render_widget(
                Paragraph::new(format!(
                    "{} {message}",
                    if *ok { "[ DONE ]" } else { "[ ERROR ]" }
                ))
                .style(Style::new().fg(if *ok {
                    NORMAL_GREEN
                } else {
                    BRIGHT_RED
                })),
                line,
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alerts_fire_once_per_transition_and_do_not_call_truncated_stream_success() {
        let mut a = Alerts::default();
        let mut s = RunState::default();
        s.ingest("progress_schema=1");
        a.ingest("progress_schema=1");
        a.observe(&s);
        a.eof(&s);
        assert!(!a.message.as_ref().unwrap().0);
        a.ingest("progress_schema=1");
        a.ingest("training_seconds=1");
        assert!(a.message.as_ref().unwrap().0);
        assert!(
            a.message
                .as_ref()
                .unwrap()
                .1
                .contains("waiting for checkpoint")
        );
        a.ingest("saved_checkpoint=model.pssa");
        assert!(a.message.as_ref().unwrap().1.contains("Checkpoint saved"));
        a.bell = false;
        a.ingest("training_seconds=1");
        assert!(!a.bell);
        s.ingest("loss=NaN tokens_per_second=1");
        a.observe(&s);
        assert!(a.bell);
        a.bell = false;
        a.observe(&s);
        assert!(!a.bell);
    }
    #[test]
    fn final_progress_without_summary_is_not_success() {
        let mut a = Alerts::default();
        let mut s = RunState::default();
        for line in [
            "progress_schema=1 updates_total=1",
            "optimizer_updates=1 updates_total=1 tokens_per_second=10",
        ] {
            s.ingest(line);
            a.ingest(line);
        }
        assert_eq!(s.updates_done, Some(1));
        assert_eq!(s.updates_total, Some(1));
        a.observe(&s);
        a.eof(&s);
        assert!(!a.message.as_ref().unwrap().0);
    }
    #[test]
    fn notification_line_renders() {
        let mut a = Alerts::default();
        a.notify(true, "Benchmark finished");
        let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(79, 24)).unwrap();
        t.draw(|f| a.draw(f)).unwrap();
        let s: String = t
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(s.contains("[ DONE ] Benchmark finished"));
    }
}
