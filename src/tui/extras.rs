//! Phase 11 orchestration; keeps the shared app shell edits additive.
use super::{
    RunState,
    alerts::Alerts,
    benchmark::Benchmark,
    chat::Chat,
    inspector::Inspector,
    kaggle::{Event, Kaggle},
    overlay::{Action, Overlay},
    runs::{Action as RunAction, Runs},
};
use crossterm::event::{KeyCode, KeyEvent};
use std::path::PathBuf;
pub(super) struct Extras {
    kaggle: Kaggle,
    inspector: Inspector,
    runs: Runs,
    benchmark: Benchmark,
    overlay: Overlay,
    alerts: Alerts,
}
impl Extras {
    pub fn new(chain: PathBuf) -> Self {
        Self {
            kaggle: Kaggle::new(),
            inspector: Inspector::default(),
            runs: Runs::new(chain.clone()),
            benchmark: Benchmark::new(chain),
            overlay: Overlay::default(),
            alerts: Alerts::default(),
        }
    }
    pub fn ingest(&mut self, line: &str) {
        if line.contains("progress_schema=") {
            self.inspector = Inspector::default();
        }
        self.inspector.ingest(line);
        self.alerts.ingest(line);
    }
    pub fn eof(&mut self, state: &RunState) {
        self.alerts.eof(state);
    }
    pub fn take_bell(&mut self) -> bool {
        std::mem::take(&mut self.alerts.bell)
    }
    pub fn poll(&mut self, state: &mut RunState, tab: &mut usize, chat: &mut Chat) {
        let remote = self.kaggle.poll();
        let mut finished = None;
        while let Some(event) = self.kaggle.take_event() {
            match event {
                Event::Started { reference } => {
                    let chain_dir = state.chain_dir.clone();
                    *state = RunState { chain_dir, corpus: Some(format!("Kaggle: {reference}")), ..Default::default() };
                    self.inspector = Inspector::default();
                    state.ingest("progress_schema=1"); self.alerts.ingest("progress_schema=1");
                    *tab = 0;
                }
                Event::Done => finished = Some((true,"Kaggle training finished.".to_string())),
                Event::Error(e) => finished = Some((false,format!("Kaggle: {e}"))),
                Event::Detached => finished = Some((false,"Kaggle log follow stopped; remote notebook may STILL be running. Stop it on kaggle.com to end quota use.".into())),
            }
        }
        for line in remote {
            self.ingest(&line);
            state.ingest(&line);
        }
        if let Some((ok, message)) = finished {
            state.training_active = false;
            if !ok {
                state.problem = Some(message.clone());
            }
            self.alerts.notify(ok, message);
        }
        self.inspector.poll(state, *tab == 7);
        self.runs.poll(*tab == 8);
        if let Some(action) = self.runs.action.take() {
            match action {
                RunAction::Monitor(opened) => {
                    *state = opened;
                    self.inspector = Inspector::default();
                    self.alerts = Alerts::default();
                    *tab = 0;
                }
                RunAction::Chat(path) => {
                    chat.open_checkpoint(&path);
                    *tab = 4;
                }
            }
        }
        self.benchmark.poll();
        for (ok, message) in [
            self.runs.notification.take(),
            self.benchmark.notification.take(),
        ]
        .into_iter()
        .flatten()
        {
            self.alerts.notify(ok, message);
        }
        self.alerts.observe(state);
    }
    /// (consumed, quit). Tab remains owned by the shared shell; forms own Esc.
    pub fn key(&mut self, key: KeyEvent, tab: &mut usize) -> (bool, bool) {
        let text_entry = matches!(*tab, 4 | 5)
            || (*tab == 6 && !self.kaggle.is_busy())
            || (*tab == 8 && self.runs.editing())
            || (*tab == 9 && self.benchmark.editing());
        let (handled, action) = self.overlay.key(key, text_entry);
        if let Some(action) = action {
            match action {
                Action::Tab(next) => *tab = next,
                Action::Benchmark => {
                    *tab = 9;
                    self.benchmark.start();
                }
                Action::Quit => return (true, true),
            }
        }
        if handled {
            return (true, false);
        }
        if key.code == KeyCode::Tab
            || (!text_entry
                && matches!(
                    key.code,
                    KeyCode::Left | KeyCode::Right | KeyCode::Char('q')
                ))
        {
            return (false, false);
        }
        match *tab {
            6 => self.kaggle.key(key),
            7 => self.inspector.key(key),
            8 => self.runs.key(key),
            9 => self.benchmark.key(key),
            _ => return (false, false),
        }
        (true, false)
    }
    pub fn draw(&mut self, f: &mut ratatui::Frame, state: &RunState, tab: usize) {
        let area = super::feature_area(f.area());
        match tab {
            6 => self.kaggle.draw(f, area),
            7 => self.inspector.draw(f, area, state),
            8 => self.runs.draw(f, area),
            9 => self.benchmark.draw(f, area),
            _ => {}
        }
        self.alerts.draw(f);
        self.overlay.draw(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    #[test]
    fn palette_reaches_all_new_tabs_without_swallowing_text_or_tab() {
        let mut extras = Extras::new("missing-chain".into());
        let mut tab = 4;
        assert!(
            !extras
                .key(
                    KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE),
                    &mut tab
                )
                .0
        );
        assert!(
            extras
                .key(
                    KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL),
                    &mut tab
                )
                .0
        );
        for c in "runs".chars() {
            extras.key(
                KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE),
                &mut tab,
            );
        }
        extras.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &mut tab);
        assert_eq!(tab, 8);
        assert!(
            !extras
                .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE), &mut tab)
                .0
        );
        extras.key(KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE), &mut tab);
        let (_, quit) = extras.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &mut tab);
        assert!(!quit);
    }
    #[test]
    fn all_extras_render_inside_real_shell_at_all_sizes() {
        let mut extras = Extras::new("missing-chain".into());
        let state = RunState::default();
        for (w, h) in [(120, 32), (79, 24), (24, 8), (1, 1)] {
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            for tab in 6..=9 {
                t.draw(|f| {
                    super::super::draw(f, &state, tab);
                    extras.draw(f, &state, tab);
                })
                .unwrap();
                if w == 120 {
                    let text: String = t
                        .backend()
                        .buffer()
                        .content
                        .iter()
                        .map(|c| c.symbol())
                        .collect();
                    assert!(text.contains(super::super::TABS[tab]));
                }
            }
        }
    }
}
