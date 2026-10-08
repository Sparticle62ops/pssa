//! Phase 11 orchestration; keyboard globals/help stay in the shared registry.
use super::{
    BENCHMARK_TAB, KAGGLE_TAB, MEMORY_TAB, RUNS_TAB, RunState,
    alerts::Alerts,
    benchmark::Benchmark,
    chat::Chat,
    inspector::Inspector,
    kaggle::{Event, Kaggle},
    keybindings::Context,
    runs::{Action as RunAction, Runs},
};
use crossterm::event::{KeyCode, KeyEvent};
use std::path::PathBuf;
pub(super) struct Extras {
    kaggle: Kaggle,
    inspector: Inspector,
    runs: Runs,
    benchmark: Benchmark,
    alerts: Alerts,
    remote_monitor: bool,
    memory_live: bool,
}
impl Extras {
    pub fn new(chain: PathBuf) -> Self {
        Self {
            kaggle: Kaggle::new(),
            inspector: Inspector::default(),
            runs: Runs::new(chain.clone()),
            benchmark: Benchmark::new(chain),
            alerts: Alerts::default(),
            remote_monitor: false,
            memory_live: false,
        }
    }
    pub fn set_chain_dir(&mut self, chain: PathBuf) {
        self.runs.add_root(chain);
        self.remote_monitor = false;
        self.inspector = Inspector::default();
        self.alerts = Alerts::default();
    }
    pub fn remote_monitor(&self) -> bool {
        self.remote_monitor
    }
    pub fn remote_busy(&self) -> bool {
        self.kaggle.is_busy()
    }
    pub(super) fn training_busy(&self) -> bool {
        self.remote_busy() || self.benchmark.busy()
    }
    pub fn ingest(&mut self, line: &str) {
        if line.contains("progress_schema=") {
            self.inspector = Inspector::default();
        }
        if !self.remote_monitor
            && let Some(path) = line.trim().strip_prefix("saved_checkpoint=").or_else(|| {
                line.split_once("checkpoint_target=")
                    .map(|(_, path)| path.trim())
            })
            && path != "-"
            && !path.is_empty()
            && let Some(parent) = std::path::Path::new(path).parent()
        {
            self.runs.add_root(if parent.as_os_str().is_empty() {
                ".".into()
            } else {
                parent.to_path_buf()
            });
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
    fn kaggle_event(&mut self, event: Event, state: &mut RunState, tab: &mut usize) {
        let (ok, message) = match event {
            Event::Started { reference } => {
                let chain_dir = state.chain_dir.clone();
                *state = RunState { chain_dir, corpus: Some(format!("Kaggle: {reference}")), ..Default::default() };
                self.remote_monitor = true;
                self.inspector = Inspector::default();
                state.ingest("progress_schema=1");
                self.alerts.ingest("progress_schema=1");
                *tab = 0;
                return;
            }
            Event::Done => (true, "Kaggle training finished.".to_string()),
            Event::Error(e) => (false, format!("Kaggle: {e}")),
            Event::Detached => (false, "Kaggle log follow stopped; remote notebook may STILL be running. Stop it on kaggle.com to end quota use.".into()),
        };
        // Credential/preview errors are notifications, not failures of a local
        // trainer or historical run displayed in the shared monitor.
        if self.remote_monitor {
            state.training_active = false;
            if !ok {
                state.problem = Some(message.clone());
            }
        }
        self.alerts.notify(ok, message);
    }
    #[cfg(test)]
    pub fn poll(&mut self, state: &mut RunState, tab: &mut usize, chat: &mut Chat) {
        self.poll_with(state, tab, chat, |_| {});
    }
    pub(super) fn poll_with(
        &mut self,
        state: &mut RunState,
        tab: &mut usize,
        chat: &mut Chat,
        mut remote_line: impl FnMut(&str),
    ) {
        let remote = self.kaggle.poll();
        let mut finished = Vec::new();
        while let Some(event) = self.kaggle.take_event() {
            if matches!(event, Event::Started { .. }) {
                self.kaggle_event(event, state, tab);
            } else {
                finished.push(event);
            }
        }
        for line in remote {
            remote_line(&line);
            self.ingest(&line);
            state.ingest(&line);
        }
        for event in finished {
            self.kaggle_event(event, state, tab);
        }
        // Remote paths are not local checkpoint files. Still collect streamed
        // occupancy/snippets, but never load a coincidentally matching path.
        self.inspector.poll(
            state,
            *tab == MEMORY_TAB && !self.remote_monitor && !self.memory_live,
        );
        self.runs.poll(*tab == RUNS_TAB);
        if let Some(action) = self.runs.action.take() {
            match action {
                RunAction::Monitor(_) if state.training_active || self.remote_busy() => {
                    self.alerts.notify(
                        false,
                        "A training run is active; wait before opening recorded history.",
                    );
                }
                RunAction::Monitor(opened) => {
                    *state = opened;
                    self.remote_monitor = false;
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
    pub fn context(&self, tab: usize) -> Option<Context> {
        let editing = match tab {
            KAGGLE_TAB => !self.kaggle.is_busy(),
            MEMORY_TAB => false,
            RUNS_TAB => self.runs.editing(),
            BENCHMARK_TAB => self.benchmark.editing(),
            _ => return None,
        };
        Some(Context::for_tab(tab, editing))
    }
    pub fn start_benchmark(&mut self) {
        self.benchmark.start();
    }
    /// Only tab-local keys resolved by the registry reach these modules.
    pub fn key(&mut self, key: KeyEvent, tab: usize, state: &RunState) {
        match tab {
            KAGGLE_TAB => {
                if (state.training_active && !self.remote_monitor || self.benchmark.busy())
                    && matches!(key.code, KeyCode::Enter | KeyCode::Char('y' | 'Y'))
                {
                    self.alerts.notify(
                        false,
                        "A local training run is active; wait before launching Kaggle.",
                    );
                } else {
                    self.kaggle.key(key);
                }
            }
            MEMORY_TAB => {
                if key.code == KeyCode::Char('v') {
                    self.memory_live = !self.memory_live;
                } else if !self.memory_live {
                    self.inspector.key(key);
                }
            }
            RUNS_TAB => self.runs.key(key),
            BENCHMARK_TAB
                if (state.training_active || self.remote_busy())
                    && !self.benchmark.editing()
                    && key.code == KeyCode::Char('b') =>
            {
                self.alerts.notify(
                    false,
                    "A training run is active; wait before starting a benchmark.",
                );
            }
            BENCHMARK_TAB => self.benchmark.key(key),
            _ => {}
        }
    }
    pub fn draw(&mut self, f: &mut ratatui::Frame, state: &RunState, tab: usize, _chat: &Chat) {
        let area = super::feature_area(f.area());
        match tab {
            KAGGLE_TAB => self.kaggle.draw(f, area),
            MEMORY_TAB => {
                if self.memory_live {
                    _chat.memory_view().draw(f, area);
                } else {
                    self.inspector.draw(f, area, state);
                }
            }
            RUNS_TAB => self.runs.draw(f, area),
            BENCHMARK_TAB => self.benchmark.draw(f, area),
            _ => {}
        }
        self.alerts.draw(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    #[test]
    fn extras_contexts_use_shared_tab_mapping_and_preserve_input() {
        use super::super::keybindings::{self, Action};
        let mut extras = Extras::new("missing-chain".into());
        for (tab, context) in [
            (KAGGLE_TAB, Context::KaggleInput),
            (MEMORY_TAB, Context::Memory),
            (RUNS_TAB, Context::Runs),
            (BENCHMARK_TAB, Context::Benchmark),
        ] {
            assert_eq!(extras.context(tab), Some(context));
            assert_eq!(
                keybindings::action(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE), context),
                Some(Action::NextTab)
            );
        }
        assert_eq!(extras.context(5), None, "setup remains owned by the shell");
        extras.key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            BENCHMARK_TAB,
            &RunState::default(),
        );
        assert_eq!(extras.context(BENCHMARK_TAB), Some(Context::BenchmarkEdit));
        assert_eq!(
            keybindings::action(
                KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE),
                Context::BenchmarkEdit
            ),
            Some(Action::Extras)
        );
    }
    #[test]
    fn kaggle_prelaunch_error_and_runs_cannot_replace_local_training() {
        let mut extras = Extras::new("missing-chain".into());
        let mut state = RunState {
            training_active: true,
            corpus: Some("local".into()),
            ..Default::default()
        };
        let mut tab = 0;
        extras.kaggle_event(
            Event::Error("invalid notebook folder".into()),
            &mut state,
            &mut tab,
        );
        assert!(state.training_active);
        assert!(state.problem.is_none());
        assert_eq!(state.corpus.as_deref(), Some("local"));
        extras.runs.action = Some(RunAction::Monitor(RunState::default()));
        let mut chat = Chat::new("unused-chats".into(), "missing-chain".into());
        extras.poll(&mut state, &mut tab, &mut chat);
        assert!(state.training_active);
        assert_eq!(state.corpus.as_deref(), Some("local"));
        extras.key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            KAGGLE_TAB,
            &state,
        );
        assert!(!extras.remote_busy());
    }
    #[test]
    fn all_extras_render_without_overwriting_shell_at_all_sizes() {
        let mut extras = Extras::new("missing-chain".into());
        let state = RunState::default();
        let chat = Chat::new("unused-chats".into(), "missing-chain".into());
        for (w, h) in [(160, 40), (120, 32), (79, 24), (24, 8), (1, 1)] {
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            for tab in KAGGLE_TAB..=BENCHMARK_TAB {
                t.draw(|f| super::super::draw(f, &state, tab)).unwrap();
                let shell = t.backend().buffer().clone();
                t.draw(|f| {
                    super::super::draw(f, &state, tab);
                    extras.draw(f, &state, tab, &chat);
                })
                .unwrap();
                let content = super::super::feature_area(ratatui::layout::Rect::new(0, 0, w, h));
                for y in 0..content.y {
                    for x in 0..w {
                        assert_eq!(
                            t.backend().buffer()[(x, y)],
                            shell[(x, y)],
                            "header/tab overwritten on tab {tab} at {w}x{h}"
                        );
                    }
                }
            }
        }
    }
}
