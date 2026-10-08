//! Small, sequential grids over the wizard's existing `train` command.
//! The shell owns/polls the child. Dropping pending work never touches a trainer.
use super::{
    AMBER, RunState, accent, panel, panel_area,
    process::clean,
    setup::{RunSpec, Setup, TrainingRun},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_RUNS: usize = 32;
const LABELS: [&str; 3] = ["Learning rates", "Latent sizes", "Batch lanes"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}
impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Pending => "queued",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}
struct Trial {
    grid: [String; 3],
    output: PathBuf,
    spec: Option<RunSpec>,
    status: Status,
    loss: Option<f64>,
    speed: Option<f64>,
}

pub(super) struct Sweep {
    grid: [String; 3],
    selected: usize,
    edit: Option<String>,
    trials: Vec<Trial>,
    current: Option<usize>,
    note: String,
}
impl Default for Sweep {
    fn default() -> Self {
        Self {
            grid: Default::default(),
            selected: 0,
            edit: None,
            trials: Vec::new(),
            current: None,
            note: "Blank grids use setup. Comma-separated values; max 8 per field / 32 runs."
                .into(),
        }
    }
}

fn values(input: &str, fallback: &str) -> Result<Vec<String>, String> {
    let input = if input.trim().is_empty() {
        fallback
    } else {
        input
    };
    let mut values = Vec::new();
    for value in input
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|v| !v.is_empty())
    {
        if !values.iter().any(|v| v == value) {
            values.push(value.to_string());
        }
        if values.len() > 8 {
            return Err("Use at most 8 values per grid field.".into());
        }
    }
    if values.is_empty() {
        return Err("A grid must contain a number, or be blank to use setup.".into());
    }
    Ok(values)
}

impl Sweep {
    pub(super) fn editing(&self) -> bool {
        self.edit.is_some()
    }

    pub(super) fn key(&mut self, key: KeyEvent, setup: &Setup) {
        if let Some(edit) = &mut self.edit {
            match key.code {
                KeyCode::Esc => self.edit = None,
                KeyCode::Enter => self.grid[self.selected] = self.edit.take().unwrap(),
                KeyCode::Backspace => {
                    edit.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => edit.clear(),
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                        && edit.len() + c.len_utf8() <= 256 =>
                {
                    edit.push(c)
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(2),
            KeyCode::Enter => self.edit = Some(self.grid[self.selected].clone()),
            KeyCode::Char('r') => {
                self.grid = setup.sweep_base().0;
                self.note = "Grid refreshed from setup. Already queued runs are unchanged.".into();
            }
            KeyCode::Char('p') => {
                if let Err(error) = self.queue(setup) {
                    self.note = error;
                }
            }
            KeyCode::Char('d') => {
                self.stop_pending();
                self.note =
                    "Pending runs stopped. An active trainer is NOT cancelled or detached.".into();
            }
            _ => {}
        }
    }

    fn queue(&mut self, setup: &Setup) -> Result<(), String> {
        if self.current.is_some() || self.trials.iter().any(|t| t.status == Status::Pending) {
            return Err(
                "A sweep is already queued. d stops pending runs, never the active trainer.".into(),
            );
        }
        let (defaults, base) = setup.sweep_base();
        let grids: Vec<Vec<String>> = self
            .grid
            .iter()
            .zip(defaults.iter())
            .map(|(input, fallback)| values(input, fallback))
            .collect::<Result<_, _>>()?;
        let count = grids.iter().map(Vec::len).product::<usize>();
        if count > MAX_RUNS {
            return Err(format!("Grid has {count} runs; limit is {MAX_RUNS}."));
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = base.join(format!(
            "sweep-{stamp}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut trials = Vec::new();
        // Validate the entire snapshot before queuing anything. Never launch here.
        for lr in &grids[0] {
            for latent in &grids[1] {
                for batch in &grids[2] {
                    let output = root.join(format!("trial-{:03}", trials.len() + 1));
                    let spec = setup.sweep_spec(lr, latent, batch, &output)?;
                    trials.push(Trial {
                        grid: [lr.clone(), latent.clone(), batch.clone()],
                        output,
                        spec: Some(spec),
                        status: Status::Pending,
                        loss: None,
                        speed: None,
                    });
                }
            }
        }
        self.trials = trials;
        self.note = format!(
            "Queued {count} runs; waits for local AND remote training. Logs: {}/trial-NNN/train.log",
            clean(&root.display().to_string())
        );
        Ok(())
    }

    fn stop_pending(&mut self) {
        for trial in &mut self.trials {
            if trial.status == Status::Pending {
                trial.status = Status::Skipped;
                trial.spec = None;
            }
        }
    }

    fn begin_next(&mut self, busy: bool) -> Option<RunSpec> {
        if busy || self.current.is_some() {
            return None;
        }
        let index = self
            .trials
            .iter()
            .position(|t| t.status == Status::Pending)?;
        let trial = &mut self.trials[index];
        let spec = trial.spec.take()?;
        trial.status = Status::Running;
        self.current = Some(index);
        Some(spec)
    }

    fn complete(&mut self, output: &Path, success: bool, state: &RunState) {
        let Some(index) = self.current else { return };
        let trial = &mut self.trials[index];
        if trial.output != output {
            // A browser/remote monitor must not donate its metrics to this trial.
            return;
        }
        trial.status = if success {
            Status::Done
        } else {
            Status::Failed
        };
        if state.chain_dir == output {
            trial.loss = state
                .epoch_loss
                .or(state.live_loss)
                .filter(|v| v.is_finite());
            trial.speed = state
                .throughput
                .as_deref()
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.replace(',', "").parse::<f64>().ok())
                .or(state.tok_s)
                .filter(|v| v.is_finite() && *v >= 0.0);
        }
        self.current = None;
        self.note = format!(
            "Trial {} {}. Metrics are recorded training loss/throughput, NOT held-out scores.",
            index + 1,
            trial.status.label()
        );
    }

    /// Call AFTER the shell has polled TrainingRun and delivered its EOF alerts.
    /// `busy` must include the shell's remote/local one-run gate. Returns true only
    /// when a new child was installed AND initialized; update chat/extras roots then.
    /// This never polls, kills, detaches, or removes the shell's active child.
    pub(super) fn poll(
        &mut self,
        training: &mut Option<TrainingRun>,
        state: &mut RunState,
        busy: bool,
    ) -> bool {
        if let Some(run) = training.as_ref() {
            if let Some(success) = run.succeeded() {
                self.complete(run.output_dir(), success, state);
            } else if run.active() && let Some(index) = self.current {
                let trial = &mut self.trials[index];
                if trial.output == run.output_dir() && state.chain_dir == trial.output {
                    trial.loss = state.live_loss.filter(|value| value.is_finite());
                    trial.speed = state.tok_s.filter(|value| value.is_finite() && *value >= 0.0);
                }
            }
        }
        let blocked =
            busy || state.training_active || training.as_ref().is_some_and(TrainingRun::active);
        let Some(spec) = self.begin_next(blocked) else {
            return false;
        };
        match spec.launch(blocked) {
            Ok(run) => {
                run.initialize(state);
                *training = Some(run);
                self.note =
                    "Training one trial. d stops only pending runs; logs remain durable.".into();
                true
            }
            Err(error) => {
                if let Some(index) = self.current.take() {
                    self.trials[index].status = Status::Failed;
                }
                self.stop_pending();
                self.note = format!("Launch failed; pending sweep stopped: {}", clean(&error));
                false
            }
        }
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        if area.is_empty() {
            return;
        }
        let show_plot =
            area.width >= 70 && area.height >= 15 && self.trials.iter().any(|t| t.loss.is_some());
        let parts = Layout::vertical([
            Constraint::Length(if area.height < 12 { 4 } else { 6 }),
            Constraint::Min(0),
            Constraint::Length(if show_plot && area.height < 20 {
                1
            } else if area.height < 12 {
                2
            } else {
                4
            }),
        ])
        .split(area);
        let (results, plot) = if show_plot {
            let columns =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(parts[1]);
            (columns[0], Some(columns[1]))
        } else {
            (parts[1], None)
        };
        let compact = results.width < 80;
        let mut fields = Vec::new();
        for (index, label) in LABELS.iter().enumerate() {
            let value = if index == self.selected {
                self.edit.as_deref().unwrap_or(&self.grid[index])
            } else {
                &self.grid[index]
            };
            let value = if value.is_empty() { "(setup)" } else { value };
            fields.push(Line::styled(
                format!(
                    "{} {label}: {}{}",
                    if index == self.selected { "▶" } else { " " },
                    clean(value),
                    if index == self.selected && self.editing() {
                        "▏"
                    } else {
                        ""
                    }
                ),
                if index == self.selected {
                    accent().add_modifier(Modifier::REVERSED)
                } else {
                    accent()
                },
            ));
        }
        fields.push(Line::from(
            "Comma-separated grids / blank = wizard value / 32 runs max",
        ));
        let base_area = panel_area(f, parts[0]);
        f.render_widget(
            Paragraph::new(fields).block(panel(" sweep / wizard base ")),
            base_area,
        );
        let mut rows = vec![Line::styled(
            if compact {
                "№ state lr/latent/batch │ loss │ tok/s"
            } else {
                "№    state       learning rate  latent  batch    training loss     tok/s"
            },
            accent(),
        )];
        if self.trials.is_empty() {
            rows.push(Line::from(
                "No runs queued. Configure setup, edit grids, then p.",
            ));
        }
        let height = usize::from(results.height.saturating_sub(3));
        let focus = self.current.unwrap_or_else(|| {
            self.trials
                .iter()
                .rposition(|t| t.status != Status::Pending)
                .unwrap_or(0)
        });
        let start = focus.saturating_sub(height.saturating_sub(1));
        for (i, trial) in self.trials.iter().enumerate().skip(start).take(height) {
            let loss = trial.loss.map_or("—".into(), super::charts::number);
            let speed = trial.speed.map_or("—".into(), |n| format!("{n:.0}"));
            let [lr, latent, batch] = &trial.grid;
            let row = if compact {
                format!(
                    "{} {:7} {lr}/{latent}/{batch} │ {loss} │ {speed}",
                    i + 1,
                    trial.status.label()
                )
            } else {
                format!(
                    "{:<4} {:10} {lr:<14} {latent:<7} {batch:<8} {loss:<17} {speed}",
                    i + 1,
                    trial.status.label()
                )
            };
            rows.push(Line::styled(
                row,
                if trial.status == Status::Running {
                    accent()
                } else {
                    Style::default()
                },
            ));
        }
        let results_area = panel_area(f, results);
        f.render_widget(
            Paragraph::new(rows).block(panel(" results / training only ")),
            results_area,
        );
        if let Some(area) = plot {
            // Later shadows from the base/results panels must not erase the
            // chart's top or left border inside the shared feature rectangle.
            let points: Vec<_> = self
                .trials
                .iter()
                .enumerate()
                .map(|(i, trial)| ((i + 1) as f64, trial.loss.unwrap_or(f64::NAN)))
                .collect();
            super::charts::draw(
                f,
                area,
                super::charts::Plot {
                    title: " sweep / loss by trial ",
                    caption: "Lower = better training fit",
                    x: "trial",
                    integer_x: true,
                    y: "loss",
                    x_bounds: super::charts::domain(&points),
                    y_bounds: super::charts::bounds(points.iter().map(|p| p.1)),
                },
                &[super::charts::Series::line(
                    "loss",
                    &points,
                    super::NORMAL_GREEN,
                )],
            );
        }
        let mut footer = vec![Line::styled(
            if self.editing() {
                "Enter save / Esc cancel / Ctrl+U clear / type"
            } else {
                "↑/↓ select / Enter edit / r setup / p queue / d stop pending"
            },
            accent(),
        )];
        footer.push(Line::styled(clean(&self.note), Style::new().fg(AMBER)));
        f.render_widget(Paragraph::new(footer).wrap(Wrap { trim: false }), parts[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use std::fs;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "pssa-sweep-{}-{} path's spaces",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join("source file.txt"), "small offline corpus").unwrap();
            Self(root)
        }
        fn setup(&self) -> Setup {
            let mut setup = Setup::default();
            let press = |s: &mut Setup, code| {
                s.key(KeyEvent::new(code, KeyModifiers::NONE));
            };
            press(&mut setup, KeyCode::Down);
            press(&mut setup, KeyCode::Enter);
            for c in self.0.join("source file.txt").to_str().unwrap().chars() {
                press(&mut setup, KeyCode::Char(c));
            }
            press(&mut setup, KeyCode::Enter);
            for _ in 0..3 {
                press(&mut setup, KeyCode::Right);
            }
            press(&mut setup, KeyCode::Enter);
            setup.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
            for c in self.0.to_str().unwrap().chars() {
                press(&mut setup, KeyCode::Char(c));
            }
            press(&mut setup, KeyCode::Enter);
            setup
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn queues_validated_snapshots_with_unique_durable_paths_and_never_overlaps() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        let mut sweep = Sweep {
            grid: ["0.001,0.002".into(), "32,64".into(), "1,2".into()],
            ..Default::default()
        };
        sweep.queue(&setup).unwrap();
        assert_eq!(sweep.trials.len(), 8);
        let first_path = sweep.trials[0].output.clone();
        assert!(first_path.starts_with(&fixture.0));
        assert!(
            !first_path.exists(),
            "queuing never launches or creates run outputs"
        );
        let mut training = None;
        let mut busy_state = RunState {
            training_active: true,
            ..Default::default()
        };
        assert!(!sweep.poll(&mut training, &mut busy_state, false));
        busy_state.training_active = false;
        assert!(!sweep.poll(&mut training, &mut busy_state, true));
        assert!(training.is_none());
        assert!(sweep.current.is_none());
        assert!(
            sweep.begin_next(true).is_none(),
            "remote/local busy must block launch"
        );
        assert!(sweep.begin_next(false).is_some());
        assert!(
            sweep.begin_next(false).is_none(),
            "active trial blocks even if shell busy is false"
        );
        let mut state = RunState {
            chain_dir: first_path.clone(),
            ..Default::default()
        };
        state.ingest("loss=3 tokens_per_second=100");
        state.ingest("epoch 1/1 loss=2.5 tokens=64 updates=1");
        state.ingest("throughput      90 tokens/second");
        sweep.complete(Path::new("unrelated run"), true, &state);
        assert!(sweep.begin_next(false).is_none());
        sweep.complete(&first_path, true, &state);
        assert_eq!(sweep.trials[0].loss, Some(2.5));
        assert_eq!(sweep.trials[0].speed, Some(90.0));
        assert!(sweep.begin_next(false).is_some());
        sweep.stop_pending();
        assert_eq!(sweep.trials[1].status, Status::Running);
        assert!(
            sweep.trials[2..]
                .iter()
                .all(|t| t.status == Status::Skipped)
        );
        assert!(sweep.begin_next(false).is_none());
        let second_path = sweep.trials[1].output.clone();
        sweep.complete(&second_path, false, &state);
        assert_eq!(
            sweep.trials[1].loss, None,
            "other monitor's metrics are not this run's results"
        );
        assert!(sweep.begin_next(false).is_none());
        sweep.queue(&setup).unwrap();
        assert_ne!(sweep.trials[0].output, first_path);
    }

    #[cfg(unix)]
    #[test]
    fn shell_completion_order_records_trial_before_releasing_slot_and_advances_queue() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        let mut sweep = Sweep {
            grid: ["0.001,0.002".into(), "32".into(), "1".into()],
            ..Default::default()
        };
        sweep.queue(&setup).unwrap();
        let spec = sweep.begin_next(false).unwrap();
        // An inert child tests lifecycle wiring; never launch a trainer.
        let run = spec.start(Path::new("/bin/true")).unwrap();
        let mut state = RunState::default();
        run.initialize(&mut state);
        fs::write(run.output_dir().join("train.log"), "epoch 1/1 loss=2.5 tokens=64 updates=1\nthroughput      90 tokens/second\n").unwrap();
        let mut training = Some(run);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while training.as_ref().unwrap().active() {
            training.as_mut().unwrap().poll(&mut state);
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Use the actual shell ordering; block launch until assertions complete.
        assert!(!super::super::poll_training_queue(&mut training, true, |slot| {
            sweep.poll(slot, &mut state, true)
        }));
        assert!(training.is_none());
        assert_eq!(sweep.trials[0].status, Status::Done);
        assert_eq!(sweep.trials[0].loss, Some(2.5));
        assert_eq!(sweep.trials[0].speed, Some(90.0));
        assert!(sweep.current.is_none());
        assert!(sweep.begin_next(false).is_some());
        assert_eq!(sweep.current, Some(1));
        assert_eq!(sweep.trials[1].status, Status::Running);
    }

    #[test]
    fn invalid_or_excessive_grid_is_atomic_and_edit_shortcuts_do_not_launch() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        for grid in [
            ["NaN", "32", "1"],
            ["0.001", "0", "1"],
            ["0.001", "32", "0"],
            ["1,2,3,4,5,6,7,8,9", "32", "1"],
            ["1,2,3,4", "32,64,128", "1,2,3"],
        ] {
            let mut sweep = Sweep {
                grid: grid.map(str::to_owned),
                ..Default::default()
            };
            assert!(sweep.queue(&setup).is_err());
            assert!(sweep.trials.is_empty());
        }
        let mut sweep = Sweep::default();
        sweep.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), &setup);
        for c in "prdq".chars() {
            sweep.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE), &setup);
        }
        assert_eq!(sweep.edit.as_deref(), Some("prdq"));
        assert!(sweep.trials.is_empty());
        sweep.key(
            KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
            &setup,
        );
        assert_eq!(sweep.edit.as_deref(), Some(""));
        sweep.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &setup);
        assert!(!sweep.editing());
        sweep.key(
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
            &setup,
        );
        assert_eq!(sweep.trials.len(), 1);
        assert!(sweep.current.is_none());
    }

    #[test]
    fn completed_trials_render_braille_and_axes_at_both_sizes() {
        let sweep = Sweep {
            trials: [4.0, 3.0, 2.0]
                .into_iter()
                .map(|loss| Trial {
                    grid: [".001".into(), "32".into(), "1".into()],
                    output: PathBuf::from("unused"),
                    spec: None,
                    status: Status::Done,
                    loss: Some(loss),
                    speed: Some(100.0),
                })
                .collect(),
            ..Default::default()
        };
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| sweep.draw(f, f.area())).unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "sweep / loss by trial",
                &[
                    "trial",
                    "loss",
                    "Lower = better training fit",
                    "1",
                    "2",
                    "3",
                ],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "sweep / loss by trial",
                if h == 40 { 8 } else { 3 },
            );
            terminal
                .draw(|f| sweep.draw(f, super::super::feature_area(f.area())))
                .unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "sweep / loss by trial",
                &["trial", "loss", "Lower = better training fit"],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "sweep / loss by trial",
                if h == 40 { 8 } else { 3 },
            );
        }
    }

    #[test]
    fn test_backend_sweep_wide_narrow_and_tiny() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        let mut sweep = Sweep::default();
        sweep.queue(&setup).unwrap();
        for (width, height) in [(120, 30), (79, 24), (45, 14), (10, 4), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| sweep.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 45 {
                assert!(text.contains("sweep / wizard base"));
                assert!(text.contains("queued"));
                assert!(text.contains("loss"));
                assert!(text.contains("tok/s"));
            }
        }
    }
}
