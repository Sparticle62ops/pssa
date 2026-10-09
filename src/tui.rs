//! Training dashboard and local checkpoint chat. Model/checkpoint files are
//! read-only; inference conversations are saved separately in the chats directory.
//! Piped non-TTY output remains plain logging.

mod ab;
mod alerts;
mod background;
mod benchmark;
pub(crate) mod benchmark_replay;
mod charts;
mod chat;
mod comparison;
mod depth_zoom;
mod device;
mod eval;
mod extras;
mod feed;
mod github;
mod hardware;
mod heatmap;
pub(crate) mod hf;
mod hf_backup;
mod inspector;
mod kaggle;
mod keybindings;
mod library;
mod limits;
mod local;
mod log_stream;
mod math;
mod memory_view;
mod mixer;
mod network;
mod notify;
mod overlay;
mod preview;
mod process;
mod ring;
mod runs;
mod session;
mod setup;
mod shadow;
#[cfg(feature = "speech")]
mod speech;
mod support;
mod sweep;
mod timeline;
mod update;

use crate::ui;
use background::HexBackground;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Tabs, Wrap,
    canvas::{Canvas, Line as CanvasLine, Points},
};
use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;
use std::sync::{OnceLock, mpsc};
use std::time::{Duration, Instant};

use keybindings::{BENCHMARK_TAB, HF_TAB, KAGGLE_TAB, MEMORY_TAB, RUNS_TAB, TABS};
// Five scanlines leave room for the P's stem, both S turns and the A's crossbar.
const PSSA_LOGO: [&str; 5] = [
    "███    ███   ███   ██ ",
    "█  █  █     █     █  █",
    "███    ██    ██   ████",
    "█        █     █  █  █",
    "█     ███   ███   █  █",
];
const PANEL_BG: Color = Color::Rgb(0x07, 0x12, 0x10);
const INSET_BG: Color = Color::Rgb(0x09, 0x14, 0x1a);
const SECOND_ACCENT: Color = Color::Rgb(0x57, 0x82, 0x91);
const GRID_COLOR: Color = Color::Rgb(0x19, 0x30, 0x35);
const NORMAL_GREEN: Color = Color::Rgb(0x39, 0xe0, 0x7a);
const AMBER: Color = Color::Rgb(0xff, 0xbf, 0x00);
const BRIGHT_RED: Color = Color::Rgb(0xff, 0x2f, 0x3f);
const STALL_TIMEOUT: Duration = Duration::from_secs(30);
const PROGRESS_INTERPOLATION: Duration = Duration::from_millis(450);
const BRAILLE_LEVELS: [&str; 8] = ["⡀", "⡄", "⡆", "⡇", "⣇", "⣧", "⣷", "⣿"];
const GRAPH_MAX_ZOOM: usize = 8;
const GRAPH_INTERPOLATION: Duration = Duration::from_millis(300);
const FRAME_INTERVAL: Duration = Duration::from_millis(34); // at most ~30 fps
const NEURON_CYCLE: Duration = Duration::from_millis(9_000);
const NEURON_GROW_START: f64 = 0.09;
const NEURON_GROW_END: f64 = 0.64;
const NEURON_COLLAPSE_END: f64 = 0.84;
const NEURON_NODE_COUNT: usize = 80;

#[derive(Clone, Copy, Debug, Default)]
struct MetricSample {
    step: Option<f64>,
    loss: Option<f64>,
    tokens_per_second: Option<f64>,
    learning_rate: Option<f64>,
}

impl MetricSample {
    fn has_value(self) -> bool {
        self.loss.is_some() || self.tokens_per_second.is_some() || self.learning_rate.is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum GraphView {
    #[default]
    Loss,
    Perplexity,
    TokensPerSecond,
    LearningRate,
    Comparison,
    All,
    Memory,
}

impl GraphView {
    fn next(self) -> Self {
        match self {
            Self::Loss => Self::Perplexity,
            Self::Perplexity => Self::TokensPerSecond,
            Self::TokensPerSecond => Self::LearningRate,
            Self::LearningRate => Self::Comparison,
            Self::Comparison => Self::All,
            Self::All => Self::Memory,
            Self::Memory => Self::Loss,
        }
    }
}

use feed::FeedState;

#[derive(Default)]
struct RunState {
    preview: preview::Preview,
    hardware: hardware::Hardware,
    math: math::Values,
    math_view: math::Math,
    device_picker: std::cell::RefCell<device::DevicePicker>,
    resource_limits: std::cell::RefCell<limits::Limits>,
    loop_count: usize,
    feed: Option<FeedState>,
    feed_telemetry: Option<bool>,
    feed_scroll: std::cell::Cell<u16>,
    // header card
    corpus: Option<String>,
    vocab: Option<String>,
    width: Option<String>,
    memory: Option<String>,
    schedule: Option<String>,
    // monitor tab
    // `progress_pct` is the latest target from the log. The two animation
    // fields keep the bar moving smoothly between producer updates without
    // changing the parsed value used by the rest of the dashboard.
    progress_pct: Option<f64>,
    progress_from: Option<f64>,
    progress_updated_at: Option<Instant>,
    live_loss: Option<f64>,
    loss_average: Option<f64>,
    tok_s: Option<f64>,
    eta: Option<String>,
    // Dream telemetry is runtime-only and comes from explicit training log
    // events. It never participates in progress or health calculations.
    dream_active: bool,
    dream_count: u64,
    dream_mode: Option<String>,
    dream_last_loss: Option<f64>,
    dream_update: Option<u64>,
    loss_guard_status: Option<String>,
    memory_retrieval: Option<String>,
    memory_growth: Option<String>,
    updates_done: Option<u64>,
    updates_total: Option<u64>,
    updates_remaining: Option<u64>,
    learning_rate: Option<f64>,
    memory_used: Option<u64>,
    memory_capacity: Option<u64>,
    checkpoint_number: Option<u64>,
    checkpoint_target: Option<String>,
    last_checkpoint: Option<String>,
    selected_checkpoint: Option<String>,
    checkpoint_saved_current: bool,
    configuration_source: Option<String>,
    checkpoint_revision: u64,
    // Parsed progress samples used by the real line charts. `loss_series` is
    // retained for health checks and existing log compatibility tests.
    loss_series: Vec<f64>,
    metric_series: Vec<MetricSample>,
    graph_from: Option<MetricSample>,
    graph_updated_at: Option<Instant>,
    graph_view: GraphView,
    graph_zoom: usize,
    graph_pan: usize,
    comparison_series: Vec<MetricSample>,
    comparison_label: Option<String>,
    comparison_error: Option<String>,
    // Health checks are presentation-only; they never affect training.
    problem: Option<String>,
    warning: Option<String>,
    tok_s_history: Vec<f64>,
    last_progress_at: Option<Instant>,
    run_started_at: Option<Instant>,
    elapsed_seconds: Option<f64>,
    grad_norm: Option<f64>,
    skipped_updates: Option<u64>,
    training_active: bool,
    expected_lr_base: Option<f64>,
    expected_lr_total: Option<u64>,
    expected_lr_warmup: Option<u64>,
    lr_metadata_pending: bool,
    // last finished epoch line
    epoch_loss: Option<f64>,
    epoch_tokens: Option<u64>,
    epoch_updates: Option<u64>,
    // summary card
    wall: Option<String>,
    throughput: Option<String>,
    training_seconds: Option<f64>,
    optimizer_updates: Option<u64>,
    global_update_seen: bool,
    // chain tab
    chain_dir: PathBuf,
    checkpoints: Vec<(String, Option<f64>)>,
    resumed_from: Option<String>,
    prior_steps: Option<u64>,
    current_offset: Option<u64>,
    raw_lines: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HealthLevel {
    Normal,
    Warning,
    Problem,
}

struct HealthStatus {
    level: HealthLevel,
    reason: Option<String>,
    normal_label: &'static str,
}

impl HealthStatus {
    fn color(&self) -> Color {
        match self.level {
            HealthLevel::Normal => NORMAL_GREEN,
            HealthLevel::Warning => AMBER,
            HealthLevel::Problem => BRIGHT_RED,
        }
    }

    fn label(&self) -> String {
        match (&self.level, &self.reason) {
            (HealthLevel::Normal, _) => self.normal_label.to_string(),
            (HealthLevel::Warning, Some(reason)) => format!("WARNING: {reason}"),
            (HealthLevel::Problem, Some(reason)) => format!("PROBLEM: {reason}"),
            (HealthLevel::Warning, None) => "WARNING".to_string(),
            (HealthLevel::Problem, None) => "PROBLEM".to_string(),
        }
    }
}

impl RunState {
    fn ingest(&mut self, line: &str) {
        let line = strip_ansi(line);
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        if line.starts_with("model=") && line.contains("parameters=") {
            self.math = math::Values::default();
            self.configuration_source = None;
            self.corpus = None;
            self.vocab = None;
            self.width = None;
            self.memory = None;
            self.schedule = None;
            self.expected_lr_base = None;
            if !self.lr_metadata_pending {
                self.expected_lr_total = None;
                self.expected_lr_warmup = None;
            }
            self.lr_metadata_pending = false;
            self.loop_count = 1;
        }
        self.math.ingest(line);
        if let Some(loops) = parse_kv::<usize>(line, "loops=") {
            self.loop_count = loops.clamp(1, 32);
        }
        self.raw_lines.push(line.to_string());
        if self.raw_lines.len() > 400 {
            self.raw_lines.remove(0);
        }

        // `label  value` rows from the banner and summary panels
        for (label, field) in [
            ("corpus", &mut self.corpus),
            ("vocabulary", &mut self.vocab),
            ("width", &mut self.width),
            ("memory", &mut self.memory),
            ("schedule", &mut self.schedule),
            ("wall time", &mut self.wall),
            ("throughput", &mut self.throughput),
        ] {
            if let Some(v) = parse_field(line, label) {
                *field = Some(v);
            }
        }

        if line.contains("lr_schedule=") {
            self.lr_metadata_pending = true;
            self.expected_lr_total =
                parse_kv(line, "horizon=").or_else(|| parse_kv(line, "to_step="));
            self.expected_lr_warmup = Some(parse_kv(line, "warmup=").unwrap_or(0));
        }
        if let Some(schedule) = parse_field(line, "schedule") {
            self.expected_lr_base = parse_kv::<f64>(&schedule, "base ");
            // The human-readable banner groups horizons with commas.
            self.expected_lr_total = parse_kv(&schedule.replace(',', ""), "horizon ");
            if self.expected_lr_warmup.is_none() {
                // Fresh legacy schedules do not emit warmup metadata. Their
                // first update is base / warmup (or base without warmup).
                // Infer only when the rounded banner supports that value.
                if let (Some(base), Some(first)) =
                    (self.expected_lr_base, parse_kv::<f64>(&schedule, "first="))
                {
                    let warmup = (base / first).round();
                    if base > 0.0
                        && first > 0.0
                        && warmup.is_finite()
                        && warmup >= 1.0
                        && (base / warmup - first).abs() <= 1e-8
                    {
                        self.expected_lr_warmup =
                            Some(if warmup == 1.0 { 0 } else { warmup as u64 });
                    }
                }
            }
        }
        if line.contains("progress_schema=") {
            if self.configuration_source.take().is_some() {
                self.width = None;
                self.vocab = None;
                self.memory = None;
                self.math = math::Values::default();
            }
            // Start the stall clock even before the first update arrives.
            self.last_progress_at = Some(Instant::now());
            self.run_started_at = Some(Instant::now());
            self.training_active = true;
            self.updates_done = Some(0);
            self.updates_total = None;
            self.updates_remaining = None;
            self.optimizer_updates = None;
            self.global_update_seen = false;
            self.prior_steps = parse_kv(line, "prior_updates=");
            self.elapsed_seconds = None;
            self.grad_norm = None;
            self.skipped_updates = None;
            self.live_loss = None;
            self.loss_average = None;
            self.tok_s = None;
            self.eta = None;
            self.learning_rate = None;
            self.memory_used = None;
            self.memory_capacity = None;
            self.checkpoints.clear();
            self.resumed_from = None;
            self.epoch_loss = None;
            self.epoch_tokens = None;
            self.epoch_updates = None;
            self.wall = None;
            self.throughput = None;
            self.training_seconds = None;
            self.last_checkpoint = None;
            self.selected_checkpoint = None;
            self.checkpoint_saved_current = false;
            self.checkpoint_target = None;
            self.checkpoint_number = None;
            self.problem = None;
            self.warning = None;
            self.loss_series.clear();
            self.metric_series.clear();
            self.graph_from = None;
            self.graph_updated_at = None;
            self.tok_s_history.clear();
            self.progress_pct = None;
            self.progress_from = None;
            self.progress_updated_at = None;
            self.feed = None;
            self.feed_telemetry = parse_kv(line, "feed_telemetry=");
            self.feed_scroll.set(0);
            self.dream_active = false;
            self.dream_count = 0;
            self.dream_mode = None;
            self.dream_last_loss = None;
            self.dream_update = None;
        }

        if let Some(mode) = parse_kv::<String>(line, "memory_retrieval=") {
            self.memory_retrieval = Some(mode);
        }
        if let Some(growth) = parse_kv::<String>(line, "memory_growth=") {
            self.memory_growth = Some(growth);
        }
        if line.contains("loss_guard=on") {
            self.loss_guard_status = Some(format!(
                "on / high {}x / jump {}x / patience {}",
                parse_kv::<String>(line, "loss_guard_high_factor=").unwrap_or_default(),
                parse_kv::<String>(line, "loss_guard_jump_factor=").unwrap_or_default(),
                parse_kv::<String>(line, "loss_guard_patience=").unwrap_or_default()
            ));
        }
        if line.contains("loss_guard=halted") {
            self.loss_guard_status = Some("HALTED / no checkpoint".into());
            self.training_active = false;
            self.record_problem("loss blow-up guard halted training without checkpoint");
        }
        // Dream events are intentionally separate from ordinary loss samples:
        // a rehearsal loss must not become the live training loss or chart
        // series. Accept the structured schema emitted by the trainer and the
        // shorter legacy aliases so recorded logs remain useful.
        if line.contains("dream phase=start") {
            self.dream_active = true;
            self.dream_mode = parse_kv::<String>(line, "mode=");
            self.dream_update =
                parse_kv(line, "update=").or_else(|| parse_kv(line, "global_update="));
        }
        if line.contains("dream phase=end") {
            self.dream_active = false;
            // Restart the stall clock after the dream so its length is not
            // charged to the next training update.
            if self.training_active {
                self.last_progress_at = Some(Instant::now());
            }
            self.dream_count = self.dream_count.saturating_add(1);
            self.dream_mode = parse_kv::<String>(line, "mode=").or(self.dream_mode.take());
            self.dream_update = parse_kv(line, "update=").or(self.dream_update);
            self.dream_last_loss = parse_kv::<f64>(line, "dream_loss=")
                .or_else(|| parse_kv(line, "rehearsal_loss="))
                .or_else(|| parse_kv(line, "loss="))
                .filter(|value| value.is_finite());
        }

        if let Some(feed) = FeedState::parse(line) {
            self.feed = Some(feed);
        }

        let raw_loss = parse_kv::<f64>(line, "loss=").or_else(|| parse_kv(line, "loss "));
        let is_progress = line.contains("tokens_per_second=") || line.contains("tok/s");
        if is_progress {
            if let Some((done, total)) = parse_fraction(line, "training ") {
                self.updates_done = Some(done);
                self.updates_total = Some(total);
                self.updates_remaining = Some(total.saturating_sub(done));
            }
            self.last_progress_at = Some(Instant::now());
            self.training_active = true;
            // Problems remain visible until the next progress sample checks
            // whether the run recovered; unrelated log rows cannot clear them.
            self.problem = None;
            self.warning = None;
            if let Some(v) = raw_loss {
                if !v.is_finite() {
                    self.record_problem("loss is NaN/inf");
                } else {
                    // Compare against the preceding samples so the current
                    // spike cannot hide itself by raising its own average.
                    if let Some(average) = self.recent_loss_average() {
                        if v > average * 1.5 {
                            self.record_problem(format!(
                                "loss spike {v:.4} > 1.5x recent avg {average:.4}"
                            ));
                        } else if v > average * 1.25 {
                            self.record_warning(format!(
                                "loss rising {v:.4} vs recent avg {average:.4}"
                            ));
                        }
                    }
                    self.live_loss = Some(v);
                    self.loss_series.push(v);
                }
            }
            if let Some(p) = parse_pct(line) {
                let now = Instant::now();
                let current = self.displayed_progress_at(now);
                if self.progress_pct.is_none() {
                    self.progress_from = Some(p);
                    self.progress_updated_at = Some(now);
                } else if self.progress_pct != Some(p) {
                    // Start the next tween at the currently visible value so
                    // closely spaced updates never cause a backwards jump.
                    self.progress_from = Some(current);
                    self.progress_updated_at = Some(now);
                }
                self.progress_pct = Some(p);
            }
            // ui::Progress emits key=value fields when piped, and a bar
            // with space-separated fields on an interactive terminal.
            let speed = parse_kv::<f64>(line, "tokens_per_second=")
                .or_else(|| {
                    line.split_once("tok/s")?
                        .0
                        .split_whitespace()
                        .last()?
                        .parse()
                        .ok()
                })
                .filter(|v| v.is_finite());
            if let Some(speed) = speed {
                self.tok_s = Some(speed);
                if let Some(median) = self.running_tok_s_median() {
                    if speed < median * 0.6 {
                        self.record_problem(format!(
                            "speed {speed:.0} tok/s < 60% of median {median:.0}"
                        ));
                    } else if speed < median * 0.75 {
                        self.record_warning(format!(
                            "speed {speed:.0} tok/s below median {median:.0}"
                        ));
                    }
                }
                self.tok_s_history.push(speed.max(0.0));
                if self.tok_s_history.len() > 31 {
                    self.tok_s_history.remove(0);
                }
            }
            let sample = MetricSample {
                step: parse_kv::<f64>(line, "global_update=")
                    .or_else(|| parse_kv::<f64>(line, "optimizer_updates="))
                    .or_else(|| parse_fraction(line, "training ").map(|(done, _)| done as f64))
                    .filter(|v| v.is_finite()),
                loss: raw_loss.filter(|v| v.is_finite()),
                tokens_per_second: speed,
                learning_rate: parse_kv::<f64>(line, "learning_rate=").filter(|v| v.is_finite()),
            };
            if sample.has_value() {
                self.graph_from = self.metric_series.last().copied();
                self.graph_updated_at = Some(Instant::now());
                self.metric_series.push(sample);
                if self.metric_series.len() > 600 {
                    self.metric_series.remove(0);
                }
            }
            if let Some((_, eta)) = line.split_once("eta=").or_else(|| line.split_once("eta ")) {
                // ui::duration can contain spaces, e.g. `2h 14m 09s`.
                self.eta = Some(
                    eta.split_once(" last_checkpoint=")
                        .map_or(eta, |(eta, _)| eta)
                        .trim()
                        .to_string(),
                );
            }
            if self.loss_series.len() > 600 {
                self.loss_series.remove(0);
            }
        } else if line.starts_with("epoch ")
            && line.contains("updates=")
            && raw_loss.is_some_and(|v| v.is_finite())
        {
            // Epoch summaries are distinct from live progress samples.
            let v = raw_loss.expect("is_finite checked above");
            self.epoch_loss = Some(v);
            self.epoch_tokens = parse_kv(line, "tokens=");
            self.epoch_updates = parse_kv(line, "updates=");
            self.loss_series.push(v);
            if self.loss_series.len() > 600 {
                self.loss_series.remove(0);
            }
        } else if raw_loss.is_some_and(|v| !v.is_finite()) {
            self.record_problem("loss is NaN/inf");
        }
        if let Some(norm) = parse_kv::<f64>(line, "grad_norm=") {
            self.grad_norm = Some(norm);
        }
        if let Some(skipped) = parse_kv(line, "skipped_updates=") {
            self.skipped_updates = Some(skipped);
        }
        if let Some(elapsed) = parse_kv::<f64>(line, "elapsed_seconds=")
            && elapsed.is_finite()
            && elapsed >= 0.0
        {
            self.elapsed_seconds = Some(elapsed);
        }
        if let Some(t) = parse_kv::<f64>(line, "training_seconds=")
            && t.is_finite()
            && t >= 0.0
        {
            self.training_seconds = Some(t);
            self.training_active = false;
        }
        if let Some(u) = parse_kv(line, "optimizer_updates=") {
            if !self.global_update_seen {
                self.optimizer_updates = Some(u);
            }
            self.updates_done = Some(u);
        }
        if let Some(prior) = parse_kv(line, "prior_updates=") {
            self.prior_steps = Some(prior);
        }
        if let Some(v) = parse_kv::<f64>(line, "loss_average=")
            && v.is_finite()
        {
            self.loss_average = Some(v);
        }
        if let Some((done, total)) = parse_fraction(line, "optimizer_updates=") {
            self.updates_done = Some(done);
            self.updates_total = Some(total);
            self.updates_remaining = Some(total.saturating_sub(done));
        }
        if let Some(total) = parse_kv(line, "updates_total=") {
            self.updates_total = Some(total);
            if let Some(done) = self.updates_done {
                self.updates_remaining = Some(total.saturating_sub(done));
            }
        }
        if let Some(remaining) = parse_kv(line, "updates_remaining=") {
            self.updates_remaining = Some(remaining);
        }
        if let Some(done) = parse_kv(line, "global_update=") {
            self.optimizer_updates = Some(done);
            self.global_update_seen = true;
        }
        if is_progress
            && self
                .updates_done
                .zip(self.updates_total)
                .is_some_and(|(done, total)| done >= total)
        {
            self.training_active = false;
        }
        if let Some(lr) = parse_kv::<f64>(line, "learning_rate=") {
            if lr.is_finite() {
                self.learning_rate = Some(lr);
                self.check_learning_rate(lr);
            } else {
                self.record_problem("learning rate is NaN/inf");
            }
        }
        if let Some(number) = parse_kv(line, "checkpoint_number=") {
            self.checkpoint_number = Some(number);
        }
        if let Some((used, capacity)) = parse_fraction(line, "memory_occupancy=") {
            self.memory_used = Some(used);
            self.memory_capacity = Some(capacity);
        }
        if let Some(path) = line.split("checkpoint_target=").nth(1).map(str::trim)
            && !path.is_empty()
            && path != "-"
        {
            self.checkpoint_target = Some(path.to_string());
            self.checkpoint_number = checkpoint_number(path);
            // Follow the actual output directory rather than a default chain.
            if let Some(parent) = std::path::Path::new(path).parent() {
                self.chain_dir = if parent.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    parent.to_path_buf()
                };
            }
        }
        if let Some(path) = line.split("last_checkpoint=").nth(1).map(str::trim)
            && !path.is_empty()
            && path != "-"
        {
            self.last_checkpoint = Some(path.to_string());
            if self.prior_steps.is_some_and(|prior| prior > 0)
                && self.updates_done == Some(0)
                && self.resumed_from.is_none()
            {
                self.resumed_from = Some(path.to_string());
            }
            self.checkpoint_number = checkpoint_number(path);
        }
        if line.contains("resumed_from=") {
            let value = line.split("resumed_from=").nth(1).unwrap_or("").trim();
            // The checkpoint path is followed by structured metadata, but the
            // path itself may contain spaces.
            let path = value
                .split_once(" vocab=")
                .map_or(value, |(path, _)| path)
                .trim();
            if !path.is_empty() {
                self.resumed_from = Some(path.to_string());
            }
        }
        if let Some(s) = parse_kv(line, "prior_steps=") {
            self.prior_steps = Some(s);
        }
        if let Some(v) = parse_kv(line, "offset ") {
            // `--- ck32 (corpus offset 6200000) ---`
            self.current_offset = Some(v);
        }
        if let Some(rest) = line.split("saved_checkpoint=").nth(1) {
            let path = rest.trim();
            self.note_checkpoint(path);
        }
        if let Some(rest) = line.split("checkpoint written to ").nth(1) {
            self.note_checkpoint(strip_ansi(rest.trim_end()).trim());
        }
        if line.starts_with("error:")
            || line.starts_with("Error:")
            || line.starts_with("save failed:")
        {
            self.training_active = false;
            self.record_problem(line);
        }
    }

    fn current_step(&self) -> Option<u64> {
        self.prior_steps
            .zip(self.updates_done)
            .map(|(prior, done)| prior.saturating_add(done))
            .or(self.optimizer_updates)
    }

    fn checkpoint_context(&self) -> String {
        let current = feed::value(self.current_step());
        let previous = self
            .last_checkpoint
            .as_ref()
            .map(|path| format!("; previous checkpoint {path}"))
            .unwrap_or_default();
        if !self.training_active
            && let Some(problem) = &self.problem
        {
            return if self.checkpoint_saved_current {
                format!("Checkpoint saved{previous}; reported issue at step {current}: {problem}")
            } else {
                format!(
                    "New checkpoint save unconfirmed; reported issue at step {current}: {problem}{previous}"
                )
            };
        }
        if self.checkpoint_saved_current
            && !self.training_active
            && let Some(path) = &self.last_checkpoint
        {
            return path.clone();
        }
        if self.checkpoint_target.is_none() {
            if let Some(path) = &self.selected_checkpoint {
                return format!("Selected local artifact {path}; run save not confirmed");
            }
            return format!("No save target in this log; current step {current}{previous}");
        }
        if self.training_seconds.is_some()
            || self
                .updates_done
                .zip(self.updates_total)
                .is_some_and(|(done, total)| done >= total)
        {
            return format!(
                "Training computation complete at step {current}; waiting for saved_checkpoint event{previous}"
            );
        }
        if let Some(total) = self.updates_total {
            let end = match self.prior_steps {
                Some(prior) => format!("planned step {}", prior.saturating_add(total)),
                None => format!("{total} run-local updates; global end step unrecorded"),
            };
            let save = if self.last_checkpoint.is_some() {
                "Next save"
            } else {
                "First checkpoint"
            };
            return format!("{save} at run end ({end}); current step {current}{previous}");
        }
        format!("Save target configured; save step unrecorded; current step {current}{previous}")
    }

    fn use_header_dims(&mut self, path: &std::path::Path, dims: &str) {
        // Header hints fill checkpoint-only sessions, never overwrite a real
        // training banner or supply nonexistent corpus/schedule/loop telemetry.
        if self.width.is_some() || self.run_started_at.is_some() || self.training_active {
            return;
        }
        self.width = Some(dims.to_string());
        self.vocab = parse_kv::<u64>(dims, "vocab ").map(|n| n.to_string());
        self.memory = if dims.starts_with("TRFM ") {
            Some("not applicable (transformer)".into())
        } else {
            parse_kv::<u64>(dims, "slots ")
                .zip(parse_kv::<u64>(dims, "key "))
                .map(|(slots, key)| format!("{slots} slots, key width {key}"))
        };
        self.math.ingest_header_dims(dims);
        if self.current_step().is_none() {
            self.optimizer_updates = parse_kv(dims, "step ");
        }
        self.configuration_source = Some(format!(
            "Header hints from {} (bounded read; model/checksum not validated)",
            ui::terminal_text(&path.display().to_string())
        ));
    }

    fn elapsed_context(&self) -> String {
        self.wall
            .clone()
            .or_else(|| self.training_seconds.map(ui::duration))
            .or_else(|| {
                self.elapsed_seconds
                    .map(|elapsed| format!("{} (last report)", ui::duration(elapsed)))
            })
            .or_else(|| {
                self.run_started_at.map(|at| {
                    format!(
                        "{} since run connected",
                        ui::duration(at.elapsed().as_secs_f64())
                    )
                })
            })
            .unwrap_or_else(|| "elapsed time unrecorded".into())
    }

    fn note_checkpoint(&mut self, path: &str) {
        let path = path.trim();
        if path.is_empty() || path == "-" {
            return;
        }
        self.last_checkpoint = Some(path.to_string());
        self.checkpoint_saved_current = true;
        self.checkpoint_revision = self.checkpoint_revision.saturating_add(1);
        self.checkpoint_number = checkpoint_number(path);
        let name = PathBuf::from(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
        if !(name.ends_with(".pssa") || name.ends_with(".trfm")) {
            return;
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            self.chain_dir = if parent.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                parent.to_path_buf()
            };
        }
        let recorded_loss = self.epoch_loss.or(self.live_loss);
        if let Some((_, loss)) = self.checkpoints.iter_mut().find(|(n, _)| *n == name) {
            *loss = recorded_loss.or(*loss);
        } else {
            self.checkpoints.push((name, recorded_loss));
            self.checkpoints
                .sort_by_key(|(name, _)| checkpoint_sort_key(name));
        }
    }

    /// Scan real checkpoint files, preserving metrics attached to save events.
    /// Historical metrics from matching recorded logs are loaded by Timeline.
    fn refresh_chain(&mut self) {
        let Ok(entries) = std::fs::read_dir(&self.chain_dir) else {
            self.checkpoints.clear();
            return;
        };
        let mut names: Vec<(String, Option<f64>)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if (name.ends_with(".pssa") || name.ends_with(".trfm"))
                && entry.file_type().is_ok_and(|kind| kind.is_file())
            {
                let loss = self
                    .checkpoints
                    .iter()
                    .find(|(n, _)| *n == name)
                    .and_then(|(_, loss)| *loss)
                    .or_else(|| {
                        std::fs::read_to_string(entry.path().with_extension("loss"))
                            .ok()
                            .and_then(|s| s.trim().parse::<f64>().ok())
                            .filter(|loss| loss.is_finite())
                    });
                names.push((name, loss));
            }
        }
        names.sort_by_key(|(name, _)| checkpoint_sort_key(name));
        self.checkpoints = names;
    }

    fn record_problem(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        match &mut self.problem {
            Some(existing) if !existing.contains(&reason) => {
                existing.push_str("; ");
                existing.push_str(&reason);
            }
            Some(_) => {}
            None => self.problem = Some(reason),
        }
    }

    fn record_warning(&mut self, reason: impl Into<String>) {
        if self.problem.is_some() {
            return;
        }
        let reason = reason.into();
        match &mut self.warning {
            Some(existing) if !existing.contains(&reason) => {
                existing.push_str("; ");
                existing.push_str(&reason);
            }
            Some(_) => {}
            None => self.warning = Some(reason),
        }
    }

    fn recent_loss_average(&self) -> Option<f64> {
        let recent: Vec<f64> = self.loss_series.iter().rev().take(8).copied().collect();
        (!recent.is_empty()).then(|| recent.iter().sum::<f64>() / recent.len() as f64)
    }

    fn running_tok_s_median(&self) -> Option<f64> {
        if self.tok_s_history.len() < 3 {
            return None;
        }
        let mut values = self.tok_s_history.clone();
        values.sort_by(f64::total_cmp);
        Some(values[values.len() / 2])
    }

    fn check_learning_rate(&mut self, actual: f64) {
        if actual <= 0.0 {
            self.record_problem(format!("lr {actual:.3e} must be positive"));
            return;
        }
        let (Some(base), Some(total), Some(step), Some(warmup)) = (
            self.expected_lr_base,
            self.expected_lr_total,
            self.optimizer_updates,
            self.expected_lr_warmup,
        ) else {
            return;
        };
        let (Ok(total), Ok(step), Ok(warmup)) = (
            usize::try_from(total),
            usize::try_from(step),
            usize::try_from(warmup),
        ) else {
            return;
        };
        let Ok(expected) = crate::cli::learning_rate_for_update(base as f32, step, total, warmup)
        else {
            return;
        };
        let expected = f64::from(expected);
        // Allow rounding of the banner base (eight decimal places) and the
        // progress field, but flag a real schedule mismatch.
        let tolerance = expected.abs() * 0.02 + 1e-8;
        if (actual - expected).abs() > tolerance {
            self.record_problem(format!(
                "lr {actual:.3e} outside schedule (expected {expected:.3e})"
            ));
        }
    }

    fn health_status_at(&self, now: Instant) -> HealthStatus {
        let normal_label = if self.dream_active {
            "DREAMING"
        } else if self.training_active {
            "TRAINING"
        } else if self.checkpoint_target.is_some()
            && self.training_seconds.is_some()
            && !self.checkpoint_saved_current
        {
            "SAVE PENDING"
        } else if self.last_progress_at.is_some() || self.training_seconds.is_some() {
            // Recorded history deliberately has no live stall-clock timestamp.
            // Its completion summary still distinguishes it from an empty TUI.
            "DONE"
        } else {
            "WAITING"
        };
        if let Some(reason) = &self.problem {
            return HealthStatus {
                level: HealthLevel::Problem,
                reason: Some(reason.clone()),
                normal_label,
            };
        }
        if let Some(at) = self.last_progress_at {
            // A dream phase emits no progress lines until it ends, so it must
            // not count as a stall.
            if self.training_active
                && !self.dream_active
                && now.saturating_duration_since(at) > STALL_TIMEOUT
            {
                return HealthStatus {
                    level: HealthLevel::Problem,
                    reason: Some(format!(
                        "no progress line for {}s",
                        now.saturating_duration_since(at).as_secs()
                    )),
                    normal_label,
                };
            }
        }
        if let Some(reason) = &self.warning {
            return HealthStatus {
                level: HealthLevel::Warning,
                reason: Some(reason.clone()),
                normal_label,
            };
        }
        HealthStatus {
            level: HealthLevel::Normal,
            reason: None,
            normal_label,
        }
    }

    fn health_status(&self) -> HealthStatus {
        self.health_status_at(Instant::now())
    }

    fn reported_progress(&self) -> Option<f64> {
        self.progress_pct.or_else(|| {
            let (done, total) = self.updates_done.zip(self.updates_total)?;
            (total > 0).then(|| done as f64 * 100.0 / total as f64)
        })
    }

    fn displayed_progress_at(&self, now: Instant) -> f64 {
        let target = self.reported_progress().unwrap_or(0.0).clamp(0.0, 100.0);
        let Some(started) = self.progress_updated_at else {
            return target;
        };
        let from = self.progress_from.unwrap_or(target).clamp(0.0, 100.0);
        if !self.training_active {
            return target;
        }
        let t = (now.saturating_duration_since(started).as_secs_f64()
            / PROGRESS_INTERPOLATION.as_secs_f64())
        .clamp(0.0, 1.0);
        // Smoothstep gives the bar a gentle ease-in/ease-out rather than a
        // visibly mechanical jump between the trainer's samples.
        let eased = t * t * (3.0 - 2.0 * t);
        from + (target - from) * eased
    }

    fn displayed_metric_at(&self, now: Instant) -> Option<MetricSample> {
        let target = self.metric_series.last().copied()?;
        let Some(from) = self.graph_from else {
            return Some(target);
        };
        let Some(started) = self.graph_updated_at else {
            return Some(target);
        };
        let t = (now.saturating_duration_since(started).as_secs_f64()
            / GRAPH_INTERPOLATION.as_secs_f64())
        .clamp(0.0, 1.0);
        let eased = t * t * (3.0 - 2.0 * t);
        let lerp = |before: Option<f64>, after: Option<f64>| match (before, after) {
            (Some(before), Some(after)) => Some(before + (after - before) * eased),
            (None, Some(after)) => Some(after),
            (_, None) => None,
        };
        Some(MetricSample {
            step: target.step,
            loss: lerp(from.loss, target.loss),
            tokens_per_second: lerp(from.tokens_per_second, target.tokens_per_second),
            learning_rate: lerp(from.learning_rate, target.learning_rate),
        })
    }

    fn graph_data_len(&self) -> usize {
        self.metric_series.len().max(self.comparison_series.len())
    }

    fn visible_graph_range(&self) -> (usize, usize) {
        let len = self.graph_data_len();
        if len == 0 {
            return (0, 0);
        }
        let visible = (len / self.graph_zoom.max(1)).max(2).min(len);
        let max_start = len.saturating_sub(visible);
        let start = self.graph_pan.min(max_start);
        (start, start + visible)
    }

    fn zoom_graph(&mut self, inward: bool) {
        if inward {
            self.graph_zoom = (self.graph_zoom + 1).min(GRAPH_MAX_ZOOM);
        } else {
            self.graph_zoom = self.graph_zoom.saturating_sub(1).max(1);
        }
        self.graph_pan = self.graph_pan.min(self.graph_data_len().saturating_sub(1));
    }

    fn pan_graph(&mut self, right: bool) {
        let (start, end) = self.visible_graph_range();
        let step = ((end.saturating_sub(start)) / 4).max(1);
        let max_start = self
            .graph_data_len()
            .saturating_sub(end.saturating_sub(start));
        self.graph_pan = if right {
            self.graph_pan.saturating_add(step).min(max_start)
        } else {
            self.graph_pan.saturating_sub(step)
        };
    }

    fn reset_graph_navigation(&mut self) {
        self.graph_zoom = 1;
        self.graph_pan = 0;
    }
}

/// Pull `label  value` from a banner/summary row (two-space separated).
fn parse_field(line: &str, label: &str) -> Option<String> {
    let line = line.trim();
    // ui::field indents rows; ui::panel_field additionally frames them.
    let line = line.strip_prefix('│').unwrap_or(line).trim_start();
    let rest = line.strip_prefix(label)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let value = rest.trim().trim_end_matches('│').trim_end();
    if value.is_empty() || value.starts_with('=') {
        return None;
    }
    Some(value.to_string())
}

/// Feed strings are percent-encoded UTF-8, never raw terminal control codes.
fn parse_log_value(line: &str, key: &str) -> Option<String> {
    let value = line.split_whitespace().find_map(|part| {
        let (name, value) = part.split_once('=')?;
        (name == key).then_some(value)
    })?;
    let bytes = value.as_bytes();
    let mut decoded = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if decoded.len() >= 4096 {
            return None;
        }
        if bytes[i] == b'%' {
            let digits = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            decoded.push(u8::from_str_radix(digits, 16).ok()?);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    let text = String::from_utf8(decoded).ok()?;
    Some(ui::terminal_text(&text))
}

/// Pull `key=value` (or `key value`) numeric pairs out of a line.
fn parse_kv<T: std::str::FromStr>(line: &str, key: &str) -> Option<T> {
    let (at, _) = line.match_indices(key).find(|(at, _)| {
        *at == 0 || line[..*at].ends_with(|c: char| c.is_whitespace() || matches!(c, '(' | '│'))
    })?;
    let rest = &line[at + key.len()..];
    let token = rest
        .split_whitespace()
        .next()?
        .trim_end_matches(['%', ',', 's', ')']);
    token.parse().ok()
}

fn parse_pct(line: &str) -> Option<f64> {
    line.split_whitespace().find_map(|token| {
        let value = token.trim_matches(['(', ')']).strip_suffix('%')?;
        value.parse::<f64>().ok().filter(|v| v.is_finite())
    })
}

fn parse_fraction(line: &str, key: &str) -> Option<(u64, u64)> {
    let rest = line.split(key).nth(1)?.trim_start();
    let value = rest.split_whitespace().next()?;
    let (done, total) = value.split_once('/')?;
    let done = done.parse().ok()?;
    let total = total.parse().ok()?;
    (total > 0 && done <= total).then_some((done, total))
}

fn checkpoint_sort_key(name: &str) -> (u8, u64, String) {
    let number = checkpoint_number(name).unwrap_or(u64::MAX);
    (u8::from(number == u64::MAX), number, name.to_string())
}

fn checkpoint_number(path: &str) -> Option<u64> {
    let name = PathBuf::from(path)
        .file_name()?
        .to_string_lossy()
        .into_owned();
    let stem = name
        .strip_suffix(".pssa")
        .or_else(|| name.strip_suffix(".trfm"))?;
    stem.strip_prefix("ck")?.parse().ok()
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for skip in chars.by_ref() {
                if skip == 'm' || skip.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn run_app(
    rx: mpsc::Receiver<String>,
    chain_dir: PathBuf,
    compare_path: Option<PathBuf>,
    chats_dir: PathBuf,
) -> io::Result<()> {
    let mut chat = chat::Chat::new(chats_dir, chain_dir.clone());
    let mut setup = setup::Setup::default();
    let mut local = local::Local::new(chain_dir.clone());
    let mut help = keybindings::Help::default();
    let mut palette = overlay::Overlay::default();
    let mut training: Option<setup::TrainingRun> = None;
    let mut hf_login = hf::Login::new();
    let mut extras = extras::Extras::new(chain_dir.clone());
    let mut network = network::Network::new();
    let (_session, mut terminal) = session::Session::start()?;
    let mut background = HexBackground::default();
    let mut last_frame = Instant::now();
    let mut state = RunState {
        chain_dir: chain_dir.clone(),
        ..RunState::default()
    };
    let mut tab = 0usize;
    let mut last_chain_scan = std::time::Instant::now() - Duration::from_secs(60);
    let mut comparison = compare_path.map(|path| (path, comparison::ComparisonLog::default()));
    let mut input_closed = false;
    let mut next_frame = Instant::now();

    loop {
        // Drain stdin. A completed producer should leave the final dashboard
        // frame visible once, then let the wrapper restore the terminal. Opted-in
        // deliveries may finish after EOF; they never hold up the producer.
        while !input_closed {
            match rx.try_recv() {
                Ok(line) => {
                    extras.ingest(&line);
                    network.ingest(&line);
                    state.ingest(&line);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    input_closed = true;
                    if training.is_none() {
                        extras.eof(&state);
                        finish_piped_stream(&mut state, network.stream_eof());
                    }
                    break;
                }
            }
        }
        if !extras.remote_monitor() && last_chain_scan.elapsed() > Duration::from_secs(5) {
            state.refresh_chain();
            last_chain_scan = std::time::Instant::now();
        }
        if let Some((path, loader)) = &mut comparison {
            loader.poll(path, &mut state);
        }

        if Instant::now() >= next_frame
            || (input_closed && training.is_none() && !network.deliveries_pending())
        {
            let now = Instant::now();
            background.advance(now.saturating_duration_since(last_frame));
            last_frame = now;
            chat.poll();
            let training_finished = if let Some(run) = &mut training {
                let was_active = run.active();
                run.poll_with(&mut state, |line| {
                    extras.ingest(line);
                    network.ingest(line);
                });
                was_active && !run.active()
            } else {
                false
            };
            if training_finished {
                extras.eof(&state);
                if let Some(run) = training.as_ref() {
                    network.eof(run.succeeded().unwrap_or(false));
                }
            }
            hf_login.poll();
            extras.poll_with(&mut state, &mut tab, &mut chat, |event| {
                network.remote_line(event)
            });
            if poll_training_queue(&mut training, training_finished, |training| {
                network.poll(
                    &mut state,
                    tab,
                    training,
                    extras.training_busy(),
                    extras.remote_monitor(),
                )
            }) {
                chat.set_model_dir(state.chain_dir.clone());
                extras.set_chain_dir(state.chain_dir.clone());
            }
            if !extras.remote_monitor() {
                chat.set_model_dir(state.chain_dir.clone());
            }
            if extras.take_bell() {
                use std::io::Write;
                let _ = io::stdout().write_all(b"\x07");
                let _ = io::stdout().flush();
            }
            let checkpoint = if extras.remote_monitor() {
                None
            } else {
                preview::Preview::candidate(&state)
            };
            let context = state.checkpoint_context();
            state.preview.set_waiting_context(context);
            state
                .preview
                .poll(checkpoint, state.loop_count, extras.remote_monitor());
            state.hardware.poll(tab == keybindings::HARDWARE_TAB);
            local.poll(&mut setup, &mut state, extras.remote_monitor());
            terminal.draw(|f| {
                draw_with_background(
                    f,
                    &state,
                    tab,
                    &background,
                    Some(&mut chat),
                    Some(&mut setup),
                );
                if tab == HF_TAB {
                    hf_login.draw(f, feature_area(f.area()));
                }
                local.draw(f, tab);
                extras.draw(f, &state, tab, &chat);
                network.draw(f, tab);
                palette.draw(f);
                if help.open {
                    help.draw(f);
                }
            })?;
            next_frame = Instant::now() + FRAME_INTERVAL;
        }
        if input_closed && training.is_none() && !network.deliveries_pending() {
            break;
        }

        // Events may wake poll early. Gate drawing separately so key repeats
        // cannot drive animation above the frame cap. The dedicated stdin
        // reader still feeds the unbounded channel independently of rendering.
        if crossterm::event::poll(next_frame.saturating_duration_since(Instant::now()))? {
            let event = crossterm::event::read()?;
            match event {
                crossterm::event::Event::Mouse(mouse) => background.mouse(mouse.column, mouse.row),
                crossterm::event::Event::FocusLost | crossterm::event::Event::Resize(_, _) => {
                    background.leave();
                }
                _ => {}
            }
            if let crossterm::event::Event::Key(key) = event {
                if key.kind == crossterm::event::KeyEventKind::Press {
                    use keybindings::{Action, Context};
                    let old_tab = tab;
                    let context = if help.open {
                        Context::Help
                    } else if palette.open {
                        Context::Palette
                    } else {
                        network
                            .context(tab)
                            .or_else(|| extras.context(tab))
                            .unwrap_or_else(|| {
                                let editing = if tab == keybindings::LIMITS_TAB {
                                    state.resource_limits.borrow().editing()
                                } else {
                                    (tab == 5 && setup.editing()) || local.editing(tab)
                                };
                                Context::for_tab(tab, editing)
                            })
                    };
                    match keybindings::action(key, context) {
                        Some(Action::Quit) => break,
                        Some(Action::NextTab) => {
                            palette.open = false;
                            tab = (tab + 1) % TABS.len();
                            help.open = false;
                        }
                        Some(Action::PreviousTab) => {
                            tab = (tab + TABS.len() - 1) % TABS.len();
                        }
                        Some(Action::ToggleHelp) => {
                            palette.open = false;
                            help.open = !help.open;
                            help.scroll = 0;
                        }
                        Some(Action::TogglePalette) => {
                            help.open = false;
                            palette.toggle();
                        }
                        Some(Action::Palette) => match palette.key(key) {
                            Some(overlay::Action::Tab(next)) => tab = next,
                            Some(overlay::Action::Benchmark) => {
                                tab = BENCHMARK_TAB;
                                if state.training_active
                                    || extras.training_busy()
                                    || training.as_ref().is_some_and(|run| run.active())
                                {
                                    network.blocked("A training run is active; wait before starting a benchmark.");
                                } else {
                                    extras.start_benchmark();
                                }
                            }
                            Some(overlay::Action::Quit) => break,
                            None => {}
                        },
                        Some(Action::Hf) => hf_login.key(key),
                        Some(Action::Extras) => {
                            if training.as_ref().is_some_and(|run| run.active())
                                && ((tab == BENCHMARK_TAB
                                    && extras.context(tab) == Some(Context::Benchmark)
                                    && key.code == crossterm::event::KeyCode::Char('b'))
                                    || (tab == KAGGLE_TAB
                                        && matches!(
                                            key.code,
                                            crossterm::event::KeyCode::Enter
                                                | crossterm::event::KeyCode::Char('y' | 'Y')
                                        )))
                            {
                                network.blocked(
                                    "A trainer is still running; wait for its process to exit.",
                                );
                            } else {
                                extras.key(key, tab, &state);
                            }
                        }
                        Some(Action::Network) => network.key(key, &mut tab, &mut setup, &mut chat),
                        Some(Action::ScrollHelp(lines)) => {
                            help.scroll = help.scroll.saturating_add_signed(lines);
                        }
                        Some(Action::HelpTop) => help.scroll = 0,
                        Some(Action::HelpBottom) => help.scroll = u16::MAX,
                        Some(Action::OpenTab(next)) => {
                            tab = next;
                            help.open = false;
                            palette.open = false;
                        }
                        Some(Action::Device) => {
                            if let Some(backend) = state.device_picker.borrow_mut().key(key) {
                                setup.set_backend(backend);
                            }
                        }
                        Some(Action::Limits) => {
                            if let Some(limits) = state.resource_limits.borrow_mut().key(key) {
                                setup.set_limits(limits);
                            }
                        }
                        Some(Action::Heatmap) => {
                            if tab == 4 {
                                chat.heatmap = !chat.heatmap;
                            } else {
                                state.preview.heatmap = !state.preview.heatmap;
                            }
                        }
                        Some(Action::Preview) => state.preview.toggle(),
                        Some(Action::FeedScroll(lines)) => state
                            .feed_scroll
                            .set(state.feed_scroll.get().saturating_add_signed(lines)),
                        Some(Action::FeedTop) => state.feed_scroll.set(0),
                        Some(Action::FeedBottom) => state.feed_scroll.set(u16::MAX),
                        Some(Action::MathScroll(lines)) => state
                            .math_view
                            .scroll
                            .set(state.math_view.scroll.get().saturating_add_signed(lines)),
                        Some(Action::MathTop) => state.math_view.scroll.set(0),
                        Some(Action::MathBottom) => state.math_view.scroll.set(u16::MAX),
                        Some(Action::Chat) => chat.key(key),
                        Some(Action::Local) => local.key(&mut tab, key, &mut chat, &mut setup),
                        Some(Action::Setup) => {
                            if setup.key(key)
                                && let Some(run) = setup.launch(
                                    state.training_active
                                        || extras.training_busy()
                                        || training.as_ref().is_some_and(|run| run.active()),
                                )
                            {
                                run.initialize(&mut state);
                                network.started();
                                chat.set_model_dir(state.chain_dir.clone());
                                extras.set_chain_dir(state.chain_dir.clone());
                                training = Some(run);
                                tab = 0;
                            }
                        }
                        Some(Action::Pan(newer)) => state.pan_graph(newer),
                        Some(Action::CycleGraph) => state.graph_view = state.graph_view.next(),
                        Some(Action::Graph(view)) => state.graph_view = view,
                        Some(Action::Zoom(closer)) => state.zoom_graph(closer),
                        Some(Action::ResetGraph) => state.reset_graph_navigation(),
                        None => {}
                    }
                    if tab != old_tab {
                        if tab == keybindings::DEVICE_TAB {
                            let mut picker = state.device_picker.borrow_mut();
                            picker.set_backend(setup.backend());
                            picker.ensure_probe();
                        } else if tab == keybindings::LIMITS_TAB {
                            if let Ok(limits) = setup.limits() {
                                let applied = state.resource_limits.borrow().applied();
                                if limits != applied {
                                    state.resource_limits.borrow_mut().set_limits(limits);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn finish_piped_stream(state: &mut RunState, completion: Option<bool>) {
    if completion == Some(false) {
        state.record_problem(
            "Training stream ended without a completion summary followed by a confirmed final checkpoint save",
        );
    }
    state.training_active = false;
}

fn release_finished_training(training: &mut Option<setup::TrainingRun>, was_active: bool) -> bool {
    if was_active && training.as_ref().is_some_and(|run| !run.active()) {
        *training = None;
        true
    } else {
        false
    }
}

// Let the sweep consume completion before freeing the shared child slot. It may
// install the next trial, in which case release_finished_training keeps it alive.
fn poll_training_queue(
    training: &mut Option<setup::TrainingRun>,
    finished: bool,
    poll_queue: impl FnOnce(&mut Option<setup::TrainingRun>) -> bool,
) -> bool {
    let started = poll_queue(training);
    release_finished_training(training, finished);
    started
}

fn accent() -> Style {
    Style::new().fg(NORMAL_GREEN)
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .style(Style::new().bg(PANEL_BG))
        .borders(Borders::ALL)
        .border_style(accent().add_modifier(Modifier::DIM))
        .title(Line::styled(title, accent().add_modifier(Modifier::BOLD)))
}

/// Paint a panel's one-cell offset shadow before returning its original rect.
/// On narrow terminals and with NO_COLOR this is an exact no-op, preserving
/// the long-standing compact layout.
fn panel_area(f: &mut ratatui::Frame, area: Rect) -> Rect {
    let frame_area = f.area();
    shadow::paint(f, area, frame_area, shadow::color(PANEL_BG));
    area
}

fn divider(width: u16) -> Line<'static> {
    Line::styled("╌".repeat(width.into()), Style::new().fg(SECOND_ACCENT))
}

fn status_badge(health: &HealthStatus) -> Line<'static> {
    let label = match health.level {
        HealthLevel::Problem => "ERROR",
        HealthLevel::Warning | HealthLevel::Normal => match health.normal_label {
            "DONE" => "DONE",
            "SAVE PENDING" => "SAVE PENDING",
            "WAITING" => "WAITING",
            "DREAMING" => "DREAMING",
            _ => "TRAINING",
        },
    };
    let mut spans = vec![Span::styled(
        format!("[ {label} ]"),
        Style::new().fg(health.color()).add_modifier(Modifier::BOLD),
    )];
    if let Some(reason) = &health.reason {
        spans.push(Span::styled(
            format!("  {reason}"),
            Style::new().fg(health.color()),
        ));
    }
    Line::from(spans)
}

fn draw_header_stats(f: &mut ratatui::Frame, area: Rect, state: &RunState, health: &HealthStatus) {
    let area = panel_area(f, area);
    // Give a health reason the whole border title instead of truncating it
    // behind the telemetry label at the 80-column breakpoint.
    let title = health
        .reason
        .as_ref()
        .map(|reason| format!(" {reason} "))
        .unwrap_or_else(|| " live telemetry ".into());
    let block = panel("")
        .style(Style::new().bg(INSET_BG))
        .border_style(Style::new().fg(SECOND_ACCENT))
        .title(Line::styled(title, Style::new().fg(health.color())));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let done = state
        .updates_done
        .or(state.optimizer_updates)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".into());
    let total = state
        .updates_total
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".into());
    let badge = HealthStatus {
        level: health.level,
        reason: None,
        normal_label: health.normal_label,
    };
    let mut status = status_badge(&badge);
    status
        .spans
        .push(Span::styled(format!("  step {done}/{total}"), accent()));
    let loss = state
        .live_loss
        .or(state.epoch_loss)
        .map(charts::number)
        .unwrap_or_else(|| "-".into());
    let speed = state
        .tok_s
        .map(|v| format!("{v:.0}"))
        .unwrap_or_else(|| "-".into());
    let eta = state.eta.as_deref().unwrap_or("-");
    // The header has only three rows: a compact, explicitly labeled trace
    // with its value range and time direction instead of a block sparkline.
    let values = &state.loss_series[state.loss_series.len().saturating_sub(12)..];
    let finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    let range = if finite.is_empty() {
        "—".into()
    } else {
        let low = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!(
            "{}/{}/{}",
            charts::number(low),
            charts::number((low + high) / 2.0),
            charts::number(high)
        )
    };
    f.render_widget(
        Paragraph::new(vec![
            status,
            Line::from(vec![Span::styled(
                format!("loss {loss}  tok/s {speed}  ETA {eta}"),
                Style::new().fg(health.color()),
            )]),
            Line::styled(
                format!("loss {} {range} step → / lower better", loss_trace(state)),
                Style::new().fg(SECOND_ACCENT),
            ),
        ]),
        inner,
    );
}

fn loss_trace(state: &RunState) -> String {
    let values = &state.loss_series[state.loss_series.len().saturating_sub(12)..];
    charts::sparkline(values, 12)
}

// Empty states are instrument cards, not invented run data. A dim dot grid
// occupies only unused rows below the hints, never behind readable text.
fn draw_info_card(f: &mut ratatui::Frame, area: Rect, title: &str, lines: Vec<Line<'_>>) {
    let area = panel_area(f, area);
    let block = panel(title)
        .style(Style::new().bg(INSET_BG))
        .border_style(Style::new().fg(SECOND_ACCENT));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let rows = paragraph
        .line_count(inner.width)
        .min(usize::from(inner.height)) as u16;
    f.render_widget(paragraph, Rect::new(inner.x, inner.y, inner.width, rows));
    for y in (inner.y + rows + 1..inner.bottom()).step_by(2) {
        for x in (inner.x + 1..inner.right()).step_by(4) {
            f.buffer_mut()[(x, y)].set_symbol("·").set_fg(GRID_COLOR);
        }
    }
}

fn card_pair(f: &ratatui::Frame, area: Rect) -> std::rc::Rc<[Rect]> {
    let gap = if shadow::enabled(f.area()) { 1 } else { 0 };
    if area.width >= 70 {
        let usable = area.width.saturating_sub(gap);
        let left = usable / 2;
        std::rc::Rc::from([
            Rect::new(area.x, area.y, left, area.height),
            Rect::new(
                area.x.saturating_add(left).saturating_add(gap),
                area.y,
                usable.saturating_sub(left),
                area.height,
            ),
        ])
    } else {
        let usable = area.height.saturating_sub(gap);
        let top = usable / 2;
        std::rc::Rc::from([
            Rect::new(area.x, area.y, area.width, top),
            Rect::new(
                area.x,
                area.y.saturating_add(top).saturating_add(gap),
                area.width,
                usable.saturating_sub(top),
            ),
        ])
    }
}

fn progress_gradient(index: u16, width: u16) -> Color {
    let denominator = f64::from(width.saturating_sub(1).max(1));
    let t = (f64::from(index) / denominator).clamp(0.0, 1.0);
    let channel = |start: u8, end: u8| {
        (f64::from(start) + (f64::from(end) - f64::from(start)) * t).round() as u8
    };
    Color::Rgb(
        channel(0x16, 0x39),
        channel(0x73, 0xe0),
        channel(0x45, 0x7a),
    )
}

fn shine_position(width: u16) -> Option<u16> {
    if width == 0 {
        return None;
    }
    static ANIMATION_START: OnceLock<Instant> = OnceLock::new();
    let elapsed = ANIMATION_START.get_or_init(Instant::now).elapsed();
    Some(((elapsed.as_millis() / 45) as u16) % width)
}

#[derive(Clone, Copy, Debug)]
struct NeuronNode {
    x: f64,
    y: f64,
    parent: usize,
    cross_link: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NeuronFrame {
    cycle: u64,
    phase: f64,
    growth: f64,
    scale: f64,
    network_alpha: f64,
    collapsed: bool,
}

/// A tiny deterministic generator keeps the animation cheap without adding a
/// dependency or sharing mutable state with the trainer. The cycle number is
/// part of the seed, so each net is different while the collapsed endpoint is
/// always the same root dot.
#[derive(Clone, Copy)]
struct NeuronRng(u64);

impl NeuronRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 7;
        value ^= value >> 9;
        value ^= value << 8;
        self.0 = value;
        value
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn smoothstep(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn neuron_frame_at(elapsed: Duration) -> NeuronFrame {
    let cycle_millis = NEURON_CYCLE.as_millis();
    let elapsed_millis = elapsed.as_millis();
    let cycle = (elapsed_millis / cycle_millis) as u64;
    let phase = (elapsed_millis % cycle_millis) as f64 / cycle_millis as f64;
    if phase < NEURON_GROW_START {
        return NeuronFrame {
            cycle,
            phase,
            growth: 0.0,
            scale: 1.0,
            network_alpha: 1.0,
            collapsed: false,
        };
    }
    if phase < NEURON_GROW_END {
        let progress = (phase - NEURON_GROW_START) / (NEURON_GROW_END - NEURON_GROW_START);
        let growth = progress.powf(1.6);
        // Growth accelerates while the camera gently pulls back, leaving
        // the finished net near 80% of the panel rather than a tiny cluster.
        let zoom = smoothstep((progress - 0.18) / 0.82);
        return NeuronFrame {
            cycle,
            phase,
            growth,
            scale: 1.0 - 0.12 * zoom,
            network_alpha: 1.0,
            collapsed: false,
        };
    }
    if phase < NEURON_COLLAPSE_END {
        let collapse = (phase - NEURON_GROW_END) / (NEURON_COLLAPSE_END - NEURON_GROW_END);
        return NeuronFrame {
            cycle,
            phase,
            growth: 1.0,
            scale: 0.88 * (1.0 - smoothstep(collapse)) + 0.03 * smoothstep(collapse),
            network_alpha: 1.0 - smoothstep(collapse),
            collapsed: false,
        };
    }
    NeuronFrame {
        cycle,
        phase,
        growth: 1.0,
        scale: 0.0,
        network_alpha: 0.0,
        collapsed: true,
    }
}

fn neuron_network(cycle: u64) -> Vec<NeuronNode> {
    let seed = 0x9e37_79b9_7f4a_7c15_u64.wrapping_add(cycle.wrapping_mul(0x517c_c1b7_2722_0a95));
    let mut rng = NeuronRng::new(seed);
    let mut nodes = Vec::with_capacity(NEURON_NODE_COUNT);
    nodes.push(NeuronNode {
        x: 0.0,
        y: 0.0,
        parent: 0,
        cross_link: None,
    });
    let rotation = rng.unit() * std::f64::consts::TAU;
    for index in 1..NEURON_NODE_COUNT {
        // A jittered golden-angle spread grows outwards without piling up
        // random walks near the seed. Short local edges keep the tendril look.
        let angle = rotation + index as f64 * 2.399963229728653 + (rng.unit() - 0.5) * 0.5;
        let radius =
            (index as f64 / (NEURON_NODE_COUNT - 1) as f64).sqrt() * (0.9 + rng.unit() * 0.1);
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;
        let distance = |node: &NeuronNode| (node.x - x).powi(2) + (node.y - y).powi(2);
        let parent = (0..index)
            .min_by(|&a, &b| distance(&nodes[a]).total_cmp(&distance(&nodes[b])))
            .unwrap();
        // Join neighbouring branches after their nodes appear. At most one
        // extra edge per alternate node bounds both generation and painting.
        let cross_link = if index > 8 && index % 2 == 0 {
            (1..index)
                .filter(|&other| {
                    other != parent
                        && other != nodes[parent].parent
                        && nodes[other].parent != parent
                })
                .min_by(|&a, &b| distance(&nodes[a]).total_cmp(&distance(&nodes[b])))
        } else {
            None
        };
        nodes.push(NeuronNode {
            x,
            y,
            parent,
            cross_link,
        });
    }
    // Fit both axes independently; the renderer adapts x to the braille
    // viewport aspect. Leave a margin even before the gentle camera zoom.
    let extent_x = nodes
        .iter()
        .map(|node| node.x.abs())
        .fold(0.0_f64, f64::max);
    let extent_y = nodes
        .iter()
        .map(|node| node.y.abs())
        .fold(0.0_f64, f64::max);
    for node in nodes.iter_mut().skip(1) {
        node.x *= 0.94 / extent_x;
        node.y *= 0.94 / extent_y;
    }
    nodes
}

fn neuron_green(intensity: f64) -> Color {
    let intensity = intensity.clamp(0.0, 1.0);
    Color::Rgb(
        (f64::from(0x39) * intensity).round() as u8,
        (f64::from(0xe0) * intensity).round() as u8,
        (f64::from(0x7a) * intensity).round() as u8,
    )
}

#[cfg(test)]
fn draw_neuron_animation_at(f: &mut ratatui::Frame, area: Rect, elapsed: Duration) {
    let frame = neuron_frame_at(elapsed);
    draw_neuron_frame(f, area, frame, "");
}

fn draw_neuron_frame(f: &mut ratatui::Frame, area: Rect, frame: NeuronFrame, title: &str) {
    let area = panel_area(f, area);
    // Preserve polish's untitled black canvas and phase 9's reusable renderer.
    let block = panel(title).style(Style::new().bg(Color::Black));
    let inner = block.inner(area);
    if inner.width < 2 || inner.height < 2 {
        f.render_widget(block, area);
        return;
    }
    // No topology work is needed for the seed-only part of the cycle or a
    // panel too small to draw. Active frames stay bounded at 80 nodes/114 edges.
    let nodes = if frame.growth > 0.0 && !frame.collapsed {
        neuron_network(frame.cycle)
    } else {
        Vec::new()
    };
    // Braille cells have two by four dots. Match the virtual camera's aspect
    // to those pixels, keeping the seed circular on wide and tall panels.
    let pixel_width = f64::from(inner.width) * 2.0 - 1.0;
    let pixel_height = f64::from(inner.height) * 4.0 - 1.0;
    let aspect = pixel_width / pixel_height;
    let dot = [(-1.0, 0.0), (0.0, -1.0), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)].map(|(x, y)| {
        (
            ((pixel_width / 2.0).floor() + x + 0.25) * 2.0 * aspect / pixel_width - aspect,
            1.0 - ((pixel_height / 2.0).floor() + y + 0.25) * 2.0 / pixel_height,
        )
    });
    let canvas = Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .background_color(Color::Black)
        .x_bounds([-aspect, aspect])
        .y_bounds([-1.0, 1.0])
        .paint(move |ctx| {
            if !frame.collapsed {
                for (index, node) in nodes.iter().enumerate().skip(1) {
                    let node_progress =
                        frame.growth * (NEURON_NODE_COUNT + 1) as f64 - (index - 1) as f64;
                    if node_progress <= 0.0 {
                        continue;
                    }
                    for (target, delay) in std::iter::once((node.parent, 0.0))
                        .chain(node.cross_link.map(|target| (target, 1.0)))
                    {
                        let edge_progress = smoothstep(node_progress - delay);
                        if edge_progress <= 0.0 {
                            continue;
                        }
                        let parent = nodes[target];
                        let end_x = parent.x + (node.x - parent.x) * edge_progress;
                        let end_y = parent.y + (node.y - parent.y) * edge_progress;
                        ctx.draw(&CanvasLine::new(
                            parent.x * frame.scale * aspect,
                            parent.y * frame.scale,
                            end_x * frame.scale * aspect,
                            end_y * frame.scale,
                            neuron_green(0.42 * frame.network_alpha * edge_progress),
                        ));
                    }
                }
                for (index, node) in nodes.iter().enumerate().skip(1) {
                    let node_progress =
                        frame.growth * (NEURON_NODE_COUNT + 1) as f64 - index as f64;
                    if node_progress <= 0.0 {
                        continue;
                    }
                    // Fade in only once the tendril has reached this node.
                    let fade = smoothstep(node_progress) * frame.network_alpha;
                    ctx.draw(&Points {
                        coords: &[(node.x * frame.scale * aspect, node.y * frame.scale)],
                        color: neuron_green(fade),
                    });
                }
            }
            // A fixed five-pixel filled seed stays visible as the net folds
            // into it. Its size and position never change across cycle seams.
            ctx.draw(&Points {
                coords: &dot,
                color: neuron_green(1.0),
            });
        });
    f.render_widget(canvas, area);
}

fn progress_label(state: &RunState, width: u16) -> String {
    if state.progress_pct.is_none() && state.updates_total.is_none() {
        return "No training progress recorded".into();
    }
    // Numeric labels use reported progress, not the visual bar's tween.
    let pct = state
        .reported_progress()
        .map_or_else(|| "unrecorded".into(), |pct| format!("{pct:>3.0}%"));
    let done = state
        .updates_done
        .or(state.optimizer_updates)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());
    let total = state
        .updates_total
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());
    let eta = state.eta.as_deref().unwrap_or("-");
    if width >= 48 {
        format!("{pct}   updates {done}/{total}   ETA {eta}")
    } else if width >= 28 {
        format!("{pct}  {done}/{total}  ETA {eta}")
    } else {
        format!("{pct} {done}/{total}")
    }
}

fn draw_progress(f: &mut ratatui::Frame, area: Rect, state: &RunState, pct: f64) {
    if area.height == 1 {
        let label = progress_label(state, area.width);
        let meter_width = area.width.saturating_sub(label.len() as u16 + 1);
        let meter = charts::bar(pct / 100.0, meter_width as usize);
        for (i, glyph) in meter.chars().enumerate() {
            f.buffer_mut().set_string(
                area.x + i as u16,
                area.y,
                glyph.to_string(),
                Style::new().fg(progress_gradient(i as u16, meter_width)),
            );
        }
        f.buffer_mut().set_stringn(
            area.x + meter_width,
            area.y,
            format!(" {label}"),
            (area.width - meter_width) as usize,
            accent(),
        );
        return;
    }
    let area = panel_area(f, area);
    let block = panel(" progress / completed training steps ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.is_empty() {
        return;
    }

    let ratio = (pct / 100.0).clamp(0.0, 1.0);
    let total_eighths = (ratio * f64::from(inner.width) * 8.0).round() as u16;
    let full_cells = total_eighths / 8;
    let partial = usize::from(total_eighths % 8);
    let shine = shine_position(inner.width);
    for index in 0..inner.width {
        let (symbol, mut style) = if index < full_cells {
            (
                BRAILLE_LEVELS[7],
                Style::new().fg(progress_gradient(index, inner.width)),
            )
        } else if index == full_cells && partial > 0 {
            (
                BRAILLE_LEVELS[partial - 1],
                Style::new().fg(progress_gradient(index, inner.width)),
            )
        } else {
            ("·", Style::new().fg(Color::Rgb(0x0d, 0x2b, 0x1d)))
        };
        let filled = index < full_cells || (index == full_cells && partial > 0);
        if shine == Some(index) && filled {
            // A single-cell glint keeps the bar lively without obscuring its
            // green gradient or turning it into a second status color.
            style = Style::new().fg(Color::Rgb(0x9a, 0xff, 0xb9));
        }
        style = style.bg(Color::Black);
        if let Some(cell) = f.buffer_mut().cell_mut((inner.x + index, inner.y)) {
            cell.set_symbol(symbol);
            cell.set_style(style);
        }
    }
    if inner.height >= 2 {
        let label = progress_label(state, inner.width);
        f.buffer_mut().set_stringn(
            inner.x,
            inner.y + 1,
            label,
            usize::from(inner.width),
            Style::new().fg(NORMAL_GREEN).add_modifier(Modifier::BOLD),
        );
    }
}

// The additive feature tabs use the same content rectangle as the shell.
fn feature_area(screen: Rect) -> Rect {
    let area = HexBackground::content_area(screen);
    if area.width < 30 || area.height < 10 {
        return area;
    }
    let header = if screen.width >= 80 && screen.height >= 22 {
        7
    } else {
        5
    };
    Rect::new(
        area.x,
        area.y + header,
        area.width,
        area.height.saturating_sub(header),
    )
}

/// Measure the rendered data region, not the panel allocation. Also check the
/// entire chart border so adjacent panels/shadows cannot silently erase it.
#[cfg(test)]
fn assert_chart_rows(buffer: &ratatui::buffer::Buffer, title: &str, minimum: u16) {
    for top in buffer.area.y..buffer.area.bottom() {
        for left in buffer.area.x..buffer.area.right() {
            if buffer[(left, top)].symbol() != "┌" {
                continue;
            }
            let Some(right) =
                (left + 1..buffer.area.right()).find(|&x| buffer[(x, top)].symbol() == "┐")
            else {
                continue;
            };
            let heading: String = (left..=right).map(|x| buffer[(x, top)].symbol()).collect();
            if !heading.contains(title) {
                continue;
            }
            let bottom = (top + 1..buffer.area.bottom())
                .find(|&y| buffer[(left, y)].symbol() == "└")
                .expect("chart bottom border must stay in the screen");
            assert_eq!(buffer[(right, bottom)].symbol(), "┘", "{title}");
            for y in top + 1..bottom {
                assert_eq!(buffer[(left, y)].symbol(), "│", "{title}: left border");
                assert_eq!(buffer[(right, y)].symbol(), "│", "{title}: right border");
            }
            let axis = (top + 1..bottom)
                .find(|&y| {
                    (left + 1..right - 1).any(|x| {
                        buffer[(x, y)].symbol() == "└"
                            && (x + 1..right).all(|xx| buffer[(xx, y)].symbol() == "─")
                    })
                })
                .expect("chart must have a separate x-axis stroke");
            assert_eq!(axis + 2, bottom, "{title}: separate tick-label row");
            let plot_rows = axis - top - 1;
            assert!(
                plot_rows >= minimum,
                "{title}: only {plot_rows} actual plot rows, need {minimum} at {:?}",
                buffer.area
            );
            return;
        }
    }
    panic!("missing chart {title} at {:?}", buffer.area);
}

#[cfg(test)]
fn draw(f: &mut ratatui::Frame, state: &RunState, tab: usize) {
    draw_with_background(f, state, tab, &HexBackground::default(), None, None);
}

fn draw_with_background(
    f: &mut ratatui::Frame,
    state: &RunState,
    tab: usize,
    background: &HexBackground,
    chat: Option<&mut chat::Chat>,
    setup: Option<&mut setup::Setup>,
) {
    let area = f.area();
    if area.is_empty() {
        return;
    }
    f.render_widget(
        Block::default().style(Style::new().fg(Color::Gray).bg(Color::Black)),
        area,
    );
    background.draw(f);
    let area = HexBackground::content_area(area);
    let health = state.health_status();
    // Do not squeeze bordered widgets into one-cell fragments on tiny screens.
    if area.width < 30 || area.height < 10 {
        if tab == 5 {
            if let Some(setup) = setup {
                setup.draw(f, area);
            } else {
                setup::Setup::default().draw(f, area);
            }
            return;
        }
        if tab == 4
            && let Some(chat) = chat
        {
            chat.draw(f, area);
            return;
        }
        let detail = match tab {
            keybindings::SAMPLE_TAB => state.preview.note.clone(),
            keybindings::HARDWARE_TAB => "CPU / RAM / GPU telemetry / enlarge for detail".into(),
            keybindings::MATH_TAB => "PSSA equations / live dimensions / enlarge to read".into(),
            keybindings::DEVICE_TAB => "CPU / CUDA / WebGPU / Enter selects".into(),
            keybindings::LIMITS_TAB => "Threads / RAM / batch / tokens / Enter edits".into(),
            0 => {
                let progress = state
                    .progress_pct
                    .map(|pct| format!("{pct:.0}%"))
                    .unwrap_or_else(|| "-".into());
                let loss = state
                    .live_loss
                    .or(state.epoch_loss)
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".into());
                format!("{progress}  loss {loss}")
            }
            1 => format!(
                "{} checkpoints in {}",
                state.checkpoints.len(),
                state.chain_dir.display()
            ),
            2 => format!("width {}", state.width.as_deref().unwrap_or("-")),
            4 => "Local checkpoint chat / enlarge to compose".into(),
            _ => state.feed.as_ref().map_or_else(
                || "Waiting for feed samples".into(),
                |feed| {
                    format!(
                        "{} / {} rows / {} tokens",
                        feed.dataset,
                        feed::value(feed.rows),
                        feed::value(feed.tokens)
                    )
                },
            ),
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    format!("{} / PSSA", TABS[tab.min(TABS.len() - 1)]),
                    accent(),
                ),
                status_badge(&health),
                Line::from(detail),
                Line::from("q quit / Tab tabs / ? help"),
                Line::from("Enlarge for full view"),
            ]),
            area,
        );
        return;
    }
    let large_title = f.area().width >= 80 && f.area().height >= 22;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(if large_title { 5 } else { 3 }),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);
    let header = chunks[0];
    if large_title {
        f.render_widget(
            Paragraph::new(PSSA_LOGO.map(|row| Line::styled(row, accent())).to_vec()),
            Rect::new(header.x, header.y, 24, 5),
        );
        draw_header_stats(
            f,
            Rect::new(header.x + 24, header.y, header.width - 24, 5),
            state,
            &health,
        );
    } else {
        let title = if area.width >= 45 {
            "PSSA / pssa tui  Ctrl+C quit  Tab tabs  F1 help"
        } else {
            "PSSA / pssa tui  F1 help"
        };
        f.render_widget(
            Paragraph::new(vec![Line::styled(title, accent()), status_badge(&health)]),
            header,
        );
    }
    // Derive the width from the actual labels, padding and separators so adding
    // a tab cannot silently clip its name or steal space from the controls hint.
    // The complete strip uses compact separators when it fits; otherwise
    // complete feature groups keep every tab discoverable at 80+ columns.
    let full_width = TABS.iter().map(|tab| tab.len()).sum::<usize>() + TABS.len() - 1;
    let (first_tab, end_tab) = if usize::from(area.width) < full_width + 9 && f.area().width >= 80 {
        if tab < HF_TAB {
            (0, HF_TAB)
        } else if tab < keybindings::SAMPLE_TAB {
            (HF_TAB, keybindings::SAMPLE_TAB)
        } else if tab < keybindings::LIBRARY_TAB {
            (keybindings::SAMPLE_TAB, keybindings::LIBRARY_TAB)
        } else if tab < keybindings::BACKUP_TAB {
            (keybindings::LIBRARY_TAB, keybindings::BACKUP_TAB)
        } else if tab < keybindings::SUPPORT_TAB {
            (keybindings::BACKUP_TAB, keybindings::SUPPORT_TAB)
        } else {
            (keybindings::SUPPORT_TAB, TABS.len())
        }
    } else {
        (0, TABS.len())
    };
    let visible_tabs = &TABS[first_tab..end_tab];
    let compact_tabs = first_tab == 0 && end_tab == TABS.len();
    let tabs_width = if compact_tabs {
        visible_tabs.iter().map(|tab| tab.len()).sum::<usize>() + visible_tabs.len() - 1
    } else {
        visible_tabs.iter().map(|tab| tab.len() + 2).sum::<usize>() + (visible_tabs.len() - 1) * 3
    } as u16;
    let nav = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(tabs_width), Constraint::Min(0)])
        .split(chunks[1]);
    if area.width < tabs_width {
        f.render_widget(
            Paragraph::new(format!(
                " {} / Tab tabs / F1 help",
                TABS[tab.min(TABS.len() - 1)]
            ))
            .style(accent()),
            nav[0],
        );
    } else {
        f.render_widget(
            Tabs::new(visible_tabs.iter().copied())
                .select(tab.saturating_sub(first_tab))
                .style(accent().add_modifier(Modifier::DIM))
                .highlight_style(
                    accent()
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                        .remove_modifier(Modifier::DIM),
                )
                .divider(if compact_tabs { "·" } else { " / " })
                .padding(
                    if compact_tabs { "" } else { " " },
                    if compact_tabs { "" } else { " " },
                ),
            nav[0],
        );
    }
    if large_title {
        f.render_widget(
            Paragraph::new(if nav[1].width < 32 {
                " F1 help "
            } else {
                match tab {
                    0 => "F1 help / g/1-7 views / +/- zoom",
                    4 => "F1 help / /help commands",
                    5 => "F1 help / arrows move / Enter edit",
                    HF_TAB..=BENCHMARK_TAB => "F1 help / Ctrl+K palette",
                    _ => "F1 help / arrows tabs / q quit",
                }
            })
            .right_aligned(),
            nav[1],
        );
    }
    f.render_widget(Paragraph::new(divider(area.width)), chunks[2]);

    match tab {
        keybindings::SAMPLE_TAB => state.preview.draw(f, chunks[3], false),
        keybindings::HARDWARE_TAB => state.hardware.draw(f, chunks[3]),
        keybindings::MATH_TAB => state.math_view.draw(f, chunks[3], state),
        keybindings::DEVICE_TAB => state.device_picker.borrow_mut().draw(f, chunks[3]),
        keybindings::LIMITS_TAB => state.resource_limits.borrow_mut().draw(f, chunks[3]),
        0 => draw_monitor(f, chunks[3], state),
        1 => draw_chain(f, chunks[3], state),
        2 => draw_model(f, chunks[3], state),
        4 => {
            if let Some(chat) = chat {
                chat.draw(f, chunks[3]);
            } else {
                chat::Chat::new(PathBuf::from("chats"), state.chain_dir.clone()).draw(f, chunks[3]);
            }
        }
        5 => {
            if let Some(setup) = setup {
                setup.draw(f, chunks[3]);
            } else {
                setup::Setup::default().draw(f, chunks[3]);
            }
        }
        HF_TAB.. => {} // rendered by independent feature modules after the shell
        _ => draw_feed(f, chunks[3], state),
    }
}

fn draw_feed(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    feed::draw(f, area, state);
}

#[cfg(test)]
fn draw_feed_at(f: &mut ratatui::Frame, area: Rect, state: &RunState, _elapsed: Duration) {
    feed::draw(f, area, state);
}

#[derive(Clone, Copy)]
enum MetricKind {
    Loss,
    Perplexity,
    TokensPerSecond,
    LearningRate,
}

fn metric_value(sample: MetricSample, kind: MetricKind) -> Option<f64> {
    let value = match kind {
        MetricKind::Loss => sample.loss?,
        MetricKind::Perplexity => sample.loss?.exp().min(1.0e9),
        MetricKind::TokensPerSecond => sample.tokens_per_second?,
        MetricKind::LearningRate => sample.learning_rate?,
    };
    value.is_finite().then_some(value.max(0.0))
}

fn moving_loss_at(
    series: &[MetricSample],
    index: usize,
    latest: Option<MetricSample>,
) -> Option<f64> {
    let start = index.saturating_sub(7);
    let mut values = Vec::new();
    for i in start..=index {
        let sample = if i + 1 == series.len() {
            latest.unwrap_or(series[i])
        } else {
            series[i]
        };
        if let Some(loss) = sample.loss.filter(|v| v.is_finite()) {
            values.push(loss);
        }
    }
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn metric_points(
    series: &[MetricSample],
    kind: MetricKind,
    start: usize,
    end: usize,
    latest: Option<MetricSample>,
) -> Vec<(f64, f64)> {
    (start..end.min(series.len()))
        .filter_map(|index| {
            let sample = if index + 1 == series.len() {
                latest.unwrap_or(series[index])
            } else {
                series[index]
            };
            let value = if matches!(kind, MetricKind::Loss) {
                sample.loss
            } else {
                metric_value(sample, kind)
            };
            let value = match value {
                Some(value) if value.is_finite() => value.max(0.0),
                Some(_) => f64::NAN,
                // Progress-only steps are not invalid loss measurements.
                None if matches!(kind, MetricKind::Loss) => return None,
                None => f64::NAN,
            };
            Some((index as f64, value))
        })
        .collect()
}

fn moving_loss_points(
    series: &[MetricSample],
    start: usize,
    end: usize,
    latest: Option<MetricSample>,
) -> Vec<(f64, f64)> {
    (start..end.min(series.len()))
        .filter_map(|index| {
            // The average smooths measured losses, skipping unmeasured steps.
            let loss = series[index].loss?;
            let value = if loss.is_finite() {
                moving_loss_at(series, index, latest).unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            Some((index as f64, value))
        })
        .collect()
}

fn normalize_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let bounds: Option<(f64, f64)> = points
        .iter()
        .map(|(_, y)| *y)
        .filter(|y| y.is_finite())
        .fold(None, |bounds, y| {
            Some(match bounds {
                Some((min, max)) => (min.min(y), max.max(y)),
                None => (y, y),
            })
        });
    let Some((min, max)) = bounds else {
        return Vec::new();
    };
    if (max - min).abs() <= f64::EPSILON {
        return points
            .iter()
            .map(|(x, y)| (*x, if y.is_finite() { 0.5 } else { f64::NAN }))
            .collect();
    }
    points
        .iter()
        .map(|(x, y)| (*x, (*y - min) / (max - min)))
        .collect()
}

fn draw_graph(f: &mut ratatui::Frame, area: Rect, state: &RunState, now: Instant) {
    let view = state.graph_view;
    if view == GraphView::Memory {
        draw_memory_graph(f, area, state);
        return;
    }
    let (start, end) = state.visible_graph_range();
    if end == 0 {
        let waiting_area = panel_area(f, area);
        if view == GraphView::Comparison
            && let Some(error) = &state.comparison_error
        {
            f.render_widget(
                Paragraph::new(error.as_str())
                    .wrap(Wrap { trim: false })
                    .block(panel(" graph / comparison unavailable ")),
                waiting_area,
            );
        } else {
            f.render_widget(
                Paragraph::new("Waiting for progress samples").block(panel(" graph / waiting ")),
                waiting_area,
            );
        }
        return;
    }

    let latest = state.displayed_metric_at(now);
    let (
        mut loss,
        mut moving_loss,
        mut compare_loss,
        mut perplexity,
        mut tokens,
        mut learning_rate,
    ) = match view {
        GraphView::Loss => (
            Some(metric_points(
                &state.metric_series,
                MetricKind::Loss,
                start,
                end,
                latest,
            )),
            Some(moving_loss_points(&state.metric_series, start, end, latest)),
            None,
            None,
            None,
            None,
        ),
        GraphView::Perplexity => (
            None,
            None,
            None,
            Some(metric_points(
                &state.metric_series,
                MetricKind::Perplexity,
                start,
                end,
                latest,
            )),
            None,
            None,
        ),
        GraphView::TokensPerSecond => (
            None,
            None,
            None,
            None,
            Some(metric_points(
                &state.metric_series,
                MetricKind::TokensPerSecond,
                start,
                end,
                latest,
            )),
            None,
        ),
        GraphView::LearningRate => (
            None,
            None,
            None,
            None,
            None,
            Some(metric_points(
                &state.metric_series,
                MetricKind::LearningRate,
                start,
                end,
                latest,
            )),
        ),
        GraphView::Comparison => (
            None,
            Some(moving_loss_points(&state.metric_series, start, end, latest)),
            Some(moving_loss_points(
                &state.comparison_series,
                start,
                end,
                None,
            )),
            None,
            None,
            None,
        ),
        GraphView::All => (
            Some(normalize_points(&metric_points(
                &state.metric_series,
                MetricKind::Loss,
                start,
                end,
                latest,
            ))),
            None,
            None,
            Some(normalize_points(&metric_points(
                &state.metric_series,
                MetricKind::Perplexity,
                start,
                end,
                latest,
            ))),
            Some(normalize_points(&metric_points(
                &state.metric_series,
                MetricKind::TokensPerSecond,
                start,
                end,
                latest,
            ))),
            Some(normalize_points(&metric_points(
                &state.metric_series,
                MetricKind::LearningRate,
                start,
                end,
                latest,
            ))),
        ),
        GraphView::Memory => unreachable!(),
    };
    // Never mix training-step coordinates with observation indices, including
    // when comparing a modern log against one without step telemetry.
    let ordered_steps = |samples: &[MetricSample]| {
        samples
            .iter()
            .all(|s| s.step.is_some_and(|step| step.is_finite()))
            && samples.windows(2).all(|pair| pair[0].step <= pair[1].step)
    };
    let has_steps = ordered_steps(&state.metric_series)
        && (view != GraphView::Comparison || ordered_steps(&state.comparison_series));
    if has_steps {
        for points in [
            &mut loss,
            &mut moving_loss,
            &mut perplexity,
            &mut tokens,
            &mut learning_rate,
        ]
        .into_iter()
        .flatten()
        {
            for (x, _) in points {
                *x = state.metric_series[*x as usize]
                    .step
                    .expect("checked above");
            }
        }
        if let Some(points) = &mut compare_loss {
            for (x, _) in points {
                *x = state.comparison_series[*x as usize]
                    .step
                    .expect("checked above");
            }
        }
    }
    let mut data: Vec<(&'static str, &[(f64, f64)], Color)> = Vec::new();
    match view {
        GraphView::Loss => {
            data.push(("loss", loss.as_deref().unwrap_or(&[]), NORMAL_GREEN));
            data.push((
                "moving avg",
                moving_loss.as_deref().unwrap_or(&[]),
                SECOND_ACCENT,
            ));
        }
        GraphView::Perplexity => {
            data.push((
                "perplexity",
                perplexity.as_deref().unwrap_or(&[]),
                NORMAL_GREEN,
            ));
        }
        GraphView::TokensPerSecond => {
            data.push(("tokens/sec", tokens.as_deref().unwrap_or(&[]), NORMAL_GREEN));
        }
        GraphView::LearningRate => {
            data.push((
                "learning rate",
                learning_rate.as_deref().unwrap_or(&[]),
                NORMAL_GREEN,
            ));
        }
        GraphView::Comparison => {
            data.push((
                "current loss",
                moving_loss.as_deref().unwrap_or(&[]),
                NORMAL_GREEN,
            ));
            if compare_loss
                .as_deref()
                .is_some_and(|points| !points.is_empty())
            {
                data.push((
                    "compare loss",
                    compare_loss.as_deref().unwrap_or(&[]),
                    Color::Rgb(0x55, 0xd7, 0xff),
                ));
            }
        }
        GraphView::All => {
            data.push(("loss", loss.as_deref().unwrap_or(&[]), NORMAL_GREEN));
            data.push((
                "perplexity",
                perplexity.as_deref().unwrap_or(&[]),
                Color::Rgb(0x55, 0xd7, 0xff),
            ));
            data.push((
                "tokens/sec",
                tokens.as_deref().unwrap_or(&[]),
                Color::Rgb(0xc7, 0x92, 0xea),
            ));
            data.push((
                "learning rate",
                learning_rate.as_deref().unwrap_or(&[]),
                Color::Rgb(0xff, 0xd1, 0x66),
            ));
        }
        GraphView::Memory => unreachable!(),
    }
    let normalized = view == GraphView::All;

    let series: Vec<_> = data
        .iter()
        .map(|(name, points, color)| charts::Series::line(name, points, *color))
        .collect();
    let x_start = data
        .iter()
        .flat_map(|(_, p, _)| p.iter().map(|p| p.0))
        .fold(f64::INFINITY, f64::min);
    let x_end = data
        .iter()
        .flat_map(|(_, p, _)| p.iter().map(|p| p.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let x_bounds = if x_start.is_finite() {
        [x_start, x_end.max(x_start + 2.0)]
    } else {
        [0.0, 2.0]
    };
    let y_bounds = if normalized {
        [0.0, 1.0]
    } else {
        charts::bounds(data.iter().flat_map(|(_, p, _)| p.iter().map(|p| p.1)))
    };
    let mut title = match view {
        GraphView::Loss => " graph / loss + moving average ",
        GraphView::Perplexity => " graph / perplexity (exp loss) ",
        GraphView::TokensPerSecond => " graph / tokens per second ",
        GraphView::LearningRate => " graph / learning rate ",
        GraphView::Comparison => {
            if state.comparison_error.is_some() {
                " graph / comparison unavailable "
            } else if state.comparison_series.is_empty() {
                " graph / comparison (use --compare LOG) "
            } else {
                " graph / comparison "
            }
        }
        GraphView::All => " graph / normalized overlay ",
        GraphView::Memory => unreachable!(),
    }
    .to_string();
    if view == GraphView::Comparison && !state.comparison_series.is_empty() {
        if let Some(label) = state.comparison_label.as_deref() {
            title = format!(" graph / comparison / {label} ");
        }
    }
    let (y, caption) = match view {
        GraphView::Loss => (
            "loss",
            "Lower = better predictions; green: loss, blue: 8-sample average",
        ),
        GraphView::Perplexity => ("perplexity", "Lower = less surprise about the next word"),
        GraphView::TokensPerSecond => (
            "tokens / second",
            "Higher = more training text processed each second",
        ),
        GraphView::LearningRate => (
            "learning rate",
            "How much each training step changes the model",
        ),
        GraphView::Comparison => (
            "loss",
            "Lower = better; green: current average, blue: comparison",
        ),
        GraphView::All => (
            "relative level",
            "Own ranges: green loss / blue surprise / purple speed / gold rate",
        ),
        GraphView::Memory => unreachable!(),
    };
    let caption = if view == GraphView::Comparison {
        state.comparison_error.as_deref().unwrap_or(caption)
    } else {
        caption
    };
    charts::draw(
        f,
        area,
        charts::Plot {
            title: &title,
            caption,
            x: if has_steps { "training step" } else { "sample" },
            integer_x: true,
            y,
            x_bounds,
            y_bounds,
        },
        &series,
    );
}

fn draw_memory_graph(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let mut boundary = Vec::with_capacity(97);
    for step in 0..=96 {
        let angle = std::f64::consts::TAU * step as f64 / 96.0;
        boundary.push((angle.cos(), angle.sin()));
    }
    let occupancy = state.memory_used.unwrap_or(0) as usize;
    let capacity = state.memory_capacity.unwrap_or(0) as usize;
    let plotted = occupancy.min(256);
    let mut entries = Vec::with_capacity(plotted);
    for index in 0..plotted {
        let fraction = (index as f64 + 0.5) / plotted.max(1) as f64;
        let radius = 0.12 + 0.78 * fraction.sqrt();
        let angle = index as f64 * 2.399963229728653;
        entries.push((radius * angle.cos(), radius * angle.sin()));
    }
    // HalfBlock has two vertical pixels per terminal row. Use the actual
    // drawable plot rectangle, not the outer panel size, so the unit circle
    // remains round after labels, borders, and the x-axis rows are removed.
    // Terminal cells are about twice as tall as wide, hence the factor of two.
    let geometry = charts::Plot {
        title: " memory / Poincare disk ",
        caption: "",
        x: "disk x",
        integer_x: false,
        y: "disk y",
        x_bounds: [-1.0, 1.0],
        y_bounds: [-1.0, 1.0],
    };
    let x_extent = charts::graph_rect(area, &geometry)
        .map(|graph| {
            // HalfBlock has `width × (2 * height)` physical pixels. Use the
            // last drawable pixel in each direction, rather than the cell
            // count, so the unit circle stays round at every panel size.
            let x_pixels = f64::from(graph.width.saturating_sub(1));
            let y_pixels = f64::from(graph.height.saturating_mul(2).saturating_sub(1));
            (x_pixels / y_pixels).max(1.0)
        })
        .unwrap_or(1.0);
    let x_bounds = [-x_extent, x_extent];
    let disk_points = [charts::Series {
        name: "occupied entries",
        points: &entries,
        color: NORMAL_GREEN,
        scatter: true,
    }];
    let occupancy_label = if capacity > 0 {
        format!("{occupancy}/{capacity}")
    } else {
        "not reported".to_string()
    };
    let title = format!(" memory / Poincare disk / occupancy only ({occupancy_label}) ");
    charts::draw(
        f,
        area,
        charts::Plot {
            title: &title,
            caption: "Dots = used slots, not learned positions",
            x: "disk x",
            integer_x: false,
            y: "disk y",
            x_bounds,
            y_bounds: [-1.0, 1.0],
        },
        &disk_points,
    );
    // Render the boundary separately with HalfBlock pixels. It is deliberately
    // not part of the occupied-entry series: a continuous wall should not make
    // the small occupancy dots look like a connected trace.
    charts::draw_solid_overlay(
        f,
        area,
        charts::Plot {
            x_bounds,
            ..geometry
        },
        &boundary,
        Color::Rgb(0x32, 0x8f, 0x60),
    );
}

fn draw_monitor(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let area = if area.height >= 27 {
        let gap = if shadow::enabled(f.area()) && area.height >= 21 {
            1
        } else {
            0
        };
        let sections = if gap == 0 {
            Layout::vertical([Constraint::Min(0), Constraint::Length(5)]).split(area)
        } else {
            Layout::vertical([
                Constraint::Min(0),
                Constraint::Length(gap),
                Constraint::Length(5),
            ])
            .split(area)
        };
        let preview_index = if gap == 0 { 1 } else { 2 };
        state.preview.draw(f, sections[preview_index], true);
        sections[0]
    } else {
        area
    };
    // Prefer plot rows to a bordered progress meter and decorative gaps.
    // At 120x40 the shell leaves 29 rows, or 23 after the live preview:
    // progress 1 + chart 13 + metrics/input 9. Thirteen allocated rows leave
    // nine data rows after borders and both x-axis rows; shadows stay outside.
    // At 80x24, keep progress 1 + chart 9 + full-width metrics/input 7.
    let show_graph = area.height >= 14;
    let compact_graph = show_graph && area.height < 28;
    let gap = if shadow::enabled(f.area()) && area.height >= 28 {
        1
    } else {
        0
    };
    let chunks = if gap == 0 {
        Layout::vertical([
            Constraint::Length(if compact_graph { 1 } else { 4 }),
            Constraint::Min(if show_graph { 7 } else { 0 }),
            Constraint::Length(if show_graph && area.height < 21 {
                7
            } else if show_graph {
                9
            } else {
                area.height.saturating_sub(4)
            }),
        ])
        .split(area)
    } else {
        Layout::vertical([
            Constraint::Length(4),
            Constraint::Length(gap),
            Constraint::Min(if show_graph { 7 } else { 0 }),
            Constraint::Length(gap),
            Constraint::Length(if show_graph {
                9
            } else {
                area.height.saturating_sub(4)
            }),
        ])
        .split(area)
    };
    let graph_index = if gap == 0 { 1 } else { 2 };
    let metrics_index = if gap == 0 { 2 } else { 4 };

    let now = Instant::now();
    let display_pct = state.displayed_progress_at(now);
    draw_progress(f, chunks[0], state, display_pct);

    let done = state
        .updates_done
        .or(state.optimizer_updates)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());
    let total = state
        .updates_total
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());

    if show_graph {
        draw_graph(f, chunks[graph_index], state, now);
    }

    let memory = match (state.memory_used, state.memory_capacity) {
        (Some(used), Some(capacity)) if capacity > 0 => {
            format!(
                "{used}/{capacity} ({:.0}%)",
                used as f64 * 100.0 / capacity as f64
            )
        }
        _ => "not reported".into(),
    };
    let checkpoint = match (state.checkpoint_number, state.checkpoint_target.as_deref()) {
        (Some(number), Some(path)) => format!("#{number}  {path}"),
        (_, Some(path)) => path.to_string(),
        _ => "not configured".into(),
    };
    let speed = state
        .tok_s
        .map(|value| format!("{value:.0}"))
        .unwrap_or_else(|| "-".into());
    let epoch_loss = state
        .epoch_loss
        .map(charts::number)
        .unwrap_or_else(|| "-".into());
    let epoch_tokens = state
        .epoch_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into());
    let epoch_updates = state
        .epoch_updates
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into());
    let dream_mode = state.dream_mode.as_deref().unwrap_or("-");
    let dream_loss = state
        .dream_last_loss
        .map(charts::number)
        .unwrap_or_else(|| "-".into());
    let dream = if state.dream_active {
        format!(
            "DREAMING / #{} / mode {dream_mode} / update {}",
            state.dream_count.saturating_add(1),
            state
                .dream_update
                .map(|update| update.to_string())
                .unwrap_or_else(|| "-".into())
        )
    } else if state.dream_count > 0 {
        format!(
            "{} completed / mode {dream_mode} / last loss {dream_loss}",
            state.dream_count
        )
    } else {
        "no dream events recorded".into()
    };
    // Put current step, save timing and reported elapsed time first: these
    // must remain visible in the real 80x24 shell, not beneath clipped rows.
    let saved = state.checkpoint_context();
    let mut lines = vec![
        Line::from(format!(
            "step {} / optimizer {done}/{total} / lr {}",
            feed::value(state.current_step()),
            state
                .learning_rate
                .map_or_else(|| "unrecorded".into(), |lr| format!("{lr:.6e}"))
        )),
        Line::from(saved),
        Line::from(format!(
            "elapsed {} / {speed} tok/s / memory {memory}",
            state.elapsed_context()
        )),
        Line::from(format!(
            "checkpoint target {checkpoint} / loss guard {}",
            state.loss_guard_status.as_deref().unwrap_or("unreported")
        )),
    ];
    let input_line = match &state.feed {
        Some(feed) => format!("actual input / Feed tab: {}", feed.snippet),
        None => "No input window recorded / Feed tab".into(),
    };
    lines.push(Line::from(input_line));
    lines.push(Line::from(format!("memory read {} / growth {}",
        state.memory_retrieval.as_deref().unwrap_or("unreported"),
        state.memory_growth.as_deref().unwrap_or("none reported"))));
    lines.push(Line::from(format!(
        "gradient norm {} / skipped updates {} / epoch loss {epoch_loss}",
        state
            .grad_norm
            .map_or_else(|| "unrecorded".into(), |norm| format!("{norm:.3e}")),
        feed::value(state.skipped_updates)
    )));
    lines.push(Line::from(format!(
        "dream {dream} / epoch tokens {epoch_tokens} / updates {epoch_updates}"
    )));
    // Split off the input preview only when enough rows and columns remain
    // for the real run metrics. Smaller monitors use one full-width card.
    let metrics_area = if f.area().width >= 110 && chunks[metrics_index].height >= 9 {
        let bottom = if gap == 0 {
            Layout::horizontal([
                Constraint::Length((f.area().width / 4).clamp(30, 48)),
                Constraint::Min(0),
            ])
            .split(chunks[metrics_index])
        } else {
            Layout::horizontal([
                Constraint::Length((f.area().width / 4).clamp(30, 48)),
                Constraint::Length(gap),
                Constraint::Min(0),
            ])
            .split(chunks[metrics_index])
        };
        // The old neuron/ring animation was not model telemetry. Show the
        // trainer's latest real input instead, with no invented motion.
        feed::draw_compact(f, bottom[0], state);
        let metrics_index = if gap == 0 { 1 } else { 2 };
        bottom[metrics_index]
    } else {
        // Preserve readable metrics rather than squeezing two tiny panels
        // side by side on a narrow or very short terminal.
        chunks[metrics_index]
    };
    let metrics_area = panel_area(f, metrics_area);
    if metrics_area.height < 9 {
        // Five content rows fit the normal 80x24 monitor. Do not silently clip
        // real safeguard/dream measurements below a seven-row bordered card.
        let norm = state
            .grad_norm
            .map_or_else(|| "unrecorded".into(), |n| format!("{n:.3e}"));
        let dream = if state.dream_active {
            "active".into()
        } else if state.dream_count > 0 {
            state.dream_count.to_string()
        } else {
            "no events".into()
        };
        lines = vec![
            lines[0].clone(),
            lines[1].clone(),
            lines[2].clone(),
            lines[4].clone(),
            Line::from(format!(
                "grad {norm} / skipped {} / guard {} / dream {dream} / epoch {epoch_loss}",
                feed::value(state.skipped_updates),
                state
                    .loss_guard_status
                    .as_deref()
                    .map(|s| s.split(" / ").next().unwrap_or(s))
                    .unwrap_or("unreported")
            )),
        ];
    }
    f.render_widget(
        Paragraph::new(lines).block(panel(if show_graph {
            " run metrics / actual input "
        } else {
            " run metrics / compact / actual input "
        })),
        metrics_area,
    );
}

fn draw_chain(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let mut rows = vec![
        Line::from(format!("directory  {}", state.chain_dir.display())),
        divider(area.width.saturating_sub(2)),
    ];
    if state.checkpoints.is_empty() && area.height >= 10 {
        let gap = if shadow::enabled(f.area()) { 1 } else { 0 };
        let sections = Layout::vertical([
            Constraint::Length(5),
            Constraint::Length(gap),
            Constraint::Min(0),
        ])
        .split(area);
        rows.push(Line::styled(state.checkpoint_context(), accent()));
        let summary_area = panel_area(f, sections[0]);
        f.render_widget(
            Paragraph::new(rows).block(panel(" chain / checkpoints ")),
            summary_area,
        );
        let cards = card_pair(f, sections[2]);
        draw_info_card(
            f,
            cards[0],
            " checkpoint index / 0 files ",
            vec![
                Line::styled("FILE                 LOSS", Style::new().fg(SECOND_ACCENT)),
                Line::from(state.checkpoint_context()),
                Line::from("Only saved files and recorded losses appear; no synthetic history."),
            ],
        );
        draw_info_card(
            f,
            cards[1],
            " connect a chain ",
            vec![
                Line::styled(state.checkpoint_context(), accent()),
                Line::from("pssa tui --chain \"path/to/chain\""),
                Line::from("Use the directory where your run saves checkpoints."),
                Line::from("Read-only view / existing files stay untouched."),
            ],
        );
        return;
    } else if state.checkpoints.is_empty() {
        rows.push(Line::from(state.checkpoint_context()));
    } else {
        let recorded_save = state
            .last_checkpoint
            .as_deref()
            .and_then(|path| std::path::Path::new(path).file_name())
            .and_then(|name| name.to_str());
        rows.extend(state.checkpoints.iter().map(|(name, loss)| {
            let latest = recorded_save == Some(name.as_str());
            let marker = if latest { "●" } else { "○" };
            let loss_text = loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "—".into());
            let suffix = if latest { "  (last recorded save)" } else { "" };
            Line::styled(
                format!("{marker} {name}  loss {loss_text}{suffix}"),
                accent(),
            )
        }));
    }
    let chain_area = panel_area(f, area);
    f.render_widget(
        Paragraph::new(rows)
            .wrap(Wrap { trim: false })
            .block(panel(" chain / checkpoints ")),
        chain_area,
    );
}

fn draw_model(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let mut lines = vec![
        Line::from(
            state
                .configuration_source
                .as_deref()
                .unwrap_or("Configuration from the training log"),
        ),
        divider(area.width.saturating_sub(2)),
    ];
    for (label, value) in [
        ("corpus", &state.corpus),
        ("vocabulary", &state.vocab),
        ("width", &state.width),
        ("memory", &state.memory),
        ("schedule", &state.schedule),
    ] {
        lines.push(Line::from(vec![
            Span::styled(format!("{label:<14}"), Style::new().fg(SECOND_ACCENT)),
            Span::raw(value.as_deref().unwrap_or("not reported")),
        ]));
    }
    // Long corpus paths and schedules take priority over the hint cards.
    let configuration = Paragraph::new(lines).wrap(Wrap { trim: false });
    let required = configuration
        .line_count(area.width.saturating_sub(2).max(1))
        .saturating_add(2)
        .min(usize::from(area.height)) as u16;
    let gap = if shadow::enabled(f.area()) && area.height >= 22 {
        1
    } else {
        0
    };
    let sections = Layout::vertical([
        Constraint::Length(if area.height.saturating_sub(required) >= 6 {
            required
        } else {
            area.height
        }),
        Constraint::Length(gap),
        Constraint::Min(0),
    ])
    .split(area);
    let configuration_area = panel_area(f, sections[0]);
    f.render_widget(
        configuration.block(panel(" model / configuration ")),
        configuration_area,
    );
    if sections[2].height >= 6 {
        let cards = card_pair(f, sections[2]);
        draw_info_card(
            f,
            cards[0],
            " log connection ",
            vec![
                Line::styled(
                    if state.corpus.is_some() || state.width.is_some() {
                        "[ configuration received ]"
                    } else {
                        "[ awaiting training banner ]"
                    },
                    accent(),
                ),
                Line::from("pssa train [flags] --no-tui | pssa tui"),
                Line::from(if state.configuration_source.is_some() {
                    "Header dimensions only; corpus/schedule/loops require a recorded training log."
                } else {
                    "Fields above show only values reported by the run."
                }),
            ],
        );
        draw_info_card(
            f,
            cards[1],
            " run context ",
            vec![
                Line::from(format!(
                    "resume  {}",
                    state.resumed_from.as_deref().unwrap_or("not reported")
                )),
                Line::from(format!("saved   {}", state.checkpoint_context())),
                Line::styled(
                    "Read-only / no model or checkpoint changes",
                    Style::new().fg(SECOND_ACCENT),
                ),
            ],
        );
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    if args.first().is_some_and(|s| s == "--preview-worker") {
        if args.len() != 3 {
            return Err("internal preview expects checkpoint and loops".into());
        }
        let loops = args[2]
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=32).contains(n))
            .ok_or("invalid preview loops")?;
        return preview::worker(&args[1], loops);
    }
    if let Some(result) = eval::run_worker(args) {
        return result;
    }
    // Keep the default useful on a local checkout; Kaggle callers can pass
    // their mounted chain explicitly (the training scripts already do).
    let mut chain_dir = PathBuf::from("chain");
    let mut compare_path = None;
    let mut chats_dir = PathBuf::from("chats");
    let mut seen_options = std::collections::HashSet::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--chain" | "-c" => {
                if !seen_options.insert("--chain") {
                    return Err("option '--chain' was specified more than once".into());
                }
                let value = args.get(i + 1).ok_or_else(|| {
                    "option '--chain' requires a directory; usage: pssa tui [-c|--chain DIR]"
                        .to_string()
                })?;
                if value.starts_with('-') {
                    return Err(format!(
                        "option '{}' requires a directory; usage: pssa tui [-c|--chain DIR]",
                        args[i]
                    ));
                }
                chain_dir = PathBuf::from(value);
                i += 1;
            }
            "--chats-dir" => {
                if !seen_options.insert("--chats-dir") {
                    return Err("option '--chats-dir' was specified more than once".into());
                }
                let value = args
                    .get(i + 1)
                    .filter(|v| !v.starts_with('-'))
                    .ok_or("--chats-dir requires a directory")?;
                chats_dir = PathBuf::from(value);
                i += 1;
            }
            "--compare" => {
                if !seen_options.insert("--compare") {
                    return Err("option '--compare' was specified more than once".into());
                }
                let value = args.get(i + 1).ok_or_else(|| {
                    "option '--compare' requires a log file; usage: pssa tui [--compare LOG]"
                        .to_string()
                })?;
                if value.starts_with('-') {
                    return Err(
                        "option '--compare' requires a log file; usage: pssa tui [--compare LOG]"
                            .to_string(),
                    );
                }
                compare_path = Some(PathBuf::from(value));
                i += 1;
            }
            other => {
                return Err(format!(
                    "unknown tui flag '{other}'; usage: pssa tui [-c|--chain DIR] [--compare LOG]"
                ));
            }
        }
        i += 1;
    }

    if let Some(path) = compare_path.as_deref()
        && std::fs::metadata(path).is_ok_and(|metadata| !metadata.is_file())
    {
        return Err("--compare requires a regular log file, not a directory/device/pipe".into());
    }

    if io::stdin().is_terminal() {
        if io::stdout().is_terminal() {
            // Retain the sender so the standalone app stays open until quit.
            let (_tx, rx) = mpsc::channel();
            return run_app(rx, chain_dir, compare_path, chats_dir).map_err(|e| e.to_string());
        }
        println!(
            "Usage: pssa tui [-c|--chain DIR] [--compare LOG] [--chats-dir DIR] (interactive TTY required)"
        );
        return Ok(());
    }
    if !io::stdout().is_terminal() {
        // A dashboard cannot repaint a pipe.  Preserve the producer's plain
        // structured log instead of failing halfway through a headless run.
        for line in io::stdin().lock().lines() {
            println!("{}", strip_ansi(&line.map_err(|e| e.to_string())?));
        }
        return Ok(());
    }

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin().lock();
        for line in stdin.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    run_app(rx, chain_dir, compare_path, chats_dir).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_area_accounts_for_gutters_header_tabs_and_divider() {
        assert_eq!(
            feature_area(Rect::new(0, 0, 120, 40)),
            Rect::new(4, 9, 112, 29),
        );
        assert_eq!(
            feature_area(Rect::new(0, 0, 80, 24)),
            Rect::new(2, 8, 76, 15),
        );
    }

    #[test]
    fn refresh_chain_drops_deleted_checkpoints_and_missing_directories() {
        let chain = std::env::temp_dir().join(format!(
            "pssa-tui-chain-refresh-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&chain).unwrap();
        std::fs::write(chain.join("ck01.pssa"), b"fixture").unwrap();
        let mut state = RunState {
            chain_dir: chain.clone(),
            ..RunState::default()
        };
        state.refresh_chain();
        assert_eq!(state.checkpoints, [("ck01.pssa".into(), None)]);
        std::fs::remove_file(chain.join("ck01.pssa")).unwrap();
        state.refresh_chain();
        assert!(state.checkpoints.is_empty());
        state.checkpoints.push(("stale.pssa".into(), None));
        std::fs::remove_dir(&chain).unwrap();
        state.refresh_chain();
        assert!(state.checkpoints.is_empty());
    }

    #[test]
    fn parses_piped_progress_and_multicomponent_eta() {
        let mut state = RunState::default();
        state.ingest("  training 97/100 (97%) loss=4.077800 tokens_per_second=146 eta=45.8s");
        assert_eq!(state.progress_pct, Some(97.0));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(state.eta.as_deref(), Some("45.8s"));
        assert_eq!(state.epoch_loss, None);
        assert_eq!(state.loss_series, [4.0778]);
        state.ingest("  training 20/100 (20%) loss=4.0 tokens_per_second=150 eta=2h 14m 09s");
        assert_eq!(state.eta.as_deref(), Some("2h 14m 09s"));
    }

    #[test]
    fn parses_dream_start_and_completion_without_polluting_training_loss() {
        let mut state = RunState::default();
        state.ingest("progress_schema=2 updates_total=12");
        state.ingest("dream phase=start update=4 mode=both");
        assert!(state.dream_active);
        assert_eq!(state.dream_count, 0);
        assert_eq!(state.dream_mode.as_deref(), Some("both"));
        assert_eq!(state.dream_update, Some(4));
        state.ingest("dream phase=end update=4 mode=both dream_loss=0.125 entries_replayed=2");
        assert!(!state.dream_active);
        assert_eq!(state.dream_count, 1);
        assert_eq!(state.dream_last_loss, Some(0.125));
        assert!(state.loss_series.is_empty());
        assert_eq!(state.health_status().normal_label, "TRAINING");
        state.ingest("dream phase=start update=8 mode=memory");
        assert_eq!(state.health_status().normal_label, "DREAMING");
    }

    #[test]
    fn parses_interactive_progress_without_confusing_epoch_summary() {
        let mut state = RunState::default();
        state
            .ingest("\r  \x1b[2mtraining\x1b[0m ███░  97%  loss 4.0778  146 tok/s  eta 14m 09s   ");
        assert_eq!(state.progress_pct, Some(97.0));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(state.eta.as_deref(), Some("14m 09s"));
        state.ingest("  epoch 1/1 loss=4.0123 tokens=1200 updates=10");
        assert_eq!(state.epoch_loss, Some(4.0123));
        assert_eq!(state.epoch_tokens, Some(1200));
        assert_eq!(state.epoch_updates, Some(10));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.loss_series, [4.0778, 4.0123]);
    }

    #[test]
    fn parses_structured_training_metrics_and_checkpoint_events() {
        let mut state = RunState::default();
        state.ingest(
            "progress_schema=2 updates_total=42 prior_updates=7 checkpoint_target=/tmp/ck08.pssa",
        );
        state.ingest(
            "training 3/42 (7%) loss=4.125000 loss_average=4.250000 tokens_per_second=321 optimizer_updates=3 updates_total=42 updates_remaining=39 global_update=10 learning_rate=2.5e-4 memory_occupancy=12/512 eta=4m 2s",
        );
        assert_eq!(state.updates_done, Some(3));
        assert_eq!(state.updates_total, Some(42));
        assert_eq!(state.updates_remaining, Some(39));
        assert_eq!(state.learning_rate, Some(2.5e-4));
        assert_eq!(state.memory_used, Some(12));
        assert_eq!(state.memory_capacity, Some(512));
        assert_eq!(state.prior_steps, Some(7));
        assert_eq!(state.checkpoint_number, Some(8));
        assert_eq!(state.checkpoint_target.as_deref(), Some("/tmp/ck08.pssa"));
        assert_eq!(state.last_checkpoint, None);
        assert_eq!(state.eta.as_deref(), Some("4m 2s"));
        state.ingest("checkpoint_number=9 checkpoint_status=saved");
        assert_eq!(state.checkpoint_number, Some(9));
        state.ingest("last_checkpoint=/tmp/ck09.pssa");
        assert_eq!(state.last_checkpoint.as_deref(), Some("/tmp/ck09.pssa"));
    }

    #[test]
    fn parses_indented_ansi_and_panel_fields_but_not_label_prefixes() {
        let mut state = RunState::default();
        state.ingest("  \x1b[2mcorpus\x1b[0m          /tmp/training text.txt  ");
        state.ingest("  vocabulary      2048 BPE tokens");
        state.ingest("  width           256");
        state.ingest("  memory          512 slots");
        state.ingest("  schedule        1 epoch(s), 446 updates, lr 0.001");
        state.ingest("  │ wall time       2h 14m 09s                 │");
        state.ingest("  │ throughput      146 tokens/s              │");
        assert_eq!(state.corpus.as_deref(), Some("/tmp/training text.txt"));
        assert_eq!(state.vocab.as_deref(), Some("2048 BPE tokens"));
        assert_eq!(state.width.as_deref(), Some("256"));
        assert_eq!(state.memory.as_deref(), Some("512 slots"));
        assert_eq!(
            state.schedule.as_deref(),
            Some("1 epoch(s), 446 updates, lr 0.001")
        );
        assert_eq!(state.wall.as_deref(), Some("2h 14m 09s"));
        assert_eq!(state.throughput.as_deref(), Some("146 tokens/s"));
        for line in [
            "corpus=other",
            "corpus_path other",
            "corpuses other",
            "corpus   ",
        ] {
            assert_eq!(parse_field(line, "corpus"), None, "{line}");
        }
    }

    #[test]
    fn renamed_environment_precedence() {
        // A subprocess avoids mutating the environment of parallel tests.
        const CURRENT: &str = "PSSA_RENAME_FIXTURE";
        const LEGACY: &str = "OXIDE_RENAME_FIXTURE"; // Legacy fallback fixture.
        for (current, legacy, expected) in [
            (None, None, None),
            (None, Some("legacy"), Some("legacy")),
            (Some("current"), None, Some("current")),
            (Some("current"), Some("legacy"), Some("current")),
            (Some(""), Some("legacy"), Some("")),
        ] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", "tui::tests::renamed_environment_fixture"])
                .env_remove(CURRENT)
                .env_remove(LEGACY)
                .env_remove("PSSA_RENAME_EXPECTED");
            for (key, value) in [
                (CURRENT, current),
                (LEGACY, legacy),
                ("PSSA_RENAME_EXPECTED", expected),
            ] {
                if let Some(value) = value {
                    child.env(key, value);
                }
            }
            let output = child.output().unwrap();
            assert!(output.status.success(), "{output:?}");
        }
    }

    #[test]
    fn renamed_environment_fixture() {
        assert_eq!(
            crate::env_var_os("PSSA_RENAME_FIXTURE", "OXIDE_RENAME_FIXTURE"),
            std::env::var_os("PSSA_RENAME_EXPECTED"),
        );
    }

    #[test]
    fn producer_output_round_trips_through_parser() {
        // A child process makes ui::Progress actually write to a pipe, avoiding
        // unstable stdout-capture APIs or a duplicate copy of its format string.
        // Terse output keeps the serial harness's test-name prefix off the first
        // producer line when RUST_TEST_THREADS=1 is inherited by the child.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tui::tests::ui_producer_fixture",
                "--nocapture",
                "--format",
                "terse",
            ])
            .env("PSSA_TUI_PRODUCER_FIXTURE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut state = RunState::default();
        for line in String::from_utf8(output.stdout).unwrap().lines() {
            state.ingest(line);
        }
        assert_eq!(state.corpus.as_deref(), Some("/tmp/producer corpus.txt"));
        assert_eq!(state.width.as_deref(), Some("256"));
        assert_eq!(state.wall.as_deref(), Some("14m 09s"));
        assert_eq!(state.progress_pct, Some(100.0));
        assert_eq!(state.live_loss, Some(4.123456));
        assert!(state.tok_s.unwrap() > 0.0);
        assert_eq!(state.eta.as_deref(), Some("0.0s"));
    }

    #[test]
    fn ui_producer_fixture() {
        if crate::env_var_os("PSSA_TUI_PRODUCER_FIXTURE", "OXIDE_TUI_PRODUCER_FIXTURE").is_none() {
            return;
        }
        // libtest's single-threaded progress prefix has no trailing newline.
        println!();
        ui::field("corpus", "/tmp/producer corpus.txt");
        ui::field("width", "256");
        ui::panel_field("wall time", "14m 09s");
        let mut progress = ui::Progress::new("training", 4);
        progress.update(4, 512, 4.123456);
        progress.finish();
    }

    #[test]
    fn preserves_resume_paths_containing_spaces() {
        let mut state = RunState::default();
        state.ingest(
            "resumed_from=/tmp/run with spaces/ck01.pssa vocab=2048 d_latent=256 depth=1 prior_steps=42",
        );
        assert_eq!(
            state.resumed_from.as_deref(),
            Some("/tmp/run with spaces/ck01.pssa")
        );
    }

    #[test]
    fn progress_history_is_bounded_and_nonfinite_numbers_are_ignored() {
        let mut state = RunState::default();
        for _ in 0..650 {
            state.ingest("training 1/2 (50%) loss=4.0 tokens_per_second=146 eta=1.0s");
        }
        assert_eq!(state.loss_series.len(), 600);
        assert_eq!(state.raw_lines.len(), 400);
        state.ingest("training 1/2 (NaN%) loss=NaN tokens_per_second=NaN eta=unknown");
        assert_eq!(state.progress_pct, Some(50.0));
        assert_eq!(state.live_loss, Some(4.0));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(parse_pct("(inf%)"), None);
    }

    #[test]
    fn truncated_piped_stream_is_not_reported_as_done() {
        let mut state = RunState::default();
        state.ingest("progress_schema=2");
        finish_piped_stream(&mut state, Some(false));
        assert_eq!(state.health_status().level, HealthLevel::Problem);
        assert!(state.health_status().label().contains("stream ended"));

        let mut empty = RunState::default();
        finish_piped_stream(&mut empty, None);
        assert_eq!(empty.health_status().level, HealthLevel::Normal);
        assert_eq!(empty.health_status().normal_label, "WAITING");
    }

    #[test]
    fn memory_retrieval_events_are_not_loss_samples() {
        let mut state = RunState::default();
        state.ingest("memory_retrieval=top-4 memory_capacity=256");
        state.ingest("memory_growth=64->256 memory_capacity=256");
        assert_eq!(state.memory_retrieval.as_deref(), Some("top-4"));
        assert_eq!(state.memory_growth.as_deref(), Some("64->256"));
        assert!(state.loss_series.is_empty());
    }

    #[test]
    fn loss_guard_events_show_settings_and_terminal_failure() {
        let mut state = RunState::default();
        state.ingest(
            "loss_guard=on loss_guard_high_factor=4 loss_guard_jump_factor=8 loss_guard_patience=3",
        );
        assert!(
            state
                .loss_guard_status
                .as_deref()
                .unwrap()
                .contains("patience 3")
        );
        state.ingest("loss_guard=halted update_index=3 loss=28000000 high_limit=30 consecutive=3 patience=3; training aborted without checkpoint");
        assert!(!state.training_active);
        assert_eq!(
            state.loss_guard_status.as_deref(),
            Some("HALTED / no checkpoint")
        );
        assert!(state.health_status().label().contains("without checkpoint"));
    }

    #[test]
    fn health_uses_one_green_and_explains_red_conditions() {
        let mut state = RunState::default();
        for _ in 0..3 {
            state.ingest("training 1/10 (10%) loss=4.0 tokens_per_second=100 eta=1s");
        }
        assert_eq!(state.health_status().level, HealthLevel::Normal);
        assert_eq!(state.health_status().color(), NORMAL_GREEN);

        state.ingest("training 2/10 (20%) loss=7.0 tokens_per_second=100 eta=1s");
        let health = state.health_status();
        assert_eq!(health.level, HealthLevel::Problem);
        assert!(health.label().contains("loss spike"));
        assert_eq!(health.color(), BRIGHT_RED);

        state.ingest("training 3/10 (30%) loss=4.0 tokens_per_second=50 eta=1s");
        let health = state.health_status();
        assert_eq!(health.level, HealthLevel::Problem);
        assert!(health.label().contains("speed"));
    }

    #[test]
    fn schedule_mismatch_and_stall_are_explicit_problems() {
        let mut state = RunState::default();
        state.ingest(
            "schedule 1 epoch(s), 100 updates, lr first=0.00100000 last=0.00001000 (base 0.00100000, horizon 100)",
        );
        state.ingest("lr_schedule=fixed horizon=100 from_step=0 to_step=100 warmup=0");
        state.ingest(
            "training 1/100 (1%) loss=4.0 tokens_per_second=100 optimizer_updates=1 global_update=1 learning_rate=5.0e-4",
        );
        assert!(state.health_status().label().contains("outside schedule"));

        state.problem = None;
        state.last_progress_at = Some(Instant::now() - STALL_TIMEOUT - Duration::from_secs(1));
        let health = state.health_status();
        assert_eq!(health.level, HealthLevel::Problem);
        assert!(health.label().contains("no progress line"));
    }

    #[test]
    fn dream_phase_does_not_trip_the_stall_check() {
        let mut state = RunState::default();
        state.ingest("progress_schema=1");
        state.ingest("dream phase=start mode=memory update=10");
        state.last_progress_at = Some(Instant::now() - STALL_TIMEOUT - Duration::from_secs(5));
        let health = state.health_status();
        assert_ne!(health.level, HealthLevel::Problem);
        assert_eq!(health.normal_label, "DREAMING");

        state.ingest("dream phase=end mode=memory update=10 dream_loss=2.5");
        let health = state.health_status();
        assert_ne!(health.level, HealthLevel::Problem);
    }

    #[test]
    fn test_backend_tab_order_is_unique_and_every_label_is_visible() {
        use ratatui::{Terminal, backend::TestBackend};
        assert_eq!(
            TABS,
            [
                "monitor",
                "chain",
                "model",
                "feed",
                "inference",
                "setup",
                "HF login",
                "Kaggle",
                "memory",
                "runs",
                "benchmark",
                "sample",
                "hardware",
                "math",
                "devices",
                "limits",
                "library",
                "mixer",
                "eval",
                "HF backup",
                "cloud log",
                "phone ping",
                "updates",
                "support",
                "GitHub",
                "sweeps",
                "timeline",
            ]
        );
        let unique: std::collections::HashSet<_> = TABS.iter().collect();
        assert_eq!(unique.len(), TABS.len());
        for tab in 0..TABS.len() {
            let mut terminal = Terminal::new(TestBackend::new(300, 24)).unwrap();
            terminal
                .draw(|f| draw(f, &RunState::default(), tab))
                .unwrap();
            let rows: Vec<String> = terminal
                .backend()
                .buffer()
                .content()
                .chunks(300)
                .map(|row| row.iter().map(|cell| cell.symbol()).collect())
                .collect();
            let nav = rows
                .iter()
                .find(|row| TABS.iter().all(|label| row.contains(label)))
                .unwrap();
            let mut previous = 0;
            for label in TABS {
                assert_eq!(nav.matches(label).count(), 1, "tab {tab}: {nav}");
                let position = nav.find(label).unwrap();
                assert!(position >= previous);
                previous = position + label.len();
            }
            assert!(nav.contains("help"), "help must be discoverable: {nav}");
            for width in [79, 80, 120] {
                let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
                terminal
                    .draw(|f| draw(f, &RunState::default(), tab))
                    .unwrap();
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect();
                if width < 80 {
                    assert!(
                        text.contains(&format!("{} / Tab tabs / F1 help", TABS[tab])),
                        "compact navigation must show selected tab {tab} at {width}"
                    );
                } else {
                    let nav: String = terminal
                        .backend()
                        .buffer()
                        .content()
                        .chunks(width as usize)
                        .nth(6)
                        .unwrap()
                        .iter()
                        .map(|cell| cell.symbol())
                        .collect();
                    let page = if tab < HF_TAB {
                        &TABS[..HF_TAB]
                    } else if tab < keybindings::SAMPLE_TAB {
                        &TABS[HF_TAB..keybindings::SAMPLE_TAB]
                    } else if tab < keybindings::LIBRARY_TAB {
                        &TABS[keybindings::SAMPLE_TAB..keybindings::LIBRARY_TAB]
                    } else if tab < keybindings::BACKUP_TAB {
                        &TABS[keybindings::LIBRARY_TAB..keybindings::BACKUP_TAB]
                    } else if tab < keybindings::SUPPORT_TAB {
                        &TABS[keybindings::BACKUP_TAB..keybindings::SUPPORT_TAB]
                    } else {
                        &TABS[keybindings::SUPPORT_TAB..]
                    };
                    for label in page {
                        assert_eq!(nav.matches(label).count(), 1, "tab {tab} at {width}: {nav}");
                    }
                    assert!(nav.contains("F1 help"), "tab {tab} at {width}: {nav}");
                }
            }
        }
    }

    #[test]
    fn test_backend_renders_retro_header_and_closed_status_box() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| draw(frame, &RunState::default(), 2))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let row = |y: u16| -> String {
            (0..80)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .unwrap()
                        .symbol()
                        .chars()
                        .next()
                        .unwrap_or(' ')
                })
                .collect()
        };
        // Wide layouts now have a one-row background gutter.
        assert!(row(1).contains("███"));
        assert!(row(6).contains("model"));
        assert!(row(6).contains("F1 help"));
        assert!(!row(6).contains("modelt"));
        assert!(row(7).contains("╌"));
        assert!(row(1).contains("┌"));
        assert!(row(5).contains("└"));
        assert!(row(2).contains("[ WAITING ]"));
    }

    #[test]
    fn test_backend_logo_has_distinct_p_s_s_a_strokes() {
        use ratatui::{Terminal, backend::TestBackend};
        // Independent pixel masks catch the old OCCO-like glyphs, not merely
        // whether the renderer copied the PSSA_LOGO constant into the buffer.
        let letters = [
            ["1110", "1001", "1110", "1000", "1000"], // P: open lower bowl + stem
            ["0111", "1000", "0110", "0001", "1110"], // S: opposite turns
            ["0111", "1000", "0110", "0001", "1110"],
            ["0110", "1001", "1111", "1001", "1001"], // A: crossbar + legs
        ];
        for (width, height) in [(80, 24), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| draw(f, &RunState::default(), 2)).unwrap();
            let buffer = terminal.backend().buffer();
            let origin = HexBackground::content_area(buffer.area);
            for (letter, rows) in letters.iter().enumerate() {
                for (y, mask) in rows.iter().enumerate() {
                    for (x, pixel) in mask.chars().enumerate() {
                        let cell =
                            &buffer[(origin.x + (letter * 6 + x) as u16, origin.y + y as u16)];
                        assert_eq!(cell.symbol(), if pixel == '1' { "█" } else { " " });
                        assert_eq!(cell.fg, NORMAL_GREEN);
                    }
                }
            }
        }
    }

    #[test]
    fn test_backend_header_shows_live_stats_on_every_tab_and_missing_values() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        for (step, loss) in [(1, 4.0), (2, 3.0), (3, 2.0)] {
            state.ingest(&format!("training {step}/10 (30%) loss={loss} tokens_per_second=125 optimizer_updates={step} updates_total=10 eta=2h 14m 09s"));
        }
        for (width, height) in [(80, 24), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            for tab in 0..TABS.len() {
                terminal.draw(|f| draw(f, &state, tab)).unwrap();
                let buffer = terminal.backend().buffer();
                let area = HexBackground::content_area(buffer.area);
                let header: String = (area.y..area.y + 5)
                    .flat_map(|y| (area.x + 24..area.right()).map(move |x| buffer[(x, y)].symbol()))
                    .collect();
                for expected in [
                    "[ TRAINING ]",
                    "step 3/10",
                    "loss 2",
                    "tok/s 125",
                    "ETA 2h 14m 09s",
                    "step → / lower better",
                    "2/3/4",
                ] {
                    assert!(header.contains(expected), "missing {expected}: {header}");
                }
                assert!(header.contains(&loss_trace(&state)));
                assert!(
                    loss_trace(&state)
                        .chars()
                        .filter(|c| ('\u{2801}'..='\u{28ff}').contains(c))
                        .count()
                        >= 10
                );
            }
            terminal.draw(|f| draw(f, &RunState::default(), 2)).unwrap();
            let buffer = terminal.backend().buffer();
            let text: String = buffer.content().iter().map(|c| c.symbol()).collect();
            for expected in ["step -/-", "loss -", "tok/s -", "ETA -", "············"] {
                assert!(text.contains(expected));
            }
        }
        // Empty monitor metrics must not turn missing measurements into zeros.
        let mut tiny = Terminal::new(TestBackend::new(20, 8)).unwrap();
        tiny.draw(|f| draw(f, &RunState::default(), 0)).unwrap();
        let tiny_text: String = tiny
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(tiny_text.contains("WAITING"));
        assert!(tiny_text.contains("loss -"));
        assert!(!tiny_text.contains("loss 0.0000"));
        assert!(!tiny_text.contains("0 tokens/s"));
        assert!(!tiny_text.contains("last epoch  loss 0"));

        // Invalid measurements do not become a plausible zero-valued trace.
        state.loss_series = vec![f64::NAN, 2.0, f64::INFINITY];
        let trace = loss_trace(&state);
        assert!(trace.starts_with('·') && trace.ends_with('·'));
        assert_eq!(
            trace
                .chars()
                .filter(|c| ('\u{2801}'..='\u{28ff}').contains(c))
                .count(),
            1
        );
    }

    #[test]
    fn test_backend_header_prioritizes_health_reason_on_non_monitor_tabs() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.ingest("training 1/3 (33%) loss=4.0 tokens_per_second=100 eta=2s");
        state.ingest("training 2/3 (67%) loss=7.0 tokens_per_second=100 eta=1s");
        let reason = state.health_status().reason.unwrap();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        for tab in 1..TABS.len() {
            terminal.draw(|f| draw(f, &state, tab)).unwrap();
            let buffer = terminal.backend().buffer();
            let area = HexBackground::content_area(buffer.area);
            let title: String = (area.x + 24..area.right())
                .map(|x| buffer[(x, area.y)].symbol())
                .collect();
            assert!(title.contains(&reason), "clipped health reason: {title}");
            assert!(buffer.content().iter().any(|c| c.fg == BRIGHT_RED));
        }
    }

    #[test]
    fn test_backend_compact_empty_feed_keeps_wrapped_connection_hint() {
        use ratatui::{Terminal, backend::TestBackend};
        for width in [60, 79] {
            let mut terminal = Terminal::new(TestBackend::new(width, 12)).unwrap();
            terminal.draw(|f| draw(f, &RunState::default(), 3)).unwrap();
            let buffer = terminal.backend().buffer();
            let text: String = buffer.content().iter().map(|c| c.symbol()).collect();
            for expected in ["Waiting for dataset telemetry", "--data CORPUS", "--no-tui"] {
                assert!(
                    text.contains(expected),
                    "{width} columns missing {expected}"
                );
            }
            assert_eq!(buffer[(0, 11)].symbol(), "└");
            assert_eq!(buffer[(width - 1, 11)].symbol(), "┘");
            assert!(!text.contains("source text"));
        }
    }

    #[test]
    fn test_backend_empty_tabs_have_cards_hints_and_two_tinted_surfaces() {
        use ratatui::{Terminal, backend::TestBackend};
        for (tab, labels) in [
            (
                1,
                [
                    "chain / checkpoints",
                    "checkpoint index",
                    "connect a chain",
                    "--chain",
                ],
            ),
            (
                2,
                [
                    "model / configuration",
                    "log connection",
                    "run context",
                    "--no-tui",
                ],
            ),
            (
                3,
                [
                    "dataset feed",
                    "dataset telemetry",
                    "--data CORPUS",
                    "--no-tui",
                ],
            ),
        ] {
            for (width, height) in [(80, 24), (120, 40)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal
                    .draw(|f| draw(f, &RunState::default(), tab))
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let text: String = buffer.content().iter().map(|c| c.symbol()).collect();
                for label in labels {
                    assert!(text.contains(label), "{width}x{height} missing {label}");
                }
                for color in [PANEL_BG, INSET_BG] {
                    assert!(buffer.content().iter().any(|c| c.bg == color));
                }
                assert!(buffer.content().iter().any(|c| c.fg == SECOND_ACCENT));
                assert!(!buffer.content().iter().any(|c| c.fg == BRIGHT_RED));
                if height >= 40 && tab != 3 {
                    assert!(
                        buffer
                            .content()
                            .iter()
                            .any(|c| c.symbol() == "·" && c.fg == GRID_COLOR)
                    );
                }
            }
        }
    }

    #[test]
    fn test_backend_populated_chain_and_model_keep_reported_values() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.chain_dir = PathBuf::from("my chain");
        state.checkpoints = vec![
            ("ck01.pssa".into(), Some(3.25)),
            ("ck02.pssa".into(), Some(2.5)),
        ];
        state.ingest("width 128 latent / 64 state");
        state.ingest("corpus local corpus.txt");
        state.resumed_from = Some("my chain/ck01.pssa".into());
        for (tab, values) in [
            (
                1,
                ["my chain", "ck01.pssa", "3.2500", "ck02.pssa", "2.5000"],
            ),
            (
                2,
                [
                    "128 latent / 64 state",
                    "local corpus.txt",
                    "configuration received",
                    "my chain/ck01.pssa",
                    "not reported",
                ],
            ),
        ] {
            let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
            terminal.draw(|f| draw(f, &state, tab)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            for value in values {
                assert!(text.contains(value), "missing {value}");
            }
        }
    }

    #[test]
    fn test_backend_renders_braille_progress_gradient_and_metrics() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use std::collections::HashSet;

        let mut state = RunState::default();
        state.ingest(
            "training 13/100 (13%) loss=4.0 tokens_per_second=100 optimizer_updates=13 updates_total=100 eta=12s",
        );
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
        assert!(
            BRAILLE_LEVELS[..7].iter().any(|glyph| text.contains(glyph)),
            "progress retains fractional precision"
        );
        assert!(text.contains("13%"));
        assert!(text.contains("updates 13/100"));
        assert!(text.contains("ETA 12s"));

        let colors: HashSet<_> = buffer
            .content()
            .iter()
            .filter(|cell| BRAILLE_LEVELS.contains(&cell.symbol()))
            .map(|cell| cell.fg)
            .collect();
        assert!(
            colors.len() >= 2,
            "the filled bar should use a green gradient"
        );
    }

    #[test]
    fn progress_bar_interpolates_between_targets() {
        let mut state = RunState::default();
        state.ingest("training 10/100 (10%) loss=4.0 tokens_per_second=100 eta=1s");
        state.ingest("training 50/100 (50%) loss=4.0 tokens_per_second=100 eta=1s");
        assert!(state.progress_from.unwrap() >= 10.0);
        assert!(state.progress_from.unwrap() < 50.0);
        let visible = state.displayed_progress_at(Instant::now());
        assert!(visible >= state.progress_from.unwrap());
        assert!(visible <= 50.0);
    }

    #[test]
    fn partial_progress_uses_recorded_counters_instead_of_a_fake_zero_percent() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.ingest("progress_schema=2 prior_updates=100 updates_total=500");
        assert_eq!(state.reported_progress(), Some(0.0));
        state.ingest("optimizer_updates=120 updates_total=500");
        assert_eq!(state.reported_progress(), Some(24.0));
        assert_eq!(state.displayed_progress_at(Instant::now()), 24.0);
        for (width, height) in [(80, 24), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| draw(f, &state, 0)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("24%"), "{width}x{height}: {text}");
            assert!(text.contains("120/500"), "{width}x{height}: {text}");
        }
        state.updates_done = None;
        assert_eq!(state.reported_progress(), None);
        assert!(progress_label(&state, 80).contains("unrecorded"));
        assert!(!progress_label(&state, 80).contains("0%"));
        state.updates_done = Some(120);
        state.updates_total = Some(0);
        assert_eq!(state.reported_progress(), None);
        state.ingest("training 120/500 (24%) loss=2 tokens_per_second=100");
        assert_eq!(state.reported_progress(), Some(24.0));
    }

    #[test]
    fn chart_samples_include_derived_metrics_and_moving_average() {
        let mut state = RunState::default();
        for (step, loss) in [4.0, 3.5, 3.0, 2.5].into_iter().enumerate() {
            state.ingest(&format!(
                "training {}/10 ({}%) loss={loss} tokens_per_second={} learning_rate=0.00{} eta=1s",
                step + 1,
                (step + 1) * 10,
                100 + step,
                step + 1
            ));
        }
        assert_eq!(state.metric_series.len(), 4);
        let (start, end) = state.visible_graph_range();
        let loss = metric_points(
            &state.metric_series,
            MetricKind::Loss,
            start,
            end,
            state.displayed_metric_at(Instant::now()),
        );
        let moving = moving_loss_points(
            &state.metric_series,
            start,
            end,
            state.displayed_metric_at(Instant::now()),
        );
        let perplexity = metric_points(
            &state.metric_series,
            MetricKind::Perplexity,
            start,
            end,
            None,
        );
        assert_eq!(loss.len(), 4);
        assert_eq!(moving.len(), 4);
        assert_eq!(perplexity.len(), 4);
        assert!((perplexity[0].1 - 4.0_f64.exp()).abs() < 1e-10);
        assert!(moving[3].1 < moving[0].1);
    }

    #[test]
    fn missing_loss_observations_are_skipped_when_smoothed_or_normalized() {
        let series = [
            MetricSample {
                loss: Some(5.0),
                ..Default::default()
            },
            MetricSample::default(),
            MetricSample::default(),
            MetricSample {
                loss: Some(4.0),
                ..Default::default()
            },
        ];
        let loss = metric_points(&series, MetricKind::Loss, 0, series.len(), None);
        let moving = moving_loss_points(&series, 0, series.len(), None);
        assert_eq!(loss, [(0.0, 5.0), (3.0, 4.0)]);
        assert_eq!(moving, [(0.0, 5.0), (3.0, 4.5)]);
        for points in [loss, moving] {
            assert!(points.iter().all(|(_, y)| y.is_finite()));
            let normalized = normalize_points(&points);
            assert_eq!(normalized.len(), 2);
            assert_eq!(normalized[0].1, 1.0);
            assert_eq!(normalized[1].1, 0.0);
        }
        let state = RunState {
            metric_series: vec![series[0], series[1]],
            graph_from: Some(series[0]),
            graph_updated_at: Some(Instant::now()),
            ..Default::default()
        };
        assert!(
            state
                .displayed_metric_at(Instant::now())
                .unwrap()
                .loss
                .is_none()
        );
    }

    #[test]
    fn non_finite_loss_observations_remain_chart_gaps() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let series = [
                MetricSample {
                    loss: Some(5.0),
                    ..Default::default()
                },
                MetricSample {
                    loss: Some(invalid),
                    ..Default::default()
                },
                MetricSample {
                    loss: Some(4.0),
                    ..Default::default()
                },
            ];
            for points in [
                metric_points(&series, MetricKind::Loss, 0, series.len(), None),
                moving_loss_points(&series, 0, series.len(), None),
            ] {
                assert_eq!(points.len(), 3);
                assert_eq!(points[0], (0.0, 5.0));
                assert_eq!(points[1].0, 1.0);
                assert!(points[1].1.is_nan());
                assert_eq!(points[2].0, 2.0);
                assert!(points[2].1.is_finite());
                assert!(normalize_points(&points)[1].1.is_nan());
            }
        }
    }

    #[test]
    fn comparison_with_missing_steps_uses_one_honest_sample_axis() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        for step in [100, 200, 300] {
            state.ingest(&format!(
                "training 1/3 (33%) loss=3 tokens_per_second=100 global_update={step}"
            ));
        }
        state.graph_view = GraphView::Comparison;
        state.comparison_series = state.metric_series.clone();
        state.comparison_series[1].step = None;
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &state, 0)).unwrap();
            charts::assert_named_plot(
                terminal.backend().buffer(),
                "graph / comparison",
                &["sample", "loss", "0", "1", "2"],
            );
        }
    }

    #[test]
    fn test_backend_renders_braille_chart_views_and_memory_disk() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = RunState::default();
        for (step, loss) in [4.0, 3.8, 3.4, 3.1, 2.9, 2.7].into_iter().enumerate() {
            state.ingest(&format!(
                "training {}/10 ({}%) loss={loss} tokens_per_second={} learning_rate=0.001 eta=1s",
                step + 1,
                (step + 1) * 10,
                100 + step
            ));
        }
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
        assert!(text.contains("graph / loss + moving average"));
        assert!(buffer.content().iter().any(|cell| {
            cell.symbol()
                .chars()
                .next()
                .is_some_and(|ch| ('\u{2800}'..='\u{28ff}').contains(&ch))
        }));

        state.graph_view = GraphView::All;
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("graph / normalized overlay"));

        state.graph_view = GraphView::Memory;
        state.memory_used = Some(12);
        state.memory_capacity = Some(64);
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("Poincare disk"));
        assert!(text.contains("12/64"));
    }

    #[test]
    fn poincare_boundary_is_aspect_correct_and_one_connected_solid_outline() {
        use ratatui::{Terminal, backend::TestBackend, style::Color};
        use std::collections::{HashSet, VecDeque};

        let state = RunState {
            graph_view: GraphView::Memory,
            memory_used: Some(12),
            memory_capacity: Some(64),
            ..RunState::default()
        };
        for (width, height) in [(120, 40), (80, 24), (240, 80), (80, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
            let buffer = terminal.backend().buffer();
            let outline_color = Color::Rgb(0x32, 0x8f, 0x60);
            let mut pixels = HashSet::new();
            let mut dots = 0;
            for (index, cell) in buffer.content().iter().enumerate() {
                let x = (index % usize::from(width)) as i32;
                let y = (index / usize::from(width)) as i32 * 2;
                if cell.fg == outline_color {
                    assert!(
                        matches!(cell.symbol(), "▀" | "▄" | "█"),
                        "non-solid boundary glyph at {x},{y}"
                    );
                    if matches!(cell.symbol(), "▀" | "█") {
                        pixels.insert((x, y));
                    }
                    if matches!(cell.symbol(), "▄" | "█") {
                        pixels.insert((x, y + 1));
                    }
                } else if cell.fg == NORMAL_GREEN
                    && cell
                        .symbol()
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                {
                    dots += 1;
                }
            }
            assert!(pixels.len() >= 12, "solid disk outline was not rendered");
            assert!(dots > 0, "occupied slots must remain separate small dots");
            let min_x = pixels.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixels.iter().map(|(x, _)| *x).max().unwrap();
            let min_y = pixels.iter().map(|(_, y)| *y).min().unwrap();
            let max_y = pixels.iter().map(|(_, y)| *y).max().unwrap();
            assert!(
                ((max_x - min_x) - (max_y - min_y)).abs() <= 2,
                "disk is an oval at {width}x{height}: {}x{} physical pixels",
                max_x - min_x,
                max_y - min_y
            );
            let start = *pixels.iter().next().unwrap();
            let mut seen = HashSet::from([start]);
            let mut queue = VecDeque::from([start]);
            while let Some((x, y)) = queue.pop_front() {
                for nx in x - 1..=x + 1 {
                    for ny in y - 1..=y + 1 {
                        if pixels.contains(&(nx, ny)) && seen.insert((nx, ny)) {
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
            assert_eq!(
                seen.len(),
                pixels.len(),
                "boundary has a gap between adjacent pixels"
            );

            // Connectivity alone would still allow an open arc. Flood-fill the
            // interior in physical half-block pixels: a closed outline must
            // prevent escape through any missing adjacent boundary cell.
            let center = ((min_x + max_x) / 2, (min_y + max_y) / 2);
            assert!(!pixels.contains(&center));
            let mut seen = HashSet::from([center]);
            let mut queue = VecDeque::from([center]);
            while let Some((x, y)) = queue.pop_front() {
                assert!(
                    x >= min_x && x <= max_x && y >= min_y && y <= max_y,
                    "boundary has a gap: interior escaped at {width}x{height}"
                );
                for next in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    if !pixels.contains(&next) && seen.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    #[test]
    fn monitor_loss_axes_keep_ticks_and_titles_out_of_the_curve() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        for (step, loss) in [(1, 5.0), (75, 4.0), (150, 3.0)] {
            state.ingest(&format!(
                "training {step}/150 (50%) loss={loss} global_update={step} tokens_per_second=100"
            ));
        }
        state.graph_updated_at = None;
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| draw(f, &state, 0)).unwrap();
            let b = terminal.backend().buffer();
            let content = HexBackground::content_area(b.area);
            let row = |y| {
                (content.x..content.right())
                    .map(|x| b[(x, y)].symbol())
                    .collect::<String>()
            };
            let top = (content.y..content.bottom())
                .find(|&y| row(y).contains("graph / loss"))
                .unwrap();
            let bottom = (top + 1..content.bottom())
                .find(|&y| b[(content.x, y)].symbol() == "└")
                .unwrap();
            let data_top = top + 1;
            let data_bottom = bottom - 3;
            // The visible loss range is 3..5, padded to 2.9..5.1, not 2..6.
            let dot_height = (data_bottom - data_top + 1) * 4 - 1;
            let ticks: Vec<_> = (data_top..=data_bottom)
                .filter_map(|y| {
                    let text = row(y);
                    let value = text
                        .strip_prefix('│')?
                        .split('│')
                        .next()?
                        .trim()
                        .parse::<f64>()
                        .ok()?;
                    Some((y, value))
                })
                .collect();
            assert!(
                ticks.len() >= 2 && ticks.len() <= 5,
                "actual fitted ticks: {ticks:?}"
            );
            for &(y, value) in &ticks {
                let expected =
                    data_top + (((5.1 - value) * f64::from(dot_height) / 2.2) as u16 / 4);
                assert!(
                    y.abs_diff(expected) <= 1,
                    "tick {value} misplaced at {y}, expected {expected}"
                );
            }
            assert!(ticks.windows(2).all(|p| p[0].1 > p[1].1));
            let labels = row(bottom - 1);
            for token in ["1", "76", "150", "training", "step"] {
                assert!(
                    labels
                        .split(|c: char| c.is_whitespace() || c == '│')
                        .any(|s| s == token),
                    "missing {token}: {labels}"
                );
            }
            assert!(!labels.contains("75.5"));
            assert!(row(bottom).contains("green: loss"));
            let axis = content.x + 4;
            let right = (axis + 1..content.right())
                .find(|&x| b[(x, data_top)].symbol() == "│")
                .unwrap();
            for y in data_top..=data_bottom {
                for x in axis + 1..right {
                    let cell = &b[(x, y)];
                    assert_eq!(cell.bg, PANEL_BG);
                    assert!(
                        cell.symbol() == " "
                            || cell
                                .symbol()
                                .chars()
                                .all(|c| ('\u{2800}'..='\u{28ff}').contains(&c)),
                        "label over curve: {}",
                        row(y)
                    );
                }
            }
        }
    }

    #[test]
    fn every_monitor_view_has_readable_braille_axes_in_the_real_shell() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        for (i, loss) in [4.0, 3.2, 3.5, 2.5].into_iter().enumerate() {
            state.ingest(&format!("training {}/4 (50%) loss={loss} tokens_per_second={} learning_rate={} global_update={} memory_occupancy=12/64",
                i + 1, 100 + i * 10, 0.001 / (i + 1) as f64, (i + 1) * 100));
        }
        state.graph_updated_at = None;
        state.comparison_series = state.metric_series.clone();
        for (w, h) in [(120, 40), (80, 24)] {
            for (view, y, caption) in [
                (GraphView::Loss, "loss", "Lower = better predictions"),
                (GraphView::Perplexity, "perplexity", "Lower = less surprise"),
                (
                    GraphView::TokensPerSecond,
                    "tokens / second",
                    "Higher = more training",
                ),
                (
                    GraphView::LearningRate,
                    "learning rate",
                    "How much each training step",
                ),
                (GraphView::Comparison, "loss", "Lower = better"),
                (GraphView::All, "relative level", "Own ranges"),
                (GraphView::Memory, "disk y", "Dots = used slots"),
            ] {
                state.graph_view = view;
                let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
                terminal.draw(|f| draw(f, &state, 0)).unwrap();
                let buffer = terminal.backend().buffer();
                let content = HexBackground::content_area(buffer.area);
                // Isolate the plot, excluding header trace, progress and neurons.
                let top = (content.y..content.bottom())
                    .find(|&row| {
                        let text: String = (content.x..content.right())
                            .map(|x| buffer[(x, row)].symbol())
                            .collect();
                        text.contains(if view == GraphView::Memory {
                            "memory / Poincare"
                        } else {
                            "graph /"
                        })
                    })
                    .unwrap();
                let bottom = (top + 1..content.bottom())
                    .find(|&row| buffer[(content.x, row)].symbol() == "└")
                    .unwrap();
                assert_chart_rows(
                    buffer,
                    if view == GraphView::Memory {
                        "memory / Poincare"
                    } else {
                        "graph /"
                    },
                    if h == 40 { 8 } else { 3 },
                );
                let rect = Rect::new(content.x, top, content.width - 1, bottom - top + 1);
                let plot_labels = [
                    y,
                    caption,
                    if view == GraphView::Memory {
                        "disk x"
                    } else {
                        "training step"
                    },
                ];
                if view == GraphView::Memory {
                    charts::assert_plot_with_halfblocks(buffer, rect, &plot_labels);
                } else {
                    charts::assert_plot(buffer, rect, &plot_labels);
                }
                if view != GraphView::Memory {
                    charts::assert_plot(buffer, rect, &["100", "250", "400"]);
                }
                if view == GraphView::Loss {
                    assert!(buffer.content().iter().any(|c| {
                        c.fg == SECOND_ACCENT
                            && c.symbol()
                                .chars()
                                .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                    }));
                }
            }
        }
    }

    #[test]
    fn neuron_animation_grows_collapses_and_reseeds_at_cycle_boundary() {
        let first = neuron_frame_at(Duration::ZERO);
        let growing = neuron_frame_at(Duration::from_millis(3_000));
        let collapsing = neuron_frame_at(Duration::from_millis(6_800));
        let settled = neuron_frame_at(NEURON_CYCLE - Duration::from_millis(1));
        let next = neuron_frame_at(NEURON_CYCLE);

        assert_eq!(first.growth, 0.0);
        assert!(growing.growth > 0.0 && growing.growth < 1.0);
        assert!(collapsing.network_alpha < 1.0);
        assert!(settled.collapsed);
        // The rendered endpoint is the same root dot as the next cycle's
        // first frame, while the next cycle receives a fresh random network.
        assert_eq!(settled.scale, 0.0);
        assert_eq!(settled.network_alpha, 0.0);
        assert_eq!(next.phase, 0.0);
        assert_eq!(next.scale, first.scale);
        assert_eq!(next.network_alpha, first.network_alpha);
        assert_ne!(
            neuron_network(first.cycle + 1)[1].x,
            neuron_network(first.cycle)[1].x
        );
    }

    #[test]
    fn neuron_growth_accelerates_while_camera_pulls_back_and_network_stays_bounded() {
        let early = neuron_frame_at(Duration::from_millis(2_000));
        let middle = neuron_frame_at(Duration::from_millis(3_000));
        let late = neuron_frame_at(Duration::from_millis(4_000));
        assert!(late.growth - middle.growth > middle.growth - early.growth);
        assert!(early.scale > middle.scale && middle.scale > late.scale);
        for cycle in 0..100 {
            let nodes = neuron_network(cycle);
            assert_eq!(nodes.len(), NEURON_NODE_COUNT);
            assert!((70..=90).contains(&nodes.len()));
            assert_eq!((nodes[0].x, nodes[0].y), (0.0, 0.0));
            let cross_links = nodes
                .iter()
                .filter(|node| node.cross_link.is_some())
                .count();
            assert!((30..=40).contains(&cross_links), "a mesh, not just a tree");
            let branching = (0..nodes.len())
                .filter(|&parent| nodes.iter().skip(1).filter(|n| n.parent == parent).count() > 1)
                .count();
            assert!(
                branching >= 12,
                "at least 15% of nodes branch into multiple tendrils"
            );
            for (index, node) in nodes.iter().enumerate().skip(1) {
                assert!(node.parent < index, "branches attach to existing nodes");
                assert!(node.x.is_finite() && node.x.abs() < 0.95);
                assert!(node.y.is_finite() && node.y.abs() < 0.95);
                if let Some(other) = node.cross_link {
                    assert!(other < index && other != node.parent);
                    assert_ne!(other, nodes[node.parent].parent);
                    assert_ne!(nodes[other].parent, node.parent);
                }
            }
        }
    }

    fn braille_pixels(buffer: &ratatui::buffer::Buffer) -> u32 {
        buffer
            .content()
            .iter()
            .filter_map(|cell| cell.symbol().chars().next())
            .filter(|ch| ('\u{2800}'..='\u{28ff}').contains(ch))
            .map(|ch| (ch as u32 - 0x2800).count_ones())
            .sum()
    }

    #[test]
    fn test_backend_renders_neuron_animation_frames_on_braille_canvas() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut terminal = Terminal::new(TestBackend::new(52, 12)).unwrap();
        for elapsed in [
            Duration::ZERO,
            Duration::from_millis(2_500),
            Duration::from_millis(6_800),
            NEURON_CYCLE - Duration::from_millis(1),
        ] {
            terminal
                .draw(|frame| draw_neuron_animation_at(frame, frame.area(), elapsed))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
            assert!(!text.contains("neuron"));
            for x in 1..51 {
                assert_eq!(buffer[(x, 0)].symbol(), "─", "plain, untitled border");
            }
            assert!(buffer.content().iter().any(|cell| {
                cell.symbol()
                    .chars()
                    .next()
                    .is_some_and(|ch| ('\u{2800}'..='\u{28ff}').contains(&ch))
            }));
            assert!(buffer.content().iter().any(|cell| cell.fg == NORMAL_GREEN));
            if elapsed == Duration::ZERO || elapsed == NEURON_CYCLE - Duration::from_millis(1) {
                assert_eq!(braille_pixels(buffer), 5, "one filled circular seed");
            } else if elapsed == Duration::from_millis(2_500) {
                assert!(braille_pixels(buffer) > 5, "branches grow beyond the seed");
                assert!(buffer.content().iter().any(|cell| {
                    matches!(cell.fg, Color::Rgb(r, g, b) if r > 0 && r < 0x39 && g > b && b > r)
                }), "branches and newly born nodes fade in green");
            }
        }

        // The final collapsed frame and the next cycle's seed are the same
        // rendered dot, not merely the same animation state numerically.
        terminal
            .draw(|frame| {
                draw_neuron_animation_at(
                    frame,
                    frame.area(),
                    NEURON_CYCLE - Duration::from_millis(1),
                )
            })
            .unwrap();
        let collapsed = terminal.backend().buffer().content().to_vec();
        terminal
            .draw(|frame| draw_neuron_animation_at(frame, frame.area(), NEURON_CYCLE))
            .unwrap();
        assert_eq!(collapsed, terminal.backend().buffer().content());
    }

    #[test]
    fn test_backend_finished_neuron_net_fills_both_panel_axes_without_clipping() {
        use ratatui::{Terminal, backend::TestBackend};

        // Include short/wide and tall/narrow viewports: fitting a square net
        // in world coordinates used to leave most of a wide panel empty.
        for (width, height) in [(30, 9), (48, 12), (80, 8), (24, 18)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            for cycle in 0..10 {
                let elapsed = NEURON_CYCLE * cycle + Duration::from_millis(5_760);
                terminal
                    .draw(|f| draw_neuron_animation_at(f, f.area(), elapsed))
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let mut pixels = Vec::new();
                for y in 1..height - 1 {
                    for x in 1..width - 1 {
                        let ch = buffer[(x, y)].symbol().chars().next().unwrap();
                        if !('\u{2800}'..='\u{28ff}').contains(&ch) {
                            continue;
                        }
                        let bits = ch as u32 - 0x2800;
                        for (bit, (dx, dy)) in [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                            (0, 3),
                            (1, 3),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            if bits & (1 << bit) != 0 {
                                pixels.push(((x - 1) * 2 + dx, (y - 1) * 4 + dy));
                            }
                        }
                    }
                }
                assert!(
                    pixels.len() > 100,
                    "finished net has substantial visible detail"
                );
                for (axis, extent) in [2 * (width - 2), 4 * (height - 2)].into_iter().enumerate() {
                    let coordinates: Vec<_> = pixels
                        .iter()
                        .map(|&(x, y)| if axis == 0 { x } else { y })
                        .collect();
                    let min = *coordinates.iter().min().unwrap();
                    let max = *coordinates.iter().max().unwrap();
                    let coverage = f64::from(max - min) / f64::from(extent - 1);
                    assert!(
                        (0.75..=0.90).contains(&coverage),
                        "{width}x{height}, cycle {cycle}, axis {axis}: coverage {coverage}"
                    );
                    assert!(min > 0 && max < extent - 1, "net leaves a margin");
                }
                assert_eq!(buffer[(0, height - 1)].symbol(), "└");
                assert_eq!(buffer[(width - 1, height - 1)].symbol(), "┘");
            }
        }
    }

    #[test]
    fn test_backend_neuron_seam_survives_resizes_and_new_seeds_change_the_net() {
        use ratatui::{Terminal, backend::TestBackend};

        for (width, height) in [(0, 0), (1, 1), (2, 2), (12, 5), (24, 9), (52, 12)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|f| draw_neuron_animation_at(f, f.area(), Duration::ZERO))
                .unwrap();
            let seed = terminal.backend().buffer().clone();
            for cycle in 1..=4 {
                for elapsed in [
                    NEURON_CYCLE * cycle - Duration::from_millis(1),
                    NEURON_CYCLE * cycle,
                ] {
                    terminal
                        .draw(|f| draw_neuron_animation_at(f, f.area(), elapsed))
                        .unwrap();
                    assert_eq!(terminal.backend().buffer(), &seed);
                }
            }
            if width >= 24 {
                terminal
                    .draw(|f| draw_neuron_animation_at(f, f.area(), Duration::from_secs(3)))
                    .unwrap();
                let first_net = terminal.backend().buffer().clone();
                terminal
                    .draw(|f| {
                        draw_neuron_animation_at(f, f.area(), NEURON_CYCLE + Duration::from_secs(3))
                    })
                    .unwrap();
                assert_ne!(terminal.backend().buffer(), &first_net);
                assert_eq!(terminal.backend().buffer()[(0, 0)].symbol(), "┌");
                assert_eq!(
                    terminal.backend().buffer()[(width - 1, height - 1)].symbol(),
                    "┘"
                );
            }
        }
    }

    #[test]
    fn test_backend_monitor_shows_actual_input_instead_of_synthetic_neurons() {
        use ratatui::{Terminal, backend::TestBackend};

        let mut state = RunState::default();
        for active in [false, true] {
            if active {
                state.ingest("training 1/2 (50%) loss=4.0 tokens_per_second=100 eta=1s");
            }
            for (width, height) in [
                (80, 24),
                (120, 40),
                (160, 40),
                (192, 40),
                (240, 40),
                (60, 20),
                (30, 10),
                (10, 5),
            ] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                for tab in 0..TABS.len() {
                    terminal.draw(|f| draw(f, &state, tab)).unwrap();
                    let text: String = terminal
                        .backend()
                        .buffer()
                        .content()
                        .iter()
                        .map(|cell| cell.symbol())
                        .collect();
                    assert!(text.contains(TABS[tab]));
                    if width >= 80 {
                        assert!(!text.contains("neuron /"));
                        if tab == 0 {
                            assert!(text.contains("actual input"));
                            assert!(text.contains("No input window recorded"));
                            assert!(text.contains("run metrics"));
                            assert!(!text.contains("not written yet"));
                        }
                        let expected = [
                            "run metrics",
                            "chain / checkpoints",
                            "model / configuration",
                            "dataset feed",
                            "conversation",
                            "parameters",
                            "HF login",
                            "Kaggle",
                            "memory",
                            "runs",
                            "benchmark",
                            "live sample",
                            "hardware",
                            "math / read-only",
                            "runtime compute",
                            "resource limits",
                            "library",
                            "mixer",
                            "eval",
                            "HF backup",
                            "cloud log",
                            "phone ping",
                            "updates",
                            "support",
                            "GitHub",
                            "sweeps",
                            "timeline",
                        ];
                        // These six bodies belong to the shell. Local screens
                        // render afterward and have their own TestBackend tests.
                        if let Some(expected) = expected.get(tab) {
                            assert!(text.contains(*expected));
                        }
                    } else if width >= 30 && tab == 0 {
                        let buffer = terminal.backend().buffer();
                        let metrics_row = buffer
                            .content()
                            .chunks(width as usize)
                            .find(|row| {
                                row.iter()
                                    .map(|c| c.symbol())
                                    .collect::<String>()
                                    .contains("run metrics")
                            })
                            .unwrap();
                        assert_eq!(
                            metrics_row[0].symbol(),
                            "┌",
                            "narrow metrics retain full width"
                        );
                        assert_eq!(metrics_row[width as usize - 1].symbol(), "┐");
                    }
                }
            }
        }
    }

    #[test]
    fn resumed_completion_requires_a_new_confirmed_save_and_preserves_failure_context() {
        let mut state = RunState::default();
        state.ingest(
            "progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/tmp/new.pssa",
        );
        state.ingest("last_checkpoint=/tmp/old.pssa");
        state.ingest("training 500/500 (100%) loss=2 tokens_per_second=90 optimizer_updates=500 global_update=600");
        state.ingest("training_seconds=12 optimizer_updates=500");
        assert!(
            state
                .checkpoint_context()
                .contains("waiting for saved_checkpoint event")
        );
        assert!(
            state
                .checkpoint_context()
                .contains("previous checkpoint /tmp/old.pssa")
        );
        assert_eq!(state.health_status().normal_label, "SAVE PENDING");
        state.ingest("error: cannot save checkpoint: disk full");
        assert!(
            state
                .checkpoint_context()
                .contains("checkpoint save unconfirmed")
        );
        assert!(state.checkpoint_context().contains("disk full"));
        assert!(
            state
                .checkpoint_context()
                .contains("previous checkpoint /tmp/old.pssa")
        );
        assert!(state.health_status().level == HealthLevel::Problem);
        state.ingest(
            "progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/tmp/new.pssa",
        );
        state.ingest("training_seconds=12 optimizer_updates=500");
        state.ingest("saved_checkpoint=/tmp/new.pssa");
        assert_eq!(state.checkpoint_context(), "/tmp/new.pssa");
        assert_eq!(state.health_status().normal_label, "DONE");
        state.problem = Some("loss spike (derived health alert)".into());
        assert!(state.checkpoint_context().contains("Checkpoint saved"));
        assert!(state.checkpoint_context().contains("reported issue"));
        assert!(
            !state.checkpoint_context().contains("run stopped"),
            "a health heuristic is not a producer exit outcome"
        );
    }

    #[test]
    fn partial_logs_do_not_invent_global_end_steps_or_filename_latest_order() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.ingest("progress_schema=1 updates_total=500 checkpoint_target=/tmp/new.pssa");
        assert!(state.checkpoint_context().contains("500 run-local updates"));
        assert!(!state.checkpoint_context().contains("planned step 500"));
        state.ingest("loss=2 tokens_per_second=90 optimizer_updates=500 global_update=600");
        state.ingest("training_seconds=12 optimizer_updates=500");
        assert_eq!(
            state.current_step(),
            Some(600),
            "a local summary cannot erase a recorded global step"
        );
        state.checkpoints = vec![("ck01.pssa".into(), None), ("model.pssa".into(), None)];
        state.ingest("saved_checkpoint=/tmp/ck01.pssa");
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| draw_chain(f, f.area(), &state)).unwrap();
        let rows: Vec<String> = terminal
            .backend()
            .buffer()
            .content()
            .chunks(80)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();
        assert!(
            rows.iter()
                .any(|row| row.contains("ck01.pssa") && row.contains("last recorded save"))
        );
        assert!(
            rows.iter()
                .filter(|row| row.contains("model.pssa"))
                .all(|row| !row.contains("last recorded save"))
        );
    }

    #[test]
    fn checkpoint_only_model_and_math_use_real_header_dimensions_without_inventing_runtime_fields()
    {
        use ratatui::{Terminal, backend::TestBackend};
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/depth_one_main85d9d33_trained.pssa");
        let dims = library::checkpoint_dims(&path).unwrap();
        let mut state = RunState::default();
        state.use_header_dims(&path, &dims);
        assert_eq!(state.width.as_deref(), Some(dims.as_str()));
        assert!(state.corpus.is_none() && state.schedule.is_none() && state.feed.is_none());
        assert!(state.memory_used.is_none() && state.learning_rate.is_none());
        for tab in [2, keybindings::MATH_TAB] {
            let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
            terminal
                .draw(|f| {
                    if tab == 2 {
                        draw_model(f, f.area(), &state);
                    } else {
                        state.math_view.draw(f, f.area(), &state);
                    }
                })
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("Header hints from"), "{text}");
        }
        state.ingest("model=pssa parameters=123 vocab=256 depth=2 loops=3");
        assert!(state.configuration_source.is_none());
        assert!(state.width.is_none());
        state.ingest("progress_schema=2 updates_total=100 prior_updates=0");
        state.use_header_dims(&path, &dims);
        assert!(
            state.width.is_none(),
            "old headers cannot fill a new run's missing banner"
        );
    }

    #[test]
    fn checkpoint_waiting_context_tracks_real_run_end_step_and_output_directory() {
        let mut state = RunState::default();
        state.ingest("progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/tmp/real run/model.pssa");
        state.ingest("training 120/500 (24%) loss=3 tokens_per_second=90 optimizer_updates=120 global_update=220");
        assert_eq!(state.chain_dir, PathBuf::from("/tmp/real run"));
        assert!(state.checkpoint_context().contains("step 600"));
        assert!(state.checkpoint_context().contains("current step 220"));
        state.ingest("training_seconds=3 optimizer_updates=500");
        assert!(state.checkpoint_context().contains("complete at step 600"));
        state.ingest("saved_checkpoint=/tmp/real run/model.pssa");
        assert_eq!(state.checkpoint_context(), "/tmp/real run/model.pssa");
        assert_eq!(state.checkpoints[0].1, Some(3.0));
        state.ingest(
            "progress_schema=2 updates_total=10 prior_updates=0 checkpoint_target=model.pssa",
        );
        assert_eq!(state.chain_dir, PathBuf::from("."));
        assert!(state.last_checkpoint.is_none());
        assert!(state.live_loss.is_none());
        assert!(state.checkpoints.is_empty());
        assert!(state.checkpoint_context().contains("current step 0"));
    }

    #[test]
    fn real_save_timing_elapsed_and_input_are_visible_in_the_monitor_shell() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.ingest(
            "progress_schema=2 updates_total=500 prior_updates=0 checkpoint_target=/tmp/model.pssa",
        );
        state.ingest("training 120/500 (24%) loss=3 tokens_per_second=90 optimizer_updates=120 global_update=120 elapsed_seconds=12.5 grad_norm=0.25 skipped_updates=2 feed_dataset=corpus.txt feed_snippet=actual%20dataset%20window feed_token_ids=1%2C2");
        for (width, height) in [(80, 24), (120, 40)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| draw(f, &state, 0)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            for expected in [
                "step 120",
                "planned step 500",
                "current step 120",
                "elapsed 12.5s",
                "actual dataset window",
            ] {
                assert!(
                    text.contains(expected),
                    "{width}x{height} missing {expected}: {text}"
                );
            }
        }
        assert_eq!(state.grad_norm, Some(0.25));
        assert_eq!(state.skipped_updates, Some(2));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| draw(f, &state, 0)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("grad 2.500e-1 / skipped 2"), "{text}");
    }

    #[test]
    fn reported_elapsed_resume_and_failed_save_context_do_not_invent_data() {
        let mut state = RunState::default();
        state.ingest(
            "progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/tmp/new.pssa",
        );
        state.ingest("last_checkpoint=/tmp/old.pssa");
        state.ingest("training 120/500 (24%) loss=3 tokens_per_second=90 optimizer_updates=120 global_update=220 elapsed_seconds=12.5");
        assert!(
            state
                .checkpoint_context()
                .contains("previous checkpoint /tmp/old.pssa")
        );
        assert!(state.checkpoint_context().contains("planned step 600"));
        assert_eq!(state.resumed_from.as_deref(), Some("/tmp/old.pssa"));
        state.run_started_at = Some(Instant::now() - Duration::from_secs(600));
        state.training_active = false;
        assert_eq!(state.elapsed_context(), "12.5s (last report)");
        state.ingest(
            "progress_schema=2 updates_total=10 prior_updates=0 checkpoint_target=/tmp/new.pssa",
        );
        state.training_active = false;
        state.problem = Some("save failed: disk full".into());
        assert!(
            state
                .checkpoint_context()
                .contains("save failed: disk full")
        );
        assert!(state.elapsed_seconds.is_none());
        assert!(state.resumed_from.is_none());
        assert!(state.grad_norm.is_none());
        state.ingest("progress_schema=1");
        assert!(
            state.current_step().is_none(),
            "legacy unknown prior must not reuse another run's counter"
        );
    }

    #[test]
    fn new_reported_schedule_replaces_the_previous_warmup() {
        let mut state = RunState::default();
        state.ingest("lr_schedule=per-run horizon=100 from_step=0 to_step=100 warmup=10");
        state.ingest("schedule 1 epoch(s), 100 updates, lr first=0.00010000 last=0.00001000 (base 0.00100000, horizon 100)");
        state.ingest("progress_schema=2 prior_updates=0 updates_total=100");
        assert_eq!(state.expected_lr_warmup, Some(10));
        state.ingest("lr_schedule=per-run horizon=200 from_step=0 to_step=200 warmup=0");
        state.ingest("schedule 1 epoch(s), 200 updates, lr first=0.00100000 last=0.00001000 (base 0.00100000, horizon 200)");
        state.ingest("progress_schema=2 prior_updates=0 updates_total=200");
        assert_eq!(state.expected_lr_warmup, Some(0));
        assert_eq!(state.expected_lr_total, Some(200));
    }

    #[test]
    fn metric_keys_never_match_another_fields_suffix() {
        assert_eq!(
            parse_kv::<f64>("loss_average=9 dream_loss=8 loss=3", "loss="),
            Some(3.0)
        );
        assert_eq!(
            parse_kv::<u64>("feed_tokens=90 epoch_tokens=10 tokens=5", "tokens="),
            Some(5)
        );
        assert_eq!(
            parse_kv::<u64>("feed_step=20 skipped_updates=7 updates=4", "updates="),
            Some(4)
        );
        assert_eq!(
            parse_kv::<f64>("(base 0.00100000, horizon 100)", "base "),
            Some(0.001)
        );
        let mut state = RunState::default();
        state.ingest("training 1/10 (10%) loss_average=9 loss=3 tokens_per_second=90 optimizer_updates=1 global_update=1");
        state.ingest("dream phase=end update=1 mode=memory dream_loss=NaN");
        assert_eq!(state.live_loss, Some(3.0));
        assert_eq!(state.loss_average, Some(9.0));
        assert!(
            state.problem.is_none(),
            "unreported training loss must not become dream loss"
        );
    }

    #[test]
    fn rejects_oversized_encoded_strings_instead_of_accepting_a_partial_window() {
        let line = format!("feed_snippet={}%ZZ", "x".repeat(4096));
        assert!(parse_log_value(&line, "feed_snippet").is_none());
    }

    #[test]
    fn feed_log_roundtrip_preserves_unicode_metrics_and_paths() {
        let mut state = RunState::default();
        let snippet = "héllo 世界 100% loss=NaN\nnext";
        state.ingest(&format!("training 1/2 (50%) loss=4 tokens_per_second=100 feed_dataset=owner/name feed_config=small feed_split=train feed_field=body.text feed_rows=2 feed_tokens=17 feed_row=3 feed_snippet={} feed_token_ids=12%2C34%2C56 eta=2h 14m 09s last_checkpoint=/tmp/my chain/ck02.pssa", ui::encode_log_value(snippet)));
        let feed = state.feed.as_ref().unwrap();
        assert_eq!(feed.dataset, "owner/name");
        assert_eq!(feed.snippet, "héllo 世界 100% loss=NaN next");
        assert_eq!(feed.token_ids, [12, 34, 56]);
        assert_eq!(
            (feed.rows, feed.tokens, feed.row),
            (Some(2), Some(17), Some(3))
        );
        assert_eq!(state.live_loss, Some(4.0));
        assert_eq!(state.progress_pct, Some(50.0));
        assert_eq!(state.eta.as_deref(), Some("2h 14m 09s"));
        assert_eq!(
            state.last_checkpoint.as_deref(),
            Some("/tmp/my chain/ck02.pssa")
        );
        assert!(state.problem.is_none());
        state.ingest("progress_schema=2 updates_total=2");
        assert!(state.feed.is_none(), "new run clears previous samples");
        assert_eq!(parse_log_value("feed_snippet=%ZZ", "feed_snippet"), None);
        assert_eq!(parse_log_value("feed_snippet=%", "feed_snippet"), None);
        assert_eq!(parse_log_value("feed_snippet=%FF", "feed_snippet"), None);
        assert_eq!(
            parse_log_value("feed_snippet=%1B%0A", "feed_snippet").as_deref(),
            Some("  ")
        );
    }

    #[test]
    fn test_backend_renders_feed_frames_and_narrow_fallbacks() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut state = RunState::default();
        state.ingest("feed_dataset=owner/name feed_config=default feed_split=train feed_field=text feed_rows=7 feed_tokens=128 feed_row=2 feed_snippet=hello%20world feed_token_ids=12%2C34%2C56");
        let mut terminal = Terminal::new(TestBackend::new(80, 19)).unwrap();
        let mut frames = Vec::new();
        for millis in [0, 420, 840] {
            terminal
                .draw(|f| draw_feed_at(f, f.area(), &state, Duration::from_millis(millis)))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let text: String = buffer.content().iter().map(|c| c.symbol()).collect();
            assert!(text.contains("owner/name"));
            assert!(text.contains("7 rows consumed / 128 tokens / selected row 2"));
            assert!(text.contains("Token pieces → IDs"));
            assert!(text.contains("IDs: 12, 34, 56"));
            assert_eq!(buffer.cell((0, 0)).unwrap().symbol(), "┌");
            assert_eq!(buffer.cell((79, 18)).unwrap().symbol(), "┘");
            assert!(!buffer.content().iter().any(|c| c.fg == BRIGHT_RED));
            frames.push(buffer.clone());
        }
        assert_eq!(
            frames[0], frames[1],
            "recorded input does not invent motion"
        );
        assert_eq!(frames[1], frames[2]);
        for (width, height) in [(120, 40), (80, 24), (60, 20), (30, 10), (10, 5), (1, 1)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| draw(f, &state, 3)).unwrap();
        }
    }

    #[test]
    fn test_backend_hex_background_preserves_panels_and_resize_fallback() {
        use ratatui::{Terminal, backend::TestBackend};

        let mut pressed = HexBackground::default();
        pressed.mouse(0, 8);
        for _ in 0..10 {
            pressed.advance(FRAME_INTERVAL);
        }
        // The same hovered state must remain safe after resizing, on every tab.
        for (width, height) in [(120, 40), (80, 24), (79, 24), (30, 10), (1, 1)] {
            for tab in 0..TABS.len() {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal
                    .draw(|f| {
                        draw_with_background(f, &RunState::default(), tab, &pressed, None, None)
                    })
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let content = HexBackground::content_area(buffer.area);
                // Monitor has a clock-driven neuron animation; the other
                // default tabs are static, so compare every panel cell exactly.
                if tab != 0 {
                    let hovered = buffer.clone();
                    terminal
                        .draw(|f| draw(f, &RunState::default(), tab))
                        .unwrap();
                    let resting = terminal.backend().buffer();
                    for y in content.y..content.bottom() {
                        for x in content.x..content.right() {
                            assert_eq!(hovered[(x, y)], resting[(x, y)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_backend_renders_done_status_badge() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = RunState::default();
        state.ingest("training 1/1 (100%) loss=4.0 tokens_per_second=100 optimizer_updates=1 updates_total=1");
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("[ DONE ]"));
    }

    #[test]
    fn test_backend_renders_problem_reason_in_bright_red() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = RunState::default();
        state.ingest("training 1/2 (50%) loss=4.0 tokens_per_second=100 eta=1s");
        state.ingest("training 2/2 (100%) loss=7.0 tokens_per_second=100 eta=0s");
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
        assert!(text.contains("loss spike"));
        assert!(buffer.content().iter().any(|cell| cell.fg == BRIGHT_RED));
    }
}
