//! Live read-only dashboard for oxide training runs.
//!
//! `oxide train ... | oxide tui` renders the run as a full-screen dashboard in
//! your own terminal. The TUI never writes to the model, the checkpoints or the
//! logs: it reads the structured `key=value` lines the CLI already prints and
//! the chain directory, and draws. When stdin is not a pipe it prints its help
//! and exits, and Ctrl+C / q always hands the terminal back cleanly.

use crate::ui;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Tabs, Wrap};
use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const TABS: [&str; 3] = ["monitor", "chain", "model"];
const PSSA_LOGO: [&str; 3] = [
    "███  ████  ████   ███",
    "█ █  █     █      █ █",
    "███  ████  ████   ███",
];
const NORMAL_GREEN: Color = Color::Rgb(0x39, 0xe0, 0x7a);
const AMBER: Color = Color::Rgb(0xff, 0xbf, 0x00);
const BRIGHT_RED: Color = Color::Rgb(0xff, 0x2f, 0x3f);
const STALL_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct RunState {
    // header card
    corpus: Option<String>,
    vocab: Option<String>,
    width: Option<String>,
    memory: Option<String>,
    schedule: Option<String>,
    // monitor tab
    progress_pct: Option<f64>,
    live_loss: Option<f64>,
    loss_average: Option<f64>,
    tok_s: Option<f64>,
    eta: Option<String>,
    updates_done: Option<u64>,
    updates_total: Option<u64>,
    updates_remaining: Option<u64>,
    learning_rate: Option<f64>,
    memory_used: Option<u64>,
    memory_capacity: Option<u64>,
    checkpoint_number: Option<u64>,
    checkpoint_target: Option<String>,
    last_checkpoint: Option<String>,
    // losses over time (for the sparkline)
    loss_series: Vec<f64>,
    // Health checks are presentation-only; they never affect training.
    problem: Option<String>,
    warning: Option<String>,
    tok_s_history: Vec<f64>,
    last_progress_at: Option<Instant>,
    training_active: bool,
    expected_lr_base: Option<f64>,
    expected_lr_total: Option<u64>,
    expected_lr_warmup: Option<u64>,
    // last finished epoch line
    epoch_loss: Option<f64>,
    epoch_tokens: Option<u64>,
    epoch_updates: Option<u64>,
    // summary card
    wall: Option<String>,
    throughput: Option<String>,
    training_seconds: Option<f64>,
    optimizer_updates: Option<u64>,
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
            // Start the stall clock even before the first update arrives.
            self.last_progress_at = Some(Instant::now());
            self.training_active = true;
            self.problem = None;
            self.warning = None;
            self.loss_series.clear();
            self.tok_s_history.clear();
        }

        let raw_loss = parse_kv::<f64>(line, "loss=").or_else(|| parse_kv(line, "loss "));
        let is_progress = line.contains("tokens_per_second=") || line.contains("tok/s");
        if is_progress {
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
            if let Some((_, eta)) = line.split_once("eta=").or_else(|| line.split_once("eta ")) {
                // ui::duration can contain spaces, e.g. `2h 14m 09s`.
                self.eta = Some(eta.trim().to_string());
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
        if let Some(t) = parse_kv(line, "training_seconds=") {
            self.training_seconds = Some(t);
            self.training_active = false;
        }
        if let Some(u) = parse_kv(line, "optimizer_updates=") {
            self.optimizer_updates = Some(u);
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
        }
        if let Some(path) = line.split("last_checkpoint=").nth(1).map(str::trim)
            && !path.is_empty()
            && path != "-"
        {
            self.last_checkpoint = Some(path.to_string());
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
    }

    fn note_checkpoint(&mut self, path: &str) {
        let path = path.trim();
        if path.is_empty() || path == "-" {
            return;
        }
        self.last_checkpoint = Some(path.to_string());
        self.checkpoint_number = checkpoint_number(path);
        let name = PathBuf::from(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
        if !(name.ends_with(".pssa") || name.ends_with(".trfm")) {
            return;
        }
        if !self.checkpoints.iter().any(|(n, _)| *n == name) {
            self.checkpoints.push((name, None));
            self.checkpoints
                .sort_by_key(|(name, _)| checkpoint_sort_key(name));
        }
    }

    /// Scan the chain directory for checkpoint files; also carry any loss
    /// recorded on disk in sibling `.loss` files (written by future runs).
    fn refresh_chain(&mut self) {
        let Ok(entries) = std::fs::read_dir(&self.chain_dir) else {
            return;
        };
        let mut names: Vec<(String, Option<f64>)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pssa") || name.ends_with(".trfm") {
                let loss = std::fs::read_to_string(entry.path().with_extension("loss"))
                    .ok()
                    .and_then(|s| s.trim().parse::<f64>().ok());
                names.push((name, loss));
            }
        }
        names.sort_by_key(|(name, _)| checkpoint_sort_key(name));
        for (name, loss) in names {
            if let Some(existing) = self.checkpoints.iter_mut().find(|(n, _)| *n == name) {
                if loss.is_some() {
                    existing.1 = loss;
                }
            } else {
                self.checkpoints.push((name, loss));
            }
        }
        self.checkpoints
            .sort_by_key(|(name, _)| checkpoint_sort_key(name));
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
        let normal_label = if self.training_active {
            "TRAINING"
        } else if self.last_progress_at.is_some() {
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
            if self.training_active && now.saturating_duration_since(at) > STALL_TIMEOUT {
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

/// Pull `key=value` (or `key value`) numeric pairs out of a line.
fn parse_kv<T: std::str::FromStr>(line: &str, key: &str) -> Option<T> {
    let rest = line.split(key).nth(1)?;
    let token = rest
        .trim_start()
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

fn parse_field_exact(line: &str, label: &str) -> Option<String> {
    let padded = format!("{:<16}", label);
    line.strip_prefix(&padded).map(|v| v.trim().to_string())
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

fn run_app(rx: mpsc::Receiver<String>, chain_dir: PathBuf) -> io::Result<()> {
    let mut terminal = ratatui::init();
    let mut state = RunState {
        chain_dir: chain_dir.clone(),
        ..RunState::default()
    };
    let mut tab = 0usize;
    let mut last_chain_scan = std::time::Instant::now() - Duration::from_secs(60);
    let mut input_closed = false;

    loop {
        // Drain stdin. A completed producer should leave the final dashboard
        // frame visible once, then let the wrapper restore the terminal.
        loop {
            match rx.try_recv() {
                Ok(line) => state.ingest(&line),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    input_closed = true;
                    state.training_active = false;
                    break;
                }
            }
        }
        if last_chain_scan.elapsed() > Duration::from_secs(5) {
            state.refresh_chain();
            last_chain_scan = std::time::Instant::now();
        }

        terminal.draw(|f| draw(f, &state, tab))?;
        if input_closed {
            break;
        }

        // poll events for up to 200ms, then loop back to stdin
        if crossterm::event::poll(Duration::from_millis(200))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if key.kind == crossterm::event::KeyEventKind::Press {
                    match key.code {
                        crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => {
                            break;
                        }
                        crossterm::event::KeyCode::Tab | crossterm::event::KeyCode::Right => {
                            tab = (tab + 1) % TABS.len();
                        }
                        crossterm::event::KeyCode::Left => {
                            tab = (tab + TABS.len() - 1) % TABS.len();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    ratatui::restore();
    Ok(())
}

fn accent() -> Style {
    Style::new().fg(NORMAL_GREEN)
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(accent().add_modifier(Modifier::DIM))
        .title(Line::styled(title, accent().add_modifier(Modifier::BOLD)))
}

fn divider(width: u16) -> Line<'static> {
    Line::styled(
        "╌".repeat(width.into()),
        accent().add_modifier(Modifier::DIM),
    )
}

fn status_badge(health: &HealthStatus) -> Line<'static> {
    let label = match health.level {
        HealthLevel::Problem => "ERROR",
        HealthLevel::Warning | HealthLevel::Normal => {
            if health.normal_label == "DONE" {
                "DONE"
            } else {
                "TRAINING"
            }
        }
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

fn draw(f: &mut ratatui::Frame, state: &RunState, tab: usize) {
    let area = f.area();
    if area.is_empty() {
        return;
    }
    f.render_widget(
        Block::default().style(Style::new().fg(Color::Gray).bg(Color::Black)),
        area,
    );
    let health = state.health_status();
    // Do not squeeze bordered widgets into one-cell fragments on tiny screens.
    if area.width < 30 || area.height < 10 {
        let detail = match tab {
            0 => format!(
                "{:.0}%  loss {:.4}",
                state.progress_pct.unwrap_or(0.0),
                state.live_loss.or(state.epoch_loss).unwrap_or(0.0)
            ),
            1 => format!(
                "{} checkpoints in {}",
                state.checkpoints.len(),
                state.chain_dir.display()
            ),
            _ => format!("width {}", state.width.as_deref().unwrap_or("-")),
        };
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    format!("PSSA / {}", TABS[tab.min(TABS.len() - 1)]),
                    accent(),
                ),
                status_badge(&health),
                Line::from(detail),
                Line::from("q quit / tab switch"),
                Line::from("Enlarge for full view"),
            ]),
            area,
        );
        return;
    }
    let large_title = area.width >= 80 && area.height >= 22;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);
    let header = chunks[0];
    if area.width >= 40 {
        f.render_widget(
            Paragraph::new(PSSA_LOGO.map(|row| Line::styled(row, accent())).to_vec()),
            Rect::new(header.x, header.y, 24, 3),
        );
        f.render_widget(
            Paragraph::new(status_badge(&health)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(health.color())),
            ),
            Rect::new(header.x + 24, header.y, header.width - 24, 3),
        );
    } else {
        let title = if area.width >= 34 {
            "PSSA / oxide tui  q quit  tab switch"
        } else {
            "PSSA / oxide tui  q quit"
        };
        f.render_widget(
            Paragraph::new(vec![Line::styled(title, accent()), status_badge(&health)]),
            header,
        );
    }
    let nav = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(29), Constraint::Min(0)])
        .split(chunks[1]);
    f.render_widget(
        Tabs::new(TABS)
            .select(tab)
            .style(accent().add_modifier(Modifier::DIM))
            .highlight_style(
                accent()
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                    .remove_modifier(Modifier::DIM),
            )
            .divider(" / ")
            .padding(" ", " "),
        nav[0],
    );
    if large_title {
        f.render_widget(
            Paragraph::new("q quit   tab / ← → switch").right_aligned(),
            nav[1],
        );
    }
    f.render_widget(Paragraph::new(divider(area.width)), chunks[2]);

    match tab {
        0 => draw_monitor(f, chunks[3], state),
        1 => draw_chain(f, chunks[3], state),
        _ => draw_model(f, chunks[3], state),
    }
}

fn draw_monitor(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    // Prefer the metrics to a squashed graph on short terminals. Each visible
    // panel retains at least one content row and a complete top/bottom border.
    let show_history = area.height >= 16;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(if show_history { 3 } else { 0 }),
            Constraint::Length(if show_history { 10 } else { area.height - 3 }),
        ])
        .split(area);

    let pct = state.progress_pct.unwrap_or(0.0);
    let loss = state.live_loss.or(state.epoch_loss).unwrap_or(0.0);
    let average = state.loss_average.or(state.live_loss).unwrap_or(loss);
    let health = state.health_status();
    let health_style = Style::new().fg(health.color()).add_modifier(Modifier::BOLD);
    let done = state
        .updates_done
        .or(state.optimizer_updates)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());
    let total = state
        .updates_total
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".into());
    let label = if area.width >= 80 {
        format!("{pct:.0}%  |  updates {done}/{total}  |  loss {loss:.4}  avg {average:.4}")
    } else {
        format!("{pct:.0}%  {done}/{total}  loss {loss:.4}")
    };
    // Phase 2 only restyles the frame; keep the existing gauge and sparkline.
    f.render_widget(
        Gauge::default()
            .label(Span::styled(label, health_style))
            .ratio((pct / 100.0).clamp(0.0, 1.0))
            .gauge_style(Style::new().fg(health.color()).bg(Color::Black))
            .block(panel(" monitor ")),
        chunks[0],
    );

    if show_history {
        let block = panel(" loss history (raw updates) ");
        if state.loss_series.is_empty() {
            f.render_widget(
                Paragraph::new("Waiting for progress samples...").block(block),
                chunks[1],
            );
        } else {
            let data: Vec<u64> = state
                .loss_series
                .iter()
                .map(|l| (l.max(0.0) * 1000.0) as u64)
                .collect();
            f.render_widget(
                Sparkline::default()
                    .block(block)
                    .data(&data)
                    .style(Style::new().fg(health.color())),
                chunks[1],
            );
        }
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
    let lines = vec![
        Line::from(vec![
            Span::raw("status      "),
            Span::styled(health.label(), health_style),
        ]),
        Line::from(format!(
            "speed       {:.0} tokens/s  ETA {}",
            state.tok_s.unwrap_or(0.0),
            state.eta.as_deref().unwrap_or("-")
        )),
        Line::from(format!(
            "optimizer   {done}/{total} updates  {} remaining  lr {}",
            state
                .updates_remaining
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            state
                .learning_rate
                .map(|lr| format!("{lr:.6e}"))
                .unwrap_or_else(|| "-".into())
        )),
        Line::from(format!("memory bank {memory}")),
        Line::from(format!("checkpoint  {checkpoint}")),
        Line::from(format!(
            "last saved  {}",
            state
                .last_checkpoint
                .as_deref()
                .unwrap_or("not written yet")
        )),
        Line::from(format!(
            "last epoch  loss {:.4}   tokens {}   updates {}",
            state.epoch_loss.unwrap_or(0.0),
            state.epoch_tokens.unwrap_or(0),
            state.epoch_updates.unwrap_or(0)
        )),
        Line::from(format!(
            "run         wall {}   resumed from {}   prior steps {}",
            state.wall.as_deref().unwrap_or("-"),
            state.resumed_from.as_deref().unwrap_or("-"),
            state
                .prior_steps
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into())
        )),
    ];
    f.render_widget(
        Paragraph::new(lines).block(panel(if show_history {
            " run metrics "
        } else {
            " run metrics / compact "
        })),
        chunks[2],
    );
}

fn draw_chain(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let mut rows = vec![
        Line::from(format!("directory  {}", state.chain_dir.display())),
        divider(area.width.saturating_sub(2)),
    ];
    if state.checkpoints.is_empty() {
        rows.push(Line::from("No checkpoint files found yet."));
    } else {
        let last = state.checkpoints.len().saturating_sub(1);
        rows.extend(
            state
                .checkpoints
                .iter()
                .enumerate()
                .map(|(i, (name, loss))| {
                    let marker = if i == last { "●" } else { "○" };
                    let loss_text = loss
                        .map(|l| format!("{l:.4}"))
                        .unwrap_or_else(|| "—".into());
                    let suffix = if i == last { "  (latest)" } else { "" };
                    Line::styled(
                        format!("{marker} {name}  loss {loss_text}{suffix}"),
                        accent(),
                    )
                }),
        );
    }
    f.render_widget(
        Paragraph::new(rows)
            .wrap(Wrap { trim: false })
            .block(panel(" chain / checkpoints ")),
        area,
    );
}

fn draw_model(f: &mut ratatui::Frame, area: Rect, state: &RunState) {
    let mut lines = vec![
        Line::from("Configuration from the training log"),
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
            Span::styled(format!("{label:<14}"), accent()),
            Span::raw(value.as_deref().unwrap_or("not reported")),
        ]));
    }
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" model / configuration ")),
        area,
    );
}

pub fn run(args: &[String]) -> Result<(), String> {
    // Keep the default useful on a local checkout; Kaggle callers can pass
    // their mounted chain explicitly (the training scripts already do).
    let mut chain_dir = PathBuf::from("chain");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--chain" | "-c" => {
                let value = args.get(i + 1).ok_or_else(|| {
                    "option '--chain' requires a directory; usage: oxide tui [-c|--chain DIR]"
                        .to_string()
                })?;
                if value.starts_with('-') {
                    return Err(format!(
                        "option '{}' requires a directory; usage: oxide tui [-c|--chain DIR]",
                        args[i]
                    ));
                }
                chain_dir = PathBuf::from(value);
                i += 1;
            }
            other => {
                return Err(format!(
                    "unknown tui flag '{other}'; usage: oxide tui [-c|--chain DIR]"
                ));
            }
        }
        i += 1;
    }

    if io::stdin().is_terminal() {
        println!("Usage: oxide_ai_pssa train ... --no-tui | oxide_ai_pssa tui [-c|--chain DIR]");
        println!(
            "Example: oxide_ai_pssa train data/corpus.txt -o chain/ck01.pssa --no-tui | oxide_ai_pssa tui --chain chain"
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

    if let Err(e) = run_app(rx, chain_dir) {
        ratatui::restore();
        return Err(e.to_string());
    }
    Ok(())
}

#[allow(dead_code)]
fn unused_helpers() {
    let _ = parse_field_exact(
        "  schedule        1 epoch(s), 446 updates, lr 0.001",
        "schedule",
    );
    let _ = ui::bold("");
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn producer_output_round_trips_through_parser() {
        // A child process makes ui::Progress actually write to a pipe, avoiding
        // unstable stdout-capture APIs or a duplicate copy of its format string.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tui::tests::ui_producer_fixture", "--nocapture"])
            .env("OXIDE_TUI_PRODUCER_FIXTURE", "1")
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
        if std::env::var_os("OXIDE_TUI_PRODUCER_FIXTURE").is_none() {
            return;
        }
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
        assert!(row(0).contains("███"));
        assert!(row(3).contains("monitor"));
        assert!(row(3).contains("chain"));
        assert!(row(3).contains("model"));
        assert!(!row(3).contains("modelt"));
        assert!(row(4).contains("╌"));
        assert!(row(0).contains("┌"));
        assert!(row(2).contains("└"));
        assert!(row(1).contains("[ TRAINING ]"));
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
