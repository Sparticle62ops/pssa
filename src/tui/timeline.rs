//! Read-only checkpoint history: filenames + bounded recorded logs, never weights.
//! A single background scan is allowed at a time, including across run switches.
use super::{
    AMBER, RunState, accent, checkpoint_sort_key, panel, panel_area, parse_field, process::clean,
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

const MAX_ENTRIES: usize = 512;
const MAX_LOGS: usize = 128;
const MAX_DIR_ENTRIES: usize = 4096;
const MAX_LOG_BYTES: usize = 2 * 1024 * 1024;
const MAX_SCAN_BYTES: usize = 8 * 1024 * 1024;
const MAX_LINE: usize = 16 * 1024;
const MAX_LOG_LINES: usize = 20_000;
const SCAN_TIME: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, Default)]
struct Metrics {
    loss: Option<f64>,
    speed: Option<f64>,
    update: Option<u64>,
    observed: bool,
}
impl Metrics {
    fn ingest(&mut self, line: &str) {
        if field(line, "progress_schema").is_some() {
            *self = Self::default();
        }
        // Only genuine training progress/epoch rows; no held-out or unrelated
        // key containing `loss=` may silently become a training measurement.
        if field(line, "tokens_per_second").is_some() || line.starts_with("epoch ") {
            self.observed = true;
            if let Some(value) = field(line, "loss") {
                self.loss = value.parse::<f64>().ok().filter(|v| v.is_finite());
            }
            if let Some(value) = field(line, "tokens_per_second") {
                self.speed = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0);
            }
        }
        if let Some(value) = parse_field(line, "throughput") {
            self.observed = true;
            self.speed = value
                .split_whitespace()
                .next()
                .and_then(|v| v.replace(',', "").parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0);
        }
        if let Some(value) =
            field(line, "global_update").or_else(|| field(line, "optimizer_updates"))
        {
            self.update = value.split('/').next().and_then(|v| v.parse().ok());
        }
    }
}
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|part| {
        let (name, value) = part.split_once('=')?;
        (name == key).then_some(value)
    })
}
fn saved_path(line: &str) -> Option<&str> {
    let path = line
        .strip_prefix("saved_checkpoint=")
        .or_else(|| {
            line.split_once("checkpoint written to ")
                .map(|(_, path)| path)
        })?
        .trim();
    (!path.is_empty() && path != "-").then_some(path)
}

#[derive(Clone, Debug)]
struct Entry {
    path: PathBuf,
    metrics: Metrics,
    source: Option<String>,
    explicit: bool,
}
struct Scan {
    entries: Vec<Entry>,
    note: String,
}

// Lexical identity preserves spaces and never opens a checkpoint. In particular,
// `/another/run/model.pssa` must NOT be treated as this directory's model.pssa.
fn absolute(path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            _ => out.push(component.as_os_str()),
        }
    }
    out
}
fn event_index(path: &str, root: &Path, indices: &HashMap<PathBuf, usize>) -> Option<usize> {
    let path = Path::new(path);
    indices.get(&absolute(path)).copied().or_else(|| {
        // Bare sibling names are useful in copied/legacy logs. Never strip an
        // explicit directory prefix: that would misassociate another run.
        (path.components().count() == 1)
            .then(|| indices.get(&root.join(path)).copied())
            .flatten()
    })
}

fn associate(
    lines: impl IntoIterator<Item = String>,
    source: &str,
    log: Option<&Path>,
    root: &Path,
    entries: &mut [Entry],
    indices: &HashMap<PathBuf, usize>,
) {
    let mut metrics = Metrics::default();
    let mut target = None;
    let mut target_seen = false;
    let mut finished = false;
    let mut explicit = false;
    for raw in lines {
        if raw.len() > MAX_LINE {
            continue;
        }
        let line = clean(&raw);
        let line = line.trim();
        if field(line, "progress_schema").is_some() {
            target = None;
            target_seen = false;
            finished = false;
            explicit = false;
        }
        metrics.ingest(line);
        if let Some((_, path)) = line.split_once("checkpoint_target=") {
            target_seen = true;
            target = event_index(path.trim(), root, indices);
        }
        if field(line, "training_seconds").is_some() {
            finished = true;
        }
        if let Some(path) = saved_path(line) {
            explicit = true;
            if let Some(index) = event_index(path, root, indices) {
                // A truncated recent window may contain the save event but no
                // preceding metrics. Do not erase a fuller durable log record.
                if metrics.observed || !entries[index].explicit {
                    entries[index].metrics = metrics.clone();
                    entries[index].source = Some(format!("{source} (save event)"));
                    entries[index].explicit = true;
                }
            }
        }
        // `last_checkpoint=` on progress refers to the PREVIOUS save; never
        // attach that progress sample to it or interpolate missing history.
    }
    if !finished || explicit {
        return;
    }
    // A finished, checkpoint-specific legacy log is an honest fallback, not
    // an excuse to copy a generic train.log's metrics to every checkpoint.
    let fallback = target.or_else(|| {
        if target_seen {
            return None;
        }
        let log = log?;
        let candidates: Vec<_> = entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.path.with_extension("log") == log
                    || (log.file_name().is_some_and(|n| n == "train.log")
                        && entry.path.file_name().is_some_and(|n| n == "model.pssa"))
            })
            .map(|(i, _)| i)
            .collect();
        (candidates.len() == 1).then(|| candidates[0])
    });
    if let Some(index) = fallback
        && !entries[index].explicit
    {
        entries[index].metrics = metrics;
        entries[index].source = Some(format!("{source} (finished log)"));
    }
}

fn read_log(path: &Path, remaining: &mut usize) -> Result<(Vec<String>, bool), String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    let limit = MAX_LOG_BYTES.min(*remaining);
    let mut truncated = len > limit as u64;
    let mut data = Vec::new();
    file.take(limit as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    *remaining = remaining.saturating_sub(data.len());
    // Do not interpret half of a line/path at the byte limit as a real record.
    if truncated {
        data.truncate(data.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1));
    }
    let mut lines = Vec::new();
    for (index, line) in data.split(|b| *b == b'\n').enumerate() {
        if index >= MAX_LOG_LINES {
            truncated = true;
            break;
        }
        if line.len() <= MAX_LINE {
            lines.push(String::from_utf8_lossy(line).into_owned());
        } else {
            truncated = true;
        }
    }
    Ok((lines, truncated))
}

fn scan(root: &Path, recent: Vec<String>) -> Scan {
    let started = Instant::now();
    let root = absolute(root);
    let mut entries = Vec::new();
    let mut logs = Vec::new();
    let mut limited = false;
    let mut unreadable = 0;
    let Ok(directory) = fs::read_dir(&root) else {
        return Scan {
            entries,
            note: "Cannot read the current run directory. r retries; no weights are loaded.".into(),
        };
    };
    for (visited, item) in directory.enumerate() {
        if visited >= MAX_DIR_ENTRIES || started.elapsed() > SCAN_TIME {
            limited = true;
            break;
        }
        let Ok(item) = item else {
            unreadable += 1;
            continue;
        };
        if !item.file_type().is_ok_and(|kind| kind.is_file()) {
            continue;
        }
        let path = item.path();
        match path.extension().and_then(|s| s.to_str()) {
            Some("pssa" | "trfm") => {
                if entries.len() >= MAX_ENTRIES {
                    limited = true;
                    continue;
                }
                entries.push(Entry {
                    path,
                    metrics: Metrics::default(),
                    source: None,
                    explicit: false,
                });
            }
            Some("log") => {
                if logs.len() >= MAX_LOGS {
                    limited = true;
                    continue;
                }
                let modified = item.metadata().ok().and_then(|m| m.modified().ok());
                logs.push((modified, path));
            }
            _ => {}
        }
    }
    entries.sort_by_key(|e| {
        checkpoint_sort_key(&e.path.file_name().unwrap_or_default().to_string_lossy())
    });
    let indices: HashMap<_, _> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.path.clone(), i))
        .collect();
    logs.sort();
    let mut remaining = MAX_SCAN_BYTES;
    for (_, log) in logs {
        if remaining == 0 || started.elapsed() > SCAN_TIME {
            limited = true;
            break;
        }
        match read_log(&log, &mut remaining) {
            Ok((lines, truncated)) => {
                limited |= truncated;
                associate(
                    lines,
                    &clean(&log.display().to_string()),
                    Some(&log),
                    &root,
                    &mut entries,
                    &indices,
                );
            }
            Err(_) => unreadable += 1,
        }
    }
    // Live piped sessions may have no durable local log. Only explicit save
    // events in this bounded recent window qualify, never the latest live loss.
    if !recent.is_empty() {
        associate(
            recent,
            "recent monitor log",
            None,
            &root,
            &mut entries,
            &indices,
        );
    }
    let note = format!(
        "{} checkpoints. Training metrics only; — = not recorded.{}{}",
        entries.len(),
        if limited {
            " Scan capped (512 checkpoints / 128 logs / 8 MiB / 2s); history may be incomplete."
        } else {
            ""
        },
        if unreadable > 0 {
            " Some directory/log entries could not be read."
        } else {
            ""
        }
    );
    Scan { entries, note }
}

pub(super) enum Action {
    Chat(PathBuf),
    Resume(PathBuf),
}

pub(super) struct Timeline {
    root: Option<PathBuf>,
    entries: Vec<Entry>,
    selected: usize,
    worker: Option<(PathBuf, Receiver<Scan>)>,
    last_scan: Option<Instant>,
    refresh: bool,
    note: String,
    waiting_context: String,
    live: Option<(Option<u64>, Option<f64>, Option<f64>)>,
}
impl Default for Timeline {
    fn default() -> Self {
        Self {
            root: None,
            entries: Vec::new(),
            selected: 0,
            worker: None,
            last_scan: None,
            refresh: true,
            note: "Open a run to inspect its saved checkpoints. No weights are loaded.".into(),
            waiting_context: "No run connected; no checkpoint history to inspect.".into(),
            live: None,
        }
    }
}
impl Timeline {
    pub(super) fn poll(&mut self, state: &RunState, visible: bool) {
        self.waiting_context = state.checkpoint_context();
        self.live = state.training_active.then_some((state.current_step(), state.live_loss, state.tok_s));
        let root = if state.chain_dir.as_os_str().is_empty() {
            state
                .last_checkpoint
                .as_ref()
                .or(state.checkpoint_target.as_ref())
                .and_then(|p| Path::new(p).parent())
                .map(absolute)
        } else {
            Some(absolute(&state.chain_dir))
        };
        if self.root != root {
            self.root = root;
            self.entries.clear();
            self.selected = 0;
            self.refresh = true;
            self.note = "Current run changed; waiting for its checkpoint scan.".into();
            // Keep the old receiver until it completes: never multiply workers
            // when a user rapidly browses between runs.
        }
        let result = self
            .worker
            .as_ref()
            .map(|(root, rx)| (root.clone(), rx.try_recv()));
        match result {
            Some((root, Ok(scan))) => {
                self.worker = None;
                if self.root.as_ref() == Some(&root) {
                    let selected = self.entries.get(self.selected).map(|e| e.path.clone());
                    self.entries = scan.entries;
                    self.selected = selected
                        .and_then(|p| self.entries.iter().position(|e| e.path == p))
                        .unwrap_or_else(|| self.selected.min(self.entries.len().saturating_sub(1)));
                    self.note = scan.note;
                }
            }
            Some((_, Err(TryRecvError::Disconnected))) => {
                self.worker = None;
                self.note = "History worker stopped; r retries.".into();
            }
            _ => {}
        }
        if !visible || self.worker.is_some() {
            return;
        }
        let Some(root) = self.root.clone() else {
            return;
        };
        if !self.refresh
            && self
                .last_scan
                .is_some_and(|t| t.elapsed() < Duration::from_secs(5))
        {
            return;
        }
        self.refresh = false;
        self.last_scan = Some(Instant::now());
        let mut bytes = 0;
        let recent = state
            .raw_lines
            .iter()
            .rev()
            .take(400)
            .take_while(|line| {
                bytes += line.len();
                bytes <= 1024 * 1024
            })
            .filter(|line| line.len() <= MAX_LINE)
            .cloned()
            .collect::<Vec<_>>();
        let recent = recent.into_iter().rev().collect();
        let (tx, rx) = mpsc::sync_channel(1);
        match std::thread::Builder::new()
            .name("tui-timeline".into())
            .spawn({
                let root = root.clone();
                move || {
                    let _ = tx.send(scan(&root, recent));
                }
            }) {
            Ok(_) => self.worker = Some((root, rx)),
            Err(error) => {
                self.note = format!("Cannot start history scan: {}", clean(&error.to_string()))
            }
        }
    }

    pub(super) fn recorded_losses(&self) -> impl Iterator<Item = (String, f64)> + '_ {
        self.entries.iter().filter_map(|entry| {
            Some((entry.path.file_name()?.to_string_lossy().into_owned(), entry.metrics.loss?))
        })
    }

    pub(super) fn key(&mut self, key: KeyEvent) -> Option<Action> {
        match key.code {
            KeyCode::Left | KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Right | KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.entries.len().saturating_sub(1))
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.entries.len().saturating_sub(1),
            KeyCode::Char('r') => self.refresh = true,
            KeyCode::Char('c') | KeyCode::Enter => {
                let entry = self.entries.get(self.selected)?;
                return Some(if key.code == KeyCode::Char('c') {
                    Action::Chat(entry.path.clone())
                } else {
                    Action::Resume(entry.path.clone())
                });
            }
            _ => {}
        }
        None
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        if area.is_empty() {
            return;
        }
        let compact = area.width < 80;
        let show_plot = area.height >= 14 && self.entries.iter().any(|e| e.metrics.loss.is_some());
        let parts = Layout::vertical([
            Constraint::Length(if show_plot {
                // Thirteen panel rows retain eight plot rows after all chrome;
                // shorter screens still leave space for selection and controls.
                if area.height >= 26 { 13 } else { 8 }
            } else if area.height < 12 {
                3
            } else {
                5
            }),
            Constraint::Min(0),
            Constraint::Length(if area.height < 20 { 2 } else { 4 }),
        ])
        .split(area);
        if show_plot {
            let points: Vec<_> = self
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| ((i + 1) as f64, e.metrics.loss.unwrap_or(f64::NAN)))
                .collect();
            let selected: Vec<_> = points.get(self.selected).copied().into_iter().collect();
            super::charts::draw(
                f,
                parts[0],
                super::charts::Plot {
                    title: &format!(
                        " timeline / saved checkpoints / selected #{} ",
                        self.selected + 1
                    ),
                    caption: "Lower loss = better fit; gaps = unrecorded, blue = selected",
                    x: "checkpoint",
                    integer_x: true,
                    y: "loss",
                    x_bounds: super::charts::domain(&points),
                    y_bounds: super::charts::bounds(points.iter().map(|p| p.1)),
                },
                &[
                    super::charts::Series::line("recorded loss", &points, super::NORMAL_GREEN),
                    super::charts::Series {
                        name: "selected",
                        points: &selected,
                        color: super::SECOND_ACCENT,
                        scatter: true,
                    },
                ],
            );
        } else {
            let timeline_area = panel_area(f, parts[0]);
            let block = panel(" timeline / saved checkpoints ");
            let inner = block.inner(timeline_area);
            f.render_widget(block, timeline_area);
            if self.entries.is_empty() {
                f.render_widget(
                    Paragraph::new(self.waiting_context.as_str()).wrap(Wrap { trim: false }),
                    inner,
                );
            } else {
                // A fixed-width horizontal window avoids truncating the selected
                // checkpoint off the right edge on narrow terminals.
                let slots = usize::from(inner.width / 9).max(1);
                let start = self
                    .selected
                    .saturating_sub(slots / 2)
                    .min(self.entries.len().saturating_sub(slots));
                let mut marks = Vec::new();
                for index in start..(start + slots).min(self.entries.len()) {
                    marks.push(Span::styled(
                        format!(
                            " {}#{:03}{} ",
                            if index == self.selected { '[' } else { '─' },
                            index + 1,
                            if index == self.selected { ']' } else { '─' }
                        ),
                        if index == self.selected {
                            accent().add_modifier(Modifier::REVERSED)
                        } else {
                            accent()
                        },
                    ));
                }
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from(marks),
                        Line::from(format!(
                            "checkpoint {} / {}   ← older / newer →",
                            self.selected + 1,
                            self.entries.len()
                        )),
                    ]),
                    inner,
                );
            }
        }
        let mut rows = Vec::new();
        if let Some(entry) = self.entries.get(self.selected) {
            rows.push(Line::styled(
                clean(&entry.path.display().to_string()),
                accent(),
            ));
            rows.push(Line::from(format!(
                "Recorded training loss: {}",
                entry
                    .metrics
                    .loss
                    .map_or("— (not recorded)".into(), super::charts::number)
            )));
            rows.push(Line::from(format!(
                "Recorded throughput: {}",
                entry
                    .metrics
                    .speed
                    .map_or("— (not recorded)".into(), |v| format!("{v:.0} tok/s"))
            )));
            rows.push(Line::from(format!(
                "Recorded update: {}",
                entry.metrics.update.map_or("—".into(), |v| v.to_string())
            )));
            rows.push(Line::from(format!(
                "Source: {}",
                entry
                    .source
                    .as_deref()
                    .unwrap_or("no matching saved history")
            )));
            if !compact {
                rows.push(Line::from(
                    "Lines join recorded samples only; not held-out scores or recovered weights.",
                ));
            }
        } else {
            rows.push(Line::from(self.waiting_context.clone()));
            if let Some((step, loss, speed)) = self.live {
                rows.push(Line::from(format!("Unsaved live run: step {} / loss {} / {} tok/s",
                    step.map_or("unrecorded".into(), |value| value.to_string()),
                    loss.map_or("unrecorded".into(), super::charts::number),
                    speed.map_or("unrecorded".into(), |value| format!("{value:.0}")))));
                rows.push(Line::from("Live metrics are not checkpoint history until a save event arrives."));
            }
            rows.push(Line::from("Keep train.log beside checkpoints; missing history cannot be recovered from weights."));
        }
        rows.push(Line::styled(
            "Enter prepares setup ONLY; review configuration before START.",
            Style::new().fg(AMBER),
        ));
        let selected_area = panel_area(f, parts[1]);
        f.render_widget(
            Paragraph::new(rows)
                .wrap(Wrap { trim: false })
                .block(panel(" selected checkpoint / read-only ")),
            selected_area,
        );
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    "←/→ scrub / Home End / c chat / Enter resume setup / r refresh",
                    accent(),
                ),
                Line::from(if self.worker.is_some() {
                    "Scanning filenames and bounded logs in background…".to_owned()
                } else {
                    clean(&self.note)
                }),
            ])
            .wrap(Wrap { trim: false }),
            parts[2],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use ratatui::{Terminal, backend::TestBackend};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "pssa-timeline-{}-{} path's spaces",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn checkpoint(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, "not a checkpoint; must never be decoded").unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn log_events_preserve_spaces_and_do_not_attach_future_samples_to_last_checkpoint() {
        let fixture = Fixture::new();
        let first = fixture.checkpoint("ck01.pssa");
        let second = fixture.checkpoint("checkpoint two.pssa");
        let missing = fixture.checkpoint("unrecorded.pssa");
        fs::write(fixture.0.join("chain.log"), format!(
            "progress_schema=2 checkpoint_target={}\nloss=4 tokens_per_second=100 global_update=10\nepoch 1/1 loss=3.5 tokens=64 updates=10\nsaved_checkpoint={}\nprogress_schema=2 checkpoint_target={}\nloss=2 tokens_per_second=200 global_update=20 last_checkpoint={}\nepoch 1/1 loss=1.5 tokens=64 updates=20\nthroughput      180 tokens/second\ntraining_seconds=2\nsaved_checkpoint={}\n", first.display(), first.display(), second.display(), first.display(), second.display())).unwrap();
        let result = scan(&fixture.0, vec![]);
        let get = |p: &Path| result.entries.iter().find(|e| e.path == p).unwrap();
        assert_eq!(get(&first).metrics.loss, Some(3.5));
        assert_eq!(get(&first).metrics.speed, Some(100.0));
        assert_eq!(get(&second).metrics.loss, Some(1.5));
        assert_eq!(get(&second).metrics.speed, Some(180.0));
        assert_eq!(get(&second).metrics.update, Some(20));
        assert!(get(&missing).metrics.loss.is_none());
        assert_eq!(
            fs::read_to_string(first).unwrap(),
            "not a checkpoint; must never be decoded"
        );
    }

    #[test]
    fn sibling_logs_and_train_log_are_not_smeared_across_other_checkpoints() {
        let fixture = Fixture::new();
        let model = fixture.checkpoint("model.pssa");
        let named = fixture.checkpoint("named checkpoint.pssa");
        let other = fixture.checkpoint("ck09.pssa");
        fs::write(
            fixture.0.join("train.log"),
            "loss=3 tokens_per_second=30\ntraining_seconds=1\n",
        )
        .unwrap();
        fs::write(
            named.with_extension("log"),
            "loss=4 tokens_per_second=40\ntraining_seconds=1\n",
        )
        .unwrap();
        fs::write(fixture.0.join("unrelated.log"), "loss=99 tokens_per_second=99\nsaved_checkpoint=/another/run/ck09.pssa\ntraining_seconds=1\n").unwrap();
        let result = scan(&fixture.0, vec![]);
        let get = |p: &Path| result.entries.iter().find(|e| e.path == p).unwrap();
        assert_eq!(get(&model).metrics.loss, Some(3.0));
        assert_eq!(get(&named).metrics.loss, Some(4.0));
        assert!(get(&other).metrics.loss.is_none());
        assert_eq!(
            result.entries[0].path, other,
            "numeric chain order precedes arbitrary names"
        );
    }

    #[test]
    fn explicit_other_run_target_blocks_filename_fallback_and_recent_window_keeps_history() {
        let fixture = Fixture::new();
        let model = fixture.checkpoint("model.pssa");
        fs::write(fixture.0.join("train.log"), "progress_schema=2 checkpoint_target=/another/run/model.pssa\nloss=99 tokens_per_second=99\ntraining_seconds=1\n").unwrap();
        assert!(scan(&fixture.0, vec![]).entries[0].metrics.loss.is_none());
        fs::write(
            fixture.0.join("train.log"),
            format!(
                "loss=3 tokens_per_second=30\ntraining_seconds=1\nsaved_checkpoint={}\n",
                model.display()
            ),
        )
        .unwrap();
        let result = scan(
            &fixture.0,
            vec![format!("saved_checkpoint={}", model.display())],
        );
        assert_eq!(result.entries[0].metrics.loss, Some(3.0));
        assert_eq!(result.entries[0].metrics.speed, Some(30.0));
    }

    #[test]
    fn live_save_events_have_exact_association_and_invalid_values_stay_missing() {
        let fixture = Fixture::new();
        let path = fixture.checkpoint("saved with spaces.pssa");
        let result = scan(
            &fixture.0,
            vec![
                "loss=NaN tokens_per_second=inf".into(),
                format!("saved_checkpoint={}", path.display()),
                "loss=1 tokens_per_second=100".into(),
                format!("last_checkpoint={}", path.display()),
            ],
        );
        assert!(result.entries[0].metrics.loss.is_none());
        assert!(result.entries[0].metrics.speed.is_none());
        assert!(
            result.entries[0]
                .source
                .as_ref()
                .unwrap()
                .contains("recent monitor")
        );
    }

    #[test]
    fn bounded_reader_drops_partial_paths_and_oversized_lines() {
        let fixture = Fixture::new();
        let log = fixture.0.join("bounded.log");
        fs::write(
            &log,
            "loss=3 tokens_per_second=30\nsaved_checkpoint=a path with spaces.pssa\n",
        )
        .unwrap();
        let mut bytes = 40;
        let (lines, capped) = read_log(&log, &mut bytes).unwrap();
        assert!(capped);
        assert_eq!(bytes, 0);
        assert_eq!(lines, vec!["loss=3 tokens_per_second=30", ""]);
        fs::write(
            &log,
            format!(
                "{}\nloss=2 tokens_per_second=20\n",
                "x".repeat(MAX_LINE + 1)
            ),
        )
        .unwrap();
        let mut bytes = MAX_SCAN_BYTES;
        let (lines, _) = read_log(&log, &mut bytes).unwrap();
        assert!(!lines.iter().any(|l| l.starts_with('x')));
    }

    #[test]
    fn navigation_actions_only_return_checkpoint_paths_and_resume_does_not_launch() {
        let fixture = Fixture::new();
        let first = fixture.checkpoint("ck01.pssa");
        let last = fixture.checkpoint("ck10.pssa");
        let mut timeline = Timeline {
            entries: scan(&fixture.0, vec![]).entries,
            ..Default::default()
        };
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        timeline.key(key(KeyCode::End));
        assert!(matches!(timeline.key(key(KeyCode::Enter)), Some(Action::Resume(p)) if p == last));
        assert!(!fixture.0.join("train.log").exists());
        timeline.key(key(KeyCode::Home));
        assert!(
            matches!(timeline.key(key(KeyCode::Char('c'))), Some(Action::Chat(p)) if p == first)
        );
        timeline.key(key(KeyCode::Left));
        assert_eq!(timeline.selected, 0);
        timeline.key(key(KeyCode::Right));
        assert_eq!(timeline.selected, 1);
    }

    #[test]
    fn background_scan_keeps_one_worker_and_discards_old_run_results() {
        let a = Fixture::new();
        let b = Fixture::new();
        a.checkpoint("ck01.pssa");
        let wanted = b.checkpoint("ck02.pssa");
        let mut state = RunState {
            chain_dir: a.0.clone(),
            ..Default::default()
        };
        let mut timeline = Timeline::default();
        timeline.poll(&state, false);
        assert!(
            timeline.worker.is_none(),
            "hidden screen does not launch scans"
        );
        timeline.poll(&state, true);
        state.chain_dir = b.0.clone();
        timeline.poll(&state, true);
        let deadline = Instant::now() + Duration::from_secs(5);
        while (timeline.entries.is_empty() || timeline.worker.is_some())
            && Instant::now() < deadline
        {
            timeline.poll(&state, true);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(timeline.entries.len(), 1);
        assert_eq!(timeline.entries[0].path, wanted);
    }

    #[test]
    fn unsaved_timeline_shows_current_run_without_inventing_checkpoint_history() {
        let mut state = RunState::default();
        state.ingest("progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/tmp/live run/model.pssa");
        state.ingest("training 120/500 (24%) loss=3 tokens_per_second=90 optimizer_updates=120 global_update=220");
        let mut timeline = Timeline::default();
        timeline.poll(&state, false);
        assert!(timeline.entries.is_empty());
        assert!(timeline.worker.is_none());
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        terminal.draw(|f| timeline.draw(f, f.area())).unwrap();
        let text: String = terminal.backend().buffer().content().iter().map(|cell| cell.symbol()).collect();
        for expected in ["step 600", "current step 220", "Unsaved live run: step 220", "loss 3", "90 tok/s"] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
    }

    #[test]
    fn timeline_loss_has_braille_axes_at_both_sizes() {
        let timeline = Timeline {
            entries: [4.0, 3.0, 2.0]
                .into_iter()
                .enumerate()
                .map(|(i, loss)| Entry {
                    path: PathBuf::from(format!("ck{}.pssa", i + 1)),
                    metrics: Metrics {
                        loss: Some(loss),
                        ..Default::default()
                    },
                    source: None,
                    explicit: true,
                })
                .collect(),
            selected: 1,
            ..Default::default()
        };
        for (w, h) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| timeline.draw(f, f.area())).unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "timeline / saved checkpoints",
                &[
                    "checkpoint",
                    "loss",
                    "Lower loss = better fit",
                    "1",
                    "2",
                    "3",
                ],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "timeline / saved checkpoints",
                if h == 40 { 8 } else { 3 },
            );
            assert!(terminal.backend().buffer().content().iter().any(|c| {
                c.fg == super::super::SECOND_ACCENT
                    && c.symbol()
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
            }));
            terminal
                .draw(|f| timeline.draw(f, super::super::feature_area(f.area())))
                .unwrap();
            super::super::charts::assert_named_plot(
                terminal.backend().buffer(),
                "timeline / saved checkpoints",
                &["checkpoint", "loss", "Lower loss = better fit"],
            );
            super::super::assert_chart_rows(
                terminal.backend().buffer(),
                "timeline / saved checkpoints",
                if h == 40 { 8 } else { 3 },
            );
        }
    }

    #[test]
    fn test_backend_timeline_wide_narrow_and_tiny() {
        let fixture = Fixture::new();
        fixture.checkpoint("checkpoint with spaces.pssa");
        let timeline = Timeline {
            entries: scan(&fixture.0, vec![]).entries,
            ..Default::default()
        };
        for (width, height) in [(120, 30), (79, 24), (45, 18), (10, 4), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| timeline.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 45 {
                assert!(text.contains("timeline / saved checkpoints"));
                assert!(text.contains("Recorded training loss"));
                assert!(text.contains("not recorded"));
            }
        }
    }
}
