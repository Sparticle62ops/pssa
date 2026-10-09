//! Bounded run discovery, lazy checkpoint metadata and isolated read-only scoring.
use super::{
    RunState, accent, panel, panel_area,
    process::{Job, clean},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::mpsc,
    time::UNIX_EPOCH,
};

struct Entry {
    path: PathBuf,
    modified: Option<u64>,
    ppl: Option<f64>,
}
fn text(path: &Path, limit: u64) -> Result<String, String> {
    let mut s = String::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(limit + 1)
        .read_to_string(&mut s)
        .map_err(|e| e.to_string())?;
    if s.len() as u64 > limit {
        return Err(format!(
            "File exceeds {} MiB display limit",
            limit / 1024 / 1024
        ));
    }
    Ok(s)
}
fn score_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".score.json");
    PathBuf::from(name)
}
fn save_score(path: &Path, value: &serde_json::Value) -> Result<PathBuf, String> {
    use std::io::Write;
    let target = score_path(path);
    let temporary = target.with_extension(format!("json.tmp-{}", std::process::id()));
    let data = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = file.write_all(&data).and_then(|_| {
        drop(file);
        fs::rename(&temporary, &target)
    });
    if let Err(e) = result {
        let _ = fs::remove_file(&temporary);
        return Err(format!("Score complete but cannot save metrics: {e}"));
    }
    Ok(target)
}
fn scan(roots: &[PathBuf]) -> Vec<Entry> {
    let mut found = Vec::new();
    let mut dirs: Vec<_> = roots.iter().cloned().map(|p| (p, 0)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut visited = 0;
    while let Some((dir, depth)) = dirs.pop() {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            visited += 1;
            if visited > 10_000 || found.len() >= 1000 {
                break;
            }
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() && depth < 2 {
                dirs.push((path, depth + 1));
                continue;
            }
            if !kind.is_file()
                || !path
                    .extension()
                    .is_some_and(|e| e == "pssa" || e == "trfm" || e == "log")
            {
                continue;
            }
            let canonical = fs::canonicalize(&path).unwrap_or(path.clone());
            if !seen.insert(canonical) {
                continue;
            }
            let modified = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            let sidecar = score_path(&path);
            // Preserve older saved scores until a format-specific score exists.
            let sidecar = if sidecar.exists() {
                sidecar
            } else {
                path.with_extension("score.json")
            };
            let ppl = text(&sidecar, 64 * 1024)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v["perplexity"].as_f64())
                .filter(|n| n.is_finite());
            found.push(Entry {
                path,
                modified,
                ppl,
            });
        }
        if visited > 10_000 || found.len() >= 1000 {
            break;
        }
    }
    found.sort_by(|a, b| b.modified.cmp(&a.modified).then(a.path.cmp(&b.path)));
    found
}
fn date(seconds: u64) -> String {
    // Gregorian civil date from days since Unix epoch, UTC (no timezone dependency).
    let z = (seconds / 86400) as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    format!("{:04}-{month:02}-{day:02} UTC", y + i64::from(month <= 2))
}
fn details(path: &Path) -> Result<String, String> {
    if path.extension().is_some_and(|e| e == "log") {
        return Ok("Training log • Enter opens recorded monitor history".into());
    }
    let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > 64 * 1024 * 1024 {
        return Err("Automatic metadata loading is capped at 64 MiB; chat/score remain available on explicit request.".into());
    }
    if path.extension().is_some_and(|e| e == "trfm") {
        let m = crate::transformer_checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
        Ok(format!(
            "Transformer • vocab {} • width {} • FF {} • chunk {} • lr {} • updates {}",
            m.cfg.d_vocab, m.cfg.d_model, m.cfg.d_ff, m.cfg.chunk_len, m.cfg.lr, m.step_counter
        ))
    } else {
        let m = crate::checkpoint::load_checkpoint(path)
            .map_err(|e| e.to_string())?
            .model;
        Ok(format!(
            "PSSA • vocab {} • latent {} • state {} • depth {} • chunk {} • lr {} • updates {}",
            m.cfg.d_vocab,
            m.cfg.d_latent,
            m.cfg.d_state,
            m.cfg.depth,
            m.cfg.chunk_len,
            m.cfg.lr,
            m.step_counter
        ))
    }
}
pub(super) enum Action {
    Monitor(RunState),
    Chat(PathBuf),
}
pub(super) struct Runs {
    roots: Vec<PathBuf>,
    entries: Vec<Entry>,
    selected: usize,
    scan: Option<mpsc::Receiver<Vec<Entry>>>,
    detail: Option<mpsc::Receiver<(PathBuf, Result<String, String>)>>,
    loaded: Option<PathBuf>,
    description: String,
    open: Option<mpsc::Receiver<Result<RunState, String>>>,
    score_input: Option<String>,
    job: Option<(PathBuf, Job)>,
    score_json: Option<serde_json::Value>,
    note: String,
    pub action: Option<Action>,
    pub notification: Option<(bool, String)>,
}
impl Runs {
    pub fn new(chain: PathBuf) -> Self {
        let mut r = Self {
            roots: vec![chain, "data".into(), "runs".into(), "comparison".into()],
            entries: Vec::new(),
            selected: 0,
            scan: None,
            detail: None,
            loaded: None,
            description: String::new(),
            open: None,
            score_input: None,
            job: None,
            score_json: None,
            note: "r refresh • ↑/↓ select • Enter monitor • c chat • s score held-out file".into(),
            action: None,
            notification: None,
        };
        r.refresh();
        r
    }
    pub fn editing(&self) -> bool {
        self.score_input.is_some()
    }
    pub fn add_root(&mut self, root: PathBuf) {
        if !self.roots.contains(&root) {
            self.roots.push(root);
        }
        // Discard a scan of the previous roots before requesting updated results.
        self.scan = None;
        self.refresh();
    }
    fn refresh(&mut self) {
        if self.scan.is_some() {
            return;
        }
        let roots = self.roots.clone();
        let (tx, rx) = mpsc::channel();
        self.scan = Some(rx);
        std::thread::spawn(move || {
            let _ = tx.send(scan(&roots));
        });
    }
    fn selected_path(&self) -> Option<PathBuf> {
        self.entries.get(self.selected).map(|e| e.path.clone())
    }
    pub fn poll(&mut self, visible: bool) {
        if let Some(entries) = self.scan.as_ref().and_then(|r| r.try_recv().ok()) {
            self.scan = None;
            self.entries = entries;
            self.selected = self.selected.min(self.entries.len().saturating_sub(1));
            self.loaded = None;
        }
        if let Some((path, result)) = self.detail.as_ref().and_then(|r| r.try_recv().ok()) {
            self.detail = None;
            if self.selected_path().as_ref() == Some(&path) {
                self.description = result.unwrap_or_else(|e| format!("Unreadable checkpoint: {e}"));
            }
        }
        if visible && self.detail.is_none() && self.selected_path() != self.loaded {
            self.loaded = self.selected_path();
            self.description = "Reading selected checkpoint in background…".into();
            if let Some(path) = self.loaded.clone() {
                let (tx, rx) = mpsc::channel();
                self.detail = Some(rx);
                std::thread::spawn(move || {
                    let result = details(&path);
                    let _ = tx.send((path, result));
                });
            }
        }
        if let Some(result) = self.open.as_ref().and_then(|r| r.try_recv().ok()) {
            self.open = None;
            match result {
                Ok(s) => self.action = Some(Action::Monitor(s)),
                Err(e) => self.note = e,
            }
        }
        if let Some((path, job)) = &mut self.job {
            let done = match job.poll() {
                Ok((lines, done)) => {
                    for line in lines {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
                            && v.get("perplexity").is_some()
                        {
                            self.score_json = Some(v);
                        }
                        self.note = clean(&line);
                    }
                    done
                }
                Err(e) => {
                    self.note = e.to_string();
                    Some(false)
                }
            };
            if let Some(ok) = done {
                let result = if ok {
                    self.score_json
                        .take()
                        .ok_or_else(|| "Score produced no metrics".to_string())
                        .and_then(|v| {
                            // Atomic rename replaces, never follows, a sidecar symlink.
                            let target = save_score(path, &v)?;
                            Ok(format!(
                                "Held-out perplexity {} • {}",
                                v["perplexity"],
                                target.display()
                            ))
                        })
                } else {
                    Err(format!("Score failed: {}", self.note))
                };
                let success = result.is_ok();
                self.note = result.unwrap_or_else(|e| e);
                self.notification = Some((success, self.note.clone()));
                self.job = None;
                self.refresh();
            }
        }
    }
    fn open_monitor(&mut self, path: PathBuf) {
        if self.open.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.open = Some(rx);
        std::thread::spawn(move || {
            let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let mut state = RunState {
                chain_dir: root.clone(),
                ..Default::default()
            };
            let mut log = path.with_extension("log");
            if !log.exists()
                && path
                    .file_name()
                    .is_some_and(|name| name == "model.pssa" || name == "model.trfm")
            {
                log = path.with_file_name("train.log");
            }
            let result = if log.exists() {
                text(&log, 8 * 1024 * 1024).map(|s| {
                    for line in s.lines() {
                        state.ingest(line);
                    }
                })
            } else {
                Ok(())
            };
            // Replayed trainer targets may describe an obsolete remote folder.
            // Browsing a copied run must stay rooted in the selected local run.
            state.chain_dir = root;
            let local_artifact = |recorded: Option<&str>| {
                let name = Path::new(recorded?).file_name()?;
                let candidate = state.chain_dir.join(name);
                (candidate
                    .extension()
                    .is_some_and(|ext| ext == "pssa" || ext == "trfm")
                    && candidate.is_file())
                .then_some(candidate)
            };
            let last = local_artifact(state.last_checkpoint.as_deref());
            let target = local_artifact(state.checkpoint_target.as_deref());
            state.last_checkpoint = last.map(|p| p.display().to_string());
            state.checkpoint_target = target.map(|p| p.display().to_string());
            if path.extension().is_some_and(|e| e != "log") {
                // File existence is not a trainer save event. Keep the selected
                // read-only inference artifact separate from save provenance.
                state.selected_checkpoint = Some(path.display().to_string());
                if let Some(dims) = super::library::checkpoint_dims(&path) {
                    state.use_header_dims(&path, &dims);
                }
            }
            state.checkpoint_number = state
                .last_checkpoint
                .as_deref()
                .or(state.checkpoint_target.as_deref())
                .and_then(super::checkpoint_number);
            state.refresh_chain();
            state.training_active = false;
            state.last_progress_at = None;
            if state.metric_series.is_empty() {
                state.warning = Some(
                    "No recorded progress log; checkpoint only (history is not recoverable)."
                        .into(),
                );
            }
            let _ = tx.send(result.map(|_| state));
        });
    }
    pub fn key(&mut self, key: KeyEvent) {
        if let Some(input) = &mut self.score_input {
            match key.code {
                KeyCode::Esc => self.score_input = None,
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    input.clear()
                }
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key.modifiers.contains(KeyModifiers::CONTROL)
                        && input.len() < 4096 =>
                {
                    input.push(c)
                }
                KeyCode::Enter => {
                    let data = input.clone();
                    if !Path::new(&data).is_file() {
                        self.note = "Choose an existing held-out local file (no download).".into();
                        return;
                    }
                    let Some(path) = self.selected_path() else {
                        return;
                    };
                    let command = if path.extension().is_some_and(|e| e == "trfm") {
                        "score-transformer"
                    } else {
                        "score"
                    };
                    match Job::start(&[
                        command.into(),
                        data,
                        "--model".into(),
                        path.display().to_string(),
                    ]) {
                        Ok(job) => {
                            self.job = Some((path, job));
                            self.score_json = None;
                            self.score_input = None;
                            self.note =
                                "Scoring in child process; Esc cancels. Checkpoint is read-only."
                                    .into();
                        }
                        Err(e) => self.note = e,
                    }
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.entries.len().saturating_sub(1))
            }
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Enter => {
                if let Some(path) = self.selected_path() {
                    self.open_monitor(path);
                }
            }
            KeyCode::Char('c') | KeyCode::Char('s') => {
                if let Some(path) = self.selected_path() {
                    if path.extension().is_some_and(|e| e == "log") {
                        self.note = "Select a checkpoint to chat or score, not a log.".into();
                    } else if key.code == KeyCode::Char('c') {
                        self.action = Some(Action::Chat(path));
                    } else if self.job.is_none() {
                        self.score_input = Some(String::new());
                        self.note = "Use genuinely held-out text: checkpoint files do not record corpus provenance.".into();
                    }
                }
            }
            KeyCode::Esc if self.job.is_some() => {
                self.job = None;
                self.note = "Score cancelled.".into();
            }
            _ => {}
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Min(0), Constraint::Length(8)]).split(area);
        let mut rows = Vec::new();
        if self.entries.is_empty() {
            rows.push(Line::from(
                "No runs yet. Scans --chain, data/, runs/, comparison/ (depth 2). r refresh",
            ));
        }
        let height = chunks[0].height.saturating_sub(2) as usize;
        let start = self.selected.saturating_sub(height.saturating_sub(1));
        for (i, e) in self.entries.iter().enumerate().skip(start).take(height) {
            rows.push(Line::from(format!(
                "{} {}  ppl {}  {}",
                if i == self.selected { "▶" } else { " " },
                e.modified
                    .map(date)
                    .unwrap_or_else(|| "mtime unavailable".into()),
                e.ppl
                    .map_or("unscored".into(), |n| format!("{n:.3} (saved/unverified)")),
                clean(&e.path.display().to_string())
            )));
        }
        let runs_area = panel_area(f, chunks[0]);
        f.render_widget(
            Paragraph::new(rows).block(panel(" runs / checkpoints + logs / historical scores ")),
            runs_area,
        );
        let mut detail = vec![
            Line::styled(self.description.as_str(), accent()),
            Line::from(self.note.as_str()),
            Line::from("Saved perplexity is historical sidecar data; checkpoint identity is unverified and the score may be stale."),
            Line::from("↑/↓ select • r refresh • Enter monitor • c chat • s score • Esc cancel"),
            Line::from(
                "Perplexity comes from saved held-out .score.json, never inferred from weights.",
            ),
        ];
        if let Some(input) = &self.score_input {
            detail.insert(
                0,
                Line::from(format!(
                    "Held-out file: {input}▌ (Enter score / Ctrl+U clear / Esc cancel)"
                )),
            );
        }
        let detail_area = panel_area(f, chunks[1]);
        f.render_widget(
            Paragraph::new(detail)
                .wrap(Wrap { trim: false })
                .block(panel(" selected run / read-only ")),
            detail_area,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovery_deduplicates_and_reopens_score_sidecar() {
        let dir = std::env::temp_dir().join(format!("pssa-runs-ui-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("path with spaces.pssa");
        fs::write(&p, b"fixture").unwrap();
        fs::write(p.with_extension("score.json"), r#"{"perplexity":12.5}"#).unwrap();
        let entries = scan(&[dir.clone(), dir.clone()]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].ppl, Some(12.5));
        assert!(details(&p).is_err());
        #[cfg(unix)]
        {
            fs::remove_file(p.with_extension("score.json")).unwrap();
            std::os::unix::fs::symlink(&p, score_path(&p)).unwrap();
            save_score(&p, &serde_json::json!({"perplexity":8.5})).unwrap();
            assert_eq!(fs::read(&p).unwrap(), b"fixture");
            assert_eq!(scan(&[dir.clone()])[0].ppl, Some(8.5));
        }
        assert_eq!(date(0), "1970-01-01 UTC");
        assert_eq!(date(1709164800), "2024-02-29 UTC");
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn replaced_checkpoint_never_presents_a_saved_sidecar_as_a_verified_current_score() {
        use ratatui::{Terminal, backend::TestBackend};
        let temp = super::super::library::tests::Temp::new();
        let path = temp.0.join("model.pssa");
        fs::write(&path, "checkpoint A").unwrap();
        save_score(&path, &serde_json::json!({"perplexity": 8.5})).unwrap();
        fs::write(&path, "checkpoint B").unwrap();
        let mut runs = Runs::new(temp.0.clone());
        runs.entries = scan(&[temp.0.clone()]);
        assert_eq!(
            runs.entries[0].ppl,
            Some(8.5),
            "saved metric remains historical data"
        );
        for (width, height) in [(120, 40), (80, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| runs.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(text.contains("8.500 (saved/unverified)"), "{text}");
            assert!(text.contains("historical scores"), "{text}");
            assert!(text.contains("score may be stale"), "{text}");
        }
    }

    #[test]
    fn open_saved_log_restores_monitor_and_missing_history_is_explicit() {
        let dir = std::env::temp_dir().join(format!("pssa-runs-open-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("saved checkpoint.pssa");
        fs::write(&path, b"not loaded for history").unwrap();
        fs::write(path.with_extension("log"),"progress_schema=1\nloss=2.5 tokens_per_second=50 memory_occupancy=1/4\ntraining_seconds=2\n").unwrap();
        let mut runs = Runs::new(dir.clone());
        runs.open_monitor(path.clone());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while runs.action.is_none() && std::time::Instant::now() < deadline {
            runs.poll(false);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let Some(Action::Monitor(state)) = runs.action.take() else {
            panic!("monitor did not open")
        };
        assert_eq!(state.live_loss, Some(2.5));
        assert_eq!(state.metric_series.len(), 1);
        assert!(!state.training_active);
        assert!(state.last_checkpoint.is_none());
        assert_eq!(state.selected_checkpoint.as_deref(), path.to_str());
        fs::remove_file(path.with_extension("log")).unwrap();
        runs.open_monitor(path);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while runs.action.is_none() && std::time::Instant::now() < deadline {
            runs.poll(false);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let Some(Action::Monitor(state)) = runs.action.take() else {
            panic!("checkpoint did not open")
        };
        assert!(state.metric_series.is_empty());
        assert!(state.warning.unwrap().contains("not recoverable"));
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn copied_remote_log_keeps_selected_local_root_and_artifact_paths() {
        let dir = std::env::temp_dir().join(format!(
            "pssa-runs-copied-{} path with spaces",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pssa");
        let previous = dir.join("ck01.pssa");
        let log = dir.join("train.log");
        fs::write(&path, "local model; not decoded or modified").unwrap();
        fs::write(&previous, "local previous artifact").unwrap();
        fs::write(&log, "progress_schema=2 updates_total=500 prior_updates=100 checkpoint_target=/kaggle/working/obsolete run/model.pssa\nloss=4 tokens_per_second=40 optimizer_updates=0 global_update=100\nsaved_checkpoint=/kaggle/working/obsolete run/ck01.pssa\nloss=2.5 tokens_per_second=50 optimizer_updates=500 global_update=600\ntraining_seconds=2\nsaved_checkpoint=/kaggle/working/obsolete run/model.pssa\n").unwrap();
        for selected in [&path, &log] {
            let mut runs = Runs::new(dir.clone());
            runs.open_monitor(selected.clone());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while runs.action.is_none() && std::time::Instant::now() < deadline {
                runs.poll(false);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let Some(Action::Monitor(state)) = runs.action.take() else {
                panic!("copied run did not open: {}", selected.display())
            };
            assert_eq!(state.chain_dir, dir);
            assert_eq!(state.last_checkpoint.as_deref(), path.to_str());
            assert_eq!(state.checkpoint_target.as_deref(), path.to_str());
            assert_eq!(state.checkpoint_context(), path.display().to_string());
            assert_eq!(state.current_step(), Some(600));
            assert_eq!(
                state.checkpoints,
                vec![
                    ("ck01.pssa".into(), Some(4.0)),
                    ("model.pssa".into(), Some(2.5))
                ]
            );
            assert!(!state.training_active);
            assert_eq!(
                fs::read_to_string(&path).unwrap(),
                "local model; not decoded or modified"
            );
            assert_eq!(
                fs::read_to_string(&previous).unwrap(),
                "local previous artifact"
            );
        }
        // A remote filename without a corresponding local artifact is not a
        // usable local save target, even when a different checkpoint is present.
        fs::write(&log, "progress_schema=2 checkpoint_target=/kaggle/working/missing/model.trfm\nsaved_checkpoint=/kaggle/working/missing/model.trfm\n").unwrap();
        let mut runs = Runs::new(dir.clone());
        runs.open_monitor(log);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while runs.action.is_none() && std::time::Instant::now() < deadline {
            runs.poll(false);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let Some(Action::Monitor(state)) = runs.action.take() else {
            panic!("copied log did not open")
        };
        assert_eq!(state.chain_dir, dir);
        assert!(state.last_checkpoint.is_none());
        assert!(state.checkpoint_target.is_none());
        assert!(
            state
                .checkpoints
                .iter()
                .all(|(name, _)| name != "model.trfm")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn opening_an_existing_artifact_does_not_claim_unsaved_metrics_belong_to_it() {
        let dir = std::env::temp_dir().join(format!("pssa-runs-unsaved-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pssa");
        fs::write(&path, "existing checkpoint must remain untouched").unwrap();
        fs::write(dir.join("train.log"), format!(
            "progress_schema=2 updates_total=10 prior_updates=0 checkpoint_target={}\ntraining 10/10 (100%) loss=2 tokens_per_second=50 global_update=10\ntraining_seconds=1 optimizer_updates=10\nsave failed: disk full\n", path.display()
        )).unwrap();
        let mut runs = Runs::new(dir.clone());
        runs.open_monitor(path.clone());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while runs.action.is_none() && std::time::Instant::now() < deadline {
            runs.poll(false);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let Some(Action::Monitor(state)) = runs.action.take() else {
            panic!("history did not open")
        };
        assert_eq!(
            state.live_loss,
            Some(2.0),
            "the log's training measurement is still real"
        );
        assert_eq!(state.checkpoints, [("model.pssa".into(), None)]);
        assert!(state.last_checkpoint.is_none());
        assert!(!state.checkpoint_saved_current);
        assert_eq!(state.selected_checkpoint.as_deref(), path.to_str());
        assert!(
            state
                .checkpoint_context()
                .contains("save failed: disk full")
        );
        assert!(state.health_status().level == super::super::HealthLevel::Problem);
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "existing checkpoint must remain untouched"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn scoring_different_checkpoint_formats_keeps_separate_sidecars() {
        let dir = std::env::temp_dir().join(format!("pssa-runs-scores-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for (ext, ppl) in [("pssa", 8.5), ("trfm", 9.5)] {
            let path = dir.join(format!("model.{ext}"));
            fs::write(&path, "read-only fixture").unwrap();
            save_score(&path, &serde_json::json!({"perplexity": ppl})).unwrap();
        }
        let entries = scan(&[dir.clone()]);
        assert_eq!(entries.len(), 2);
        for entry in entries {
            assert_eq!(
                entry.ppl,
                Some(if entry.path.extension().unwrap() == "pssa" {
                    8.5
                } else {
                    9.5
                })
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wizard_output_is_discoverable_and_reopens_train_log() {
        let dir = std::env::temp_dir().join(format!("pssa-runs-wizard-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.log"),
            "progress_schema=1\nloss=2.5 tokens_per_second=50\ntraining_seconds=2\n",
        )
        .unwrap();
        for name in ["model.pssa", "model.trfm"] {
            let path = dir.join(name);
            fs::write(&path, "not loaded for history").unwrap();
            let mut runs = Runs::new("missing-chain".into());
            runs.add_root(dir.clone());
            runs.open_monitor(path.clone());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while (runs.action.is_none() || runs.scan.is_some())
                && std::time::Instant::now() < deadline
            {
                runs.poll(false);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(runs.entries.iter().any(|entry| entry.path == path));
            let Some(Action::Monitor(state)) = runs.action.take() else {
                panic!("wizard history did not open: {name}")
            };
            assert_eq!(state.live_loss, Some(2.5));
            assert_eq!(state.tok_s, Some(50.0));
            assert_eq!(state.metric_series.len(), 1);
            assert!(state.warning.is_none());
            assert!(!state.training_active);
            assert!(
                state.last_progress_at.is_none(),
                "recorded history has no stall clock"
            );
            assert_eq!(state.health_status().normal_label, "DONE");
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
            terminal.draw(|f| super::super::draw(f, &state, 0)).unwrap();
            let screen: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(screen.contains("[ DONE ]"));
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn score_editor_remains_visible_above_long_details_on_narrow_screens() {
        let mut runs = Runs::new("missing-chain".into());
        runs.description = "long metadata ".repeat(100);
        runs.score_input = Some("held-out.txt".into());
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(79, 16)).unwrap();
        terminal.draw(|f| runs.draw(f, f.area())).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(screen.contains("Held-out file: held-out.txt"));
    }

    #[test]
    fn runs_empty_and_selected_screens_render() {
        let mut r = Runs::new("missing-test-chain".into());
        r.entries = vec![Entry {
            path: "models/a checkpoint.pssa".into(),
            modified: None,
            ppl: Some(9.5),
        }];
        r.description = "PSSA • latent 256 • state 16 • depth 1".into();
        for (w, h) in [(120, 30), (79, 24), (24, 8), (1, 1)] {
            let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            t.draw(|f| r.draw(f, f.area())).unwrap();
            if w >= 79 {
                let s: String = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(s.contains("ppl 9.500"));
                assert!(s.contains("latent 256"));
            }
        }
    }
}
