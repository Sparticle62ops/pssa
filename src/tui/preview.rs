//! Read-only checkpoint sampling in a bounded, low-priority CPU child.
//! No model loads, token generation, or pipe waits occur on the UI/trainer thread.
use super::{
    RunState, accent,
    heatmap::{self, TokenMark},
    panel, panel_area,
};
use crate::{
    cli::CLIHandler,
    inference::{InferenceConfig, PSSAInferenceEngine},
};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant, SystemTime},
};

const INTERVAL: Duration = Duration::from_secs(60);
const TIMEOUT: Duration = Duration::from_secs(20);
const MAX_OUTPUT: u64 = 128 * 1024;
const MAX_CHECKPOINT: u64 = 128 * 1024 * 1024;

// Also used for bounded hardware queries. Drain stdout on a separate reader;
// a pipe-full child cannot hold up drawing. Drop kills/reaps abandoned workers.
pub(super) struct Process {
    child: Option<Child>,
    output: mpsc::Receiver<Result<Vec<u8>, String>>,
    started: Instant,
    timeout: Duration,
}
impl Process {
    pub(super) fn start(mut command: Command, timeout: Duration) -> Result<Self, String> {
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("worker has no stdout")?;
        let (tx, output) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let result = stdout
                .take(MAX_OUTPUT + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())
                .and_then(|_| {
                    if bytes.len() as u64 > MAX_OUTPUT {
                        Err("worker output exceeded limit".into())
                    } else {
                        Ok(bytes)
                    }
                });
            let _ = tx.send(result);
        });
        Ok(Self {
            child: Some(child),
            output,
            started: Instant::now(),
            timeout,
        })
    }
    pub(super) fn poll(&mut self) -> Option<Result<Vec<u8>, String>> {
        let child = self.child.as_mut()?;
        if self.started.elapsed() >= self.timeout {
            self.stop();
            return Some(Err("worker timed out (training is unaffected)".into()));
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                // Exit can race the stdout reader's final send; wait until a later tick.
                match self.output.try_recv() {
                    Ok(result) => {
                        self.child = None;
                        Some(if status.success() {
                            result
                        } else {
                            Err(format!("worker exited {status}"))
                        })
                    }
                    Err(mpsc::TryRecvError::Empty) => None,
                    Err(_) => {
                        self.child = None;
                        Some(Err("worker output closed".into()))
                    }
                }
            }
            Ok(None) => None,
            Err(e) => {
                let error = e.to_string();
                self.stop();
                Some(Err(error))
            }
        }
    }
    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Revision {
    path: PathBuf,
    modified: SystemTime,
    bytes: u64,
    loops: usize,
}

pub(super) struct Preview {
    job: Option<Process>,
    remote: bool,
    revision: Option<Revision>,
    last_scan: Option<Instant>,
    last_attempt: Option<Instant>,
    pub enabled: bool,
    pub heatmap: bool,
    pub text: String,
    pub marks: Vec<TokenMark>,
    pub note: String,
    prompt: String,
    checkpoint: String,
    waiting_context: String,
}
impl Default for Preview {
    fn default() -> Self {
        Self {
            job: None,
            remote: false,
            revision: None,
            last_scan: None,
            last_attempt: None,
            enabled: true,
            heatmap: false,
            text: String::new(),
            marks: Vec::new(),
            prompt: String::new(),
            checkpoint: String::new(),
            note: "No checkpoint yet / no training save target connected".into(),
            waiting_context: "No checkpoint yet / no training save target connected".into(),
        }
    }
}
impl Preview {
    pub(super) fn candidate(state: &RunState) -> Option<PathBuf> {
        // Do not sample an old chain while a fresh wizard run has no saved state.
        if let Some(path) = state
            .last_checkpoint
            .as_deref()
            .or(state.selected_checkpoint.as_deref())
            .or(state.resumed_from.as_deref())
        {
            let path = PathBuf::from(path);
            if path.extension().is_some_and(|x| x == "pssa" || x == "trfm") && path.is_file() {
                return Some(path);
            }
        }
        // An unsaved connected run must never borrow a previous run's model,
        // even when its output directory already contains other checkpoints.
        if state.training_active || state.run_started_at.is_some() {
            return None;
        }
        state
            .checkpoints
            .iter()
            .rev()
            .filter(|(name, _)| name.ends_with(".pssa") || name.ends_with(".trfm"))
            .map(|(name, _)| state.chain_dir.join(name))
            .find(|path| path.is_file())
    }

    pub(super) fn set_waiting_context(&mut self, context: String) {
        self.waiting_context = context;
    }

    fn clear_missing_sample(&mut self) {
        self.job = None;
        self.revision = None;
        self.text.clear();
        self.marks.clear();
        self.checkpoint.clear();
        self.note = if self.enabled {
            self.waiting_context.clone()
        } else {
            "Preview paused / F7 resumes".into()
        };
    }
    pub(super) fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if !self.enabled {
            self.job = None;
            self.revision = None;
            self.note = "Preview paused / F7 resumes".into();
        } else {
            self.note = "Waiting for next throttled checkpoint sample".into();
        }
    }
    pub(super) fn poll(&mut self, path: Option<PathBuf>, loops: usize, remote: bool) {
        if self.remote != remote {
            // Drop cancels an outstanding local worker. Never present an old
            // local sample as output from a remote run (or vice versa).
            *self = Self {
                remote,
                enabled: self.enabled,
                heatmap: self.heatmap,
                waiting_context: self.waiting_context.clone(),
                ..Self::default()
            };
            if !self.enabled {
                self.note = "Preview paused / F7 resumes".into();
            }
        }
        if remote {
            // A Kaggle path may coincidentally exist locally. Do not stat it or
            // launch inference, even when preview is enabled by the user.
            self.note =
                "Preview unavailable for remote Kaggle checkpoints / local files are not sampled"
                    .into();
            return;
        }
        if path.is_none() {
            self.clear_missing_sample();
            return;
        }
        let requested = path
            .as_ref()
            .map(|path| heatmap::clean(&path.to_string_lossy()))
            .unwrap();
        if !self.checkpoint.is_empty() && self.checkpoint != requested {
            self.job = None;
            self.revision = None;
            self.text.clear();
            self.marks.clear();
            self.prompt.clear();
            self.checkpoint = requested.clone();
            self.note = if self.enabled {
                "Run changed; waiting for next throttled checkpoint sample".into()
            } else {
                "Preview paused / F7 resumes".into()
            };
        }
        let source_exists = path.as_ref().is_some_and(|path| path.is_file());
        if !source_exists {
            self.clear_missing_sample();
            if self.enabled {
                self.note = format!("Checkpoint file unavailable: {requested}");
            }
            return;
        }
        if self.checkpoint.is_empty() {
            self.checkpoint = requested.clone();
            if self.enabled {
                self.note =
                    "Checkpoint file available; waiting for next throttled read-only sample".into();
            }
        }
        if let Some(job) = &mut self.job {
            if let Some(result) = job.poll() {
                self.job = None;
                match result.and_then(|bytes| self.accept(&bytes)) {
                    Ok(()) => {
                        self.note = if self.checkpoint.ends_with(".trfm") {
                            "Read-only transformer CPU sample / confidence not exposed by this sampler / at most 1/min".into()
                        } else {
                            "Read-only CPU sample / refresh on changed checkpoint, at most 1/min"
                                .into()
                        }
                    }
                    Err(e) => {
                        self.note = format!("Preview unavailable: {e}");
                        self.revision = None;
                    }
                }
            }
            return;
        }
        if !self.enabled
            || self.last_attempt.is_some_and(|t| t.elapsed() < INTERVAL)
            || self
                .last_scan
                .is_some_and(|t| t.elapsed() < Duration::from_secs(5))
        {
            return;
        }
        self.last_scan = Some(Instant::now());
        let Some(path) = path else {
            return;
        };
        let Ok(metadata) = path.metadata() else {
            self.clear_missing_sample();
            return;
        };
        if !metadata.is_file() {
            self.clear_missing_sample();
            return;
        }
        let revision = Revision {
            path: path.clone(),
            bytes: metadata.len(),
            modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            loops: loops.clamp(1, 32),
        };
        if self.revision.as_ref() == Some(&revision) {
            return;
        }
        // Allow an in-progress non-atomic save to settle; the checkpoint reader
        // validates it, and failure retries after the throttle instead of writing it.
        if revision.modified.elapsed().unwrap_or_default() < Duration::from_secs(2) {
            self.checkpoint = requested;
            self.note = "Checkpoint file changed; waiting for the 2s writer-settle guard before read-only sampling".into();
            return;
        }
        self.last_attempt = Some(Instant::now());
        if revision.bytes > MAX_CHECKPOINT {
            self.note = "Preview skipped: checkpoint >128 MiB (protecting training RAM)".into();
            self.revision = Some(revision);
            return;
        }
        let result = std::env::current_exe()
            .map_err(|e| e.to_string())
            .and_then(|exe| {
                let nice = cfg!(unix) && Path::new("/usr/bin/nice").is_file();
                let mut cmd =
                    if cfg!(target_os = "linux") && Path::new("/usr/bin/prlimit").is_file() {
                        let mut cmd = Command::new("/usr/bin/prlimit");
                        cmd.args(["--as=536870912:536870912", "--"]);
                        if nice {
                            cmd.args(["/usr/bin/nice", "-n", "19"]);
                        }
                        cmd.arg(exe);
                        cmd
                    } else if nice {
                        let mut cmd = Command::new("/usr/bin/nice");
                        cmd.args(["-n", "19"]).arg(exe);
                        cmd
                    } else {
                        Command::new(exe)
                    };
                cmd.arg("tui")
                    .arg("--preview-worker")
                    .arg(&path)
                    .arg(revision.loops.to_string())
                    .env("RAYON_NUM_THREADS", "1")
                    .env("TOKENIZERS_PARALLELISM", "false");
                Process::start(cmd, TIMEOUT)
            });
        let checkpoint = heatmap::clean(&path.to_string_lossy());
        if checkpoint != self.checkpoint {
            self.text.clear();
            self.marks.clear();
            self.prompt.clear();
        }
        self.checkpoint = checkpoint;
        self.revision = Some(revision);
        match result {
            Ok(job) => {
                self.job = Some(job);
                self.note = "Sampling checkpoint in low-priority CPU worker…".into();
            }
            Err(e) => {
                self.note = format!("Cannot start preview: {e}");
                self.revision = None;
            }
        }
    }
    fn accept(&mut self, bytes: &[u8]) -> Result<(), String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if let Some(error) = v["error"].as_str() {
            return Err(heatmap::clean(error));
        }
        let text = v["text"].as_str().ok_or("sample has no text")?;
        let marks = heatmap::from_json(&v["confidence"])?;
        if !heatmap::valid(text, &marks) {
            return Err("invalid sample confidence".into());
        }
        self.text = text.to_owned();
        self.marks = marks;
        self.prompt = heatmap::clean(v["prompt"].as_str().unwrap_or(""));
        Ok(())
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect, compact: bool) {
        let area = panel_area(f, area);
        let mut lines = if compact {
            Vec::new()
        } else {
            vec![
                Line::styled("F7 pause/resume / F6 confidence / Tab tabs", accent()),
                Line::from(format!(
                    "Checkpoint: {}",
                    if self.checkpoint.is_empty() {
                        "not saved locally"
                    } else {
                        &self.checkpoint
                    }
                )),
                Line::from(format!(
                    "Prompt: {}",
                    if self.prompt.is_empty() {
                        "selected from saved tokenizer after load"
                    } else {
                        &self.prompt
                    }
                )),
                Line::from("32 tokens / 20s timeout / one CPU thread / no GPU"),
                Line::from("Linux prlimit when installed: 512 MiB worker address-space cap"),
            ]
        };
        lines.push(Line::from(heatmap::clean(&self.note)));
        lines.push(heatmap::legend(self.heatmap));
        if !self.text.is_empty() {
            lines.extend(heatmap::lines(&self.text, &self.marks, self.heatmap));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" live sample / F7 pause ")),
            area,
        );
    }
}

fn available_memory() -> Option<u64> {
    let host = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|mem| {
            mem.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
                .map(|kib| kib.saturating_mul(1024))
        });
    // Container memory pressure can be much tighter than the host's /proc view.
    let number = |path| {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    };
    let cgroup = number("/sys/fs/cgroup/memory.max")
        .zip(number("/sys/fs/cgroup/memory.current"))
        .or_else(|| {
            number("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                .zip(number("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        })
        .map(|(limit, used)| limit.saturating_sub(used));
    match (host, cgroup) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Internal subprocess endpoint; never enters the terminal session or trainer.
pub(super) fn worker(path: &str, loops: usize) -> Result<(), String> {
    let result = (|| {
        if std::fs::metadata(path).map_err(|e| e.to_string())?.len() > MAX_CHECKPOINT {
            return Err("checkpoint exceeds preview memory guard".into());
        }
        // Refuse under pressure instead of competing with training for the last RAM.
        let bytes = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
        if available_memory()
            .is_some_and(|free| free < bytes.saturating_mul(8).max(256 * 1024 * 1024))
        {
            return Err("not enough free RAM for a safe preview".into());
        }
        if Path::new(path)
            .extension()
            .is_some_and(|extension| extension == "trfm")
        {
            let model = crate::transformer_checkpoint::load_checkpoint(path)
                .map_err(|error| error.to_string())?;
            let tok = model.tokenizer()?;
            let prompt = prompt_for(&tok)?;
            drop(model);
            let text = crate::transformer_inference::generate_controlled(
                path,
                &prompt,
                &InferenceConfig {
                    max_new_tokens: 32,
                    ..Default::default()
                },
                &mut |_, _| {},
                &|| {
                    std::thread::sleep(Duration::from_millis(10));
                    false
                },
            )?;
            // The transformer sampler does not expose probabilities. Empty
            // confidence means unmeasured, never synthesized confidence.
            return Ok(serde_json::json!({"prompt":prompt,"text":text,"confidence":[]}));
        }
        let (mut model, tok) = CLIHandler::load_for_inference(path, None)?;
        model.set_loops(loops)?;
        let prompt = prompt_for(&tok)?;
        let mut marks = Vec::new();
        let text = PSSAInferenceEngine::try_new(&mut model, &tok)?.try_generate_chat_turn_scored(
            &prompt,
            &InferenceConfig {
                max_new_tokens: 32,
                ..Default::default()
            },
            |text, count, p| heatmap::record(&mut marks, text, count, p),
            || {
                // A short cooperative pause further yields CPU between tokens;
                // the parent enforces the hard wall-time bound even during load.
                std::thread::sleep(Duration::from_millis(10));
                false
            },
        )?;
        Ok::<_, String>(
            serde_json::json!({"prompt":prompt,"text":text,"confidence":heatmap::to_json(&marks)}),
        )
    })();
    let value = result.unwrap_or_else(|e| serde_json::json!({"error":e}));
    println!("{value}");
    Ok(())
}

fn prompt_for(tok: &crate::dataset::Tokenizer) -> Result<String, String> {
    match tok.kind() {
        crate::dataset::TokenizerKind::Bpe => Ok("The".to_owned()),
        crate::dataset::TokenizerKind::Word => (1..tok.vocab_size)
            .filter_map(|id| tok.id_to_token.get(&id))
            .find(|piece| piece.chars().any(char::is_alphanumeric))
            .cloned()
            .ok_or_else(|| "no usable prompt token".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn recently_written_checkpoint_reports_settling_instead_of_no_checkpoint() {
        let temp = super::super::library::tests::Temp::new();
        let path = temp.0.join("recent checkpoint.pssa");
        std::fs::write(&path, "recent checkpoint fixture; not loaded").unwrap();
        let mut preview = Preview::default();
        preview.poll(Some(path.clone()), 1, false);
        assert!(preview.note.contains("writer-settle"));
        assert!(!preview.note.contains("No checkpoint"));
        assert_eq!(preview.checkpoint, path.display().to_string());
        assert!(preview.job.is_none());
    }

    #[test]
    fn throttled_saved_checkpoint_and_remote_transitions_keep_honest_source_context() {
        let temp = super::super::library::tests::Temp::new();
        let path = temp.0.join("saved but throttled.pssa");
        std::fs::write(&path, "checkpoint fixture; not loaded while throttled").unwrap();
        let mut preview = Preview {
            last_attempt: Some(Instant::now()),
            ..Preview::default()
        };
        preview.set_waiting_context(path.display().to_string());
        preview.poll(Some(path.clone()), 1, false);
        assert_eq!(preview.checkpoint, path.display().to_string());
        assert!(preview.note.contains("throttled"));
        assert!(!preview.note.contains("No checkpoint"));
        assert!(preview.job.is_none());
        preview.set_waiting_context(
            "First checkpoint at run end (planned step 500); current step 120".into(),
        );
        preview.poll(None, 1, true);
        preview.poll(None, 1, false);
        assert!(preview.note.contains("planned step 500"));
        assert!(preview.note.contains("current step 120"));
    }

    #[test]
    fn placeholder_sample_and_heatmap_render_wide_and_compact() {
        for (width, height) in [(120, 26), (79, 20), (40, 14), (1, 1)] {
            let mut preview = Preview::default();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| preview.draw(f, f.area(), false)).unwrap();
            if width >= 40 {
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("No checkpoint yet"));
                preview.accept(br#"{"prompt":"hello","text":"sample","confidence":[{"end":6,"probability":0.9}]}"#).unwrap();
                preview.heatmap = true;
                terminal.draw(|f| preview.draw(f, f.area(), true)).unwrap();
                assert!(
                    terminal
                        .backend()
                        .buffer()
                        .content()
                        .iter()
                        .any(|c| c.symbol() == "s" && c.fg == super::super::NORMAL_GREEN)
                );
            }
        }
    }

    #[test]
    fn waiting_sample_has_real_save_context_and_transformer_files_are_candidates() {
        let mut state = RunState::default();
        state.ingest(
            "progress_schema=2 updates_total=500 prior_updates=0 checkpoint_target=/tmp/model.trfm",
        );
        state.ingest("training 120/500 (24%) loss=3 tokens_per_second=100 optimizer_updates=120 global_update=120");
        let mut preview = Preview::default();
        preview.set_waiting_context(state.checkpoint_context());
        preview.poll(None, 1, false);
        assert!(preview.note.contains("step 500"));
        assert!(preview.note.contains("current step 120"));
        assert!(preview.job.is_none());
        let path = std::env::temp_dir().join(format!(
            "pssa-preview-transformer-candidate-{}.trfm",
            std::process::id()
        ));
        std::fs::write(&path, b"candidate fixture").unwrap();
        state.last_checkpoint = Some(path.to_string_lossy().into_owned());
        assert_eq!(Preview::candidate(&state), Some(path.clone()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fresh_runs_do_not_sample_unrelated_files_after_directory_rescans() {
        let root =
            std::env::temp_dir().join(format!("pssa-preview-unrelated-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let old = root.join("old.pssa");
        std::fs::write(&old, b"old checkpoint fixture").unwrap();
        let mut state = RunState {
            chain_dir: root.clone(),
            ..Default::default()
        };
        state.refresh_chain();
        assert_eq!(
            Preview::candidate(&state),
            Some(old.clone()),
            "explicit historical browsing can sample real files"
        );
        state.ingest(&format!(
            "progress_schema=2 updates_total=500 prior_updates=0 checkpoint_target={}",
            root.join("new.pssa").display()
        ));
        state.refresh_chain();
        assert!(!state.checkpoints.is_empty());
        assert_eq!(Preview::candidate(&state), None);
        state.ingest("training_seconds=1 optimizer_updates=500");
        assert_eq!(
            Preview::candidate(&state),
            None,
            "completion before first save is not a license to sample an old file"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changing_checkpoint_drops_old_text_immediately_even_during_throttle() {
        let temp = super::super::library::tests::Temp::new();
        let path = temp.0.join("new.pssa");
        std::fs::write(&path, "fixture; not read while throttled").unwrap();
        let mut preview = Preview {
            checkpoint: "old.pssa".into(),
            text: "old generated text".into(),
            last_attempt: Some(Instant::now()),
            ..Default::default()
        };
        preview.poll(Some(path.clone()), 1, false);
        assert!(preview.text.is_empty());
        assert!(preview.marks.is_empty());
        assert!(preview.job.is_none());
        assert!(preview.note.contains("Run changed"));
        std::fs::remove_file(&path).unwrap();
        preview.poll(Some(path.clone()), 1, false);
        assert!(preview.checkpoint.is_empty());
        assert!(preview.note.contains("Checkpoint file unavailable"));
        assert!(preview.note.contains(path.to_str().unwrap()));
        assert!(preview.job.is_none());
    }

    #[test]
    fn errors_retain_last_sample_and_pausing_never_starts_work() {
        let mut preview = Preview::default();
        preview.text = "keep me".into();
        assert!(
            preview
                .accept(br#"{"error":"incomplete checkpoint"}"#)
                .is_err()
        );
        assert_eq!(preview.text, "keep me");
        preview.toggle();
        preview.poll(Some(PathBuf::from("missing.pssa")), 1, false);
        assert!(preview.job.is_none());
        assert!(preview.last_attempt.is_none());
        preview.poll(None, 1, false);
        assert!(preview.note.contains("paused"));
        let candidate = std::env::temp_dir().join(format!(
            "pssa-preview-candidate-{}.pssa",
            std::process::id()
        ));
        std::fs::write(&candidate, b"fixture").unwrap();
        let mut state = RunState::default();
        state.last_checkpoint = Some(candidate.to_string_lossy().into_owned());
        assert_eq!(Preview::candidate(&state), Some(candidate.clone()));
        std::fs::remove_file(&candidate).unwrap();
        assert!(Preview::candidate(&state).is_none());
        state.last_checkpoint = Some("a.trfm".into());
        assert!(Preview::candidate(&state).is_none());

        let mut stale = Preview::default();
        stale.text = "old sample".into();
        stale.checkpoint = "deleted.pssa".into();
        stale.poll(Preview::candidate(&state), 1, false);
        assert!(stale.text.is_empty());
        assert!(stale.checkpoint.is_empty());
        assert!(stale.note.contains("No checkpoint"));
    }

    #[cfg(unix)]
    #[test]
    fn remote_paths_never_start_local_samples_and_switching_cancels_stale_work() {
        let path =
            std::env::temp_dir().join(format!("pssa-preview-remote-{}.pssa", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        // A sparse file exercises the local budget guard without loading a
        // model. Give it a settled timestamp so the local poll must inspect it.
        file.set_len(MAX_CHECKPOINT + 1).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))
            .unwrap();
        let mut command = Command::new("sleep");
        command.arg("30");
        let mut preview = Preview {
            job: Some(Process::start(command, Duration::from_secs(60)).unwrap()),
            text: "previous local sample".into(),
            checkpoint: "old.pssa".into(),
            heatmap: true,
            ..Preview::default()
        };
        preview.poll(Some(path.clone()), 1, true);
        assert!(
            preview.job.is_none(),
            "source change must cancel the old worker"
        );
        assert!(preview.text.is_empty());
        assert!(preview.checkpoint.is_empty());
        assert!(
            preview.last_scan.is_none(),
            "remote path must not even be inspected"
        );
        assert!(preview.last_attempt.is_none());
        assert!(preview.note.contains("remote Kaggle"));
        preview.toggle();
        preview.toggle();
        preview.poll(Some(path.clone()), 1, true);
        assert!(preview.job.is_none());
        assert!(
            preview.last_attempt.is_none(),
            "F7 cannot override the remote boundary"
        );
        preview.poll(Some(path.clone()), 1, false);
        assert!(preview.last_scan.is_some());
        assert!(
            preview.last_attempt.is_some(),
            "returning to local runs restores sampling"
        );
        assert!(preview.note.contains("checkpoint >128 MiB"));
        assert!(preview.heatmap, "source changes retain display preferences");
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_output_is_drained_and_timeouts_do_not_wait_for_generation() {
        let mut command = Command::new("sh");
        command.args(["-c", "printf sample"]);
        let mut process = Process::start(command, Duration::from_secs(2)).unwrap();
        let output = loop {
            if let Some(output) = process.poll() {
                break output.unwrap();
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(output, b"sample");
        let mut command = Command::new("sleep");
        command.arg("5");
        let mut process = Process::start(command, Duration::ZERO).unwrap();
        assert!(process.poll().unwrap().unwrap_err().contains("timed out"));
        assert!(process.child.is_none());
    }
}
