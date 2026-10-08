//! Shared, bounded network/UI utilities. This module is never used by the trainer.
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub(super) fn clean(text: &str) -> String {
    super::process::clean(text).chars().take(4096).collect()
}

pub(super) fn offline() -> bool {
    ["PSSA_OFFLINE", "HF_HUB_OFFLINE"].iter().any(|key| {
        std::env::var(key)
            .is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
    })
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn config_path() -> Result<PathBuf, String> {
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .ok_or("No config directory; set HOME or XDG_CONFIG_HOME")?;
    Ok(root.join("pssa/tui-network.json"))
}

fn read_config(path: &Path) -> Result<Value, String> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(json!({})),
        Err(_) => return Err("Cannot read TUI network config".into()),
    };
    let mut text = String::new();
    file.take(65_537)
        .read_to_string(&mut text)
        .map_err(|_| "Cannot read TUI network config")?;
    if text.len() > 65_536 {
        return Err("TUI network config exceeds 64 KiB".into());
    }
    let value: Value =
        serde_json::from_str(&text).map_err(|_| "Invalid TUI network config JSON")?;
    if !value.is_object() {
        return Err("TUI network config must be an object".into());
    }
    Ok(value)
}

pub(super) fn load_config(section: &str) -> Value {
    config_path()
        .and_then(|p| read_config(&p))
        .ok()
        .and_then(|v| v.get(section).cloned())
        .unwrap_or(Value::Null)
}

pub(super) fn save_config(section: &str, value: &Value) -> Result<(), String> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().map_err(|_| "Config lock unavailable")?;
    save_section(&config_path()?, section, value)
}

fn save_section(path: &Path, section: &str, value: &Value) -> Result<(), String> {
    let mut config = read_config(path)?;
    config[section] = value.clone();
    let bytes = serde_json::to_vec_pretty(&config).map_err(|_| "Cannot encode network config")?;
    if bytes.len() > 65_536 {
        return Err("TUI network config exceeds 64 KiB".into());
    }
    private_write(path, &bytes)
}

/// Same private-file policy as the HF login: create-new temporary, mode 0600,
/// atomic rename; never follow a token-file symlink or log its contents.
pub(super) fn private_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("Missing private-file directory")?;
    fs::create_dir_all(parent).map_err(|_| "Cannot create private-file directory")?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temp)
        .map_err(|_| "Cannot create private temporary file")?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result.map_err(|_| "Cannot save private file".into())
}

/// Injectable boundary used by GitHub browsing and the release checker.
/// Production callers use fixed api.github.com URLs; tests use in-memory fakes.
pub(super) trait Http: Send + Sync {
    fn get(&self, url: &str, token: Option<&str>) -> Result<Value, String>;
}

pub(super) struct Web;
impl Http for Web {
    fn get(&self, url: &str, token: Option<&str>) -> Result<Value, String> {
        if offline() {
            return Err("Offline / request skipped".into());
        }
        if !url.starts_with("https://api.github.com/") {
            return Err("Refusing untrusted GitHub API host".into());
        }
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout(Duration::from_secs(5))
            .build();
        let mut req = agent
            .get(url)
            .set("User-Agent", "pssa-tui/0.5.0")
            .set("Accept", "application/vnd.github+json");
        if let Some(token) = token {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
        let response = req.call().map_err(|e| match e {
            ureq::Error::Status(code, _) => {
                format!("GitHub HTTP {code}; check access or rate limit")
            }
            _ => "GitHub unavailable / check network and retry".into(),
        })?;
        if response.status() != 200 {
            return Err(format!("GitHub HTTP {}", response.status()));
        }
        let mut body = String::new();
        response
            .into_reader()
            .take(1_048_577)
            .read_to_string(&mut body)
            .map_err(|_| "Cannot read GitHub response")?;
        if body.len() > 1_048_576 {
            return Err("GitHub response exceeds 1 MiB".into());
        }
        serde_json::from_str(&body).map_err(|_| "Invalid GitHub response".into())
    }
}

fn run_desktop(mut cmd: Command, input: Option<&str>) -> Result<(), String> {
    let mut child = cmd
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "Desktop helper unavailable; use the displayed text")?;
    if let Some(text) = input {
        if let Some(mut stdin) = child.stdin.take() {
            if stdin.write_all(text.as_bytes()).is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Clipboard unavailable; use the displayed text".into());
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => return Err("Desktop helper failed; use the displayed text".into()),
            _ if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Desktop helper timed out; use the displayed text".into());
            }
            _ => std::thread::sleep(Duration::from_millis(25)),
        }
    }
}

/// Lower only this worker thread's CPU/I/O priority on Linux, not the TUI or
/// trainer. Missing platform helpers are harmless; bounded reads still apply.
pub(super) fn background_io() {
    #[cfg(target_os = "linux")]
    if let Ok(thread) = fs::read_link("/proc/thread-self") {
        if let Some(tid) = thread.file_name().and_then(|s| s.to_str()) {
            for (helper, flags) in [
                ("renice", vec!["-n", "19", "-p", tid]),
                ("ionice", vec!["-c", "3", "-p", tid]),
            ] {
                let mut cmd = Command::new(helper);
                cmd.args(flags);
                let _ = run_desktop(cmd, None);
            }
        }
    }
}

/// Called only from feature workers, never from the render/training thread.
pub(super) fn open_browser(url: &str) -> Result<(), String> {
    if !url.starts_with("https://") || url.chars().any(char::is_control) {
        return Err("Invalid browser URL".into());
    }
    #[cfg(target_os = "macos")]
    let mut cmd = Command::new("open");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler");
        c
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut cmd = Command::new("xdg-open");
    cmd.arg(url);
    run_desktop(cmd, None)
}

pub(super) fn copy_text(text: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let cmd = Command::new("pbcopy");
    #[cfg(target_os = "windows")]
    let cmd = Command::new("clip");
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let cmd = if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        Command::new("wl-copy")
    } else if std::env::var_os("DISPLAY").is_some() {
        let mut cmd = Command::new("xclip");
        cmd.args(["-selection", "clipboard"]);
        cmd
    } else {
        return Err("No desktop clipboard; select and copy the displayed SOL address".into());
    };
    run_desktop(cmd, Some(text))
}

/// Shell-owned coordinator: only canonical live save events schedule uploads.
/// Historical run scans and remote filesystem paths can never trigger a backup.
pub(super) struct Network {
    backup: super::hf_backup::Backup,
    logs: super::log_stream::LogStream,
    notify: super::notify::Notify,
    updates: super::update::Update,
    support: super::support::Support,
    github: super::github::Github,
    sweep: super::sweep::Sweep,
    timeline: super::timeline::Timeline,
    local_events: RunEvents,
    cloud_events: RunEvents,
    kaggle_events: RunEvents,
    local_closed: bool,
    remote_timeline: bool,
    message: Option<String>,
}

#[derive(Default)]
struct RunEvents {
    active: bool,
    completed: bool,
    died: bool,
    saved: bool,
}
impl RunEvents {
    fn ingest(&mut self, line: &str) -> Option<super::notify::Event> {
        use super::notify::Event;
        if line.contains("progress_schema=") {
            *self = Self {
                active: true,
                ..Self::default()
            };
        }
        if saved_checkpoint(line).is_some() {
            self.saved = true;
        }
        if line.contains("tokens_per_second=") && !self.completed && !self.died {
            self.active = true;
        }
        if line.starts_with("error:")
            || line.starts_with("Error:")
            || line.starts_with("thread 'main' panicked")
        {
            if !self.died {
                self.died = true;
                self.active = false;
                return Some(Event::Died);
            }
        } else if line.contains("training_seconds=") && !self.completed && !self.died {
            self.completed = true;
            self.active = false;
            return Some(Event::Finished);
        }
        None
    }
    fn eof(&mut self, success: bool) -> Option<super::notify::Event> {
        use super::notify::Event;
        let event = if !success && !self.died {
            Some(Event::Died)
        } else if success && !self.completed && !self.died {
            Some(Event::Finished)
        } else {
            None
        };
        self.active = false;
        self.completed |= success;
        self.died |= !success;
        event
    }
}
fn saved_checkpoint(line: &str) -> Option<&str> {
    line.strip_prefix("saved_checkpoint=")
        .filter(|p| !p.is_empty() && !p.chars().any(char::is_control))
}
impl Network {
    pub(super) fn new() -> Self {
        Self {
            backup: Default::default(),
            logs: Default::default(),
            notify: Default::default(),
            updates: super::update::Update::load(),
            support: Default::default(),
            github: Default::default(),
            sweep: Default::default(),
            timeline: Default::default(),
            local_events: Default::default(),
            cloud_events: Default::default(),
            kaggle_events: Default::default(),
            local_closed: false,
            remote_timeline: false,
            message: None,
        }
    }
    pub(super) fn ingest(&mut self, line: &str) {
        let stripped = super::strip_ansi(line);
        let line = stripped.trim();
        if line.contains("progress_schema=") {
            self.local_closed = false;
        }
        // Completion of owned/piped trainers is decided by EOF/exit, not a
        // summary that can precede a failed checkpoint write.
        let event = self.local_events.ingest(line);
        if matches!(event, Some(super::notify::Event::Died)) {
            self.notify.event(super::notify::Event::Died);
        }
        if let Some(path) = saved_checkpoint(line) {
            self.backup.checkpoint(path);
            self.notify.event(super::notify::Event::Checkpoint);
        }
    }
    pub(super) fn started(&mut self) {
        self.local_events = RunEvents {
            active: true,
            ..Default::default()
        };
        self.local_closed = false;
    }
    pub(super) fn blocked(&mut self, message: &str) {
        self.message = Some(message.into());
    }
    pub(super) fn deliveries_pending(&self) -> bool {
        self.backup.pending() || self.notify.pending()
    }
    pub(super) fn stream_eof(&mut self) -> Option<bool> {
        // Piped logs have no child exit status. Require the completion summary
        // AND a successful save, not presentation-only loss/spike health flags.
        let active = self.local_events.active
            || self.local_events.completed
            || self.local_events.died;
        if !active {
            return None;
        }
        let success = self.local_events.completed
            && self.local_events.saved
            && !self.local_events.died;
        self.eof(success);
        Some(success)
    }
    pub(super) fn eof(&mut self, success: bool) {
        if self.local_closed
            || !(self.local_events.active || self.local_events.completed || self.local_events.died)
        {
            return;
        }
        self.local_closed = true;
        // Local summaries were not sent early; release the successful terminal
        // event here exactly once, after the owned child's durable log is drained.
        if self.local_events.completed && !self.local_events.died {
            self.local_events.completed = false;
        }
        if let Some(event) = self.local_events.eof(success) {
            self.notify.event(event);
        }
    }
    pub(super) fn remote_line(&mut self, line: &str) {
        let stripped = super::strip_ansi(line);
        let line = stripped.trim();
        if let Some(event) = self.kaggle_events.ingest(line) {
            self.notify.event(event);
        }
        if saved_checkpoint(line).is_some() {
            self.notify.event(super::notify::Event::Checkpoint);
        }
    }
    pub(super) fn context(&self, tab: usize) -> Option<super::keybindings::Context> {
        use super::keybindings::*;
        let editing = match tab {
            BACKUP_TAB => self.backup.editing(),
            LOG_STREAM_TAB => self.logs.editing(),
            NOTIFY_TAB => self.notify.editing(),
            GITHUB_TAB => self.github.editing(),
            SWEEP_TAB => self.sweep.editing(),
            UPDATE_TAB | SUPPORT_TAB | TIMELINE_TAB => false,
            _ => return None,
        };
        Some(Context::for_tab(tab, editing))
    }
    /// Call after polling the current child and delivering its EOF; installing
    /// the next sweep child uses the same sole TrainingRun slot as the wizard.
    pub(super) fn poll(
        &mut self,
        state: &mut super::RunState,
        tab: usize,
        training: &mut Option<super::setup::TrainingRun>,
        busy: bool,
        remote: bool,
    ) -> bool {
        use super::keybindings::*;
        self.backup.poll();
        for line in self.logs.poll() {
            if let Some(event) = self.cloud_events.ingest(&line) {
                self.notify.event(event);
            }
            if saved_checkpoint(&line).is_some() {
                self.notify.event(super::notify::Event::Checkpoint);
            }
        }
        self.notify.poll();
        self.updates.poll();
        self.support.poll();
        self.github.poll(tab == GITHUB_TAB);
        self.remote_timeline = remote;
        if !remote {
            self.timeline.poll(state, tab == TIMELINE_TAB || tab == 1);
            for (name, loss) in self.timeline.recorded_losses() {
                if let Some((_, recorded)) =
                    state.checkpoints.iter_mut().find(|(file, _)| *file == name)
                {
                    // Save-event metrics are newer than a bounded historical
                    // scan, including when a checkpoint overwrites the same path.
                    recorded.get_or_insert(loss);
                }
            }
        }
        let started = self.sweep.poll(training, state, busy);
        if started {
            self.started();
        }
        started
    }
    pub(super) fn key(
        &mut self,
        key: crossterm::event::KeyEvent,
        tab: &mut usize,
        setup: &mut super::setup::Setup,
        chat: &mut super::chat::Chat,
    ) {
        use super::keybindings::*;
        self.message = None;
        match *tab {
            BACKUP_TAB => self.backup.key(key),
            LOG_STREAM_TAB => self.logs.key(key),
            NOTIFY_TAB => self.notify.key(key),
            UPDATE_TAB => self.updates.key(key),
            SUPPORT_TAB => self.support.key(key),
            GITHUB_TAB => self.github.key(key),
            SWEEP_TAB => self.sweep.key(key, setup),
            TIMELINE_TAB if self.remote_timeline => self.blocked(
                "Remote checkpoints are not local files. Sync the run to use its timeline.",
            ),
            TIMELINE_TAB => {
                if let Some(action) = self.timeline.key(key) {
                    match action {
                        super::timeline::Action::Chat(path) => {
                            chat.open_checkpoint(&path);
                            *tab = 4;
                        }
                        super::timeline::Action::Resume(path) => match setup.prepare_resume(path) {
                            Ok(()) => *tab = 5,
                            Err(e) => self.message = Some(e),
                        },
                    }
                }
            }
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut ratatui::Frame, tab: usize) {
        use super::keybindings::*;
        let area = super::feature_area(f.area());
        match tab {
            BACKUP_TAB => self.backup.draw(f, area),
            LOG_STREAM_TAB => self.logs.draw(f, area),
            NOTIFY_TAB => self.notify.draw(f, area),
            UPDATE_TAB => self.updates.draw(f, area),
            SUPPORT_TAB => self.support.draw(f, area),
            GITHUB_TAB => self.github.draw(f, area),
            SWEEP_TAB => self.sweep.draw(f, area),
            TIMELINE_TAB if self.remote_timeline => {
                let timeline_area = super::panel_area(f, area);
                f.render_widget(
                    ratatui::widgets::Paragraph::new(
                        "Remote checkpoints unavailable locally / sync the run first",
                    )
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .block(super::panel(" timeline ")),
                    timeline_area,
                )
            }
            TIMELINE_TAB => self.timeline.draw(f, area),
            _ => {}
        }
        if tab == 0 {
            self.updates.banner(f);
            let screen = f.area();
            if self.deliveries_pending() && screen.height >= 10 {
                let line =
                    ratatui::layout::Rect::new(screen.x, screen.bottom() - 2, screen.width, 1);
                f.render_widget(ratatui::widgets::Clear, line);
                f.render_widget(
                    ratatui::widgets::Paragraph::new(
                        "[ NETWORK ] Background delivery / q quits without waiting",
                    )
                    .style(super::accent()),
                    line,
                );
            }
        }
        if let Some(message) = &self.message {
            let screen = f.area();
            if !screen.is_empty() {
                f.render_widget(
                    ratatui::widgets::Paragraph::new(clean(message)).style(super::accent()),
                    ratatui::layout::Rect::new(screen.x, screen.bottom() - 1, screen.width, 1),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_roundtrip_preserves_other_sections_and_permissions() {
        let root = std::env::temp_dir().join(format!("pssa-network-config-{}", std::process::id()));
        let path = root.join("network.json");
        save_section(&path, "updates", &json!({"enabled":false})).unwrap();
        save_section(&path, "backup", &json!({"every":3})).unwrap();
        let value = read_config(&path).unwrap();
        assert_eq!(value["updates"]["enabled"], false);
        assert_eq!(value["backup"]["every"], 3);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        fs::write(&path, "malformed").unwrap();
        assert!(save_section(&path, "updates", &json!({})).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "malformed");
        fs::remove_dir_all(root).unwrap();
    }
    fn fixture() -> Network {
        Network {
            backup: super::super::hf_backup::Backup::with_uploader(
                json!({"every":0}),
                std::sync::Arc::new(|_, _, _| panic!("unexpected upload")),
            ),
            logs: super::super::log_stream::LogStream::with_opener(std::sync::Arc::new(|_, _| {
                panic!("unexpected log request")
            })),
            notify: super::super::notify::Notify::with_sender(
                json!({"enabled":false}),
                std::sync::Arc::new(|_, _, _, _| panic!("unexpected notification")),
            ),
            updates: Default::default(),
            support: Default::default(),
            github: Default::default(),
            sweep: Default::default(),
            timeline: Default::default(),
            local_events: Default::default(),
            cloud_events: Default::default(),
            kaggle_events: Default::default(),
            local_closed: false,
            remote_timeline: false,
            message: None,
        }
    }
    #[test]
    fn historical_losses_fill_gaps_without_replacing_same_path_save_metrics() {
        let root = std::env::temp_dir().join(format!("pssa-network-losses-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let saved = root.join("model.pssa");
        let missing = root.join("ck01.pssa");
        fs::write(&saved, "read-only saved fixture").unwrap();
        fs::write(&missing, "read-only historical fixture").unwrap();
        fs::write(root.join("train.log"), format!(
            "loss=9 tokens_per_second=40\nsaved_checkpoint={}\nloss=4 tokens_per_second=80\nsaved_checkpoint={}\n",
            saved.display(), missing.display()
        )).unwrap();
        let mut state = super::super::RunState {
            chain_dir: root.clone(),
            live_loss: Some(2.0),
            ..Default::default()
        };
        state.note_checkpoint(&saved.display().to_string());
        state.raw_lines.clear(); // Fresh explicit metrics can outlive the bounded recent log.
        state.refresh_chain();
        let mut ui = fixture();
        let mut training = None;
        let deadline = Instant::now() + Duration::from_secs(5);
        while ui.timeline.recorded_losses().count() < 2 && Instant::now() < deadline {
            ui.poll(&mut state, 1, &mut training, false, false);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(ui.timeline.recorded_losses().count(), 2);
        let loss = |state: &super::super::RunState, name: &str| {
            state
                .checkpoints
                .iter()
                .find(|(file, _)| file == name)
                .unwrap()
                .1
        };
        assert_eq!(
            loss(&state, "model.pssa"),
            Some(2.0),
            "cached historical loss is not newer than an explicit save"
        );
        assert_eq!(
            loss(&state, "ck01.pssa"),
            Some(4.0),
            "missing chain history is still filled"
        );

        fs::write(&saved, "overwritten saved fixture").unwrap();
        state.ingest("loss=1 tokens_per_second=100 global_update=600");
        state.ingest(&format!("saved_checkpoint={}", saved.display()));
        ui.poll(&mut state, 0, &mut training, false, false);
        assert_eq!(loss(&state, "model.pssa"), Some(1.0));
        assert_eq!(
            ui.timeline.recorded_losses().count(),
            0,
            "same-root save invalidates history even off-tab"
        );
        assert_eq!(
            fs::read_to_string(&saved).unwrap(),
            "overwritten saved fixture"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canonical_save_events_and_terminal_transitions_are_not_progress_snapshots() {
        use super::super::notify::Event;
        assert_eq!(
            saved_checkpoint("saved_checkpoint=runs/path with spaces/model.pssa"),
            Some("runs/path with spaces/model.pssa")
        );
        for line in [
            "last_checkpoint=model.pssa",
            "resumed_from=model.pssa",
            "checkpoint_status=saved",
            "saved_checkpoint=",
        ] {
            assert!(saved_checkpoint(line).is_none());
        }
        let mut events = RunEvents::default();
        assert_eq!(events.ingest("progress_schema=2"), None);
        assert_eq!(events.ingest("training_seconds=2"), Some(Event::Finished));
        assert_eq!(events.ingest("training_seconds=2"), None);
        assert_eq!(events.eof(true), None);
        events.ingest("progress_schema=2");
        assert_eq!(events.eof(false), Some(Event::Died));
        assert_eq!(events.eof(false), None);
        assert_eq!(events.ingest("error: later detail"), None);
    }
    #[test]
    fn all_network_tabs_preserve_shell_and_render_at_narrow_widths_offline() {
        use super::super::{RunState, keybindings::*};
        use ratatui::{Terminal, backend::TestBackend};
        let ui = fixture();
        let state = RunState::default();
        for (w, h) in [
            (160, 40),
            (100, 32),
            (80, 24),
            (79, 24),
            (24, 8),
            (1, 1),
            (0, 0),
        ] {
            for tab in [
                BACKUP_TAB,
                LOG_STREAM_TAB,
                NOTIFY_TAB,
                UPDATE_TAB,
                SUPPORT_TAB,
                GITHUB_TAB,
                SWEEP_TAB,
                TIMELINE_TAB,
            ] {
                let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
                t.draw(|f| super::super::draw(f, &state, tab)).unwrap();
                let shell = t.backend().buffer().clone();
                t.draw(|f| {
                    super::super::draw(f, &state, tab);
                    ui.draw(f, tab);
                })
                .unwrap();
                let content = super::super::feature_area(ratatui::layout::Rect::new(0, 0, w, h));
                for y in 0..content.y {
                    for x in 0..w {
                        assert_eq!(
                            shell[(x, y)],
                            t.backend().buffer()[(x, y)],
                            "shell overwritten at tab {tab}, {w}x{h}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn piped_completion_requires_summary_and_save_not_health_heuristics() {
        for saved in [false, true] {
            let mut ui = fixture();
            ui.ingest("progress_schema=2");
            ui.ingest("training_seconds=2");
            if saved {
                ui.ingest("saved_checkpoint=path with spaces/model.pssa");
            }
            assert_eq!(ui.stream_eof(), Some(saved));
            assert_eq!(ui.local_events.died, !saved);
            assert!(
                !ui.deliveries_pending(),
                "opt-out defaults must preserve immediate EOF"
            );
        }
    }
    #[test]
    fn local_completion_waits_for_exit_and_does_not_notify_again_on_repeated_eof() {
        let mut ui = fixture();
        ui.started();
        ui.ingest("training_seconds=2");
        assert!(!ui.local_closed);
        ui.eof(true);
        assert!(ui.local_closed);
        ui.eof(false);
        assert!(
            !ui.local_events.died,
            "repeated EOF must not invent a crash"
        );
    }
    #[test]
    fn github_transport_rejects_untrusted_origins_before_sending_credentials() {
        assert!(
            Web.get("https://example.com/leak", Some("fixture-secret"))
                .is_err()
        );
    }
}
