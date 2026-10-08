//! Optional notifications. OFF by default; the only outbound payloads are fixed
//! checkpoint/finished/died summaries, never log lines, paths or model prompts.
//! One worker, three coalesced event kinds, bounded timeout, explicit retry only.
use super::{accent, network, panel, panel_area};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

const SECTION: &str = "notify";
const MAX_TARGET: usize = 2048;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Checkpoint,
    Finished,
    Died,
}
impl Event {
    fn name(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Finished => "finished",
            Self::Died => "died",
        }
    }
    fn message(self) -> &'static str {
        match self {
            Self::Checkpoint => "PSSA: a new checkpoint was saved.",
            Self::Finished => "PSSA: training finished.",
            Self::Died => "PSSA: training stopped with an error.",
        }
    }
}
/// Worker-only injectable boundary. The endpoint can contain a private webhook
/// path: never echo it, HTTP diagnostics, or response bodies in returned errors.
pub(super) type Sender =
    Arc<dyn Fn(&str, bool, Event, &AtomicBool) -> Result<(), String> + Send + Sync>;

pub(super) struct Notify {
    enabled: bool,
    webhook: bool,
    target: String,
    selected: usize,
    input: Option<String>,
    status: String,
    queue: VecDeque<Event>,
    pending: Option<mpsc::Receiver<Result<(), String>>>,
    cancel: Arc<AtomicBool>,
    last: Option<Event>,
    failures: u64,
    history: VecDeque<String>,
    sender: Sender,
    persist: bool,
}
impl Default for Notify {
    fn default() -> Self {
        Self::new(network::load_config(SECTION), Arc::new(send), true)
    }
}
impl Notify {
    #[cfg(test)]
    pub(super) fn with_sender(config: Value, sender: Sender) -> Self {
        Self::new(config, sender, false)
    }
    fn new(config: Value, sender: Sender, persist: bool) -> Self {
        let webhook = config["mode"].as_str() == Some("webhook");
        let target = config["target"]
            .as_str()
            .filter(|s| s.len() <= MAX_TARGET)
            .unwrap_or("")
            .to_owned();
        let enabled =
            config["enabled"].as_bool().unwrap_or(false) && endpoint(webhook, &target).is_ok();
        Self {
            enabled,
            webhook,
            target,
            selected: 0,
            input: None,
            status: if enabled {
                "Notifications enabled (fixed summaries only)"
            } else {
                "Notifications OFF. Configure ntfy topic or webhook; d explicitly enables."
            }
            .into(),
            queue: VecDeque::new(),
            pending: None,
            cancel: Arc::new(AtomicBool::new(false)),
            last: None,
            failures: 0,
            history: VecDeque::new(),
            sender,
            persist,
        }
    }
    pub(super) fn editing(&self) -> bool {
        self.input.is_some()
    }
    pub(super) fn pending(&self) -> bool {
        self.pending.is_some() || !self.queue.is_empty()
    }
    fn config(&self) -> Value {
        json!({"enabled":self.enabled,"mode":if self.webhook {"webhook"} else {"ntfy"},"target":self.target})
    }
    fn save(&mut self) {
        if self.persist && network::save_config(SECTION, &self.config()).is_err() {
            self.status = "Settings active for this session, but could not be saved".into();
        }
    }
    pub(super) fn event(&mut self, kind: Event) {
        if !self.enabled {
            return;
        }
        if network::offline() {
            self.status = "Notification skipped: offline mode".into();
            return;
        }
        // At most one pending event of each kind (3 total), even during bursts.
        if !self.queue.contains(&kind) {
            self.queue.push_back(kind);
        }
        self.start_next();
    }
    fn start_next(&mut self) {
        if self.pending.is_some() || !self.enabled {
            return;
        }
        let Some(event) = self.queue.pop_front() else {
            return;
        };
        if network::offline() {
            self.queue.clear();
            self.status = "Notification skipped: offline mode".into();
            return;
        }
        let url = match endpoint(self.webhook, &self.target) {
            Ok(url) => url,
            Err(e) => {
                self.status = e.into();
                return;
            }
        };
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&self.cancel);
        let sender = Arc::clone(&self.sender);
        let webhook = self.webhook;
        let (tx, rx) = mpsc::sync_channel(1);
        self.last = Some(event);
        match std::thread::Builder::new()
            .name("tui-notify".into())
            .spawn(move || {
                let _ = tx.send(sender(&url, webhook, event, &cancel));
            }) {
            Ok(_) => {
                self.pending = Some(rx);
                self.status = format!("Sending {} notification…", event.name());
            }
            Err(_) => self.status = "Cannot start notification worker; r retries".into(),
        }
    }
    pub(super) fn poll(&mut self) {
        let result = match self.pending.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(result)) => result,
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                Err("Notification worker stopped".into())
            }
            _ => return,
        };
        self.pending = None;
        if result.is_err() && !self.cancel.load(Ordering::Relaxed) {
            self.failures = self.failures.saturating_add(1);
        }
        self.status = if self.cancel.load(Ordering::Relaxed) {
            "Notifications OFF or reconfigured; an in-flight request may already have arrived"
                .into()
        } else if result.is_ok() {
            "Notification delivered (fixed summary only)".into()
        } else {
            "Notification failed or timed out; training unaffected. r retries the last event."
                .into()
        };
        if self.history.len() == 8 {
            self.history.pop_front();
        }
        self.history.push_back(format!(
            "{}: {}",
            self.last.map_or("event", Event::name),
            self.status
        ));
        self.start_next();
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Release {
            return;
        }
        if let Some(input) = self.input.as_mut() {
            match key.code {
                KeyCode::Esc => self.input = None,
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    input.clear()
                }
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                        && input.len() + c.len_utf8() <= MAX_TARGET =>
                {
                    input.push(c)
                }
                KeyCode::Enter => {
                    let value = input.trim().to_owned();
                    if self.selected == 0 {
                        match value.as_str() {
                            "ntfy" => self.webhook = false,
                            "webhook" => self.webhook = true,
                            _ => {
                                self.status = "Mode must be ntfy or webhook".into();
                                return;
                            }
                        }
                    } else {
                        self.target = value;
                    }
                    // Editing a target never implicitly authorizes delivery to it.
                    self.enabled = false;
                    self.queue.clear();
                    self.cancel.store(true, Ordering::Relaxed);
                    self.input = None;
                    self.status =
                        "Applied, notifications OFF. d enables after checking destination.".into();
                    self.save();
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.selected = 0,
            KeyCode::Down => self.selected = 1,
            KeyCode::Enter => {
                self.input = Some(if self.selected == 0 {
                    if self.webhook { "webhook" } else { "ntfy" }.into()
                } else {
                    self.target.clone()
                })
            }
            KeyCode::Char('d') => {
                if !self.enabled {
                    if let Err(e) = endpoint(self.webhook, &self.target) {
                        self.status = e.into();
                        return;
                    }
                }
                self.enabled = !self.enabled;
                self.queue.clear();
                if !self.enabled {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                self.status = if self.enabled {
                    "Notifications ON; only fixed event summaries will be sent"
                } else {
                    "Notifications OFF; queued events discarded"
                }
                .into();
                self.save();
            }
            KeyCode::Char('r') => {
                if !self.enabled {
                    self.status = "Notifications are OFF; d enables".into();
                } else if let Some(event) = self.last {
                    self.event(event);
                } else {
                    self.status =
                        "No previous notification to retry; p sends a test checkpoint event".into();
                }
            }
            KeyCode::Char('p') => {
                if self.enabled {
                    self.event(Event::Checkpoint);
                } else {
                    self.status = "Test not sent: notifications are OFF; d enables".into();
                }
            }
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let mode = if self.selected == 0 {
            self.input
                .as_deref()
                .unwrap_or(if self.webhook { "webhook" } else { "ntfy" })
        } else if self.webhook {
            "webhook"
        } else {
            "ntfy"
        };
        let target = if self.selected == 1 {
            self.input.as_deref().unwrap_or(&self.target)
        } else {
            &self.target
        };
        // Both ntfy topics and webhook paths can act as bearer secrets. Never
        // render them, including while editing or in transport errors.
        let mask = if target.is_empty() {
            "(not configured)".into()
        } else {
            "•".repeat(target.chars().count().min(24))
        };
        let mut lines = vec![
            Line::styled("NOTIFICATIONS / explicit opt-in", accent()),
            Line::from(format!(
                "Delivery: {}",
                if self.enabled { "ON" } else { "OFF" }
            )),
            Line::from(format!(
                "{} Mode: {}",
                if self.selected == 0 { "▶" } else { " " },
                network::clean(mode)
            )),
            Line::from(format!(
                "{} {}: {mask}{}",
                if self.selected == 1 { "▶" } else { " " },
                if self.webhook {
                    "Webhook URL"
                } else {
                    "ntfy.sh topic"
                },
                if self.selected == 1 && self.editing() {
                    "▏"
                } else {
                    ""
                }
            )),
            Line::from(self.status.as_str()),
            Line::from(format!(
                "Failed attempts this session: {} (no automatic retry)",
                self.failures
            )),
            Line::from("↑/↓ field • Enter edit/apply • Esc cancel • Ctrl+U clear"),
            Line::from("d enable/disable • p send test checkpoint • r retry last"),
            Line::from(
                "Only checkpoint / finished / died summaries; no logs, paths, prompts or credentials.",
            ),
            Line::from(
                "ntfy topics are public unless protected; use an unguessable topic. Generic webhook: JSON POST.",
            ),
            Line::from(
                "Targets are saved in local TUI config; use a dedicated/revocable webhook URL.",
            ),
        ];
        if !self.history.is_empty() {
            lines.push(Line::styled("DELIVERY LOG / latest attempts", accent()));
            lines.extend(
                self.history
                    .iter()
                    .rev()
                    .take(3)
                    .map(|s| Line::from(s.as_str())),
            );
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" Notifications ")),
            area,
        );
    }
}
impl Drop for Notify {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
fn endpoint(webhook: bool, target: &str) -> Result<String, &'static str> {
    if !webhook {
        if target.is_empty()
            || target.len() > 128
            || !target
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
        {
            return Err("ntfy topic needs 1..128 letters, digits, - or _");
        }
        return Ok(format!("https://ntfy.sh/{target}"));
    }
    if target.len() > MAX_TARGET
        || target.chars().any(|c| c.is_control() || c.is_whitespace())
        || target.contains(['\\', '#'])
    {
        return Err("Invalid webhook URL; use HTTPS without user-info or fragments");
    }
    let rest = target
        .strip_prefix("https://")
        .or_else(|| target.strip_prefix("http://"))
        .ok_or("Webhook requires HTTPS (HTTP allowed only for local fixtures)")?;
    let authority = rest.split(['/', '?']).next().unwrap_or("");
    if authority.is_empty()
        || authority.contains('@')
        || !authority
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':' | b'[' | b']'))
    {
        return Err("Invalid webhook authority; credentials in user-info are forbidden");
    }
    if target.starts_with("http://")
        && !["localhost", "127.0.0.1", "[::1]"].iter().any(|host| {
            authority == *host
                || authority
                    .strip_prefix(host)
                    .is_some_and(|port| port.starts_with(':') && port[1..].parse::<u16>().is_ok())
        })
    {
        return Err("Remote webhooks require HTTPS");
    }
    Ok(target.into())
}
fn send(url: &str, webhook: bool, event: Event, cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("Notification cancelled".into());
    }
    if network::offline() {
        return Err("Notification skipped: offline mode".into());
    }
    deliver(url, webhook, event)
}
fn deliver(url: &str, webhook: bool, event: Event) -> Result<(), String> {
    // No auth source is ever loaded; no redirect may forward a private URL or
    // payload. Do not consume an untrusted/unbounded response body.
    let request = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout(Duration::from_secs(10))
        .timeout_connect(Duration::from_secs(5))
        .build()
        .post(url);
    let result = if webhook {
        request.set("Content-Type", "application/json").send_string(
            &json!({"event":event.name(),"message":event.message(),"source":"pssa"}).to_string(),
        )
    } else {
        request
            .set("Content-Type", "text/plain; charset=utf-8")
            .set("Title", "PSSA training")
            .send_string(event.message())
    };
    match result {
        Ok(response) if (200..300).contains(&response.status()) => Ok(()),
        _ => Err("Notification rejected or connection failed".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delivery_log_retains_failures_without_secrets_and_stays_bounded() {
        let mut notify =
            Notify::with_sender(json!({}), Arc::new(|_, _, _, _| panic!("no network")));
        for i in 0..10 {
            let (tx, rx) = mpsc::sync_channel(1);
            tx.send(if i % 2 == 0 {
                Err("PRIVATE-webhook-error".into())
            } else {
                Ok(())
            })
            .unwrap();
            notify.pending = Some(rx);
            notify.last = Some(Event::Finished);
            notify.poll();
        }
        assert_eq!(notify.failures, 5);
        assert_eq!(notify.history.len(), 8);
        assert!(notify.history.iter().any(|s| s.contains("failed")));
        assert!(notify.history.iter().all(|s| !s.contains("PRIVATE")));
        assert!(notify.history.back().unwrap().contains("delivered"));
    }
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::Mutex,
    };
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    #[test]
    fn default_disabled_event_queue_is_bounded_and_payload_is_fixed() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let copy = Arc::clone(&sent);
        let mut notify = Notify::with_sender(
            Value::Null,
            Arc::new(move |_, _, event, _| {
                copy.lock().unwrap().push(event);
                Ok(())
            }),
        );
        notify.event(Event::Died);
        assert!(notify.pending.is_none());
        assert!(sent.lock().unwrap().is_empty());
        notify.target = "fixture-topic".into();
        notify.enabled = true;
        let (_tx, rx) = mpsc::sync_channel(1);
        notify.pending = Some(rx);
        for _ in 0..100 {
            for event in [Event::Checkpoint, Event::Finished, Event::Died] {
                notify.event(event);
            }
        }
        assert!(notify.queue.len() <= 3);
        notify.key(key(KeyCode::Char('d')));
        assert!(!notify.enabled);
        assert!(notify.queue.is_empty());
        assert_eq!(
            Event::Checkpoint.message(),
            "PSSA: a new checkpoint was saved."
        );
    }
    #[test]
    fn configuration_round_trip_and_invalid_target_stays_off() {
        let notify = Notify::with_sender(
            json!({"enabled":true,"mode":"webhook","target":"https://fixture.test/private-hook"}),
            Arc::new(|_, _, _, _| panic!("construction must not send")),
        );
        let restored = Notify::with_sender(
            notify.config(),
            Arc::new(|_, _, _, _| panic!("construction must not send")),
        );
        assert!(restored.enabled);
        assert!(restored.webhook);
        assert_eq!(restored.target, "https://fixture.test/private-hook");
        assert!(restored.pending.is_none());
        let invalid = Notify::with_sender(
            json!({"enabled":true,"mode":"webhook","target":"https://user:secret@host/hook"}),
            Arc::new(|_, _, _, _| panic!("invalid target must not send")),
        );
        assert!(!invalid.enabled);
    }
    #[test]
    fn worker_failure_redacts_destination_and_preserves_training() {
        let mut notify = Notify::with_sender(
            json!({"enabled":true,"target":"private-topic"}),
            Arc::new(|_, _, _, _| Err("private-topic hf_NEVER echo".into())),
        );
        notify.event(Event::Finished);
        for _ in 0..200 {
            notify.poll();
            if notify.pending.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(notify.pending.is_none());
        assert!(!notify.status.contains("private-topic"));
        assert!(!notify.status.contains("hf_NEVER"));
        if !network::offline() {
            assert!(notify.status.contains("failed"));
        }
    }
    #[test]
    fn edit_requires_reenable_and_cancels_without_changing_saved_value() {
        let mut notify = Notify::with_sender(
            json!({"enabled":true,"target":"topic"}),
            Arc::new(|_, _, _, _| panic!("no send")),
        );
        notify.key(key(KeyCode::Down));
        notify.key(key(KeyCode::Enter));
        notify.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        notify.key(key(KeyCode::Char('x')));
        notify.key(key(KeyCode::Esc));
        assert_eq!(notify.target, "topic");
        assert!(notify.enabled);
        notify.key(key(KeyCode::Enter));
        notify.key(key(KeyCode::Backspace));
        notify.key(key(KeyCode::Enter));
        assert_eq!(notify.target, "topi");
        assert!(!notify.enabled);
        for invalid in [
            "http://evil.test/hook",
            "https://user:secret@host/hook",
            "https://host/path\n",
            "https://host/path#secret",
        ] {
            assert!(endpoint(true, invalid).is_err());
        }
        assert!(endpoint(true, "http://127.0.0.1:321/hook").is_ok());
        assert!(endpoint(false, "private-topic").is_ok());
    }
    #[test]
    fn local_webhook_receives_summary_only_and_redirect_is_not_followed() {
        for status in ["200 OK", "302 Found"] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let worker = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buf = [0; 1024];
                loop {
                    let n = stream.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..pos]);
                        let len = head
                            .lines()
                            .find_map(|s| {
                                s.to_ascii_lowercase()
                                    .strip_prefix("content-length: ")
                                    .and_then(|s| s.parse::<usize>().ok())
                            })
                            .unwrap();
                        if bytes.len() >= pos + 4 + len {
                            break;
                        }
                    }
                    assert!(bytes.len() < 8192);
                }
                let request = String::from_utf8(bytes).unwrap();
                assert!(!request.to_ascii_lowercase().contains("authorization:"));
                let body = request.split_once("\r\n\r\n").unwrap().1;
                let value: Value = serde_json::from_str(body).unwrap();
                assert_eq!(value["event"], "finished");
                assert_eq!(value.as_object().unwrap().len(), 3);
                write!(stream,"HTTP/1.1 {status}\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            });
            let result = deliver(
                &format!("http://{addr}/private-webhook"),
                true,
                Event::Finished,
            );
            worker.join().unwrap();
            assert_eq!(result.is_ok(), status == "200 OK");
        }
    }
    #[test]
    fn notification_screen_masks_secret_wide_narrow_tiny() {
        for (w, h) in [(120, 24), (79, 18), (24, 8), (4, 2), (1, 1)] {
            let mut notify = Notify::with_sender(
                json!({"mode":"webhook","target":"https://host/NEVER_SHOW_SECRET"}),
                Arc::new(|_, _, _, _| panic!("no send")),
            );
            notify.selected = 1;
            notify.key(key(KeyCode::Enter));
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| notify.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(!text.contains("NEVER_SHOW_SECRET"));
            if w >= 79 {
                assert!(text.contains("OFF"));
                assert!(text.contains("••••••••"));
            }
        }
    }
}
