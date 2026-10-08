//! Independent cloud/synced-file log viewer, never a replacement for the main
//! training monitor. One cancellable worker streams into an eight-line queue;
//! UI history, line size, work per tick and retry delays are bounded.
use super::{accent, network, panel, panel_area};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    collections::VecDeque,
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
const MAX_SOURCE: usize = 4096;
const MAX_LINE: usize = 8192;
const HISTORY: usize = 400;
const MAX_SKIP: u64 = 8 * 1024 * 1024;
const MAX_TICK: usize = 64;

#[derive(Clone, Default)]
pub(super) struct Cursor {
    pub offset: u64,
    pub file_id: Option<(u64, u64)>,
}
/// A fixture can supply any bounded Read implementation. `offset` is the byte
/// position represented by the first new byte (0 signals rotation/reset), while
/// `skip` discards a replayed HTTP prefix when the server ignores Range.
pub(super) struct Opened {
    pub reader: Box<dyn Read + Send>,
    pub offset: u64,
    pub skip: u64,
    pub file_id: Option<(u64, u64)>,
}
pub(super) type Opener = Arc<dyn Fn(&str, &Cursor) -> Result<Opened, String> + Send + Sync>;
enum Message {
    Line(String),
    Status(&'static str),
}
struct Worker {
    rx: mpsc::Receiver<Message>,
    cancel: Arc<AtomicBool>,
}

pub(super) struct LogStream {
    source: String,
    input: Option<String>,
    status: String,
    history: VecDeque<String>,
    scroll: usize,
    worker: Option<Worker>,
    restart: bool,
    opener: Opener,
}
impl Default for LogStream {
    fn default() -> Self {
        Self::with_opener(Arc::new(open))
    }
}
impl LogStream {
    pub(super) fn with_opener(opener: Opener) -> Self {
        Self { source: String::new(), input: None,
            status: "Enter an HTTPS log URL or local synced file; p follows. No connection until requested.".into(),
            history: VecDeque::new(), scroll: 0, worker: None, restart: false, opener }
    }
    pub(super) fn editing(&self) -> bool {
        self.input.is_some()
    }
    fn start(&mut self) {
        if self.worker.is_some() {
            return;
        }
        if let Err(e) = validate_source(&self.source) {
            self.status = e.into();
            return;
        }
        if remote(&self.source) && network::offline() {
            self.status = "Cloud log skipped: offline mode; local files still work".into();
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let opener = Arc::clone(&self.opener);
        let source = self.source.clone();
        let (tx, rx) = mpsc::sync_channel(8);
        match std::thread::Builder::new()
            .name("cloud-log-follow".into())
            .spawn(move || follow(&source, &opener, &flag, &tx))
        {
            Ok(_) => {
                self.worker = Some(Worker { rx, cancel });
                self.history.clear();
                self.scroll = 0;
                self.status = "Following log on a worker…".into();
            }
            Err(_) => self.status = "Cannot start log worker; r retries".into(),
        }
    }
    fn stop(&mut self) {
        self.restart = false;
        if let Some(worker) = &self.worker {
            worker.cancel.store(true, Ordering::Relaxed);
        }
    }
    /// Newly completed, cleaned lines only. The shell may ignore the return
    /// value: this view owns its history and must not reset the training monitor.
    pub(super) fn poll(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        let mut ended = false;
        if let Some(worker) = &self.worker {
            for _ in 0..MAX_TICK {
                match worker.rx.try_recv() {
                    Ok(Message::Line(line)) if !worker.cancel.load(Ordering::Relaxed) => {
                        if self.history.len() == HISTORY {
                            self.history.pop_front();
                        }
                        self.history.push_back(line.clone());
                        lines.push(line);
                    }
                    Ok(Message::Status(status)) if !worker.cancel.load(Ordering::Relaxed) => {
                        self.status = status.into()
                    }
                    Ok(_) => {}
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        ended = true;
                        break;
                    }
                }
            }
        }
        if ended {
            if self
                .worker
                .as_ref()
                .is_some_and(|worker| !worker.cancel.load(Ordering::Relaxed))
            {
                self.status = "Log reader stopped; r restarts".into();
            }
            self.worker = None;
            if std::mem::take(&mut self.restart) {
                self.start();
            }
        }
        lines
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
                        && input.len() + c.len_utf8() <= MAX_SOURCE =>
                {
                    input.push(c)
                }
                KeyCode::Enter => {
                    let value = input.trim().to_owned();
                    if let Err(e) = validate_source(&value) {
                        self.status = e.into();
                        return;
                    }
                    self.stop();
                    self.source = value;
                    self.input = None;
                    self.status =
                        "Source applied; p starts following (source stays in memory only)".into();
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Enter => self.input = Some(self.source.clone()),
            // There is one editable field; Up/Down consistently select it.
            KeyCode::Up | KeyCode::Down => {}
            KeyCode::Char('p') => {
                if self
                    .worker
                    .as_ref()
                    .is_some_and(|w| w.cancel.load(Ordering::Relaxed))
                {
                    self.restart = true;
                    self.status = "Waiting for the old reader to stop before restarting…".into();
                } else {
                    self.start();
                }
            }
            KeyCode::Char('r') => {
                self.stop();
                if self.worker.is_some() {
                    self.restart = true;
                    self.status = "Refreshing after the previous reader stops…".into();
                } else {
                    self.start();
                }
            }
            KeyCode::Char('d') => {
                self.stop();
                self.status = "Follow paused; p restarts from the beginning".into();
            }
            KeyCode::PageUp => {
                self.scroll = (self.scroll + 10).min(self.history.len().saturating_sub(1))
            }
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::End => self.scroll = 0,
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let block = panel(" Cloud log stream ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.is_empty() {
            return;
        }
        let value = self.input.as_deref().unwrap_or(&self.source);
        let shown = if remote(value) {
            "[URL hidden: may contain a private access query]".into()
        } else {
            network::clean(value)
        };
        let lines = vec![
            Line::styled("CLOUD LOG / separate live view", accent()),
            Line::from(format!(
                "▶ Source: {shown}{}",
                if self.editing() { "▏" } else { "" }
            )),
            Line::from(self.status.as_str()),
            Line::from(
                "Enter edit/apply • Esc cancel • Ctrl+U clear • p follow • r refresh • d pause",
            ),
            Line::from("PgUp/PgDn history • End live • green checkpoint / cyan throughput"),
        ];
        let header = Paragraph::new(lines).wrap(Wrap { trim: false });
        let header_height = header.line_count(inner.width).min(inner.height as usize) as u16;
        let rows = Layout::vertical([Constraint::Length(header_height), Constraint::Min(0)])
            .split(inner);
        f.render_widget(header, rows[0]);
        if rows[1].is_empty() {
            return;
        }
        let available = rows[1].height as usize;
        let end = self.history.len().saturating_sub(self.scroll);
        let mut lines = Vec::new();
        let mut wrapped_height = 0;
        // Select from the tail by rendered rows, not by logical line count.
        for line in self.history.iter().take(end).rev() {
            let line = Line::styled(line.as_str(), line_style(line));
            wrapped_height += Paragraph::new(line.clone())
                .wrap(Wrap { trim: false })
                .line_count(rows[1].width);
            lines.push(line);
            if wrapped_height >= available {
                break;
            }
        }
        lines.reverse();
        if self.history.is_empty() {
            lines.push(Line::from(if self.source.is_empty() {
                "No cloud log source configured. Local training data is in Monitor and Feed."
            } else {
                "Waiting for complete lines. Partial lines survive append/reconnect."
            }));
        }
        let scroll = wrapped_height.saturating_sub(available).min(u16::MAX as usize) as u16;
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((scroll, 0)),
            rows[1],
        );
    }
}
impl Drop for LogStream {
    fn drop(&mut self) {
        self.stop();
    }
}
fn line_style(line: &str) -> Style {
    if line.contains("saved_checkpoint=") || line.contains("checkpoint written to ") {
        accent().add_modifier(Modifier::BOLD)
    } else if line.contains("tokens_per_second=")
        || line.contains("tok/s")
        || line.contains("tokens/s")
    {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}
fn remote(source: &str) -> bool {
    source.starts_with("https://") || source.starts_with("http://")
}
fn validate_source(source: &str) -> Result<(), &'static str> {
    if source.is_empty() || source.len() > MAX_SOURCE || source.chars().any(char::is_control) {
        return Err("Enter a log URL or a regular local file path");
    }
    if remote(source) {
        if source.contains(['\\', '#']) || source.chars().any(char::is_whitespace) {
            return Err("Invalid log URL; use a URL without user-info or fragments");
        }
        let authority = source
            .split_once("://")
            .unwrap()
            .1
            .split(['/', '?'])
            .next()
            .unwrap_or("");
        if authority.is_empty()
            || authority.contains('@')
            || !authority
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':' | b'[' | b']'))
        {
            return Err("Invalid log URL authority; credentials in user-info are forbidden");
        }
        if source.starts_with("http://")
            && !["localhost", "127.0.0.1", "[::1]"].iter().any(|host| {
                authority == *host
                    || authority.strip_prefix(host).is_some_and(|port| {
                        port.starts_with(':') && port[1..].parse::<u16>().is_ok()
                    })
            })
        {
            return Err("Remote logs require HTTPS (HTTP only for localhost fixtures)");
        }
    } else if source.contains("://") {
        return Err("Only HTTPS URLs and local files are supported");
    }
    Ok(())
}
fn open(source: &str, cursor: &Cursor) -> Result<Opened, String> {
    validate_source(source).map_err(str::to_owned)?;
    if remote(source) {
        open_http(source, cursor)
    } else {
        open_file(source, cursor)
    }
}
fn open_file(source: &str, cursor: &Cursor) -> Result<Opened, String> {
    let meta = std::fs::symlink_metadata(source).map_err(|_| "Cannot inspect local log")?;
    if !meta.is_file() {
        return Err("Synced log must be a regular file (not a device, FIFO or symlink)".into());
    }
    let mut file = File::open(source).map_err(|_| "Cannot open local log")?;
    let meta = file
        .metadata()
        .map_err(|_| "Cannot inspect opened local log")?;
    if !meta.is_file() {
        return Err("Log is not a regular file".into());
    }
    #[cfg(unix)]
    let file_id = {
        use std::os::unix::fs::MetadataExt;
        Some((meta.dev(), meta.ino()))
    };
    #[cfg(not(unix))]
    let file_id = None;
    let offset =
        if meta.len() < cursor.offset || cursor.file_id.is_some() && cursor.file_id != file_id {
            0
        } else {
            cursor.offset
        };
    file.seek(SeekFrom::Start(offset))
        .map_err(|_| "Cannot seek synced log")?;
    Ok(Opened {
        reader: Box::new(file),
        offset,
        skip: 0,
        file_id,
    })
}
fn open_http(source: &str, cursor: &Cursor) -> Result<Opened, String> {
    let result = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout(Duration::from_secs(20))
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(5))
        .build()
        .get(source)
        .set("Accept-Encoding", "identity")
        .set("Range", &format!("bytes={}-", cursor.offset))
        .call();
    let response = match result {
        Ok(r) => r,
        Err(ureq::Error::Status(416, r)) => {
            let total = r
                .header("Content-Range")
                .and_then(|s| s.strip_prefix("bytes */"))
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or("Invalid range response")?;
            if total > cursor.offset {
                return Err("Server rejected an available log range".into());
            }
            return Ok(Opened {
                reader: Box::new(std::io::empty()),
                offset: if total < cursor.offset {
                    0
                } else {
                    cursor.offset
                },
                skip: 0,
                file_id: None,
            });
        }
        Err(_) => return Err("Cannot connect to log endpoint".into()),
    };
    if response
        .header("Content-Encoding")
        .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
    {
        return Err(
            "Log endpoint must return uncompressed bytes for reliable range cursors".into(),
        );
    }
    let (offset, skip) = match response.status() {
        206 => {
            let start = response
                .header("Content-Range")
                .and_then(|s| s.strip_prefix("bytes "))
                .and_then(|s| s.split_once('-'))
                .and_then(|(s, _)| s.parse::<u64>().ok())
                .ok_or("Invalid log Content-Range")?;
            if start != cursor.offset {
                return Err("Log range does not match requested cursor".into());
            }
            (start, 0)
        }
        200 => {
            if cursor.offset > MAX_SKIP {
                return Err(
                    "Log server ignores Range; use a local synced file for logs above 8 MiB".into(),
                );
            }
            (cursor.offset, cursor.offset)
        }
        _ => return Err("Log redirect refused or unsupported HTTP response".into()),
    };
    Ok(Opened {
        reader: response.into_reader(),
        offset,
        skip,
        file_id: None,
    })
}
fn sleep(cancel: &AtomicBool, duration: Duration) {
    let end = Instant::now() + duration;
    while !cancel.load(Ordering::Relaxed) && Instant::now() < end {
        std::thread::sleep(
            Duration::from_millis(25).min(end.saturating_duration_since(Instant::now())),
        );
    }
}
fn emit(tx: &mpsc::SyncSender<Message>, cancel: &AtomicBool, mut message: Message) -> bool {
    while !cancel.load(Ordering::Relaxed) {
        match tx.try_send(message) {
            Ok(()) => return true,
            Err(mpsc::TrySendError::Disconnected(_)) => return false,
            Err(mpsc::TrySendError::Full(back)) => {
                message = back;
                sleep(cancel, Duration::from_millis(10));
            }
        }
    }
    false
}
#[derive(Default)]
struct Lines {
    partial: Vec<u8>,
    truncated: bool,
}
impl Lines {
    fn feed(&mut self, bytes: &[u8], mut line: impl FnMut(String) -> bool) -> bool {
        for &byte in bytes {
            if byte == b'\n' {
                let raw = String::from_utf8_lossy(&self.partial);
                let mut text = network::clean(raw.trim_end_matches('\r'));
                if self.truncated {
                    text.push_str(" … [line truncated]");
                }
                self.partial.clear();
                self.truncated = false;
                if !line(text) {
                    return false;
                }
            } else if self.partial.len() < MAX_LINE {
                self.partial.push(byte);
            } else {
                self.truncated = true;
            }
        }
        true
    }
}
fn follow(source: &str, opener: &Opener, cancel: &AtomicBool, tx: &mpsc::SyncSender<Message>) {
    let mut cursor = Cursor::default();
    let mut lines = Lines::default();
    let mut failures = 0u32;
    while !cancel.load(Ordering::Relaxed) {
        if remote(source) && network::offline() {
            let _ = tx.try_send(Message::Status(
                "Cloud follow paused: offline mode; retrying without network",
            ));
            sleep(cancel, Duration::from_secs(2));
            continue;
        }
        let result = (|| -> Result<bool, String> {
            let mut opened = opener(source, &cursor)?;
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            if opened.offset != cursor.offset {
                lines = Lines::default();
                cursor.offset = opened.offset;
                let _ = tx.try_send(Message::Status(
                    "Log rotated/truncated; following from the beginning",
                ));
            }
            cursor.file_id = opened.file_id;
            let mut buffer = [0u8; 8192];
            let mut skip = opened.skip;
            while skip > 0 {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(false);
                }
                let size = skip.min(buffer.len() as u64) as usize;
                let n = opened
                    .reader
                    .read(&mut buffer[..size])
                    .map_err(|_| "Log replay read failed")?;
                if n == 0 {
                    cursor.offset = 0;
                    lines = Lines::default();
                    return Ok(false);
                }
                skip -= n as u64;
            }
            let mut progressed = false;
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(progressed);
                }
                if remote(source) && network::offline() {
                    return Err("Offline".into());
                }
                let n = opened
                    .reader
                    .read(&mut buffer)
                    .map_err(|_| "Log read disconnected")?;
                if n == 0 {
                    break;
                }
                progressed = true;
                failures = 0;
                cursor.offset = cursor.offset.saturating_add(n as u64);
                if !lines.feed(&buffer[..n], |line| emit(tx, cancel, Message::Line(line))) {
                    return Ok(progressed);
                }
            }
            Ok(progressed)
        })();
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        match result {
            Ok(progressed) => {
                failures = 0;
                let _ = tx.try_send(Message::Status(if progressed {
                    "Following live log; waiting for more lines"
                } else {
                    "Connected; waiting for append (partial lines retained)"
                }));
                sleep(
                    cancel,
                    if remote(source) {
                        Duration::from_millis(1500)
                    } else {
                        Duration::from_millis(300)
                    },
                );
            }
            Err(_) => {
                // Raw errors/URLs can contain access query strings and are never
                // displayed. Cursor/partial bytes survive reconnects.
                let _=tx.try_send(Message::Status("Log unavailable/disconnected; retrying (1–30s backoff). Check URL/file; large HTTP logs need Range."));
                let delay = backoff(failures);
                failures = failures.saturating_add(1);
                sleep(cancel, delay);
            }
        }
    }
}
fn backoff(failures: u32) -> Duration {
    Duration::from_secs((1u64 << failures.min(5)).min(30))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Cursor as Reader, Write},
        sync::Mutex,
    };
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    #[test]
    fn partial_utf8_crlf_and_oversized_lines_survive_reconnect() {
        let mut parser = Lines::default();
        let mut output = Vec::new();
        assert!(parser.feed(b"saved_checkpoint=some ", |s| {
            output.push(s);
            true
        }));
        assert!(output.is_empty());
        parser.feed(&[0xe2, 0x82], |s| {
            output.push(s);
            true
        });
        parser.feed(&[0xac, b'\r', b'\n'], |s| {
            output.push(s);
            true
        });
        assert_eq!(output, ["saved_checkpoint=some €"]);
        parser.feed(&vec![b'x'; MAX_LINE * 3], |_| true);
        assert_eq!(parser.partial.len(), MAX_LINE);
        parser.feed(b"\nok\n", |s| {
            output.push(s);
            true
        });
        assert!(output[1].ends_with("[line truncated]"));
        assert_eq!(output[2], "ok");
        assert_eq!(backoff(0), Duration::from_secs(1));
        assert_eq!(backoff(100), Duration::from_secs(30));
        assert_ne!(
            line_style("saved_checkpoint=model.pssa"),
            line_style("ordinary")
        );
        assert_ne!(line_style("tokens_per_second=42"), line_style("ordinary"));
    }
    #[test]
    fn synced_file_appends_truncates_and_rotates_without_replaying() {
        let dir = std::env::temp_dir().join(format!("pssa-log-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("log with spaces.txt");
        std::fs::write(&path, b"first\npart").unwrap();
        let mut cursor = Cursor::default();
        let mut first = open_file(path.to_str().unwrap(), &cursor).unwrap();
        let mut text = String::new();
        first.reader.read_to_string(&mut text).unwrap();
        assert_eq!(text, "first\npart");
        cursor.offset = text.len() as u64;
        cursor.file_id = first.file_id;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"ial\n")
            .unwrap();
        let mut next = open_file(path.to_str().unwrap(), &cursor).unwrap();
        text.clear();
        next.reader.read_to_string(&mut text).unwrap();
        assert_eq!(text, "ial\n");
        cursor.offset += text.len() as u64;
        std::fs::write(&path, b"new\n").unwrap();
        assert_eq!(
            open_file(path.to_str().unwrap(), &cursor).unwrap().offset,
            0
        );
        #[cfg(unix)]
        {
            std::fs::rename(&path, dir.join("old")).unwrap();
            std::fs::write(&path, b"replacement longer than the old file\n").unwrap();
            assert_eq!(
                open_file(path.to_str().unwrap(), &cursor).unwrap().offset,
                0
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn worker_follows_fixture_and_ui_history_is_bounded() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let copy = Arc::clone(&calls);
        let opener: Opener = Arc::new(move |_, cursor| {
            copy.lock().unwrap().push(cursor.offset);
            let bytes = if cursor.offset == 0 {
                b"first\npartial".to_vec()
            } else if cursor.offset == 13 {
                b" line\n".to_vec()
            } else {
                Vec::new()
            };
            Ok(Opened {
                reader: Box::new(Reader::new(bytes)),
                offset: cursor.offset,
                skip: 0,
                file_id: None,
            })
        });
        let mut view = LogStream::with_opener(opener);
        view.source = "fixture-file".into();
        assert!(view.worker.is_none());
        view.key(key(KeyCode::Char('p')));
        let mut received = Vec::new();
        for _ in 0..300 {
            received.extend(view.poll());
            if received.len() >= 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(received, ["first", "partial line"]);
        view.key(key(KeyCode::Char('d')));
        assert!(view.worker.as_ref().unwrap().cancel.load(Ordering::Relaxed));
        assert!(!calls.lock().unwrap().is_empty());
        let (tx, rx) = mpsc::sync_channel(HISTORY + 100);
        view.worker = Some(Worker {
            rx,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        for i in 0..HISTORY + 100 {
            tx.send(Message::Line(i.to_string())).unwrap();
        }
        for _ in 0..10 {
            view.poll();
        }
        assert_eq!(view.history.len(), HISTORY);
        assert_eq!(view.history.front().unwrap(), "100");
    }
    #[test]
    fn local_http_range_no_credentials_and_redirect_refusal() {
        use std::net::TcpListener;
        for redirect in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let worker = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buf = [0; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = stream.read(&mut buf).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&buf[..n]);
                    assert!(request.len() < 8192);
                }
                let request = String::from_utf8(request).unwrap().to_ascii_lowercase();
                assert!(request.contains("range: bytes=5-"));
                assert!(!request.contains("authorization:"));
                let response = if redirect {
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                } else {
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-8/9\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnew\n"
                };
                stream.write_all(response.as_bytes()).unwrap();
            });
            let result = open_http(
                &format!("http://{addr}/private-query?token=NEVER"),
                &Cursor {
                    offset: 5,
                    file_id: None,
                },
            );
            worker.join().unwrap();
            if redirect {
                assert!(result.is_err());
            } else {
                let mut opened = result.unwrap();
                let mut text = String::new();
                opened.reader.read_to_string(&mut text).unwrap();
                assert_eq!(text, "new\n");
                assert_eq!(opened.offset, 5);
            }
        }
    }
    #[test]
    fn live_and_end_show_the_newest_wrapped_log_output() {
        for (width, height) in [(120, 40), (80, 24)] {
            let mut view = LogStream::with_opener(Arc::new(|_, _| panic!("draw must not open")));
            for i in 0..40 {
                view.history.push_back(format!("old-{i} {}", "x".repeat(320)));
            }
            view.history
                .push_back(format!("{} NEWEST_ENTRY", "y".repeat(320)));
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            let mut render = |view: &LogStream| -> String {
                terminal.draw(|f| view.draw(f, f.area())).unwrap();
                terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect()
            };
            let live = render(&view);
            assert!(live.contains("NEWEST_ENTRY"), "live at {width}x{height}");
            assert!(live.contains("CLOUD LOG / separate live view"));
            assert!(live.contains("▶ Source:"));
            view.key(key(KeyCode::PageUp));
            assert!(!render(&view).contains("NEWEST_ENTRY"));
            view.key(key(KeyCode::End));
            assert!(
                render(&view).contains("NEWEST_ENTRY"),
                "End at {width}x{height}"
            );

            // A newly arrived line can itself be taller than the entire viewport.
            let (tx, rx) = mpsc::sync_channel(1);
            view.worker = Some(Worker {
                rx,
                cancel: Arc::new(AtomicBool::new(false)),
            });
            tx.send(Message::Line(format!(
                "{} APPENDED_ENTRY",
                "界 ".repeat(1024)
            )))
            .unwrap();
            view.poll();
            assert!(
                render(&view).contains("APPENDED_ENTRY"),
                "append at {width}x{height}"
            );
        }
    }

    #[test]
    fn editing_validation_and_render_wide_narrow_tiny() {
        let mut view = LogStream::with_opener(Arc::new(|_, _| panic!("draw/edit must not open")));
        view.key(key(KeyCode::Enter));
        for c in "my log.txt".chars() {
            view.key(key(KeyCode::Char(c)));
        }
        view.key(key(KeyCode::Enter));
        assert_eq!(view.source, "my log.txt");
        view.key(key(KeyCode::Enter));
        view.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        view.key(key(KeyCode::Esc));
        assert_eq!(view.source, "my log.txt");
        for bad in [
            "ftp://host/log",
            "http://remote.test/log",
            "https://user:secret@host/log",
            "https://host/path\n",
        ] {
            assert!(validate_source(bad).is_err());
        }
        view.source = "https://host/log?token=NEVER_SHOW".into();
        view.history
            .push_back("saved_checkpoint=fixture.pssa".into());
        view.history.push_back("tokens_per_second=42".into());
        for (w, h) in [(120, 24), (79, 18), (24, 8), (4, 2), (1, 1)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| view.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(!text.contains("NEVER_SHOW"));
            if w >= 79 {
                assert!(text.contains("fixture.pssa"));
                assert!(text.contains("tokens_per_second=42"));
            }
        }
    }
}
