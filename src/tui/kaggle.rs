//! Kaggle notebook/script push + version-pinned execution logs (no Kaggle CLI).
//!
//! Integration: call `poll()` on every UI tick, even while this tab is hidden.
//! Drain `take_event()`: apply `Started` (reset RunState) BEFORE ingesting the
//! returned lines, and apply `Done`/`Error`/`Detached` AFTER ingesting them.
//! `Started` means Kaggle accepted a new version, not that a GPU is allocated.
//! `Error` may describe a failed launch OR lost monitoring; it never implies
//! the remote run was cancelled. Esc/Drop only detach locally.
//!
//! Protocol follows Kaggle's official kaggle-cli kernels_push/kernels_logs_stream
//! and kagglesdk kernels.KernelsApiService (SaveKernel/GetKernelSessionStatus).
//! See https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
//! and src/kaggle/api/kaggle_api_extended.py in that repository. Live SSE may
//! be unavailable on older deployments; an explicit error is shown, not fake
//! progress. No remote service access is needed by the tests below.
use super::{accent, panel, process::clean};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    fs,
    io::{BufRead, BufReader, Read},
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    },
    time::{Duration, Instant},
};

const API: &str = "https://api.kaggle.com/v1";
const MAX_CODE: usize = 4 * 1024 * 1024;
const MAX_JSON: usize = 256 * 1024;
const MAX_LOG: usize = 16 * 1024 * 1024;
const MAX_EVENT: usize = 256 * 1024;
const MAX_LINE: usize = 16 * 1024;
const QUEUE: usize = 64;
const HISTORY: usize = 120;
const INTERVAL: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

// Secrets have no Debug implementation and never leave the worker/confirmation
// snapshot. Raw HTTP errors/bodies are deliberately not interpolated in errors.
struct Credentials {
    username: String,
    key: String,
    authorization: String,
}
impl Credentials {
    fn parse(username: String, key: String) -> Result<Self, String> {
        if !slug_part(&username)
            || key.is_empty()
            || key.len() > 8192
            || !key.bytes().all(|b| b.is_ascii_graphic())
        {
            return Err("Invalid Kaggle username/key; check the credential source".into());
        }
        let authorization = format!("Basic {}", base64(format!("{username}:{key}").as_bytes()));
        Ok(Self {
            username,
            key,
            authorization,
        })
    }
    fn redact(&self, text: &str) -> String {
        // Clean first too: ANSI/control characters must not conceal a key from
        // redaction and then be stripped into an exposed credential.
        let text = clean(text);
        text.replace(&self.authorization, "[redacted]")
            .replace(
                self.authorization.trim_start_matches("Basic "),
                "[redacted]",
            )
            .replace(&self.key, "[redacted]")
    }
}
fn base64(bytes: &[u8]) -> String {
    // RFC 4648; avoid a new dependency just for Basic authentication.
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = *chunk.get(1).unwrap_or(&0);
        let c = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(a >> 2) as usize] as char);
        out.push(TABLE[(((a & 3) << 4) | (b >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(((b & 15) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
fn bounded_read(input: impl Read, limit: usize) -> Result<String, String> {
    let mut body = String::new();
    input
        .take(limit as u64 + 1)
        .read_to_string(&mut body)
        .map_err(|_| "Cannot read bounded UTF-8 content".to_owned())?;
    if body.len() > limit {
        return Err("Content exceeds the safe size limit".into());
    }
    Ok(body)
}
fn file_text(path: &Path, limit: usize) -> Result<String, String> {
    // Reject devices/FIFOs before opening; all filesystem work is off UI thread.
    let meta = fs::metadata(path).map_err(|_| "Cannot read required file")?;
    if !meta.is_file() || meta.len() > limit as u64 {
        return Err("Required file must be regular and within the safe size limit".into());
    }
    let file = fs::File::open(path).map_err(|_| "Cannot open required file")?;
    bounded_read(file, limit)
}
fn resolve_credentials(
    username: Option<String>,
    key: Option<String>,
    path: Option<&Path>,
) -> Result<Credentials, String> {
    match (username, key) {
        (Some(username), Some(key)) => Credentials::parse(username, key),
        (None, None) => {
            let path = path.ok_or("Set KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json")?;
            let text = file_text(path, 32 * 1024).map_err(
                |_| "Cannot read ~/.kaggle/kaggle.json; set KAGGLE_USERNAME + KAGGLE_KEY",
            )?;
            let value: Value =
                serde_json::from_str(&text).map_err(|_| "Invalid Kaggle credential JSON")?;
            Credentials::parse(
                value["username"]
                    .as_str()
                    .ok_or("Credential file needs username and key")?
                    .into(),
                value["key"]
                    .as_str()
                    .ok_or("Credential file needs username and key")?
                    .into(),
            )
        }
        _ => {
            Err("Set BOTH KAGGLE_USERNAME and KAGGLE_KEY, or unset both to use kaggle.json".into())
        }
    }
}
fn credentials() -> Result<Credentials, String> {
    fn env(name: &str) -> Result<Option<String>, String> {
        match std::env::var(name) {
            Ok(s) => Ok(Some(s)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(_) => Err("Kaggle credential environment must be valid UTF-8".into()),
        }
    }
    let path = std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".kaggle/kaggle.json"));
    resolve_credentials(env("KAGGLE_USERNAME")?, env("KAGGLE_KEY")?, path.as_deref())
}
fn slug_part(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
fn reference(s: &str) -> bool {
    s.split_once('/')
        .is_some_and(|(owner, slug)| slug_part(owner) && slug_part(slug))
}
fn boolean(meta: &Value, field: &str, default: bool) -> Result<bool, String> {
    match meta.get(field) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("true") => Ok(true),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("false") => Ok(false),
        _ => Err(format!("Metadata {field} must be true or false")),
    }
}
struct Prepared {
    credentials: Credentials,
    payload: Value,
    slug: String,
    summary: String,
}
fn prepare(folder: &Path, credentials: Credentials) -> Result<Prepared, String> {
    let folder = folder
        .canonicalize()
        .map_err(|_| "Cannot open notebook folder")?;
    let meta: Value =
        serde_json::from_str(&file_text(&folder.join("kernel-metadata.json"), MAX_JSON)?)
            .map_err(|_| "Invalid kernel-metadata.json")?;
    let slug = meta["id"]
        .as_str()
        .filter(|s| reference(s))
        .ok_or("Metadata id must be owner/notebook-slug (no version)")?;
    if !slug
        .split_once('/')
        .unwrap()
        .0
        .eq_ignore_ascii_case(&credentials.username)
    {
        return Err("Metadata owner must match the authenticated Kaggle username".into());
    }
    // Numeric id takes precedence on Kaggle and could target a different kernel.
    if meta.get("id_no").is_some_and(|v| !v.is_null()) {
        return Err("Remove id_no; this tab targets the explicit owner/slug only".into());
    }
    let title = meta["title"]
        .as_str()
        .filter(|s| (5..=100).contains(&s.len()) && !s.chars().any(char::is_control))
        .ok_or("Metadata title must contain 5–100 bytes without control characters")?;
    let language = meta["language"]
        .as_str()
        .filter(|s| ["python", "r", "rmarkdown"].contains(s))
        .ok_or("Metadata language must be python, r, or rmarkdown")?;
    let kind = meta["kernel_type"]
        .as_str()
        .filter(|s| ["notebook", "script"].contains(s))
        .ok_or("Metadata kernel_type must be notebook or script")?;
    let code = meta["code_file"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("Metadata needs code_file")?;
    let relative = Path::new(code);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err("code_file must be relative and remain inside the notebook folder".into());
    }
    let source_path = folder
        .join(relative)
        .canonicalize()
        .map_err(|_| "Cannot locate code_file")?;
    if !source_path.starts_with(&folder) {
        return Err("code_file symlink escapes the notebook folder".into());
    }
    let mut text = file_text(&source_path, MAX_CODE)?;
    if kind == "notebook" {
        let mut notebook: Value =
            serde_json::from_str(&text).map_err(|_| "code_file is not notebook JSON")?;
        let cells = notebook
            .get_mut("cells")
            .and_then(Value::as_array_mut)
            .ok_or("Notebook must have a cells array")?;
        for cell in cells {
            if !cell.is_object() {
                return Err("Notebook cells must be JSON objects".into());
            }
            if cell["cell_type"] == "code" {
                cell["outputs"] = json!([]);
                cell["execution_count"] = Value::Null;
            }
            if let Some(parts) = cell["source"].as_array() {
                let mut source = String::new();
                for part in parts {
                    source.push_str(part.as_str().ok_or("Invalid notebook cell source")?);
                }
                cell["source"] = Value::String(source);
            }
        }
        text = notebook.to_string();
    }
    let private = boolean(&meta, "is_private", true)?;
    let gpu = boolean(&meta, "enable_gpu", false)?;
    let tpu = boolean(&meta, "enable_tpu", false)?;
    let internet = boolean(&meta, "enable_internet", false)?;
    if gpu && tpu {
        return Err("Choose GPU or TPU, not both".into());
    }
    let mut payload = json!({
        "slug": slug, "newTitle": title, "text": text,
        "language": if kind == "notebook" && language == "rmarkdown" { "r" } else { language },
        "kernelType": kind, "isPrivate": private, "enableGpu": gpu, "enableTpu": tpu,
        "enableInternet": internet, "kernelExecutionType": "SAVE_AND_RUN_ALL",
    });
    for (input, output) in [
        ("dataset_sources", "datasetDataSources"),
        ("kernel_sources", "kernelDataSources"),
        ("competition_sources", "competitionDataSources"),
        ("model_sources", "modelDataSources"),
        ("keywords", "categoryIds"),
    ] {
        let sources = meta.get(input).cloned().unwrap_or_else(|| json!([]));
        let array = sources
            .as_array()
            .ok_or("Metadata source lists must be arrays")?;
        if array.len() > 100
            || array.iter().any(|v| {
                v.as_str()
                    .is_none_or(|s| s.len() > 512 || s.chars().any(char::is_control))
            })
        {
            return Err("Invalid or oversized metadata source list".into());
        }
        payload[output] = sources;
    }
    for (input, output) in [
        ("machine_shape", "machineShape"),
        ("docker_image", "dockerImage"),
        ("docker_image_pinning_type", "dockerImagePinningType"),
    ] {
        if let Some(value) = meta.get(input).filter(|v| !v.is_null()) {
            let s = value
                .as_str()
                .filter(|s| s.len() <= 256 && !s.chars().any(char::is_control))
                .ok_or("Invalid metadata accelerator/container setting")?;
            if !s.is_empty() {
                payload[output] = Value::String(s.into());
            }
        }
    }
    let body = payload.to_string();
    if body.len() > MAX_CODE * 2 {
        return Err("Encoded notebook exceeds the 8 MiB upload limit".into());
    }
    if body.contains(&credentials.key) || body.contains(&credentials.authorization) {
        return Err(
            "Refusing to upload metadata/source containing the local Kaggle credential".into(),
        );
    }
    let summary = format!(
        "Target: {slug}\nTitle: {title}\nVisibility: {} • GPU: {gpu} • TPU: {tpu} • Internet: {internet}\nAccelerator: {} • Upload: {} bytes\nCreates/updates a version and EXECUTES code using Kaggle quota.\nOnly code_file is uploaded; no local datasets/checkpoints are uploaded.",
        if private { "PRIVATE" } else { "PUBLIC" },
        payload["machineShape"].as_str().unwrap_or("Kaggle default"),
        body.len(),
    );
    let summary = summary
        .lines()
        .map(|line| credentials.redact(line))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Prepared {
        slug: slug.into(),
        summary,
        credentials,
        payload,
    })
}

#[derive(Debug)]
struct ApiError {
    message: String,
    retryable: bool,
    idle_timeout: bool,
}
impl ApiError {
    fn invalid(message: &str) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            idle_timeout: false,
        }
    }
    fn transport() -> Self {
        Self {
            message: "Kaggle request failed or timed out; check network/TLS".into(),
            retryable: true,
            idle_timeout: false,
        }
    }
    fn read(error: std::io::Error) -> Self {
        let mut result = Self::transport();
        result.idle_timeout = matches!(
            error.kind(),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
        );
        result
    }
    fn status(status: u16) -> Self {
        let hint = match status {
            401 | 403 => {
                "access denied; verify credentials, notebook access and Kaggle account verification"
            }
            404 => "notebook/version or live-log endpoint unavailable",
            429 => "rate limited; wait before retrying",
            300..=399 => "redirect refused to protect credentials",
            _ => "request rejected by Kaggle",
        };
        Self {
            message: format!("HTTP {status}: {hint}"),
            retryable: status == 404 || status == 429 || status >= 500,
            idle_timeout: false,
        }
    }
}
struct Client {
    agent: ureq::Agent,
    base: String,
    credentials: Credentials,
}
impl Client {
    fn new(credentials: Credentials) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .redirects(0)
                .timeout_connect(Duration::from_secs(5))
                .timeout_read(Duration::from_secs(10))
                .timeout_write(Duration::from_secs(10))
                .timeout(REQUEST_TIMEOUT)
                .build(),
            base: API.into(),
            credentials,
        }
    }
    fn response(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<ureq::Response, ApiError> {
        // base is never user-configurable; tests alone replace it with loopback.
        let req = self
            .agent
            .request(method, &format!("{}{path}", self.base))
            .set("User-Agent", "oxide-ai/0.5.0")
            .set("Authorization", &self.credentials.authorization);
        let result = match body {
            Some(value) => req
                .set("Content-Type", "application/json")
                .send_string(&value.to_string()),
            None => req
                .set("Accept", "text/event-stream, application/json, text/plain")
                .call(),
        };
        let response = result.map_err(|e| match e {
            ureq::Error::Status(status, _) => ApiError::status(status),
            _ => ApiError::transport(),
        })?;
        if !(200..300).contains(&response.status()) {
            return Err(ApiError::status(response.status()));
        }
        Ok(response)
    }
    fn rpc(&self, method: &str, body: &Value) -> Result<Value, ApiError> {
        let response = self.response(
            "POST",
            &format!("/kernels.KernelsApiService/{method}"),
            Some(body),
        )?;
        let body = bounded_read(response.into_reader(), MAX_JSON)
            .map_err(|_| ApiError::invalid("Invalid or oversized Kaggle response"))?;
        let value: Value = serde_json::from_str(&body)
            .map_err(|_| ApiError::invalid("Invalid Kaggle JSON response"))?;
        if value["code"].as_u64().is_some_and(|n| n >= 400)
            || value["error"].as_str().is_some_and(|s| !s.is_empty())
        {
            return Err(ApiError::invalid(
                "Kaggle rejected the operation; check metadata and account access on Kaggle",
            ));
        }
        Ok(value)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Queued,
    Running,
    Complete,
    Failed,
    Cancelled,
}
fn status(value: &Value) -> Result<Status, ApiError> {
    // SDK enum JSON may be names or numeric values. Missing/unknown is NEVER
    // interpreted as successful completion.
    match value["status"]
        .as_str()
        .map(str::to_ascii_uppercase)
        .as_deref()
    {
        Some("QUEUED" | "NEW_SCRIPT") => Ok(Status::Queued),
        Some("RUNNING" | "CANCEL_REQUESTED") => Ok(Status::Running),
        Some("COMPLETE") => Ok(Status::Complete),
        Some("ERROR") => Ok(Status::Failed),
        Some("CANCEL_ACKNOWLEDGED") => Ok(Status::Cancelled),
        _ => match value["status"].as_u64() {
            Some(0 | 6) => Ok(Status::Queued),
            Some(1 | 4) => Ok(Status::Running),
            Some(2) => Ok(Status::Complete),
            Some(3) => Ok(Status::Failed),
            Some(5) => Ok(Status::Cancelled),
            _ => Err(ApiError::invalid(
                "Kaggle returned an unknown execution status",
            )),
        },
    }
}
fn pushed(value: &Value, fallback: &str) -> Result<(String, u64), String> {
    if value["error"].as_str().is_some_and(|s| !s.is_empty()) {
        return Err("Kaggle rejected notebook launch; inspect metadata/access on Kaggle".into());
    }
    let version = value["versionNumber"]
        .as_u64()
        .or_else(|| value["versionNumber"].as_str().and_then(|v| v.parse().ok()))
        .filter(|v| *v > 0)
        .ok_or("Launch response has no version; check Kaggle before retrying (run may exist)")?;
    let slug = value["ref"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback);
    if !reference(slug) {
        return Err(
            "Launch returned an invalid notebook reference; check Kaggle before retrying".into(),
        );
    }
    Ok((slug.into(), version))
}

/// Parent-facing lifecycle, separate from notebook stdout/progress strings.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Event {
    Started {
        reference: String,
    },
    Done,
    Error(String),
    /// Monitoring stopped locally; the Kaggle run may still be consuming quota.
    Detached,
}
enum Message {
    Prepared(Box<Prepared>),
    Started(String),
    Status(String),
    Line(String),
    Finished(Result<(), String>),
}
struct Worker {
    rx: Receiver<Message>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
fn send(tx: &SyncSender<Message>, cancel: &AtomicBool, mut message: Message) -> bool {
    loop {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        match tx.try_send(message) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(m)) => {
                message = m;
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
fn pause(cancel: &AtomicBool, duration: Duration) -> bool {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        std::thread::sleep(
            Duration::from_millis(50).min(until.saturating_duration_since(Instant::now())),
        );
    }
    !cancel.load(Ordering::Relaxed)
}

/// Replayed SSE entries are append-only for one pinned version. Store a count,
/// not all past logs. Partial stdout chunks are assembled before redaction so
/// a key split across events cannot be displayed in pieces.
#[derive(Default)]
struct Tail {
    seen: usize,
    partial: String,
    overlong: bool,
}
impl Tail {
    fn text(
        &mut self,
        text: &str,
        credentials: &Credentials,
        emit: &mut impl FnMut(String) -> bool,
    ) -> bool {
        for c in text.chars() {
            if c == '\n' || c == '\r' {
                if !self.flush(credentials, emit) {
                    return false;
                }
            } else if !self.overlong {
                if self.partial.len() + c.len_utf8() <= MAX_LINE {
                    self.partial.push(c);
                } else {
                    self.partial.clear();
                    self.overlong = true;
                }
            }
        }
        true
    }
    fn flush(&mut self, credentials: &Credentials, emit: &mut impl FnMut(String) -> bool) -> bool {
        let text = if self.overlong {
            self.overlong = false;
            self.partial.clear();
            "[Kaggle: oversized log line omitted]".into()
        } else {
            credentials.redact(&std::mem::take(&mut self.partial))
        };
        if text.len() > MAX_LINE {
            return emit("[Kaggle: oversized redacted log line omitted]".into());
        }
        text.is_empty() || emit(text)
    }
    fn event(
        &mut self,
        index: usize,
        value: &Value,
        credentials: &Credentials,
        emit: &mut impl FnMut(String) -> bool,
    ) -> Result<bool, ApiError> {
        let data = value["data"]
            .as_str()
            .ok_or_else(|| ApiError::invalid("Invalid Kaggle log event"))?;
        if index < self.seen {
            return Ok(true);
        }
        if !self.text(data, credentials, emit) {
            return Ok(false);
        }
        self.seen = index + 1;
        Ok(true)
    }
}
// read_until alone grows without bound on malformed SSE. This checks the cap
// before allocating another chunk. An overlong event is an explicit failure.
fn read_record(reader: &mut impl BufRead, limit: usize) -> Result<Option<String>, ApiError> {
    let mut line = Vec::new();
    loop {
        let chunk = reader.fill_buf().map_err(ApiError::read)?;
        if chunk.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                String::from_utf8(line)
                    .map(Some)
                    .map_err(|_| ApiError::invalid("Invalid UTF-8 in Kaggle logs"))
            };
        }
        let take = chunk
            .iter()
            .position(|b| *b == b'\n')
            .map_or(chunk.len(), |i| i + 1);
        if line.len() + take > limit {
            return Err(ApiError::invalid("Kaggle log event exceeds 256 KiB"));
        }
        let ended = chunk[take - 1] == b'\n';
        line.extend_from_slice(&chunk[..take]);
        reader.consume(take);
        if ended {
            return String::from_utf8(line)
                .map(Some)
                .map_err(|_| ApiError::invalid("Invalid UTF-8 in Kaggle logs"));
        }
    }
}
fn stream_logs(
    client: &Client,
    slug: &str,
    version: u64,
    tail: &mut Tail,
    cancel: &AtomicBool,
    emit: &mut impl FnMut(String) -> bool,
) -> Result<(), ApiError> {
    let response = client.response(
        "GET",
        &format!("/kernels/logs/stream/{slug}?versionLabel=v{version}"),
        None,
    )?;
    let is_sse = response
        .header("Content-Type")
        .unwrap_or("")
        .to_ascii_lowercase()
        .starts_with("text/event-stream");
    if !is_sse {
        // Completed-session endpoint returns the persisted [{time, data, ...}]
        // blob. Reject HTML and unknown bodies rather than render auth/errors.
        let body = bounded_read(response.into_reader(), MAX_LOG).map_err(|_| {
            ApiError::invalid("Kaggle log snapshot exceeds 16 MiB or is unreadable")
        })?;
        let value: Value = serde_json::from_str(&body).map_err(|_| {
            ApiError::invalid("Expected Kaggle JSON log entries, not an HTML/text response")
        })?;
        let events = value
            .as_array()
            .ok_or_else(|| ApiError::invalid("Invalid Kaggle log snapshot"))?;
        for (index, event) in events.iter().enumerate() {
            if cancel.load(Ordering::Relaxed)
                || !tail.event(index, event, &client.credentials, emit)?
            {
                break;
            }
        }
        return Ok(());
    }
    let mut reader = BufReader::new(response.into_reader());
    let mut index = 0;
    let mut bytes = 0;
    while !cancel.load(Ordering::Relaxed) {
        let Some(record) = read_record(&mut reader, MAX_EVENT)? else {
            break;
        };
        bytes += record.len();
        if bytes > MAX_LOG {
            return Err(ApiError::invalid(
                "Kaggle log stream/replay exceeds the 16 MiB safety limit",
            ));
        }
        let Some(data) = record.trim_end_matches(['\n', '\r']).strip_prefix("data:") else {
            continue;
        };
        let data = data.trim_start();
        if data == "END_OF_LOG" {
            break;
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|_| ApiError::invalid("Invalid Kaggle SSE log event"))?;
        if !tail.event(index, &value, &client.credentials, emit)? {
            break;
        }
        index += 1;
    }
    Ok(())
}
fn launch(prepared: Prepared, tx: &SyncSender<Message>, cancel: &AtomicBool) -> Result<(), String> {
    launch_with_client(
        Client::new(prepared.credentials),
        prepared.payload,
        &prepared.slug,
        tx,
        cancel,
    )
}
fn launch_with_client(
    client: Client,
    payload: Value,
    requested_slug: &str,
    tx: &SyncSender<Message>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }
    // SaveKernel is NOT retried: a timeout can still have created a version.
    let value = client.rpc("SaveKernel", &payload).map_err(|e| {
        format!(
            "{}. Check Kaggle before retrying: launch may have succeeded.",
            e.message
        )
    })?;
    let (slug, version) = pushed(&value, requested_slug)?;
    if !send(
        tx,
        cancel,
        Message::Started(client.credentials.redact(&format!("{slug}/v{version}"))),
    ) {
        return Ok(());
    }
    for field in [
        "invalidTags",
        "invalidDatasetSources",
        "invalidCompetitionSources",
        "invalidKernelSources",
        "invalidModelSources",
    ] {
        if value[field].as_array().is_some_and(|a| !a.is_empty()) {
            send(
                tx,
                cancel,
                Message::Line(
                    "[Kaggle: omitted invalid sources/tags; inspect the notebook on Kaggle]".into(),
                ),
            );
            break;
        }
    }
    let (owner, name) = slug.split_once('/').ok_or("Invalid notebook reference")?;
    let request =
        json!({"userName": owner, "kernelSlug": name, "versionLabel": format!("v{version}")});
    let mut tail = Tail::default();
    let mut failures = 0;
    let started = Instant::now();
    let mut emit = |line| send(tx, cancel, Message::Line(line));
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(24 * 60 * 60) {
            return Err("Monitoring stopped after 24 hours; check the remote run on Kaggle".into());
        }
        let state = match client
            .rpc("GetKernelSessionStatus", &request)
            .and_then(|v| status(&v))
        {
            Ok(state) => state,
            Err(e) => {
                failures += 1;
                if !e.retryable || failures >= 5 {
                    return Err(format!(
                        "{}; monitoring stopped, remote run may continue",
                        e.message
                    ));
                }
                send(
                    tx,
                    cancel,
                    Message::Status(format!("{}; retry {failures}/5", e.message)),
                );
                if !pause(cancel, INTERVAL * failures) {
                    return Ok(());
                }
                continue;
            }
        };
        send(
            tx,
            cancel,
            Message::Status(format!(
                "Kaggle {state:?} • version {version} • Esc detaches (does not cancel remote)"
            )),
        );
        let before = tail.seen;
        let logs = stream_logs(&client, &slug, version, &mut tail, cancel, &mut emit);
        match logs {
            Err(e) if !e.retryable => {
                return Err(format!("{}; remote run may continue", e.message));
            }
            Err(e) => {
                // A quiet, healthy SSE connection hits our deadline too. Keep
                // monitoring while the separate status endpoint is healthy;
                // an idle training/install phase is not a transport failure.
                failures = if tail.seen > before
                    || (e.idle_timeout && matches!(state, Status::Queued | Status::Running))
                {
                    0
                } else {
                    failures + 1
                };
                if failures >= 5 {
                    return Err(format!(
                        "{}; log monitoring stopped, remote run may continue",
                        e.message
                    ));
                }
            }
            Ok(()) => {
                failures = 0;
                match state {
                    Status::Complete => {
                        tail.flush(&client.credentials, &mut emit);
                        return Ok(());
                    }
                    Status::Failed | Status::Cancelled => {
                        tail.flush(&client.credentials, &mut emit);
                        return Err(
                            "Kaggle execution failed/cancelled; inspect notebook logs on Kaggle"
                                .into(),
                        );
                    }
                    _ => {}
                }
            }
        }
        if !pause(cancel, INTERVAL * failures.max(1)) {
            return Ok(());
        }
    }
}

pub(super) struct Kaggle {
    folder: String,
    status: String,
    prepared: Option<Box<Prepared>>,
    worker: Option<Worker>,
    events: VecDeque<Event>,
    history: VecDeque<String>,
    detaching: bool,
}
impl Kaggle {
    /// No credential reads, filesystem access or network on construction.
    pub fn new() -> Self {
        Self {
            folder: String::new(),
            status: "Enter the folder containing kernel-metadata.json, then Enter to review (not launch).".into(),
            prepared: None, worker: None, events: VecDeque::new(), history: VecDeque::new(), detaching: false,
        }
    }
    pub fn take_event(&mut self) -> Option<Event> {
        self.events.pop_front()
    }
    pub fn is_busy(&self) -> bool {
        self.worker.is_some() || self.prepared.is_some()
    }
    fn start_worker(
        &mut self,
        operation: impl FnOnce(&SyncSender<Message>, &AtomicBool) -> Result<(), String> + Send + 'static,
    ) {
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let cancel = Arc::new(AtomicBool::new(false));
        let child_cancel = cancel.clone();
        let result = std::thread::Builder::new()
            .name("kaggle-tab".into())
            .spawn(move || {
                if let Err(error) = operation(&tx, &child_cancel) {
                    send(&tx, &child_cancel, Message::Finished(Err(error)));
                }
            });
        match result {
            Ok(_) => {
                self.worker = Some(Worker { rx, cancel });
                self.detaching = false;
            }
            Err(_) => self.fail("Cannot start the Kaggle background worker".into()),
        }
    }
    fn fail(&mut self, error: String) {
        self.status = error.clone();
        self.events.push_back(Event::Error(error));
    }
    /// At most 64 bounded messages per tick; never waits for IO or a worker.
    pub fn poll(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        for _ in 0..QUEUE {
            let Some(worker) = self.worker.as_ref() else {
                break;
            };
            let message = match worker.rx.try_recv() {
                Ok(message) => message,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.worker = None;
                    if self.detaching {
                        self.events.push_back(Event::Detached);
                        self.status = "Detached locally. Remote run may still consume quota; stop it on Kaggle.".into();
                        self.detaching = false;
                    } else {
                        self.fail(
                            "Kaggle worker stopped unexpectedly; check remote run before retrying"
                                .into(),
                        );
                    }
                    break;
                }
            };
            if self.detaching {
                continue;
            }
            match message {
                Message::Prepared(prepared) => {
                    self.prepared = Some(prepared);
                    self.worker = None;
                    self.status =
                        "Review the snapshot below. Press Y to upload AND RUN; N/Esc cancels."
                            .into();
                    break;
                }
                Message::Started(reference) => {
                    self.history.clear();
                    self.events.push_back(Event::Started { reference });
                }
                Message::Status(status) => self.status = status,
                Message::Line(line) => {
                    if self.history.len() == HISTORY {
                        self.history.pop_front();
                    }
                    self.history.push_back(line.clone());
                    lines.push(line);
                }
                Message::Finished(result) => {
                    self.worker = None;
                    match result {
                        Ok(()) => {
                            self.status = "Kaggle run complete".into();
                            self.events.push_back(Event::Done);
                        }
                        Err(error) => self.fail(error),
                    }
                    break;
                }
            }
        }
        lines
    }
    pub fn key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if let Some(worker) = self.worker.as_ref() {
            if key.code == KeyCode::Esc {
                worker.cancel.store(true, Ordering::Relaxed);
                self.detaching = true;
                self.status =
                    "Detaching after bounded IO; this does NOT cancel the remote run.".into();
            }
            return;
        }
        let plain = !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
        if self.prepared.is_some() {
            match key.code {
                KeyCode::Char('y' | 'Y') if plain => {
                    let prepared = *self.prepared.take().unwrap();
                    self.status = "Uploading and launching once… Esc detaches only; a run may still be created.".into();
                    self.start_worker(move |tx, cancel| {
                        launch(prepared, tx, cancel)?;
                        send(tx, cancel, Message::Finished(Ok(())));
                        Ok(())
                    });
                }
                KeyCode::Esc | KeyCode::Char('n' | 'N') => {
                    self.prepared = None;
                    self.status = "Launch cancelled; nothing uploaded".into();
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Enter if self.events.is_empty() => {
                if self.folder.trim().is_empty() {
                    self.status = "Enter a notebook folder first".into();
                    return;
                }
                let folder = PathBuf::from(self.folder.trim());
                self.status =
                    "Reading local credentials + notebook snapshot (no network yet)…".into();
                self.start_worker(move |tx, cancel| {
                    let prepared = prepare(&folder, credentials()?)?;
                    send(tx, cancel, Message::Prepared(Box::new(prepared)));
                    Ok(())
                });
            }
            KeyCode::Esc => self.folder.clear(),
            KeyCode::Backspace => {
                self.folder.pop();
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.folder.clear()
            }
            KeyCode::Char(c)
                if plain && !c.is_control() && self.folder.len() + c.len_utf8() <= 4096 =>
            {
                self.folder.push(c)
            }
            _ => {}
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        let sections = Layout::vertical([
            Constraint::Length(if self.prepared.is_some() { 18 } else { 10 }),
            Constraint::Min(0),
        ])
        .split(area);
        let mut text = vec![
            Line::styled("KAGGLE / remote notebook run", accent()),
            Line::from(format!("Folder: {}", clean(&self.folder))),
            Line::from(self.status.as_str()),
            Line::from("Credentials: KAGGLE_USERNAME + KAGGLE_KEY, else ~/.kaggle/kaggle.json"),
            Line::from("No key is displayed or saved. Keep kaggle.json private (chmod 600)."),
            Line::from("Enter review • Y confirm • N/Esc cancel • Ctrl+U clear • Tab tabs"),
            Line::from("Run oxide-ai with --no-tui in the notebook for parseable monitor metrics."),
        ];
        if let Some(prepared) = &self.prepared {
            text.extend(prepared.summary.lines().map(|s| Line::from(s.to_owned())));
        }
        f.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(panel(" Kaggle launch ")),
            sections[0],
        );
        let visible = sections[1].height.saturating_sub(2) as usize;
        let logs: Vec<Line<'_>> = self
            .history
            .iter()
            .skip(self.history.len().saturating_sub(visible))
            .map(|s| Line::from(s.as_str()))
            .collect();
        f.render_widget(
            Paragraph::new(logs).block(panel(" Kaggle live log (bounded tail) ")),
            sections[1],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::TcpListener, sync::atomic::AtomicUsize};

    const KEY: &str = "fixture_NEVER_DISPLAY_key";
    fn creds() -> Credentials {
        Credentials::parse("fixtureuser".into(), KEY.into()).unwrap()
    }
    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static SERIAL: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "oxide-kaggle-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn notebook(&self) {
            fs::write(self.0.join("kernel-metadata.json"), json!({
                "id": "fixtureuser/test-notebook", "title": "Test Notebook", "code_file": "run.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": "true",
                "enable_gpu": "true", "dataset_sources": ["owner/dataset"],
            }).to_string()).unwrap();
            fs::write(self.0.join("run.ipynb"), json!({"nbformat": 4, "cells": [
                {"cell_type": "code", "source": ["print('hello')", "\n"], "outputs": [{"text":"old output"}], "execution_count": 4}
            ]}).to_string()).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn basic_auth_rfc_vectors_and_credential_precedence() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("user:key", "dXNlcjprZXk="),
        ] {
            assert_eq!(base64(plain.as_bytes()), encoded);
        }
        let dir = Fixture::new();
        let path = dir.0.join("kaggle.json");
        fs::write(
            &path,
            json!({"username":"saveduser", "key":"saved_fixture"}).to_string(),
        )
        .unwrap();
        assert_eq!(
            resolve_credentials(None, None, Some(&path))
                .unwrap()
                .username,
            "saveduser"
        );
        let from_env = resolve_credentials(
            Some("envuser".into()),
            Some("env_fixture".into()),
            Some(&path),
        )
        .unwrap();
        assert_eq!(from_env.username, "envuser");
        assert!(resolve_credentials(Some("envuser".into()), None, Some(&path)).is_err());
        assert!(resolve_credentials(None, None, None).is_err());
        assert!(Credentials::parse("user\r\n".into(), KEY.into()).is_err());
        assert!(Credentials::parse("user".into(), "bad\nkey".into()).is_err());
        fs::write(&path, "{".repeat(33 * 1024)).unwrap();
        assert!(resolve_credentials(None, None, Some(&path)).is_err());
        assert!(bounded_read("abcdef".as_bytes(), 5).is_err());
    }

    #[test]
    fn snapshot_normalizes_notebook_and_preserves_explicit_resource_choices() {
        let dir = Fixture::new();
        dir.notebook();
        let prepared = prepare(&dir.0, creds()).unwrap();
        assert_eq!(prepared.payload["slug"], "fixtureuser/test-notebook");
        assert_eq!(prepared.payload["isPrivate"], true);
        assert_eq!(prepared.payload["enableGpu"], true);
        assert_eq!(prepared.payload["enableInternet"], false);
        assert_eq!(prepared.payload["kernelExecutionType"], "SAVE_AND_RUN_ALL");
        assert_eq!(
            prepared.payload["datasetDataSources"],
            json!(["owner/dataset"])
        );
        let notebook: Value =
            serde_json::from_str(prepared.payload["text"].as_str().unwrap()).unwrap();
        assert_eq!(notebook["cells"][0]["source"], "print('hello')\n");
        assert_eq!(notebook["cells"][0]["outputs"], json!([]));
        assert_eq!(notebook["cells"][0]["execution_count"], Value::Null);
        assert!(prepared.summary.contains("PRIVATE"));
        assert!(!prepared.summary.contains(KEY));
        assert!(prepared.summary.lines().count() >= 5);
        // Changing a file after review never changes the upload snapshot.
        fs::write(dir.0.join("run.ipynb"), "changed after review").unwrap();
        assert_eq!(notebook["cells"][0]["source"], "print('hello')\n");
    }

    #[test]
    fn metadata_rejects_ambiguous_owner_path_boolean_and_embedded_key() {
        let dir = Fixture::new();
        dir.notebook();
        let base: Value = serde_json::from_str(
            &file_text(&dir.0.join("kernel-metadata.json"), MAX_JSON).unwrap(),
        )
        .unwrap();
        for (field, invalid) in [
            ("id", json!("otheruser/test-notebook")),
            ("id", json!("fixtureuser/../escape")),
            ("id_no", json!(123)),
            ("enable_gpu", json!("maybe")),
            ("code_file", json!("../outside.py")),
            ("code_file", json!("/etc/passwd")),
            ("title", json!(KEY)),
            ("dataset_sources", json!(true)),
        ] {
            let mut meta = base.clone();
            meta[field] = invalid;
            fs::write(dir.0.join("kernel-metadata.json"), meta.to_string()).unwrap();
            let err = prepare(&dir.0, creds())
                .err()
                .expect("invalid metadata accepted");
            assert!(!err.contains(KEY));
        }
        assert!(prepare(&dir.0.join("missing"), creds()).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = Fixture::new();
            fs::write(outside.0.join("outside.ipynb"), "{}").unwrap();
            fs::remove_file(dir.0.join("run.ipynb")).unwrap();
            symlink(outside.0.join("outside.ipynb"), dir.0.join("run.ipynb")).unwrap();
            fs::write(dir.0.join("kernel-metadata.json"), base.to_string()).unwrap();
            assert!(prepare(&dir.0, creds()).is_err());
        }
    }

    #[test]
    fn malformed_notebook_values_fail_without_panicking() {
        let dir = Fixture::new();
        dir.notebook();
        for invalid in [json!(true), json!([]), json!({}), json!({"cells": [true]})] {
            fs::write(dir.0.join("run.ipynb"), invalid.to_string()).unwrap();
            assert!(prepare(&dir.0, creds()).is_err());
        }
    }

    #[test]
    fn tail_replay_partial_lines_redaction_controls_and_bounds() {
        let credentials = creds();
        let mut tail = Tail::default();
        let mut lines = Vec::new();
        let mut emit = |s| {
            lines.push(s);
            true
        };
        tail.event(
            0,
            &json!({"data":"training 1/2 (50%) loss=4.0\r\nfixture_NEVER_"}),
            &credentials,
            &mut emit,
        )
        .unwrap();
        tail.event(
            0,
            &json!({"data":"training 1/2 (50%) loss=4.0\r\nfixture_NEVER_"}),
            &credentials,
            &mut emit,
        )
        .unwrap();
        tail.event(
            1,
            &json!({"data":"DISPLAY_key\n\u{1b}[31m世界\u{1b}[0m\n"}),
            &credentials,
            &mut emit,
        )
        .unwrap();
        tail.text(&"x".repeat(MAX_LINE + 20), &credentials, &mut emit);
        tail.text("\nlast partial", &credentials, &mut emit);
        tail.flush(&credentials, &mut emit);
        assert_eq!(
            lines,
            [
                "training 1/2 (50%) loss=4.0",
                "[redacted]",
                "世界",
                "[Kaggle: oversized log line omitted]",
                "last partial"
            ]
        );
        assert_eq!(tail.seen, 2);
        assert_eq!(credentials.redact(&credentials.authorization), "[redacted]");
        assert_eq!(
            credentials.redact("fixture_\u{1b}[31mNEVER_DISPLAY_key"),
            "[redacted]"
        );
        assert!(
            read_record(
                &mut BufReader::new("x".repeat(MAX_EVENT + 1).as_bytes()),
                MAX_EVENT
            )
            .is_err()
        );
        assert_eq!(
            read_record(&mut BufReader::new("世界\n".as_bytes()), MAX_EVENT).unwrap(),
            Some("世界\n".into())
        );
    }

    #[test]
    fn remote_status_and_push_validation_never_invent_success() {
        for (wire, expected) in [
            (json!("queued"), Status::Queued),
            (json!("RUNNING"), Status::Running),
            (json!(2), Status::Complete),
            (json!("ERROR"), Status::Failed),
            (json!(5), Status::Cancelled),
        ] {
            assert_eq!(status(&json!({"status":wire})).unwrap(), expected);
        }
        assert!(status(&json!({})).is_err());
        assert!(status(&json!({"status":"unknown"})).is_err());
        assert!(ApiError::read(std::io::ErrorKind::TimedOut.into()).idle_timeout);
        assert!(!ApiError::read(std::io::ErrorKind::ConnectionReset.into()).idle_timeout);
        assert!(
            pushed(&json!({"error":KEY}), "fixtureuser/test-notebook")
                .unwrap_err()
                .find(KEY)
                .is_none()
        );
        assert!(pushed(&json!({}), "fixtureuser/test-notebook").is_err());
        assert!(pushed(&json!({"versionNumber":0}), "fixtureuser/test-notebook").is_err());
        assert!(
            pushed(
                &json!({"versionNumber":3,"ref":"https://host/evil"}),
                "fixtureuser/test-notebook"
            )
            .is_err()
        );
        assert_eq!(
            pushed(&json!({"versionNumber":"3"}), "fixtureuser/test-notebook").unwrap(),
            ("fixtureuser/test-notebook".into(), 3)
        );
    }

    type Requests = std::thread::JoinHandle<Vec<(String, Value)>>;
    fn server(responses: Vec<(u16, &'static str, String)>) -> (Client, Requests) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, extra_headers, body) in responses {
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "local test server did not receive a request"
                            );
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(e) => panic!("test listener failed: {e}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                loop {
                    let line = read_record(&mut reader, MAX_EVENT).unwrap().unwrap();
                    head.push_str(&line);
                    if line == "\r\n" {
                        break;
                    }
                }
                let length: usize = head
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                    .map(|(_, value)| value.trim().parse().unwrap())
                    .unwrap_or(0);
                let mut bytes = vec![0; length];
                reader.read_exact(&mut bytes).unwrap();
                let request = if bytes.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes).unwrap()
                };
                requests.push((head, request));
                write!(stream, "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            requests
        });
        let mut client = Client::new(creds());
        client.base = format!("http://{address}/v1");
        (client, handle)
    }

    #[test]
    fn api_push_pinned_status_and_logs_feed_existing_runstate_before_done() {
        let line = "training 1/2 (50%) loss=4.0 tokens_per_second=123 eta=1s\n";
        let (client, server) = server(vec![
            (
                200,
                "",
                json!({"ref":"fixtureuser/test-notebook", "versionNumber":7}).to_string(),
            ),
            (200, "", json!({"status":"COMPLETE"}).to_string()),
            (
                200,
                "Content-Type: application/json\r\n",
                json!([{"data":line},{"data":format!("{KEY}\n")}]).to_string(),
            ),
        ]);
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let cancel = Arc::new(AtomicBool::new(false));
        let mut tab = Kaggle::new();
        tab.worker = Some(Worker {
            rx,
            cancel: cancel.clone(),
        });
        launch_with_client(
            client,
            json!({"slug":"fixtureuser/test-notebook"}),
            "fixtureuser/test-notebook",
            &tx,
            &cancel,
        )
        .unwrap();
        tx.send(Message::Finished(Ok(()))).unwrap();
        let lines = tab.poll();
        assert_eq!(
            tab.take_event(),
            Some(Event::Started {
                reference: "fixtureuser/test-notebook/v7".into()
            })
        );
        let mut state = super::super::RunState::default();
        for line in &lines {
            state.ingest(line);
        }
        assert_eq!(state.progress_pct, Some(50.0));
        assert_eq!(state.live_loss, Some(4.0));
        assert_eq!(state.tok_s, Some(123.0));
        assert_eq!(tab.take_event(), Some(Event::Done));
        assert!(tab.take_event().is_none());
        assert!(lines.iter().all(|line| !line.contains(KEY)));
        assert!(!tab.is_busy());
        let requests = server.join().unwrap();
        assert!(
            requests[0]
                .0
                .starts_with("POST /v1/kernels.KernelsApiService/SaveKernel ")
        );
        assert!(requests[0].0.contains(&creds().authorization));
        assert_eq!(requests[1].1["versionLabel"], "v7");
        assert!(
            requests[2].0.starts_with(
                "GET /v1/kernels/logs/stream/fixtureuser/test-notebook?versionLabel=v7 "
            )
        );
    }

    #[test]
    fn sse_reconnect_deduplicates_and_flushes_actual_new_metrics() {
        let body =
            "data: {\"data\":\"one\\n\"}\n\ndata: {\"data\":\"two\\n\"}\n\ndata: END_OF_LOG\n\n"
                .to_owned();
        let (client, server) = server(vec![
            (200, "Content-Type: text/event-stream\r\n", body.clone()),
            (200, "Content-Type: text/event-stream\r\n", body),
        ]);
        let mut tail = Tail::default();
        let mut lines = Vec::new();
        let cancel = AtomicBool::new(false);
        for _ in 0..2 {
            stream_logs(
                &client,
                "fixtureuser/test-notebook",
                1,
                &mut tail,
                &cancel,
                &mut |s| {
                    lines.push(s);
                    true
                },
            )
            .unwrap();
        }
        assert_eq!(lines, ["one", "two"]);
        server.join().unwrap();
    }

    #[test]
    fn http_errors_and_redirects_never_echo_credentials_or_retry_push() {
        for status in [302, 401, 403, 429, 500] {
            let (client, server) = server(vec![(
                status,
                "Location: http://127.0.0.1:1/leak\r\n",
                KEY.into(),
            )]);
            let (tx, _rx) = mpsc::sync_channel(QUEUE);
            let err = launch_with_client(
                client,
                json!({}),
                "fixtureuser/test-notebook",
                &tx,
                &AtomicBool::new(false),
            )
            .unwrap_err();
            assert!(err.contains(&status.to_string()));
            assert!(!err.contains(KEY));
            assert!(err.contains("before retrying"));
            assert_eq!(server.join().unwrap().len(), 1);
        }
    }

    #[test]
    fn explicit_confirmation_repeat_events_cancel_and_disconnect_are_safe() {
        let dir = Fixture::new();
        dir.notebook();
        let mut tab = Kaggle::new();
        tab.key(press(KeyCode::Enter));
        assert!(!tab.is_busy());
        tab.prepared = Some(Box::new(prepare(&dir.0, creds()).unwrap()));
        tab.key(press(KeyCode::Enter));
        assert!(tab.prepared.is_some()); // repeated Enter is never launch confirmation
        let mut repeat = press(KeyCode::Char('y'));
        repeat.kind = KeyEventKind::Repeat;
        tab.key(repeat);
        assert!(tab.prepared.is_some());
        tab.key(press(KeyCode::Char('n')));
        assert!(tab.prepared.is_none());
        assert!(tab.worker.is_none());
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let cancel = Arc::new(AtomicBool::new(false));
        tab.worker = Some(Worker {
            rx,
            cancel: cancel.clone(),
        });
        tab.key(press(KeyCode::Esc));
        assert!(cancel.load(Ordering::Relaxed));
        tx.send(Message::Started("late".into())).unwrap();
        drop(tx);
        assert!(tab.poll().is_empty());
        assert_eq!(tab.take_event(), Some(Event::Detached));
        assert!(tab.take_event().is_none());
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        tab.worker = Some(Worker {
            rx,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        drop(tx);
        tab.poll();
        assert!(matches!(tab.take_event(), Some(Event::Error(_))));
    }

    #[test]
    fn queue_backpressure_cancels_and_visible_history_is_bounded() {
        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(Message::Line("first".into())).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let worker =
            std::thread::spawn(move || send(&tx, &worker_cancel, Message::Line("blocked".into())));
        cancel.store(true, Ordering::Relaxed);
        assert!(!worker.join().unwrap());
        drop(rx);
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let mut tab = Kaggle::new();
        tab.worker = Some(Worker {
            rx,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        for n in 0..HISTORY * 2 {
            tx.send(Message::Line(n.to_string())).unwrap();
            assert_eq!(tab.poll().len(), 1);
        }
        assert_eq!(tab.history.len(), HISTORY);
        assert_eq!(tab.history.front().unwrap(), &HISTORY.to_string());
    }

    #[test]
    fn test_backend_confirmation_tail_and_tiny_layouts_do_not_expose_key() {
        let dir = Fixture::new();
        dir.notebook();
        let mut tab = Kaggle::new();
        tab.folder = "fixture notebook folder".into();
        tab.prepared = Some(Box::new(prepare(&dir.0, creds()).unwrap()));
        tab.status = "Press Y to upload AND RUN; N/Esc cancels.".into();
        tab.history
            .push_back(creds().redact(&format!("training loss=4.0 {KEY}")));
        for (width, height) in [(110, 32), (79, 24), (24, 8), (4, 2), (1, 1)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| tab.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(!text.contains(KEY));
            assert!(!text.contains(&creds().authorization));
            if width >= 79 {
                assert!(text.contains("upload AND RUN"));
                assert!(text.contains("PRIVATE"));
                assert!(text.contains("fixtureuser/test-notebook"));
            }
        }
    }
}
