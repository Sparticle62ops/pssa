//! Event-driven HF backups. The shell calls `checkpoint(path)` exactly once per
//! NEW canonical save event, never for progress snapshots/resume/file scans.
//! All checkpoint reads, hashing and HTTP run on one worker; busy automatic jobs
//! coalesce to the latest checkpoint. No worker is joined by the UI.
use super::{accent, hf, network, panel, panel_area};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

const MAX_FIELD: usize = 4096;
const MAX_CHECKPOINT: u64 = 5 * 1024 * 1024 * 1024;
const MAX_RESPONSE: u64 = 256 * 1024;
const SECTION: &str = "hf_backup";

/// Injectable boundary: invoked only on the backup worker. Do not include
/// credentials or raw server response/error text in returned diagnostics.
pub(super) type Uploader = Arc<dyn Fn(&str, &str, &AtomicBool) -> Result<(), String> + Send + Sync>;

pub(super) struct Backup {
    path: String,
    repo: String,
    every: u32,
    seen: u32,
    selected: usize,
    input: Option<String>,
    status: String,
    pending: Option<mpsc::Receiver<Result<(), String>>>,
    cancel: Arc<AtomicBool>,
    queued: Option<(String, String)>,
    uploader: Uploader,
    persist: bool,
}
impl Default for Backup {
    fn default() -> Self {
        Self::new(network::load_config(SECTION), Arc::new(upload), true)
    }
}
impl Backup {
    /// Fixture/embedding constructor: no config IO and no network on construction.
    #[cfg(test)]
    pub(super) fn with_uploader(config: Value, uploader: Uploader) -> Self {
        Self::new(config, uploader, false)
    }
    fn new(config: Value, uploader: Uploader, persist: bool) -> Self {
        let repo = config["repo"]
            .as_str()
            .filter(|s| valid_repo(s))
            .unwrap_or("")
            .into();
        let every = config["every"]
            .as_u64()
            .filter(|n| *n <= 100_000)
            .unwrap_or(3) as u32;
        Self {
            path: config["checkpoint"]
                .as_str()
                .filter(|s| s.len() <= MAX_FIELD && !s.chars().any(char::is_control))
                .unwrap_or("")
                .into(),
            repo,
            every,
            seen: 0,
            selected: 0,
            input: None,
            status: "Choose a checkpoint and an EXISTING owner/repo. p pushes; every=0 is off."
                .into(),
            pending: None,
            cancel: Arc::new(AtomicBool::new(false)),
            queued: None,
            uploader,
            persist,
        }
    }
    pub(super) fn editing(&self) -> bool {
        self.input.is_some()
    }
    pub(super) fn pending(&self) -> bool {
        self.pending.is_some() || self.queued.is_some()
    }
    fn config(&self) -> Value {
        json!({"checkpoint":self.path,"repo":self.repo,"every":self.every})
    }
    fn save(&mut self) {
        if self.persist && network::save_config(SECTION, &self.config()).is_err() {
            self.status = "Settings active for this session, but could not be saved".into();
        }
    }
    /// Repeated saves to the same path are distinct events. The caller, not a
    /// path-based cache, is responsible for canonical event deduplication.
    pub(super) fn checkpoint(&mut self, path: &str) {
        if path.is_empty() || path.len() > MAX_FIELD || path.chars().any(char::is_control) {
            return;
        }
        if self.path.is_empty() {
            self.path = path.into();
        }
        if self.every == 0 {
            return;
        }
        self.seen = self.seen.saturating_add(1);
        if self.seen < self.every {
            return;
        }
        self.seen = 0;
        if self.repo.is_empty() {
            self.status = "Auto backup skipped: choose an existing HF repository first".into();
        } else if network::offline() {
            self.status = "Auto backup skipped: offline mode".into();
        } else if self.pending.is_some() {
            self.queued = Some((path.into(), self.repo.clone()));
            self.status = "Upload busy; keeping only the latest due checkpoint".into();
        } else {
            self.start(path.into(), self.repo.clone());
        }
    }
    fn start(&mut self, path: String, repo: String) {
        if self.pending.is_some() {
            self.status = "Upload already in progress; no second worker started".into();
            return;
        }
        if path.is_empty() || path.len() > MAX_FIELD || !valid_repo(&repo) {
            self.status = "Enter a checkpoint path and an existing owner/repo".into();
            return;
        }
        if network::offline() {
            self.status = "Backup skipped: offline mode".into();
            return;
        }
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&self.cancel);
        let uploader = Arc::clone(&self.uploader);
        let (tx, rx) = mpsc::sync_channel(1);
        match std::thread::Builder::new()
            .name("hf-backup".into())
            .spawn(move || {
                let result = uploader(&path, &repo, &cancel);
                let _ = tx.send(result);
            }) {
            Ok(_) => {
                self.pending = Some(rx);
                self.status = "Uploading on a worker (training never waits)…".into();
            }
            Err(_) => self.status = "Cannot start backup worker; r retries".into(),
        }
    }
    pub(super) fn poll(&mut self) {
        let result = match self.pending.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(result)) => result,
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                Err("Backup worker stopped; r retries".into())
            }
            _ => return,
        };
        self.pending = None;
        self.status = if self.cancel.load(Ordering::Relaxed) {
            "Backup disabled; in-flight request may already have reached HF".into()
        } else {
            result
                .map(|_| "Checkpoint + model card pushed to HF".into())
                .unwrap_or_else(|e| network::clean(&e).chars().take(300).collect())
        };
        if let Some((path, repo)) = self.queued.take() {
            if self.every > 0 {
                self.start(path, repo);
            }
        }
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
                        && input.len() + c.len_utf8() <= MAX_FIELD =>
                {
                    input.push(c)
                }
                KeyCode::Enter => {
                    let value = input.trim().to_owned();
                    let valid = match self.selected {
                        0 => {
                            self.path = value;
                            true
                        }
                        1 if valid_repo(&value) || value.is_empty() => {
                            self.repo = value;
                            true
                        }
                        2 => match value.parse::<u32>() {
                            Ok(n) if n <= 100_000 => {
                                self.every = n;
                                self.seen = 0;
                                true
                            }
                            _ => false,
                        },
                        _ => false,
                    };
                    if valid {
                        self.input = None;
                        self.queued = None;
                        if self.every == 0 {
                            self.cancel.store(true, Ordering::Relaxed);
                        }
                        self.status =
                            "Settings applied. p pushes manually; 0 disables automatic backup."
                                .into();
                        self.save();
                    } else {
                        self.status =
                            "Repository must be owner/name; interval must be 0..100000".into();
                    }
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(2),
            KeyCode::Enter => {
                self.input = Some(match self.selected {
                    0 => self.path.clone(),
                    1 => self.repo.clone(),
                    _ => self.every.to_string(),
                })
            }
            KeyCode::Char('p' | 'r') => self.start(self.path.clone(), self.repo.clone()),
            KeyCode::Char('d') => {
                self.every = if self.every == 0 { 3 } else { 0 };
                self.seen = 0;
                self.queued = None;
                if self.every == 0 {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                self.status = if self.every == 0 {
                    "Automatic backup OFF; pending upload cancelled where possible"
                } else {
                    "Automatic backup ON: every 3 new save events"
                }
                .into();
                self.save();
            }
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let mut lines = vec![Line::styled("HF HUB / checkpoint backup", accent())];
        for (i, (name, value)) in [
            ("Checkpoint", self.path.clone()),
            ("Existing repo", self.repo.clone()),
            ("Every N saves (0=off)", self.every.to_string()),
        ]
        .into_iter()
        .enumerate()
        {
            let value = if self.selected == i {
                self.input.as_ref().unwrap_or(&value)
            } else {
                &value
            };
            lines.push(Line::from(format!(
                "{} {name}: {}{}",
                if i == self.selected { "▶" } else { " " },
                network::clean(value),
                if i == self.selected && self.editing() {
                    "▏"
                } else {
                    ""
                }
            )));
        }
        lines.extend([
            Line::from(self.status.as_str()),
            Line::from("↑/↓ field • Enter edit/apply • Esc cancel • Ctrl+U clear"),
            Line::from("p push • r retry • d toggle automatic backup"),
            Line::from(
                "Phase-10 login needs write access. No repo is created. Offline/logged out: skip.",
            ),
            Line::from(
                "Replaces model.pssa/model.trfm + README.md on main. Maximum 5 GiB (basic LFS).",
            ),
            Line::from("Only new save events count; newest due job wins while busy."),
        ]);
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" HF backup ")),
            area,
        );
    }
}
impl Drop for Backup {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
fn valid_repo(s: &str) -> bool {
    fn part(s: &str) -> bool {
        !s.is_empty()
            && s.len() <= 96
            && !s.starts_with(['.', '-'])
            && !s.ends_with(['.', '-'])
            && !s.contains("..")
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    }
    s.split_once('/')
        .is_some_and(|(owner, repo)| part(owner) && part(repo))
}
fn active(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("Backup cancelled".into())
    } else if network::offline() {
        Err("Backup skipped: offline mode".into())
    } else {
        Ok(())
    }
}
fn response(result: Result<ureq::Response, ureq::Error>) -> Result<ureq::Response, String> {
    let response = result.map_err(|e| match e {
        ureq::Error::Status(401 | 403, _) => {
            "HF access denied: log in with write access to this repo".into()
        }
        ureq::Error::Status(404, _) => {
            "HF repository not found; create it on HF or check access".into()
        }
        ureq::Error::Status(code, _) => format!("Upload rejected (HTTP {code}); r retries"),
        _ => "Upload connection failed or timed out; r retries".into(),
    })?;
    if !(200..300).contains(&response.status()) {
        return Err("Upload redirect refused; credentials were not forwarded".into());
    }
    Ok(response)
}
fn json_response(result: Result<ureq::Response, ureq::Error>) -> Result<Value, String> {
    let mut body = Vec::new();
    response(result)?
        .into_reader()
        .take(MAX_RESPONSE + 1)
        .read_to_end(&mut body)
        .map_err(|_| "Cannot read HF response")?;
    if body.len() as u64 > MAX_RESPONSE {
        return Err("HF response exceeds safe size limit".into());
    }
    serde_json::from_slice(&body).map_err(|_| "Invalid HF response".into())
}
/// Protocol boundary for offline upload fixtures; production never accepts a
/// caller-selected authenticated origin.
trait HubTransport {
    fn batch(&self, repo: &str, oid: &str, size: u64) -> Result<Value, String>;
    fn storage(
        &self,
        method: &str,
        action: &Value,
        size: u64,
        body: &mut dyn Read,
    ) -> Result<(), String>;
    fn commit(&self, repo: &str, body: &str) -> Result<Value, String>;
}
struct Hub(hf::HubCredential);
impl HubTransport for Hub {
    fn batch(&self, repo: &str, oid: &str, size: u64) -> Result<Value, String> {
        json_response(self.0.request("POST", &format!("https://huggingface.co/{repo}.git/info/lfs/objects/batch"))?
            .set("Accept", "application/vnd.git-lfs+json")
            .set("Content-Type", "application/vnd.git-lfs+json")
            .send_string(&json!({"operation":"upload", "transfers":["basic"], "hash_algo":"sha256", "ref":{"name":"main"}, "objects":[{"oid":oid,"size":size}]}).to_string()))
    }
    fn storage(
        &self,
        method: &str,
        action: &Value,
        size: u64,
        body: &mut dyn Read,
    ) -> Result<(), String> {
        response(
            storage_request(method, action, &self.0)?
                .set("Content-Length", &size.to_string())
                .set(
                    "Content-Type",
                    if method == "PUT" {
                        "application/octet-stream"
                    } else {
                        "application/vnd.git-lfs+json"
                    },
                )
                .send(body),
        )
        .map(|_| ())
    }
    fn commit(&self, repo: &str, body: &str) -> Result<Value, String> {
        json_response(
            self.0
                .request(
                    "POST",
                    &format!("https://huggingface.co/api/models/{repo}/commit/main"),
                )?
                .set("Content-Type", "application/x-ndjson")
                .send_string(body),
        )
    }
}

/// Upload checkpoint via Git LFS basic transfer, then atomically commit its LFS
/// reference and a small README. Only the trusted HF origin gets the HF token.
/// The signed storage URL gets no HF Authorization header, cookies or redirects.
fn upload(path: &str, repo: &str, cancel: &AtomicBool) -> Result<(), String> {
    active(cancel)?;
    if !valid_repo(repo) {
        return Err("Invalid HF repository".into());
    }
    let credential = hf::hub_credential()?.ok_or("Backup skipped: logged out of Hugging Face")?;
    network::background_io();
    upload_with(path, repo, cancel, &Hub(credential))
}
fn upload_with(
    path: &str,
    repo: &str,
    cancel: &AtomicBool,
    hub: &dyn HubTransport,
) -> Result<(), String> {
    active(cancel)?;
    if !valid_repo(repo) {
        return Err("Invalid HF repository".into());
    }
    let path = Path::new(path);
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .filter(|s| matches!(*s, "pssa" | "trfm"))
        .ok_or("Choose a .pssa or .trfm checkpoint")?;
    let meta = path
        .symlink_metadata()
        .map_err(|_| "Cannot read checkpoint metadata")?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_CHECKPOINT {
        return Err("Checkpoint must be a regular, nonempty file no larger than 5 GiB".into());
    }
    let mut file = File::open(path).map_err(|_| "Cannot open checkpoint")?;
    let before = file.metadata().map_err(|_| "Cannot inspect checkpoint")?;
    if !before.is_file() || before.len() != meta.len() {
        return Err("Checkpoint changed; retry after its save completes".into());
    }
    let size = before.len();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        active(cancel)?;
        let n = file
            .read(&mut buffer)
            .map_err(|_| "Cannot read checkpoint")?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if bytes > size {
            return Err("Checkpoint changed during hashing; retry".into());
        }
        hash.update(&buffer[..n]);
    }
    if bytes != size {
        return Err("Checkpoint changed during hashing; retry".into());
    }
    let oid = hash.finish();
    active(cancel)?;
    let batch = hub.batch(repo, &oid, size)?;
    if batch
        .get("transfer")
        .and_then(Value::as_str)
        .is_some_and(|s| s != "basic")
    {
        return Err(
            "HF requested an unsupported upload transfer; choose a smaller checkpoint".into(),
        );
    }
    let object = batch["objects"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or("HF returned no upload object")?;
    if object.get("error").is_some()
        || object["oid"].as_str() != Some(&oid)
        || object["size"].as_u64() != Some(size)
    {
        return Err("HF rejected the checkpoint object".into());
    }
    if let Some(action) = object["actions"].get("upload") {
        active(cancel)?;
        file.seek(SeekFrom::Start(0))
            .map_err(|_| "Cannot rewind checkpoint")?;
        let mut checked = CheckedFile {
            file: &mut file,
            cancel,
            hash: Sha256::new(),
            bytes: 0,
        };
        hub.storage("PUT", action, size, &mut checked)?;
        if checked.bytes != size || checked.hash.finish() != oid {
            return Err("Checkpoint changed during upload; no commit was created".into());
        }
        if let Some(verify) = object["actions"].get("verify") {
            active(cancel)?;
            let body = json!({"oid":oid,"size":size}).to_string();
            hub.storage("POST", verify, body.len() as u64, &mut body.as_bytes())?;
        }
    }
    let after = file
        .metadata()
        .map_err(|_| "Cannot inspect checkpoint after upload")?;
    if after.len() != size || before.modified().ok() != after.modified().ok() {
        return Err("Checkpoint changed; no commit was created, retry after save".into());
    }
    active(cancel)?;
    let card = format!(
        "---\ntags:\n- pssa\n---\n\n# PSSA checkpoint\n\nUploaded with the PSSA TUI.\n\n- File: `model.{extension}`\n- Checkpoint bytes: {size}\n- SHA-256: `{oid}`\n\nLoad this checkpoint with the matching PSSA version.\nTraining dataset, metrics, license and hardware are not inferred; add verified details before sharing.\n"
    );
    let body = commit_body(extension, &oid, size, &card);
    let result = hub.commit(repo, &body)?;
    if result.get("error").is_some() || result["commitOid"].as_str().is_none() {
        return Err(
            "HF did not confirm the checkpoint commit; inspect the repo before retrying".into(),
        );
    }
    Ok(())
}
fn storage_request(
    method: &str,
    action: &Value,
    credential: &hf::HubCredential,
) -> Result<ureq::Request, String> {
    let url = action["href"]
        .as_str()
        .ok_or("HF did not return a storage URL")?;
    let authority = url
        .strip_prefix("https://")
        .and_then(|s| s.split('/').next())
        .ok_or("Refusing insecure storage URL")?;
    if authority.is_empty()
        || authority.contains(['@', '\\', '#', '?'])
        || url.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err("Refusing invalid storage URL".into());
    }
    let trusted = url.starts_with("https://huggingface.co/");
    let mut req = if trusted {
        credential.request(method, url)?
    } else {
        ureq::AgentBuilder::new()
            .redirects(0)
            .timeout(Duration::from_secs(120))
            .timeout_connect(Duration::from_secs(10))
            .build()
            .request(method, url)
    };
    if let Some(headers) = action["header"].as_object() {
        for (name, value) in headers {
            // Never allow an API response to forward credentials to a different
            // origin, replace Host, or override transfer framing/Content-Length.
            let lower = name.to_ascii_lowercase();
            if lower == "chunk_size" {
                return Err(
                    "Multipart HF uploads are not supported; choose a smaller checkpoint".into(),
                );
            }
            if lower == "authorization" || lower == "cookie" {
                continue;
            }
            if lower.starts_with("x-amz-") || lower == "content-type" {
                let value = value
                    .as_str()
                    .filter(|s| s.len() <= 8192 && !s.chars().any(char::is_control))
                    .ok_or("Invalid storage upload header")?;
                req = req.set(name, value);
            }
        }
    }
    Ok(req)
}
struct CheckedFile<'a> {
    file: &'a mut File,
    cancel: &'a AtomicBool,
    hash: Sha256,
    bytes: u64,
}
impl Read for CheckedFile<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        active(self.cancel).map_err(std::io::Error::other)?;
        let n = self.file.read(output)?;
        self.hash.update(&output[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}
fn commit_body(extension: &str, oid: &str, size: u64, card: &str) -> String {
    [json!({"key":"header","value":{"summary":"Upload PSSA checkpoint and model card","description":""}}),
     json!({"key":"lfsFile","value":{"path":format!("model.{extension}"),"algo":"sha256","oid":oid,"size":size}}),
     json!({"key":"file","value":{"path":"README.md","encoding":"base64","content":base64(card.as_bytes())}})]
        .into_iter().map(|v| format!("{v}\n")).collect()
}
fn base64(input: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let a = chunk[0];
        let b = *chunk.get(1).unwrap_or(&0);
        let c = *chunk.get(2).unwrap_or(&0);
        output.push(TABLE[(a >> 2) as usize] as char);
        output.push(TABLE[(((a & 3) << 4) | (b >> 4)) as usize] as char);
        output.push(if chunk.len() > 1 {
            TABLE[(((b & 15) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            TABLE[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

// FIPS 180-4 SHA-256, fixed 64-byte state: no dependency, subprocess or whole
// checkpoint allocation. Tested against standard vectors and split boundaries.
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    used: usize,
    length: u64,
}
impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            used: 0,
            length: 0,
        }
    }
    fn update(&mut self, mut bytes: &[u8]) {
        self.length += bytes.len() as u64;
        while !bytes.is_empty() {
            let n = (64 - self.used).min(bytes.len());
            self.block[self.used..self.used + n].copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used == 64 {
                self.compress();
                self.used = 0;
            }
        }
    }
    fn compress(&mut self) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (i, chunk) in self.block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for i in 16..64 {
            let x = w[i - 15];
            let y = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10));
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let t1 = h
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
    fn finish(mut self) -> String {
        let bits = self.length * 8;
        self.update(&[0x80]);
        while self.used != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn settle(backup: &mut Backup) {
        for _ in 0..200 {
            backup.poll();
            if backup.pending.is_none() {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("fixture worker did not finish");
    }
    #[test]
    fn hashing_and_model_card_have_bounded_streaming_protocol() {
        for (input, expected) in [
            (
                "",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                "abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ] {
            let mut hash = Sha256::new();
            for byte in input.as_bytes() {
                hash.update(&[*byte]);
            }
            assert_eq!(hash.finish(), expected);
        }
        let mut hash = Sha256::new();
        for _ in 0..1000 {
            hash.update(&[b'a'; 1000]);
        }
        assert_eq!(
            hash.finish(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        let body = commit_body("pssa", "oid", 123, "model card");
        let records: Vec<Value> = body
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(records.len(), 3);
        assert_eq!(records[1]["key"], "lfsFile");
        assert_eq!(records[1]["value"]["size"], 123);
        assert_eq!(records[2]["value"]["path"], "README.md");
        assert!(!body.contains("local/path"));
    }
    #[test]
    fn streaming_upload_commits_weights_and_card_only_after_verified_transfer() {
        struct Fixture {
            commits: Mutex<Vec<String>>,
            uploaded: Mutex<u64>,
            corrupt: bool,
        }
        impl HubTransport for Fixture {
            fn batch(&self, repo: &str, oid: &str, size: u64) -> Result<Value, String> {
                assert_eq!(repo, "owner/model");
                Ok(
                    json!({"objects":[{"oid":oid,"size":size,"actions":{"upload":{"href":"fixture"},"verify":{"href":"fixture"}}}]}),
                )
            }
            fn storage(
                &self,
                method: &str,
                _: &Value,
                size: u64,
                body: &mut dyn Read,
            ) -> Result<(), String> {
                let mut buffer = [0u8; 1024];
                let mut total = 0;
                loop {
                    let n = body.read(&mut buffer).map_err(|_| "fixture read")?;
                    if n == 0 {
                        break;
                    }
                    total += n as u64;
                    if self.corrupt && method == "PUT" {
                        break;
                    }
                }
                if method == "PUT" {
                    *self.uploaded.lock().unwrap() = total;
                }
                if !self.corrupt {
                    assert_eq!(total, size);
                }
                Ok(())
            }
            fn commit(&self, _: &str, body: &str) -> Result<Value, String> {
                self.commits.lock().unwrap().push(body.into());
                Ok(json!({"commitOid":"fixture"}))
            }
        }
        let path = std::env::temp_dir().join(format!("backup-stream-{}.pssa", std::process::id()));
        std::fs::write(&path, [b'x'; 128 * 1024]).unwrap();
        for corrupt in [false, true] {
            let fixture = Fixture {
                commits: Mutex::new(Vec::new()),
                uploaded: Mutex::new(0),
                corrupt,
            };
            let result = upload_with(
                path.to_str().unwrap(),
                "owner/model",
                &AtomicBool::new(false),
                &fixture,
            );
            if network::offline() {
                assert!(result.unwrap_err().contains("offline"));
                continue;
            }
            assert_eq!(result.is_ok(), !corrupt);
            let commits = fixture.commits.lock().unwrap();
            assert_eq!(commits.len(), usize::from(!corrupt));
            if !corrupt {
                assert_eq!(*fixture.uploaded.lock().unwrap(), 128 * 1024);
                let records: Vec<Value> = commits[0]
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(records[1]["value"]["path"], "model.pssa");
                assert_eq!(records[2]["value"]["path"], "README.md");
                assert!(!commits[0].contains("backup-stream-"));
            }
        }
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn configuration_round_trip_and_invalid_values_use_safe_defaults() {
        let backup = Backup::with_uploader(
            json!({"repo":"owner/model","every":9,"checkpoint":"path with spaces.pssa"}),
            Arc::new(|_, _, _| panic!("no upload")),
        );
        let restored =
            Backup::with_uploader(backup.config(), Arc::new(|_, _, _| panic!("no upload")));
        assert_eq!(restored.path, "path with spaces.pssa");
        assert_eq!(restored.repo, "owner/model");
        assert_eq!(restored.every, 9);
        let invalid = Backup::with_uploader(
            json!({"repo":"evil/../repo","every":-1,"checkpoint":"bad\npath"}),
            Arc::new(|_, _, _| panic!("no upload")),
        );
        assert!(invalid.repo.is_empty());
        assert!(invalid.path.is_empty());
        assert_eq!(invalid.every, 3);
    }
    #[test]
    fn new_save_events_include_same_path_and_off_suppresses_work() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let copy = Arc::clone(&sent);
        let mut backup = Backup::with_uploader(
            json!({"repo":"owner/model"}),
            Arc::new(move |path, repo, _| {
                copy.lock()
                    .unwrap()
                    .push((path.to_owned(), repo.to_owned()));
                Ok(())
            }),
        );
        assert!(backup.pending.is_none());
        backup.checkpoint("same path.pssa");
        backup.checkpoint("same path.pssa");
        assert!(backup.pending.is_none());
        backup.checkpoint("same path.pssa");
        settle(&mut backup);
        // Offline environment is respected even by fixture upload jobs.
        assert_eq!(sent.lock().unwrap().len(), usize::from(!network::offline()));
        backup.key(key(KeyCode::Char('d')));
        assert_eq!(backup.every, 0);
        for _ in 0..9 {
            backup.checkpoint("other.pssa");
        }
        assert!(backup.pending.is_none());
    }
    #[test]
    fn busy_events_coalesce_without_spawning_unbounded_workers() {
        let mut backup = Backup::with_uploader(
            json!({"repo":"owner/model","every":1}),
            Arc::new(|_, _, _| Ok(())),
        );
        let (_tx, rx) = mpsc::sync_channel(1);
        backup.pending = Some(rx);
        for i in 0..100 {
            backup.checkpoint(&format!("checkpoint {i}.pssa"));
        }
        if !network::offline() {
            assert_eq!(backup.queued.as_ref().unwrap().0, "checkpoint 99.pssa");
        }
        backup.key(key(KeyCode::Char('d')));
        assert!(backup.queued.is_none());
        assert!(backup.cancel.load(Ordering::Relaxed));
    }
    #[test]
    fn field_editing_validation_and_cancel() {
        let mut backup =
            Backup::with_uploader(Value::Null, Arc::new(|_, _, _| panic!("no upload")));
        backup.key(key(KeyCode::Down));
        backup.key(key(KeyCode::Enter));
        for c in "owner/model".chars() {
            backup.key(key(KeyCode::Char(c)));
        }
        backup.key(key(KeyCode::Enter));
        assert_eq!(backup.repo, "owner/model");
        backup.key(key(KeyCode::Down));
        backup.key(key(KeyCode::Enter));
        backup.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        backup.key(key(KeyCode::Char('0')));
        backup.key(key(KeyCode::Enter));
        assert_eq!(backup.every, 0);
        backup.key(key(KeyCode::Enter));
        backup.key(key(KeyCode::Backspace));
        backup.key(key(KeyCode::Esc));
        assert!(!backup.editing());
        assert_eq!(backup.every, 0);
        for bad in [
            "owner/../leak",
            "owner/repo?token=x",
            "owner/repo\n",
            "https://evil/x",
            "owner/a/b",
        ] {
            assert!(!valid_repo(bad));
        }
    }
    #[test]
    fn backup_screen_wide_narrow_tiny() {
        for (w, h) in [(120, 24), (79, 18), (24, 8), (4, 2), (1, 1)] {
            let backup = Backup::with_uploader(
                json!({"repo":"fixture/model"}),
                Arc::new(|_, _, _| panic!("draw must not upload")),
            );
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
            terminal.draw(|f| backup.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            if w >= 79 {
                assert!(text.contains("fixture/model"));
                assert!(text.contains("Checkpoint"));
            }
        }
    }
}
