//! Hugging Face credentials shared by the TUI and plain CLI dataset loader.
//! Secrets deliberately have no Debug implementation. Never report HTTP bodies
//! or transport diagnostics from authenticated requests (either can echo headers).
use super::{accent, panel, panel_area};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

static SIGNED_OUT: AtomicBool = AtomicBool::new(false);
static USE_SAVED: AtomicBool = AtomicBool::new(false);
const CHILD_SIGNED_OUT: &str = "PSSA_HF_SIGNED_OUT";

// Pass authentication policy, never credentials, to wizard children. An
// explicit login must supersede an older inherited HF_TOKEN; logout must also
// work if removing the saved token failed. Do not change the process environment.
pub(super) fn configure_child(command: &mut std::process::Command) {
    child_policy(
        command,
        SIGNED_OUT.load(Ordering::Relaxed),
        USE_SAVED.load(Ordering::Relaxed),
    );
}
fn child_policy(command: &mut std::process::Command, signed_out: bool, use_saved: bool) {
    if signed_out || use_saved {
        command.env_remove("HF_TOKEN");
    }
    if signed_out {
        command.env(CHILD_SIGNED_OUT, "1");
    } else if use_saved {
        command.env_remove(CHILD_SIGNED_OUT);
        // Also clear the inherited legacy fallback after an explicit login.
        command.env_remove("OXIDE_HF_SIGNED_OUT");
    }
}
struct Token(String);
impl Token {
    fn parse(value: String) -> Result<Self, String> {
        let value = value.trim();
        if value.is_empty() || value.len() > 8192 || !value.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("Invalid token: enter a non-empty HF access token without spaces".into());
        }
        Ok(Self(value.into()))
    }
}
fn token_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".cache/huggingface/token"))
        .ok_or_else(|| "HOME is not set; cannot locate the HF token file".into())
}
fn resolve(env: Option<String>, path: &Path) -> Result<Option<Token>, String> {
    if let Some(value) = env.filter(|s| !s.trim().is_empty()) {
        return Token::parse(value).map(Some);
    }
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Cannot read the HF token file".into()),
    };
    let mut value = String::new();
    file.take(8194)
        .read_to_string(&mut value)
        .map_err(|_| "Cannot read the HF token file")?;
    Token::parse(value).map(Some)
}
fn existing() -> Result<Option<Token>, String> {
    let use_saved = USE_SAVED.load(Ordering::Relaxed);
    if SIGNED_OUT.load(Ordering::Relaxed)
        || (!use_saved
            && crate::env_var_os(CHILD_SIGNED_OUT, "OXIDE_HF_SIGNED_OUT")
                .is_some_and(|v| v == "1"))
    {
        return Ok(None);
    }
    let env = if use_saved {
        None
    } else {
        std::env::var("HF_TOKEN").ok()
    };
    // HF_TOKEN also works in containers without HOME.
    if let Some(value) = env.as_ref().filter(|s| !s.trim().is_empty()) {
        return Token::parse(value.clone()).map(Some);
    }
    match token_path() {
        Ok(path) => resolve(None, &path),
        Err(_) => Ok(None), // public datasets also work without a home directory
    }
}
/// Opaque Phase-10 credential for Hub uploads. Resolve only on a worker; the
/// token never enters upload configuration, UI state, URLs, or diagnostics.
pub(super) struct HubCredential(Token);
pub(super) fn hub_credential() -> Result<Option<HubCredential>, String> {
    existing().map(|token| token.map(HubCredential))
}
impl HubCredential {
    pub(super) fn request(&self, method: &str, endpoint: &str) -> Result<ureq::Request, String> {
        if SIGNED_OUT.load(Ordering::Relaxed) {
            return Err("Backup skipped: logged out of Hugging Face".into());
        }
        if !endpoint.starts_with("https://huggingface.co/") {
            return Err("Refusing HF authentication to an untrusted origin".into());
        }
        Ok(ureq::AgentBuilder::new()
            .redirects(0)
            .timeout(Duration::from_secs(120))
            .timeout_connect(Duration::from_secs(10))
            .build()
            .request(method, endpoint)
            .set("Authorization", &format!("Bearer {}", self.0.0)))
    }
}

fn store(path: &Path, token: &Token) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("Cannot locate the HF token directory")?;
    fs::create_dir_all(parent).map_err(|_| "Cannot create the HF token directory")?;
    let temp = parent.join(format!(".token-{}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| -> std::io::Result<()> {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        created = true;
        file.write_all(token.0.as_bytes())?;
        file.sync_all()?;
        // Atomic replacement never follows an existing token symlink and resets
        // permissions even if the old file was world-readable.
        fs::rename(&temp, path)
    })();
    if result.is_err() && created {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(|_| {
        "Cannot securely save the HF token (requires a writable token directory)".into()
    })
}
fn remove(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("Cannot remove the saved HF token; check file permissions".into()),
    }
}
fn http_error(status: u16) -> String {
    match status {
        401 | 403 => format!(
            "HTTP {status}: authentication/access denied. Log in with a read token; for gated datasets request/accept access on https://huggingface.co/datasets/OWNER/NAME and wait for approval"
        ),
        404 => {
            "HTTP 404: dataset/config not found, or private dataset requires an authorized HF token"
                .into()
        }
        _ => format!("HTTP {status}: Hugging Face rejected the request"),
    }
}
fn request(endpoint: &str, token: Option<&Token>) -> Result<serde_json::Value, String> {
    // Never forward a Bearer header across redirects.
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout(Duration::from_secs(20))
        .build();
    let mut req = agent.get(endpoint).set("User-Agent", "pssa/0.5.0");
    if let Some(token) = token {
        req = req.set("Authorization", &format!("Bearer {}", token.0));
    }
    let response = req.call().map_err(|e| match e {
        ureq::Error::Status(status, _) => http_error(status),
        _ => "Hugging Face request failed; check network/TLS and retry".into(),
    })?;
    if response.status() != 200 {
        return Err(http_error(response.status()));
    }
    let mut body = String::new();
    response
        .into_reader()
        .take(16 * 1024 * 1024 + 1)
        .read_to_string(&mut body)
        .map_err(|_| "Cannot read Hugging Face response")?;
    if body.len() > 16 * 1024 * 1024 {
        return Err("Hugging Face response exceeds 16 MiB".into());
    }
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|_| "Invalid Hugging Face JSON response")?;
    if value.get("error").is_some() {
        return Err(
            "Hugging Face returned an error; check dataset access approval and configuration"
                .into(),
        );
    }
    Ok(value)
}
pub(crate) fn dataset_json(endpoint: &str) -> Result<serde_json::Value, String> {
    if !endpoint.starts_with("https://datasets-server.huggingface.co/") {
        return Err("Refusing to send HF credentials to an untrusted host".into());
    }
    request(endpoint, existing()?.as_ref())
}
fn whoami(token: &Token) -> Result<String, String> {
    let value = request("https://huggingface.co/api/whoami-v2", Some(token))?;
    identity(&value, token)
}
fn identity(value: &serde_json::Value, token: &Token) -> Result<String, String> {
    value["name"]
        .as_str()
        .filter(|name| {
            !name.is_empty()
                && !name.contains(&token.0)
                && name.len() <= 100
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
        })
        .map(str::to_owned)
        .ok_or_else(|| "HF did not return a valid account name".into())
}

pub(super) struct Login {
    input: String,
    user: Option<String>,
    status: String,
    pending: Option<mpsc::Receiver<Result<(String, Option<Token>), String>>>,
}
impl Login {
    pub fn new() -> Self {
        let mut login = Self {
            input: String::new(),
            user: None,
            status: "Enter a read token, then Enter to log in. Ctrl+L logs out.".into(),
            pending: None,
        };
        login.verify(None);
        login
    }
    fn verify(&mut self, entered: Option<Token>) {
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        self.status = "Checking HF identity…".into();
        std::thread::spawn(move || {
            let result = (|| {
                if let Some(token) = entered {
                    return whoami(&token).map(|name| (name, Some(token)));
                }
                let token =
                    existing()?.ok_or("Not logged in. Enter a read token, or set HF_TOKEN.")?;
                whoami(&token).map(|name| (name, None))
            })();
            let _ = tx.send(result);
        });
    }
    pub fn poll(&mut self) {
        let result = match self.pending.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(result)) => result,
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                Err("HF identity check stopped; retry".into())
            }
            _ => return,
        };
        self.pending = None;
        match result {
            Ok((name, token)) => {
                if let Some(token) = token {
                    if let Err(e) = token_path().and_then(|path| store(&path, &token)) {
                        self.status = e;
                        return;
                    }
                    USE_SAVED.store(true, Ordering::Relaxed);
                    SIGNED_OUT.store(false, Ordering::Relaxed);
                }
                self.user = Some(name);
                self.status =
                    "Authenticated. Gated datasets may still require approval on the HF site."
                        .into();
            }
            Err(e) => {
                self.user = None;
                self.status = e;
            }
        }
    }
    fn logout(&mut self, path: Result<PathBuf, String>) {
        // Drop the receiver so a late whoami cannot sign back in or save a token.
        self.pending = None;
        self.input.clear();
        self.user = None;
        SIGNED_OUT.store(true, Ordering::Relaxed);
        self.status = match path.and_then(|path| remove(&path)) {
            Ok(()) => "Logged out. HF_TOKEN ignored for this process; unset it in your shell to sign out future CLI runs.".into(),
            Err(e) => e,
        };
    }
    pub fn key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
            self.logout(token_path());
            return;
        }
        match key.code {
            KeyCode::Enter if self.pending.is_none() => {
                let input = std::mem::take(&mut self.input);
                if input.is_empty() {
                    self.verify(None);
                } else {
                    match Token::parse(input) {
                        Ok(token) => self.verify(Some(token)),
                        Err(e) => self.status = e,
                    }
                }
            }
            KeyCode::Esc => self.input.clear(),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && !c.is_control()
                    && self.input.len() < 8192 =>
            {
                self.input.push(c)
            }
            _ => {}
        }
    }
    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        let area = panel_area(f, area);
        let mask = "•".repeat(
            self.input
                .chars()
                .count()
                .min(area.width.saturating_sub(12) as usize),
        );
        f.render_widget(
            Paragraph::new(vec![
                Line::styled("HUGGING FACE / secure dataset access", accent()),
                Line::from(format!(
                    "Account: {}",
                    self.user.as_deref().unwrap_or("signed out")
                )),
                Line::from(""),
                Line::from(format!("Token: {mask}")),
                Line::from(self.status.as_str()),
                Line::from(""),
                Line::from("Enter verify/save • Esc clear • Ctrl+L logout • Tab tabs"),
                Line::from("Reads HF_TOKEN, then ~/.cache/huggingface/token; saves mode 0600."),
                Line::from("Create a read token: https://huggingface.co/settings/tokens"),
                Line::from(
                    "Logout removes the file; it cannot edit the parent shell's environment.",
                ),
            ])
            .wrap(Wrap { trim: false })
            .block(panel(" HF login ")),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_credentials_are_opaque_and_origin_restricted() {
        let credential = HubCredential(Token::parse("hf_upload_fixture".into()).unwrap());
        let request = credential
            .request(
                "POST",
                "https://huggingface.co/api/models/owner/repo/commit/main",
            )
            .unwrap();
        assert_eq!(
            request.header("Authorization"),
            Some("Bearer hf_upload_fixture")
        );
        for origin in [
            "https://huggingface.co.evil.test/upload",
            "https://huggingface.co@evil.test/upload",
            "http://huggingface.co/upload",
            "https://storage.example/upload",
        ] {
            let error = credential.request("PUT", origin).err().unwrap();
            assert!(!error.contains("hf_upload_fixture"));
            assert!(error.contains("untrusted"));
        }
    }
    #[test]
    fn wizard_children_inherit_login_policy_without_token_arguments() {
        use std::ffi::OsStr;
        for (signed_out, use_saved) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut command = std::process::Command::new("trainer");
            child_policy(&mut command, signed_out, use_saved);
            let env: std::collections::HashMap<_, _> = command.get_envs().collect();
            if signed_out || use_saved {
                assert_eq!(env.get(OsStr::new("HF_TOKEN")), Some(&None));
            } else {
                assert!(
                    env.is_empty(),
                    "startup preserves caller credential precedence"
                );
            }
            if signed_out {
                assert_eq!(
                    env.get(OsStr::new(CHILD_SIGNED_OUT)),
                    Some(&Some(OsStr::new("1")))
                );
            } else if use_saved {
                assert_eq!(env.get(OsStr::new(CHILD_SIGNED_OUT)), Some(&None));
                assert_eq!(env.get(OsStr::new("OXIDE_HF_SIGNED_OUT")), Some(&None));
            }
            assert_eq!(command.get_args().count(), 0);
        }
    }
    #[test]
    fn credentials_round_trip_permissions_and_precedence() {
        let dir = std::env::temp_dir().join(format!("pssa-hf-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("token");
        store(&path, &Token::parse("hf_fixture".into()).unwrap()).unwrap();
        assert_eq!(resolve(None, &path).unwrap().unwrap().0, "hf_fixture");
        assert_eq!(
            resolve(Some("hf_env".into()), &path).unwrap().unwrap().0,
            "hf_env"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
            store(&path, &Token("hf_new".into())).unwrap();
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        remove(&path).unwrap();
        remove(&path).unwrap();
        assert!(resolve(None, &path).unwrap().is_none());
        fs::remove_dir_all(dir).unwrap();
        assert!(Token::parse("hf_bad\r\nAuthorization: nope".into()).is_err());
    }
    #[test]
    fn identity_validation_never_displays_echoed_credentials() {
        let token = Token("hf_fixture".into());
        assert_eq!(
            identity(&serde_json::json!({"name":"fixture-user"}), &token).unwrap(),
            "fixture-user"
        );
        for value in [
            serde_json::json!({"name":"hf_fixture"}),
            serde_json::json!({"name":"bad\u{1b}[31m"}),
            serde_json::json!({}),
        ] {
            let error = identity(&value, &token).unwrap_err();
            assert!(!error.contains("hf_fixture"));
        }
    }
    #[test]
    fn bearer_header_and_error_redaction() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            let n = stream.read(&mut request).unwrap();
            assert!(
                String::from_utf8_lossy(&request[..n]).contains("Authorization: Bearer hf_fixture")
            );
            let body = "{\"error\":\"hf_fixture\"}";
            write!(
                stream,
                "HTTP/1.1 403 Forbidden\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let error = request(
            &format!("http://{addr}/rows"),
            Some(&Token("hf_fixture".into())),
        )
        .unwrap_err();
        worker.join().unwrap();
        assert!(error.contains("approval"));
        assert!(!error.contains("hf_fixture"));
        assert!(
            dataset_json("https://example.com/")
                .unwrap_err()
                .contains("untrusted")
        );
    }
    #[test]
    fn redirects_never_forward_credentials() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0; 4096];
            let _ = stream.read(&mut buf).unwrap();
            write!(stream, "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        });
        let error = request(
            &format!("http://{addr}/rows"),
            Some(&Token("hf_fixture".into())),
        )
        .unwrap_err();
        worker.join().unwrap();
        assert!(error.contains("302"));
        assert!(!error.contains("hf_fixture"));
    }
    #[test]
    fn logout_cancels_pending_identity_without_saving_late_reply() {
        let (tx, rx) = mpsc::channel();
        let mut login = Login {
            input: "hf_fixture".into(),
            user: Some("old".into()),
            status: String::new(),
            pending: Some(rx),
        };
        let path = std::env::temp_dir().join(format!("pssa-hf-logout-{}", std::process::id()));
        store(&path, &Token("hf_fixture".into())).unwrap();
        login.logout(Ok(path.clone()));
        assert!(
            tx.send(Ok(("late".into(), Some(Token("hf_fixture".into())))))
                .is_err()
        );
        login.poll();
        assert!(login.user.is_none());
        assert!(login.input.is_empty());
        assert!(!path.exists());
        assert!(existing().unwrap().is_none());
        SIGNED_OUT.store(false, Ordering::Relaxed);
    }
    #[test]
    fn token_entry_edits_and_clears_without_submission() {
        let mut login = Login {
            input: String::new(),
            user: None,
            status: String::new(),
            pending: None,
        };
        for c in "hf_fixture".chars() {
            login.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
        login.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(login.input, "hf_fixtur");
        login.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(login.input.is_empty());
        assert!(login.pending.is_none());
    }
    #[test]
    fn test_backend_masks_token_at_every_size() {
        for (width, height) in [(100, 24), (79, 18), (24, 8), (4, 2)] {
            let login = Login {
                input: "hf_NEVER_SHOW_THIS".into(),
                user: Some("fixture-user".into()),
                status: "Ready".into(),
                pending: None,
            };
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| login.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(!text.contains("hf_NEVER"));
            if width >= 79 {
                assert!(text.contains("••••••••"));
                assert!(text.contains("fixture-user"));
            }
        }
    }
}
