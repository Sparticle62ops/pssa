//! Token login and read-only public community browsing. No GitHub write APIs.
use super::{
    accent,
    network::{self, Http, Web},
    panel, panel_area,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::Value;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
};

const REPO: &str = "https://api.github.com/repos/Sparticle62ops/pssa";
// No Debug implementation: a worker reply must never accidentally print a token.
struct Token(String);
impl Token {
    fn parse(value: String) -> Result<Self, String> {
        let value = value.trim();
        if value.is_empty() || value.len() > 8192 || !value.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("Enter a GitHub token without spaces (maximum 8192 bytes)".into());
        }
        Ok(Self(value.into()))
    }
}
fn token_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".cache/pssa/github/token"))
        .ok_or_else(|| "HOME is not set; cannot locate GitHub credentials".into())
}
fn resolve(env: Option<String>, path: &Path) -> Result<Option<Token>, String> {
    if let Some(value) = env.filter(|v| !v.trim().is_empty()) {
        return Token::parse(value).map(Some);
    }
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Cannot read GitHub token file".into()),
    };
    let mut value = String::new();
    file.take(8194)
        .read_to_string(&mut value)
        .map_err(|_| "Cannot read GitHub token file")?;
    Token::parse(value).map(Some)
}
fn existing() -> Result<Option<Token>, String> {
    let env = ["GH_TOKEN", "GITHUB_TOKEN"]
        .iter()
        .find_map(|key| std::env::var(key).ok().filter(|s| !s.trim().is_empty()));
    if let Some(value) = env {
        return Token::parse(value).map(Some);
    }
    token_path().map_or(Ok(None), |path| resolve(None, &path))
}
fn identity(http: &dyn Http, token: &Token) -> Result<String, String> {
    let value = http.get("https://api.github.com/user", Some(&token.0))?;
    value["login"]
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 100
                && !s.contains(&token.0)
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
        })
        .map(str::to_owned)
        .ok_or_else(|| "GitHub returned an invalid account name".into())
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Pr,
    Issue,
}
impl Kind {
    fn label(self) -> &'static str {
        if self == Self::Pr { "PR" } else { "ISSUE" }
    }
    fn web_path(self) -> &'static str {
        if self == Self::Pr { "pull" } else { "issues" }
    }
}
struct Item {
    number: u64,
    title: String,
    url: String,
}
fn items(value: &Value, kind: Kind, token: Option<&str>) -> Result<Vec<Item>, String> {
    let array = value.as_array().ok_or("Invalid GitHub list response")?;
    Ok(array
        .iter()
        .take(50)
        .filter(|v| kind != Kind::Issue || v.get("pull_request").is_none())
        .filter_map(|v| {
            let number = v["number"].as_u64().filter(|n| *n > 0)?;
            let title = v["title"].as_str()?;
            let title = if token.is_some_and(|secret| title.contains(secret)) {
                "[redacted title]".into()
            } else {
                network::clean(title).chars().take(256).collect()
            };
            Some(Item {
                number,
                title,
                url: format!(
                    "https://github.com/Sparticle62ops/pssa/{}/{number}",
                    kind.web_path()
                ),
            })
        })
        .collect())
}
enum Reply {
    Login {
        name: String,
        token: Token,
        save: bool,
    },
    List(Vec<Item>),
}
pub(super) struct Github {
    editing: bool,
    input: String,
    user: Option<String>,
    token: Option<Token>,
    signed_out: bool,
    initialized: bool,
    kind: Kind,
    rows: Vec<Item>,
    selected: usize,
    status: String,
    http: Arc<dyn Http>,
    pending: Option<mpsc::Receiver<Result<Reply, String>>>,
    browser: Option<mpsc::Receiver<Result<(), String>>>,
}
impl Default for Github {
    fn default() -> Self {
        Self {
            editing: false,
            input: String::new(),
            user: None,
            token: None,
            signed_out: false,
            initialized: false,
            kind: Kind::Pr,
            rows: Vec::new(),
            selected: 0,
            status: "Public read-only browsing / l token login / Ctrl+L logout".into(),
            http: Arc::new(Web),
            pending: None,
            browser: None,
        }
    }
}
impl Github {
    pub(super) fn editing(&self) -> bool {
        self.editing
    }
    fn login(&mut self, entered: Option<Token>) {
        if network::offline() {
            self.status = "Offline / login skipped".into();
            return;
        }
        if self.pending.is_some() {
            return;
        }
        let (tx, rx) = mpsc::sync_channel(1);
        self.pending = Some(rx);
        self.status = "Checking GitHub identity…".into();
        let http = Arc::clone(&self.http);
        std::thread::spawn(move || {
            let result = (|| {
                let save = entered.is_some();
                let token = entered
                    .map_or_else(existing, |token| Ok(Some(token)))?
                    .ok_or("No GitHub token; public browsing remains available")?;
                let name = identity(http.as_ref(), &token)?;
                Ok(Reply::Login { name, token, save })
            })();
            let _ = tx.send(result);
        });
    }
    fn refresh(&mut self) {
        if network::offline() {
            self.status = "Offline / cached community list retained".into();
            return;
        }
        if self.pending.is_some() {
            self.status = "Request already in progress".into();
            return;
        }
        let token = self.token.as_ref().map(|t| t.0.clone());
        let http = Arc::clone(&self.http);
        let kind = self.kind;
        let (tx, rx) = mpsc::sync_channel(1);
        self.pending = Some(rx);
        self.status = "Loading open community items…".into();
        std::thread::spawn(move || {
            let url = if kind == Kind::Pr {
                format!("{REPO}/pulls?state=open&per_page=50")
            } else {
                // /issues also contains PRs and can hide all real issues on a
                // busy repository's first page. Search explicitly for issues.
                "https://api.github.com/search/issues?q=repo%3ASparticle62ops%2Fpssa%20is%3Aissue%20is%3Aopen&sort=updated&order=desc&per_page=50".into()
            };
            let result = http
                .get(&url, token.as_deref())
                .and_then(|v| {
                    items(
                        if kind == Kind::Issue { &v["items"] } else { &v },
                        kind,
                        token.as_deref(),
                    )
                })
                .map(Reply::List);
            let _ = tx.send(result);
        });
    }
    fn logout(&mut self, path: Result<PathBuf, String>) {
        self.pending = None;
        self.token = None;
        self.user = None;
        self.input.clear();
        self.editing = false;
        self.signed_out = true;
        self.initialized = true;
        self.status = match path.and_then(|p| match fs::remove_file(p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err("Signed out in this process; could not remove token file".into()),
        }) {
            Ok(()) => "Signed out. GH_TOKEN/GITHUB_TOKEN ignored for this process; unset them for future sessions.".into(),
            Err(e) => e,
        };
    }
    pub(super) fn poll(&mut self, visible: bool) {
        if visible && !self.initialized {
            self.initialized = true;
            if !self.signed_out {
                self.login(None);
            } else {
                self.refresh();
            }
        }
        let reply = match self.pending.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(reply)) => Some(reply),
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                Some(Err("GitHub worker stopped; r retries".into()))
            }
            _ => None,
        };
        if let Some(reply) = reply {
            self.pending = None;
            match reply {
                Ok(Reply::Login { name, token, save }) => {
                    if save {
                        if let Err(e) = token_path()
                            .and_then(|p| network::private_write(&p, token.0.as_bytes()))
                        {
                            self.status = e;
                            return;
                        }
                    }
                    self.user = Some(name);
                    self.token = Some(token);
                    self.signed_out = false;
                    self.editing = false;
                    self.refresh();
                }
                Ok(Reply::List(rows)) => {
                    self.rows = rows;
                    self.selected = self.selected.min(self.rows.len().saturating_sub(1));
                    self.status =
                        "Read-only / first 50 open items; o opens full discussion in browser"
                            .into();
                }
                Err(e) => {
                    // Missing credentials should not prevent public browsing.
                    let public = e.starts_with("No GitHub token;");
                    self.status = e;
                    if public {
                        self.refresh();
                    }
                }
            }
        }
        if let Some(result) = self.browser.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.browser = None;
            self.status = result
                .map(|_| "Discussion opened in browser".into())
                .unwrap_or_else(|e| e);
        }
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('l') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.logout(token_path());
            return;
        }
        if self.editing {
            match key.code {
                KeyCode::Esc => {
                    self.input.clear();
                    self.editing = false;
                }
                KeyCode::Enter if self.pending.is_none() => {
                    let value = std::mem::take(&mut self.input);
                    if value.trim().is_empty() {
                        self.login(None);
                    } else {
                        match Token::parse(value) {
                            Ok(token) => self.login(Some(token)),
                            Err(e) => self.status = e,
                        }
                    }
                }
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.input.clear()
                }
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                        && self.input.len() + c.len_utf8() <= 8192 =>
                {
                    self.input.push(c)
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('l') => {
                self.editing = true;
                self.input.clear();
            }
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Char('p' | 'i') if self.pending.is_none() => {
                self.kind = if key.code == KeyCode::Char('p') {
                    Kind::Pr
                } else {
                    Kind::Issue
                };
                self.rows.clear();
                self.selected = 0;
                self.refresh();
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.rows.len().saturating_sub(1))
            }
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(8),
            KeyCode::PageDown => {
                self.selected = self
                    .selected
                    .saturating_add(8)
                    .min(self.rows.len().saturating_sub(1))
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.rows.len().saturating_sub(1),
            KeyCode::Enter | KeyCode::Char('o') if self.browser.is_none() => {
                if let Some(row) = self.rows.get(self.selected) {
                    let url = row.url.clone();
                    let (tx, rx) = mpsc::sync_channel(1);
                    self.browser = Some(rx);
                    std::thread::spawn(move || {
                        let _ = tx.send(network::open_browser(&url));
                    });
                }
            }
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let mut lines = vec![
            Line::styled("GITHUB / COMMUNITY / READ ONLY", accent()),
            Line::from(format!(
                "Account: {}",
                self.user.as_deref().unwrap_or("public / signed out")
            )),
        ];
        if self.editing {
            lines.extend([
                Line::from(format!(
                    "Token: {}",
                    "•".repeat(
                        self.input
                            .chars()
                            .count()
                            .min(area.width.saturating_sub(10) as usize)
                    )
                )),
                Line::from("Enter verify/save / Esc cancel / Ctrl+U clear / Ctrl+L logout"),
                Line::from("Fine-grained token: only read access needed. No write actions exist."),
                Line::from("Reads GH_TOKEN, GITHUB_TOKEN, then ~/.cache/pssa/github/token (0600)."),
            ]);
        } else {
            lines.extend([
                Line::from("p PRs / i issues / r refresh / l login / Ctrl+L logout"),
                Line::from("Up/Down PgUp/PgDn Home/End select / Enter or o browser"),
            ]);
        }
        lines.push(Line::from(self.status.as_str()));
        if self.rows.is_empty() {
            lines.push(Line::from(
                "No items loaded. Public browsing does not require login.",
            ));
        }
        let count = area
            .height
            .saturating_sub(if self.editing { 13 } else { 10 })
            .max(1) as usize;
        let start = self.selected.saturating_sub(count.saturating_sub(1));
        for (i, item) in self.rows.iter().enumerate().skip(start).take(count) {
            lines.push(Line::styled(
                format!(
                    "{} {} #{} {}",
                    if i == self.selected { "▶" } else { " " },
                    self.kind.label(),
                    item.number,
                    item.title
                ),
                accent(),
            ));
        }
        if let Some(item) = self.rows.get(self.selected) {
            lines.push(Line::from(item.url.as_str()));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" GitHub ")),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use serde_json::json;
    struct Fake;
    impl Http for Fake {
        fn get(&self, url: &str, token: Option<&str>) -> Result<Value, String> {
            assert_eq!(token, Some("fixture-token"));
            if url.ends_with("/user") {
                Ok(json!({"login":"fixture-user"}))
            } else if url.starts_with("https://api.github.com/search/issues?") {
                assert!(url.contains("is%3Aissue%20is%3Aopen"));
                Ok(json!({"items":[{"number":18,"title":"An open issue"}]}))
            } else {
                assert!(url.starts_with(REPO));
                Ok(json!([{"number":17,"title":"A useful change"}]))
            }
        }
    }
    #[test]
    fn token_login_and_read_only_list_are_mockable() {
        assert_eq!(
            identity(&Fake, &Token("fixture-token".into())).unwrap(),
            "fixture-user"
        );
        let mut page = Github {
            token: Some(Token("fixture-token".into())),
            http: Arc::new(Fake),
            ..Github::default()
        };
        page.refresh();
        let reply = page
            .pending
            .take()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let Reply::List(rows) = reply else {
            panic!("not a list")
        };
        assert_eq!(
            rows[0].url,
            "https://github.com/Sparticle62ops/pssa/pull/17"
        );
        page.kind = Kind::Issue;
        page.refresh();
        let reply = page
            .pending
            .take()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let Reply::List(rows) = reply else {
            panic!("not an issue list")
        };
        assert_eq!(
            rows[0].url,
            "https://github.com/Sparticle62ops/pssa/issues/18"
        );
    }
    #[test]
    fn tokens_roundtrip_privately_and_logout_discards_late_reply() {
        let dir = std::env::temp_dir().join(format!("pssa-github-test-{}", std::process::id()));
        let path = dir.join("token");
        network::private_write(&path, b"fixture-token").unwrap();
        assert_eq!(resolve(None, &path).unwrap().unwrap().0, "fixture-token");
        assert_eq!(
            resolve(Some("env-token".into()), &path).unwrap().unwrap().0,
            "env-token"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let (tx, rx) = mpsc::sync_channel(1);
        let mut page = Github {
            pending: Some(rx),
            input: "fixture-token".into(),
            ..Github::default()
        };
        page.logout(Ok(path.clone()));
        assert!(
            tx.send(Ok(Reply::Login {
                name: "late".into(),
                token: Token("secret".into()),
                save: true
            }))
            .is_err()
        );
        assert!(page.user.is_none());
        assert!(page.input.is_empty());
        assert!(!path.exists());
        fs::remove_dir_all(dir).unwrap();
        assert!(Token::parse("bad\nheader".into()).is_err());
    }
    #[test]
    fn issue_list_filters_prs_and_never_trusts_remote_links_or_tokens() {
        let values = json!([
            {"number":2,"title":"issue\u{1b}[31m","html_url":"https://evil.test/"},
            {"number":3,"title":"PR","pull_request":{}},
            {"number":4,"title":"echo fixture-token"}
        ]);
        let rows = items(&values, Kind::Issue, Some("fixture-token")).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].url.ends_with("/issues/2"));
        assert!(!rows[0].title.contains('\u{1b}'));
        assert_eq!(rows[1].title, "[redacted title]");
    }
    #[test]
    fn github_screen_masks_tokens_wide_narrow_and_tiny() {
        for (w, h) in [(120, 28), (79, 24), (32, 18), (1, 1), (0, 0)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            let mut page = Github {
                editing: true,
                input: "fixture-token-NEVER-SHOW".into(),
                ..Github::default()
            };
            for editing in [true, false] {
                page.editing = editing;
                t.draw(|f| page.draw(f, f.area())).unwrap();
                let text: String = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(!text.contains("NEVER-SHOW"));
                assert!(!text.contains("fixture-token"));
                if w >= 79 {
                    assert!(text.contains("READ ONLY"));
                    if editing {
                        assert!(text.contains("•••••"));
                    }
                }
            }
        }
    }
}
