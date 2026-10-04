//! Release-only, opt-out update hints. Never downloads/builds/updates the app.
use super::{
    accent,
    network::{self, Http, Web},
    panel,
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    sync::{Arc, mpsc},
};

const API: &str = "https://api.github.com/repos/Sparticle62ops/pssa/releases/latest";
const DAY: u64 = 86_400;
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, PartialEq, Eq)]
struct Version {
    core: [u64; 3],
    pre: Vec<String>,
}
impl Version {
    fn parse(text: &str) -> Option<Self> {
        let text = text.strip_prefix('v').unwrap_or(text);
        let text = if let Some((core, build)) = text.split_once('+') {
            if build.split('.').any(|part| {
                part.is_empty() || !part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            }) {
                return None;
            }
            core
        } else {
            text
        };
        let (core, pre) = text
            .split_once('-')
            .map_or((text, None), |(a, b)| (a, Some(b)));
        let parts: Vec<_> = core.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let mut numbers = [0; 3];
        for (n, part) in numbers.iter_mut().zip(parts) {
            if part.is_empty()
                || !part.bytes().all(|b| b.is_ascii_digit())
                || (part.len() > 1 && part.starts_with('0'))
            {
                return None;
            }
            *n = part.parse().ok()?;
        }
        let mut identifiers = Vec::new();
        if let Some(pre) = pre {
            for part in pre.split('.') {
                if part.is_empty() || !part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                {
                    return None;
                }
                if part.bytes().all(|b| b.is_ascii_digit())
                    && part.len() > 1
                    && part.starts_with('0')
                {
                    return None;
                }
                identifiers.push(part.to_owned());
            }
        }
        Some(Self {
            core: numbers,
            pre: identifiers,
        })
    }
    fn compare(&self, other: &Self) -> Ordering {
        let core = self.core.cmp(&other.core);
        if core != Ordering::Equal {
            return core;
        }
        match (self.pre.is_empty(), other.pre.is_empty()) {
            (true, false) => return Ordering::Greater,
            (false, true) => return Ordering::Less,
            _ => {}
        }
        for (a, b) in self.pre.iter().zip(&other.pre) {
            let numeric = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
            let order = match (numeric(a), numeric(b)) {
                (true, true) => a.len().cmp(&b.len()).then_with(|| a.cmp(b)),
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => a.cmp(b),
            };
            if order != Ordering::Equal {
                return order;
            }
        }
        self.pre.len().cmp(&other.pre.len())
    }
}
pub(super) fn newer(candidate: &str, current: &str) -> bool {
    Version::parse(candidate)
        .zip(Version::parse(current))
        .is_some_and(|(a, b)| a.compare(&b) == Ordering::Greater)
}
fn due(last: u64, now: u64) -> bool {
    last == 0 || now.saturating_sub(last) >= DAY
}

#[derive(Clone)]
struct Release {
    version: String,
    url: String,
}
fn release(value: &Value) -> Result<Release, String> {
    if value["draft"].as_bool().unwrap_or(false) || value["prerelease"].as_bool().unwrap_or(false) {
        return Err("No stable release available".into());
    }
    let version = value["tag_name"]
        .as_str()
        .filter(|s| s.len() <= 100 && Version::parse(s).is_some())
        .ok_or("Release has no valid version")?;
    let url = value["html_url"]
        .as_str()
        .filter(|s| {
            s.starts_with("https://github.com/Sparticle62ops/pssa/releases/tag/")
                && s.len() < 2048
                && s.bytes().all(|b| b.is_ascii_graphic())
        })
        .ok_or("Release has no trusted URL")?;
    Ok(Release {
        version: version.into(),
        url: url.into(),
    })
}

pub(super) struct Update {
    enabled: bool,
    last_check: u64,
    latest: Option<Release>,
    status: String,
    http: Arc<dyn Http>,
    pending: Option<mpsc::Receiver<Result<Release, String>>>,
    browser: Option<mpsc::Receiver<Result<(), String>>>,
}
impl Default for Update {
    fn default() -> Self {
        Self {
            enabled: true,
            last_check: 0,
            latest: None,
            status: "Checks stable GitHub releases at most once per day".into(),
            http: Arc::new(Web),
            pending: None,
            browser: None,
        }
    }
}
impl Update {
    pub(super) fn load() -> Self {
        let v = network::load_config("updates");
        Self {
            enabled: v["enabled"].as_bool().unwrap_or(true),
            last_check: v["last_check"].as_u64().unwrap_or(0),
            latest: release(&v["release"]).ok(),
            ..Self::default()
        }
    }
    fn config(&self) -> Value {
        json!({"enabled":self.enabled,"last_check":self.last_check,
            "release":self.latest.as_ref().map(|r| json!({"tag_name":r.version,"html_url":r.url}))})
    }
    fn check(&mut self) {
        if !self.enabled {
            self.status = "Update checks disabled".into();
            return;
        }
        if network::offline() {
            self.status = "Offline / update check skipped".into();
            return;
        }
        if self.pending.is_some() {
            return;
        }
        let now = network::now();
        if !due(self.last_check, now) {
            self.status = "Already checked today; retry after 24 hours".into();
            return;
        }
        let previous = self.last_check;
        self.last_check = now;
        // Persist the attempt BEFORE any request, including failed attempts. A
        // restart/manual refresh must not circumvent the daily request limit.
        if let Err(e) = network::save_config("updates", &self.config()) {
            self.last_check = previous;
            self.status = format!("Check skipped: {e}");
            return;
        }
        self.start();
    }
    fn start(&mut self) {
        let (tx, rx) = mpsc::sync_channel(1);
        self.pending = Some(rx);
        self.status = "Checking stable release…".into();
        let http = Arc::clone(&self.http);
        std::thread::spawn(move || {
            let _ = tx.send(http.get(API, None).and_then(|v| release(&v)));
        });
    }
    pub(super) fn poll(&mut self) {
        if let Some(result) = self.pending.as_ref().and_then(|r| r.try_recv().ok()) {
            self.pending = None;
            match result {
                Ok(latest) => {
                    self.status = if newer(&latest.version, VERSION) {
                        "Update available / o opens release"
                    } else {
                        "You are on the latest stable version"
                    }
                    .into();
                    self.latest = Some(latest);
                    if let Err(e) = network::save_config("updates", &self.config()) {
                        self.status = e;
                    }
                }
                Err(_) => self.status = "Release check unavailable / will retry tomorrow".into(),
            }
        }
        if let Some(result) = self.browser.as_ref().and_then(|r| r.try_recv().ok()) {
            self.browser = None;
            self.status = result
                .map(|_| "Release opened in browser".into())
                .unwrap_or_else(|e| e);
        }
        if self.enabled && self.pending.is_none() && due(self.last_check, network::now()) {
            // Failure to persist config must not retry disk I/O at frame rate.
            if !self.status.starts_with("Check skipped:") {
                self.check();
            }
        }
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') => self.check(),
            KeyCode::Char('d') => {
                self.enabled = !self.enabled;
                if !self.enabled {
                    self.pending = None;
                }
                self.status = if self.enabled {
                    "Update checks enabled"
                } else {
                    "Update checks disabled"
                }
                .into();
                if let Err(e) = network::save_config("updates", &self.config()) {
                    self.status = e;
                }
            }
            KeyCode::Enter | KeyCode::Char('o') if self.browser.is_none() => {
                if let Some(r) = &self.latest {
                    let url = r.url.clone();
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
        f.render_widget(Paragraph::new(vec![
            Line::styled("UPDATES / STABLE RELEASES", accent()),
            Line::from(format!("Installed: {VERSION}")),
            Line::from(format!("Daily checks: {}", if self.enabled { "ON" } else { "OFF" })),
            Line::from(format!("Latest: {}", self.latest.as_ref().map_or("not checked", |r| r.version.as_str()))),
            Line::from(self.latest.as_ref().map_or("", |r| r.url.as_str())),
            Line::from(self.status.as_str()),
            Line::from(""),
            Line::from("r check (daily limit) / d toggle checks / o open release"),
            Line::from("No auto-update, git changes, downloads or builds. Offline failures stay quiet."),
        ]).wrap(Wrap { trim: false }).block(panel(" updates ")), area);
    }
    pub(super) fn banner(&self, f: &mut Frame) {
        let area = f.area();
        if !self.enabled || area.height < 10 || network::offline() {
            return;
        }
        if let Some(r) = self.latest.as_ref().filter(|r| newer(&r.version, VERSION)) {
            let line = Rect::new(area.x, area.bottom() - 2, area.width, 1);
            f.render_widget(Clear, line);
            f.render_widget(
                Paragraph::new(format!("[ UPDATE {} ] Ctrl+K → Open updates", r.version))
                    .style(accent()),
                line,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    #[test]
    fn semantic_versions_compare_numbers_prereleases_and_build_metadata() {
        for (a, b) in [
            ("v0.10.0", "0.9.9"),
            ("1.0.0", "1.0.0-rc.9"),
            ("1.0.0-rc.10", "1.0.0-rc.2"),
            ("1.0.0-beta", "1.0.0-9"),
            ("1.0.0-rc.1.1", "1.0.0-rc.1"),
        ] {
            assert!(newer(a, b), "{a} > {b}");
            assert!(!newer(b, a));
        }
        for a in [
            "0.5.0",
            "v0.5.0+build",
            "junk",
            "1.0",
            "1.0.0-",
            "01.0.0",
            "1.0.0-01",
            "1.0.0+bad\u{1b}[31m",
            "1.0.0+",
            "1.0.0+a..b",
        ] {
            assert!(!newer(a, "0.5.0"), "{a}");
        }
    }
    #[test]
    fn daily_limit_includes_failures_restarts_and_clock_rollback() {
        assert!(due(0, 100));
        assert!(!due(100, 100));
        assert!(!due(100, 99));
        assert!(!due(100, 100 + DAY - 1));
        assert!(due(100, 100 + DAY));
        assert!(!due(
            json!({"last_check":100})["last_check"].as_u64().unwrap(),
            101
        ));
    }
    struct Fake;
    impl Http for Fake {
        fn get(&self, url: &str, token: Option<&str>) -> Result<Value, String> {
            assert_eq!(url, API);
            assert!(token.is_none());
            Ok(
                json!({"tag_name":"v0.6.0","html_url":"https://github.com/Sparticle62ops/pssa/releases/tag/v0.6.0"}),
            )
        }
    }
    #[test]
    fn release_worker_is_mockable_and_rejects_untrusted_links() {
        let mut u = Update {
            http: Arc::new(Fake),
            ..Update::default()
        };
        u.start();
        let release = u
            .pending
            .take()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(release.version, "v0.6.0");
        assert!(
            super::release(&json!({"tag_name":"v0.6.0","html_url":"https://evil.test/"})).is_err()
        );
    }
    #[test]
    fn update_screen_and_banner_render_wide_narrow_and_tiny() {
        for (w, h) in [(120, 28), (79, 24), (35, 18), (1, 1), (0, 0)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            let u = Update {
                latest: Some(Release {
                    version: "v9.0.0".into(),
                    url: "https://github.com/Sparticle62ops/pssa/releases/tag/v9.0.0".into(),
                }),
                ..Update::default()
            };
            t.draw(|f| {
                u.draw(f, f.area());
                u.banner(f);
            })
            .unwrap();
            let text: String = t
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            if w >= 79 {
                assert!(text.contains("0.5.0"));
                assert!(text.contains("v9.0.0"));
            }
            assert!(u.pending.is_none());
        }
    }
}
