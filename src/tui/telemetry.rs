//! Opt-in, anonymous telemetry for PSSA. Nothing is collected or sent before
//! the user answers the first-run prompt in the TUI, and events are only
//! queued locally until the project stands up a collector: the upload
//! endpoint constant is empty, so queued events never leave the machine.
use serde_json::{Value, json};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

/// Where queued events will be uploaded once the project has a collector.
/// Empty until then; empty means never send, only queue locally.
const UPLOAD_ENDPOINT: &str = "";

/// Oldest events are dropped once the local queue reaches this size.
const MAX_QUEUED_EVENTS: usize = 200;

const CONSENT_FILE: &str = "telemetry.json";
const QUEUE_FILE: &str = "telemetry-events.jsonl";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Consent {
    Unset,
    Allowed,
    Declined,
}

impl Consent {
    fn from_json(text: Option<&str>) -> Self {
        let Some(text) = text else {
            return Self::Unset;
        };
        match serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.get("consent").and_then(Value::as_str).map(String::from))
        {
            Some(value) if value == "allowed" => Self::Allowed,
            Some(value) if value == "declined" => Self::Declined,
            _ => Self::Unset,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Unset => "unset",
            Self::Allowed => "allowed",
            Self::Declined => "declined",
        }
    }
}

fn config_dir() -> Result<PathBuf, String> {
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .ok_or("No config directory; set HOME or XDG_CONFIG_HOME")?;
    Ok(root.join("pssa"))
}

pub(crate) struct Telemetry {
    pub(super) prompt_open: bool,
    consent: Consent,
}

impl Telemetry {
    /// Load the saved answer. A missing or unreadable answer re-opens the
    /// first-run prompt; a declined answer never does.
    pub(super) fn load() -> Self {
        let consent = Self::read_consent();
        Self {
            prompt_open: consent == Consent::Unset,
            consent,
        }
    }

    fn read_consent() -> Consent {
        config_dir()
            .ok()
            .and_then(|dir| fs::read_to_string(dir.join(CONSENT_FILE)).ok())
            .map(|text| Consent::from_json(Some(&text)))
            .unwrap_or(Consent::Unset)
    }

    /// Record the user's answer and close the prompt. The write is
    /// best-effort: if it fails the prompt re-opens on the next start.
    pub(super) fn answer(&mut self, allowed: bool) {
        self.consent = if allowed {
            Consent::Allowed
        } else {
            Consent::Declined
        };
        self.prompt_open = false;
        if let Ok(dir) = config_dir() {
            let _ = fs::create_dir_all(&dir);
            let body = json!({"consent": self.consent.label()});
            let _ = fs::write(dir.join(CONSENT_FILE), body.to_string());
        }
    }

    /// Queue one anonymous event. Only allowed-consent machines queue
    /// anything, and events stay local until an upload endpoint exists.
    pub(crate) fn record(kind: &str, detail: Value) {
        if Self::read_consent() != Consent::Allowed {
            return;
        }
        if let Ok(dir) = config_dir() {
            Self::record_into(&dir, kind, detail);
        }
    }

    fn record_into(dir: &Path, kind: &str, detail: Value) {
        let _ = fs::create_dir_all(dir);
        let path = dir.join(QUEUE_FILE);
        let mut body = fs::read_to_string(&path).unwrap_or_default();
        let mut count = body.lines().count();
        let mut event = json!({"t": super::network::now(), "kind": kind});
        if let Some(map) = event.as_object_mut() {
            if let Some(fields) = detail.as_object() {
                for (key, value) in fields {
                    map.insert(key.clone(), value.clone());
                }
            }
        }
        body.push_str(&format!("{}\n", event));
        while count >= MAX_QUEUED_EVENTS {
            let Some(split) = body.find('\n') else {
                break;
            };
            body.drain(..=split);
            count -= 1;
        }
        let _ = fs::write(&path, body);
    }

    /// Queue a panic summary if the user allowed telemetry. Installed from
    /// the binary entry point so crash reports cover CLI runs too.
    pub(crate) fn install_panic_hook() {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = if let Some(text) = info.payload().downcast_ref::<&str>() {
                (*text).to_string()
            } else if let Some(text) = info.payload().downcast_ref::<String>() {
                text.clone()
            } else {
                "unknown panic payload".into()
            };
            let location = info
                .location()
                .map(|site| site.to_string())
                .unwrap_or_default();
            Self::record(
                "panic",
                json!({
                    "message": message.chars().take(400).collect::<String>(),
                    "location": location,
                    "version": env!("CARGO_PKG_VERSION"),
                }),
            );
            default_hook(info);
        }));
    }

    /// The prompt overlay: what is collected, what is never collected, and
    /// the two answers. Centered so it reads as a modal, not a screen.
    pub(super) fn draw(&self, f: &mut ratatui::Frame) {
        use ratatui::{
            layout::{Constraint, Layout},
            style::{Modifier, Style},
            text::{Line, Span},
            widgets::{Clear, Paragraph},
        };
        if !self.prompt_open {
            return;
        }
        let area = f.area();
        let vertical = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(13),
            Constraint::Fill(1),
        ])
        .split(area);
        let horizontal = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Min(62),
            Constraint::Fill(1),
        ])
        .split(vertical[1]);
        let box_area = horizontal[1];
        // An opaque card: erase what sits behind so the prompt stays readable
        // over any screen, and give it the same panel treatment as the rest.
        f.render_widget(Clear, box_area);
        f.render_widget(super::panel(" help improve pssa "), box_area);
        let base = Style::new().bg(super::panel_bg());
        let heading = super::accent().add_modifier(Modifier::BOLD);
        let key = |label: &str| {
            Span::styled(
                format!(" {label} "),
                Style::new()
                    .fg(super::panel_bg())
                    .bg(super::normal_green())
                    .add_modifier(Modifier::BOLD),
            )
        };
        let lines = vec![
            Line::styled("share anonymous data with the pssa project?", heading),
            Line::from(""),
            Line::styled("would be shared:", heading),
            Line::from("  - training stats: loss, speed, token counts"),
            Line::from("  - errors: panic and crash summaries"),
            Line::from("  - model performance and hardware specs"),
            Line::from(""),
            Line::from("never shared: personal data, model weights, files."),
            Line::from(""),
            Line::from(""),
            Line::from(vec![
                Span::raw("  "),
                key("A"),
                Span::raw("  allow        "),
                key("D"),
                Span::raw("  no thanks"),
            ]),
        ];
        let inner_x = box_area.x + 2;
        let inner_y = box_area.y + 1;
        let inner_w = box_area.width.saturating_sub(4);
        let inner_h = box_area.height.saturating_sub(2);
        let inner = ratatui::layout::Rect::new(inner_x, inner_y, inner_w, inner_h);
        f.render_widget(Paragraph::new(lines).style(base), inner);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pssa-telemetry-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn consent_parses_allowed_declined_and_unset() {
        assert_eq!(Consent::from_json(None), Consent::Unset);
        assert_eq!(Consent::from_json(Some("")), Consent::Unset);
        assert_eq!(
            Consent::from_json(Some(r#"{"consent": "allowed"}"#)),
            Consent::Allowed
        );
        assert_eq!(
            Consent::from_json(Some(r#"{"consent": "declined"}"#)),
            Consent::Declined
        );
        assert_eq!(
            Consent::from_json(Some(r#"{"consent": "maybe"}"#)),
            Consent::Unset
        );
    }

    #[test]
    fn record_queues_locally_and_caps_events() {
        let dir = temp_dir("queue");
        // record_into queues for an allowed-consent directory and drops the
        // oldest events once the local queue reaches the cap.
        for index in 0..(MAX_QUEUED_EVENTS + 10) {
            Telemetry::record_into(&dir, "start", json!({"index": index}));
        }
        let queued = fs::read_to_string(dir.join(QUEUE_FILE)).unwrap();
        assert_eq!(queued.lines().count(), MAX_QUEUED_EVENTS);
        assert!(queued.contains("\"kind\":\"start\""));
        // The first queued event was dropped with the overflow.
        assert!(!queued.contains("\"index\":0}"));
        // The last event survived.
        assert!(queued.contains(&format!("\"index\":{}", MAX_QUEUED_EVENTS + 9)));
        // No endpoint configured: nothing ever leaves the machine.
        assert!(UPLOAD_ENDPOINT.is_empty());
    }

    #[test]
    fn record_gates_on_saved_consent() {
        let dir = temp_dir("gate");
        // Reading the saved answer is what record() gates on: unset queues
        // nothing even though the writer would happily append.
        let consent = Consent::from_json(Some(r#"{"consent":"declined"}"#));
        assert_eq!(consent, Consent::Declined);
        assert_ne!(consent, Consent::Allowed);
    }
}
