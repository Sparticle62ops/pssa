//! Own terminal modes in one place, including panic=abort release builds.
use crossterm::{
    cursor::Show,
    event::{DisableFocusChange, DisableMouseCapture, EnableFocusChange, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use std::{
    io,
    sync::{
        Once,
        atomic::{AtomicBool, Ordering},
    },
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

pub(super) struct Session;

impl Session {
    fn arm() -> Self {
        PANIC_HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                // Drop does not run with panic=abort. Restore before invoking
                // the original hook, so diagnostics appear in the normal screen.
                restore();
                previous(info);
            }));
        });
        ACTIVE.store(true, Ordering::SeqCst);
        Self
    }

    pub(super) fn start() -> io::Result<(Self, ratatui::DefaultTerminal)> {
        let session = Self::arm();
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        let terminal =
            ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(io::stdout()))?;
        // Unsupported terminals ignore these requests: no mouse events means
        // a static grid. A failed request must not prevent keyboard-only use.
        if execute!(io::stdout(), EnableMouseCapture, EnableFocusChange).is_err() {
            let _ = execute!(io::stdout(), DisableMouseCapture, DisableFocusChange);
        }
        Ok((session, terminal))
    }
}

fn restore() {
    if ACTIVE.swap(false, Ordering::SeqCst) {
        // Attempt each cleanup independently even if an earlier write fails.
        let _ = execute!(io::stdout(), DisableMouseCapture);
        let _ = execute!(io::stdout(), DisableFocusChange);
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), Show);
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        restore();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn terminal_cleanup_fixture() {
        let Some(mode) = crate::env_var_os("PSSA_TUI_CLEANUP_FIXTURE", "OXIDE_TUI_CLEANUP_FIXTURE")
            .and_then(|value| value.into_string().ok())
        else {
            return;
        };
        let _session = super::Session::arm();
        if mode == "panic" {
            panic!("terminal cleanup fixture");
        }
        if mode == "error" {
            fn fail() -> std::io::Result<()> {
                let _session = super::Session;
                Err(std::io::Error::other("fixture"))
            }
            assert!(fail().is_err());
        }
    }

    #[test]
    fn mouse_capture_restored_on_exit_error_and_panic() {
        // Subprocesses isolate the global panic hook and stdout from the runner.
        for mode in ["exit", "error", "panic"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tui::session::tests::terminal_cleanup_fixture",
                    "--nocapture",
                ])
                .env("PSSA_TUI_CLEANUP_FIXTURE", mode)
                .output()
                .unwrap();
            assert_eq!(output.status.success(), mode != "panic");
            let text = String::from_utf8_lossy(&output.stdout);
            for code in [
                "\x1b[?1000l",
                "\x1b[?1002l",
                "\x1b[?1003l",
                "\x1b[?1006l",
                "\x1b[?1004l",
                "\x1b[?25h",
                "\x1b[?1049l",
            ] {
                assert!(text.contains(code), "{mode}: missing {code:?}: {text:?}");
            }
            assert!(text.find("\x1b[?1000l").unwrap() < text.find("\x1b[?1049l").unwrap());
        }
    }
}
