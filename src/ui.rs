//! Terminal presentation for the oxide command line.
//!
//! Colour, cursor control and screen clearing are emitted only when stdout is
//! an interactive terminal and `NO_COLOR` is unset, so piped output, CI logs
//! and Kaggle notebook logs stay plain parseable text. Every machine-readable
//! `key=value` line the CLI has always printed keeps its exact spelling; the
//! helpers here add presentation around those lines rather than replacing
//! them.

use std::io::{self, IsTerminal, Write};
use std::sync::OnceLock;
use std::time::Instant;

static COLOR: OnceLock<bool> = OnceLock::new();

/// True when it is safe to emit ANSI escapes on stdout.
pub fn color_enabled() -> bool {
    *COLOR.get_or_init(|| io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
}

fn paint(code: &str, text: &str) -> String {
    if color_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

pub fn bold(t: &str) -> String {
    paint("1", t)
}
pub fn dim(t: &str) -> String {
    paint("2", t)
}
pub fn cyan(t: &str) -> String {
    paint("36", t)
}
pub fn green(t: &str) -> String {
    paint("32", t)
}
pub fn yellow(t: &str) -> String {
    paint("33", t)
}
pub fn red(t: &str) -> String {
    paint("31", t)
}
pub fn magenta(t: &str) -> String {
    paint("35", t)
}

/// Visible width of a string, ignoring ANSI escape sequences.
pub fn visible_len(s: &str) -> usize {
    let mut n = 0usize;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            n += 1;
        }
    }
    n
}

/// Inner width of framed output. `COLUMNS` when the shell exports it,
/// otherwise a comfortable default, always clamped to something printable.
pub fn width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.trim().parse::<usize>().ok())
        .unwrap_or(78)
        .clamp(58, 96)
}

/// Home the cursor and wipe the screen. No-op unless interactive.
pub fn clear_screen() {
    if color_enabled() {
        print!("\x1b[2J\x1b[H");
        let _ = io::stdout().flush();
    }
}

fn pad(text: &str, to: usize) -> String {
    let len = visible_len(text);
    if len >= to {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(to - len))
    }
}

/// `╭─ title ──────╮`
pub fn panel_top(title: &str) {
    let inner = width() - 4;
    let head = if title.is_empty() {
        "─".repeat(inner + 2)
    } else {
        let label = format!("─ {} ", bold(&cyan(title)));
        let used = visible_len(&label);
        format!("{label}{}", "─".repeat((inner + 2).saturating_sub(used)))
    };
    println!("  {}", dim(&format!("╭{head}╮")));
}

/// A line of content inside the current panel.
pub fn panel_row(text: &str) {
    let inner = width() - 4;
    println!("  {} {} {}", dim("│"), pad(text, inner), dim("│"));
}

pub fn panel_blank() {
    panel_row("");
}

/// `label  value` aligned inside a panel.
pub fn panel_field(label: &str, value: &str) {
    panel_row(&format!("{}{}", pad(&dim(label), 16), value));
}

pub fn panel_bottom() {
    let inner = width() - 4;
    println!("  {}", dim(&format!("╰{}╯", "─".repeat(inner + 2))));
}

/// Block-letter wordmark, drawn once at the top of the home screen.
pub fn logo() {
    const ART: [&str; 5] = [
        " ██████  ██   ██ ██ ██████  ███████",
        "██    ██  ██ ██  ██ ██   ██ ██     ",
        "██    ██   ████  ██ ██   ██ █████  ",
        "██    ██  ██ ██  ██ ██   ██ ██     ",
        " ██████  ██   ██ ██ ██████  ███████",
    ];
    println!();
    for line in ART {
        println!("   {}", cyan(line));
    }
}

pub fn rule() {
    println!("  {}", dim(&"─".repeat(width() - 2)));
}

/// The title block printed once at the start of a long command.
pub fn banner(command: &str, subtitle: &str) {
    let name = format!("oxide {command}");
    println!();
    println!("  {}  {}", bold(&cyan(&name)), dim(subtitle));
    println!("  {}", dim(&"─".repeat(width() - 2)));
}

/// A `label  value` row inside a banner block, label column padded to 16.
pub fn field(label: &str, value: &str) {
    println!("  {:<16}{}", dim(label), value);
}

pub fn section(title: &str) {
    println!();
    println!("  {}", bold(title));
}

pub fn success(text: &str) {
    println!("  {} {}", green("✓"), text);
}

pub fn warn(text: &str) {
    eprintln!("  {} {}", yellow("!"), text);
}

pub fn failure(text: &str) {
    eprintln!("  {} {}", red("✗"), text);
}

pub fn note(text: &str) {
    println!("  {}", dim(text));
}

/// `1234567` renders as `1,234,567`.
pub fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Bytes as `2.1 MB`, `914 kB` or `320 B`.
pub fn bytes(n: u64) -> String {
    const UNIT: [(u64, &str); 3] = [(1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "kB")];
    for (scale, suffix) in UNIT {
        if n >= scale {
            return format!("{:.1} {suffix}", n as f64 / scale as f64);
        }
    }
    format!("{n} B")
}

/// Seconds as `2h 14m 09s`, `14m 09s` or `9.4s`.
pub fn duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".into();
    }
    let total = seconds as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{seconds:.1}s")
    }
}

/// A single step of work with an indeterminate length.
///
/// Interactively this animates a braille spinner in place and resolves to a
/// tick with the elapsed time. In a log it prints one plain `step … done`
/// line, so a headless run stays greppable.
pub struct Step {
    label: String,
    started: Instant,
    frame: usize,
    interactive: bool,
}

impl Step {
    pub fn start(label: &str) -> Self {
        let mut step = Self {
            label: label.to_string(),
            started: Instant::now(),
            frame: 0,
            interactive: color_enabled(),
        };
        if step.interactive {
            step.tick();
        } else {
            println!("  {} …", label);
        }
        step
    }

    /// Repaint the spinner. Cheap enough to call inside a polling loop.
    pub fn tick(&mut self) {
        if !self.interactive {
            return;
        }
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        self.frame = (self.frame + 1) % FRAMES.len();
        print!("\r  {} {}   ", cyan(FRAMES[self.frame]), dim(&self.label));
        let _ = io::stdout().flush();
    }

    pub fn done(self, detail: &str) {
        let elapsed = dim(&format!(
            "({})",
            duration(self.started.elapsed().as_secs_f64())
        ));
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!(" {}", dim(detail))
        };
        if self.interactive {
            print!("\r{}\r", " ".repeat(width()));
        }
        println!("  {} {}{tail} {elapsed}", green("✓"), self.label);
        let _ = io::stdout().flush();
    }
}

/// A progress display for a loop with a known number of steps.
///
/// On a terminal this repaints a single bar in place. Everywhere else it
/// prints a plain checkpoint line at a fixed interval, so a six-hour headless
/// run leaves a readable log instead of thousands of carriage returns.
pub struct Progress {
    label: String,
    total: usize,
    started: Instant,
    last_draw: Instant,
    tokens: usize,
    interactive: bool,
    drawn: bool,
}

impl Progress {
    pub fn new(label: &str, total: usize) -> Self {
        Self {
            label: label.to_string(),
            total: total.max(1),
            started: Instant::now(),
            last_draw: Instant::now(),
            tokens: 0,
            interactive: color_enabled(),
            drawn: false,
        }
    }

    /// Record `token_delta` freshly processed tokens at step `done`.
    pub fn update(&mut self, done: usize, token_delta: usize, loss: f64) {
        self.tokens += token_delta;
        let min_gap = if self.interactive { 0.08 } else { 30.0 };
        if self.last_draw.elapsed().as_secs_f64() < min_gap && done < self.total {
            return;
        }
        self.last_draw = Instant::now();

        let elapsed = self.started.elapsed().as_secs_f64();
        let fraction = (done as f64 / self.total as f64).clamp(0.0, 1.0);
        let rate = if elapsed > 0.0 {
            self.tokens as f64 / elapsed
        } else {
            0.0
        };
        let remaining = if fraction > 0.0 {
            elapsed / fraction - elapsed
        } else {
            f64::NAN
        };

        if self.interactive {
            let bar_width = (width() as i64 - 52).clamp(16, 40) as usize;
            let filled = (fraction * bar_width as f64).round() as usize;
            let bar = format!(
                "{}{}",
                cyan(&"█".repeat(filled)),
                dim(&"░".repeat(bar_width - filled))
            );
            print!(
                "\r  {} {bar} {:>3.0}%  loss {:.4}  {:.0} tok/s  eta {}   ",
                dim(&self.label),
                fraction * 100.0,
                loss,
                rate,
                duration(remaining)
            );
            let _ = io::stdout().flush();
            self.drawn = true;
        } else {
            println!(
                "  {} {}/{} ({:.0}%) loss={loss:.6} tokens_per_second={rate:.0} eta={}",
                self.label,
                done,
                self.total,
                fraction * 100.0,
                duration(remaining)
            );
        }
    }

    /// Clear the in-place bar so the next line starts clean.
    pub fn finish(&mut self) {
        if self.interactive && self.drawn {
            print!("\r{}\r", " ".repeat(width() + 20));
            let _ = io::stdout().flush();
        }
        self.drawn = false;
    }
}
