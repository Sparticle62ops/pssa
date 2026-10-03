//! Terminal presentation for the oxide command line.
//!
//! Colour, cursor control and screen clearing are emitted only when stdout is
//! an interactive terminal and `NO_COLOR` is unset, so piped output, CI logs
//! and Kaggle notebook logs stay plain parseable text. Every machine-readable
//! `key=value` line the CLI has always printed keeps its exact spelling; the
//! helpers here add presentation around those lines rather than replacing
//! them.

use std::cell::Cell;
use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::sync::OnceLock;
use std::time::Instant;

static COLOR: OnceLock<bool> = OnceLock::new();
thread_local! { static PLAIN: Cell<bool> = const { Cell::new(false) }; }

/// Scoped presentation policy: no process-wide environment changes and no
/// leakage between library calls or tests on different threads.
pub struct PlainOutput(bool);
pub fn plain_output(enabled: bool) -> PlainOutput {
    PlainOutput(PLAIN.with(|plain| plain.replace(plain.get() || enabled)))
}
impl Drop for PlainOutput {
    fn drop(&mut self) {
        PLAIN.with(|plain| plain.set(self.0));
    }
}

/// True when it is safe to emit ANSI escapes on stdout.
pub fn color_enabled() -> bool {
    !PLAIN.with(Cell::get)
        && *COLOR.get_or_init(|| {
            io::stdout().is_terminal()
                && std::env::var_os("NO_COLOR").is_none()
                && std::env::var("TERM").is_ok_and(|term| term != "dumb")
        })
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

/// A bounded inline training dashboard. No raw mode, alternate screen or
/// input interception: Ctrl+C keeps its usual meaning. Headless output emits
/// a first, five-second and final sample, always as newline-delimited text.
pub struct Progress {
    label: String,
    total: usize,
    started: Instant,
    last_draw: Instant,
    tokens: usize,
    loss_history: VecDeque<(f64, usize)>,
    learning_rate: Option<f32>,
    checkpoint_path: Option<String>,
    last_checkpoint: Option<String>,
    prior_updates: usize,
    memory_occupancy: Option<(usize, usize)>,
    // Optional Hugging Face feed metadata. These fields are presentation-only
    // and are emitted as additional key=value fields, so older log consumers
    // can ignore them unchanged.
    feed_dataset: Option<String>,
    feed_config: Option<String>,
    feed_split: Option<String>,
    feed_field: Option<String>,
    feed_rows: Option<usize>,
    feed_tokens: Option<usize>,
    feed_row: Option<usize>,
    feed_snippet: Option<String>,
    feed_token_ids: Option<String>,
    interactive: bool,
    emitted: bool,
    drawn_rows: u16,
}

impl Progress {
    pub fn new(label: &str, total: usize) -> Self {
        Self::new_with_tui(label, total, true)
    }

    /// Construct a progress display, optionally disabling cursor control even
    /// when stdout is a terminal.  This is the implementation behind
    /// `--no-tui`: the run still emits useful, rate-limited plain log lines.
    pub fn new_with_tui(label: &str, total: usize, tui: bool) -> Self {
        Self {
            label: label.to_string(),
            total: total.max(1),
            started: Instant::now(),
            last_draw: Instant::now(),
            tokens: 0,
            loss_history: VecDeque::with_capacity(8),
            learning_rate: None,
            checkpoint_path: None,
            last_checkpoint: None,
            prior_updates: 0,
            memory_occupancy: None,
            feed_dataset: None,
            feed_config: None,
            feed_split: None,
            feed_field: None,
            feed_rows: None,
            feed_tokens: None,
            feed_row: None,
            feed_snippet: None,
            feed_token_ids: None,
            interactive: tui && color_enabled(),
            emitted: false,
            drawn_rows: 0,
        }
    }

    /// Set the checkpoint destination shown by the live dashboard. This is a
    /// pending target until `checkpoint_saved` is emitted after a successful
    /// writer return.
    pub fn set_checkpoint_path(&mut self, path: &str) {
        self.checkpoint_path = Some(path.to_string());
    }

    pub fn set_prior_updates(&mut self, prior: usize) {
        self.prior_updates = prior;
    }

    /// Show the checkpoint that this run resumed from until a newer save is
    /// reported. This is intentionally presentation-only and is never stored
    /// in a checkpoint.
    pub fn set_last_checkpoint(&mut self, path: &str) {
        let path = path.trim();
        if !path.is_empty() && path != "-" {
            self.last_checkpoint = Some(path.to_string());
        }
    }

    /// Set the latest optimizer learning rate and optional memory-bank
    /// occupancy shown on the next update.
    pub fn set_metrics(&mut self, learning_rate: Option<f32>, memory: Option<(usize, usize)>) {
        self.learning_rate = learning_rate.filter(|x| x.is_finite() && *x > 0.0);
        self.memory_occupancy =
            memory.filter(|(used, capacity)| *used <= *capacity && *capacity > 0);
    }

    /// Set the optional source/sample metadata shown by the feed tab. The
    /// sample is deliberately short and the IDs are already formatted by the
    /// caller so the progress logger never needs to know tokenizer details.
    pub fn set_feed(
        &mut self,
        dataset: &str,
        config: Option<&str>,
        split: &str,
        field: &str,
        rows: usize,
        tokens: usize,
        row: usize,
        snippet: &str,
        token_ids: &str,
    ) {
        self.feed_dataset = Some(dataset.to_string());
        self.feed_config = config.map(str::to_string);
        self.feed_split = Some(split.to_string());
        self.feed_field = Some(field.to_string());
        self.feed_rows = Some(rows);
        self.feed_tokens = Some(tokens);
        self.feed_row = Some(row);
        self.feed_snippet = Some(snippet.to_string());
        self.feed_token_ids = Some(token_ids.to_string());
    }

    /// Let optional presentation work share the logger's existing throttle.
    pub(crate) fn should_emit(&self, done: usize) -> bool {
        let min_gap = if self.interactive { 0.2 } else { 5.0 };
        !self.emitted || self.last_draw.elapsed().as_secs_f64() >= min_gap || done >= self.total
    }

    /// Record `token_delta` freshly processed tokens at step `done`.
    pub fn update(&mut self, done: usize, token_delta: usize, loss: f64) {
        self.update_with_metrics(done, token_delta, loss, None, None);
    }

    /// Record progress and the metrics that are useful when a run is left
    /// unattended.  The last eight finite losses form the short moving
    /// average, while the raw loss remains available for machine parsers.
    pub fn update_with_metrics(
        &mut self,
        done: usize,
        token_delta: usize,
        loss: f64,
        learning_rate: Option<f32>,
        memory: Option<(usize, usize)>,
    ) {
        self.tokens += token_delta;
        self.set_metrics(learning_rate, memory);
        if loss.is_finite() {
            if self.loss_history.len() == 8 {
                self.loss_history.pop_front();
            }
            self.loss_history.push_back((loss, token_delta));
        }
        if !self.should_emit(done) {
            return;
        }
        self.last_draw = Instant::now();
        self.emitted = true;

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
        let average = self.loss_average().unwrap_or(loss);
        let remaining_updates = self.total.saturating_sub(done);
        let lr = self
            .learning_rate
            .map(|x| format!("{x:.6e}"))
            .unwrap_or_else(|| "-".into());
        let memory = self
            .memory_occupancy
            .map(|(used, capacity)| format!("{used}/{capacity}"))
            .unwrap_or_else(|| "-".into());
        let checkpoint = self.checkpoint_path.clone().unwrap_or_else(|| "-".into());
        let number = self
            .checkpoint_path
            .as_deref()
            .and_then(checkpoint_number)
            .map(|number| number.to_string())
            .unwrap_or_else(|| "-".into());
        let last_checkpoint = self.last_checkpoint.clone().unwrap_or_else(|| "-".into());
        let last_checkpoint_field = self
            .last_checkpoint
            .as_deref()
            .map(|path| format!(" last_checkpoint={path}"))
            .unwrap_or_default();
        let feed_fields = self.feed_fields();
        let global = self.prior_updates.saturating_add(done);

        if self.interactive {
            self.clear();
            let mut rows = vec![
                format!(
                    "{} {:>3.0}% | updates {done}/{} | {remaining_updates} remaining (global {global})",
                    self.label,
                    fraction * 100.0,
                    self.total
                ),
                format!("loss {loss:.6} | avg(8, token-weighted) {average:.6}"),
                format!(
                    "{rate:.0} tokens/s | ETA {} | elapsed {}",
                    duration(remaining),
                    duration(elapsed)
                ),
                format!("learning rate {lr} | memory bank {memory}"),
                format!("checkpoint #{number} pending: {checkpoint}"),
                format!("last checkpoint: {last_checkpoint}"),
            ];
            if let Some(dataset) = self.feed_dataset.as_deref() {
                rows.push(format!(
                    "feed {dataset} selected row {} / {} rows consumed / {} tokens: {}",
                    self.feed_row.unwrap_or(0),
                    self.feed_rows.unwrap_or(0),
                    self.feed_tokens.unwrap_or(0),
                    terminal_text(self.feed_snippet.as_deref().unwrap_or("-"))
                ));
            }
            let (columns, height) = crossterm::terminal::size().unwrap_or((80, 24));
            // Never wrap a row; leave the final column free. ratatui truncates
            // by display width (including wide Unicode paths), not bytes.
            let mut buffer = ratatui::buffer::Buffer::empty(ratatui::layout::Rect::new(
                0,
                0,
                columns.saturating_sub(1),
                1,
            ));
            for row in rows.iter().take(usize::from(height.saturating_sub(1))) {
                buffer.reset();
                buffer.set_string(0, 0, row, ratatui::style::Style::default());
                let text: String = buffer.content.iter().map(|cell| cell.symbol()).collect();
                println!("\r{text}");
                self.drawn_rows += 1;
            }
            let _ = io::stdout().flush();
        } else {
            println!(
                "  {} {}/{} ({:.0}%) loss={loss:.6} loss_average={average:.6} tokens_per_second={rate:.0} optimizer_updates={done} updates_total={} updates_remaining={remaining_updates} global_update={global} learning_rate={lr} memory_occupancy={memory} checkpoint_number={number} elapsed_seconds={elapsed:.3}{feed_fields} eta={}{last_checkpoint_field}",
                self.label,
                done,
                self.total,
                fraction * 100.0,
                self.total,
                duration(remaining),
            );
            let _ = io::stdout().flush();
        }
    }

    fn feed_fields(&self) -> String {
        let Some(dataset) = self.feed_dataset.as_deref() else {
            return String::new();
        };
        format!(
            " feed_dataset={} feed_config={} feed_split={} feed_field={} feed_rows={} feed_tokens={} feed_row={} feed_snippet={} feed_token_ids={}",
            encode_log_value(dataset),
            encode_log_value(self.feed_config.as_deref().unwrap_or("auto")),
            encode_log_value(self.feed_split.as_deref().unwrap_or("train")),
            encode_log_value(self.feed_field.as_deref().unwrap_or("text")),
            self.feed_rows.unwrap_or(0),
            self.feed_tokens.unwrap_or(0),
            self.feed_row.unwrap_or(0),
            encode_log_value(self.feed_snippet.as_deref().unwrap_or("")),
            encode_log_value(self.feed_token_ids.as_deref().unwrap_or("")),
        )
    }

    fn loss_average(&self) -> Option<f64> {
        let tokens: usize = self.loss_history.iter().map(|(_, n)| n).sum();
        (tokens > 0).then(|| {
            self.loss_history
                .iter()
                .map(|(l, n)| l * *n as f64)
                .sum::<f64>()
                / tokens as f64
        })
    }

    fn clear(&mut self) {
        if self.drawn_rows > 0 {
            let _ = crossterm::execute!(
                io::stdout(),
                crossterm::cursor::MoveUp(self.drawn_rows),
                crossterm::cursor::MoveToColumn(0),
                crossterm::terminal::Clear(crossterm::terminal::ClearType::FromCursorDown)
            );
        }
        self.drawn_rows = 0;
    }

    /// Leave the final frame in scrollback and allow epoch/summary lines below.
    pub fn finish(&mut self) {
        self.drawn_rows = 0;
    }
}

/// Dataset text is untrusted terminal content, even when decoded from a log.
pub(crate) fn terminal_text(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

pub(crate) fn encode_log_value(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// Kaggle chain convention; no assumption about arbitrary checkpoint names.
pub(crate) fn checkpoint_number(path: &str) -> Option<u64> {
    std::path::Path::new(path)
        .file_stem()?
        .to_str()?
        .strip_prefix("ck")?
        .parse()
        .ok()
}

pub(crate) fn checkpoint_saved(path: &str) {
    println!("saved_checkpoint={path}");
    let number = checkpoint_number(path)
        .map(|number| number.to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("checkpoint_number={number} checkpoint_status=saved");
    println!("last_checkpoint={path}");
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn average_uses_only_last_eight_updates_and_weights_targets() {
        let mut progress = Progress::new_with_tui("test", 100, false);
        progress.emitted = true;
        for _ in 0..8 {
            progress.update(1, 1, 2.0);
        }
        progress.update(2, 7, 4.0);
        assert_eq!(progress.loss_history.len(), 8);
        assert_eq!(progress.loss_average(), Some(3.0));
        progress.update(3, 1, f64::NAN);
        assert_eq!(progress.loss_average(), Some(3.0));
    }

    #[test]
    fn feed_terminal_text_preserves_unicode_without_control_sequences() {
        assert_eq!(
            terminal_text("héllo 世界\n\r\t\x1b[2J\u{009b}31m"),
            "héllo 世界    [2J 31m"
        );
    }

    #[test]
    fn feed_metadata_is_optional_escaped_and_shares_progress_throttle() {
        let mut progress = Progress::new_with_tui("training", 10, false);
        assert!(progress.feed_fields().is_empty());
        assert!(progress.should_emit(1));
        progress.set_feed(
            "owner/name",
            None,
            "train",
            "text",
            2,
            8,
            3,
            "a b\n%é",
            "1,2",
        );
        let fields = progress.feed_fields();
        assert!(fields.contains(" feed_dataset=owner/name"));
        assert!(fields.contains(" feed_rows=2 feed_tokens=8 feed_row=3"));
        assert!(fields.contains(" feed_snippet=a%20b%0A%25%C3%A9"));
        assert!(fields.contains(" feed_token_ids=1%2C2"));
        assert!(!fields.contains('\n'));
        progress.emitted = true;
        progress.last_draw = Instant::now();
        assert!(!progress.should_emit(2));
        assert!(
            progress.should_emit(10),
            "final preview is never throttled away"
        );
        progress.last_draw = Instant::now() - std::time::Duration::from_secs(6);
        assert!(progress.should_emit(2));
    }

    #[test]
    fn chain_number_handles_both_checkpoint_types_and_spaces() {
        assert_eq!(checkpoint_number("/tmp/chain dir/ck32.pssa"), Some(32));
        assert_eq!(checkpoint_number("ck02.trfm"), Some(2));
        assert_eq!(checkpoint_number("model.pssa"), None);
    }
}
