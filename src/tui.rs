//! Live read-only dashboard for oxide training runs.
//!
//! `oxide train ... | oxide tui` renders the run as a full-screen dashboard in
//! your own terminal. The TUI never writes to the model, the checkpoints or the
//! logs: it reads the structured `key=value` lines the CLI already prints and
//! the chain directory, and draws. When stdin is not a pipe it prints its help
//! and exits, and Ctrl+C / q always hands the terminal back cleanly.

use crate::ui;
use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

const TABS: [&str; 3] = ["monitor", "chain", "model"];

#[derive(Default)]
struct RunState {
    // header card
    corpus: Option<String>,
    vocab: Option<String>,
    width: Option<String>,
    memory: Option<String>,
    schedule: Option<String>,
    // monitor tab
    progress_pct: Option<f64>,
    live_loss: Option<f64>,
    tok_s: Option<f64>,
    eta: Option<String>,
    // losses over time (for the sparkline)
    loss_series: Vec<f64>,
    // last finished epoch line
    epoch_loss: Option<f64>,
    epoch_tokens: Option<u64>,
    epoch_updates: Option<u64>,
    // summary card
    wall: Option<String>,
    throughput: Option<String>,
    training_seconds: Option<f64>,
    optimizer_updates: Option<u64>,
    // chain tab
    chain_dir: PathBuf,
    checkpoints: Vec<(String, Option<f64>)>,
    resumed_from: Option<String>,
    prior_steps: Option<u64>,
    current_offset: Option<u64>,
    raw_lines: Vec<String>,
}

impl RunState {
    fn ingest(&mut self, line: &str) {
        let line = strip_ansi(line);
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        self.raw_lines.push(line.to_string());
        if self.raw_lines.len() > 400 {
            self.raw_lines.remove(0);
        }

        // `label  value` rows from the banner and summary panels
        for (label, field) in [
            ("corpus", &mut self.corpus),
            ("vocabulary", &mut self.vocab),
            ("width", &mut self.width),
            ("memory", &mut self.memory),
            ("schedule", &mut self.schedule),
            ("wall time", &mut self.wall),
            ("throughput", &mut self.throughput),
        ] {
            if let Some(v) = parse_field(line, label) {
                *field = Some(v);
            }
        }

        let loss = parse_kv::<f64>(line, "loss=")
            .or_else(|| parse_kv(line, "loss "))
            .filter(|v| v.is_finite());
        if let Some(v) = loss {
            if line.starts_with("epoch ") && line.contains("updates=") {
                // Epoch summaries are distinct from live progress samples.
                self.epoch_loss = Some(v);
                self.epoch_tokens = parse_kv(line, "tokens=");
                self.epoch_updates = parse_kv(line, "updates=");
                self.loss_series.push(v);
            } else if line.contains("tokens_per_second=") || line.contains("tok/s") {
                // ui::Progress emits key=value fields when piped, and a bar
                // with space-separated fields on an interactive terminal.
                self.live_loss = Some(v);
                self.loss_series.push(v);
                if let Some(p) = parse_pct(line) {
                    self.progress_pct = Some(p);
                }
                self.tok_s = parse_kv::<f64>(line, "tokens_per_second=")
                    .or_else(|| line.split_once("tok/s")?.0.split_whitespace().last()?.parse().ok())
                    .filter(|v| v.is_finite());
                if let Some((_, eta)) = line.split_once("eta=").or_else(|| line.split_once("eta ")) {
                    // ui::duration can contain spaces, e.g. `2h 14m 09s`.
                    self.eta = Some(eta.trim().to_string());
                }
            }
            if self.loss_series.len() > 600 {
                self.loss_series.remove(0);
            }
        }
        if let Some(t) = parse_kv(line, "training_seconds=") {
            self.training_seconds = Some(t);
        }
        if let Some(u) = parse_kv(line, "optimizer_updates=") {
            self.optimizer_updates = Some(u);
        }
        if line.contains("resumed_from=") {
            let value = line.split("resumed_from=").nth(1).unwrap_or("").trim();
            let head = value.split_whitespace().next().unwrap_or("");
            if !head.is_empty() {
                self.resumed_from = Some(head.to_string());
            }
        }
        if let Some(s) = parse_kv(line, "prior_steps=") {
            self.prior_steps = Some(s);
        }
        if let Some(v) = parse_kv(line, "offset ") {
            // `--- ck32 (corpus offset 6200000) ---`
            self.current_offset = Some(v);
        }
        if let Some(rest) = line.split("saved_checkpoint=").nth(1) {
            let path = rest.trim();
            self.note_checkpoint(path);
        }
        if let Some(rest) = line.split("checkpoint written to ").nth(1) {
            self.note_checkpoint(strip_ansi(rest.trim_end()).trim());
        }
    }

    fn note_checkpoint(&mut self, path: &str) {
        let name = PathBuf::from(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
        if !name.ends_with(".pssa") {
            return;
        }
        if !self.checkpoints.iter().any(|(n, _)| *n == name) {
            self.checkpoints.push((name, None));
        }
    }

    /// Scan the chain directory for checkpoint files; also carry any loss
    /// recorded on disk in sibling `.loss` files (written by future runs).
    fn refresh_chain(&mut self) {
        let Ok(entries) = std::fs::read_dir(&self.chain_dir) else {
            return;
        };
        let mut names: Vec<(String, Option<f64>)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pssa") {
                let loss = std::fs::read_to_string(
                    entry.path().with_extension("loss"),
                )
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok());
                names.push((name, loss));
            }
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, _) in names {
            if !self.checkpoints.iter().any(|(n, _)| *n == name) {
                self.checkpoints.push((name, None));
            }
        }
    }
}

/// Pull `label  value` from a banner/summary row (two-space separated).
fn parse_field(line: &str, label: &str) -> Option<String> {
    let line = line.trim();
    // ui::field indents rows; ui::panel_field additionally frames them.
    let line = line.strip_prefix('│').unwrap_or(line).trim_start();
    let rest = line.strip_prefix(label)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let value = rest.trim().trim_end_matches('│').trim_end();
    if value.is_empty() || value.starts_with('=') {
        return None;
    }
    Some(value.to_string())
}

/// Pull `key=value` (or `key value`) numeric pairs out of a line.
fn parse_kv<T: std::str::FromStr>(line: &str, key: &str) -> Option<T> {
    let rest = line.split(key).nth(1)?;
    let token = rest.trim_start().split_whitespace().next()?.trim_end_matches(['%', ',', 's']);
    token.parse().ok()
}

fn parse_pct(line: &str) -> Option<f64> {
    line.split_whitespace().find_map(|token| {
        let value = token.trim_matches(['(', ')']).strip_suffix('%')?;
        value.parse::<f64>().ok().filter(|v| v.is_finite())
    })
}

fn parse_field_exact(line: &str, label: &str) -> Option<String> {
    let padded = format!("{:<16}", label);
    line.strip_prefix(&padded).map(|v| v.trim().to_string())
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for skip in chars.by_ref() {
                if skip == 'm' || skip.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn loss_color(loss: f64) -> ratatui::style::Color {
    match loss {
        l if l < 3.5 => ratatui::style::Color::Green,
        l if l < 4.0 => ratatui::style::Color::LightGreen,
        l if l < 4.2 => ratatui::style::Color::Yellow,
        _ => ratatui::style::Color::Red,
    }
}

fn run_app(rx: mpsc::Receiver<String>, chain_dir: PathBuf) -> io::Result<()> {
    let mut terminal = ratatui::init();
    let mut state = RunState {
        chain_dir: chain_dir.clone(),
        ..RunState::default()
    };
    let mut tab = 0usize;
    let mut last_chain_scan = std::time::Instant::now() - Duration::from_secs(60);

    loop {
        // drain stdin
        while let Ok(line) = rx.try_recv() {
            state.ingest(&line);
        }
        if last_chain_scan.elapsed() > Duration::from_secs(5) {
            state.refresh_chain();
            last_chain_scan = std::time::Instant::now();
        }

        terminal.draw(|f| draw(f, &state, tab))?;

        // poll events for up to 200ms, then loop back to stdin
        if crossterm::event::poll(Duration::from_millis(200))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if key.kind == crossterm::event::KeyEventKind::Press {
                    match key.code {
                        crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => break,
                        crossterm::event::KeyCode::Tab | crossterm::event::KeyCode::Right => {
                            tab = (tab + 1) % TABS.len();
                        }
                        crossterm::event::KeyCode::Left => {
                            tab = (tab + TABS.len() - 1) % TABS.len();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    ratatui::restore();
    Ok(())
}

fn draw(f: &mut ratatui::Frame, state: &RunState, tab: usize) {
    let area = f.area();
    let tabs = ratatui::widgets::Tabs::new(TABS)
        .select(tab)
        .highlight_style(ratatui::style::Style::new().fg(ratatui::style::Color::Cyan).add_modifier(ratatui::style::Modifier::BOLD))
        .padding("", "");
    let title = format!("oxide tui  |  q quit  tab switch");
    f.render_widget(
        ratatui::widgets::Paragraph::new(ratatui::text::Line::styled(title, ratatui::style::Style::new().add_modifier(ratatui::style::Modifier::DIM))),
        area,
    );
    let tabs_area = ratatui::layout::Rect { x: area.x, y: area.y, width: area.width, height: 1 };
    f.render_widget(tabs, tabs_area);

    let body = ratatui::layout::Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height.saturating_sub(1),
    };
    match tab {
        0 => draw_monitor(f, body, state),
        1 => draw_chain(f, body, state),
        _ => draw_model(f, body, state),
    }
}

fn draw_monitor(f: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &RunState) {
    let chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(7),
            ratatui::layout::Constraint::Min(5),
            ratatui::layout::Constraint::Length(6),
        ])
        .split(area);

    // progress + gauges
    let pct = state.progress_pct.unwrap_or(0.0);
    let loss = state.live_loss.or(state.epoch_loss).unwrap_or(0.0);
    let gauge = ratatui::widgets::Gauge::default()
        .label(format!(
            "loss {:.4}  |  {:.0}%  |  {:.0} tok/s  |  eta {}",
            loss,
            pct,
            state.tok_s.unwrap_or(0.0),
            state.eta.as_deref().unwrap_or("-")
        ))
        .ratio((pct / 100.0).clamp(0.0, 1.0))
        .gauge_style(
            ratatui::style::Style::new()
                .fg(loss_color(loss))
                .bg(ratatui::style::Color::Black),
        );
    f.render_widget(
        ratatui::widgets::Block::default().title(" monitor ").borders(ratatui::widgets::Borders::ALL),
        chunks[0],
    );
    let inner = ratatui::layout::Rect {
        x: chunks[0].x + 1,
        y: chunks[0].y + 1,
        width: chunks[0].width.saturating_sub(2),
        height: 3,
    };
    f.render_widget(gauge, inner);

    // loss sparkline
    let data: Vec<u64> = state
        .loss_series
        .iter()
        .map(|l| (*l * 1000.0) as u64)
        .collect();
    let spark = ratatui::widgets::Sparkline::default()
        .block(ratatui::widgets::Block::default().title(" loss ").borders(ratatui::widgets::Borders::ALL))
        .data(&data)
        .style(ratatui::style::Style::new().fg(loss_color(loss)));
    f.render_widget(spark, chunks[1]);

    // latest epoch + run stats
    let lines = vec![
        ratatui::text::Line::from(format!(
            "last epoch   loss {:.4}   tokens {}   updates {}",
            state.epoch_loss.unwrap_or(0.0),
            state.epoch_tokens.unwrap_or(0),
            state.epoch_updates.unwrap_or(0)
        )),
        ratatui::text::Line::from(format!(
            "this run     wall {}   training_seconds={:.1}   optimizer_updates={}",
            state.wall.as_deref().unwrap_or("-"),
            state.training_seconds.unwrap_or(0.0),
            state.optimizer_updates.unwrap_or(0)
        )),
        ratatui::text::Line::from(format!(
            "chain        resumed from {} at {} prior steps",
            state.resumed_from.as_deref().unwrap_or("-"),
            state.prior_steps.map(|s| s.to_string()).unwrap_or_else(|| "-".into())
        )),
    ];
    f.render_widget(
        ratatui::widgets::Paragraph::new(lines)
            .block(ratatui::widgets::Block::default().title(" run ").borders(ratatui::widgets::Borders::ALL)),
        chunks[2],
    );
}

fn draw_chain(f: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &RunState) {
    let rows: Vec<ratatui::text::Line> = if state.checkpoints.is_empty() {
        vec![ratatui::text::Line::from(format!(
            "no .pssa files found in {}",
            state.chain_dir.display()
        ))]
    } else {
        let last = state.checkpoints.len().saturating_sub(1);
        state
            .checkpoints
            .iter()
            .enumerate()
            .map(|(i, (name, loss))| {
                let marker = if i == last { "●" } else { "●" };
                let style = loss
                    .map(loss_color)
                    .unwrap_or(ratatui::style::Color::Green);
                let loss_text = loss
                    .map(|l| format!("{l:.4}"))
                    .unwrap_or_else(|| "—".into());
                let suffix = if i == last { "  (latest)" } else { "" };
                ratatui::text::Line::styled(
                    format!("{marker} {name}  loss {loss_text}{suffix}"),
                    ratatui::style::Style::new().fg(style),
                )
            })
            .collect()
    };
    let block = ratatui::widgets::Paragraph::new(rows)
        .block(
            ratatui::widgets::Block::default()
                .title(format!(" chain ({}) ", state.chain_dir.display()))
                .borders(ratatui::widgets::Borders::ALL),
        );
    f.render_widget(block, area);
}

fn draw_model(f: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &RunState) {
    let rows = [
        ("corpus", state.corpus.clone()),
        ("vocabulary", state.vocab.clone()),
        ("width", state.width.clone()),
        ("memory", state.memory.clone()),
        ("schedule", state.schedule.clone()),
    ];
    let lines: Vec<ratatui::text::Line> = rows
        .iter()
        .map(|(label, value)| {
            let value = value.clone().unwrap_or_else(|| "-".into());
            ratatui::text::Line::from(format!("{label:<14}{value}"))
        })
        .collect();
    f.render_widget(
        ratatui::widgets::Paragraph::new(lines)
            .block(ratatui::widgets::Block::default().title(" model ").borders(ratatui::widgets::Borders::ALL)),
        area,
    );
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut chain_dir = PathBuf::from("/kaggle/working/chain");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--chain" | "-c" if i + 1 < args.len() => {
                chain_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            other => return Err(format!("unknown tui flag '{other}'; usage: oxide tui [--chain DIR]")),
        }
        i += 1;
    }

    if io::stdin().is_terminal() {
        return Err(
            "oxide tui reads a training run from stdin; pipe it: oxide train ... | oxide tui".into(),
        );
    }
    if !io::stdout().is_terminal() {
        return Err(
            "oxide tui needs a real terminal to draw in; run it interactively, e.g. inside tmux or ssh".into(),
        );
    }

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin().lock();
        for line in stdin.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    if let Err(e) = run_app(rx, chain_dir) {
        ratatui::restore();
        return Err(e.to_string());
    }
    Ok(())
}

#[allow(dead_code)]
fn unused_helpers() {
    let _ = parse_field_exact("  schedule        1 epoch(s), 446 updates, lr 0.001", "schedule");
    let _ = ui::bold("");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_piped_progress_and_multicomponent_eta() {
        let mut state = RunState::default();
        state.ingest("  training 97/100 (97%) loss=4.077800 tokens_per_second=146 eta=45.8s");
        assert_eq!(state.progress_pct, Some(97.0));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(state.eta.as_deref(), Some("45.8s"));
        assert_eq!(state.epoch_loss, None);
        assert_eq!(state.loss_series, [4.0778]);
        state.ingest("  training 20/100 (20%) loss=4.0 tokens_per_second=150 eta=2h 14m 09s");
        assert_eq!(state.eta.as_deref(), Some("2h 14m 09s"));
    }

    #[test]
    fn parses_interactive_progress_without_confusing_epoch_summary() {
        let mut state = RunState::default();
        state.ingest("\r  \x1b[2mtraining\x1b[0m ███░  97%  loss 4.0778  146 tok/s  eta 14m 09s   ");
        assert_eq!(state.progress_pct, Some(97.0));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(state.eta.as_deref(), Some("14m 09s"));
        state.ingest("  epoch 1/1 loss=4.0123 tokens=1200 updates=10");
        assert_eq!(state.epoch_loss, Some(4.0123));
        assert_eq!(state.epoch_tokens, Some(1200));
        assert_eq!(state.epoch_updates, Some(10));
        assert_eq!(state.live_loss, Some(4.0778));
        assert_eq!(state.loss_series, [4.0778, 4.0123]);
    }

    #[test]
    fn parses_indented_ansi_and_panel_fields_but_not_label_prefixes() {
        let mut state = RunState::default();
        state.ingest("  \x1b[2mcorpus\x1b[0m          /tmp/training text.txt  ");
        state.ingest("  vocabulary      2048 BPE tokens");
        state.ingest("  width           256");
        state.ingest("  memory          512 slots");
        state.ingest("  schedule        1 epoch(s), 446 updates, lr 0.001");
        state.ingest("  │ wall time       2h 14m 09s                 │");
        state.ingest("  │ throughput      146 tokens/s              │");
        assert_eq!(state.corpus.as_deref(), Some("/tmp/training text.txt"));
        assert_eq!(state.vocab.as_deref(), Some("2048 BPE tokens"));
        assert_eq!(state.width.as_deref(), Some("256"));
        assert_eq!(state.memory.as_deref(), Some("512 slots"));
        assert_eq!(state.schedule.as_deref(), Some("1 epoch(s), 446 updates, lr 0.001"));
        assert_eq!(state.wall.as_deref(), Some("2h 14m 09s"));
        assert_eq!(state.throughput.as_deref(), Some("146 tokens/s"));
        for line in ["corpus=other", "corpus_path other", "corpuses other", "corpus   "] {
            assert_eq!(parse_field(line, "corpus"), None, "{line}");
        }
    }

    #[test]
    fn producer_output_round_trips_through_parser() {
        // A child process makes ui::Progress actually write to a pipe, avoiding
        // unstable stdout-capture APIs or a duplicate copy of its format string.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tui::tests::ui_producer_fixture", "--nocapture"])
            .env("OXIDE_TUI_PRODUCER_FIXTURE", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let mut state = RunState::default();
        for line in String::from_utf8(output.stdout).unwrap().lines() {
            state.ingest(line);
        }
        assert_eq!(state.corpus.as_deref(), Some("/tmp/producer corpus.txt"));
        assert_eq!(state.width.as_deref(), Some("256"));
        assert_eq!(state.wall.as_deref(), Some("14m 09s"));
        assert_eq!(state.progress_pct, Some(100.0));
        assert_eq!(state.live_loss, Some(4.123456));
        assert!(state.tok_s.unwrap() > 0.0);
        assert_eq!(state.eta.as_deref(), Some("0.0s"));
    }

    #[test]
    fn ui_producer_fixture() {
        if std::env::var_os("OXIDE_TUI_PRODUCER_FIXTURE").is_none() {
            return;
        }
        ui::field("corpus", "/tmp/producer corpus.txt");
        ui::field("width", "256");
        ui::panel_field("wall time", "14m 09s");
        let mut progress = ui::Progress::new("training", 4);
        progress.update(4, 512, 4.123456);
        progress.finish();
    }

    #[test]
    fn progress_history_is_bounded_and_nonfinite_numbers_are_ignored() {
        let mut state = RunState::default();
        for _ in 0..650 {
            state.ingest("training 1/2 (50%) loss=4.0 tokens_per_second=146 eta=1.0s");
        }
        assert_eq!(state.loss_series.len(), 600);
        assert_eq!(state.raw_lines.len(), 400);
        state.ingest("training 1/2 (NaN%) loss=NaN tokens_per_second=NaN eta=unknown");
        assert_eq!(state.progress_pct, Some(50.0));
        assert_eq!(state.live_loss, Some(4.0));
        assert_eq!(state.tok_s, Some(146.0));
        assert_eq!(parse_pct("(inf%)"), None);
    }
}
