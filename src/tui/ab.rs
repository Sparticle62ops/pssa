//! Opt-in, read-only checkpoint comparison for the chat tab.
//!
//! One worker loads A, generates, drops it, then does B. There is deliberately no
//! model cache or second worker: two large models must never coexist just to show
//! their replies side by side. Cancellation is cooperative; loading and a single
//! forward pass cannot be interrupted, and the UI never joins this worker.
use super::{AMBER, accent, panel, panel_area};
use crate::{
    cli::CLIHandler,
    dataset::{Tokenizer, TokenizerKind},
    inference::{InferenceConfig, PSSAInferenceEngine, unknown_prompt_error},
    linalg::SimpleRng,
    transformer::TransformerModel,
    transformer_checkpoint,
};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const MAX_OUTPUT: usize = 64 * 1024;
const MAX_PROMPT: usize = 256 * 1024;
pub(super) const HELP: &str = "/ab toggles A/B • /ab a PATH • /ab b PATH • /ab off\nPaths may contain spaces (optional surrounding quotes). PSSA / .trfm supported.\nSame prompt + system + attachments; no prior chat history. Replies are not saved.\nA then B: one model in RAM. Rates include prefill, exclude loading/waiting.";

fn clean(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

fn bounded(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Ready,
    Waiting,
    Loading,
    Generating,
    Done,
    Stopped,
    Failed,
    Limited,
}
impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "READY",
            Self::Waiting => "WAITING",
            Self::Loading => "LOADING",
            Self::Generating => "GENERATING",
            Self::Done => "DONE",
            Self::Stopped => "STOPPED",
            Self::Failed => "ERROR",
            Self::Limited => "OUTPUT LIMIT",
        }
    }
}

#[derive(Clone, Default)]
struct Output {
    text: String,
    tokens: usize,
    elapsed: f64,
    phase: Phase,
    error: Option<String>,
}
impl Output {
    fn rate_label(&self) -> String {
        if self.elapsed > 0.0 {
            format!("{:.1}", self.rate())
        } else {
            "unmeasured".into()
        }
    }

    fn rate(&self) -> f64 {
        if self.elapsed > 0.0 {
            self.tokens as f64 / self.elapsed
        } else {
            0.0
        }
    }
}
#[derive(Default)]
struct Progress {
    sides: [Output; 2],
    done: bool,
}
struct Worker {
    progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Generation-only clock and bounded snapshots. Hitting the output cap stops
/// this side, not the other checkpoint. No per-token channel can grow unbounded.
struct Reporter<'a> {
    progress: &'a Mutex<Progress>,
    index: usize,
    cancel: &'a AtomicBool,
    limited: AtomicBool,
    started: Option<Instant>,
}
impl Reporter<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.limited.load(Ordering::Relaxed)
    }
    fn generating(&mut self) {
        self.started = Some(Instant::now());
        self.progress
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sides[self.index]
            .phase = Phase::Generating;
    }
    fn update(&self, text: &str, tokens: usize) {
        if text.len() >= MAX_OUTPUT {
            self.limited.store(true, Ordering::Relaxed);
        }
        let mut p = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        let side = &mut p.sides[self.index];
        side.text = bounded(text, MAX_OUTPUT).to_owned();
        side.tokens = tokens;
        side.elapsed = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
    }
}

#[derive(Default)]
pub(super) struct Comparison {
    enabled: bool,
    models: [String; 2],
    prompt: String,
    sides: [Output; 2],
    worker: Option<Worker>,
    scroll: [u16; 2],
    follow: bool,
}
impl Comparison {
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }
    pub(super) fn busy(&self) -> bool {
        self.worker.is_some()
    }
    pub(super) fn command(&mut self, arg: &str, default_model: &str) -> Result<String, String> {
        if self.busy() {
            return Err(
                "A/B is busy; Esc stops it. Wait for the worker before switching models or modes."
                    .into(),
            );
        }
        let (cmd, path) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
        match cmd {
            "" | "on" | "off" if path.trim().is_empty() => {
                self.enabled = match cmd {
                    "on" => true,
                    "off" => false,
                    _ => !self.enabled,
                };
                if self.enabled && self.models[0].is_empty() && !default_model.is_empty() {
                    self.models[0] = default_model.into();
                }
            }
            "a" | "b" => {
                let path = path.trim();
                let path = path
                    .strip_prefix('"')
                    .and_then(|p| p.strip_suffix('"'))
                    .or_else(|| path.strip_prefix('\'').and_then(|p| p.strip_suffix('\'')))
                    .unwrap_or(path);
                validate_path(path)?;
                self.models[usize::from(cmd == "b")] = path.into();
                // Never label an old reply with a newly selected checkpoint.
                self.sides = std::array::from_fn(|_| Output::default());
                self.prompt.clear();
                self.scroll = [0; 2];
                self.follow = true;
                self.enabled = true;
            }
            _ => return Err(format!("Usage: {HELP}")),
        }
        Ok(if self.enabled {
            format!("A/B enabled. {HELP}")
        } else {
            "Single chat restored; its history and checkpoint are unchanged. /ab returns to comparison.".into()
        })
    }
    pub(super) fn start(&mut self, prompt: String, cfg: &InferenceConfig) -> Result<(), String> {
        self.start_with(prompt, cfg, run_checkpoint)
    }
    fn start_with<F>(
        &mut self,
        prompt: String,
        cfg: &InferenceConfig,
        mut run: F,
    ) -> Result<(), String>
    where
        F: FnMut(&str, &str, &InferenceConfig, &mut Reporter<'_>) -> Result<(), String>
            + Send
            + 'static,
    {
        if self.busy() {
            return Err("A/B is busy; Esc stops generation.".into());
        }
        if prompt.trim().is_empty() || prompt.len() > MAX_PROMPT {
            return Err(
                "A/B prompt must be nonempty and at most 256 KiB; reduce text/attachments.".into(),
            );
        }
        PSSAInferenceEngine::validate(cfg)?;
        if !(1..=4096).contains(&cfg.max_new_tokens) {
            return Err("max-tokens must be in 1..=4096".into());
        }
        for (index, model) in self.models.iter().enumerate() {
            validate_path(model).map_err(|e| {
                format!(
                    "{}: {e}; select with /ab {} PATH",
                    label(index),
                    label(index).to_ascii_lowercase()
                )
            })?;
        }
        let progress = Arc::new(Mutex::new(Progress {
            sides: std::array::from_fn(|_| Output {
                phase: Phase::Waiting,
                ..Output::default()
            }),
            done: false,
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (progress.clone(), cancel.clone());
        let models = self.models.clone();
        let worker_prompt = prompt.clone();
        let cfg = InferenceConfig { ..*cfg };
        std::thread::Builder::new()
            .name("checkpoint-ab".into())
            .spawn(move || {
                for (index, path) in models.iter().enumerate() {
                    if c.load(Ordering::Relaxed) {
                        p.lock().unwrap_or_else(|e| e.into_inner()).sides[index].phase =
                            Phase::Stopped;
                        continue;
                    }
                    p.lock().unwrap_or_else(|e| e.into_inner()).sides[index].phase = Phase::Loading;
                    let mut reporter = Reporter {
                        progress: &p,
                        index,
                        cancel: &c,
                        limited: AtomicBool::new(false),
                        started: None,
                    };
                    // A bad checkpoint must not leave the UI permanently busy.
                    // The normal error path still lets the other side run.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(path, &worker_prompt, &cfg, &mut reporter)
                    }))
                    .unwrap_or_else(|_| Err("inference worker panicked".into()));
                    let mut p = p.lock().unwrap_or_else(|e| e.into_inner());
                    let side = &mut p.sides[index];
                    side.elapsed = reporter.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
                    side.phase = match result {
                        Err(error) => {
                            side.error = Some(bounded(&error, MAX_OUTPUT).to_owned());
                            Phase::Failed
                        }
                        Ok(()) if c.load(Ordering::Relaxed) => Phase::Stopped,
                        Ok(()) if reporter.limited.load(Ordering::Relaxed) => Phase::Limited,
                        Ok(()) => Phase::Done,
                    };
                }
                p.lock().unwrap_or_else(|e| e.into_inner()).done = true;
            })
            .map_err(|e| format!("cannot start A/B worker: {e}"))?;
        self.prompt = prompt;
        self.sides = std::array::from_fn(|_| Output {
            phase: Phase::Waiting,
            ..Output::default()
        });
        self.scroll = [0; 2];
        self.follow = true;
        self.worker = Some(Worker { progress, cancel });
        Ok(())
    }
    /// Returns a completion note once; snapshots remain visible after finishing.
    pub(super) fn poll(&mut self) -> Option<String> {
        let worker = self.worker.as_ref()?;
        let p = worker.progress.lock().unwrap_or_else(|e| e.into_inner());
        self.sides.clone_from(&p.sides);
        if !p.done {
            return None;
        }
        drop(p);
        self.worker = None;
        Some(format!(
            "A: {} • B: {}. Rates exclude load/wait, include prefill. Replies are not saved; /ab off restores single chat.",
            self.sides[0].phase.label(),
            self.sides[1].phase.label()
        ))
    }
    pub(super) fn stop(&mut self) {
        if let Some(worker) = &self.worker {
            worker.cancel.store(true, Ordering::Relaxed);
        }
    }
    pub(super) fn page(&mut self, down: bool) {
        self.follow = false;
        for scroll in &mut self.scroll {
            *scroll = if down {
                scroll.saturating_add(8)
            } else {
                scroll.saturating_sub(8)
            };
        }
    }
    pub(super) fn follow(&mut self) {
        self.follow = true;
    }
    pub(super) fn draw(
        &mut self,
        f: &mut ratatui::Frame,
        area: Rect,
        input: &str,
        note: &str,
        cfg: &InferenceConfig,
    ) {
        if area.is_empty() {
            return;
        }
        if area.width < 44 || area.height < 16 {
            f.render_widget(
                Paragraph::new(format!(
                    "A/B • A then B (one model in RAM)\nA {} tok • {} tok/s\nB {} tok • {} tok/s\n{}\n▶ {}\nEnlarge for side-by-side replies • Esc stop",
                    self.sides[0].tokens, self.sides[0].rate_label(),
                    self.sides[1].tokens, self.sides[1].rate_label(),
                    clean(note), clean(input)
                )).style(accent()),
                area,
            );
            return;
        }
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);
        let stopping = self
            .worker
            .as_ref()
            .is_some_and(|w| w.cancel.load(Ordering::Relaxed));
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    if stopping {
                        "A/B • STOP REQUESTED (waiting for load/step)"
                    } else {
                        "A/B • same prompt • A then B • one model in RAM"
                    },
                    accent(),
                ),
                Line::from(format!(
                    "T {:.2} • p {:.2} • k {} • max {} • rep {:.2}",
                    cfg.temperature,
                    cfg.top_p,
                    cfg.top_k,
                    cfg.max_new_tokens,
                    cfg.repetition_penalty
                )),
                Line::from(format!(
                    "Prompt: {}",
                    clean(&self.prompt).replace('\n', " ")
                )),
            ]),
            chunks[0],
        );
        let gap = if super::shadow::enabled(f.area()) {
            1
        } else {
            0
        };
        let columns = Layout::horizontal([
            Constraint::Percentage(50),
            Constraint::Length(gap),
            Constraint::Percentage(50),
        ])
        .split(chunks[1]);
        for (index, column) in columns.iter().copied().step_by(2).enumerate() {
            let column = panel_area(f, column);
            let side = &self.sides[index];
            let kind = if self.models[index].ends_with(".trfm") {
                "transformer"
            } else {
                "PSSA"
            };
            let title = format!(" {} / {} ", label(index), kind);
            let block = panel(&title);
            let inner = block.inner(column);
            f.render_widget(block, column);
            let rows = Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).split(inner);
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(if self.models[index].is_empty() {
                        format!("Select /ab {} PATH", label(index).to_ascii_lowercase())
                    } else {
                        clean(&self.models[index])
                    }),
                    Line::styled(side.phase.label(), accent()),
                    Line::from(format!("{} tok • {} tok/s", side.tokens, side.rate_label())),
                ]),
                rows[0],
            );
            let mut lines: Vec<Line> = clean(&side.text)
                .lines()
                .map(|s| Line::from(s.to_owned()))
                .collect();
            if let Some(error) = &side.error {
                lines.push(Line::styled(
                    format!("Error: {}", clean(error)),
                    ratatui::style::Style::new().fg(AMBER),
                ));
            }
            if side.phase == Phase::Limited {
                lines.push(Line::from("Reply capped at 64 KiB."));
            }
            let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
            let max = (paragraph.line_count(rows[1].width).min(u16::MAX as usize) as u16)
                .saturating_sub(rows[1].height);
            self.scroll[index] = if self.follow {
                max
            } else {
                self.scroll[index].min(max)
            };
            f.render_widget(paragraph.scroll((self.scroll[index], 0)), rows[1]);
        }
        f.render_widget(
            Paragraph::new(clean(note))
                .wrap(Wrap { trim: false })
                .style(ratatui::style::Style::new().fg(AMBER)),
            chunks[2],
        );
        let prompt_area = panel_area(f, chunks[3]);
        let input = clean(input);
        let tail = super::setup::visible_tail(&input, prompt_area.width.saturating_sub(4) as usize);
        f.render_widget(
            Paragraph::new(format!("▶ {tail}")).block(panel(" shared prompt / command ")),
            prompt_area,
        );
        f.render_widget(
            Paragraph::new("Enter send • Esc stop • PgUp/Dn • End follow • /ab off • /help")
                .style(accent()),
            chunks[4],
        );
    }
}

fn label(index: usize) -> &'static str {
    if index == 0 { "A" } else { "B" }
}
fn validate_path(path: &str) -> Result<(), String> {
    if !matches!(
        Path::new(path).extension().and_then(|e| e.to_str()),
        Some("pssa" | "trfm")
    ) {
        return Err("expected a .pssa or .trfm checkpoint path".into());
    }
    if !Path::new(path).is_file() {
        return Err("checkpoint path is not a regular file".into());
    }
    Ok(())
}

fn run_checkpoint(
    path: &str,
    prompt: &str,
    cfg: &InferenceConfig,
    reporter: &mut Reporter<'_>,
) -> Result<(), String> {
    if reporter.cancelled() {
        return Ok(());
    }
    if path.ends_with(".trfm") {
        let model = transformer_checkpoint::load_checkpoint(path).map_err(|e| e.to_string())?;
        if reporter.cancelled() {
            return Ok(());
        }
        let tok = model.tokenizer()?;
        reporter.generating();
        generate_transformer(&model, &tok, prompt, cfg, reporter)
    } else {
        let (mut model, tok) = CLIHandler::load_for_inference(path, None)?;
        if reporter.cancelled() {
            return Ok(());
        }
        reporter.generating();
        // The existing engine owns PSSA sampling, recurrent state and decoding.
        let text = PSSAInferenceEngine::try_new(&mut model, &tok)?
            .try_generate_chat_turn_controlled(
                prompt,
                cfg,
                |text, count| reporter.update(text, count),
                || reporter.cancelled(),
            )?;
        let count = reporter
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sides[reporter.index]
            .tokens;
        reporter.update(&text, count);
        Ok(())
    }
}

/// Decode complete UTF-8, replacing invalid sequences but holding an incomplete
/// final sequence. This also handles an invalid byte before that final suffix.
fn decoded_prefix(mut bytes: &[u8]) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                out.push_str(std::str::from_utf8(&bytes[..valid]).expect("validated UTF-8"));
                if let Some(bad) = error.error_len() {
                    out.push('\u{FFFD}');
                    bytes = &bytes[valid + bad..];
                } else {
                    break;
                }
            }
        }
    }
    out
}

/// The baseline's public generate() loads and runs synchronously without progress
/// or cancellation. This adapter uses its same context logits, seeded sampler,
/// token counts and sentence stopping policy; it does not implement model math.
fn generate_transformer(
    model: &TransformerModel,
    tok: &Tokenizer,
    prompt: &str,
    cfg: &InferenceConfig,
    reporter: &Reporter<'_>,
) -> Result<(), String> {
    let mut ids = tok.try_encode(prompt, true)?;
    if ids.is_empty() {
        return Err("prompt is empty after tokenization".into());
    }
    if ids.iter().all(|&id| id == 0) {
        return Err(unknown_prompt_error(tok, prompt));
    }
    let prompt_len = ids.len();
    let mut logits = vec![0.0; tok.vocab_size];
    let mut probs = vec![0.0; tok.vocab_size];
    let mut candidates = Vec::with_capacity(tok.vocab_size);
    let mut rng = SimpleRng::new(1337);
    let mut sentences = 0;
    let mut raw = Vec::new();
    let mut out = String::new();
    for step in 0..cfg.max_new_tokens {
        if reporter.cancelled() {
            break;
        }
        model.logits_for_context(&ids, &mut logits)?;
        if reporter.cancelled() {
            break;
        }
        let id = PSSAInferenceEngine::sample(
            &mut rng,
            cfg,
            &ids,
            &mut logits,
            &mut probs,
            &mut candidates,
        )?;
        ids.push(id);
        if tok.kind() == TokenizerKind::Bpe {
            let bytes = tok
                .token_bytes(id)
                .ok_or_else(|| format!("missing BPE bytes for token {id}"))?;
            let remaining = MAX_OUTPUT.saturating_sub(raw.len());
            raw.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
            if bytes.len() >= remaining {
                reporter.limited.store(true, Ordering::Relaxed);
            }
            // Hold an incomplete UTF-8 suffix until another token arrives. Lossy
            // decoding of completed invalid sequences matches baseline output.
            out = decoded_prefix(&raw);
        } else {
            out = tok.decode(&ids[prompt_len..]);
            if matches!(tok.id_to_token[&id].as_str(), "." | "?" | "!") {
                sentences += 1;
            }
        }
        reporter.update(&out, step + 1);
        if sentences >= 2 {
            break;
        }
    }
    if tok.kind() == TokenizerKind::Bpe {
        out = String::from_utf8_lossy(&raw).into_owned();
    }
    reporter.update(&out, ids.len() - prompt_len);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checkpoint,
        pssa::{PSSAConfigV2, PSSALayerV2},
        transformer::TransformerConfig,
    };
    use ratatui::{Terminal, backend::TestBackend};
    use std::{fs, path::PathBuf, sync::mpsc, time::Duration};

    struct Fixture {
        dir: PathBuf,
        models: [String; 2],
    }
    impl Fixture {
        fn new() -> Self {
            static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "pssa ab {} {}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            let tok = Tokenizer::from_vocabulary(&["<unk>".into(), "hello".into(), "world".into()])
                .unwrap();
            let mut pssa = PSSALayerV2::new(
                PSSAConfigV2 {
                    d_vocab: 3,
                    d_latent: 4,
                    d_state: 2,
                    d_mem_key: 2,
                    mem_capacity: 2,
                    chunk_len: 2,
                    ..Default::default()
                },
                7,
            );
            pssa.vocabulary = tok.ordered_vocabulary().unwrap();
            pssa.unembed_w.data.fill(0.0);
            let mut trfm = TransformerModel::new(
                TransformerConfig {
                    d_vocab: 3,
                    d_model: 4,
                    n_heads: 2,
                    d_ff: 5,
                    chunk_len: 4,
                    ..Default::default()
                },
                7,
            )
            .unwrap();
            trfm.vocabulary = tok.ordered_vocabulary().unwrap();
            trfm.unembed.data.fill(0.0);
            let a = dir.join("checkpoint a.pssa");
            let b = dir.join("baseline b.trfm");
            checkpoint::save_model(&pssa, &a).unwrap();
            transformer_checkpoint::save_model(&trfm, &b).unwrap();
            Self {
                dir,
                models: [a.to_str().unwrap().into(), b.to_str().unwrap().into()],
            }
        }
        fn comparison(&self) -> Comparison {
            Comparison {
                enabled: true,
                models: self.models.clone(),
                ..Comparison::default()
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
    fn config() -> InferenceConfig {
        InferenceConfig {
            temperature: 0.0,
            max_new_tokens: 4,
            ..InferenceConfig::default()
        }
    }
    fn finish(ab: &mut Comparison) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while ab.busy() && Instant::now() < deadline {
            ab.poll();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!ab.busy(), "comparison worker did not finish");
    }
    fn screen(ab: &mut Comparison, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|f| ab.draw(f, f.area(), "shared input", "A/B note", &config()))
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .chunks(width.max(1) as usize)
            .map(|row| row.iter().map(|c| c.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn selection_accepts_spaces_and_quotes_without_loading_and_clears_stale_results() {
        let fixture = Fixture::new();
        let empty = fixture.dir.join("not loaded yet.trfm");
        fs::write(&empty, []).unwrap();
        let mut ab = Comparison::default();
        ab.command("on", &fixture.models[0]).unwrap();
        assert!(ab.enabled());
        assert_eq!(ab.models[0], fixture.models[0]);
        ab.command(&format!("b \"{}\"", empty.display()), "")
            .unwrap();
        assert_eq!(ab.models[1], empty.to_str().unwrap());
        assert!(!ab.busy());
        ab.sides[0].text = "old reply".into();
        ab.prompt = "old prompt".into();
        ab.command(&format!("a '{}'", fixture.models[0]), "")
            .unwrap();
        assert!(ab.sides[0].text.is_empty());
        assert!(ab.prompt.is_empty());
        let previous = ab.models.clone();
        for command in [
            "a /missing file.pssa",
            "b /etc/passwd",
            "unexpected",
            "on extra",
        ] {
            assert!(ab.command(command, "").is_err(), "{command}");
            assert_eq!(ab.models, previous);
        }
        ab.command("off", "").unwrap();
        assert!(!ab.enabled());
        ab.command("", "").unwrap();
        assert!(ab.enabled());
    }

    #[test]
    fn mixed_checkpoints_generate_real_replies_and_leave_files_unchanged() {
        let fixture = Fixture::new();
        let original: Vec<_> = fixture
            .models
            .iter()
            .map(|p| fs::read(p).unwrap())
            .collect();
        let mut ab = fixture.comparison();
        let prompt = "System: hello\n\nUser: hello world\n\nAssistant:";
        let baseline =
            crate::transformer_inference::generate(&fixture.models[1], prompt, &config()).unwrap();
        assert_eq!(baseline, "hello hello hello hello");
        for reverse in [false, true] {
            if reverse {
                ab.models.swap(0, 1);
            }
            ab.start(prompt.into(), &config()).unwrap();
            finish(&mut ab);
            for side in &ab.sides {
                assert_eq!(side.phase, Phase::Done, "{:?}", side.error);
                assert_eq!(side.tokens, 4);
                assert_eq!(side.text, baseline);
                assert!(side.elapsed > 0.0);
                assert!(side.rate().is_finite() && side.rate() > 0.0);
            }
            assert_eq!(ab.prompt, prompt);
        }
        for (path, bytes) in fixture.models.iter().zip(original) {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
    }

    #[test]
    fn bpe_checkpoints_match_existing_generation_engines() {
        let fixture = Fixture::new();
        let tok = Tokenizer::from_corpus_bpe("hello world 世界. hello world 世界!", 280).unwrap();
        let mut pssa = PSSALayerV2::new(
            PSSAConfigV2 {
                d_vocab: tok.vocab_size,
                d_latent: 4,
                d_state: 2,
                d_mem_key: 2,
                mem_capacity: 2,
                chunk_len: 2,
                ..Default::default()
            },
            19,
        );
        pssa.vocabulary = tok.ordered_vocabulary().unwrap();
        pssa.tokenizer_json = tok.serialized_metadata();
        let mut trfm = TransformerModel::new(
            TransformerConfig {
                d_vocab: tok.vocab_size,
                d_model: 4,
                n_heads: 2,
                d_ff: 5,
                chunk_len: 4,
                ..Default::default()
            },
            19,
        )
        .unwrap();
        trfm.vocabulary = tok.ordered_vocabulary().unwrap();
        trfm.tokenizer_json = tok.serialized_metadata();
        checkpoint::save_model(&pssa, &fixture.models[0]).unwrap();
        transformer_checkpoint::save_model(&trfm, &fixture.models[1]).unwrap();
        let cfg = InferenceConfig {
            max_new_tokens: 7,
            ..InferenceConfig::default()
        };
        let prompt = "hello 世界";
        let pssa_expected = PSSAInferenceEngine::try_new(&mut pssa, &tok)
            .unwrap()
            .try_generate_chat_turn(prompt, &cfg, |_| {})
            .unwrap();
        let trfm_expected =
            crate::transformer_inference::generate(&fixture.models[1], prompt, &cfg).unwrap();
        let mut ab = fixture.comparison();
        ab.start(prompt.into(), &cfg).unwrap();
        finish(&mut ab);
        assert_eq!(ab.sides[0].phase, Phase::Done, "{:?}", ab.sides[0].error);
        assert_eq!(ab.sides[1].phase, Phase::Done, "{:?}", ab.sides[1].error);
        assert_eq!(ab.sides[0].text, pssa_expected);
        assert_eq!(ab.sides[1].text, trfm_expected);
        assert_eq!(ab.sides[0].tokens, 7);
        assert_eq!(ab.sides[1].tokens, 7);
    }

    #[test]
    fn checkpoint_error_is_local_and_other_side_still_generates() {
        let fixture = Fixture::new();
        let broken = fixture.dir.join("broken.pssa");
        fs::write(&broken, b"not a checkpoint").unwrap();
        let mut ab = fixture.comparison();
        ab.models[0] = broken.to_str().unwrap().into();
        ab.start("hello".into(), &config()).unwrap();
        finish(&mut ab);
        assert_eq!(ab.sides[0].phase, Phase::Failed);
        assert!(ab.sides[0].error.as_ref().is_some_and(|e| !e.is_empty()));
        assert_eq!(ab.sides[1].phase, Phase::Done);
        assert_eq!(ab.sides[1].tokens, 4);
        assert!(screen(&mut ab, 140, 26).contains("Error:"));
        let unknown = "unknownwordnotinvocabulary";
        ab.models = fixture.models.clone();
        ab.start(unknown.into(), &config()).unwrap();
        finish(&mut ab);
        for side in &ab.sides {
            assert_eq!(side.phase, Phase::Failed);
            assert!(side.error.as_ref().unwrap().contains("vocabulary"));
        }
    }

    #[test]
    fn same_prompt_settings_and_sequential_order_with_independent_output_caps() {
        let fixture = Fixture::new();
        let mut ab = fixture.comparison();
        let (tx, rx) = mpsc::channel();
        ab.start_with(
            "shared 世界 prompt".into(),
            &config(),
            move |path, prompt, cfg, reporter| {
                tx.send((
                    path.to_owned(),
                    prompt.to_owned(),
                    cfg.max_new_tokens,
                    cfg.temperature,
                ))
                .unwrap();
                assert_eq!(
                    reporter.progress.lock().unwrap().sides[reporter.index].phase,
                    Phase::Loading
                );
                assert!(
                    reporter.started.is_none(),
                    "loading must not start rate clock"
                );
                if reporter.index == 1 {
                    assert_eq!(
                        reporter.progress.lock().unwrap().sides[0].phase,
                        Phase::Limited
                    );
                    assert!(!reporter.cancelled(), "A's output limit must not stop B");
                }
                reporter.generating();
                reporter.update(&"世".repeat(MAX_OUTPUT), 9);
                assert!(reporter.cancelled());
                Ok(())
            },
        )
        .unwrap();
        finish(&mut ab);
        let received: Vec<_> = rx.try_iter().collect();
        assert_eq!(received.len(), 2);
        for (index, (path, prompt, tokens, temperature)) in received.iter().enumerate() {
            assert_eq!(path, &fixture.models[index]);
            assert_eq!(prompt, "shared 世界 prompt");
            assert_eq!(*tokens, 4);
            assert_eq!(*temperature, 0.0);
            assert_eq!(ab.sides[index].phase, Phase::Limited);
            assert_eq!(ab.sides[index].text.len(), MAX_OUTPUT / 3 * 3);
            assert!(ab.sides[index].text.ends_with('世'));
        }
    }

    #[test]
    fn stop_is_nonblocking_retains_partial_reply_and_never_starts_b() {
        let fixture = Fixture::new();
        let mut ab = fixture.comparison();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        ab.start_with("hello".into(), &config(), move |_, _, _, reporter| {
            assert_eq!(reporter.index, 0, "B must not load after cancellation");
            reporter.generating();
            reporter.update("partial reply", 2);
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(reporter.cancelled());
            Ok(())
        })
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        ab.stop();
        ab.poll();
        assert!(ab.busy(), "keep gate held until the old worker exits");
        assert_eq!(ab.sides[0].text, "partial reply");
        assert!(ab.command("off", "").is_err());
        assert!(ab.start("second".into(), &config()).is_err());
        assert!(screen(&mut ab, 120, 24).contains("STOP REQUESTED"));
        release_tx.send(()).unwrap();
        finish(&mut ab);
        assert_eq!(ab.sides[0].phase, Phase::Stopped);
        assert_eq!(ab.sides[1].phase, Phase::Stopped);
        assert_eq!(ab.sides[1].tokens, 0);
        assert_eq!(ab.sides[0].text, "partial reply");
    }

    #[test]
    fn dropping_comparison_cancels_without_joining_a_blocked_loader() {
        let fixture = Fixture::new();
        let mut ab = fixture.comparison();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        ab.start_with("hello".into(), &config(), move |_, _, _, reporter| {
            assert_eq!(reporter.index, 0);
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            done_tx.send(reporter.cancelled()).unwrap();
            Ok(())
        })
        .unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let cancel = ab.worker.as_ref().unwrap().cancel.clone();
        drop(ab);
        assert!(cancel.load(Ordering::Relaxed));
        release_tx.send(()).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_secs(10)).unwrap());
    }

    #[test]
    fn input_validation_is_bounded_and_utf8_streaming_holds_only_incomplete_suffixes() {
        let fixture = Fixture::new();
        let mut ab = fixture.comparison();
        assert!(ab.start("".into(), &config()).is_err());
        assert!(ab.start("x".repeat(MAX_PROMPT + 1), &config()).is_err());
        for max_new_tokens in [0, 4097] {
            assert!(
                ab.start(
                    "hello".into(),
                    &InferenceConfig {
                        max_new_tokens,
                        ..config()
                    }
                )
                .is_err()
            );
        }
        assert!(
            ab.start(
                "hello".into(),
                &InferenceConfig {
                    temperature: f32::NAN,
                    ..config()
                }
            )
            .is_err()
        );
        assert!(!ab.busy());
        assert_eq!(decoded_prefix(&[0xe4, 0xb8]), "");
        assert_eq!(decoded_prefix(&[0xe4, 0xb8, 0x96]), "世");
        assert_eq!(decoded_prefix(&[0xff, 0xe4, 0xb8]), "\u{fffd}");
        assert_eq!(decoded_prefix(&[0xff, 0xe4, 0xb8, 0x96]), "\u{fffd}世");
    }

    #[test]
    fn idle_comparison_rates_are_unmeasured_until_generation_is_timed() {
        let mut ab = Comparison {
            enabled: true,
            ..Comparison::default()
        };
        for (w, h) in [(100, 24), (40, 14)] {
            let text = screen(&mut ab, w, h);
            assert!(text.contains("unmeasured tok/s"), "{text}");
            assert!(!text.contains("0.0 tok/s"));
        }
        ab.sides[0].tokens = 2;
        ab.sides[0].elapsed = 1.0;
        assert!(screen(&mut ab, 100, 24).contains("2 tok • 2.0 tok/s"));
    }

    #[test]
    fn test_backend_shows_side_by_side_rates_scroll_and_small_screen_fallback() {
        let mut ab = Comparison {
            enabled: true,
            models: ["a.pssa".into(), "b.trfm".into()],
            follow: true,
            ..Comparison::default()
        };
        ab.sides = [
            Output {
                text: "left reply".into(),
                tokens: 20,
                elapsed: 2.0,
                phase: Phase::Done,
                error: None,
            },
            Output {
                text: "right reply".into(),
                tokens: 12,
                elapsed: 3.0,
                phase: Phase::Generating,
                error: None,
            },
        ];
        for (width, height) in [(160, 40), (100, 24), (60, 20), (44, 16)] {
            let rendered = screen(&mut ab, width, height);
            assert!(rendered.contains("A / PSSA"));
            assert!(rendered.contains("B / transformer"));
            assert!(rendered.contains("20 tok • 10.0 tok/s"));
            assert!(rendered.contains("12 tok • 4.0 tok/s"));
            assert!(rendered.contains("left reply"));
            assert!(rendered.contains("right reply"));
            let reply_row = rendered
                .lines()
                .find(|row| row.contains("left reply"))
                .unwrap();
            assert!(
                reply_row.contains("right reply"),
                "answers must occupy the same row in different columns"
            );
        }
        for side in &mut ab.sides {
            side.text = (0..80).map(|n| format!("reply {n} 世界\n")).collect();
        }
        assert!(screen(&mut ab, 100, 24).contains("reply 79"));
        let previous = ab.scroll;
        ab.page(false);
        screen(&mut ab, 100, 24);
        assert!(ab.scroll[0] < previous[0] && ab.scroll[1] < previous[1]);
        ab.follow();
        assert!(screen(&mut ab, 100, 24).contains("reply 79"));
        for (width, height) in [(40, 14), (25, 8), (1, 1), (0, 0)] {
            let rendered = screen(&mut ab, width, height);
            if width >= 25 {
                assert!(rendered.contains("A/B"));
                assert!(rendered.contains("10.0 tok/s"));
                assert!(rendered.contains("4.0 tok/s"));
            }
        }
    }
}
