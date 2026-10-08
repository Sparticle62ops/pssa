//! Local chat UI. Checkpoints are read-only; chat documents are separate JSON files.
use super::{
    AMBER, NORMAL_GREEN, accent,
    heatmap::{self, TokenMark},
    memory_view::{MemorySnapshot, MemoryView},
    panel, panel_area,
};
use crate::{
    cli::CLIHandler,
    inference::{InferenceConfig, PSSAInferenceEngine},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const MAX_TEXT: usize = 64 * 1024;
const MAX_CONTEXT: usize = 256 * 1024;
const MAX_CHAT: u64 = 4 * 1024 * 1024;
const HELP: &str = "/ab [on|off|a PATH|b PATH]  /model [path]  /new  /chats  /open ID  /rename NAME  /delete ID\n/temp F (alias /temperature)  /top-p F  /top-k N  /max-tokens N  /repetition-penalty F\n/system TEXT  /attach PATH  /detach  /copy  /speech  /stop  /help\nF1 shows all keyboard controls; //TEXT sends a leading slash.";

fn clean(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

#[derive(Clone)]
struct Message {
    role: String,
    text: String,
    confidence: Vec<TokenMark>,
}

struct Document {
    id: String,
    name: String,
    model: String,
    system: String,
    config: InferenceConfig,
    messages: Vec<Message>,
}

impl Document {
    fn new(model: String) -> Self {
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            id: format!(
                "{stamp}-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ),
            name: "New chat".into(),
            model,
            system: String::new(),
            config: InferenceConfig::default(),
            messages: Vec::new(),
        }
    }
    fn value(&self) -> Value {
        json!({"version":1,"id":self.id,"name":self.name,"model":self.model,"system":self.system,
            "settings":{"temperature":self.config.temperature,"top_p":self.config.top_p,"top_k":self.config.top_k,"max_tokens":self.config.max_new_tokens,"repetition_penalty":self.config.repetition_penalty},
            "messages":self.messages.iter().map(|m|json!({"role":m.role,"text":m.text,"confidence":heatmap::to_json(&m.confidence)})).collect::<Vec<_>>()})
    }
    fn parse(v: Value) -> Result<Self, String> {
        fn text(v: &Value, k: &str) -> Result<String, String> {
            v[k].as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("invalid chat field: {k}"))
        }
        if v["version"] != 1 {
            return Err("unsupported chat JSON version".into());
        }
        let s = &v["settings"];
        let number = |k: &str| {
            s[k].as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| format!("invalid setting: {k}"))
        };
        let integer = |k: &str| {
            s[k].as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("invalid setting: {k}"))
        };
        let config = InferenceConfig {
            temperature: number("temperature")?,
            top_p: number("top_p")?,
            top_k: integer("top_k")?,
            max_new_tokens: integer("max_tokens")?,
            repetition_penalty: number("repetition_penalty")?,
        };
        validate(&config)?;
        let messages = v["messages"]
            .as_array()
            .ok_or("invalid messages")?
            .iter()
            .map(|m| {
                let role = text(m, "role")?;
                if !matches!(role.as_str(), "user" | "assistant") {
                    return Err("invalid message role".into());
                }
                let body = text(m, "text")?;
                let confidence: Vec<TokenMark> = match m.get("confidence") {
                    Some(v) => heatmap::from_json(v)?,
                    None => Vec::new(), // Existing saved chats have no probabilities.
                };
                if !heatmap::valid(&body, &confidence) {
                    return Err("invalid token confidence boundaries".into());
                }
                Ok(Message {
                    role,
                    text: body,
                    confidence,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let doc = Self {
            id: text(&v, "id")?,
            name: text(&v, "name")?,
            model: text(&v, "model")?,
            system: text(&v, "system")?,
            config,
            messages,
        };
        if !valid_id(&doc.id) {
            return Err("invalid chat ID".into());
        }
        Ok(doc)
    }
    fn context(&self) -> Result<String, String> {
        let mut out = String::new();
        if !self.system.is_empty() {
            out.push_str(&format!("System: {}\n\n", self.system));
        }
        for m in &self.messages {
            out.push_str(if m.role == "user" {
                "User: "
            } else {
                "Assistant: "
            });
            out.push_str(&m.text);
            out.push_str("\n\n");
        }
        out.push_str("Assistant:");
        if out.len() > MAX_CONTEXT {
            return Err(
                "Context exceeds 256 KiB; start /new (history is never silently truncated)".into(),
            );
        }
        Ok(out)
    }
}
fn validate(cfg: &InferenceConfig) -> Result<(), String> {
    PSSAInferenceEngine::validate(cfg)?;
    if cfg.max_new_tokens == 0 || cfg.max_new_tokens > 4096 {
        return Err("max-tokens must be in 1..=4096".into());
    }
    Ok(())
}
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 100 && id.bytes().all(|b| b.is_ascii_digit() || b == b'-')
}
fn read_bounded(path: &Path, limit: u64) -> Result<String, String> {
    if !fs::metadata(path).map_err(|e| e.to_string())?.is_file() {
        return Err("expected a regular file".into());
    }
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err(format!("file exceeds {limit} bytes"));
    }
    String::from_utf8(bytes)
        .map_err(|_| "not supported by this model yet: only UTF-8 text files".into())
}
struct Store {
    dir: PathBuf,
}
impl Store {
    fn path(&self, id: &str) -> Result<PathBuf, String> {
        if !valid_id(id) {
            return Err("invalid chat ID; use /chats".into());
        }
        Ok(self.dir.join(format!("{id}.json")))
    }
    fn save(&self, doc: &Document) -> Result<(), String> {
        fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
        let path = self.path(&doc.id)?;
        let bytes = serde_json::to_vec_pretty(&doc.value()).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_CHAT {
            return Err("chat exceeds 4 MiB; start /new".into());
        }
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let result = (|| -> io::Result<()> {
            let mut opts = fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            let mut file = opts.open(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&tmp, &path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result.map_err(|e| format!("chat was not saved: {e}"))
    }
    fn load(&self, id: &str) -> Result<Document, String> {
        let doc = Document::parse(
            serde_json::from_str(&read_bounded(&self.path(id)?, MAX_CHAT)?)
                .map_err(|e| e.to_string())?,
        )?;
        if doc.id != id {
            return Err("chat ID does not match filename".into());
        }
        Ok(doc)
    }
    fn list(&self) -> Result<String, String> {
        if !self.dir.exists() {
            return Ok("No saved chats. Send a message to save one.".into());
        }
        let mut paths = fs::read_dir(&self.dir)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect::<Vec<_>>();
        paths.sort();
        let mut lines = vec!["Saved chats — /open ID to resume; /delete ID to remove".to_string()];
        for path in paths.iter().rev().take(200) {
            let id = path.file_stem().unwrap_or_default().to_string_lossy();
            match self.load(&id) {
                Ok(doc) => lines.push(format!(
                    "{}  {}  ({} messages)",
                    id,
                    clean(&doc.name),
                    doc.messages.len()
                )),
                Err(e) => lines.push(format!("{}  [unreadable: {}]", clean(&id), clean(&e))),
            }
        }
        Ok(lines.join("\n"))
    }
}

#[derive(Default)]
struct Progress {
    confidence: Vec<TokenMark>,
    text: String,
    tokens: usize,
    done: Option<Result<(), String>>,
    speech: bool,
    memory: Option<MemoryView>,
}
struct Job {
    progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
    started: Instant,
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub(super) struct Chat {
    doc: Document,
    store: Store,
    pub(super) input: String,
    note: String,
    attachments: Vec<(String, String)>,
    job: Option<Job>,
    ab: super::ab::Comparison,
    // Only speech owns external processes. Join its cancellable worker on quit
    // so a recorder cannot outlive the UI (never join checkpoint loading).
    speech_thread: Option<std::thread::JoinHandle<()>>,
    tokens: usize,
    elapsed: f64,
    scroll: u16,
    follow: bool,
    pending_delete: Option<String>,
    model_dir: PathBuf,
    pub(super) heatmap: bool,
    memory: MemoryView,
}
impl Chat {
    pub(super) fn open_checkpoint(&mut self, path: &Path) {
        let path_text = path.display().to_string();
        if path_text.chars().any(char::is_control) {
            self.note = "Checkpoint paths must not contain control characters.".into();
            return;
        }
        let transformer = path.extension().is_some_and(|e| e == "trfm");
        if !transformer && self.ab.enabled() {
            if let Err(e) = self.command("/ab off") {
                self.note = e;
                return;
            }
        }
        let cmd = if transformer { "/ab b" } else { "/model" };
        if let Err(e) = self.command(&format!("{cmd} {path_text}")) {
            self.note = e;
        }
    }
    pub(super) fn new(dir: PathBuf, model_dir: PathBuf) -> Self {
        Self {
            doc: Document::new(String::new()),
            store: Store { dir },
            input: String::new(),
            note: format!("Select a checkpoint with /model PATH (spaces allowed).\n{HELP}"),
            attachments: Vec::new(),
            job: None,
            ab: super::ab::Comparison::default(),
            speech_thread: None,
            tokens: 0,
            elapsed: 0.0,
            scroll: 0,
            follow: true,
            pending_delete: None,
            model_dir,
            heatmap: false,
            memory: MemoryView::default(),
        }
    }
    // Follow the wizard's output directory without changing the selected model
    // or discarding the current conversation.
    pub(super) fn set_model_dir(&mut self, model_dir: PathBuf) {
        self.model_dir = model_dir;
    }

    /// Latest read-only retrieval snapshot, retained when generation finishes.
    pub(super) fn memory_view(&self) -> &MemoryView {
        &self.memory
    }

    pub(super) fn poll(&mut self) {
        if let Some(note) = self.ab.poll() {
            self.note = note;
        }
        let Some(job) = &self.job else {
            return;
        };
        self.elapsed = job.started.elapsed().as_secs_f64();
        let mut p = job.progress.lock().unwrap_or_else(|e| e.into_inner());
        self.tokens = p.tokens;
        if let Some(memory) = &p.memory
            && memory.latest().map(|snapshot| snapshot.generated_tokens)
                != self
                    .memory
                    .latest()
                    .map(|snapshot| snapshot.generated_tokens)
        {
            self.memory.clone_from(memory);
        }
        if !p.speech
            && let Some(last) = self.doc.messages.last_mut()
        {
            last.text.clone_from(&p.text);
            last.confidence.clone_from(&p.confidence);
        }
        let Some(result) = p.done.take() else {
            return;
        };
        let speech = p.speech;
        if speech && result.is_ok() {
            self.input.push_str(&clean(&p.text));
        }
        drop(p);
        let stopped = job.cancel.load(Ordering::Relaxed);
        self.job = None;
        if let Some(thread) = self.speech_thread.take() {
            let _ = thread.join();
        }
        self.note = match result {
            Ok(()) if stopped => "Stopped. Partial reply kept.".into(),
            Ok(()) if speech => "Transcript inserted in input; review then Enter to send.".into(),
            Ok(()) => "Reply saved. /copy copies the last reply.".into(),
            Err(e) => format!("Error: {e}"),
        };
        if !speech && let Err(e) = self.store.save(&self.doc) {
            self.note.push_str(&format!("\n{e}"));
        }
    }
    fn save(&mut self) -> Result<(), String> {
        self.store.save(&self.doc)
    }
    fn send(&mut self, text: &str) -> Result<(), String> {
        if self.doc.model.is_empty() && !self.ab.enabled() {
            return Err("Pick a checkpoint first: /model PATH".into());
        }
        let mut text = text.to_owned();
        for (name, body) in &self.attachments {
            text.push_str(&format!(
                "\n\n--- attached text: {name} ---\n{body}\n--- end attachment ---"
            ));
        }
        if self.ab.enabled() {
            // Independent comparison turns never append either answer to the
            // single-chat history or give B a different prompt from A.
            let system = if self.doc.system.is_empty() {
                String::new()
            } else {
                format!("System: {}\n\n", self.doc.system)
            };
            self.ab.start(
                format!("{system}User: {text}\n\nAssistant:"),
                &self.doc.config,
            )?;
            self.attachments.clear();
            self.note = "A/B running A then B (one model in RAM). Esc stops; checkpoint loading/current step must finish. Replies are not saved.".into();
            return Ok(());
        }
        self.doc.messages.push(Message {
            role: "user".into(),
            text,
            confidence: Vec::new(),
        });
        let prompt = match self.doc.context() {
            Ok(p) => p,
            Err(e) => {
                self.doc.messages.pop();
                return Err(e);
            }
        };
        if let Err(e) = self.save() {
            self.doc.messages.pop();
            return Err(e);
        }
        self.attachments.clear();
        self.doc.messages.push(Message {
            role: "assistant".into(),
            text: String::new(),
            confidence: Vec::new(),
        });
        let model = self.doc.model.clone();
        let cfg = InferenceConfig { ..self.doc.config };
        self.memory = MemoryView::new(&model);
        let progress = Arc::new(Mutex::new(Progress {
            memory: Some(self.memory.clone()),
            ..Progress::default()
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (progress.clone(), cancel.clone());
        std::thread::spawn(move || {
            let result = (|| {
                if Path::new(&model).extension().is_some_and(|e| e.eq_ignore_ascii_case("trfm")) {
                    let text = crate::transformer_inference::generate_controlled(
                        &model, &prompt, &cfg,
                        &mut |text, tokens| {
                            let mut p = p.lock().unwrap_or_else(|e| e.into_inner());
                            p.text = text.to_owned();
                            p.tokens = tokens;
                        },
                        &|| c.load(Ordering::Relaxed),
                    )?;
                    p.lock().unwrap_or_else(|e| e.into_inner()).text = text;
                    return Ok(());
                }
                let (mut model, tok) = CLIHandler::load_for_inference(&model, None)?;
                if c.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let text = PSSAInferenceEngine::try_new(&mut model, &tok)?
                    .try_generate_chat_turn_scored_observed(
                        &prompt,
                        &cfg,
                        |text, tokens, probability| {
                            let mut p = p.lock().unwrap_or_else(|e| e.into_inner());
                            heatmap::record(&mut p.confidence, text, tokens, probability);
                            p.text = text.to_owned();
                            p.tokens = tokens;
                        },
                        || c.load(Ordering::Relaxed),
                        |count, query, selected, model| {
                            // Copy before locking: rendering only sees owned telemetry,
                            // never a live model borrow or a second retrieval operation.
                            let snapshot =
                                MemorySnapshot::capture(count, query, selected, &tok, model);
                            if let Some(memory) =
                                &mut p.lock().unwrap_or_else(|e| e.into_inner()).memory
                            {
                                memory.record(snapshot);
                            }
                        },
                    )?;
                p.lock().unwrap_or_else(|e| e.into_inner()).text = text;
                Ok(())
            })();
            p.lock().unwrap_or_else(|e| e.into_inner()).done = Some(result);
        });
        self.job = Some(Job {
            progress,
            cancel,
            started: Instant::now(),
        });
        self.tokens = 0;
        self.elapsed = 0.0;
        self.follow = true;
        self.note =
            "Loading / generating locally… Esc stops (checkpoint loading finishes first).".into();
        Ok(())
    }
    fn command(&mut self, line: &str) -> Result<(), String> {
        let (cmd, arg) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let arg = arg.trim();
        if (self.job.is_some() || self.ab.busy()) && !matches!(cmd, "/stop" | "/copy" | "/help") {
            return Err("Busy; Esc stops the current job first.".into());
        }
        if self.ab.enabled()
            && matches!(
                cmd,
                "/model" | "/new" | "/open" | "/rename" | "/delete" | "/chats" | "/copy"
            )
        {
            return Err("A/B replies are temporary; /ab off restores saved single-chat commands. Use /ab a PATH or /ab b PATH to select checkpoints.".into());
        }
        if cmd != "/delete" {
            self.pending_delete = None;
        }
        match cmd {
            "/help" => self.note = format!("{HELP}\n{}", super::ab::HELP),
            "/ab" => self.note = self.ab.command(arg, &self.doc.model)?,
            "/stop" => self.stop(),
            "/model" if arg.is_empty() => {
                let mut paths = Vec::new();
                for dir in [&self.model_dir, &PathBuf::from("data")] {
                    if let Ok(entries) = fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let p = entry.path();
                            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pssa") || e.eq_ignore_ascii_case("trfm")) {
                                paths.push(p.display().to_string());
                            }
                        }
                    }
                }
                paths.sort();
                paths.dedup();
                self.note = format!(
                    "Pick with /model PATH (PSSA/TRFM checkpoints; paths may contain spaces)\n{}",
                    paths.join("\n")
                );
            }
            "/model" => {
                if !Path::new(arg).is_file() {
                    return Err("checkpoint path is not a file".into());
                }
                self.doc.model = arg.into();
                self.memory = MemoryView::new(arg);
                self.save()?;
                self.note = format!("Checkpoint selected: {arg}. Loaded when you send.");
            }
            "/new" => {
                self.save()?;
                self.doc = Document::new(self.doc.model.clone());
                self.memory = MemoryView::new(&self.doc.model);
                self.attachments.clear();
                self.tokens = 0;
                self.elapsed = 0.0;
                self.follow = true;
                self.save()?;
                self.note = "New chat. /system sets its system prompt.".into();
            }
            "/chats" => {
                self.note = self.store.list()?;
            }
            "/open" => {
                let doc = self.store.load(arg)?;
                self.save()?;
                self.doc = doc;
                self.memory = MemoryView::new(&self.doc.model);
                self.attachments.clear();
                self.tokens = 0;
                self.elapsed = 0.0;
                self.follow = true;
                self.note =
                    "Resumed saved chat with its checkpoint, system prompt and settings.".into();
            }
            "/rename" => {
                if arg.is_empty() || arg.len() > 200 {
                    return Err("usage: /rename NAME (1–200 bytes)".into());
                }
                self.doc.name = arg.into();
                self.save()?;
                self.note = "Chat renamed.".into();
            }
            "/delete" => {
                let path = self.store.path(arg)?;
                if !path.is_file() {
                    return Err("saved chat not found; /chats lists IDs".into());
                }
                if self.pending_delete.as_deref() != Some(arg) {
                    self.pending_delete = Some(arg.into());
                    self.note = format!("Delete permanently? Repeat /delete {arg} to confirm.");
                } else {
                    fs::remove_file(path).map_err(|e| e.to_string())?;
                    self.pending_delete = None;
                    if self.doc.id == arg {
                        self.doc = Document::new(self.doc.model.clone());
                        self.memory = MemoryView::new(&self.doc.model);
                        self.attachments.clear();
                    }
                    self.note = "Chat deleted.".into();
                }
            }
            "/system" => {
                if arg.len() > MAX_TEXT {
                    return Err("system prompt exceeds 64 KiB".into());
                }
                self.doc.system = arg.into();
                self.save()?;
                self.note = "System prompt saved (empty clears it).".into();
            }
            "/temp"
            | "/temperature"
            | "/top-p"
            | "/top-k"
            | "/max-tokens"
            | "/repetition-penalty" => {
                let mut cfg = InferenceConfig { ..self.doc.config };
                match cmd {
                    "/temp" | "/temperature" => {
                        cfg.temperature = arg.parse().map_err(|_| "expected temperature number")?
                    }
                    "/top-p" => cfg.top_p = arg.parse().map_err(|_| "expected top-p number")?,
                    "/top-k" => cfg.top_k = arg.parse().map_err(|_| "expected top-k integer")?,
                    "/max-tokens" => {
                        cfg.max_new_tokens =
                            arg.parse().map_err(|_| "expected token limit integer")?
                    }
                    _ => {
                        cfg.repetition_penalty = arg
                            .parse()
                            .map_err(|_| "expected repetition penalty number")?
                    }
                }
                validate(&cfg)?;
                self.doc.config = cfg;
                self.save()?;
                self.note = format!("Saved {cmd} {arg}");
            }
            "/attach" => {
                let ext = Path::new(arg)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(
                    ext.as_str(),
                    "pdf"
                        | "png"
                        | "jpg"
                        | "jpeg"
                        | "gif"
                        | "webp"
                        | "svg"
                        | "mp3"
                        | "wav"
                        | "mp4"
                        | "zip"
                        | "pssa"
                        | "trfm"
                ) {
                    return Err("not supported by this model yet: only UTF-8 text files (no images, audio, PDFs or binary files)".into());
                }
                let text = read_bounded(Path::new(arg), MAX_TEXT as u64)?;
                if text
                    .chars()
                    .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
                {
                    return Err(
                        "not supported by this model yet: binary content is not text".into(),
                    );
                }
                if self.attachments.iter().map(|(_, s)| s.len()).sum::<usize>() + text.len()
                    > MAX_TEXT
                {
                    return Err("pending attachments exceed 64 KiB; /detach clears them".into());
                }
                self.attachments.push((arg.into(), text));
                self.note =
                    "Text attached to your next message; /detach clears pending files.".into();
            }
            "/detach" => {
                self.attachments.clear();
                self.note = "Pending attachments cleared.".into();
            }
            "/copy" => {
                let text = self
                    .doc
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "assistant" && !m.text.is_empty())
                    .ok_or("No reply to copy yet")?;
                copy_reply(&text.text)?;
                self.note =
                    "Copy requested via OSC 52; terminal must permit clipboard access.".into();
            }
            "/speech" => self.speech()?,
            _ => {
                return Err(
                    "Unknown command. /help lists controls; //TEXT sends a leading slash.".into(),
                );
            }
        }
        Ok(())
    }
    fn stop(&mut self) {
        if self.ab.busy() {
            self.ab.stop();
            self.note = "Stopping A/B… waiting for checkpoint loading/current step; partial replies remain visible.".into();
        }
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
            self.note = "Stopping… partial reply will be saved.".into();
        }
    }
    fn speech(&mut self) -> Result<(), String> {
        #[cfg(not(feature = "speech"))]
        {
            Err("Speech is optional: build with --features speech; install local whisper-cli (or main) and arecord; set PSSA_WHISPER_BIN and PSSA_WHISPER_MODEL to existing local files. No downloads are performed.".into())
        }
        #[cfg(feature = "speech")]
        {
            let config = super::speech::discover()?;
            let progress = Arc::new(Mutex::new(Progress {
                speech: true,
                ..Progress::default()
            }));
            let cancel = Arc::new(AtomicBool::new(false));
            let (p, c) = (progress.clone(), cancel.clone());
            self.speech_thread = Some(std::thread::spawn(move || {
                let result = super::speech::transcribe(config, &c);
                let mut p = p.lock().unwrap_or_else(|e| e.into_inner());
                p.done = Some(match result {
                    Ok(text) => {
                        p.text = text;
                        Ok(())
                    }
                    Err(e) => Err(e),
                });
            }));
            self.job = Some(Job {
                progress,
                cancel,
                started: Instant::now(),
            });
            self.note="Recording 10 seconds locally… Esc cancels. Transcript goes into input, never auto-sent.".into();
            Ok(())
        }
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => self.stop(),
            KeyCode::PageUp if self.ab.enabled() => self.ab.page(false),
            KeyCode::PageDown if self.ab.enabled() => self.ab.page(true),
            KeyCode::End if self.ab.enabled() => self.ab.follow(),
            KeyCode::PageUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(8);
            }
            KeyCode::PageDown => {
                self.follow = false;
                self.scroll = self.scroll.saturating_add(8);
            }
            KeyCode::End => self.follow = true,
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear()
            }
            KeyCode::Enter => {
                let line = self.input.trim().to_owned();
                if line.is_empty() {
                    return;
                }
                let result = if line.starts_with('/') && !line.starts_with("//") {
                    self.command(&line)
                } else if self.job.is_some() || self.ab.busy() {
                    Err("Busy; Esc stops generation.".into())
                } else {
                    self.send(
                        line.strip_prefix('/')
                            .filter(|_| line.starts_with("//"))
                            .unwrap_or(&line),
                    )
                };
                match result {
                    Ok(()) => self.input.clear(),
                    Err(e) => self.note = format!("Error: {e}"),
                }
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && self.input.len() < MAX_TEXT =>
            {
                self.input.push(c)
            }
            _ => {}
        }
    }
    pub(super) fn draw(&mut self, f: &mut ratatui::Frame, area: Rect) {
        if area.is_empty() {
            return;
        }
        if self.ab.enabled() {
            self.ab
                .draw(f, area, &self.input, &self.note, &self.doc.config);
            return;
        }
        if area.width < 26 || area.height < 9 {
            f.render_widget(
                Paragraph::new(format!(
                    "inference / Tab tabs\n{} tok • {:.1} tok/s\n{}\n▶ {}\nEnlarge to chat",
                    self.tokens,
                    self.rate(),
                    clean(&self.note),
                    clean(&self.input)
                ))
                .style(accent()),
                area,
            );
            return;
        }
        let cfg = &self.doc.config;
        let settings = format!(
            "T {:.2} • p {:.2} • k {} • max {} • rep {:.2}",
            cfg.temperature, cfg.top_p, cfg.top_k, cfg.max_new_tokens, cfg.repetition_penalty
        );
        let status = format!(
            "{} • {} tok • {:.1} tok/s",
            if self.job.is_some() {
                "STREAMING"
            } else {
                "READY"
            },
            self.tokens,
            self.rate()
        );
        let chunks = Layout::vertical([
            Constraint::Length(if area.width < 65 { 5 } else { 4 }),
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);
        f.render_widget(
            Paragraph::new(vec![
                Line::from(format!(
                    "{} | {}",
                    clean(&self.doc.name),
                    clean(&self.doc.model)
                )),
                Line::from(settings),
                Line::styled(status, accent()),
                heatmap::legend(self.heatmap),
            ]),
            chunks[0],
        );
        let mut lines = Vec::new();
        if !self.doc.system.is_empty() {
            lines.push(Line::styled("SYSTEM", accent()));
            for l in clean(&self.doc.system).lines() {
                lines.push(Line::from(l.to_owned()));
            }
        }
        for message in &self.doc.messages {
            lines.push(Line::styled(
                if message.role == "user" {
                    "▌ YOU"
                } else {
                    "▌ PSSA"
                },
                accent(),
            ));
            lines.extend(heatmap::lines(
                &message.text,
                &message.confidence,
                self.heatmap && message.role == "assistant",
            ));
            lines.push(Line::from(""));
        }
        for l in clean(&self.note).lines() {
            lines.push(Line::styled(
                l.to_owned(),
                ratatui::style::Style::new().fg(AMBER),
            ));
        }
        let conversation_area = panel_area(f, chunks[1]);
        let inner = panel(" conversation ").inner(conversation_area);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        // Ratatui measures wrapped display lines (including wide Unicode), not bytes.
        let total = paragraph.line_count(inner.width).min(u16::MAX as usize) as u16;
        let max = total.saturating_sub(inner.height);
        self.scroll = if self.follow {
            max
        } else {
            self.scroll.min(max)
        };
        f.render_widget(
            paragraph
                .scroll((self.scroll, 0))
                .block(panel(" conversation ")),
            conversation_area,
        );
        let title = format!(" input • {} attachment(s) ", self.attachments.len());
        let input = clean(&self.input);
        let input_area = panel_area(f, chunks[2]);
        let tail = super::setup::visible_tail(&input, input_area.width.saturating_sub(4) as usize);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("▶ ", accent()),
                Span::raw(tail),
            ]))
            .block(panel(&title)),
            input_area,
        );
        f.render_widget(
            Paragraph::new(if area.width >= 70 {
                "Enter send  Esc stop  Tab tabs  F1 keys  /help commands"
            } else {
                "Enter send / F1 keys / Tab tabs"
            })
            .style(ratatui::style::Style::new().fg(NORMAL_GREEN)),
            chunks[3],
        );
    }
    fn rate(&self) -> f64 {
        if self.elapsed > 0.0 {
            self.tokens as f64 / self.elapsed
        } else {
            0.0
        }
    }
}

impl Drop for Chat {
    fn drop(&mut self) {
        // Quit never waits for model loading/forward passes. Persist the latest
        // snapshot before signalling cancellation, including a partial reply.
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
            let progress = job.progress.lock().unwrap_or_else(|e| e.into_inner());
            if !progress.speech {
                if let Some(last) = self.doc.messages.last_mut() {
                    last.text.clone_from(&progress.text);
                    last.confidence.clone_from(&progress.confidence);
                }
                if let Err(error) = self.store.save(&self.doc) {
                    eprintln!("{error}");
                }
            }
        }
        if let Some(thread) = self.speech_thread.take() {
            let _ = thread.join();
        }
    }
}

fn copy_reply(text: &str) -> Result<(), String> {
    // OSC 52 avoids platform clipboard dependencies and works over SSH. Base64
    // is the only payload sent; generated terminal escapes cannot be executed.
    const ABC: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::new();
    for chunk in text.as_bytes().chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        encoded.push(ABC[((n >> 18) & 63) as usize] as char);
        encoded.push(ABC[((n >> 12) & 63) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            ABC[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            ABC[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    let mut out = io::stdout().lock();
    write!(out, "\x1b]52;c;{encoded}\x07")
        .and_then(|_| out.flush())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    fn fixture() -> Chat {
        let doc = Document::new(String::new());
        Chat::new(
            std::env::temp_dir().join(format!("pssa-chat-{}", doc.id)),
            PathBuf::from("missing"),
        )
    }
    #[test]
    fn long_wide_unicode_drafts_keep_the_end_visible_in_single_and_ab_inputs() {
        let mut chat = fixture();
        chat.input = format!("{}終END", "世界".repeat(100));
        for ab in [false, true] {
            chat.ab.command(if ab { "on" } else { "off" }, "").unwrap();
            for (width, height) in [(80, 24), (120, 40), (60, 20)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| chat.draw(f, f.area())).unwrap();
                let text: String = terminal.backend().buffer().content().iter().map(|c| c.symbol()).collect();
                // Wide glyphs reserve a second blank TestBackend cell.
                assert!(text.contains("END"), "draft end hidden at {width}x{height}, ab={ab}: {text}");
            }
        }
    }

    #[test]
    fn model_picker_follows_setup_output_without_replacing_conversation() {
        let mut chat = fixture();
        let output = chat.store.dir.join("wizard output");
        fs::create_dir_all(&output).unwrap();
        let checkpoint = output.join("model.pssa");
        fs::write(&checkpoint, b"picker lists paths without loading models").unwrap();
        chat.doc.model = "existing/model.pssa".into();
        let id = chat.doc.id.clone();
        chat.set_model_dir(output);
        chat.command("/model").unwrap();
        assert!(chat.note.contains(&checkpoint.display().to_string()));
        assert_eq!(chat.doc.model, "existing/model.pssa");
        assert_eq!(chat.doc.id, id);
        fs::remove_dir_all(&chat.store.dir).unwrap();
    }

    #[test]
    fn runs_browser_selection_preserves_draft_and_switches_model_modes() {
        let mut c = fixture();
        fs::create_dir_all(&c.store.dir).unwrap();
        let a = c.store.dir.join("selected model.pssa");
        let b = c.store.dir.join("selected baseline.trfm");
        fs::write(&a, b"loaded only on send").unwrap();
        fs::write(&b, b"loaded only on send").unwrap();
        c.input = "unfinished draft?".into();
        c.open_checkpoint(&a);
        assert_eq!(c.doc.model, a.display().to_string());
        assert!(!c.ab.enabled());
        c.open_checkpoint(&b);
        assert!(c.ab.enabled());
        assert!(!c.ab.busy());
        c.open_checkpoint(&a);
        assert!(!c.ab.enabled());
        assert_eq!(c.input, "unfinished draft?");
        fs::remove_dir_all(&c.store.dir).unwrap();
    }
    #[test]
    fn ab_keyboard_paths_shared_prompt_and_return_preserve_single_chat() {
        let mut c = fixture();
        fs::create_dir_all(&c.store.dir).unwrap();
        c.doc.model = "single chat model.pssa".into();
        c.doc.system = "shared system".into();
        c.doc.messages.push(Message {
            role: "user".into(),
            text: "old single-chat history".into(),
            confidence: Vec::new(),
        });
        c.save().unwrap();
        let original = c.doc.value();
        let saved = fs::read(c.store.path(&c.doc.id).unwrap()).unwrap();
        let a = c.store.dir.join("compare a.pssa");
        let b = c.store.dir.join("baseline b.trfm");
        // Selection must not parse/load a checkpoint on the event thread.
        fs::write(&a, b"invalid fixture").unwrap();
        fs::write(&b, b"invalid fixture").unwrap();
        for line in [
            format!("/ab a {}", a.display()),
            format!("/ab b \"{}\"", b.display()),
        ] {
            c.input = line;
            c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert!(c.input.is_empty(), "{}", c.note);
        }
        assert!(c.ab.enabled());
        c.attachments
            .push(("shared.txt".into(), "shared attachment".into()));
        c.input = "shared question".into();
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(c.input.is_empty(), "{}", c.note);
        assert!(c.ab.busy());
        assert!(c.job.is_none());
        assert!(c.attachments.is_empty());
        assert_eq!(c.doc.value(), original);
        assert!(
            c.command("/ab off").is_err(),
            "cannot start a second model while the old worker is alive"
        );
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        let deadline = Instant::now() + std::time::Duration::from_secs(20);
        while c.ab.busy() && Instant::now() < deadline {
            c.poll();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!c.ab.busy());
        let mut terminal = Terminal::new(TestBackend::new(120, 24)).unwrap();
        terminal.draw(|f| c.draw(f, f.area())).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("A / PSSA"));
        assert!(screen.contains("B / transformer"));
        assert!(screen.contains("shared system"));
        assert!(screen.contains("shared question"));
        assert!(screen.contains("shared attachment"));
        assert!(!screen.contains("old single-chat history"));
        c.command("/ab off").unwrap();
        assert!(!c.ab.enabled());
        assert_eq!(c.doc.value(), original);
        assert_eq!(fs::read(c.store.path(&c.doc.id).unwrap()).unwrap(), saved);
        terminal.draw(|f| c.draw(f, f.area())).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("conversation"));
        assert!(screen.contains("old single-chat history"));
        fs::remove_dir_all(&c.store.dir).unwrap();
    }

    #[test]
    fn confidence_survives_save_resume_and_old_chats_remain_readable() {
        let mut c = fixture();
        c.doc.messages.push(Message {
            role: "assistant".into(),
            text: "low high".into(),
            confidence: vec![
                TokenMark {
                    end: 4,
                    probability: 0.01,
                },
                TokenMark {
                    end: 8,
                    probability: 0.9,
                },
            ],
        });
        c.save().unwrap();
        let loaded = c.store.load(&c.doc.id).unwrap();
        assert_eq!(loaded.messages[0].confidence.len(), 2);
        assert_eq!(loaded.messages[0].confidence[1].probability, 0.9);
        let mut old = c.doc.value();
        old["messages"][0]
            .as_object_mut()
            .unwrap()
            .remove("confidence");
        assert!(
            Document::parse(old).unwrap().messages[0]
                .confidence
                .is_empty()
        );
        c.heatmap = true;
        c.note.clear();
        for width in [120, 79, 40] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal.draw(|f| c.draw(f, f.area())).unwrap();
            let cells = terminal.backend().buffer().content();
            assert!(
                cells
                    .iter()
                    .any(|cell| cell.symbol() == "h" && cell.fg == NORMAL_GREEN)
            );
            let text: String = cells.iter().map(|cell| cell.symbol()).collect();
            assert!(text.contains("F6 heatmap"));
            assert!(text.contains("low high"));
        }
        let mut invalid = c.doc.value();
        invalid["messages"][0]["confidence"][0]["end"] = json!(9999);
        assert!(Document::parse(invalid).is_err());
        fs::remove_dir_all(&c.store.dir).unwrap();
    }

    #[test]
    fn ab_missing_selection_keeps_input_and_attachments_and_never_spawns() {
        let mut c = fixture();
        c.command("/ab on").unwrap();
        c.attachments
            .push(("pending.txt".into(), "attached".into()));
        c.input = "question".into();
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(c.input, "question");
        assert_eq!(c.attachments.len(), 1);
        assert!(!c.ab.busy());
        assert!(c.job.is_none());
        assert!(c.doc.messages.is_empty());
        assert!(c.note.contains("/ab a PATH"));
        assert!(!c.store.dir.exists());
        assert!(c.command("/copy").unwrap_err().contains("/ab off"));
        c.command("/ab off").unwrap();
        assert!(c.send("question").unwrap_err().contains("/model PATH"));
    }

    #[test]
    fn json_roundtrip_rename_resume_delete_and_paths() {
        let mut c = fixture();
        c.doc.model = "path with spaces/model.pssa".into();
        c.doc.system = "Be helpful\n世界".into();
        c.doc.messages.push(Message {
            role: "user".into(),
            text: "hello".into(),
            confidence: Vec::new(),
        });
        c.command("/temp 0.25").unwrap();
        c.command("/rename unicode 世界").unwrap();
        let id = c.doc.id.clone();
        let loaded = c.store.load(&id).unwrap();
        assert_eq!(loaded.model, c.doc.model);
        assert_eq!(loaded.system, c.doc.system);
        assert_eq!(loaded.config.temperature, 0.25);
        assert!(c.store.list().unwrap().contains("unicode 世界"));
        c.command("/new").unwrap();
        c.command(&format!("/open {id}")).unwrap();
        assert_eq!(c.doc.messages[0].text, "hello");
        assert!(c.store.load("../outside").is_err());
        c.command(&format!("/delete {id}")).unwrap();
        assert!(c.store.path(&id).unwrap().exists());
        c.command(&format!("/delete {id}")).unwrap();
        assert!(!c.store.path(&id).unwrap().exists());
        assert_ne!(c.doc.id, id);
        fs::remove_dir_all(&c.store.dir).unwrap();
    }
    #[test]
    fn quitting_cancels_worker_and_saves_latest_partial_reply() {
        let mut c = fixture();
        c.doc.messages.push(Message {
            role: "assistant".into(),
            text: String::new(),
            confidence: Vec::new(),
        });
        let cancel = Arc::new(AtomicBool::new(false));
        c.job = Some(Job {
            progress: Arc::new(Mutex::new(Progress {
                text: "latest snapshot".into(),
                tokens: 2,
                ..Progress::default()
            })),
            cancel: cancel.clone(),
            started: Instant::now(),
        });
        let id = c.doc.id.clone();
        let dir = c.store.dir.clone();
        drop(c);
        assert!(cancel.load(Ordering::Relaxed));
        let store = Store { dir };
        assert_eq!(store.load(&id).unwrap().messages[0].text, "latest snapshot");
        fs::remove_dir_all(store.dir).unwrap();
    }
    #[test]
    fn quitting_waits_for_speech_worker_cleanup() {
        let mut c = fixture();
        let cancel = Arc::new(AtomicBool::new(false));
        c.job = Some(Job {
            progress: Arc::new(Mutex::new(Progress {
                speech: true,
                ..Progress::default()
            })),
            cancel: cancel.clone(),
            started: Instant::now(),
        });
        let cleaned = Arc::new(AtomicBool::new(false));
        let completed = cleaned.clone();
        c.speech_thread = Some(std::thread::spawn(move || {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            completed.store(true, Ordering::Relaxed);
        }));
        drop(c);
        assert!(cleaned.load(Ordering::Relaxed));
    }
    #[test]
    fn settings_invalid_json_and_context_limits() {
        let mut c = fixture();
        for cmd in [
            "/temp NaN",
            "/top-p 0",
            "/top-p 1.1",
            "/top-k 0",
            "/max-tokens 4097",
            "/max-tokens 0",
            "/repetition-penalty 0.9",
        ] {
            assert!(c.command(cmd).is_err(), "{cmd}");
        }
        let mut v = c.doc.value();
        v["settings"]["top_p"] = json!(2);
        assert!(Document::parse(v).is_err());
        c.doc.system = "s".repeat(MAX_CONTEXT);
        assert!(c.doc.context().is_err());
        c.doc.system = "hello".into();
        assert!(c.doc.context().unwrap().starts_with("System: hello"));
    }
    #[test]
    fn text_attachments_insert_context_and_reject_binary() {
        let mut c = fixture();
        fs::create_dir_all(&c.store.dir).unwrap();
        let text = c.store.dir.join("file with spaces.txt");
        fs::write(&text, "attached 世界").unwrap();
        c.command(&format!("/attach {}", text.display())).unwrap();
        assert_eq!(c.attachments[0].1, "attached 世界");
        // Send fails before loading, but its saved prompt includes the attachment.
        c.doc.model = "does-not-exist.pssa".into();
        c.send("question").unwrap();
        let saved = c.store.load(&c.doc.id).unwrap();
        assert!(saved.context().unwrap().contains("attached 世界"));
        assert!(c.attachments.is_empty());
        while c.job.is_some() {
            c.poll();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(c.note.contains("Error:"));
        let image = c.store.dir.join("image.png");
        fs::write(&image, [0, 255, 0]).unwrap();
        assert!(
            c.command(&format!("/attach {}", image.display()))
                .unwrap_err()
                .contains("not supported by this model yet")
        );
        fs::remove_dir_all(&c.store.dir).unwrap();
    }
    #[test]
    fn checkpoint_worker_generates_and_resumes_saved_history() {
        use crate::{
            checkpoint,
            dataset::Tokenizer,
            pssa::{PSSAConfigV2, PSSALayerV2},
        };
        let mut c = fixture();
        fs::create_dir_all(&c.store.dir).unwrap();
        let tokenizer =
            Tokenizer::from_vocabulary(&["<unk>".into(), "hello".into(), "world".into()]).unwrap();
        let mut model = PSSALayerV2::new(
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
        model.vocabulary = tokenizer.ordered_vocabulary().unwrap();
        model.unembed_w.data.fill(0.0);
        model.memory.insert(&[0.1, 0.2], &[0.4, 0.2, -0.3, 0.6]);
        model.memory.insert(&[-0.3, 0.4], &[0.1, -0.2, 0.5, 0.7]);
        let path = c.store.dir.join("tiny checkpoint.pssa");
        checkpoint::save_model(&model, &path).unwrap();
        let checkpoint_before = fs::read(&path).unwrap();
        let baseline = PSSAInferenceEngine::new(&mut model, &tokenizer)
            .try_generate_chat_turn_controlled(
                "User: hello world\n\nAssistant:",
                &InferenceConfig {
                    temperature: 0.0,
                    max_new_tokens: 4,
                    ..Default::default()
                },
                |_, _| {},
                || false,
            )
            .unwrap();
        c.command(&format!("/model {}", path.display())).unwrap();
        c.command("/temp 0").unwrap();
        c.command("/max-tokens 4").unwrap();
        c.send("hello world").unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(20);
        while c.job.is_some() && Instant::now() < deadline {
            c.poll();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(c.job.is_none(), "worker did not finish");
        assert_eq!(c.tokens, 4);
        assert_eq!(c.doc.messages[1].text, "hello hello hello hello");
        assert_eq!(c.doc.messages[1].text, baseline);
        assert_eq!(
            fs::read(&path).unwrap(),
            checkpoint_before,
            "chat must not rewrite checkpoint bytes"
        );
        let snapshot = c
            .memory_view()
            .latest()
            .expect("worker publishes memory observations");
        assert_eq!(snapshot.generated_tokens, 4);
        assert_eq!(snapshot.generated_token_id, 1);
        assert_eq!(snapshot.query_token_id, 1);
        assert_eq!(
            snapshot.layers[0].weights,
            model.inf_mem_weights[..model.memory.count]
        );
        assert_eq!(snapshot.layers[0].capacity, model.memory.capacity);
        let saved = c.doc.value();
        assert!(
            saved.get("memory").is_none(),
            "runtime observation is not chat/checkpoint state"
        );
        let id = c.doc.id.clone();
        c.command("/new").unwrap();
        assert!(c.memory_view().latest().is_none());
        c.command(&format!("/open {id}")).unwrap();
        assert!(
            c.memory_view().latest().is_none(),
            "saved chats do not invent live observations"
        );
        assert_eq!(c.doc.messages.len(), 2);
        assert!(
            c.doc
                .context()
                .unwrap()
                .contains("Assistant: hello hello hello hello")
        );
        fs::remove_dir_all(&c.store.dir).unwrap();
    }
    #[test]
    fn keyboard_and_render_scrollback_narrow_and_streaming() {
        let mut c = fixture();
        c.key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert_eq!(c.input, "q");
        c.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert!(c.input.is_empty());
        c.doc.messages.push(Message {
            role: "assistant".into(),
            text: (0..80).map(|n| format!("reply {n} 世界\n")).collect(),
            confidence: Vec::new(),
        });
        c.tokens = 23;
        c.elapsed = 2.0;
        c.note = "saved".into();
        for (w, h) in [(120, 40), (79, 24), (40, 16), (25, 8), (1, 1)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| c.draw(f, f.area())).unwrap();
            if w >= 40 {
                let text = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect::<String>();
                assert!(text.contains("23 tok"));
                assert!(text.contains("11.5 tok/s"));
                assert!(text.contains("reply 79"));
                let old = c.scroll;
                c.key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
                t.draw(|f| c.draw(f, f.area())).unwrap();
                assert!(c.scroll < old);
                c.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
            }
        }
        let progress = Arc::new(Mutex::new(Progress {
            text: "partial".into(),
            tokens: 5,
            ..Progress::default()
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        c.job = Some(Job {
            progress: progress.clone(),
            cancel: cancel.clone(),
            started: Instant::now(),
        });
        c.poll();
        assert_eq!(c.doc.messages.last().unwrap().text, "partial");
        assert_eq!(c.tokens, 5);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| c.draw(f, f.area())).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(rendered.contains("STREAMING"));
        assert!(rendered.contains("partial"));
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(cancel.load(Ordering::Relaxed));
        progress.lock().unwrap().done = Some(Ok(()));
        c.poll();
        assert!(c.job.is_none());
        assert!(c.note.contains("Stopped"));
        assert_eq!(
            c.store
                .load(&c.doc.id)
                .unwrap()
                .messages
                .last()
                .unwrap()
                .text,
            "partial"
        );
        fs::remove_dir_all(&c.store.dir).unwrap();
    }
}
