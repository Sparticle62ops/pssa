//! Local catalogue and bounded, off-thread dataset inspection. No weights are loaded.
use super::{AMBER, SECOND_ACCENT, accent, panel, panel_area};
use crate::dataset::Tokenizer;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, UNIX_EPOCH},
};

const SAMPLE_BYTES: usize = 256 * 1024;
const RECORD_LIMIT: u64 = 1024 * 1024;

pub(super) fn clean(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .collect()
}
pub(super) fn size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / 1048576.0)
    } else {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    }
}
fn date(seconds: u64) -> String {
    // Gregorian civil date from days since Unix epoch (UTC).
    let z = seconds as i64 / 86400 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    format!("{:04}-{m:02}-{d:02}", y + i64::from(m <= 2))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Pssa,
    Transformer,
    Dataset,
}
#[derive(Clone, Debug)]
pub(super) struct Entry {
    pub path: PathBuf,
    pub kind: Kind,
    pub bytes: u64,
    pub modified: u64,
    pub dims: String,
}
impl Entry {
    pub(super) fn name(&self) -> String {
        clean(&self.path.file_name().unwrap_or_default().to_string_lossy()).replace('\n', " ")
    }
}

/// At most 106 bytes; header hints only, not checksum/model validation.
pub(super) fn checkpoint_dims(path: &Path) -> Option<String> {
    let mut bytes = Vec::new();
    File::open(path)
        .ok()?
        .take(106)
        .read_to_end(&mut bytes)
        .ok()?;
    let version = u16::from_le_bytes(bytes.get(4..6)?.try_into().ok()?);
    let word =
        |at| -> Option<u64> { Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?)) };
    match (bytes.get(..4)?, version) {
        (b"PSSA", 6..=8) => Some(format!(
            "PSSA v{version} vocab {} latent {} state {} depth {}",
            word(22)?,
            word(30)?,
            word(38)?,
            if version == 8 { word(98)? } else { 1 }
        )),
        (b"TRFM", 1) => Some(format!(
            "TRFM vocab {} width {} heads {} ff {}",
            word(22)?,
            word(30)?,
            word(38)?,
            word(46)?
        )),
        (b"PSSA", 5) => {
            let n = |at| -> Option<u32> {
                Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
            };
            Some(format!(
                "PSSA v5 vocab {} latent {} state {} (legacy)",
                n(6)?,
                n(10)?,
                n(14)?
            ))
        }
        _ => None,
    }
}

pub(super) fn scan(models: &Path, datasets: &Path) -> (Vec<Entry>, String) {
    let mut out = Vec::new();
    let mut notes = Vec::new();
    for (dir, model) in [(models, true), (datasets, false)] {
        let items = match fs::read_dir(dir) {
            Ok(items) => items,
            Err(e) => {
                notes.push(format!("{}: {e}", dir.display()));
                continue;
            }
        };
        for (index, item) in items.enumerate() {
            if index >= 10_000 || out.len() >= 10_000 {
                notes.push("Catalogue scan capped at 10000 directory entries".into());
                break;
            }
            let Ok(item) = item else {
                continue;
            };
            let path = item.path();
            let ext = path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_ascii_lowercase();
            let kind = match (model, ext.as_str()) {
                (true, "pssa") => Kind::Pssa,
                (true, "trfm") => Kind::Transformer,
                (false, "txt" | "parquet" | "jsonl") => Kind::Dataset,
                _ => continue,
            };
            let Ok(meta) = item.metadata() else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let dims = if kind == Kind::Dataset {
                String::new()
            } else {
                checkpoint_dims(&path).unwrap_or_else(|| "Header unavailable / unsupported".into())
            };
            out.push(Entry {
                path,
                kind,
                bytes: meta.len(),
                modified: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs()),
                dims,
            });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    (out, notes.join("; "))
}

/// Shared persisted TUI preferences. Preserve unknown fields for other phases.
pub(super) struct Config {
    pub models: PathBuf,
    pub datasets: PathBuf,
    pub eval_prompts: PathBuf,
    path: PathBuf,
}
impl Config {
    pub(super) fn load(models: PathBuf) -> Self {
        let root = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".config")
            });
        Self::load_at(root.join("pssa/tui.json"), models)
    }
    fn load_at(path: PathBuf, models: PathBuf) -> Self {
        let value = File::open(&path)
            .ok()
            .and_then(|f| serde_json::from_reader::<_, Value>(f.take(64 * 1024)).ok())
            .unwrap_or(Value::Null);
        Self {
            models: value["models_folder"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or(models),
            datasets: value["datasets_folder"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or_else(|| "data".into()),
            eval_prompts: value["eval_prompts"]
                .as_str()
                .map(PathBuf::from)
                .unwrap_or_default(),
            path,
        }
    }
    pub(super) fn save(&self) -> Result<(), String> {
        let mut value = if self.path.exists() {
            let f = File::open(&self.path).map_err(|e| e.to_string())?;
            serde_json::from_reader::<_, Value>(f.take(64 * 1024))
                .map_err(|e| format!("Config not overwritten: {e}"))?
        } else {
            json!({})
        };
        let obj = value
            .as_object_mut()
            .ok_or("TUI config must be a JSON object")?;
        obj.insert("models_folder".into(), json!(self.models));
        obj.insert("datasets_folder".into(), json!(self.datasets));
        obj.insert("eval_prompts".into(), json!(self.eval_prompts));
        fs::create_dir_all(self.path.parent().unwrap()).map_err(|e| e.to_string())?;
        let tmp = self
            .path
            .with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&tmp, serde_json::to_vec_pretty(&value).unwrap()).map_err(|e| e.to_string())?;
        fs::rename(&tmp, &self.path).map_err(|e| e.to_string())
    }
}

/// Stream text/JSONL or use an optional, strictly local pyarrow adapter.
/// Each record is bounded, including malformed files with no line terminator.
pub(super) struct Records {
    reader: Box<dyn BufRead + Send>,
    child: Option<Child>,
    json: bool,
    pub total: Option<u64>,
    pub read_bytes: u64,
}
impl Records {
    pub(super) fn open(path: &Path) -> Result<Self, String> {
        let ext = path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        if ext == "parquet" {
            let mut child = Command::new("python3")
                .args(["-u", "-c", include_str!("../../scripts/tui_parquet.py")])
                .arg(path)
                .env("OMP_NUM_THREADS", "1")
                .env("ARROW_NUM_THREADS", "1")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("Parquet requires local python3 + pyarrow: {e}"))?;
            let reader = Box::new(BufReader::new(child.stdout.take().unwrap()));
            let mut this = Self {
                reader,
                child: Some(child),
                json: true,
                total: None,
                read_bytes: 0,
            };
            let header = this
                .raw()?
                .ok_or("Parquet reader failed (requires local python3 + pyarrow)")?;
            let value: Value = serde_json::from_str(&header).map_err(|e| e.to_string())?;
            if let Some(e) = value["error"].as_str() {
                return Err(e.into());
            }
            this.total = value["records"].as_u64();
            Ok(this)
        } else {
            Ok(Self {
                reader: Box::new(BufReader::new(File::open(path).map_err(|e| e.to_string())?)),
                child: None,
                json: ext == "jsonl",
                total: None,
                read_bytes: 0,
            })
        }
    }
    fn raw(&mut self) -> Result<Option<String>, String> {
        let mut bytes = Vec::new();
        let n = self
            .reader
            .by_ref()
            .take(RECORD_LIMIT + 1)
            .read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            if let Some(child) = &mut self.child {
                let status = child.wait().map_err(|e| e.to_string())?;
                if !status.success() {
                    return Err(format!("Local Parquet reader exited with {status}"));
                }
            }
            return Ok(None);
        }
        self.read_bytes += n as u64;
        if n as u64 > RECORD_LIMIT {
            return Err("Record exceeds 1 MiB; split long records first".into());
        }
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| "Dataset must be UTF-8".into())
    }
    pub(super) fn next(&mut self) -> Result<Option<String>, String> {
        let Some(raw) = self.raw()? else {
            return Ok(None);
        };
        if !self.json {
            return Ok(Some(raw.trim_end_matches(['\n', '\r']).into()));
        }
        let value: Value =
            serde_json::from_str(&raw).map_err(|e| format!("Invalid JSONL record: {e}"))?;
        if let Some(text) = value.as_str() {
            return Ok(Some(text.into()));
        }
        if let Some(e) = value["error"].as_str() {
            return Err(e.into());
        }
        for field in ["text", "content", "body"] {
            if let Some(text) = value[field].as_str() {
                return Ok(Some(text.into()));
            }
        }
        Err("JSONL needs a string or a text, content or body string field".into())
    }
}
impl Drop for Records {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Stats {
    pub records: u64,
    pub tokens: u64,
    pub estimated: bool,
    pub records_estimated: bool,
    pub samples: Vec<String>,
    pub sampled_records: u64,
}
/// Word-tokenizer counts are independent of its vocabulary; this uses the
/// existing tokenizer, not a bytes/4 approximation or a newly trained BPE.
pub(super) fn counting_tokenizer() -> Tokenizer {
    Tokenizer::from_vocabulary(&["<unk>".into(), "sample".into()]).expect("fixed valid vocabulary")
}
pub(super) fn dataset_stats(path: &Path) -> Result<Stats, String> {
    let bytes = fs::metadata(path).map_err(|e| e.to_string())?.len();
    let tok = counting_tokenizer();
    let mut source = Records::open(path)?;
    let mut count = 0u64;
    let mut tokens = 0u64;
    let mut text_bytes = 0;
    let mut samples = Vec::new();
    let mut eof = false;
    while text_bytes < SAMPLE_BYTES && count < 2048 {
        let Some(text) = source.next()? else {
            eof = true;
            break;
        };
        tokens += tok.try_encode(&text, true)?.len() as u64;
        text_bytes += text.len();
        count += 1;
        if samples.len() < 4 {
            samples.push(clean(&text).replace('\n', " ").chars().take(180).collect());
        }
        if count % 128 == 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let estimated = !eof && source.total.map_or(source.read_bytes < bytes, |n| n > count);
    let total = source.total.unwrap_or_else(|| {
        if estimated {
            // Rounding can hide a short unread tail. It is still a sample, not
            // an exact count, even when extrapolation rounds back to `count`.
            ((count as f64 * bytes as f64 / source.read_bytes.max(1) as f64).round() as u64)
                .max(count + 1)
        } else {
            count
        }
    });
    let tokens = if estimated {
        (tokens as f64 * total as f64 / count.max(1) as f64).round() as u64
    } else {
        tokens
    };
    Ok(Stats {
        records: total,
        tokens,
        estimated,
        records_estimated: estimated && source.total.is_none(),
        samples,
        sampled_records: count,
    })
}

pub(super) enum Pick {
    Chat(PathBuf),
    Resume(PathBuf),
    Train(PathBuf),
}
pub(super) struct Library {
    pub config: Config,
    pub entries: Vec<Entry>,
    pub revision: u64,
    selected: usize,
    rescan: bool,
    runtime_models: Option<PathBuf>,
    runtime_dataset: Option<PathBuf>,
    runtime_revision: u64,
    edit: Option<(char, String)>,
    note: String,
    scan_job: Option<mpsc::Receiver<(Vec<Entry>, String)>>,
    stats_job: Option<mpsc::Receiver<(PathBuf, Result<Stats, String>)>>,
    stats: Option<(PathBuf, Result<Stats, String>)>,
}
impl Library {
    pub(super) fn new(models: PathBuf) -> Self {
        let mut this = Self {
            config: Config::load(models),
            entries: Vec::new(),
            revision: 0,
            selected: 0,
            rescan: false,
            runtime_models: None,
            runtime_dataset: None,
            runtime_revision: 0,
            edit: None,
            note: "m models folder / d datasets folder / r rescan".into(),
            scan_job: None,
            stats_job: None,
            stats: None,
        };
        this.refresh();
        this
    }
    pub(super) fn editing(&self) -> bool {
        self.edit.is_some()
    }
    pub(super) fn watch_run(&mut self, models: Option<PathBuf>, revision: u64) {
        if self.runtime_models != models || self.runtime_revision != revision {
            self.runtime_models = models;
            self.runtime_revision = revision;
            self.refresh();
        }
    }

    pub(super) fn watch_dataset(&mut self, dataset: Option<PathBuf>) {
        if self.runtime_dataset != dataset {
            self.runtime_dataset = dataset;
            self.refresh();
        }
    }

    fn refresh(&mut self) {
        if self.scan_job.is_some() {
            self.rescan = true;
            self.note = "Folder rescan queued after the current scan".into();
            return;
        }
        let (tx, rx) = mpsc::channel();
        let (models, datasets) = (self.config.models.clone(), self.config.datasets.clone());
        let runtime = self.runtime_models.clone().filter(|path| *path != models);
        let runtime_dataset = self.runtime_dataset.clone();
        std::thread::spawn(move || {
            let mut result = scan(&models, &datasets);
            if let Some(runtime) = runtime {
                let extra = scan(&runtime, &datasets);
                for entry in extra.0 {
                    if !result.0.iter().any(|existing| existing.path == entry.path) {
                        result.0.push(entry);
                    }
                }
                result.0.sort_by(|a, b| a.path.cmp(&b.path));
            }
            if let Some(path) = runtime_dataset
                && !result.0.iter().any(|entry| entry.path == path)
                && let Ok(meta) = fs::metadata(&path)
                && meta.is_file()
            {
                result.0.push(Entry {
                    path,
                    kind: Kind::Dataset,
                    bytes: meta.len(),
                    modified: meta
                        .modified()
                        .ok()
                        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                        .map_or(0, |time| time.as_secs()),
                    dims: String::new(),
                });
                result.0.sort_by(|a, b| a.path.cmp(&b.path));
            }
            let _ = tx.send(result);
        });
        self.scan_job = Some(rx);
        self.note = "Scanning folder entries and headers in background…".into();
    }
    fn request_stats(&mut self) {
        let Some(entry) = self
            .entries
            .get(self.selected)
            .filter(|e| e.kind == Kind::Dataset)
        else {
            return;
        };
        if self.stats_job.is_some() || self.stats.as_ref().is_some_and(|(p, _)| *p == entry.path) {
            return;
        }
        let path = entry.path.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = dataset_stats(&path);
            let _ = tx.send((path, result));
        });
        self.stats_job = Some(rx);
    }
    pub(super) fn poll(&mut self) {
        if let Some(result) = self.scan_job.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.scan_job = None;
            self.entries = result.0;
            self.selected = self.selected.min(self.entries.len().saturating_sub(1));
            self.note = if result.1.is_empty() {
                format!(
                    "{} files / selected dataset stats are sampled off-thread",
                    self.entries.len()
                )
            } else {
                result.1
            };
            self.stats = None;
            self.revision += 1;
            if std::mem::take(&mut self.rescan) {
                self.refresh();
            }
        }
        if let Some(result) = self.stats_job.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.stats = Some(result);
            self.stats_job = None;
        }
        self.request_stats();
    }
    pub(super) fn key(&mut self, key: KeyEvent) -> Option<Pick> {
        if let Some((field, text)) = &mut self.edit {
            match key.code {
                KeyCode::Esc => self.edit = None,
                KeyCode::Backspace => {
                    text.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => text.clear(),
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                        && text.len() < 4096 =>
                {
                    text.push(c)
                }
                KeyCode::Enter => {
                    let path = PathBuf::from(text.as_str());
                    if !path.is_dir() {
                        self.note = "Folder must already exist".into();
                        return None;
                    }
                    if *field == 'm' {
                        self.config.models = path;
                    } else {
                        self.config.datasets = path;
                    }
                    self.edit = None;
                    match self.config.save() {
                        Ok(()) => self.refresh(),
                        Err(e) => self.note = e,
                    }
                }
                _ => {}
            }
            return None;
        }
        match key.code {
            KeyCode::Char(c @ ('m' | 'd')) => {
                self.edit = Some((
                    c,
                    if c == 'm' {
                        &self.config.models
                    } else {
                        &self.config.datasets
                    }
                    .to_string_lossy()
                    .into_owned(),
                ))
            }
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.entries.len().saturating_sub(1))
            }
            KeyCode::Enter => {
                self.stats = None;
                self.request_stats();
            }
            KeyCode::Char(c @ ('c' | 'u' | 't')) => {
                if let Some(entry) = self.entries.get(self.selected) {
                    return match (c, &entry.kind) {
                        ('c', Kind::Pssa | Kind::Transformer) => {
                            Some(Pick::Chat(entry.path.clone()))
                        }
                        ('u', Kind::Pssa | Kind::Transformer) => {
                            Some(Pick::Resume(entry.path.clone()))
                        }
                        ('t', Kind::Dataset) => Some(Pick::Train(entry.path.clone())),
                        _ => {
                            self.note = "c chat / u resume require a checkpoint; t train requires a dataset".into();
                            None
                        }
                    };
                }
            }
            _ => {}
        }
        None
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let block = panel(" library / local files ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Percentage(50),
                Constraint::Min(0),
            ])
            .split(inner);
        let header = if let Some((field, text)) = &self.edit {
            format!(
                "Edit {} folder: {}█\nEnter save / Esc cancel / Ctrl+U clear",
                if *field == 'm' { "models" } else { "datasets" },
                clean(text)
            )
        } else {
            format!(
                "models {}\ndata {}\nm/d folders  r rescan  ↑↓ select  c chat  u resume  t train",
                clean(&self.config.models.display().to_string()),
                clean(&self.config.datasets.display().to_string())
            )
        };
        f.render_widget(Paragraph::new(header).style(accent()), parts[0]);
        let start = self
            .selected
            .saturating_sub(parts[1].height.saturating_sub(1) as usize);
        let mut rows = Vec::new();
        for (i, entry) in self
            .entries
            .iter()
            .enumerate()
            .skip(start)
            .take(parts[1].height as usize)
        {
            let text = if area.width < 80 {
                format!(
                    "{} {}  {}",
                    if i == self.selected { "▶" } else { " " },
                    entry.name(),
                    size(entry.bytes)
                )
            } else {
                format!(
                    "{} {:<32.32} {:>10} {}  {}",
                    if i == self.selected { "▶" } else { " " },
                    entry.name(),
                    size(entry.bytes),
                    date(entry.modified),
                    entry.dims
                )
            };
            rows.push(Line::styled(
                text,
                if i == self.selected {
                    accent()
                } else {
                    ratatui::style::Style::new().fg(SECOND_ACCENT)
                },
            ));
        }
        if rows.is_empty() {
            rows.push(Line::from(
                "No files. Set folders with m / d, then r to rescan.",
            ));
        }
        f.render_widget(Paragraph::new(rows), parts[1]);
        let mut detail = vec![Line::styled(
            clean(&self.note),
            ratatui::style::Style::new().fg(AMBER),
        )];
        if let Some(entry) = self.entries.get(self.selected) {
            detail.push(Line::from(format!(
                "{} / {} / {} UTC",
                entry.name(),
                size(entry.bytes),
                date(entry.modified)
            )));
            if entry.kind != Kind::Dataset {
                detail.push(Line::from(entry.dims.clone()));
                detail.push(Line::from(
                    "Header hints only; weights validated when opened.",
                ));
            } else if let Some((_, result)) =
                self.stats.as_ref().filter(|(path, _)| *path == entry.path)
            {
                match result {
                    Ok(stats) => {
                        detail.push(Line::styled(
                            format!(
                                "{} records{} / ~{} word tokens{}",
                                stats.records,
                                if stats.records_estimated {
                                    " (estimate)"
                                } else {
                                    ""
                                },
                                stats.tokens,
                                if stats.estimated {
                                    " (sample estimate)"
                                } else {
                                    " (not model BPE)"
                                }
                            ),
                            accent(),
                        ));
                        detail.push(Line::from(format!(
                            "Prefix sample: {} records; estimates may be biased.",
                            stats.sampled_records
                        )));
                        detail.extend(stats.samples.iter().map(|s| Line::from(format!("│ {s}"))));
                    }
                    Err(e) => detail.push(Line::from(clean(e))),
                }
            } else {
                detail.push(Line::from("Inspecting dataset in background…"));
            }
        }
        f.render_widget(Paragraph::new(detail).wrap(Wrap { trim: false }), parts[2]);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(in crate::tui) struct Temp(pub PathBuf);
    impl Temp {
        pub fn new() -> Self {
            static ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "pssa-library-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn scan_headers_extensions_spaces_and_missing_folders() {
        let temp = Temp::new();
        let mut header = b"PSSA".to_vec();
        header.extend(8u16.to_le_bytes());
        header.resize(22, 0);
        for n in [2048u64, 32, 8, 4, 16, 64] {
            header.extend(n.to_le_bytes());
        }
        header.resize(98, 0);
        header.extend(3u64.to_le_bytes());
        fs::write(temp.0.join("model name.pssa"), &header).unwrap();
        let mut trfm = b"TRFM".to_vec();
        trfm.extend(1u16.to_le_bytes());
        trfm.resize(22, 0);
        for n in [2048u64, 32, 4, 64, 8] {
            trfm.extend(n.to_le_bytes());
        }
        fs::write(temp.0.join("model.trfm"), trfm).unwrap();
        for name in ["sample.txt", "rows.jsonl", "table.parquet", "ignore.bin"] {
            fs::write(temp.0.join(name), "x").unwrap();
        }
        fs::create_dir(temp.0.join("directory.txt")).unwrap();
        let (entries, note) = scan(&temp.0, &temp.0);
        assert_eq!(entries.len(), 5);
        assert!(note.is_empty());
        assert!(
            entries
                .iter()
                .any(|e| e.dims.contains("latent 32 state 8 depth 3"))
        );
        assert!(entries.iter().any(|e| e.dims.contains("heads 4")));
        assert!(!scan(&temp.0.join("missing"), &temp.0).1.is_empty());
        assert_eq!(date(0), "1970-01-01");
    }
    #[test]
    fn stats_exact_small_sampled_large_and_jsonl() {
        let temp = Temp::new();
        let path = temp.0.join("text.txt");
        fs::write(&path, "Hello world!\nAnother line.").unwrap();
        let stats = dataset_stats(&path).unwrap();
        assert_eq!((stats.records, stats.tokens), (2, 6));
        assert!(!stats.estimated);
        fs::write(&path, "word word\n".repeat(4096)).unwrap();
        let stats = dataset_stats(&path).unwrap();
        assert!(stats.estimated);
        assert_eq!(stats.records, 4096);
        assert_eq!(stats.tokens, 8192);
        fs::write(&path, format!("{}x", "one two three four\n".repeat(2048))).unwrap();
        let stats = dataset_stats(&path).unwrap();
        assert!(stats.estimated && stats.records_estimated, "an unread short tail is still estimated");
        assert_eq!(stats.records, 2049);
        let path = temp.0.join("rows.jsonl");
        fs::write(&path, "{\"text\":\"hello there!\"}\n\"again\"\n").unwrap();
        assert_eq!(dataset_stats(&path).unwrap().tokens, 4);
        fs::write(&path, "bad json").unwrap();
        assert!(dataset_stats(&path).is_err());
        let path = temp.0.join("long.txt");
        fs::write(&path, vec![b'a'; RECORD_LIMIT as usize + 2]).unwrap();
        assert!(dataset_stats(&path).is_err());
    }
    #[test]
    fn parquet_adapter_streams_local_text_when_pyarrow_is_available() {
        if !Command::new("python3")
            .args(["-c", "import pyarrow.parquet"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            eprintln!("optional local pyarrow unavailable; Parquet adapter fixture skipped");
            return;
        }
        let temp = Temp::new();
        let path = temp.0.join("table with spaces.parquet");
        assert!(Command::new("python3").args(["-c", "import pyarrow as a, pyarrow.parquet as p, sys; p.write_table(a.table({'text':['hello world','another row']}), sys.argv[1])"]).arg(&path).status().unwrap().success());
        let stats = dataset_stats(&path).unwrap();
        assert_eq!((stats.records, stats.tokens), (2, 4));
        assert!(!stats.estimated);
        assert_eq!(stats.samples, ["hello world", "another row"]);
    }

    #[test]
    fn config_roundtrip_preserves_other_phases() {
        let temp = Temp::new();
        let path = temp.0.join("tui.json");
        fs::write(&path, "{\"other_phase\":17}").unwrap();
        let mut config = Config::load_at(path.clone(), "chain".into());
        config.models = temp.0.join("models with spaces");
        config.datasets = temp.0.join("data with spaces");
        config.eval_prompts = temp.0.join("prompts.json");
        config.save().unwrap();
        let loaded = Config::load_at(path.clone(), "ignored".into());
        assert_eq!(loaded.models, config.models);
        assert_eq!(loaded.datasets, config.datasets);
        assert_eq!(loaded.eval_prompts, config.eval_prompts);
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["other_phase"], 17);
        fs::write(&path, "invalid config").unwrap();
        assert!(config.save().is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "invalid config");
    }
    #[test]
    fn current_run_checkpoints_refresh_without_replacing_configured_folders() {
        let temp = Temp::new();
        let configured = temp.0.join("configured");
        let runtime = temp.0.join("live run");
        fs::create_dir_all(&configured).unwrap();
        fs::create_dir_all(&runtime).unwrap();
        let mut library = Library {
            config: Config::load_at(temp.0.join("cfg"), configured.clone()),
            entries: Vec::new(),
            revision: 0,
            selected: 0,
            rescan: false,
            runtime_models: None,
            runtime_dataset: None,
            runtime_revision: 0,
            edit: None,
            note: String::new(),
            scan_job: None,
            stats_job: None,
            stats: None,
        };
        library.watch_run(Some(runtime.clone()), 0);
        fs::write(runtime.join("new.pssa"), b"checkpoint header fixture").unwrap();
        library.watch_run(Some(runtime.clone()), 1);
        let corpus = temp.0.join("external corpus.txt");
        fs::write(&corpus, "actual dataset text").unwrap();
        let configured_datasets = library.config.datasets.clone();
        library.watch_dataset(Some(corpus.clone()));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while library.scan_job.is_some() && std::time::Instant::now() < deadline {
            library.poll();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(library.config.models, configured);
        assert_eq!(library.config.datasets, configured_datasets);
        assert!(
            library
                .entries
                .iter()
                .any(|entry| entry.path == corpus && entry.kind == Kind::Dataset)
        );
        assert!(
            library
                .entries
                .iter()
                .any(|entry| entry.path == runtime.join("new.pssa"))
        );
        assert_eq!(
            library
                .entries
                .iter()
                .filter(|entry| entry.path == runtime.join("new.pssa"))
                .count(),
            1
        );
    }

    #[test]
    fn library_renders_wide_narrow_and_tiny() {
        let temp = Temp::new();
        fs::write(temp.0.join("sample.txt"), "sample text").unwrap();
        let library = Library {
            config: Config::load_at(temp.0.join("cfg"), temp.0.clone()),
            entries: scan(&temp.0, &temp.0).0,
            revision: 0,
            selected: 0,
            rescan: false,
            runtime_models: None,
            runtime_dataset: None,
            runtime_revision: 0,
            edit: None,
            note: "Ready".into(),
            scan_job: None,
            stats_job: None,
            stats: Some((
                temp.0.join("sample.txt"),
                dataset_stats(&temp.0.join("sample.txt")),
            )),
        };
        for (w, h) in [(120, 30), (79, 24), (40, 16), (1, 1), (0, 0)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| library.draw(f, f.area())).unwrap();
            if w >= 40 {
                let text: String = t
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("sample.txt"));
                assert!(text.contains("word tokens"));
            }
        }
    }
}
