//! Opt-in, deterministic token-share mixer. Exports plain UTF-8 for the existing
//! trainer; neither dataset CLI defaults nor training math are changed.
use super::{
    AMBER, accent,
    library::{self, Entry, Kind, Records, Stats},
    panel, panel_area,
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const TOKEN_CAP: u64 = 1_000_000;
const OUTPUT_CAP: u64 = 256 * 1024 * 1024;

/// Normalized token shares, without repetition. Largest possible total that
/// doesn't exhaust any enabled source, bounded by the export budget. Rounding
/// never exceeds either the source's capacity or the total budget.
pub(super) fn allocation(available: &[u64], weights: &[u16], cap: u64) -> Vec<u64> {
    if available.len() != weights.len() {
        return Vec::new();
    }
    let sum: u64 = weights.iter().map(|&w| u64::from(w)).sum();
    if sum == 0 {
        return vec![0; weights.len()];
    }
    let mut total = cap as u128;
    for (&tokens, &weight) in available.iter().zip(weights) {
        if weight > 0 {
            total = total.min(tokens as u128 * sum as u128 / weight as u128);
        }
    }
    let mut result: Vec<_> = available
        .iter()
        .zip(weights)
        .map(|(&n, &w)| (total * w as u128 / sum as u128).min(n as u128) as u64)
        .collect();
    let remainder = (total as u64).saturating_sub(result.iter().sum());
    let mut order: Vec<_> = (0..result.len())
        .filter(|&i| weights[i] > 0 && result[i] < available[i])
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(total * weights[i] as u128 % sum as u128));
    for &i in order.iter().take(remainder as usize) {
        result[i] += 1;
    }
    result
}

fn output_dir(root: &Path) -> Result<PathBuf, String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ID: AtomicU64 = AtomicU64::new(0);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let parent = root.join(".pssa-mixes");
    fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
    let dir = parent.join(format!(
        "mix-{stamp}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn prefix_for_tokens(
    text: &str,
    max: u64,
    tokenizer: &crate::dataset::Tokenizer,
) -> Result<String, String> {
    if max == 0 {
        return Ok(String::new());
    }
    let boundaries: Vec<_> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let (mut lo, mut hi) = (0, boundaries.len() - 1);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if tokenizer.try_encode(&text[..boundaries[mid]], true)?.len() as u64 <= max {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Ok(text[..boundaries[lo]].to_owned())
}

/// Sequential weighted prefixes keep the writer bounded to one record and one
/// input at a time. The manifest records actual counts as well as the targets.
fn export(root: &Path, sources: &[(PathBuf, u16, u64)], full: bool) -> Result<PathBuf, String> {
    if sources.is_empty() || (!full && sources.iter().all(|(_, _, n)| *n == 0)) {
        return Err("Mix is empty; inspect sources and increase a share".into());
    }
    let dir = output_dir(root)?;
    let result = (|| {
        let path = dir.join("corpus.txt");
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| e.to_string())?,
        );
        let tok = library::counting_tokenizer();
        let mut written = 0u64;
        let mut manifest = Vec::new();
        let mut total_tokens = 0u64;
        for (source, weight, target) in sources {
            if !full && *target == 0 {
                continue;
            }
            let mut reader = Records::open(source)?;
            let mut used = 0u64;
            let mut records = 0;
            while full || used < *target {
                let Some(text) = reader.next()? else {
                    break;
                };
                let n = tok.try_encode(&text, true)?.len() as u64;
                let text = if !full && n > target.saturating_sub(used) {
                    prefix_for_tokens(&text, target.saturating_sub(used), &tok)?
                } else {
                    text
                };
                let n = tok.try_encode(&text, true)?.len() as u64;
                written = written.saturating_add(text.len() as u64 + 1);
                if written > OUTPUT_CAP {
                    return Err("Export exceeds 256 MiB safety cap; split the dataset first".into());
                }
                out.write_all(text.as_bytes())
                    .and_then(|_| out.write_all(b"\n"))
                    .map_err(|e| e.to_string())?;
                used += n;
                records += 1;
                if records % 128 == 0 {
                    std::thread::sleep(Duration::from_millis(3));
                }
            }
            total_tokens += used;
            manifest.push(json!({"path": source, "weight": weight, "target_word_tokens": if full { serde_json::Value::Null } else { json!(target) }, "actual_word_tokens": used, "records": records}));
        }
        if total_tokens == 0 {
            return Err("Dataset contains no word tokens".into());
        }
        out.flush().map_err(|e| e.to_string())?;
        let config = json!({"version": 1, "corpus": path, "tokenizer": "existing legacy word tokenizer (not model BPE)", "order": "weighted source prefixes, no repetition", "total_word_tokens": total_tokens, "sources": manifest});
        fs::write(
            dir.join("mix.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        Ok(path)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&dir);
    }
    result
}
struct Source {
    entry: Entry,
    weight: u16,
    stats: Result<Stats, String>,
}
pub(super) struct Mixer {
    sources: Vec<Source>,
    selected: usize,
    revision: u64,
    root: PathBuf,
    note: String,
    inspect: Option<mpsc::Receiver<Vec<(Entry, Result<Stats, String>)>>>,
    export: Option<mpsc::Receiver<Result<PathBuf, String>>>,
}
impl Default for Mixer {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            selected: 0,
            revision: u64::MAX,
            root: "data".into(),
            note: "Opt-in only: defaults leave training unchanged. Enter writes corpus + mix.json."
                .into(),
            inspect: None,
            export: None,
        }
    }
}
impl Mixer {
    pub(super) fn sync(&mut self, entries: &[Entry], revision: u64, root: &Path) {
        if self.revision == revision || self.inspect.is_some() || self.export.is_some() {
            return;
        }
        self.revision = revision;
        self.root = root.to_owned();
        let entries: Vec<_> = entries
            .iter()
            .filter(|e| e.kind == Kind::Dataset)
            .take(64)
            .cloned()
            .collect();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut result = Vec::new();
            for entry in entries {
                let stats = library::dataset_stats(&entry.path);
                result.push((entry, stats));
                std::thread::sleep(Duration::from_millis(10));
            }
            let _ = tx.send(result);
        });
        self.inspect = Some(rx);
        self.note = "Inspecting up to 64 sources in background (bounded prefix samples)…".into();
    }
    pub(super) fn busy(&self) -> bool {
        self.export.is_some()
    }
    pub(super) fn poll(&mut self) -> Option<PathBuf> {
        if let Some(result) = self.inspect.as_ref().and_then(|r| r.try_recv().ok()) {
            self.inspect = None;
            self.sources = result
                .into_iter()
                .map(|(entry, stats)| {
                    let weight = if stats.is_ok() {
                        self.sources
                            .iter()
                            .find(|s| s.entry.path == entry.path)
                            .map_or(100, |s| s.weight)
                    } else {
                        // An unreadable source must not reduce every other quota
                        // to zero, including after a previously successful scan.
                        0
                    };
                    Source {
                        entry,
                        weight,
                        stats,
                    }
                })
                .collect();
            self.selected = self.selected.min(self.sources.len().saturating_sub(1));
            self.note = "↑↓ source / +/- weight (0 disables) / Enter export and fill Setup".into();
        }
        if let Some(result) = self.export.as_ref().and_then(|r| r.try_recv().ok()) {
            self.export = None;
            match result {
                Ok(path) => {
                    self.note = format!(
                        "Exported {} + mix.json; Setup Dataset filled",
                        path.display()
                    );
                    return Some(path);
                }
                Err(e) => self.note = e,
            }
        }
        None
    }
    pub(super) fn prepare_dataset(&mut self, path: PathBuf, root: &Path) -> Option<PathBuf> {
        if self.busy() {
            self.note = "An export is already running".into();
            return None;
        }
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("txt"))
        {
            return Some(path);
        }
        self.start_export(root.to_owned(), vec![(path, 100, 0)], true);
        None
    }
    fn start_export(&mut self, root: PathBuf, sources: Vec<(PathBuf, u16, u64)>, full: bool) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(export(&root, &sources, full));
        });
        self.export = Some(rx);
        self.note = "Writing UTF-8 corpus + mix.json in background (256 MiB cap)…".into();
    }
    fn quotas(&self) -> Vec<u64> {
        allocation(
            &self
                .sources
                .iter()
                .map(|s| s.stats.as_ref().map_or(0, |s| s.tokens))
                .collect::<Vec<_>>(),
            &self.sources.iter().map(|s| s.weight).collect::<Vec<_>>(),
            TOKEN_CAP,
        )
    }
    pub(super) fn key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.sources.len().saturating_sub(1))
            }
            KeyCode::Char(c @ ('+' | '-' | '=')) => {
                if let Some(source) = self.sources.get_mut(self.selected) {
                    if source.stats.is_ok() {
                        source.weight = if c == '-' {
                            source.weight.saturating_sub(5)
                        } else {
                            source.weight.saturating_add(5).min(100)
                        };
                    }
                }
            }
            KeyCode::Enter if self.export.is_none() && self.inspect.is_none() => {
                let quotas = self.quotas();
                let sources = self
                    .sources
                    .iter()
                    .zip(quotas)
                    .map(|(s, q)| (s.entry.path.clone(), s.weight, q))
                    .collect();
                self.start_export(self.root.clone(), sources, false);
            }
            _ => {}
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let quotas = self.quotas();
        let sum: u64 = self.sources.iter().map(|s| u64::from(s.weight)).sum();
        let mut lines = vec![
            Line::styled("DATASET MIX / token shares", accent()),
            Line::from("↑↓ source  +/- slider  Enter export → Setup Dataset"),
            Line::from(format!(
                "~{} word tokens total / cap {TOKEN_CAP} / no repetition",
                quotas.iter().sum::<u64>()
            )),
            Line::from("Counts use word tokens, not model BPE; big sources are estimates."),
        ];
        let visible = area.height.saturating_sub(9) as usize;
        let start = self.selected.saturating_sub(visible.saturating_sub(1));
        for (i, (source, quota)) in self
            .sources
            .iter()
            .zip(quotas)
            .enumerate()
            .skip(start)
            .take(visible)
        {
            let cells = if area.width < 80 { 8 } else { 16 };
            let filled = source.weight as usize * cells / 100;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(cells - filled));
            let share = if sum == 0 {
                0.0
            } else {
                source.weight as f64 * 100.0 / sum as f64
            };
            lines.push(Line::styled(
                format!(
                    "{} {:<16.16} [{bar}] {:>5.1}% ~{quota}",
                    if i == self.selected { "▶" } else { " " },
                    source.entry.name(),
                    share
                ),
                accent(),
            ));
            if let Err(e) = &source.stats {
                lines.push(Line::from(format!("  {}", library::clean(e))));
            }
        }
        if self.sources.is_empty() {
            lines.push(Line::from(
                "No datasets. Set the Library datasets folder first.",
            ));
        }
        lines.push(Line::from(
            "Exports weighted prefixes into .pssa-mixes/ (never overwrites inputs).",
        ));
        lines.push(Line::styled(
            library::clean(&self.note),
            ratatui::style::Style::new().fg(AMBER),
        ));
        f.render_widget(
            Paragraph::new(lines)
                .block(panel(" mixer / opt-in corpus "))
                .wrap(Wrap { trim: false }),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use library::tests::Temp;
    use ratatui::{Terminal, backend::TestBackend};
    #[test]
    fn mix_math_normalizes_limits_and_handles_zero_shares() {
        assert_eq!(allocation(&[100, 100], &[1, 3], 1000), vec![33, 100]);
        assert_eq!(allocation(&[100, 100], &[0, 0], 100), vec![0, 0]);
        assert_eq!(allocation(&[100, 0], &[1, 0], 60), vec![60, 0]);
        assert_eq!(allocation(&[100, 0], &[1, 1], 60), vec![0, 0]);
        assert_eq!(
            allocation(&[100, 100, 100], &[1, 1, 1], 10)
                .iter()
                .sum::<u64>(),
            10
        );
        assert_eq!(allocation(&[u64::MAX; 2], &[100; 2], 100), vec![50, 50]);
    }
    #[test]
    fn export_writes_trainable_text_manifest_and_leaves_inputs_alone() {
        let temp = Temp::new();
        let a = temp.0.join("a.txt");
        let b = temp.0.join("b.jsonl");
        fs::write(&a, "one two three four\n").unwrap();
        fs::write(&b, "{\"text\":\"five six seven eight\"}\n").unwrap();
        let path = export(&temp.0, &[(a.clone(), 1, 2), (b.clone(), 1, 2)], false).unwrap();
        let body =
            crate::dataset::DatasetManager::load_dataset(Some(&format!("file:{}", path.display())))
                .unwrap();
        assert_eq!(library::counting_tokenizer().encode(&body, true).len(), 4);
        assert!(body.contains("one two"));
        assert!(body.contains("five six"));
        assert!(!body.contains("text"));
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(path.parent().unwrap().join("mix.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["total_word_tokens"], 4);
        assert_eq!(fs::read_to_string(&a).unwrap(), "one two three four\n");
        assert!(export(&temp.0, &[], false).is_err());
        let full = export(&temp.0, &[(b, 100, 0)], true).unwrap();
        assert!(
            fs::read_to_string(full)
                .unwrap()
                .contains("five six seven eight")
        );
    }
    #[test]
    fn rescan_disables_failed_sources_without_losing_valid_shares() {
        let temp = Temp::new();
        let good = temp.0.join("good.txt");
        let bad = temp.0.join("bad.jsonl");
        fs::write(&good, "one two").unwrap();
        fs::write(&bad, "\"three four\"\n").unwrap();
        let entries = library::scan(&temp.0, &temp.0).0;
        let mut mixer = Mixer::default();
        mixer.sources = entries.iter().cloned().map(|entry| Source {
            stats: library::dataset_stats(&entry.path), entry, weight: 35,
        }).collect();
        fs::write(&bad, "invalid json").unwrap();
        let (tx, rx) = mpsc::channel();
        tx.send(entries.into_iter().map(|entry| {
            let stats = library::dataset_stats(&entry.path);
            (entry, stats)
        }).collect()).unwrap();
        mixer.inspect = Some(rx);
        assert!(mixer.poll().is_none());
        assert_eq!(mixer.sources.iter().find(|s| s.entry.path == good).unwrap().weight, 35);
        assert_eq!(mixer.sources.iter().find(|s| s.entry.path == bad).unwrap().weight, 0);
        assert_eq!(mixer.quotas().iter().sum::<u64>(), 2);
    }

    #[test]
    fn mixer_screen_wide_narrow_tiny() {
        let temp = Temp::new();
        fs::write(temp.0.join("a.txt"), "one two").unwrap();
        let mut mixer = Mixer::default();
        mixer.sources = library::scan(&temp.0, &temp.0)
            .0
            .into_iter()
            .map(|entry| Source {
                stats: library::dataset_stats(&entry.path),
                entry,
                weight: 100,
            })
            .collect();
        for (w, h) in [(120, 30), (79, 24), (40, 16), (1, 1), (0, 0)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| mixer.draw(f, f.area())).unwrap();
            if w >= 40 {
                let text: String = t
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(text.contains("word tokens total"));
                assert!(text.contains("a.txt"));
            }
        }
    }
}
