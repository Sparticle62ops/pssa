//! Bounded comparison-log reloads away from the raw-mode UI thread.
use super::{MetricSample, RunState};
use std::{
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

const MAX_BYTES: u64 = 8 * 1024 * 1024;

fn read_log(path: &Path) -> Result<Vec<MetricSample>, String> {
    // Reject FIFOs/devices before open, which could otherwise wait forever.
    let metadata = fs::metadata(path).map_err(|e| format!("Cannot read comparison log: {e}"))?;
    if !metadata.is_file() {
        return Err("Comparison requires a regular log file, not a directory/device/pipe".into());
    }
    if metadata.len() > MAX_BYTES {
        return Err("Comparison log exceeds the 8 MiB display cap".into());
    }
    let file = fs::File::open(path).map_err(|e| format!("Cannot open comparison log: {e}"))?;
    let mut state = RunState::default();
    let mut bytes = 0;
    for line in BufReader::new(file.take(MAX_BYTES + 1)).lines() {
        let line = line.map_err(|e| format!("Cannot parse comparison log: {e}"))?;
        bytes += line.len() as u64 + 1;
        if bytes > MAX_BYTES {
            return Err("Comparison log exceeds the 8 MiB display cap".into());
        }
        state.ingest(&line);
    }
    Ok(state.metric_series)
}

#[derive(Default)]
pub(super) struct ComparisonLog {
    pending: Option<Receiver<Result<Vec<MetricSample>, String>>>,
    last_scan: Option<Instant>,
}
impl ComparisonLog {
    pub(super) fn poll(&mut self, path: &Path, state: &mut RunState) {
        let result = self.pending.as_ref().map(Receiver::try_recv);
        match result {
            Some(Ok(result)) => {
                self.pending = None;
                match result {
                    Ok(samples) => {
                        state.comparison_series = samples;
                        state.comparison_error = None;
                        state.comparison_label = path.file_name().map(|s| s.to_string_lossy().into_owned());
                    }
                    Err(error) => {
                        state.comparison_series.clear();
                        state.comparison_label = None;
                        state.comparison_error = Some(error);
                    }
                }
            }
            Some(Err(TryRecvError::Disconnected)) => {
                self.pending = None;
                state.comparison_error = Some("Comparison worker stopped; retrying".into());
            }
            _ => {}
        }
        if self.pending.is_some() || self.last_scan.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
            return;
        }
        self.last_scan = Some(Instant::now());
        let path = PathBuf::from(path);
        let (tx, rx) = mpsc::sync_channel(1);
        match std::thread::Builder::new().name("tui-comparison".into()).spawn(move || {
            let _ = tx.send(read_log(&path));
        }) {
            Ok(_) => self.pending = Some(rx),
            Err(error) => state.comparison_error = Some(format!("Cannot start comparison worker: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn regular_comparison_is_bounded_and_invalid_sources_are_explicit() {
        let temp = super::super::library::tests::Temp::new();
        let log = temp.0.join("comparison.log");
        fs::write(&log, "loss=3 tokens_per_second=100\nloss=2 tokens_per_second=120\n").unwrap();
        let samples = read_log(&log).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[1].loss, Some(2.0));
        assert!(read_log(&temp.0).unwrap_err().contains("regular"));
        fs::File::create(&log).unwrap().set_len(MAX_BYTES + 1).unwrap();
        assert!(read_log(&log).unwrap_err().contains("8 MiB"));
    }

    #[test]
    fn pending_worker_does_not_block_drawing_or_hide_errors() {
        let (tx, rx) = mpsc::sync_channel(1);
        let mut loader = ComparisonLog { pending: Some(rx), last_scan: Some(Instant::now()) };
        let mut state = RunState { graph_view: super::super::GraphView::Comparison, ..Default::default() };
        loader.poll(Path::new("unused"), &mut state);
        assert!(loader.pending.is_some());
        tx.send(Err("Comparison requires a regular log file".into())).unwrap();
        loader.poll(Path::new("unused"), &mut state);
        assert!(loader.pending.is_none());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| super::super::draw(f, &state, 0)).unwrap();
        let text: String = terminal.backend().buffer().content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("comparison unavailable"));
        assert!(text.contains("regular log file"));
    }
}
