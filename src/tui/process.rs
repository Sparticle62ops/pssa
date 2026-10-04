//! Bounded, non-blocking child jobs for UI-only scoring and benchmarks.
//! Existing CLI commands remain the source of truth; no shell interpolation.
use std::{
    io::{self, Read},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
};

pub(super) fn clean(text: &str) -> String {
    super::strip_ansi(text).chars().filter(|c| !c.is_control() || *c == '\t').collect()
}
fn read_lines(mut input: impl Read, tx: SyncSender<String>) {
    let mut chunk = [0; 4096];
    let mut line = Vec::new();
    loop {
        let n = match input.read(&mut chunk) { Ok(0) | Err(_) => break, Ok(n) => n };
        for &b in &chunk[..n] {
            if b == b'\n' || b == b'\r' {
                if !line.is_empty() && tx.send(clean(&String::from_utf8_lossy(&line))).is_err() { return; }
                line.clear();
            } else if line.len() < 16 * 1024 { line.push(b); }
        }
    }
    if !line.is_empty() { let _ = tx.send(clean(&String::from_utf8_lossy(&line))); }
}

pub(super) struct Job {
    child: Child,
    rx: Receiver<String>,
    status: Option<ExitStatus>,
    drained: bool,
}
impl Job {
    pub fn start(args: &[String]) -> Result<Self, String> {
        let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        cmd.args(args);
        Self::spawn(cmd)
    }
    fn spawn(mut cmd: Command) -> Result<Self, String> {
        let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .env("NO_COLOR", "1").spawn().map_err(|e| format!("Cannot start child: {e}"))?;
        let (tx, rx) = mpsc::sync_channel(256);
        let stdout = child.stdout.take().ok_or("Child stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Child stderr unavailable")?;
        let err_tx = tx.clone();
        std::thread::spawn(move || read_lines(stdout, tx));
        std::thread::spawn(move || read_lines(stderr, err_tx));
        Ok(Self { child, rx, status: None, drained: false })
    }
    pub fn poll(&mut self) -> io::Result<(Vec<String>, Option<bool>)> {
        let mut lines = Vec::new();
        for _ in 0..256 {
            match self.rx.try_recv() {
                Ok(line) => lines.push(line),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { self.drained = true; break; }
            }
        }
        if self.status.is_none() { self.status = self.child.try_wait()?; }
        Ok((lines, self.status.filter(|_| self.drained).map(|s| s.success())))
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        // Only direct project CLI children (no shell or grandchildren) are used.
        if self.status.is_none() { let _ = self.child.kill(); }
        let _ = self.child.wait();
        // Dropping rx unblocks bounded readers, including on cancellation.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_lines_strip_escapes_and_keep_partial_unicode() {
        let (tx, rx) = mpsc::sync_channel(8);
        read_lines("\x1b[31merror\x1b[0m\rnext\n世界".as_bytes(), tx);
        assert_eq!(rx.iter().collect::<Vec<_>>(), ["error", "next", "世界"]);
        let (tx, rx) = mpsc::sync_channel(8);
        read_lines(vec![b'x'; 100_000].as_slice(), tx);
        assert_eq!(rx.recv().unwrap().len(), 16 * 1024);
    }
    #[cfg(unix)]
    #[test]
    fn child_delivers_stdout_stderr_before_completion_and_is_reaped_on_drop() {
        use std::time::{Duration, Instant};
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf 'one\\n'; printf 'two\\n' >&2"]);
        let mut job = Job::spawn(cmd).unwrap();
        let mut all = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (lines, done) = job.poll().unwrap(); all.extend(lines);
            if let Some(ok) = done { assert!(ok); break; }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        all.sort(); assert_eq!(all, ["one", "two"]);
        let mut cmd = Command::new("sleep"); cmd.arg("30");
        let job = Job::spawn(cmd).unwrap();
        let pid = job.child.id(); drop(job);
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
}
