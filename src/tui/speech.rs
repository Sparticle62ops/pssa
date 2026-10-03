//! Optional local speech capture. No networking or native audio dependencies.
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

pub(super) struct Config {
    binary: PathBuf,
    model: PathBuf,
    recorder: PathBuf,
}
fn executable(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    })
}
pub(super) fn discover() -> Result<Config, String> {
    let binary=std::env::var_os("OXIDE_WHISPER_BIN").map(PathBuf::from)
        .or_else(||executable("whisper-cli")).or_else(||executable("main"))
        .filter(|p|p.is_file()).ok_or("Speech needs an existing local whisper.cpp whisper-cli/main; set OXIDE_WHISPER_BIN. Nothing is downloaded.")?;
    let model=std::env::var_os("OXIDE_WHISPER_MODEL").map(PathBuf::from)
        .or_else(||["models/ggml-base.en.bin","models/ggml-base.bin","models/ggml-tiny.en.bin"].into_iter().map(PathBuf::from).find(|p|p.is_file()))
        .filter(|p|p.is_file()).ok_or("Set OXIDE_WHISPER_MODEL to an existing local whisper.cpp ggml model. Nothing is downloaded.")?;
    let recorder=executable("arecord").ok_or("Speech capture needs local arecord (Linux ALSA); no audio libraries are linked into oxide.")?;
    Ok(Config {
        binary,
        model,
        recorder,
    })
}
fn run(mut cmd: Command, cancel: &AtomicBool, timeout: Duration) -> Result<(), String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Cannot start local speech tool: {e}"))?;
    let start = Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) || start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Speech cancelled or timed out; no transcript sent".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "Local speech tool failed ({status}); check microphone, binary and model"
                    ))
                };
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e.to_string());
            }
        }
    }
}
struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn private_temp() -> Result<Temp, String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxide-speech-{}-{stamp}", std::process::id()));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&dir).map_err(|e| e.to_string())?;
    Ok(Temp(dir))
}
pub(super) fn transcribe(cfg: Config, cancel: &AtomicBool) -> Result<String, String> {
    let temp = private_temp()?;
    let wav = temp.0.join("capture.wav");
    let out = temp.0.join("transcript");
    let mut record = Command::new(cfg.recorder);
    record
        .args([
            "-q", "-d", "10", "-r", "16000", "-c", "1", "-f", "S16_LE", "-t", "wav",
        ])
        .arg(&wav);
    run(record, cancel, Duration::from_secs(20))?;
    let mut whisper = Command::new(cfg.binary);
    whisper
        .arg("-m")
        .arg(cfg.model)
        .arg("-f")
        .arg(&wav)
        .arg("-otxt")
        .arg("-of")
        .arg(&out);
    run(whisper, cancel, Duration::from_secs(180))?;
    transcript(&out.with_extension("txt"))
}
fn transcript(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|e| format!("No whisper transcript: {e}"))?
        .take(64 * 1024 + 1)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    if text.len() > 64 * 1024 {
        return Err("Transcript exceeds 64 KiB".into());
    }
    if text.trim().is_empty() {
        return Err("No speech recognized".into());
    }
    Ok(text.trim().to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_transcript_and_cleanup() {
        let dir = private_temp().unwrap();
        let path = dir.0.join("transcript.txt");
        std::fs::write(&path, " hello world \n").unwrap();
        assert_eq!(transcript(&path).unwrap(), "hello world");
        let path = dir.0.clone();
        drop(dir);
        assert!(!path.exists());
    }
    #[cfg(unix)]
    #[test]
    fn cancels_external_tool() {
        let mut cmd = Command::new("sleep");
        cmd.arg("10");
        let now = Instant::now();
        assert!(run(cmd, &AtomicBool::new(true), Duration::from_secs(20)).is_err());
        assert!(now.elapsed() < Duration::from_secs(5));
    }
}
