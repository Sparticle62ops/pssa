//! Throttled read-only host telemetry. OS reads and GPU tools run off the UI.
use super::{
    NORMAL_GREEN, SECOND_ACCENT, accent,
    charts::{bar, sparkline},
    heatmap, panel, panel_area,
    preview::Process,
};
use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    collections::VecDeque,
    process::Command,
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Clone, Default)]
struct Cpu {
    total: u64,
    idle: u64,
}
#[derive(Default)]
struct Snapshot {
    cpu: Vec<Cpu>,
    model: String,
    ram: Option<(u64, u64)>,
    gpu: Vec<String>,
}
#[derive(Default)]
pub(super) struct Hardware {
    pending: Option<mpsc::Receiver<Snapshot>>,
    last_scan: Option<Instant>,
    latest: Snapshot,
    usage: Vec<Option<f64>>,
    history: VecDeque<u64>,
}

fn parse_cpu(text: &str) -> Vec<Cpu> {
    text.lines()
        .filter(|l| {
            l.split_whitespace().next().is_some_and(|label| {
                label == "cpu"
                    || label
                        .strip_prefix("cpu")
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            })
        })
        .take(257)
        .filter_map(|line| {
            // guest / guest_nice are already included in user / nice.
            let values: Vec<u64> = line
                .split_whitespace()
                .skip(1)
                .take(8)
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            if values.len() < 4 {
                return None;
            }
            Some(Cpu {
                total: values.iter().sum(),
                idle: values[3] + values.get(4).copied().unwrap_or(0),
            })
        })
        .collect()
}
fn parse_ram(text: &str) -> Option<(u64, u64)> {
    let value = |key: &str| {
        text.lines()
            .find(|line| line.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?
            .checked_mul(1024)
    };
    let total = value("MemTotal:")?;
    let available = value("MemAvailable:")?;
    (total > 0 && available <= total).then(|| (total - available, total))
}
fn cpu_usage(old: &Cpu, new: &Cpu) -> Option<f64> {
    let total = new.total.checked_sub(old.total)?;
    let idle = new.idle.checked_sub(old.idle)?;
    (total > 0 && idle <= total).then(|| 100.0 * (total - idle) as f64 / total as f64)
}
fn gpu_rows(text: &str) -> Vec<String> {
    text.lines()
        .take(16)
        .filter_map(|line| {
            let cols: Vec<_> = line.split(',').map(str::trim).collect();
            if cols.len() != 7 {
                return None;
            }
            let field = |i: usize| {
                if cols[i].starts_with('[') || cols[i].is_empty() {
                    "n/a".into()
                } else {
                    heatmap::clean(cols[i])
                }
            };
            Some(format!(
                "{} / VRAM {} / {} MiB / util {}% / {}°C / {}W / driver {}",
                field(0),
                field(1),
                field(2),
                field(3),
                field(4),
                field(5),
                field(6)
            ))
        })
        .collect()
}
fn snapshot() -> Snapshot {
    let mut out = Snapshot::default();
    if cfg!(target_os = "linux") {
        out.cpu = std::fs::read_to_string("/proc/stat")
            .map(|s| parse_cpu(&s))
            .unwrap_or_default();
        out.ram = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| parse_ram(&s));
        out.model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|line| line.starts_with("model name") || line.starts_with("Hardware"))
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, value)| heatmap::clean(value.trim()))
            })
            .unwrap_or_else(|| "n/a".into());
    }
    let mut command = Command::new("nvidia-smi");
    command.args(["--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,driver_version", "--format=csv,noheader,nounits"]);
    if let Ok(mut process) = Process::start(command, Duration::from_secs(1)) {
        loop {
            if let Some(result) = process.poll() {
                if let Ok(bytes) = result {
                    out.gpu = gpu_rows(&String::from_utf8_lossy(&bytes));
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    if out.gpu.is_empty() {
        // WebGPU exposes adapter identity, not portable utilization/VRAM counters.
        // Enumerate once, off-thread, without allocating a device or shader cache.
        static ADAPTERS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        out.gpu = ADAPTERS
            .get_or_init(|| {
                wgpu::Instance::default()
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .map(|adapter| adapter.get_info())
                    .filter(super::device::hardware_adapter)
                    .take(16)
                    .map(|info| {
                        format!(
                            "WebGPU {} ({:?}) / VRAM n/a / utilization n/a",
                            heatmap::clean(&info.name),
                            info.backend
                        )
                    })
                    .collect()
            })
            .clone();
    }
    out
}
impl Hardware {
    pub(super) fn poll(&mut self, visible: bool) {
        if let Some(rx) = &self.pending {
            match rx.try_recv() {
                Ok(next) => {
                    self.usage = next
                        .cpu
                        .iter()
                        .enumerate()
                        .map(|(i, cpu)| self.latest.cpu.get(i).and_then(|old| cpu_usage(old, cpu)))
                        .collect();
                    if let Some(Some(usage)) = self.usage.first() {
                        self.history.push_back(*usage as u64);
                        if self.history.len() > 60 {
                            self.history.pop_front();
                        }
                    }
                    self.latest = next;
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if visible
            && self.pending.is_none()
            && self
                .last_scan
                .is_none_or(|t| t.elapsed() >= Duration::from_secs(2))
        {
            self.last_scan = Some(Instant::now());
            let (tx, rx) = mpsc::sync_channel(1);
            self.pending = Some(rx);
            std::thread::spawn(move || {
                let _ = tx.send(snapshot());
            });
        }
    }
    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let area = panel_area(f, area);
        let usage = |v: Option<f64>| {
            v.map_or_else(|| "n/a (need two samples)".into(), |n| format!("{n:.1}%"))
        };
        let mut lines = vec![
            Line::styled("HARDWARE / read-only / 2s refresh", accent()),
            Line::from(format!(
                "CPU {}",
                if self.latest.model.is_empty() {
                    "n/a"
                } else {
                    &self.latest.model
                }
            )),
            Line::from(format!(
                "CPU usage {} / {} logical CPUs",
                usage(self.usage.first().copied().flatten()),
                if self.latest.cpu.is_empty() {
                    "unavailable (OS counters not sampled)".into()
                } else {
                    self.latest.cpu.len().saturating_sub(1).to_string()
                }
            )),
            Line::from(match self.latest.ram {
                Some((used, total)) => format!(
                    "RAM {:.2} / {:.2} GiB used/total",
                    used as f64 / 1073741824.0,
                    total as f64 / 1073741824.0
                ),
                None => "RAM n/a / n/a used/total".into(),
            }),
        ];
        if !self.history.is_empty() {
            let width = area.width.saturating_sub(2) as usize;
            let values: Vec<_> = self
                .history
                .iter()
                .skip(self.history.len().saturating_sub(width.saturating_mul(2)))
                .map(|&usage| usage as f64)
                .collect();
            let low = values.iter().copied().reduce(f64::min).unwrap_or(0.0);
            let high = values.iter().copied().reduce(f64::max).unwrap_or(0.0);
            lines.push(Line::styled(
                format!("CPU history / observed {low:.0}..{high:.0}% / time: oldest → latest"),
                Style::new().fg(SECOND_ACCENT),
            ));
            lines.push(Line::styled(
                sparkline(&values, width),
                Style::new().fg(NORMAL_GREEN),
            ));
        }
        lines.push(Line::styled(
            "CPU meters 0..100% / ticks 0 / 50 / 100 / each logical CPU",
            Style::new().fg(SECOND_ACCENT),
        ));
        // Aggregate always stays visible; per-core meters use only spare rows.
        let gpu_rows = self.latest.gpu.len().max(1) * if area.width < 80 { 3 } else { 2 };
        let room = (area.height as usize)
            .saturating_sub(lines.len() + gpu_rows + 5)
            .min(16);
        for (i, value) in self.usage.iter().enumerate().skip(1).take(room) {
            lines.push(Line::styled(
                format!(
                    "cpu{:>2} [{}] {}",
                    i - 1,
                    bar(value.map_or(f64::NAN, |usage| usage / 100.0), 10),
                    usage(*value)
                ),
                accent(),
            ));
        }
        if self.usage.len().saturating_sub(1) > room {
            lines.push(Line::from(
                "Per-core detail clipped; total includes all CPUs.",
            ));
        }
        lines.push(Line::styled(
            "── GPU / available telemetry ──",
            Style::new().fg(SECOND_ACCENT),
        ));
        if self.latest.gpu.is_empty() {
            lines.push(Line::from(
                "GPU name / VRAM / utilization: n/a (no telemetry available)",
            ));
        } else {
            lines.extend(self.latest.gpu.iter().map(|row| Line::from(row.clone())));
        }
        lines.push(Line::from(
            "WebGPU availability/names: devices tab; usage counters may be n/a.",
        ));
        lines.push(Line::from(
            "Linux /proc host-wide metrics, not trainer-only. Other OS: n/a.",
        ));
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" hardware ")),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn linux_counters_exclude_guest_and_handle_resets_and_missing_ram() {
        let cpu = parse_cpu("cpu 10 0 10 80 0 0 0 0 30 0\ncpu0 10 0 10 80\nintr 99");
        assert_eq!(cpu.len(), 2);
        assert_eq!(cpu[0].total, 100);
        assert_eq!(
            cpu_usage(
                &cpu[0],
                &Cpu {
                    total: 200,
                    idle: 130
                }
            ),
            Some(50.0)
        );
        assert!(cpu_usage(&cpu[0], &cpu[0]).is_none());
        assert!(cpu_usage(&cpu[0], &Cpu::default()).is_none());
        assert_eq!(
            parse_ram("MemTotal: 100 kB\nMemAvailable: 25 kB"),
            Some((75 * 1024, 100 * 1024))
        );
        assert!(parse_ram("MemTotal: 100 kB").is_none());
        assert!(parse_ram("MemTotal: 100 kB\nMemAvailable: 101 kB").is_none());
        assert!(gpu_rows("not supported").is_empty());
        assert!(gpu_rows("Test GPU, 100, 1000, 50, 60, [N/A], 555")[0].contains("n/aW"));
    }

    #[test]
    fn hardware_available_and_unavailable_render_under_80_columns() {
        for populated in [false, true] {
            let mut hardware = Hardware::default();
            if populated {
                hardware.latest = Snapshot {
                    model: "Fixture CPU".into(),
                    ram: Some((1073741824, 3221225472)),
                    gpu: gpu_rows("Fixture GPU, 100, 1000, 50, 60, 75, 555"),
                    cpu: vec![Cpu::default(); 3],
                };
                hardware.usage = vec![Some(50.0); 3];
                hardware.history.extend([0, 25, 50, 75, 100, 50]);
            }
            for (w, h) in [(120, 40), (80, 24), (120, 30), (79, 24), (40, 22), (1, 1)] {
                let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
                t.draw(|f| hardware.draw(f, f.area())).unwrap();
                if w >= 40 {
                    let text: String = t
                        .backend()
                        .buffer()
                        .content()
                        .iter()
                        .map(|c| c.symbol())
                        .collect();
                    assert!(text.contains("CPU"));
                    assert!(text.contains("RAM"));
                    assert!(text.contains(if populated { "Fixture GPU" } else { "n/a" }));
                    assert!(!text.chars().any(|c| ('\u{2580}'..='\u{259f}').contains(&c)));
                    if w >= 80 {
                        assert!(text.contains("CPU meters 0..100%"));
                        if populated {
                            assert!(text.contains("CPU history / observed 0..100%"));
                            assert!(text.contains("oldest → latest"));
                            assert!(text.contains(&sparkline(
                                &[0.0, 25.0, 50.0, 75.0, 100.0, 50.0],
                                usize::from(w.saturating_sub(2)),
                            )));
                            assert!(text.contains("RAM 1.00 / 3.00 GiB used/total"));
                            assert!(text.contains(&format!("cpu 0 [{}] 50.0%", bar(0.5, 10))));
                            assert!(t.backend().buffer().content().iter().any(|cell| {
                                cell.fg == NORMAL_GREEN
                                    && cell
                                        .symbol()
                                        .chars()
                                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                            }));
                        } else {
                            assert!(text.contains("unavailable (OS counters not sampled)"));
                            assert!(!text.contains("0 logical CPUs"));
                            assert!(!text.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)));
                        }
                    }
                }
            }
        }
    }
}
