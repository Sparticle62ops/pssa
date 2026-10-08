//! Runtime backend picker. Discovery runs once on demand, away from rendering.
//! The existing backend APIs bind CUDA visible device 0 / WebGPU's preferred
//! adapter. Other adapters are informational, never falsely offered as bindings.
use super::{AMBER, SECOND_ACCENT, accent, panel, panel_area};
use crate::{cli::TrainingBackend, ui};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::sync::mpsc::{self, Receiver};

#[derive(Clone, Debug)]
struct Entry {
    label: String,
    backend: Option<TrainingBackend>,
    available: bool,
    detail: String,
}

fn entry(label: &str, backend: Option<TrainingBackend>, available: bool, detail: &str) -> Entry {
    Entry {
        label: label.into(),
        backend,
        available,
        detail: ui::terminal_text(detail),
    }
}

fn initial_entries() -> Vec<Entry> {
    vec![
        entry(
            "Automatic",
            Some(TrainingBackend::Auto),
            true,
            "Existing GPU/CPU fallback; unchanged default",
        ),
        entry(
            "CPU",
            Some(TrainingBackend::Cpu),
            true,
            "Available / Rayon CPU kernels",
        ),
        entry(
            "WebGPU",
            Some(TrainingBackend::WebGpu),
            false,
            "Not probed yet",
        ),
        entry("CUDA", Some(TrainingBackend::Cuda), false, "Not probed yet"),
        entry(
            "TPU / other accelerators",
            None,
            false,
            "Not supported yet by this binary",
        ),
    ]
}

pub(super) fn hardware_adapter(info: &wgpu::AdapterInfo) -> bool {
    let name = info.name.to_ascii_lowercase();
    info.device_type != wgpu::DeviceType::Cpu
        && !["llvmpipe", "lavapipe", "swiftshader", "software"]
            .iter()
            .any(|s| name.contains(s))
}

fn discover() -> Vec<Entry> {
    let mut entries = initial_entries();
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let selected = adapter.as_ref().map(|a| a.get_info());
    entries[2] = match adapter {
        Some(adapter) if hardware_adapter(&adapter.get_info()) => {
            let info = adapter.get_info();
            // Match backend.rs's device requirements without its stdout banner.
            // Shader/kernel initialization is checked again by the trainer.
            let result = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("PSSA device availability probe"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            ));
            match result {
                Ok(_) => entry(
                    "WebGPU",
                    Some(TrainingBackend::WebGpu),
                    true,
                    &format!(
                        "Available / {} ({:?}); preferred adapter",
                        info.name, info.backend
                    ),
                ),
                Err(e) => entry(
                    "WebGPU",
                    Some(TrainingBackend::WebGpu),
                    false,
                    &format!("Unavailable / device request failed: {e}"),
                ),
            }
        }
        Some(_) => entry(
            "WebGPU",
            Some(TrainingBackend::WebGpu),
            false,
            "Unavailable / software adapter refused",
        ),
        None => entry(
            "WebGPU",
            Some(TrainingBackend::WebGpu),
            false,
            "Unavailable / no compatible adapter",
        ),
    };
    #[cfg(feature = "cuda")]
    {
        entries[3] = match crate::cuda::CudaContext::init() {
            Ok(context) => entry(
                "CUDA",
                Some(TrainingBackend::Cuda),
                true,
                &format!("Available / {}; visible device 0", context.adapter_name()),
            ),
            Err(e) => entry(
                "CUDA",
                Some(TrainingBackend::Cuda),
                false,
                &format!("Unavailable / {e}"),
            ),
        };
    }
    #[cfg(not(feature = "cuda"))]
    {
        entries[3].detail = "Unavailable / binary built without --features cuda".into();
    }
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        if hardware_adapter(&info)
            && selected.as_ref().is_none_or(|chosen| {
                chosen.name != info.name
                    || chosen.backend != info.backend
                    || chosen.device != info.device
            })
        {
            entries.push(entry(
                &ui::terminal_text(&format!("{} ({:?})", info.name, info.backend)),
                None,
                false,
                "Visible WebGPU adapter; per-adapter selection not supported yet",
            ));
        }
    }
    entries
}

pub(super) struct DevicePicker {
    entries: Vec<Entry>,
    selected: usize,
    backend: TrainingBackend,
    receiver: Option<Receiver<Vec<Entry>>>,
    probed: bool,
    message: String,
}

impl Default for DevicePicker {
    fn default() -> Self {
        Self {
            entries: initial_entries(),
            selected: 0,
            backend: TrainingBackend::Auto,
            receiver: None,
            probed: false,
            message: "Up/Down select / Enter apply / r refresh".into(),
        }
    }
}

impl DevicePicker {
    /// Parent calls on first entry to the tab; no driver work occurs per frame.
    pub(super) fn ensure_probe(&mut self) {
        if !self.probed && self.receiver.is_none() {
            self.refresh();
        }
    }

    pub(super) fn refresh(&mut self) {
        if self.receiver.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.message = "Probing runtime devices in the background...".into();
        std::thread::spawn(move || {
            let _ = tx.send(discover());
        });
    }

    pub(super) fn poll(&mut self) {
        let Some(receiver) = &self.receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(entries) => {
                self.entries = entries;
                self.selected = self.selected.min(self.entries.len().saturating_sub(1));
                self.receiver = None;
                self.probed = true;
                self.message = "Up/Down select / Enter apply / r refresh".into();
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.receiver = None;
                self.probed = true;
                self.message = "Device probe failed; CPU/Auto remain available. r retries.".into();
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    pub(super) fn set_backend(&mut self, backend: TrainingBackend) {
        self.backend = backend;
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.backend == Some(backend))
        {
            self.selected = index;
        }
    }

    pub(super) fn backend(&self) -> TrainingBackend {
        self.backend
    }

    pub(super) fn next_backend(&mut self, current: TrainingBackend) -> TrainingBackend {
        self.poll();
        let choices: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.available)
            .filter_map(|e| e.backend)
            .collect();
        let index = choices.iter().position(|&b| b == current).unwrap_or(0);
        choices[(index + 1) % choices.len()]
    }

    pub(super) fn status(&self, backend: TrainingBackend) -> &str {
        self.entries
            .iter()
            .find(|e| e.backend == Some(backend))
            .map(|e| e.detail.as_str())
            .unwrap_or("Unavailable")
    }

    /// Returns a newly applied backend only; unavailable rows never change it.
    pub(super) fn key(&mut self, key: KeyEvent) -> Option<TrainingBackend> {
        self.poll();
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.entries.len().saturating_sub(1))
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.entries.len().saturating_sub(1),
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Enter | KeyCode::Char(' ') => {
                let selected = &self.entries[self.selected];
                if selected.available
                    && let Some(backend) = selected.backend
                {
                    self.backend = backend;
                    self.message = format!(
                        "{} selected for the next run; active jobs are unchanged",
                        backend.as_str()
                    );
                    return Some(backend);
                }
                self.message = selected.detail.clone();
            }
            _ => {}
        }
        None
    }

    pub(super) fn draw(&mut self, f: &mut Frame, area: Rect) {
        self.poll();
        if area.is_empty() {
            return;
        }
        let parts = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(area);
        f.render_widget(
            Paragraph::new(format!("device / selected: {}", self.backend().as_str()))
                .style(accent()),
            parts[0],
        );
        let rows = parts[1].height.saturating_sub(2) as usize;
        let first = self.selected.saturating_sub(rows.saturating_sub(1));
        let lines: Vec<_> = self
            .entries
            .iter()
            .enumerate()
            .skip(first)
            .map(|(index, entry)| {
                Line::styled(
                    format!(
                        "{} {} / {}",
                        if index == self.selected { "▶" } else { " " },
                        entry.label,
                        entry.detail
                    ),
                    if index == self.selected {
                        accent().add_modifier(Modifier::REVERSED)
                    } else if entry.available {
                        accent()
                    } else {
                        Style::new().fg(AMBER)
                    },
                )
            })
            .collect();
        let panel_rect = panel_area(f, parts[1]);
        f.render_widget(
            Paragraph::new(lines).block(panel(" runtime compute / software GPUs skipped ")),
            panel_rect,
        );
        let detail = &self.entries[self.selected].detail;
        f.render_widget(
            Paragraph::new(vec![
                Line::styled(detail.clone(), Style::new().fg(SECOND_ACCENT)),
                Line::from(
                    "Backend family only; CUDA visible device 0 / WebGPU preferred adapter.",
                ),
                Line::styled(self.message.clone(), Style::new().fg(AMBER)),
            ])
            .wrap(Wrap { trim: false }),
            parts[2],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use ratatui::{Terminal, backend::TestBackend};

    fn fixture() -> DevicePicker {
        let mut picker = DevicePicker::default();
        picker.probed = true;
        picker.entries[2] = entry(
            "WebGPU",
            Some(TrainingBackend::WebGpu),
            true,
            "Available / real GPU",
        );
        picker.entries[3].detail = "Unavailable / runtime driver missing".into();
        picker
    }

    #[test]
    fn only_runtime_available_choices_apply_and_preserve_automatic_default() {
        let mut picker = fixture();
        assert_eq!(picker.backend(), TrainingBackend::Auto);
        assert_eq!(
            picker.next_backend(TrainingBackend::Cpu),
            TrainingBackend::WebGpu
        );
        assert_eq!(
            picker.next_backend(TrainingBackend::WebGpu),
            TrainingBackend::Auto
        );
        picker.selected = 3;
        assert_eq!(
            picker.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            None
        );
        assert_eq!(picker.backend(), TrainingBackend::Auto);
        picker.selected = 1;
        assert_eq!(
            picker.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(TrainingBackend::Cpu)
        );
        assert!(picker.message.contains("active jobs are unchanged"));
    }

    #[test]
    fn every_available_backend_is_selectable_and_rendered_without_fallback() {
        let mut picker = fixture();
        picker.entries[3] = entry("CUDA", Some(TrainingBackend::Cuda), true, "Available / fixture CUDA GPU");
        for (index, backend) in [(1, TrainingBackend::Cpu), (2, TrainingBackend::WebGpu), (3, TrainingBackend::Cuda)] {
            picker.selected = index;
            assert_eq!(picker.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)), Some(backend));
            assert_eq!(picker.backend(), backend);
            for (width, height) in [(80, 24), (120, 40), (60, 20)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| picker.draw(f, super::super::feature_area(f.area()))).unwrap();
                let text: String = terminal.backend().buffer().content().iter().map(|c| c.symbol()).collect();
                assert!(text.contains(&format!("selected: {}", backend.as_str())));
            }
        }
        assert_eq!(picker.next_backend(TrainingBackend::WebGpu), TrainingBackend::Cuda);
        assert_eq!(picker.next_backend(TrainingBackend::Cuda), TrainingBackend::Auto);
    }

    #[test]
    fn software_adapters_are_not_offered() {
        let mut info = wgpu::AdapterInfo {
            name: "real hardware".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert!(hardware_adapter(&info));
        for name in [
            "llvmpipe (LLVM)",
            "lavapipe",
            "SwiftShader",
            "software renderer",
        ] {
            info.name = name.into();
            assert!(!hardware_adapter(&info));
        }
        info.name = "unknown".into();
        info.device_type = wgpu::DeviceType::Cpu;
        assert!(!hardware_adapter(&info));
    }

    #[test]
    fn test_backend_device_screen_wide_narrow_and_tiny() {
        for (width, height) in [(120, 30), (79, 24), (60, 18), (30, 10), (1, 1), (0, 0)] {
            let mut picker = fixture();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| picker.draw(f, f.area())).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            if width >= 60 {
                for expected in [
                    "selected: auto",
                    "CPU",
                    "WebGPU",
                    "CUDA",
                    "Not supported yet",
                ] {
                    assert!(text.contains(expected), "missing {expected} at {width}");
                }
            }
            picker.selected = picker.entries.len() - 1;
            terminal.draw(|f| picker.draw(f, f.area())).unwrap();
        }
    }
}
