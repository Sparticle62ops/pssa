//! Training wizard. The trainer writes to a log file, never a UI-owned pipe:
//! slow redraws (or quitting the dashboard) cannot hold up the child process.
use super::device::DevicePicker;
use super::{
    AMBER, BRIGHT_RED, NORMAL_GREEN, RunState, accent, depth_zoom::DepthZoom, panel, panel_area, ring,
};
use crate::{
    cli::{TrainingBackend, resource_limits::ResourceLimits},
    dataset::DatasetManager,
    dream::DreamMode as DreamModeValue,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Paragraph, Wrap},
};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
enum Field {
    Source,
    Dataset,
    HfConfig,
    HfSplit,
    HfField,
    Latent,
    State,
    Vocab,
    Depth,
    Loops,
    Lr,
    Epochs,
    MaxTokens,
    Seed,
    Chunk,
    Accumulate,
    Output,
    Resume,
    Backend,
    Batch,
    Threads,
    Ram,
    DreamEvery,
    DreamReplay,
    DreamMode,
    DreamLen,
    DreamLr,
    DreamSteps,
    LossHigh,
    LossJump,
    LossPatience,
    Memory,
    GrowMemory,
    MemoryTopK,
    WorldSide,
    WorldHidden,
    WorldState,
    WorldCategories,
    WorldTrain,
    WorldHeldout,
    WorldHorizon,
    WorldRollout,
    WorldEpochs,
    WorldLr,
    WorldSeed,
    WorldKl,
    WorldAuxiliary,
    WorldClip,
}
use Field::*;
const LABELS: [&str; 48] = [
    "Source",
    "Dataset",
    "HF config (optional)",
    "HF split",
    "HF text field",
    "Latent",
    "State",
    "Vocab ceiling",
    "Depth",
    "Loops",
    "Learning rate",
    "Epochs",
    "Max tokens (optional)",
    "Seed",
    "Chunk",
    "Accumulate",
    "Output directory",
    "Resume (optional)",
    "Backend",
    "Max batch lanes",
    "Threads (optional)",
    "RAM MiB (optional)",
    "Dream every updates (0 = off)",
    "Dream replay entries",
    "Dream mode",
    "Dream length",
    "Dream learning rate",
    "Dream steps",
    "Loss high factor (>1)",
    "Loss jump factor (>1)",
    "Loss patience updates",
    "Memory slots (optional)",
    "Grow memory on resume (optional)",
    "Memory top-k (CPU, optional)",
    "Grid side",
    "World hidden width",
    "World state width",
    "Latent categories",
    "Train episodes",
    "Held-out episodes",
    "Episode horizon",
    "Open-loop horizon",
    "World epochs",
    "World learning rate",
    "World seed",
    "KL weight",
    "Reward/continue weight",
    "World gradient clip",
];
const WORLD_FLAGS: [(Field, &str); 14] = [
    (WorldSide, "--side"), (WorldHidden, "--hidden"),
    (WorldState, "--state"), (WorldCategories, "--categories"),
    (WorldTrain, "--train-episodes"), (WorldHeldout, "--heldout-episodes"),
    (WorldHorizon, "--horizon"), (WorldRollout, "--rollout-horizon"),
    (WorldEpochs, "--epochs"), (WorldLr, "--lr"), (WorldSeed, "--seed"),
    (WorldKl, "--kl-weight"), (WorldAuxiliary, "--auxiliary-weight"),
    (WorldClip, "--grad-clip"),
];
const WORLD_FIELDS: [&[Field]; 4] = [
    &[Source, WorldSide, WorldTrain, WorldHeldout],
    &[WorldHidden, WorldState, WorldCategories],
    &[WorldEpochs, WorldLr, WorldSeed, WorldHorizon, WorldRollout, WorldKl, WorldAuxiliary, WorldClip],
    &[Output],
];
const PAGES: [&str; 4] = ["1 dataset", "2 model", "3 training", "4 review / launch"];
const FIELDS: [&[Field]; 4] = [
    &[Source, Dataset, HfConfig, HfSplit, HfField],
    &[Latent, State, Vocab, Depth, Loops, Memory, GrowMemory, MemoryTopK],
    &[
        Lr,
        Epochs,
        MaxTokens,
        Seed,
        Chunk,
        Accumulate,
        Batch,
        DreamEvery,
        DreamReplay,
        DreamMode,
        DreamLen,
        DreamLr,
        DreamSteps,
        LossHigh,
        LossJump,
        LossPatience,
    ],
    &[Output, Resume, Backend, Threads, Ram],
];

pub(super) struct Setup {
    values: [String; 48],
    devices: DevicePicker,
    page: usize,
    selected: usize,
    edit: Option<String>,
    message: String,
    error: bool,
    command_scroll: u16,
    command_only: bool,
    depth: DepthZoom,
    started: Instant,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            values: [
                "local",
                "",
                "",
                "train",
                "text",
                "256",
                "16",
                "2048",
                "1",
                "1",
                "0.001",
                "4",
                "",
                "42",
                "64",
                "8",
                "runs/new-run",
                "",
                "auto",
                "",
                "",
                "",
                "0",
                "32",
                "memory",
                "64",
                "0.006",
                "1",
                "4",
                "8",
                "3",
                "",
                "",
                "",
                "4", "16", "2", "16", "32", "12", "12", "8", "3",
                "0.003", "73", "0.1", "0.25", "5",
            ]
            .map(str::to_owned),
            devices: DevicePicker::default(),
            page: 0,
            selected: 0,
            edit: None,
            message: "Choose a local UTF-8 file or a Hugging Face owner/name.".into(),
            error: false,
            command_scroll: 0,
            command_only: false,
            depth: DepthZoom::default(),
            started: Instant::now(),
        }
    }
}

impl Setup {
    pub(super) fn set_dataset(&mut self, path: PathBuf) {
        self.values[Source as usize] = "local".into();
        self.values[Dataset as usize] = path.to_string_lossy().into_owned();
        self.page = 0;
        self.selected = 1;
        self.edit = None;
        self.message = "Library/mixer dataset selected; review before launch.".into();
        self.error = false;
    }

    pub(super) fn set_resume(&mut self, path: PathBuf) {
        if self.is_world_model() { self.values[Source as usize] = "local".into(); }
        self.values[Resume as usize] = path.to_string_lossy().into_owned();
        self.page = 3;
        self.selected = 1;
        self.edit = None;
        for (label, value) in super::local::resume_hints(&path) {
            if let Some(index) = LABELS.iter().position(|s| *s == label) {
                self.values[index] = value;
            }
        }
        self.message = "Resume selected; verify shape, chunk and schedule. TRFM uses CPU baseline; PSSA-only fields, including dream controls, are ignored for TRFM.".into();
        self.error = false;
    }

    fn transformer_resume(&self) -> bool {
        Path::new(self.value(Resume)).extension().is_some_and(|e| e.eq_ignore_ascii_case("trfm"))
    }

    fn checkpoint_name(&self) -> &'static str {
        if self.transformer_resume() { "model.trfm" } else { "model.pssa" }
    }

    fn value(&self, field: Field) -> &str {
        &self.values[field as usize]
    }
    fn is_world_model(&self) -> bool {
        self.value(Source) == "boxes-world"
    }
    fn fields(&self) -> &[Field] {
        if self.is_world_model() {
            WORLD_FIELDS[self.page]
        } else if self.page == 0 && self.value(Source) == "local" {
            &[Source, Dataset]
        } else {
            FIELDS[self.page]
        }
    }
    fn focused(&self) -> Option<Field> {
        self.fields().get(self.selected).copied()
    }
    pub(super) fn editing(&self) -> bool {
        self.edit.is_some()
    }

    fn fail(&mut self, message: impl Into<String>) {
        self.message = message.into();
        self.error = true;
    }

    fn update_depth(&mut self) {
        let value = if self.focused() == Some(Depth) {
            self.edit.as_deref().unwrap_or(self.value(Depth))
        } else {
            self.value(Depth)
        };
        if let Ok(depth @ 1..=32) = value.parse::<usize>() {
            self.depth.set_depth(depth, Instant::now());
        }
    }

    pub(super) fn set_backend(&mut self, backend: TrainingBackend) {
        self.values[Backend as usize] = backend.as_str().into();
        self.devices.set_backend(backend);
    }

    pub(super) fn backend(&self) -> TrainingBackend {
        TrainingBackend::parse(self.value(Backend)).unwrap_or_default()
    }

    pub(super) fn limits(&self) -> Result<ResourceLimits, String> {
        ResourceLimits::from_inputs([
            self.value(Threads),
            self.value(Ram),
            self.value(Batch),
            self.value(MaxTokens),
        ])
    }

    pub(super) fn set_limits(&mut self, limits: ResourceLimits) {
        for (field, value) in [
            (Threads, limits.threads),
            (Ram, limits.ram_mib),
            (Batch, limits.batch_size),
            (MaxTokens, limits.max_tokens),
        ] {
            self.values[field as usize] = value.map(|n| n.to_string()).unwrap_or_default();
        }
    }

    /// Snapshot the wizard's grid defaults and output root without changing it.
    pub(super) fn sweep_base(&self) -> ([String; 3], PathBuf) {
        (
            [
                self.value(Lr).into(),
                self.value(Latent).into(),
                if self.value(Batch).is_empty() {
                    "1".into()
                } else {
                    self.value(Batch).into()
                },
            ],
            PathBuf::from(safe_path(self.value(Output))),
        )
    }

    /// Use exactly the wizard validation/CLI path; queued specs freeze the base.
    pub(super) fn sweep_spec(
        &self,
        lr: &str,
        latent: &str,
        batch: &str,
        output: &Path,
    ) -> Result<RunSpec, String> {
        if self.is_world_model() {
            return Err("Boxes-world has its own bounded comparison; text-training sweeps are unavailable.".into());
        }
        if !self.value(Resume).is_empty() {
            return Err(
                "Clear Resume in setup before sweeping: checkpoint dimensions override the grid."
                    .into(),
            );
        }
        if self.value(Output).trim().is_empty() {
            return Err("Choose a base output directory in setup.".into());
        }
        let mut base = Self {
            values: self.values.clone(),
            ..Self::default()
        };
        for (field, value) in [
            (Lr, lr),
            (Latent, latent),
            (Batch, batch),
            (Output, output.to_str().ok_or("Output path must be UTF-8")?),
        ] {
            base.values[field as usize] = value.into();
        }
        base.validate()
    }

    /// Prepare review only. Never read weights, infer shapes, or launch a child.
    pub(super) fn prepare_resume(&mut self, path: PathBuf) -> Result<(), String> {
        let value = path.to_str().ok_or("Checkpoint path must be UTF-8")?;
        if value.chars().any(char::is_control) || !path.is_file() {
            return Err(
                "Resume checkpoint must be an existing file without control characters".into(),
            );
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        if self.is_world_model() { self.values[Source as usize] = "local".into(); }
        self.values[Resume as usize] = value.into();
        self.values[Output as usize] = path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("resume-{stamp}-{}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        self.page = 3;
        self.selected = 1;
        self.edit = None;
        self.command_only = false;
        self.error = false;
        self.message = "Resume prepared, NOT started. Match checkpoint shape/chunk in setup; review dataset, output and limits before START.".into();
        Ok(())
    }

    fn cycle_choice(&mut self, field: Field) {
        if field == Backend {
            self.devices.ensure_probe();
            let next = self.devices.next_backend(self.backend());
            self.set_backend(next);
            self.message = self.devices.status(next).into();
            self.error = false;
            return;
        }
        let choices: &[&str] = match field {
            Source => &["local", "hf", "boxes-world"],
            _ => return,
        };
        let index = choices
            .iter()
            .position(|v| *v == self.value(field))
            .unwrap_or(0);
        self.values[field as usize] = choices[(index + 1) % choices.len()].into();
        if self.is_world_model() {
            self.message = "Boxes-world: bounded CPU comparison, no corpus/resume/checkpoint. Output saves train.log only.".into();
            self.error = false;
        }
        self.selected = self.selected.min(self.fields().len());
    }

    /// Returns true only on an explicit activation of the review page's start button.
    pub(super) fn key(&mut self, key: KeyEvent) -> bool {
        if let Some(edit) = &mut self.edit {
            match key.code {
                KeyCode::Esc => {
                    self.edit = None;
                }
                KeyCode::Enter => {
                    let field = self.focused().unwrap();
                    self.values[field as usize] = self.edit.take().unwrap();
                    self.message = format!("{} updated", LABELS[field as usize]);
                    self.error = false;
                }
                KeyCode::Backspace => {
                    edit.pop();
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => edit.clear(),
                KeyCode::Char(c)
                    if !c.is_control()
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    if edit.len() + c.len_utf8() <= 4096 {
                        edit.push(c);
                    }
                }
                _ => {}
            }
            self.update_depth();
            return false;
        }
        if key.code == KeyCode::Char('c') {
            self.command_only = !self.command_only;
            return false;
        }
        if self.command_only {
            match key.code {
                KeyCode::PageUp | KeyCode::Up => {
                    self.command_scroll = self.command_scroll.saturating_sub(3);
                }
                KeyCode::PageDown | KeyCode::Down => {
                    self.command_scroll = self.command_scroll.saturating_add(3);
                }
                _ => {}
            }
            return false;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(self.fields().len()),
            KeyCode::Left | KeyCode::BackTab => {
                self.page = self.page.saturating_sub(1);
                self.selected = 0;
            }
            KeyCode::Right | KeyCode::F(5) => {
                self.page = (self.page + 1).min(3);
                self.selected = 0;
            }
            KeyCode::PageUp => self.command_scroll = self.command_scroll.saturating_sub(3),
            KeyCode::PageDown => self.command_scroll = self.command_scroll.saturating_add(3),
            KeyCode::Char('+') | KeyCode::Char('-')
                if matches!(self.focused(), Some(Depth | Loops)) =>
            {
                let field = self.focused().unwrap();
                let current = self.value(field).parse::<usize>().unwrap_or(1);
                let next = if key.code == KeyCode::Char('+') {
                    current.saturating_add(1)
                } else {
                    current.saturating_sub(1)
                };
                self.values[field as usize] = next.clamp(1, 32).to_string();
                self.update_depth();
            }
            KeyCode::Enter | KeyCode::Char(' ') => match self.focused() {
                Some(field @ (Source | Backend)) => self.cycle_choice(field),
                Some(field) => {
                    self.edit = Some(self.value(field).to_owned());
                    match field {
                        Threads => self.message = "Threads may change last-digit reduction rounding; blank keeps defaults.".into(),
                        Ram => self.message = "Linux prlimit address-space cap, NOT RSS/VRAM; too low can abort the child.".into(),
                        Batch => self.message = "Independent lanes (--batch-size), NOT a VRAM quota; blank = 1.".into(),
                        _ => {}
                    }
                }
                None if self.page < 3 => {
                    self.page += 1;
                    self.selected = 0;
                }
                None => return true,
            },
            _ => {}
        }
        if self.page == 3 && !self.is_world_model() {
            self.devices.ensure_probe();
        }
        false
    }

    fn number(&self, field: Field, min: usize, max: usize) -> Result<usize, String> {
        self.value(field)
            .parse::<usize>()
            .ok()
            .filter(|n| (min..=max).contains(n))
            .ok_or_else(|| {
                format!(
                    "{} must be an integer in {min}..={max}",
                    LABELS[field as usize]
                )
            })
    }

    fn validate(&self) -> Result<RunSpec, String> {
        for value in &self.values {
            if value.chars().any(char::is_control) {
                return Err("Inputs must not contain control characters".into());
            }
        }
        if self.is_world_model() {
            let args = self.args();
            crate::cli::world_model::parse(&args[1..])?;
            if self.value(Output).trim().is_empty() {
                return Err("Choose an output directory for train.log".into());
            }
            let output = PathBuf::from(safe_path(self.value(Output)));
            if output.exists() && !output.is_dir() {
                return Err("Output must be a directory, not a file".into());
            }
            if output.join("train.log").symlink_metadata().is_ok() {
                return Err("train.log already exists; choose a new output directory".into());
            }
            return Ok(RunSpec { args, output, loops: 1 });
        }
        match self.value(Source) {
            "local" => {
                let path = Path::new(self.value(Dataset));
                if self.value(Dataset).contains(',')
                    || self.value(Dataset).trim_end() != self.value(Dataset)
                {
                    return Err("Local filenames with commas or trailing whitespace are not supported by the CLI source-list syntax".into());
                }
                if !path.is_file() {
                    return Err("Dataset must be an existing local UTF-8 file".into());
                }
                let file = File::open(path).map_err(|e| format!("Cannot read dataset: {e}"))?;
                if file.metadata().map_err(|e| e.to_string())?.len() == 0 {
                    return Err("Dataset is empty".into());
                }
            }
            "hf" => {
                DatasetManager::validate_huggingface_dataset_name(self.value(Dataset))?;
                if self.value(HfSplit).trim().is_empty() || self.value(HfField).trim().is_empty() {
                    return Err("HF split and text field must not be empty".into());
                }
                if !self.value(HfConfig).is_empty() && self.value(HfConfig).trim().is_empty() {
                    return Err("HF config must be blank or contain a config name".into());
                }
                for field in [HfConfig, HfSplit, HfField] {
                    if self.value(field).starts_with('-') {
                        return Err(format!(
                            "{} must not begin with '-'",
                            LABELS[field as usize]
                        ));
                    }
                }
            }
            _ => return Err("Source must be local or hf".into()),
        }
        for (field, min, max) in [
            (Latent, 1, 4096),
            (State, 1, 4096),
            (Vocab, 257, 65_536),
            (Depth, 1, 32),
            (Loops, 1, 32),
            (Epochs, 1, 1_000_000),
            (Chunk, 1, 65_536),
            (Accumulate, 1, 1_000_000),
        ] {
            self.number(field, min, max)?;
        }
        if !self.transformer_resume() {
            for field in [Memory, GrowMemory, MemoryTopK] {
                if !self.value(field).is_empty() {
                    self.number(field, 1, 1_000_000)?;
                }
            }
            if !self.value(GrowMemory).is_empty() && self.value(Resume).is_empty() {
                return Err("Grow memory requires a resume checkpoint".into());
            }
            if !self.value(MemoryTopK).is_empty() && self.value(Backend) != "cpu" {
                return Err("Memory top-k requires Backend cpu".into());
            }
        }
        if !self.value(MaxTokens).is_empty() {
            self.number(MaxTokens, 1, usize::MAX)?;
        }
        self.value(Seed)
            .parse::<u64>()
            .map_err(|_| "Seed must be an unsigned 64-bit integer")?;
        for (field, min, max) in [
            (DreamEvery, 0, 1_000_000),
            (DreamReplay, 1, 1_000_000),
            (DreamLen, 1, 100_000),
            (DreamSteps, 1, 1_000_000),
        ] {
            self.number(field, min, max)?;
        }
        crate::loss_guard::LossGuardConfig {
            high_factor: self
                .value(LossHigh)
                .parse::<f32>()
                .map_err(|_| "Invalid loss high factor")? as f64,
            jump_factor: self
                .value(LossJump)
                .parse::<f32>()
                .map_err(|_| "Invalid loss jump factor")? as f64,
            patience: self.number(LossPatience, 1, 1_000_000)?,
        }
        .validate()?;
        DreamModeValue::parse(self.value(DreamMode))?;
        let dream_lr = self.value(DreamLr).parse::<f32>().map_err(|_| {
            "Dream learning rate must be a finite positive number"
        })?;
        if !dream_lr.is_finite() || dream_lr <= 0.0 {
            return Err("Dream learning rate must be a finite positive number".into());
        }
        let lr = self
            .value(Lr)
            .parse::<f32>()
            .map_err(|_| "Learning rate must be a finite positive number")?;
        if !lr.is_finite() || lr <= 0.0 {
            return Err("Learning rate must be a finite positive number".into());
        }
        TrainingBackend::parse(self.value(Backend))?;
        self.limits()?;
        if self.transformer_resume() {
            if self.value(Source) != "local" {
                return Err("Transformer resume requires a local dataset".into());
            }
            if self.limits()?.batch_size.is_some_and(|n| n != 1) {
                return Err("Transformer resume does not support batch lanes; Max batch lanes must be blank or 1".into());
            }
        }
        if !self.value(Resume).is_empty() {
            let path = Path::new(self.value(Resume));
            if !path.is_file() {
                return Err("Resume checkpoint must be an existing file".into());
            }
            File::open(path).map_err(|e| format!("Cannot read resume checkpoint: {e}"))?;
        }
        if self.value(Output).trim().is_empty() {
            return Err("Choose an output directory".into());
        }
        let output = safe_path(self.value(Output));
        let path = Path::new(&output);
        if path.exists() && !path.is_dir() {
            return Err("Output must be a directory, not a file".into());
        }
        for name in [self.checkpoint_name(), "train.log"] {
            if path.join(name).symlink_metadata().is_ok() {
                return Err(format!(
                    "{} already exists; choose a new output directory",
                    path.join(name).display()
                ));
            }
        }
        Ok(RunSpec {
            args: self.args(),
            output: PathBuf::from(output),
            loops: self.number(Loops, 1, 32)?,
        })
    }

    fn args(&self) -> Vec<String> {
        if self.is_world_model() {
            let mut args = vec!["world-model".into()];
            for (field, flag) in WORLD_FLAGS {
                args.extend([flag.into(), self.value(field).into()]);
            }
            return args;
        }
        let transformer = self.transformer_resume();
        let mut args = vec![if transformer { "train-transformer" } else { "train" }.into()];
        let mut push = |flag: &str, value: String| {
            args.extend([flag.to_owned(), value]);
        };
        if self.value(Source) == "hf" {
            push("--hf-dataset", self.value(Dataset).into());
            if !self.value(HfConfig).is_empty() {
                push("--hf-config", self.value(HfConfig).into());
            }
            push("--hf-split", self.value(HfSplit).into());
            push("--hf-field", self.value(HfField).into());
        } else {
            // Force local interpretation even for a file literally named 'science'.
            push("--data", format!("file:{}", safe_path(self.value(Dataset))));
        }
        for (field, flag) in [
            (Latent, "--latent"),
            (State, "--state"),
            (Vocab, "--vocab-size"),
            (Depth, "--depth"),
            (Loops, "--loops"),
            (Lr, "--lr"),
            (Epochs, "--epochs"),
            (Seed, "--seed"),
            (Chunk, "--chunk"),
            (Accumulate, "--accumulate"),
            (Backend, "--backend"),
        ] {
            if transformer && matches!(field, Latent | State | Depth | Loops | Backend) { continue; }
            push(flag, self.value(field).into());
        }
        for (field, flag) in [
            (MaxTokens, "--max-tokens"),
            (Batch, "--batch-size"),
            (Threads, "--threads"),
            (Ram, "--ram-mib"),
        ] {
            // train-transformer is a single-lane CPU baseline and does not
            // accept --batch-size, even when the shared draft explicitly says 1.
            if !(transformer && field == Batch) && !self.value(field).is_empty() {
                push(flag, self.value(field).into());
            }
        }
        if !transformer {
            for (field, flag) in [
                (Memory, "--memory"),
                (GrowMemory, "--grow-memory"),
                (MemoryTopK, "--memory-top-k"),
            ] {
                if !self.value(field).is_empty() {
                    push(flag, self.value(field).into());
                }
            }
            for (field, flag) in [
                (LossHigh, "--loss-guard-high-factor"),
                (LossJump, "--loss-guard-jump-factor"),
                (LossPatience, "--loss-guard-patience"),
            ] {
                push(flag, self.value(field).into());
            }
            for (field, flag, differs) in [
                (
                    DreamEvery,
                    "--dream-every",
                    self.value(DreamEvery)
                        .parse::<usize>()
                        .map_or(true, |value| value != 0),
                ),
                (
                    DreamReplay,
                    "--dream-replay",
                    self.value(DreamReplay)
                        .parse::<usize>()
                        .map_or(true, |value| value != 32),
                ),
                (
                    DreamMode,
                    "--dream-mode",
                    DreamModeValue::parse(self.value(DreamMode))
                        .map_or(true, |value| value != DreamModeValue::Memory),
                ),
                (
                    DreamLen,
                    "--dream-len",
                    self.value(DreamLen)
                        .parse::<usize>()
                        .map_or(true, |value| value != 64),
                ),
                (
                    DreamLr,
                    "--dream-lr",
                    self.value(DreamLr)
                        .parse::<f32>()
                        .map_or(true, |value| value != crate::dream::DEFAULT_REHEARSAL_LR),
                ),
                (
                    DreamSteps,
                    "--dream-steps",
                    self.value(DreamSteps)
                        .parse::<usize>()
                        .map_or(true, |value| value != 1),
                ),
            ] {
                if differs {
                    push(flag, self.value(field).into());
                }
            }
        }
        if !self.value(Resume).is_empty() {
            push("--resume", safe_path(self.value(Resume)));
        }
        push(
            "--out",
            Path::new(&safe_path(self.value(Output)))
                .join(self.checkpoint_name())
                .to_string_lossy()
                .into_owned(),
        );
        args.push("--no-tui".into());
        args
    }

    fn command(&self) -> String {
        let executable = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("pssa"));
        let output = safe_path(self.value(Output));
        if self.is_world_model() {
            return format!("mkdir -p -- {} && (set -C; {} {} > {} 2>&1 < /dev/null)",
                quote(&output), quote(&executable.to_string_lossy()),
                self.args().iter().map(|v| quote(v)).collect::<Vec<_>>().join(" "),
                quote(&Path::new(&output).join("train.log").to_string_lossy()));
        }
        let checkpoint = Path::new(&output).join(self.checkpoint_name());
        format!(
            "mkdir -p -- {} && test ! -e {} && test ! -L {} && (set -C; {} {} > {} 2>&1 < /dev/null)",
            quote(&output),
            quote(&checkpoint.to_string_lossy()),
            quote(&checkpoint.to_string_lossy()),
            quote(&executable.to_string_lossy()),
            self.args()
                .iter()
                .map(|v| quote(v))
                .collect::<Vec<_>>()
                .join(" "),
            quote(&Path::new(&output).join("train.log").to_string_lossy())
        )
    }

    pub(super) fn launch(&mut self, busy: bool) -> Option<TrainingRun> {
        if busy {
            self.fail("A training run is active. Wait for it to finish before starting another.");
            return None;
        }
        let result = self.validate().and_then(|spec| {
            let executable = std::env::current_exe().map_err(|e| e.to_string())?;
            spec.start(&executable)
        });
        match result {
            Ok(run) => {
                self.message = "Training started; logs are saved in train.log.".into();
                self.error = false;
                Some(run)
            }
            Err(error) => {
                self.fail(error);
                None
            }
        }
    }

    pub(super) fn draw(&mut self, f: &mut Frame, area: Rect) {
        self.devices.poll();
        if area.is_empty() {
            return;
        }
        if self.command_only {
            let parts = Layout::default()
                .constraints([Constraint::Min(0), Constraint::Length(2)])
                .split(area);
            self.draw_command(f, parts[0]);
            f.render_widget(
                Paragraph::new("c back / PgUp PgDn scroll\nTab tabs / ? help / q quit")
                    .style(accent()),
                parts[1],
            );
            return;
        }
        let compact = area.width < 60 || area.height < 14;
        let command_height = if compact {
            0
        } else if self.page == 3 {
            (area.height / 3).max(5)
        } else {
            4
        };
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(command_height),
                Constraint::Length(if compact { 2 } else { 3 }),
            ])
            .split(area);
        f.render_widget(
            Paragraph::new(format!("setup / {}", PAGES[self.page])).style(accent()),
            parts[0],
        );
        let show_animation = !self.is_world_model() && self.page == 1 && f.area().width >= 80 && parts[1].height >= 7;
        let body = if show_animation {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
                .split(parts[1])
        } else {
            Layout::default()
                .constraints([Constraint::Percentage(100)])
                .split(parts[1])
        };
        let fields = self.fields();
        let mut rows = Vec::new();
        let available = body[0].height.saturating_sub(2) as usize;
        let first = if compact || fields.len() + 1 > available {
            self.selected.saturating_sub(available.saturating_sub(2))
        } else {
            0
        };
        for (index, field) in fields.iter().enumerate().skip(first) {
            let focused = self.selected == index;
            let value = if focused {
                self.edit.as_deref().unwrap_or(self.value(*field))
            } else {
                self.value(*field)
            };
            // Reserve the actual border/label/cursor width before clipping the
            // value. Count terminal cells, not characters (paths may be CJK).
            let inner_width = body[0].width.saturating_sub(2) as usize;
            let label = LABELS[*field as usize];
            let label = if compact {
                format!("{label}: ")
            } else {
                format!("{label:<21} ")
            };
            let label: String = label.chars().take(inner_width.saturating_sub(6)).collect();
            let prefix = format!("{} {label}", if focused { "▶" } else { " " });
            let cursor = if focused && self.edit.is_some() {
                "▏"
            } else {
                ""
            };
            let room = inner_width.saturating_sub(prefix.len() + cursor.chars().count());
            let text = format!("{prefix}{}{cursor}", visible_tail(value, room));
            rows.push(Line::styled(
                text,
                if focused {
                    accent().add_modifier(Modifier::REVERSED)
                } else {
                    accent()
                },
            ));
        }
        rows.push(Line::styled(
            if self.page == 3 && self.is_world_model() {
                "[ START WORLD MODEL ]"
            } else if self.page == 3 {
                "[ START TRAINING ]"
            } else {
                "[ NEXT ▶ ]"
            },
            if self.selected == fields.len() {
                accent().add_modifier(Modifier::REVERSED)
            } else {
                accent()
            },
        ));
        if !compact {
            rows.push(Line::from(""));
            rows.push(Line::styled(
                if self.is_world_model() {
                    "CPU boxes-world / no checkpoint / same episodes, not parameter-matched."
                } else { match self.page {
                    0 => "HF is cached by the CLI. Local files must be UTF-8.",
                    1 => "Depth = stacked nets. Loops = repeated passes. +/- changes either.",
                    2 => "Blank tokens = no cap; blank batch = 1. Batch is not a VRAM cap.",
                    _ => "Blank limits = unchanged. RAM = Linux address space, NOT RSS/VRAM.",
                } },
                Style::new().fg(AMBER),
            ));
            if self.page == 3 && !self.is_world_model() {
                rows.push(Line::from(
                    "Resume: shape/chunk must match; the child validates the checkpoint.",
                ));
                rows.push(Line::from(self.devices.status(self.backend()).to_owned()));
                rows.push(Line::from(
                    "Threads can change reduction rounding; tiny RAM budgets may abort the child.",
                ));
                rows.push(Line::from("Quitting the TUI leaves training running; reopen with tail -f train.log | pssa tui."));
            }
        }
        let parameters_area = panel_area(f, body[0]);
        f.render_widget(
            Paragraph::new(rows).block(panel(" parameters ")),
            parameters_area,
        );
        if show_animation {
            let previews = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(body[1]);
            self.depth.draw(f, previews[0], Instant::now());
            let value = if self.focused() == Some(Loops) {
                self.edit.as_deref().unwrap_or(self.value(Loops))
            } else {
                self.value(Loops)
            };
            ring::draw(
                f,
                previews[1],
                value.parse::<usize>().unwrap_or(1).clamp(1, 32),
                self.started.elapsed(),
            );
        }
        if command_height > 0 {
            self.draw_command(f, parts[2]);
        }
        let mut footer = vec![Line::styled(
            &self.message,
            Style::new().fg(if self.error { BRIGHT_RED } else { AMBER }),
        )];
        footer.push(Line::from(if self.edit.is_some() {
            "Type / Enter save / Esc cancel / Tab tabs / F1 help"
        } else {
            "Enter edit / arrows move / c CLI / Tab tabs / ? help"
        }));
        f.render_widget(Paragraph::new(footer), parts[3]);
    }

    fn draw_command(&mut self, f: &mut Frame, area: Rect) {
        let paragraph = Paragraph::new(self.command())
            .style(Style::new().fg(NORMAL_GREEN))
            .wrap(Wrap { trim: false });
        let max_scroll = paragraph
            .line_count(area.width.saturating_sub(2))
            .saturating_sub(area.height.saturating_sub(2) as usize)
            .min(u16::MAX as usize) as u16;
        self.command_scroll = self.command_scroll.min(max_scroll);
        let command_area = panel_area(f, area);
        f.render_widget(
            paragraph
                .scroll((self.command_scroll, 0))
                .block(panel(" equivalent CLI / PgUp PgDn ")),
            command_area,
        );
    }
}

pub(super) fn visible_tail(value: &str, cells: usize) -> &str {
    let mut start = value.len();
    let mut used = 0;
    let mut bytes = [0; 4];
    for (index, c) in value.char_indices().rev() {
        let width = Line::raw(&*c.encode_utf8(&mut bytes)).width();
        if used + width > cells {
            break;
        }
        used += width;
        start = index;
    }
    &value[start..]
}

fn safe_path(value: &str) -> String {
    if value.starts_with('-') {
        format!("./{value}")
    } else {
        value.to_owned()
    }
}

fn quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_+-./:=@".contains(&c))
    {
        value.into()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

pub(super) struct RunSpec {
    args: Vec<String>,
    output: PathBuf,
    loops: usize,
}

impl RunSpec {
    pub(super) fn launch(self, busy: bool) -> Result<TrainingRun, String> {
        if busy {
            return Err("A local or remote training run is active".into());
        }
        for name in ["model.pssa", "train.log"] {
            if self.output.join(name).symlink_metadata().is_ok() {
                return Err(format!(
                    "{} already exists; refusing to overwrite",
                    self.output.join(name).display()
                ));
            }
        }
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        self.start(&executable)
    }

    pub(super) fn start(self, executable: &Path) -> Result<TrainingRun, String> {
        fs::create_dir_all(&self.output)
            .map_err(|e| format!("Cannot create output directory: {e}"))?;
        let path = self.output.join("train.log");
        let log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("Cannot create {}: {e}", path.display()))?;
        let spawn = (|| {
            let reader = File::open(&path)?;
            let mut command = Command::new(executable);
            command
                .args(&self.args)
                .stdin(Stdio::null())
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            super::hf::configure_child(&mut command);
            // Keep terminal job-control signals from reaching a detached run.
            // No pre_exec hook or extra dependency is needed for this on Unix.
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                command.process_group(0);
            }
            command.spawn().map(|child| (child, reader))
        })();
        match spawn {
            Ok((child, log)) => Ok(TrainingRun {
                checkpoint_name: match self.args.first().map(String::as_str) {
                    Some("world-model") => None,
                    Some("train-transformer") => Some("model.trfm"),
                    _ => Some("model.pssa"),
                },
                child,
                log,
                pending: Vec::new(),
                output: self.output,
                loops: self.loops,
                status: None,
                reported: false,
            }),
            Err(e) => {
                let _ = fs::remove_file(path);
                Err(format!("Cannot start trainer: {e}"))
            }
        }
    }
}

pub(super) struct TrainingRun {
    checkpoint_name: Option<&'static str>,
    child: Child,
    log: File,
    pending: Vec<u8>,
    output: PathBuf,
    loops: usize,
    status: Option<ExitStatus>,
    reported: bool,
}

impl TrainingRun {
    pub(super) fn output_dir(&self) -> &Path {
        &self.output
    }

    /// None until exit AND durable-log draining have both completed.
    pub(super) fn succeeded(&self) -> Option<bool> {
        self.status
            .filter(|_| self.reported)
            .map(|status| status.success())
    }

    pub(super) fn initialize(&self, state: &mut RunState) {
        let comparison_series = std::mem::take(&mut state.comparison_series);
        let comparison_label = state.comparison_label.take();
        let comparison_error = state.comparison_error.take();
        *state = RunState {
            chain_dir: self.output.clone(),
            training_active: true,
            loop_count: self.loops,
            last_progress_at: Some(Instant::now()),
            checkpoint_target: self.checkpoint_name.map(|name| {
                self.output.join(name).to_string_lossy().into_owned()
            }),
            comparison_series,
            comparison_label,
            comparison_error,
            ..RunState::default()
        };
        state.ingest(&format!(
            "Trainer pid={} / log={} / quitting the TUI leaves this run active",
            self.child.id(),
            self.output.join("train.log").display()
        ));
        state.warning = Some(format!(
            "Child pid={} / q detaches; log: {}",
            self.child.id(),
            self.output.join("train.log").display()
        ));
    }

    pub(super) fn active(&self) -> bool {
        !self.reported
    }

    /// Bounded tail work on each UI tick. Log bytes are durable even if the UI
    /// cannot keep up, and no trainer thread waits for the dashboard to drain.
    #[cfg(test)]
    pub(super) fn poll(&mut self, state: &mut RunState) {
        self.poll_with(state, |_| {});
    }

    pub(super) fn poll_with(&mut self, state: &mut RunState, mut ingest: impl FnMut(&str)) {
        if self.reported {
            return;
        }
        if self.status.is_none() {
            match self.child.try_wait() {
                Ok(status) => self.status = status,
                Err(error) => {
                    state.problem = Some(format!("Cannot inspect trainer: {error}"));
                }
            }
        }
        let mut buffer = [0u8; 16_384];
        let mut at_end = false;
        for _ in 0..4 {
            match self.log.read(&mut buffer) {
                Ok(0) => {
                    at_end = true;
                    break;
                }
                Ok(n) => {
                    for &byte in &buffer[..n] {
                        if byte == b'\n' {
                            let line = String::from_utf8_lossy(&self.pending);
                            state.ingest(&line);
                            ingest(&line);
                            self.pending.clear();
                        } else if self.pending.len() < 65_536 {
                            self.pending.push(byte);
                        }
                    }
                }
                Err(error) => {
                    state.problem = Some(format!("Cannot tail training log: {error}"));
                    at_end = true;
                    break;
                }
            }
        }
        if let Some(status) = self.status.filter(|_| at_end) {
            if !self.pending.is_empty() {
                let line = String::from_utf8_lossy(&self.pending);
                state.ingest(&line);
                ingest(&line);
                self.pending.clear();
            }
            state.training_active = false;
            state.warning = None;
            if !status.success() {
                let detail = state
                    .raw_lines
                    .iter()
                    .rev()
                    .find(|line| !line.trim().is_empty())
                    .cloned()
                    .unwrap_or_default();
                state.problem = Some(format!("Trainer exited {status}: {detail}"));
            }
            state.ingest(&format!(
                "Trainer exited {status}; log={}",
                self.output.join("train.log").display()
            ));
            state.refresh_chain();
            self.reported = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::Duration,
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "pssa-setup-{}-{} space's",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            fs::write(
                path.join("source file.txt"),
                "small local dataset with words\n",
            )
            .unwrap();
            Self(path)
        }
        fn setup(&self) -> Setup {
            let mut setup = Setup::default();
            setup.values[Dataset as usize] = self
                .0
                .join("source file.txt")
                .to_string_lossy()
                .into_owned();
            setup.values[Output as usize] =
                self.0.join("run output").to_string_lossy().into_owned();
            setup
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn press(setup: &mut Setup, key: KeyCode) -> bool {
        setup.key(KeyEvent::new(key, KeyModifiers::NONE))
    }
    fn text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn world_model_wizard_uses_real_parser_and_separate_defaults() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        let text_args = setup.args();
        assert!(!press(&mut setup, KeyCode::Enter)); // local -> hf
        assert!(!press(&mut setup, KeyCode::Enter)); // hf -> boxes-world
        assert!(setup.is_world_model());
        let spec = setup.validate().unwrap();
        let parsed = crate::cli::world_model::parse(&spec.args[1..]).unwrap();
        assert_eq!(format!("{parsed:?}"), format!("{:?}", crate::world_model::WorldModelConfig::default()));
        assert_eq!(spec.args[0], "world-model");
        assert_eq!(spec.loops, 1);
        for flag in ["--data", "--resume", "--out", "--memory", "--dream-every", "--backend", "--no-tui"] {
            assert!(!spec.args.iter().any(|s| s == flag));
        }
        assert!(!setup.command().contains("model.pssa"));
        assert!(setup.command().contains("train.log"));
        assert!(!spec.output.exists());
        assert!(setup.sweep_spec(".001", "16", "1", &spec.output).is_err());
        for (field, value) in [(WorldSide, "5"), (WorldHidden, "8"), (WorldState, "3"),
            (WorldCategories, "7"), (WorldTrain, "5"), (WorldHeldout, "3"),
            (WorldHorizon, "5"), (WorldRollout, "2"), (WorldEpochs, "2"),
            (WorldLr, "0.006"), (WorldSeed, "149"), (WorldKl, "0.2"),
            (WorldAuxiliary, "0.5"), (WorldClip, "3")] {
            setup.values[field as usize] = value.into();
        }
        let changed = setup.validate().unwrap();
        let parsed = crate::cli::world_model::parse(&changed.args[1..]).unwrap();
        let expected = crate::world_model::WorldModelConfig {
            side: 5, hidden: 8, state: 3, categories: 7, train_episodes: 5,
            heldout_episodes: 3, horizon: 5, rollout_horizon: 2, epochs: 2,
            learning_rate: 0.006, seed: 149, kl_weight: 0.2, auxiliary_weight: 0.5,
            gradient_clip: 3.0,
        };
        assert_eq!(format!("{parsed:?}"), format!("{expected:?}"));
        assert!(!press(&mut setup, KeyCode::Enter)); // returns to unchanged local draft
        assert_eq!(setup.args(), text_args);
    }

    #[test]
    fn world_model_wizard_validates_bounds_and_protects_logs() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        setup.values[Source as usize] = "boxes-world".into();
        // Irrelevant text-training fields are not used by the world-model mode.
        setup.values[Resume as usize] = "missing.pssa".into();
        setup.values[Dataset as usize] = "missing.txt".into();
        setup.values[Latent as usize] = "bad".into();
        assert!(setup.validate().is_ok());
        for (field, bad) in [(WorldSide, "2"), (WorldHidden, "49"), (WorldState, "0"),
            (WorldCategories, "33"), (WorldTrain, "129"), (WorldHeldout, "0"),
            (WorldHorizon, "25"), (WorldRollout, "13"), (WorldEpochs, "0"),
            (WorldLr, "NaN"), (WorldSeed, "-1"), (WorldKl, "0"),
            (WorldAuxiliary, "-1"), (WorldClip, "inf")] {
            let old = std::mem::replace(&mut setup.values[field as usize], bad.into());
            assert!(setup.validate().is_err(), "{}", LABELS[field as usize]);
            setup.values[field as usize] = old;
        }
        let output = PathBuf::from(setup.value(Output));
        assert!(!output.exists());
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("train.log"), "keep").unwrap();
        assert!(setup.validate().is_err());
        assert_eq!(fs::read_to_string(output.join("train.log")).unwrap(), "keep");
    }

    #[test]
    fn world_model_review_and_command_render_without_text_training_controls() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        setup.values[Source as usize] = "boxes-world".into();
        for (width, height) in [(80, 24), (120, 32), (44, 18)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            for page in 0..4 {
                setup.page = page;
                setup.selected = 0;
                terminal.draw(|f| setup.draw(f, f.area())).unwrap();
                let rendered = text(&terminal);
                assert!(!rendered.contains("Grow memory") && !rendered.contains("Dream every"));
                if page == 3 { assert!(rendered.contains("START WORLD MODEL")); }
            }
        }
        setup.page = 3;
        setup.selected = setup.fields().len();
        assert!(press(&mut setup, KeyCode::Enter));
        assert!(setup.launch(true).is_none());
        assert!(!Path::new(setup.value(Output)).exists());
    }

    #[test]
    fn library_picks_fill_wizard_and_transformer_resume_uses_existing_cli() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        let dataset = fixture.0.join("source file.txt");
        setup.set_dataset(dataset.clone());
        assert_eq!(setup.value(Dataset), dataset.to_str().unwrap());
        assert_eq!(setup.value(Source), "local");
        let resume = fixture.0.join("checkpoint name.trfm");
        let mut header = b"TRFM".to_vec();
        header.extend(1u16.to_le_bytes());
        header.resize(22, 0);
        for n in [2048u64, 32, 4, 64, 8] {
            header.extend(n.to_le_bytes());
        }
        fs::write(&resume, header).unwrap();
        setup.set_resume(resume.clone());
        assert_eq!(setup.value(Resume), resume.to_str().unwrap());
        assert_eq!(setup.value(Chunk), "8");
        let args = setup.validate().unwrap().args;
        assert_eq!(args[0], "train-transformer");
        assert!(args.windows(2).any(|w| w == ["--resume", resume.to_str().unwrap()]));
        assert!(args.windows(2).any(|w| w[0] == "--out" && w[1].ends_with("model.trfm")));
        for flag in ["--latent", "--state", "--depth", "--loops", "--backend"] {
            assert!(!args.iter().any(|s| s == flag));
        }
        assert!(setup.command().contains("model.trfm"));
    }

    #[test]
    fn transformer_resume_rejects_multi_lane_batches_and_omits_cpu_only_flag() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        let resume = fixture.0.join("model.trfm");
        fs::write(&resume, b"child validates checkpoint").unwrap();
        setup.set_resume(resume);
        for batch in ["", "1"] {
            setup.values[Batch as usize] = batch.into();
            let args = setup.validate().unwrap().args;
            assert!(!args.iter().any(|arg| arg == "--batch-size"));
        }
        setup.values[Batch as usize] = "2".into();
        let error = setup.validate().err().expect("unsupported batch must fail preflight");
        assert!(error.contains("Transformer resume"));
        assert!(error.contains("batch"));
        assert!(error.contains("blank or 1"));
    }

    #[test]
    fn sweep_specs_use_wizard_validation_and_preserve_argument_boundaries() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        let original = setup.args();
        let output = fixture.0.join("sweep output/trial-001");
        let spec = setup.sweep_spec("0.002", "64", "2", &output).unwrap();
        for (flag, value) in [("--lr", "0.002"), ("--latent", "64"), ("--batch-size", "2")] {
            let index = spec.args.iter().position(|arg| arg == flag).unwrap();
            assert_eq!(spec.args[index + 1], value);
        }
        let index = spec.args.iter().position(|arg| arg == "--out").unwrap();
        assert_eq!(Path::new(&spec.args[index + 1]), output.join("model.pssa"));
        assert_eq!(setup.args(), original, "the base wizard is not mutated");
        assert!(spec.launch(true).is_err());
        assert!(!output.exists(), "busy launch does not create output or spawn");
        let spec = setup.sweep_spec("0.002", "64", "2", &output).unwrap();
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("train.log"), "keep this log").unwrap();
        assert!(spec.launch(false).is_err(), "recheck reservations at launch time");
        assert_eq!(fs::read_to_string(output.join("train.log")).unwrap(), "keep this log");
        setup.values[Resume as usize] = "checkpoint.pssa".into();
        assert!(setup.sweep_spec("0.002", "64", "2", &output).is_err());
    }

    #[test]
    fn timeline_resume_prepares_review_without_loading_or_starting_checkpoint() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        let path = fixture.0.join("checkpoint with spaces.pssa");
        fs::write(&path, "not checkpoint bytes; never load in timeline/setup").unwrap();
        setup.prepare_resume(path.clone()).unwrap();
        assert_eq!(setup.value(Resume), path.to_str().unwrap());
        assert_eq!(setup.page, 3);
        assert_eq!(setup.selected, 1);
        assert!(!Path::new(setup.value(Output)).exists());
        assert!(setup.message.contains("NOT started"));
        assert!(setup.validate().is_ok());
        assert_eq!(fs::read_to_string(&path).unwrap(), "not checkpoint bytes; never load in timeline/setup");
        let index = setup.args().iter().position(|arg| arg == "--resume").unwrap();
        assert_eq!(setup.args()[index + 1], path.to_str().unwrap());
        assert!(setup.prepare_resume(fixture.0.join("missing.pssa")).is_err());
    }

    #[test]
    fn memory_settings_round_trip_and_validate_cpu_and_resume() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        setup.values[Memory as usize] = "64".into();
        setup.values[MemoryTopK as usize] = "4".into();
        setup.values[Backend as usize] = "cpu".into();
        let spec = setup.validate().unwrap();
        let parsed = crate::cli::parse_train_options_for_test(&spec.args[1..]).unwrap();
        assert_eq!(parsed.memory, 64);
        assert_eq!(parsed.memory_top_k, Some(4));
        assert_eq!(parsed.grow_memory, None);
        setup.values[Backend as usize] = "auto".into();
        assert!(setup.validate().is_err());
        setup.values[Backend as usize] = "cpu".into();
        setup.values[GrowMemory as usize] = "256".into();
        assert!(setup.validate().is_err());
        let checkpoint = fixture.0.join("resume.pssa");
        fs::write(&checkpoint, "validation does not load checkpoints").unwrap();
        setup.values[Resume as usize] = checkpoint.to_string_lossy().into_owned();
        let spec = setup.validate().unwrap();
        let parsed = crate::cli::parse_train_options_for_test(&spec.args[1..]).unwrap();
        assert_eq!(parsed.grow_memory, Some(256));
        setup.values[MemoryTopK as usize] = "0".into();
        assert!(setup.validate().is_err());
    }

    #[test]
    fn loss_guard_settings_round_trip_and_reject_invalid_values() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        setup.values[LossHigh as usize] = "5".into();
        setup.values[LossJump as usize] = "10".into();
        setup.values[LossPatience as usize] = "4".into();
        let spec = setup.validate().unwrap();
        let parsed = crate::cli::parse_train_options_for_test(&spec.args[1..]).unwrap();
        assert_eq!(parsed.loss_guard.high_factor, 5.0);
        assert_eq!(parsed.loss_guard.jump_factor, 10.0);
        assert_eq!(parsed.loss_guard.patience, 4);
        for (field, value) in [(LossHigh, "1"), (LossJump, "NaN"), (LossPatience, "0")] {
            let mut bad = fixture.setup();
            bad.values[field as usize] = value.into();
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn dream_defaults_and_changed_values_round_trip_through_the_real_cli_parser() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        let spec = setup.validate().unwrap();
        let dream_flags = [
            "--dream-every",
            "--dream-replay",
            "--dream-mode",
            "--dream-len",
            "--dream-lr",
            "--dream-steps",
        ];
        for flag in dream_flags {
            assert_eq!(spec.args.iter().filter(|arg| *arg == flag).count(), 0);
        }
        let parsed = crate::cli::parse_train_options_for_test(&spec.args[1..]).unwrap();
        assert_eq!(parsed.dream_every, 0);
        assert_eq!(parsed.dream_replay, 32);
        assert_eq!(parsed.dream_mode, DreamModeValue::Memory);
        assert_eq!(parsed.dream_len, 64);
        assert_eq!(parsed.dream_lr, crate::dream::DEFAULT_REHEARSAL_LR);
        assert_eq!(parsed.dream_steps, crate::dream::DEFAULT_REHEARSAL_STEPS);

        for (mode, expected_flags) in [
            ("memory", [true, true, false, true, true, true]),
            ("generate", [true, true, true, true, true, true]),
            ("both", [true, true, true, true, true, true]),
        ] {
            let mut setup = fixture.setup();
            for (field, value) in [
                (DreamEvery, "1"),
                (DreamReplay, "1"),
                (DreamMode, mode),
                (DreamLen, "1"),
                (DreamLr, "0.001"),
                (DreamSteps, "2"),
            ] {
                setup.values[field as usize] = value.into();
            }
            let spec = setup.validate().unwrap();
            for (flag, should_emit) in dream_flags.into_iter().zip(expected_flags) {
                let positions: Vec<_> = spec
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(index, arg)| (arg == flag).then_some(index))
                    .collect();
                assert_eq!(positions.len(), usize::from(should_emit), "{mode} {flag}");
                if let Some(&index) = positions.first() {
                    assert_eq!(spec.args[index + 1], setup.value(match flag {
                        "--dream-every" => DreamEvery,
                        "--dream-replay" => DreamReplay,
                        "--dream-mode" => DreamMode,
                        "--dream-len" => DreamLen,
                        "--dream-lr" => DreamLr,
                        "--dream-steps" => DreamSteps,
                        _ => unreachable!(),
                    }));
                }
            }
            let parsed = crate::cli::parse_train_options_for_test(&spec.args[1..]).unwrap();
            assert_eq!(parsed.dream_every, 1);
            assert_eq!(parsed.dream_replay, 1);
            assert_eq!(parsed.dream_mode, DreamModeValue::parse(mode).unwrap());
            assert_eq!(parsed.dream_len, 1);
            assert_eq!(parsed.dream_lr, 0.001);
            assert_eq!(parsed.dream_steps, 2);
            assert!(setup.command().contains("--dream-every 1"));
        }
    }

    #[test]
    fn dream_boundaries_and_invalid_values_match_cli_validation() {
        let fixture = Fixture::new();
        for (field, value) in [
            (DreamEvery, "0"),
            (DreamEvery, "1000000"),
            (DreamReplay, "1"),
            (DreamReplay, "1000000"),
            (DreamLen, "1"),
            (DreamLen, "100000"),
            (DreamSteps, "1"),
            (DreamSteps, "1000000"),
            (DreamLr, "1e-45"),
            (DreamLr, "3.4028235e38"),
        ] {
            let mut setup = fixture.setup();
            setup.values[field as usize] = value.into();
            let spec = setup.validate().unwrap();
            assert!(crate::cli::parse_train_options_for_test(&spec.args[1..]).is_ok());
        }
        for (field, value) in [
            (DreamEvery, "-1"),
            (DreamEvery, "1000001"),
            (DreamReplay, "0"),
            (DreamReplay, "1000001"),
            (DreamMode, "tokens"),
            (DreamLen, "0"),
            (DreamLen, "100001"),
            (DreamLr, "0"),
            (DreamLr, "-1"),
            (DreamLr, "NaN"),
            (DreamLr, "inf"),
            (DreamSteps, "0"),
            (DreamSteps, "1000001"),
        ] {
            let mut setup = fixture.setup();
            setup.values[field as usize] = value.into();
            assert!(setup.validate().is_err(), "wizard accepted {field:?}={value}");
            let args = setup.args();
            assert!(
                crate::cli::parse_train_options_for_test(&args[1..]).is_err(),
                "CLI accepted {field:?}={value}"
            );
        }
    }

    #[test]
    fn validates_every_numeric_field_and_required_paths() {
        let fixture = Fixture::new();
        assert!(fixture.setup().validate().is_ok());
        for (field, value) in [
            (Latent, "0"),
            (State, "4097"),
            (Vocab, "256"),
            (Vocab, "65537"),
            (Depth, "33"),
            (Loops, "0"),
            (Epochs, "1000001"),
            (Chunk, "65537"),
            (Accumulate, "0"),
            (Threads, "0"),
            (Threads, "65537"),
            (Ram, "0"),
            (Ram, "184467440737095516160"),
            (Batch, "65537"),
            (Batch, "0"),
            (Lr, "NaN"),
            (Lr, "inf"),
            (Lr, "-1"),
            (Seed, "-1"),
            (MaxTokens, "0"),
            (Backend, "tpu"),
            (Dataset, ""),
            (Output, ""),
            (Resume, "missing.pssa"),
        ] {
            let mut setup = fixture.setup();
            setup.values[field as usize] = value.into();
            assert!(setup.validate().is_err(), "{field:?}={value}");
        }
        let setup = fixture.setup();
        fs::create_dir_all(setup.value(Output)).unwrap();
        fs::write(Path::new(setup.value(Output)).join("model.pssa"), "keep me").unwrap();
        assert!(setup.validate().err().unwrap().contains("already exists"));
        assert_eq!(
            fs::read_to_string(Path::new(setup.value(Output)).join("model.pssa")).unwrap(),
            "keep me"
        );
    }

    #[test]
    fn rejects_unsafe_sources_and_existing_logs_without_touching_them() {
        let fixture = Fixture::new();
        for name in ["source,part.txt", "trailing space "] {
            let mut setup = fixture.setup();
            let path = fixture.0.join(name);
            fs::write(&path, "local data").unwrap();
            setup.values[Dataset as usize] = path.to_string_lossy().into_owned();
            assert!(
                setup
                    .validate()
                    .err()
                    .unwrap()
                    .contains("source-list syntax")
            );
        }
        let mut setup = fixture.setup();
        setup.values[Output as usize].push('\n');
        assert!(
            setup
                .validate()
                .err()
                .unwrap()
                .contains("control characters")
        );
        let mut setup = fixture.setup();
        fs::create_dir_all(setup.value(Output)).unwrap();
        let log = Path::new(setup.value(Output)).join("train.log");
        fs::write(&log, "existing training log").unwrap();
        assert!(setup.launch(false).is_none());
        assert!(setup.message.contains("already exists"));
        assert_eq!(fs::read_to_string(log).unwrap(), "existing training log");
    }

    #[test]
    fn optional_limits_and_device_selection_reach_real_cli_flags() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        for flag in ["--threads", "--ram-mib", "--batch-size", "--max-tokens"] {
            assert!(
                !setup.args().iter().any(|arg| arg == flag),
                "default added {flag}"
            );
        }
        setup.set_backend(TrainingBackend::Cpu);
        let limits = ResourceLimits {
            threads: Some(2),
            ram_mib: cfg!(target_os = "linux").then_some(2048),
            batch_size: Some(3),
            max_tokens: Some(77),
        };
        setup.set_limits(limits);
        assert_eq!(setup.limits().unwrap(), limits);
        let spec = setup.validate().unwrap();
        for (flag, expected) in [
            ("--threads", "2"),
            ("--batch-size", "3"),
            ("--max-tokens", "77"),
            ("--backend", "cpu"),
        ] {
            let index = spec.args.iter().position(|arg| arg == flag).unwrap();
            assert_eq!(spec.args[index + 1], expected);
            assert_eq!(spec.args.iter().filter(|arg| *arg == flag).count(), 1);
        }
        if cfg!(target_os = "linux") {
            assert!(setup.command().contains("--ram-mib 2048"));
        }
        setup.set_limits(ResourceLimits::default());
        assert_eq!(setup.limits().unwrap(), ResourceLimits::default());
    }

    #[test]
    fn hf_fields_command_and_resume_preserve_argument_boundaries() {
        let fixture = Fixture::new();
        let mut setup = fixture.setup();
        setup.values[Source as usize] = "hf".into();
        setup.values[Dataset as usize] = "owner/corpus".into();
        setup.values[HfConfig as usize] = "large config".into();
        setup.values[Resume as usize] = fixture
            .0
            .join("checkpoint with spaces.pssa")
            .to_string_lossy()
            .into_owned();
        fs::write(setup.value(Resume), "child validates actual checkpoint").unwrap();
        let spec = setup.validate().unwrap();
        assert!(!spec.args.contains(&"--data".into()));
        for (flag, value) in [
            ("--hf-dataset", "owner/corpus"),
            ("--hf-config", "large config"),
            ("--hf-split", "train"),
            ("--hf-field", "text"),
            ("--resume", setup.value(Resume)),
            ("--backend", "auto"),
        ] {
            let index = spec.args.iter().position(|arg| arg == flag).unwrap();
            assert_eq!(spec.args[index + 1], value);
        }
        assert_eq!(spec.args.last().unwrap(), "--no-tui");
        assert!(setup.command().contains("'large config'"));
        assert!(setup.command().contains("'\"'\"'"));
        setup.values[Dataset as usize] = "invalid".into();
        assert!(setup.validate().is_err());
        setup.values[Dataset as usize] = "owner/corpus".into();
        setup.values[HfSplit as usize].clear();
        assert!(setup.validate().is_err());
        setup.values[HfSplit as usize] = "train".into();
        setup.values[HfConfig as usize] = "   ".into();
        assert!(setup.validate().is_err());
        setup.values[HfConfig as usize].clear();
        assert!(setup.validate().is_ok());
        assert_eq!(safe_path("-resume with spaces"), "./-resume with spaces");
    }

    #[cfg(unix)]
    #[test]
    fn equivalent_shell_command_keeps_existing_logs_and_checkpoints() {
        let fixture = Fixture::new();
        let setup = fixture.setup();
        fs::create_dir_all(setup.value(Output)).unwrap();
        let log = Path::new(setup.value(Output)).join("train.log");
        let checkpoint = Path::new(setup.value(Output)).join("model.pssa");
        for path in [&log, &checkpoint] {
            fs::write(path, "do not overwrite").unwrap();
            let result = Command::new("/bin/sh")
                .args(["-c", &setup.command()])
                .output()
                .unwrap();
            assert!(!result.status.success());
            assert_eq!(fs::read_to_string(path).unwrap(), "do not overwrite");
            if path == &checkpoint {
                assert!(
                    !log.exists(),
                    "do not even launch against an existing checkpoint"
                );
            }
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn keyboard_edits_unicode_and_only_starts_from_explicit_review_button() {
        let mut setup = Setup::default();
        press(&mut setup, KeyCode::Down);
        press(&mut setup, KeyCode::Enter);
        for c in "héllo q".chars() {
            press(&mut setup, KeyCode::Char(c));
        }
        press(&mut setup, KeyCode::Backspace);
        press(&mut setup, KeyCode::Enter);
        assert_eq!(setup.value(Dataset), "héllo ");
        press(&mut setup, KeyCode::Enter);
        press(&mut setup, KeyCode::Char('x'));
        press(&mut setup, KeyCode::Esc);
        assert_eq!(setup.value(Dataset), "héllo ");
        assert!(!press(&mut setup, KeyCode::Right));
        setup.selected = 3;
        press(&mut setup, KeyCode::Char('+'));
        assert_eq!(setup.value(Depth), "2");
        press(&mut setup, KeyCode::Right);
        press(&mut setup, KeyCode::Right);
        assert!(!press(&mut setup, KeyCode::Enter)); // edits output, does NOT run
        press(&mut setup, KeyCode::Esc);
        setup.selected = setup.fields().len();
        assert!(press(&mut setup, KeyCode::Enter));
        assert!(setup.launch(true).is_none());
        assert!(setup.message.contains("active"));
    }

    #[test]
    fn test_backend_wizard_pages_command_validation_and_narrow_fallback() {
        let fixture = Fixture::new();
        for (width, height) in [
            (0, 0),
            (1, 1),
            (10, 5),
            (30, 10),
            (60, 20),
            (79, 24),
            (80, 24),
            (120, 40),
        ] {
            let mut setup = fixture.setup();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            for page in 0..4 {
                setup.page = page;
                terminal.draw(|f| setup.draw(f, f.area())).unwrap();
                if width >= 30 && height >= 10 {
                    assert!(text(&terminal).contains(PAGES[page]));
                }
                if width >= 80 && height >= 24 {
                    assert!(text(&terminal).contains("equivalent CLI"));
                    assert!(text(&terminal).contains("mkdir -p"));
                }
            }
            if width >= 80 && height >= 24 {
                setup.command_scroll = u16::MAX;
                terminal.draw(|f| setup.draw(f, f.area())).unwrap();
                assert!(text(&terminal).contains("/dev/null"));
                assert!(setup.command_scroll < u16::MAX);
            }
            if width >= 60 && height >= 20 {
                setup.fail("Learning rate must be positive");
                terminal.draw(|f| setup.draw(f, f.area())).unwrap();
                assert!(text(&terminal).contains("Learning rate must be positive"));
                assert!(
                    terminal
                        .backend()
                        .buffer()
                        .content()
                        .iter()
                        .any(|c| c.fg == BRIGHT_RED)
                );
            }
        }
    }

    #[test]
    fn test_backend_fullscreen_command_is_available_on_narrow_terminals() {
        let fixture = Fixture::new();
        for (width, height) in [(30, 10), (60, 20), (79, 24)] {
            let mut setup = fixture.setup();
            assert!(!press(&mut setup, KeyCode::Char('c')));
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| setup.draw(f, f.area())).unwrap();
            assert!(text(&terminal).contains("equivalent CLI"));
            assert!(text(&terminal).contains("mkdir -p"));
            setup.command_scroll = u16::MAX;
            terminal.draw(|f| setup.draw(f, f.area())).unwrap();
            assert!(text(&terminal).contains("/dev/null"));
            assert!(
                !press(&mut setup, KeyCode::Enter),
                "preview cannot launch a run"
            );
            press(&mut setup, KeyCode::Char('c'));
            terminal.draw(|f| setup.draw(f, f.area())).unwrap();
            assert!(text(&terminal).contains("setup / 1 dataset"));
        }
    }

    #[test]
    fn test_backend_long_unicode_edits_keep_the_tail_and_cursor_visible() {
        for width in [30, 60, 79, 80, 120] {
            let mut setup = Setup::default();
            setup.page = 3;
            setup.edit = Some(format!("{}末尾Z", "long 路径/".repeat(20)));
            let mut terminal = Terminal::new(TestBackend::new(width, 20)).unwrap();
            terminal.draw(|f| setup.draw(f, f.area())).unwrap();
            let cells = terminal.backend().buffer().content();
            let cursor = cells
                .iter()
                .position(|cell| cell.symbol() == "▏")
                .unwrap_or_else(|| panic!("cursor clipped at {width} columns"));
            assert!(cursor % usize::from(width) >= 3);
            assert_eq!(cells[cursor - 1].symbol(), "Z");
            // A double-width glyph occupies two cells; its continuation is blank.
            assert_eq!(cells[cursor - 3].symbol(), "尾");
        }
        assert_eq!(visible_tail("路径尾Z", 3), "尾Z");
        assert_eq!(visible_tail("路径尾Z", 2), "Z");
        assert_eq!(visible_tail("路径尾Z", 0), "");
    }

    #[test]
    fn test_backend_looped_monitor_uses_real_input_not_a_synthetic_pass_ring() {
        let mut state = RunState::default();
        state.ingest("model=pssa parameters=100 vocab=257 depth=2 loops=3");
        state.ingest("training 1/2 (50%) loss=4.0 tokens_per_second=100");
        assert_eq!(state.loop_count, 3);
        for (width, height) in [(120, 40), (80, 24), (79, 24)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|f| super::super::draw(f, &state, 0)).unwrap();
            let screen = text(&terminal);
            assert!(screen.contains("run metrics"));
            assert!(screen.contains("actual input"));
            assert!(!screen.contains("configuration diagram"));
            assert!(!screen.contains("pass 1/3"));
            assert!(!screen.contains("neuron /"));
        }
    }

    #[test]
    fn child_fixture() {
        // Invoked only with --exact by the child lifecycle test below.
        if std::env::args().any(|arg| arg == "--nocapture") {
            // More than one poll budget, including a line exceeding the cap.
            println!("{}", "x".repeat(70_000));
            println!("{}", "y".repeat(70_000));
            println!("model=pssa parameters=100 vocab=257 depth=2 loops=3");
            eprintln!("fixture stderr captured");
            println!(
                "training 1/1 (100%) loss=4.0 tokens_per_second=100 optimizer_updates=1 updates_total=1"
            );
            print!("final unterminated log line");
        }
    }

    #[test]
    fn child_log_is_durable_and_completion_is_drained_without_ui_backpressure() {
        let fixture = Fixture::new();
        let output = fixture.0.join("child output");
        let spec = RunSpec {
            output: output.clone(),
            loops: 3,
            args: vec![
                "--exact".into(),
                "tui::setup::tests::child_fixture".into(),
                "--nocapture".into(),
            ],
        };
        let mut run = spec.start(&std::env::current_exe().unwrap()).unwrap();
        let mut state = RunState::default();
        run.initialize(&mut state);
        assert!(run.active());
        assert_eq!(run.succeeded(), None);
        assert_eq!(state.loop_count, 3);
        // Wait WITHOUT draining any output: the trainer must not need the UI.
        #[cfg(target_os = "linux")]
        {
            let process = fs::read_to_string(format!("/proc/{}/stat", run.child.id())).unwrap();
            let group = process
                .rsplit_once(") ")
                .unwrap()
                .1
                .split_whitespace()
                .nth(2)
                .unwrap();
            assert_eq!(
                group.parse::<u32>().unwrap(),
                run.child.id(),
                "detached process group"
            );
        }
        assert!(run.child.wait().unwrap().success());
        let mut telemetry = Vec::new();
        run.poll_with(&mut state, |line| telemetry.push(line.to_owned()));
        assert!(
            run.active(),
            "completion must wait for the entire durable log to drain"
        );
        assert_eq!(
            run.succeeded(),
            None,
            "sweeps must still wait after child exit"
        );
        assert!(run.pending.len() <= 65_536, "bound unterminated lines");
        let deadline = Instant::now() + Duration::from_secs(5);
        while run.active() && Instant::now() < deadline {
            run.poll_with(&mut state, |line| telemetry.push(line.to_owned()));
        }
        assert!(
            telemetry
                .iter()
                .any(|line| line.contains("tokens_per_second=100")),
            "wizard telemetry must also reach extras/alerts"
        );
        assert!(
            telemetry
                .iter()
                .any(|line| line.contains("final unterminated log line"))
        );
        assert!(!run.active());
        assert!(!state.training_active);
        assert_eq!(run.succeeded(), Some(true));
        let mut slot = Some(run);
        assert!(super::super::release_finished_training(&mut slot, true));
        assert!(slot.is_none());
        assert_eq!(state.live_loss, Some(4.0));
        assert!(
            state
                .raw_lines
                .iter()
                .any(|line| line.contains("fixture stderr captured"))
        );
        assert!(
            fs::read_to_string(output.join("train.log"))
                .unwrap()
                .contains("final unterminated log line")
        );
        assert!(state.problem.is_none());
    }

    #[test]
    fn failed_child_and_failed_spawn_are_visible_and_do_not_overwrite_logs() {
        let fixture = Fixture::new();
        let output = fixture.0.join("failed output");
        let spec = RunSpec {
            output: output.clone(),
            loops: 1,
            args: vec!["--not-a-real-test-flag".into()],
        };
        let mut run = spec.start(&std::env::current_exe().unwrap()).unwrap();
        let mut state = RunState::default();
        run.initialize(&mut state);
        assert!(!run.child.wait().unwrap().success());
        run.poll(&mut state);
        assert!(!run.active());
        assert!(state.problem.as_deref().unwrap().contains("Trainer exited"));
        let spec = RunSpec {
            output: output.clone(),
            loops: 1,
            args: vec![],
        };
        assert!(spec.start(Path::new("/no/such/trainer")).is_err());
        assert!(output.join("train.log").exists());
        let output = fixture.0.join("spawn failure");
        let spec = RunSpec {
            output: output.clone(),
            loops: 1,
            args: vec![],
        };
        assert!(spec.start(Path::new("/no/such/trainer")).is_err());
        assert!(!output.join("train.log").exists());
    }
}
