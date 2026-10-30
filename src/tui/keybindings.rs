//! One registry for shortcut dispatch and the help overlay. Reused keys are
//! explicitly scoped; printable input is a fallback, never a global shortcut.
use super::{nav, panel_bg, accent, panel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};

pub(super) const MONITOR_TAB: usize = 0;
pub(super) const CHAIN_TAB: usize = 1;
pub(super) const MODEL_TAB: usize = 2;
pub(super) const FEED_TAB: usize = 3;
pub(super) const INFERENCE_TAB: usize = 4;
pub(super) const SETUP_TAB: usize = 5;
pub(super) const HF_TAB: usize = 6;
pub(super) const KAGGLE_TAB: usize = 7;
pub(super) const MEMORY_TAB: usize = 8;
pub(super) const RUNS_TAB: usize = 9;
pub(super) const BENCHMARK_TAB: usize = 10;
pub(super) const SAMPLE_TAB: usize = 11;
pub(super) const HARDWARE_TAB: usize = 12;
pub(super) const MATH_TAB: usize = 13;
pub(super) const DEVICE_TAB: usize = 14;
pub(super) const LIMITS_TAB: usize = 15;
pub(super) const LIBRARY_TAB: usize = 16;
pub(super) const MIXER_TAB: usize = 17;
pub(super) const EVAL_TAB: usize = 18;
pub(super) const BACKUP_TAB: usize = 19;
pub(super) const LOG_STREAM_TAB: usize = 20;
pub(super) const NOTIFY_TAB: usize = 21;
pub(super) const UPDATE_TAB: usize = 22;
pub(super) const SUPPORT_TAB: usize = 23;
pub(super) const GITHUB_TAB: usize = 24;
pub(super) const SWEEP_TAB: usize = 25;
pub(super) const TIMELINE_TAB: usize = 26;
/// Every renderable screen, indexed by its stable numeric id. The visible
/// navigation is the smaller category strip in `nav`; this list stays the
/// internal registry so screen modules keep one id each.
pub(super) const SCREENS: [&str; 27] = [
    "monitor",
    "chain",
    "model",
    "feed",
    "inference",
    "setup",
    "HF login",
    "Kaggle",
    "memory",
    "runs",
    "benchmark",
    "sample",
    "hardware",
    "math",
    "devices",
    "limits",
    "library",
    "mixer",
    "eval",
    "HF backup",
    "cloud log",
    "phone ping",
    "updates",
    "support",
    "GitHub",
    "sweeps",
    "timeline",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Context {
    Monitor,
    Dashboard,
    Chat,
    Setup,
    SetupEdit,
    HfInput,
    KaggleInput,
    KaggleBusy,
    Memory,
    Runs,
    RunsEdit,
    Benchmark,
    BenchmarkEdit,
    Palette,
    Library,
    LibraryEdit,
    Mixer,
    Eval,
    EvalEdit,
    Help,
    Sample,
    Math,
    Device,
    Limits,
    LimitsEdit,
    Network,
    NetworkEdit,
    Timeline,
    Feed,
    Menu,
}
impl Context {
    fn mask(self) -> u32 {
        1 << self as u32
    }

    pub(super) fn for_tab(tab: usize, editing: bool) -> Self {
        match tab {
            0 => Self::Monitor,
            3 => Self::Feed,
            4 => Self::Chat,
            5 if editing => Self::SetupEdit,
            5 => Self::Setup,
            HF_TAB => Self::HfInput,
            KAGGLE_TAB if editing => Self::KaggleInput,
            KAGGLE_TAB => Self::KaggleBusy,
            MEMORY_TAB => Self::Memory,
            RUNS_TAB if editing => Self::RunsEdit,
            RUNS_TAB => Self::Runs,
            BENCHMARK_TAB if editing => Self::BenchmarkEdit,
            BENCHMARK_TAB => Self::Benchmark,
            SAMPLE_TAB => Self::Sample,
            MATH_TAB => Self::Math,
            DEVICE_TAB => Self::Device,
            LIMITS_TAB if editing => Self::LimitsEdit,
            LIMITS_TAB => Self::Limits,
            LIBRARY_TAB if editing => Self::LibraryEdit,
            LIBRARY_TAB => Self::Library,
            MIXER_TAB => Self::Mixer,
            EVAL_TAB if editing => Self::EvalEdit,
            EVAL_TAB => Self::Eval,
            TIMELINE_TAB => Self::Timeline,
            BACKUP_TAB..=SWEEP_TAB if editing => Self::NetworkEdit,
            BACKUP_TAB..=SWEEP_TAB => Self::Network,
            _ => Self::Dashboard,
        }
    }
}

const MONITOR: u32 = 1 << Context::Monitor as u32;
const DASHBOARD: u32 = 1 << Context::Dashboard as u32;
const CHAT: u32 = 1 << Context::Chat as u32;
const SETUP: u32 = 1 << Context::Setup as u32;
const EDIT: u32 = 1 << Context::SetupEdit as u32;
const HF: u32 = 1 << Context::HfInput as u32;
const KAGGLE_INPUT: u32 = 1 << Context::KaggleInput as u32;
const KAGGLE_BUSY: u32 = 1 << Context::KaggleBusy as u32;
const MEMORY: u32 = 1 << Context::Memory as u32;
const RUNS: u32 = 1 << Context::Runs as u32;
const RUNS_EDIT: u32 = 1 << Context::RunsEdit as u32;
const BENCHMARK: u32 = 1 << Context::Benchmark as u32;
const BENCHMARK_EDIT: u32 = 1 << Context::BenchmarkEdit as u32;
const PALETTE: u32 = 1 << Context::Palette as u32;
const HELP: u32 = 1 << Context::Help as u32;
const EXTRA_EDIT: u32 = KAGGLE_INPUT | RUNS_EDIT | BENCHMARK_EDIT;
const EXTRA_BROWSE: u32 = KAGGLE_BUSY | MEMORY | RUNS | BENCHMARK;
const EXTRA: u32 = EXTRA_EDIT | EXTRA_BROWSE;
const SAMPLE: u32 = 1 << Context::Sample as u32;
const MATH: u32 = 1 << Context::Math as u32;
const FEED: u32 = 1 << Context::Feed as u32;
const DEVICE: u32 = 1 << Context::Device as u32;
const LIMITS: u32 = 1 << Context::Limits as u32;
const LIMIT_EDIT: u32 = 1 << Context::LimitsEdit as u32;
const PAGES: u32 = DASHBOARD | SAMPLE | MATH | DEVICE | LIMITS | FEED;
const LIBRARY: u32 = 1 << Context::Library as u32;
const LIBRARY_EDIT: u32 = 1 << Context::LibraryEdit as u32;
const MIXER: u32 = 1 << Context::Mixer as u32;
const EVAL: u32 = 1 << Context::Eval as u32;
const EVAL_EDIT: u32 = 1 << Context::EvalEdit as u32;
const LOCAL_BROWSE: u32 = LIBRARY | MIXER | EVAL;
const LOCAL_EDIT: u32 = LIBRARY_EDIT | EVAL_EDIT;
const NETWORK: u32 = 1 << Context::Network as u32;
const NETWORK_EDIT: u32 = 1 << Context::NetworkEdit as u32;
const TIMELINE: u32 = 1 << Context::Timeline as u32;
const MENU: u32 = 1 << Context::Menu as u32;
const BROWSE: u32 = MONITOR | PAGES | SETUP | EXTRA_BROWSE | LOCAL_BROWSE | NETWORK | TIMELINE;
const ALL: u32 = BROWSE
    | CHAT
    | EDIT
    | HF
    | EXTRA_EDIT
    | PALETTE
    | HELP
    | LIMIT_EDIT
    | LOCAL_EDIT
    | NETWORK_EDIT
    | MENU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Quit,
    NextTab,
    PreviousTab,
    BackToSections,
    MenuMove(isize),
    MenuOpen,
    ToggleHelp,
    TogglePalette,
    Palette,
    Chat,
    Setup,
    Hf,
    Extras,
    Network,
    OpenCategory(usize),
    Heatmap,
    Preview,
    MathScroll(i16),
    MathTop,
    MathBottom,
    FeedScroll(i16),
    FeedTop,
    FeedBottom,
    Device,
    Limits,
    Local,
    Pan(bool),
    CycleGraph,
    Zoom(bool),
    ResetGraph,
    ScrollHelp(i16),
    HelpTop,
    HelpBottom,
}

struct Binding {
    code: KeyCode,
    modifiers: KeyModifiers,
    label: &'static str,
    description: &'static str,
    routes: &'static [(u32, Action)],
}

macro_rules! bind {
    ($code:expr, $label:literal, $description:literal, $($scope:expr => $action:expr),+ $(,)?) => {
        Binding {
            code: $code,
            modifiers: KeyModifiers::NONE,
            label: $label,
            description: $description,
            routes: &[$(($scope, $action)),+],
        }
    };
}

use Action::*;
use KeyCode::*;

const BINDINGS: &[Binding] = &[
    Binding {
        code: Char('c'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+C",
        description: "Quit the TUI. Training started from the setup wizard keeps running.",
        routes: &[(ALL, Quit)],
    },
    bind!(Tab, "Tab", "Next category. Works even while typing in an editor.", ALL => NextTab),
    bind!(F(6), "F6", "Toggle the token confidence heatmap: how sure the model is about each token (chat, monitor, sample).", CHAT | MONITOR | SAMPLE => Heatmap),
    bind!(F(7), "F7", "Pause or resume the live output preview (monitor, sample).", MONITOR | SAMPLE => Preview),
    bind!(F(1), "F1", "Open or close this help. Works even while typing.", ALL => ToggleHelp),
    bind!(Char('?'), "?", "Open or close this help outside text editors. In an editor it just types '?'.", BROWSE | MENU | HELP => ToggleHelp),
    bind!(Char('q'), "q", "Quit from a normal tab or the section menu. When help is open, close help instead.", BROWSE | MENU => Quit, HELP => ToggleHelp),
    bind!(Char('1'), "1", "Open the Overview category menu.", BROWSE | MENU => OpenCategory(0)),
    bind!(Char('2'), "2", "Open the Model category menu.", BROWSE | MENU => OpenCategory(1)),
    bind!(Char('3'), "3", "Open the Data & Eval category menu.", BROWSE | MENU => OpenCategory(2)),
    bind!(Char('4'), "4", "Open the Training category menu.", BROWSE | MENU => OpenCategory(3)),
    bind!(Char('5'), "5", "Open the Settings category menu.", BROWSE | MENU => OpenCategory(4)),
    Binding {
        code: Char('1'),
        modifiers: KeyModifiers::ALT,
        label: "Alt+1",
        description: "Open the Overview category menu, even while typing.",
        routes: &[(ALL, OpenCategory(0))],
    },
    Binding {
        code: Char('2'),
        modifiers: KeyModifiers::ALT,
        label: "Alt+2",
        description: "Open the Model category menu, even while typing.",
        routes: &[(ALL, OpenCategory(1))],
    },
    Binding {
        code: Char('3'),
        modifiers: KeyModifiers::ALT,
        label: "Alt+3",
        description: "Open the Data & Eval category menu, even while typing.",
        routes: &[(ALL, OpenCategory(2))],
    },
    Binding {
        code: Char('4'),
        modifiers: KeyModifiers::ALT,
        label: "Alt+4",
        description: "Open the Training category menu, even while typing.",
        routes: &[(ALL, OpenCategory(3))],
    },
    Binding {
        code: Char('5'),
        modifiers: KeyModifiers::ALT,
        label: "Alt+5",
        description: "Open the Settings category menu, even while typing.",
        routes: &[(ALL, OpenCategory(4))],
    },
    bind!(Esc, "Esc", "Back out one level: close help or the palette, cancel an edit, stop chat work, cancel a running job, or leave a screen for its section menu.",
        MONITOR | PAGES | SETUP | LOCAL_BROWSE | NETWORK | TIMELINE => BackToSections, MENU => BackToSections, CHAT => Chat, EDIT => Setup, HF => Hf, EXTRA => Extras, LIMIT_EDIT => Limits, LOCAL_EDIT => Local, NETWORK_EDIT => Network, PALETTE => Palette, HELP => ToggleHelp),
    bind!(Left, "Left", "Monitor: pan to older steps. Timeline: older checkpoint. Setup: previous page.",
        MONITOR => Pan(false), SETUP => Setup, TIMELINE => Network),
    bind!(Right, "Right", "Monitor: pan to newer steps. Timeline: newer checkpoint. Setup: next page.",
        MONITOR => Pan(true), SETUP => Setup, TIMELINE => Network),
    bind!(Char('g'), "g", "Monitor: cycle the graph view: loss, perplexity, tokens per second, learning rate, comparison, all metrics, memory.", MONITOR => CycleGraph),
    bind!(Char('+'), "+", "Zoom in: monitor graph; setup depth/loops; mixer share.", MONITOR => Zoom(true), SETUP => Setup, MIXER => Local),
    bind!(Char('='), "=", "Zoom in: monitor graph; mixer share (same as +).", MONITOR => Zoom(true), MIXER => Local),
    bind!(Char('-'), "-", "Zoom out: monitor graph; setup depth/loops; mixer share.", MONITOR => Zoom(false), SETUP => Setup, MIXER => Local),
    bind!(Char('0'), "0", "Monitor: reset pan and zoom.", MONITOR => ResetGraph),
    bind!(Up, "Up", "Move up: section menu, setup fields, run and benchmark lists, devices, limits, local and network lists. Scroll up: feed, math, help.", MENU => MenuMove(-1), FEED => FeedScroll(-1), SETUP => Setup, RUNS | BENCHMARK => Extras, PALETTE => Palette, DEVICE => Device, LIMITS => Limits, LOCAL_BROWSE => Local, NETWORK | TIMELINE => Network, MATH => MathScroll(-1), HELP => ScrollHelp(-1)),
    bind!(Down, "Down", "Move down: section menu, setup fields, run and benchmark lists, devices, limits, local and network lists. Scroll down: feed, math, help.", MENU => MenuMove(1), FEED => FeedScroll(1), SETUP => Setup, RUNS | BENCHMARK => Extras, PALETTE => Palette, DEVICE => Device, LIMITS => Limits, LOCAL_BROWSE => Local, NETWORK | TIMELINE => Network, MATH => MathScroll(1), HELP => ScrollHelp(1)),
    bind!(BackTab, "Shift+Tab", "Previous category.", ALL => PreviousTab),
    bind!(F(5), "F5", "Setup: next wizard page.", SETUP => Setup),
    bind!(Char('c'), "c", "Setup: show command preview. Runs/library: chat with the checkpoint. Network/timeline: open or copy.", SETUP => Setup, RUNS => Extras, LIBRARY => Local, NETWORK | TIMELINE => Network),
    bind!(Char('r'), "r", "Refresh or retry (memory, runs, devices, library, eval, network, timeline).", MEMORY | RUNS => Extras, DEVICE => Device, LIBRARY | EVAL => Local, NETWORK | TIMELINE => Network),
    bind!(Char('v'), "v", "Memory: switch between checkpoint inspector and live chat retrieval.", MEMORY => Extras),
    bind!(Char('p'), "p", "Eval: edit the prompt. HF backup/cloud log/phone ping/sweeps/GitHub: run the tab's action.", EVAL => Local, NETWORK => Network),
    bind!(Char('i'), "i", "GitHub: open the issue list.", NETWORK => Network),
    bind!(Char('l'), "l", "GitHub: enter the masked-token login.", NETWORK => Network),
    bind!(Char('o'), "o", "Support/GitHub/updates: open the selected link in your browser.", NETWORK => Network),
    bind!(Char('s'), "s", "Runs: score a checkpoint on a held-out file.", RUNS => Extras),
    bind!(Char('b'), "b", "Benchmark: run the configured matched comparison.", BENCHMARK => Extras),
    bind!(Char('y'), "y", "Kaggle preview: confirm the upload and launch.", KAGGLE_BUSY => Extras),
    bind!(Char('Y'), "Y", "Kaggle preview: confirm the upload and launch (uppercase Y).", KAGGLE_BUSY => Extras),
    bind!(Char('n'), "n", "Kaggle preview: cancel the launch.", KAGGLE_BUSY => Extras),
    bind!(Char('N'), "N", "Kaggle preview: cancel the launch (uppercase N).", KAGGLE_BUSY => Extras),
    bind!(Char('m'), "m", "Library: edit the models folder.", LIBRARY => Local),
    bind!(Char('d'), "d", "Limits: reset the draft. Library: edit the datasets folder. Network: toggle, stop or cancel.", LIMITS => Limits, LIBRARY => Local, NETWORK => Network),
    bind!(Char('a'), "a", "Eval: pause or resume background checkpoint evaluation.", EVAL => Local),
    bind!(Char('u'), "u", "Library: fill Setup's Resume-from with the selected checkpoint.", LIBRARY => Local),
    bind!(Char('t'), "t", "Library: fill Setup's Dataset (converts JSONL/Parquet in the background).", LIBRARY => Local),
    bind!(PageUp, "PgUp", "Scroll up: chat, setup preview, memory, local, network, feed, math, help.", FEED => FeedScroll(-8), CHAT => Chat, SETUP => Setup, MEMORY => Extras, EVAL => Local, NETWORK => Network, MATH => MathScroll(-8), HELP => ScrollHelp(-8)),
    bind!(PageDown, "PgDn", "Scroll down: chat, setup preview, memory, local, network, feed, math, help.", FEED => FeedScroll(8), CHAT => Chat, SETUP => Setup, MEMORY => Extras, EVAL => Local, NETWORK => Network, MATH => MathScroll(8), HELP => ScrollHelp(8)),
    bind!(Home, "Home", "First line (feed, math, help). First field (devices, limits, network). First checkpoint (timeline).", FEED => FeedTop, HELP => HelpTop, MATH => MathTop, DEVICE => Device, LIMITS => Limits, NETWORK | TIMELINE => Network),
    bind!(End, "End", "Chat: follow. Last line (feed, math, help). Last field (devices, limits, network).", FEED => FeedBottom, CHAT => Chat, HELP => HelpBottom, MATH => MathBottom, DEVICE => Device, LIMITS => Limits, NETWORK | TIMELINE => Network),
    bind!(Enter, "Enter", "Section menu: open the highlighted section. Chat: send. Setup: edit/save/start. HF: log in. Kaggle: review. Runs: monitor/score. Benchmark/local/network: edit/save/open. Palette: run. Devices/limits: select/edit/apply. Timeline: prepare resume.", MENU => MenuOpen, CHAT => Chat, SETUP | EDIT => Setup, HF => Hf, KAGGLE_INPUT | RUNS | RUNS_EDIT | BENCHMARK | BENCHMARK_EDIT => Extras, PALETTE => Palette, DEVICE => Device, LIMITS | LIMIT_EDIT => Limits, LOCAL_BROWSE | LOCAL_EDIT => Local, NETWORK | NETWORK_EDIT | TIMELINE => Network),
    bind!(Char(' '), "Space", "Activate the selected field or button (setup, devices, limits). In a text editor it types a space.", SETUP => Setup, DEVICE => Device, LIMITS => Limits),
    bind!(Backspace, "Backspace", "Delete a character in text editors and the palette.", CHAT => Chat, EDIT => Setup, HF => Hf, EXTRA_EDIT => Extras, PALETTE => Palette, LIMIT_EDIT => Limits, LOCAL_EDIT => Local, NETWORK_EDIT => Network),
    Binding {
        code: Char('u'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+U",
        description: "Clear the input in chat, setup, Kaggle, score, benchmark, limits, local and network editors.",
        routes: &[
            (CHAT, Chat),
            (EDIT, Setup),
            (EXTRA_EDIT, Extras),
            (LIMIT_EDIT, Limits),
            (LOCAL_EDIT, Local),
            (NETWORK_EDIT, Network),
        ],
    },
    Binding {
        code: Char('l'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+L",
        description: "HF/GitHub login: log out and remove the saved token.",
        routes: &[(HF, Hf), (NETWORK | NETWORK_EDIT, Network)],
    },
    Binding {
        code: Char('k'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+K",
        description: "Open or close the command palette: jump to any tab (all tabs and editors).",
        routes: &[(ALL, TogglePalette)],
    },
];

// Crossterm includes SHIFT on some terminals for printable symbols and BackTab.
// Do not discard CTRL/ALT: modified letters must not trigger plain shortcuts.
fn normalized_modifiers(key: KeyEvent) -> KeyModifiers {
    let mut modifiers = key.modifiers;
    if matches!(key.code, Char(_) | BackTab) {
        modifiers.remove(KeyModifiers::SHIFT);
    }
    modifiers
}

impl Context {
    /// Contexts where a letter or digit must reach the editor, chat or palette
    /// query instead of acting as navigation.
    fn types_text(self) -> bool {
        matches!(
            self,
            Context::Chat
                | Context::SetupEdit
                | Context::HfInput
                | Context::KaggleInput
                | Context::RunsEdit
                | Context::BenchmarkEdit
                | Context::Palette
                | Context::LimitsEdit
                | Context::LibraryEdit
                | Context::EvalEdit
                | Context::NetworkEdit
        )
    }
}

/// j/k move like the arrow keys outside text editors, so menus and lists share
/// one grammar.
fn motion_key(key: KeyEvent, context: Context) -> KeyEvent {
    if context.types_text() || !normalized_modifiers(key).is_empty() {
        return key;
    }
    match key.code {
        Char('j') => KeyEvent::new(Down, KeyModifiers::NONE),
        Char('k') => KeyEvent::new(Up, KeyModifiers::NONE),
        _ => key,
    }
}

pub(super) fn action(key: KeyEvent, context: Context) -> Option<Action> {
    let key = motion_key(key, context);
    let modifiers = normalized_modifiers(key);
    if let Some(action) = BINDINGS
        .iter()
        .filter(|binding| binding.code == key.code && binding.modifiers == modifiers)
        .flat_map(|binding| binding.routes)
        .find_map(|&(scope, action)| (scope & context.mask() != 0).then_some(action))
    {
        return Some(action);
    }
    if matches!(key.code, Char(c) if !c.is_control()) && modifiers.is_empty() {
        return match context {
            Context::Chat => Some(Chat),
            Context::SetupEdit => Some(Setup),
            Context::HfInput => Some(Hf),
            Context::KaggleInput | Context::RunsEdit | Context::BenchmarkEdit => Some(Extras),
            Context::Palette => Some(Palette),
            Context::LimitsEdit => Some(Limits),
            Context::LibraryEdit | Context::EvalEdit => Some(Local),
            Context::NetworkEdit => Some(Network),
            _ => None,
        };
    }
    None
}

#[derive(Default)]
pub(super) struct Help {
    pub open: bool,
    pub scroll: u16,
}
impl Help {
    pub(super) fn draw(&mut self, f: &mut Frame) {
        let screen = f.area();
        let width = screen.width.min(100);
        let height = screen.height.saturating_sub(2).max(1).min(screen.height);
        let area = Rect::new(
            screen.x + (screen.width - width) / 2,
            screen.y + (screen.height - height) / 2,
            width,
            height,
        );
        f.render_widget(Clear, area);
        let block = panel(" keyboard shortcuts / all screens ");
        let inner = block.inner(area);
        let mut lines: Vec<Line<'static>> = vec![
            Line::from("Keys on the left, what they do on the right. Press F1, ? or Esc to close; PgUp/PgDn scroll."),
            Line::from("Navigation: Tab / Shift+Tab move between the five categories; 1-5 (or Alt+1-5 while typing) jump straight to one. On a category menu, arrows or j/k move, Enter opens the highlighted section, and its letter opens it directly."),
            Line::from("Esc backs out one level: close help or the palette, cancel an edit, stop chat work, cancel a running job, or leave a screen for its menu. Ctrl+K jumps to any screen."),
            Line::from(""),
        ];
        lines.extend(
            BINDINGS
                .iter()
                .map(|binding| Line::from(format!("{:<10} {}", binding.label, binding.description))),
        );
        lines.push(Line::from(""));
        lines.push(Line::from("Category menus and their section letters:"));
        lines.extend(nav::CATEGORIES.iter().enumerate().map(|(index, category)| {
            let mut text = format!("{} {}:", index + 1, category.name);
            for section in category.sections {
                text.push_str(&format!("  {} {}", section.key, section.title));
            }
            Line::from(text)
        }));
        lines.push(Line::from(
            "Other printable characters type into editors/palette, including library folders and eval prompts; limits accept digits. BPE byte groups share their minimum confidence color. Chat and A/B slash commands: /help.",
        ));
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let max = paragraph
            .line_count(inner.width)
            .saturating_sub(inner.height as usize)
            .min(u16::MAX as usize) as u16;
        self.scroll = self.scroll.min(max);
        f.render_widget(
            paragraph
                .scroll((self.scroll, 0))
                .block(block)
                .style(accent().bg(panel_bg())),
            area,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use std::collections::HashSet;

    #[test]
    fn network_keys_are_scoped_and_editors_keep_all_printable_input() {
        for tab in BACKUP_TAB..=SWEEP_TAB {
            for code in [
                Up,
                Down,
                Enter,
                Char('p'),
                Char('r'),
                Char('d'),
                Char('o'),
                Char('l'),
                Char('i'),
                Char('c'),
            ] {
                assert_eq!(
                    action(
                        KeyEvent::new(code, KeyModifiers::NONE),
                        Context::for_tab(tab, false)
                    ),
                    Some(Network)
                );
                assert!(
                    BINDINGS
                        .iter()
                        .any(|b| b.code == code && !b.description.is_empty())
                );
            }
            for c in ['q', '?', 'p', 'r', 'd', 'o', 'l', 'i', 'c'] {
                assert_eq!(
                    action(
                        KeyEvent::new(Char(c), KeyModifiers::NONE),
                        Context::for_tab(tab, true)
                    ),
                    Some(Network)
                );
            }
        }
        for code in [Left, Right, Home, End, Enter, Char('c'), Char('r')] {
            assert_eq!(
                action(KeyEvent::new(code, KeyModifiers::NONE), Context::Timeline),
                Some(Network)
            );
        }
        assert_eq!(
            action(
                KeyEvent::new(Char('l'), KeyModifiers::CONTROL),
                Context::NetworkEdit
            ),
            Some(Network)
        );
        assert_eq!(
            action(
                KeyEvent::new(Char('q'), KeyModifiers::NONE),
                Context::Network
            ),
            Some(Quit)
        );
        assert_eq!(
            action(KeyEvent::new(Esc, KeyModifiers::NONE), Context::Network),
            Some(BackToSections)
        );
        assert_eq!(
            action(KeyEvent::new(Esc, KeyModifiers::NONE), Context::Timeline),
            Some(BackToSections)
        );
        assert_eq!(
            action(KeyEvent::new(Esc, KeyModifiers::NONE), Context::NetworkEdit),
            Some(Network)
        );
        assert_eq!(
            action(
                KeyEvent::new(Char('p'), KeyModifiers::NONE),
                Context::Monitor
            ),
            None
        );
    }

    #[test]
    fn phase12_routes_preserve_editing_and_every_new_key_is_helped() {
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        for context in [Context::Monitor, Context::Chat, Context::Sample] {
            assert_eq!(action(key(F(6)), context), Some(Heatmap));
        }
        for context in [Context::SetupEdit, Context::LimitsEdit] {
            assert_eq!(action(key(F(6)), context), None);
            assert_eq!(action(key(F(9)), context), None);
        }
        assert_eq!(action(key(Char('2')), Context::LimitsEdit), Some(Limits));
        assert_eq!(action(key(Esc), Context::LimitsEdit), Some(Limits));
        assert_eq!(action(key(Char('d')), Context::Limits), Some(Limits));
        assert_eq!(action(key(Char('r')), Context::Device), Some(Device));
        assert_eq!(action(key(PageDown), Context::Math), Some(MathScroll(8)));
        for code in [F(1), F(5), F(6), F(7)] {
            assert!(
                BINDINGS
                    .iter()
                    .any(|b| b.code == code && !b.description.is_empty())
            );
        }
    }

    #[test]
    fn no_duplicate_keybindings_across_tabs_and_global_keys() {
        let mut physical_keys = HashSet::new();
        let mut labels = HashSet::new();
        for binding in BINDINGS {
            assert!(
                physical_keys.insert((
                    binding.code,
                    normalized_modifiers(KeyEvent::new(binding.code, binding.modifiers)),
                )),
                "duplicate help key: {}",
                binding.label
            );
            assert!(
                labels.insert(binding.label),
                "duplicate help label: {}",
                binding.label
            );
            let mut scopes = 0;
            for &(scope, _) in binding.routes {
                assert_ne!(scope, 0);
                assert_eq!(
                    scope & scopes,
                    0,
                    "overlapping global/tab binding: {}",
                    binding.label
                );
                scopes |= scope;
            }
        }
        // Exercise the registry used by the actual event loop in every tab/editor.
        for tab in 0..super::super::SCREENS.len() {
            for editing in [false, true] {
                let context = Context::for_tab(tab, editing);
                for binding in BINDINGS {
                    let routes: Vec<_> = binding
                        .routes
                        .iter()
                        .filter(|(scope, _)| scope & context.mask() != 0)
                        .collect();
                    assert!(routes.len() <= 1, "{} on tab {tab}", binding.label);
                    if let Some((_, expected)) = routes.first() {
                        assert_eq!(
                            action(KeyEvent::new(binding.code, binding.modifiers), context),
                            Some(*expected)
                        );
                    }
                }
                assert_eq!(
                    action(KeyEvent::new(Tab, KeyModifiers::NONE), context),
                    Some(NextTab)
                );
                assert_eq!(
                    action(KeyEvent::new(Char('c'), KeyModifiers::CONTROL), context),
                    Some(Quit)
                );
            }
        }
    }

    #[test]
    fn merged_tab_indices_contexts_and_modal_bindings_do_not_collide() {
        let tabs = [
            (0, "monitor", Context::Monitor, Context::Monitor),
            (1, "chain", Context::Dashboard, Context::Dashboard),
            (2, "model", Context::Dashboard, Context::Dashboard),
            (3, "feed", Context::Feed, Context::Feed),
            (4, "inference", Context::Chat, Context::Chat),
            (5, "setup", Context::Setup, Context::SetupEdit),
            (HF_TAB, "HF login", Context::HfInput, Context::HfInput),
            (
                KAGGLE_TAB,
                "Kaggle",
                Context::KaggleBusy,
                Context::KaggleInput,
            ),
            (MEMORY_TAB, "memory", Context::Memory, Context::Memory),
            (RUNS_TAB, "runs", Context::Runs, Context::RunsEdit),
            (
                BENCHMARK_TAB,
                "benchmark",
                Context::Benchmark,
                Context::BenchmarkEdit,
            ),
            (SAMPLE_TAB, "sample", Context::Sample, Context::Sample),
            (
                HARDWARE_TAB,
                "hardware",
                Context::Dashboard,
                Context::Dashboard,
            ),
            (MATH_TAB, "math", Context::Math, Context::Math),
            (DEVICE_TAB, "devices", Context::Device, Context::Device),
            (LIMITS_TAB, "limits", Context::Limits, Context::LimitsEdit),
            (
                LIBRARY_TAB,
                "library",
                Context::Library,
                Context::LibraryEdit,
            ),
            (MIXER_TAB, "mixer", Context::Mixer, Context::Mixer),
            (EVAL_TAB, "eval", Context::Eval, Context::EvalEdit),
            (
                BACKUP_TAB,
                "HF backup",
                Context::Network,
                Context::NetworkEdit,
            ),
            (
                LOG_STREAM_TAB,
                "cloud log",
                Context::Network,
                Context::NetworkEdit,
            ),
            (
                NOTIFY_TAB,
                "phone ping",
                Context::Network,
                Context::NetworkEdit,
            ),
            (
                UPDATE_TAB,
                "updates",
                Context::Network,
                Context::NetworkEdit,
            ),
            (
                SUPPORT_TAB,
                "support",
                Context::Network,
                Context::NetworkEdit,
            ),
            (GITHUB_TAB, "GitHub", Context::Network, Context::NetworkEdit),
            (SWEEP_TAB, "sweeps", Context::Network, Context::NetworkEdit),
            (
                TIMELINE_TAB,
                "timeline",
                Context::Timeline,
                Context::Timeline,
            ),
        ];
        let mut indices = HashSet::new();
        let mut contexts = vec![Context::Help, Context::Palette, Context::Menu];
        for (index, label, browse, edit) in tabs {
            assert!(indices.insert(index), "duplicate tab index: {label}");
            assert_eq!(SCREENS[index], label);
            assert_eq!(Context::for_tab(index, false), browse, "{label}");
            assert_eq!(Context::for_tab(index, true), edit, "{label} editor");
            contexts.extend([browse, edit]);
        }
        assert_eq!(indices.len(), SCREENS.len());
        // Every context retains a distinct bit, including all local/network
        // scopes beyond the tabs plus palette/help scopes.
        assert_eq!(
            contexts
                .iter()
                .fold(0, |mask, context| mask | context.mask()),
            ALL
        );
        assert_eq!(ALL.count_ones(), 30);
        for context in contexts {
            for binding in BINDINGS {
                let matching: Vec<_> = BINDINGS
                    .iter()
                    .filter(|other| {
                        other.code == binding.code && other.modifiers == binding.modifiers
                    })
                    .flat_map(|other| other.routes)
                    .filter(|(scope, _)| scope & context.mask() != 0)
                    .collect();
                assert!(
                    matching.len() <= 1,
                    "duplicate {} in {context:?}",
                    binding.label
                );
                if let Some((_, expected)) = matching.first() {
                    assert_eq!(
                        action(KeyEvent::new(binding.code, binding.modifiers), context),
                        Some(*expected)
                    );
                }
            }
            for (code, modifiers, expected) in [
                (Tab, KeyModifiers::NONE, NextTab),
                (F(1), KeyModifiers::NONE, ToggleHelp),
                (Char('c'), KeyModifiers::CONTROL, Quit),
                (Char('k'), KeyModifiers::CONTROL, TogglePalette),
                (Char('2'), KeyModifiers::ALT, OpenCategory(1)),
            ] {
                assert_eq!(
                    action(KeyEvent::new(code, modifiers), context),
                    Some(expected),
                    "{context:?}: {code:?}"
                );
            }
        }
    }

    #[test]
    fn phase_ten_eleven_controls_are_registered_without_global_shadowing() {
        for (context, codes, expected) in [
            (Context::HfInput, vec![Enter, Esc, Backspace], Hf),
            (Context::KaggleInput, vec![Enter, Esc, Backspace], Extras),
            (
                Context::KaggleBusy,
                vec![Char('y'), Char('Y'), Char('n'), Char('N'), Esc],
                Extras,
            ),
            (
                Context::Memory,
                vec![Char('r'), Char('v'), PageUp, PageDown],
                Extras,
            ),
            (
                Context::Runs,
                vec![Up, Down, Enter, Char('c'), Char('s'), Char('r'), Esc],
                Extras,
            ),
            (Context::RunsEdit, vec![Enter, Esc, Backspace], Extras),
            (
                Context::Benchmark,
                vec![Up, Down, Enter, Char('b'), Esc],
                Extras,
            ),
            (Context::BenchmarkEdit, vec![Enter, Esc, Backspace], Extras),
            (
                Context::Palette,
                vec![Up, Down, Enter, Esc, Backspace],
                Palette,
            ),
        ] {
            for code in codes {
                assert_eq!(
                    action(KeyEvent::new(code, KeyModifiers::NONE), context),
                    Some(expected),
                    "{context:?}: {code:?}"
                );
            }
            for (code, modifiers, expected) in [
                (Tab, KeyModifiers::NONE, NextTab),
                (F(1), KeyModifiers::NONE, ToggleHelp),
                (Char('k'), KeyModifiers::CONTROL, TogglePalette),
                (Char('c'), KeyModifiers::CONTROL, Quit),
            ] {
                assert_eq!(
                    action(KeyEvent::new(code, modifiers), context),
                    Some(expected),
                    "{context:?}"
                );
            }
        }
        assert_eq!(
            action(
                KeyEvent::new(Char('l'), KeyModifiers::CONTROL),
                Context::HfInput
            ),
            Some(Hf)
        );
        for context in [
            Context::KaggleInput,
            Context::RunsEdit,
            Context::BenchmarkEdit,
        ] {
            assert_eq!(
                action(KeyEvent::new(Char('u'), KeyModifiers::CONTROL), context),
                Some(Extras)
            );
        }
    }

    #[test]
    fn editors_keep_text_and_scoped_shortcuts_do_not_leak() {
        for (context, expected) in [
            (Context::Chat, Chat),
            (Context::SetupEdit, Setup),
            (Context::HfInput, Hf),
            (Context::KaggleInput, Extras),
            (Context::RunsEdit, Extras),
            (Context::BenchmarkEdit, Extras),
            (Context::Palette, Palette),
            (Context::LimitsEdit, Limits),
            (Context::LibraryEdit, Local),
            (Context::EvalEdit, Local),
        ] {
            for c in ['?', 'q', 'c', 'g', '1', '+', '-'] {
                assert_eq!(
                    action(KeyEvent::new(Char(c), KeyModifiers::NONE), context),
                    Some(expected)
                );
            }
            assert_eq!(
                action(KeyEvent::new(F(1), KeyModifiers::NONE), context),
                Some(ToggleHelp)
            );
        }
        assert_eq!(
            action(KeyEvent::new(Left, KeyModifiers::NONE), Context::Monitor),
            Some(Pan(false))
        );
        assert_eq!(
            action(KeyEvent::new(Left, KeyModifiers::NONE), Context::Setup),
            Some(Setup)
        );
        assert_eq!(
            action(
                KeyEvent::new(Char('?'), KeyModifiers::SHIFT),
                Context::Setup
            ),
            Some(ToggleHelp)
        );
        assert_eq!(
            action(KeyEvent::new(BackTab, KeyModifiers::SHIFT), Context::Setup),
            Some(PreviousTab)
        );
        for context in [Context::Monitor, Context::Dashboard, Context::Setup] {
            assert_eq!(
                action(KeyEvent::new(Char('q'), KeyModifiers::CONTROL), context),
                None
            );
            assert_eq!(
                action(KeyEvent::new(Char('c'), KeyModifiers::ALT), context),
                None
            );
        }
    }

    #[test]
    fn help_is_modal_and_keeps_global_controls_available() {
        for (code, expected) in [
            (Esc, ToggleHelp),
            (Char('?'), ToggleHelp),
            (Char('q'), ToggleHelp),
            (F(1), ToggleHelp),
            (Tab, NextTab),
            (Up, ScrollHelp(-1)),
            (Down, ScrollHelp(1)),
            (PageUp, ScrollHelp(-8)),
            (PageDown, ScrollHelp(8)),
            (Home, HelpTop),
            (End, HelpBottom),
        ] {
            assert_eq!(
                action(KeyEvent::new(code, KeyModifiers::NONE), Context::Help),
                Some(expected)
            );
        }
        assert_eq!(
            action(
                KeyEvent::new(Char('c'), KeyModifiers::CONTROL),
                Context::Help
            ),
            Some(Quit)
        );
        // Help must not launch training, edit input, change pages, or pan graphs.
        for code in [
            Enter,
            Char(' '),
            Char('c'),
            Char('g'),
            Char('1'),
            Left,
            Right,
            Backspace,
        ] {
            assert_eq!(
                action(KeyEvent::new(code, KeyModifiers::NONE), Context::Help),
                None
            );
        }
    }

    #[test]
    fn help_renders_each_key_once_and_scrolls_on_narrow_terminals() {
        let mut help = Help::default();
        let mut terminal = Terminal::new(TestBackend::new(120, 120)).unwrap();
        terminal.draw(|f| help.draw(f)).unwrap();
        let buffer = terminal.backend().buffer();
        let rows: Vec<String> = buffer
            .content()
            .chunks(120)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();
        for binding in BINDINGS {
            let label = format!("│{:<10} ", binding.label);
            assert_eq!(
                rows.iter().filter(|row| row.contains(&label)).count(),
                1,
                "{}",
                binding.label
            );
        }
        for (width, height) in [(80, 24), (79, 24), (30, 10), (10, 5), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            help.scroll = u16::MAX;
            terminal.draw(|f| help.draw(f)).unwrap();
            assert!(help.scroll < u16::MAX);
            if width >= 30 {
                let text: String = terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .map(|c| c.symbol())
                    .collect();
                assert!(
                    text.contains("/help."),
                    "bottom of help must remain reachable at {width}"
                );
            }
        }
    }

    #[test]
    fn one_key_grammar_moves_opens_and_jumps_categories() {
        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        // Menu: arrows or j/k move, Enter opens, Esc backs out, Tab switches.
        assert_eq!(action(key(Tab), Context::Menu), Some(NextTab));
        assert_eq!(action(key(BackTab), Context::Menu), Some(PreviousTab));
        assert_eq!(action(key(Up), Context::Menu), Some(MenuMove(-1)));
        assert_eq!(action(key(Down), Context::Menu), Some(MenuMove(1)));
        assert_eq!(action(key(Char('k')), Context::Menu), Some(MenuMove(-1)));
        assert_eq!(action(key(Char('j')), Context::Menu), Some(MenuMove(1)));
        assert_eq!(action(key(Enter), Context::Menu), Some(MenuOpen));
        assert_eq!(action(key(Esc), Context::Menu), Some(BackToSections));
        assert_eq!(action(key(Char('?')), Context::Menu), Some(ToggleHelp));
        assert_eq!(action(key(Char('q')), Context::Menu), Some(Quit));

        // 1-5 pick a category on any browse screen. Alt+1-5 also work while
        // typing, which keeps two-key reachability from chat and editors.
        for (index, digit) in ['1', '2', '3', '4', '5'].into_iter().enumerate() {
            assert_eq!(
                action(key(Char(digit)), Context::Menu),
                Some(OpenCategory(index))
            );
            for context in [
                Context::Monitor,
                Context::Runs,
                Context::KaggleBusy,
                Context::Timeline,
            ] {
                assert_eq!(
                    action(key(Char(digit)), context),
                    Some(OpenCategory(index)),
                    "{context:?}"
                );
            }
            for context in [Context::Chat, Context::SetupEdit, Context::Palette] {
                assert_eq!(
                    action(KeyEvent::new(Char(digit), KeyModifiers::ALT), context),
                    Some(OpenCategory(index)),
                    "{context:?}"
                );
            }
            // Plain digits still type inside editors and chat.
            assert!(matches!(
                action(key(Char(digit)), Context::Chat),
                Some(Chat)
            ));
            assert!(matches!(
                action(key(Char(digit)), Context::SetupEdit),
                Some(Setup)
            ));
        }

        // j/k mirror the arrows outside text editors.
        assert_eq!(action(key(Char('j')), Context::Feed), Some(FeedScroll(1)));
        assert_eq!(action(key(Char('k')), Context::Feed), Some(FeedScroll(-1)));
        assert_eq!(action(key(Char('j')), Context::Runs), Some(Extras));
        assert_eq!(action(key(Char('j')), Context::Help), Some(ScrollHelp(1)));
        assert_eq!(action(key(Char('j')), Context::Chat), Some(Chat));
        assert_eq!(action(key(Char('j')), Context::Palette), Some(Palette));

        // Redundant direct jumps are gone; g still reaches every graph view.
        for code in [F(8), F(9), F(10), F(11), F(12), Char('6'), Char('7')] {
            assert_eq!(action(key(code), Context::Monitor), None, "{code:?}");
        }
        assert_eq!(action(key(Char('g')), Context::Monitor), Some(CycleGraph));
        // Left/Right no longer cycle categories on list screens.
        assert_eq!(action(key(Left), Context::Runs), None);
        assert_eq!(action(key(Right), Context::Runs), None);
    }
}
