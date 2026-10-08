//! One registry for shortcut dispatch and the help overlay. Reused keys are
//! explicitly scoped; printable input is a fallback, never a global shortcut.
use super::{GraphView, PANEL_BG, accent, panel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    text::Line,
    widgets::{Clear, Paragraph, Wrap},
};

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
pub(super) const TABS: [&str; 27] = [
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
    | NETWORK_EDIT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Quit,
    NextTab,
    PreviousTab,
    ToggleHelp,
    TogglePalette,
    Palette,
    Chat,
    Setup,
    Hf,
    Extras,
    Network,
    OpenTab(usize),
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
    Graph(GraphView),
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
        description: "Quit TUI; wizard-launched training stays running",
        routes: &[(ALL, Quit)],
    },
    bind!(Tab, "Tab", "Next application tab (also while editing)", ALL => NextTab),
    bind!(F(6), "F6", "Chat / monitor / sample: toggle raw token confidence heatmap", CHAT | MONITOR | SAMPLE => Heatmap),
    bind!(F(7), "F7", "Monitor / sample: pause or resume live checkpoint preview", MONITOR | SAMPLE => Preview),
    bind!(F(8), "F8", "Open hardware telemetry", ALL => OpenTab(HARDWARE_TAB)),
    bind!(F(9), "F9", "Open training device picker", ALL => OpenTab(DEVICE_TAB)),
    bind!(F(10), "F10", "Open resource limits for the next run", ALL => OpenTab(LIMITS_TAB)),
    bind!(F(11), "F11", "Open live PSSA math reference", ALL => OpenTab(MATH_TAB)),
    bind!(F(12), "F12", "Open full checkpoint sample", ALL => OpenTab(SAMPLE_TAB)),
    bind!(F(1), "F1", "Open/close help, including inside text input", ALL => ToggleHelp),
    bind!(Char('?'), "?", "Open/close help outside text input; type normally in editors", BROWSE | HELP => ToggleHelp),
    bind!(Char('q'), "q", "Quit outside text input; close help", BROWSE => Quit, HELP => ToggleHelp),
    bind!(Esc, "Esc", "Dashboard/setup/local/network: quit; editors: cancel/clear; chat/extras: stop (Kaggle detaches); overlays: close",
        MONITOR | PAGES | SETUP | LOCAL_BROWSE => Quit, CHAT => Chat, EDIT => Setup, HF => Hf, EXTRA => Extras, LIMIT_EDIT => Limits, LOCAL_EDIT => Local, NETWORK => Quit, NETWORK_EDIT => Network, TIMELINE => Quit, PALETTE => Palette, HELP => ToggleHelp),
    bind!(Left, "Left", "Monitor: pan older; timeline: older checkpoint; other browse tabs: previous tab; setup: previous page",
        MONITOR => Pan(false), PAGES | EXTRA_BROWSE | LOCAL_BROWSE | NETWORK => PreviousTab, SETUP => Setup, TIMELINE => Network),
    bind!(Right, "Right", "Monitor: pan newer; timeline: newer checkpoint; other browse tabs: next tab; setup: next page",
        MONITOR => Pan(true), PAGES | EXTRA_BROWSE | LOCAL_BROWSE | NETWORK => NextTab, SETUP => Setup, TIMELINE => Network),
    bind!(Char('g'), "g", "Monitor: cycle graph view", MONITOR => CycleGraph),
    bind!(Char('1'), "1", "Monitor: loss", MONITOR => Graph(GraphView::Loss)),
    bind!(Char('2'), "2", "Monitor: perplexity", MONITOR => Graph(GraphView::Perplexity)),
    bind!(Char('3'), "3", "Monitor: tokens per second", MONITOR => Graph(GraphView::TokensPerSecond)),
    bind!(Char('4'), "4", "Monitor: learning rate", MONITOR => Graph(GraphView::LearningRate)),
    bind!(Char('5'), "5", "Monitor: comparison", MONITOR => Graph(GraphView::Comparison)),
    bind!(Char('6'), "6", "Monitor: all metrics", MONITOR => Graph(GraphView::All)),
    bind!(Char('7'), "7", "Monitor: memory", MONITOR => Graph(GraphView::Memory)),
    bind!(Char('+'), "+", "Monitor: zoom in; setup: increase depth/loops; mixer: increase share", MONITOR => Zoom(true), SETUP => Setup, MIXER => Local),
    bind!(Char('='), "=", "Monitor: zoom in; mixer: increase share", MONITOR => Zoom(true), MIXER => Local),
    bind!(Char('-'), "-", "Monitor: zoom out; setup: decrease depth/loops; mixer: decrease share", MONITOR => Zoom(false), SETUP => Setup, MIXER => Local),
    bind!(Char('0'), "0", "Monitor: reset graph navigation", MONITOR => ResetGraph),
    bind!(Up, "Up", "Setup/runs/benchmark/palette/devices/limits/local/network: previous item; feed/math/help: scroll up", FEED => FeedScroll(-1), SETUP => Setup, RUNS | BENCHMARK => Extras, PALETTE => Palette, DEVICE => Device, LIMITS => Limits, LOCAL_BROWSE => Local, NETWORK | TIMELINE => Network, MATH => MathScroll(-1), HELP => ScrollHelp(-1)),
    bind!(Down, "Down", "Setup/runs/benchmark/palette/devices/limits/local/network: next item; feed/math/help: scroll down", FEED => FeedScroll(1), SETUP => Setup, RUNS | BENCHMARK => Extras, PALETTE => Palette, DEVICE => Device, LIMITS => Limits, LOCAL_BROWSE => Local, NETWORK | TIMELINE => Network, MATH => MathScroll(1), HELP => ScrollHelp(1)),
    bind!(BackTab, "Shift+Tab", "Setup: previous wizard page", SETUP => Setup),
    bind!(F(5), "F5", "Setup: next wizard page", SETUP => Setup),
    bind!(Char('c'), "c", "Setup: command preview; runs/library: chat with checkpoint; network/timeline: open or copy", SETUP => Setup, RUNS => Extras, LIBRARY => Local, NETWORK | TIMELINE => Network),
    bind!(Char('r'), "r", "Memory/runs/devices/library/eval/network/timeline: refresh or retry", MEMORY | RUNS => Extras, DEVICE => Device, LIBRARY | EVAL => Local, NETWORK | TIMELINE => Network),
    bind!(Char('v'), "v", "Memory: switch checkpoint inspector / live chat retrieval", MEMORY => Extras),
    bind!(Char('p'), "p", "Eval: edit prompt; HF backup/cloud log/phone ping/sweeps/GitHub: act", EVAL => Local, NETWORK => Network),
    bind!(Char('i'), "i", "GitHub: open issue list", NETWORK => Network),
    bind!(Char('l'), "l", "GitHub: enter masked token login", NETWORK => Network),
    bind!(Char('o'), "o", "Support/GitHub/updates: open selected link in browser", NETWORK => Network),
    bind!(Char('s'), "s", "Runs: score checkpoint on a held-out file", RUNS => Extras),
    bind!(Char('b'), "b", "Benchmark: run configured matched comparison", BENCHMARK => Extras),
    bind!(Char('y'), "y", "Kaggle preview: confirm upload and launch", KAGGLE_BUSY => Extras),
    bind!(Char('Y'), "Y", "Kaggle preview: confirm upload and launch (uppercase)", KAGGLE_BUSY => Extras),
    bind!(Char('n'), "n", "Kaggle preview: cancel launch", KAGGLE_BUSY => Extras),
    bind!(Char('N'), "N", "Kaggle preview: cancel launch (uppercase)", KAGGLE_BUSY => Extras),
    bind!(Char('m'), "m", "Library: edit models folder", LIBRARY => Local),
    bind!(Char('d'), "d", "Limits: reset draft; library: edit datasets folder; network: toggle/stop/cancel", LIMITS => Limits, LIBRARY => Local, NETWORK => Network),
    bind!(Char('a'), "a", "Eval: pause/resume background checkpoint evaluation", EVAL => Local),
    bind!(Char('u'), "u", "Library: fill Setup Resume from selected checkpoint", LIBRARY => Local),
    bind!(Char('t'), "t", "Library: fill Setup Dataset (converts JSONL/Parquet off-thread)", LIBRARY => Local),
    bind!(PageUp, "PgUp", "Chat/setup preview/memory/local/network/feed/math/help: scroll up", FEED => FeedScroll(-8), CHAT => Chat, SETUP => Setup, MEMORY => Extras, EVAL => Local, NETWORK => Network, MATH => MathScroll(-8), HELP => ScrollHelp(-8)),
    bind!(PageDown, "PgDn", "Chat/setup preview/memory/local/network/feed/math/help: scroll down", FEED => FeedScroll(8), CHAT => Chat, SETUP => Setup, MEMORY => Extras, EVAL => Local, NETWORK => Network, MATH => MathScroll(8), HELP => ScrollHelp(8)),
    bind!(Home, "Home", "Feed/math/help: first line; devices/limits/network: first field; timeline: first checkpoint", FEED => FeedTop, HELP => HelpTop, MATH => MathTop, DEVICE => Device, LIMITS => Limits, NETWORK | TIMELINE => Network),
    bind!(End, "End", "Chat: follow; feed/math/help: last line; devices/limits/network: last field", FEED => FeedBottom, CHAT => Chat, HELP => HelpBottom, MATH => MathBottom, DEVICE => Device, LIMITS => Limits, NETWORK | TIMELINE => Network),
    bind!(Enter, "Enter", "Chat: send; setup: edit/save/start; HF: login; Kaggle: review; runs: monitor/score; benchmark/local/network: edit/save/open; palette: execute; devices/limits: select/edit/apply; timeline: prepare resume", CHAT => Chat, SETUP | EDIT => Setup, HF => Hf, KAGGLE_INPUT | RUNS | RUNS_EDIT | BENCHMARK | BENCHMARK_EDIT => Extras, PALETTE => Palette, DEVICE => Device, LIMITS | LIMIT_EDIT => Limits, LOCAL_BROWSE | LOCAL_EDIT => Local, NETWORK | NETWORK_EDIT | TIMELINE => Network),
    bind!(Char(' '), "Space", "Setup/devices/limits: activate selected field/button; text editors: type space", SETUP => Setup, DEVICE => Device, LIMITS => Limits),
    bind!(Backspace, "Backspace", "Text editors/palette", CHAT => Chat, EDIT => Setup, HF => Hf, EXTRA_EDIT => Extras, PALETTE => Palette, LIMIT_EDIT => Limits, LOCAL_EDIT => Local, NETWORK_EDIT => Network),
    Binding {
        code: Char('u'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+U",
        description: "Chat/setup/Kaggle/score/benchmark/limits/local/network editor: clear input",
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
        description: "HF/GitHub login: log out and remove saved token",
        routes: &[(HF, Hf), (NETWORK | NETWORK_EDIT, Network)],
    },
    Binding {
        code: Char('k'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+K",
        description: "Open/close command palette (all tabs and editors)",
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

pub(super) fn action(key: KeyEvent, context: Context) -> Option<Action> {
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
        let block = panel(" controls / all tabs ");
        let inner = block.inner(area);
        let mut lines: Vec<Line<'static>> = BINDINGS
            .iter()
            .map(|binding| Line::from(format!("{:<10} {}", binding.label, binding.description)))
            .collect();
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
                .style(accent().bg(PANEL_BG)),
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
            Some(Quit)
        );
        assert_eq!(
            action(KeyEvent::new(Esc, KeyModifiers::NONE), Context::Timeline),
            Some(Quit)
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
            assert_eq!(action(key(F(9)), context), Some(OpenTab(DEVICE_TAB)));
        }
        assert_eq!(action(key(Char('2')), Context::LimitsEdit), Some(Limits));
        assert_eq!(action(key(Esc), Context::LimitsEdit), Some(Limits));
        assert_eq!(action(key(Char('d')), Context::Limits), Some(Limits));
        assert_eq!(action(key(Char('r')), Context::Device), Some(Device));
        assert_eq!(action(key(PageDown), Context::Math), Some(MathScroll(8)));
        for code in [F(6), F(7), F(8), F(9), F(10), F(11), F(12)] {
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
        for tab in 0..super::super::TABS.len() {
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
        let mut contexts = vec![Context::Help, Context::Palette];
        for (index, label, browse, edit) in tabs {
            assert!(indices.insert(index), "duplicate tab index: {label}");
            assert_eq!(TABS[index], label);
            assert_eq!(Context::for_tab(index, false), browse, "{label}");
            assert_eq!(Context::for_tab(index, true), edit, "{label} editor");
            contexts.extend([browse, edit]);
        }
        assert_eq!(indices.len(), TABS.len());
        // Every context retains a distinct bit, including all local/network
        // scopes beyond the tabs plus palette/help scopes.
        assert_eq!(
            contexts
                .iter()
                .fold(0, |mask, context| mask | context.mask()),
            ALL
        );
        assert_eq!(ALL.count_ones(), 29);
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
                (F(8), KeyModifiers::NONE, OpenTab(HARDWARE_TAB)),
                (F(9), KeyModifiers::NONE, OpenTab(DEVICE_TAB)),
                (F(10), KeyModifiers::NONE, OpenTab(LIMITS_TAB)),
                (F(11), KeyModifiers::NONE, OpenTab(MATH_TAB)),
                (F(12), KeyModifiers::NONE, OpenTab(SAMPLE_TAB)),
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
            Some(Setup)
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
}
