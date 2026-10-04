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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Context {
    Monitor,
    Dashboard,
    Chat,
    Setup,
    SetupEdit,
    Help,
}
impl Context {
    fn mask(self) -> u8 {
        1 << self as u8
    }

    pub(super) fn for_tab(tab: usize, editing: bool) -> Self {
        match tab {
            0 => Self::Monitor,
            4 => Self::Chat,
            5 if editing => Self::SetupEdit,
            5 => Self::Setup,
            _ => Self::Dashboard,
        }
    }
}

const MONITOR: u8 = 1 << Context::Monitor as u8;
const DASHBOARD: u8 = 1 << Context::Dashboard as u8;
const CHAT: u8 = 1 << Context::Chat as u8;
const SETUP: u8 = 1 << Context::Setup as u8;
const EDIT: u8 = 1 << Context::SetupEdit as u8;
const HELP: u8 = 1 << Context::Help as u8;
const BROWSE: u8 = MONITOR | DASHBOARD | SETUP;
const ALL: u8 = BROWSE | CHAT | EDIT | HELP;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Quit,
    NextTab,
    PreviousTab,
    ToggleHelp,
    Chat,
    Setup,
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
    routes: &'static [(u8, Action)],
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
    bind!(F(1), "F1", "Open/close help, including inside text input", ALL => ToggleHelp),
    bind!(Char('?'), "?", "Open/close help outside text input; type normally in editors", BROWSE | HELP => ToggleHelp),
    bind!(Char('q'), "q", "Quit outside text input; close help", BROWSE => Quit, HELP => ToggleHelp),
    bind!(Esc, "Esc", "Quit dashboard/setup; cancel edit; stop chat; close help",
        BROWSE => Quit, CHAT => Chat, EDIT => Setup, HELP => ToggleHelp),
    bind!(Left, "Left", "Monitor: pan older; dashboard: previous tab; setup: previous page",
        MONITOR => Pan(false), DASHBOARD => PreviousTab, SETUP => Setup),
    bind!(Right, "Right", "Monitor: pan newer; dashboard: next tab; setup: next page",
        MONITOR => Pan(true), DASHBOARD => NextTab, SETUP => Setup),
    bind!(Char('g'), "g", "Monitor: cycle graph view", MONITOR => CycleGraph),
    bind!(Char('1'), "1", "Monitor: loss", MONITOR => Graph(GraphView::Loss)),
    bind!(Char('2'), "2", "Monitor: perplexity", MONITOR => Graph(GraphView::Perplexity)),
    bind!(Char('3'), "3", "Monitor: tokens per second", MONITOR => Graph(GraphView::TokensPerSecond)),
    bind!(Char('4'), "4", "Monitor: learning rate", MONITOR => Graph(GraphView::LearningRate)),
    bind!(Char('5'), "5", "Monitor: comparison", MONITOR => Graph(GraphView::Comparison)),
    bind!(Char('6'), "6", "Monitor: all metrics", MONITOR => Graph(GraphView::All)),
    bind!(Char('7'), "7", "Monitor: memory", MONITOR => Graph(GraphView::Memory)),
    bind!(Char('+'), "+", "Monitor: zoom in; setup: increase selected depth/loops", MONITOR => Zoom(true), SETUP => Setup),
    bind!(Char('='), "=", "Monitor: zoom in", MONITOR => Zoom(true)),
    bind!(Char('-'), "-", "Monitor: zoom out; setup: decrease selected depth/loops", MONITOR => Zoom(false), SETUP => Setup),
    bind!(Char('0'), "0", "Monitor: reset graph navigation", MONITOR => ResetGraph),
    bind!(Up, "Up", "Setup: previous field / preview scroll up; help: scroll up", SETUP => Setup, HELP => ScrollHelp(-1)),
    bind!(Down, "Down", "Setup: next field / preview scroll down; help: scroll down", SETUP => Setup, HELP => ScrollHelp(1)),
    bind!(BackTab, "Shift+Tab", "Setup: previous wizard page", SETUP => Setup),
    bind!(F(5), "F5", "Setup: next wizard page", SETUP => Setup),
    bind!(Char('c'), "c", "Setup: toggle full command preview", SETUP => Setup),
    bind!(PageUp, "PgUp", "Chat, setup command preview, help: scroll up", CHAT => Chat, SETUP => Setup, HELP => ScrollHelp(-8)),
    bind!(PageDown, "PgDn", "Chat, setup command preview, help: scroll down", CHAT => Chat, SETUP => Setup, HELP => ScrollHelp(8)),
    bind!(Home, "Home", "Help: first line", HELP => HelpTop),
    bind!(End, "End", "Chat: follow latest output; help: last line", CHAT => Chat, HELP => HelpBottom),
    bind!(Enter, "Enter", "Chat: send/command; setup: activate/edit/save (launch only on start button)", CHAT => Chat, SETUP | EDIT => Setup),
    bind!(Char(' '), "Space", "Setup: activate/edit selected field or button; editors: type a space", SETUP => Setup),
    bind!(Backspace, "Backspace", "Chat/setup editor: delete last character", CHAT => Chat, EDIT => Setup),
    Binding {
        code: Char('u'),
        modifiers: KeyModifiers::CONTROL,
        label: "Ctrl+U",
        description: "Chat/setup editor: clear input",
        routes: &[(CHAT, Chat), (EDIT, Setup)],
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
            "Other printable characters type into chat/setup editors. Chat slash commands: /help.",
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
    fn editors_keep_text_and_scoped_shortcuts_do_not_leak() {
        for (context, expected) in [(Context::Chat, Chat), (Context::SetupEdit, Setup)] {
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
        let mut terminal = Terminal::new(TestBackend::new(120, 80)).unwrap();
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
