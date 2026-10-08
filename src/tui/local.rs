//! Thin integration for the local-model/data screens, kept out of the app shell.
use super::{
    RunState, feature_area as content_area,
    chat::Chat,
    eval::Eval,
    keybindings::{EVAL_TAB, LIBRARY_TAB, MIXER_TAB},
    library::{Library, Pick},
    mixer::Mixer,
    setup::Setup,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{Frame, layout::Rect};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub(super) struct Local {
    pub library: Library,
    pub mixer: Mixer,
    eval: Eval,
    config_error: String,
}
impl Local {
    pub(super) fn new(chain: PathBuf) -> Self {
        let library = Library::new(chain);
        let mut eval = Eval::new();
        if !library.config.eval_prompts.as_os_str().is_empty() {
            eval.set_prompt_file(Some(library.config.eval_prompts.clone()));
        }
        Self {
            library,
            mixer: Mixer::default(),
            eval,
            config_error: String::new(),
        }
    }
    pub(super) fn editing(&self, tab: usize) -> bool {
        (tab == LIBRARY_TAB && self.library.editing()) || (tab == EVAL_TAB && self.eval.editing())
    }
    pub(super) fn poll(&mut self, setup: &mut Setup, state: &RunState, remote: bool) {
        let run_dir = (!remote && (state.training_active || state.checkpoint_target.is_some()
            || state.last_checkpoint.is_some())).then(|| state.chain_dir.clone());
        self.library.watch_run(run_dir, state.checkpoint_revision);
        self.library.poll();
        self.mixer.sync(
            &self.library.entries,
            self.library.revision,
            &self.library.config.datasets,
        );
        if let Some(path) = self.mixer.poll() {
            setup.set_dataset(path);
        }
        let (dir, latest) = eval_target(state, &self.library.config.models, remote);
        self.eval.set_run_context(if remote {
            "Remote checkpoints are not local files; sync a saved checkpoint to evaluate it.".into()
        } else {
            state.checkpoint_context()
        });
        self.eval.watch(dir, latest);
        self.eval.poll();
    }
    pub(super) fn key(
        &mut self,
        tab: &mut usize,
        key: KeyEvent,
        chat: &mut Chat,
        setup: &mut Setup,
    ) {
        match *tab {
            LIBRARY_TAB => match self.library.key(key) {
                Some(Pick::Chat(path)) => {
                    let draft =
                        std::mem::replace(&mut chat.input, format!("/model {}", path.display()));
                    chat.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                    chat.input = draft;
                    *tab = 4;
                }
                Some(Pick::Resume(path)) => {
                    setup.set_resume(path);
                    *tab = 5;
                }
                Some(Pick::Train(path)) => {
                    if let Some(path) = self
                        .mixer
                        .prepare_dataset(path, &self.library.config.datasets)
                    {
                        setup.set_dataset(path);
                        *tab = 5;
                    } else {
                        *tab = MIXER_TAB;
                    }
                }
                None => {}
            },
            MIXER_TAB => self.mixer.key(key),
            EVAL_TAB => {
                let before = self.eval.prompt_file().map(Path::to_path_buf);
                self.eval.key(key);
                if before.as_deref() != self.eval.prompt_file() {
                    self.library.config.eval_prompts = self
                        .eval
                        .prompt_file()
                        .map(Path::to_path_buf)
                        .unwrap_or_default();
                    self.config_error = self.library.config.save().err().unwrap_or_default();
                }
            }
            _ => {}
        }
    }
    pub(super) fn draw(&mut self, f: &mut Frame, tab: usize) {
        if !matches!(tab, LIBRARY_TAB | MIXER_TAB | EVAL_TAB) {
            return;
        }
        let area = content_area(f.area());
        match tab {
            LIBRARY_TAB => self.library.draw(f, area),
            MIXER_TAB => self.mixer.draw(f, area),
            EVAL_TAB => self.eval.draw(f, area),
            _ => {}
        }
        if !self.config_error.is_empty() && area.height > 0 {
            f.render_widget(
                ratatui::widgets::Paragraph::new(format!(
                    "Config not saved: {}",
                    self.config_error
                )),
                Rect::new(area.x, area.bottom() - 1, area.width, 1),
            );
        }
    }
}

// Kaggle log paths are remote, even when a same-named local file exists.
// Only the explicitly configured local library remains eligible in that mode.
fn eval_target<'a>(state: &'a RunState, models: &'a Path, remote: bool) -> (&'a Path, Option<&'a Path>) {
    if !remote && (state.training_active || state.last_checkpoint.is_some() || state.checkpoint_target.is_some()) {
        (&state.chain_dir, state.last_checkpoint.as_deref().map(Path::new))
    } else {
        (models, None)
    }
}

/// Populate existing wizard fields from a bounded header read. Labels, rather
/// than field indexes, keep this additive when other phases add wizard fields.
pub(super) fn resume_hints(path: &Path) -> Vec<(&'static str, String)> {
    let mut bytes = Vec::new();
    if File::open(path)
        .and_then(|f| f.take(106).read_to_end(&mut bytes))
        .is_err()
        || bytes.len() < 62
    {
        return Vec::new();
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    let word = |at| {
        bytes
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    let fields: &[(&str, usize)] = match (&bytes[..4], version) {
        (b"PSSA", 6..=8) => &[
            ("Vocab ceiling", 22),
            ("Latent", 30),
            ("State", 38),
            ("Chunk", 62),
        ],
        (b"TRFM", 1) => &[("Vocab ceiling", 22), ("Chunk", 54)],
        _ => return Vec::new(),
    };
    let mut result: Vec<_> = fields
        .iter()
        .filter_map(|&(label, at)| {
            word(at)
                .filter(|n| *n > 0 && *n <= 65536)
                .map(|n| (label, if label == "Vocab ceiling" { n.max(257) } else { n }.to_string()))
        })
        .collect();
    if &bytes[..4] == b"PSSA" {
        result.push((
            "Depth",
            if version == 8 {
                word(98).unwrap_or(1).clamp(1, 32)
            } else {
                1
            }
            .to_string(),
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_screen_body_matches_shell_wide_and_narrow() {
        for (w, h) in [(120, 40), (79, 24), (40, 12), (20, 5), (0, 0)] {
            let body = content_area(Rect::new(0, 0, w, h));
            assert!(body.right() <= w && body.bottom() <= h);
        }
    }
    #[test]
    fn resume_hints_use_only_header_and_keep_paths_with_spaces() {
        let temp = super::super::library::tests::Temp::new();
        let path = temp.0.join("resume name.pssa");
        let mut bytes = b"PSSA".to_vec();
        bytes.extend(8u16.to_le_bytes());
        bytes.resize(22, 0);
        for n in [2048u64, 32, 8, 4, 16, 64] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.resize(98, 0);
        bytes.extend(3u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let hints = resume_hints(&path);
        assert!(hints.contains(&("Latent", "32".into())));
        assert!(hints.contains(&("Chunk", "64".into())));
        assert!(hints.contains(&("Depth", "3".into())));
    }
}
