//! Category and section navigation. Five category tabs hold the existing
//! screens as sections, so the top strip stays short while every screen keeps
//! its own state and renderer. A category digit plus a section letter opens
//! any screen from any other in two keystrokes.

use super::keybindings::{
    BACKUP_TAB, BENCHMARK_TAB, CHAIN_TAB, DEVICE_TAB, EVAL_TAB, FEED_TAB, GITHUB_TAB,
    HARDWARE_TAB, HF_TAB, INFERENCE_TAB, KAGGLE_TAB, LIBRARY_TAB, LIMITS_TAB, LOG_STREAM_TAB,
    MATH_TAB, MEMORY_TAB, MIXER_TAB, MODEL_TAB, MONITOR_TAB, NOTIFY_TAB, RUNS_TAB, SAMPLE_TAB,
    SETUP_TAB, SUPPORT_TAB, SWEEP_TAB, TIMELINE_TAB, UPDATE_TAB,
};

pub(super) struct Section {
    pub tab: usize,
    pub title: &'static str,
    /// Letter that opens this section from the category menu.
    pub key: char,
    pub blurb: &'static str,
}

pub(super) struct Category {
    pub name: &'static str,
    pub blurb: &'static str,
    pub sections: &'static [Section],
}

pub(super) const CATEGORIES: [Category; 5] = [
    Category {
        name: "Overview",
        blurb: "What is happening right now",
        sections: &[
            Section {
                tab: MONITOR_TAB,
                title: "Monitor",
                key: 'm',
                blurb: "live training progress, loss and speed",
            },
            Section {
                tab: FEED_TAB,
                title: "Live feed",
                key: 'f',
                blurb: "dataset samples while the run is streaming",
            },
            Section {
                tab: HARDWARE_TAB,
                title: "Hardware",
                key: 'h',
                blurb: "CPU, memory and GPU use",
            },
        ],
    },
    Category {
        name: "Model",
        blurb: "The network, its checkpoints and its behaviour",
        sections: &[
            Section {
                tab: MODEL_TAB,
                title: "Model",
                key: 'v',
                blurb: "configuration read from the training log",
            },
            Section {
                tab: CHAIN_TAB,
                title: "Checkpoints",
                key: 'c',
                blurb: "saved checkpoints in the training folder",
            },
            Section {
                tab: INFERENCE_TAB,
                title: "Chat",
                key: 'i',
                blurb: "talk to a saved checkpoint",
            },
            Section {
                tab: MIXER_TAB,
                title: "Mixer",
                key: 'x',
                blurb: "mix datasets and set the sampling share",
            },
            Section {
                tab: MEMORY_TAB,
                title: "Memory view",
                key: 'r',
                blurb: "inspect retained memory and live retrieval",
            },
            Section {
                tab: MATH_TAB,
                title: "Math",
                key: 'p',
                blurb: "PSSA equations and live dimensions",
            },
        ],
    },
    Category {
        name: "Data & Eval",
        blurb: "Files, samples and scores",
        sections: &[
            Section {
                tab: LIBRARY_TAB,
                title: "Library",
                key: 'l',
                blurb: "model and dataset folders",
            },
            Section {
                tab: SAMPLE_TAB,
                title: "Sample",
                key: 's',
                blurb: "live output and token confidence",
            },
            Section {
                tab: EVAL_TAB,
                title: "Eval",
                key: 'e',
                blurb: "score a checkpoint on held-out prompts",
            },
            Section {
                tab: BENCHMARK_TAB,
                title: "Benchmark",
                key: 'b',
                blurb: "matched comparison against a baseline",
            },
        ],
    },
    Category {
        name: "Training",
        blurb: "Start, watch and resume runs",
        sections: &[
            Section {
                tab: SETUP_TAB,
                title: "Setup",
                key: 'u',
                blurb: "start, resume or configure a training run",
            },
            Section {
                tab: RUNS_TAB,
                title: "Runs",
                key: 'r',
                blurb: "run history; monitor or score a checkpoint",
            },
            Section {
                tab: SWEEP_TAB,
                title: "Sweeps",
                key: 'w',
                blurb: "try parameter ranges in sequence",
            },
            Section {
                tab: TIMELINE_TAB,
                title: "Timeline",
                key: 't',
                blurb: "checkpoints over time; prepare a resume",
            },
        ],
    },
    Category {
        name: "Settings",
        blurb: "Accounts, devices and delivery",
        sections: &[
            Section {
                tab: HF_TAB,
                title: "Hugging Face",
                key: 'h',
                blurb: "sign in for uploads and downloads",
            },
            Section {
                tab: KAGGLE_TAB,
                title: "Kaggle",
                key: 'k',
                blurb: "upload and run training on Kaggle",
            },
            Section {
                tab: BACKUP_TAB,
                title: "HF backup",
                key: 'b',
                blurb: "upload checkpoints to Hugging Face",
            },
            Section {
                tab: LOG_STREAM_TAB,
                title: "Cloud log",
                key: 'g',
                blurb: "read the cloud training log",
            },
            Section {
                tab: NOTIFY_TAB,
                title: "Phone alerts",
                key: 'p',
                blurb: "message your phone when a run finishes",
            },
            Section {
                tab: UPDATE_TAB,
                title: "Updates",
                key: 'u',
                blurb: "check for a newer release",
            },
            Section {
                tab: SUPPORT_TAB,
                title: "Support",
                key: 's',
                blurb: "links and support information",
            },
            Section {
                tab: GITHUB_TAB,
                title: "GitHub",
                key: 'G',
                blurb: "issues and a masked-token login",
            },
            Section {
                tab: DEVICE_TAB,
                title: "Devices",
                key: 'd',
                blurb: "choose CPU, CUDA or WebGPU",
            },
            Section {
                tab: LIMITS_TAB,
                title: "Limits",
                key: 'l',
                blurb: "threads, memory, batch and token caps",
            },
        ],
    },
];

/// Which category holds a screen id.
pub(super) fn category_of(tab: usize) -> usize {
    locate(tab).map(|(category, _)| category).unwrap_or(0)
}

fn locate(tab: usize) -> Option<(usize, usize)> {
    CATEGORIES.iter().enumerate().find_map(|(category, entry)| {
        entry
            .sections
            .iter()
            .position(|section| section.tab == tab)
            .map(|section| (category, section))
    })
}

/// Current category, highlighted section, and whether the section menu is
/// showing instead of the section itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Nav {
    pub category: usize,
    pub section: usize,
    pub menu: bool,
}

impl Nav {
    /// The screen the navigation currently points at, even while its category
    /// menu is open. Keeps polling and header state aligned with the cursor.
    pub(super) fn tab(&self) -> usize {
        CATEGORIES[self.category].sections[self.section].tab
    }

    /// Point at a screen and show it, used when a module opens a screen itself
    /// (for example Library -> Chat).
    pub(super) fn follow(&mut self, tab: usize) -> bool {
        let Some((category, section)) = locate(tab) else {
            return false;
        };
        self.category = category;
        self.section = section;
        self.menu = false;
        true
    }

    /// Show the section menu for a category, starting at that category's first
    /// section. Used by Tab and the 1-5 category keys.
    pub(super) fn select_category(&mut self, category: usize) {
        self.category = category % CATEGORIES.len();
        self.section = 0;
        self.menu = true;
    }

    pub(super) fn next_category(&mut self) {
        self.select_category(self.category + 1);
    }

    pub(super) fn previous_category(&mut self) {
        self.select_category((self.category + CATEGORIES.len() - 1) % CATEGORIES.len());
    }

    pub(super) fn move_section(&mut self, delta: isize) {
        let last = CATEGORIES[self.category].sections.len() - 1;
        self.section = self.section.saturating_add_signed(delta).min(last);
    }

    pub(super) fn open_section(&mut self) {
        self.menu = false;
    }

    /// Escape from a screen: highlight that screen and show its category menu.
    pub(super) fn back_to_menu(&mut self, tab: usize) {
        if let Some((category, section)) = locate(tab) {
            self.category = category;
            self.section = section;
        }
        self.menu = true;
    }

    /// Section letter pressed on the menu, if any.
    pub(super) fn section_for_key(&self, key: char) -> Option<usize> {
        CATEGORIES[self.category]
            .sections
            .iter()
            .position(|section| section.key == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::keybindings::SCREENS;
    use std::collections::HashSet;

    #[test]
    fn every_screen_is_a_section_exactly_once() {
        let mut seen = HashSet::new();
        for category in &CATEGORIES {
            assert!(!category.sections.is_empty(), "{}", category.name);
            assert!(!category.blurb.is_empty());
            for section in category.sections {
                assert!(
                    seen.insert(section.tab),
                    "{} listed twice",
                    SCREENS[section.tab]
                );
                assert!(
                    !section.blurb.is_empty() && !section.title.is_empty(),
                    "{}",
                    SCREENS[section.tab]
                );
            }
        }
        assert_eq!(seen.len(), SCREENS.len());
        for tab in 0..SCREENS.len() {
            assert!(seen.contains(&tab), "{} has no section", SCREENS[tab]);
        }
    }

    #[test]
    fn category_names_and_section_letters_are_unique() {
        let names: HashSet<_> = CATEGORIES.iter().map(|c| c.name).collect();
        assert_eq!(names.len(), CATEGORIES.len());
        for category in &CATEGORIES {
            let keys: HashSet<_> = category.sections.iter().map(|s| s.key).collect();
            assert_eq!(keys.len(), category.sections.len(), "{}", category.name);
            for section in category.sections {
                assert!(!section.key.is_whitespace());
            }
        }
    }

    #[test]
    fn any_screen_reaches_any_screen_in_two_keys() {
        // A category digit then a section letter, from every possible screen.
        for from in 0..SCREENS.len() {
            let mut nav = Nav::default();
            assert!(nav.follow(from), "{}", SCREENS[from]);
            assert_eq!(nav.tab(), from);
            for (index, category) in CATEGORIES.iter().enumerate() {
                for (section, target) in category.sections.iter().enumerate() {
                    let mut nav = nav;
                    nav.select_category(index); // first key: category digit
                    assert!(nav.menu);
                    assert_eq!(
                        nav.section_for_key(target.key), // second key: section letter
                        Some(section),
                        "{} -> {}",
                        SCREENS[from],
                        SCREENS[target.tab]
                    );
                    nav.section = section;
                    nav.open_section();
                    assert_eq!(nav.tab(), target.tab, "{}", SCREENS[from]);
                    assert!(!nav.menu);
                }
            }
        }
    }

    #[test]
    fn arrows_enter_and_escape_keep_their_place() {
        let mut nav = Nav::default();
        nav.select_category(4);
        assert_eq!(nav.category, 4);
        assert!(nav.menu);
        nav.move_section(3);
        assert_eq!(nav.section, 3);
        nav.move_section(100);
        assert_eq!(nav.section, CATEGORIES[4].sections.len() - 1);
        nav.move_section(-100);
        assert_eq!(nav.section, 0);
        nav.open_section();
        assert_eq!(nav.tab(), CATEGORIES[4].sections[0].tab);
        nav.back_to_menu(nav.tab());
        assert!(nav.menu);
        assert_eq!(nav.section, 0);
        nav.next_category();
        assert_eq!(nav.category, 0);
        assert!(nav.menu);
        nav.previous_category();
        assert_eq!(nav.category, 4);
    }

    #[test]
    fn follow_finds_the_category_and_section_for_every_screen() {
        for tab in 0..SCREENS.len() {
            let mut nav = Nav::default();
            assert!(nav.follow(tab), "{}", SCREENS[tab]);
            assert_eq!(nav.tab(), tab);
            assert!(!nav.menu);
            assert_eq!(category_of(tab), nav.category);
        }
        assert!(!Nav::default().follow(usize::MAX));
    }
}
