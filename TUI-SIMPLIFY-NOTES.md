# TUI simplification notes

Goal: make the PSSA TUI easier to navigate and understand without changing what it
can do. No dependencies, CLI flags, or output formats changed. The Halloween
palette (`fn spooky()` / `PSSA_SPOOKY`) was left untouched.

Verification: `cargo test --profile fast --lib` stays green (513 passed, 2 ignored)
after every commit.

## 1. Navigation

- **F1 help is easier to scan.** The overlay now opens with a short legend that
  states the shared vocabulary: keys are listed on the left, F1/?/Esc close,
  PgUp/PgDn scroll, `Tab` = next tab everywhere, `Shift+Tab` / `Left` / `Right`
  move between most browse tabs, `Ctrl+K` jumps to any tab, and Esc always
  "backs out" (close overlay, cancel edit, stop work, or quit).
- **Shift+Tab now means "previous tab" on browse pages.** Previously `Tab` cycled
  forward globally but `Shift+Tab` only did Setup wizard pages. Now browse tabs
  (except Setup, which keeps Shift+Tab for its wizard) route `Shift+Tab` to the
  previous tab, so forward/back tab switching is symmetric and discoverable.
- **Header and per-tab hints were rewritten in plain language** and now name the
  real controls: `Tab/Shift+Tab tabs`, `1-7 pick graph`, `+/- zoom`, `0 reset`,
  `Ctrl+K jump`, `F5/Shift+Tab pages`. The misleading "arrows tabs" hint and a
  duplicate hint arm were removed.
- **Audited the key table** for redundancy. The remaining aliases (`+`/`=` zoom,
  `y`/`Y` and `n`/`N` for shift/caps-lock, `q`/Esc/Ctrl+C quit) are intentional
  and removing them would drop real ergonomics, so they were kept.

## 2. Understandability

- **Every F1 help description was rewritten in plain sentences**, preserving all
  original facts (scopes, exception cases, and the Kaggle-detaches nuance) with no
  new jargon.
- **Pane footers use one vocabulary.** Control hints now consistently use `·`
  separators, `Tab next tab`, `PgUp/PgDn`, and `Esc` wording across chat, setup,
  benchmark, HF, Kaggle, eval, inspector, preview, and A/B panes.
- **Compact status lines say what the pane is for.** Hardware/math/devices/limits
  now read as sentences ("CPU, memory and GPU usage…", "Choose CPU, CUDA or
  WebGPU…") instead of slash-separated terse fields.
- **Jargon-heavy pane titles were reworded**: "runtime compute" → "available
  compute", "opt-in corpus" → "extra data you opt into", "enforcement /
  semantics" → "how limits are applied", "bounded tail" → "latest lines",
  "equivalent CLI" → "same run as a CLI command", "wizard base" → "base settings
  from setup".
- **Loss-trend status label** changed from cryptic `step → / lower better` to
  `steps → · lower better` (same facts, clearer grammar; verified it still fits
  the 80-column header).

## 3. Cheap code simplification (behavior unchanged)

- Collapsed the status-badge label match that mapped `"DONE" => "DONE"` etc. into
  a single pattern arm returning the existing label.
- Removed a duplicate header-hint match arm (feature tabs used the same text as
  the default) and consolidated help-line construction.

## Tests updated (only where wording was intentionally changed)

- Header/footer/loss-trend string expectations and pane-title expectations in the
  TUI test modules were updated to the new wording. No assertions were weakened:
  the exhaustive route uniqueness, modal help, narrow-terminal, and Halloween
  checks all still run unchanged.
