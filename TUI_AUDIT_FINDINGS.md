# PSSA TUI audit findings

Branch: `tui-audit` (base `c6c0cd3`). The audit covers `src/tui/**`, the TUI
launch path in `src/cli.rs`, ratatui `TestBackend` coverage, and headless
pseudo-terminal sessions at 80x24, 120x40, and 60x20.

## Confirmed bugs

| Bug | How found | Fix commit |
| --- | --- | --- |
| `tui::hardware::parse_ram` evaluated `total - available` eagerly inside `then_some`, so malformed `/proc/meminfo` with `MemAvailable > MemTotal` panicked instead of reporting missing RAM. | `CARGO_BUILD_JOBS=1 cargo test --lib tui::hardware::tests::linux_counters_exclude_guest_and_handle_resets_and_missing_ram -- --nocapture` through `/workspace/bin/csrun`; failure at `src/tui/hardware.rs:85`. | `a24ce5c` |
| `RunState::refresh_chain` only added scanned checkpoints and retained files deleted from disk or from a missing chain directory, leaving stale chain entries and preview candidates. | Read-only audit of `refresh_chain` and `preview::candidate`; regression test creates, scans, deletes, and rescans a checkpoint plus a missing directory. | `fb5d677` |
| Escape was documented as quitting network browse tabs but routed `Context::Network` and `Context::Timeline` to no-op handlers; only editors should consume Escape to cancel. | Read-only keybinding audit plus headless pty navigation of network/timeline tabs; regression assertions cover browse, timeline, and editor contexts. | `e8693de` |
| The empty/no-input dashboard rendered `[ TRAINING ]` because `status_badge` treated every non-DONE normal state as training, hiding the documented waiting state. | Headless pty launch with an empty chain at 80x24/120x40/60x20 showed `[ TRAINING ]` before any producer log; the default `HealthStatus` was `WAITING`. | `f629582` |
| A wizard-launched child was polled to completion but retained in `training`, so a piped TUI never reached its EOF exit predicate and stayed open after the final frame. | Read-only event-loop audit and setup child lifecycle trace: `TrainingRun::active()` became false, but `training` stayed `Some`; the loop exits only when it is `None`. | `35f4480` |
| Repeating `--chain`/`-c`, `--chats-dir`, or `--compare` silently overwrote the earlier TUI launch setting, making a typo select a different chain or log. | CLI argument audit plus `tests/tui_chain_args.rs::tui_rejects_repeated_options_and_mixed_chain_aliases`; duplicate spellings now return an error before TTY detection. | `34c2012` |
| A piped producer that closed without a completion summary and saved checkpoint was shown with the green `DONE` badge because EOF only cleared `training_active`; the alert layer reported an error at the same time. | Code audit of the stdin-disconnect path and `Network::stream_eof`; regression test `tui::tests::truncated_piped_stream_is_not_reported_as_done` exercises an incomplete stream and keeps an empty TUI in `WAITING`. | `916fdbb` |
| An empty monitor fabricated `0 tokens/s`, `loss 0.0000`, and zero epoch counters even though no producer had reported measurements; the tiny fallback also showed `0% loss 0.0000`. | Read-only monitor audit plus `tui::tests::test_backend_header_shows_live_stats_on_every_tab_and_missing_values` at 20x8; missing values now render as `-`. | `18759e8` |
| After a sampled checkpoint was deleted, the stale `last_checkpoint` candidate and generated sample remained visible; missing metadata returned without clearing preview state. | Preview/state audit plus `tui::preview::tests::errors_retain_last_sample_and_pausing_never_starts_work`: a real temporary `.pssa` candidate is deleted, candidate resolution becomes `None`, and the old text/checkpoint are cleared. | `cf41510` |

## Checks and non-findings

- WebGPU parity software-adapter skip is tracked in `76073a9`.
- The TUI unit/TestBackend suite now reports 268 passing tests through
  `/workspace/bin/csrun`; `tests/tui_chain_args.rs` reports 4 passing tests.
  The targeted hardware and checkpoint regressions, the Escape routing
  regression, and the full TUI module suite all pass; only the repository's
  existing dead-code warnings remain.
- Headless pty sweeps rendered all 27 tabs, every requested size, the help and
  palette overlays, the setup review/CLI preview, device picker unavailable
  rows, limits, math, sample, and empty/no-checkpoint states without a crash.
  Escape-to-quit behavior was also exercised deliberately.

## Not fixed + why

None yet.
