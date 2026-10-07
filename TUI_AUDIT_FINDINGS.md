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
| A transformer-resume wizard accepted shared `Max batch lanes` values and emitted `--batch-size`, which `train-transformer` does not accept; even an explicit single lane failed after starting the child. | Wizard/CLI argument audit; `transformer_resume_rejects_multi_lane_batches_and_omits_cpu_only_flag` failed on the old flag emission through csrun. | `d18e2fa` |
| The completed-child cleanup ran before the sweep controller could read its result, permanently leaving the first trial `Running` and blocking the remaining queue. | Independent code audit; `shell_completion_order_records_trial_before_releasing_slot_and_advances_queue` uses an inert `/bin/true` child, the shell's actual ordering, and recorded metrics. | `d8dfae5` |
| Runs only looked for `train.log` beside `model.pssa`; wizard-produced `model.trfm` checkpoints incorrectly lost their recorded history and claimed it was not recoverable. | Independent code audit; wizard-history regression now reopens both formats and checks loss, throughput, samples and no missing-history warning. | `ab70842` |
| Successfully reopened run history always showed `[ WAITING ]`, because the history loader clears the live stall timestamp and the status label ignored the restored completion summary. | Independent audit plus the wizard-history TestBackend regression for both checkpoint formats. | `8544b9c` |
| `--compare` opened and parsed an unbounded file on the raw-mode event-loop thread; a FIFO could hang the UI indefinitely and large logs blocked redraw/input. Errors were silently discarded. | Independent audit; a timeout-guarded FIFO subprocess regression, bounded loader tests, and a TestBackend unavailable-comparison assertion pass through csrun. | `e66aed5` |
| Single-chat and A/B prompt tails were clipped by character count rather than terminal-cell width, so long CJK drafts hid the newest text off the right edge. | TestBackend regression with a long `世界` draft ending in `END` at 80x24, 120x40 and 60x20, in both modes. | `c2a3012` |

## Checks and non-findings

- WebGPU parity software-adapter skip is tracked in `76073a9`.
- Final checks through `/workspace/bin/csrun`: TUI unit/TestBackend suite
  passed **275 tests**; `tests/tui_chain_args.rs` passed **5 tests** and
  `tests/tui_preview.rs` passed **2 tests**. `tests/wgpu_parity.rs` passed
  with adapter skips on this no-GPU host. All four device-picker tests also
  pass with `--features cuda`, including CPU/CUDA/WebGPU application and labels.
  `cargo check --features cuda`, `cargo clippy --lib --tests`, and
  `cargo build --release` passed. Clippy reports repository-wide warnings;
  this is not a warning-free clippy result.
- `scripts/tui_audit_pty.py` drives the release binary with an incremental VT
  screen reader and private temporary HOME/config/chain/chat paths. The final
  default-feature run passed: 111 captured screens, all 27 tabs at 80x24,
  120x40 and 60x20, plus 20x8; help/palette, wizard pages/command preview,
  scoped editors, graph controls, function keys, CPU selection, loud unavailable
  WebGPU/CUDA rows, implicit CLI launch, q/Ctrl+C/Escape quit and editor cancel.
  Screens are saved on the codespace in `/tmp/pssa-tui-audit-screens-final.txt`.
  No live trainer, upload or authenticated service call was launched.

## Coverage limits / not fixed + why

- No remaining confirmed TUI bug is intentionally left unfixed.
- Codespace checks cannot verify a successful real CUDA/WebGPU device binding
  or numerical GPU parity. CPU/CUDA/WebGPU selection and labels have
  deterministic TestBackend fixtures; real no-GPU refusal is exercised in the
  release PTY sessions. Hardware-backed execution still needs molab.
- Authenticated Kaggle/HF/GitHub uploads, live cloud delivery and actual training
  launches were not performed. Their validation, lifecycle and error paths are
  covered by the existing offline/injected tests and the PTY editor/navigation
  sweep; this is not an end-to-end production-service verification.
