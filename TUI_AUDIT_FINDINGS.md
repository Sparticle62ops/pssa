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

## Checks and non-findings

- WebGPU parity software-adapter skip is tracked in `76073a9`.
- The existing TUI unit/TestBackend suite covered 266 passing tests before the
  hardware fix; its one failure was the bug listed above. The targeted hardware
  and checkpoint regressions, the Escape routing regression, and the full TUI
  module suite are being rerun after these fixes.
- Headless pty sweeps rendered all 27 tabs, every requested size, the help and
  palette overlays, the setup review/CLI preview, device picker unavailable
  rows, limits, math, sample, and empty/no-checkpoint states without a crash.
  Escape-to-quit behavior was also exercised deliberately.

## Not fixed + why

None yet.
