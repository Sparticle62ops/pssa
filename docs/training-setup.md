# Training from the TUI

Run `pssa tui` in a terminal, then press **Tab** to reach **setup**.
The existing command-line training and piped-log monitor remain available.

## Wizard

1. **Dataset**: choose `local` (one existing UTF-8 file) or `hf` (an
   `owner/name` dataset). HF config is optional; split defaults to `train`
   and text field to `text`. The existing CLI downloads/caches HF data.
2. **Model**: latent/state dimensions, BPE vocabulary ceiling (257–65,536), depth, loops.
   Depth and loops range from 1 to 32. On wide terminals the previews show
   stacked neuron nets and an oval with electricity making one lap per pass.
   They illustrate the selected configuration, not measured model activity.
3. **Training**: learning rate, epochs, optional max tokens, seed, chunk,
   accumulation, and the dream controls: dream cadence (`--dream-every`,
   default `0`/off), replay count (`--dream-replay`, default `32`), mode
   (`--dream-mode`, default `memory`), generated length (`--dream-len`, default
   `64`), rehearsal rate (`--dream-lr`, default `0.006`), and rehearsal passes
   (`--dream-steps`, default `1`). Blank max tokens means no cap. Dream values
   use the same validation as the CLI and are ignored for transformer runs.
4. **Review / launch**: output directory, optional resume checkpoint, backend,
   and the equivalent shell command. Dream flags appear in that command only
   when they differ from their defaults. Select **START TRAINING** and press Enter.
   The monitor then reports active dreams, completed count, mode, and last dream
   loss separately from the ordinary training loss.
   On success the app switches to the monitor.

**Up/Down** selects a field or button. **Enter** edits a field (or cycles source
and backend); type text, **Backspace** deletes, **Ctrl+U** clears, **Enter** saves,
**Esc** cancels the edit. **Left/Right** changes pages while not editing.
**+/-** adjusts a selected depth or loops value. **PgUp/PgDn** scrolls the command.
**c** opens/closes a full-screen command preview, including on narrow terminals.
**Tab** still switches application tabs, even while editing. The tab order is
monitor, chain, model, feed, inference, setup. **?** opens the shared keyboard
help while browsing; **F1** opens it anywhere. In chat and setup text fields,
`?` and `q` remain literal input. Help lists each shortcut once with its scope;
scroll with the arrow/page keys, or use Home/End. Esc closes help without
quitting or cancelling the underlying editor. On narrow terminals, the wizard
retains its keyboard-editable fields and omits the animation previews.

Numeric inputs, local-file readability, HF names, and output paths are validated.
The CLI's source-list syntax cannot represent local filenames containing commas
or ending in whitespace; the wizard reports these before starting.
The child validates dataset contents, device availability and checkpoint contents
without loading a model into the UI process. When resuming, latent/state/depth
and chunk must match the checkpoint; vocabulary comes from that checkpoint.
Errors from the child appear in the monitor and remain in `train.log`.

## Output and lifecycle

The wizard creates `model.pssa` and `train.log` in the chosen directory. It
refuses to overwrite either file: use a new directory for each run, including
resumed runs. Only one wizard-launched run may be active at a time.

The trainer is a separate process, uses `--no-tui`, and writes stdout/stderr
straight to the log file, not a pipe owned by the dashboard. Slow redraws cannot
block training. On Unix the child also has its own process group, separate from
terminal job-control signals. **Quitting the TUI leaves the child training**. The PID appears
at launch in the monitor; to stop a detached run, use normal OS process controls.
In another terminal, reopen its monitor with:

```sh
tail -f 'runs/my run/train.log' | pssa tui --chain 'runs/my run'
```

The app stays open after a child finishes so the final status remains visible.
The output directory is also the monitor's checkpoint chain directory and is
included in the inference tab's `/model` picker. Starting training does not
replace the checkpoint already selected for an existing conversation.

## Backend CLI option

`train --backend auto|cpu|webgpu|cuda` is additive. Omitting it (or using `auto`)
keeps the original automatic GPU selection and CPU fallback. `cpu` does not
probe GPUs. Explicit `webgpu` and `cuda` fail visibly rather than silently
falling back. CUDA requires a binary built with `--features cuda`; the wizard
only offers CUDA in such a build. This is backend selection, not an individual
GPU/device picker. Existing CLI flags, checkpoint formats, and training math
are unchanged.
