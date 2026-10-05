# Gradient clipping and memory value cap: validation

Implementation checkpoints: `9d5eab2` (CPU fused clipping and CUDA kernels),
`bb87f58` (resident CUDA training integration), and `bf1ef18` (stacked-backward
embedding-row marking regression fix). The final follow-up checks and timing
probe use `bf1ef18`.

## Backend and remaining host work

`--backend auto` tries CUDA first when built with `--features cuda`, then WebGPU,
then CPU. The training log's `backend=` line establishes which backend actually
initialized; a CUDA-feature build alone does not establish that CUDA is in use.
Explicit `--backend cuda` fails rather than silently selecting another backend.

With CUDA selected and gradient clipping enabled, dense gradients and Adam
moments remain on the device through optimizer updates. The global f64 L2 norm
is reduced on the device, one norm scalar is read back, and scaled AdamW runs on
the device. A separate scalar finite-state check replaces the full host scan of
parameters, gradients, and moments. Gradients and moments are downloaded at
final handoff. Host-computed embedding, normalization, and SSM-rate gradients
are uploaded as needed; embedding rows use sparse row marks. The `bf1ef18`
regression verifies these marks across accumulated stacked-backward chunks.

Parameter data is still downloaded after each update as a host mirror required
by the current recurrence, retrieval, embedding, and looped stages. Host EMA
changes to fast adapter weights are explicitly refreshed on the device. This is
not a fully device-resident training implementation and does not eliminate all
per-update transfers.

Memory-value cap arithmetic uses CUDA after device selection: all capacity rows
once at startup, then only each accepted inserted value. Capped values are
returned to the existing host-owned bank; retrieval and metadata are unchanged.
Rejected protected writes do not run the cap kernel. The memory bank is not
made device-resident. CPU/WebGPU retain the CPU safeguard path; CPU clipping
uses one norm pass and fuses scaling into the required AdamW update pass.

## Light serial checks

All follow-up Cargo commands use `CARGO_BUILD_JOBS=1`, run **one at a time**, and
retain the existing release profile and shared target directory. No full test
suite or `--no-run` check was used. The default-feature checks were already
recorded as passing on `bf1ef18` before this final follow-up; the CUDA-feature
checks were rerun during the follow-up.

| Command (prefix each with `CARGO_BUILD_JOBS=1`) | Result |
| --- | --- |
| `cargo build --release` | Passed (previous follow-up) |
| `cargo test --release --test training_safeguards` | 5 passed (previous follow-up) |
| `cargo test --release --lib training_safeguards_tests` | 8 passed (previous follow-up) |
| `cargo build --release --features cuda` | Passed |
| `cargo test --release --features cuda --lib training_safeguards_tests` | 8 passed, 2 GPU-required tests ignored |
| `cargo test --release --features cuda --lib memory::tests` | 9 passed, 1 GPU-required test ignored |
| `cargo test --release --features cuda --test training_safeguards` | 5 passed (CPU execution in the CUDA-feature build) |

The CPU historical-reference test checks exact optimizer state and gradient bits
across repeated updates, all depth-two tensors, nonzero moments, weight decay,
signed zero, huge finite gradients, below-limit gradients, and subnormal clip
limits. Other checks cover nonfinite-update skipping, the optimizer-counter
limit, loaded-bank capping, protected writes, and byte-compatible flags-off
resume. Existing dead-code and deprecated linker-optimization warnings remain.

## Bounded before/after timing

Reproduce serially, without another build or workload running:

```sh
CARGO_BUILD_JOBS=1 cargo run --release --example grad_clip_probe -- 2000000 7
```

This compares the historical separate norm/scale/Adam passes against fused CPU
clipping in the **same process**, not two separately built revisions. Both
models have 2,000,800 trainable scalars and deterministic equal initial state;
the probe visits every Adam gradient in historical order, warms up twice, and
measures seven updates with alternating execution order. Construction and
gradient resets are excluded. These are CPU elapsed milliseconds per clipped
optimizer update, not CUDA host-dispatch time or end-to-end training throughput.

| Probe revision | Before median (ms/update) | After median (ms/update) | Change |
| --- | ---: | ---: | --- |
| `9d5eab2` (earlier recorded run) | 16.697 | 16.775 | +0.078 ms (+0.47%) |
| `bf1ef18` (final repetition) | 12.807490 | 16.459833 | +3.652343 ms (+28.52%) |

Final samples, in measurement order (ms/update):

- Before: `[10.654882, 11.182238, 10.458300, 14.266724, 14.908730, 14.922148, 12.807490]`
- After: `[15.492827, 16.816615, 17.156144, 16.185642, 19.802745, 16.459833, 15.893656]`

The final run used Rust `1.98.1 (48a229cea 2026-09-01)` on Linux x86_64/KVM,
with two virtual CPUs and 4 GiB RAM. No other Cargo command or test workload was
running during measurement. Before samples ranged from 10.458300 to 14.922148
ms; after samples ranged from 15.492827 to 19.802745 ms.

There is **no demonstrated CPU speedup**. In the final repetition the fused CPU
path was **28.52% slower** on this bounded workload. Absolute times vary between
the recorded sandbox runs, but that does not erase this repetition's observed
CPU regression. Exact CPU numerical equivalence is separately checked by the
passing historical-reference test. The intended CUDA residency improvement
cannot be measured in this environment, and these CPU timings do not prove
recovered GPU throughput.

## Unverified GPU behavior

The sandbox has two virtual x86_64 CPUs, 4 GiB RAM, and no GPU or CUDA toolkit.
CUDA-feature compilation is checked, but the three ignored numerical/real-CLI
GPU tests have **not** been executed. The molab figures (~2,200 tokens/s without
the safeguards versus ~260 with them on an RTX Pro 6000 Blackwell) are
user-provided historical observations, not sandbox measurements.

Before claiming recovered GPU throughput, run the ignored CUDA optimizer,
CUDA CLI, and CUDA memory-cap tests on a GPU, then repeat the original molab
training workload and confirm the log reports `backend=cuda`. Remaining bulk
weight mirroring and host stages may still limit that workload.
