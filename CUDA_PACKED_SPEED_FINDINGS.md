# Packed CUDA speed audit (baseline: 23d7bba)

## Baseline inventory — recorded before implementation

Scope: one successful `SequenceBatch::forward` + `backward`, depth=1,
loops=1, B active lanes, N packed tokens, each lane length <=1024. This is a
microbatch step, **not** AdamW/gradient clipping/memory insertion. No GPU exists
in this sandbox; the reported molab measurements are supplied baseline evidence,
not measurements made here: training widths CPU 0.294 s / CUDA 0.536 s; small
CPU 0.001 s / CUDA 0.038 s. The supplied timings do not establish a repeatable warmed baseline here; no
GPU timing has been collected in this sandbox.

Counts below enumerate explicit PTX launches, cuBLAS submissions, CUDA allocation
calls and copy calls in source. cuBLAS can internally launch multiple kernels;
its actual kernel count requires Nsight. Allocation includes `clone_htod` and
`CudaSlice::clone` (the latter **allocates and copies**, not a shared handle).
`alloc_zeros` also submits a memset. Pageable-copy driver staging can block even
though the API name says Async. cudarc 0.19.10 ordinary host-slice guards do not
synchronize; the explicit readback synchronize is required for host ownership.
All device buffers use the default stream; there are no explicit context/device
synchronizes per step. On devices without async allocation support, each buffer
free additionally stream-synchronizes (not the expected Blackwell path).

### Ranked by expected cost

1. **B separate forward memory pipelines and B readback waits**
   (`forward_lane_cuda`, `memory_forward_after_ssm`, `run_memory_forward`).
   Each lane: 4 PTX launches (query add, retrieval, sigmoid, injection),
   5 cuBLAS calls (Qx, Qh, value reduction, gate, projection), 9 HtoD calls
   (x_norm, initial weights tape, four matrices, keys/norms/values),
   8 DtoH memory tapes, 18 allocations and 1 DtoD copy (cloned SSM y).
   With an empty bank: value GEMM is replaced by a memset.
   Large gate/projection matrices and bank values are re-uploaded **B times**.
2. **B independent SSM backward upload/allocate/readback cycles**
   (`Lane::backward_gpu`, `CudaContext::ssm_backward`). Each: 3 PTX launches
   (reverse maps, scan, local VJP), 12 HtoD calls (delta, raw delta, B, C,
   rates, derivatives, x, full states, A/B maps, gz, gy), 5 DtoH calls
   (gdelta, gB, gC, per-token gA, gx), 23 allocations, 1 stream wait.
   The forward resident buffers are not reused by this API.
3. **B SSM forward allocation/upload cycles** (`make_ssm_buffers`). Each:
   3 PTX launches (prepare, scan, materialize), 8 HtoD calls (seven cloned
   inputs including carry, plus duplicate initial-state row), 4 DtoH calls
   (A/B maps, full states, y), 16 allocations including scan summaries.
   The SSM readbacks share the memory readback's one explicit wait.
4. **32 host-visible dense GEMMs outside the per-lane forward pipeline**:
   8 forward (3 projections, adapter down/up, MLP1/2, head), 24 backward
   (2 head, 4 MLP, 2 adapter up, 2 adapter down, 8 memory, 6 SSM projections).
   Each submits one cuBLAS operation, copies lhs HtoD and output DtoH, then
   stream-synchronizes. The 12 backward weight-gradient GEMMs additionally
   upload rhs activation tapes and the seeded output gradient. Input-adjoint
   weights reuse the cache. High-water lhs/rhs/output buffers allocate only
   on growth; eight unique forward matrices populate the cache cold. Cache
   invalidation after AdamW discards those allocations in the baseline.
5. **Packed memory backward kernels**: gate local VJP: 1 PTX, 3 HtoD,
   2 DtoH, 5 allocations, 1 wait. Retrieval VJP: 1 PTX, 8 HtoD, 2 DtoH,
   10 allocations, 1 wait. Both re-upload forward tapes and bank contents.
6. **Softplus after delta projection**: 1 PTX, 1 HtoD, 1 DtoH,
   1 allocation, 1 wait. Embedding/norm, SiLU/residual, softmax/loss,
   gradient aggregation and embedding scatter are CPU work.

### Totals (occupied memory bank, warmed common GEMM pool/cache)

| Operation | Formula | B=32 |
|---|---:|---:|
| Explicit PTX launches | 10B + 3 | 323 |
| cuBLAS submissions | 5B + 32 | 192 |
| Combined submissions (not cuBLAS internal kernels) | 15B + 35 | 515 |
| Explicit stream synchronizes | 2B + 35 | 99 |
| Explicit context/device synchronizes | 0 | 0 |
| HtoD copies | 29B + 68 | 996 |
| DtoH copies | 17B + 37 | 581 |
| DtoD copies | B | 32 |
| Device allocations | 57B + 16 | 1840 |

The allocation total counts 34/lane forward + 23/lane backward, plus 15 memory
backward allocations and 1 softplus allocation. Cold common GEMM workspace
growth/cache misses are additional, shape/order-dependent allocations and eight
additional weight uploads. For lanes >1024, each recursive scan level adds
2 PTX launches and 4 scratch allocations, plus the recursive leaf's 2 summary
allocations instead of the single-level 2; use `launch_scan`'s recurrence to count
ragged lengths individually. Each alloc_zeros submits one memset (forward 17/lane,
backward 11/lane, gate-backward 2, retrieval-backward 2; clone allocations do not).
No per-lane carry-only copy is needed in addition to the full state readback.

`PSSA_CUDA_PROFILE=1` reports host `StageTrace` durations (`gemm.upload`,
`gemm.submit`, `gemm.readback_wait`, `memory.upload`, `memory.query_submit`,
`memory.retrieval_submit`, `memory.gate_projection_submit`,
`memory.readback_wait`, `batch.forward.ssm`, `batch.backward.ssm`). These are
wall-clock submission/readback boundaries, not CUDA-event kernel times. Real
GPU profiling and fair warmed before/after timing remain necessary.

## Implementation — source complete, verification BLOCKED

**Do not treat this branch as validated or as a measured speedup.** The resumed
implementation has been reviewed and completed, but every remote check is
blocked before execution by unauthenticated GitHub CLI codespace access. There
is no GPU here, and the instruction to run heavy commands only through `csrun`
has been respected. No local Cargo/compiler/PTX-emulator/PTXAS command or
training run was started. No authentication, remotes, or other worktrees were
changed; README graphics, LICENSE and checkpoint formats are untouched.

### Changes

* `src/cuda/packed.rs`: a `SequenceBatch`-owned fixed-capacity CUDA workspace.
  Stable lane IDs occupy `grid.y`; token tapes use packed offsets, while state
  tapes use `[lane, chunk+1, latent*state]`. Forward projections, SSM, memory,
  and their backward GEMMs all use device views, without the host-slice GEMM
  boundaries inside this pipeline. The workspace allocates **58 device buffers
  once** (49 f32 tape/weight buffers, seven private matrix-gradient buffers,
  two metadata buffers); ragged/sparse batches reuse them without allocation.
* `src/cuda/stages.ptx`: four packed entries (ordered exclusive scan,
  materialization, reverse maps, local VJP). One invocation covers all lanes;
  each channel's scan walks only its own sequence. Initial carries are an
  immutable device snapshot, so terminal threads cannot race with early-token
  materialization. Omitted lanes keep their carry. Resets and `state_mut`
  invalidate the host carry mirror; otherwise the next step uses the existing
  device carry without uploading it again.
* `src/cuda/packed.ptx`: deterministic latent-order B/C reduction and
  reverse-token rate-gradient reduction. No atomic floating-point sums.
  Reverse-map storage is reused for B/C contributions after the scan; the dead
  exclusive-A prefix buffer is reused for rate-gradient contributions. No
  full token-state/adjoint maps are downloaded or re-uploaded for backward.
* Existing memory, elementwise and cuBLAS kernels are reused. Elementwise/local
  blocks use 256 threads to avoid a 1024-thread register-resource constraint.
  Serial per-token retrieval kernels use one block per token so a short batch
  does not concentrate all bank/value walks in one SM. This launch geometry
  needs real GPU profiling; it is not a measured occupancy improvement.
* Scoped CLI optimizer gradients remain device-owned. Seven private adjoints
  are seeded from current gradients, GEMMs accumulate into those private
  buffers, and successful backward publishes them by allocation swaps under
  the optimizer lock. Failure cannot partially update optimizer gradients.
  Outside that scope (including `gpu_train_check`) public host gradients are
  published only after all readbacks succeed.
* `src/sequence_batch.rs`: errors warn loudly and drain submitted work before
  replay. CPU fallback rebuilds the missing forward tapes from saved initial
  states, preserves user carry edits, and accumulates adjoints exactly once.
  A failed CUDA backward uses CPU for the entire remaining backward path, not
  more GEMMs on the failing stream. Prior resident gradient contributions must
  be recovered successfully; inability to recover returns an explicit error
  rather than pretending CPU replay can produce correct accumulated gradients.
  The CPU recurrence, checkpoint format, and stacked/looped replay behavior
  have not been intentionally changed.
* `src/cuda.rs` and `safeguards.rs`: cache invalidation retains device weight
  allocations for in-place refresh; optimizer norm/invalid-flag and memory-cap
  buffers are pooled. The DtoD driver symbol is explicitly preflighted. TN
  dispatch seeds from the host mirror when transitioning from CPU-owned
  gradients, rather than reviving an old resident contribution.
* `examples/gpu_train_check.rs`: both packed configs still check finite loss,
  every parameter gradient, and every lane carry at relative error **<1e-3**.
  The production-width config now actually processes **all 32 lanes**, not
  merely two lanes in a 32-lane workspace. Step 0 is checked cold/warmup; step
  1 reverses packing order and retains carries, and reports warmed CPU/GPU
  forward+backward seconds. `PSSA_REQUIRE_CUDA=1` rejects unavailable CUDA,
  WebGPU substitution, and packed CPU replay instead of accepting false parity.

### Expected counts after batching

The baseline table above describes ordinary host-visible gradient execution,
which is also what `gpu_train_check` exercises. For the same nonempty bank,
fixed shape, warmed common GEMM pool and caches, the new counts are:

| Operation | Before (B=32) | After (any B within the workspace) |
|---|---:|---:|
| Explicit PTX launches | 323 | 20 |
| cuBLAS submissions | 192 | 37 |
| Combined submissions | 515 | 57 |
| Explicit stream synchronizes | 99 | 17 |
| Explicit context/device synchronizes | 0 | 0 |
| HtoD copies | 996 | 41 + C |
| DtoH copies | 581 | 28 |
| DtoD copies | 32 | 2 |
| Device allocations | 1840 | 0 steady-state; 58 packed workspace cold |

`C=1` when any lane reset, host carry edit, CPU replay or new CUDA workspace
requires a carry upload; otherwise `C=0`. These are source-level counts, **not
GPU trace measurements**. cuBLAS internal kernels/allocations and driver/JIT
activity are not counted as explicit source submissions.

Detailed after inventory (each row is the entire packed batch, not per lane):

| Pipeline | PTX | cuBLAS | waits | HtoD | DtoH | DtoD | warm allocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Packed forward | 8 | 8 | 1 | 15+C | 4 | 2 | 0 |
| Packed backward | 12 | 14 | 1 | 1 | 9 | 0 | 0 |
| Remaining host-visible forward GEMMs | 0 | 5 | 5 | 5 | 5 | 0 | 0 |
| Remaining host-visible backward GEMMs | 0 | 10 | 10 | 20 | 10 | 0 | 0 |

Forward PTX: softplus, prepare, scan, materialize, query-add, retrieval, sigmoid,
injection. Forward GEMMs: delta/B/C projections, Qx/Qh, value reduction, gate,
projection. Backward PTX: gate-local, retrieval, Qx input-add, reverse maps,
scan, local SSM VJP, B/C reduction, rate reduction, direct SSM input-add, three
projection input-adds. Backward GEMMs: seven input adjoints and seven matrix
adjoints. Remaining GEMMs: adapter down/up, MLP1/2, head forward; two head, four
MLP and four adapter backward calls.

Packed forward uploads: offsets, lengths, normalized embeddings, rates,
derivatives, seven matrices and three occupied-bank slices (+ optional carry).
DtoD: carry snapshot and raw-delta snapshot. Forward readbacks: SSM readout,
memory injection, projected keys for host terminal-memory insertion/diagnostics,
public carries. Backward uploads only the residual adjoint. Backward readbacks:
seven matrix gradients, embedding/norm input adjoint, and per-lane rate sums.
The last two remain CPU consumers; per-lane rates are folded in stable lane
order on the host. The host key/readout/carry mirrors preserve existing public
access, memory insertion and nonfinite-loss diagnostics, not backward storage.

**Scoped CUDA CLI training:** after the first ownership transition, the packed
pipeline has 9+C HtoD, 6 DtoH and 16 DtoD calls; the whole microbatch has **29+C
HtoD, 16 DtoH, 16 DtoD, the same 57 submissions/17 waits, and zero warm device
allocations**. Seven weight snapshots replace uploads with DtoD; seven gradient
seeds use DtoD; five exterior TN calls have two input uploads and no gradient
readback/seed. First use can instead upload seven private gradient seeds and
five exterior TN seeds (12 extra HtoD, seven fewer DtoD). Scope registration,
AdamW/clipping, memory insertion and scope handoff are outside this microbatch
inventory. The baseline scoped counts differ from the baseline host-gradient
table: omit twelve TN readbacks and twelve output-seed uploads, i.e. 984 HtoD
and 569 DtoH at B=32 (same submissions/waits/allocations).

Warmup qualifications:

* A cold common GEMM pool can grow lhs/rhs/output buffers several times; five
  exterior matrices populate the standalone cache. After an optimizer update,
  standalone execution refreshes those five cached matrices with five HtoD
  copies but no reallocations; the scoped path refreshes the effective adapter
  up matrix with one HtoD copy. Shape growth beyond a pool's high-water mark
  may allocate. Changing CUDA context constructs a new packed workspace.
* An empty bank removes the value GEMM (37 -> 36 total submissions to cuBLAS)
  and three bank uploads; one value-tape memset replaces that GEMM. All scans
  still cover all active lanes in one invocation, including lengths >1024;
  there is no host-recursive scan allocation/launch escalation.
* Pooled clip norm buffers allocate four buffers once; the cap invalid flag
  allocates once and value scratch grows only at high-water marks. These are
  excluded from the microbatch totals, along with registration and AdamW.
* Pageable-copy staging, synchronous malloc on older drivers, non-async frees,
  event bookkeeping, cuBLAS internal work and PTX module loading are not
  hidden by these counts. All application operations use the default stream;
  there are no explicit per-lane waits or device-wide synchronizes.

### Driver-free regression coverage added (NOT executed here)

The tests interpret the actual embedded PTX, not a Rust rewrite, with strict
address validation and cross-thread write-ownership checks. They cover:

* ragged/reordered/omitted/reset lanes, one-token and all-empty kernel launches,
  all 32 lanes, adjacent boundaries and over-launched threads;
* immutable initial carries even when terminal threads run first;
* active -> omitted -> active lane reuse with different packed offsets;
* production latent=3584/state=16 boundary channels, rows and fixed state tapes;
* reuse of reverse/A-prefix maps for local contributions and rate reduction;
* B/C latent summation and a cancellation fixture that distinguishes reverse
  rate summation from forward summation;
* positive relative accuracy for representable sigmoid tails (-20/-87/-88),
  rather than an absolute floor that would accept zero;
* truncated allocation/overlapping-write detection, disjoint row ownership,
  invalid bank/parameter sizes, metadata/grid overflow, and u32 flattened-index
  limits including every lane's extra terminal row;
* the reused memory PTX's one-token-per-block launch geometry, including an
  over-launched block, against full CPU loss/gradient reference tests;
* a forced failed resident-backward CPU replay retaining carries and existing
  accumulated gradients without advancing a lane twice.

An explicitly ignored CUDA-only unit test additionally checks the resident
allocation-swap path through repeated accumulation, carry reset, clipped AdamW,
zeroing and final gradients/moments/weights. It cannot run without real CUDA.

### Remote check attempts and blocker

Every command below was attempted through `/workspace/bin/csrun`. **Each wrapper
invocation exited 4 before executing its command**, printing:

```text
To get started with GitHub CLI, please run:  gh auth login -s codespace
```

The associated tar broken-pipe error is a consequence of that failure. The
shell loop collecting these attempts exited 0, but that is **not** a successful
build/test result; its per-command exits were all 4.

```sh
/workspace/bin/csrun "cargo check --release --features cuda --all-targets"
/workspace/bin/csrun "cargo build --release"
/workspace/bin/csrun "cargo test --release --lib"
/workspace/bin/csrun "cargo clippy --release --all-targets"
/workspace/bin/csrun "cargo build --release --examples"
/workspace/bin/csrun "cargo build --release --features cuda"
/workspace/bin/csrun "cargo test --release --features cuda --lib"
/workspace/bin/csrun "cargo clippy --release --features cuda --all-targets"
/workspace/bin/csrun "cargo build --release --features cuda --examples"
/workspace/bin/csrun "cargo test --release --features cuda --test training_safeguards cuda_ -- --nocapture"
/workspace/bin/csrun "cargo test --release --features cuda --test sequence_batch"
/workspace/bin/csrun "command -v ptxas && cargo test --release --test training_safeguards embedded_cuda_ptx_assembles_with_ptxas_when_available -- --nocapture"
/workspace/bin/csrun "cargo fmt --all -- --check"
```

The earlier baseline clippy attempt at 23d7bba was blocked by the same issue.
Therefore compilation, tests, example compilation, PTXAS validation and **no
new clippy warnings vs baseline are all UNVERIFIED**. The assembly test includes
`stages.ptx`, `safeguards.ptx` and `packed.ptx`, each at **sm_120/sm_90/sm_80**.
The `command -v ptxas` prerequisite intentionally prevents a missing-tool skip
from being called validation. Use a CUDA 12.8+ toolkit (or a compatible PTXAS
that supports all three targets); `PTXAS` can override the executable path.

Locally, only source/diff review and `git diff --check` were performed; the
latter passed. Free `/workspace` disk remained above 650 MiB (minimum required
300 MB). Re-run all remote checks once the wrapper's codespace access is
restored; do not substitute a local build for this gate.

### Exact molab GPU verification commands (not run in this sandbox)

Run these in the checked-out branch on the RTX PRO 6000 Blackwell host:

```sh
# Full example: existing depth/loop checks, both packed configurations,
# all gradients/carries, cold and warmed CPU-vs-GPU step seconds.
PSSA_REQUIRE_CUDA=1 cargo run --release --features cuda --example gpu_train_check

# Same numerical checks with host submission/readback stage timing.
PSSA_REQUIRE_CUDA=1 PSSA_CUDA_PROFILE=1 cargo run --release --features cuda --example gpu_train_check

# Require PTXAS to be present; assemble all three PTX files at all targets.
command -v ptxas
PTXAS="$(command -v ptxas)" cargo test --release --features cuda --test training_safeguards embedded_cuda_ptx_assembles_with_ptxas_when_available -- --nocapture

# Targeted resident optimizer transaction check, not a corpus training run.
cargo test --release --features cuda --lib cuda_packed_resident_gradient_transactions_match_cpu -- --ignored --exact cuda::packed::tests::cuda_packed_resident_gradient_transactions_match_cpu --nocapture
```

For comparisons, fix CPU thread count (e.g. `RAYON_NUM_THREADS=4`) and record
hardware/toolkit/driver versions. Run the unprofiled example repeatedly and
compare its warmed **step 1**, not cold step 0, using identical lane lengths,
packing order, widths and memory occupancy. The old example exercised only two
production-width lanes; its supplied 0.294/0.536-second numbers must **not** be
compared directly with the new all-32-lane measurement as a speedup. A fair
baseline needs the same all-lane diagnostic workload at 23d7bba. `PSSA_CUDA_PROFILE`
includes logging overhead and is for attribution, not final speed claims.

New profile labels: `packed.forward`, `packed.upload`,
`packed.forward.readback_wait`, `packed.backward`,
`packed.backward.readback_wait`, alongside exterior `gemm.*` and stage labels.
These remain host wall-clock timings, not CUDA events. A GPU trace should verify
57 explicit submissions/17 host waits and show no per-lane allocation/wait
cycles; cuBLAS may expand into multiple internal kernels.

### What remains unverified

1. **Required remote build/test/clippy/example gates**, including baseline
   warning comparison and driver-free emulator assertions. Source review is
   not a replacement for running them.
2. **Actual PTXAS/JIT acceptance** at all three architectures; assembly test
   wiring is present but did not execute.
3. **GPU numerical parity <1e-3** for both packed configs, every gradient/carry,
   lane repacking and device-resident carries; the supplied 9.7e-6 error is for
   the old 23d7bba path, not this implementation.
4. **Resident optimizer ownership swaps**, gradient accumulation after zeroing,
   clipping/AdamW and recovery under real CUDA driver failures.
5. **Speedup versus CPU and a fair old-CUDA baseline**, GPU launch/resource
   behavior, bank-occupancy scaling and peak VRAM. Resident storage trades
   higher retained VRAM for removal of per-lane churn; scans are serial in time
   per channel, and retrieval/VJP remains serial per token. Very long chunks or
   full banks may need further GPU tuning after measurement.
6. The entire model is **not** yet device-resident: adapter/MLP/head CPU tape
   boundaries still account for fifteen GEMM waits, and loss, embedding/norm,
   rate folding and memory insertion remain host-owned. Those are visible
   follow-up limits, not evidence that the requested speed target has passed.
