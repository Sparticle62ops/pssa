# Packed CUDA speed audit (baseline: 23d7bba)

## Baseline inventory — recorded before implementation

Scope: one successful `SequenceBatch::forward` + `backward`, depth=1,
loops=1, B active lanes, N packed tokens, each lane length <=1024. This is a
microbatch step, **not** AdamW/gradient clipping/memory insertion. No GPU exists
in this sandbox; the reported molab measurements are supplied baseline evidence,
not measurements made here: training widths CPU 0.294 s / CUDA 0.536 s; small
CPU 0.001 s / CUDA 0.038 s. Both examples include cold-start effects.

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

## Implementation and verification

Pending. The baseline remote clippy command was attempted before edits through
`/workspace/bin/csrun`; the wrapper exits 4 because GitHub CLI codespace access
is unauthenticated. No local cargo command or training run was started.
