# CUDA packed-training findings

## Baseline traced before the packed-path change

This trace is for the checkout at `3d46578` (`cuda-packed`, before the
implementation in this work). The affected case is `pssa train ...
--backend cuda --batch-size 32` with depth 1 and one loop.

### Entry point and per-step dispatch

1. `src/cli.rs:650` (`Cli::train_corpus`) validates the training options and
   constructs `PSSALayerV2`.
2. `src/cli.rs:784-818` selects `TrainingBackend::Cuda`; the selected
   `Device::Cuda` is assigned to `model.device`. This does not select a
   different trainer for CUDA.
3. `src/training.rs:28-69` creates the independent-lane `sequence_plan`.
   Since `batch_size > 1`, `src/cli.rs:838-845` creates a
   `SequenceBatch`. The CPU fallback also uses this object; CUDA does not
   bypass it.
4. For each optimizer group, `src/cli.rs:964-977` clears gradients. For each
   microbatch, `src/cli.rs:980-998` creates borrowed `Sequence` views and calls
   `SequenceBatch::forward`; then `src/cli.rs:1024-1032` calls
   `SequenceBatch::backward`. Memory insertion is deliberately after all
   retrieval backwards at `src/cli.rs:1053-1070`, and AdamW follows at
   `src/cli.rs:1073` onward.
5. The depth-one, one-loop packed forward path is
   `src/sequence_batch.rs:258-344`:
   `stage_embed_norm` -> `stage_projections` -> one SSM operation per lane ->
   copy lane `y` rows into the packed tape -> `stage_memory_packed` ->
   `stage_adapter` -> `stage_mlp` -> `stage_logits_loss`.
6. Its backward path is `src/sequence_batch.rs:363-480`:
   `bwd_stage_logits` -> `bwd_stage_mlp` -> `bwd_stage_adapter` ->
   `bwd_stage_adapter_down` -> `bwd_stage_memory` -> one SSM backward per lane
   -> packed projection adjoints -> RMSNorm/embedding scatter.

Stacked depth or multiple loops takes the replay path at
`src/sequence_batch.rs:250-256` and `:359-361`; that is not the packed
single-block path audited here.

## What was already CUDA-backed in the baseline

The baseline was not entirely host-only. For a working CUDA context:

- `stage_projections` (`src/gpu_batch.rs:688-730`) sends the three projection
  GEMMs through cuBLAS and sends softplus through the CUDA elementwise kernel.
- `Lane::forward_gpu` (`src/sequence_batch.rs:592-626`) calls
  `GpuDispatch::ssm_forward`, which uses `ssm_prepare`, `affine_scan`,
  `scan_apply`, and `ssm_materialize` in `src/cuda/stages.ptx`. It did this one
  lane at a time and immediately read the result back to host memory.
- `stage_memory_packed` (`src/gpu_batch.rs:742-831`) calls
  `GpuDispatch::memory_forward`, which uses cuBLAS plus the existing
  `memory_forward`, `sigmoid`, and `sigmoid_mul` kernels. Its inputs and all
  outputs crossed the host boundary for every packed microbatch.
- `stage_adapter`, `stage_mlp`, `stage_logits_loss`, and their CUDA-aware
  backward variants use cuBLAS for their dense matrix products. Their
  elementwise nonlinearities, softmax/loss, and host tape bookkeeping remain
  on the CPU.
- `Lane::backward_gpu` (`src/sequence_batch.rs:628-670`) calls
  `GpuDispatch::ssm_backward`, which uses `ssm_backward_maps`, the existing
  affine scan kernels, and `ssm_backward_local`; its outputs are read back for
  lane aggregation.
- `bwd_stage_memory` (`src/gpu_batch.rs:2068-2334`) calls the existing CUDA
  `memory_backward_local` and `memory_backward` kernels and CUDA GEMMs when
  available. It still uploads host tapes and reads all query adjoints back;
  its shared-weight reductions and fallback implementation are host code.

## Forward work still on the CPU in the baseline CUDA packed path

The following work is CPU work even when all CUDA launches succeed:

- `stage_embed_norm` (`src/gpu_batch.rs:676-686`): embedding row gather and
  input RMSNorm.
- Packed lane scheduling, carry publication, host tape slicing, and copying
  each lane's CUDA SSM output into the packed `y_ssm` tape
  (`src/sequence_batch.rs:274-335`). The SSM arithmetic is CUDA on the success
  path, but there is no packed device-resident carry/tape across lanes and
  stages.
- The host-side synchronization/copies inside `GpuDispatch::ssm_forward` and
  `GpuDispatch::memory_forward`; these are especially costly at 32 lanes.
- Elementwise adapter SiLU in `stage_adapter`/`stage_mlp`, latent aggregation,
  MLP SiLU, and residual assembly (`src/gpu_batch.rs:921-1063`). Their GEMMs
  are CUDA, but their surrounding arithmetic is host-side.
- Softmax probabilities, cross-entropy, and loss reduction in
  `stage_logits_loss` (`src/gpu_batch.rs:1064-1118`). The logits GEMM is CUDA;
  the complete loss/probability loop is CPU.
- If any CUDA stage fails, the baseline explicitly warns and falls back to the
  host SSM scan (`Lane::forward` -> `affine_scan_in_place`) and/or the host
  memory projection, hyperbolic retrieval, sigmoid gate, and injection path
  (`src/gpu_batch.rs:832-918`).

## Backward work still on the CPU in the baseline CUDA packed path

- Cross-entropy adjoint construction (`bwd_stage_logits_blocked`,
  `src/gpu_batch.rs:1636-1651`) and all stage trace/tape orchestration.
- SiLU derivatives for the MLP and adapter (`src/gpu_batch.rs:1805-1810`
  and `:1944-1952`). The large matrix products may be CUDA, but these loops
  are host-side.
- Per-lane result copying, shared `a_mat` gradient reduction in reverse lane
  order, and packed gradient assembly (`src/sequence_batch.rs:377-428`).
- SSM rate-gradient aggregation (`Lane::backward_gpu` at
  `src/sequence_batch.rs:663-668`) and the final RMSNorm plus sparse embedding
  gradient scatter (`src/sequence_batch.rs:466-480`).
- `bwd_stage_memory`'s host-side shared-gradient bookkeeping and all of its
  retrieval math when `memory_backward_local`, `memory_backward`, or a dense
  CUDA operation fails. On a successful CUDA launch, the two retrieval kernels
  are CUDA, but each call still uploads host tapes and reads the result back;
  there is no packed device-resident backward memory tape.
- If the packed SSM backward launch fails, `SequenceBatch` warns and runs
  `Lane::backward`, whose recurrence is CPU (`affine_scan_in_place` for long
  lanes, ordered recurrence for short lanes).

## Main cause found

The packed path owned independent lane carries, so it could not safely call the
existing single-sequence resident API once for the concatenated token rows: a
normal scan would carry state from one lane into the next. The baseline avoided
that correctness issue by doing one host-visible SSM call per lane and then
running a separate packed memory call. That preserved numbers but forced
repeated device/host boundaries and left the packed path without an SSM-to-memory
resident handoff.

The implementation below keeps each lane's carry as the initial row of its own
CUDA scan, consumes that scan immediately with `memory_forward_after_ssm`, and
only then publishes the lane's host tape rows. CPU behavior and the explicit,
warning-bearing fallback remain unchanged.

## Implemented packed CUDA route

For the depth-one, one-loop path, `src/sequence_batch.rs` now does the following
when `Device::Cuda` is active:

- `stage_embed_norm` and `stage_projections` still prepare the packed host tape;
  their shared GEMMs and CUDA softplus use the existing dispatch.
- Each active lane calls `GpuDispatch::ssm_forward_resident` with its own initial
  carry. The existing `ssm_prepare`, `affine_scan`, `scan_apply`, and
  `ssm_materialize` kernels therefore scan only that lane and cannot leak state
  into a neighboring lane.
- The lane's resident SSM result is immediately consumed by
  `GpuDispatch::memory_forward_after_ssm`. This reuses the existing memory
  projection, `memory_forward`, cuBLAS value reduction, sigmoid, and sigmoid
  injection kernels without first materializing `y_ssm` on the host. The output
  tapes needed by later host stages and backward are then read back into the
  lane/packed tape, and the terminal state becomes that lane's next carry.
- A completely successful CUDA lane pass skips `stage_memory_packed`, avoiding a
  second memory computation. If any lane launch or readback fails, all carries
  are restored to their pre-forward snapshots, a warning is emitted, and every
  lane is replayed through the CPU SSM path before the ordinary packed memory
  fallback. This prevents a partial CUDA batch from advancing state twice.
- Backward still performs the packed logits/MLP/adapter orchestration, then
  `bwd_stage_memory` uses `memory_backward_local` and `memory_backward` (the
  retrieval VJP) plus CUDA GEMMs when available. Each independent lane then
  calls `GpuDispatch::ssm_backward`, which reuses `ssm_backward_maps`, the
  existing affine scan kernels, and `ssm_backward_local`. Shared gradients are
  reduced on the host in deterministic lane/token order.
- Any CUDA SSM backward failure emits a warning and recomputes all lanes through
  the existing CPU recurrence. Memory backward failures likewise warn and use
  the existing host VJP for the failed substage. No CUDA error is converted into
  an unannounced numerical result.

The scan kernel remains one independent sequence at a time: its existing PTX
ABI has no lane dimension. This is intentional for correctness with lane-owned
carries and reuses the validated kernels without adding a new PTX ABI. The
SSM-to-memory handoff is device-resident; the host still owns packed tape
bookkeeping, the later adapter/MLP/logit elementwise stages, and the tape
readbacks required by the current backward API.

## CPU work that remains after the change

The following is expected and explicit on the successful CUDA packed path:

- token-ID embedding gather, input RMSNorm, lane scheduling/carry publication,
  packed tape slicing, and final lane gradient reductions;
- adapter and MLP SiLU/residual elementwise work, softmax/cross-entropy and
  loss reduction, memory shared-gradient bookkeeping, SSM rate-gradient
  reduction, final RMSNorm, and sparse embedding scatter;
- host/device synchronization and readback of the lane tapes between the
  resident SSM/memory handoff and the host-owned later stages.

The large GEMMs, CUDA softplus, SSM forward/backward kernels, memory forward
kernels, memory backward kernels, and supported CUDA dense adjoints are no
longer replaced by the packed host scan/memory implementation merely because
there are multiple lanes. CPU devices retain the original staged/ordered path.

## Verification commands

All sandbox build/test commands were run through the required remote wrapper:

```text
/workspace/bin/csrun "cargo build --release"
/workspace/bin/csrun "cargo test --release --lib"
/workspace/bin/csrun "cargo clippy --release --all-targets"
/workspace/bin/csrun "cargo check --release --examples"
```

On the molab RTX PRO 6000 Blackwell box, run the packed CPU/GPU parity and timing
check (with the CUDA feature enabled) as:

```text
CUDA_VISIBLE_DEVICES=0 cargo run --release --features cuda --example gpu_train_check
```

The example checks packed forward loss, every exposed parameter gradient, and
each lane carry at the small configuration and at `d_vocab=2048,
d_latent=3584, d_state=16, d_mem_key=32, mem_capacity=512, depth=1,
loops=1, batch_size=32`; it also prints CPU and GPU step seconds/tokens per
second. Its production-shape reference uses two tokens per lane to keep the
CPU comparison bounded while preserving the production parameter dimensions.

## Unverified until a real CUDA run

This sandbox has no CUDA device, so the following remain unverified here:

- Blackwell PTX JIT/load success and numerical parity of the resident
  SSM-to-memory path, packed memory backward, and per-lane CUDA SSM backward;
- the `< 1e-3` production-shape loss/gradient/carry result and the reported
  CPU-vs-GPU timings on the molab hardware;
- actual utilization, memory residency, and whether the remaining host tape
  synchronization is the dominant bottleneck at batch size 32.

The CPU packed reference tests and the release build/checks do not substitute
for that real-GPU result.
