//! Batched stage decomposition of the chunk training math.
//!
//! The historical `forward_train_chunk` / `backward_chunk` walk the chunk one
//! token at a time, issuing one matvec per weight matrix per token. This module
//! re-expresses the exact same math as stage functions over the whole chunk so
//! every weight-touching stage is a single batched GEMM over all L tokens in
//! the chunk. That stage structure is what the wgpu GPU dispatch in
//! `backend.rs` maps one-to-one onto compute kernels; on CPU these functions
//! ARE the dispatch path, and they are the allocation-free twin used to verify
//! the GPU kernels numerically (`cpu-twin check`).
//!
//! Numerical contract with the reference implementations
//! (`PSSALayerV2::forward_train_chunk` / `backward_chunk`): results agree to
//! f32 roundoff, not bitwise equality. The affine SSM scan reassociates time
//! and projection input adjoints group sums by matrix. Shared SSM weight
//! gradients still accumulate in descending token order. Nonlinear memory
//! retrieval, bank writes and refractory updates are NOT affine scan operators.

use crate::linalg::{dot_slice, sigmoid, softplus};
use crate::memory::HyperbolicEpisodicBankV2;
use crate::pssa::{PSSAContinuousBlockV2, PSSALayerV2};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Stage module failures are already reported with the driver's JIT log by the
/// CUDA loader. Suppress repeated fallback chatter for that permanent error;
/// other transient stage failures are also reported only once per message.
pub(crate) fn warn_gpu_fallback_once(error: &str, message: String) {
    if error.contains("stage module failed to load") {
        return;
    }
    static REPORTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mut reported = REPORTED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if reported.insert(message.clone()) {
        eprintln!("{message}");
    }
}

// Parallelize large, independent dense adjoints without shared atomics.
fn blocked_backward(tokens: usize, rows: usize, cols: usize) -> bool {
    tokens.saturating_mul(rows).saturating_mul(cols) >= 1_048_576
}

fn parallel_backward(tokens: usize, rows: usize, cols: usize) -> bool {
    blocked_backward(tokens, rows, cols) && rayon::current_num_threads() > 1
}

// Retrieval is independent per token while the bank is read-only. Avoid
// waking Rayon for tiny probes, but spread training-sized linear scans over
// all available workers.
fn parallel_memory_work(tokens: usize, count: usize, key: usize, value: usize) -> bool {
    tokens
        .saturating_mul(count)
        .saturating_mul(key.saturating_add(value))
        >= 1_048_576
        && rayon::current_num_threads() > 1
}

/// The tree scan has more arithmetic and synchronization than the sequential
/// recurrence at the short chunk lengths used by the trainer. Keep that fast
/// path allocation-free and serial; only enable the scan once there is enough
/// time parallelism to amortize the tree and rayon scheduling overhead.
const PARALLEL_SCAN_MIN_LEN: usize = 256;

pub(crate) fn parallel_scan_enabled(len: usize, _stride: usize) -> bool {
    len >= PARALLEL_SCAN_MIN_LEN && rayon::current_num_threads() > 1
}

/// Compose two diagonal affine maps. `first` is applied before `second`:
///
/// ```text
/// x -> first_a*x + first_b -> second_a*x + second_b
/// ```
///
/// The SSM has one such scalar map for every latent/state channel, so maps at
/// different channels are independent and the composition is associative.
#[inline(always)]
fn compose_affine(first_a: f32, first_b: f32, second_a: f32, second_b: f32) -> (f32, f32) {
    (second_a * first_a, second_a * first_b + second_b)
}

/// Work-efficient Blelloch exclusive scan of diagonal affine maps.
///
/// The first `len` rows of `scan_a`/`scan_b` contain the per-token maps on
/// entry. The remaining rows are padded with the identity map. Each tree
/// level operates on disjoint time intervals and is therefore parallelized
/// with rayon; only the O(log T) tree levels remain sequential. The result at
/// row `t` is the composition of tokens before `t`.
///
/// The caller owns the workspaces so this remains allocation-free in both the
/// single-sequence stage path and the independent-lane sequence batch path.
pub(crate) fn affine_scan_in_place(
    scan_a: &mut [f32],
    scan_b: &mut [f32],
    len: usize,
    stride: usize,
) {
    assert!(len > 0 && stride > 0);
    let width = len.next_power_of_two();
    assert!(scan_a.len() >= width * stride && scan_b.len() >= width * stride);
    let a = &mut scan_a[..width * stride];
    let b = &mut scan_b[..width * stride];
    a[len * stride..]
        .par_chunks_mut(stride)
        .for_each(|row| row.fill(1.0));
    b[len * stride..].fill(0.0);

    // Up-sweep: reduce each subtree to its right-hand endpoint.
    let mut step = 1;
    while step < width {
        let chunk_len = 2 * step * stride;
        a.par_chunks_mut(chunk_len)
            .zip(b.par_chunks_mut(chunk_len))
            .for_each(|(a_chunk, b_chunk)| {
                let left = (step - 1) * stride;
                let right = (2 * step - 1) * stride;
                for j in 0..stride {
                    let (next_a, next_b) = compose_affine(
                        a_chunk[left + j],
                        b_chunk[left + j],
                        a_chunk[right + j],
                        b_chunk[right + j],
                    );
                    a_chunk[right + j] = next_a;
                    b_chunk[right + j] = next_b;
                }
            });
        step *= 2;
    }

    // The exclusive root prefix is the identity.
    a[(width - 1) * stride..].fill(1.0);
    b[(width - 1) * stride..].fill(0.0);

    // Down-sweep: distribute each parent prefix to its two children.
    let mut step = width / 2;
    while step > 0 {
        let chunk_len = 2 * step * stride;
        a.par_chunks_mut(chunk_len)
            .zip(b.par_chunks_mut(chunk_len))
            .for_each(|(a_chunk, b_chunk)| {
                let left = (step - 1) * stride;
                let right = (2 * step - 1) * stride;
                for j in 0..stride {
                    let parent_a = a_chunk[right + j];
                    let parent_b = b_chunk[right + j];
                    let left_a = a_chunk[left + j];
                    let left_b = b_chunk[left + j];
                    a_chunk[left + j] = parent_a;
                    b_chunk[left + j] = parent_b;
                    let (next_a, next_b) = compose_affine(parent_a, parent_b, left_a, left_b);
                    a_chunk[right + j] = next_a;
                    b_chunk[right + j] = next_b;
                }
            });
        step /= 2;
    }
}

/// G[L,R] * W[R,C], owning complete output rows with contiguous weight reads.
pub(crate) fn dense_input_adjoint(
    g: &[f32],
    w: &[f32],
    l: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    // Reuse each contiguous weight row across four tokens before advancing.
    // Per-element summation order is unchanged; tasks own complete output tiles.
    let tile = |tile_idx: usize, dst: &mut [f32]| {
        dst.fill(0.0);
        for r in 0..rows {
            let weights = &w[r * cols..(r + 1) * cols];
            for (t, output) in dst.chunks_mut(cols).enumerate() {
                let scale = g[(tile_idx * 4 + t) * rows + r];
                for (dst, &weight) in output.iter_mut().zip(weights) {
                    *dst += scale * weight;
                }
            }
        }
    };
    if parallel_backward(l, rows, cols) {
        out.par_chunks_mut(4 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    } else {
        out.chunks_mut(4 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    }
}

/// dW[R,C] += G[L,R]^T * X[L,C]. Each task owns rows, accumulating tokens
/// in reverse order just like the reference TBPTT path (including existing grads).
pub(crate) fn dense_weight_adjoint(
    g: &[f32],
    x: &[f32],
    l: usize,
    rows: usize,
    cols: usize,
    grad: &mut [f32],
) {
    let tile = |tile_idx: usize, dst: &mut [f32]| {
        let first_row = tile_idx * 8;
        for t in (0..l).rev() {
            let input = &x[t * cols..(t + 1) * cols];
            for (r, output) in dst.chunks_mut(cols).enumerate() {
                let scale = g[t * rows + first_row + r];
                for (dst, &value) in output.iter_mut().zip(input) {
                    *dst += scale * value;
                }
            }
        }
    };
    if parallel_backward(l, rows, cols) {
        grad.par_chunks_mut(8 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    } else {
        grad.chunks_mut(8 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    }
}

/// Weight adjoint with forward token order. Adapter gradients are accumulated
/// in forward-token order by the scalar reference, so this variant preserves
/// each output element's f32 reduction order while parallelizing independent
/// weight rows for large packed batches.
pub(crate) fn dense_weight_adjoint_forward(
    g: &[f32],
    x: &[f32],
    l: usize,
    rows: usize,
    cols: usize,
    grad: &mut [f32],
) {
    let tile = |tile_idx: usize, dst: &mut [f32]| {
        let first_row = tile_idx * 8;
        for t in 0..l {
            let input = &x[t * cols..(t + 1) * cols];
            for (r, output) in dst.chunks_mut(cols).enumerate() {
                let scale = g[t * rows + first_row + r];
                for (dst, &value) in output.iter_mut().zip(input) {
                    *dst += scale * value;
                }
            }
        }
    };
    if parallel_backward(l, rows, cols) {
        grad.par_chunks_mut(8 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    } else {
        grad.chunks_mut(8 * cols)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    }
}

/// Adapter up-projection input adjoint, including the consolidated slow copy.
/// Each token owns one output row, and its channel reduction retains the same
/// i-major order as `PlasticAdapterV2::total_up_matvec_transpose`.
fn adapter_up_input_adjoint(
    g: &[f32],
    fast: &[f32],
    slow: &[f32],
    l: usize,
    d_m: usize,
    rank: usize,
    out: &mut [f32],
) {
    let tile = |tile_idx: usize, dst: &mut [f32]| {
        dst.fill(0.0);
        let first_token = tile_idx * 4;
        for i in 0..d_m {
            let fast_row = &fast[i * rank..(i + 1) * rank];
            let slow_row = &slow[i * rank..(i + 1) * rank];
            for (local_t, row) in dst.chunks_mut(rank).enumerate() {
                let t = first_token + local_t;
                if t >= l {
                    break;
                }
                let scale = g[t * d_m + i];
                for r in 0..rank {
                    row[r] += scale * (fast_row[r] + slow_row[r]);
                }
            }
        }
    };
    if parallel_backward(l, d_m, rank) {
        out.par_chunks_mut(4 * rank)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    } else {
        out.chunks_mut(4 * rank)
            .enumerate()
            .for_each(|(i, dst)| tile(i, dst));
    }
}

/// Batched matvec over L rows: `out[t, r] = dot(W[r, :], x[t, :])`.
/// Same accumulation order as `ParamMatrix::matvec`, applied per row.
#[inline(always)]
fn batched_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], l: usize, out: &mut [f32]) {
    crate::backend::gemm_cpu_into(x, w, l, rows, cols, 1, out)
        .expect("validated model GEMM dimensions");
}

/// Clone the layer's GPU context out, so stage functions can keep borrowing
/// tape fields while the dispatch runs. `None` on the CPU path.
#[inline]
fn gpu_ctx(m: &PSSALayerV2) -> Option<crate::backend::GpuDispatch> {
    m.device.gpu()
}

/// Device-aware batched matvec: on a GPU device this is one `dispatch_gemm`
/// call (X [1,L,K], W [rows,K], Y [1,L,rows]). L is the total number of
/// packed token rows, including independent sequences when present. Never use
/// M=1, batch=L: that wastes 15/16 of WebGPU's row tile and gives CUDA GEMVs.
/// The backends also fold legacy shared-weight batches at their boundary.
#[inline]
fn batched_matvec_dev(
    gpu: Option<&crate::backend::GpuDispatch>,
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    l: usize,
    out: &mut [f32],
) {
    if let Some(ctx) = gpu {
        ctx.dispatch_gemm_into(&x[..l * cols], w, l, rows, cols, 1, out)
            .expect("validated model GEMM dimensions");
    } else {
        batched_matvec(w, rows, cols, x, l, out);
    }
}

/// Device-aware row-major C(M,N) = A(M,K) * B(K,N), reusing model scratch.
#[inline]
fn gemm_nn_dev_into(
    gpu: Option<&crate::backend::GpuDispatch>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    match gpu {
        Some(ctx) => ctx
            .gemm_nn_into(a, b, m, k, n, out)
            .expect("validated model GEMM dimensions"),
        None => crate::backend::gemm_nn_cpu_into(a, b, m, k, n, out)
            .expect("validated model GEMM dimensions"),
    }
}

/// Device-aware row-major C(K,N) = A(M,K)^T * B(M,N), accumulated into an
/// existing gradient so optimizer-group accumulation remains intact.
#[inline]
fn gemm_tn_dev_accumulate(
    gpu: Option<&crate::backend::GpuDispatch>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    match gpu {
        Some(ctx) => ctx
            .gemm_tn_accumulate_into(a, b, m, k, n, out)
            .expect("validated model GEMM dimensions"),
        None => crate::backend::gemm_tn_cpu_accumulate_into(a, b, m, k, n, out)
            .expect("validated model GEMM dimensions"),
    }
}

/// Materialize SSM states and outputs from an exclusive affine scan. `local_*`
/// are the token maps, while `scan_*` contain the exclusive prefix maps. The
/// state update and readout are independent for each time row once the scan is
/// complete, so this pass is also parallel over time.
pub(crate) fn materialize_ssm_scan(
    initial: &[f32],
    local_a: &[f32],
    local_b: &[f32],
    x_norm: &[f32],
    scan_a: &[f32],
    scan_b: &[f32],
    c_proj: &[f32],
    states_out: &mut [f32],
    y_out: &mut [f32],
    len: usize,
    d_m: usize,
    d_s: usize,
) {
    let stride = d_m * d_s;
    assert_eq!(initial.len(), stride);
    assert_eq!(local_a.len(), len * stride);
    assert_eq!(local_b.len(), len * stride);
    assert_eq!(x_norm.len(), len * d_m);
    assert_eq!(scan_a.len(), len * stride);
    assert_eq!(scan_b.len(), len * stride);
    assert_eq!(c_proj.len(), len * d_s);
    assert_eq!(states_out.len(), len * stride);
    assert_eq!(y_out.len(), len * d_m);

    states_out
        .par_chunks_mut(stride)
        .zip(y_out.par_chunks_mut(d_m))
        .enumerate()
        .for_each(|(t, (state_row, y_row))| {
            let state_off = t * stride;
            let c_off = t * d_s;
            for i in 0..d_m {
                let mut y_i = 0.0f32;
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let h_before = scan_a[state_off + idx] * initial[idx] + scan_b[state_off + idx];
                    let h = local_a[state_off + idx] * h_before
                        + local_b[state_off + idx] * x_norm[t * d_m + i];
                    state_row[idx] = h;
                    y_i += h * c_proj[c_off + j];
                }
                y_row[i] = y_i;
            }
        });
}

// =============================================================================
// FORWARD STAGES
// =============================================================================

/// Stage 1: embedding gather for every token, followed by affine RMSNorm.
#[inline]
pub fn stage_embed_norm(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "forward.embed_norm", seq_len);
    let (embed_w, m) = (&m.embed_w, &mut m.block);
    let d_m = m.cfg.d_latent;
    for t in 0..seq_len {
        let x_id = m.tape.x_ids[t];
        let e_t = &embed_w.data[x_id * d_m..(x_id + 1) * d_m];
        m.tape.x_raw[t * d_m..(t + 1) * d_m].copy_from_slice(e_t);
    }
    stage_input_norm_block(m, seq_len);
}

#[inline]
pub(crate) fn stage_input_norm_block(m: &mut PSSAContinuousBlockV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    for t in 0..seq_len {
        let raw_off = t * d_m;
        let e_t = &m.tape.x_raw[raw_off..raw_off + d_m];
        let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();
        m.tape.inv_rms[t] = inv_rms;
        for i in 0..d_m {
            m.tape.x_norm[raw_off + i] =
                m.norm_gamma.data[i] * (e_t[i] * inv_rms) + m.norm_beta.data[i];
        }
    }
}

/// Stage 2: data-dependent projections (delta/b/c/gate) as batched GEMMs,
/// plus the softplus activation on the raw delta.
#[inline]
pub fn stage_projections(m: &mut PSSALayerV2, seq_len: usize) {
    assert!(!m.local_mixing_enabled(), "local mixing requires the full CPU forward entry point");
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "forward.projections", seq_len);
    let gpu = gpu_ctx(m);
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let l = seq_len;
    let xn = &m.tape.x_norm[..l * d_m];

    batched_matvec_dev(
        gpu.as_ref(),
        &m.w_delta.data,
        d_m,
        d_m,
        xn,
        l,
        &mut m.tape.delta_raw[..l * d_m],
    );
    // Reference keeps the raw projection in `delta_raw` and the softplus in
    // `delta`; the backward pass takes sigmoid(delta_raw), so both are needed.
    if let Some(gpu) = gpu.as_ref() {
        m.tape.delta[..l * d_m].copy_from_slice(&m.tape.delta_raw[..l * d_m]);
        if let Err(error) = gpu.softplus_in_place(&mut m.tape.delta[..l * d_m]) {
            warn_gpu_fallback_once(
                &error,
                format!("warning: GPU softplus failed; using CPU elementwise path: {error}"),
            );
            for i in 0..l * d_m {
                m.tape.delta[i] = softplus(m.tape.delta_raw[i]);
            }
        }
    } else {
        for i in 0..l * d_m {
            m.tape.delta[i] = softplus(m.tape.delta_raw[i]);
        }
    }
    batched_matvec_dev(
        gpu.as_ref(),
        &m.w_b.data,
        d_s,
        d_m,
        xn,
        l,
        &mut m.tape.b_proj[..l * d_s],
    );
    batched_matvec_dev(
        gpu.as_ref(),
        &m.w_c.data,
        d_s,
        d_m,
        xn,
        l,
        &mut m.tape.c_proj[..l * d_s],
    );
}

/// The original ordered recurrence is the fast path for short chunks. It is
/// deliberately separate from the scalar PSSA reference implementation: the
/// staged path still computes the same tape fields, but avoids a tree, rayon
/// jobs, and their per-dispatch bookkeeping when parallelism cannot pay back.
#[inline]
fn stage_ssm_scan_sequential(m: &mut PSSALayerV2, seq_len: usize) {
    let m = &mut m.block;
    m.refresh_ssm_rates();
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    for t in 0..seq_len {
        let del_off = t * d_m;
        let b_off = t * d_s;
        let c_off = t * d_s;
        let h_prev_off = t * (d_m * d_s);
        let h_next_off = (t + 1) * (d_m * d_s);
        let ssm_off = t * (d_m * d_s);
        let y_off = t * d_m;
        let xn_off = t * d_m;

        for i in 0..d_m {
            let d_i = m.tape.delta[del_off + i];
            let mut y_i = 0.0f32;
            for j in 0..d_s {
                let idx = i * d_s + j;
                let bar_a = (d_i * m.ssm_rates[idx]).exp();
                let bar_b = d_i * m.tape.b_proj[b_off + j];
                m.tape.bar_a[ssm_off + idx] = bar_a;
                m.tape.bar_b[ssm_off + idx] = bar_b;
                let h_val =
                    bar_a * m.tape.h_states[h_prev_off + idx] + bar_b * m.tape.x_norm[xn_off + i];
                m.tape.h_states[h_next_off + idx] = h_val;
                y_i += h_val * m.tape.c_proj[c_off + j];
            }
            m.tape.y_ssm[y_off + i] = y_i;
        }
    }
}

/// Stage 3: multi-channel SSM recurrent scan. Each channel is a diagonal
/// affine recurrence, so its `(bar_a, bar_b*x)` maps compose associatively.
/// The Blelloch tree parallelizes time while retaining the local maps in the
/// tape for the unchanged backward oracle.
#[inline]
pub fn stage_ssm_scan(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace = crate::training_diagnostics::StageTrace::new(&m.device, "forward.ssm", seq_len);
    if let Some(gpu) = gpu_ctx(m) {
        let d_m = m.cfg.d_latent;
        let d_s = m.cfg.d_state;
        m.block.refresh_ssm_rates();
        let b = &mut m.block;
        let result = gpu.ssm_forward_resident(
            &b.tape.delta[..seq_len * d_m],
            &b.tape.delta_raw[..seq_len * d_m],
            &b.tape.b_proj[..seq_len * d_s],
            &b.tape.x_norm[..seq_len * d_m],
            &b.ssm_rates,
            &b.tape.c_proj[..seq_len * d_s],
            &b.h_persistent,
            seq_len,
            d_m,
            d_s,
        );
        if let Err(error) = result {
            warn_gpu_fallback_once(
                &error,
                format!("warning: GPU SSM forward failed; using CPU scan: {error}"),
            );
        } else {
            return;
        }
    }
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let stride = d_m * d_s;
    if !parallel_scan_enabled(seq_len, stride) {
        stage_ssm_scan_sequential(m, seq_len);
        return;
    }
    let executor = m.scan_executor.clone();
    executor.run(|| stage_ssm_scan_parallel(m, seq_len));
}

#[inline]
fn stage_ssm_scan_parallel(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let stride = d_m * d_s;
    let m = &mut m.block;
    m.refresh_ssm_rates();

    // Build the independent per-token affine maps in parallel. The scan is
    // below; keeping bar_a/bar_b local preserves the tape contract for
    // backward while scan_b holds the input-dependent affine translation.
    let rates = &m.ssm_rates;
    let delta = &m.tape.delta;
    let b_proj = &m.tape.b_proj;
    let x_norm = &m.tape.x_norm;
    m.tape
        .bar_a
        .par_chunks_mut(stride)
        .zip(m.tape.bar_b.par_chunks_mut(stride))
        .zip(m.ssm_scan_b.par_chunks_mut(stride))
        .take(seq_len)
        .enumerate()
        .for_each(|(t, ((a_row, b_row), scan_b_row))| {
            let del_off = t * d_m;
            let b_off = t * d_s;
            let xn_off = t * d_m;
            for i in 0..d_m {
                let d_i = delta[del_off + i];
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let bar_a = (d_i * rates[idx]).exp();
                    let bar_b = d_i * b_proj[b_off + j];
                    a_row[idx] = bar_a;
                    b_row[idx] = bar_b;
                    scan_b_row[idx] = bar_b * x_norm[xn_off + i];
                }
            }
        });

    m.ssm_scan_a[..seq_len * stride].copy_from_slice(&m.tape.bar_a[..seq_len * stride]);
    affine_scan_in_place(&mut m.ssm_scan_a, &mut m.ssm_scan_b, seq_len, stride);
    let scan_a = &m.ssm_scan_a[..seq_len * stride];
    let scan_b = &m.ssm_scan_b[..seq_len * stride];
    let tape = &mut m.tape;
    let (h0, h_out) = tape.h_states.split_at_mut(stride);
    materialize_ssm_scan(
        h0,
        &tape.bar_a[..seq_len * stride],
        &tape.bar_b[..seq_len * stride],
        &tape.x_norm[..seq_len * d_m],
        scan_a,
        scan_b,
        &tape.c_proj[..seq_len * d_s],
        &mut h_out[..seq_len * stride],
        &mut tape.y_ssm[..seq_len * d_m],
        seq_len,
        d_m,
        d_s,
    );
}

#[inline]
fn retrieve_memory_row(
    memory: &HyperbolicEpisodicBankV2,
    q_euc: &[f32],
    q_poincare: &mut [f32],
    q_norm: &mut f32,
    value: &mut [f32],
    weights: &mut [f32],
    tau: f32,
) {
    *q_norm = HyperbolicEpisodicBankV2::diffeomorphic_project(q_euc, q_poincare);
    memory.retrieve_soft_into(q_poincare, tau, value, weights);
}

fn retrieve_memory_rows(
    memory: &HyperbolicEpisodicBankV2,
    q_euc: &[f32],
    q_poincare: &mut [f32],
    q_norm: &mut [f32],
    values: &mut [f32],
    weights: &mut [f32],
    seq_len: usize,
    d_key: usize,
    d_value: usize,
    capacity: usize,
    tau: f32,
    parallel: bool,
) {
    assert_eq!(q_euc.len(), seq_len * d_key);
    assert_eq!(q_poincare.len(), seq_len * d_key);
    assert_eq!(q_norm.len(), seq_len);
    assert_eq!(values.len(), seq_len * d_value);
    assert_eq!(weights.len(), seq_len * capacity);
    if parallel {
        q_euc
            .par_chunks(d_key)
            .zip(q_poincare.par_chunks_mut(d_key))
            .zip(q_norm.par_iter_mut())
            .zip(values.par_chunks_mut(d_value))
            .zip(weights.par_chunks_mut(capacity))
            .for_each(|((((q_euc, q_poincare), q_norm), value), weights)| {
                retrieve_memory_row(memory, q_euc, q_poincare, q_norm, value, weights, tau);
            });
    } else {
        q_euc
            .chunks(d_key)
            .zip(q_poincare.chunks_mut(d_key))
            .zip(q_norm.iter_mut())
            .zip(values.chunks_mut(d_value))
            .zip(weights.chunks_mut(capacity))
            .for_each(|((((q_euc, q_poincare), q_norm), value), weights)| {
                retrieve_memory_row(memory, q_euc, q_poincare, q_norm, value, weights, tau);
            });
    }
}

/// Stage 4: Poincare query projection, diffeomorphic projection, soft
/// retrieval, memory gate and injection for every token. Nonlinear retrieval
/// stays outside the affine scan and reads an immutable bank. Protected writes
/// (including the history-dependent refractory gate) remain ordered in the
/// trainer, after all backwards using this bank have completed.
#[inline]
pub fn stage_memory(m: &mut PSSALayerV2, seq_len: usize) {
    stage_memory_impl(m, seq_len, true);
}

/// Packed SSM carries are lane-owned; there is no resident full-batch scan.
#[inline]
pub(crate) fn stage_memory_packed(m: &mut PSSALayerV2, seq_len: usize) {
    stage_memory_impl(m, seq_len, false);
}

fn stage_memory_impl(m: &mut PSSALayerV2, seq_len: usize, resident_ssm: bool) {
    let _trace = crate::training_diagnostics::StageTrace::new(&m.device, "forward.memory", seq_len);
    let gpu = gpu_ctx(m);
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_k = m.cfg.d_mem_key;
    let mem_cap = m.cfg.mem_capacity;

    if let Some(gpu) = gpu.as_ref() {
        let resident_error = if resident_ssm {
            match gpu.memory_forward_after_ssm(
                &m.tape.x_norm[..seq_len * d_m],
                &m.w_qx.data,
                &m.w_qh.data,
                &m.w_gate.data,
                &m.w_proj.data,
                &m.memory.keys,
                &m.memory.norm_sq,
                &m.memory.values,
                seq_len,
                d_m,
                d_k,
                d_m,
                mem_cap,
                m.memory.count,
                m.cfg.tau_mem,
                &mut m.tape.bar_a[..seq_len * d_m * m.cfg.d_state],
                &mut m.tape.bar_b[..seq_len * d_m * m.cfg.d_state],
                &mut m.tape.h_states[..(seq_len + 1) * d_m * m.cfg.d_state],
                &mut m.tape.y_ssm[..seq_len * d_m],
                &mut m.tape.q_euc[..seq_len * d_k],
                &mut m.tape.q_poincare[..seq_len * d_k],
                &mut m.tape.q_norm[..seq_len],
                &mut m.tape.mem_weights[..seq_len * mem_cap],
                &mut m.tape.m_val[..seq_len * d_m],
                &mut m.tape.g_mem[..seq_len * d_m],
                &mut m.tape.m_proj[..seq_len * d_m],
                &mut m.tape.m_inj[..seq_len * d_m],
            ) {
                Ok(()) => return,
                Err(error) => Some(error),
            }
        } else {
            None
        };
        // SequenceBatch publishes a packed host y_ssm, not the last lane's
        // resident device result. Dispatch it directly instead of consuming a
        // differently shaped resident result and relying on an error fallback.
        let direct = gpu.memory_forward(
            &m.tape.x_norm[..seq_len * d_m],
            &m.tape.y_ssm[..seq_len * d_m],
            &m.w_qx.data,
            &m.w_qh.data,
            &m.w_gate.data,
            &m.w_proj.data,
            &m.memory.keys,
            &m.memory.norm_sq,
            &m.memory.values,
            seq_len,
            d_m,
            d_k,
            d_m,
            mem_cap,
            m.memory.count,
            m.cfg.tau_mem,
            &mut m.tape.q_euc[..seq_len * d_k],
            &mut m.tape.q_poincare[..seq_len * d_k],
            &mut m.tape.q_norm[..seq_len],
            &mut m.tape.mem_weights[..seq_len * mem_cap],
            &mut m.tape.m_val[..seq_len * d_m],
            &mut m.tape.g_mem[..seq_len * d_m],
            &mut m.tape.m_proj[..seq_len * d_m],
            &mut m.tape.m_inj[..seq_len * d_m],
        );
        if let Err(direct_error) = direct {
            warn_gpu_fallback_once(
                &direct_error,
                format!(
                    "warning: GPU memory forward failed; using host retrieval: {direct_error}; resident path: {resident_error:?}"
                ),
            );
        } else {
            return;
        }
    }

    if let Some(gpu) = gpu.as_ref() {
        // Query projection is two shared-weight GEMMs. The second result uses
        // the backward query workspace as a temporary before the CPU-only
        // hyperbolic projection and retrieval stages.
        batched_matvec_dev(
            Some(gpu),
            &m.w_qx.data,
            d_k,
            d_m,
            &m.tape.x_norm[..seq_len * d_m],
            seq_len,
            &mut m.tape.q_euc[..seq_len * d_k],
        );
        batched_matvec_dev(
            Some(gpu),
            &m.w_qh.data,
            d_k,
            d_m,
            &m.tape.y_ssm[..seq_len * d_m],
            seq_len,
            &mut m.bwd_g_query_euc[..seq_len * d_k],
        );
        for (q, h) in m.tape.q_euc[..seq_len * d_k]
            .iter_mut()
            .zip(&m.bwd_g_query_euc[..seq_len * d_k])
        {
            *q += h;
        }
    } else {
        for t in 0..seq_len {
            let xn_off = t * d_m;
            let y_off = t * d_m;
            let q_off = t * d_k;
            for r in 0..d_k {
                let row_x = &m.w_qx.data[r * d_m..(r + 1) * d_m];
                let row_h = &m.w_qh.data[r * d_m..(r + 1) * d_m];
                m.tape.q_euc[q_off + r] = dot_slice(row_x, &m.tape.x_norm[xn_off..xn_off + d_m])
                    + dot_slice(row_h, &m.tape.y_ssm[y_off..y_off + d_m]);
            }
        }
    }
    let parallel = parallel_memory_work(seq_len, m.memory.count, d_k, d_m);
    retrieve_memory_rows(
        &m.memory,
        &m.tape.q_euc[..seq_len * d_k],
        &mut m.tape.q_poincare[..seq_len * d_k],
        &mut m.tape.q_norm[..seq_len],
        &mut m.tape.m_val[..seq_len * d_m],
        &mut m.tape.mem_weights[..seq_len * mem_cap],
        seq_len,
        d_k,
        d_m,
        mem_cap,
        m.cfg.tau_mem,
        parallel,
    );

    // Gate, memory projection and injection as batched GEMMs over the chunk.
    batched_matvec_dev(
        gpu.as_ref(),
        &m.w_gate.data,
        d_m,
        d_m,
        &m.tape.x_norm[..seq_len * d_m],
        seq_len,
        &mut m.tape.g_mem[..seq_len * d_m],
    );
    for v in &mut m.tape.g_mem[..seq_len * d_m] {
        *v = sigmoid(*v);
    }
    batched_matvec_dev(
        gpu.as_ref(),
        &m.w_proj.data,
        d_m,
        d_m,
        &m.tape.m_val[..seq_len * d_m],
        seq_len,
        &mut m.tape.m_proj[..seq_len * d_m],
    );
    for t in 0..seq_len {
        let m_off = t * d_m;
        for i in 0..d_m {
            m.tape.m_inj[m_off + i] = m.tape.g_mem[m_off + i] * m.tape.m_proj[m_off + i];
        }
    }
}

/// Stage 5: zero-init plastic adapter (down projection, SiLU, up projection)
/// over the whole chunk.
#[inline]
pub fn stage_adapter(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "forward.adapter", seq_len);
    let gpu = gpu_ctx(m);
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let down = &m.adapters[0].down_proj.data;
    if let Some(gpu) = gpu.as_ref() {
        batched_matvec_dev(
            Some(gpu),
            down,
            rank,
            d_m,
            &m.tape.x_norm[..seq_len * d_m],
            seq_len,
            &mut m.tape.adapter_hidden[..seq_len * rank],
        );
    } else {
        for t in 0..seq_len {
            let x_off = t * d_m;
            let out_off = t * rank;
            for r in 0..rank {
                m.tape.adapter_hidden[out_off + r] = dot_slice(
                    &down[r * d_m..(r + 1) * d_m],
                    &m.tape.x_norm[x_off..x_off + d_m],
                );
            }
        }
    }
}

/// Adapter up-projection contribution for token t, written into `out`
/// (`out[i] = sum_r (U_fast + U_slow)[i, r] * act[r]`).
#[inline(always)]
fn adapter_up_into(
    ad: &crate::adapter::PlasticAdapterV2,
    d_m: usize,
    act: &[f32],
    out: &mut [f32],
) {
    let rank = ad.rank;
    for i in 0..d_m {
        let off = i * rank;
        let mut total = 0.0f32;
        for r in 0..rank {
            total += (ad.up_proj.data[off + r] + ad.consolidated_up[off + r]) * act[r];
        }
        out[i] = total;
    }
}

/// Stage 6: latent aggregation and SiLU MLP expansion, batched.
#[inline]
pub fn stage_mlp(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace = crate::training_diagnostics::StageTrace::new(&m.device, "forward.mlp", seq_len);
    let gpu = gpu_ctx(m);
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_mlp = d_m * 2;
    let ssm_scale = 1.0 / (m.cfg.d_state as f32).sqrt();

    // Use the model-owned scratch buffer rather than a fixed-size temporary.
    // The CLI permits latent widths larger than 256, and the old stack array
    // indexed past its end for those otherwise valid configurations.
    for t in 0..seq_len {
        let ad_off = t * m.adapters[0].rank;
        for r in 0..m.adapters[0].rank {
            let h = m.tape.adapter_hidden[ad_off + r];
            m.tape.adapter_act[ad_off + r] = h * sigmoid(h);
        }
    }
    if let Some(gpu) = gpu.as_ref() {
        let rank = m.adapters[0].rank;
        m.refresh_adapter_up_effective();
        // Reuse bwd_g_zraw as a packed adapter output scratch. It is filled
        // again by the backward pass after this forward stage completes.
        batched_matvec_dev(
            Some(gpu),
            &m.adapter_up_effective,
            d_m,
            rank,
            &m.tape.adapter_act[..seq_len * rank],
            seq_len,
            &mut m.bwd_g_zraw[..seq_len * d_m],
        );
        for t in 0..seq_len {
            let z_off = t * d_m;
            for i in 0..d_m {
                m.tape.z_raw[z_off + i] = m.tape.y_ssm[z_off + i] * ssm_scale
                    + m.tape.m_inj[z_off + i]
                    + m.bwd_g_zraw[z_off + i];
            }
        }
    } else {
        for t in 0..seq_len {
            let z_off = t * d_m;
            let m_off = t * d_m;
            let y_off = t * d_m;
            let ad_off = t * m.adapters[0].rank;
            let act = &m.tape.adapter_act[ad_off..ad_off + m.adapters[0].rank];
            adapter_up_into(&m.adapters[0], d_m, act, &mut m.buf_ad_out[..d_m]);
            for i in 0..d_m {
                m.tape.z_raw[z_off + i] = (m.tape.y_ssm[y_off + i] * ssm_scale)
                    + m.tape.m_inj[m_off + i]
                    + m.buf_ad_out[i];
            }
        }
    }

    batched_matvec_dev(
        gpu.as_ref(),
        &m.mlp_w1.data,
        d_mlp,
        d_m,
        &m.tape.z_raw[..seq_len * d_m],
        seq_len,
        &mut m.tape.mlp_hidden[..seq_len * d_mlp],
    );
    for t_i in 0..seq_len * d_mlp {
        let h = m.tape.mlp_hidden[t_i];
        m.tape.mlp_act[t_i] = h * sigmoid(h);
    }
    batched_matvec_dev(
        gpu.as_ref(),
        &m.mlp_w2.data,
        d_m,
        d_mlp,
        &m.tape.mlp_act[..seq_len * d_mlp],
        seq_len,
        &mut m.tape.z_final[..seq_len * d_m],
    );
    for t in 0..seq_len {
        let z_off = t * d_m;
        for i in 0..d_m {
            m.tape.z_final[z_off + i] += m.tape.z_raw[z_off + i];
        }
    }
}

/// Stage 7: unembed logits, stable softmax probabilities, and per-token
/// cross-entropy losses for the whole chunk.
#[inline]
pub fn stage_logits_loss(m: &mut PSSALayerV2, seq_len: usize) -> f32 {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "forward.logits_loss", seq_len);
    let gpu = gpu_ctx(m);
    let d_m = m.cfg.d_latent;
    let d_v = m.cfg.d_vocab;
    let (unembed_w, m) = (&m.unembed_w, &mut m.block);
    let logit_scale = 1.0 / (d_m as f32).sqrt();

    batched_matvec_dev(
        gpu.as_ref(),
        &unembed_w.data,
        d_v,
        d_m,
        &m.tape.z_final[..seq_len * d_m],
        seq_len,
        &mut m.tape.logits[..seq_len * d_v],
    );
    for v in &mut m.tape.logits[..seq_len * d_v] {
        *v *= logit_scale;
    }

    let mut total_loss = 0.0f32;
    for t in 0..seq_len {
        let log_off = t * d_v;
        let tgt_id = m.tape.target_ids[t];

        let mut max_l = f32::NEG_INFINITY;
        for i in 0..d_v {
            let l = m.tape.logits[log_off + i];
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..d_v {
            let exp_l = (m.tape.logits[log_off + i] - max_l).exp();
            m.tape.probs[log_off + i] = exp_l;
            sum_exp += exp_l;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-8);
        for i in 0..d_v {
            m.tape.probs[log_off + i] *= inv_sum;
        }

        let nll_loss = (max_l - m.tape.logits[log_off + tgt_id]) + sum_exp.ln();
        m.tape.losses[t] = nll_loss;
        total_loss += nll_loss;
    }

    total_loss / (seq_len as f32)
}

/// Run the dense staged schedule for one continuous block whose raw inputs are
/// already in the block tape. Stacked depth uses this CPU schedule for every
/// block instead of falling back to the scalar token-at-a-time implementation.
#[inline]
fn stacked_prepare_block(block: &mut crate::pssa::PSSAContinuousBlockV2, seq_len: usize) {
    let state_width = block.cfg.d_latent * block.cfg.d_state;
    block.tape.h_states[..state_width].copy_from_slice(&block.h_persistent);
    assert!(seq_len <= block.tape.max_l);
}

#[inline]
fn stacked_finish_block(block: &mut crate::pssa::PSSAContinuousBlockV2, seq_len: usize) {
    let state_width = block.cfg.d_latent * block.cfg.d_state;
    let last = seq_len * state_width;
    block
        .h_persistent
        .copy_from_slice(&block.tape.h_states[last..last + state_width]);
}

#[inline]
fn stacked_input_norm(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    inputs: &[f32],
    seq_len: usize,
) {
    let d = block.cfg.d_latent;
    assert_eq!(inputs.len(), seq_len * d);
    block.tape.x_raw[..inputs.len()].copy_from_slice(inputs);
    for t in 0..seq_len {
        let off = t * d;
        let raw = &block.tape.x_raw[off..off + d];
        let sum_sq: f32 = raw.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / d as f32 + 1e-5).sqrt();
        block.tape.inv_rms[t] = inv_rms;
        for i in 0..d {
            block.tape.x_norm[off + i] = block.norm_gamma.data[i]
                * (block.tape.x_raw[off + i] * inv_rms)
                + block.norm_beta.data[i];
        }
    }
}

#[inline]
fn stacked_projections(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let s = block.cfg.d_state;
    let xn = &block.tape.x_norm[..seq_len * d];
    batched_matvec_dev(
        gpu,
        &block.w_delta.data,
        d,
        d,
        xn,
        seq_len,
        &mut block.tape.delta_raw[..seq_len * d],
    );
    for i in 0..seq_len * d {
        block.tape.delta[i] = softplus(block.tape.delta_raw[i]);
    }
    batched_matvec_dev(
        gpu,
        &block.w_b.data,
        s,
        d,
        xn,
        seq_len,
        &mut block.tape.b_proj[..seq_len * s],
    );
    batched_matvec_dev(
        gpu,
        &block.w_c.data,
        s,
        d,
        xn,
        seq_len,
        &mut block.tape.c_proj[..seq_len * s],
    );
}

/// The stacked path keeps the short-chunk ordered recurrence. Its expensive
/// dense projections and adjoints are still whole-chunk GEMMs; retaining the
/// ordered recurrence also preserves the scalar depth math for every layer.
#[inline]
fn stacked_ssm(block: &mut crate::pssa::PSSAContinuousBlockV2, seq_len: usize) {
    block.refresh_ssm_rates();
    let d = block.cfg.d_latent;
    let s = block.cfg.d_state;
    let stride = d * s;
    for t in 0..seq_len {
        let d_off = t * d;
        let b_off = t * s;
        let c_off = t * s;
        let h_prev = t * stride;
        let h_next = (t + 1) * stride;
        let xn = t * d;
        for i in 0..d {
            let delta = block.tape.delta[d_off + i];
            let mut y = 0.0f32;
            for j in 0..s {
                let idx = i * s + j;
                let a = (delta * block.ssm_rates[idx]).exp();
                let b = delta * block.tape.b_proj[b_off + j];
                block.tape.bar_a[t * stride + idx] = a;
                block.tape.bar_b[t * stride + idx] = b;
                let h = a * block.tape.h_states[h_prev + idx] + b * block.tape.x_norm[xn + i];
                block.tape.h_states[h_next + idx] = h;
                y += h * block.tape.c_proj[c_off + j];
            }
            block.tape.y_ssm[t * d + i] = y;
        }
    }
}

#[inline]
fn stacked_memory(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let k = block.cfg.d_mem_key;
    let cap = block.cfg.mem_capacity;
    if let Some(gpu) = gpu {
        batched_matvec_dev(
            Some(gpu),
            &block.w_qx.data,
            k,
            d,
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            &mut block.tape.q_euc[..seq_len * k],
        );
        batched_matvec_dev(
            Some(gpu),
            &block.w_qh.data,
            k,
            d,
            &block.tape.y_ssm[..seq_len * d],
            seq_len,
            &mut block.bwd_g_query_euc[..seq_len * k],
        );
        for (q, h) in block.tape.q_euc[..seq_len * k]
            .iter_mut()
            .zip(&block.bwd_g_query_euc[..seq_len * k])
        {
            *q += h;
        }
    } else {
        for t in 0..seq_len {
            let x = t * d;
            let q = t * k;
            for r in 0..k {
                let row = r * d;
                block.tape.q_euc[q + r] =
                    dot_slice(&block.w_qx.data[row..row + d], &block.tape.x_norm[x..x + d])
                        + dot_slice(&block.w_qh.data[row..row + d], &block.tape.y_ssm[x..x + d]);
            }
        }
    }
    for t in 0..seq_len {
        let x = t * d;
        let q = t * k;
        block.tape.q_norm[t] = crate::memory::HyperbolicEpisodicBankV2::diffeomorphic_project(
            &block.tape.q_euc[q..q + k],
            &mut block.tape.q_poincare[q..q + k],
        );
        block.memory.retrieve_soft_into(
            &block.tape.q_poincare[q..q + k],
            block.cfg.tau_mem,
            &mut block.tape.m_val[x..x + d],
            &mut block.tape.mem_weights[t * cap..(t + 1) * cap],
        );
    }
    batched_matvec_dev(
        gpu,
        &block.w_gate.data,
        d,
        d,
        &block.tape.x_norm[..seq_len * d],
        seq_len,
        &mut block.tape.g_mem[..seq_len * d],
    );
    for x in &mut block.tape.g_mem[..seq_len * d] {
        *x = sigmoid(*x);
    }
    batched_matvec_dev(
        gpu,
        &block.w_proj.data,
        d,
        d,
        &block.tape.m_val[..seq_len * d],
        seq_len,
        &mut block.tape.m_proj[..seq_len * d],
    );
    for i in 0..seq_len * d {
        block.tape.m_inj[i] = block.tape.g_mem[i] * block.tape.m_proj[i];
    }
}

#[inline]
fn stacked_adapter(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let rank = block.adapters[0].rank;
    if let Some(gpu) = gpu {
        batched_matvec_dev(
            Some(gpu),
            &block.adapters[0].down_proj.data,
            rank,
            d,
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            &mut block.tape.adapter_hidden[..seq_len * rank],
        );
        for i in 0..seq_len * rank {
            let h = block.tape.adapter_hidden[i];
            block.tape.adapter_act[i] = h * sigmoid(h);
        }
        block.refresh_adapter_up_effective();
        batched_matvec_dev(
            Some(gpu),
            &block.adapter_up_effective,
            d,
            rank,
            &block.tape.adapter_act[..seq_len * rank],
            seq_len,
            &mut block.bwd_g_zraw[..seq_len * d],
        );
        for i in 0..seq_len * d {
            block.tape.z_raw[i] = block.tape.y_ssm[i] / (block.cfg.d_state as f32).sqrt()
                + block.tape.m_inj[i]
                + block.bwd_g_zraw[i];
        }
    } else {
        for t in 0..seq_len {
            let x = t * d;
            let a = t * rank;
            for r in 0..rank {
                block.tape.adapter_hidden[a + r] = dot_slice(
                    &block.adapters[0].down_proj.data[r * d..(r + 1) * d],
                    &block.tape.x_norm[x..x + d],
                );
                let h = block.tape.adapter_hidden[a + r];
                block.tape.adapter_act[a + r] = h * sigmoid(h);
            }
            for i in 0..d {
                let row = i * rank;
                let mut value = 0.0f32;
                for r in 0..rank {
                    value += (block.adapters[0].up_proj.data[row + r]
                        + block.adapters[0].consolidated_up[row + r])
                        * block.tape.adapter_act[a + r];
                }
                block.buf_ad_out[i] = value;
                block.tape.z_raw[t * d + i] = block.tape.y_ssm[t * d + i]
                    / (block.cfg.d_state as f32).sqrt()
                    + block.tape.m_inj[t * d + i]
                    + value;
            }
        }
    }
}

#[inline]
fn stacked_mlp(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let mlp = 2 * d;
    batched_matvec_dev(
        gpu,
        &block.mlp_w1.data,
        mlp,
        d,
        &block.tape.z_raw[..seq_len * d],
        seq_len,
        &mut block.tape.mlp_hidden[..seq_len * mlp],
    );
    for i in 0..seq_len * mlp {
        let h = block.tape.mlp_hidden[i];
        block.tape.mlp_act[i] = h * sigmoid(h);
    }
    batched_matvec_dev(
        gpu,
        &block.mlp_w2.data,
        d,
        mlp,
        &block.tape.mlp_act[..seq_len * mlp],
        seq_len,
        &mut block.tape.z_final[..seq_len * d],
    );
    for i in 0..seq_len * d {
        block.tape.z_final[i] += block.tape.z_raw[i];
    }
}

#[inline]
fn stacked_forward_block(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    inputs: &[f32],
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    stacked_input_norm(block, inputs, seq_len);
    stacked_prepare_block(block, seq_len);
    stacked_projections(block, seq_len, gpu);
    stacked_ssm(block, seq_len);
    stacked_memory(block, seq_len, gpu);
    stacked_adapter(block, seq_len, gpu);
    stacked_mlp(block, seq_len, gpu);
    stacked_finish_block(block, seq_len);
}

/// Run a continuous block whose base tape row has already been populated.
/// This is also used for each saved Ouro loop of an extra depth block after
/// its loop slot is copied into the base workspace.
pub(crate) fn forward_continuous_block_batched(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: &crate::backend::GpuDispatch,
) {
    stage_input_norm_block(block, seq_len);
    stacked_prepare_block(block, seq_len);
    stacked_projections(block, seq_len, Some(gpu));
    stacked_ssm(block, seq_len);
    stacked_memory(block, seq_len, Some(gpu));
    stacked_adapter(block, seq_len, Some(gpu));
    stacked_mlp(block, seq_len, Some(gpu));
    stacked_finish_block(block, seq_len);
}

#[inline]
fn stacked_logits_loss(
    m: &mut PSSALayerV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) -> f32 {
    let d = m.cfg.d_latent;
    let v = m.cfg.d_vocab;
    let final_z = &m.continuous_inputs[..seq_len * d];
    let (unembed, block) = (&m.unembed_w.data, &mut m.block);
    batched_matvec_dev(
        gpu,
        unembed,
        v,
        d,
        final_z,
        seq_len,
        &mut block.tape.logits[..seq_len * v],
    );
    let scale = 1.0 / (d as f32).sqrt();
    for x in &mut block.tape.logits[..seq_len * v] {
        *x *= scale;
    }
    let mut total = 0.0f32;
    for t in 0..seq_len {
        let off = t * v;
        let target = block.tape.target_ids[t];
        let mut max_l = f32::NEG_INFINITY;
        for i in 0..v {
            max_l = max_l.max(block.tape.logits[off + i]);
        }
        let mut sum = 0.0f32;
        for i in 0..v {
            let e = (block.tape.logits[off + i] - max_l).exp();
            block.tape.probs[off + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-8);
        for i in 0..v {
            block.tape.probs[off + i] *= inv;
        }
        let loss = (max_l - block.tape.logits[off + target]) + sum.ln();
        block.tape.losses[t] = loss;
        total += loss;
    }
    total / seq_len as f32
}

fn forward_train_chunk_stacked_batched(
    m: &mut PSSALayerV2,
    token_ids: &[usize],
    target_ids: &[usize],
) -> f32 {
    assert!(!token_ids.is_empty());
    assert_eq!(token_ids.len(), target_ids.len());
    assert!(token_ids.len() <= m.cfg.chunk_len);
    assert!(
        token_ids
            .iter()
            .chain(target_ids)
            .all(|&id| id < m.cfg.d_vocab)
    );
    let seq_len = token_ids.len();
    let d = m.cfg.d_latent;
    let n = seq_len * d;
    m.block.tape.x_ids[..seq_len].copy_from_slice(token_ids);
    m.block.tape.target_ids[..seq_len].copy_from_slice(target_ids);
    let gpu = gpu_ctx(m);
    stage_embed_norm(m, seq_len);
    stacked_prepare_block(&mut m.block, seq_len);
    stacked_projections(&mut m.block, seq_len, gpu.as_ref());
    stacked_ssm(&mut m.block, seq_len);
    stacked_memory(&mut m.block, seq_len, gpu.as_ref());
    stacked_adapter(&mut m.block, seq_len, gpu.as_ref());
    stacked_mlp(&mut m.block, seq_len, gpu.as_ref());
    stacked_finish_block(&mut m.block, seq_len);
    m.continuous_inputs[..n].copy_from_slice(&m.block.tape.z_final[..n]);
    for layer in 0..m.extra_blocks.len() {
        let input = &m.continuous_inputs[..n];
        stacked_forward_block(&mut m.extra_blocks[layer], input, seq_len, gpu.as_ref());
        let scale = m.residual_scales[layer];
        for i in 0..n {
            m.layer_activations[layer][i] =
                input[i] + scale * m.extra_blocks[layer].tape.z_final[i];
        }
        m.continuous_inputs[..n].copy_from_slice(&m.layer_activations[layer][..n]);
    }
    stacked_logits_loss(m, seq_len, gpu.as_ref())
}

/// Full batched forward pass over a chunk, matching `forward_train_chunk` to
/// f32 roundoff. Dense stages batch token rows; the diagonal SSM recurrence uses
/// a host-side parallel Blelloch scan for both CPU and GPU dense backends.
pub fn forward_train_chunk_batched(
    m: &mut PSSALayerV2,
    token_ids: &[usize],
    target_ids: &[usize],
) -> f32 {
    if m.local_mixing_enabled() {
        assert!(!m.device.is_gpu(), "local mixing is CPU-only");
        return m.forward_train_chunk(token_ids, target_ids);
    }
    if m.loops() > 1 {
        return m.forward_train_chunk(token_ids, target_ids);
    }
    if m.depth() > 1 {
        if m.loops() == 1 {
            return forward_train_chunk_stacked_batched(m, token_ids, target_ids);
        }
        return m.forward_train_chunk(token_ids, target_ids);
    }
    assert!(!token_ids.is_empty(), "training chunk must be nonempty");
    assert_eq!(
        token_ids.len(),
        target_ids.len(),
        "token and target counts must match"
    );
    assert!(
        token_ids.len() <= m.cfg.chunk_len,
        "training chunk length exceeds configured tape capacity"
    );
    let seq_len = token_ids.len();
    assert!(seq_len > 0);
    assert!(
        token_ids[..seq_len].iter().all(|&id| id < m.cfg.d_vocab)
            && target_ids[..seq_len].iter().all(|&id| id < m.cfg.d_vocab),
        "token IDs must be in vocabulary"
    );

    m.tape.x_ids[..seq_len].copy_from_slice(&token_ids[..seq_len]);
    m.tape.target_ids[..seq_len].copy_from_slice(&target_ids[..seq_len]);
    m.block.tape.h_states[..m.cfg.d_latent * m.cfg.d_state].copy_from_slice(&m.block.h_persistent);

    stage_embed_norm(m, seq_len);
    stage_projections(m, seq_len);
    stage_ssm_scan(m, seq_len);
    stage_memory(m, seq_len);
    stage_adapter(m, seq_len);
    stage_mlp(m, seq_len);
    let loss = stage_logits_loss(m, seq_len);

    let last_h_off = seq_len * (m.cfg.d_latent * m.cfg.d_state);
    m.block.h_persistent.copy_from_slice(
        &m.block.tape.h_states[last_h_off..last_h_off + m.cfg.d_latent * m.cfg.d_state],
    );

    loss
}

// =============================================================================
// BACKWARD STAGES (REVERSE TIME, BATCHED)
// =============================================================================

/// Backward Stage 7: full-vocabulary cross-entropy adjoint + unembed gradient,
/// batched over all L tokens.
#[inline]
pub fn bwd_stage_logits(m: &mut PSSALayerV2, seq_len: usize, scale_loss: f32) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "backward.logits", seq_len);
    if let Some(gpu) = gpu_ctx(m).filter(|g| g.accelerates_backward()) {
        bwd_stage_logits_blocked(m, seq_len, scale_loss, Some(&gpu));
        return;
    }
    if blocked_backward(seq_len, m.cfg.d_vocab, m.cfg.d_latent) {
        bwd_stage_logits_blocked(m, seq_len, scale_loss, None);
    } else {
        bwd_stage_logits_scalar(m, seq_len, scale_loss);
    }
}

/// Blocked form of the logits backward pass: build the per-token logit adjoints
/// once, then take both the input gradient and the unembedding weight gradient
/// as single GEMMs. Passing `gpu = None` runs the CPU twins, which is how the
/// restructure is verified against [`bwd_stage_logits_scalar`].
pub fn bwd_stage_logits_blocked(
    m: &mut PSSALayerV2,
    seq_len: usize,
    scale_loss: f32,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d_m = m.cfg.d_latent;
    let d_v = m.cfg.d_vocab;
    let (unembed_w, m) = (&mut m.unembed_w, &mut m.block);
    let logit_scale = 1.0 / (d_m as f32).sqrt();

    // G [L, d_v]: dLoss/dlogit for every token.
    let g_logit = &mut m.bwd_g_logits[..seq_len * d_v];
    for t in 0..seq_len {
        let log_off = t * d_v;
        let tgt_id = m.tape.target_ids[t];
        for i in 0..d_v {
            let indicator = if i == tgt_id { 1.0 } else { 0.0 };
            g_logit[log_off + i] =
                (m.tape.probs[log_off + i] - indicator) * scale_loss * logit_scale;
        }
    }

    if gpu.is_none() {
        dense_input_adjoint(
            g_logit,
            &unembed_w.data,
            seq_len,
            d_v,
            d_m,
            &mut m.bwd_g_zfinal[..seq_len * d_m],
        );
        dense_weight_adjoint(
            g_logit,
            &m.tape.z_final[..seq_len * d_m],
            seq_len,
            d_v,
            d_m,
            &mut unembed_w.grad,
        );
        return;
    }

    // grad_z_final [L, d_m] = G [L, d_v] * W [d_v, d_m]
    gemm_nn_dev_into(
        gpu,
        &g_logit,
        &unembed_w.data,
        seq_len,
        d_v,
        d_m,
        &mut m.bwd_g_zfinal[..seq_len * d_m],
    );

    // unembed grad [d_v, d_m] += G^T [d_v, L] * Z [L, d_m]
    gemm_tn_dev_accumulate(
        gpu,
        &g_logit,
        &m.tape.z_final[..seq_len * d_m],
        seq_len,
        d_v,
        d_m,
        &mut unembed_w.grad,
    );
}

/// Fused per-token reference form, kept as the CPU path and the twin.
pub fn bwd_stage_logits_scalar(m: &mut PSSALayerV2, seq_len: usize, scale_loss: f32) {
    let d_m = m.cfg.d_latent;
    let d_v = m.cfg.d_vocab;
    let (unembed_w, m) = (&mut m.unembed_w, &mut m.block);
    let logit_scale = 1.0 / (d_m as f32).sqrt();

    for t in 0..seq_len {
        let z_off = t * d_m;
        let log_off = t * d_v;
        let tgt_id = m.tape.target_ids[t];

        // Per-token grad_z_final adjoint (stored for downstream stages).
        for j in 0..d_m {
            m.bwd_g_zfinal[z_off + j] = 0.0;
        }
        for i in 0..d_v {
            let indicator = if i == tgt_id { 1.0 } else { 0.0 };
            let g_logit = (m.tape.probs[log_off + i] - indicator) * scale_loss * logit_scale;
            let row_off = i * d_m;
            for j in 0..d_m {
                m.bwd_g_zfinal[z_off + j] += g_logit * unembed_w.data[row_off + j];
                unembed_w.grad[row_off + j] += g_logit * m.tape.z_final[z_off + j];
            }
        }
    }
}

/// Backward Stage 6: SiLU MLP adjoint, batched over all L tokens.
#[inline]
pub fn bwd_stage_mlp(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace = crate::training_diagnostics::StageTrace::new(&m.device, "backward.mlp", seq_len);
    if let Some(gpu) = gpu_ctx(m).filter(|g| g.accelerates_backward()) {
        bwd_stage_mlp_blocked(m, seq_len, Some(&gpu));
        return;
    }
    if blocked_backward(seq_len, m.cfg.d_latent, 2 * m.cfg.d_latent) {
        bwd_stage_mlp_blocked(m, seq_len, None);
    } else {
        bwd_stage_mlp_scalar(m, seq_len);
    }
}

/// Blocked form of the MLP backward pass: four GEMMs over the whole chunk
/// instead of four matvecs per token. `gpu = None` runs the CPU twins, which is
/// how this is verified against [`bwd_stage_mlp_scalar`].
pub fn bwd_stage_mlp_blocked(
    m: &mut PSSALayerV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_mlp = d_m * 2;
    let l = seq_len;

    let gz = &m.bwd_g_zfinal[..l * d_m];
    let g_hidden = &mut m.bwd_g_mlp[..l * d_mlp];
    if gpu.is_none() {
        dense_input_adjoint(gz, &m.mlp_w2.data, l, d_m, d_mlp, g_hidden);
        dense_weight_adjoint(
            gz,
            &m.tape.mlp_act[..l * d_mlp],
            l,
            d_m,
            d_mlp,
            &mut m.mlp_w2.grad,
        );
        for (i, grad) in g_hidden.iter_mut().enumerate() {
            let h = m.tape.mlp_hidden[i];
            let sig_h = sigmoid(h);
            *grad *= sig_h * (1.0 + h * (1.0 - sig_h));
        }
        dense_input_adjoint(
            g_hidden,
            &m.mlp_w1.data,
            l,
            d_mlp,
            d_m,
            &mut m.bwd_g_zraw[..l * d_m],
        );
        dense_weight_adjoint(
            g_hidden,
            &m.tape.z_raw[..l * d_m],
            l,
            d_mlp,
            d_m,
            &mut m.mlp_w1.grad,
        );
        for (dst, &residual) in m.bwd_g_zraw[..l * d_m].iter_mut().zip(gz) {
            *dst += residual;
        }
        return;
    }

    // g_mlp_act [L, d_mlp] = gz [L, d_m] * mlp_w2 [d_m, d_mlp]
    gemm_nn_dev_into(gpu, &gz, &m.mlp_w2.data, l, d_m, d_mlp, g_hidden);

    // mlp_w2 grad [d_m, d_mlp] += gz^T * mlp_act [L, d_mlp]
    gemm_tn_dev_accumulate(
        gpu,
        &gz,
        &m.tape.mlp_act[..l * d_mlp],
        l,
        d_m,
        d_mlp,
        &mut m.mlp_w2.grad,
    );

    // SiLU derivative, elementwise over the chunk.
    for i in 0..l * d_mlp {
        let h = m.tape.mlp_hidden[i];
        let sig_h = sigmoid(h);
        g_hidden[i] *= sig_h * (1.0 + h * (1.0 - sig_h));
    }

    // g_zraw_mlp [L, d_m] = g_hidden [L, d_mlp] * mlp_w1 [d_mlp, d_m]
    gemm_nn_dev_into(
        gpu,
        &g_hidden,
        &m.mlp_w1.data,
        l,
        d_mlp,
        d_m,
        &mut m.bwd_g_zraw[..l * d_m],
    );

    // mlp_w1 grad [d_mlp, d_m] += g_hidden^T * z_raw [L, d_m]
    gemm_tn_dev_accumulate(
        gpu,
        &g_hidden,
        &m.tape.z_raw[..l * d_m],
        l,
        d_mlp,
        d_m,
        &mut m.mlp_w1.grad,
    );

    for i in 0..l * d_m {
        m.bwd_g_zraw[i] += m.bwd_g_zfinal[i];
    }
}

/// Fused per-token reference form, kept as the CPU path and the twin.
pub fn bwd_stage_mlp_scalar(m: &mut PSSALayerV2, seq_len: usize) {
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_mlp = d_m * 2;
    let l = seq_len;

    for t in 0..l {
        let mlp_off = t * d_mlp;
        let z_off = t * d_m;

        // g_mlp_act = mlp_w2^T @ grad_z_final[t]
        m.mlp_w2
            .matvec_transpose(&m.bwd_g_zfinal[z_off..z_off + d_m], &mut m.buf_g_mlp_act);

        // mlp_w2 grad: grad_z_final[t,i] * mlp_act[t,j]
        for i in 0..d_m {
            let gz_i = m.bwd_g_zfinal[z_off + i];
            let row_off = i * d_mlp;
            for j in 0..d_mlp {
                m.mlp_w2.grad[row_off + j] += gz_i * m.tape.mlp_act[mlp_off + j];
            }
        }

        // SiLU derivative
        for i in 0..d_mlp {
            let h = m.tape.mlp_hidden[mlp_off + i];
            let sig_h = sigmoid(h);
            let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
            m.buf_g_mlp_hidden[i] = m.buf_g_mlp_act[i] * silu_prime;
        }

        // g_zraw_mlp = mlp_w1^T @ g_mlp_hidden
        m.mlp_w1
            .matvec_transpose(&m.buf_g_mlp_hidden, &mut m.buf_g_zraw_mlp);

        // mlp_w1 grad: g_mlp_hidden[t,i] * z_raw[t,j]
        for i in 0..d_mlp {
            let gh_i = m.buf_g_mlp_hidden[i];
            let row_off = i * d_m;
            for j in 0..d_m {
                m.mlp_w1.grad[row_off + j] += gh_i * m.tape.z_raw[z_off + j];
            }
        }

        // per-token grad_z_raw = grad_z_final + g_zraw_mlp (stored downstream)
        for i in 0..d_m {
            m.bwd_g_zraw[z_off + i] = m.bwd_g_zfinal[z_off + i] + m.buf_g_zraw_mlp[i];
        }
    }
}

/// Backward Stage 5: plastic adapter adjoint, batched over all L tokens.
#[inline]
pub fn bwd_stage_adapter(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "backward.adapter", seq_len);
    let gpu = gpu_ctx(m).filter(|g| g.accelerates_backward());
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let l = seq_len;

    if let Some(gpu) = gpu.as_ref() {
        // The effective adapter-up matrix was materialized by the matching
        // forward stage and remains unchanged until backward completes.
        gemm_nn_dev_into(
            Some(gpu),
            &m.bwd_g_zraw[..l * d_m],
            &m.adapter_up_effective,
            l,
            d_m,
            rank,
            &mut m.bwd_g_ad_down[..l * rank],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &m.bwd_g_zraw[..l * d_m],
            &m.tape.adapter_act[..l * rank],
            l,
            d_m,
            rank,
            &mut m.adapters[0].up_proj.grad,
        );
    } else {
        // CPU twin: retain the historical reduction order exactly.
        adapter_up_input_adjoint(
            &m.bwd_g_zraw[..l * d_m],
            &m.adapters[0].up_proj.data,
            &m.adapters[0].consolidated_up,
            l,
            d_m,
            rank,
            &mut m.bwd_g_ad_down[..l * rank],
        );
        dense_weight_adjoint_forward(
            &m.bwd_g_zraw[..l * d_m],
            &m.tape.adapter_act[..l * rank],
            l,
            d_m,
            rank,
            &mut m.adapters[0].up_proj.grad,
        );
    }

    // SiLU derivative on adapter hidden, stored per token.
    for t in 0..l {
        let ad_off = t * rank;
        for r in 0..rank {
            let h = m.tape.adapter_hidden[ad_off + r];
            let sig_h = sigmoid(h);
            let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
            m.bwd_g_ad_down[ad_off + r] *= silu_prime;
        }
    }
}

/// Backward Stage 4b: adapter down-projection adjoint into grad_x_norm plus
/// down_proj gradients, batched over all L tokens. (Kept separate from
/// bwd_stage_adapter because grad_x_norm accumulates across stages.)
#[inline]
pub fn bwd_stage_adapter_down(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "backward.adapter_down", seq_len);
    let gpu = gpu_ctx(m).filter(|g| g.accelerates_backward());
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let l = seq_len;
    m.bwd_g_xnorm[..l * d_m].fill(0.0);

    if let Some(gpu) = gpu.as_ref() {
        gemm_nn_dev_into(
            Some(gpu),
            &m.bwd_g_ad_down[..l * rank],
            &m.adapters[0].down_proj.data,
            l,
            rank,
            d_m,
            &mut m.bwd_g_xnorm[..l * d_m],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &m.bwd_g_ad_down[..l * rank],
            &m.tape.x_norm[..l * d_m],
            l,
            rank,
            d_m,
            &mut m.adapters[0].down_proj.grad,
        );
    } else {
        // CPU twin: retain the historical reduction order exactly.
        dense_input_adjoint(
            &m.bwd_g_ad_down[..l * rank],
            &m.adapters[0].down_proj.data,
            l,
            rank,
            d_m,
            &mut m.bwd_g_xnorm[..l * d_m],
        );
        dense_weight_adjoint_forward(
            &m.bwd_g_ad_down[..l * rank],
            &m.tape.x_norm[..l * d_m],
            l,
            rank,
            d_m,
            &mut m.adapters[0].down_proj.grad,
        );
    }
}

#[inline]
fn memory_query_adjoint(
    memory: &HyperbolicEpisodicBankV2,
    q_poincare: &[f32],
    q_euc: &[f32],
    g_m_val: &[f32],
    m_val: &[f32],
    weights: &[f32],
    tau: f32,
    g_query_pnc: &mut [f32],
    g_query_euc: &mut [f32],
) {
    let d_key = memory.dim_key;
    let d_value = memory.dim_val;
    assert_eq!(q_poincare.len(), d_key);
    assert_eq!(q_euc.len(), d_key);
    assert_eq!(g_m_val.len(), d_value);
    assert_eq!(m_val.len(), d_value);
    assert_eq!(weights.len(), memory.capacity);
    g_query_pnc.fill(0.0);
    let q_sq = HyperbolicEpisodicBankV2::squared_norm(q_poincare) as f64;
    for entry in 0..memory.count {
        let key_off = entry * d_key;
        let key = &memory.keys[key_off..key_off + d_key];
        let key_sq = memory.norm_sq[entry] as f64;
        let mut dot_g_value_minus_mean = 0.0f64;
        let value_off = entry * d_value;
        for j in 0..d_value {
            dot_g_value_minus_mean +=
                g_m_val[j] as f64 * (memory.values[value_off + j] - m_val[j]) as f64;
        }
        let g_score = weights[entry] as f64 * dot_g_value_minus_mean;
        let mut sq = 0.0f64;
        for k in 0..d_key {
            let diff = q_poincare[k] as f64 - key[k] as f64;
            sq += diff * diff;
        }
        if sq > 0.0 {
            let denom = (1.0 - q_sq) * (1.0 - key_sq);
            assert!(denom > 0.0 && denom.is_finite());
            let z = sq / denom;
            let dd_dz = 1.0 / (z * (1.0 + z)).sqrt();
            for k in 0..d_key {
                let diff = q_poincare[k] as f64 - key[k] as f64;
                let ddenom = -2.0 * q_poincare[k] as f64 * (1.0 - key_sq);
                let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                g_query_pnc[k] += (g_score * (-1.0 / tau as f64) * dd_dz * dz) as f32;
            }
        }
    }
    HyperbolicEpisodicBankV2::projection_adjoint(q_euc, g_query_pnc, g_query_euc);
}

/// Backward Stage 4: memory injection adjoint (gate, w_proj, memory query
/// adjoint incl. hyperbolic distance chain), accumulating grad_x_norm.
/// Retrieval VJPs are independent per token and use Rayon for training-sized
/// banks; shared parameter rows retain the reference reverse-token reduction.
#[inline]
pub fn bwd_stage_memory(m: &mut PSSALayerV2, seq_len: usize) {
    let _trace =
        crate::training_diagnostics::StageTrace::new(&m.device, "backward.memory", seq_len);
    let gpu = gpu_ctx(m).filter(|g| g.accelerates_backward());
    let m = &mut m.block;
    let d_m = m.cfg.d_latent;
    let d_k = m.cfg.d_mem_key;
    let mem_cap = m.cfg.mem_capacity;
    let l = seq_len;

    // Keep the two token-local gate adjoints in model-owned storage. Their
    // rows are independent, while the shared weight reductions below retain
    // the scalar forward-token order for each parameter element.
    let g_m_proj_out = &mut m.bwd_g_zfinal[..l * d_m];
    let (g_gate_pre, gate_x) = m.bwd_g_mlp[..2 * l * d_m].split_at_mut(l * d_m);

    let g_zraw = &m.bwd_g_zraw[..l * d_m];
    let g_mem = &m.tape.g_mem[..l * d_m];
    let m_proj = &m.tape.m_proj[..l * d_m];
    if let Some(gpu) = gpu.as_ref() {
        if let Err(error) =
            gpu.memory_backward_local(g_zraw, g_mem, m_proj, g_m_proj_out, g_gate_pre)
        {
            warn_gpu_fallback_once(
                &error,
                format!(
                    "warning: GPU memory elementwise backward failed; using host path: {error}"
                ),
            );
            for i in 0..g_zraw.len() {
                g_m_proj_out[i] = g_zraw[i] * g_mem[i];
                g_gate_pre[i] = g_zraw[i] * m_proj[i] * g_mem[i] * (1.0 - g_mem[i]);
            }
        }
    } else {
        let calculate_gate = |(t, (g_mp, g_gate)): (usize, (&mut [f32], &mut [f32]))| {
            let off = t * d_m;
            for i in 0..d_m {
                let gz_i = g_zraw[off + i];
                let g_mem_i = g_mem[off + i];
                g_mp[i] = gz_i * g_mem_i;
                g_gate[i] = gz_i * m_proj[off + i] * g_mem_i * (1.0 - g_mem_i);
            }
        };
        if parallel_backward(l, d_m, d_m) {
            g_m_proj_out
                .par_chunks_mut(d_m)
                .zip(g_gate_pre.par_chunks_mut(d_m))
                .enumerate()
                .for_each(calculate_gate);
        } else {
            g_m_proj_out
                .chunks_mut(d_m)
                .zip(g_gate_pre.chunks_mut(d_m))
                .enumerate()
                .for_each(calculate_gate);
        }
    }

    // The input adjoint is first formed per token, then added in token order
    // so its f32 accumulation order remains identical to the scalar path.
    if let Some(gpu) = gpu.as_ref() {
        gemm_nn_dev_into(Some(gpu), g_gate_pre, &m.w_gate.data, l, d_m, d_m, gate_x);
        for (dst, src) in m.bwd_g_xnorm[..l * d_m].iter_mut().zip(gate_x.iter()) {
            *dst += src;
        }
        // Reuse the second half of the same scratch for the projection input
        // adjoint after its gate contribution has been consumed.
        gemm_nn_dev_into(Some(gpu), g_m_proj_out, &m.w_proj.data, l, d_m, d_m, gate_x);
        gemm_tn_dev_accumulate(
            Some(gpu),
            g_gate_pre,
            &m.tape.x_norm[..l * d_m],
            l,
            d_m,
            d_m,
            &mut m.w_gate.grad,
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            g_m_proj_out,
            &m.tape.m_val[..l * d_m],
            l,
            d_m,
            d_m,
            &mut m.w_proj.grad,
        );
    } else {
        dense_input_adjoint(g_gate_pre, &m.w_gate.data, l, d_m, d_m, gate_x);
        for (dst, src) in m.bwd_g_xnorm[..l * d_m].iter_mut().zip(gate_x.iter()) {
            *dst += src;
        }
        dense_input_adjoint(g_m_proj_out, &m.w_proj.data, l, d_m, d_m, gate_x);
        dense_weight_adjoint_forward(
            g_gate_pre,
            &m.tape.x_norm[..l * d_m],
            l,
            d_m,
            d_m,
            &mut m.w_gate.grad,
        );
        dense_weight_adjoint_forward(
            g_m_proj_out,
            &m.tape.m_val[..l * d_m],
            l,
            d_m,
            d_m,
            &mut m.w_proj.grad,
        );
    }

    // Every retrieval VJP reads the same immutable bank and writes one query
    // row. Materialize those rows in parallel, then reduce shared CPU weight
    // gradients in the historical reverse-token order.
    let parallel = parallel_memory_work(l, m.memory.count, d_k, d_m);
    let memory = &m.memory;
    let q_poincare = &m.tape.q_poincare[..l * d_k];
    let q_euc = &m.tape.q_euc[..l * d_k];
    let m_values = &m.tape.m_val[..l * d_m];
    let mem_weights = &m.tape.mem_weights[..l * mem_cap];
    let g_m_values = &gate_x[..l * d_m];
    let query_pnc = &mut m.bwd_g_query_pnc[..l * d_k];
    let query_euc = &mut m.bwd_g_query_euc[..l * d_k];
    if let Some(gpu) = gpu.as_ref() {
        if let Err(error) = gpu.memory_backward_retrieval(
            q_poincare,
            q_euc,
            g_m_values,
            m_values,
            mem_weights,
            &memory.keys,
            &memory.norm_sq,
            &memory.values,
            l,
            memory.count,
            mem_cap,
            d_k,
            d_m,
            m.cfg.tau_mem,
            query_pnc,
            query_euc,
        ) {
            warn_gpu_fallback_once(
                &error,
                format!("warning: GPU memory retrieval backward failed; using host path: {error}"),
            );
            for (t, (pnc, euc)) in query_pnc
                .chunks_mut(d_k)
                .zip(query_euc.chunks_mut(d_k))
                .enumerate()
            {
                memory_query_adjoint(
                    memory,
                    &q_poincare[t * d_k..(t + 1) * d_k],
                    &q_euc[t * d_k..(t + 1) * d_k],
                    &g_m_values[t * d_m..(t + 1) * d_m],
                    &m_values[t * d_m..(t + 1) * d_m],
                    &mem_weights[t * mem_cap..(t + 1) * mem_cap],
                    m.cfg.tau_mem,
                    pnc,
                    euc,
                );
            }
        }
    } else if parallel {
        query_pnc
            .par_chunks_mut(d_k)
            .zip(query_euc.par_chunks_mut(d_k))
            .enumerate()
            .for_each(|(t, (pnc, euc))| {
                memory_query_adjoint(
                    memory,
                    &q_poincare[t * d_k..(t + 1) * d_k],
                    &q_euc[t * d_k..(t + 1) * d_k],
                    &g_m_values[t * d_m..(t + 1) * d_m],
                    &m_values[t * d_m..(t + 1) * d_m],
                    &mem_weights[t * mem_cap..(t + 1) * mem_cap],
                    m.cfg.tau_mem,
                    pnc,
                    euc,
                );
            });
    } else {
        query_pnc
            .chunks_mut(d_k)
            .zip(query_euc.chunks_mut(d_k))
            .enumerate()
            .for_each(|(t, (pnc, euc))| {
                memory_query_adjoint(
                    memory,
                    &q_poincare[t * d_k..(t + 1) * d_k],
                    &q_euc[t * d_k..(t + 1) * d_k],
                    &g_m_values[t * d_m..(t + 1) * d_m],
                    &m_values[t * d_m..(t + 1) * d_m],
                    &mem_weights[t * mem_cap..(t + 1) * mem_cap],
                    m.cfg.tau_mem,
                    pnc,
                    euc,
                );
            });
    }
    if let Some(gpu) = gpu.as_ref() {
        gemm_tn_dev_accumulate(
            Some(gpu),
            query_euc,
            &m.tape.x_norm[..l * d_m],
            l,
            d_k,
            d_m,
            &mut m.w_qx.grad,
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            query_euc,
            &m.tape.y_ssm[..l * d_m],
            l,
            d_k,
            d_m,
            &mut m.w_qh.grad,
        );
        gemm_nn_dev_into(
            Some(gpu),
            query_euc,
            &m.w_qx.data,
            l,
            d_k,
            d_m,
            &mut m.bwd_g_mlp[..l * d_m],
        );
        for (dst, src) in m.bwd_g_xnorm[..l * d_m]
            .iter_mut()
            .zip(&m.bwd_g_mlp[..l * d_m])
        {
            *dst += src;
        }
        gemm_nn_dev_into(
            Some(gpu),
            query_euc,
            &m.w_qh.data,
            l,
            d_k,
            d_m,
            &mut m.bwd_g_ysm[..l * d_m],
        );
    } else {
        m.g_y_ssm.fill(0.0);
        for t in (0..l).rev() {
            let q_off = t * d_k;
            let m_off = t * d_m;
            let xn = &m.tape.x_norm[m_off..m_off + d_m];
            let y = &m.tape.y_ssm[m_off..m_off + d_m];
            let query = &query_euc[q_off..q_off + d_k];
            for r_i in 0..d_k {
                let gq = query[r_i];
                let row = r_i * d_m;
                for j in 0..d_m {
                    m.w_qx.grad[row + j] += gq * xn[j];
                    m.w_qh.grad[row + j] += gq * y[j];
                    m.bwd_g_xnorm[m_off + j] += gq * m.w_qx.data[row + j];
                    m.g_y_ssm[j] += gq * m.w_qh.data[row + j];
                }
            }
            m.bwd_g_ysm[m_off..m_off + d_m].copy_from_slice(&m.g_y_ssm);
            m.g_y_ssm.fill(0.0);
        }
    }
}

/// CUDA SSM backward. Reverse affine maps, the tiled scan, and token-local
/// derivatives stay on the device. Only the existing dense projection boundary
/// and the final RMSNorm/embedding scatter return to the host-owned tape.
#[inline]
fn bwd_stage_ssm_cuda(
    m: &mut PSSALayerV2,
    seq_len: usize,
    input_is_embedding: bool,
    gpu: &crate::backend::GpuDispatch,
) -> Result<(), String> {
    let pending_step = m.step_counter + 1;
    let (embed_w, embed_row_marks, block) = (&mut m.embed_w, &mut m.embed_row_marks, &mut m.block);
    block.refresh_ssm_rates();
    let d_m = block.cfg.d_latent;
    let d_s = block.cfg.d_state;
    let l = seq_len;
    let stride = d_m * d_s;
    let ssm_scale = 1.0 / (d_s as f32).sqrt();
    if input_is_embedding {
        for t in 0..l {
            embed_row_marks[block.tape.x_ids[t]] = pending_step;
        }
    }

    // bwd_g_mlp is dead after the memory stage and is a model-owned temporary
    // for the device recurrence adjoint. The memory/adapter contribution already
    // in bwd_g_xnorm is preserved and the CUDA result is added below.
    let mut g_ssm_x = vec![0.0f32; l * d_m];
    gpu.ssm_backward(
        &block.tape.delta[..l * d_m],
        &block.tape.delta_raw[..l * d_m],
        &block.tape.b_proj[..l * d_s],
        &block.tape.c_proj[..l * d_s],
        &block.ssm_rates,
        &block.ssm_rate_derivatives,
        &block.tape.x_norm[..l * d_m],
        &block.tape.h_states[..(l + 1) * stride],
        &block.tape.bar_a[..l * stride],
        &block.tape.bar_b[..l * stride],
        &block.bwd_g_zraw[..l * d_m],
        &block.bwd_g_ysm[..l * d_m],
        l,
        d_m,
        d_s,
        ssm_scale,
        &mut block.bwd_ssm_delta[..l * d_m],
        &mut block.bwd_ssm_b[..l * d_s],
        &mut block.bwd_ssm_c[..l * d_s],
        &mut block.bwd_ssm_a[..l * stride],
        &mut g_ssm_x,
    )?;
    for (dst, src) in block.bwd_g_xnorm[..l * d_m].iter_mut().zip(g_ssm_x) {
        *dst += src;
    }

    for (g, w, rows) in [
        (&block.bwd_ssm_delta[..l * d_m], &mut block.w_delta, d_m),
        (&block.bwd_ssm_b[..l * d_s], &mut block.w_b, d_s),
        (&block.bwd_ssm_c[..l * d_s], &mut block.w_c, d_s),
    ] {
        gemm_nn_dev_into(
            Some(gpu),
            g,
            &w.data,
            l,
            rows,
            d_m,
            &mut block.bwd_g_mlp[..l * d_m],
        );
        for (dst, src) in block.bwd_g_xnorm[..l * d_m]
            .iter_mut()
            .zip(&block.bwd_g_mlp[..l * d_m])
        {
            *dst += src;
        }
        gemm_tn_dev_accumulate(
            Some(gpu),
            g,
            &block.tape.x_norm[..l * d_m],
            l,
            rows,
            d_m,
            &mut w.grad,
        );
    }
    for t in (0..l).rev() {
        for idx in 0..stride {
            block.a_mat.grad[idx] += block.bwd_ssm_a[t * stride + idx];
        }
    }

    // This final chain is deliberately identical to the CPU twin. It consumes
    // the device-produced x_norm adjoint and owns the sparse embedding scatter.
    for t in (0..l).rev() {
        let x_id = block.tape.x_ids[t];
        let off = t * d_m;
        let inv_rms = block.tape.inv_rms[t];
        let e_t = &block.tape.x_raw[off..off + d_m];
        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            let gx_i = block.bwd_g_xnorm[off + i];
            block.norm_beta.grad[i] += gx_i;
            block.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);
            dot_gx_e += gx_i * block.norm_gamma.data[i] * e_t[i];
        }
        if input_is_embedding {
            let row = x_id * d_m;
            for i in 0..d_m {
                let g_unnorm = block.bwd_g_xnorm[off + i] * block.norm_gamma.data[i];
                embed_w.grad[row + i] +=
                    inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / d_m as f32));
            }
        }
    }
    Ok(())
}

/// Compatibility fallback for a CUDA driver/kernel error. The normal path above
/// keeps the recurrence device-resident; this ordered implementation preserves
/// training progress when a deployment has an incompatible stage module.
#[inline]
fn bwd_stage_ssm_sequential_gpu(
    m: &mut PSSALayerV2,
    seq_len: usize,
    input_is_embedding: bool,
    gpu: &crate::backend::GpuDispatch,
) {
    let pending_step = m.step_counter + 1;
    let (embed_w, embed_row_marks, block) = (&mut m.embed_w, &mut m.embed_row_marks, &mut m.block);
    block.refresh_ssm_rates();
    let d_m = block.cfg.d_latent;
    let d_s = block.cfg.d_state;
    let l = seq_len;
    let stride = d_m * d_s;
    let ssm_scale = 1.0 / (d_s as f32).sqrt();

    block.grad_h_next.fill(0.0);
    if input_is_embedding {
        for t in 0..l {
            embed_row_marks[block.tape.x_ids[t]] = pending_step;
        }
    }

    for t in (0..l).rev() {
        let del_off = t * d_m;
        let m_off = t * d_m;
        block.buf_g_delta.fill(0.0);
        block.buf_g_b_proj.fill(0.0);
        block.buf_g_c_proj.fill(0.0);
        block.buf_g_h_prev.fill(0.0);

        for i in 0..d_m {
            let gz_i = block.bwd_g_zraw[m_off + i];
            let g_y_i = gz_i * ssm_scale + block.bwd_g_ysm[m_off + i];
            let d_i = block.tape.delta[del_off + i];
            let xn_i = block.tape.x_norm[m_off + i];
            for j in 0..d_s {
                let idx = i * d_s + j;
                let h_next = block.tape.h_states[(t + 1) * stride + idx];
                let c_val = block.tape.c_proj[t * d_s + j];
                let bar_a = block.tape.bar_a[t * stride + idx];
                let a_physical = block.ssm_rates[idx];
                let b_val = block.tape.b_proj[t * d_s + j];
                let g_h_total = g_y_i * c_val + block.grad_h_next[idx];
                block.buf_g_c_proj[j] += g_y_i * h_next;
                block.buf_g_h_prev[idx] += g_h_total * bar_a;
                block.bwd_ssm_a[t * stride + idx] = g_h_total
                    * (d_i * bar_a)
                    * block.tape.h_states[t * stride + idx]
                    * block.ssm_rate_derivatives[idx];
                block.buf_g_delta[i] += g_h_total
                    * (a_physical * bar_a * block.tape.h_states[t * stride + idx] + b_val * xn_i);
                block.buf_g_b_proj[j] += g_h_total * (d_i * xn_i);
                block.bwd_g_xnorm[m_off + i] += g_h_total * block.tape.bar_b[t * stride + idx];
            }
        }
        block.grad_h_next.copy_from_slice(&block.buf_g_h_prev);
        for i in 0..d_m {
            block.bwd_ssm_delta[m_off + i] =
                block.buf_g_delta[i] * sigmoid(block.tape.delta_raw[del_off + i]);
        }
        for j in 0..d_s {
            block.bwd_ssm_b[t * d_s + j] = block.buf_g_b_proj[j];
            block.bwd_ssm_c[t * d_s + j] = block.buf_g_c_proj[j];
        }
    }

    for (g, w, rows) in [
        (&block.bwd_ssm_delta[..l * d_m], &mut block.w_delta, d_m),
        (&block.bwd_ssm_b[..l * d_s], &mut block.w_b, d_s),
        (&block.bwd_ssm_c[..l * d_s], &mut block.w_c, d_s),
    ] {
        gemm_nn_dev_into(
            Some(gpu),
            g,
            &w.data,
            l,
            rows,
            d_m,
            &mut block.bwd_g_mlp[..l * d_m],
        );
        for (dst, src) in block.bwd_g_xnorm[..l * d_m]
            .iter_mut()
            .zip(&block.bwd_g_mlp[..l * d_m])
        {
            *dst += src;
        }
        gemm_tn_dev_accumulate(
            Some(gpu),
            g,
            &block.tape.x_norm[..l * d_m],
            l,
            rows,
            d_m,
            &mut w.grad,
        );
    }
    for t in (0..l).rev() {
        for idx in 0..stride {
            block.a_mat.grad[idx] += block.bwd_ssm_a[t * stride + idx];
        }
    }

    for t in (0..l).rev() {
        let x_id = block.tape.x_ids[t];
        let d_off = t * d_m;
        let inv_rms = block.tape.inv_rms[t];
        let e_t = &block.tape.x_raw[d_off..d_off + d_m];
        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            let gx_i = block.bwd_g_xnorm[d_off + i];
            block.norm_beta.grad[i] += gx_i;
            block.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);
            dot_gx_e += gx_i * block.norm_gamma.data[i] * e_t[i];
        }
        if input_is_embedding {
            let emb_row_off = x_id * d_m;
            for i in 0..d_m {
                let g_unnorm = block.bwd_g_xnorm[d_off + i] * block.norm_gamma.data[i];
                embed_w.grad[emb_row_off + i] +=
                    inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / d_m as f32));
            }
        }
    }
}

/// Ordered SSM adjoint for short chunks. This mirrors the old staged CPU
/// recurrence and uses only model-owned scalar scratch, avoiding rayon's job
/// setup on the path where a tree scan would be work-inefficient.
#[inline]
fn bwd_stage_ssm_sequential(m: &mut PSSALayerV2, seq_len: usize, input_is_embedding: bool) {
    let pending_step = m.step_counter + 1;
    let (embed_w, embed_row_marks, m) = (&mut m.embed_w, &mut m.embed_row_marks, &mut m.block);
    m.refresh_ssm_rates();
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let l = seq_len;
    let ssm_scale = 1.0 / (d_s as f32).sqrt();

    m.grad_h_next.fill(0.0);
    if input_is_embedding {
        for t in 0..l {
            embed_row_marks[m.tape.x_ids[t]] = pending_step;
        }
    }

    for t in (0..l).rev() {
        let x_id = m.tape.x_ids[t];
        let del_off = t * d_m;
        let m_off = t * d_m;
        m.buf_g_delta.fill(0.0);
        m.buf_g_b_proj.fill(0.0);
        m.buf_g_c_proj.fill(0.0);
        m.buf_g_h_prev.fill(0.0);

        for i in 0..d_m {
            let gz_i = m.bwd_g_zraw[m_off + i];
            let g_y_i = gz_i * ssm_scale + m.bwd_g_ysm[m_off + i];
            let d_i = m.tape.delta[del_off + i];
            let xn_i = m.tape.x_norm[m_off + i];
            for j in 0..d_s {
                let idx = i * d_s + j;
                let h_next = m.tape.h_states[(t + 1) * (d_m * d_s) + idx];
                let c_val = m.tape.c_proj[t * d_s + j];
                let bar_a = m.tape.bar_a[t * (d_m * d_s) + idx];
                let a_physical = m.ssm_rates[idx];
                let b_val = m.tape.b_proj[t * d_s + j];
                let g_h_total = g_y_i * c_val + m.grad_h_next[idx];
                m.buf_g_c_proj[j] += g_y_i * h_next;
                m.buf_g_h_prev[idx] += g_h_total * bar_a;
                m.a_mat.grad[idx] += g_h_total
                    * (d_i * bar_a)
                    * m.tape.h_states[t * (d_m * d_s) + idx]
                    * m.ssm_rate_derivatives[idx];
                m.buf_g_delta[i] += g_h_total
                    * (a_physical * bar_a * m.tape.h_states[t * (d_m * d_s) + idx] + b_val * xn_i);
                m.buf_g_b_proj[j] += g_h_total * (d_i * xn_i);
                m.bwd_g_xnorm[m_off + i] += g_h_total * m.tape.bar_b[t * (d_m * d_s) + idx];
            }
        }
        m.grad_h_next.copy_from_slice(&m.buf_g_h_prev);

        for i in 0..d_m {
            let gd_i = m.buf_g_delta[i] * sigmoid(m.tape.delta_raw[del_off + i]);
            let row_off = i * d_m;
            for j in 0..d_m {
                m.bwd_g_xnorm[m_off + j] += gd_i * m.w_delta.data[row_off + j];
                m.w_delta.grad[row_off + j] += gd_i * m.tape.x_norm[m_off + j];
            }
        }
        for j in 0..d_s {
            let gb_j = m.buf_g_b_proj[j];
            let gc_j = m.buf_g_c_proj[j];
            let row_off = j * d_m;
            for k in 0..d_m {
                m.bwd_g_xnorm[m_off + k] +=
                    gb_j * m.w_b.data[row_off + k] + gc_j * m.w_c.data[row_off + k];
                m.w_b.grad[row_off + k] += gb_j * m.tape.x_norm[m_off + k];
                m.w_c.grad[row_off + k] += gc_j * m.tape.x_norm[m_off + k];
            }
        }

        let inv_rms = m.tape.inv_rms[t];
        let e_t = &m.tape.x_raw[m_off..m_off + d_m];
        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            let gx_i = m.bwd_g_xnorm[m_off + i];
            m.norm_beta.grad[i] += gx_i;
            m.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);
            dot_gx_e += gx_i * m.norm_gamma.data[i] * e_t[i];
        }
        if input_is_embedding {
            let emb_row_off = x_id * d_m;
            for i in 0..d_m {
                let g_unnorm = m.bwd_g_xnorm[m_off + i] * m.norm_gamma.data[i];
                embed_w.grad[emb_row_off + i] +=
                    inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / d_m as f32));
            }
        }
    }
}

/// Backward Stage 3: reverse affine scan for the SSM adjoint, followed by
/// token-local derivatives and the projection/RMSNorm chains. The memory bank
/// query adjoint has already been computed by `bwd_stage_memory`; it is not part
/// of this associative recurrence.
#[inline]
pub fn bwd_stage_ssm(m: &mut PSSALayerV2, seq_len: usize) {
    bwd_stage_ssm_with_input(m, seq_len, true);
}

/// SSM backward with an explicit input kind. The normal batched path starts
/// at the embedding, while an Ouro pass starts at a continuous residual and
/// must return that residual adjoint instead of writing another embedding row.
pub(crate) fn bwd_stage_ssm_with_input(
    m: &mut PSSALayerV2,
    seq_len: usize,
    input_is_embedding: bool,
) {
    let _trace = crate::training_diagnostics::StageTrace::new(&m.device, "backward.ssm", seq_len);
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let stride = d_m * d_s;
    let gpu = gpu_ctx(m).filter(|g| g.accelerates_backward());
    if let Some(gpu) = gpu.as_ref() {
        if let Err(error) = bwd_stage_ssm_cuda(m, seq_len, input_is_embedding, gpu) {
            warn_gpu_fallback_once(
                &error,
                format!("warning: GPU SSM backward failed; using host fallback: {error}"),
            );
            bwd_stage_ssm_sequential_gpu(m, seq_len, input_is_embedding, gpu);
        }
        return;
    }
    if !parallel_scan_enabled(seq_len, stride) {
        bwd_stage_ssm_sequential(m, seq_len, input_is_embedding);
        return;
    }
    let executor = m.scan_executor.clone();
    executor.run(|| bwd_stage_ssm_parallel(m, seq_len, input_is_embedding, None));
}

#[inline]
fn bwd_stage_ssm_parallel(
    m: &mut PSSALayerV2,
    seq_len: usize,
    input_is_embedding: bool,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let stride = d_m * d_s;
    let pending_step = m.step_counter + 1;
    let (embed_w, embed_row_marks, m) = (&mut m.embed_w, &mut m.embed_row_marks, &mut m.block);
    m.refresh_ssm_rates();
    let l = seq_len;
    let ssm_scale = 1.0 / (d_s as f32).sqrt();

    if input_is_embedding {
        for t in 0..l {
            embed_row_marks[m.tape.x_ids[t]] = pending_step;
        }
    }

    // Let p_t be the adjoint propagated through the transition at t:
    // p_t = A_t * (r_t + p_(t+1)), where r_t is the direct readout adjoint.
    // In reverse time each token is therefore the affine map
    // p -> A_t*p + A_t*r_t. The exclusive reverse prefix gives p_(t+1)
    // (the terminal adjoint is zero because TBPTT detaches chunk boundaries).
    // Reverse-map construction is token-local too; only the reverse prefix
    // itself needs the tree's ordered composition.
    let bar_a = &m.tape.bar_a;
    let c_proj = &m.tape.c_proj;
    let g_zraw = &m.bwd_g_zraw;
    let g_ysm = &m.bwd_g_ysm;
    m.ssm_scan_a[..l * stride]
        .par_chunks_mut(stride)
        .zip(m.ssm_scan_b[..l * stride].par_chunks_mut(stride))
        .enumerate()
        .for_each(|(u, (a_row, b_row))| {
            let t = l - 1 - u;
            let t_state = t * stride;
            let t_c = t * d_s;
            let gy_off = t * d_m;
            for i in 0..d_m {
                let gy = g_zraw[gy_off + i] * ssm_scale + g_ysm[gy_off + i];
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let a = bar_a[t_state + idx];
                    let r = gy * c_proj[t_c + j];
                    a_row[idx] = a;
                    b_row[idx] = a * r;
                }
            }
        });
    affine_scan_in_place(&mut m.ssm_scan_a, &mut m.ssm_scan_b, l, stride);

    // The scan has removed the reverse-time dependency. All local derivatives
    // now write disjoint token rows, so this bulk is parallel over time.
    let delta_grads = &mut m.bwd_ssm_delta[..l * d_m];
    let b_grads = &mut m.bwd_ssm_b[..l * d_s];
    let c_grads = &mut m.bwd_ssm_c[..l * d_s];
    let a_grads = &mut m.bwd_ssm_a[..l * stride];
    let x_grads = &mut m.bwd_g_xnorm[..l * d_m];
    delta_grads.fill(0.0);
    b_grads.fill(0.0);
    c_grads.fill(0.0);
    a_grads.fill(0.0);
    delta_grads
        .par_chunks_mut(d_m)
        .zip(b_grads.par_chunks_mut(d_s))
        .zip(c_grads.par_chunks_mut(d_s))
        .zip(a_grads.par_chunks_mut(stride))
        .zip(x_grads.par_chunks_mut(d_m))
        .enumerate()
        .for_each(|(t, ((((delta_row, b_row), c_row), a_row), x_row))| {
            let state_off = t * stride;
            let d_off = t * d_m;
            let s_off = t * d_s;
            let future_u = l - 1 - t;
            let future_p = &m.ssm_scan_b[future_u * stride..(future_u + 1) * stride];
            let gy_row = &m.bwd_g_zraw[d_off..d_off + d_m];
            let mem_gy_row = &m.bwd_g_ysm[d_off..d_off + d_m];
            for i in 0..d_m {
                let gy = gy_row[i] * ssm_scale + mem_gy_row[i];
                let d_i = m.tape.delta[d_off + i];
                let x_i = m.tape.x_norm[d_off + i];
                for j in 0..d_s {
                    let idx = i * d_s + j;
                    let q = gy * m.tape.c_proj[s_off + j] + future_p[idx];
                    let a = m.tape.bar_a[state_off + idx];
                    let h_prev = m.tape.h_states[state_off + idx];
                    let h_next = m.tape.h_states[state_off + stride + idx];
                    let b = m.tape.b_proj[s_off + j];
                    let rate = m.ssm_rates[idx];

                    c_row[j] += gy * h_next;
                    a_row[idx] = q * (d_i * a) * h_prev * m.ssm_rate_derivatives[idx];
                    delta_row[i] += q * (rate * a * h_prev + b * x_i);
                    b_row[j] += q * (d_i * x_i);
                    x_row[i] += q * m.tape.bar_b[state_off + idx];
                }
            }
            for i in 0..d_m {
                delta_row[i] *= sigmoid(m.tape.delta_raw[d_off + i]);
            }
        });

    // These are ordinary token reductions, not temporal dependencies. Reuse
    // the existing deterministic dense-adjoint kernels for the shared
    // projection weights and add their input adjoints to the memory/adapter
    // contributions already in bwd_g_xnorm.
    for (g, w, rows) in [
        (&m.bwd_ssm_delta[..l * d_m], &mut m.w_delta, d_m),
        (&m.bwd_ssm_b[..l * d_s], &mut m.w_b, d_s),
        (&m.bwd_ssm_c[..l * d_s], &mut m.w_c, d_s),
    ] {
        if let Some(gpu) = gpu {
            gemm_nn_dev_into(
                Some(gpu),
                g,
                &w.data,
                l,
                rows,
                d_m,
                &mut m.bwd_g_mlp[..l * d_m],
            );
            gemm_tn_dev_accumulate(
                Some(gpu),
                g,
                &m.tape.x_norm[..l * d_m],
                l,
                rows,
                d_m,
                &mut w.grad,
            );
        } else {
            dense_input_adjoint(g, &w.data, l, rows, d_m, &mut m.bwd_g_mlp[..l * d_m]);
            dense_weight_adjoint(g, &m.tape.x_norm[..l * d_m], l, rows, d_m, &mut w.grad);
        }
        for (dst, src) in m.bwd_g_xnorm[..l * d_m]
            .iter_mut()
            .zip(&m.bwd_g_mlp[..l * d_m])
        {
            *dst += src;
        }
    }
    for t in (0..l).rev() {
        for idx in 0..stride {
            m.a_mat.grad[idx] += m.bwd_ssm_a[t * stride + idx];
        }
    }

    // RMSNorm parameters and embedding rows can alias across tokens. Their
    // sums are mathematically associative, but retain the reference reverse
    // order for deterministic f32 accumulation without atomics.
    for t in (0..l).rev() {
        let x_id = m.tape.x_ids[t];
        let d_off = t * d_m;
        let inv_rms = m.tape.inv_rms[t];
        let e_t = &m.tape.x_raw[d_off..d_off + d_m];
        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            let gx_i = m.bwd_g_xnorm[d_off + i];
            m.norm_beta.grad[i] += gx_i;
            m.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);
            dot_gx_e += gx_i * m.norm_gamma.data[i] * e_t[i];
        }
        if input_is_embedding {
            let emb_row_off = x_id * d_m;
            for i in 0..d_m {
                let g_unnorm = m.bwd_g_xnorm[d_off + i] * m.norm_gamma.data[i];
                embed_w.grad[emb_row_off + i] +=
                    inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / d_m as f32));
            }
        }
    }
}

/// Return the adjoint through the affine RMSNorm into the raw continuous
/// input. This is separate from the embedding write in `bwd_stage_ssm` so a
/// later shared loop can feed its residual derivative to the previous pass.
pub(crate) fn bwd_stage_input_norm(
    b: &PSSAContinuousBlockV2,
    seq_len: usize,
    input_adjoints: &mut [f32],
) {
    let d_m = b.cfg.d_latent;
    assert_eq!(input_adjoints.len(), seq_len * d_m);
    for t in 0..seq_len {
        let off = t * d_m;
        let inv_rms = b.tape.inv_rms[t];
        let e_t = &b.tape.x_raw[off..off + d_m];
        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            dot_gx_e += b.bwd_g_xnorm[off + i] * b.norm_gamma.data[i] * e_t[i];
        }
        for i in 0..d_m {
            let g_unnorm = b.bwd_g_xnorm[off + i] * b.norm_gamma.data[i];
            input_adjoints[off + i] =
                inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / d_m as f32));
        }
    }
}

#[inline]
fn stacked_backward_mlp(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let mlp = 2 * d;
    let gz = &block.bwd_g_zfinal[..seq_len * d];
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(
            Some(gpu),
            gz,
            &block.mlp_w2.data,
            seq_len,
            d,
            mlp,
            &mut block.bwd_g_mlp[..seq_len * mlp],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            gz,
            &block.tape.mlp_act[..seq_len * mlp],
            seq_len,
            d,
            mlp,
            &mut block.mlp_w2.grad,
        );
    } else {
        dense_input_adjoint(
            gz,
            &block.mlp_w2.data,
            seq_len,
            d,
            mlp,
            &mut block.bwd_g_mlp[..seq_len * mlp],
        );
        dense_weight_adjoint(
            gz,
            &block.tape.mlp_act[..seq_len * mlp],
            seq_len,
            d,
            mlp,
            &mut block.mlp_w2.grad,
        );
    }
    for i in 0..seq_len * mlp {
        let h = block.tape.mlp_hidden[i];
        let sig = sigmoid(h);
        block.bwd_g_mlp[i] *= sig * (1.0 + h * (1.0 - sig));
    }
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_mlp[..seq_len * mlp],
            &block.mlp_w1.data,
            seq_len,
            mlp,
            d,
            &mut block.bwd_g_zraw[..seq_len * d],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_mlp[..seq_len * mlp],
            &block.tape.z_raw[..seq_len * d],
            seq_len,
            mlp,
            d,
            &mut block.mlp_w1.grad,
        );
    } else {
        dense_input_adjoint(
            &block.bwd_g_mlp[..seq_len * mlp],
            &block.mlp_w1.data,
            seq_len,
            mlp,
            d,
            &mut block.bwd_g_zraw[..seq_len * d],
        );
        dense_weight_adjoint(
            &block.bwd_g_mlp[..seq_len * mlp],
            &block.tape.z_raw[..seq_len * d],
            seq_len,
            mlp,
            d,
            &mut block.mlp_w1.grad,
        );
    }
    for i in 0..seq_len * d {
        block.bwd_g_zraw[i] += block.bwd_g_zfinal[i];
    }
}

#[inline]
fn stacked_backward_adapter(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let rank = block.adapters[0].rank;
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_zraw[..seq_len * d],
            &block.adapter_up_effective,
            seq_len,
            d,
            rank,
            &mut block.bwd_g_ad_down[..seq_len * rank],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_zraw[..seq_len * d],
            &block.tape.adapter_act[..seq_len * rank],
            seq_len,
            d,
            rank,
            &mut block.adapters[0].up_proj.grad,
        );
    } else {
        adapter_up_input_adjoint(
            &block.bwd_g_zraw[..seq_len * d],
            &block.adapters[0].up_proj.data,
            &block.adapters[0].consolidated_up,
            seq_len,
            d,
            rank,
            &mut block.bwd_g_ad_down[..seq_len * rank],
        );
        dense_weight_adjoint_forward(
            &block.bwd_g_zraw[..seq_len * d],
            &block.tape.adapter_act[..seq_len * rank],
            seq_len,
            d,
            rank,
            &mut block.adapters[0].up_proj.grad,
        );
    }
    for t in 0..seq_len {
        for r in 0..rank {
            let off = t * rank + r;
            let h = block.tape.adapter_hidden[off];
            let sig = sigmoid(h);
            block.bwd_g_ad_down[off] *= sig * (1.0 + h * (1.0 - sig));
        }
    }
}

#[inline]
fn stacked_backward_adapter_down(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let rank = block.adapters[0].rank;
    block.bwd_g_xnorm[..seq_len * d].fill(0.0);
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_ad_down[..seq_len * rank],
            &block.adapters[0].down_proj.data,
            seq_len,
            rank,
            d,
            &mut block.bwd_g_xnorm[..seq_len * d],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_ad_down[..seq_len * rank],
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            rank,
            d,
            &mut block.adapters[0].down_proj.grad,
        );
    } else {
        dense_input_adjoint(
            &block.bwd_g_ad_down[..seq_len * rank],
            &block.adapters[0].down_proj.data,
            seq_len,
            rank,
            d,
            &mut block.bwd_g_xnorm[..seq_len * d],
        );
        dense_weight_adjoint_forward(
            &block.bwd_g_ad_down[..seq_len * rank],
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            rank,
            d,
            &mut block.adapters[0].down_proj.grad,
        );
    }
}

#[inline]
fn stacked_backward_memory(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let k = block.cfg.d_mem_key;
    let cap = block.cfg.mem_capacity;
    let g_m_proj = &mut block.bwd_g_zfinal[..seq_len * d];
    let (g_gate, gate_x) = block.bwd_g_mlp[..2 * seq_len * d].split_at_mut(seq_len * d);
    let g_zraw = &block.bwd_g_zraw[..seq_len * d];
    let g_mem = &block.tape.g_mem[..seq_len * d];
    let m_proj = &block.tape.m_proj[..seq_len * d];
    for t in 0..seq_len {
        let off = t * d;
        for i in 0..d {
            let gz = g_zraw[off + i];
            let gm = g_mem[off + i];
            g_m_proj[off + i] = gz * gm;
            g_gate[off + i] = gz * m_proj[off + i] * gm * (1.0 - gm);
        }
    }
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(Some(gpu), g_gate, &block.w_gate.data, seq_len, d, d, gate_x);
        for i in 0..seq_len * d {
            block.bwd_g_xnorm[i] += gate_x[i];
        }
        gemm_nn_dev_into(
            Some(gpu),
            g_m_proj,
            &block.w_proj.data,
            seq_len,
            d,
            d,
            gate_x,
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            g_gate,
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            d,
            d,
            &mut block.w_gate.grad,
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            g_m_proj,
            &block.tape.m_val[..seq_len * d],
            seq_len,
            d,
            d,
            &mut block.w_proj.grad,
        );
    } else {
        dense_input_adjoint(g_gate, &block.w_gate.data, seq_len, d, d, gate_x);
        for i in 0..seq_len * d {
            block.bwd_g_xnorm[i] += gate_x[i];
        }
        dense_input_adjoint(g_m_proj, &block.w_proj.data, seq_len, d, d, gate_x);
        dense_weight_adjoint_forward(
            g_gate,
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            d,
            d,
            &mut block.w_gate.grad,
        );
        dense_weight_adjoint_forward(
            g_m_proj,
            &block.tape.m_val[..seq_len * d],
            seq_len,
            d,
            d,
            &mut block.w_proj.grad,
        );
    }

    for t in (0..seq_len).rev() {
        let q_off = t * k;
        let m_off = t * d;
        let q = &block.tape.q_poincare[q_off..q_off + k];
        let q_sq = crate::memory::HyperbolicEpisodicBankV2::squared_norm(q) as f64;
        block.g_query_pnc.fill(0.0);
        let g_m_val = &gate_x[m_off..m_off + d];
        for entry in 0..block.memory.count {
            let key_off = entry * k;
            let key = &block.memory.keys[key_off..key_off + k];
            let key_sq = block.memory.norm_sq[entry] as f64;
            let value_off = entry * d;
            let mut dot = 0.0f64;
            for j in 0..d {
                dot += g_m_val[j] as f64
                    * (block.memory.values[value_off + j] - block.tape.m_val[m_off + j]) as f64;
            }
            let g_score = block.tape.mem_weights[t * cap + entry] as f64 * dot;
            let mut sq = 0.0f64;
            for j in 0..k {
                let diff = q[j] as f64 - key[j] as f64;
                sq += diff * diff;
            }
            if sq > 0.0 {
                let denom = (1.0 - q_sq) * (1.0 - key_sq);
                assert!(denom > 0.0 && denom.is_finite());
                let z = sq / denom;
                let dd_dz = 1.0 / (z * (1.0 + z)).sqrt();
                for j in 0..k {
                    let diff = q[j] as f64 - key[j] as f64;
                    let ddenom = -2.0 * q[j] as f64 * (1.0 - key_sq);
                    let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                    block.g_query_pnc[j] +=
                        (g_score * (-1.0 / block.cfg.tau_mem as f64) * dd_dz * dz) as f32;
                }
            }
        }
        crate::memory::HyperbolicEpisodicBankV2::projection_adjoint(
            &block.tape.q_euc[q_off..q_off + k],
            &block.g_query_pnc,
            &mut block.g_query_euc,
        );
        if gpu.is_some() {
            block.bwd_g_query_euc[q_off..q_off + k].copy_from_slice(&block.g_query_euc);
            block.bwd_g_ysm[m_off..m_off + d].fill(0.0);
        } else {
            block.g_y_ssm.fill(0.0);
            let xn = &block.tape.x_norm[m_off..m_off + d];
            let y = &block.tape.y_ssm[m_off..m_off + d];
            for r in 0..k {
                let gq = block.g_query_euc[r];
                let row = r * d;
                for j in 0..d {
                    block.w_qx.grad[row + j] += gq * xn[j];
                    block.w_qh.grad[row + j] += gq * y[j];
                    block.bwd_g_xnorm[m_off + j] += gq * block.w_qx.data[row + j];
                    block.g_y_ssm[j] += gq * block.w_qh.data[row + j];
                }
            }
            block.bwd_g_ysm[m_off..m_off + d].copy_from_slice(&block.g_y_ssm);
        }
    }
    if let Some(gpu) = gpu {
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_query_euc[..seq_len * k],
            &block.tape.x_norm[..seq_len * d],
            seq_len,
            k,
            d,
            &mut block.w_qx.grad,
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_query_euc[..seq_len * k],
            &block.tape.y_ssm[..seq_len * d],
            seq_len,
            k,
            d,
            &mut block.w_qh.grad,
        );
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_query_euc[..seq_len * k],
            &block.w_qx.data,
            seq_len,
            k,
            d,
            &mut block.bwd_g_mlp[..seq_len * d],
        );
        for (dst, src) in block.bwd_g_xnorm[..seq_len * d]
            .iter_mut()
            .zip(&block.bwd_g_mlp[..seq_len * d])
        {
            *dst += src;
        }
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_query_euc[..seq_len * k],
            &block.w_qh.data,
            seq_len,
            k,
            d,
            &mut block.bwd_g_ysm[..seq_len * d],
        );
    }
}

#[inline]
fn stacked_backward_ssm(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    seq_len: usize,
    input_adjoints: &mut [f32],
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = block.cfg.d_latent;
    let s = block.cfg.d_state;
    let stride = d * s;
    assert_eq!(input_adjoints.len(), seq_len * d);
    block.refresh_ssm_rates();
    block.grad_h_next.fill(0.0);
    for t in (0..seq_len).rev() {
        let d_off = t * d;
        block.buf_g_delta.fill(0.0);
        block.buf_g_b_proj.fill(0.0);
        block.buf_g_c_proj.fill(0.0);
        block.buf_g_h_prev.fill(0.0);
        for i in 0..d {
            let gz = block.bwd_g_zraw[d_off + i];
            let gy = gz / (s as f32).sqrt() + block.bwd_g_ysm[d_off + i];
            let delta = block.tape.delta[d_off + i];
            for j in 0..s {
                let idx = i * s + j;
                let h_next = block.tape.h_states[(t + 1) * stride + idx];
                let h_prev = block.tape.h_states[t * stride + idx];
                let c = block.tape.c_proj[t * s + j];
                let a = block.tape.bar_a[t * stride + idx];
                let rate = block.ssm_rates[idx];
                let b = block.tape.b_proj[t * s + j];
                let gh = gy * c + block.grad_h_next[idx];
                block.buf_g_c_proj[j] += gy * h_next;
                block.buf_g_h_prev[idx] += gh * a;
                block.a_mat.grad[idx] +=
                    gh * (delta * a) * h_prev * block.ssm_rate_derivatives[idx];
                block.buf_g_delta[i] += gh * (rate * a * h_prev + b * block.tape.x_norm[d_off + i]);
                block.buf_g_b_proj[j] += gh * (delta * block.tape.x_norm[d_off + i]);
                block.bwd_g_xnorm[d_off + i] += gh * block.tape.bar_b[t * stride + idx];
            }
        }
        block.grad_h_next.copy_from_slice(&block.buf_g_h_prev);
        if gpu.is_some() {
            for i in 0..d {
                block.bwd_ssm_delta[d_off + i] =
                    block.buf_g_delta[i] * sigmoid(block.tape.delta_raw[d_off + i]);
            }
            for j in 0..s {
                block.bwd_ssm_b[t * s + j] = block.buf_g_b_proj[j];
                block.bwd_ssm_c[t * s + j] = block.buf_g_c_proj[j];
            }
        } else {
            for i in 0..d {
                let gd = block.buf_g_delta[i] * sigmoid(block.tape.delta_raw[d_off + i]);
                let row = i * d;
                for j in 0..d {
                    block.bwd_g_xnorm[d_off + j] += gd * block.w_delta.data[row + j];
                    block.w_delta.grad[row + j] += gd * block.tape.x_norm[d_off + j];
                }
            }
            for j in 0..s {
                let row = j * d;
                for k in 0..d {
                    block.bwd_g_xnorm[d_off + k] += block.buf_g_b_proj[j] * block.w_b.data[row + k]
                        + block.buf_g_c_proj[j] * block.w_c.data[row + k];
                    block.w_b.grad[row + k] += block.buf_g_b_proj[j] * block.tape.x_norm[d_off + k];
                    block.w_c.grad[row + k] += block.buf_g_c_proj[j] * block.tape.x_norm[d_off + k];
                }
            }
        }

        if gpu.is_none() {
            let inv_rms = block.tape.inv_rms[t];
            let raw = &block.tape.x_raw[d_off..d_off + d];
            let mut dot = 0.0f32;
            for i in 0..d {
                let gx = block.bwd_g_xnorm[d_off + i];
                block.norm_beta.grad[i] += gx;
                block.norm_gamma.grad[i] += gx * (raw[i] * inv_rms);
                dot += gx * block.norm_gamma.data[i] * raw[i];
            }
            for i in 0..d {
                let gx = block.bwd_g_xnorm[d_off + i] * block.norm_gamma.data[i];
                input_adjoints[d_off + i] =
                    inv_rms * (gx - raw[i] * (dot * inv_rms * inv_rms / d as f32));
            }
        }
    }
    if let Some(gpu) = gpu {
        let d = block.cfg.d_latent;
        let s = block.cfg.d_state;
        for (g, w, rows) in [
            (&block.bwd_ssm_delta[..seq_len * d], &mut block.w_delta, d),
            (&block.bwd_ssm_b[..seq_len * s], &mut block.w_b, s),
            (&block.bwd_ssm_c[..seq_len * s], &mut block.w_c, s),
        ] {
            gemm_nn_dev_into(
                Some(gpu),
                g,
                &w.data,
                seq_len,
                rows,
                d,
                &mut block.bwd_g_mlp[..seq_len * d],
            );
            for (dst, src) in block.bwd_g_xnorm[..seq_len * d]
                .iter_mut()
                .zip(&block.bwd_g_mlp[..seq_len * d])
            {
                *dst += src;
            }
            gemm_tn_dev_accumulate(
                Some(gpu),
                g,
                &block.tape.x_norm[..seq_len * d],
                seq_len,
                rows,
                d,
                &mut w.grad,
            );
        }
        for t in (0..seq_len).rev() {
            let d_off = t * d;
            let inv_rms = block.tape.inv_rms[t];
            let raw = &block.tape.x_raw[d_off..d_off + d];
            let mut dot = 0.0f32;
            for i in 0..d {
                let gx = block.bwd_g_xnorm[d_off + i];
                block.norm_beta.grad[i] += gx;
                block.norm_gamma.grad[i] += gx * (raw[i] * inv_rms);
                dot += gx * block.norm_gamma.data[i] * raw[i];
            }
            for i in 0..d {
                let gx = block.bwd_g_xnorm[d_off + i] * block.norm_gamma.data[i];
                input_adjoints[d_off + i] =
                    inv_rms * (gx - raw[i] * (dot * inv_rms * inv_rms / d as f32));
            }
        }
    }
}

#[inline]
fn stacked_backward_block(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    output_adjoints: &[f32],
    seq_len: usize,
    input_adjoints: &mut [f32],
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    block.bwd_g_zfinal[..seq_len * block.cfg.d_latent].copy_from_slice(output_adjoints);
    stacked_backward_mlp(block, seq_len, gpu);
    stacked_backward_adapter(block, seq_len, gpu);
    stacked_backward_adapter_down(block, seq_len, gpu);
    stacked_backward_memory(block, seq_len, gpu);
    stacked_backward_ssm(block, seq_len, input_adjoints, gpu);
}

/// Backward a continuous block from the base tape workspace. The outer layer
/// owns residual and loop bookkeeping; this function only performs one block
/// VJP and returns its detached raw-input adjoint.
pub(crate) fn backward_continuous_block_batched(
    block: &mut crate::pssa::PSSAContinuousBlockV2,
    output_adjoints: &[f32],
    seq_len: usize,
    input_adjoints: &mut [f32],
    gpu: &crate::backend::GpuDispatch,
) {
    stacked_backward_block(block, output_adjoints, seq_len, input_adjoints, Some(gpu));
}

#[inline]
pub(crate) fn stacked_backward_logits(
    m: &mut PSSALayerV2,
    seq_len: usize,
    scale_loss: f32,
    gpu: Option<&crate::backend::GpuDispatch>,
) {
    let d = m.cfg.d_latent;
    let v = m.cfg.d_vocab;
    let logit_scale = 1.0 / (d as f32).sqrt();
    let final_z = &m.continuous_inputs[..seq_len * d];
    let (unembed, block) = (&mut m.unembed_w, &mut m.block);
    for t in 0..seq_len {
        let log_off = t * v;
        let target = block.tape.target_ids[t];
        for i in 0..v {
            let indicator = if i == target { 1.0 } else { 0.0 };
            block.bwd_g_logits[log_off + i] =
                (block.tape.probs[log_off + i] - indicator) * scale_loss * logit_scale;
        }
    }
    if let Some(gpu) = gpu {
        gemm_nn_dev_into(
            Some(gpu),
            &block.bwd_g_logits[..seq_len * v],
            &unembed.data,
            seq_len,
            v,
            d,
            &mut block.bwd_g_zfinal[..seq_len * d],
        );
        gemm_tn_dev_accumulate(
            Some(gpu),
            &block.bwd_g_logits[..seq_len * v],
            final_z,
            seq_len,
            v,
            d,
            &mut unembed.grad,
        );
    } else {
        dense_input_adjoint(
            &block.bwd_g_logits[..seq_len * v],
            &unembed.data,
            seq_len,
            v,
            d,
            &mut block.bwd_g_zfinal[..seq_len * d],
        );
        dense_weight_adjoint(
            &block.bwd_g_logits[..seq_len * v],
            final_z,
            seq_len,
            v,
            d,
            &mut unembed.grad,
        );
    }
}

fn backward_chunk_stacked_batched(m: &mut PSSALayerV2, seq_len: usize, accumulation_scale: f32) {
    let gpu = gpu_ctx(m).filter(|g| g.accelerates_backward());
    assert!(seq_len > 0 && seq_len <= m.cfg.chunk_len);
    assert!(accumulation_scale.is_finite());
    let d = m.cfg.d_latent;
    let n = seq_len * d;
    stacked_backward_logits(
        m,
        seq_len,
        accumulation_scale / seq_len as f32,
        gpu.as_ref(),
    );
    m.output_adjoints[..n].copy_from_slice(&m.block.bwd_g_zfinal[..n]);
    let depth = m.depth();
    m.boundary_adjoints[depth][..n].copy_from_slice(&m.output_adjoints[..n]);
    for layer in (0..m.extra_blocks.len()).rev() {
        let scale = m.residual_scales[layer];
        for i in 0..n {
            m.residual_block_adjoints[i] = scale * m.output_adjoints[i];
        }
        stacked_backward_block(
            &mut m.extra_blocks[layer],
            &m.residual_block_adjoints[..n],
            seq_len,
            &mut m.residual_input_adjoints[..n],
            gpu.as_ref(),
        );
        for i in 0..n {
            m.output_adjoints[i] += m.residual_input_adjoints[i];
        }
        m.boundary_adjoints[layer + 1][..n].copy_from_slice(&m.output_adjoints[..n]);
    }
    stacked_backward_block(
        &mut m.block,
        &m.output_adjoints[..n],
        seq_len,
        &mut m.input_adjoints[..n],
        gpu.as_ref(),
    );
    m.boundary_adjoints[0][..n].copy_from_slice(&m.input_adjoints[..n]);
    let pending_step = m
        .step_counter
        .checked_add(1)
        .expect("optimizer step counter overflow");
    for t in (0..seq_len).rev() {
        let id = m.block.tape.x_ids[t];
        // Resident CUDA clipping uploads/clears only the rows touched by this
        // optimizer group, including every replayed lane in stacked batches.
        m.embed_row_marks[id] = pending_step;
        let row = id * d;
        for i in 0..d {
            m.embed_w.grad[row + i] += m.input_adjoints[t * d + i];
        }
    }
}

/// Full batched backward pass over a chunk: reverse affine SSM scan followed
/// by local derivatives and deterministic shared-weight reductions, matching
/// the scalar `backward_chunk` to f32 roundoff.
pub fn backward_chunk_batched(m: &mut PSSALayerV2, seq_len: usize, accumulation_scale: f32) {
    if m.local_mixing_enabled() {
        assert!(!m.device.is_gpu(), "local mixing is CPU-only");
        return m.backward_chunk(seq_len, accumulation_scale);
    }
    if m.loops() > 1 {
        return m.backward_chunk(seq_len, accumulation_scale);
    }
    if m.depth() > 1 {
        if m.loops() == 1 {
            return backward_chunk_stacked_batched(m, seq_len, accumulation_scale);
        }
        m.backward_chunk(seq_len, accumulation_scale);
        return;
    }
    assert!(
        seq_len > 0 && seq_len <= m.cfg.chunk_len,
        "backward sequence length must be within tape capacity"
    );
    assert!(accumulation_scale.is_finite());
    let scale_loss = accumulation_scale / (seq_len as f32);

    bwd_stage_logits(m, seq_len, scale_loss);
    bwd_stage_mlp(m, seq_len);
    bwd_stage_adapter(m, seq_len);
    bwd_stage_adapter_down(m, seq_len);
    bwd_stage_memory(m, seq_len);
    bwd_stage_ssm(m, seq_len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_scan_keeps_cpu_exclusive_prefix_contract_for_short_and_padded_rows() {
        for &(len, stride) in &[(1usize, 1usize), (3, 2), (7, 3), (33, 4)] {
            let width = len.next_power_of_two();
            let mut a = vec![0.0; width * stride];
            let mut b = vec![0.0; width * stride];
            for t in 0..len {
                for c in 0..stride {
                    a[t * stride + c] = 0.8 + 0.01 * (t + c) as f32;
                    b[t * stride + c] = -0.2 + 0.03 * (2 * t + c) as f32;
                }
            }
            let input_a = a.clone();
            let input_b = b.clone();
            affine_scan_in_place(&mut a, &mut b, len, stride);
            for c in 0..stride {
                let mut pa = 1.0f32;
                let mut pb = 0.0f32;
                for t in 0..len {
                    assert!((a[t * stride + c] - pa).abs() < 2e-6);
                    assert!((b[t * stride + c] - pb).abs() < 1e-4);
                    let aa = input_a[t * stride + c];
                    let bb = input_b[t * stride + c];
                    pb = aa * pb + bb;
                    pa *= aa;
                }
            }
        }
    }
}
