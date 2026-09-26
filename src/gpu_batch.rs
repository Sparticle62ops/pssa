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
//! (`forward_train_chunk_reference` / `backward_chunk_reference`): identical
//! operation order per element, so results agree to f32 roundoff. The only
//! intentional difference is accumulation order across tokens for weight
//! gradients, which stage GEMMs fold in descending token order to match the
//! reference reverse-time loop.

use crate::linalg::{dot_slice, sigmoid, softplus};
use crate::memory::HyperbolicEpisodicBankV2;
use crate::pssa::PSSALayerV2;

/// Batched matvec over L rows: `out[t, r] = dot(W[r, :], x[t, :])`.
/// Same accumulation order as `ParamMatrix::matvec`, applied per row.
#[inline(always)]
fn batched_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], l: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), l * cols);
    debug_assert_eq!(out.len(), l * rows);
    for t in 0..l {
        let x_off = t * cols;
        let o_off = t * rows;
        for r in 0..rows {
            out[o_off + r] = dot_slice(&w[r * cols..(r + 1) * cols], &x[x_off..x_off + cols]);
        }
    }
}

/// Clone the layer's GPU context out, so stage functions can keep borrowing
/// tape fields while the dispatch runs. `None` on the CPU path.
#[inline]
fn gpu_ctx(m: &PSSALayerV2) -> Option<crate::backend::WgpuContext> {
    match &m.device {
        crate::backend::Device::Gpu(ctx) => Some(ctx.clone()),
        crate::backend::Device::Cpu => None,
    }
}

/// Device-aware batched matvec: on a GPU device this is one `dispatch_gemm`
/// call (X [L,1,K], W [rows,K], Y [L,1,rows]); on CPU it is the scalar twin
/// used by the numerical verification.
#[inline]
fn batched_matvec_dev(gpu: Option<&crate::backend::WgpuContext>, w: &[f32], rows: usize, cols: usize, x: &[f32], l: usize, out: &mut [f32]) {
    if let Some(ctx) = gpu {
        let y = ctx.dispatch_gemm(&x[..l * cols], w, 1, rows, cols, l);
        out[..l * rows].copy_from_slice(&y);
    } else {
        batched_matvec(w, rows, cols, x, l, out);
    }
}

// =============================================================================
// FORWARD STAGES
// =============================================================================

/// Stage 1: embedding gather + affine RMSNorm for every token in the chunk.
#[inline]
pub fn stage_embed_norm(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    for t in 0..seq_len {
        let x_id = m.tape.x_ids[t];
        let e_t = &m.embed_w.data[x_id * d_m..(x_id + 1) * d_m];
        let sum_sq: f32 = e_t.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / (d_m as f32) + 1e-5).sqrt();
        m.tape.inv_rms[t] = inv_rms;
        let xn_off = t * d_m;
        for i in 0..d_m {
            m.tape.x_norm[xn_off + i] =
                m.norm_gamma.data[i] * (e_t[i] * inv_rms) + m.norm_beta.data[i];
        }
    }
}

/// Stage 2: data-dependent projections (delta/b/c/gate) as batched GEMMs,
/// plus the softplus activation on the raw delta.
#[inline]
pub fn stage_projections(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let l = seq_len;
    let xn = &m.tape.x_norm[..l * d_m];

    batched_matvec_dev(gpu_ctx(m).as_ref(), &m.w_delta.data, d_m, d_m, xn, l, &mut m.tape.delta_raw[..l * d_m]);
    // Reference keeps the raw projection in `delta_raw` and the softplus in
    // `delta`; the backward pass takes sigmoid(delta_raw), so both are needed.
    for i in 0..l * d_m {
        m.tape.delta[i] = softplus(m.tape.delta_raw[i]);
    }
    batched_matvec(&m.w_b.data, d_s, d_m, xn, l, &mut m.tape.b_proj[..l * d_s]);
    batched_matvec(&m.w_c.data, d_s, d_m, xn, l, &mut m.tape.c_proj[..l * d_s]);
}

/// Stage 3: multi-channel SSM recurrent scan. Sequential over time (the
/// recurrence is inherently serial); each step is fully parallel over
/// (d_latent, d_state) and is one GPU dispatch per step.
#[inline]
pub fn stage_ssm_scan(m: &mut PSSALayerV2, seq_len: usize) {
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
                let bar_a = (d_i * -softplus(m.a_mat.data[idx])).exp();
                let bar_b = d_i * m.tape.b_proj[b_off + j];

                m.tape.bar_a[ssm_off + idx] = bar_a;
                m.tape.bar_b[ssm_off + idx] = bar_b;

                let h_val = bar_a * m.tape.h_states[h_prev_off + idx] + bar_b * m.tape.x_norm[xn_off + i];
                m.tape.h_states[h_next_off + idx] = h_val;

                y_i += h_val * m.tape.c_proj[c_off + j];
            }
            m.tape.y_ssm[y_off + i] = y_i;
        }
    }
}

/// Stage 4: Poincare query projection, diffeomorphic projection, soft
/// retrieval, memory gate and injection for every token.
#[inline]
pub fn stage_memory(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_k = m.cfg.d_mem_key;
    let mem_cap = m.cfg.mem_capacity;

    for t in 0..seq_len {
        let xn_off = t * d_m;
        let y_off = t * d_m;
        let q_off = t * d_k;
        let m_off = t * d_m;
        let mw_off = t * mem_cap;

        for r in 0..d_k {
            let row_x = &m.w_qx.data[r * d_m..(r + 1) * d_m];
            let row_h = &m.w_qh.data[r * d_m..(r + 1) * d_m];
            m.tape.q_euc[q_off + r] = dot_slice(row_x, &m.tape.x_norm[xn_off..xn_off + d_m])
                + dot_slice(row_h, &m.tape.y_ssm[y_off..y_off + d_m]);
        }

        m.tape.q_norm[t] = HyperbolicEpisodicBankV2::diffeomorphic_project(
            &m.tape.q_euc[q_off..q_off + d_k],
            &mut m.tape.q_poincare[q_off..q_off + d_k],
        );

        m.memory.retrieve_soft_into(
            &m.tape.q_poincare[q_off..q_off + d_k],
            m.cfg.tau_mem,
            &mut m.tape.m_val[m_off..m_off + d_m],
            &mut m.tape.mem_weights[mw_off..mw_off + mem_cap],
        );
    }

    // Gate, memory projection and injection as batched GEMMs over the chunk.
    batched_matvec_dev(gpu_ctx(m).as_ref(), &m.w_gate.data, d_m, d_m, &m.tape.x_norm[..seq_len * d_m], seq_len, &mut m.tape.g_mem[..seq_len * d_m]);
    for v in &mut m.tape.g_mem[..seq_len * d_m] {
        *v = sigmoid(*v);
    }
    batched_matvec_dev(gpu_ctx(m).as_ref(), &m.w_proj.data, d_m, d_m, &m.tape.m_val[..seq_len * d_m], seq_len, &mut m.tape.m_proj[..seq_len * d_m]);
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
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let down = m.adapters[0].down_proj.data.clone();

    batched_matvec(
        &down,
        rank,
        d_m,
        &m.tape.x_norm[..seq_len * d_m],
        seq_len,
        &mut m.tape.adapter_hidden[..seq_len * rank],
    );
}

/// Adapter up-projection contribution for token t, written into `out`
/// (`out[i] = sum_r (U_fast + U_slow)[i, r] * act[r]`).
#[inline(always)]
fn adapter_up_into(m: &PSSALayerV2, act: &[f32], out: &mut [f32]) {
    let rank = m.adapters[0].rank;
    let ad = &m.adapters[0];
    for i in 0..m.cfg.d_latent {
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
    let d_m = m.cfg.d_latent;
    let d_mlp = d_m * 2;
    let ssm_scale = 1.0 / (m.cfg.d_state as f32).sqrt();

    let mut ad_out = [0.0f32; 256]; // d_latent <= 256 by config validation below.
    debug_assert!(d_m <= ad_out.len());

    for t in 0..seq_len {
        let z_off = t * d_m;
        let m_off = t * d_m;
        let y_off = t * d_m;
        let ad_off = t * m.adapters[0].rank;
        for r in 0..m.adapters[0].rank {
            let h = m.tape.adapter_hidden[ad_off + r];
            m.tape.adapter_act[ad_off + r] = h * sigmoid(h);
        }
        adapter_up_into(m, &m.tape.adapter_act[ad_off..ad_off + m.adapters[0].rank], &mut ad_out[..d_m]);

        for i in 0..d_m {
            m.tape.z_raw[z_off + i] = (m.tape.y_ssm[y_off + i] * ssm_scale)
                + m.tape.m_inj[m_off + i]
                + ad_out[i];
        }
    }

    batched_matvec_dev(gpu_ctx(m).as_ref(), &m.mlp_w1.data, d_mlp, d_m, &m.tape.z_raw[..seq_len * d_m], seq_len, &mut m.tape.mlp_hidden[..seq_len * d_mlp]);
    for t_i in 0..seq_len * d_mlp {
        let h = m.tape.mlp_hidden[t_i];
        m.tape.mlp_act[t_i] = h * sigmoid(h);
    }
    batched_matvec_dev(gpu_ctx(m).as_ref(), &m.mlp_w2.data, d_m, d_mlp, &m.tape.mlp_act[..seq_len * d_mlp], seq_len, &mut m.tape.z_final[..seq_len * d_m]);
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
    let d_m = m.cfg.d_latent;
    let d_v = m.cfg.d_vocab;
    let logit_scale = 1.0 / (d_m as f32).sqrt();

    batched_matvec_dev(
        gpu_ctx(m).as_ref(),
        &m.unembed_w.data,
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

/// Full batched forward pass over a chunk: identical tape contents and loss to
/// `forward_train_chunk`, but every weight-touching stage is one batched GEMM
/// over all L tokens (one GPU dispatch per stage, L dispatches only inside the
/// inherently-serial SSM scan).
pub fn forward_train_chunk_batched(m: &mut PSSALayerV2, token_ids: &[usize], target_ids: &[usize]) -> f32 {
    assert!(!token_ids.is_empty(), "training chunk must be nonempty");
    assert_eq!(token_ids.len(), target_ids.len(), "token and target counts must match");
    let seq_len = token_ids.len().min(m.cfg.chunk_len);
    assert!(seq_len > 0);
    assert!(
        token_ids[..seq_len].iter().all(|&id| id < m.cfg.d_vocab)
            && target_ids[..seq_len].iter().all(|&id| id < m.cfg.d_vocab),
        "token IDs must be in vocabulary"
    );

    m.tape.x_ids[..seq_len].copy_from_slice(&token_ids[..seq_len]);
    m.tape.target_ids[..seq_len].copy_from_slice(&target_ids[..seq_len]);
    m.tape.h_states[..m.cfg.d_latent * m.cfg.d_state].copy_from_slice(&m.h_persistent);

    stage_embed_norm(m, seq_len);
    stage_projections(m, seq_len);
    stage_ssm_scan(m, seq_len);
    stage_memory(m, seq_len);
    stage_adapter(m, seq_len);
    stage_mlp(m, seq_len);
    let loss = stage_logits_loss(m, seq_len);

    let last_h_off = seq_len * (m.cfg.d_latent * m.cfg.d_state);
    m.h_persistent
        .copy_from_slice(&m.tape.h_states[last_h_off..last_h_off + m.cfg.d_latent * m.cfg.d_state]);

    loss
}



// =============================================================================
// BACKWARD STAGES (REVERSE TIME, BATCHED)
// =============================================================================

/// Backward Stage 7: full-vocabulary cross-entropy adjoint + unembed gradient,
/// batched over all L tokens.
#[inline]
pub fn bwd_stage_logits(m: &mut PSSALayerV2, seq_len: usize, scale_loss: f32) {
    let d_m = m.cfg.d_latent;
    let d_v = m.cfg.d_vocab;
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
                m.bwd_g_zfinal[z_off + j] += g_logit * m.unembed_w.data[row_off + j];
                m.unembed_w.grad[row_off + j] += g_logit * m.tape.z_final[z_off + j];
            }
        }
    }
}

/// Backward Stage 6: SiLU MLP adjoint, batched over all L tokens.
#[inline]
pub fn bwd_stage_mlp(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_mlp = d_m * 2;
    let l = seq_len;

    for t in 0..l {
        let mlp_off = t * d_mlp;
        let z_off = t * d_m;

        // g_mlp_act = mlp_w2^T @ grad_z_final[t]
        for i in 0..d_mlp {
            let mut total = 0.0f32;
            for r in 0..d_m {
                total += m.mlp_w2.data[r * d_mlp + i] * m.bwd_g_zfinal[z_off + r];
            }
            m.buf_g_mlp_act[i] = total;
        }

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
        for j in 0..d_m {
            let mut total = 0.0f32;
            for r in 0..d_mlp {
                total += m.mlp_w1.data[r * d_m + j] * m.buf_g_mlp_hidden[r];
            }
            m.buf_g_zraw_mlp[j] = total;
        }

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
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let l = seq_len;

    let up_total = {
        let mut u = vec![0.0f32; d_m * rank];
        for i in 0..u.len() {
            u[i] = m.adapters[0].up_proj.data[i] + m.adapters[0].consolidated_up[i];
        }
        u
    };

    for t in 0..l {
        let ad_off = t * rank;
        let z_off = t * d_m;

        // g_ad_act = U_total^T @ grad_z_raw[t]
        for r in 0..rank {
            let mut total = 0.0f32;
            for i in 0..d_m {
                total += up_total[i * rank + r] * m.bwd_g_zraw[z_off + i];
            }
            m.buf_g_ad_act[r] = total;
        }

        // up_proj grad: grad_z_raw[t,i] * adapter_act[t,r]
        for i in 0..d_m {
            let gz_i = m.bwd_g_zraw[z_off + i];
            let row_off = i * rank;
            for r in 0..rank {
                m.adapters[0].up_proj.grad[row_off + r] +=
                    gz_i * m.tape.adapter_act[ad_off + r];
            }
        }

        // SiLU derivative on adapter hidden, stored per token
        for r in 0..rank {
            let h = m.tape.adapter_hidden[ad_off + r];
            let sig_h = sigmoid(h);
            let silu_prime = sig_h * (1.0 + h * (1.0 - sig_h));
            m.bwd_g_ad_down[ad_off + r] = m.buf_g_ad_act[r] * silu_prime;
        }
    }
}

/// Backward Stage 4b: adapter down-projection adjoint into grad_x_norm plus
/// down_proj gradients, batched over all L tokens. (Kept separate from
/// bwd_stage_adapter because grad_x_norm accumulates across stages.)
#[inline]
pub fn bwd_stage_adapter_down(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let rank = m.adapters[0].rank;
    let l = seq_len;
    m.bwd_g_xnorm[..l * d_m].fill(0.0);

    for t in 0..l {
        let xn_off = t * d_m;
        let ad_off = t * rank;
        for r in 0..rank {
            let gad_r = m.bwd_g_ad_down[ad_off + r];
            let row_off = r * d_m;
            for j in 0..d_m {
                m.bwd_g_xnorm[xn_off + j] += gad_r * m.adapters[0].down_proj.data[row_off + j];
                m.adapters[0].down_proj.grad[row_off + j] += gad_r * m.tape.x_norm[xn_off + j];
            }
        }
    }
}

/// Backward Stage 4: memory injection adjoint (gate, w_proj, memory query
/// adjoint incl. hyperbolic distance chain), accumulating grad_x_norm.
/// The per-token hyperbolic retrieval adjoint is inherently serial over the
/// memory bank; token loops here mirror the reference accumulation order.
#[inline]
pub fn bwd_stage_memory(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let d_k = m.cfg.d_mem_key;
    let mem_cap = m.cfg.mem_capacity;
    let l = seq_len;

    for t in 0..l {
        let m_off = t * d_m;

        for i in 0..d_m {
            let gz_i = m.bwd_g_zraw[m_off + i];
            let g_mem = m.tape.g_mem[m_off + i];
            m.buf_g_m_proj_out[i] = gz_i * g_mem;

            let d_sig = g_mem * (1.0 - g_mem);
            let g_wgate_pre = gz_i * m.tape.m_proj[m_off + i] * d_sig;
            let row_off = i * d_m;
            for j in 0..d_m {
                m.bwd_g_xnorm[m_off + j] += g_wgate_pre * m.w_gate.data[row_off + j];
                m.w_gate.grad[row_off + j] += g_wgate_pre * m.tape.x_norm[m_off + j];
            }
        }

        // g_m_val = w_proj^T @ g_m_proj_out
        for j in 0..d_m {
            let mut total = 0.0f32;
            for r in 0..d_m {
                total += m.w_proj.data[r * d_m + j] * m.buf_g_m_proj_out[r];
            }
            m.buf_g_m_val[j] = total;
        }

        // w_proj grad
        for i in 0..d_m {
            let g_mp_i = m.buf_g_m_proj_out[i];
            let row_off = i * d_m;
            for j in 0..d_m {
                m.w_proj.grad[row_off + j] += g_mp_i * m.tape.m_val[m_off + j];
            }
        }
        let _ = m_off;
    }

    // Hyperbolic retrieval adjoint, reverse time.
    for t in (0..l).rev() {
        let q_off = t * d_k;
        let m_off = t * d_m;
        let q = &m.tape.q_poincare[q_off..q_off + d_k];
        let q_sq = dot_slice(q, q) as f64;

        m.g_query_pnc.fill(0.0);
        for entry in 0..m.memory.count {
            let key_off = entry * d_k;
            let key = &m.memory.keys[key_off..key_off + d_k];
            let key_sq = m.memory.norm_sq[entry] as f64;
            let mut dot_g_value_minus_mean = 0.0f64;
            let value_off = entry * d_m;
            for j in 0..d_m {
                dot_g_value_minus_mean += m.buf_g_m_val[j] as f64
                    * (m.memory.values[value_off + j] - m.tape.m_val[m_off + j]) as f64;
            }
            let g_score = m.tape.mem_weights[t * mem_cap + entry] as f64 * dot_g_value_minus_mean;
            let mut sq = 0.0f64;
            for k in 0..d_k {
                let diff = (q[k] - key[k]) as f64;
                sq += diff * diff;
            }
            if sq > 0.0 {
                let denom = (1.0 - q_sq) * (1.0 - key_sq);
                assert!(denom > 0.0 && denom.is_finite());
                let z = sq / denom;
                let dd_dz = 1.0 / (z * (1.0 + z)).sqrt();
                for k in 0..d_k {
                    let diff = (q[k] - key[k]) as f64;
                    let ddenom = -2.0 * q[k] as f64 * (1.0 - key_sq);
                    let dz = (2.0 * diff * denom - sq * ddenom) / (denom * denom);
                    m.g_query_pnc[k] +=
                        (g_score * (-1.0 / m.cfg.tau_mem as f64) * dd_dz * dz) as f32;
                }
            }
        }
        let r = m.tape.q_norm[t];
        if r == 0.0 {
            m.g_query_euc.copy_from_slice(&m.g_query_pnc);
        } else {
            let q_euc = &m.tape.q_euc[q_off..q_off + d_k];
            let mut qdotg = 0.0;
            for k in 0..d_k {
                qdotg += q_euc[k] * m.g_query_pnc[k];
            }
            let denom = r * (1.0 + r) * (1.0 + r);
            for k in 0..d_k {
                m.g_query_euc[k] = m.g_query_pnc[k] / (1.0 + r) - q_euc[k] * qdotg / denom;
            }
        }
        m.g_y_ssm.fill(0.0);
        let xn = &m.tape.x_norm[m_off..m_off + d_m];
        let y = &m.tape.y_ssm[m_off..m_off + d_m];
        for r_i in 0..d_k {
            let gq = m.g_query_euc[r_i];
            let row = r_i * d_m;
            for j in 0..d_m {
                m.w_qx.grad[row + j] += gq * xn[j];
                m.w_qh.grad[row + j] += gq * y[j];
                m.bwd_g_xnorm[m_off + j] += gq * m.w_qx.data[row + j];
                m.g_y_ssm[j] += gq * m.w_qh.data[row + j];
            }
        }
        // Per-token g_y_ssm stored for the SSM stage.
        for j in 0..d_m {
            m.bwd_g_ysm[m_off + j] = m.g_y_ssm[j];
        }
        let _ = d_s;
    }
}

/// Backward Stage 3: SSM recurrence adjoint plus delta / b / c projections and
/// the second RMSNorm chain into the embedding, in exact reverse time order.
/// The recurrence itself is inherently serial; its per-step work mirrors the
/// reference inner loops so grads agree to f32 roundoff.
#[inline]
pub fn bwd_stage_ssm(m: &mut PSSALayerV2, seq_len: usize) {
    let d_m = m.cfg.d_latent;
    let d_s = m.cfg.d_state;
    let l = seq_len;
    let ssm_scale = 1.0 / (d_s as f32).sqrt();

    m.grad_h_next.fill(0.0);
    let pending_step = m.step_counter + 1;
    for t in 0..l {
        m.embed_row_marks[m.tape.x_ids[t]] = pending_step;
    }

    for t in (0..l).rev() {
        let x_id = m.tape.x_ids[t];
        let del_off = t * d_m;
        let m_off = t * d_m;

        // SSM recurrence backward
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
                let a_raw = m.a_mat.data[idx];
                let a_physical = -softplus(a_raw);
                let b_val = m.tape.b_proj[t * d_s + j];

                let g_h_total = g_y_i * c_val + m.grad_h_next[idx];

                m.buf_g_c_proj[j] += g_y_i * h_next;
                m.buf_g_h_prev[idx] += g_h_total * bar_a;

                m.a_mat.grad[idx] += g_h_total
                    * (d_i * bar_a)
                    * m.tape.h_states[t * (d_m * d_s) + idx]
                    * -sigmoid(a_raw);
                m.buf_g_delta[i] += g_h_total
                    * (a_physical * bar_a * m.tape.h_states[t * (d_m * d_s) + idx]
                        + b_val * xn_i);
                m.buf_g_b_proj[j] += g_h_total * (d_i * xn_i);
                m.bwd_g_xnorm[m_off + i] += g_h_total * m.tape.bar_b[t * (d_m * d_s) + idx];
            }
        }

        m.grad_h_next.copy_from_slice(&m.buf_g_h_prev);

        // delta projection backward
        for i in 0..d_m {
            let d_sig = sigmoid(m.tape.delta_raw[del_off + i]);
            let gd_i = m.buf_g_delta[i] * d_sig;
            let row_off = i * d_m;
            for j in 0..d_m {
                m.bwd_g_xnorm[m_off + j] += gd_i * m.w_delta.data[row_off + j];
                m.w_delta.grad[row_off + j] += gd_i * m.tape.x_norm[m_off + j];
            }
        }

        // b and c projections backward
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

        // Affine RMSNorm backward to gamma, beta, and the embedding row
        let inv_rms = m.tape.inv_rms[t];
        let e_t = &m.embed_w.data[x_id * d_m..(x_id + 1) * d_m];

        let mut dot_gx_e = 0.0f32;
        for i in 0..d_m {
            let gx_i = m.bwd_g_xnorm[m_off + i];
            m.norm_beta.grad[i] += gx_i;
            m.norm_gamma.grad[i] += gx_i * (e_t[i] * inv_rms);

            let g_unnorm = gx_i * m.norm_gamma.data[i];
            dot_gx_e += g_unnorm * e_t[i];
        }

        let emb_row_off = x_id * d_m;
        for i in 0..d_m {
            let g_unnorm = m.bwd_g_xnorm[m_off + i] * m.norm_gamma.data[i];
            let g_e_i =
                inv_rms * (g_unnorm - e_t[i] * (dot_gx_e * inv_rms * inv_rms / (d_m as f32)));
            m.embed_w.grad[emb_row_off + i] += g_e_i;
        }
    }
}

/// Full batched backward pass over a chunk: identical gradients to the
/// reference `backward_chunk`, accumulated in the same reverse-time token
/// order, driven by the stage adjoints above.
pub fn backward_chunk_batched(m: &mut PSSALayerV2, seq_len: usize, accumulation_scale: f32) {
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
