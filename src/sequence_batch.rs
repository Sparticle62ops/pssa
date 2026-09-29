//! Independent sequence lanes with packed (not padded) dense token rows.
//!
//! Parameters and episodic memory are shared. Memory must stay unchanged between
//! forward and backward; insertions belong after backward. Each lane owns its SSM
//! carry/tape and reverse-time scratch, so scans run in parallel without atomics
//! or cross-sequence adjoints. Dense stages use the same CPU/GPU dispatch as L=1
//! chunk training, but see the sum of the active sequence lengths as GEMM rows.
//!
//! Stacked models use a CPU-only serial replay fallback: each lane retains every
//! layer's carry and its chunk inputs, then recomputes its tape for backward.
//! This preserves independent lanes and token weighting, not packed-GEMM speed.
//! Parameters and memory must remain unchanged between forward and backward.
use crate::{
    gpu_batch as stages,
    linalg::sigmoid,
    pssa::{ChunkActivationTape, PSSALayerV2},
};
use rayon::prelude::*;

pub struct Sequence<'a> {
    /// Stable lane index; omitted lanes keep their carry and do no work.
    pub lane: usize,
    pub inputs: &'a [usize],
    pub targets: &'a [usize],
    /// Reset only this lane, e.g. at a document boundary.
    pub reset: bool,
}

struct ReplayLane {
    initial_carry: Vec<f32>,
    inputs: Vec<usize>,
    targets: Vec<usize>,
    terminal_keys: Vec<f32>,
    terminal_values: Vec<f32>,
}

struct Lane {
    replay: Option<ReplayLane>,
    offset: usize,
    len: usize,
    carry: Vec<f32>,
    h: Vec<f32>,
    bar_a: Vec<f32>,
    bar_b: Vec<f32>,
    scan_a: Vec<f32>,
    scan_b: Vec<f32>,
    y: Vec<f32>,
    ga: Vec<f32>,
    ga_tokens: Vec<f32>,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    gx: Vec<f32>,
}

/// Runtime-only workspace. Does not change model configuration or checkpoint
/// bytes. A forward must be followed by backward before another forward.
pub struct SequenceBatch {
    lanes: Vec<Lane>,
    shape: [usize; 8],
    // The fallback restores model-owned carries; lane carries are runtime-only.
    model_carry: Vec<f32>,
    losses: Vec<f32>,
    active_lanes: Vec<usize>,
    tokens: usize,
    pending: bool,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    gx: Vec<f32>,
}

fn shape(m: &PSSALayerV2) -> [usize; 8] {
    [
        m.cfg.chunk_len,
        m.cfg.d_latent,
        m.cfg.d_state,
        m.cfg.d_vocab,
        m.cfg.d_mem_key,
        m.cfg.mem_capacity,
        m.adapters[0].rank,
        m.depth(),
    ]
}

impl SequenceBatch {
    pub fn new(m: &mut PSSALayerV2, batch_size: usize) -> Result<Self, String> {
        if batch_size == 0 {
            return Err(
                "batch size must be positive; use --batch-size 1 for serial training".into(),
            );
        }
        let [l, d, s, v, k, mem, rank, depth] = shape(m);
        if depth > 1 && m.device.is_gpu() {
            return Err("stacked sequence batching is CPU-only; use Device::Cpu".into());
        }
        let rows = l
            .checked_mul(batch_size)
            .ok_or("batch tape size overflow; reduce --batch-size")?;
        let scan_len = l.next_power_of_two();
        let state_width = d * s;
        // Conservative bound includes both the packed model tape and lane-local
        // recurrent tapes/scratch. Reuse the checked 1 GiB model allocation guard.
        let mut budget = m.cfg.clone();
        budget.chunk_len = l
            .checked_add(1)
            .and_then(|n| n.checked_mul(batch_size))
            .and_then(|n| n.checked_mul(2))
            .ok_or("batch allocation overflow; reduce --batch-size")?;
        crate::checkpoint::validate_model_config(&budget)
            .map_err(|e| format!("batch workspace: {e}; reduce --batch-size or --chunk"))?;
        m.block.tape = ChunkActivationTape::new(rows, v, d, s, k, mem, rank);
        // Packed terminal rows let the trainer defer per-layer memory writes
        // until ALL replayed backwards complete, using the usual insertion API.
        for block in &mut m.extra_blocks {
            block.tape = ChunkActivationTape::new(rows, 0, d, s, k, mem, rank);
        }
        m.bwd_g_zfinal.resize(rows * d, 0.0);
        m.bwd_g_zraw.resize(rows * d, 0.0);
        m.bwd_g_ad_down.resize(rows * rank, 0.0);
        m.bwd_g_xnorm.resize(rows * d, 0.0);
        m.bwd_g_ysm.resize(rows * d, 0.0);
        m.bwd_g_logits.resize(rows * v, 0.0);
        m.bwd_g_mlp.resize(rows * 2 * d, 0.0);
        let lanes = (0..batch_size)
            .map(|_| Lane {
                replay: (depth > 1).then(|| ReplayLane {
                    initial_carry: vec![0.0; depth * d * s],
                    inputs: Vec::with_capacity(l),
                    targets: Vec::with_capacity(l),
                    terminal_keys: vec![0.0; depth * k],
                    terminal_values: vec![0.0; depth * d],
                }),
                offset: 0,
                len: 0,
                carry: vec![0.0; depth * d * s],
                h: vec![0.0; (l + 1) * d * s],
                bar_a: vec![0.0; l * d * s],
                bar_b: vec![0.0; l * d * s],
                scan_a: vec![0.0; scan_len * state_width],
                scan_b: vec![0.0; scan_len * state_width],
                y: vec![0.0; l * d],
                ga: vec![0.0; d * s],
                ga_tokens: vec![0.0; l * state_width],
                gd: vec![0.0; l * d],
                gb: vec![0.0; l * s],
                gc: vec![0.0; l * s],
                gx: vec![0.0; l * d],
            })
            .collect();
        Ok(Self {
            lanes,
            shape: shape(m),
            model_carry: vec![0.0; depth * d * s],
            losses: vec![0.0; rows],
            active_lanes: Vec::with_capacity(batch_size),
            tokens: 0,
            pending: false,
            gd: vec![0.0; rows * d],
            gb: vec![0.0; rows * s],
            gc: vec![0.0; rows * s],
            gx: vec![0.0; rows * d],
        })
    }

    /// Flattened recurrent carries in layer order (depth * latent * state).
    pub fn state(&self, lane: usize) -> &[f32] {
        &self.lanes[lane].carry
    }
    pub fn state_mut(&mut self, lane: usize) -> &mut [f32] {
        &mut self.lanes[lane].carry
    }
    pub fn reset_states(&mut self) {
        for lane in &mut self.lanes {
            lane.carry.fill(0.0);
        }
    }

    fn check_model(&self, m: &PSSALayerV2) -> Result<(), String> {
        if m.depth() > 1 && m.device.is_gpu() {
            return Err("stacked sequence batching is CPU-only; use Device::Cpu".into());
        }
        let rows = self.shape[0] * self.lanes.len();
        if shape(m) != self.shape || m.block.tape.max_l < rows
            || m.extra_blocks.iter().any(|block| block.tape.max_l < rows)
        {
            return Err(
                "sequence batch workspace does not match model; create a new workspace".into(),
            );
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        m: &mut PSSALayerV2,
        sequences: &[Sequence<'_>],
    ) -> Result<f32, String> {
        self.check_model(m)?;
        if self.pending {
            return Err("finish batch backward before the next forward".into());
        }
        if sequences.is_empty() || sequences.len() > self.lanes.len() {
            return Err("sequence count must be between 1 and the configured batch size".into());
        }
        // Validate the whole batch before mutating any carry/tape.
        for (i, seq) in sequences.iter().enumerate() {
            if seq.lane >= self.lanes.len()
                || sequences[..i].iter().any(|other| other.lane == seq.lane)
            {
                return Err(
                    "each sequence needs a distinct lane within the configured batch size".into(),
                );
            }
            if seq.inputs.is_empty()
                || seq.inputs.len() > m.cfg.chunk_len
                || seq.inputs.len() != seq.targets.len()
            {
                return Err("each sequence needs matching nonempty input/target slices no longer than --chunk".into());
            }
            if seq
                .inputs
                .iter()
                .chain(seq.targets)
                .any(|&id| id >= m.cfg.d_vocab)
            {
                return Err("batch token ID is outside the model vocabulary".into());
            }
        }
        if m.depth() > 1 {
            return Ok(self.forward_stacked(m, sequences));
        }
        self.tokens = 0;
        for lane in &mut self.lanes {
            lane.len = 0;
        }
        for seq in sequences {
            let lane = &mut self.lanes[seq.lane];
            lane.offset = self.tokens;
            lane.len = seq.inputs.len();
            if seq.reset {
                lane.carry.fill(0.0);
            }
            let end = self.tokens + lane.len;
            m.tape.x_ids[self.tokens..end].copy_from_slice(seq.inputs);
            m.tape.target_ids[self.tokens..end].copy_from_slice(seq.targets);
            self.tokens = end;
        }
        stages::stage_embed_norm(m, self.tokens);
        stages::stage_projections(m, self.tokens);
        m.refresh_ssm_rates();
        self.lanes
            .par_iter_mut()
            .filter(|lane| lane.len > 0)
            .for_each(|lane| lane.forward(m));
        for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
            let start = lane.offset * m.cfg.d_latent;
            m.block.tape.y_ssm[start..start + lane.len * m.cfg.d_latent]
                .copy_from_slice(&lane.y[..lane.len * m.cfg.d_latent]);
        }
        stages::stage_memory(m, self.tokens);
        stages::stage_adapter(m, self.tokens);
        stages::stage_mlp(m, self.tokens);
        let loss = stages::stage_logits_loss(m, self.tokens);
        self.pending = true;
        Ok(loss)
    }

    /// Accumulate the token-mean gradient, multiplied by accumulation_scale.
    /// For a larger optimizer group pass batch_tokens / group_tokens. Carry is
    /// retained for the next forward, but TBPTT never differentiates across calls.
    pub fn backward(&mut self, m: &mut PSSALayerV2, accumulation_scale: f32) -> Result<(), String> {
        self.check_model(m)?;
        if !self.pending || !accumulation_scale.is_finite() {
            return Err(
                "batch backward needs a pending forward and a finite accumulation scale".into(),
            );
        }
        if m.depth() > 1 {
            self.backward_stacked(m, accumulation_scale);
            return Ok(());
        }
        let n = self.tokens;
        stages::bwd_stage_logits(m, n, accumulation_scale / n as f32);
        stages::bwd_stage_mlp(m, n);
        stages::bwd_stage_adapter(m, n);
        stages::bwd_stage_adapter_down(m, n);
        stages::bwd_stage_memory(m, n);
        m.refresh_ssm_rates();
        self.lanes
            .par_iter_mut()
            .filter(|lane| lane.len > 0)
            .for_each(|lane| lane.backward(m));
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
            let start = lane.offset * d;
            let end = start + lane.len * d;
            self.gd[start..end].copy_from_slice(&lane.gd[..lane.len * d]);
            self.gb[lane.offset * s..(lane.offset + lane.len) * s]
                .copy_from_slice(&lane.gb[..lane.len * s]);
            self.gc[lane.offset * s..(lane.offset + lane.len) * s]
                .copy_from_slice(&lane.gc[..lane.len * s]);
            for (dst, src) in m.bwd_g_xnorm[start..end].iter_mut().zip(&lane.gx) {
                *dst += src;
            }
            for (dst, src) in m.a_mat.grad.iter_mut().zip(&lane.ga) {
                *dst += src;
            }
        }
        // Projection adjoints share weights across ALL sequences, like the head.
        for (g, w, rows) in [
            (&self.gd, &mut m.block.w_delta, d),
            (&self.gb, &mut m.block.w_b, s),
            (&self.gc, &mut m.block.w_c, s),
        ] {
            stages::dense_input_adjoint(g, &w.data, n, rows, d, &mut self.gx[..n * d]);
            stages::dense_weight_adjoint(g, &m.block.tape.x_norm, n, rows, d, &mut w.grad);
            for (dst, src) in m.block.bwd_g_xnorm[..n * d].iter_mut().zip(&self.gx) {
                *dst += src;
            }
        }
        for t in (0..n).rev() {
            let id = m.tape.x_ids[t];
            m.embed_row_marks[id] = m.step_counter + 1;
            let inv = m.tape.inv_rms[t];
            let e = &m.embed_w.data[id * d..(id + 1) * d];
            let gx = &m.block.bwd_g_xnorm[t * d..(t + 1) * d];
            let mut dot = 0.0;
            for i in 0..d {
                m.block.norm_beta.grad[i] += gx[i];
                m.block.norm_gamma.grad[i] += gx[i] * (e[i] * inv);
                dot += gx[i] * m.block.norm_gamma.data[i] * e[i];
            }
            for i in 0..d {
                m.embed_w.grad[id * d + i] +=
                    inv * (gx[i] * m.block.norm_gamma.data[i] - e[i] * (dot * inv * inv / d as f32));
            }
        }
        self.pending = false;
        Ok(())
    }

    fn forward_stacked(&mut self, m: &mut PSSALayerV2, sequences: &[Sequence<'_>]) -> f32 {
        copy_carry_from_model(m, &mut self.model_carry);
        self.tokens = 0;
        self.active_lanes.clear();
        for lane in &mut self.lanes {
            lane.len = 0;
        }
        let d = m.cfg.d_latent;
        let k = m.cfg.d_mem_key;
        for seq in sequences {
            self.active_lanes.push(seq.lane);
            let lane = &mut self.lanes[seq.lane];
            lane.offset = self.tokens;
            lane.len = seq.inputs.len();
            if seq.reset {
                lane.carry.fill(0.0);
            }
            let replay = lane.replay.as_mut().expect("stacked lane workspace");
            replay.initial_carry.copy_from_slice(&lane.carry);
            replay.inputs.clear();
            replay.inputs.extend_from_slice(seq.inputs);
            replay.targets.clear();
            replay.targets.extend_from_slice(seq.targets);
            copy_carry_to_model(&lane.carry, m);
            m.forward_train_chunk(seq.inputs, seq.targets);
            copy_carry_from_model(m, &mut lane.carry);
            self.losses[self.tokens..self.tokens + lane.len]
                .copy_from_slice(&m.block.tape.losses[..lane.len]);
            let last = lane.len - 1;
            for (i, block) in std::iter::once(&m.block).chain(&m.extra_blocks).enumerate() {
                replay.terminal_keys[i * k..(i + 1) * k]
                    .copy_from_slice(&block.tape.q_poincare[last * k..(last + 1) * k]);
                replay.terminal_values[i * d..(i + 1) * d]
                    .copy_from_slice(&block.tape.z_final[last * d..(last + 1) * d]);
            }
            self.tokens += lane.len;
        }
        copy_carry_to_model(&self.model_carry, m);
        self.publish_stacked_terminals(m);
        self.pending = true;
        self.losses[..self.tokens].iter().sum::<f32>() / self.tokens as f32
    }

    fn backward_stacked(&mut self, m: &mut PSSALayerV2, accumulation_scale: f32) {
        copy_carry_from_model(m, &mut self.model_carry);
        for &index in &self.active_lanes {
            let lane = &self.lanes[index];
            let replay = lane.replay.as_ref().expect("stacked lane workspace");
            copy_carry_to_model(&replay.initial_carry, m);
            // Memory and parameters have not changed since forward. Replaying
            // avoids keeping a full activation tape for every lane AND layer.
            m.forward_train_chunk(&replay.inputs, &replay.targets);
            m.backward_chunk(lane.len, accumulation_scale * lane.len as f32 / self.tokens as f32);
        }
        copy_carry_to_model(&self.model_carry, m);
        self.publish_stacked_terminals(m);
        self.pending = false;
    }

    fn publish_stacked_terminals(&self, m: &mut PSSALayerV2) {
        let d = m.cfg.d_latent;
        let k = m.cfg.d_mem_key;
        m.block.tape.losses[..self.tokens].copy_from_slice(&self.losses[..self.tokens]);
        for &index in &self.active_lanes {
            let lane = &self.lanes[index];
            let replay = lane.replay.as_ref().expect("stacked lane workspace");
            let last = lane.offset + lane.len - 1;
            for (i, block) in std::iter::once(&mut m.block).chain(&mut m.extra_blocks).enumerate() {
                block.tape.q_poincare[last * k..(last + 1) * k]
                    .copy_from_slice(&replay.terminal_keys[i * k..(i + 1) * k]);
                block.tape.z_final[last * d..(last + 1) * d]
                    .copy_from_slice(&replay.terminal_values[i * d..(i + 1) * d]);
            }
        }
    }
}

fn copy_carry_from_model(m: &PSSALayerV2, out: &mut [f32]) {
    let hs = m.cfg.d_latent * m.cfg.d_state;
    for (block, dst) in std::iter::once(&m.block).chain(&m.extra_blocks).zip(out.chunks_exact_mut(hs)) {
        dst.copy_from_slice(&block.h_persistent);
    }
}

fn copy_carry_to_model(carry: &[f32], m: &mut PSSALayerV2) {
    let hs = m.cfg.d_latent * m.cfg.d_state;
    for (block, src) in std::iter::once(&mut m.block).chain(&mut m.extra_blocks).zip(carry.chunks_exact(hs)) {
        block.h_persistent.copy_from_slice(src);
    }
}

impl Lane {
    fn forward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        let offset = self.offset;
        self.h[..hs].copy_from_slice(&self.carry);

        // Keep the local maps for backward while scanning (bar_a, bar_b*x).
        // Every worker owns a complete token row in each output buffer.
        self.bar_a[..l * hs]
            .par_chunks_mut(hs)
            .zip(self.bar_b[..l * hs].par_chunks_mut(hs))
            .zip(
                self.scan_a[..l * hs]
                    .par_chunks_mut(hs)
                    .zip(self.scan_b[..l * hs].par_chunks_mut(hs)),
            )
            .enumerate()
            .for_each(|(t, ((a_row, b_row), (scan_a_row, scan_b_row)))| {
                let row = offset + t;
                for i in 0..d {
                    let delta = m.tape.delta[row * d + i];
                    let x = m.tape.x_norm[row * d + i];
                    for j in 0..s {
                        let idx = i * s + j;
                        let a = (delta * m.ssm_rates[idx]).exp();
                        let b = delta * m.tape.b_proj[row * s + j];
                        a_row[idx] = a;
                        b_row[idx] = b;
                        scan_a_row[idx] = a;
                        scan_b_row[idx] = b * x;
                    }
                }
            });
        stages::affine_scan_in_place(&mut self.scan_a, &mut self.scan_b, l, hs);
        let (initial, states_out) = self.h.split_at_mut(hs);
        stages::materialize_ssm_scan(
            initial,
            &self.bar_a[..l * hs],
            &self.bar_b[..l * hs],
            &m.tape.x_norm[offset * d..(offset + l) * d],
            &self.scan_a[..l * hs],
            &self.scan_b[..l * hs],
            &m.tape.c_proj[offset * s..(offset + l) * s],
            &mut states_out[..l * hs],
            &mut self.y[..l * d],
            l,
            d,
            s,
        );
        self.carry
            .copy_from_slice(&self.h[l * hs..(l + 1) * hs]);
    }

    fn backward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        let offset = self.offset;
        let scale = 1.0 / (s as f32).sqrt();
        let local_a = &self.bar_a[..l * hs];

        // In reverse time p_t = A_t * (r_t + p_(t+1)), r_t = gy_t*C_t.
        // The exclusive prefix of (A_t, A_t*r_t) yields the future adjoint
        // p_(t+1), with a zero terminal adjoint at this lane's TBPTT boundary.
        self.scan_a[..l * hs]
            .par_chunks_mut(hs)
            .zip(self.scan_b[..l * hs].par_chunks_mut(hs))
            .enumerate()
            .for_each(|(u, (a_row, b_row))| {
                let t = l - 1 - u;
                let row = offset + t;
                for i in 0..d {
                    let gy = m.bwd_g_zraw[row * d + i] * scale + m.bwd_g_ysm[row * d + i];
                    for j in 0..s {
                        let idx = i * s + j;
                        let a = local_a[t * hs + idx];
                        let r = gy * m.tape.c_proj[row * s + j];
                        a_row[idx] = a;
                        b_row[idx] = a * r;
                    }
                }
            });
        stages::affine_scan_in_place(&mut self.scan_a, &mut self.scan_b, l, hs);

        let future = &self.scan_b[..l * hs];
        let local_b = &self.bar_b[..l * hs];
        let states = &self.h[..(l + 1) * hs];
        self.gd[..l * d]
            .par_chunks_mut(d)
            .zip(self.gb[..l * s].par_chunks_mut(s))
            .zip(self.gc[..l * s].par_chunks_mut(s))
            .zip(self.gx[..l * d].par_chunks_mut(d))
            .zip(self.ga_tokens[..l * hs].par_chunks_mut(hs))
            .enumerate()
            .for_each(|(t, ((((gd, gb), gc), gx), ga))| {
                let row = offset + t;
                let future_p = &future[(l - 1 - t) * hs..(l - t) * hs];
                gd.fill(0.0);
                gb.fill(0.0);
                gc.fill(0.0);
                gx.fill(0.0);
                for i in 0..d {
                    let gy = m.bwd_g_zraw[row * d + i] * scale + m.bwd_g_ysm[row * d + i];
                    let delta = m.tape.delta[row * d + i];
                    let x = m.tape.x_norm[row * d + i];
                    for j in 0..s {
                        let idx = i * s + j;
                        let a = local_a[t * hs + idx];
                        let b = m.tape.b_proj[row * s + j];
                        let prev = states[t * hs + idx];
                        let gh = gy * m.tape.c_proj[row * s + j] + future_p[idx];
                        gc[j] += gy * states[(t + 1) * hs + idx];
                        ga[idx] = gh * (delta * a) * prev * m.ssm_rate_derivatives[idx];
                        gd[i] += gh * (m.ssm_rates[idx] * a * prev + b * x);
                        gb[j] += gh * (delta * x);
                        gx[i] += gh * local_b[t * hs + idx];
                    }
                    gd[i] *= sigmoid(m.tape.delta_raw[row * d + i]);
                }
            });

        // Shared rate gradients retain a deterministic reverse-time reduction;
        // no token worker writes the lane aggregate or another lane's buffers.
        self.ga.fill(0.0);
        for ga in self.ga_tokens[..l * hs].chunks_exact(hs).rev() {
            for (dst, src) in self.ga.iter_mut().zip(ga) {
                *dst += src;
            }
        }
    }
}
