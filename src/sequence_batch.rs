//! Independent sequence lanes with packed (not padded) dense token rows.
//!
//! Parameters and episodic memory are shared. Memory must stay unchanged between
//! forward and backward; insertions belong after backward. Each lane owns its SSM
//! carry/tape and reverse-time scratch, so scans run in parallel without atomics
//! or cross-sequence adjoints. Dense stages use the same CPU/GPU dispatch as L=1
//! chunk training, but see the sum of the active sequence lengths as GEMM rows.
//!
//! Stacked and looped models use a serial replay fallback for their recurrent
//! lane scheduling: each lane retains every layer/pass carry and its chunk
//! inputs, then recomputes its tape for backward. Dense stages in each replay
//! still use the model's GPU dispatch when one is available.
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
    gh: Vec<f32>,
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
    #[cfg(feature = "cuda")]
    cuda: Option<crate::cuda::packed::PackedWorkspace>,
    cuda_pending: bool,
    last_cuda_step: bool,
    cuda_carry_dirty: bool,
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
        if m.local_mixing_enabled() {
            return Err("local mixing experiments require single-lane training".into());
        }
        if batch_size == 0 {
            return Err(
                "batch size must be positive; use --batch-size 1 for serial training".into(),
            );
        }
        let [l, d, s, v, k, mem, rank, depth] = shape(m);
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
        let loops = m.loops();
        m.block.tape = ChunkActivationTape::new_with_loops(rows, v, d, s, k, mem, rank, loops);
        // Packed terminal rows let the trainer defer per-layer memory writes
        // until ALL replayed backwards complete, using the usual insertion API.
        for block in &mut m.extra_blocks {
            block.tape = ChunkActivationTape::new_with_loops(rows, 0, d, s, k, mem, rank, loops);
        }
        m.bwd_g_zfinal.resize(rows * d, 0.0);
        m.bwd_g_zraw.resize(rows * d, 0.0);
        m.bwd_g_ad_down.resize(rows * rank, 0.0);
        m.bwd_g_xnorm.resize(rows * d, 0.0);
        m.bwd_g_ysm.resize(rows * d, 0.0);
        m.bwd_g_query_euc.resize(rows * k, 0.0);
        m.bwd_g_query_pnc.resize(rows * k, 0.0);
        m.bwd_g_logits.resize(rows * v, 0.0);
        m.bwd_g_mlp.resize(rows * 2 * d, 0.0);
        let lanes = (0..batch_size)
            .map(|_| Lane {
                replay: (depth > 1 || loops > 1).then(|| ReplayLane {
                    initial_carry: vec![0.0; depth * loops * d * s],
                    inputs: Vec::with_capacity(l),
                    targets: Vec::with_capacity(l),
                    terminal_keys: vec![0.0; depth * k],
                    terminal_values: vec![0.0; depth * d],
                }),
                offset: 0,
                len: 0,
                carry: vec![0.0; depth * loops * d * s],
                h: vec![0.0; (l + 1) * d * s],
                bar_a: vec![0.0; l * d * s],
                bar_b: vec![0.0; l * d * s],
                scan_a: vec![0.0; scan_len * state_width],
                scan_b: vec![0.0; scan_len * state_width],
                y: vec![0.0; l * d],
                gh: vec![0.0; d * s],
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
            model_carry: vec![0.0; depth * loops * d * s],
            losses: vec![0.0; rows],
            active_lanes: Vec::with_capacity(batch_size),
            tokens: 0,
            pending: false,
            gd: vec![0.0; rows * d],
            gb: vec![0.0; rows * s],
            gc: vec![0.0; rows * s],
            gx: vec![0.0; rows * d],
            #[cfg(feature = "cuda")]
            cuda: None,
            cuda_pending: false,
            last_cuda_step: false,
            cuda_carry_dirty: true,
        })
    }

    /// Diagnostic: the most recent packed forward/backward used the resident
    /// CUDA stages without replaying their CPU fallback. False for CPU/replay.
    pub fn last_step_used_cuda(&self) -> bool {
        self.last_cuda_step
    }

    /// Flattened recurrent carries in layer order (depth * latent * state).
    pub fn state(&self, lane: usize) -> &[f32] {
        &self.lanes[lane].carry
    }
    pub fn state_mut(&mut self, lane: usize) -> &mut [f32] {
        self.cuda_carry_dirty = true;
        &mut self.lanes[lane].carry
    }
    pub fn reset_states(&mut self) {
        self.cuda_carry_dirty = true;
        for lane in &mut self.lanes {
            lane.carry.fill(0.0);
        }
    }

    fn check_model(&self, m: &PSSALayerV2) -> Result<(), String> {
        if m.local_mixing_enabled() {
            return Err("local mixing experiments require single-lane training".into());
        }
        let rows = self.shape[0] * self.lanes.len();
        if shape(m) != self.shape
            || m.block.tape.max_l < rows
            || m.extra_blocks.iter().any(|block| block.tape.max_l < rows)
        {
            return Err(
                "sequence batch workspace does not match model; create a new workspace".into(),
            );
        }
        Ok(())
    }

    /// A lane recurrence is already serial within each document. Only wake
    /// Rayon when there are multiple active lanes and enough aggregate state
    /// work to keep the workers busy; short batches stay entirely synchronous.
    fn parallel_lanes(&self, m: &PSSALayerV2) -> bool {
        let active = self.lanes.iter().filter(|lane| lane.len > 0).count();
        active > 1
            && self
                .tokens
                .saturating_mul(m.cfg.d_latent)
                .saturating_mul(m.cfg.d_state)
                >= 1_048_576
            && rayon::current_num_threads() > 1
    }

    pub fn forward(
        &mut self,
        m: &mut PSSALayerV2,
        sequences: &[Sequence<'_>],
    ) -> Result<f32, String> {
        let _trace = crate::training_diagnostics::StageTrace::new(
            &m.device,
            "batch.forward",
            sequences.iter().map(|s| s.inputs.len()).sum(),
        );
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
        self.last_cuda_step = false;
        if m.depth() > 1 || m.loops() > 1 {
            let loss = self.forward_stacked(m, sequences);
            if !loss.is_finite() {
                self.pending = false;
                return Err(crate::training_diagnostics::loss_error(m, self.tokens));
            }
            return Ok(loss);
        }
        self.cuda_pending = false;
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
                self.cuda_carry_dirty = true;
            }
            let end = self.tokens + lane.len;
            m.tape.x_ids[self.tokens..end].copy_from_slice(seq.inputs);
            m.tape.target_ids[self.tokens..end].copy_from_slice(seq.targets);
            self.tokens = end;
        }
        stages::stage_embed_norm(m, self.tokens);
        // CUDA has a packed resident implementation. WebGPU does not yet
        // share one queue submission across independent lane carries, so run
        // the same device SSM dispatcher once per active lane and keep the
        // packed memory stage on device. A failed WebGPU dispatch is loud and
        // falls back only for that lane.
        let wgpu = match m.device.gpu() {
            Some(crate::backend::GpuDispatch::Wgpu(ctx)) => {
                Some(crate::backend::GpuDispatch::Wgpu(ctx))
            }
            _ => None,
        };
        if let Some(gpu) = wgpu.as_ref() {
            self.cuda_pending = false;
            self.cuda_carry_dirty = true;
            m.refresh_ssm_rates();
            stages::stage_projections(m, self.tokens);
            for lane in self.lanes.iter_mut().filter(|lane| lane.len > 0) {
                if let Err(error) = lane.forward_wgpu(m, gpu) {
                    eprintln!(
                        "warning: WebGPU packed SSM forward failed; using CPU lane scan: {error}"
                    );
                    lane.forward(m);
                }
            }
            for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
                let start = lane.offset * m.cfg.d_latent;
                m.block.tape.y_ssm[start..start + lane.len * m.cfg.d_latent]
                    .copy_from_slice(&lane.y[..lane.len * m.cfg.d_latent]);
            }
            stages::stage_memory_packed(m, self.tokens);
        } else {
            m.refresh_ssm_rates();
            let mut cuda_failed = false;
            self.cuda_pending = match self.forward_cuda(m) {
                Ok(resident) => resident,
                Err(error) => {
                    eprintln!("warning: CUDA packed forward failed; replaying CPU stages: {error}");
                    cuda_failed = true;
                    false
                }
            };
            if !self.cuda_pending {
                self.cuda_carry_dirty = true;
                // A failed CUDA transaction publishes no lane carries. Explicitly
                // use CPU stages for recovery rather than retry a damaged stream.
                let device = cuda_failed
                    .then(|| std::mem::replace(&mut m.device, crate::backend::Device::Cpu));
                stages::stage_projections(m, self.tokens);
                if self.parallel_lanes(m) {
                    m.scan_executor.run(|| {
                        self.lanes
                            .par_iter_mut()
                            .filter(|lane| lane.len > 0)
                            .for_each(|lane| lane.forward(m));
                    });
                } else {
                    self.lanes
                        .iter_mut()
                        .filter(|lane| lane.len > 0)
                        .for_each(|lane| lane.forward(m));
                }
                for lane in self.lanes.iter().filter(|lane| lane.len > 0) {
                    let start = lane.offset * m.cfg.d_latent;
                    m.block.tape.y_ssm[start..start + lane.len * m.cfg.d_latent]
                        .copy_from_slice(&lane.y[..lane.len * m.cfg.d_latent]);
                }
                stages::stage_memory_packed(m, self.tokens);
                if let Some(device) = device {
                    m.device = device;
                }
            }
        }
        stages::stage_adapter(m, self.tokens);
        stages::stage_mlp(m, self.tokens);
        let loss = stages::stage_logits_loss(m, self.tokens);
        if !loss.is_finite() {
            return Err(crate::training_diagnostics::loss_error(m, self.tokens));
        }
        self.pending = true;
        Ok(loss)
    }

    /// Accumulate the token-mean gradient, multiplied by accumulation_scale.
    /// For a larger optimizer group pass batch_tokens / group_tokens. Carry is
    /// retained for the next forward, but TBPTT never differentiates across calls.
    pub fn backward(&mut self, m: &mut PSSALayerV2, accumulation_scale: f32) -> Result<(), String> {
        let _trace =
            crate::training_diagnostics::StageTrace::new(&m.device, "batch.backward", self.tokens);
        self.check_model(m)?;
        if !self.pending || !accumulation_scale.is_finite() {
            return Err(
                "batch backward needs a pending forward and a finite accumulation scale".into(),
            );
        }
        if m.depth() > 1 || m.loops() > 1 {
            self.backward_stacked(m, accumulation_scale);
            return Ok(());
        }
        let n = self.tokens;
        stages::bwd_stage_logits(m, n, accumulation_scale / n as f32);
        stages::bwd_stage_mlp(m, n);
        stages::bwd_stage_adapter(m, n);
        stages::bwd_stage_adapter_down(m, n);
        m.refresh_ssm_rates();
        let mut recovery_device = None;
        let resident_backward = if self.cuda_pending {
            match self.backward_cuda(m) {
                Ok(()) => true,
                Err(error) => {
                    eprintln!(
                        "warning: CUDA packed backward failed; replaying CPU tapes/adjoints: {error}"
                    );
                    self.cuda_carry_dirty = true;
                    // Resident optimizer gradients still hold only PRIOR
                    // successful transactions. Recover them before CPU replay;
                    // inability to recover is an error, never a wrong gradient.
                    #[cfg(feature = "cuda")]
                    if let Some(workspace) = self.cuda.as_ref() {
                        workspace.recover_host_gradients(m).map_err(|failure| {
                            workspace.finish_failed_transaction(format!(
                                "cannot recover prior CUDA gradients for CPU replay: {failure}"
                            ))
                        })?;
                    }
                    recovery_device = Some(std::mem::replace(
                        &mut m.device,
                        crate::backend::Device::Cpu,
                    ));
                    self.rebuild_host_tapes(m);
                    stages::bwd_stage_memory(m, n);
                    false
                }
            }
        } else {
            stages::bwd_stage_memory(m, n);
            false
        };
        self.cuda_pending = false;
        if !resident_backward {
            let wgpu = match m.device.gpu() {
                Some(crate::backend::GpuDispatch::Wgpu(ctx)) => {
                    Some(crate::backend::GpuDispatch::Wgpu(ctx))
                }
                _ => None,
            };
            if let Some(gpu) = wgpu.as_ref() {
                for lane in self.lanes.iter_mut().filter(|lane| lane.len > 0) {
                    if let Err(error) = lane.backward_wgpu(m, gpu) {
                        eprintln!(
                            "warning: WebGPU packed SSM backward failed; using CPU lane scan: {error}"
                        );
                        lane.backward(m);
                    }
                }
            } else if self.parallel_lanes(m) {
                m.scan_executor.run(|| {
                    self.lanes
                        .par_iter_mut()
                        .filter(|lane| lane.len > 0)
                        .for_each(|lane| lane.backward(m));
                });
            } else {
                self.lanes
                    .iter_mut()
                    .filter(|lane| lane.len > 0)
                    .for_each(|lane| lane.backward(m));
            }
        }
        let _trace = crate::training_diagnostics::StageTrace::new(
            &m.device,
            "batch.backward.projections_norm",
            n,
        );
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        if !resident_backward {
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
            // Projection adjoints share weights across ALL sequences, like the
            // head. CUDA uses the same packed GEMMs as single-lane backward;
            // WebGPU keeps the deterministic CPU twin until it has transposed
            // backward kernels.
            let gpu = m.device.gpu().filter(|g| g.accelerates_backward());
            for (g, w, rows) in [
                (&self.gd[..n * d], &mut m.block.w_delta, d),
                (&self.gb[..n * s], &mut m.block.w_b, s),
                (&self.gc[..n * s], &mut m.block.w_c, s),
            ] {
                if let Some(gpu) = gpu.as_ref() {
                    gpu.gemm_nn_into(g, &w.data, n, rows, d, &mut self.gx[..n * d])
                        .expect("validated model GEMM dimensions");
                    gpu.gemm_tn_accumulate_into(
                        g,
                        &m.block.tape.x_norm[..n * d],
                        n,
                        rows,
                        d,
                        &mut w.grad,
                    )
                    .expect("validated model GEMM dimensions");
                } else {
                    stages::dense_input_adjoint(g, &w.data, n, rows, d, &mut self.gx[..n * d]);
                    stages::dense_weight_adjoint(
                        g,
                        &m.block.tape.x_norm[..n * d],
                        n,
                        rows,
                        d,
                        &mut w.grad,
                    );
                }
                for (dst, src) in m.block.bwd_g_xnorm[..n * d].iter_mut().zip(&self.gx) {
                    *dst += src;
                }
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
                m.embed_w.grad[id * d + i] += inv
                    * (gx[i] * m.block.norm_gamma.data[i] - e[i] * (dot * inv * inv / d as f32));
            }
        }
        if let Some(device) = recovery_device {
            m.device = device;
        }
        self.last_cuda_step = resident_backward;
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
            let loops = m.loops();
            for (i, block) in std::iter::once(&m.block).chain(&m.extra_blocks).enumerate() {
                let loop_l = if loops == 1 {
                    0
                } else {
                    (loops - 1) * block.tape.max_l
                };
                replay.terminal_keys[i * k..(i + 1) * k].copy_from_slice(
                    &block.tape.q_poincare[(loop_l + last) * k..(loop_l + last + 1) * k],
                );
                replay.terminal_values[i * d..(i + 1) * d].copy_from_slice(
                    &block.tape.z_final[(loop_l + last) * d..(loop_l + last + 1) * d],
                );
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
            m.backward_chunk(
                lane.len,
                accumulation_scale * lane.len as f32 / self.tokens as f32,
            );
        }
        copy_carry_to_model(&self.model_carry, m);
        self.publish_stacked_terminals(m);
        self.pending = false;
    }

    fn publish_stacked_terminals(&self, m: &mut PSSALayerV2) {
        let d = m.cfg.d_latent;
        let k = m.cfg.d_mem_key;
        let loops = m.loops();
        m.block.tape.losses[..self.tokens].copy_from_slice(&self.losses[..self.tokens]);
        for &index in &self.active_lanes {
            let lane = &self.lanes[index];
            let replay = lane.replay.as_ref().expect("stacked lane workspace");
            let last = lane.offset + lane.len - 1;
            for (i, block) in std::iter::once(&mut m.block)
                .chain(&mut m.extra_blocks)
                .enumerate()
            {
                let loop_l = if loops == 1 {
                    0
                } else {
                    (loops - 1) * block.tape.max_l
                };
                block.tape.q_poincare[(loop_l + last) * k..(loop_l + last + 1) * k]
                    .copy_from_slice(&replay.terminal_keys[i * k..(i + 1) * k]);
                block.tape.z_final[(loop_l + last) * d..(loop_l + last + 1) * d]
                    .copy_from_slice(&replay.terminal_values[i * d..(i + 1) * d]);
            }
        }
    }
}

fn copy_carry_from_model(m: &PSSALayerV2, out: &mut [f32]) {
    m.copy_recurrent_state_to(out);
}

fn copy_carry_to_model(carry: &[f32], m: &mut PSSALayerV2) {
    m.copy_recurrent_state_from(carry);
}

impl SequenceBatch {
    fn forward_cuda(&mut self, m: &mut PSSALayerV2) -> Result<bool, String> {
        #[cfg(feature = "cuda")]
        if let Some(crate::backend::GpuDispatch::Cuda(ctx)) = m.device.gpu() {
            if self
                .cuda
                .as_ref()
                .is_none_or(|workspace| !workspace.matches(&ctx))
            {
                self.cuda = Some(crate::cuda::packed::PackedWorkspace::new(
                    &ctx,
                    m,
                    self.lanes.len(),
                )?);
                self.cuda_carry_dirty = true;
            }
            let workspace = self
                .cuda
                .as_mut()
                .expect("initialized packed CUDA workspace");
            let hs = m.cfg.d_latent * m.cfg.d_state;
            for (index, lane) in self.lanes.iter_mut().enumerate() {
                // CPU replay uses only this small initial row; all normal CUDA
                // forward/backward tapes stay resident in the workspace.
                lane.h[..hs].copy_from_slice(&lane.carry);
                workspace.offsets[index] = lane.offset as u32;
                workspace.lengths[index] = lane.len as u32;
                if self.cuda_carry_dirty {
                    workspace.carries[index * hs..(index + 1) * hs].copy_from_slice(&lane.carry);
                }
            }
            if let Err(error) = workspace.forward(m, self.tokens, self.cuda_carry_dirty) {
                return Err(workspace.finish_failed_transaction(error));
            }
            for (index, lane) in self.lanes.iter_mut().enumerate() {
                lane.carry
                    .copy_from_slice(&workspace.carries[index * hs..(index + 1) * hs]);
            }
            self.cuda_carry_dirty = false;
            return Ok(true);
        }
        let _ = m;
        Ok(false)
    }

    fn backward_cuda(&mut self, m: &mut PSSALayerV2) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        if let Some(workspace) = self.cuda.as_mut() {
            return workspace
                .backward(m, self.tokens)
                .map_err(|error| workspace.finish_failed_transaction(error));
        }
        let _ = m;
        Err("CUDA packed forward workspace is unavailable".into())
    }

    fn rebuild_host_tapes(&mut self, m: &mut PSSALayerV2) {
        stages::stage_projections(m, self.tokens);
        let hs = m.cfg.d_latent * m.cfg.d_state;
        for lane in self.lanes.iter_mut().filter(|lane| lane.len > 0) {
            lane.gh.copy_from_slice(&lane.carry);
            lane.carry.copy_from_slice(&lane.h[..hs]);
            lane.forward(m);
            lane.carry.copy_from_slice(&lane.gh);
            let start = lane.offset * m.cfg.d_latent;
            m.block.tape.y_ssm[start..start + lane.len * m.cfg.d_latent]
                .copy_from_slice(&lane.y[..lane.len * m.cfg.d_latent]);
        }
        stages::stage_memory_packed(m, self.tokens);
    }
}

impl Lane {
    fn forward_wgpu(
        &mut self,
        m: &PSSALayerV2,
        gpu: &crate::backend::GpuDispatch,
    ) -> Result<(), String> {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        let row_d = self.offset * d;
        let row_s = self.offset * s;
        let block = &m.block;
        gpu.ssm_forward(
            &block.tape.delta[row_d..row_d + l * d],
            &block.tape.delta_raw[row_d..row_d + l * d],
            &block.tape.b_proj[row_s..row_s + l * s],
            &block.tape.x_norm[row_d..row_d + l * d],
            &block.ssm_rates,
            &block.ssm_rate_derivatives,
            &block.tape.c_proj[row_s..row_s + l * s],
            &self.carry,
            l,
            d,
            s,
            &mut self.bar_a[..l * hs],
            &mut self.bar_b[..l * hs],
            &mut self.h[..(l + 1) * hs],
            &mut self.y[..l * d],
        )?;
        self.carry.copy_from_slice(&self.h[l * hs..(l + 1) * hs]);
        Ok(())
    }

    fn backward_wgpu(
        &mut self,
        m: &PSSALayerV2,
        gpu: &crate::backend::GpuDispatch,
    ) -> Result<(), String> {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        let row_d = self.offset * d;
        let row_s = self.offset * s;
        let block = &m.block;
        gpu.ssm_backward(
            &block.tape.delta[row_d..row_d + l * d],
            &block.tape.delta_raw[row_d..row_d + l * d],
            &block.tape.b_proj[row_s..row_s + l * s],
            &block.tape.c_proj[row_s..row_s + l * s],
            &block.ssm_rates,
            &block.ssm_rate_derivatives,
            &block.tape.x_norm[row_d..row_d + l * d],
            &self.h[..(l + 1) * hs],
            &self.bar_a[..l * hs],
            &self.bar_b[..l * hs],
            &block.bwd_g_zraw[row_d..row_d + l * d],
            &block.bwd_g_ysm[row_d..row_d + l * d],
            l,
            d,
            s,
            1.0 / (s as f32).sqrt(),
            &mut self.gd[..l * d],
            &mut self.gb[..l * s],
            &mut self.gc[..l * s],
            &mut self.ga_tokens[..l * hs],
            &mut self.gx[..l * d],
        )?;
        self.ga.fill(0.0);
        for token in self.ga_tokens[..l * hs].chunks_exact(hs).rev() {
            for (dst, src) in self.ga.iter_mut().zip(token) {
                *dst += src;
            }
        }
        Ok(())
    }

    /// Ordered lane recurrence for short sequences. The packed dense stages
    /// still run as usual; only the scan itself stays serial when its tree
    /// would cost more than the recurrence it replaces.
    fn forward_sequential(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        self.h[..hs].copy_from_slice(&self.carry);
        for t in 0..self.len {
            let row = self.offset + t;
            for i in 0..d {
                let delta = m.tape.delta[row * d + i];
                let mut y = 0.0;
                for j in 0..s {
                    let idx = i * s + j;
                    let a = (delta * m.ssm_rates[idx]).exp();
                    self.bar_a[t * hs + idx] = a;
                    let h = a * self.h[t * hs + idx]
                        + (delta * m.tape.b_proj[row * s + j]) * m.tape.x_norm[row * d + i];
                    self.h[(t + 1) * hs + idx] = h;
                    y += h * m.tape.c_proj[row * s + j];
                }
                self.y[t * d + i] = y;
            }
        }
        self.carry
            .copy_from_slice(&self.h[self.len * hs..(self.len + 1) * hs]);
    }

    fn forward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        if !stages::parallel_scan_enabled(l, hs) {
            self.forward_sequential(m);
            return;
        }
        m.scan_executor.run(|| self.forward_parallel(m));
    }

    fn forward_parallel(&mut self, m: &PSSALayerV2) {
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
        self.carry.copy_from_slice(&self.h[l * hs..(l + 1) * hs]);
    }

    fn backward_sequential(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let scale = 1.0 / (s as f32).sqrt();
        self.gh.fill(0.0);
        self.ga.fill(0.0);
        self.gd[..self.len * d].fill(0.0);
        self.gb[..self.len * s].fill(0.0);
        self.gc[..self.len * s].fill(0.0);
        self.gx[..self.len * d].fill(0.0);
        for t in (0..self.len).rev() {
            let row = self.offset + t;
            for i in 0..d {
                let gy = m.bwd_g_zraw[row * d + i] * scale + m.bwd_g_ysm[row * d + i];
                let delta = m.tape.delta[row * d + i];
                let x = m.tape.x_norm[row * d + i];
                for j in 0..s {
                    let idx = i * s + j;
                    let a = self.bar_a[t * hs + idx];
                    let b = m.tape.b_proj[row * s + j];
                    let prev = self.h[t * hs + idx];
                    let gh = gy * m.tape.c_proj[row * s + j] + self.gh[idx];
                    self.gc[t * s + j] += gy * self.h[(t + 1) * hs + idx];
                    self.gh[idx] = gh * a;
                    self.ga[idx] += gh * (delta * a) * prev * m.ssm_rate_derivatives[idx];
                    self.gd[t * d + i] += gh * (m.ssm_rates[idx] * a * prev + b * x);
                    self.gb[t * s + j] += gh * (delta * x);
                    self.gx[t * d + i] += gh * (delta * b);
                }
                self.gd[t * d + i] *= sigmoid(m.tape.delta_raw[row * d + i]);
            }
        }
    }

    fn backward(&mut self, m: &PSSALayerV2) {
        let d = m.cfg.d_latent;
        let s = m.cfg.d_state;
        let hs = d * s;
        let l = self.len;
        if !stages::parallel_scan_enabled(l, hs) {
            self.backward_sequential(m);
            return;
        }
        m.scan_executor.run(|| self.backward_parallel(m));
    }

    fn backward_parallel(&mut self, m: &PSSALayerV2) {
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

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::pssa::PSSAConfigV2;

    #[test]
    fn failed_resident_backward_replays_once_and_preserves_carry_edits() {
        let cfg = PSSAConfigV2 {
            d_vocab: 11,
            d_latent: 7,
            d_state: 3,
            d_mem_key: 4,
            mem_capacity: 4,
            chunk_len: 6,
            ..Default::default()
        };
        let mut reference = PSSALayerV2::new(cfg.clone(), 7);
        let mut recovered = PSSALayerV2::new(cfg, 7);
        for model in [&mut reference, &mut recovered] {
            model.memory.insert(&[0.1, -0.1, 0.05, 0.0], &[0.03; 7]);
            model.w_delta.grad.fill(0.125); // Accumulation must not be cleared.
        }
        let mut a = SequenceBatch::new(&mut reference, 3).unwrap();
        let mut b = SequenceBatch::new(&mut recovered, 3).unwrap();
        for batch in [&mut a, &mut b] {
            batch.state_mut(0).fill(0.04);
            batch.state_mut(2).fill(-0.02);
        }
        let sequences = [
            Sequence {
                lane: 2,
                inputs: &[1, 4, 3],
                targets: &[4, 3, 2],
                reset: false,
            },
            Sequence {
                lane: 0,
                inputs: &[5, 6],
                targets: &[6, 7],
                reset: false,
            },
        ];
        assert_eq!(
            a.forward(&mut reference, &sequences).unwrap(),
            b.forward(&mut recovered, &sequences).unwrap()
        );
        // Model the failure boundary without a CUDA device: the private CUDA
        // workspace is absent, so backward emits a warning and replays tapes.
        b.cuda_pending = true;
        a.state_mut(2).fill(0.3);
        b.state_mut(2).fill(0.3);
        a.backward(&mut reference, 0.7).unwrap();
        b.backward(&mut recovered, 0.7).unwrap();
        for lane in 0..3 {
            assert_eq!(a.state(lane), b.state(lane));
        }
        macro_rules! equal { ($($field:ident),*) => { $(assert_eq!(reference.$field.grad, recovered.$field.grad, stringify!($field));)* }; }
        equal!(
            embed_w, norm_gamma, norm_beta, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj,
            mlp_w1, mlp_w2, unembed_w
        );
        assert_eq!(
            reference.adapters[0].down_proj.grad,
            recovered.adapters[0].down_proj.grad
        );
        assert_eq!(
            reference.adapters[0].up_proj.grad,
            recovered.adapters[0].up_proj.grad
        );
        assert!(!b.last_step_used_cuda());
        // A following sparse forward must start from the preserved carry, not
        // advance it a second time or resurrect a stale resident carry.
        assert_eq!(
            a.forward(&mut reference, &sequences).unwrap(),
            b.forward(&mut recovered, &sequences).unwrap()
        );
        a.backward(&mut reference, 1.0).unwrap();
        b.backward(&mut recovered, 1.0).unwrap();
        equal!(
            embed_w, a_mat, w_delta, w_b, w_c, w_qx, w_qh, w_gate, w_proj
        );
    }
}
