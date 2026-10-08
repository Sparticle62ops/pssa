//! CUDA safeguard arithmetic. Embedded PTX needs the driver, not libnvrtc.
//! Training may retain dense gradients and Adam moments between updates; host
//! weights remain current because recurrence/retrieval are still host-owned.
use super::stage_bounds::{cap_shape, optimizer_shape, product};
use super::*;
use crate::pssa::{AdamTensor, PSSAConfigV2};
use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};

struct Kernels {
    partials: CudaFunction,
    finish: CudaFunction,
    adam: CudaFunction,
    cap: CudaFunction,
}

struct TensorState {
    host_data: usize,
    data: CudaSlice<f32>,
    grad: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    // TN gradients are accumulated exclusively on-device after registration.
    dense: bool,
}

struct NormWorkspace {
    descriptors: CudaSlice<u64>,
    partials: CudaSlice<f64>,
    norm: CudaSlice<f64>,
    invalid: CudaSlice<u32>,
}

#[derive(Default)]
pub(super) struct Safeguards {
    kernels: Option<Kernels>,
    tensors: HashMap<usize, TensorState>,
    cap_workspace: Option<CudaSlice<f32>>,
    cap_invalid: Option<CudaSlice<u32>>,
    norm_workspace: Option<NormWorkspace>,
    resident: bool,
    fresh_gradients: bool,
    finite: bool,
}

impl Safeguards {
    pub(super) fn weight(&self, host: &[f32]) -> Option<&CudaSlice<f32>> {
        if !self.resident {
            return None;
        }
        self.tensors
            .values()
            .find(|t| t.host_data == host.as_ptr() as usize && t.data.len() == host.len())
            .map(|t| &t.data)
    }
}

fn validate_packed_gradients(
    state: &Safeguards,
    gradients: &[&[f32]],
    private: &[CudaSlice<f32>],
) -> Result<(), String> {
    if gradients.len() != private.len() {
        return Err("CUDA packed gradient count changed".into());
    }
    for (index, (host, buffer)) in gradients.iter().zip(private).enumerate() {
        let key = host.as_ptr() as usize;
        if gradients[..index]
            .iter()
            .any(|prior| prior.as_ptr() as usize == key)
        {
            return Err("CUDA packed gradients alias; refusing publication".into());
        }
        let device = state
            .tensors
            .get(&key)
            .ok_or("unregistered packed CUDA gradient")?;
        if device.grad.len() != host.len() || buffer.len() != host.len() {
            return Err("packed CUDA gradient length changed".into());
        }
    }
    Ok(())
}

fn error(e: impl std::fmt::Debug) -> String {
    format!("CUDA safeguard operation failed ({e:?})")
}

impl CudaContext {
    fn safeguard_kernels<'a>(&self, state: &'a mut Safeguards) -> Result<&'a Kernels, String> {
        if state.kernels.is_none() {
            let module = self
                .stream
                .context()
                .load_module(cudarc::nvrtc::Ptx::from_src(include_str!("safeguards.ptx")))
                .map_err(error)?;
            state.kernels = Some(Kernels {
                partials: module.load_function("norm_partials").map_err(error)?,
                finish: module.load_function("norm_finish").map_err(error)?,
                adam: module.load_function("scaled_adam").map_err(error)?,
                cap: module.load_function("cap_values").map_err(error)?,
            });
        }
        Ok(state.kernels.as_ref().unwrap())
    }

    fn upload_tensor(&self, tensor: &AdamTensor<'_>) -> Result<TensorState, String> {
        optimizer_shape(
            tensor.data.len(),
            tensor.grad.len(),
            tensor.m.len(),
            tensor.v.len(),
        )?;
        Ok(TensorState {
            host_data: tensor.data.as_ptr() as usize,
            data: self.stream.clone_htod(tensor.data).map_err(error)?,
            grad: self.stream.clone_htod(tensor.grad).map_err(error)?,
            m: self.stream.clone_htod(tensor.m).map_err(error)?,
            v: self.stream.clone_htod(tensor.v).map_err(error)?,
            dense: false,
        })
    }

    /// Scoped to CLI training. Initialize from checkpoint moments, then retain
    /// them until finish_safeguarded_training restores the public host state.
    pub(crate) fn begin_safeguarded_training(
        &self,
        tensors: &[AdamTensor<'_>],
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        self.safeguard_kernels(&mut state)?;
        state.tensors.clear();
        for t in tensors {
            state
                .tensors
                .insert(t.grad.as_ptr() as usize, self.upload_tensor(t)?);
        }
        self.stream.synchronize().map_err(error)?;
        state.resident = true;
        state.fresh_gradients = true;
        state.finite = true;
        Ok(())
    }

    /// Publish device-resident Adam weights to their host mirrors before the
    /// host-only dream phase. The optimizer remains resident; its adapter is
    /// refreshed again after dream consolidation.
    pub(crate) fn sync_safeguarded_weights(
        &self,
        tensors: &mut [AdamTensor<'_>],
    ) -> Result<(), String> {
        let state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Ok(());
        }
        for t in tensors {
            let device = state
                .tensors
                .get(&(t.grad.as_ptr() as usize))
                .ok_or("CUDA optimizer tensor registration changed")?;
            self.stream
                .memcpy_dtoh(&device.data, t.data)
                .map_err(error)?;
        }
        self.stream.synchronize().map_err(error)
    }

    /// Publish host-only dream changes back into the resident optimizer before
    /// the next GPU forward. The ordinary training path keeps device weights
    /// authoritative; replay temporarily moves ownership to the host.
    pub(crate) fn refresh_safeguarded_weights(
        &self,
        tensors: &mut [AdamTensor<'_>],
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Ok(());
        }
        for t in tensors {
            let device = state
                .tensors
                .get_mut(&(t.grad.as_ptr() as usize))
                .ok_or("CUDA optimizer tensor registration changed")?;
            self.stream
                .memcpy_htod(t.data, &mut device.data)
                .map_err(error)?;
        }
        self.stream.synchronize().map_err(error)
    }

    pub(crate) fn finish_safeguarded_training(
        &self,
        tensors: &mut [AdamTensor<'_>],
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        for t in tensors {
            let device = state
                .tensors
                .get(&(t.grad.as_ptr() as usize))
                .ok_or("CUDA optimizer tensor registration changed")?;
            self.stream
                .memcpy_dtoh(&device.grad, t.grad)
                .map_err(error)?;
            self.stream.memcpy_dtoh(&device.m, t.m).map_err(error)?;
            self.stream.memcpy_dtoh(&device.v, t.v).map_err(error)?;
        }
        self.stream.synchronize().map_err(error)?;
        state.resident = false;
        state.tensors.clear();
        Ok(())
    }

    pub(crate) fn safeguarded_parameters_finite(&self) -> Option<bool> {
        let state = self.safeguards.lock().unwrap_or_else(|e| e.into_inner());
        state.resident.then_some(state.finite)
    }

    pub(crate) fn zero_safeguarded_gradients(
        &self,
        tensors: &mut [AdamTensor<'_>],
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        let fresh = state.fresh_gradients;
        for (index, t) in tensors.iter_mut().enumerate() {
            let device = state
                .tensors
                .get_mut(&(t.grad.as_ptr() as usize))
                .ok_or("CUDA optimizer tensor registration changed")?;
            self.stream.memset_zeros(&mut device.grad).map_err(error)?;
            // Embedding rows were cleared sparsely by the model. Dense TN
            // gradients have no host consumers inside this training scope.
            if (index == 0 && fresh) || (index != 0 && !device.dense) {
                t.grad.fill(0.0);
            }
        }
        state.fresh_gradients = false;
        Ok(())
    }

    /// Epoch EMA transfers part of the fast adapter into its slow host copy.
    /// Keep the resident fast weights in sync without resetting Adam moments.
    pub(crate) fn refresh_safeguarded_weight(
        &self,
        p: &crate::pssa::ParamMatrix,
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        if state.resident {
            let device = state
                .tensors
                .get_mut(&(p.grad.as_ptr() as usize))
                .ok_or("CUDA optimizer tensor registration changed")?;
            self.stream
                .memcpy_htod(&p.data, &mut device.data)
                .map_err(error)?;
            self.stream.synchronize().map_err(error)?;
        }
        Ok(())
    }

    /// Packed stages publish these gradients transactionally to their host
    /// mirrors. Recover any prior resident-TN contribution before changing
    /// ownership; clipping must subsequently upload, not ignore, the mirror.
    pub(super) fn host_owned_gradients(&self, gradients: &mut [&mut [f32]]) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Ok(());
        }
        let mut copied = false;
        for host in gradients.iter_mut() {
            let device = state
                .tensors
                .get(&(host.as_ptr() as usize))
                .ok_or("unregistered packed CUDA gradient")?;
            if device.grad.len() != host.len() {
                return Err("packed CUDA gradient length changed".into());
            }
            if device.dense {
                self.stream
                    .memcpy_dtoh(&device.grad, *host)
                    .map_err(error)?;
                copied = true;
            }
        }
        if copied {
            self.stream.synchronize().map_err(error)?;
        }
        for host in gradients {
            state
                .tensors
                .get_mut(&(host.as_ptr() as usize))
                .unwrap()
                .dense = false;
        }
        Ok(())
    }

    /// Seed private packed adjoints without modifying the optimizer's current
    /// gradients. Their allocations can be swapped only after the entire CUDA
    /// transaction succeeds, so a failed kernel never requires GPU rollback.
    pub(super) fn seed_packed_gradients(
        &self,
        gradients: &[&[f32]],
        private: &mut [CudaSlice<f32>],
    ) -> Result<bool, String> {
        let state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Ok(false);
        }
        validate_packed_gradients(&state, gradients, private)?;
        for (host, buffer) in gradients.iter().zip(private) {
            let device = &state.tensors[&(host.as_ptr() as usize)];
            if device.dense {
                self.stream
                    .memcpy_dtod(&device.grad, buffer)
                    .map_err(error)?;
            } else {
                self.stream.memcpy_htod(*host, buffer).map_err(error)?;
            }
        }
        Ok(true)
    }

    /// No driver operation occurs during publication. Validate ALL entries
    /// before swapping any allocation; the old optimizer buffers become the
    /// next packed transaction's scratch, without allocating or copying.
    pub(super) fn publish_packed_gradients(
        &self,
        gradients: &[&[f32]],
        private: &mut [CudaSlice<f32>],
    ) -> Result<(), String> {
        let mut state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Err("CUDA training scope ended during packed backward".into());
        }
        validate_packed_gradients(&state, gradients, private)?;
        for (host, buffer) in gradients.iter().zip(private) {
            let device = state.tensors.get_mut(&(host.as_ptr() as usize)).unwrap();
            std::mem::swap(&mut device.grad, buffer);
            device.dense = true;
        }
        Ok(())
    }

    // False preserves the historical host-slice GEMM API outside scoped training.
    pub(super) fn resident_tn(
        &self,
        a: &[f32],
        b: &[f32],
        cfg: GemmConfig<f32>,
        out: &mut [f32],
    ) -> Result<bool, String> {
        let mut workspace = self.workspace.lock().map_err(error)?;
        let mut state = self.safeguards.lock().map_err(error)?;
        if !state.resident {
            return Ok(false);
        }
        let device = state
            .tensors
            .get_mut(&(out.as_ptr() as usize))
            .ok_or("unregistered CUDA training gradient")?;
        if device.grad.len() != out.len() {
            return Err("CUDA gradient length changed".into());
        }
        reserve_device(&mut workspace.lhs, &self.stream, a.len())?;
        reserve_device(&mut workspace.rhs, &self.stream, b.len())?;
        let Workspace { lhs, rhs, .. } = &mut *workspace;
        let mut a_dev = lhs.as_mut().unwrap().slice_mut(..a.len());
        self.stream.memcpy_htod(a, &mut a_dev).map_err(error)?;
        let mut b_dev = rhs.as_mut().unwrap().slice_mut(..b.len());
        self.stream.memcpy_htod(b, &mut b_dev).map_err(error)?;
        // A packed CPU replay may have accumulated into the host mirror after
        // handing ownership back. Do not resurrect a stale device contribution.
        if !device.dense {
            self.stream
                .memcpy_htod(out, &mut device.grad)
                .map_err(error)?;
        }
        // SAFETY: caller performed exact cuBLAS shape validation; device grad
        // stays alive under the optimizer guard and all operations use one stream.
        unsafe { self.blas.gemm(cfg, &b_dev, &a_dev, &mut device.grad) }.map_err(error)?;
        // Borrowed host inputs may be mutated by the next CPU stage. Ensure their
        // asynchronous uploads finish before releasing those borrows.
        self.stream.synchronize().map_err(error)?;
        device.dense = true;
        Ok(true)
    }

    pub(crate) fn clip_adamw(
        &self,
        tensors: &mut [AdamTensor<'_>],
        cfg: &PSSAConfigV2,
        lr: f32,
        step: Option<usize>,
        max_norm: f32,
        embedding_rows: &[usize],
    ) -> Result<f64, String> {
        let count = u32::try_from(tensors.len())
            .map_err(|_| "CUDA optimizer tensor count exceeds u32; refusing launch")?;
        for t in tensors.iter() {
            optimizer_shape(t.data.len(), t.grad.len(), t.m.len(), t.v.len())?;
        }
        for &row in embedding_rows {
            let rows = row
                .checked_add(1)
                .ok_or("CUDA embedding row overflow; refusing launch")?;
            let end = product(rows, cfg.d_latent, "embedding row end")?;
            if cfg.d_latent == 0 || tensors.first().is_none_or(|t| end > t.grad.len()) {
                return Err(format!(
                    "CUDA embedding row {row} is out of bounds; refusing launch"
                ));
            }
        }
        let mut state = self.safeguards.lock().map_err(error)?;
        self.safeguard_kernels(&mut state)?;
        if !state.resident {
            state.tensors.clear();
            for t in tensors.iter() {
                state
                    .tensors
                    .insert(t.grad.as_ptr() as usize, self.upload_tensor(t)?);
            }
        } else {
            for (index, t) in tensors.iter().enumerate() {
                let device = state
                    .tensors
                    .get_mut(&(t.grad.as_ptr() as usize))
                    .ok_or("CUDA optimizer tensor registration changed")?;
                if device.grad.len() != t.grad.len() {
                    return Err("CUDA registered gradient length changed; refusing launch".into());
                }
                optimizer_shape(
                    device.data.len(),
                    device.grad.len(),
                    device.m.len(),
                    device.v.len(),
                )?;
                if index == 0 {
                    // Embedding scatter is host-owned, but only used rows have
                    // gradients. Device zeroing already cleared all other rows.
                    for &row in embedding_rows {
                        let start = row * cfg.d_latent;
                        let end = start + cfg.d_latent;
                        self.stream
                            .memcpy_htod(
                                &t.grad[start..end],
                                &mut device.grad.slice_mut(start..end),
                            )
                            .map_err(error)?;
                    }
                } else if !device.dense {
                    // Norm/SSM-rate (and looped-head) gradients are still CPU
                    // computed; upload them without a host norm/scale walk.
                    self.stream
                        .memcpy_htod(t.grad, &mut device.grad)
                        .map_err(error)?;
                }
            }
        }
        let mut descriptors = Vec::with_capacity(tensors.len() * 2);
        for t in tensors.iter() {
            let device = &state.tensors[&(t.grad.as_ptr() as usize)];
            let (ptr, record) = device.grad.device_ptr(&self.stream);
            descriptors.extend([ptr, t.grad.len() as u64]);
            drop(record);
        }
        if state
            .norm_workspace
            .as_ref()
            .is_none_or(|w| w.descriptors.len() < descriptors.len())
        {
            state.norm_workspace = Some(NormWorkspace {
                descriptors: self
                    .stream
                    .alloc_zeros::<u64>(descriptors.len())
                    .map_err(error)?,
                partials: self.stream.alloc_zeros::<f64>(256).map_err(error)?,
                norm: self.stream.alloc_zeros::<f64>(1).map_err(error)?,
                invalid: self.stream.alloc_zeros::<u32>(1).map_err(error)?,
            });
        }
        let resident = state.resident;
        let Safeguards {
            kernels,
            tensors: device_tensors,
            norm_workspace,
            ..
        } = &mut *state;
        let NormWorkspace {
            descriptors: descriptors_dev,
            partials,
            norm: norm_dev,
            invalid,
        } = norm_workspace.as_mut().unwrap();
        self.stream
            .memcpy_htod(&descriptors, descriptors_dev)
            .map_err(error)?;
        let kernels = kernels.as_ref().unwrap();
        // SAFETY: descriptors reference exactly the registered, live gradient
        // allocations. Fixed 256-thread blocks match PTX shared storage; the
        // second stage reads exactly 256 partials. No unordered atomic FP sums.
        unsafe {
            self.stream
                .launch_builder(&kernels.partials)
                .arg(&*descriptors_dev)
                .arg(&count)
                .arg(&mut *partials)
                .launch(LaunchConfig {
                    grid_dim: (256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(error)?;
            self.stream
                .launch_builder(&kernels.finish)
                .arg(&*partials)
                .arg(&mut *norm_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(error)?;
        }
        let mut norm = [0.0f64];
        self.stream
            .memcpy_dtoh(&*norm_dev, &mut norm)
            .map_err(error)?;
        self.stream.synchronize().map_err(error)?;
        if !norm[0].is_finite() {
            return Ok(norm[0]);
        }
        let scale = if norm[0] > max_norm as f64 {
            max_norm as f64 / norm[0]
        } else {
            1.0
        };
        let step = step.ok_or("optimizer step counter overflow")?;
        let bias1 = 1.0 - cfg.beta1.powf(step as f32);
        let bias2 = 1.0 - cfg.beta2.powf(step as f32);
        // Norm kernels assign every partial/result. Only the atomic invalid
        // flag must be explicitly reset when these buffers are reused.
        self.stream.memset_zeros(invalid).map_err(error)?;
        for t in tensors.iter_mut() {
            let device = device_tensors.get_mut(&(t.grad.as_ptr() as usize)).unwrap();
            let len = t.grad.len() as u64;
            let launch_len = optimizer_shape(
                device.data.len(),
                device.grad.len(),
                device.m.len(),
                device.v.len(),
            )?;
            // SAFETY: all four allocations have len elements, kernel bounds
            // checks every access; f64 scale rounds to f32 before Adam moments.
            unsafe {
                self.stream
                    .launch_builder(&kernels.adam)
                    .arg(&mut device.data)
                    .arg(&mut device.grad)
                    .arg(&mut device.m)
                    .arg(&mut device.v)
                    .arg(&len)
                    .arg(&lr)
                    .arg(&cfg.beta1)
                    .arg(&cfg.beta2)
                    .arg(&t.weight_decay)
                    .arg(&cfg.eps)
                    .arg(&bias1)
                    .arg(&bias2)
                    .arg(&scale)
                    .arg(&mut *invalid)
                    .launch(LaunchConfig::for_num_elems(launch_len))
                    .map_err(error)?;
            }
            // Host recurrence still consumes weights. Moments/dense gradients
            // are downloaded only at handoff, not after every CLI update.
            self.stream
                .memcpy_dtoh(&device.data, t.data)
                .map_err(error)?;
            if !resident {
                self.stream
                    .memcpy_dtoh(&device.grad, t.grad)
                    .map_err(error)?;
                self.stream.memcpy_dtoh(&device.m, t.m).map_err(error)?;
                self.stream.memcpy_dtoh(&device.v, t.v).map_err(error)?;
            }
        }
        let mut invalid_host = [0u32];
        self.stream
            .memcpy_dtoh(&*invalid, &mut invalid_host)
            .map_err(error)?;
        self.stream.synchronize().map_err(error)?;
        state.finite = invalid_host[0] == 0;
        if !resident {
            state.tensors.clear();
        }
        Ok(norm[0])
    }

    pub(crate) fn cap_memory_values(
        &self,
        values: &mut [f32],
        width: usize,
        cap: f32,
    ) -> Result<(), String> {
        let (rows, width_arg) = cap_shape(values.len(), width, cap)?;
        if values.is_empty() {
            return Ok(());
        }
        let mut state = self.safeguards.lock().map_err(error)?;
        self.safeguard_kernels(&mut state)?;
        let Safeguards {
            kernels,
            cap_workspace,
            cap_invalid,
            ..
        } = &mut *state;
        reserve_device(cap_workspace, &self.stream, values.len())?;
        let mut values_dev = cap_workspace.as_mut().unwrap().slice_mut(..values.len());
        self.stream
            .memcpy_htod(values, &mut values_dev)
            .map_err(error)?;
        let kernels = kernels.as_ref().unwrap();
        if cap_invalid.is_none() {
            *cap_invalid = Some(self.stream.alloc_zeros::<u32>(1).map_err(error)?);
        }
        let invalid = cap_invalid.as_mut().unwrap();
        self.stream.memset_zeros(invalid).map_err(error)?;
        let width = width_arg;
        let cap = cap as f64;
        // SAFETY: validated whole rows, positive width/cap, live allocations;
        // each thread owns one row including the one-ULP correction loop.
        unsafe {
            self.stream
                .launch_builder(&kernels.cap)
                .arg(&mut values_dev)
                .arg(&rows)
                .arg(&width)
                .arg(&cap)
                .arg(&mut *invalid)
                .launch(LaunchConfig::for_num_elems(rows))
                .map_err(error)?;
        }
        let mut bad = [0u32];
        self.stream
            .memcpy_dtoh(&*invalid, &mut bad)
            .map_err(error)?;
        self.stream.synchronize().map_err(error)?;
        if bad[0] != 0 {
            return Err("capped memory value must be finite".into());
        }
        self.stream
            .memcpy_dtoh(&values_dev, values)
            .map_err(error)?;
        self.stream.synchronize().map_err(error)
    }
}
