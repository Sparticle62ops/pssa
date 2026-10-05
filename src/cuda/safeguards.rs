//! CUDA safeguard arithmetic. Embedded PTX needs the driver, not libnvrtc.
//! Training may retain dense gradients and Adam moments between updates; host
//! weights remain current because recurrence/retrieval are still host-owned.
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

#[derive(Default)]
pub(super) struct Safeguards {
    kernels: Option<Kernels>,
    tensors: HashMap<usize, TensorState>,
    cap_workspace: Option<CudaSlice<f32>>,
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
        let descriptors_dev = self.stream.clone_htod(&descriptors).map_err(error)?;
        let mut partials = self.stream.alloc_zeros::<f64>(256).map_err(error)?;
        let mut norm_dev = self.stream.alloc_zeros::<f64>(1).map_err(error)?;
        let kernels = state.kernels.as_ref().unwrap();
        let count = u32::try_from(tensors.len()).map_err(error)?;
        // SAFETY: descriptors reference exactly the registered, live gradient
        // allocations. Fixed 256-thread blocks match PTX shared storage; the
        // second stage reads exactly 256 partials. No unordered atomic FP sums.
        unsafe {
            self.stream
                .launch_builder(&kernels.partials)
                .arg(&descriptors_dev)
                .arg(&count)
                .arg(&mut partials)
                .launch(LaunchConfig {
                    grid_dim: (256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(error)?;
            self.stream
                .launch_builder(&kernels.finish)
                .arg(&partials)
                .arg(&mut norm_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(error)?;
        }
        let mut norm = [0.0f64];
        self.stream
            .memcpy_dtoh(&norm_dev, &mut norm)
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
        let mut invalid = self.stream.alloc_zeros::<u32>(1).map_err(error)?;
        let resident = state.resident;
        let Safeguards {
            kernels,
            tensors: device_tensors,
            ..
        } = &mut *state;
        for t in tensors.iter_mut() {
            let device = device_tensors.get_mut(&(t.grad.as_ptr() as usize)).unwrap();
            let len = t.grad.len() as u64;
            let launch_len = u32::try_from(len).map_err(error)?;
            // SAFETY: all four allocations have len elements, kernel bounds
            // checks every access; f64 scale rounds to f32 before Adam moments.
            unsafe {
                self.stream
                    .launch_builder(&kernels.as_ref().unwrap().adam)
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
                    .arg(&mut invalid)
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
            .memcpy_dtoh(&invalid, &mut invalid_host)
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
        assert!(width > 0 && values.len() % width == 0 && cap.is_finite() && cap > 0.0);
        if values.is_empty() {
            return Ok(());
        }
        let mut state = self.safeguards.lock().map_err(error)?;
        self.safeguard_kernels(&mut state)?;
        let Safeguards {
            kernels,
            cap_workspace,
            ..
        } = &mut *state;
        reserve_device(cap_workspace, &self.stream, values.len())?;
        let mut values_dev = cap_workspace.as_mut().unwrap().slice_mut(..values.len());
        self.stream
            .memcpy_htod(values, &mut values_dev)
            .map_err(error)?;
        let kernels = kernels.as_ref().unwrap();
        let mut invalid = self.stream.alloc_zeros::<u32>(1).map_err(error)?;
        let rows = u32::try_from(values.len() / width).map_err(error)?;
        let width = u32::try_from(width).map_err(error)?;
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
                .arg(&mut invalid)
                .launch(LaunchConfig::for_num_elems(rows))
                .map_err(error)?;
        }
        let mut bad = [0u32];
        self.stream.memcpy_dtoh(&invalid, &mut bad).map_err(error)?;
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
