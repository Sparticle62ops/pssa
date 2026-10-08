//! CUDA SSM scan, episodic-memory retrieval, and token-local arithmetic.
//!
//! The decomposition is backend-neutral: the affine scan is a tiled exclusive
//! Blelloch scan. One workgroup/block owns one independent latent/state channel
//! and one lane owns one sequence row within a tile. Tile totals are scanned by
//! the same kernel recursively, then composed into each tile's local prefix.
//! Memory retrieval and its VJP use one invocation per token and keep the slot
//! loop inside that invocation. A WGSL port can reuse these ownership rules.

use super::*;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use std::{ffi::CString, ptr};

use super::stage_bounds::{MemoryShape, SsmShape, elems, gemm_shape, lengths, product, scan_shape};

pub(super) struct Kernels {
    pub(super) module: Arc<cudarc::driver::CudaModule>,
    prepare: CudaFunction,
    scan: CudaFunction,
    scan_apply: CudaFunction,
    materialize: CudaFunction,
    backward_maps: CudaFunction,
    backward_local: CudaFunction,
    memory_forward: CudaFunction,
    memory_backward: CudaFunction,
    memory_backward_local: CudaFunction,
    add: CudaFunction,
    sigmoid_mul: CudaFunction,
    sigmoid: CudaFunction,
    softplus: CudaFunction,
}

struct MemoryBuffers {
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    q_euc: CudaSlice<f32>,
    // Query projection scratch is [len,d_k], not [len,d_m].
    q_ssm: CudaSlice<f32>,
    q_pnc: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    weights: CudaSlice<f32>,
    m_val: CudaSlice<f32>,
    g_mem: CudaSlice<f32>,
    m_proj: CudaSlice<f32>,
    m_inj: CudaSlice<f32>,
    keys: Option<CudaSlice<f32>>,
    norm_sq: Option<CudaSlice<f32>>,
    values: Option<CudaSlice<f32>>,
}

struct ForwardBuffers {
    seq_len: usize,
    d_m: usize,
    d_s: usize,
    stride: usize,
    // Backward uploads its host tape separately; only forward results stay resident.
    bar_a: CudaSlice<f32>,
    bar_b: CudaSlice<f32>,
    states: CudaSlice<f32>,
    y_ssm: CudaSlice<f32>,
    memory: Option<MemoryBuffers>,
}

#[derive(Default)]
pub(super) struct StageState {
    kernels: Option<Kernels>,
    // A module-load failure is permanent for this context. Remember it so a
    // malformed embedded PTX cannot trigger a fresh JIT attempt on every
    // fallback call.
    load_error: Option<String>,
    forward: Option<ForwardBuffers>,
}

const STAGE_PTX: &str = include_str!("stages.ptx");

fn error(e: impl std::fmt::Debug) -> String {
    format!("CUDA scan/memory stage failed ({e:?})")
}

fn driver_error(e: &cudarc::driver::DriverError) -> String {
    let name = e
        .error_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown CUDA driver error".to_owned());
    let description = e
        .error_string()
        .map(|description| description.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "no CUDA driver description".to_owned());
    format!("{name}: {description}")
}

/// Ask the driver for its JIT log after cudarc's ordinary loader reports a
/// failure. cudarc 0.19 exposes `load_module`, but not the `cuModuleLoadDataEx`
/// JIT log options, so use its raw driver bindings for this diagnostic retry.
fn jit_error_log(ctx: &cudarc::driver::CudaContext) -> Option<String> {
    ctx.bind_to_thread().ok()?;
    let source = CString::new(STAGE_PTX).ok()?;
    let mut log = vec![0u8; 16 * 1024];
    let mut log_size = log.len() as u32;
    let mut module = ptr::null_mut();
    let mut options = [
        cudarc::driver::sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        cudarc::driver::sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut values = [log.as_mut_ptr().cast(), (&mut log_size as *mut u32).cast()];
    let result = unsafe {
        cudarc::driver::sys::cuModuleLoadDataEx(
            &mut module,
            source.as_ptr().cast(),
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    if !module.is_null() {
        let _ = unsafe { cudarc::driver::result::module::unload(module) };
    }
    let end = log.iter().position(|&byte| byte == 0).unwrap_or(log.len());
    let message = String::from_utf8_lossy(&log[..end]).trim().to_owned();
    (result != cudarc::driver::sys::CUresult::CUDA_SUCCESS && !message.is_empty())
        .then_some(message)
}

fn stage_load_error(
    ctx: &cudarc::driver::CudaContext,
    error: &cudarc::driver::DriverError,
) -> String {
    let detail = driver_error(error);
    match jit_error_log(ctx) {
        Some(log) => format!("{detail}; CUDA JIT error log: {log}"),
        None => format!("{detail}; CUDA driver returned no JIT error log"),
    }
}

impl CudaContext {
    pub(super) fn stage_kernels<'a>(
        &self,
        state: &'a mut StageState,
    ) -> Result<&'a Kernels, String> {
        if let Some(error) = state.load_error.as_ref() {
            return Err(error.clone());
        }
        if state.kernels.is_none() {
            let context = self.stream.context();
            let loaded = (|| -> Result<Kernels, cudarc::driver::DriverError> {
                let module = context.load_module(cudarc::nvrtc::Ptx::from_src(STAGE_PTX))?;
                Ok(Kernels {
                    module: module.clone(),
                    prepare: module.load_function("ssm_prepare")?,
                    scan: module.load_function("affine_scan")?,
                    scan_apply: module.load_function("scan_apply")?,
                    materialize: module.load_function("ssm_materialize")?,
                    backward_maps: module.load_function("ssm_backward_maps")?,
                    backward_local: module.load_function("ssm_backward_local")?,
                    memory_forward: module.load_function("memory_forward")?,
                    memory_backward: module.load_function("memory_backward")?,
                    memory_backward_local: module.load_function("memory_backward_local")?,
                    add: module.load_function("add_in_place")?,
                    sigmoid_mul: module.load_function("sigmoid_mul")?,
                    sigmoid: module.load_function("sigmoid_in_place")?,
                    softplus: module.load_function("softplus_in_place")?,
                })
            })();
            match loaded {
                Ok(kernels) => state.kernels = Some(kernels),
                Err(error) => {
                    let message = format!(
                        "CUDA scan/memory stage module failed to load ({})",
                        stage_load_error(context, &error)
                    );
                    // This is deliberately one diagnostic per context. The
                    // operation-level fallbacks still receive the same error,
                    // but a bad PTX is never silent or retried in a loop.
                    eprintln!("error: {message}");
                    state.load_error = Some(message.clone());
                    return Err(message);
                }
            }
        }
        Ok(state.kernels.as_ref().unwrap())
    }

    fn gemm_device(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: usize,
        k: usize,
        n: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        gemm_shape(m, k, n, a.len(), b.len(), out.len())?;
        // Model weights are row-major [output,input]; transpose their
        // column-major view, just like the public forward matvec path.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            beta: 0.0,
            lda: k as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, b, a, out) }.map_err(error)
    }

    /// Scan arbitrary sequence lengths in tiled workgroups.  The temporary
    /// tile-total arrays are device-only; recursive calls never cross the host
    /// boundary.  `out_a/out_b` are exclusive prefixes on return.
    fn launch_scan(
        &self,
        kernels: &Kernels,
        in_a: &CudaSlice<f32>,
        in_b: &CudaSlice<f32>,
        out_a: &mut CudaSlice<f32>,
        out_b: &mut CudaSlice<f32>,
        len: usize,
        stride: usize,
    ) -> Result<(), String> {
        let (capacity, tiles) = scan_shape(len, stride)?;
        let total = product(len, stride, "scan elements")?;
        lengths(
            "affine scan",
            &[
                ("in_a", in_a.len(), total),
                ("in_b", in_b.len(), total),
                ("out_a", out_a.len(), total),
                ("out_b", out_b.len(), total),
            ],
        )?;
        let len_arg = elems(len, "scan length")?;
        let stride_arg = elems(stride, "scan stride")?;
        let capacity_arg = elems(capacity, "scan tile width")?;
        let tiles_arg = elems(tiles, "scan tile count")?;
        let mut summary_a = self
            .stream
            .alloc_zeros::<f32>(product(tiles, stride, "scan summaries")?)
            .map_err(error)?;
        let mut summary_b = self
            .stream
            .alloc_zeros::<f32>(product(tiles, stride, "scan summaries")?)
            .map_err(error)?;
        unsafe {
            self.stream
                .launch_builder(&kernels.scan)
                .arg(in_a)
                .arg(in_b)
                .arg(&mut *out_a)
                .arg(&mut *out_b)
                .arg(&mut summary_a)
                .arg(&mut summary_b)
                .arg(&len_arg)
                .arg(&stride_arg)
                .arg(&capacity_arg)
                .launch(LaunchConfig {
                    grid_dim: (stride_arg, tiles_arg, 1),
                    block_dim: (capacity_arg, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(error)?;
        }
        if tiles == 1 {
            return Ok(());
        }
        let mut prefix_a = self
            .stream
            .alloc_zeros::<f32>(product(tiles, stride, "scan prefixes")?)
            .map_err(error)?;
        let mut prefix_b = self
            .stream
            .alloc_zeros::<f32>(product(tiles, stride, "scan prefixes")?)
            .map_err(error)?;
        self.launch_scan(
            kernels,
            &summary_a,
            &summary_b,
            &mut prefix_a,
            &mut prefix_b,
            tiles,
            stride,
        )?;
        unsafe {
            self.stream
                .launch_builder(&kernels.scan_apply)
                .arg(&mut *out_a)
                .arg(&mut *out_b)
                .arg(&prefix_a)
                .arg(&prefix_b)
                .arg(&len_arg)
                .arg(&stride_arg)
                .arg(&capacity_arg)
                .launch(LaunchConfig::for_num_elems(elems(
                    product(len, stride, "scan elements")?,
                    "scan elements",
                )?))
                .map_err(error)?;
        }
        Ok(())
    }

    fn make_ssm_buffers(
        &self,
        kernels: &Kernels,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        x_norm: &[f32],
        rates: &[f32],
        c_proj: &[f32],
        initial: &[f32],
        seq_len: usize,
        d_m: usize,
        d_s: usize,
    ) -> Result<ForwardBuffers, String> {
        let SsmShape {
            stride,
            token_m,
            token_s,
            token_state,
            state_len,
        } = SsmShape::new(seq_len, d_m, d_s)?;
        if delta.len() != token_m
            || delta_raw.len() != token_m
            || b_proj.len() != token_s
            || x_norm.len() != token_m
            || rates.len() != stride
            || c_proj.len() != token_s
            || initial.len() != stride
        {
            return Err("CUDA SSM forward shape mismatch".into());
        }
        let d_delta = self.stream.clone_htod(delta).map_err(error)?;
        let d_b = self.stream.clone_htod(b_proj).map_err(error)?;
        let d_x = self.stream.clone_htod(x_norm).map_err(error)?;
        let d_rates = self.stream.clone_htod(rates).map_err(error)?;
        let d_c = self.stream.clone_htod(c_proj).map_err(error)?;
        let d_initial = self.stream.clone_htod(initial).map_err(error)?;
        let mut d_bar_a = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_bar_b = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_scan_a = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_scan_b = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_scan_input_b = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_states = self.stream.alloc_zeros::<f32>(state_len).map_err(error)?;
        self.stream
            .memcpy_htod(initial, &mut d_states.slice_mut(..stride))
            .map_err(error)?;
        let mut d_y = self.stream.alloc_zeros::<f32>(token_m).map_err(error)?;
        let len_arg = elems(seq_len, "SSM length")?;
        let dm_arg = elems(d_m, "latent width")?;
        let ds_arg = elems(d_s, "state width")?;
        unsafe {
            self.stream
                .launch_builder(&kernels.prepare)
                .arg(&d_delta)
                .arg(&d_b)
                .arg(&d_x)
                .arg(&d_rates)
                .arg(&mut d_bar_a)
                .arg(&mut d_bar_b)
                .arg(&mut d_scan_input_b)
                .arg(&len_arg)
                .arg(&dm_arg)
                .arg(&ds_arg)
                .launch(LaunchConfig::for_num_elems(elems(
                    token_state,
                    "SSM elements",
                )?))
                .map_err(error)?;
        }
        self.launch_scan(
            kernels,
            &d_bar_a,
            &d_scan_input_b,
            &mut d_scan_a,
            &mut d_scan_b,
            seq_len,
            stride,
        )?;
        unsafe {
            self.stream
                .launch_builder(&kernels.materialize)
                .arg(&d_initial)
                .arg(&d_bar_a)
                .arg(&d_bar_b)
                .arg(&d_x)
                .arg(&d_scan_a)
                .arg(&d_scan_b)
                .arg(&d_c)
                .arg(&mut d_states)
                .arg(&mut d_y)
                .arg(&len_arg)
                .arg(&dm_arg)
                .arg(&ds_arg)
                .launch(LaunchConfig::for_num_elems(elems(token_m, "SSM outputs")?))
                .map_err(error)?;
        }
        Ok(ForwardBuffers {
            seq_len,
            d_m,
            d_s,
            stride,
            bar_a: d_bar_a,
            bar_b: d_bar_b,
            states: d_states,
            y_ssm: d_y,
            memory: None,
        })
    }

    fn readback_ssm(
        &self,
        fwd: &ForwardBuffers,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y_ssm: &mut [f32],
    ) -> Result<(), String> {
        let SsmShape {
            token_state,
            token_m,
            state_len,
            ..
        } = SsmShape::new(fwd.seq_len, fwd.d_m, fwd.d_s)?;
        if bar_a.len() != token_state
            || bar_b.len() != token_state
            || states.len() < state_len
            || y_ssm.len() != token_m
        {
            return Err("CUDA SSM output shape mismatch".into());
        }
        self.stream.memcpy_dtoh(&fwd.bar_a, bar_a).map_err(error)?;
        self.stream.memcpy_dtoh(&fwd.bar_b, bar_b).map_err(error)?;
        self.stream
            .memcpy_dtoh(&fwd.states, &mut states[..state_len])
            .map_err(error)?;
        self.stream.memcpy_dtoh(&fwd.y_ssm, y_ssm).map_err(error)?;
        Ok(())
    }

    pub(crate) fn ssm_forward_resident(
        &self,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        x_norm: &[f32],
        rates: &[f32],
        c_proj: &[f32],
        initial: &[f32],
        seq_len: usize,
        d_m: usize,
        d_s: usize,
    ) -> Result<(), String> {
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let fwd = self.make_ssm_buffers(
            kernels, delta, delta_raw, b_proj, x_norm, rates, c_proj, initial, seq_len, d_m, d_s,
        )?;
        state.forward = Some(fwd);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ssm_forward(
        &self,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        x_norm: &[f32],
        rates: &[f32],
        rate_deriv: &[f32],
        c_proj: &[f32],
        initial: &[f32],
        seq_len: usize,
        d_m: usize,
        d_s: usize,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y_ssm: &mut [f32],
    ) -> Result<(), String> {
        let _ = rate_deriv;
        self.ssm_forward_resident(
            delta, delta_raw, b_proj, x_norm, rates, c_proj, initial, seq_len, d_m, d_s,
        )?;
        let state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let fwd = state.forward.as_ref().ok_or("CUDA SSM state was lost")?;
        self.readback_ssm(fwd, bar_a, bar_b, states, y_ssm)?;
        self.stream.synchronize().map_err(error)
    }

    fn make_memory_buffers(
        &self,
        x: CudaSlice<f32>,
        y: CudaSlice<f32>,
        seq_len: usize,
        d_m: usize,
        d_k: usize,
        d_val: usize,
        capacity: usize,
        weights: &[f32],
    ) -> Result<MemoryBuffers, String> {
        let shape = MemoryShape::new(seq_len, d_m, d_k, d_val, capacity, 0, 1.0)?;
        let (x_len, q_len, value_len, weight_len) =
            (shape.x, shape.query, shape.value, shape.weights);
        if x.len() != x_len || y.len() != x_len || weights.len() != weight_len {
            return Err("CUDA memory buffer shape mismatch".into());
        }
        Ok(MemoryBuffers {
            x,
            y,
            q_euc: self.stream.alloc_zeros::<f32>(q_len).map_err(error)?,
            q_ssm: self.stream.alloc_zeros::<f32>(q_len).map_err(error)?,
            q_pnc: self.stream.alloc_zeros::<f32>(q_len).map_err(error)?,
            q_norm: self.stream.alloc_zeros::<f32>(seq_len).map_err(error)?,
            weights: self.stream.clone_htod(weights).map_err(error)?,
            m_val: self.stream.alloc_zeros::<f32>(value_len).map_err(error)?,
            g_mem: self.stream.alloc_zeros::<f32>(x_len).map_err(error)?,
            m_proj: self.stream.alloc_zeros::<f32>(x_len).map_err(error)?,
            m_inj: self.stream.alloc_zeros::<f32>(x_len).map_err(error)?,
            keys: None,
            norm_sq: None,
            values: None,
        })
    }

    fn run_memory_forward(
        &self,
        kernels: &Kernels,
        mem: &mut MemoryBuffers,
        w_qx: &[f32],
        w_qh: &[f32],
        w_gate: &[f32],
        w_proj: &[f32],
        keys: &[f32],
        norm_sq: &[f32],
        values: &[f32],
        seq_len: usize,
        d_m: usize,
        d_k: usize,
        d_val: usize,
        capacity: usize,
        count: usize,
        tau: f32,
    ) -> Result<(), String> {
        let shape = MemoryShape::new(seq_len, d_m, d_k, d_val, capacity, count, tau)?;
        lengths(
            "memory forward",
            &[
                ("w_qx", w_qx.len(), shape.w_query),
                ("w_qh", w_qh.len(), shape.w_query),
                ("w_gate", w_gate.len(), shape.w_gate),
                ("w_proj", w_proj.len(), shape.w_proj),
                ("keys", keys.len(), shape.keys),
                ("norm_sq", norm_sq.len(), capacity),
                ("values", values.len(), shape.values),
                ("x", mem.x.len(), shape.x),
                ("y", mem.y.len(), shape.x),
                ("q_euc", mem.q_euc.len(), shape.query),
                ("q_ssm", mem.q_ssm.len(), shape.query),
                ("q_pnc", mem.q_pnc.len(), shape.query),
                ("q_norm", mem.q_norm.len(), seq_len),
                ("weights", mem.weights.len(), shape.weights),
                ("m_val", mem.m_val.len(), shape.value),
                ("g_mem", mem.g_mem.len(), shape.x),
                ("m_proj", mem.m_proj.len(), shape.x),
                ("m_inj", mem.m_inj.len(), shape.x),
            ],
        )?;
        let upload_trace = crate::training_diagnostics::StageTrace::cuda("memory.upload", seq_len);
        let dwqx = self.stream.clone_htod(w_qx).map_err(error)?;
        let dwqh = self.stream.clone_htod(w_qh).map_err(error)?;
        let dwg = self.stream.clone_htod(w_gate).map_err(error)?;
        let dwp = self.stream.clone_htod(w_proj).map_err(error)?;
        let dkeys = self.stream.clone_htod(keys).map_err(error)?;
        let dnorm = self.stream.clone_htod(norm_sq).map_err(error)?;
        let dvalues = self.stream.clone_htod(values).map_err(error)?;
        drop(upload_trace);
        let query_trace =
            crate::training_diagnostics::StageTrace::cuda("memory.query_submit", seq_len);
        self.gemm_device(&mem.x, &dwqx, seq_len, d_m, d_k, &mut mem.q_euc)?;
        self.gemm_device(&mem.y, &dwqh, seq_len, d_m, d_k, &mut mem.q_ssm)?;
        let q_len = elems(product(seq_len, d_k, "query count")?, "query count")?;
        unsafe {
            self.stream
                .launch_builder(&kernels.add)
                .arg(&mut mem.q_euc)
                .arg(&mem.q_ssm)
                .arg(&q_len)
                .launch(LaunchConfig::for_num_elems(q_len))
                .map_err(error)?;
        }
        drop(query_trace);
        let retrieval_trace =
            crate::training_diagnostics::StageTrace::cuda("memory.retrieval_submit", seq_len);
        let len_arg = elems(seq_len, "memory length")?;
        let count_arg = elems(count, "memory count")?;
        let capacity_arg = elems(capacity, "memory capacity")?;
        let key_arg = elems(d_k, "key width")?;
        // Score/query work is token-local, but value retrieval is a large
        // [tokens,capacity] * [capacity,value] reduction. Do not serialize it
        // inside one PTX thread per token; cuBLAS owns the packed reduction.
        let val_arg = 0u32;
        unsafe {
            self.stream
                .launch_builder(&kernels.memory_forward)
                .arg(&mem.q_euc)
                .arg(&dkeys)
                .arg(&dnorm)
                .arg(&dvalues)
                .arg(&mut mem.q_pnc)
                .arg(&mut mem.q_norm)
                .arg(&mut mem.m_val)
                .arg(&mut mem.weights)
                .arg(&len_arg)
                .arg(&count_arg)
                .arg(&capacity_arg)
                .arg(&key_arg)
                .arg(&val_arg)
                .arg(&tau)
                .launch(LaunchConfig::for_num_elems(len_arg))
                .map_err(error)?;
        }
        gemm_shape(
            seq_len,
            capacity,
            d_val,
            mem.weights.len(),
            dvalues.len(),
            mem.m_val.len(),
        )?;
        if count == 0 {
            self.stream.memset_zeros(&mut mem.m_val).map_err(error)?;
        } else {
            let value_gemm = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: d_val as i32,
                n: seq_len as i32,
                k: count as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: d_val as i32,
                ldb: capacity as i32,
                ldc: d_val as i32,
            };
            // Only occupied slots participate; 0 * a stale NaN/Inf in an
            // inactive value row must not contaminate an otherwise valid read.
            let active_values = dvalues.slice(..count * d_val);
            unsafe {
                self.blas
                    .gemm(value_gemm, &active_values, &mem.weights, &mut mem.m_val)
            }
            .map_err(error)?;
        }
        drop(retrieval_trace);
        let _projection_trace =
            crate::training_diagnostics::StageTrace::cuda("memory.gate_projection_submit", seq_len);
        self.gemm_device(&mem.x, &dwg, seq_len, d_m, d_m, &mut mem.g_mem)?;
        let x_len = elems(
            product(seq_len, d_m, "memory element count")?,
            "memory element count",
        )?;
        unsafe {
            self.stream
                .launch_builder(&kernels.sigmoid)
                .arg(&mut mem.g_mem)
                .arg(&x_len)
                .launch(LaunchConfig::for_num_elems(x_len))
                .map_err(error)?;
        }
        self.gemm_device(&mem.m_val, &dwp, seq_len, d_val, d_m, &mut mem.m_proj)?;
        unsafe {
            self.stream
                .launch_builder(&kernels.sigmoid_mul)
                .arg(&mem.g_mem)
                .arg(&mem.m_proj)
                .arg(&mut mem.m_inj)
                .arg(&x_len)
                .launch(LaunchConfig::for_num_elems(x_len))
                .map_err(error)?;
        }
        mem.keys = Some(dkeys);
        mem.norm_sq = Some(dnorm);
        mem.values = Some(dvalues);
        Ok(())
    }

    fn readback_memory(
        &self,
        mem: &MemoryBuffers,
        q_euc: &mut [f32],
        q_pnc: &mut [f32],
        q_norm: &mut [f32],
        weights: &mut [f32],
        m_val: &mut [f32],
        g_mem: &mut [f32],
        m_proj: &mut [f32],
        m_inj: &mut [f32],
    ) -> Result<(), String> {
        let _trace =
            crate::training_diagnostics::StageTrace::cuda("memory.readback_wait", q_norm.len());
        self.stream.memcpy_dtoh(&mem.q_euc, q_euc).map_err(error)?;
        self.stream.memcpy_dtoh(&mem.q_pnc, q_pnc).map_err(error)?;
        self.stream
            .memcpy_dtoh(&mem.q_norm, q_norm)
            .map_err(error)?;
        self.stream
            .memcpy_dtoh(&mem.weights, weights)
            .map_err(error)?;
        self.stream.memcpy_dtoh(&mem.m_val, m_val).map_err(error)?;
        self.stream.memcpy_dtoh(&mem.g_mem, g_mem).map_err(error)?;
        self.stream
            .memcpy_dtoh(&mem.m_proj, m_proj)
            .map_err(error)?;
        self.stream.memcpy_dtoh(&mem.m_inj, m_inj).map_err(error)?;
        self.stream.synchronize().map_err(error)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn memory_forward_after_ssm(
        &self,
        x_norm: &[f32],
        w_qx: &[f32],
        w_qh: &[f32],
        w_gate: &[f32],
        w_proj: &[f32],
        keys: &[f32],
        norm_sq: &[f32],
        values: &[f32],
        seq_len: usize,
        d_m: usize,
        d_k: usize,
        d_val: usize,
        capacity: usize,
        count: usize,
        tau: f32,
        bar_a: &mut [f32],
        bar_b: &mut [f32],
        states: &mut [f32],
        y_ssm: &mut [f32],
        q_euc: &mut [f32],
        q_pnc: &mut [f32],
        q_norm: &mut [f32],
        weights: &mut [f32],
        m_val: &mut [f32],
        g_mem: &mut [f32],
        m_proj: &mut [f32],
        m_inj: &mut [f32],
    ) -> Result<(), String> {
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let mut fwd = state
            .forward
            .take()
            .ok_or("CUDA SSM result is not resident")?;
        let kernels = match self.stage_kernels(&mut state) {
            Ok(kernels) => kernels,
            Err(error) => {
                state.forward = Some(fwd);
                return Err(error);
            }
        };
        if let Err(error) = MemoryShape::new(seq_len, d_m, d_k, d_val, capacity, count, tau) {
            state.forward = Some(fwd);
            return Err(error);
        }
        let resident_stride = match product(fwd.d_s, d_m, "resident SSM stride") {
            Ok(stride) => stride,
            Err(error) => {
                state.forward = Some(fwd);
                return Err(error);
            }
        };
        if fwd.seq_len != seq_len || fwd.d_m != d_m || resident_stride != fwd.stride {
            state.forward = Some(fwd);
            return Err("CUDA resident SSM shape mismatch".into());
        }
        // Keep the resident SSM result recoverable. A memory launch or readback
        // can fail after the SSM has already advanced, and the caller's direct
        // host fallback must not consume the previous/stale y_ssm tape.
        let result = (|| -> Result<(), String> {
            let x = self.stream.clone_htod(x_norm).map_err(error)?;
            let y = fwd.y_ssm.clone();
            let mut mem =
                self.make_memory_buffers(x, y, seq_len, d_m, d_k, d_val, capacity, weights)?;
            self.run_memory_forward(
                kernels, &mut mem, w_qx, w_qh, w_gate, w_proj, keys, norm_sq, values, seq_len, d_m,
                d_k, d_val, capacity, count, tau,
            )?;
            self.readback_ssm(&fwd, bar_a, bar_b, states, y_ssm)?;
            self.readback_memory(
                &mem, q_euc, q_pnc, q_norm, weights, m_val, g_mem, m_proj, m_inj,
            )?;
            fwd.memory = Some(mem);
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.forward = Some(fwd);
                Ok(())
            }
            Err(error) => {
                // This synchronization/readback is the last safe opportunity
                // to publish the current resident SSM output before fallback.
                let recovery = self.readback_ssm(&fwd, bar_a, bar_b, states, y_ssm);
                state.forward = Some(fwd);
                match recovery {
                    Ok(()) => Err(error),
                    Err(recovery_error) => Err(format!(
                        "{error}; unable to publish resident SSM for fallback: {recovery_error}"
                    )),
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn memory_forward(
        &self,
        x_norm: &[f32],
        y_ssm: &[f32],
        w_qx: &[f32],
        w_qh: &[f32],
        w_gate: &[f32],
        w_proj: &[f32],
        keys: &[f32],
        norm_sq: &[f32],
        values: &[f32],
        seq_len: usize,
        d_m: usize,
        d_k: usize,
        d_val: usize,
        capacity: usize,
        count: usize,
        tau: f32,
        q_euc: &mut [f32],
        q_pnc: &mut [f32],
        q_norm: &mut [f32],
        weights: &mut [f32],
        m_val: &mut [f32],
        g_mem: &mut [f32],
        m_proj: &mut [f32],
        m_inj: &mut [f32],
    ) -> Result<(), String> {
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let shape = MemoryShape::new(seq_len, d_m, d_k, d_val, capacity, count, tau)?;
        lengths(
            "memory input",
            &[
                ("x_norm", x_norm.len(), shape.x),
                ("y_ssm", y_ssm.len(), shape.x),
            ],
        )?;
        let x = self.stream.clone_htod(x_norm).map_err(error)?;
        let y = self.stream.clone_htod(y_ssm).map_err(error)?;
        let mut mem =
            self.make_memory_buffers(x, y, seq_len, d_m, d_k, d_val, capacity, weights)?;
        self.run_memory_forward(
            kernels, &mut mem, w_qx, w_qh, w_gate, w_proj, keys, norm_sq, values, seq_len, d_m,
            d_k, d_val, capacity, count, tau,
        )?;
        self.readback_memory(
            &mem, q_euc, q_pnc, q_norm, weights, m_val, g_mem, m_proj, m_inj,
        )
    }

    pub(crate) fn memory_backward_local(
        &self,
        g_zraw: &[f32],
        g_mem: &[f32],
        m_proj: &[f32],
        g_m_proj: &mut [f32],
        g_gate: &mut [f32],
    ) -> Result<(), String> {
        if g_zraw.len() != g_mem.len()
            || g_zraw.len() != m_proj.len()
            || g_zraw.len() != g_m_proj.len()
            || g_zraw.len() != g_gate.len()
        {
            return Err("CUDA memory backward local shape mismatch".into());
        }
        if g_zraw.is_empty() {
            return Ok(());
        }
        let len = elems(g_zraw.len(), "memory backward local count")?;
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let dz = self.stream.clone_htod(g_zraw).map_err(error)?;
        let dm = self.stream.clone_htod(g_mem).map_err(error)?;
        let dp = self.stream.clone_htod(m_proj).map_err(error)?;
        let mut out_m = self
            .stream
            .alloc_zeros::<f32>(g_m_proj.len())
            .map_err(error)?;
        let mut out_g = self
            .stream
            .alloc_zeros::<f32>(g_gate.len())
            .map_err(error)?;
        unsafe {
            self.stream
                .launch_builder(&kernels.memory_backward_local)
                .arg(&dz)
                .arg(&dm)
                .arg(&dp)
                .arg(&mut out_m)
                .arg(&mut out_g)
                .arg(&len)
                .launch(LaunchConfig::for_num_elems(len))
                .map_err(error)?;
        }
        self.stream.memcpy_dtoh(&out_m, g_m_proj).map_err(error)?;
        self.stream.memcpy_dtoh(&out_g, g_gate).map_err(error)?;
        self.stream.synchronize().map_err(error)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ssm_backward(
        &self,
        delta: &[f32],
        delta_raw: &[f32],
        b_proj: &[f32],
        c_proj: &[f32],
        rates: &[f32],
        rate_deriv: &[f32],
        x_norm: &[f32],
        states: &[f32],
        bar_a: &[f32],
        bar_b: &[f32],
        g_zraw: &[f32],
        g_ysm: &[f32],
        seq_len: usize,
        d_m: usize,
        d_s: usize,
        scale: f32,
        g_delta: &mut [f32],
        g_b: &mut [f32],
        g_c: &mut [f32],
        g_a: &mut [f32],
        g_x: &mut [f32],
    ) -> Result<(), String> {
        let SsmShape {
            stride,
            token_m,
            token_s,
            token_state,
            state_len,
        } = SsmShape::new(seq_len, d_m, d_s)?;
        if delta.len() != token_m
            || delta_raw.len() != token_m
            || b_proj.len() != token_s
            || c_proj.len() != token_s
            || rates.len() != stride
            || rate_deriv.len() != stride
            || x_norm.len() != token_m
            || states.len() < state_len
            || bar_a.len() != token_state
            || bar_b.len() != token_state
            || g_zraw.len() != token_m
            || g_ysm.len() != token_m
            || g_delta.len() != token_m
            || g_b.len() != token_s
            || g_c.len() != token_s
            || g_a.len() != token_state
            || g_x.len() != token_m
        {
            return Err("CUDA SSM backward shape mismatch".into());
        }
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let d_delta = self.stream.clone_htod(delta).map_err(error)?;
        let d_raw = self.stream.clone_htod(delta_raw).map_err(error)?;
        let d_b = self.stream.clone_htod(b_proj).map_err(error)?;
        let d_c = self.stream.clone_htod(c_proj).map_err(error)?;
        let d_rates = self.stream.clone_htod(rates).map_err(error)?;
        let d_deriv = self.stream.clone_htod(rate_deriv).map_err(error)?;
        let d_x = self.stream.clone_htod(x_norm).map_err(error)?;
        let d_states = self
            .stream
            .clone_htod(&states[..state_len])
            .map_err(error)?;
        let d_bar_a = self.stream.clone_htod(bar_a).map_err(error)?;
        let d_bar_b = self.stream.clone_htod(bar_b).map_err(error)?;
        let d_gz = self.stream.clone_htod(g_zraw).map_err(error)?;
        let d_gy = self.stream.clone_htod(g_ysm).map_err(error)?;
        let mut d_rev_a = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_rev_b = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_scan_a = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_scan_b = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_gdelta = self.stream.alloc_zeros::<f32>(token_m).map_err(error)?;
        let mut d_gb = self.stream.alloc_zeros::<f32>(token_s).map_err(error)?;
        let mut d_gc = self.stream.alloc_zeros::<f32>(token_s).map_err(error)?;
        let mut d_ga = self.stream.alloc_zeros::<f32>(token_state).map_err(error)?;
        let mut d_gx = self.stream.alloc_zeros::<f32>(token_m).map_err(error)?;
        let len_arg = elems(seq_len, "SSM length")?;
        let dm_arg = elems(d_m, "latent width")?;
        let ds_arg = elems(d_s, "state width")?;
        let stride_arg = elems(stride, "SSM stride")?;
        unsafe {
            self.stream
                .launch_builder(&kernels.backward_maps)
                .arg(&d_bar_a)
                .arg(&d_c)
                .arg(&d_gz)
                .arg(&d_gy)
                .arg(&mut d_rev_a)
                .arg(&mut d_rev_b)
                .arg(&len_arg)
                .arg(&dm_arg)
                .arg(&ds_arg)
                .arg(&scale)
                .launch(LaunchConfig::for_num_elems(elems(
                    token_state,
                    "SSM backward maps",
                )?))
                .map_err(error)?;
        }
        self.launch_scan(
            kernels,
            &d_rev_a,
            &d_rev_b,
            &mut d_scan_a,
            &mut d_scan_b,
            seq_len,
            stride,
        )?;
        unsafe {
            self.stream
                .launch_builder(&kernels.backward_local)
                .arg(&d_delta)
                .arg(&d_raw)
                .arg(&d_b)
                .arg(&d_c)
                .arg(&d_rates)
                .arg(&d_deriv)
                .arg(&d_x)
                .arg(&d_states)
                .arg(&d_bar_a)
                .arg(&d_bar_b)
                .arg(&d_scan_b)
                .arg(&d_gz)
                .arg(&d_gy)
                .arg(&mut d_gdelta)
                .arg(&mut d_gb)
                .arg(&mut d_gc)
                .arg(&mut d_ga)
                .arg(&mut d_gx)
                .arg(&len_arg)
                .arg(&dm_arg)
                .arg(&ds_arg)
                .arg(&stride_arg)
                .arg(&scale)
                .launch(LaunchConfig::for_num_elems(len_arg))
                .map_err(error)?;
        }
        self.stream.memcpy_dtoh(&d_gdelta, g_delta).map_err(error)?;
        self.stream.memcpy_dtoh(&d_gb, g_b).map_err(error)?;
        self.stream.memcpy_dtoh(&d_gc, g_c).map_err(error)?;
        self.stream.memcpy_dtoh(&d_ga, g_a).map_err(error)?;
        self.stream.memcpy_dtoh(&d_gx, g_x).map_err(error)?;
        self.stream.synchronize().map_err(error)
    }

    pub(crate) fn memory_backward_retrieval(
        &self,
        q_pnc: &[f32],
        q_euc: &[f32],
        g_m: &[f32],
        m_val: &[f32],
        weights: &[f32],
        keys: &[f32],
        norm_sq: &[f32],
        values: &[f32],
        seq_len: usize,
        count: usize,
        capacity: usize,
        d_key: usize,
        d_val: usize,
        tau: f32,
        query_pnc: &mut [f32],
        query_euc: &mut [f32],
    ) -> Result<(), String> {
        // Retrieval has no dense latent GEMM; use unit input width here.
        let shape = MemoryShape::new(seq_len, 1, d_key, d_val, capacity, count, tau)?;
        lengths(
            "memory backward",
            &[
                ("q_pnc", q_pnc.len(), shape.query),
                ("q_euc", q_euc.len(), shape.query),
                ("g_m", g_m.len(), shape.value),
                ("m_val", m_val.len(), shape.value),
                ("weights", weights.len(), shape.weights),
                ("keys", keys.len(), shape.keys),
                ("norm_sq", norm_sq.len(), capacity),
                ("values", values.len(), shape.values),
                ("query_pnc", query_pnc.len(), shape.query),
                ("query_euc", query_euc.len(), shape.query),
            ],
        )?;
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let dq = self.stream.clone_htod(q_pnc).map_err(error)?;
        let de = self.stream.clone_htod(q_euc).map_err(error)?;
        let dgm = self.stream.clone_htod(g_m).map_err(error)?;
        let dmv = self.stream.clone_htod(m_val).map_err(error)?;
        let dw = self.stream.clone_htod(weights).map_err(error)?;
        let dk = self.stream.clone_htod(keys).map_err(error)?;
        let dn = self.stream.clone_htod(norm_sq).map_err(error)?;
        let dv = self.stream.clone_htod(values).map_err(error)?;
        let mut dqp = self
            .stream
            .alloc_zeros::<f32>(query_pnc.len())
            .map_err(error)?;
        let mut dqe = self
            .stream
            .alloc_zeros::<f32>(query_euc.len())
            .map_err(error)?;
        let len_arg = elems(seq_len, "memory length")?;
        let count_arg = elems(count, "memory count")?;
        let capacity_arg = elems(capacity, "memory capacity")?;
        let key_arg = elems(d_key, "key width")?;
        let val_arg = elems(d_val, "value width")?;
        unsafe {
            self.stream
                .launch_builder(&kernels.memory_backward)
                .arg(&dq)
                .arg(&de)
                .arg(&dgm)
                .arg(&dmv)
                .arg(&dw)
                .arg(&dk)
                .arg(&dn)
                .arg(&dv)
                .arg(&mut dqp)
                .arg(&mut dqe)
                .arg(&len_arg)
                .arg(&count_arg)
                .arg(&capacity_arg)
                .arg(&key_arg)
                .arg(&val_arg)
                .arg(&tau)
                .launch(LaunchConfig::for_num_elems(len_arg))
                .map_err(error)?;
        }
        self.stream.memcpy_dtoh(&dqp, query_pnc).map_err(error)?;
        self.stream.memcpy_dtoh(&dqe, query_euc).map_err(error)?;
        self.stream.synchronize().map_err(error)
    }

    pub(crate) fn softplus_in_place(&self, values: &mut [f32]) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        let mut state = self.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let kernels = self.stage_kernels(&mut state)?;
        let len = elems(values.len(), "softplus count")?;
        let mut device = self.stream.clone_htod(values).map_err(error)?;
        unsafe {
            self.stream
                .launch_builder(&kernels.softplus)
                .arg(&mut device)
                .arg(&len)
                .launch(LaunchConfig::for_num_elems(len))
                .map_err(error)?;
        }
        self.stream.memcpy_dtoh(&device, values).map_err(error)?;
        self.stream.synchronize().map_err(error)
    }
}
