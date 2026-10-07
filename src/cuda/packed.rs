//! Independent-lane CUDA workspace owned by SequenceBatch, not by a context's
//! single-sequence tape. All scratch is allocated once; views select the active
//! packed rows. Forward tapes survive every intervening host/dense stage.
use super::stage_bounds::{MemoryShape, elems, lengths, product};
use super::*;
use crate::pssa::PSSALayerV2;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

struct PackedKernels {
    prepare: CudaFunction,
    scan: CudaFunction,
    materialize: CudaFunction,
    maps: CudaFunction,
    local: CudaFunction,
    memory: CudaFunction,
    memory_local: CudaFunction,
    retrieval: CudaFunction,
    add: CudaFunction,
    sigmoid: CudaFunction,
    multiply: CudaFunction,
    softplus: CudaFunction,
    reduce: CudaFunction,
    reduce_bc: CudaFunction,
}

pub(crate) struct PackedWorkspace {
    ctx: CudaContext,
    kernels: PackedKernels,
    chunk: usize,
    lanes: usize,
    d: usize,
    s: usize,
    k: usize,
    cap: usize,
    pub offsets: Vec<u32>,
    pub lengths: Vec<u32>,
    pub carries: Vec<f32>,
    offsets_dev: CudaSlice<u32>,
    lengths_dev: CudaSlice<u32>,
    x: CudaSlice<f32>,
    delta: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
    rates: CudaSlice<f32>,
    deriv: CudaSlice<f32>,
    a: CudaSlice<f32>,
    bb: CudaSlice<f32>,
    scan_a: CudaSlice<f32>,
    scan_b: CudaSlice<f32>,
    rev_a: CudaSlice<f32>,
    rev_b: CudaSlice<f32>,
    initial: CudaSlice<f32>,
    carry: CudaSlice<f32>,
    states: CudaSlice<f32>,
    y: CudaSlice<f32>,
    qe: CudaSlice<f32>,
    qs: CudaSlice<f32>,
    qp: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    weights: CudaSlice<f32>,
    mv: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    mp: CudaSlice<f32>,
    inj: CudaSlice<f32>,
    keys: CudaSlice<f32>,
    norms: CudaSlice<f32>,
    values: CudaSlice<f32>,
    wd: CudaSlice<f32>,
    wb: CudaSlice<f32>,
    wc: CudaSlice<f32>,
    wqx: CudaSlice<f32>,
    wqh: CudaSlice<f32>,
    wg: CudaSlice<f32>,
    wp: CudaSlice<f32>,
    gz: CudaSlice<f32>,
    gmp: CudaSlice<f32>,
    gg: CudaSlice<f32>,
    gmv: CudaSlice<f32>,
    gqp: CudaSlice<f32>,
    gqe: CudaSlice<f32>,
    gx: CudaSlice<f32>,
    gy: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    gd: CudaSlice<f32>,
    gb: CudaSlice<f32>,
    gc: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
    // Weight adjoints are computed into private buffers: beta=0 for host-owned
    // gradients, beta=1 after seeding scoped optimizer gradients. Neither a
    // launch error nor a failed readback can partially publish model gradients.
    weight_grads: Vec<CudaSlice<f32>>,
    host_grads: Vec<Vec<f32>>,
    host_gx: Vec<f32>,
    host_ga: Vec<f32>,
    ready: bool,
}

// cuDARC's convenience launcher uses 1024-thread blocks. The packed local
// VJP/retrieval kernels have long-lived pointer and f64 registers; 256 threads
// avoid making their maximum block size depend on the JIT register allocation.
fn packed_launch(elements: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (elements.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// The reused retrieval kernels walk a complete bank (and value VJP) in one
// thread per token. Spread those serial walks across SMs instead of packing a
// short 32-lane batch into a single block. They have no barriers/shared memory.
fn packed_row_launch(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn err(e: impl std::fmt::Debug) -> String {
    format!("CUDA packed workspace failed ({e:?})")
}

impl PackedWorkspace {
    pub fn finish_failed_transaction(&self, failure: String) -> String {
        // Drain any submitted copies before CPU replay mutates borrowed tapes.
        match self.ctx.stream.synchronize() {
            Ok(()) => failure,
            Err(wait) => format!("{failure}; CUDA recovery wait also failed: {wait:?}"),
        }
    }

    pub fn recover_host_gradients(&self, m: &mut PSSALayerV2) -> Result<(), String> {
        self.ctx.host_owned_gradients(&mut [
            &mut m.block.w_delta.grad,
            &mut m.block.w_b.grad,
            &mut m.block.w_c.grad,
            &mut m.block.w_qx.grad,
            &mut m.block.w_qh.grad,
            &mut m.block.w_gate.grad,
            &mut m.block.w_proj.grad,
        ])
    }

    pub fn matches(&self, ctx: &CudaContext) -> bool {
        Arc::ptr_eq(&self.ctx.stream, &ctx.stream)
    }

    pub fn new(ctx: &CudaContext, m: &PSSALayerV2, lanes: usize) -> Result<Self, String> {
        if lanes == 0 || lanes > 65535 {
            return Err("CUDA packed lane grid invalid; refusing launch".into());
        }
        let (chunk, d, s, k, cap) = (
            m.cfg.chunk_len,
            m.cfg.d_latent,
            m.cfg.d_state,
            m.cfg.d_mem_key,
            m.cfg.mem_capacity,
        );
        let n = product(lanes, chunk, "packed rows")?;
        validate_layout(&vec![0; lanes], &vec![0; lanes], chunk, 0, d, s)?;
        if n > i32::MAX as usize {
            return Err("CUDA packed row count exceeds i32; refusing launch".into());
        }
        let ms = MemoryShape::new(n, d, k, d, cap, m.memory.count, m.cfg.tau_mem)?;
        let hs = product(d, s, "packed stride")?;
        let maps = product(n, hs, "packed maps")?;
        let state_rows = product(
            lanes,
            chunk.checked_add(1).ok_or("packed terminal overflow")?,
            "packed state rows",
        )?;
        let states = product(state_rows, hs, "packed states")?;
        let carries = product(lanes, hs, "packed carries")?;
        let projection = product(s, d, "packed projection weights")?;
        let stream = &ctx.stream;
        // Cache the loaded stage module and its permanent failure diagnostic.
        let mut stages = ctx.stages.lock().map_err(|_| "CUDA stage lock poisoned")?;
        let module = ctx.stage_kernels(&mut stages)?.module.clone();
        let reduction = stream
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(include_str!("packed.ptx")))
            .map_err(err)?;
        let kernels = PackedKernels {
            prepare: module.load_function("ssm_prepare").map_err(err)?,
            scan: module.load_function("packed_scan").map_err(err)?,
            materialize: module.load_function("packed_materialize").map_err(err)?,
            maps: module.load_function("packed_backward_maps").map_err(err)?,
            local: module.load_function("packed_backward_local").map_err(err)?,
            memory: module.load_function("memory_forward").map_err(err)?,
            memory_local: module.load_function("memory_backward_local").map_err(err)?,
            retrieval: module.load_function("memory_backward").map_err(err)?,
            add: module.load_function("add_in_place").map_err(err)?,
            sigmoid: module.load_function("sigmoid_in_place").map_err(err)?,
            multiply: module.load_function("sigmoid_mul").map_err(err)?,
            softplus: module.load_function("softplus_in_place").map_err(err)?,
            reduce: reduction.load_function("packed_reduce_rate").map_err(err)?,
            reduce_bc: reduction.load_function("packed_reduce_bc").map_err(err)?,
        };
        drop(stages);
        let alloc = |len| stream.alloc_zeros::<f32>(len).map_err(err);
        let grad_sizes = [
            ms.w_gate, projection, projection, ms.w_query, ms.w_query, ms.w_gate, ms.w_proj,
        ];
        let weight_grads = grad_sizes
            .iter()
            .map(|&len| alloc(len))
            .collect::<Result<Vec<_>, _>>()?;
        let host_grads = grad_sizes.iter().map(|&len| vec![0.0; len]).collect();
        Ok(Self {
            ctx: ctx.clone(),
            kernels,
            chunk,
            lanes,
            d,
            s,
            k,
            cap,
            offsets: vec![0; lanes],
            lengths: vec![0; lanes],
            carries: vec![0.0; carries],
            offsets_dev: stream.alloc_zeros(lanes).map_err(err)?,
            lengths_dev: stream.alloc_zeros(lanes).map_err(err)?,
            x: alloc(ms.x)?,
            delta: alloc(ms.x)?,
            raw: alloc(ms.x)?,
            b: alloc(product(n, s, "packed B")?)?,
            c: alloc(product(n, s, "packed C")?)?,
            rates: alloc(hs)?,
            deriv: alloc(hs)?,
            a: alloc(maps)?,
            bb: alloc(maps)?,
            scan_a: alloc(maps)?,
            scan_b: alloc(maps)?,
            rev_a: alloc(maps)?,
            rev_b: alloc(maps)?,
            initial: alloc(carries)?,
            carry: alloc(carries)?,
            states: alloc(states)?,
            y: alloc(ms.x)?,
            qe: alloc(ms.query)?,
            qs: alloc(ms.query)?,
            qp: alloc(ms.query)?,
            qn: alloc(n)?,
            weights: alloc(ms.weights)?,
            mv: alloc(ms.value)?,
            gate: alloc(ms.x)?,
            mp: alloc(ms.x)?,
            inj: alloc(ms.x)?,
            keys: alloc(ms.keys)?,
            norms: alloc(cap)?,
            values: alloc(ms.values)?,
            wd: alloc(ms.w_gate)?,
            wb: alloc(projection)?,
            wc: alloc(projection)?,
            wqx: alloc(ms.w_query)?,
            wqh: alloc(ms.w_query)?,
            wg: alloc(ms.w_gate)?,
            wp: alloc(ms.w_proj)?,
            gz: alloc(ms.x)?,
            gmp: alloc(ms.x)?,
            gg: alloc(ms.x)?,
            gmv: alloc(ms.x)?,
            gqp: alloc(ms.query)?,
            gqe: alloc(ms.query)?,
            gx: alloc(ms.x)?,
            gy: alloc(ms.x)?,
            tmp: alloc(ms.x)?,
            gd: alloc(ms.x)?,
            gb: alloc(product(n, s, "packed gB")?)?,
            gc: alloc(product(n, s, "packed gC")?)?,
            reduced: alloc(carries)?,
            weight_grads,
            host_grads,
            host_gx: vec![0.0; ms.x],
            host_ga: vec![0.0; carries],
            ready: false,
        })
    }

    pub fn forward(
        &mut self,
        m: &mut PSSALayerV2,
        n: usize,
        dirty_carry: bool,
    ) -> Result<(), String> {
        self.ready = false;
        validate_layout(&self.offsets, &self.lengths, self.chunk, n, self.d, self.s)?;
        validate_model_buffers(m, n)?;
        let _trace = crate::training_diagnostics::StageTrace::cuda("packed.forward", n);
        let _dispatch = self
            .ctx
            .workspace
            .lock()
            .map_err(|_| "CUDA workspace lock poisoned")?;
        let (d, s, k, hs) = (self.d, self.s, self.k, self.d * self.s);
        let (nr, dm, ds, stride, chunk) = (
            elems(n, "packed tokens")?,
            d as u32,
            s as u32,
            hs as u32,
            self.chunk as u32,
        );
        let (xlen, qlen) = (
            elems(n * d, "packed latent")?,
            elems(n * k, "packed query")?,
        );
        let scan_launch = self.lane_launch(hs)?;
        let mat_launch = self.lane_launch(self.chunk * d)?;
        let stream = &self.ctx.stream;
        let upload = crate::training_diagnostics::StageTrace::cuda("packed.upload", n);
        stream
            .memcpy_htod(&self.offsets, &mut self.offsets_dev)
            .map_err(err)?;
        stream
            .memcpy_htod(&self.lengths, &mut self.lengths_dev)
            .map_err(err)?;
        if dirty_carry {
            stream
                .memcpy_htod(&self.carries, &mut self.carry)
                .map_err(err)?;
        }
        stream
            .memcpy_dtod(&self.carry, &mut self.initial)
            .map_err(err)?;
        stream
            .memcpy_htod(&m.tape.x_norm[..n * d], &mut self.x)
            .map_err(err)?;
        stream
            .memcpy_htod(&m.ssm_rates, &mut self.rates)
            .map_err(err)?;
        stream
            .memcpy_htod(&m.ssm_rate_derivatives, &mut self.deriv)
            .map_err(err)?;
        for (host, device) in [
            (&m.w_delta.data, &mut self.wd),
            (&m.w_b.data, &mut self.wb),
            (&m.w_c.data, &mut self.wc),
            (&m.w_qx.data, &mut self.wqx),
            (&m.w_qh.data, &mut self.wqh),
            (&m.w_gate.data, &mut self.wg),
            (&m.w_proj.data, &mut self.wp),
        ] {
            // Scoped CUDA Adam already owns current device weights. Snapshot
            // those without bouncing their host mirrors back over PCIe.
            let safeguards = self.ctx.safeguards.lock().map_err(err)?;
            if let Some(resident) = safeguards.weight(host) {
                stream.memcpy_dtod(resident, device).map_err(err)?;
            } else {
                stream.memcpy_htod(host, device).map_err(err)?;
            }
        }
        // Only occupied bank rows are read by retrieval. Unused capacity need
        // not be copied; each forward overwrites its active weight rows.
        if m.memory.count != 0 {
            stream
                .memcpy_htod(&m.memory.keys[..m.memory.count * k], &mut self.keys)
                .map_err(err)?;
            stream
                .memcpy_htod(&m.memory.norm_sq[..m.memory.count], &mut self.norms)
                .map_err(err)?;
            stream
                .memcpy_htod(&m.memory.values[..m.memory.count * d], &mut self.values)
                .map_err(err)?;
        }
        drop(upload);
        self.ctx
            .packed_gemm(&self.x, &self.wd, n, d, d, false, &mut self.raw)?;
        stream
            .memcpy_dtod(&self.raw.slice(..n * d), &mut self.delta.slice_mut(..n * d))
            .map_err(err)?;
        self.ctx
            .packed_gemm(&self.x, &self.wb, n, d, s, false, &mut self.b)?;
        self.ctx
            .packed_gemm(&self.x, &self.wc, n, d, s, false, &mut self.c)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.softplus)
                .arg(&mut self.delta)
                .arg(&xlen)
                .launch(packed_launch(xlen))
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.prepare)
                .arg(&self.delta)
                .arg(&self.b)
                .arg(&self.x)
                .arg(&self.rates)
                .arg(&mut self.a)
                .arg(&mut self.bb)
                .arg(&mut self.rev_a)
                .arg(&nr)
                .arg(&dm)
                .arg(&ds)
                .launch(packed_launch(elems(n * hs, "packed maps")?))
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.scan)
                .arg(&self.a)
                .arg(&self.rev_a)
                .arg(&mut self.scan_a)
                .arg(&mut self.scan_b)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&stride)
                .launch(scan_launch)
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.materialize)
                .arg(&self.initial)
                .arg(&self.a)
                .arg(&self.bb)
                .arg(&self.x)
                .arg(&self.scan_a)
                .arg(&self.scan_b)
                .arg(&self.c)
                .arg(&mut self.states)
                .arg(&mut self.y)
                .arg(&nr)
                .arg(&dm)
                .arg(&ds)
                .arg(&mut self.carry)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&chunk)
                .launch(mat_launch)
                .map_err(err)?;
        }
        self.ctx
            .packed_gemm(&self.x, &self.wqx, n, d, k, false, &mut self.qe)?;
        self.ctx
            .packed_gemm(&self.y, &self.wqh, n, d, k, false, &mut self.qs)?;
        self.ctx
            .packed_add(&self.kernels.add, &mut self.qe, &self.qs, qlen)?;
        let (count, cap, dk, dv) = (m.memory.count as u32, self.cap as u32, k as u32, 0u32);
        let tau = m.cfg.tau_mem;
        unsafe {
            stream
                .launch_builder(&self.kernels.memory)
                .arg(&self.qe)
                .arg(&self.keys)
                .arg(&self.norms)
                .arg(&self.values)
                .arg(&mut self.qp)
                .arg(&mut self.qn)
                .arg(&mut self.mv)
                .arg(&mut self.weights)
                .arg(&nr)
                .arg(&count)
                .arg(&cap)
                .arg(&dk)
                .arg(&dv)
                .arg(&tau)
                .launch(packed_row_launch(nr))
                .map_err(err)?;
        }
        if count == 0 {
            stream
                .memset_zeros(&mut self.mv.slice_mut(..n * d))
                .map_err(err)?;
        } else {
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: d as i32,
                n: n as i32,
                k: count as i32,
                alpha: 1.0,
                beta: 0.0,
                lda: d as i32,
                ldb: self.cap as i32,
                ldc: d as i32,
            };
            unsafe {
                self.ctx.blas.gemm(
                    cfg,
                    &self.values.slice(..m.memory.count * d),
                    &self.weights.slice(..n * self.cap),
                    &mut self.mv.slice_mut(..n * d),
                )
            }
            .map_err(err)?;
        }
        self.ctx
            .packed_gemm(&self.x, &self.wg, n, d, d, false, &mut self.gate)?;
        self.ctx
            .packed_gemm(&self.mv, &self.wp, n, d, d, false, &mut self.mp)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.sigmoid)
                .arg(&mut self.gate)
                .arg(&xlen)
                .launch(packed_launch(xlen))
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.multiply)
                .arg(&self.gate)
                .arg(&self.mp)
                .arg(&mut self.inj)
                .arg(&xlen)
                .launch(packed_launch(xlen))
                .map_err(err)?;
        }
        let _readback =
            crate::training_diagnostics::StageTrace::cuda("packed.forward.readback_wait", n);
        // Only CPU consumers cross this boundary: residual assembly, terminal
        // memory insertion, and the public host carry API. No backward tapes.
        stream
            .memcpy_dtoh(&self.y.slice(..n * d), &mut m.block.tape.y_ssm[..n * d])
            .map_err(err)?;
        stream
            .memcpy_dtoh(&self.inj.slice(..n * d), &mut m.block.tape.m_inj[..n * d])
            .map_err(err)?;
        stream
            .memcpy_dtoh(
                &self.qp.slice(..n * k),
                &mut m.block.tape.q_poincare[..n * k],
            )
            .map_err(err)?;
        stream
            .memcpy_dtoh(&self.carry, &mut self.carries)
            .map_err(err)?;
        stream.synchronize().map_err(err)?;
        self.ready = true;
        Ok(())
    }

    fn lane_launch(&self, elements: usize) -> Result<LaunchConfig, String> {
        let mut launch = packed_launch(elems(elements, "packed lane launch")?);
        launch.grid_dim.1 = self.lanes as u32;
        Ok(launch)
    }

    pub fn backward(&mut self, m: &mut PSSALayerV2, n: usize) -> Result<(), String> {
        if !self.ready {
            return Err("CUDA packed backward has no resident forward".into());
        }
        self.ready = false;
        let _trace = crate::training_diagnostics::StageTrace::cuda("packed.backward", n);
        let _dispatch = self
            .ctx
            .workspace
            .lock()
            .map_err(|_| "CUDA workspace lock poisoned")?;
        let resident_gradients = self
            .ctx
            .seed_packed_gradients(&packed_gradients(m), &mut self.weight_grads)?;
        let beta = if resident_gradients { 1.0 } else { 0.0 };
        let (d, s, k, hs) = (self.d, self.s, self.k, self.d * self.s);
        let (nr, dm, ds, stride, chunk, scale) = (
            n as u32,
            d as u32,
            s as u32,
            hs as u32,
            self.chunk as u32,
            1.0 / (s as f32).sqrt(),
        );
        let xlen = elems(n * d, "packed backward latent")?;
        let maps_launch = self.lane_launch(self.chunk * hs)?;
        let scan_launch = self.lane_launch(hs)?;
        let local_launch = self.lane_launch(self.chunk * d)?;
        let bc_launch = self.lane_launch(self.chunk * s)?;
        let stream = &self.ctx.stream;
        stream
            .memcpy_htod(&m.bwd_g_zraw[..n * d], &mut self.gz)
            .map_err(err)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.memory_local)
                .arg(&self.gz)
                .arg(&self.gate)
                .arg(&self.mp)
                .arg(&mut self.gmp)
                .arg(&mut self.gg)
                .arg(&xlen)
                .launch(packed_launch(xlen))
                .map_err(err)?;
        }
        self.ctx
            .packed_gemm(&self.gg, &self.wg, n, d, d, true, &mut self.gx)?;
        self.ctx
            .packed_gemm(&self.gmp, &self.wp, n, d, d, true, &mut self.gmv)?;
        self.ctx
            .packed_weight_grad(&self.gg, &self.x, n, d, d, beta, &mut self.weight_grads[5])?;
        self.ctx
            .packed_weight_grad(&self.gmp, &self.mv, n, d, d, beta, &mut self.weight_grads[6])?;
        let (count, cap, dk, dv, tau) = (
            m.memory.count as u32,
            self.cap as u32,
            k as u32,
            d as u32,
            m.cfg.tau_mem,
        );
        unsafe {
            stream
                .launch_builder(&self.kernels.retrieval)
                .arg(&self.qp)
                .arg(&self.qe)
                .arg(&self.gmv)
                .arg(&self.mv)
                .arg(&self.weights)
                .arg(&self.keys)
                .arg(&self.norms)
                .arg(&self.values)
                .arg(&mut self.gqp)
                .arg(&mut self.gqe)
                .arg(&nr)
                .arg(&count)
                .arg(&cap)
                .arg(&dk)
                .arg(&dv)
                .arg(&tau)
                .launch(packed_row_launch(nr))
                .map_err(err)?;
        }
        self.ctx
            .packed_weight_grad(&self.gqe, &self.x, n, k, d, beta, &mut self.weight_grads[3])?;
        self.ctx
            .packed_weight_grad(&self.gqe, &self.y, n, k, d, beta, &mut self.weight_grads[4])?;
        self.ctx
            .packed_gemm(&self.gqe, &self.wqx, n, k, d, true, &mut self.tmp)?;
        self.ctx
            .packed_add(&self.kernels.add, &mut self.gx, &self.tmp, xlen)?;
        self.ctx
            .packed_gemm(&self.gqe, &self.wqh, n, k, d, true, &mut self.gy)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.maps)
                .arg(&self.a)
                .arg(&self.c)
                .arg(&self.gz)
                .arg(&self.gy)
                .arg(&mut self.rev_a)
                .arg(&mut self.rev_b)
                .arg(&nr)
                .arg(&dm)
                .arg(&ds)
                .arg(&scale)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .launch(maps_launch)
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.scan)
                .arg(&self.rev_a)
                .arg(&self.rev_b)
                .arg(&mut self.scan_a)
                .arg(&mut self.scan_b)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&stride)
                .launch(scan_launch)
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.local)
                .arg(&self.delta)
                .arg(&self.raw)
                .arg(&self.b)
                .arg(&self.c)
                .arg(&self.rates)
                .arg(&self.deriv)
                .arg(&self.x)
                .arg(&self.states)
                .arg(&self.a)
                .arg(&self.bb)
                .arg(&self.scan_b)
                .arg(&self.gz)
                .arg(&self.gy)
                .arg(&mut self.gd)
                // Reverse maps are dead after scan; reuse them for token-local
                // B/C contributions, then reduce in the CPU latent order.
                .arg(&mut self.rev_a)
                .arg(&mut self.rev_b)
                // Exclusive A prefixes are dead after the reverse scan;
                // reuse that full map for token-local rate gradients.
                .arg(&mut self.scan_a)
                .arg(&mut self.tmp)
                .arg(&nr)
                .arg(&dm)
                .arg(&ds)
                .arg(&stride)
                .arg(&scale)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&chunk)
                .launch(local_launch)
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.reduce_bc)
                .arg(&self.rev_a)
                .arg(&self.rev_b)
                .arg(&mut self.gb)
                .arg(&mut self.gc)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&dm)
                .arg(&ds)
                .launch(bc_launch)
                .map_err(err)?;
            stream
                .launch_builder(&self.kernels.reduce)
                .arg(&self.scan_a)
                .arg(&mut self.reduced)
                .arg(&self.offsets_dev)
                .arg(&self.lengths_dev)
                .arg(&stride)
                .launch(scan_launch)
                .map_err(err)?;
        }
        self.ctx
            .packed_add(&self.kernels.add, &mut self.gx, &self.tmp, xlen)?;
        for (g, w, rows) in [
            (&self.gd, &self.wd, d),
            (&self.gb, &self.wb, s),
            (&self.gc, &self.wc, s),
        ] {
            // Input adjoints are added in exactly delta/B/C order.
            self.ctx
                .packed_gemm(g, w, n, rows, d, true, &mut self.tmp)?;
            self.ctx
                .packed_add(&self.kernels.add, &mut self.gx, &self.tmp, xlen)?;
        }
        self.ctx
            .packed_weight_grad(&self.gd, &self.x, n, d, d, beta, &mut self.weight_grads[0])?;
        self.ctx
            .packed_weight_grad(&self.gb, &self.x, n, s, d, beta, &mut self.weight_grads[1])?;
        self.ctx
            .packed_weight_grad(&self.gc, &self.x, n, s, d, beta, &mut self.weight_grads[2])?;
        let _readback =
            crate::training_diagnostics::StageTrace::cuda("packed.backward.readback_wait", n);
        if !resident_gradients {
            for (device, host) in self.weight_grads.iter().zip(&mut self.host_grads) {
                stream.memcpy_dtoh(device, host).map_err(err)?;
            }
        }
        stream
            .memcpy_dtoh(&self.gx.slice(..n * d), &mut self.host_gx[..n * d])
            .map_err(err)?;
        stream
            .memcpy_dtoh(&self.reduced, &mut self.host_ga)
            .map_err(err)?;
        stream.synchronize().map_err(err)?;
        // Transactional publication: no parameter gradient is modified until
        // all GPU computations and copies have succeeded. Scoped training swaps
        // private allocations into the optimizer, without a PCIe round-trip.
        if resident_gradients {
            self.ctx
                .publish_packed_gradients(&packed_gradients(m), &mut self.weight_grads)?;
        }
        let block = &mut m.block;
        if !resident_gradients {
            for (dst, src) in [
                &mut block.w_delta.grad,
                &mut block.w_b.grad,
                &mut block.w_c.grad,
                &mut block.w_qx.grad,
                &mut block.w_qh.grad,
                &mut block.w_gate.grad,
                &mut block.w_proj.grad,
            ]
            .into_iter()
            .zip(&self.host_grads)
            {
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst += src;
                }
            }
        }
        for lane in 0..self.lanes {
            for (dst, src) in block
                .a_mat
                .grad
                .iter_mut()
                .zip(&self.host_ga[lane * hs..(lane + 1) * hs])
            {
                *dst += src;
            }
        }
        for (dst, src) in block.bwd_g_xnorm[..n * d].iter_mut().zip(&self.host_gx) {
            *dst += src;
        }
        Ok(())
    }
}

impl CudaContext {
    // Forward X*W^T or input adjoint G*W. Always beta=0, exact active views;
    // no host boundary, allocation or implicit CPU fallback.
    #[allow(clippy::too_many_arguments)]
    fn packed_gemm(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        m: usize,
        k: usize,
        n: usize,
        adjoint: bool,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let cfg = GemmConfig {
            transa: if adjoint {
                cublasOperation_t::CUBLAS_OP_N
            } else {
                cublasOperation_t::CUBLAS_OP_T
            },
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            beta: 0.0,
            lda: if adjoint { n as i32 } else { k as i32 },
            ldb: k as i32,
            ldc: n as i32,
        };
        unsafe {
            self.blas.gemm(
                cfg,
                &w.slice(..k * n),
                &a.slice(..m * k),
                &mut out.slice_mut(..m * n),
            )
        }
        .map_err(err)
    }
    #[allow(clippy::too_many_arguments)]
    fn packed_weight_grad(
        &self,
        g: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        m: usize,
        k: usize,
        n: usize,
        beta: f32,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: k as i32,
            k: m as i32,
            alpha: 1.0,
            beta,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        unsafe {
            self.blas.gemm(
                cfg,
                &x.slice(..m * n),
                &g.slice(..m * k),
                &mut out.slice_mut(..k * n),
            )
        }
        .map_err(err)
    }
    fn packed_add(
        &self,
        kernel: &CudaFunction,
        out: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        len: u32,
    ) -> Result<(), String> {
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(out)
                .arg(src)
                .arg(&len)
                .launch(packed_launch(len))
        }
        .map_err(err)?;
        Ok(())
    }
}

fn packed_gradients(m: &PSSALayerV2) -> [&[f32]; 7] {
    [
        &m.block.w_delta.grad,
        &m.block.w_b.grad,
        &m.block.w_c.grad,
        &m.block.w_qx.grad,
        &m.block.w_qh.grad,
        &m.block.w_gate.grad,
        &m.block.w_proj.grad,
    ]
}

/// Public model fields may be edited between steps. Revalidate every operand
/// before copies or slicing, rather than trusting the workspace's initial shape.
fn validate_model_buffers(m: &PSSALayerV2, n: usize) -> Result<(), String> {
    let (d, s, k, cap) = (
        m.cfg.d_latent,
        m.cfg.d_state,
        m.cfg.d_mem_key,
        m.cfg.mem_capacity,
    );
    let ms = MemoryShape::new(n, d, k, d, cap, m.memory.count, m.cfg.tau_mem)?;
    let projection = product(s, d, "packed projection weights")?;
    let hs = product(d, s, "packed rates")?;
    lengths(
        "packed model",
        &[
            ("rates", m.ssm_rates.len(), hs),
            ("rate derivatives", m.ssm_rate_derivatives.len(), hs),
            ("rate gradient", m.a_mat.grad.len(), hs),
            ("keys", m.memory.keys.len(), ms.keys),
            ("norms", m.memory.norm_sq.len(), cap),
            ("values", m.memory.values.len(), ms.values),
        ],
    )?;
    for (parameter, expected) in [
        (&m.w_delta, ms.w_gate),
        (&m.w_b, projection),
        (&m.w_c, projection),
        (&m.w_qx, ms.w_query),
        (&m.w_qh, ms.w_query),
        (&m.w_gate, ms.w_gate),
        (&m.w_proj, ms.w_proj),
    ] {
        lengths(
            "packed parameter",
            &[
                ("weight", parameter.data.len(), expected),
                ("gradient", parameter.grad.len(), expected),
            ],
        )?;
    }
    Ok(())
}

/// Validate lane ownership before any unsafe launch. Offsets may be reordered,
/// but active intervals must form an exact, disjoint partition of packed rows.
fn validate_layout(
    offsets: &[u32],
    lengths: &[u32],
    chunk: usize,
    n: usize,
    d: usize,
    s: usize,
) -> Result<(), String> {
    if offsets.len() != lengths.len()
        || offsets.is_empty()
        || offsets.len() > 65535
        || chunk == 0
        || d == 0
        || s == 0
        || [n, chunk, d, s].into_iter().any(|v| v > i32::MAX as usize)
    {
        return Err("CUDA packed dimensions/grid invalid; refusing launch".into());
    }
    let hs = product(d, s, "packed stride")?;
    product(n, hs, "packed token maps")?;
    product(
        product(
            offsets.len(),
            chunk
                .checked_add(1)
                .ok_or("CUDA packed terminal overflow; refusing launch")?,
            "packed lane states",
        )?,
        hs,
        "packed state elements",
    )?;
    let mut total = 0usize;
    for (lane, (&off, &len)) in offsets.iter().zip(lengths).enumerate() {
        if len == 0 {
            continue;
        }
        let (off, len) = (off as usize, len as usize);
        if len > chunk || off.checked_add(len).is_none_or(|end| end > n) {
            return Err("CUDA packed lane range exceeds tape; refusing launch".into());
        }
        for (&other_off, &other_len) in offsets[..lane].iter().zip(&lengths[..lane]) {
            if other_len != 0
                && off < other_off as usize + other_len as usize
                && (other_off as usize) < off + len
            {
                return Err("CUDA packed lanes overlap; refusing launch".into());
            }
        }
        total += len;
    }
    if total != n {
        return Err("CUDA packed lanes leave unowned rows; refusing launch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{packed_launch, packed_row_launch, validate_layout, validate_model_buffers};
    use crate::pssa::{PSSAConfigV2, PSSALayerV2};

    #[test]
    fn packed_launch_covers_exact_active_range_with_register_safe_blocks() {
        for elements in [1, 255, 256, 257, 3584 * 16, u32::MAX] {
            let launch = packed_launch(elements);
            assert_eq!(launch.block_dim, (256, 1, 1));
            assert_eq!(launch.grid_dim.1, 1);
            let capacity = u64::from(launch.grid_dim.0) * 256;
            assert!(capacity >= u64::from(elements));
            assert!(capacity - u64::from(elements) < 256);
        }
        for rows in [1, 32, 64, 2048, i32::MAX as u32] {
            let launch = packed_row_launch(rows);
            assert_eq!(launch.grid_dim, (rows, 1, 1));
            assert_eq!(launch.block_dim, (1, 1, 1));
        }
    }

    #[test]
    fn malformed_bank_or_parameter_is_rejected_before_copy_or_launch() {
        let cfg = PSSAConfigV2 {
            d_vocab: 11,
            d_latent: 7,
            d_state: 3,
            d_mem_key: 4,
            mem_capacity: 4,
            chunk_len: 6,
            ..Default::default()
        };
        let mut m = PSSALayerV2::new(cfg, 7);
        assert!(validate_model_buffers(&m, 5).is_ok());
        m.memory.count = 5;
        assert!(
            validate_model_buffers(&m, 5)
                .unwrap_err()
                .contains("refusing launch")
        );
        m.memory.count = 0;
        m.cfg.tau_mem = f32::NAN;
        assert!(validate_model_buffers(&m, 5).is_err());
        m.cfg.tau_mem = 0.7;
        m.w_delta.data.pop();
        assert!(
            validate_model_buffers(&m, 5)
                .unwrap_err()
                .contains("weight")
        );
        m.w_delta.data.push(0.0);
        m.w_b.grad.pop();
        assert!(
            validate_model_buffers(&m, 5)
                .unwrap_err()
                .contains("gradient")
        );
        m.w_b.grad.push(0.0);
        m.memory.keys.pop();
        assert!(validate_model_buffers(&m, 5).unwrap_err().contains("keys"));
    }

    #[test]
    fn packed_index_limits_include_lane_terminal_rows() {
        let stride = 3584 * 16;
        let state_rows = u32::MAX as usize / stride;
        // A one-lane state tape has one more row than its packed token maps.
        let largest_chunk = state_rows - 1;
        assert!(validate_layout(
            &[0],
            &[largest_chunk as u32],
            largest_chunk,
            largest_chunk,
            3584,
            16,
        ).is_ok());
        assert!(validate_layout(
            &[0],
            &[state_rows as u32],
            state_rows,
            state_rows,
            3584,
            16,
        ).unwrap_err().contains("u32 PTX index"));
        assert!(validate_layout(&[0x8000_0000], &[1], 1, 1, 1, 2).is_err());
        // Per-lane fixed state storage can overflow even with very few tokens.
        assert!(validate_layout(
            &[0, 1], &[1, 1], largest_chunk, 2, 3584, 16,
        ).is_err());
    }

    #[test]
    #[ignore = "requires CUDA; checks swapped packed gradients through accumulation and AdamW"]
    fn cuda_packed_resident_gradient_transactions_match_cpu() {
        use crate::backend::Device;
        use crate::sequence_batch::{Sequence, SequenceBatch};

        fn close(actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len());
            let mut error = 0.0f32;
            let mut scale = 1e-6f32;
            for (&a, &b) in actual.iter().zip(expected) {
                assert!(a.is_finite() && b.is_finite());
                error = error.max((a - b).abs());
                scale = scale.max(a.abs()).max(b.abs());
            }
            assert!(error / scale < 1e-3, "relative error {}", error / scale);
        }

        let cfg = PSSAConfigV2 {
            d_vocab: 11,
            d_latent: 7,
            d_state: 3,
            d_mem_key: 4,
            mem_capacity: 4,
            chunk_len: 6,
            ..Default::default()
        };
        let ctx = crate::cuda::CudaContext::init().expect("CUDA GPU required");
        let mut cpu = PSSALayerV2::new(cfg.clone(), 7);
        let mut gpu = PSSALayerV2::new(cfg, 7);
        gpu.device = Device::Cuda(ctx.clone());
        for model in [&mut cpu, &mut gpu] {
            model.memory.insert(&[0.1, -0.1, 0.05, 0.0], &[0.03; 7]);
        }
        let mut a = SequenceBatch::new(&mut cpu, 3).unwrap();
        let mut b = SequenceBatch::new(&mut gpu, 3).unwrap();
        ctx.begin_safeguarded_training(&gpu.adam_tensors()).unwrap();
        for update in 0..3 {
            cpu.zero_gradients();
            gpu.zero_gradients();
            for micro in 0..2 {
                let sequences = [
                    Sequence {
                        lane: 2,
                        inputs: &[1, 4, 3],
                        targets: &[4, 3, 2],
                        reset: update == 0 && micro == 0,
                    },
                    Sequence {
                        lane: 0,
                        inputs: &[5, 6],
                        targets: &[6, 7],
                        reset: update == 1 && micro == 0,
                    },
                ];
                let loss_a = a.forward(&mut cpu, &sequences).unwrap();
                let loss_b = b.forward(&mut gpu, &sequences).unwrap();
                close(&[loss_b], &[loss_a]);
                let scale = if micro == 0 { 0.4 } else { 0.6 };
                a.backward(&mut cpu, scale).unwrap();
                b.backward(&mut gpu, scale).unwrap();
                assert!(b.last_step_used_cuda(), "packed CUDA fell back");
                for lane in 0..3 {
                    close(b.state(lane), a.state(lane));
                }
            }
            cpu.apply_adamw_with_grad_clip(1e-3, 1.0);
            gpu.apply_adamw_with_grad_clip(1e-3, 1.0);
        }
        ctx.finish_safeguarded_training(&mut gpu.adam_tensors()).unwrap();
        for (actual, expected) in gpu.adam_tensors().iter().zip(cpu.adam_tensors()) {
            close(actual.data, expected.data);
            close(actual.grad, expected.grad);
            close(actual.m, expected.m);
            close(actual.v, expected.v);
        }
    }

    #[test]
    fn lane_ownership_rejects_overlap_holes_overflow_and_invalid_grid() {
        assert!(validate_layout(&[3, 0, 0], &[2, 3, 0], 3, 5, 7, 2).is_ok());
        for (o, l) in [
            (vec![0, 1], vec![2, 2]),
            (vec![0, 3], vec![2, 1]),
            (vec![0, 4], vec![2, 2]),
            (vec![0, 2], vec![4, 1]),
        ] {
            assert!(
                validate_layout(&o, &l, 3, 4, 7, 2)
                    .unwrap_err()
                    .contains("refusing launch")
            );
        }
        assert!(validate_layout(&[0], &[1], usize::MAX, 1, 7, 2).is_err());
        assert!(validate_layout(&[0], &[1], 1, 1, usize::MAX, 2).is_err());
        assert!(validate_layout(&vec![0; 65536], &vec![0; 65536], 1, 1, 1, 1).is_err());
        assert!(validate_layout(&[0], &[1], 1, 1, 65536, 65536).is_err());
    }
}
