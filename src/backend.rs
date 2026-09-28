use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Validate all element and byte counts before allocation or backend calls.
/// Shared by CUDA, WebGPU and the allocation-free CPU kernel.
pub(crate) fn checked_gemm_sizes(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    x_len: usize,
    w_len: usize,
) -> Result<(usize, usize, usize), String> {
    if [m, n, k, batch].contains(&0) {
        return Err("GEMM dimensions must be positive".into());
    }
    let product = |dims: &[usize]| -> Result<usize, String> {
        let len = dims
            .iter()
            .try_fold(1usize, |v, &d| v.checked_mul(d))
            .ok_or_else(|| "GEMM element count overflow".to_string())?;
        checked_f32_bytes(len)?;
        Ok(len)
    };
    let input = product(&[batch, m, k])?;
    let weights = product(&[n, k])?;
    let output = product(&[batch, m, n])?;
    if x_len != input || w_len != weights {
        return Err(format!(
            "GEMM operand lengths must be {input} and {weights}, got {x_len} and {w_len}"
        ));
    }
    Ok((input, weights, output))
}

/// All forward GEMMs share W across batches. Flattening [B,M,K] to [B*M,K]
/// preserves row-major bytes and gives tiled WebGPU/cuBLAS a real matrix, not
/// B skinny products. Keep this at the backend boundary as well as in training.
pub(crate) fn shared_gemm_rows(m: usize, batch: usize) -> Result<usize, String> {
    m.checked_mul(batch)
        .filter(|&rows| rows > 0)
        .ok_or_else(|| "GEMM row count must be positive and fit usize".into())
}

pub(crate) fn checked_f32_bytes(len: usize) -> Result<u64, String> {
    len.checked_mul(std::mem::size_of::<f32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .map(|bytes| bytes as u64)
        .ok_or_else(|| "GEMM buffer byte size overflow".to_string())
}

pub(crate) fn zeroed_output(len: usize) -> Result<Vec<f32>, String> {
    checked_f32_bytes(len)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|e| format!("GEMM host allocation failed: {e}"))?;
    output.resize(len, 0.0);
    Ok(output)
}

// =============================================================================
// COMPLETE EMBEDDED WGSL COMPUTE SHADERS
// =============================================================================

pub const WGSL_COMPUTE_KERNELS: &str = r#"
// -----------------------------------------------------------------------------
// 1. Tiled Parallel GEMM: Y = X * W^T (Batch & Sequence Aware)
// -----------------------------------------------------------------------------
struct GemmUniforms {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
};

@group(0) @binding(0) var<uniform> gemm_cfg: GemmUniforms;
@group(0) @binding(1) var<storage, read> X_buf: array<f32>;
@group(0) @binding(2) var<storage, read> W_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y_buf: array<f32>;

var<workgroup> tile_x: array<array<f32, 16>, 16>;
var<workgroup> tile_w: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn gemm_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;

    var acc: f32 = 0.0;
    let num_tiles = gemm_cfg.K / 16u + select(0u, 1u, gemm_cfg.K % 16u != 0u);

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let x_k = t * 16u + local_id.x;
        let w_k = t * 16u + local_id.y;

        if (row < gemm_cfg.M && x_k < gemm_cfg.K) {
            let x_idx = (batch * gemm_cfg.M * gemm_cfg.K) + (row * gemm_cfg.K) + x_k;
            tile_x[local_id.y][local_id.x] = X_buf[x_idx];
        } else {
            tile_x[local_id.y][local_id.x] = 0.0;
        }

        if (col < gemm_cfg.N && w_k < gemm_cfg.K) {
            let w_idx = (col * gemm_cfg.K) + w_k;
            tile_w[local_id.y][local_id.x] = W_buf[w_idx];
        } else {
            tile_w[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + tile_x[local_id.y][k] * tile_w[k][local_id.x];
        }

        workgroupBarrier();
    }

    if (row < gemm_cfg.M && col < gemm_cfg.N) {
        let y_idx = (batch * gemm_cfg.M * gemm_cfg.N) + (row * gemm_cfg.N) + col;
        Y_buf[y_idx] = acc;
    }
}

// -----------------------------------------------------------------------------
// 2. Parallel Affine RMSNorm: out = gamma * (x / RMS(x)) + beta
// -----------------------------------------------------------------------------
struct NormUniforms {
    dim: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> norm_cfg: NormUniforms;
@group(0) @binding(1) var<storage, read> raw_x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> norm_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn affine_rmsnorm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim = norm_cfg.dim;
    let base_off = token_idx * dim;

    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let val = raw_x[base_off + i];
        sum_sq = sum_sq + val * val;
    }

    let inv_rms = 1.0 / sqrt(sum_sq / f32(dim) + norm_cfg.eps);
    let feat_idx = global_id.x;

    if (feat_idx < dim) {
        let v = raw_x[base_off + feat_idx];
        norm_out[base_off + feat_idx] = gamma[feat_idx] * (v * inv_rms) + beta[feat_idx];
    }
}

// -----------------------------------------------------------------------------
// 3. Fused SiLU Non-Linearity & Residual Addition
// -----------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> act_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> act_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn silu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&act_in)) {
        let v = act_in[idx];
        let sig = 1.0 / (1.0 + exp(-v));
        act_out[idx] = v * sig;
    }
}

// -----------------------------------------------------------------------------
// 4. In-Place VRAM-Native AdamW Optimizer Kernel
// -----------------------------------------------------------------------------
struct AdamWUniforms {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
    step: f32,
    len: u32,
};

@group(0) @binding(0) var<uniform> adam_cfg: AdamWUniforms;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> m_moments: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_moments: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn adamw_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= adam_cfg.len) {
        return;
    }

    let g = grads[idx];
    var p = params[idx];

    // Decoupled Weight Decay
    if (adam_cfg.weight_decay > 0.0) {
        p = p - adam_cfg.lr * adam_cfg.weight_decay * p;
    }

    let m_new = adam_cfg.beta1 * m_moments[idx] + (1.0 - adam_cfg.beta1) * g;
    let v_new = adam_cfg.beta2 * v_moments[idx] + (1.0 - adam_cfg.beta2) * g * g;

    m_moments[idx] = m_new;
    v_moments[idx] = v_new;

    let bias1 = 1.0 - pow(adam_cfg.beta1, adam_cfg.step);
    let bias2 = 1.0 - pow(adam_cfg.beta2, adam_cfg.step);

    let m_hat = m_new / bias1;
    let v_hat = v_new / bias2;

    params[idx] = p - adam_cfg.lr * m_hat / (sqrt(v_hat) + adam_cfg.eps);
}
"#;

// =============================================================================
// DEVICE CONTEXT & GPU PIPELINE CONTROLLER
// =============================================================================

#[derive(Default)]
struct WgpuWorkspace {
    input: Option<wgpu::Buffer>,
    output: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,
    uniforms: Option<wgpu::Buffer>,
}

fn reserve_wgpu_buffer(
    device: &wgpu::Device,
    slot: &mut Option<wgpu::Buffer>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) {
    if slot.as_ref().is_none_or(|buffer| buffer.size() < size) {
        *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }));
    }
}

fn begin_error_scopes(device: &wgpu::Device) {
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
}

fn finish_error_scopes(device: &wgpu::Device) -> Result<(), String> {
    let validation = pollster::block_on(device.pop_error_scope());
    let allocation = pollster::block_on(device.pop_error_scope());
    match validation.or(allocation) {
        Some(error) => Err(format!("WebGPU operation failed: {error}")),
        None => Ok(()),
    }
}

fn checked_wgpu_sizes(
    limits: &wgpu::Limits,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    x_len: usize,
    w_len: usize,
) -> Result<(usize, usize, usize), String> {
    let sizes = checked_gemm_sizes(m, n, k, batch, x_len, w_len)?;
    let m = shared_gemm_rows(m, batch)?;
    let batch = 1;
    // WGSL indexing arithmetic is u32, not just the uniform dimensions.
    if [m, n, k, batch, sizes.0, sizes.1, sizes.2]
        .iter()
        .any(|&v| v > u32::MAX as usize)
    {
        return Err("WebGPU GEMM dimensions or element offsets exceed u32".into());
    }
    for elements in [sizes.0, sizes.1, sizes.2] {
        let bytes = checked_f32_bytes(elements)?;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(format!(
                "WebGPU GEMM buffer needs {bytes} bytes, exceeding device buffer/storage-binding limits"
            ));
        }
    }
    let max_groups = limits.max_compute_workgroups_per_dimension as usize;
    if n.div_ceil(16) > max_groups || m.div_ceil(16) > max_groups || batch > max_groups {
        return Err("WebGPU GEMM dispatch exceeds device workgroup-count limit".into());
    }
    if limits.max_compute_workgroup_size_x < 16
        || limits.max_compute_workgroup_size_y < 16
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_storage_size < 2048
    {
        return Err("WebGPU device cannot run the 16x16 tiled GEMM kernel".into());
    }
    Ok(sizes)
}

#[derive(Clone)]
pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub gemm_pipeline: Arc<wgpu::ComputePipeline>,
    pub norm_pipeline: Arc<wgpu::ComputePipeline>,
    pub silu_pipeline: Arc<wgpu::ComputePipeline>,
    pub adamw_pipeline: Arc<wgpu::ComputePipeline>,
    /// Weight matrices resident on the device, keyed by (host pointer, len).
    /// Weights only change when the optimizer steps, so every GEMM between two
    /// steps reuses the uploaded copy instead of re-sending it. Cleared by
    /// `invalidate_weights` right after each AdamW step.
    weight_cache: Arc<Mutex<HashMap<(usize, usize), Arc<wgpu::Buffer>>>>,
    // High-water buffers, not a per-shape cache. Also serializes error scopes
    // and staging-map lifetimes for every clone of this context.
    workspace: Arc<Mutex<WgpuWorkspace>>,
}

impl WgpuContext {
    pub fn init_blocking() -> Result<Self, String> {
        Self::init_with_software_policy(false)
    }

    fn init_with_software_policy(allow_software: bool) -> Result<Self, String> {
        // Library initialization may run after Rayon or other application
        // threads start. Configure XDG_RUNTIME_DIR externally when needed;
        // mutating process-wide environment here is not thread-safe on Unix.
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "No compatible WebGPU compute adapter found.".to_string())?;

        let info = adapter.get_info();
        // A software rasterizer (llvmpipe / lavapipe) reports device_type Cpu.
        // It is slower than our scalar CPU path end to end (buffer upload +
        // dispatch + blocking readback per stage), so refuse it and let the
        // caller fall back to CPU rather than train 12x slower.
        if info.device_type == wgpu::DeviceType::Cpu && !allow_software {
            return Err(format!(
                "software GPU adapter refused ({} / {:?}): slower than cpu",
                info.name, info.backend
            ));
        }
        println!(
            "gpu adapter: {} (backend={:?}, type={:?}, driver={})",
            info.name, info.backend, info.device_type, info.driver
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("PSSA V2 GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create WebGPU device: {}", e))?;

        begin_error_scopes(&device);
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PSSA V2 Compute Shaders"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_COMPUTE_KERNELS)),
        });

        let gemm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMM Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "gemm_main",
        });

        let norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "affine_rmsnorm_main",
        });

        let silu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SiLU Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "silu_main",
        });

        let adamw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AdamW Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "adamw_main",
        });

        finish_error_scopes(&device)?;
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            gemm_pipeline: Arc::new(gemm_pipeline),
            norm_pipeline: Arc::new(norm_pipeline),
            silu_pipeline: Arc::new(silu_pipeline),
            adamw_pipeline: Arc::new(adamw_pipeline),
            weight_cache: Arc::new(Mutex::new(HashMap::new())),
            workspace: Arc::new(Mutex::new(WgpuWorkspace::default())),
        })
    }

    /// Upload a weight matrix once and keep it on the device until the next
    /// optimizer step. Without this every GEMM re-sends the full matrix over
    /// PCIe, which for small per-stage batches costs more than the arithmetic.
    pub fn weight_buffer(&self, w: &[f32]) -> Arc<wgpu::Buffer> {
        let key = (w.as_ptr() as usize, w.len());
        let mut cache = self.weight_cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }
        let buf = Arc::new(self.create_buffer_init("gemm_w_resident", w, true));
        cache.insert(key, buf.clone());
        buf
    }

    /// Drop every resident weight copy. Must be called whenever host-side
    /// weights change (i.e. straight after an AdamW step), or the GPU would
    /// keep multiplying by stale parameters.
    pub fn invalidate_weights(&self) {
        self.weight_cache.lock().unwrap().clear();
    }

    pub fn create_buffer_init(&self, label: &str, data: &[f32], read_only: bool) -> wgpu::Buffer {
        let usage = if read_only {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        };

        // Do not map an allocation before its error scope can be checked.
        // DeviceExt::create_buffer_init maps immediately and can panic on an
        // invalid/OOM handle even when buffer creation is inside a scope.
        let contents = bytemuck::cast_slice(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage,
            mapped_at_creation: false,
        });
        if !contents.is_empty() {
            self.queue.write_buffer(&buffer, 0, contents);
        }
        buffer
    }

    pub fn read_buffer_blocking(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>, String> {
        let byte_len = checked_f32_bytes(count)?;
        if byte_len == 0
            || byte_len > buffer.size()
            || byte_len > self.device.limits().max_buffer_size
            || !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC)
        {
            return Err("GPU readback requires a nonempty, in-bounds COPY_SRC buffer".into());
        }
        let _guard = self
            .workspace
            .lock()
            .map_err(|_| "WebGPU workspace lock poisoned")?;
        begin_error_scopes(&self.device);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Staging Buffer"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));
        finish_error_scopes(&self.device)?;
        let mut result = zeroed_output(count)?;
        self.map_readback_into(&staging, &mut result)?;
        Ok(result)
    }

    fn map_readback_into(&self, staging: &wgpu::Buffer, output: &mut [f32]) -> Result<(), String> {
        let byte_len = checked_f32_bytes(output.len())?;
        let slice = staging.slice(..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        self.device.poll(wgpu::Maintain::Wait);
        let mapped_result = receiver
            .recv()
            .map_err(|_| "GPU readback callback was dropped".to_string())?;
        mapped_result.map_err(|e| format!("GPU readback mapping failed: {e:?}"))?;

        let mapped = slice.get_mapped_range();
        output.copy_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        staging.unmap();
        Ok(())
    }

    /// Run the embedded tiled GEMM kernel on the GPU.
    /// X is [batch, M, K] row-major, W is [N, K] row-major, output is [batch, M, N].
    pub fn dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Vec<f32> {
        match self.try_dispatch_gemm(x, w, m, n, k, batch) {
            Ok(y) => y,
            Err(error) => {
                eprintln!("warning: WebGPU GEMM failed; using CPU fallback: {error}");
                crate::backend::gemm_cpu_reference(x, w, m, n, k, batch)
            }
        }
    }

    /// Strict GPU execution: malformed/unsupported requests are errors, never
    /// successful CPU results. Intended for probes and hardware parity tests.
    pub fn try_dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, _, len) =
            checked_wgpu_sizes(&self.device.limits(), m, n, k, batch, x.len(), w.len())?;
        let mut out = zeroed_output(len)?;
        self.try_dispatch_gemm_into(x, w, m, n, k, batch, &mut out)?;
        Ok(out)
    }

    /// Reuses caller output and high-water device scratch; wgpu command objects
    /// and readback synchronization can still allocate host bookkeeping.
    /// The caller's output is untouched on validation failure. Resource and
    /// dispatch validation errors are scoped; native wgpu driver/device-loss
    /// failures in submit/poll may still be fatal in this dependency version.
    pub fn try_dispatch_gemm_into(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let (x_len, _, out_len) =
            checked_wgpu_sizes(&self.device.limits(), m, n, k, batch, x.len(), w.len())?;
        if out.len() != out_len {
            return Err("WebGPU GEMM output length mismatch".into());
        }
        let m = shared_gemm_rows(m, batch)?;
        let batch = 1;
        let x_bytes = checked_f32_bytes(x_len)?;
        let out_bytes = checked_f32_bytes(out_len)?;
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| "WebGPU workspace lock poisoned")?;
        begin_error_scopes(&self.device);
        reserve_wgpu_buffer(
            &self.device,
            &mut workspace.input,
            x_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "gemm_x",
        );
        reserve_wgpu_buffer(
            &self.device,
            &mut workspace.output,
            out_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "gemm_y",
        );
        reserve_wgpu_buffer(
            &self.device,
            &mut workspace.staging,
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "gemm_readback",
        );
        reserve_wgpu_buffer(
            &self.device,
            &mut workspace.uniforms,
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "gemm_uniforms",
        );
        let cfg_buf = workspace.uniforms.as_ref().unwrap();
        let x_buf = workspace.input.as_ref().unwrap();
        let y_buf = workspace.output.as_ref().unwrap();
        let staging = workspace.staging.as_ref().unwrap();
        let cfg: [u32; 4] = [m as u32, n as u32, k as u32, batch as u32];
        self.queue
            .write_buffer(cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.queue.write_buffer(x_buf, 0, bytemuck::cast_slice(x));
        let w_buf = self.weight_buffer(w);

        let layout = self.gemm_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(16) as u32, m.div_ceil(16) as u32, batch as u32);
        }
        // Copy in the same submission rather than a second readback encoder.
        encoder.copy_buffer_to_buffer(y_buf, 0, staging, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));
        if let Err(error) = finish_error_scopes(&self.device) {
            // Failed allocations may have left invalid handles in either cache.
            *workspace = WgpuWorkspace::default();
            self.invalidate_weights();
            return Err(error);
        }
        self.map_readback_into(staging, out)
    }
}

/// Reference CPU GEMM with identical layout, used to verify the GPU kernel.
pub fn gemm_cpu_reference(
    x: &[f32],
    w: &[f32],
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) -> Vec<f32> {
    let Ok((_, _, out_len)) = checked_gemm_sizes(m, n, k, batch, x.len(), w.len()) else {
        return Vec::new();
    };
    let Ok(mut y) = zeroed_output(out_len) else {
        return Vec::new();
    };
    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[b * m * k + i * k + kk] * w[j * k + kk];
                }
                y[b * m * n + i * n + j] = acc;
            }
        }
    }
    y
}

/// Allocation-free, row-major Y = X * W^T. Blocks four input rows so each
/// weight row is reused while hot, and uses coarse, disjoint Rayon output tiles
/// only for large work. Each dot uses the same accumulation order as matvec.
pub fn gemm_cpu_into(
    x: &[f32],
    w: &[f32],
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    out: &mut [f32],
) -> Result<(), String> {
    let (_, _, output) = checked_gemm_sizes(m, n, k, batch, x.len(), w.len())?;
    if out.len() != output {
        return Err("CPU GEMM output length mismatch".into());
    }
    const ROW_TILE: usize = 4;
    const WEIGHT_TILE: usize = 32;
    let tile_len = n.checked_mul(ROW_TILE).unwrap_or(output).min(output);
    let rows_per_tile = tile_len / n;
    let calculate = |(tile, dst): (usize, &mut [f32])| {
        let first_row = tile * rows_per_tile;
        for first_weight in (0..n).step_by(WEIGHT_TILE) {
            for j in first_weight..(first_weight + WEIGHT_TILE).min(n) {
                let weights = &w[j * k..(j + 1) * k];
                for (t, row) in dst.chunks_exact_mut(n).enumerate() {
                    let offset = (first_row + t) * k;
                    row[j] = crate::linalg::dot_slice(&x[offset..offset + k], weights);
                }
            }
        }
    };
    // Avoid waking a large pool for small state/rank projections.
    if output.saturating_mul(k) >= 4 * 1024 * 1024 && output / n >= 8 {
        out.par_chunks_mut(tile_len).enumerate().for_each(calculate);
    } else {
        out.chunks_mut(tile_len).enumerate().for_each(calculate);
    }
    Ok(())
}

/// Row-major C(M,N) = A(M,K) * B(K,N). CPU twin for the backward-pass GEMMs.
pub fn gemm_nn_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let Ok((_, _, c_len)) = checked_gemm_sizes(m, n, k, 1, a.len(), b.len()) else {
        return Vec::new();
    };
    let Ok(mut c) = zeroed_output(c_len) else {
        return Vec::new();
    };
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk];
            if av == 0.0 {
                continue;
            }
            let b_row = &b[kk * n..kk * n + n];
            let c_row = &mut c[i * n..i * n + n];
            for j in 0..n {
                c_row[j] += av * b_row[j];
            }
        }
    }
    c
}

/// Row-major C(K,N) = A(M,K)^T * B(M,N). CPU twin for the weight-gradient GEMMs.
pub fn gemm_tn_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let Ok((_, _, c_len)) = checked_gemm_sizes(k, n, m, 1, a.len(), b.len()) else {
        return Vec::new();
    };
    let Ok(mut c) = zeroed_output(c_len) else {
        return Vec::new();
    };
    for t in 0..m {
        for kk in 0..k {
            let av = a[t * k + kk];
            if av == 0.0 {
                continue;
            }
            let b_row = &b[t * n..t * n + n];
            let c_row = &mut c[kk * n..kk * n + n];
            for j in 0..n {
                c_row[j] += av * b_row[j];
            }
        }
    }
    c
}

// =============================================================================
// HARDWARE-AGNOSTIC DEVICE & TENSOR PRIMITIVES
// =============================================================================

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(WgpuContext),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaContext),
}

/// A GPU context cloned out of a [`Device`], so the batched stage functions can
/// dispatch without caring which backend is underneath. Both variants honour
/// the same row-major layout contract and are checked against
/// [`gemm_cpu_reference`].
#[derive(Clone)]
pub enum GpuDispatch {
    Wgpu(WgpuContext),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaContext),
}

impl GpuDispatch {
    /// Y[b] = X[b] * W^T, X [batch,M,K], W [N,K], Y [batch,M,N].
    pub fn dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Vec<f32> {
        match self {
            GpuDispatch::Wgpu(ctx) => ctx.dispatch_gemm(x, w, m, n, k, batch),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(ctx) => ctx.dispatch_gemm(x, w, m, n, k, batch),
        }
    }

    /// Strict dispatch, intended for hardware probes: an error cannot be
    /// mistaken for a GPU result computed by a silent CPU fallback.
    pub fn try_dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Wgpu(ctx) => ctx.try_dispatch_gemm(x, w, m, n, k, batch),
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.try_dispatch_gemm(x, w, m, n, k, batch),
        }
    }

    /// Strict output-into dispatch, reusing backend workspaces and caller storage.
    pub fn try_dispatch_gemm_into(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        match self {
            Self::Wgpu(ctx) => ctx.try_dispatch_gemm_into(x, w, m, n, k, batch, out),
            #[cfg(feature = "cuda")]
            Self::Cuda(ctx) => ctx.try_dispatch_gemm_into(x, w, m, n, k, batch, out),
        }
    }

    /// Production output-into dispatch. Valid shapes fall back safely on GPU
    /// failure; malformed buffers return Err without modifying the output.
    pub fn dispatch_gemm_into(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let (_, _, expected) = checked_gemm_sizes(m, n, k, batch, x.len(), w.len())?;
        if out.len() != expected {
            return Err("GEMM output length mismatch".into());
        }
        if let Err(error) = self.try_dispatch_gemm_into(x, w, m, n, k, batch, out) {
            eprintln!("warning: GPU GEMM failed; using CPU fallback: {error}");
            gemm_cpu_into(x, w, m, n, k, batch, out)?;
        }
        Ok(())
    }

    /// Drop device-resident weight copies after an optimizer step.
    pub fn invalidate_weights(&self) {
        match self {
            GpuDispatch::Wgpu(ctx) => ctx.invalidate_weights(),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(ctx) => ctx.invalidate_weights(),
        }
    }

    /// Row-major C(M,N) = A(M,K) * B(K,N).
    pub fn gemm_nn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            GpuDispatch::Wgpu(_) => gemm_nn_cpu(a, b, m, k, n),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(ctx) => ctx.gemm_nn(a, b, m, k, n),
        }
    }

    /// Row-major C(K,N) = A(M,K)^T * B(M,N).
    pub fn gemm_tn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            GpuDispatch::Wgpu(_) => gemm_tn_cpu(a, b, m, k, n),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(ctx) => ctx.gemm_tn(a, b, m, k, n),
        }
    }

    /// Whether this backend actually accelerates the backward-pass GEMM shapes.
    /// The WGSL kernel only implements the forward X * W^T layout, so on WebGPU
    /// the backward pass stays on its fused CPU loops instead of paying to
    /// materialize intermediates for a CPU twin.
    pub fn accelerates_backward(&self) -> bool {
        match self {
            GpuDispatch::Wgpu(_) => false,
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(_) => true,
        }
    }

    pub fn backend_label(&self) -> String {
        match self {
            GpuDispatch::Wgpu(_) => "webgpu".to_string(),
            #[cfg(feature = "cuda")]
            GpuDispatch::Cuda(ctx) => format!("cuda ({})", ctx.adapter_name()),
        }
    }
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        self.gpu().is_some()
    }

    /// The dispatch handle for this device, or `None` on CPU.
    pub fn gpu(&self) -> Option<GpuDispatch> {
        match self {
            Device::Cpu => None,
            Device::Gpu(ctx) => Some(GpuDispatch::Wgpu(ctx.clone())),
            #[cfg(feature = "cuda")]
            Device::Cuda(ctx) => Some(GpuDispatch::Cuda(ctx.clone())),
        }
    }

    /// Try to bring up a real GPU compute device. Native CUDA is preferred when
    /// the build has it and a driver is present, because cuBLAS beats the WGSL
    /// kernel; WebGPU is the portable fallback. Returns Err with the reason each
    /// backend gave when neither is usable.
    pub fn try_gpu() -> Result<Device, String> {
        #[cfg(feature = "cuda")]
        let cuda_err = match crate::cuda::CudaContext::init() {
            Ok(ctx) => return Ok(Device::Cuda(ctx)),
            Err(e) => e,
        };

        match WgpuContext::init_blocking() {
            Ok(ctx) => Ok(Device::Gpu(ctx)),
            #[cfg(feature = "cuda")]
            Err(e) => Err(format!("{cuda_err}; {e}")),
            #[cfg(not(feature = "cuda"))]
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn shared_weight_layout_folds_sequences_and_tokens_without_overflow() {
        assert_eq!(shared_gemm_rows(1, 64).unwrap(), 64);
        assert_eq!(shared_gemm_rows(64, 8).unwrap(), 512);
        assert_eq!(shared_gemm_rows(17, 3).unwrap(), 51);
        assert!(shared_gemm_rows(0, 1).is_err());
        assert!(shared_gemm_rows(usize::MAX, 2).is_err());
        // 64 one-row dispatches formerly used 64 row tiles; now just four.
        assert_eq!(shared_gemm_rows(1, 64).unwrap().div_ceil(16), 4);
    }

    #[test]
    fn webgpu_limits_are_checked_without_a_device_or_allocating_operands() {
        let limits = wgpu::Limits::default();
        assert!(checked_wgpu_sizes(&limits, 1, 1, 1, 65535, 65535, 1).is_ok());
        // Even legacy M=1 callers now use full 16-row tiles, not dispatch Z.
        assert!(checked_wgpu_sizes(&limits, 1, 1, 1, 65536, 65536, 1).is_ok());
        let too_many_rows = 65535 * 16 + 1;
        assert!(checked_wgpu_sizes(&limits, 1, 1, 1, too_many_rows, too_many_rows, 1)
            .unwrap_err().contains("workgroup"));
        // The same contiguous chunk can be represented as M=L, batch=1.
        assert!(checked_wgpu_sizes(&limits, 65536, 1, 1, 1, 65536, 1).is_ok());
        let mut small = limits.clone();
        small.max_compute_workgroups_per_dimension = 1;
        for (m, n) in [(17, 1), (1, 17)] {
            assert!(checked_wgpu_sizes(&small, m, n, 1, 1, m, n).is_err());
        }
        small = limits.clone();
        small.max_buffer_size = 4;
        assert!(checked_wgpu_sizes(&small, 2, 1, 1, 1, 2, 1).is_err());
        small = limits.clone();
        small.max_storage_buffer_binding_size = 4;
        assert!(checked_wgpu_sizes(&small, 1, 2, 1, 1, 1, 2).is_err());
        small = limits.clone();
        small.max_compute_invocations_per_workgroup = 128;
        assert!(checked_wgpu_sizes(&small, 1, 1, 1, 1, 1, 1).is_err());
        assert!(checked_wgpu_sizes(&limits, usize::MAX, 2, 2, 2, 0, 0).is_err());
        assert!(checked_wgpu_sizes(&limits, 2, 2, 2, 1, 1, 1).is_err());
    }

    #[test]
    fn webgpu_strict_dispatch_and_error_scopes_when_an_adapter_is_available() {
        // Software Vulkan is useful for API/validation coverage, not evidence of
        // hardware acceleration. Production initialization still refuses it.
        let ctx = match WgpuContext::init_with_software_policy(true) {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("WebGPU adapter unavailable; device checks skipped: {error}");
                return;
            }
        };
        assert!(ctx.try_dispatch_gemm(&[1.0], &[1.0], 2, 2, 2, 1).is_err());
        let gpu = GpuDispatch::Wgpu(ctx.clone());
        let x: Vec<_> = (0..3 * 5 * 7).map(|i| (i % 13) as f32 * 0.125).collect();
        let w: Vec<_> = (0..11 * 7).map(|i| (i % 17) as f32 * -0.0625).collect();
        let expected = gemm_cpu_reference(&x, &w, 5, 11, 7, 3);
        let mut out = vec![0.0; expected.len()];
        for _ in 0..2 {
            gpu.try_dispatch_gemm_into(&x, &w, 5, 11, 7, 3, &mut out)
                .unwrap();
            assert_eq!(out.len(), expected.len());
            assert!(
                out.iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() < 1e-5)
            );
        }
        // Reuse a larger workspace for a smaller dispatch: no stale output tail.
        assert_eq!(
            gpu.try_dispatch_gemm(&x[..7], &w[..7], 1, 1, 7, 1).unwrap(),
            gemm_cpu_reference(&x[..7], &w[..7], 1, 1, 7, 1)
        );
        let large = vec![1.0; 65536];
        assert_eq!(
            gpu.try_dispatch_gemm(&large, &[2.0], 1, 1, 1, 65536).unwrap(),
            vec![2.0; 65536]
        );
        gpu.dispatch_gemm_into(&large, &[2.0], 1, 1, 1, 65536, &mut vec![0.0; 65536])
            .unwrap();
        // Deliberately violate a resource limit *inside* the same error-scope
        // helper used by dispatch. This must produce Err, not an uncaptured panic.
        begin_error_scopes(&ctx.device);
        let _invalid = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("validation-regression"),
            size: ctx.device.limits().max_buffer_size + 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(finish_error_scopes(&ctx.device).is_err());
    }
}

#[derive(Clone)]
pub enum TensorBuffer {
    Cpu(Vec<f32>),
    Gpu(Arc<wgpu::Buffer>),
}

impl TensorBuffer {
    pub fn as_cpu_slice(&self) -> &[f32] {
        match self {
            TensorBuffer::Cpu(vec) => vec.as_slice(),
            TensorBuffer::Gpu(_) => panic!(
                "Attempted to read GPU tensor directly as CPU slice without staging readback."
            ),
        }
    }

    pub fn as_cpu_mut_slice(&mut self) -> &mut [f32] {
        match self {
            TensorBuffer::Cpu(vec) => vec.as_mut_slice(),
            TensorBuffer::Gpu(_) => panic!("Attempted to mutate GPU tensor directly as CPU slice."),
        }
    }
}

#[derive(Clone)]
pub struct ParamTensor {
    pub shape: Vec<usize>,
    pub device: Device,
    pub data: TensorBuffer,
    pub grad: TensorBuffer,
    pub m: TensorBuffer,
    pub v: TensorBuffer,
}

impl ParamTensor {
    pub fn new_cpu(shape: Vec<usize>, init_val: f32) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            device: Device::Cpu,
            data: TensorBuffer::Cpu(vec![init_val; size]),
            grad: TensorBuffer::Cpu(vec![0.0; size]),
            m: TensorBuffer::Cpu(vec![0.0; size]),
            v: TensorBuffer::Cpu(vec![0.0; size]),
        }
    }

    pub fn new_gpu(ctx: &WgpuContext, shape: Vec<usize>, init_data: &[f32]) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(size, init_data.len());

        let data_buf = ctx.create_buffer_init("param_data", init_data, false);
        let grad_buf = ctx.create_buffer_init("param_grad", &vec![0.0f32; size], false);
        let m_buf = ctx.create_buffer_init("param_m", &vec![0.0f32; size], false);
        let v_buf = ctx.create_buffer_init("param_v", &vec![0.0f32; size], false);

        Self {
            shape,
            device: Device::Gpu(ctx.clone()),
            data: TensorBuffer::Gpu(Arc::new(data_buf)),
            grad: TensorBuffer::Gpu(Arc::new(grad_buf)),
            m: TensorBuffer::Gpu(Arc::new(m_buf)),
            v: TensorBuffer::Gpu(Arc::new(v_buf)),
        }
    }

    pub fn zero_grad(&mut self) {
        match &mut self.grad {
            TensorBuffer::Cpu(vec) => vec.fill(0.0),
            TensorBuffer::Gpu(buf) => {
                if let Device::Gpu(ctx) = &self.device {
                    let zero_vec = vec![0.0f32; self.shape.iter().product()];
                    ctx.queue
                        .write_buffer(buf, 0, bytemuck::cast_slice(&zero_vec));
                }
            }
        }
    }
}
