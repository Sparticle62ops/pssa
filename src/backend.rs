use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    let num_tiles = (gemm_cfg.K + 15u) / 16u;

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

#[derive(Clone)]
pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub gemm_pipeline: Arc<wgpu::ComputePipeline>,
    pub norm_pipeline: Arc<wgpu::ComputePipeline>,
    pub silu_pipeline: Arc<wgpu::ComputePipeline>,
    pub adamw_pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuContext {
    pub fn init_blocking() -> Result<Self, String> {
        // Headless boxes (Kaggle included) often run without XDG_RUNTIME_DIR,
        // in which case the Vulkan loader refuses to start and wgpu silently
        // downgrades to the GL backend on llvmpipe: a software rasterizer
        // running on the CPU. Point the loader at a writable scratch dir so
        // real GPU drivers get a chance to come up.
        if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
            let dir = std::env::temp_dir().join("oxide-xdg-runtime");
            let _ = std::fs::create_dir_all(&dir);
            // SAFETY: single-threaded bring-up; no other thread reads the env.
            unsafe { std::env::set_var("XDG_RUNTIME_DIR", &dir) };
        }

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
        if info.device_type == wgpu::DeviceType::Cpu {
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

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            gemm_pipeline: Arc::new(gemm_pipeline),
            norm_pipeline: Arc::new(norm_pipeline),
            silu_pipeline: Arc::new(silu_pipeline),
            adamw_pipeline: Arc::new(adamw_pipeline),
        })
    }

    pub fn create_buffer_init(&self, label: &str, data: &[f32], read_only: bool) -> wgpu::Buffer {
        let usage = if read_only {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        };

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    pub fn read_buffer_blocking(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let byte_len = (count * std::mem::size_of::<f32>()) as u64;

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

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| sender.send(res).unwrap());

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        result
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
        assert_eq!(x.len(), batch * m * k, "X buffer length mismatch");
        assert_eq!(w.len(), n * k, "W buffer length mismatch");
        let out_len = batch * m * n;

        let cfg: [u32; 4] = [m as u32, n as u32, k as u32, batch as u32];
        let cfg_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gemm_uniforms"),
                contents: bytemuck::cast_slice(&cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let x_buf = self.create_buffer_init("gemm_x", x, true);
        let w_buf = self.create_buffer_init("gemm_w", w, true);
        let y_buf = self.create_buffer_init("gemm_y", &vec![0.0f32; out_len], false);

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
            let gx = ((n + 15) / 16) as u32;
            let gy = ((m + 15) / 16) as u32;
            pass.dispatch_workgroups(gx.max(1), gy.max(1), (batch as u32).max(1));
        }
        self.queue.submit(Some(encoder.finish()));

        self.read_buffer_blocking(&y_buf, out_len)
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
    let mut y = vec![0.0f32; batch * m * n];
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

// =============================================================================
// HARDWARE-AGNOSTIC DEVICE & TENSOR PRIMITIVES
// =============================================================================

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(WgpuContext),
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    /// Try to bring up a real WebGPU compute device. Returns Err with the
    /// adapter-level reason when no GPU is usable on this machine.
    pub fn try_gpu() -> Result<Device, String> {
        WgpuContext::init_blocking().map(Device::Gpu)
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
