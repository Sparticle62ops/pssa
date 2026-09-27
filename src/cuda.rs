//! Native CUDA backend: cuBLAS SGEMM with device-resident weight caching.
//!
//! This mirrors the semantics of [`crate::backend::WgpuContext::dispatch_gemm`]
//! exactly, so the two GPU backends are interchangeable and both are checked
//! against `gemm_cpu_reference`.
//!
//! Layout contract (row-major, identical to the WGSL kernel):
//!   X: [batch, M, K]
//!   W: [N, K]          (shared across the batch)
//!   Y: [batch, M, N]   Y[b] = X[b] * W^T
//!
//! cuBLAS is column-major, so a row-major A of shape (r, c) is the same bytes
//! as a column-major (c, r). Writing the row-major product in column-major
//! terms gives Y^c (N x M) = W^c^T (N x K) * X^c (K x M), which is the
//! transa=T / transb=N configuration below with leading dimensions K, K, N.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaSlice, CudaStream, CudaContext as DriverContext};

/// A live CUDA device plus a cuBLAS handle and the resident weight cache.
#[derive(Clone)]
pub struct CudaContext {
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    name: String,
    /// Weight matrices already uploaded, keyed by host pointer + length so a
    /// parameter is transferred once and reused until the optimizer moves it.
    weight_cache: Arc<Mutex<HashMap<(usize, usize), Arc<CudaSlice<f32>>>>>,
}

impl CudaContext {
    /// Bring up device 0. Returns Err with the driver-level reason when no
    /// usable CUDA device is present, so the caller can fall back.
    pub fn init() -> Result<CudaContext, String> {
        let ctx = DriverContext::new(0).map_err(|e| format!("no CUDA device ({e:?})"))?;
        let name = ctx.name().unwrap_or_else(|_| "unknown CUDA device".to_string());
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("cuBLAS unavailable on {name} ({e:?})"))?;
        Ok(CudaContext {
            stream,
            blas: Arc::new(blas),
            name,
            weight_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    /// Upload `w` once and keep it on the card. The key is the host pointer and
    /// length, which is stable for a parameter tensor across training steps.
    fn weight_buffer(&self, w: &[f32]) -> Result<Arc<CudaSlice<f32>>, String> {
        let key = (w.as_ptr() as usize, w.len());
        if let Some(buf) = self.weight_cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let buf = Arc::new(
            self.stream
                .clone_htod(w)
                .map_err(|e| format!("weight upload failed ({e:?})"))?,
        );
        self.weight_cache.lock().unwrap().insert(key, buf.clone());
        Ok(buf)
    }

    /// Drop every cached weight. Must be called after an optimizer step, or the
    /// GPU would keep multiplying by the pre-update parameters.
    pub fn invalidate_weights(&self) {
        self.weight_cache.lock().unwrap().clear();
    }

    /// Y[b] = X[b] * W^T for every b, with the row-major layout documented above.
    /// Falls back to the CPU reference if any CUDA call fails, so a driver
    /// hiccup degrades throughput instead of corrupting a training run.
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
                eprintln!("warning: CUDA GEMM failed; using CPU fallback: {error}");
                crate::backend::gemm_cpu_reference(x, w, m, n, k, batch)
            }
        }
    }

    fn try_dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<Vec<f32>, String> {
        if m == 0 || n == 0 || k == 0 || batch == 0 {
            return Err("CUDA GEMM dimensions must be positive".into());
        }
        let x_len = batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(k))
            .ok_or_else(|| "CUDA GEMM input size overflow".to_string())?;
        let w_len = n
            .checked_mul(k)
            .ok_or_else(|| "CUDA GEMM weight size overflow".to_string())?;
        let out_len = batch
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| "CUDA GEMM output size overflow".to_string())?;
        if x.len() != x_len || w.len() != w_len {
            return Err("CUDA GEMM buffer length mismatch".into());
        }

        let w_dev = self.weight_buffer(w)?;
        let x_dev = self
            .stream
            .clone_htod(x)
            .map_err(|e| format!("activation upload failed ({e:?})"))?;
        let mut y_dev = self
            .stream
            .alloc_zeros::<f32>(out_len)
            .map_err(|e| format!("output alloc failed ({e:?})"))?;

        let gemm = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            beta: 0.0f32,
            lda: k as i32,
            ldb: k as i32,
            ldc: n as i32,
        };

        if batch == 1 {
            unsafe { self.blas.gemm(gemm, w_dev.as_ref(), &x_dev, &mut y_dev) }
                .map_err(|e| format!("cuBLAS sgemm failed ({e:?})"))?;
        } else {
            let cfg = StridedBatchedConfig {
                gemm,
                batch_size: batch as i32,
                // The weight matrix is shared across the batch: stride 0.
                stride_a: 0,
                stride_b: (m * k) as i64,
                stride_c: (m * n) as i64,
            };
            unsafe {
                self.blas
                    .gemm_strided_batched(cfg, w_dev.as_ref(), &x_dev, &mut y_dev)
            }
            .map_err(|e| format!("cuBLAS batched sgemm failed ({e:?})"))?;
        }

        self.stream
            .clone_dtoh(&y_dev)
            .map_err(|e| format!("readback failed ({e:?})"))
    }
}

impl CudaContext {
    /// C(M,N) = A(M,K) * B(K,N), all row-major. Used by the backward pass,
    /// where the second operand is not transposed.
    pub fn gemm_nn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.try_gemm_nn(a, b, m, k, n).unwrap_or_else(|error| {
            eprintln!("warning: CUDA backward GEMM (NN) failed; using CPU fallback: {error}");
            crate::backend::gemm_nn_cpu(a, b, m, k, n)
        })
    }

    /// C(K,N) = A(M,K)^T * B(M,N), all row-major. This is the weight-gradient
    /// shape: contract over the token axis.
    pub fn gemm_tn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.try_gemm_tn(a, b, m, k, n).unwrap_or_else(|error| {
            eprintln!("warning: CUDA backward GEMM (TN) failed; using CPU fallback: {error}");
            crate::backend::gemm_tn_cpu(a, b, m, k, n)
        })
    }

    fn try_gemm_nn(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        // Column-major identity: C^c(N,M) = B^c(N,K) * A^c(K,M), no transposes.
        let a_dev = self
            .stream
            .clone_htod(a)
            .map_err(|e| format!("gemm_nn lhs upload failed ({e:?})"))?;
        let b_dev = self.weight_buffer(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| format!("gemm_nn output alloc failed ({e:?})"))?;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            beta: 0.0f32,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, b_dev.as_ref(), &a_dev, &mut c_dev) }
            .map_err(|e| format!("cuBLAS gemm_nn failed ({e:?})"))?;
        self.stream
            .clone_dtoh(&c_dev)
            .map_err(|e| format!("gemm_nn readback failed ({e:?})"))
    }

    fn try_gemm_tn(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        // Column-major identity: C^c(N,K) = B^c(N,M) * (A^c)^T(M,K).
        // Both operands are activations here, so neither is weight-cached.
        let a_dev = self
            .stream
            .clone_htod(a)
            .map_err(|e| format!("gemm_tn lhs upload failed ({e:?})"))?;
        let b_dev = self
            .stream
            .clone_htod(b)
            .map_err(|e| format!("gemm_tn rhs upload failed ({e:?})"))?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>(k * n)
            .map_err(|e| format!("gemm_tn output alloc failed ({e:?})"))?;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: k as i32,
            k: m as i32,
            alpha: 1.0f32,
            beta: 0.0f32,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }
            .map_err(|e| format!("cuBLAS gemm_tn failed ({e:?})"))?;
        self.stream
            .clone_dtoh(&c_dev)
            .map_err(|e| format!("gemm_tn readback failed ({e:?})"))
    }
}
