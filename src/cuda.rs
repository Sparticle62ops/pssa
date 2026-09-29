//! Native CUDA/cuBLAS backend. Public entry points validate the complete shape
//! before allocation or unsafe FFI. Strict `try_*` methods never execute on CPU.
//! Forward layout: X [batch,M,K], W [N,K], Y [batch,M,N] = X * W^T.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext as DriverContext, CudaSlice, CudaStream};

use crate::backend::{checked_gemm_sizes, shared_gemm_rows, zeroed_output};

// cudarc 0.19 lazily loads symbols with unwrap/panic, including error formatting
// and Drop paths. Preflight *every* symbol used by this backend before calling
// it, not just cuInit/cublasCreate. This also works under release panic=abort.
// Keep this list in sync when adding CUDA operations/updating cudarc.
const DRIVER_SYMBOLS: &[&str] = &[
    "cuInit",
    "cuDeviceGet",
    "cuDeviceGetAttribute",
    "cuDeviceGetName",
    "cuDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxRelease_v2",
    "cuCtxGetCurrent",
    "cuCtxSetCurrent",
    "cuGetErrorName",
    "cuGetErrorString",
    "cuMemAlloc_v2",
    "cuMemFree_v2",
    "cuMemAllocAsync",
    "cuMemFreeAsync",
    "cuMemcpyHtoDAsync_v2",
    "cuMemcpyDtoHAsync_v2",
    "cuMemsetD8Async",
    "cuStreamSynchronize",
    "cuStreamWaitEvent",
    "cuEventCreate",
    "cuEventRecord",
    "cuEventSynchronize",
    "cuEventDestroy_v2",
];
const BLAS_SYMBOLS: &[&str] = &[
    "cublasCreate_v2",
    "cublasDestroy_v2",
    "cublasSetStream_v2",
    "cublasSgemm_v2",
    "cublasSgemmStridedBatched",
];

fn check_symbols(
    library: &str,
    names: &[&str],
    mut present: impl FnMut(&str) -> bool,
) -> Result<(), String> {
    for &name in names {
        if !present(name) {
            return Err(format!(
                "{library} is missing required symbol {name}; install a compatible CUDA driver/cuBLAS or use CPU"
            ));
        }
    }
    Ok(())
}

fn preflight_libraries() -> Result<(), String> {
    static CHECK: OnceLock<Result<(), String>> = OnceLock::new();
    CHECK
        .get_or_init(|| {
            // SAFETY: only the installed CUDA libraries are loaded; queried symbol
            // pointers are checked for presence, never called with a guessed ABI.
            unsafe {
                if !cudarc::driver::sys::is_culib_present() {
                    return Err(
                        "CUDA driver library unavailable; install the driver or use CPU/WebGPU"
                            .into(),
                    );
                }
                let driver = cudarc::driver::sys::culib();
                check_symbols("CUDA driver", DRIVER_SYMBOLS, |name| {
                    driver
                        .get::<unsafe extern "C" fn()>(name.as_bytes())
                        .is_ok()
                })?;
                if !cudarc::cublas::sys::is_culib_present() {
                    return Err(
                        "cuBLAS library unavailable; install cuBLAS or use CPU/WebGPU".into(),
                    );
                }
                let blas = cudarc::cublas::sys::culib();
                check_symbols("cuBLAS", BLAS_SYMBOLS, |name| {
                    blas.get::<unsafe extern "C" fn()>(name.as_bytes()).is_ok()
                })
            }
        })
        .clone()
}

/// Check cuBLAS's signed dimensions *and* all allocation/stride products before
/// casting or using cudarc (whose unsafe GEMM intentionally omits these checks).
fn checked_cuda_sizes(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a_len: usize,
    b_len: usize,
) -> Result<(usize, usize, usize), String> {
    if [m, n, k, batch].iter().any(|&d| d > i32::MAX as usize) {
        return Err("CUDA GEMM dimensions/batch exceed the cuBLAS i32 limit".into());
    }
    checked_gemm_sizes(m, n, k, batch, a_len, b_len)
}

#[derive(Default)]
struct Workspace {
    lhs: Option<CudaSlice<f32>>,
    rhs: Option<CudaSlice<f32>>,
    output: Option<CudaSlice<f32>>,
}

fn reserve_device(
    slot: &mut Option<CudaSlice<f32>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<(), String> {
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        // Zero-initialize once when growing; subsequent beta=0 GEMMs overwrite
        // the entire requested view, without re-zeroing or reallocation.
        *slot = Some(
            stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| format!("CUDA workspace allocation failed ({e:?})"))?,
        );
    }
    Ok(())
}

/// A live CUDA device, cuBLAS handle, resident weights and bounded high-water
/// workspaces. Clones serialize dispatch so scratch and the handle remain safe.
#[derive(Clone)]
pub struct CudaContext {
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    name: Arc<str>,
    weight_cache: Arc<Mutex<HashMap<(usize, usize), Arc<CudaSlice<f32>>>>>,
    workspace: Arc<Mutex<Workspace>>,
}

impl CudaContext {
    pub fn init() -> Result<Self, String> {
        preflight_libraries()?;
        let ctx = DriverContext::new(0).map_err(|e| format!("no CUDA device ({e:?})"))?;
        let name: Arc<str> = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string())
            .into();
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("cuBLAS unavailable on {name} ({e:?})"))?;
        Ok(Self {
            stream,
            blas: Arc::new(blas),
            name,
            weight_cache: Arc::new(Mutex::new(HashMap::new())),
            workspace: Arc::new(Mutex::new(Workspace::default())),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    fn weight_buffer(&self, w: &[f32]) -> Result<Arc<CudaSlice<f32>>, String> {
        let key = (w.as_ptr() as usize, w.len());
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| "CUDA weight cache lock poisoned")?;
        if let Some(buf) = cache.get(&key) {
            return Ok(buf.clone());
        }
        let buf = Arc::new(
            self.stream
                .clone_htod(w)
                .map_err(|e| format!("weight upload failed ({e:?})"))?,
        );
        cache.insert(key, buf.clone());
        Ok(buf)
    }

    /// Call whenever cached host weights change (normally after AdamW).
    pub fn invalidate_weights(&self) {
        self.weight_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    pub fn dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Vec<f32> {
        self.try_dispatch_gemm(x, w, m, n, k, batch)
            .unwrap_or_else(|error| {
                eprintln!("warning: CUDA GEMM failed; using CPU fallback: {error}");
                crate::backend::gemm_cpu_reference(x, w, m, n, k, batch)
            })
    }

    /// Strict GPU execution, without a silent CPU fallback.
    pub fn try_dispatch_gemm(
        &self,
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, _, len) = checked_cuda_sizes(m, n, k, batch, x.len(), w.len())?;
        let mut out = zeroed_output(len)?;
        self.try_dispatch_gemm_into(x, w, m, n, k, batch, &mut out)?;
        Ok(out)
    }

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
        let (_, _, len) = checked_cuda_sizes(m, n, k, batch, x.len(), w.len())?;
        if out.len() != len {
            return Err("CUDA GEMM output length mismatch".into());
        }
        // Shared weights make [batch,M,K] exactly one [batch*M,K] matrix.
        // cuBLAS should see a wide SGEMM, never a collection of skinny GEMVs.
        let m = shared_gemm_rows(m, batch)?;
        if m > i32::MAX as usize {
            return Err("CUDA flattened GEMM rows exceed the cuBLAS i32 limit".into());
        }
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
        self.execute_into(x, w, cfg, 1, 0, 0, true, false, out)
    }

    pub fn gemm_nn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.try_gemm_nn(a, b, m, k, n).unwrap_or_else(|error| {
            eprintln!("warning: CUDA backward GEMM (NN) failed; using CPU fallback: {error}");
            crate::backend::gemm_nn_cpu(a, b, m, k, n)
        })
    }

    pub fn gemm_tn(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.try_gemm_tn(a, b, m, k, n).unwrap_or_else(|error| {
            eprintln!("warning: CUDA backward GEMM (TN) failed; using CPU fallback: {error}");
            crate::backend::gemm_tn_cpu(a, b, m, k, n)
        })
    }

    /// Row-major C(M,N) = A(M,K) * B(K,N), strictly on CUDA.
    pub fn try_gemm_nn(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, _, len) = checked_cuda_sizes(m, n, k, 1, a.len(), b.len())?;
        let mut out = zeroed_output(len)?;
        // Column-major identity C^c(N,M) = B^c(N,K) * A^c(K,M).
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            beta: 0.0,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        self.execute_into(a, b, cfg, 1, 0, 0, true, false, &mut out)?;
        Ok(out)
    }

    /// Row-major C(K,N) = A(M,K)^T * B(M,N), strictly on CUDA.
    pub fn try_gemm_tn(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, _, len) = checked_cuda_sizes(k, n, m, 1, a.len(), b.len())?;
        let mut out = zeroed_output(len)?;
        // Column-major identity C^c(N,K) = B^c(N,M) * (A^c)^T(M,K).
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: k as i32,
            k: m as i32,
            alpha: 1.0,
            beta: 0.0,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        self.execute_into(a, b, cfg, 1, 0, 0, false, false, &mut out)?;
        Ok(out)
    }

    /// Row-major C(M,N) = A(M,K) * B(K,N), writing into caller-owned output.
    pub fn gemm_nn_into(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.try_gemm_nn_into(a, b, m, k, n, out).or_else(|error| {
            eprintln!("warning: CUDA backward GEMM (NN) failed; using CPU fallback: {error}");
            crate::backend::gemm_nn_cpu_into(a, b, m, k, n, out)
        })
    }

    pub fn try_gemm_nn_into(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let (_, _, len) = checked_cuda_sizes(m, n, k, 1, a.len(), b.len())?;
        if out.len() != len {
            return Err("CUDA NN GEMM output length mismatch".into());
        }
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            beta: 0.0,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        self.execute_into(a, b, cfg, 1, 0, 0, true, false, out)
    }

    /// Accumulate row-major A(M,K)^T * B(M,N) into caller-owned output.
    pub fn gemm_tn_accumulate_into(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        self.try_gemm_tn_accumulate_into(a, b, m, k, n, out)
            .or_else(|error| {
                eprintln!("warning: CUDA backward GEMM (TN) failed; using CPU fallback: {error}");
                crate::backend::gemm_tn_cpu_accumulate_into(a, b, m, k, n, out)
            })
    }

    pub fn try_gemm_tn_accumulate_into(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let (_, _, len) = checked_cuda_sizes(k, n, m, 1, a.len(), b.len())?;
        if out.len() != len {
            return Err("CUDA TN GEMM output length mismatch".into());
        }
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: k as i32,
            k: m as i32,
            alpha: 1.0,
            beta: 1.0,
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };
        self.execute_into(a, b, cfg, 1, 0, 0, false, true, out)
    }

    // Only called after full shape/byte checks; each view exactly covers the
    // validated operands/output. The workspace guard serializes cloned handles.
    fn execute_into(
        &self,
        a: &[f32],
        b: &[f32],
        gemm: GemmConfig<f32>,
        batch: usize,
        stride_b: i64,
        stride_c: i64,
        cache_rhs: bool,
        seed_output: bool,
        out: &mut [f32],
    ) -> Result<(), String> {
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| "CUDA workspace lock poisoned")?;
        let Workspace { lhs, rhs, output } = &mut *workspace;
        reserve_device(lhs, &self.stream, a.len())?;
        reserve_device(output, &self.stream, out.len())?;
        let mut a_dev = lhs.as_mut().unwrap().slice_mut(..a.len());
        self.stream
            .memcpy_htod(a, &mut a_dev)
            .map_err(|e| format!("CUDA lhs upload failed ({e:?})"))?;
        let cached;
        let b_dev = if cache_rhs {
            cached = self.weight_buffer(b)?;
            cached.slice(..b.len())
        } else {
            reserve_device(rhs, &self.stream, b.len())?;
            let mut view = rhs.as_mut().unwrap().slice_mut(..b.len());
            self.stream
                .memcpy_htod(b, &mut view)
                .map_err(|e| format!("CUDA rhs upload failed ({e:?})"))?;
            rhs.as_ref().unwrap().slice(..b.len())
        };
        let mut c_dev = output.as_mut().unwrap().slice_mut(..out.len());
        if seed_output {
            self.stream
                .memcpy_htod(out, &mut c_dev)
                .map_err(|e| format!("CUDA output seed upload failed ({e:?})"))?;
        }
        // SAFETY: exact views, checked signed dimensions/strides and leading
        // dimensions satisfy the row-/column-major identities above.
        if batch == 1 {
            unsafe { self.blas.gemm(gemm, &b_dev, &a_dev, &mut c_dev) }
                .map_err(|e| format!("cuBLAS sgemm failed ({e:?})"))?;
        } else {
            let cfg = StridedBatchedConfig {
                gemm,
                batch_size: batch as i32,
                stride_a: 0,
                stride_b,
                stride_c,
            };
            unsafe {
                self.blas
                    .gemm_strided_batched(cfg, &b_dev, &a_dev, &mut c_dev)
            }
            .map_err(|e| format!("cuBLAS batched sgemm failed ({e:?})"))?;
        }
        self.stream
            .memcpy_dtoh(&c_dev, out)
            .map_err(|e| format!("CUDA readback failed ({e:?})"))?;
        // HostSlice synchronizes on guard Drop and records asynchronous errors
        // in the context. Surface those now, not on an unrelated next dispatch.
        self.stream
            .context()
            .check_err()
            .map_err(|e| format!("CUDA readback synchronization failed ({e:?})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_validation_requires_exact_lengths_signed_dimensions_and_bytes() {
        assert_eq!(
            checked_cuda_sizes(2, 3, 5, 4, 40, 15).unwrap(),
            (40, 15, 24)
        );
        for (m, n, k, batch, a, b) in [
            (2, 2, 2, 1, 1, 1),
            (2, 2, 2, 1, 5, 4),
            (0, 2, 2, 1, 0, 4),
            (1, 1, 1, 0, 0, 1),
            (i32::MAX as usize + 1, 1, 1, 1, 0, 0),
            (1, 1, 1, i32::MAX as usize + 1, 0, 0),
            (
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                i32::MAX as usize,
                0,
                0,
            ),
        ] {
            assert!(checked_cuda_sizes(m, n, k, batch, a, b).is_err());
        }
    }

    #[test]
    fn missing_symbols_are_errors_including_cleanup_and_error_paths() {
        for (library, names) in [("CUDA driver", DRIVER_SYMBOLS), ("cuBLAS", BLAS_SYMBOLS)] {
            assert!(check_symbols(library, names, |_| true).is_ok());
            for &missing in names {
                let error = check_symbols(library, names, |name| name != missing).unwrap_err();
                assert!(error.contains(missing), "{error}");
            }
        }
    }
}
