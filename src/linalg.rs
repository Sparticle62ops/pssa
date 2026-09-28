use std::f32;

pub struct SimpleRng {
    pub state: u64,
}

impl SimpleRng {
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 32) as u32
    }

    #[inline(always)]
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        let norm = (self.next_u32() as f32) / (u32::MAX as f32);
        low + norm * (high - low)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    pub fn dot(&self, other: &Vector) -> f32 {
        dot_slice(&self.data, &other.data)
    }

    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline(always)]
    pub fn scale_in_place(&mut self, s: f32) {
        for x in self.data.iter_mut() {
            *x *= s;
        }
    }

    #[inline(always)]
    pub fn clamp_norm_in_place(&mut self, max_norm: f32) {
        let n = self.norm();
        if n > max_norm {
            self.scale_in_place(max_norm / (n + 1e-7));
        }
    }

    #[inline(always)]
    pub fn rms_norm_into(&self, out: &mut Vector) {
        let len = self.data.len();
        if out.data.len() != len {
            out.data.resize(len, 0.0);
        }
        if len == 0 {
            return;
        }
        rms_norm_slice(&self.data, &mut out.data);
    }

    #[inline(always)]
    pub fn scale(&self, s: f32) -> Vector {
        Vector {
            data: self.data.iter().map(|a| a * s).collect(),
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "vector operands must have equal lengths"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    #[inline(always)]
    pub fn clamp_norm(&self, max_norm: f32) -> Vector {
        let n = self.norm();
        if n > max_norm {
            Vector {
                data: self
                    .data
                    .iter()
                    .map(|a| a * (max_norm / (n + 1e-7)))
                    .collect(),
            }
        } else {
            self.clone()
        }
    }

    #[inline(always)]
    pub fn rms_norm(&self) -> Vector {
        let mut out = Vector::zeros(self.len());
        self.rms_norm_into(&mut out);
        out
    }
}

/// Repository RMSNorm convention, with epsilon inside the root. Returns the
/// inverse RMS for backward; affine gamma/beta are applied by the model.
pub fn rms_norm_slice(input: &[f32], out: &mut [f32]) -> f32 {
    assert_eq!(input.len(), out.len());
    if input.is_empty() { return 0.0; }
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let inv = 1.0 / (sum_sq / input.len() as f32 + 1e-5).sqrt();
    for (y, x) in out.iter_mut().zip(input) { *y = x * inv; }
    inv
}

/// Returns the dot product of two equally sized slices.
///
/// Generic x86_64 builds select AVX2/FMA using cached runtime feature detection;
/// other CPUs retain the portable implementation. Tiny slices avoid SIMD setup.
/// Fused multiply-add has one rounding rather than the scalar multiply followed
/// by add's two, so finite results require a mixed absolute/relative tolerance
/// rather than bitwise equality.
#[inline(always)]
pub fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot product operands must have equal lengths"
    );

    #[cfg(target_arch = "x86_64")]
    if a.len() >= 32 && avx2_fma_available() {
        // SAFETY: runtime detection establishes both target features and the
        // helper only reads within the equally sized input slices.
        return unsafe { dot_slice_avx2_fma(a, b) };
    }

    dot_slice_portable(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_fma_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_slice_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0;
    // Four independent dependency chains cover 32 floats per iteration.
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );

    while i + 32 <= len {
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i)),
                _mm256_loadu_ps(b_ptr.add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i + 8)),
                _mm256_loadu_ps(b_ptr.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i + 16)),
                _mm256_loadu_ps(b_ptr.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i + 24)),
                _mm256_loadu_ps(b_ptr.add(i + 24)),
                acc3,
            );
        }
        i += 32;
    }

    // Consume the remaining complete vectors before the final 0..7 scalars.
    while i + 8 <= len {
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i)),
                _mm256_loadu_ps(b_ptr.add(i)),
                acc0,
            );
        }
        i += 8;
    }

    let pair01 = _mm256_add_ps(acc0, acc1);
    let pair23 = _mm256_add_ps(acc2, acc3);
    let lanes = _mm256_add_ps(pair01, pair23);
    let folded = _mm_add_ps(
        _mm256_castps256_ps128(lanes),
        _mm256_extractf128_ps(lanes, 1),
    );
    let pair = _mm_hadd_ps(folded, folded);
    let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pair, pair));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline(always)]
fn dot_slice_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut i = 0;
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    while i + 4 <= a.len() {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline(always)]
pub unsafe fn dot_256_raw(a: *const f32, b: *const f32) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    unsafe {
        for i in (0..256).step_by(8) {
            s0 += *a.add(i) * *b.add(i);
            s1 += *a.add(i + 1) * *b.add(i + 1);
            s2 += *a.add(i + 2) * *b.add(i + 2);
            s3 += *a.add(i + 3) * *b.add(i + 3);
            s4 += *a.add(i + 4) * *b.add(i + 4);
            s5 += *a.add(i + 5) * *b.add(i + 5);
            s6 += *a.add(i + 6) * *b.add(i + 6);
            s7 += *a.add(i + 7) * *b.add(i + 7);
        }
    }
    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

#[inline(always)]
pub unsafe fn dot_128_raw(a: *const f32, b: *const f32) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    unsafe {
        for i in (0..128).step_by(8) {
            s0 += *a.add(i) * *b.add(i);
            s1 += *a.add(i + 1) * *b.add(i + 1);
            s2 += *a.add(i + 2) * *b.add(i + 2);
            s3 += *a.add(i + 3) * *b.add(i + 3);
            s4 += *a.add(i + 4) * *b.add(i + 4);
            s5 += *a.add(i + 5) * *b.add(i + 5);
            s6 += *a.add(i + 6) * *b.add(i + 6);
            s7 += *a.add(i + 7) * *b.add(i + 7);
        }
    }
    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

#[inline(always)]
pub unsafe fn dot_32_raw(a: *const f32, b: *const f32) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    unsafe {
        for i in (0..32).step_by(4) {
            s0 += *a.add(i) * *b.add(i);
            s1 += *a.add(i + 1) * *b.add(i + 1);
            s2 += *a.add(i + 2) * *b.add(i + 2);
            s3 += *a.add(i + 3) * *b.add(i + 3);
        }
    }
    (s0 + s1) + (s2 + s3)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random_xavier(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        let data = (0..rows * cols)
            .map(|_| rng.gen_range_f32(-limit, limit))
            .collect();
        Self { rows, cols, data }
    }

    /// Multiply into an existing output, validating public matrix storage before
    /// reading any row. Shape violations panic, as for the other numeric APIs.
    #[inline(always)]
    pub fn matvec_into(&self, v: &[f32], out: &mut [f32]) {
        let len = self.rows.checked_mul(self.cols)
            .expect("matrix dimensions overflow usize");
        assert_eq!(self.data.len(), len, "matrix storage must match its dimensions");
        assert_eq!(self.cols, v.len(), "matrix input length must match its columns");
        assert_eq!(self.rows, out.len(), "matrix output length must match its rows");
        if self.cols == 0 {
            out.fill(0.0);
            return;
        }
        for (row, value) in self.data.chunks_exact(self.cols).zip(out) {
            *value = dot_slice(row, v);
        }
    }

    #[inline(always)]
    pub fn matvec(&self, v: &Vector) -> Vector {
        let mut out = vec![0.0; self.rows];
        self.matvec_into(&v.data, &mut out);
        Vector { data: out }
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            let self_offset = i * self.cols;
            let out_offset = i * other.cols;
            for k in 0..self.cols {
                let a = self.data[self_offset + k];
                let other_offset = k * other.cols;
                for j in 0..other.cols {
                    out.data[out_offset + j] += a * other.data[other_offset + j];
                }
            }
        }
        out
    }

    pub fn outer_product(u: &Vector, v: &Vector) -> Matrix {
        let rows = u.len();
        let cols = v.len();
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            let ui = u.data[i];
            let row_offset = i * cols;
            for j in 0..cols {
                data[row_offset + j] = ui * v.data[j];
            }
        }
        Matrix { rows, cols, data }
    }

    pub fn add_assign_scaled(&mut self, delta: &Matrix, scale: f32) {
        assert_eq!(self.rows, delta.rows, "matrix row counts must match");
        assert_eq!(self.cols, delta.cols, "matrix column counts must match");
        assert_eq!(self.data.len(), self.rows * self.cols);
        assert_eq!(delta.data.len(), delta.rows * delta.cols);
        for (a, b) in self.data.iter_mut().zip(&delta.data) {
            *a += *b * scale;
        }
    }

    #[inline(always)]
    pub fn get_row_slice(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline(always)]
    pub fn get_row(&self, row: usize) -> Vector {
        Vector {
            data: self.get_row_slice(row).to_vec(),
        }
    }

    /// Invert with f64 elimination and row-scale-relative partial pivoting.
    /// Reject malformed storage, singular matrices, and nonfinite input/output.
    pub fn invert(&self) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Cannot invert non-square matrix".to_string());
        }
        let n = self.rows;
        let len = n.checked_mul(n)
            .ok_or_else(|| "Matrix dimensions overflow usize".to_string())?;
        if self.data.len() != len {
            return Err("Matrix storage must match its dimensions".to_string());
        }
        if self.data.iter().any(|x| !x.is_finite()) {
            return Err("Cannot invert a matrix containing nonfinite values".to_string());
        }
        let stride = n.checked_mul(2)
            .ok_or_else(|| "Augmented matrix dimensions overflow usize".to_string())?;
        let augmented_len = len.checked_mul(2)
            .ok_or_else(|| "Augmented matrix dimensions overflow usize".to_string())?;
        let mut augmented = Vec::new();
        augmented.try_reserve_exact(augmented_len)
            .map_err(|e| format!("Cannot allocate augmented matrix: {e}"))?;
        augmented.resize(augmented_len, 0.0f64);
        let mut row_scales = vec![0.0f64; n];

        for i in 0..n {
            for j in 0..n {
                let value = self.data[i * n + j] as f64;
                augmented[i * stride + j] = value;
                row_scales[i] = row_scales[i].max(value.abs());
            }
            if row_scales[i] == 0.0 {
                return Err("Matrix is singular and cannot be inverted".to_string());
            }
            augmented[i * stride + n + i] = 1.0;
        }

        for i in 0..n {
            // Scaled partial pivoting makes singularity decisions invariant to
            // uniform rescaling, even when the original entries are very small.
            let mut pivot_row = i;
            let mut max_ratio = augmented[i * stride + i].abs() / row_scales[i];
            for k in (i + 1)..n {
                let ratio = augmented[k * stride + i].abs() / row_scales[k];
                if ratio > max_ratio {
                    max_ratio = ratio;
                    pivot_row = k;
                }
            }

            if !max_ratio.is_finite() || max_ratio <= f64::EPSILON * n as f64 {
                return Err("Matrix is singular and cannot be inverted".to_string());
            }

            if pivot_row != i {
                for col in 0..stride {
                    augmented.swap(i * stride + col, pivot_row * stride + col);
                }
                row_scales.swap(i, pivot_row);
            }

            let pivot = augmented[i * stride + i];
            for col in 0..stride {
                augmented[i * stride + col] /= pivot;
                if !augmented[i * stride + col].is_finite() {
                    return Err("Matrix inverse produced nonfinite values".to_string());
                }
            }

            for row in 0..n {
                if row != i {
                    let factor = augmented[row * stride + i];
                    // A small nonzero factor can still be significant relative
                    // to the matrix scale; only exact zero is safe to skip.
                    if factor != 0.0 {
                        for col in 0..stride {
                            let sub = factor * augmented[i * stride + col];
                            augmented[row * stride + col] -= sub;
                            if !augmented[row * stride + col].is_finite() {
                                return Err("Matrix inverse produced nonfinite values".to_string());
                            }
                        }
                    }
                }
            }
        }

        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let value = augmented[i * stride + n + j] as f32;
                if !value.is_finite() {
                    return Err("Matrix inverse is not representable as finite f32 values".to_string());
                }
                inv.data[i * n + j] = value;
            }
        }

        Ok(inv)
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

pub fn softmax(v: &Vector) -> Vector {
    if v.data.is_empty() {
        return Vector { data: Vec::new() };
    }
    // Subtracting the maximum is required for finite probabilities when a
    // caller supplies ordinary logits rather than already-normalized values.
    let max = v.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.data.iter().map(|x| (*x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Vector {
        data: exps.iter().map(|x| x / sum).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_and_runtime_kernels_match_f64_for_unaligned_slices_and_every_tail() {
        for len in (0..=97).chain([127, 128, 129, 255, 256, 257, 1023, 1024, 1025]) {
            let a: Vec<f32> = (0..len + 3).map(|i| ((i * 37 % 127) as f32 - 63.0) * 0.03125).collect();
            let b: Vec<f32> = (0..len + 5).map(|i| ((i * 53 % 113) as f32 - 56.0) * -0.0234375).collect();
            let (a, b) = (&a[1..len + 1], &b[3..len + 3]);
            let expected: f64 = a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum();
            let check = |actual: f32| {
                assert!(actual.is_finite());
                assert!((actual as f64 - expected).abs() <= 2.0e-6 + 2.0e-5 * expected.abs(),
                    "length={len}, actual={actual}, expected={expected}");
            };
            check(dot_slice_portable(a, b));
            check(dot_slice(a, b));
            #[cfg(target_arch = "x86_64")]
            if avx2_fma_available() {
                // SAFETY: feature detection and equally sized valid slices.
                check(unsafe { dot_slice_avx2_fma(a, b) });
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cached_detection_matches_host_and_dispatches_the_selected_kernel() {
        let expected = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        assert_eq!(avx2_fma_available(), expected);
        assert_eq!(avx2_fma_available(), expected);
        // Non-dyadic values exercise the distinct FMA rounding/reduction path.
        let a: Vec<f32> = (0..257).map(|i| (i as f32 * 0.19).sin()).collect();
        let b: Vec<f32> = (0..257).map(|i| (i as f32 * 0.31).cos()).collect();
        let selected = if expected {
            // SAFETY: host detection and equal valid lengths.
            unsafe { dot_slice_avx2_fma(&a, &b) }
        } else {
            dot_slice_portable(&a, &b)
        };
        assert_eq!(dot_slice(&a, &b).to_bits(), selected.to_bits());
    }

    /// Opt-in microbenchmark: run an optimized generic-target test binary with
    /// `--ignored --nocapture dot_kernel_microbenchmark`. Not a training timing.
    #[test]
    #[ignore = "timing probe; run optimized and without competing CPU workloads"]
    fn dot_kernel_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        // Match the pre-runtime-dispatch public wrapper, including its shape
        // assertion: it lets LLVM eliminate the portable loop's bounds checks.
        #[inline(always)]
        fn portable(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len(), "dot product operands must have equal lengths");
            dot_slice_portable(a, b)
        }

        fn time(mut kernel: impl FnMut() -> f32, iterations: usize) -> f64 {
            for _ in 0..1000 { black_box(kernel()); }
            let start = Instant::now();
            for _ in 0..iterations { black_box(kernel()); }
            start.elapsed().as_secs_f64() * 1.0e9 / iterations as f64
        }
        #[cfg(target_arch = "x86_64")]
        println!("runtime_avx2_fma={}", avx2_fma_available());
        for len in [16, 32, 64, 128, 256, 257, 512, 2048] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.19).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.31).cos()).collect();
            let iterations = (20_000_000 / len).max(50_000);
            let mut baseline = Vec::new();
            let mut runtime = Vec::new();
            for round in 0..7 {
                // Alternate order to reduce first/second-run frequency bias.
                if round % 2 == 0 {
                    baseline.push(time(|| portable(black_box(&a), black_box(&b)), iterations));
                    runtime.push(time(|| dot_slice(black_box(&a), black_box(&b)), iterations));
                } else {
                    runtime.push(time(|| dot_slice(black_box(&a), black_box(&b)), iterations));
                    baseline.push(time(|| portable(black_box(&a), black_box(&b)), iterations));
                }
            }
            baseline.sort_by(f64::total_cmp);
            runtime.sort_by(f64::total_cmp);
            println!("length={len} portable_median_ns={:.2} runtime_median_ns={:.2} speedup={:.2}x",
                baseline[3], runtime[3], baseline[3] / runtime[3]);
        }
    }
}
