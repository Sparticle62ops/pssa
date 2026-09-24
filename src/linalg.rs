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
        let sum_sq: f32 = self.data.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (len as f32) + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..len {
            out.data[i] = self.data[i] * inv_rms;
        }
    }

    #[inline(always)]
    pub fn scale(&self, s: f32) -> Vector {
        Vector {
            data: self.data.iter().map(|a| a * s).collect(),
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Vector) -> Vector {
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

/// Returns the dot product of two equally sized slices.
///
/// The AVX2/FMA implementation is selected at compile time when those target
/// features are enabled. Fused multiply-add has one rounding rather than the
/// scalar multiply followed by add's two, so finite results are checked with a
/// mixed absolute/relative tolerance rather than bitwise equality.
#[inline(always)]
pub fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot product operands must have equal lengths"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: the helper only reads within the equally sized input slices.
        return unsafe { dot_slice_avx2_fma(a, b) };
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    dot_slice_portable(a, b)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
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

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
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

    #[inline(always)]
    pub fn matvec_into(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(self.cols, v.len());
        assert_eq!(self.rows, out.len());
        let v_ptr = v.as_ptr();
        let data_ptr = self.data.as_ptr();
        for i in 0..self.rows {
            unsafe {
                let row_ptr = data_ptr.add(i * self.cols);
                out[i] = if self.cols == 256 {
                    dot_256_raw(row_ptr, v_ptr)
                } else if self.cols == 128 {
                    dot_128_raw(row_ptr, v_ptr)
                } else if self.cols == 32 {
                    dot_32_raw(row_ptr, v_ptr)
                } else {
                    let mut sum = 0.0f32;
                    for j in 0..self.cols {
                        sum += *row_ptr.add(j) * *v_ptr.add(j);
                    }
                    sum
                };
            }
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

    pub fn invert(&self) -> Result<Matrix, String> {
        if self.rows != self.cols {
            return Err("Cannot invert non-square matrix".to_string());
        }
        let n = self.rows;
        let mut augmented = vec![0.0f64; n * 2 * n];

        for i in 0..n {
            for j in 0..n {
                augmented[i * 2 * n + j] = self.data[i * n + j] as f64;
            }
            augmented[i * 2 * n + n + i] = 1.0f64;
        }

        for i in 0..n {
            let mut pivot_row = i;
            let mut max_val = augmented[i * 2 * n + i].abs();
            for k in (i + 1)..n {
                let val = augmented[k * 2 * n + i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = k;
                }
            }

            if max_val < 1e-12 {
                return Err("Matrix is singular and cannot be inverted".to_string());
            }

            if pivot_row != i {
                for col in 0..(2 * n) {
                    let tmp = augmented[i * 2 * n + col];
                    augmented[i * 2 * n + col] = augmented[pivot_row * 2 * n + col];
                    augmented[pivot_row * 2 * n + col] = tmp;
                }
            }

            let pivot = augmented[i * 2 * n + i];
            for col in 0..(2 * n) {
                augmented[i * 2 * n + col] /= pivot;
            }

            for row in 0..n {
                if row != i {
                    let factor = augmented[row * 2 * n + i];
                    if factor.abs() > 1e-12 {
                        for col in 0..(2 * n) {
                            let sub = factor * augmented[i * 2 * n + col];
                            augmented[row * 2 * n + col] -= sub;
                        }
                    }
                }
            }
        }

        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.data[i * n + j] = augmented[i * 2 * n + n + j] as f32;
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
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn softmax(v: &Vector) -> Vector {
    let max = v.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.data.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Vector {
        data: exps.iter().map(|x| x / (sum + 1e-8)).collect(),
    }
}
