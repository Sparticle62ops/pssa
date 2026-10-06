//! Driver-free validation of every stage's flattened PTX index and allocation.
//! PTX element indices are u32, but byte offsets use mul.wide.u32 (u64).
//! Reject unsupported shapes before uploading, allocating, or launching work.

pub(super) fn elems(value: usize, what: &str) -> Result<u32, String> {
    let index = u32::try_from(value).map_err(|_| {
        format!(
            "CUDA {what}: {value} elements exceeds the u32 PTX index limit {}; refusing launch",
            u32::MAX
        )
    })?;
    if value
        .checked_mul(4)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(format!(
            "CUDA {what}: f32 allocation byte size overflow; refusing launch"
        ));
    }
    Ok(index)
}

pub(super) fn product(a: usize, b: usize, what: &str) -> Result<usize, String> {
    let value = a
        .checked_mul(b)
        .ok_or_else(|| format!("CUDA {what}: {a} * {b} size overflow; refusing launch"))?;
    elems(value, what)?;
    Ok(value)
}

pub(super) fn lengths(stage: &str, buffers: &[(&str, usize, usize)]) -> Result<(), String> {
    for &(name, actual, expected) in buffers {
        if actual != expected {
            return Err(format!(
                "CUDA {stage}: {name} has {actual} elements, expected {expected}; refusing launch"
            ));
        }
    }
    Ok(())
}

pub(super) fn scan_capacity(len: usize) -> usize {
    if len <= 512 {
        len.max(1).next_power_of_two()
    } else {
        1024
    }
}

pub(super) fn scan_shape(len: usize, stride: usize) -> Result<(usize, usize), String> {
    if len == 0 || stride == 0 {
        return Err("CUDA affine scan dimensions must be positive; refusing launch".into());
    }
    product(len, stride, "affine scan elements")?;
    let capacity = scan_capacity(len);
    let tiles = len.div_ceil(capacity);
    // Supported sm_50+ devices: grid.x <= 2^31-1, grid.y <= 65535.
    // Static shared arrays hold 1024 floats each (8192 bytes total).
    if stride > i32::MAX as usize || tiles > 65535 {
        return Err(format!(
            "CUDA affine scan grid ({stride}, {tiles}) exceeds grid limits (2147483647, 65535); refusing launch"
        ));
    }
    product(tiles, stride, "affine scan summaries")?;
    Ok((capacity, tiles))
}

pub(super) struct SsmShape {
    pub stride: usize,
    pub token_m: usize,
    pub token_s: usize,
    pub token_state: usize,
    pub state_len: usize,
}

impl SsmShape {
    pub fn new(len: usize, dm: usize, ds: usize) -> Result<Self, String> {
        if len == 0 || dm == 0 || ds == 0 {
            return Err("CUDA SSM dimensions must be positive; refusing launch".into());
        }
        let stride = product(dm, ds, "SSM stride")?;
        let token_m = product(len, dm, "SSM latent tape")?;
        let token_s = product(len, ds, "SSM projection tape")?;
        let token_state = product(len, stride, "SSM state maps")?;
        let rows = len
            .checked_add(1)
            .ok_or("CUDA SSM state row count overflow; refusing launch")?;
        // materialize and backward_local index h[t+1], not only h[t].
        let state_len = product(rows, stride, "SSM states including terminal row")?;
        scan_shape(len, stride)?;
        Ok(Self {
            stride,
            token_m,
            token_s,
            token_state,
            state_len,
        })
    }
}

pub(super) struct MemoryShape {
    pub x: usize,
    pub query: usize,
    pub value: usize,
    pub weights: usize,
    pub keys: usize,
    pub values: usize,
    pub w_query: usize,
    pub w_gate: usize,
    pub w_proj: usize,
}

impl MemoryShape {
    pub fn new(
        len: usize,
        dm: usize,
        dk: usize,
        dv: usize,
        capacity: usize,
        count: usize,
        tau: f32,
    ) -> Result<Self, String> {
        if len == 0
            || capacity == 0
            || [dm, dk, dv]
                .iter()
                .any(|&d| d == 0 || d > i32::MAX as usize)
        {
            return Err(
                "CUDA memory dimensions must be positive and GEMM widths fit i32; refusing launch"
                    .into(),
            );
        }
        elems(len, "memory length")?;
        elems(capacity, "memory capacity")?;
        if count > capacity || !tau.is_finite() || tau <= 0.0 {
            return Err(format!(
                "CUDA memory invalid count/capacity ({count}/{capacity}) or temperature ({tau}); refusing launch"
            ));
        }
        Ok(Self {
            x: product(len, dm, "memory input")?,
            query: product(len, dk, "memory query")?,
            value: product(len, dv, "memory value")?,
            weights: product(len, capacity, "memory weights")?,
            keys: product(capacity, dk, "memory keys")?,
            values: product(capacity, dv, "memory bank values")?,
            w_query: product(dk, dm, "memory query weights")?,
            w_gate: product(dm, dm, "memory gate weights")?,
            w_proj: product(dm, dv, "memory projection weights")?,
        })
    }
}

pub(super) fn gemm_shape(
    m: usize,
    k: usize,
    n: usize,
    a: usize,
    b: usize,
    out: usize,
) -> Result<(), String> {
    if [m, k, n].iter().any(|&d| d == 0 || d > i32::MAX as usize) {
        return Err(
            "CUDA stage GEMM dimensions must be positive and fit i32; refusing launch".into(),
        );
    }
    lengths(
        "stage GEMM",
        &[
            ("input", a, product(m, k, "stage GEMM input")?),
            (
                "weights [output,input]",
                b,
                product(n, k, "stage GEMM weights")?,
            ),
            ("output", out, product(m, n, "stage GEMM output")?),
        ],
    )
}

pub(super) fn optimizer_shape(
    data: usize,
    grad: usize,
    moment: usize,
    variance: usize,
) -> Result<u32, String> {
    if grad == 0 {
        return Err("CUDA Adam tensor must be nonempty; refusing launch".into());
    }
    let len = elems(grad, "Adam tensor")?;
    lengths(
        "Adam tensor",
        &[
            ("data", data, grad),
            ("moment", moment, grad),
            ("variance", variance, grad),
        ],
    )?;
    Ok(len)
}

pub(super) fn cap_shape(len: usize, width: usize, cap: f32) -> Result<(u32, u32), String> {
    if width == 0 || len % width != 0 || !cap.is_finite() || cap <= 0.0 {
        return Err("CUDA memory cap requires whole rows, positive width and finite positive cap; refusing launch".into());
    }
    elems(len, "memory cap elements")?;
    Ok((
        elems(len / width, "memory cap rows")?,
        elems(width, "memory cap width")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_shapes_and_offsets_above_two_gib_are_valid() {
        let s = SsmShape::new(64, 3584, 16).unwrap();
        assert_eq!(
            (s.stride, s.token_state, s.state_len),
            (57344, 3670016, 3727360)
        );
        let packed = MemoryShape::new(32 * 64, 3584, 32, 3584, 512, 512, 0.7).unwrap();
        assert_eq!(packed.value, 7340032);
        let large = SsmShape::new(16384, 3584, 16).unwrap();
        assert!(large.token_state as u64 * 4 > i32::MAX as u64);
        assert!(large.state_len as u64 * 4 < u32::MAX as u64);
    }

    #[test]
    fn rejects_wrapped_maps_terminal_row_and_memory_products() {
        for (l, dm, ds, detail) in [
            (usize::MAX, 3584, 16, "overflow"),
            (1 << 20, 3584, 16, "u32 PTX index"),
            (65535, 65536, 1, "terminal row"),
            (0, 3584, 16, "positive"),
        ] {
            let err = SsmShape::new(l, dm, ds).err().unwrap();
            assert!(err.contains(detail), "{err}");
            assert!(err.contains("refusing launch"), "{err}");
        }
        assert!(MemoryShape::new(32, 3584, 32, 3584, usize::MAX, 1, 0.7).is_err());
        assert!(MemoryShape::new(1 << 24, 3584, 32, 3584, 512, 1, 0.7).is_err());
        assert!(MemoryShape::new(32, 3584, 32, 3584, 512, 513, 0.7).is_err());
        assert!(MemoryShape::new(32, 3584, 32, 3584, 512, 1, f32::NAN).is_err());
    }

    #[test]
    fn scan_grid_and_shared_capacity_are_bounded() {
        assert_eq!(scan_shape(1, 1).unwrap(), (1, 1));
        assert_eq!(scan_shape(513, 57344).unwrap(), (1024, 1));
        assert_eq!(scan_shape(1025, 57344).unwrap(), (1024, 2));
        assert!(
            scan_shape(65535 * 1024 + 1, 1)
                .unwrap_err()
                .contains("grid limits")
        );
        assert!(
            scan_shape(1, i32::MAX as usize + 1)
                .unwrap_err()
                .contains("grid limits")
        );
        assert!(scan_shape(1024, 1 << 23).is_err());
        for len in [1, 32, 64, 511, 512, 513, 1025, 1 << 20] {
            let width = scan_capacity(len);
            assert!(width.is_power_of_two() && width <= 1024 && width * 4 <= 4096);
        }
    }

    #[test]
    fn safeguard_allocations_are_validated_before_launch() {
        assert_eq!(cap_shape(512 * 3584, 3584, 512.0).unwrap(), (512, 3584));
        assert!(cap_shape(512 * 3584 - 1, 3584, 512.0).is_err());
        assert!(cap_shape(16, 0, 512.0).is_err());
        assert!(cap_shape(16, 4, f32::INFINITY).is_err());
        assert!(optimizer_shape(105000000, 105000000, 105000000, 105000000).is_ok());
        assert!(optimizer_shape(16, 17, 17, 17).is_err());
        assert!(optimizer_shape(17, 17, 16, 17).is_err());
        assert!(optimizer_shape(17, 17, 17, 16).is_err());
        assert!(optimizer_shape(0, 0, 0, 0).is_err());
        assert!(optimizer_shape(usize::MAX, usize::MAX, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn query_scratch_is_sized_by_key_width_and_gemm_rejects_short_buffers() {
        let s = MemoryShape::new(4, 3, 4, 2, 3, 2, 0.7).unwrap();
        assert_eq!((s.x, s.query, s.value), (12, 16, 8));
        assert!(gemm_shape(4, 3, 4, 12, 12, s.query).is_ok());
        let err = gemm_shape(4, 3, 4, 12, 12, s.x).unwrap_err();
        assert!(err.contains("output has 12 elements, expected 16"), "{err}");
        assert!(gemm_shape(4, 3, 4, 11, 12, 16).is_err());
        assert!(gemm_shape(4, 3, 4, 12, 11, 16).is_err());
    }
}
