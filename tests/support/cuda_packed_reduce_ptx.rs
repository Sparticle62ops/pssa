//! Execute the ordered rate-gradient reduction PTX, including lane ownership.
#[path = "ptx_emulator.rs"]
#[allow(dead_code)] // Shared interpreter API; this module uses a subset.
mod ptx_emulator;
use ptx_emulator::{
    Arg::{Buffer as B, U32 as U},
    Machine,
};
const PTX: &str = include_str!("../../src/cuda/packed.ptx");

#[test]
fn packed_bc_reduction_keeps_latent_order_and_does_not_touch_padding() {
    let (d, s, rows) = (3, 2, 6);
    let offsets = [3, u32::MAX, 1];
    let lengths = [2, 0, 2];
    let mut b = vec![123.0; rows * d * s];
    let mut c = b.clone();
    let mut gb = vec![123.0; rows * s];
    let mut gc = gb.clone();
    for row in 1..5 {
        for j in 0..s {
            let bi = [1e10, -1e10, (row + j) as f32];
            let ci = [1e10, (row + j) as f32, -1e10];
            gb[row * s + j] = 0.0;
            gc[row * s + j] = 0.0;
            for i in 0..d {
                let index = row * d * s + i * s + j;
                b[index] = bi[i];
                c[index] = ci[i];
                gb[row * s + j] += bi[i];
                gc[row * s + j] += ci[i];
            }
        }
    }
    let mut m = Machine::default();
    m.put("b", &b);
    m.put("c", &c);
    m.put("gb", &vec![123.0; rows * s]);
    m.put("gc", &vec![123.0; rows * s]);
    m.put_u32("offsets", &offsets);
    m.put_u32("lengths", &lengths);
    let args = [
        B("b"),
        B("c"),
        B("gb"),
        B("gc"),
        B("offsets"),
        B("lengths"),
        U(d as u32),
        U(s as u32),
    ];
    for _ in 0..2 {
        m.launch(PTX, "packed_reduce_bc", &args, (2, 3), 8);
        assert_eq!(m.get("gb"), gb);
        assert_eq!(m.get("gc"), gc);
    }
    assert_eq!(m.get("b"), b);
    assert_eq!(m.get("c"), c);
}

#[test]
fn packed_rate_reduction_is_reverse_ordered_ragged_and_lane_local() {
    let stride = 3;
    let offsets = [2, u32::MAX, 0];
    let lengths = [3, 0, 2];
    let grad = [
        1e10, 0.1, -0.2, 1.0, 0.2, 0.3, 1e10, 0.3, 0.4, -1e10, 0.4, 0.5, 1.0, 0.5, 0.6,
    ];
    let mut expected = vec![0.0f32; offsets.len() * stride];
    for (lane, (&off, &len)) in offsets.iter().zip(&lengths).enumerate() {
        for t in (0..len as usize).rev() {
            for channel in 0..stride {
                expected[lane * stride + channel] += grad[(off as usize + t) * stride + channel];
            }
        }
    }
    // Lane zero: forward [1e10, -1e10, 1] sums to 1, but the required
    // reverse-token reduction sums to 0. A forward-loop mutation must fail.
    assert_eq!(expected[0], 0.0);
    assert_eq!((grad[6] + grad[9]) + grad[12], 1.0);
    let mut m = Machine::default();
    m.put("grad", &grad);
    m.put("out", &[123.0; 9]);
    m.put_u32("offsets", &offsets);
    m.put_u32("lengths", &lengths);
    m.launch(
        PTX,
        "packed_reduce_rate",
        &[
            B("grad"),
            B("out"),
            B("offsets"),
            B("lengths"),
            U(stride as u32),
        ],
        (1, 3),
        8,
    );
    assert_eq!(m.get("out"), expected);
}

#[test]
fn packed_rate_reduction_training_stride_boundary_channels_and_guard() {
    let stride = 3584 * 16;
    let mut grad = vec![0.25; 3 * stride];
    grad[stride - 1] = 1.0;
    grad[2 * stride - 1] = -0.3;
    let mut m = Machine::default();
    m.put("grad", &grad);
    m.put("out", &vec![123.0; 3 * stride]);
    m.put_u32("offsets", &[1, u32::MAX, 0]);
    m.put_u32("lengths", &[2, 0, 1]);
    let args = [
        B("grad"),
        B("out"),
        B("offsets"),
        B("lengths"),
        U(stride as u32),
    ];
    let selected = [
        (0, 0),
        (0, stride - 1),
        (1, 0),
        (1, stride - 1),
        (2, 0),
        (2, stride - 1),
        (0, stride),
        (2, stride + 1),
    ];
    m.launch_threads(PTX, "packed_reduce_rate", &args, &selected, 256);
    let out = m.get("out");
    assert_eq!(out[0], 0.5);
    assert_eq!(out[stride - 1], 0.25 + -0.3);
    assert_eq!(out[stride], 0.0);
    assert_eq!(out[2 * stride - 1], 0.0);
    assert_eq!(out[2 * stride], 0.25);
    assert_eq!(out[3 * stride - 1], 1.0);
    assert_eq!(out[1], 123.0);
}
