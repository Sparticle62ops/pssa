//! CPU replay of the remaining token-local arithmetic stage kernels.
#[path = "ptx_emulator.rs"]
mod ptx_emulator;
use ptx_emulator::{
    Arg::{Buffer as B, U32 as U},
    Machine,
};
const PTX: &str = include_str!("../../src/cuda/stages.ptx");

#[test]
fn cuda_token_local_ptx_training_stride_extreme_activations() {
    let (rows, dm, width) = (32usize, 3584usize, 1024usize);
    let values = [
        -f32::MAX,
        -104.0,
        -90.0,
        -88.0,
        -87.5,
        -80.0,
        -40.0,
        -24.0,
        -20.0,
        -16.0,
        -10.0,
        0.0,
        20.0,
        21.0,
        90.0,
        f32::MAX,
    ];
    let len = rows * dm;
    let mut m = Machine::default();
    let mut input = vec![0.0; len];
    // Keep the actual flattened token/channel stride, including both sides of
    // a launch-block boundary, but do not replay all 114688 identical lanes.
    let probes = values
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .chain([
            (dm - 1, -24.0),
            (dm, -20.0),
            (16 * dm, 90.0),
            (len - 1, -80.0),
        ])
        .collect::<Vec<_>>();
    for &(i, x) in &probes {
        input[i] = x;
    }
    for (kernel, cpu) in [
        ("sigmoid_in_place", pssa::linalg::sigmoid as fn(f32) -> f32),
        (
            "softplus_in_place",
            pssa::linalg::softplus as fn(f32) -> f32,
        ),
    ] {
        m.put("a", &input);
        for &(i, _) in &probes {
            m.launch_thread(PTX, kernel, &[B("a"), U(len as u32)], i, width);
        }
        m.launch_thread(PTX, kernel, &[B("a"), U(len as u32)], len, width);
        let actual = m.get("a");
        for &(i, x) in &probes {
            let expected = cpu(x);
            let got = actual[i];
            assert!(
                got.is_finite()
                    && (got - expected).abs() <= 3e-6 * expected.abs() + 2.0 * f32::from_bits(1),
                "{kernel}[token={},channel={}], x={x}: {got} != {expected}",
                i / dm,
                i % dm
            );
            if expected > 0.0 {
                assert!(
                    got > 0.0,
                    "{kernel}[{i}] lost a representable positive tail at {x}"
                );
            }
        }
    }
}

#[test]
fn cuda_token_local_ptx_bounds_and_memory_gate_math() {
    let (len, width) = (7usize, 4usize);
    let a = vec![-21., -1., 0., 0.25, 1., 3., 21.];
    let b = vec![0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
    let gate = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    let mut m = Machine::default();
    m.put("a", &a);
    m.put("b", &b);
    m.put("gate", &gate);
    m.zeros("out", len);
    m.zeros("gout", len);
    let grid = (len.div_ceil(width), 1);
    m.launch(
        PTX,
        "sigmoid_mul",
        &[B("gate"), B("b"), B("out"), U(len as u32)],
        grid,
        width,
    );
    let expected = gate
        .iter()
        .zip(&b)
        .map(|(&g, &p)| g * p)
        .collect::<Vec<_>>();
    assert_eq!(m.get("out"), expected);
    m.launch(
        PTX,
        "memory_backward_local",
        &[
            B("a"),
            B("gate"),
            B("b"),
            B("out"),
            B("gout"),
            U(len as u32),
        ],
        grid,
        width,
    );
    assert_eq!(
        m.get("out"),
        a.iter()
            .zip(&gate)
            .map(|(&z, &g)| z * g)
            .collect::<Vec<_>>()
    );
    for (i, got) in m.get("gout").iter().enumerate() {
        assert!((got - a[i] * b[i] * gate[i] * (1. - gate[i])).abs() < 1e-6);
    }
    m.launch(
        PTX,
        "add_in_place",
        &[B("a"), B("b"), U(len as u32)],
        grid,
        width,
    );
    assert_eq!(
        m.get("a"),
        a.iter().zip(&b).map(|(&x, &y)| x + y).collect::<Vec<_>>()
    );
    m.put("a", &a);
    m.launch(
        PTX,
        "sigmoid_in_place",
        &[B("a"), U(len as u32)],
        grid,
        width,
    );
    for (&got, &x) in m.get("a").iter().zip(&a) {
        assert!((got - pssa::linalg::sigmoid(x)).abs() < 1e-6);
    }
    m.put("a", &a);
    m.launch(
        PTX,
        "softplus_in_place",
        &[B("a"), U(len as u32)],
        grid,
        width,
    );
    for (&got, &x) in m.get("a").iter().zip(&a) {
        assert!((got - pssa::linalg::softplus(x)).abs() < 1e-6);
    }
}
