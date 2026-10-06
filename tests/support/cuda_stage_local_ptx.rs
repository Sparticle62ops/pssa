//! CPU replay of the remaining token-local arithmetic stage kernels.
#[path = "ptx_emulator.rs"]
mod ptx_emulator;
use ptx_emulator::{
    Arg::{Buffer as B, U32 as U},
    Machine,
};
const PTX: &str = include_str!("../../src/cuda/stages.ptx");

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
