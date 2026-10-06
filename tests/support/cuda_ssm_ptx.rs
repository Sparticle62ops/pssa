//! CPU-only execution of the actual embedded PTX against an ordered SSM oracle.
//! These tests run without the cuda feature or a driver. GPU parity tests remain
//! necessary for JIT behavior, asynchronous copies, and real device scheduling.
#[path = "ptx_emulator.rs"]
mod ptx_emulator;
use ptx_emulator::{
    Arg::{Buffer as B, F32 as F, U32 as U},
    Machine,
};

const PTX: &str = include_str!("../../src/cuda/stages.ptx");

struct Ssm {
    l: usize,
    dm: usize,
    ds: usize,
    delta: Vec<f32>,
    raw: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    rates: Vec<f32>,
    deriv: Vec<f32>,
    x: Vec<f32>,
    states: Vec<f32>,
    a: Vec<f32>,
    bar_b: Vec<f32>,
    y: Vec<f32>,
    gz: Vec<f32>,
    gy: Vec<f32>,
    future: Vec<f32>,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    ga: Vec<f32>,
    gx: Vec<f32>,
}

impl Ssm {
    fn new(l: usize, dm: usize, ds: usize) -> Self {
        let stride = dm * ds;
        let mut s = Self {
            l,
            dm,
            ds,
            delta: (0..l * dm).map(|i| 0.2 + 0.03 * (i % 11) as f32).collect(),
            raw: (0..l * dm).map(|i| -0.8 + 0.17 * (i % 13) as f32).collect(),
            b: (0..l * ds)
                .map(|i| 0.03 * ((i % 17) as f32 - 7.0))
                .collect(),
            c: (0..l * ds)
                .map(|i| 0.04 * ((i % 19) as f32 - 8.0))
                .collect(),
            rates: (0..stride).map(|i| -0.2 - 0.02 * i as f32).collect(),
            deriv: (0..stride).map(|i| -0.3 - 0.05 * i as f32).collect(),
            x: (0..l * dm)
                .map(|i| 0.09 * ((i % 23) as f32 - 9.0))
                .collect(),
            states: vec![0.0; (l + 1) * stride],
            a: vec![0.0; l * stride],
            bar_b: vec![0.0; l * stride],
            y: vec![0.0; l * dm],
            gz: (0..l * dm).map(|i| 0.02 * ((i % 7) as f32 - 2.0)).collect(),
            gy: (0..l * dm).map(|i| 0.03 * ((i % 5) as f32 - 3.0)).collect(),
            future: vec![0.0; l * stride],
            gd: vec![0.0; l * dm],
            gb: vec![0.0; l * ds],
            gc: vec![0.0; l * ds],
            ga: vec![0.0; l * stride],
            gx: vec![0.0; l * dm],
        };
        for idx in 0..stride {
            s.states[idx] = 0.01 * (idx as f32 - 2.0);
        }
        for t in 0..l {
            for i in 0..dm {
                for j in 0..ds {
                    let idx = i * ds + j;
                    let a = (s.delta[t * dm + i] * s.rates[idx]).exp();
                    let b = s.delta[t * dm + i] * s.b[t * ds + j];
                    s.a[t * stride + idx] = a;
                    s.bar_b[t * stride + idx] = b;
                    let h = a * s.states[t * stride + idx] + b * s.x[t * dm + i];
                    s.states[(t + 1) * stride + idx] = h;
                    s.y[t * dm + i] += h * s.c[t * ds + j];
                }
            }
        }
        let mut future = vec![0.0; stride];
        for t in (0..l).rev() {
            s.future[(l - 1 - t) * stride..(l - t) * stride].copy_from_slice(&future);
            for i in 0..dm {
                let gy = s.gz[t * dm + i] * s.scale() + s.gy[t * dm + i];
                for j in 0..ds {
                    let idx = i * ds + j;
                    let q = gy * s.c[t * ds + j] + future[idx];
                    let a = s.a[t * stride + idx];
                    let prev = s.states[t * stride + idx];
                    s.gc[t * ds + j] += gy * s.states[(t + 1) * stride + idx];
                    s.ga[t * stride + idx] = q * (s.delta[t * dm + i] * a) * prev * s.deriv[idx];
                    s.gd[t * dm + i] +=
                        q * (s.rates[idx] * a * prev + s.b[t * ds + j] * s.x[t * dm + i]);
                    s.gb[t * ds + j] += q * (s.delta[t * dm + i] * s.x[t * dm + i]);
                    s.gx[t * dm + i] += q * s.bar_b[t * stride + idx];
                    future[idx] = q * a;
                }
                s.gd[t * dm + i] *= pssa::linalg::sigmoid(s.raw[t * dm + i]);
            }
        }
        s
    }
    fn scale(&self) -> f32 {
        1.0 / (self.ds as f32).sqrt()
    }
    fn machine(&self) -> Machine {
        let mut m = Machine::default();
        for (name, data) in [
            ("delta", &self.delta),
            ("raw", &self.raw),
            ("b", &self.b),
            ("c", &self.c),
            ("rates", &self.rates),
            ("deriv", &self.deriv),
            ("x", &self.x),
            ("states", &self.states),
            ("a", &self.a),
            ("bar_b", &self.bar_b),
            ("gz", &self.gz),
            ("gy", &self.gy),
            ("future", &self.future),
        ] {
            m.put(name, data);
        }
        for (name, len) in [
            ("gd", self.l * self.dm),
            ("gb", self.l * self.ds),
            ("gc", self.l * self.ds),
            ("ga", self.l * self.dm * self.ds),
            ("gx", self.l * self.dm),
        ] {
            // Nonzero seeds ensure local outputs overwrite stale tape contents.
            m.put(name, &vec![123.0; len]);
        }
        m
    }
    fn local(&self, m: &mut Machine) {
        // Positional ABI matches stages.rs::ssm_backward, including gB then gC.
        m.launch(
            PTX,
            "ssm_backward_local",
            &[
                B("delta"),
                B("raw"),
                B("b"),
                B("c"),
                B("rates"),
                B("deriv"),
                B("x"),
                B("states"),
                B("a"),
                B("bar_b"),
                B("future"),
                B("gz"),
                B("gy"),
                B("gd"),
                B("gb"),
                B("gc"),
                B("ga"),
                B("gx"),
                U(self.l as u32),
                U(self.dm as u32),
                U(self.ds as u32),
                U((self.dm * self.ds) as u32),
                F(self.scale()),
            ],
            // Small blocks deliberately exercise cross-block row ownership.
            (self.l.div_ceil(32), 1),
            32,
        );
    }
    fn maps(&self, m: &mut Machine) {
        let n = self.l * self.dm * self.ds;
        m.zeros("rev_a", n);
        m.zeros("rev_b", n);
        m.launch(
            PTX,
            "ssm_backward_maps",
            &[
                B("a"),
                B("c"),
                B("gz"),
                B("gy"),
                B("rev_a"),
                B("rev_b"),
                U(self.l as u32),
                U(self.dm as u32),
                U(self.ds as u32),
                F(self.scale()),
            ],
            (n.div_ceil(32), 1),
            32,
        );
    }
    fn check(&self, m: &Machine) {
        for (name, expected) in [
            ("gd", &self.gd),
            ("gb", &self.gb),
            ("gc", &self.gc),
            ("ga", &self.ga),
            ("gx", &self.gx),
        ] {
            close(name, &m.get(name), expected);
        }
        // These are the exact [l,d_s]^T * [l,d_m] reductions feeding w_b/w_c.
        for (name, expected) in [("gb", &self.gb), ("gc", &self.gc)] {
            let increment = pssa::backend::gemm_tn_cpu(expected, &self.x, self.l, self.ds, self.dm);
            let magnitude = increment.iter().map(|v| v.abs()).fold(0.0, f32::max);
            assert!(magnitude > 1e-6, "fixture must exercise w from {name}");
            let mut actual_w = vec![0.125; self.ds * self.dm];
            let mut expected_w = actual_w.clone();
            for count in 1..=2 {
                // shared parameter gradients must accumulate, not assign
                pssa::backend::gemm_tn_cpu_accumulate_into(
                    &m.get(name),
                    &self.x,
                    self.l,
                    self.ds,
                    self.dm,
                    &mut actual_w,
                )
                .unwrap();
                pssa::backend::gemm_tn_cpu_accumulate_into(
                    expected,
                    &self.x,
                    self.l,
                    self.ds,
                    self.dm,
                    &mut expected_w,
                )
                .unwrap();
                let error = actual_w
                    .iter()
                    .zip(&expected_w)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0, f32::max);
                assert!(
                    error / (count as f32 * magnitude) < 1e-3,
                    "w from {name}: relative error {}",
                    error / (count as f32 * magnitude)
                );
                close(&format!("w from {name}"), &actual_w, &expected_w);
            }
        }
    }
}

fn close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 3e-6 + 3e-5 * e.abs(),
            "{name}[{i}]: {a} != {e}"
        );
    }
}

fn scan(
    m: &mut Machine,
    a: &str,
    b: &str,
    len: usize,
    stride: usize,
    capacity: usize,
    tag: &str,
) -> (String, String) {
    let tiles = len.div_ceil(capacity);
    let oa = format!("{tag}-oa");
    let ob = format!("{tag}-ob");
    let sa = format!("{tag}-sa");
    let sb = format!("{tag}-sb");
    m.zeros(&oa, len * stride);
    m.zeros(&ob, len * stride);
    m.zeros(&sa, tiles * stride);
    m.zeros(&sb, tiles * stride);
    m.launch(
        PTX,
        "affine_scan",
        &[
            B(a),
            B(b),
            B(&oa),
            B(&ob),
            B(&sa),
            B(&sb),
            U(len as u32),
            U(stride as u32),
            U(capacity as u32),
        ],
        (stride, tiles),
        capacity,
    );
    if tiles > 1 {
        let (pa, pb) = scan(
            m,
            &sa,
            &sb,
            tiles,
            stride,
            tiles.next_power_of_two().min(1024),
            &format!("{tag}-prefix"),
        );
        m.launch(
            PTX,
            "scan_apply",
            &[
                B(&oa),
                B(&ob),
                B(&pa),
                B(&pb),
                U(len as u32),
                U(stride as u32),
                U(capacity as u32),
            ],
            ((len * stride).div_ceil(32), 1),
            32,
        );
    }
    (oa, ob)
}

#[test]
fn cuda_ssm_ptx_local_gradients_match_ordered_cpu_without_gpu() {
    for (l, dm, ds) in [(1, 1, 1), (3, 3, 2), (7, 2, 3), (33, 3, 2)] {
        let s = Ssm::new(l, dm, ds);
        let mut m = s.machine();
        s.local(&mut m);
        s.check(&m);
        s.local(&mut m);
        s.check(&m); // reused output buffers must not grow/stay stale
    }
}

#[test]
fn cuda_ssm_ptx_reverse_maps_match_cpu_without_gpu() {
    let s = Ssm::new(7, 3, 2);
    let mut m = s.machine();
    s.maps(&mut m);
    let stride = s.dm * s.ds;
    let mut a = vec![0.0; s.l * stride];
    let mut b = a.clone();
    for u in 0..s.l {
        let t = s.l - 1 - u;
        for i in 0..s.dm {
            for j in 0..s.ds {
                let idx = i * s.ds + j;
                a[u * stride + idx] = s.a[t * stride + idx];
                b[u * stride + idx] = s.a[t * stride + idx]
                    * (s.gz[t * s.dm + i] * s.scale() + s.gy[t * s.dm + i])
                    * s.c[t * s.ds + j];
            }
        }
    }
    close("reverse A", &m.get("rev_a"), &a);
    close("reverse B", &m.get("rev_b"), &b);
}

#[test]
fn cuda_ssm_ptx_tiled_exclusive_scan_matches_cpu_without_gpu() {
    // Forced small tiles cover recursive composition cheaply; 1025 also uses
    // production's maximum 1024-thread tile and a padded final tile.
    for (len, stride, capacity) in [(1, 1, 1), (3, 2, 4), (7, 3, 8), (33, 2, 8), (1025, 2, 1024)] {
        let mut m = Machine::default();
        let a: Vec<_> = (0..len * stride)
            .map(|i| 0.8 + 0.01 * (i % 11) as f32)
            .collect();
        let b: Vec<_> = (0..len * stride)
            .map(|i| -0.2 + 0.03 * (i % 17) as f32)
            .collect();
        m.put("a", &a);
        m.put("b", &b);
        let (oa, ob) = scan(&mut m, "a", "b", len, stride, capacity, "scan");
        let mut ea = vec![0.0; len * stride];
        let mut eb = ea.clone();
        for c in 0..stride {
            let (mut pa, mut pb) = (1.0, 0.0);
            for t in 0..len {
                ea[t * stride + c] = pa;
                eb[t * stride + c] = pb;
                pb = a[t * stride + c] * pb + b[t * stride + c];
                pa *= a[t * stride + c];
            }
        }
        close("exclusive A", &m.get(&oa), &ea);
        close("exclusive B", &m.get(&ob), &eb);
    }
}

#[test]
fn cuda_ssm_ptx_forward_and_backward_match_cpu_without_gpu() {
    for (l, dm, ds, capacity) in [(1, 1, 1, 1), (7, 3, 2, 8), (33, 3, 2, 8), (16, 32, 4, 16)] {
        let s = Ssm::new(l, dm, ds);
        let stride = dm * ds;
        let mut m = s.machine();
        m.zeros("a", l * stride);
        m.zeros("bar_b", l * stride);
        m.zeros("input_b", l * stride);
        m.launch(
            PTX,
            "ssm_prepare",
            &[
                B("delta"),
                B("b"),
                B("x"),
                B("rates"),
                B("a"),
                B("bar_b"),
                B("input_b"),
                U(l as u32),
                U(dm as u32),
                U(ds as u32),
            ],
            ((l * stride).div_ceil(32), 1),
            32,
        );
        close("bar A", &m.get("a"), &s.a);
        close("bar B", &m.get("bar_b"), &s.bar_b);
        let (oa, ob) = scan(&mut m, "a", "input_b", l, stride, capacity, "forward");
        m.put("initial", &s.states[..stride]);
        let mut initial_states = vec![0.0; (l + 1) * stride];
        initial_states[..stride].copy_from_slice(&s.states[..stride]);
        m.put("states", &initial_states);
        m.zeros("y", l * dm);
        m.launch(
            PTX,
            "ssm_materialize",
            &[
                B("initial"),
                B("a"),
                B("bar_b"),
                B("x"),
                B(&oa),
                B(&ob),
                B("c"),
                B("states"),
                B("y"),
                U(l as u32),
                U(dm as u32),
                U(ds as u32),
            ],
            ((l * dm).div_ceil(32), 1),
            32,
        );
        close("states", &m.get("states"), &s.states);
        close("readout", &m.get("y"), &s.y);
        s.maps(&mut m);
        let (_, future) = scan(&mut m, "rev_a", "rev_b", l, stride, capacity, "reverse");
        m.put("future", &m.get(&future));
        close("future adjoint", &m.get("future"), &s.future);
        s.local(&mut m);
        s.check(&m);
    }
}
