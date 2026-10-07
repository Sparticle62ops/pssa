//! Driver-free execution of the actual packed SSM PTX against ordered CPU loops.
//! Metadata words are u32; pointer parameters are u64. Stable lane IDs need not
//! follow packed token order. GPU/JIT/copy scheduling still requires GPU tests.
#[path = "ptx_emulator.rs"]
#[allow(dead_code)] // Shared interpreter API; this module uses a subset.
mod ptx_emulator;
use ptx_emulator::{
    Arg::{Buffer as B, F32 as F, U32 as U},
    Machine,
};

const PTX: &str = include_str!("../../src/cuda/stages.ptx");
const REDUCE_PTX: &str = include_str!("../../src/cuda/packed.ptx");
const STALE: f32 = 123.0;
const BLOCK: usize = 8;

struct Packed {
    dm: usize,
    ds: usize,
    chunk: usize,
    rows: usize,
    offsets: Vec<u32>,
    lengths: Vec<u32>,
    initial: Vec<f32>,
    delta: Vec<f32>,
    raw: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    rates: Vec<f32>,
    deriv: Vec<f32>,
    x: Vec<f32>,
    a: Vec<f32>,
    bar_b: Vec<f32>,
    input_b: Vec<f32>,
    gz: Vec<f32>,
    gy: Vec<f32>,
    prefix_a: Vec<f32>,
    prefix_b: Vec<f32>,
    states: Vec<f32>,
    carries: Vec<f32>,
    y: Vec<f32>,
    rev_a: Vec<f32>,
    rev_b: Vec<f32>,
    future: Vec<f32>,
    gd: Vec<f32>,
    gb: Vec<f32>,
    gc: Vec<f32>,
    ga: Vec<f32>,
    gx: Vec<f32>,
    token_gb: Vec<f32>,
    token_gc: Vec<f32>,
}

impl Packed {
    fn new(
        dm: usize,
        ds: usize,
        chunk: usize,
        rows: usize,
        offsets: &[u32],
        lengths: &[u32],
    ) -> Self {
        assert_eq!(offsets.len(), lengths.len());
        let stride = dm * ds;
        let n = rows * stride;
        let mut s = Self {
            dm,
            ds,
            chunk,
            rows,
            offsets: offsets.to_vec(),
            lengths: lengths.to_vec(),
            initial: (0..offsets.len() * stride)
                .map(|i| {
                    // Lane two models a reset; the others have distinct carries.
                    if i / stride == 2 {
                        0.0
                    } else {
                        0.07 * (i / stride + 1) as f32 + 0.01 * ((i % 11) as f32 - 5.0)
                    }
                })
                .collect(),
            delta: (0..rows * dm)
                .map(|i| 0.2 + 0.03 * (i % 11) as f32)
                .collect(),
            raw: (0..rows * dm)
                .map(|i| [-100.0, -90.0, -20.0, -0.8, 0.0, 0.5, 20.0, 90.0][i % 8])
                .collect(),
            b: (0..rows * ds)
                .map(|i| 0.03 * ((i % 17) as f32 - 7.0))
                .collect(),
            c: (0..rows * ds)
                .map(|i| 0.04 * ((i % 19) as f32 - 8.0))
                .collect(),
            rates: (0..stride).map(|i| -0.2 - 0.02 * (i % 23) as f32).collect(),
            deriv: (0..stride).map(|i| -0.3 - 0.05 * (i % 13) as f32).collect(),
            x: (0..rows * dm)
                .map(|i| 0.09 * ((i % 23) as f32 - 9.0))
                .collect(),
            a: vec![0.0; n],
            bar_b: vec![0.0; n],
            input_b: vec![0.0; n],
            gz: (0..rows * dm)
                .map(|i| 0.02 * ((i % 7) as f32 - 2.0))
                .collect(),
            gy: (0..rows * dm)
                .map(|i| 0.03 * ((i % 5) as f32 - 3.0))
                .collect(),
            prefix_a: vec![STALE; n],
            prefix_b: vec![STALE; n],
            states: vec![STALE; offsets.len() * (chunk + 1) * stride],
            carries: Vec::new(),
            y: vec![STALE; rows * dm],
            rev_a: vec![STALE; n],
            rev_b: vec![STALE; n],
            future: vec![STALE; n],
            gd: vec![STALE; rows * dm],
            gb: vec![STALE; rows * ds],
            gc: vec![STALE; rows * ds],
            ga: vec![STALE; n],
            gx: vec![STALE; rows * dm],
            token_gb: vec![STALE; n],
            token_gc: vec![STALE; n],
        };
        s.refresh();
        s
    }

    fn stride(&self) -> usize {
        self.dm * self.ds
    }

    fn scale(&self) -> f32 {
        1.0 / (self.ds as f32).sqrt()
    }

    fn refresh(&mut self) {
        let stride = self.stride();
        self.carries = self.initial.clone();
        self.states.fill(STALE);
        self.y.fill(STALE);
        for lane in 0..self.lengths.len() {
            let len = self.lengths[lane] as usize;
            if len == 0 {
                // An empty lane's offset deliberately need not be dereferenceable.
                continue;
            }
            assert!(len <= self.chunk);
            let offset = self.offsets[lane] as usize;
            assert!(offset + len <= self.rows);
            let state_base = lane * (self.chunk + 1) * stride;
            self.states[state_base..state_base + stride]
                .copy_from_slice(&self.initial[lane * stride..(lane + 1) * stride]);
            let mut pa = vec![1.0; stride];
            let mut pb = vec![0.0; stride];
            for t in 0..len {
                let row = offset + t;
                for i in 0..self.dm {
                    let di = row * self.dm + i;
                    self.y[di] = 0.0;
                    for j in 0..self.ds {
                        let channel = i * self.ds + j;
                        let index = row * stride + channel;
                        let a = (self.delta[di] * self.rates[channel]).exp();
                        let b = self.delta[di] * self.b[row * self.ds + j];
                        let input_b = b * self.x[di];
                        self.a[index] = a;
                        self.bar_b[index] = b;
                        self.input_b[index] = input_b;
                        self.prefix_a[index] = pa[channel];
                        self.prefix_b[index] = pb[channel];
                        pa[channel] = a * pa[channel];
                        pb[channel] = a * pb[channel] + input_b;
                        // Ordered recurrence, not a transcription of materialize.
                        let h = a * self.states[state_base + t * stride + channel] + input_b;
                        self.states[state_base + (t + 1) * stride + channel] = h;
                        self.carries[lane * stride + channel] = h;
                        self.y[di] += h * self.c[row * self.ds + j];
                    }
                }
            }
            let mut future = vec![0.0; stride];
            for t in (0..len).rev() {
                let row = offset + t;
                let reverse_row = offset + len - 1 - t;
                self.future[reverse_row * stride..(reverse_row + 1) * stride]
                    .copy_from_slice(&future);
                self.gb[row * self.ds..(row + 1) * self.ds].fill(0.0);
                self.gc[row * self.ds..(row + 1) * self.ds].fill(0.0);
                for i in 0..self.dm {
                    let di = row * self.dm + i;
                    let gy = self.gz[di] * self.scale() + self.gy[di];
                    self.gd[di] = 0.0;
                    self.gx[di] = 0.0;
                    for j in 0..self.ds {
                        let channel = i * self.ds + j;
                        let index = row * stride + channel;
                        let ri = reverse_row * stride + channel;
                        let a = self.a[index];
                        let r = gy * self.c[row * self.ds + j];
                        let q = r + future[channel];
                        self.rev_a[ri] = a;
                        self.rev_b[ri] = a * r;
                        let prev = self.states[state_base + t * stride + channel];
                        let next = self.states[state_base + (t + 1) * stride + channel];
                        self.ga[index] = q * (self.delta[di] * a) * prev * self.deriv[channel];
                        self.gd[di] += q
                            * (self.rates[channel] * a * prev
                                + self.b[row * self.ds + j] * self.x[di]);
                        self.token_gb[index] = q * (self.delta[di] * self.x[di]);
                        self.token_gc[index] = gy * next;
                        self.gb[row * self.ds + j] += self.token_gb[index];
                        self.gc[row * self.ds + j] += self.token_gc[index];
                        self.gx[di] += q * self.bar_b[index];
                        future[channel] = a * q;
                    }
                    self.gd[di] *= pssa::linalg::sigmoid(self.raw[di]);
                }
            }
        }
    }

    fn machine(&self) -> Machine {
        let mut m = Machine::default();
        for (name, data) in [
            ("initial", &self.initial),
            ("delta", &self.delta),
            ("raw", &self.raw),
            ("b", &self.b),
            ("c", &self.c),
            ("rates", &self.rates),
            ("deriv", &self.deriv),
            ("x", &self.x),
            ("a", &self.a),
            ("bar_b", &self.bar_b),
            ("input_b", &self.input_b),
            ("gz", &self.gz),
            ("gy", &self.gy),
        ] {
            m.put(name, data);
        }
        m.put_u32("offsets", &self.offsets);
        m.put_u32("lengths", &self.lengths);
        m.put("carries", &self.initial);
        for (name, len) in [
            ("prefix_a", self.rows * self.stride()),
            ("prefix_b", self.rows * self.stride()),
            ("states", self.states.len()),
            ("y", self.y.len()),
            ("rev_a", self.rev_a.len()),
            ("rev_b", self.rev_b.len()),
            ("future_a", self.future.len()),
            ("future", self.future.len()),
            ("gd", self.gd.len()),
            ("gb", self.gb.len()),
            ("gc", self.gc.len()),
            ("ga", self.ga.len()),
            ("gx", self.gx.len()),
            ("token_gb", self.token_gb.len()),
            ("token_gc", self.token_gc.len()),
        ] {
            m.put(name, &vec![STALE; len]);
        }
        m
    }

    fn materialize_args(&self) -> Vec<ptx_emulator::Arg<'static>> {
        // The original len slot is intentionally wrong: metadata replaces it.
        vec![
            B("initial"),
            B("a"),
            B("bar_b"),
            B("x"),
            B("prefix_a"),
            B("prefix_b"),
            B("c"),
            B("states"),
            B("y"),
            U(u32::MAX),
            U(self.dm as u32),
            U(self.ds as u32),
            B("carries"),
            B("offsets"),
            B("lengths"),
            U(self.chunk as u32),
        ]
    }

    fn maps_args(&self) -> Vec<ptx_emulator::Arg<'static>> {
        vec![
            B("a"),
            B("c"),
            B("gz"),
            B("gy"),
            B("rev_a"),
            B("rev_b"),
            U(u32::MAX),
            U(self.dm as u32),
            U(self.ds as u32),
            F(self.scale()),
            B("offsets"),
            B("lengths"),
        ]
    }

    fn local_args(&self) -> Vec<ptx_emulator::Arg<'static>> {
        // Preserve gB-before-gC and the original per-token gA output ABI.
        vec![
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
            B("token_gb"),
            B("token_gc"),
            B("ga"),
            B("gx"),
            U(u32::MAX),
            U(self.dm as u32),
            U(self.ds as u32),
            U(self.stride() as u32),
            F(self.scale()),
            B("offsets"),
            B("lengths"),
            U(self.chunk as u32),
        ]
    }

    fn launch(&self, m: &mut Machine, name: &str, args: &[ptx_emulator::Arg<'_>], count: usize) {
        // Deliberately oversubscribe both x blocks and threads on every lane.
        m.launch(
            PTX,
            name,
            args,
            (count.div_ceil(BLOCK) + 1, self.lengths.len()),
            BLOCK,
        );
    }

    fn scan(
        &self,
        m: &mut Machine,
        a: &'static str,
        b: &'static str,
        oa: &'static str,
        ob: &'static str,
    ) {
        self.launch(
            m,
            "packed_scan",
            &[
                B(a),
                B(b),
                B(oa),
                B(ob),
                B("offsets"),
                B("lengths"),
                U(self.stride() as u32),
            ],
            self.stride(),
        );
    }

    fn reduce_bc_args(&self) -> Vec<ptx_emulator::Arg<'static>> {
        vec![
            B("token_gb"),
            B("token_gc"),
            B("gb"),
            B("gc"),
            B("offsets"),
            B("lengths"),
            U(self.dm as u32),
            U(self.ds as u32),
        ]
    }

    fn check_gradients(&self, m: &Machine) {
        for (name, expected) in [
            ("gd", &self.gd),
            ("gb", &self.gb),
            ("gc", &self.gc),
            ("ga", &self.ga),
            ("gx", &self.gx),
        ] {
            close(name, &m.get(name), expected);
        }
        self.check_reductions(m, None);
    }

    fn check_reductions(&self, m: &Machine, selected_rows: Option<&[usize]>) {
        let stride = self.stride();
        let actual = m.get("ga");
        let mut a_sum = vec![0.0; stride];
        let mut e_sum = a_sum.clone();
        for row in 0..self.rows {
            let active = self
                .lengths
                .iter()
                .zip(&self.offsets)
                .any(|(&len, &offset)| {
                    len != 0 && row >= offset as usize && row < offset as usize + len as usize
                });
            if !active || selected_rows.is_some_and(|rows| !rows.contains(&row)) {
                continue;
            }
            for c in 0..stride {
                a_sum[c] += actual[row * stride + c];
                e_sum[c] += self.ga[row * stride + c];
            }
        }
        // The host reduces these token rows into the shared rate gradient.
        close("summed gA", &a_sum, &e_sum);
        if selected_rows.is_some() {
            return;
        }
        for (name, expected) in [("gb", &self.gb), ("gc", &self.gc)] {
            let actual = m.get(name);
            let mut aw = vec![0.125; stride];
            let mut ew = aw.clone();
            for _ in 0..2 {
                for lane in 0..self.lengths.len() {
                    for t in 0..self.lengths[lane] as usize {
                        let row = self.offsets[lane] as usize + t;
                        for j in 0..self.ds {
                            for i in 0..self.dm {
                                let index = j * self.dm + i;
                                aw[index] += actual[row * self.ds + j] * self.x[row * self.dm + i];
                                ew[index] +=
                                    expected[row * self.ds + j] * self.x[row * self.dm + i];
                            }
                        }
                    }
                }
                close(&format!("accumulated w from {name}"), &aw, &ew);
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
        // Padding and unwritten rows must be exactly untouched, not just close.
        if e == STALE {
            assert_eq!(a.to_bits(), e.to_bits(), "{name}[{i}] touched padding");
        }
    }
}

fn exercise(s: &Packed, m: &mut Machine) {
    s.scan(m, "a", "input_b", "prefix_a", "prefix_b");
    close("exclusive forward A", &m.get("prefix_a"), &s.prefix_a);
    close("exclusive forward B", &m.get("prefix_b"), &s.prefix_b);
    let max_len = s.lengths.iter().copied().max().unwrap_or(0) as usize;
    s.launch(
        m,
        "packed_materialize",
        &s.materialize_args(),
        max_len * s.dm,
    );
    close("fixed lane states", &m.get("states"), &s.states);
    close("packed readout", &m.get("y"), &s.y);
    close("terminal carries", &m.get("carries"), &s.carries);
    assert_eq!(
        m.get("initial"),
        s.initial,
        "the carry snapshot is immutable"
    );
    s.launch(
        m,
        "packed_backward_maps",
        &s.maps_args(),
        max_len * s.stride(),
    );
    close("reverse A", &m.get("rev_a"), &s.rev_a);
    close("reverse B", &m.get("rev_b"), &s.rev_b);
    s.scan(m, "rev_a", "rev_b", "future_a", "future");
    close("exclusive future adjoint", &m.get("future"), &s.future);
    for _ in 0..2 {
        // Local gradients assign/zero their rows even when the tapes are reused.
        s.launch(m, "packed_backward_local", &s.local_args(), max_len * s.dm);
        close("token B contributions", &m.get("token_gb"), &s.token_gb);
        close("token C contributions", &m.get("token_gc"), &s.token_gc);
        m.launch(
            REDUCE_PTX,
            "packed_reduce_bc",
            &s.reduce_bc_args(),
            ((max_len * s.ds).div_ceil(BLOCK) + 1, s.lengths.len()),
            BLOCK,
        );
        s.check_gradients(m);
    }
    for (name, expected) in [
        ("delta", &s.delta),
        ("raw", &s.raw),
        ("b", &s.b),
        ("c", &s.c),
        ("rates", &s.rates),
        ("deriv", &s.deriv),
        ("x", &s.x),
        ("a", &s.a),
        ("bar_b", &s.bar_b),
        ("gz", &s.gz),
        ("gy", &s.gy),
    ] {
        assert_eq!(&m.get(name), expected, "{name} must remain read-only");
    }
}

#[test]
fn cuda_packed_ptx_forward_backward_ragged_reordered_and_empty_lanes() {
    for s in [
        Packed::new(1, 1, 1, 3, &[1], &[1]),
        // Packed rows 1..6, 6..8, and 8..11 are adjacent but reordered by lane.
        Packed::new(3, 2, 5, 12, &[8, u32::MAX, 1, 6], &[3, 0, 5, 2]),
        Packed::new(2, 3, 33, 45, &[11, 1, u32::MAX, 4], &[33, 2, 0, 7]),
        Packed::new(3, 2, 5, 1, &[u32::MAX, u32::MAX], &[0, 0]),
    ] {
        let mut m = s.machine();
        exercise(&s, &mut m);
    }
}

#[test]
fn cuda_packed_ptx_all_32_lanes_with_ragged_reversed_packing() {
    let lengths: Vec<u32> = (0..32)
        .map(|lane| if lane % 5 == 0 { 0 } else { lane % 6 + 1 })
        .collect();
    let mut offsets = vec![u32::MAX; 32];
    let mut rows = 1;
    for lane in (0..32).rev() {
        if lengths[lane] != 0 {
            offsets[lane] = rows;
            rows += lengths[lane];
        }
    }
    let s = Packed::new(3, 2, 6, rows as usize + 1, &offsets, &lengths);
    exercise(&s, &mut s.machine());
}

#[test]
fn cuda_packed_ptx_sigmoid_retains_representable_negative_tails() {
    for raw in [-20.0f32, -87.0, -88.0] {
        let mut s = Packed::new(1, 1, 1, 1, &[0], &[1]);
        // The delta adjoint before sigmoid is exactly 1: gh=C=gy=b=x=1,
        // initial carry and rates are zero. An absolute error floor would
        // otherwise allow a broken kernel to erase these small gradients.
        s.initial.fill(0.0);
        s.rates.fill(0.0);
        s.raw.fill(raw);
        s.b.fill(1.0);
        s.c.fill(1.0);
        s.x.fill(1.0);
        s.gz.fill(0.0);
        s.gy.fill(1.0);
        s.refresh();
        let mut m = s.machine();
        exercise(&s, &mut m);
        let actual = m.get("gd")[0];
        let expected = 1.0 / (1.0 + (-raw).exp());
        assert!(actual > 0.0, "sigmoid({raw}) tail was erased");
        assert!((actual - expected).abs() / expected < 3e-5);
    }
}

#[test]
fn cuda_packed_ptx_materialize_terminal_threads_before_initial_threads() {
    let s = Packed::new(3, 2, 5, 12, &[8, u32::MAX, 1, 6], &[3, 0, 5, 2]);
    let mut m = s.machine();
    s.scan(&mut m, "a", "input_b", "prefix_a", "prefix_b");
    let mut threads = Vec::new();
    for lane in 0..s.lengths.len() {
        let count = s.lengths[lane] as usize * s.dm;
        // All fit in the same block, so this order writes terminal carries
        // before earlier tokens read the immutable initial carry snapshot.
        threads.push((lane, count));
        for index in (0..count).rev() {
            threads.push((lane, index));
        }
    }
    m.launch_threads(
        PTX,
        "packed_materialize",
        &s.materialize_args(),
        &threads,
        32,
    );
    close("reverse-scheduled states", &m.get("states"), &s.states);
    close("reverse-scheduled carries", &m.get("carries"), &s.carries);
    close("reverse-scheduled readout", &m.get("y"), &s.y);
    assert_eq!(m.get("initial"), s.initial);
}

#[test]
fn cuda_packed_ptx_ordered_exclusive_scan_noncommuting_maps() {
    let stride = 17;
    let rows = 72;
    let offsets = [6, u32::MAX, 1, 2];
    let lengths = [65, 0, 1, 3];
    let a: Vec<_> = (0..rows * stride)
        .map(|i| 0.65 + 0.03 * (i % 11) as f32)
        .collect();
    let b: Vec<_> = (0..rows * stride)
        .map(|i| 0.04 * ((i % 19) as f32 - 8.0))
        .collect();
    let mut m = Machine::default();
    m.put("a", &a);
    m.put("b", &b);
    m.put("oa", &vec![STALE; a.len()]);
    m.put("ob", &vec![STALE; b.len()]);
    m.put_u32("offsets", &offsets);
    m.put_u32("lengths", &lengths);
    m.launch(
        PTX,
        "packed_scan",
        &scan_args("a", "b", "oa", "ob", stride),
        (stride.div_ceil(BLOCK) + 1, offsets.len()),
        BLOCK,
    );
    let mut ea = vec![STALE; a.len()];
    let mut eb = ea.clone();
    for (&offset, &len) in offsets.iter().zip(&lengths) {
        for channel in 0..stride {
            let (mut pa, mut pb) = (1.0, 0.0);
            for t in 0..len as usize {
                let index = (offset as usize + t) * stride + channel;
                ea[index] = pa;
                eb[index] = pb;
                pa = a[index] * pa;
                pb = a[index] * pb + b[index];
            }
        }
    }
    close("ordered exclusive A", &m.get("oa"), &ea);
    close("ordered exclusive B", &m.get("ob"), &eb);
}

#[test]
fn cuda_packed_ptx_dead_scan_maps_can_be_reused_for_local_gradients() {
    let s = Packed::new(3, 2, 5, 10, &[6, u32::MAX, 0], &[4, 0, 6 - 1]);
    let mut m = s.machine();
    exercise(&s, &mut m);
    // Match production aliasing: reverse-map inputs and exclusive A prefixes
    // are dead; only the exclusive B prefix remains a local-VJP input.
    let mut args = s.local_args();
    args[14] = B("rev_a");
    args[15] = B("rev_b");
    args[16] = B("future_a");
    s.launch(&mut m, "packed_backward_local", &args, 5 * s.dm);
    close("reused token B", &m.get("rev_a"), &s.token_gb);
    close("reused token C", &m.get("rev_b"), &s.token_gc);
    close("reused rate map", &m.get("future_a"), &s.ga);
    // The same reusable map is immediately consumed by the rate reduction.
    m.zeros("reduced", s.lengths.len() * s.stride());
    m.launch(
        REDUCE_PTX,
        "packed_reduce_rate",
        &[
            B("future_a"), B("reduced"), B("offsets"), B("lengths"),
            U(s.stride() as u32),
        ],
        (1, s.lengths.len()),
        BLOCK,
    );
    let mut expected = vec![0.0; s.lengths.len() * s.stride()];
    for (lane, (&offset, &len)) in s.offsets.iter().zip(&s.lengths).enumerate() {
        for t in (0..len as usize).rev() {
            for channel in 0..s.stride() {
                expected[lane * s.stride() + channel] +=
                    s.ga[(offset as usize + t) * s.stride() + channel];
            }
        }
    }
    close("reused rate reduction", &m.get("reduced"), &expected);
}

#[test]
fn cuda_packed_ptx_carry_snapshot_reuse_and_lane_reset() {
    let mut s = Packed::new(3, 2, 5, 12, &[8, u32::MAX, 1, 6], &[3, 0, 5, 2]);
    let mut m = s.machine();
    exercise(&s, &mut m);
    s.initial = m.get("carries");
    let stride = s.stride();
    s.initial[..stride].fill(0.0); // Reset only lane zero, retaining adjacent lanes.
    s.refresh();
    // Model the host's carry -> immutable snapshot DtoD between chunk launches.
    m.put("carries", &s.initial);
    m.put("initial", &s.initial);
    exercise(&s, &mut m);
}

#[test]
fn cuda_packed_ptx_carries_follow_lane_ids_through_omission_and_repacking() {
    let mut s = Packed::new(3, 2, 5, 8, &[0, 3, u32::MAX], &[3, 5, 0]);
    let mut m = s.machine();
    for (step, (offsets, lengths)) in [
        ([0, 3, u32::MAX], [3, 5, 0]),
        ([5, 0, u32::MAX], [3, 5, 0]),
        ([0, u32::MAX, 3], [3, 0, 5]),
        ([5, 0, u32::MAX], [3, 5, 0]),
    ]
    .into_iter()
    .enumerate()
    {
        let previous_states = m.get("states");
        s.initial = m.get("carries");
        if step == 2 {
            let stride = s.stride();
            s.initial[..stride].fill(0.0);
        }
        s.offsets = offsets.to_vec();
        s.lengths = lengths.to_vec();
        s.refresh();
        m.put_u32("offsets", &offsets);
        m.put_u32("lengths", &lengths);
        m.put("initial", &s.initial);
        m.put("carries", &s.initial);
        s.scan(&mut m, "a", "input_b", "prefix_a", "prefix_b");
        s.launch(&mut m, "packed_materialize", &s.materialize_args(), 5 * s.dm);
        close("repacked carries", &m.get("carries"), &s.carries);
        close("repacked readout", &m.get("y"), &s.y);
        let actual_states = m.get("states");
        for (lane, &len) in lengths.iter().enumerate() {
            let base = lane * (s.chunk + 1) * s.stride();
            let written = if len == 0 { 0 } else { (len as usize + 1) * s.stride() };
            close(
                "active lane states",
                &actual_states[base..base + written],
                &s.states[base..base + written],
            );
            let end = base + (s.chunk + 1) * s.stride();
            assert_eq!(
                actual_states[base + written..end],
                previous_states[base + written..end],
                "inactive/padded states changed on lane {lane}"
            );
        }
    }
}

fn scan_args(
    a: &'static str,
    b: &'static str,
    oa: &'static str,
    ob: &'static str,
    stride: usize,
) -> Vec<ptx_emulator::Arg<'static>> {
    vec![
        B(a),
        B(b),
        B(oa),
        B(ob),
        B("offsets"),
        B("lengths"),
        U(stride as u32),
    ]
}

#[test]
fn cuda_packed_ptx_production_width_first_last_rows_and_channels() {
    let s = Packed::new(3584, 16, 5, 7, &[3, u32::MAX, 1], &[3, 0, 2]);
    let stride = s.stride();
    let mut m = s.machine();
    let channels = [
        (0, 0),
        (0, stride - 1),
        (0, stride),
        (1, 0),
        (2, 0),
        (2, stride - 1),
        (2, stride),
    ];
    m.launch_threads(
        PTX,
        "packed_scan",
        &scan_args("a", "input_b", "prefix_a", "prefix_b", stride),
        &channels,
        256,
    );
    for (name, oracle) in [("prefix_a", &s.prefix_a), ("prefix_b", &s.prefix_b)] {
        let mut expected = vec![STALE; oracle.len()];
        for &(lane, channel) in &channels {
            if channel >= stride {
                continue;
            }
            for t in 0..s.lengths[lane] as usize {
                let index = (s.offsets[lane] as usize + t) * stride + channel;
                expected[index] = oracle[index];
            }
        }
        close(name, &m.get(name), &expected);
    }
    // Materialize needs every j for the sampled first/last latent channels.
    m.put("prefix_a", &s.prefix_a);
    m.put("prefix_b", &s.prefix_b);
    let mut materialize_threads = vec![(1, 0)];
    let mut map_threads = vec![(1, 0)];
    let mut local_threads = vec![(1, 0)];
    let mut selected_rows = Vec::new();
    let mut states = vec![STALE; s.states.len()];
    let mut carries = s.initial.clone();
    let mut y = vec![STALE; s.y.len()];
    let mut rev_a = vec![STALE; s.rev_a.len()];
    let mut rev_b = vec![STALE; s.rev_b.len()];
    for lane in [0, 2] {
        let len = s.lengths[lane] as usize;
        let offset = s.offsets[lane] as usize;
        let state_base = lane * (s.chunk + 1) * stride;
        for t in [0, len - 1] {
            selected_rows.push(offset + t);
            local_threads.extend((0..s.dm).map(|i| (lane, t * s.dm + i)));
            for i in [0, s.dm - 1] {
                materialize_threads.push((lane, t * s.dm + i));
                let di = (offset + t) * s.dm + i;
                y[di] = s.y[di];
                for j in 0..s.ds {
                    let channel = i * s.ds + j;
                    let index = state_base + (t + 1) * stride + channel;
                    states[index] = s.states[index];
                    if t == 0 {
                        states[state_base + channel] = s.initial[lane * stride + channel];
                    }
                    if t == len - 1 {
                        carries[lane * stride + channel] = s.carries[lane * stride + channel];
                    }
                }
            }
            for channel in [0, stride - 1] {
                map_threads.push((lane, t * stride + channel));
                let index = (offset + t) * stride + channel;
                rev_a[index] = s.rev_a[index];
                rev_b[index] = s.rev_b[index];
            }
        }
        materialize_threads.push((lane, len * s.dm));
        map_threads.push((lane, len * stride));
        local_threads.push((lane, len * s.dm));
    }
    m.launch_threads(
        PTX,
        "packed_materialize",
        &s.materialize_args(),
        &materialize_threads,
        256,
    );
    close("sampled states", &m.get("states"), &states);
    close("sampled carries", &m.get("carries"), &carries);
    close("sampled y", &m.get("y"), &y);
    assert_eq!(m.get("initial"), s.initial);
    m.launch_threads(
        PTX,
        "packed_backward_maps",
        &s.maps_args(),
        &map_threads,
        256,
    );
    close("sampled reverse A", &m.get("rev_a"), &rev_a);
    close("sampled reverse B", &m.get("rev_b"), &rev_b);
    m.put("rev_a", &s.rev_a);
    m.put("rev_b", &s.rev_b);
    m.launch_threads(
        PTX,
        "packed_scan",
        &scan_args("rev_a", "rev_b", "future_a", "future", stride),
        &channels,
        256,
    );
    let mut future = vec![STALE; s.future.len()];
    for &(lane, channel) in &channels {
        if channel >= stride {
            continue;
        }
        for t in 0..s.lengths[lane] as usize {
            let index = (s.offsets[lane] as usize + t) * stride + channel;
            future[index] = s.future[index];
        }
    }
    close("sampled future", &m.get("future"), &future);
    // Local distributes each sampled token across ALL 3584 latent threads;
    // the separate B/C kernel retains the original ascending reduction order.
    m.put("states", &s.states);
    m.put("future", &s.future);
    m.launch_threads(
        PTX,
        "packed_backward_local",
        &s.local_args(),
        &local_threads,
        256,
    );
    let mut reduction_threads = vec![(1, 0)];
    for lane in [0, 2] {
        let len = s.lengths[lane] as usize;
        for t in [0, len - 1] {
            reduction_threads.extend((0..s.ds).map(|j| (lane, t * s.ds + j)));
        }
        reduction_threads.push((lane, len * s.ds));
    }
    m.launch_threads(
        REDUCE_PTX,
        "packed_reduce_bc",
        &s.reduce_bc_args(),
        &reduction_threads,
        256,
    );
    for (name, oracle, width) in [
        ("gd", &s.gd, s.dm),
        ("gb", &s.gb, s.ds),
        ("gc", &s.gc, s.ds),
        ("ga", &s.ga, stride),
        ("gx", &s.gx, s.dm),
    ] {
        let mut expected = vec![STALE; oracle.len()];
        for &row in &selected_rows {
            expected[row * width..(row + 1) * width]
                .copy_from_slice(&oracle[row * width..(row + 1) * width]);
        }
        close(name, &m.get(name), &expected);
    }
    s.check_reductions(&m, Some(&selected_rows));
}

#[test]
#[should_panic(expected = "overlapping thread writes")]
fn cuda_packed_ptx_emulator_detects_overlapping_lane_token_ranges() {
    let s = Packed::new(1, 1, 1, 2, &[1, 1], &[1, 1]);
    let mut m = s.machine();
    s.scan(&mut m, "a", "input_b", "prefix_a", "prefix_b");
}

#[test]
#[should_panic(expected = "invalid address")]
fn cuda_packed_ptx_emulator_detects_truncated_fixed_state_storage() {
    let s = Packed::new(1, 1, 1, 2, &[1], &[1]);
    let mut m = Machine::default();
    for (name, data) in [
        ("initial", &s.initial),
        ("a", &s.a),
        ("bar_b", &s.bar_b),
        ("x", &s.x),
        ("prefix_a", &s.prefix_a),
        ("prefix_b", &s.prefix_b),
        ("c", &s.c),
        ("carries", &s.initial),
        ("y", &s.y),
    ] {
        m.put(name, data);
    }
    m.put_u32("offsets", &s.offsets);
    m.put_u32("lengths", &s.lengths);
    m.zeros("states", 1); // Row zero exists, but the terminal row is missing.
    s.launch(&mut m, "packed_materialize", &s.materialize_args(), 1);
}
