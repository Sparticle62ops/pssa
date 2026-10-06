//! CPU execution of the embedded memory kernels, including training-size rows.
#[path = "ptx_emulator.rs"]
mod ptx_emulator;
use pssa::memory::HyperbolicEpisodicBankV2 as Bank;
use ptx_emulator::{
    Arg::{Buffer as B, F32 as F, U32 as U},
    Machine,
};

const PTX: &str = include_str!("../../src/cuda/stages.ptx");

fn close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() < 2e-5 + 2e-4 * b.abs(),
            "element {i}: {a} != {b}"
        );
    }
}

struct Memory {
    l: usize,
    dk: usize,
    dv: usize,
    bank: Bank,
    qe: Vec<f32>,
    qp: Vec<f32>,
    norms: Vec<f32>,
    weights: Vec<f32>,
    mv: Vec<f32>,
    gm: Vec<f32>,
}

impl Memory {
    fn new(l: usize, dk: usize, dv: usize, cap: usize, count: usize) -> Self {
        let mut bank = Bank::new(cap, dk, dv);
        for e in 0..count {
            let key = (0..dk)
                .map(|k| 0.015 * ((k % 5) as f32 - 2.0) + 0.01 * e as f32)
                .collect::<Vec<_>>();
            let value = (0..dv)
                .map(|j| 0.02 * ((j % 7) as f32 - 3.0) + 0.1 * e as f32)
                .collect::<Vec<_>>();
            bank.insert(&key, &value);
        }
        let mut qe = (0..l * dk)
            .map(|i| 0.03 * ((i % 11) as f32 - 5.0))
            .collect::<Vec<_>>();
        // The zero-norm projection branch must copy the query adjoint,
        // not the (zero) Euclidean query input.
        qe[..dk].fill(0.0);
        let mut s = Self {
            l,
            dk,
            dv,
            bank,
            qe,
            qp: vec![0.; l * dk],
            norms: vec![0.; l],
            weights: vec![0.; l * cap],
            mv: vec![0.; l * dv],
            gm: (0..l * dv)
                .map(|i| 0.001 * ((i % 13) as f32 - 6.0))
                .collect(),
        };
        for t in 0..l {
            s.norms[t] = Bank::diffeomorphic_project(
                &s.qe[t * dk..(t + 1) * dk],
                &mut s.qp[t * dk..(t + 1) * dk],
            );
            s.bank.retrieve_soft_into(
                &s.qp[t * dk..(t + 1) * dk],
                0.7,
                &mut s.mv[t * dv..(t + 1) * dv],
                &mut s.weights[t * cap..(t + 1) * cap],
            );
        }
        s
    }

    fn machine(&self) -> Machine {
        let mut m = Machine::default();
        for (name, data) in [
            ("qe", &self.qe),
            ("qp", &self.qp),
            ("norms", &self.norms),
            ("weights", &self.weights),
            ("mv", &self.mv),
            ("gm", &self.gm),
            ("keys", &self.bank.keys),
            ("nsq", &self.bank.norm_sq),
            ("values", &self.bank.values),
        ] {
            m.put(name, data);
        }
        // Stale seeds verify every output is assigned, including empty banks.
        m.put("gqp", &vec![17.; self.l * self.dk]);
        m.put("gqe", &vec![-19.; self.l * self.dk]);
        m
    }

    fn backward_args(&self) -> Vec<ptx_emulator::Arg<'static>> {
        vec![
            B("qp"),
            B("qe"),
            B("gm"),
            B("mv"),
            B("weights"),
            B("keys"),
            B("nsq"),
            B("values"),
            B("gqp"),
            B("gqe"),
            U(self.l as u32),
            U(self.bank.count as u32),
            U(self.bank.capacity as u32),
            U(self.dk as u32),
            U(self.dv as u32),
            F(0.7),
        ]
    }

    fn adjoint(&self, t: usize) -> (Vec<f32>, Vec<f32>) {
        let q = &self.qp[t * self.dk..(t + 1) * self.dk];
        let mut gp = vec![0.; self.dk];
        let qsq = q.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        for e in 0..self.bank.count {
            let dot = (0..self.dv)
                .map(|j| {
                    self.gm[t * self.dv + j] as f64
                        * (self.bank.values[e * self.dv + j] - self.mv[t * self.dv + j]) as f64
                })
                .sum::<f64>();
            let sq = (0..self.dk)
                .map(|k| (q[k] as f64 - self.bank.keys[e * self.dk + k] as f64).powi(2))
                .sum::<f64>();
            if sq == 0. {
                continue;
            }
            let denom = (1. - qsq) * (1. - self.bank.norm_sq[e] as f64);
            let z = sq / denom;
            let coeff =
                self.weights[t * self.bank.capacity + e] as f64 * dot * (-1. / 0.7f32 as f64)
                    / (z * (1. + z)).sqrt();
            for k in 0..self.dk {
                let diff = q[k] as f64 - self.bank.keys[e * self.dk + k] as f64;
                let dd = -2. * q[k] as f64 * (1. - self.bank.norm_sq[e] as f64);
                gp[k] += (coeff * (2. * diff * denom - sq * dd) / (denom * denom)) as f32;
            }
        }
        let mut ge = vec![0.; self.dk];
        Bank::projection_adjoint(&self.qe[t * self.dk..(t + 1) * self.dk], &gp, &mut ge);
        (gp, ge)
    }
}

#[test]
fn cuda_memory_backward_ptx_addresses_and_gradients() {
    for (l, dk, dv, cap, count) in [(1, 2, 3, 3, 0), (4, 2, 3, 3, 2), (5, 3, 2, 2, 2)] {
        let s = Memory::new(l, dk, dv, cap, count);
        let mut m = s.machine();
        m.launch(
            PTX,
            "memory_backward",
            &s.backward_args(),
            (l.div_ceil(4), 1),
            4,
        );
        for t in 0..l {
            let (gp, ge) = s.adjoint(t);
            close(&m.get("gqp")[t * dk..(t + 1) * dk], &gp);
            close(&m.get("gqe")[t * dk..(t + 1) * dk], &ge);
        }
    }
}

#[test]
fn cuda_memory_backward_ptx_training_width_boundary_rows() {
    let s = Memory::new(32, 32, 3584, 512, 2);
    let mut m = s.machine();
    for t in [0, 15, 16, 31, 32] {
        m.launch_thread(PTX, "memory_backward", &s.backward_args(), t, 1024);
        if t < 32 {
            let (gp, ge) = s.adjoint(t);
            close(&m.get("gqp")[t * s.dk..(t + 1) * s.dk], &gp);
            close(&m.get("gqe")[t * s.dk..(t + 1) * s.dk], &ge);
        }
    }
}

#[test]
fn cuda_memory_forward_ptx_training_value_width_exceeds_slot_capacity() {
    let s = Memory::new(32, 32, 3584, 512, 2);
    let mut m = s.machine();
    m.put("weights", &vec![23.; s.l * s.bank.capacity]);
    m.put("mv", &vec![23.; s.l * s.dv]);
    for t in [0, 31, 32] {
        m.launch_thread(
            PTX,
            "memory_forward",
            &[
                B("qe"),
                B("keys"),
                B("nsq"),
                B("values"),
                B("qp"),
                B("norms"),
                B("mv"),
                B("weights"),
                U(s.l as u32),
                U(s.bank.count as u32),
                U(s.bank.capacity as u32),
                U(s.dk as u32),
                U(s.dv as u32),
                F(0.7),
            ],
            t,
            1024,
        );
        if t < s.l {
            close(
                &m.get("weights")[t * s.bank.capacity..(t + 1) * s.bank.capacity],
                &s.weights[t * s.bank.capacity..(t + 1) * s.bank.capacity],
            );
            close(
                &m.get("mv")[t * s.dv..(t + 1) * s.dv],
                &s.mv[t * s.dv..(t + 1) * s.dv],
            );
        }
    }
}

#[test]
fn cuda_memory_forward_ptx_nonempty_and_empty_banks() {
    for (l, dk, dv, cap, count) in [(1, 2, 3, 3, 0), (4, 2, 3, 3, 2), (5, 3, 2, 2, 2)] {
        let s = Memory::new(l, dk, dv, cap, count);
        let mut m = s.machine();
        m.put("qp", &vec![23.; l * dk]);
        m.put("norms", &vec![23.; l]);
        m.put("weights", &vec![23.; l * cap]);
        m.put("mv", &vec![23.; l * dv]);
        m.launch(
            PTX,
            "memory_forward",
            &[
                B("qe"),
                B("keys"),
                B("nsq"),
                B("values"),
                B("qp"),
                B("norms"),
                B("mv"),
                B("weights"),
                U(l as u32),
                U(count as u32),
                U(cap as u32),
                U(dk as u32),
                U(dv as u32),
                F(0.7),
            ],
            (l.div_ceil(4), 1),
            4,
        );
        close(&m.get("qp"), &s.qp);
        close(&m.get("norms"), &s.norms);
        close(&m.get("weights"), &s.weights);
        close(&m.get("mv"), &s.mv);
    }
}
