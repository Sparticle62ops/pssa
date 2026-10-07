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
            a.is_finite() && b.is_finite() && (a - b).abs() < 2e-5 + 2e-4 * b.abs(),
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
        s.refresh(0.7);
        s
    }

    fn refresh(&mut self, tau: f32) {
        for t in 0..self.l {
            let (dk, dv, cap) = (self.dk, self.dv, self.bank.capacity);
            self.norms[t] = Bank::diffeomorphic_project(
                &self.qe[t * dk..(t + 1) * dk],
                &mut self.qp[t * dk..(t + 1) * dk],
            );
            self.weights[t * cap..(t + 1) * cap].fill(0.0);
            self.bank.retrieve_soft_into(
                &self.qp[t * dk..(t + 1) * dk],
                tau,
                &mut self.mv[t * dv..(t + 1) * dv],
                &mut self.weights[t * cap..(t + 1) * cap],
            );
        }
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

    fn forward_args(&self, tau: f32) -> Vec<ptx_emulator::Arg<'static>> {
        vec![
            B("qe"),
            B("keys"),
            B("nsq"),
            B("values"),
            B("qp"),
            B("norms"),
            B("mv"),
            B("weights"),
            U(self.l as u32),
            U(self.bank.count as u32),
            U(self.bank.capacity as u32),
            U(self.dk as u32),
            U(self.dv as u32),
            F(tau),
        ]
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
        // Production differentiates using the rounded, validated f32 norm.
        let qsq = Bank::squared_norm(q) as f64;
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
fn cuda_memory_ptx_training_stride_extreme_projection_and_adjoint() {
    let mut s = Memory::new(32, 32, 3584, 512, 2);
    for (t, magnitude) in [(1, 16_777_216.0), (15, f32::MAX), (16, 1024.0), (31, 1e-30)] {
        s.qe[t * s.dk..(t + 1) * s.dk].fill(0.0);
        s.qe[t * s.dk] = magnitude;
        if t == 15 {
            s.qe[t * s.dk + 1] = -magnitude;
        }
    }
    s.refresh(0.7);
    let mut m = s.machine();
    for t in [0, 1, 15, 16, 31, 32] {
        m.launch_thread(PTX, "memory_forward", &s.forward_args(0.7), t, 1024);
        if t == s.l {
            continue;
        }
        let actual_q = m.get("qp");
        let q = &actual_q[t * s.dk..(t + 1) * s.dk];
        assert!(
            q.iter().all(|x| x.is_finite()) && Bank::squared_norm(q) < 1.0,
            "first failure: qp[token={t}] must remain in the open ball: {q:?}"
        );
        assert_eq!(q, &s.qp[t * s.dk..(t + 1) * s.dk], "qp[token={t}]");
        close(&m.get("norms")[t..t + 1], &s.norms[t..t + 1]);
        close(
            &m.get("weights")[t * s.bank.capacity..(t + 1) * s.bank.capacity],
            &s.weights[t * s.bank.capacity..(t + 1) * s.bank.capacity],
        );
        close(
            &m.get("mv")[t * s.dv..(t + 1) * s.dv],
            &s.mv[t * s.dv..(t + 1) * s.dv],
        );
        m.launch_thread(PTX, "memory_backward", &s.backward_args(), t, 1024);
        let (gp, ge) = s.adjoint(t);
        close(&m.get("gqp")[t * s.dk..(t + 1) * s.dk], &gp);
        let actual_ge = m.get("gqe");
        close(&actual_ge[t * s.dk..(t + 1) * s.dk], &ge);
        if t == 1 {
            assert_eq!(
                actual_ge[t * s.dk],
                0.0,
                "saturated axis-aligned query has exactly zero radial derivative"
            );
        }
    }
}

#[test]
fn cuda_memory_ptx_min_subtracted_softmax_survives_tiny_temperature() {
    // Same value/slot strides as training, but only three occupied slots. This
    // catches dividing distances before max subtraction and stale slot tails.
    let mut s = Memory::new(32, 32, 3584, 512, 0);
    for (coordinate, value) in [(0.5, 1.0), (-0.5, 3.0), (0.75, 100.0)] {
        let mut key = vec![0.0; s.dk];
        key[0] = coordinate;
        s.bank.insert(&key, &vec![value; s.dv]);
    }
    s.qe.fill(0.0);
    for tau in [0.7, 1e-40, f32::from_bits(1)] {
        s.refresh(tau);
        let mut m = s.machine();
        m.put("weights", &vec![23.0; s.l * s.bank.capacity]);
        m.put("mv", &vec![23.0; s.l * s.dv]);
        m.launch_thread(PTX, "memory_forward", &s.forward_args(tau), 31, 1024);
        let weights = m.get("weights");
        close(
            &weights[31 * s.bank.capacity..32 * s.bank.capacity],
            &s.weights[31 * s.bank.capacity..32 * s.bank.capacity],
        );
        close(
            &m.get("mv")[31 * s.dv..32 * s.dv],
            &s.mv[31 * s.dv..32 * s.dv],
        );
        assert!(
            (weights[31 * s.bank.capacity..32 * s.bank.capacity]
                .iter()
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-6
        );
    }
}

// A full production-model loss/backward check, not just an activation probe.
// Only the memory stage is replayed in PTX; surrounding dense/SSM/MLP work uses
// production CPU stages. This checks its integration and every parameter VJP,
// but deliberately does not claim CUDA scheduling/cuBLAS/full-device parity.
fn training_model() -> pssa::pssa::PSSALayerV2 {
    use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
    let cfg = PSSAConfigV2 {
        depth: 1,
        d_vocab: 11,
        d_latent: 2048,
        d_state: 16,
        d_mem_key: 32,
        mem_capacity: 512,
        chunk_len: 32,
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.25,
    };
    // The production constructor caps the entire model at 1 GiB, rejecting
    // latent=3584 (1,515,608,088 declared bytes). Use 2048 for this full-model
    // integration check; the separate loss/VJP regression keeps 3584 strides.
    // Reuse one model plus a gradient snapshot rather than cloning Adam state.
    let mut m = PSSALayerV2::new(cfg, 71);
    let dm = m.cfg.d_latent;
    for id in 0..m.cfg.d_vocab {
        m.embed_w.data[id * dm] = 1.0;
    }
    m.w_qx.data.fill(0.0);
    m.w_qh.data.fill(0.0);
    m.w_gate.data.fill(0.0);
    m.w_proj.data.fill(0.0);
    for i in 0..dm {
        m.w_gate.data[i * dm + i] = 0.1;
        m.w_proj.data[i * dm + i] = 0.25;
        // Keep the MLP path/its two parameter gradients non-vacuous.
        m.mlp_w2.data[i * (2 * dm) + i] = 0.01;
    }
    for (i, x) in m.adapters[0].up_proj.data.iter_mut().enumerate() {
        *x = 0.001 * ((i % 7) as f32 - 3.0);
    }
    for (i, x) in m.h_persistent.iter_mut().enumerate() {
        *x = 0.001 * ((i % 17) as f32 - 8.0);
    }
    for e in 0..3 {
        let mut key = vec![0.0; m.cfg.d_mem_key];
        key[0] = [0.2, -0.5, 0.8][e];
        // A valid legacy near-boundary key whose f64 squared norm rounds to
        // 1 in f32. Production squared_norm keeps its stored norm open-ball.
        if e == 2 {
            key[1] = f32::from_bits(0.6f32.to_bits() - 1);
        }
        let values = (0..dm)
            .map(|j| 0.05 * (((j + 3 * e) % 13) as f32 - 6.0))
            .collect::<Vec<_>>();
        m.memory.insert(&key, &values);
    }
    m
}

fn gradients(m: &pssa::pssa::PSSALayerV2) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("embed", m.embed_w.grad.clone()),
        ("unembed", m.unembed_w.grad.clone()),
        ("gamma", m.norm_gamma.grad.clone()),
        ("beta", m.norm_beta.grad.clone()),
        ("A", m.a_mat.grad.clone()),
        ("delta", m.w_delta.grad.clone()),
        ("B", m.w_b.grad.clone()),
        ("C", m.w_c.grad.clone()),
        ("qx", m.w_qx.grad.clone()),
        ("qh", m.w_qh.grad.clone()),
        ("gate", m.w_gate.grad.clone()),
        ("proj", m.w_proj.grad.clone()),
        ("mlp1", m.mlp_w1.grad.clone()),
        ("mlp2", m.mlp_w2.grad.clone()),
        ("down", m.adapters[0].down_proj.grad.clone()),
        ("up", m.adapters[0].up_proj.grad.clone()),
    ]
}

fn check_training_gradients(m: &pssa::pssa::PSSALayerV2, expected: &[(&str, Vec<f32>)]) {
    // Borrow actual gradients: a second gradient snapshot wastes ~360 MB.
    let actual = [
        &m.embed_w.grad,
        &m.unembed_w.grad,
        &m.norm_gamma.grad,
        &m.norm_beta.grad,
        &m.a_mat.grad,
        &m.w_delta.grad,
        &m.w_b.grad,
        &m.w_c.grad,
        &m.w_qx.grad,
        &m.w_qh.grad,
        &m.w_gate.grad,
        &m.w_proj.grad,
        &m.mlp_w1.grad,
        &m.mlp_w2.grad,
        &m.adapters[0].down_proj.grad,
        &m.adapters[0].up_proj.grad,
    ];
    for ((name, expected), actual) in expected.iter().zip(actual) {
        let scale = expected.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mut error = 0.0f32;
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && e.is_finite(),
                "nonfinite {name} gradient: {a} vs {e}"
            );
            error = error.max((a - e).abs());
        }
        assert!(
            error <= 2e-6 + 5e-4 * scale,
            "{name} gradient: max error {error}, scale {scale}"
        );
    }
}

fn replay_memory_forward(m: &mut pssa::pssa::PSSALayerV2, l: usize) -> Memory {
    let (dm, dk, cap) = (m.cfg.d_latent, m.cfg.d_mem_key, m.cfg.mem_capacity);
    let mut s = Memory {
        l,
        dk,
        dv: dm,
        bank: m.memory.clone(),
        qe: m.tape.q_euc[..l * dk].to_vec(),
        qp: vec![23.0; l * dk],
        norms: vec![23.0; l],
        weights: vec![23.0; l * cap],
        mv: vec![23.0; l * dm],
        gm: vec![0.0; l * dm],
    };
    let mut machine = s.machine();
    machine.launch(
        PTX,
        "memory_forward",
        &s.forward_args(m.cfg.tau_mem),
        (l + 1, 1), // Packed retrieval spreads serial token walks across blocks.
        1, // The extra block checks the first out-of-range row.
    );
    s.qp = machine.get("qp");
    s.norms = machine.get("norms");
    s.weights = machine.get("weights");
    s.mv = machine.get("mv");
    m.tape.q_poincare[..l * dk].copy_from_slice(&s.qp);
    m.tape.q_norm[..l].copy_from_slice(&s.norms);
    m.tape.mem_weights[..l * cap].copy_from_slice(&s.weights);
    m.tape.m_val[..l * dm].copy_from_slice(&s.mv);
    // Dense boundaries are real production CPU GEMMs, not a second oracle.
    let b = &mut m.block;
    pssa::backend::gemm_cpu_into(
        &b.tape.x_norm[..l * dm],
        &b.w_gate.data,
        l,
        dm,
        dm,
        1,
        &mut b.tape.g_mem[..l * dm],
    )
    .unwrap();
    machine.put("gate", &m.tape.g_mem[..l * dm]);
    machine.launch(
        PTX,
        "sigmoid_in_place",
        &[B("gate"), U((l * dm) as u32)],
        ((l * dm).div_ceil(1024), 1),
        1024,
    );
    m.tape.g_mem[..l * dm].copy_from_slice(&machine.get("gate"));
    let b = &mut m.block;
    pssa::backend::gemm_cpu_into(
        &s.mv,
        &b.w_proj.data,
        l,
        dm,
        dm,
        1,
        &mut b.tape.m_proj[..l * dm],
    )
    .unwrap();
    machine.put("mp", &m.tape.m_proj[..l * dm]);
    machine.zeros("inj", l * dm);
    machine.launch(
        PTX,
        "sigmoid_mul",
        &[B("gate"), B("mp"), B("inj"), U((l * dm) as u32)],
        ((l * dm).div_ceil(1024), 1),
        1024,
    );
    m.tape.m_inj[..l * dm].copy_from_slice(&machine.get("inj"));
    s
}

fn replay_memory_backward(m: &mut pssa::pssa::PSSALayerV2, s: &mut Memory) {
    let (l, dm, dk) = (s.l, s.dv, s.dk);
    let mut machine = s.machine();
    machine.put("gz", &m.bwd_g_zraw[..l * dm]);
    machine.put("gate", &m.tape.g_mem[..l * dm]);
    machine.put("mp", &m.tape.m_proj[..l * dm]);
    machine.zeros("gmp", l * dm);
    machine.zeros("ggate", l * dm);
    machine.launch(
        PTX,
        "memory_backward_local",
        &[
            B("gz"),
            B("gate"),
            B("mp"),
            B("gmp"),
            B("ggate"),
            U((l * dm) as u32),
        ],
        ((l * dm).div_ceil(1024), 1),
        1024,
    );
    let (gmp, ggate) = (machine.get("gmp"), machine.get("ggate"));
    let b = &mut m.block;
    let mut gx = vec![0.0; l * dm];
    pssa::backend::gemm_nn_cpu_into(&ggate, &b.w_gate.data, l, dm, dm, &mut gx).unwrap();
    for (dst, &x) in b.bwd_g_xnorm[..l * dm].iter_mut().zip(&gx) {
        *dst += x;
    }
    pssa::backend::gemm_nn_cpu_into(&gmp, &b.w_proj.data, l, dm, dm, &mut s.gm).unwrap();
    pssa::backend::gemm_tn_cpu_accumulate_into(
        &ggate,
        &b.tape.x_norm[..l * dm],
        l,
        dm,
        dm,
        &mut b.w_gate.grad,
    )
    .unwrap();
    pssa::backend::gemm_tn_cpu_accumulate_into(&gmp, &s.mv, l, dm, dm, &mut b.w_proj.grad).unwrap();
    machine.put("gm", &s.gm);
    let mut args = s.backward_args();
    args[15] = F(b.cfg.tau_mem);
    machine.launch(PTX, "memory_backward", &args, (l + 1, 1), 1);
    b.bwd_g_query_pnc[..l * dk].copy_from_slice(&machine.get("gqp"));
    let ge = machine.get("gqe");
    b.bwd_g_query_euc[..l * dk].copy_from_slice(&ge);
    b.bwd_g_ysm[..l * dm].fill(0.0);
    // Retain production's reverse-token query reduction order, including
    // shared parameter accumulation. All nonlinear VJPs above came from PTX.
    for t in (0..l).rev() {
        for k in 0..dk {
            let g = ge[t * dk + k];
            for j in 0..dm {
                b.w_qx.grad[k * dm + j] += g * b.tape.x_norm[t * dm + j];
                b.w_qh.grad[k * dm + j] += g * b.tape.y_ssm[t * dm + j];
                b.bwd_g_xnorm[t * dm + j] += g * b.w_qx.data[k * dm + j];
                b.bwd_g_ysm[t * dm + j] += g * b.w_qh.data[k * dm + j];
            }
        }
    }
}

#[test]
fn cuda_memory_ptx_loss_and_all_parameter_gradients_match_cpu() {
    use pssa::gpu_batch as stages;
    let mut m = training_model();
    let ids = (0..32).map(|t| t % m.cfg.d_vocab).collect::<Vec<_>>();
    let targets = (0..32).map(|t| (t + 1) % m.cfg.d_vocab).collect::<Vec<_>>();
    // Later cases reuse the carry from earlier chunks and simulate resumed,
    // large-but-finite query weights/stored values. The tiny tau case has a
    // unique nearest slot so the unchanged CPU backward itself remains finite.
    for (case, tau, query_scale, value_scale) in [
        ("ordinary", 0.7, 0.01, 1.0),
        ("resumed-saturated", 0.7, 16_777_216.0, 1e12),
        ("resumed-tiny-tau", 1e-40, 0.01, 1.0),
    ] {
        m.cfg.tau_mem = tau;
        m.block.cfg.tau_mem = tau;
        m.w_qx.data[0] = query_scale;
        let dm = m.cfg.d_latent;
        for e in 0..m.memory.count {
            for j in 0..dm {
                m.memory.values[e * dm + j] =
                    value_scale * 0.05 * (((j + 3 * e) % 13) as f32 - 6.0);
            }
        }
        m.zero_gradients();
        let expected_loss = stages::forward_train_chunk_batched(&mut m, &ids, &targets);
        stages::backward_chunk_batched(&mut m, 32, 0.7);
        assert!(expected_loss.is_finite(), "{case}: CPU loss must be finite");
        let expected = gradients(&m);
        for name in [
            "embed", "unembed", "gate", "proj", "mlp1", "mlp2", "down", "up",
        ] {
            assert!(
                expected
                    .iter()
                    .find(|(n, _)| *n == name)
                    .unwrap()
                    .1
                    .iter()
                    .any(|x| x.abs() > 1e-12),
                "{case}: vacuous {name} gradient"
            );
        }
        if case == "ordinary" {
            assert!(
                expected
                    .iter()
                    .find(|(n, _)| *n == "qx")
                    .unwrap()
                    .1
                    .iter()
                    .any(|x| x.abs() > 1e-10)
            );
        }
        m.zero_gradients();
        let mut s = replay_memory_forward(&mut m, 32);
        stages::stage_adapter(&mut m, 32);
        stages::stage_mlp(&mut m, 32);
        let actual_loss = stages::stage_logits_loss(&mut m, 32);
        assert!(
            actual_loss.is_finite()
                && (actual_loss - expected_loss).abs() <= 2e-5 + 2e-4 * expected_loss.abs(),
            "{case}: loss {actual_loss} != {expected_loss}"
        );
        stages::bwd_stage_logits(&mut m, 32, 0.7 / 32.0);
        stages::bwd_stage_mlp(&mut m, 32);
        stages::bwd_stage_adapter(&mut m, 32);
        stages::bwd_stage_adapter_down(&mut m, 32);
        replay_memory_backward(&mut m, &mut s);
        stages::bwd_stage_ssm(&mut m, 32);
        check_training_gradients(&m, &expected);
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
