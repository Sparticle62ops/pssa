#![cfg(feature = "cuda")]

use pssa::cuda::CudaContext;

#[test]
fn cuda_initialization_child() {
    let Some(expected) = std::env::var_os("PSSA_CUDA_CHILD_EXPECT_ERROR")
        .or_else(|| std::env::var_os("OXIDE_CUDA_CHILD_EXPECT_ERROR"))
        .and_then(|value| value.into_string().ok())
    else {
        return;
    };
    match CudaContext::init() {
        Ok(_) => panic!("expected initialization failure containing {expected}"),
        Err(error) => assert!(error.contains(&expected), "unexpected error: {error}"),
    }
}

#[test]
fn cuda_init_is_fallible_and_strict_shapes_are_checked_if_available() {
    let ctx = match CudaContext::init() {
        Ok(ctx) => ctx,
        Err(error) => {
            eprintln!("CUDA initialization returned a recoverable error: {error}");
            assert!(!error.is_empty());
            return;
        }
    };
    assert!(ctx.try_dispatch_gemm(&[1.0], &[1.0], 2, 2, 2, 1).is_err());
    assert!(ctx.try_gemm_nn(&[1.0], &[1.0], 2, 2, 2).is_err());
    assert!(ctx.try_gemm_tn(&[1.0], &[1.0], 2, 2, 2).is_err());
    assert!(
        ctx.try_gemm_nn(&[], &[], i32::MAX as usize + 1, 1, 1)
            .is_err()
    );
    assert!(ctx.try_gemm_tn(&[], &[], 1, usize::MAX, 1).is_err());
    assert!(ctx.gemm_nn(&[1.0], &[1.0], 2, 2, 2).is_empty());
    assert!(ctx.gemm_tn(&[1.0], &[1.0], 2, 2, 2).is_empty());
    let w: Vec<_> = (0..11 * 7).map(|i| (i % 17) as f32 * -0.0625).collect();
    for scale in [0.125, 0.25] {
        let x: Vec<_> = (0..3 * 5 * 7).map(|i| (i % 13) as f32 * scale).collect();
        let expected = pssa::backend::gemm_cpu_reference(&x, &w, 5, 11, 7, 3);
        let actual = ctx.try_dispatch_gemm(&x, &w, 5, 11, 7, 3).unwrap();
        assert_close(&actual, &expected);
        assert_close(
            &ctx.try_dispatch_gemm(&x, &w, 1, 11, 7, 15).unwrap(),
            &expected,
        );
        assert_close(
            &ctx.try_dispatch_gemm(&x, &w, 15, 11, 7, 1).unwrap(),
            &expected,
        );
        let b: Vec<_> = (0..7 * 11).map(|i| (i % 23) as f32 * 0.0625).collect();
        assert_close(
            &ctx.try_gemm_nn(&x, &b, 15, 7, 11).unwrap(),
            &pssa::backend::gemm_nn_cpu(&x, &b, 15, 7, 11),
        );
        let b: Vec<_> = (0..15 * 11).map(|i| (i % 23) as f32 * 0.0625).collect();
        assert_close(
            &ctx.try_gemm_tn(&x, &b, 15, 7, 11).unwrap(),
            &pssa::backend::gemm_tn_cpu(&x, &b, 15, 7, 11),
        );
        let mut accumulated = vec![0.25; 7 * 11];
        ctx.try_gemm_tn_accumulate_into(&x, &b, 15, 7, 11, &mut accumulated)
            .unwrap();
        let expected = pssa::backend::gemm_tn_cpu(&x, &b, 15, 7, 11);
        for (actual, product) in accumulated.iter().zip(expected) {
            assert!((actual - (0.25 + product)).abs() < 1e-4);
        }
        // nn's second operand is cached; host storage can be reused next loop.
        ctx.invalidate_weights();
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.is_finite() && (a - b).abs() < 1e-4)
    );
}

fn assert_close_stage(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 3e-3,
            "{a} != {b}"
        );
    }
}

#[test]
#[ignore]
fn cuda_ssm_scan_forward_and_backward_match_cpu_twin() {
    let ctx = CudaContext::init().expect("CUDA is required for the ignored parity test");
    let gpu = pssa::backend::GpuDispatch::Cuda(ctx);
    let (l, dm, ds) = (7usize, 3usize, 2usize);
    let delta: Vec<f32> = vec![
        0.2, 0.4, 0.3, 0.5, 0.25, 0.6, 0.35, 0.45, 0.55, 0.3, 0.7, 0.2, 0.5, 0.4, 0.65, 0.25, 0.35,
        0.8, 0.45, 0.3, 0.6,
    ];
    let delta_raw = delta.iter().map(|x| x - 0.1).collect::<Vec<_>>();
    let b_proj = (0..l * ds)
        .map(|i| 0.03 * (i as f32 - 4.0))
        .collect::<Vec<_>>();
    let x_norm = (0..l * dm)
        .map(|i| 0.1 + 0.02 * i as f32)
        .collect::<Vec<_>>();
    let rates = vec![-0.2, -0.35, -0.25, -0.45, -0.3, -0.15];
    let rate_deriv = vec![0.7, 0.8, 0.9, 0.6, 0.75, 0.85];
    let c_proj = (0..l * ds)
        .map(|i| -0.04 + 0.01 * i as f32)
        .collect::<Vec<_>>();
    let initial = vec![0.01, -0.02, 0.03, -0.04, 0.02, -0.01];
    let stride = dm * ds;
    let mut cpu_a = vec![0.0; l * stride];
    let mut cpu_b = vec![0.0; l * stride];
    let mut cpu_states = vec![0.0; (l + 1) * stride];
    let mut cpu_y = vec![0.0; l * dm];
    cpu_states[..stride].copy_from_slice(&initial);
    for t in 0..l {
        for i in 0..dm {
            for j in 0..ds {
                let c = i * ds + j;
                let a = (delta[t * dm + i] * rates[c]).exp();
                let b = delta[t * dm + i] * b_proj[t * ds + j];
                cpu_a[t * stride + c] = a;
                cpu_b[t * stride + c] = b;
                let h = a * cpu_states[t * stride + c] + b * x_norm[t * dm + i];
                cpu_states[(t + 1) * stride + c] = h;
                cpu_y[t * dm + i] += h * c_proj[t * ds + j];
            }
        }
    }
    let mut a = vec![0.0; l * stride];
    let mut b = vec![0.0; l * stride];
    let mut states = vec![0.0; (l + 1) * stride];
    let mut y = vec![0.0; l * dm];
    gpu.ssm_forward(
        &delta,
        &delta_raw,
        &b_proj,
        &x_norm,
        &rates,
        &rate_deriv,
        &c_proj,
        &initial,
        l,
        dm,
        ds,
        &mut a,
        &mut b,
        &mut states,
        &mut y,
    )
    .unwrap();
    assert_close_stage(&a, &cpu_a);
    assert_close_stage(&b, &cpu_b);
    assert_close_stage(&states, &cpu_states);
    assert_close_stage(&y, &cpu_y);

    let gz = (0..l * dm)
        .map(|i| 0.02 * (i as f32 + 1.0))
        .collect::<Vec<_>>();
    let gym = (0..l * dm)
        .map(|i| -0.01 + 0.005 * i as f32)
        .collect::<Vec<_>>();
    let scale = 1.0 / (ds as f32).sqrt();
    let mut exp_gd = vec![0.0; l * dm];
    let mut exp_gb = vec![0.0; l * ds];
    let mut exp_gc = vec![0.0; l * ds];
    let mut exp_ga = vec![0.0; l * stride];
    let mut exp_gx = vec![0.0; l * dm];
    let mut future = vec![0.0; stride];
    for t in (0..l).rev() {
        let mut next = vec![0.0; stride];
        for i in 0..dm {
            let gy = gz[t * dm + i] * scale + gym[t * dm + i];
            for j in 0..ds {
                let c = i * ds + j;
                let q = gy * c_proj[t * ds + j] + future[c];
                let hprev = cpu_states[t * stride + c];
                exp_gc[t * ds + j] += gy * cpu_states[(t + 1) * stride + c];
                exp_ga[t * stride + c] =
                    q * delta[t * dm + i] * cpu_a[t * stride + c] * hprev * rate_deriv[c];
                exp_gd[t * dm + i] += q
                    * (rates[c] * cpu_a[t * stride + c] * hprev
                        + b_proj[t * ds + j] * x_norm[t * dm + i]);
                exp_gb[t * ds + j] += q * delta[t * dm + i] * x_norm[t * dm + i];
                exp_gx[t * dm + i] += q * cpu_b[t * stride + c];
                next[c] = q * cpu_a[t * stride + c];
            }
            exp_gd[t * dm + i] *= pssa::linalg::sigmoid(delta_raw[t * dm + i]);
        }
        future = next;
    }
    let mut gd = vec![0.0; l * dm];
    let mut gb = vec![0.0; l * ds];
    let mut gc = vec![0.0; l * ds];
    let mut ga = vec![0.0; l * stride];
    let mut gx = vec![0.0; l * dm];
    gpu.ssm_backward(
        &delta,
        &delta_raw,
        &b_proj,
        &c_proj,
        &rates,
        &rate_deriv,
        &x_norm,
        &cpu_states,
        &cpu_a,
        &cpu_b,
        &gz,
        &gym,
        l,
        dm,
        ds,
        scale,
        &mut gd,
        &mut gb,
        &mut gc,
        &mut ga,
        &mut gx,
    )
    .unwrap();
    assert_close_stage(&gd, &exp_gd);
    assert_close_stage(&gb, &exp_gb);
    assert_close_stage(&gc, &exp_gc);
    assert_close_stage(&ga, &exp_ga);
    assert_close_stage(&gx, &exp_gx);

    // Exercise the exact token-gradient -> shared-weight boundary that feeds
    // w_b/w_c. Check relative error as well as the stage's absolute tolerance:
    // a small but entirely missing gradient must not pass as "close to zero".
    for (name, actual, expected) in [("w_b", &gb, &exp_gb), ("w_c", &gc, &exp_gc)] {
        let expected_w = pssa::backend::gemm_tn_cpu(expected, &x_norm, l, ds, dm);
        let magnitude = expected_w.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(magnitude > 1e-6, "fixture must exercise {name}");
        let mut actual_w = vec![0.0; ds * dm];
        for count in 1..=2 {
            gpu.gemm_tn_accumulate_into(actual, &x_norm, l, ds, dm, &mut actual_w)
                .unwrap();
            let error = actual_w
                .iter()
                .zip(&expected_w)
                .map(|(a, e)| (a - count as f32 * e).abs())
                .fold(0.0, f32::max);
            assert!(
                error / (count as f32 * magnitude) < 1e-3,
                "{name} gradient relative error: {}",
                error / (count as f32 * magnitude)
            );
        }
    }
}

#[test]
#[ignore]
fn cuda_memory_forward_and_backward_match_cpu_twin() {
    let ctx = CudaContext::init().expect("CUDA is required for the ignored parity test");
    let gpu = pssa::backend::GpuDispatch::Cuda(ctx);
    let (l, dm, dk, cap, count) = (4usize, 3usize, 2usize, 3usize, 2usize);
    let x = vec![
        0.2, -0.1, 0.3, 0.1, 0.4, -0.2, -0.3, 0.2, 0.05, 0.25, -0.15, 0.35,
    ];
    let y = vec![
        0.1, 0.3, -0.2, -0.2, 0.05, 0.4, 0.25, -0.3, 0.2, 0.1, 0.15, -0.05,
    ];
    let w_qx = vec![0.2, -0.1, 0.3, -0.2, 0.15, 0.05];
    let w_qh = vec![-0.1, 0.25, 0.2, 0.15, -0.2, 0.1];
    let w_gate = vec![0.1, -0.2, 0.15, 0.2, 0.05, -0.1, -0.15, 0.1, 0.2];
    let w_proj = vec![0.2, 0.1, -0.1, -0.15, 0.25, 0.05, 0.1, -0.2, 0.15];
    let keys = vec![0.1, -0.2, -0.15, 0.2, 0.0, 0.0];
    let norm_sq = vec![0.05, 0.0625, 0.0];
    let values = vec![0.2, -0.1, 0.3, -0.4, 0.1, 0.25, 0.0, 0.0, 0.0];
    let tau = 0.7;
    let mut qe = vec![0.0; l * dk];
    let mut qp = vec![0.0; l * dk];
    let mut qn = vec![0.0; l];
    let mut weights = vec![0.0; l * cap];
    let mut mv = vec![0.0; l * dm];
    let mut gm = vec![0.0; l * dm];
    let mut mp = vec![0.0; l * dm];
    let mut inj = vec![0.0; l * dm];
    gpu.memory_forward(
        &x,
        &y,
        &w_qx,
        &w_qh,
        &w_gate,
        &w_proj,
        &keys,
        &norm_sq,
        &values,
        l,
        dm,
        dk,
        dm,
        cap,
        count,
        tau,
        &mut qe,
        &mut qp,
        &mut qn,
        &mut weights,
        &mut mv,
        &mut gm,
        &mut mp,
        &mut inj,
    )
    .unwrap();
    let mut cqe = vec![0.0; l * dk];
    let mut cqp = vec![0.0; l * dk];
    let mut cqn = vec![0.0; l];
    let mut cw = vec![0.0; l * cap];
    let mut cmv = vec![0.0; l * dm];
    let mut cgm = vec![0.0; l * dm];
    let mut cmp = vec![0.0; l * dm];
    let mut cinj = vec![0.0; l * dm];
    let bank = pssa::memory::HyperbolicEpisodicBankV2 {
        capacity: cap,
        count,
        dim_key: dk,
        dim_val: dm,
        write_head: 0,
        keys: keys.clone(),
        values: values.clone(),
        value_cap: None,
        norm_sq: norm_sq.clone(),
        confidence: vec![1.0; cap],
        last_seen_step: vec![0; cap],
    };
    for t in 0..l {
        for k in 0..dk {
            cqe[t * dk + k] = (0..dm)
                .map(|j| w_qx[k * dm + j] * x[t * dm + j] + w_qh[k * dm + j] * y[t * dm + j])
                .sum();
        }
        cqn[t] = pssa::memory::HyperbolicEpisodicBankV2::diffeomorphic_project(
            &cqe[t * dk..(t + 1) * dk],
            &mut cqp[t * dk..(t + 1) * dk],
        );
        bank.retrieve_soft_into(
            &cqp[t * dk..(t + 1) * dk],
            tau,
            &mut cmv[t * dm..(t + 1) * dm],
            &mut cw[t * cap..(t + 1) * cap],
        );
        for i in 0..dm {
            cgm[t * dm + i] =
                pssa::linalg::sigmoid((0..dm).map(|j| w_gate[i * dm + j] * x[t * dm + j]).sum());
            cmp[t * dm + i] = (0..dm).map(|j| w_proj[i * dm + j] * cmv[t * dm + j]).sum();
            cinj[t * dm + i] = cgm[t * dm + i] * cmp[t * dm + i];
        }
    }
    assert_close_stage(&qe, &cqe);
    assert_close_stage(&qp, &cqp);
    assert_close_stage(&qn, &cqn);
    assert_close_stage(&weights, &cw);
    assert_close_stage(&mv, &cmv);
    assert_close_stage(&gm, &cgm);
    assert_close_stage(&mp, &cmp);
    assert_close_stage(&inj, &cinj);
    let gmval = (0..l * dm)
        .map(|i| 0.03 * (i as f32 - 2.0))
        .collect::<Vec<_>>();
    let mut exp_qp = vec![0.0; l * dk];
    let mut exp_qe = vec![0.0; l * dk];
    for t in 0..l {
        let q = &cqp[t * dk..(t + 1) * dk];
        let qe0 = &cqe[t * dk..(t + 1) * dk];
        let out = &mut exp_qp[t * dk..(t + 1) * dk];
        let qsq: f64 = q.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        for e in 0..count {
            let mut dot = 0.;
            let mut sq = 0.;
            for j in 0..dm {
                dot += gmval[t * dm + j] as f64 * (values[e * dm + j] - cmv[t * dm + j]) as f64;
            }
            for k in 0..dk {
                let diff = q[k] as f64 - keys[e * dk + k] as f64;
                sq += diff * diff;
            }
            if sq > 0. {
                let denom = (1. - qsq) * (1. - norm_sq[e] as f64);
                let z = sq / denom;
                let coeff =
                    cw[t * cap + e] as f64 * dot * (-1. / tau as f64) / (z * (1. + z)).sqrt();
                for k in 0..dk {
                    let diff = q[k] as f64 - keys[e * dk + k] as f64;
                    let dd = -2. * q[k] as f64 * (1. - norm_sq[e] as f64);
                    out[k] += (coeff * (2. * diff * denom - sq * dd) / (denom * denom)) as f32;
                }
            }
        }
        pssa::memory::HyperbolicEpisodicBankV2::projection_adjoint(
            qe0,
            out,
            &mut exp_qe[t * dk..(t + 1) * dk],
        );
    }
    let mut got_qp = vec![0.; l * dk];
    let mut got_qe = vec![0.; l * dk];
    gpu.memory_backward_retrieval(
        &cqp,
        &cqe,
        &gmval,
        &cmv,
        &cw,
        &keys,
        &norm_sq,
        &values,
        l,
        count,
        cap,
        dk,
        dm,
        tau,
        &mut got_qp,
        &mut got_qe,
    )
    .unwrap();
    assert_close_stage(&got_qp, &exp_qp);
    assert_close_stage(&got_qe, &exp_qe);
}

#[test]
#[ignore = "requires a CUDA GPU; exercises the actual training memory strides"]
fn cuda_memory_training_strides_extreme_queries_and_inactive_values_match_cpu() {
    use pssa::memory::HyperbolicEpisodicBankV2 as Bank;
    let gpu = pssa::backend::GpuDispatch::Cuda(CudaContext::init().expect("CUDA required"));
    let (l, dm, dk, cap) = (32, 3584, 32, 512);
    let mut x = vec![0.0; l * dm];
    for t in 0..l {
        x[t * dm] = [0.25, 16_777_216.0, f32::MAX, 1e-30][t % 4];
    }
    let y = vec![0.0; l * dm];
    let mut w_qx = vec![0.0; dk * dm];
    w_qx[0] = 1.0;
    let w_qh = vec![0.0; dk * dm];
    let w_gate = vec![0.0; dm * dm];
    let mut w_proj = vec![0.0; dm * dm];
    for i in 0..dm {
        w_proj[i * dm + i] = 1.0;
    }
    for count in [0, 2, cap] {
        let mut bank = Bank::new(cap, dk, dm);
        for e in 0..count {
            let mut key = vec![0.0; dk];
            key[0] = 0.25 * ((e % 3) as f32 - 1.0);
            bank.insert(&key, &vec![0.01 * ((e % 7) as f32 - 3.0); dm]);
        }
        // CPU retrieval never reads inactive slots. Nor may the new GEMM:
        // multiplying these by cleared weights would still yield NaN.
        bank.values[count * dm..].fill(f32::NAN);
        let mut qe = vec![23.0; l * dk];
        let mut qp = qe.clone();
        let mut qn = vec![23.0; l];
        let mut weights = vec![23.0; l * cap];
        let mut mv = vec![23.0; l * dm];
        let mut gm = mv.clone();
        let mut mp = mv.clone();
        let mut inj = mv.clone();
        gpu.memory_forward(
            &x,
            &y,
            &w_qx,
            &w_qh,
            &w_gate,
            &w_proj,
            &bank.keys,
            &bank.norm_sq,
            &bank.values,
            l,
            dm,
            dk,
            dm,
            cap,
            count,
            0.7,
            &mut qe,
            &mut qp,
            &mut qn,
            &mut weights,
            &mut mv,
            &mut gm,
            &mut mp,
            &mut inj,
        )
        .unwrap();
        let mut expected_qe = vec![0.0; l * dk];
        let mut expected_qp = expected_qe.clone();
        let mut expected_qn = vec![0.0; l];
        let mut expected_weights = vec![0.0; l * cap];
        let mut expected_mv = vec![0.0; l * dm];
        for t in 0..l {
            expected_qe[t * dk] = x[t * dm];
            expected_qn[t] = Bank::diffeomorphic_project(
                &expected_qe[t * dk..(t + 1) * dk],
                &mut expected_qp[t * dk..(t + 1) * dk],
            );
            bank.retrieve_soft_into(
                &expected_qp[t * dk..(t + 1) * dk],
                0.7,
                &mut expected_mv[t * dm..(t + 1) * dm],
                &mut expected_weights[t * cap..(t + 1) * cap],
            );
        }
        assert_close_stage(&qe, &expected_qe);
        assert_close_stage(&qp, &expected_qp);
        assert_close_stage(&qn, &expected_qn);
        assert_close_stage(&weights, &expected_weights);
        assert_close_stage(&mv, &expected_mv);
        assert_close_stage(&gm, &vec![0.5; l * dm]);
        assert_close_stage(&mp, &expected_mv);
        assert_close_stage(
            &inj,
            &expected_mv.iter().map(|x| 0.5 * x).collect::<Vec<_>>(),
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn cuda_missing_driver_symbol_returns_error_in_a_fresh_process() {
    // A loadable but incomplete driver used to pass an availability-only guard,
    // then panic in cudarc's lazy symbol loader. Isolate loader state/env in a
    // subprocess; no unsafe process-global environment mutation in the test.
    let dir = std::env::temp_dir().join(format!("pssa-cuda-symbol-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("stub.c");
    std::fs::write(
        &source,
        "int cuInit(unsigned int flags) { (void)flags; return 0; }\n",
    )
    .unwrap();
    let compiler = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    let compile = std::process::Command::new(compiler)
        .args(["-shared", "-fPIC", "-o"])
        .arg(dir.join("libcuda.so"))
        .arg(&source)
        .output();
    let compile = match compile {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("C compiler unavailable; CUDA stub-library subprocess test skipped");
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        Err(error) => panic!("could not build CUDA stub: {error}"),
    };
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "cuda_initialization_child", "--nocapture"])
        .env("LD_LIBRARY_PATH", &dir)
        .env(
            "PSSA_CUDA_CHILD_EXPECT_ERROR",
            "missing required symbol cuDeviceGet",
        )
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
