use oxide_ai_pssa::backend::{gemm_cpu_into, gemm_cpu_reference, gemm_nn_cpu, gemm_tn_cpu};

fn values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.03125)
        .collect()
}

#[test]
fn shared_weight_batch_and_flat_matrix_have_identical_layouts() {
    for (batch, m, n, k) in [(64, 1, 19, 7), (3, 17, 31, 13), (8, 64, 9, 5)] {
        let x = values(batch * m * k);
        let w = values(n * k);
        assert_eq!(
            gemm_cpu_reference(&x, &w, m, n, k, batch),
            gemm_cpu_reference(&x, &w, batch * m, n, k, 1),
        );
    }
}

#[test]
fn cpu_gemm_twins_cover_non_multiple_tail_shapes() {
    let (batch, m, n, k) = (3, 5, 7, 11);
    let x = values(batch * m * k);
    let b = values(k * n);
    let mut w = vec![0.0; n * k];
    for j in 0..n {
        for kk in 0..k {
            w[j * k + kk] = b[kk * n + j];
        }
    }
    let expected = gemm_cpu_reference(&x, &w, m, n, k, batch);
    for b in 0..batch {
        let actual = gemm_nn_cpu(&x[b * m * k..(b + 1) * m * k], &values(k * n), m, k, n);
        assert!(
            actual
                .iter()
                .zip(&expected[b * m * n..(b + 1) * m * n])
                .all(|(a, e)| (a - e).abs() < 1e-6)
        );
    }
}

#[test]
fn cpu_transposed_gemm_matches_reference_definition() {
    let (m, k, n) = (5, 11, 7);
    let a = values(m * k);
    let b = values(m * n);
    let actual = gemm_tn_cpu(&a, &b, m, k, n);
    let mut expected = vec![0.0; k * n];
    for kk in 0..k {
        for j in 0..n {
            for i in 0..m {
                expected[kk * n + j] += a[i * k + kk] * b[i * n + j];
            }
        }
    }
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() < 1e-6)
    );
}

#[test]
fn cpu_gemm_rejects_bad_buffer_lengths_without_panicking() {
    assert!(gemm_cpu_reference(&[1.0], &[1.0], 2, 2, 2, 1).is_empty());
    assert!(gemm_nn_cpu(&[1.0], &[1.0], 2, 2, 2).is_empty());
    assert!(gemm_tn_cpu(&[1.0], &[1.0], 2, 2, 2).is_empty());
    for (m, n, k, batch) in [(0, 1, 1, 1), (1, 1, 0, 1), (usize::MAX, 2, 2, 2)] {
        assert!(gemm_cpu_reference(&[], &[], m, n, k, batch).is_empty());
        assert!(gemm_nn_cpu(&[], &[], m, k, n).is_empty());
        assert!(gemm_tn_cpu(&[], &[], m, k, n).is_empty());
        let mut out = [123.0];
        assert!(gemm_cpu_into(&[], &[], m, n, k, batch, &mut out).is_err());
        assert_eq!(out, [123.0]);
    }
    let mut out = [123.0; 2];
    assert!(gemm_cpu_into(&[1.0], &[2.0], 1, 1, 1, 1, &mut out).is_err());
    assert_eq!(out, [123.0; 2]);
}

#[test]
fn blocked_cpu_output_into_matches_dot_order_for_tails_and_parallel_tiles() {
    for (batch, m, n, k) in [(3, 5, 7, 11), (1, 1, 1, 1), (1, 17, 513, 513)] {
        let x = values(batch * m * k);
        let w = values(n * k);
        let mut out = vec![f32::NAN; batch * m * n];
        gemm_cpu_into(&x, &w, m, n, k, batch, &mut out).unwrap();
        for t in 0..batch * m {
            for j in 0..n {
                assert_eq!(
                    out[t * n + j],
                    oxide_ai_pssa::linalg::dot_slice(
                        &x[t * k..(t + 1) * k],
                        &w[j * k..(j + 1) * k]
                    )
                );
            }
        }
    }
}

#[test]
fn backend_initialization_environment_child() {
    if std::env::var_os("OXIDE_BACKEND_ENV_CHILD").is_none() {
        return;
    }
    assert!(std::env::var_os("XDG_RUNTIME_DIR").is_none());
    let _ = oxide_ai_pssa::backend::WgpuContext::init_blocking();
    assert!(
        std::env::var_os("XDG_RUNTIME_DIR").is_none(),
        "backend mutated process environment"
    );
}

#[test]
fn backend_initialization_does_not_set_process_environment() {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "backend_initialization_environment_child",
            "--nocapture",
        ])
        .env("OXIDE_BACKEND_ENV_CHILD", "1")
        .env_remove("XDG_RUNTIME_DIR")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// Run with `cargo test --release --test backend backend_kernel_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "manual wall-clock kernel benchmark, not a CI assertion"]
fn backend_kernel_benchmark() {
    use std::{hint::black_box, time::Instant};
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for (m, n, k) in [(64, 256, 256), (64, 2048, 256), (64, 16, 256)] {
                let x = values(m * k);
                let w = values(n * k);
                let mut out = vec![0.0; m * n];
                let serial = |out: &mut [f32]| {
                    for t in 0..m { for j in 0..n {
                        out[t * n + j] = oxide_ai_pssa::linalg::dot_slice(&x[t * k..(t + 1) * k], &w[j * k..(j + 1) * k]);
                    }}
                };
                serial(&mut out);
                gemm_cpu_into(&x, &w, m, n, k, 1, &mut out).unwrap();
                let repetitions = 50;
                let start = Instant::now();
                for _ in 0..repetitions { serial(black_box(&mut out)); }
                let before = start.elapsed();
                let start = Instant::now();
                for _ in 0..repetitions { gemm_cpu_into(&x, &w, m, n, k, 1, black_box(&mut out)).unwrap(); }
                let after = start.elapsed();
                println!("threads={threads} shape={m}x{n}x{k} serial={before:?} blocked={after:?} speedup={:.2}", before.as_secs_f64() / after.as_secs_f64());
            }
        });
    }
}
