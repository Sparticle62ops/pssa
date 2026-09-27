use oxide_ai_pssa::backend::{gemm_cpu_reference, gemm_nn_cpu, gemm_tn_cpu};

fn values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.03125)
        .collect()
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
        let actual = gemm_nn_cpu(
            &x[b * m * k..(b + 1) * m * k],
            &values(k * n),
            m,
            k,
            n,
        );
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
}
