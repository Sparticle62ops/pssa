use oxide_ai_pssa::linalg::{Matrix, Vector, dot_slice, sigmoid, softplus};

fn fixture(len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(len);
    let mut b = Vec::with_capacity(len);
    for i in 0..len {
        a.push(((i * 37 % 127) as f32 - 63.0) * 0.03125);
        b.push(((i * 53 % 113) as f32 - 56.0) * -0.0234375);
    }
    if len > 0 {
        a[0] = -0.0;
        b[0] = 3.0;
    }
    if len > 1 {
        a[1] = f32::from_bits(1); // smallest positive subnormal
        b[1] = -1.0;
    }
    if len > 2 {
        a[2] = -f32::MIN_POSITIVE;
        b[2] = 0.5;
    }
    if len > 3 {
        a[3] = 1.0e-20;
        b[3] = -1.0e20;
    }
    (a, b)
}

#[test]
fn dot_slice_matches_f64_reference_across_vector_and_tail_lengths() {
    let lengths = (0..=65)
        .chain([127, 128, 255, 256, 257, 1024])
        .collect::<Vec<_>>();
    for len in lengths {
        let (a, b) = fixture(len);
        let reference: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let actual = dot_slice(&a, &b);
        let tolerance = 2.0e-6 + 2.0e-5 * reference.abs() as f32;
        assert!(actual.is_finite(), "length {len}: result was {actual}");
        assert!(
            (actual - reference as f32).abs() <= tolerance,
            "length {len}: actual={actual:?}, reference={reference:?}, tolerance={tolerance:?}"
        );
    }
}

#[test]
fn dot_slice_rejects_mismatched_lengths() {
    assert!(std::panic::catch_unwind(|| dot_slice(&[1.0, 2.0], &[1.0])).is_err());
}

#[test]
fn vector_and_matrix_additions_reject_shape_mismatches() {
    assert!(
        std::panic::catch_unwind(|| {
            Vector::from_slice(&[1.0, 2.0]).add(&Vector::from_slice(&[3.0]));
        })
        .is_err()
    );

    let mut lhs = Matrix::zeros(2, 2);
    let rhs = Matrix::zeros(1, 2);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lhs.add_assign_scaled(&rhs, 1.0);
        }))
        .is_err()
    );
}

#[test]
fn empty_rms_norm_is_empty_and_finite() {
    let mut out = Vector::from_slice(&[1.0]);
    Vector::from_slice(&[]).rms_norm_into(&mut out);
    assert!(out.data.is_empty());
}

#[test]
fn matvec_rejects_malformed_public_storage_before_writing() {
    for cols in [1, 5, 32, 33, 128, 256] {
        for len in [0, cols - 1, cols + 1] {
            let matrix = Matrix {
                rows: 1,
                cols,
                data: vec![1.0; len],
            };
            let input = vec![1.0; cols];
            let mut out = [123.0];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.matvec_into(&input, &mut out);
            }));
            assert!(result.is_err(), "cols={cols}, storage={len}");
            assert_eq!(out, [123.0], "validation must precede output writes");
        }
    }
}

#[test]
fn matvec_rejects_overflowing_dimensions_and_mismatched_operands() {
    let overflow = Matrix {
        rows: usize::MAX,
        cols: 2,
        data: vec![],
    };
    let error = std::panic::catch_unwind(|| overflow.matvec_into(&[], &mut []))
        .expect_err("overflowing shape must be rejected");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains("dimensions overflow"), "{message}");

    let matrix = Matrix::zeros(2, 3);
    assert!(std::panic::catch_unwind(|| matrix.matvec_into(&[1.0; 2], &mut [0.0; 2])).is_err());
    assert!(std::panic::catch_unwind(|| matrix.matvec_into(&[1.0; 3], &mut [0.0; 1])).is_err());
}

#[test]
fn matvec_handles_empty_shapes_and_numeric_tails() {
    let mut out = [123.0; 3];
    Matrix::zeros(3, 0).matvec_into(&[], &mut out);
    assert_eq!(out, [0.0; 3]);
    Matrix::zeros(0, 3).matvec_into(&[1.0; 3], &mut []);
    Matrix::zeros(0, 0).matvec_into(&[], &mut []);

    for cols in [1, 5, 31, 32, 33, 127, 128, 129, 255, 256, 257] {
        let (row, input) = fixture(cols);
        let mut data = row.clone();
        data.extend(row.iter().map(|x| -x));
        let matrix = Matrix {
            rows: 2,
            cols,
            data,
        };
        let mut out = [0.0; 2];
        matrix.matvec_into(&input, &mut out);
        let expected: f64 = row
            .iter()
            .zip(input)
            .map(|(&a, b)| a as f64 * b as f64)
            .sum();
        let tolerance = 2.0e-6 + 2.0e-5 * expected.abs();
        assert!((out[0] as f64 - expected).abs() <= tolerance, "cols={cols}");
        assert!((out[1] as f64 + expected).abs() <= tolerance, "cols={cols}");
    }
}

#[test]
fn softplus_preserves_representable_negative_tail_and_derivative() {
    for x in [
        -100.0f32, -80.0, -40.0, -20.0, -17.0, -8.0, 0.0, 8.0, 20.0, 21.0, 100.0,
    ] {
        let actual = softplus(x);
        let expected = (x as f64).exp().ln_1p();
        let tolerance = 4.0 * f32::from_bits(1) as f64 + 2.0e-7 * expected;
        assert!(actual > 0.0 && actual.is_finite(), "x={x}, actual={actual}");
        assert!(
            (actual as f64 - expected).abs() <= tolerance,
            "x={x}, actual={actual}, expected={expected}"
        );
    }
    for x in [-20.0f32, -17.0] {
        let step = 0.01;
        let numeric = (softplus(x + step) - softplus(x - step)) / (2.0 * step);
        let expected = sigmoid(x);
        assert!(
            (numeric / expected - 1.0).abs() < 1.0e-3,
            "x={x}, derivative={numeric}, sigmoid={expected}"
        );
    }
    assert_eq!(softplus(f32::NEG_INFINITY), 0.0);
    assert_eq!(softplus(f32::INFINITY), f32::INFINITY);
    assert!(softplus(f32::NAN).is_nan());
}

fn assert_inverse_residual(matrix: &Matrix, inverse: &Matrix) {
    for (a, b) in [(matrix, inverse), (inverse, matrix)] {
        for row in 0..matrix.rows {
            for col in 0..matrix.cols {
                let product: f64 = (0..matrix.rows)
                    .map(|k| a.data[row * a.cols + k] as f64 * b.data[k * b.cols + col] as f64)
                    .sum();
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (product - expected).abs() < 2.0e-6,
                    "residual at ({row}, {col}): {product} versus {expected}"
                );
            }
        }
    }
}

#[test]
fn inversion_is_scale_invariant_and_eliminates_small_nonzero_factors() {
    // At scale 1e-11 the old absolute elimination cutoff lost the 0.09 term.
    for scale in [1.0e-30f32, 1.0e-20, 1.0e-11, 1.0, 1.0e20, 1.0e30] {
        for values in [[1.0, 0.0, 0.09, 1.0], [0.0, 2.0, 3.0, 4.0]] {
            let matrix = Matrix {
                rows: 2,
                cols: 2,
                data: values.iter().map(|x| x * scale).collect(),
            };
            let inverse = matrix
                .invert()
                .unwrap_or_else(|e| panic!("scale={scale}: {e}"));
            assert_inverse_residual(&matrix, &inverse);
        }
        let singular = Matrix {
            rows: 2,
            cols: 2,
            data: vec![scale, scale, scale, scale],
        };
        assert!(singular.invert().is_err(), "scale={scale}");
    }
    let diagonal = Matrix {
        rows: 2,
        cols: 2,
        data: vec![1.0e-30, 0.0, 0.0, 1.0e30],
    };
    assert_inverse_residual(&diagonal, &diagonal.invert().unwrap());
    assert_eq!(Matrix::zeros(0, 0).invert().unwrap(), Matrix::zeros(0, 0));
}

#[test]
fn inversion_rejects_nonfinite_inputs_results_and_malformed_shapes() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in 0..4 {
            let mut matrix = Matrix {
                rows: 2,
                cols: 2,
                data: vec![1.0, 0.0, 0.0, 1.0],
            };
            matrix.data[index] = value;
            assert!(matrix.invert().is_err(), "index={index}, value={value}");
        }
    }
    let overflowing_inverse = Matrix {
        rows: 1,
        cols: 1,
        data: vec![f32::from_bits(1)],
    };
    assert!(overflowing_inverse.invert().is_err());
    for matrix in [
        Matrix {
            rows: 2,
            cols: 2,
            data: vec![],
        },
        Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0; 5],
        },
        Matrix {
            rows: usize::MAX,
            cols: usize::MAX,
            data: vec![],
        },
        Matrix::zeros(2, 3),
        Matrix::zeros(2, 2),
    ] {
        assert!(matrix.invert().is_err());
    }
}
