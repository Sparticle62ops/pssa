use oxide_ai_pssa::linalg::dot_slice;

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
