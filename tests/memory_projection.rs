use oxide_ai_pssa::memory::HyperbolicEpisodicBankV2 as Bank;

#[test]
fn finite_extreme_queries_project_strictly_inside_the_ball() {
    for q in [
        vec![0.0],
        vec![16_777_216.0],
        vec![-f32::MAX],
        vec![f32::MAX, -f32::MAX, f32::MAX],
        vec![f32::MAX; 4096],
        vec![f32::from_bits(1), -f32::from_bits(1)],
    ] {
        let mut projected = vec![0.0; q.len()];
        let norm = Bank::diffeomorphic_project(&q, &mut projected);
        assert!(norm.is_finite());
        assert!(projected.iter().all(|x| x.is_finite()));
        assert!(Bank::squared_norm(&projected) < 1.0);
        if q.iter().any(|x| x.abs() > 1.0) {
            assert!(
                Bank::squared_norm(&projected) > 0.99,
                "large input collapsed to zero"
            );
        }
        let mut bank = Bank::new(1, q.len(), 2);
        bank.insert(&projected, &[2.0, -3.0]);
        let mut out = [0.0; 2];
        let mut weights = [0.0];
        bank.retrieve_soft_into(&projected, f32::from_bits(1), &mut out, &mut weights);
        assert_eq!(out, [2.0, -3.0]);
        assert_eq!(weights, [1.0]);
        let mut adjoint = vec![0.0; q.len()];
        Bank::projection_adjoint(&q, &vec![0.25; q.len()], &mut adjoint);
        assert!(adjoint.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn tiny_temperature_keeps_nearest_entries_and_ties_finite() {
    let mut bank = Bank::new(3, 1, 2);
    bank.insert(&[0.5], &[1.0, 2.0]);
    let mut out = [0.0; 2];
    let mut weights = [0.0; 3];
    for tau in [1e-40, f32::from_bits(1)] {
        bank.retrieve_soft_into(&[0.0], tau, &mut out, &mut weights);
        assert_eq!(out, [1.0, 2.0]);
        assert_eq!(weights[0], 1.0);
    }
    bank.insert(&[-0.5], &[3.0, -2.0]);
    bank.insert(&[0.75], &[100.0, 100.0]);
    bank.retrieve_soft_into(&[0.0], f32::from_bits(1), &mut out, &mut weights);
    assert_eq!(weights, [0.5, 0.5, 0.0]);
    assert_eq!(out, [2.0, 0.0]);
}

#[test]
fn projection_adjoint_matches_finite_differences_before_and_after_saturation() {
    let upstream = [0.7, -0.4, 0.3];
    for (q, h) in [
        ([0.0, 0.0, 0.0], 1e-4),
        ([0.2, -0.7, 1.1], 1e-3),
        ([2e7, -3e7, 1e7], 1e4),
    ] {
        let mut analytic = [0.0; 3];
        Bank::projection_adjoint(&q, &upstream, &mut analytic);
        for i in 0..3 {
            let evaluate = |delta: f32| {
                let mut input = q;
                input[i] += delta;
                let mut output = [0.0; 3];
                Bank::diffeomorphic_project(&input, &mut output);
                output
                    .iter()
                    .zip(upstream)
                    .map(|(&x, g)| x as f64 * g as f64)
                    .sum::<f64>()
            };
            let numeric = (evaluate(h) - evaluate(-h)) / (2.0 * h as f64);
            let analytic = analytic[i] as f64;
            let limit = 2e-4 * analytic.abs().max(numeric.abs()) + 2e-8 / h as f64;
            assert!(
                (numeric - analytic).abs() < limit,
                "q={q:?} i={i} analytic={analytic} numeric={numeric} limit={limit}"
            );
        }
    }
    let mut radial = [1.0];
    Bank::projection_adjoint(&[16_777_216.0], &[1.0], &mut radial);
    assert_eq!(radial, [0.0], "saturated one-dimensional map is constant");
}
