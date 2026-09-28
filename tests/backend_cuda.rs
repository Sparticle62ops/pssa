#![cfg(feature = "cuda")]

use oxide_ai_pssa::cuda::CudaContext;

#[test]
fn cuda_initialization_child() {
    let Ok(expected) = std::env::var("OXIDE_CUDA_CHILD_EXPECT_ERROR") else {
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
        let expected = oxide_ai_pssa::backend::gemm_cpu_reference(&x, &w, 5, 11, 7, 3);
        let actual = ctx.try_dispatch_gemm(&x, &w, 5, 11, 7, 3).unwrap();
        assert_close(&actual, &expected);
        let b: Vec<_> = (0..7 * 11).map(|i| (i % 23) as f32 * 0.0625).collect();
        assert_close(
            &ctx.try_gemm_nn(&x, &b, 15, 7, 11).unwrap(),
            &oxide_ai_pssa::backend::gemm_nn_cpu(&x, &b, 15, 7, 11),
        );
        let b: Vec<_> = (0..15 * 11).map(|i| (i % 23) as f32 * 0.0625).collect();
        assert_close(
            &ctx.try_gemm_tn(&x, &b, 15, 7, 11).unwrap(),
            &oxide_ai_pssa::backend::gemm_tn_cpu(&x, &b, 15, 7, 11),
        );
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

#[cfg(target_os = "linux")]
#[test]
fn cuda_missing_driver_symbol_returns_error_in_a_fresh_process() {
    // A loadable but incomplete driver used to pass an availability-only guard,
    // then panic in cudarc's lazy symbol loader. Isolate loader state/env in a
    // subprocess; no unsafe process-global environment mutation in the test.
    let dir = std::env::temp_dir().join(format!("oxide-cuda-symbol-test-{}", std::process::id()));
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
            "OXIDE_CUDA_CHILD_EXPECT_ERROR",
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
