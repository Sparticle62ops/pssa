use oxide_ai_pssa::gpu_batch::{backward_chunk_batched, forward_train_chunk_batched};
use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};

fn model(raw_rate: f32, raw_delta: f32, carry: f32, input: f32) -> PSSALayerV2 {
    let mut cfg = PSSAConfigV2::default();
    cfg.d_latent = 1;
    cfg.d_state = 1;
    cfg.d_mem_key = 1;
    cfg.d_vocab = 3;
    cfg.chunk_len = 2;
    cfg.mem_capacity = 1;
    let mut m = PSSALayerV2::new(cfg, 9);
    m.norm_gamma.data[0] = 0.0;
    m.norm_beta.data[0] = 1.0;
    m.a_mat.data[0] = raw_rate;
    m.w_delta.data[0] = raw_delta;
    m.h_persistent[0] = carry;
    m.w_b.data[0] = input;
    m.w_c.data[0] = 1.0;
    m.unembed_w.data.copy_from_slice(&[-1.0, 0.0, 1.0]);
    m
}

#[test]
fn rate_cache_refreshes_after_direct_public_parameter_mutation_on_every_path() {
    for staged in [false, true] {
        let mut cached = model(-1.0, 0.2, 1.0, 0.0);
        for rate in [-1.0, -20.0, 2.0, -1.0] {
            cached.a_mat.data[0] = rate;
            cached.h_persistent[0] = 1.0;
            let mut fresh = model(rate, 0.2, 1.0, 0.0);
            fresh.forward_train_chunk(&[0, 1], &[1, 2]);
            if staged {
                forward_train_chunk_batched(&mut cached, &[0, 1], &[1, 2]);
            } else {
                cached.forward_train_chunk(&[0, 1], &[1, 2]);
            }
            assert_eq!(cached.h_persistent, fresh.h_persistent);
            cached.zero_gradients();
            fresh.backward_chunk(2, 1.0);
            if staged {
                backward_chunk_batched(&mut cached, 2, 1.0);
            } else {
                cached.backward_chunk(2, 1.0);
            }
            assert!((cached.a_mat.grad[0] - fresh.a_mat.grad[0]).abs() < 1e-7);
        }
        cached.a_mat.data[0] = 0.7;
        cached.h_persistent[0] = 1.0;
        let mut fresh = model(0.7, 0.2, 1.0, 0.0);
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        cached.forward_inference(0, &mut a);
        fresh.forward_inference(0, &mut b);
        assert_eq!(a, b);
        assert_eq!(cached.h_persistent, fresh.h_persistent);
    }
}
