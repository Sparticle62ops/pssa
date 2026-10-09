//! Matched-token CPU ablation. Byte vocabulary, disjoint train/held-out bytes,
//! identical scalar execution path, initialization, optimizer and write policy.
//! No checkpoints: the experimental weights remain attached during evaluation.
//! Example: cargo run --profile fast --example local_mixing_probe -- --seed 7 --sweep
use pssa::{cli::learning_rate_for_update, pssa::{GradientClipOutcome, PSSAConfigV2, PSSALayerV2}};
use std::time::Instant;

fn value(args: &[String], flag: &str, default: usize) -> usize {
    args.iter().position(|x| x == flag).map_or(default, |i| args[i+1].parse().expect(flag))
}
fn score(m: &mut PSSALayerV2, tokens: &[usize]) -> f64 {
    m.reset_recurrent_state();
    let mut sum = 0.;
    for start in (0..tokens.len()-1).step_by(m.cfg.chunk_len) {
        let n = m.cfg.chunk_len.min(tokens.len()-1-start);
        sum += m.forward_train_chunk(&tokens[start..start+n], &tokens[start+1..start+1+n]) as f64*n as f64;
    }
    sum / (tokens.len()-1) as f64
}
fn run(seed: u64, conv: bool, skip: bool, train: &[usize], heldout: &[usize]) {
    let mut m = PSSALayerV2::new(PSSAConfigV2 {
        d_vocab: 256, d_latent: 64, d_state: 8, d_mem_key: 8, mem_capacity: 32,
        chunk_len: 32, lr: 0.003, ..Default::default()
    }, seed);
    m.set_local_conv(conv).unwrap(); m.set_ssm_skip(skip).unwrap();
    m.memory.set_value_cap(Some(1.0));
    let initial_ce = score(&mut m, heldout);
    m.reset_recurrent_state();
    let chunk = m.cfg.chunk_len;
    let accumulate = 8;
    let group_tokens = chunk*accumulate;
    let targets = train.len()-1;
    let updates = targets.div_ceil(group_tokens);
    let warmup = 8.min(updates.saturating_sub(1));
    let mut train_loss = 0.;
    let timer = Instant::now();
    for (step, start) in (0..targets).step_by(group_tokens).enumerate() {
        m.zero_gradients();
        let total = group_tokens.min(targets-start);
        for off in (start..start+total).step_by(chunk) {
            let n = chunk.min(start+total-off);
            let loss = m.forward_train_chunk(&train[off..off+n], &train[off+1..off+1+n]);
            assert!(loss.is_finite());
            m.backward_chunk(n, n as f32 / total as f32);
            m.insert_training_memory(loss,n);
            train_loss += loss as f64*n as f64;
        }
        let lr = learning_rate_for_update(0.003,step+1,updates,warmup).unwrap();
        assert!(matches!(m.apply_adamw_with_grad_clip(lr,1.0),GradientClipOutcome::Applied { .. }));
        m.ema_consolidate_plasticity();
    }
    let elapsed = timer.elapsed().as_secs_f64();
    let eval_timer = Instant::now();
    let ce = score(&mut m,heldout);
    let eval_elapsed = eval_timer.elapsed().as_secs_f64();
    println!("seed={seed} conv={conv} skip={skip} params={} targets={targets} updates={} initial_ce={initial_ce:.6} train_ce={:.6} heldout_ce={ce:.6} train_seconds={elapsed:.6} train_tokens_sec={:.2} eval_tokens_sec={:.2}",
        m.parameter_count(),m.step_counter,train_loss/targets as f64,targets as f64/elapsed,(heldout.len()-1) as f64/eval_elapsed);
}
fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|x| x == flag);
    let train_n = value(&args,"--train-tokens",65536);
    let eval_n = value(&args,"--eval-tokens",8192);
    let eval_start = value(&args,"--eval-start",262144);
    let seed = value(&args,"--seed",7) as u64;
    assert!(train_n > 0 && train_n <= 200000 && eval_n > 0 && eval_start > train_n);
    let path = args.iter().position(|x| x == "--data").map_or("data/downloaded.txt", |i| args[i+1].as_str());
    let bytes = std::fs::read(path).expect("read corpus");
    assert!(eval_start+eval_n < bytes.len());
    let train: Vec<usize> = bytes[..train_n+1].iter().map(|&x| x as usize).collect();
    let heldout: Vec<usize> = bytes[eval_start..eval_start+eval_n+1].iter().map(|&x| x as usize).collect();
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h,&b| (h^b as u64).wrapping_mul(0x100000001b3));
    println!("data={path} fnv1a64={hash:016x} tokenizer=bytes vocab=256 train_bytes=0..{} eval_bytes={eval_start}..{} width=64 state=8 key=8 memory=32 chunk=32 accumulate=8 lr=0.003 cosine_warmup=8 clip=1 memory_cap=1 threads=1 engine=scalar",train_n+1,eval_start+eval_n+1);
    if has("--sweep") {
        // Opposite run order on the second seed reduces simple order bias.
        let mut modes = vec![(false,false),(true,false),(false,true),(true,true)];
        if seed % 2 == 0 { modes.reverse(); }
        for (conv,skip) in modes { run(seed,conv,skip,&train,&heldout); }
    } else {
        run(seed,has("--local-conv"),has("--ssm-skip"),&train,&heldout);
    }
}
