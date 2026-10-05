use pssa::gpu_batch::*;
use pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::time::Instant;

const L: usize = 512;
const ROUNDS: usize = 5;
fn model() -> PSSALayerV2 {
    let cfg = PSSAConfigV2 { d_vocab: 256, d_latent: 64, d_state: 8, d_mem_key: 16, mem_capacity: 64, chunk_len: L, lr: 1e-5, ..PSSAConfigV2::default() };
    let mut m = PSSALayerV2::new(cfg, 0x51a7);
    let mut key = vec![0.0; m.cfg.d_mem_key]; let mut value = vec![0.0; m.cfg.d_latent];
    for e in 0..m.cfg.mem_capacity { for i in 0..key.len() { key[i] = 0.00002 * (e as f32 + 1.0) * (i as f32 + 1.0); } for i in 0..value.len() { value[i] = 0.002 * ((e + i) % 13) as f32 - 0.01; } m.memory.insert(&key, &value); }
    m
}
fn prepare(m:&mut PSSALayerV2, x:&[usize], y:&[usize]) { m.tape.x_ids[..L].copy_from_slice(x); m.tape.target_ids[..L].copy_from_slice(y); stage_embed_norm(m,L); stage_projections(m,L); stage_ssm_scan(m,L); stage_memory(m,L); stage_adapter(m,L); stage_mlp(m,L); stage_logits_loss(m,L); }
fn median(v:&mut [f64])->f64 { v.sort_by(|a,b|a.total_cmp(b)); v[v.len()/2] }
fn main() {
    let seed=model(); let x:Vec<_>=(0..L).map(|i|(i*17+3)%seed.cfg.d_vocab).collect(); let y:Vec<_>=(0..L).map(|i|(i*23+11)%seed.cfg.d_vocab).collect();
    let names=["embed_norm","projections","ssm_scan","memory","adapter","mlp","logits"]; let mut f=Vec::new();
    for stage in 0..7 { let mut samples=Vec::new(); for _ in 0..ROUNDS { let mut m=model(); m.tape.x_ids[..L].copy_from_slice(&x); m.tape.target_ids[..L].copy_from_slice(&y); if stage>0 {stage_embed_norm(&mut m,L);} if stage>1 {stage_projections(&mut m,L);} if stage>2 {stage_ssm_scan(&mut m,L);} if stage>3 {stage_memory(&mut m,L);} if stage>4 {stage_adapter(&mut m,L);} if stage>5 {stage_mlp(&mut m,L);} let st=Instant::now(); match stage {0=>stage_embed_norm(&mut m,L),1=>stage_projections(&mut m,L),2=>stage_ssm_scan(&mut m,L),3=>stage_memory(&mut m,L),4=>stage_adapter(&mut m,L),5=>stage_mlp(&mut m,L),_=>{stage_logits_loss(&mut m,L);}} samples.push(st.elapsed().as_secs_f64()*1000.0); } f.push(median(&mut samples)); }
    println!("forward_ms {:?}", names.iter().zip(&f).collect::<Vec<_>>());
    let names=["logits","mlp","adapter","adapter_down","memory","ssm"]; let mut b=Vec::new();
    for stage in 0..6 { let mut samples=Vec::new(); for _ in 0..ROUNDS { let mut m=model(); prepare(&mut m,&x,&y); m.zero_gradients(); if stage>0 {bwd_stage_logits(&mut m,L,1.0/L as f32);} if stage>1 {bwd_stage_mlp(&mut m,L);} if stage>2 {bwd_stage_adapter(&mut m,L);} if stage>3 {bwd_stage_adapter_down(&mut m,L);} if stage>4 {bwd_stage_memory(&mut m,L);} let st=Instant::now(); match stage {0=>bwd_stage_logits(&mut m,L,1.0/L as f32),1=>bwd_stage_mlp(&mut m,L),2=>bwd_stage_adapter(&mut m,L),3=>bwd_stage_adapter_down(&mut m,L),4=>bwd_stage_memory(&mut m,L),_=>bwd_stage_ssm(&mut m,L)} samples.push(st.elapsed().as_secs_f64()*1000.0); } b.push(median(&mut samples)); }
    println!("backward_ms {:?}", names.iter().zip(&b).collect::<Vec<_>>());
}
