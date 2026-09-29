//! Count all threads, not just the caller, after initializing the reusable Rayon
//! pool. Keep the original cold-path allocation tests in allocations.rs intact.
use oxide_ai_pssa::{
    gpu_batch,
    pssa::{PSSAConfigV2, PSSALayerV2},
    sequence_batch::{Sequence, SequenceBatch},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct CountingAllocator;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if ACTIVE.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if ACTIVE.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, len: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, len) };
        if ACTIVE.load(Ordering::Relaxed) {
            REALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn staged_and_packed_scans_reuse_storage_on_all_threads() {
    let cfg = PSSAConfigV2 {
        d_vocab: 17, d_latent: 16, d_state: 3, d_mem_key: 4,
        mem_capacity: 4, chunk_len: 257, ..Default::default()
    };
    let mut staged = PSSALayerV2::new(cfg.clone(), 91);
    let mut packed = PSSALayerV2::new(cfg, 91);
    let mut batch = SequenceBatch::new(&mut packed, 2).unwrap();
    let inputs: Vec<_> = (0..257).map(|i| i % 17).collect();
    let targets: Vec<_> = (0..257).map(|i| (i + 1) % 17).collect();
    let step = |staged: &mut PSSALayerV2, packed: &mut PSSALayerV2, batch: &mut SequenceBatch, len| {
        staged.zero_gradients();
        gpu_batch::forward_train_chunk_batched(staged, &inputs[..len], &targets[..len]);
        gpu_batch::backward_chunk_batched(staged, len, 1.0);
        staged.apply_adamw(staged.cfg.lr);
        staged.ema_consolidate_plasticity();
        packed.zero_gradients();
        batch.forward(packed, &[
            Sequence { lane: 0, inputs: &inputs[..len], targets: &targets[..len], reset: true },
            Sequence { lane: 1, inputs: &inputs[..len - 1], targets: &targets[..len - 1], reset: false },
        ]).unwrap();
        batch.backward(packed, 1.0).unwrap();
        packed.apply_adamw(packed.cfg.lr);
    };
    // One-time scheduler/worker setup is permitted, but no per-step storage.
    for len in [257, 256, 64] {
        step(&mut staged, &mut packed, &mut batch, len);
    }
    ALLOCS.store(0, Ordering::Relaxed);
    REALLOCS.store(0, Ordering::Relaxed);
    ACTIVE.store(true, Ordering::SeqCst);
    for round in 0..8 {
        for len in [257, 256, 64] {
            step(&mut staged, &mut packed, &mut batch, len);
            eprintln!("round={round} len={len} alloc={} realloc={}", ALLOCS.load(Ordering::Relaxed), REALLOCS.load(Ordering::Relaxed));
        }
    }
    ACTIVE.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCS.load(Ordering::Relaxed), 0, "all-thread live allocations");
    assert_eq!(REALLOCS.load(Ordering::Relaxed), 0, "all-thread live reallocations");
}
