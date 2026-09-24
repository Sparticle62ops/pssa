use oxide_ai_pssa::pssa::{PSSAConfigV2, PSSALayerV2};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct TestThreadAllocator;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static REALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record(reallocation: bool) {
    // These const thread-local cells are warmed before tracking starts. Threads
    // other than this test thread retain their default inactive state.
    ACTIVE.with(|active| {
        if active.get() {
            let counter = if reallocation {
                &REALLOCATIONS
            } else {
                &ALLOCATIONS
            };
            counter.with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for TestThreadAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record(false);
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        record(true);
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: TestThreadAllocator = TestThreadAllocator;

fn warm_thread_local_counters() {
    ACTIVE.with(|active| active.set(false));
    ALLOCATIONS.with(|count| count.set(0));
    REALLOCATIONS.with(|count| count.set(0));
}

fn start_counting() {
    ALLOCATIONS.with(|count| count.set(0));
    REALLOCATIONS.with(|count| count.set(0));
    ACTIVE.with(|active| active.set(true));
}

fn stop_counting() -> (usize, usize) {
    ACTIVE.with(|active| active.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    let reallocations = REALLOCATIONS.with(Cell::get);
    (allocations, reallocations)
}

fn config() -> PSSAConfigV2 {
    PSSAConfigV2 {
        d_vocab: 17,
        d_latent: 16,
        d_state: 3,
        d_mem_key: 4,
        mem_capacity: 4,
        chunk_len: 3,
        lr: 1e-5,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.7,
        ema_alpha: 0.1,
    }
}

#[test]
fn live_model_training_inference_memory_and_consolidation_allocate_nothing() {
    // All model, state, input, output, and direct-memory-operation storage is
    // constructed before activation and excluded from the assertion.
    let mut model = PSSALayerV2::new(config(), 91);
    let key = [0.01, -0.02, 0.015, -0.01];
    let value = [
        -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
        0.07, 0.08,
    ];
    for _ in 0..model.cfg.mem_capacity {
        model.memory.insert(&key, &value);
    }
    let tokens = [1, 5, 9];
    let targets = [2, 6, 10];
    let mut logits = vec![0.0; model.cfg.d_vocab];
    let mut retrieved = vec![0.0; model.cfg.d_latent];
    let mut weights = vec![0.0; model.cfg.mem_capacity];

    warm_thread_local_counters();
    start_counting();
    for _ in 0..3 {
        model.reset_recurrent_state();
        model.forward_train_chunk(&tokens, &targets);
        model.zero_gradients();
        model.backward_chunk(tokens.len(), 1.0);
        model.apply_adamw(model.cfg.lr);
        model.reset_recurrent_state();
        model.forward_inference(tokens[0], &mut logits);
        model
            .memory
            .retrieve_soft_into(&key, model.cfg.tau_mem, &mut retrieved, &mut weights);
        model.memory.insert(&key, &value);
        model.ema_consolidate_plasticity();
    }
    let counts = stop_counting();

    assert_eq!(counts.0, 0, "live paths allocated {} times", counts.0);
    assert_eq!(counts.1, 0, "live paths reallocated {} times", counts.1);
}
