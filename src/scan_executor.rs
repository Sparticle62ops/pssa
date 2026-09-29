//! Persistent handoff into Rayon for the staged affine scans.
//!
//! The scan tapes are already model/lane-owned and sized to the configured
//! maximum. Rayon's external injector, however, allocates queue blocks as jobs
//! pass through it, even after warmup. Keep one worker alive and hand it borrowed
//! jobs through a reusable slot; nested parallel iterators use worker-local
//! deques instead. No per-step heap closures or length-dependent storage.
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[derive(Clone, Default)]
pub struct ScanExecutor {
    worker: Arc<OnceLock<Worker>>,
}

impl ScanExecutor {
    pub(crate) fn run<F: FnOnce() + Send>(&self, f: F) {
        // Nested lane scans already run on a worker. Reentering the handoff
        // would deadlock; their parallel iterators need no external injection.
        if rayon::current_thread_index().is_some() {
            f();
            return;
        }
        let worker = self.worker.get_or_init(Worker::new);
        // Only one caller may own the borrowed job slot until completion.
        let _caller = worker.caller.lock().unwrap();
        let mut job = (Some(f), None);
        unsafe fn execute<F: FnOnce() + Send>(ptr: *mut ()) {
            // SAFETY: run keeps this exact tuple alive and exclusively borrowed
            // until the worker publishes completion under the state mutex.
            let job = unsafe { &mut *ptr.cast::<(Option<F>, Option<std::thread::Result<()>>)>() };
            job.1 = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                job.0.take().unwrap(),
            )));
        }
        let mut state = worker.shared.state.lock().unwrap();
        state.job = Some(Job {
            data: (&mut job as *mut (Option<F>, Option<std::thread::Result<()>>)).cast(),
            execute: execute::<F>,
        });
        state.busy = true;
        worker.shared.wake.notify_one();
        while state.busy {
            state = worker.shared.done.wait(state).unwrap();
        }
        drop(state);
        // Release serialization before propagating a user panic, avoiding a
        // poisoned caller mutex. The worker and its storage remain reusable.
        drop(_caller);
        if let Err(payload) = job.1.unwrap() {
            std::panic::resume_unwind(payload);
        }
    }
}

struct Job {
    data: *mut (),
    execute: unsafe fn(*mut ()),
}
// SAFETY: only Send closures enter this slot; the submitting caller waits for
// completion before accessing or dropping their borrowed stack data.
unsafe impl Send for Job {}

#[derive(Default)]
struct State {
    job: Option<Job>,
    busy: bool,
    shutdown: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

struct Worker {
    shared: Arc<Shared>,
    caller: Mutex<()>,
    _pool: rayon::ThreadPool,
}

impl Worker {
    fn new() -> Self {
        let shared = Arc::new(Shared::default());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .build()
            .expect("affine scan worker pool");
        let worker = shared.clone();
        // This is the only heap job submitted to this pool. Its lifetime covers
        // every scan step; the mutex/condvars and worker scratch never resize.
        pool.spawn(move || {
            let mut state = worker.state.lock().unwrap();
            loop {
                while state.job.is_none() && !state.shutdown {
                    state = worker.wake.wait(state).unwrap();
                }
                if state.shutdown {
                    break;
                }
                let job = state.job.take().unwrap();
                drop(state);
                // SAFETY: the caller retains the borrowed job until busy is
                // cleared below, including when its closure panics.
                unsafe { (job.execute)(job.data) };
                state = worker.state.lock().unwrap();
                state.busy = false;
                worker.done.notify_one();
            }
        });
        Self {
            shared,
            caller: Mutex::new(()),
            _pool: pool,
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // All run calls retain an executor handle, so no borrowed job remains.
        let mut state = self.shared.state.lock().unwrap();
        state.shutdown = true;
        self.shared.wake.notify_one();
    }
}
