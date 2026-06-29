// Timeline Semaphore — CPU/GPU Overlap Synchronization
//
// Mirrors the Vulkan 1.2 VkSemaphoreTypeTimeline API using Rust atomics + Notify.
// This allows the HeteroScheduler to overlap CPU attention (layer N) with GPU
// FFN execution (layer N-1) without busy-spinning on the OS thread.
//
// API contract (same as Vulkan timeline semaphores):
//   - The semaphore holds a monotonically increasing u64 counter.
//   - signal(v): advance the counter to max(current, v), wake all waiters.
//   - wait_for(v): block until counter >= v.
//   - Signals are permanent: once signalled to v, waiting on any w ≤ v resolves immediately.
//
// Heterogeneous pipeline usage:
//
//   CPU thread (attention layer N):
//     attn_done.wait_for(N - 1)   // wait for previous GPU FFN to finish
//     compute_attention(N)
//     attn_done.signal(N)         // signal GPU: attention N is ready
//
//   GPU thread (FFN layer N-1):
//     attn_done.wait_for(N - 1)   // wait for CPU attention N-1
//     dispatch_ffn(N - 1)
//     ffn_done.signal(N - 1)
//
// This gives perfect CPU/GPU pipeline overlap when compute times are similar.
// Maximum throughput = max(t_attention_per_layer, t_ffn_per_layer) per layer.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

// ─── Core semaphore ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Inner {
    value:    AtomicU64,
    waiters:  Mutex<Vec<Arc<std::sync::Condvar>>>,
}

/// A monotonically-increasing timeline semaphore.
/// Clone to share across threads — all clones reference the same counter.
#[derive(Clone, Debug)]
pub struct TimelineSemaphore {
    inner: Arc<Inner>,
}

impl TimelineSemaphore {
    /// Create a new semaphore with initial value `initial`.
    pub fn new(initial: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                value:   AtomicU64::new(initial),
                waiters: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Read the current timeline value without waiting.
    #[inline]
    pub fn current(&self) -> u64 {
        self.inner.value.load(Ordering::Acquire)
    }

    /// Signal the semaphore to at least `value`.
    /// If `value` ≤ current, this is a no-op.
    /// Wakes all threads waiting on any value ≤ `value`.
    pub fn signal(&self, value: u64) {
        let prev = self.inner.value.fetch_max(value, Ordering::AcqRel);
        if value > prev {
            // Wake all waiters — they each re-check their threshold.
            let waiters = self.inner.waiters.lock().unwrap();
            for cv in waiters.iter() {
                cv.notify_all();
            }
        }
    }

    /// Block the current thread until `current() >= value`.
    /// Returns immediately if already satisfied.
    pub fn wait_for(&self, value: u64) {
        if self.current() >= value { return; }

        let cv = Arc::new(std::sync::Condvar::new());
        {
            let mut waiters = self.inner.waiters.lock().unwrap();
            waiters.push(cv.clone());
        }

        let mutex = std::sync::Mutex::new(());
        let guard = mutex.lock().unwrap();
        let _guard = cv.wait_while(guard, |_| {
            self.inner.value.load(Ordering::Acquire) < value
        }).unwrap();

        // Remove our condvar from the waiter list.
        let mut waiters = self.inner.waiters.lock().unwrap();
        waiters.retain(|w| !Arc::ptr_eq(w, &cv));
    }

    /// Block until `current() >= value` with a timeout.
    /// Returns `true` if condition was met, `false` if timed out.
    pub fn wait_for_timeout(&self, value: u64, timeout: Duration) -> bool {
        if self.current() >= value { return true; }

        let cv = Arc::new(std::sync::Condvar::new());
        {
            let mut waiters = self.inner.waiters.lock().unwrap();
            waiters.push(cv.clone());
        }

        let mutex = std::sync::Mutex::new(());
        let guard = mutex.lock().unwrap();
        let (_guard2, timed_out) = cv.wait_timeout_while(
            guard,
            timeout,
            |_| self.inner.value.load(Ordering::Acquire) < value,
        ).unwrap();

        let mut waiters = self.inner.waiters.lock().unwrap();
        waiters.retain(|w| !Arc::ptr_eq(w, &cv));

        !timed_out.timed_out()
    }
}

// ─── Layer pipeline fence ──────────────────────────────────────────────────────

/// Pair of timeline semaphores that implement the CPU↔GPU layer pipeline.
///
/// `attn`: CPU signals after attention layer N completes (GPU waits on this).
/// `ffn`:  GPU signals after FFN layer N-1 completes (CPU waits on this).
///
/// Use `LayerFence::for_n_layers(n)` to get a pair pre-seeded at layer 0.
pub struct LayerFence {
    /// CPU signals here after completing attention for layer N.
    pub attn: TimelineSemaphore,
    /// GPU/CPU signals here after completing FFN for layer N.
    pub ffn:  TimelineSemaphore,
}

impl LayerFence {
    /// Create a fence pair for a model with `n_layers` transformer layers.
    /// Both semaphores start at 0 (nothing completed).
    pub fn new() -> Self {
        Self {
            attn: TimelineSemaphore::new(0),
            ffn:  TimelineSemaphore::new(0),
        }
    }

    /// CPU calls this before starting attention for `layer`.
    /// Waits for GPU to complete FFN for layer-1 (so KV cache is stable).
    #[inline]
    pub fn cpu_wait_before_attn(&self, layer: u64) {
        if layer > 0 { self.ffn.wait_for(layer); }
    }

    /// CPU calls this after finishing attention for `layer`.
    #[inline]
    pub fn cpu_signal_attn_done(&self, layer: u64) {
        self.attn.signal(layer + 1);
    }

    /// GPU/FFN thread calls this before starting FFN for `layer`.
    /// Waits for CPU to complete attention for the same layer.
    #[inline]
    pub fn gpu_wait_before_ffn(&self, layer: u64) {
        self.attn.wait_for(layer + 1);
    }

    /// GPU/FFN thread calls this after finishing FFN for `layer`.
    #[inline]
    pub fn gpu_signal_ffn_done(&self, layer: u64) {
        self.ffn.signal(layer + 1);
    }
}

impl Default for LayerFence {
    fn default() -> Self { Self::new() }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn semaphore_initial_value() {
        let s = TimelineSemaphore::new(5);
        assert_eq!(s.current(), 5);
    }

    #[test]
    fn semaphore_signal_advances_counter() {
        let s = TimelineSemaphore::new(0);
        s.signal(3);
        assert_eq!(s.current(), 3);
        s.signal(1); // lower value → no-op
        assert_eq!(s.current(), 3);
        s.signal(10);
        assert_eq!(s.current(), 10);
    }

    #[test]
    fn semaphore_wait_for_already_satisfied() {
        let s = TimelineSemaphore::new(100);
        s.wait_for(50); // already satisfied — must return immediately
        assert_eq!(s.current(), 100);
    }

    #[test]
    fn semaphore_wait_for_unblocks_on_signal() {
        let s = Arc::new(TimelineSemaphore::new(0));
        let s2 = s.clone();

        let waiter = thread::spawn(move || {
            s2.wait_for(1);
        });

        thread::sleep(Duration::from_millis(10));
        s.signal(1);
        waiter.join().expect("waiter thread panicked");
        assert_eq!(s.current(), 1);
    }

    #[test]
    fn semaphore_multiple_waiters_all_wake() {
        let s = Arc::new(TimelineSemaphore::new(0));
        let handles: Vec<_> = (0..4).map(|_| {
            let sc = s.clone();
            thread::spawn(move || { sc.wait_for(1); })
        }).collect();

        thread::sleep(Duration::from_millis(10));
        s.signal(1);

        for h in handles { h.join().expect("waiter panicked"); }
    }

    #[test]
    fn semaphore_wait_for_timeout_returns_false_on_timeout() {
        let s = TimelineSemaphore::new(0);
        let satisfied = s.wait_for_timeout(99, Duration::from_millis(20));
        assert!(!satisfied, "timeout: semaphore never signalled");
    }

    #[test]
    fn semaphore_wait_for_timeout_returns_true_when_signalled() {
        let s = Arc::new(TimelineSemaphore::new(0));
        let s2 = s.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            s2.signal(1);
        });
        let satisfied = s.wait_for_timeout(1, Duration::from_millis(500));
        assert!(satisfied, "semaphore was signalled within timeout");
    }

    #[test]
    fn layer_fence_cpu_gpu_pipeline_ordering() {
        let fence = Arc::new(LayerFence::new());
        let fc = fence.clone();
        let n_layers = 4u64;

        // GPU thread: processes FFN after CPU completes attention
        let gpu = thread::spawn(move || {
            let mut ffn_log = Vec::new();
            for layer in 0..n_layers {
                fc.gpu_wait_before_ffn(layer);
                ffn_log.push(layer);
                fc.gpu_signal_ffn_done(layer);
            }
            ffn_log
        });

        // CPU thread: processes attention, then waits for GPU FFN before next layer
        let mut attn_log = Vec::new();
        for layer in 0..n_layers {
            fence.cpu_wait_before_attn(layer);
            attn_log.push(layer);
            fence.cpu_signal_attn_done(layer);
        }

        let ffn_log = gpu.join().expect("GPU thread panicked");

        // Both sides must have processed all layers in order
        assert_eq!(attn_log, (0..n_layers).collect::<Vec<_>>());
        assert_eq!(ffn_log,  (0..n_layers).collect::<Vec<_>>());
    }

    #[test]
    fn semaphore_clone_shares_state() {
        let s1 = TimelineSemaphore::new(0);
        let s2 = s1.clone();
        s1.signal(42);
        assert_eq!(s2.current(), 42, "clone must share the same counter");
    }
}
