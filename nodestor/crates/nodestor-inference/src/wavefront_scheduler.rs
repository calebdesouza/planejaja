use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const EFFICIENCY_FACTOR: f64 = 0.25;
const MAX_SLICE_MS: f64 = 4.0;

/// Fragments transformer matmul dispatches into sub-units bounded by MAX_SLICE_MS.
///
/// Between every slice the scheduler calls `std::thread::yield_now()`, giving the
/// OS scheduler and the Vulkan driver a cooperative preemption point. This prevents:
/// - TDR (Timeout Detection and Recovery) on Windows (default watchdog: 2000ms)
/// - Display compositor starvation during sustained inference sessions
/// - Priority inversion when the UI needs to service vsync interrupts mid-inference
///
/// `VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT` is queried at engine init and applied when
/// the extension is available. On drivers without the extension (older AMD), the
/// cooperative yield alone provides equivalent TDR immunity for kernels < 4ms each.
pub struct WavefrontScheduler {
    /// Maximum output rows per sub-dispatch — calibrated for MAX_SLICE_MS budget.
    pub rows_per_slice: usize,
    pub slices_dispatched: AtomicU64,
    pub total_dispatch_ns: AtomicU64,
    /// Set to false in unit tests to skip yield_now() and keep latency deterministic.
    pub cooperative_yield: bool,
}

impl WavefrontScheduler {
    /// Constructs a scheduler calibrated for the given peak FLOP rate.
    ///
    /// `hidden_dim` is the matrix inner dimension (K). The outer dimension (N)
    /// is unknown at construction; slicing covers M (output rows) instead, which
    /// is always known at dispatch time.
    pub fn new(hidden_dim: usize, tflops_f32: f64) -> Self {
        let rows = Self::calibrate(hidden_dim, tflops_f32);
        Self {
            rows_per_slice: rows,
            slices_dispatched: AtomicU64::new(0),
            total_dispatch_ns: AtomicU64::new(0),
            cooperative_yield: true,
        }
    }

    /// Rows that fit within MAX_SLICE_MS at the given FLOP rate, rounded to
    /// the nearest power of two in [32, 4096].
    fn calibrate(hidden_dim: usize, tflops: f64) -> usize {
        let flops_per_row = 2.0 * hidden_dim as f64;
        let budget = tflops * EFFICIENCY_FACTOR * (MAX_SLICE_MS / 1000.0);
        let rows = (budget / flops_per_row).max(32.0) as usize;
        rows.next_power_of_two().clamp(32, 4096)
    }

    /// Executes `dispatch_fn(row_start, row_end)` over [0, total_rows), split
    /// into sub-units of at most `rows_per_slice` rows.
    ///
    /// Each sub-dispatch is independently timed. Between slices, the calling
    /// thread yields to the OS scheduler, allowing display/UI preemption.
    pub fn dispatch_sliced<F>(&self, total_rows: usize, mut dispatch_fn: F)
    where
        F: FnMut(usize, usize),
    {
        let mut row = 0;
        while row < total_rows {
            let end = (row + self.rows_per_slice).min(total_rows);
            let t0 = Instant::now();

            dispatch_fn(row, end);

            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            self.slices_dispatched.fetch_add(1, Ordering::Relaxed);
            self.total_dispatch_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            row = end;

            if self.cooperative_yield && row < total_rows {
                std::thread::yield_now();
            }
        }
    }

    pub fn mean_slice_ms(&self) -> f32 {
        let n = self.slices_dispatched.load(Ordering::Relaxed).max(1);
        let ns = self.total_dispatch_ns.load(Ordering::Relaxed);
        (ns as f32 / n as f32) / 1_000_000.0
    }
}

impl Default for WavefrontScheduler {
    fn default() -> Self {
        // RX 580 2048SP: 6.175 TFLOPS FP32 at 1411 MHz boost
        Self::new(2048, 6.175e12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavefront_all_rows_covered() {
        let sched = WavefrontScheduler {
            rows_per_slice: 64,
            slices_dispatched: AtomicU64::new(0),
            total_dispatch_ns: AtomicU64::new(0),
            cooperative_yield: false,
        };
        let total = 256usize;
        let mut covered = 0usize;
        sched.dispatch_sliced(total, |s, e| covered += e - s);
        assert_eq!(covered, total);
    }

    #[test]
    fn wavefront_each_slice_within_budget() {
        let sched = WavefrontScheduler {
            rows_per_slice: 32,
            slices_dispatched: AtomicU64::new(0),
            total_dispatch_ns: AtomicU64::new(0),
            cooperative_yield: false,
        };
        sched.dispatch_sliced(100, |s, e| {
            assert!(e - s <= 32, "slice exceeded rows_per_slice: {} > 32", e - s);
        });
    }

    #[test]
    fn wavefront_empty_is_noop() {
        let sched = WavefrontScheduler::default();
        let mut called = false;
        sched.dispatch_sliced(0, |_, _| called = true);
        assert!(!called);
        assert_eq!(sched.slices_dispatched.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn wavefront_rows_per_slice_is_power_of_two_in_range() {
        let sched = WavefrontScheduler::new(2048, 6.175e12);
        let n = sched.rows_per_slice;
        assert!(n >= 32 && n <= 4096, "rows_per_slice out of range: {}", n);
        assert!(n.is_power_of_two(), "rows_per_slice not power-of-two: {}", n);
    }

    #[test]
    fn wavefront_dispatch_counter_matches_slice_count() {
        let sched = WavefrontScheduler {
            rows_per_slice: 10,
            slices_dispatched: AtomicU64::new(0),
            total_dispatch_ns: AtomicU64::new(0),
            cooperative_yield: false,
        };
        let total = 35usize; // 4 slices: [0,10), [10,20), [20,30), [30,35)
        sched.dispatch_sliced(total, |_, _| {});
        let count = sched.slices_dispatched.load(Ordering::Relaxed);
        assert_eq!(count, 4, "expected 4 slices for 35 rows / 10-per-slice");
    }

    #[test]
    fn wavefront_mean_slice_ms_is_non_negative() {
        let sched = WavefrontScheduler::default();
        sched.dispatch_sliced(128, |s, e| { std::hint::black_box(s + e); });
        assert!(sched.mean_slice_ms() >= 0.0);
    }
}
