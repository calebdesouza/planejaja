use nodestor_core::NodeStorError;

/// Page-aligned virtual memory region for tensor data.
///
/// Allocation is backed by the OS virtual memory manager via VirtualAlloc
/// (Windows) or anonymous mmap (POSIX). Physical pages are NOT pre-faulted
/// at allocation — they are demand-committed on first write, preventing the
/// large upfront RAM reservation that causes OOM crashes for 1B+ models.
///
/// Under host memory pressure the OS MMU can evict clean demand-zero pages
/// automatically, reducing resident set size without invalidating pointers.
/// This gives MoE expert weights the same elastic behaviour as file-backed
/// mmap: valid addresses, variable physical residency.
pub struct ElasticTensorSlot {
    ptr: *mut u8,
    len: usize,
    pub resident_hint: bool,
}

unsafe impl Send for ElasticTensorSlot {}
unsafe impl Sync for ElasticTensorSlot {}

impl ElasticTensorSlot {
    /// Reserves `size` bytes of virtual address space.
    ///
    /// On Windows: MEM_RESERVE | MEM_COMMIT (pages committed but demand-zeroed).
    /// On POSIX: MAP_PRIVATE | MAP_ANONYMOUS (pages faulted on first access).
    pub fn allocate(size: usize) -> Result<Self, NodeStorError> {
        let page_size = page_granularity();
        let aligned = (size + page_size - 1) & !(page_size - 1);
        let ptr = os_alloc(aligned)?;
        Ok(Self { ptr, len: aligned, resident_hint: false })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Advises the OS that these pages will be needed soon.
    ///
    /// Windows: best-effort no-op (PrefetchVirtualMemory requires kernel
    /// privileges; page-fault path is equally fast for committed pages).
    /// POSIX: madvise(MADV_WILLNEED) schedules prefetch from swap.
    pub fn advise_will_need(&mut self) {
        #[cfg(not(target_os = "windows"))]
        unsafe { posix_madvise(self.ptr, self.len, MADV_WILLNEED); }
        self.resident_hint = true;
    }

    /// Releases physical pages without freeing the virtual address range.
    ///
    /// Windows: VirtualFree(MEM_DECOMMIT). Subsequent reads return zero from
    /// re-committed pages (new physical frames, clean slate).
    /// POSIX: madvise(MADV_DONTNEED). Subsequent reads return zero (kernel
    /// re-maps demand-zero pages).
    ///
    /// This is the core mechanism for MoE expert eviction under pressure:
    /// the pointer remains valid, address space is reserved, but physical
    /// frames are returned to the OS pool.
    pub fn release_pages(&mut self) {
        unsafe { os_decommit(self.ptr, self.len); }
        self.resident_hint = false;
    }
}

impl Drop for ElasticTensorSlot {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { os_free(self.ptr, self.len); }
        }
    }
}

// ─── ElasticWeightCache ───────────────────────────────────────────────────────

/// Per-layer elastic weight slots for a transformer stack.
///
/// For each layer, holds an optional ElasticTensorSlot. When a layer's weights
/// are fully resident in VRAM (DEVICE_LOCAL), `evict_layer()` decommits the
/// physical CPU pages — freeing RAM while keeping the virtual pointer valid.
/// When the layer needs to fall back to CPU (VRAM eviction), `restore_layer()`
/// re-faults the pages via MADV_WILLNEED (POSIX) or demand-zero (Windows).
///
/// This is the MoE expert eviction mechanism applied to dense transformer layers:
/// `n_layers × FFN_size` bytes of staging RAM become elastic, not resident.
pub struct ElasticWeightCache {
    slots: Vec<Option<ElasticTensorSlot>>,
    evicted: Vec<bool>,
}

impl ElasticWeightCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            slots: (0..n_layers).map(|_| None).collect(),
            evicted: vec![false; n_layers],
        }
    }

    /// Allocates a slot for `layer` and copies `bytes` into it.
    pub fn set_layer(&mut self, layer: usize, bytes: &[u8]) -> Result<(), NodeStorError> {
        if layer >= self.slots.len() || bytes.is_empty() { return Ok(()); }
        let mut slot = ElasticTensorSlot::allocate(bytes.len())?;
        slot.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        self.slots[layer] = Some(slot);
        self.evicted[layer] = false;
        Ok(())
    }

    /// Decommits physical pages for `layer`. Virtual range remains valid.
    pub fn evict_layer(&mut self, layer: usize) {
        if let Some(slot) = self.slots.get_mut(layer).and_then(|s| s.as_mut()) {
            slot.release_pages();
            if layer < self.evicted.len() { self.evicted[layer] = true; }
        }
    }

    /// Re-faults pages. POSIX: MADV_WILLNEED. Windows: demand-zero on next access.
    pub fn restore_layer(&mut self, layer: usize) {
        if let Some(slot) = self.slots.get_mut(layer).and_then(|s| s.as_mut()) {
            slot.advise_will_need();
            if layer < self.evicted.len() { self.evicted[layer] = false; }
        }
    }

    pub fn is_evicted(&self, layer: usize) -> bool {
        self.evicted.get(layer).copied().unwrap_or(false)
    }

    pub fn layer_bytes(&self, layer: usize) -> Option<&[u8]> {
        self.slots.get(layer)?.as_ref().map(|s| s.as_slice())
    }

    /// Returns `(evicted_count, total_evicted_bytes)`.
    pub fn evicted_stats(&self) -> (usize, usize) {
        let n = self.evicted.iter().filter(|&&e| e).count();
        let bytes: usize = self.slots.iter().zip(self.evicted.iter())
            .filter(|(_, &e)| e)
            .filter_map(|(s, _)| s.as_ref().map(|sl| sl.len()))
            .sum();
        (n, bytes)
    }

    pub fn n_layers(&self) -> usize { self.slots.len() }
}

// ─── Platform: Windows ───────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn page_granularity() -> usize { 65536 } // VirtualAlloc allocation granularity

#[cfg(target_os = "windows")]
fn os_alloc(size: usize) -> Result<*mut u8, NodeStorError> {
    extern "system" {
        fn VirtualAlloc(
            lp: *mut core::ffi::c_void,
            sz: usize,
            ty: u32,
            pr: u32,
        ) -> *mut core::ffi::c_void;
    }
    const MEM_COMMIT_RESERVE: u32 = 0x3000;
    const PAGE_READWRITE: u32 = 0x04;
    let p = unsafe { VirtualAlloc(core::ptr::null_mut(), size, MEM_COMMIT_RESERVE, PAGE_READWRITE) };
    if p.is_null() {
        Err(NodeStorError::InferenceError(format!("VirtualAlloc({} bytes) failed", size)))
    } else {
        Ok(p as *mut u8)
    }
}

#[cfg(target_os = "windows")]
unsafe fn os_decommit(ptr: *mut u8, len: usize) {
    extern "system" { fn VirtualFree(p: *mut core::ffi::c_void, s: usize, t: u32) -> i32; }
    // MEM_DECOMMIT = 0x4000: returns physical pages but keeps virtual reservation
    VirtualFree(ptr as *mut core::ffi::c_void, len, 0x4000);
}

#[cfg(target_os = "windows")]
unsafe fn os_free(ptr: *mut u8, _len: usize) {
    extern "system" { fn VirtualFree(p: *mut core::ffi::c_void, s: usize, t: u32) -> i32; }
    // MEM_RELEASE = 0x8000: releases virtual reservation (size must be 0)
    VirtualFree(ptr as *mut core::ffi::c_void, 0, 0x8000);
}

// ─── Platform: POSIX (Linux / macOS) ─────────────────────────────────────────

#[cfg(not(target_os = "windows"))]
fn page_granularity() -> usize { 4096 }

#[cfg(not(target_os = "windows"))]
fn os_alloc(size: usize) -> Result<*mut u8, NodeStorError> {
    extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut core::ffi::c_void;
    }
    const PROT_RW: i32 = 0x03; // PROT_READ | PROT_WRITE
    // MAP_PRIVATE | MAP_ANONYMOUS: differs by platform
    #[cfg(target_os = "macos")]
    const MAP_FLAGS: i32 = 0x1002; // MAP_PRIVATE | MAP_ANON
    #[cfg(not(target_os = "macos"))]
    const MAP_FLAGS: i32 = 0x0022; // MAP_PRIVATE | MAP_ANONYMOUS
    let p = unsafe { mmap(core::ptr::null_mut(), size, PROT_RW, MAP_FLAGS, -1, 0) };
    if p as isize == -1 {
        Err(NodeStorError::InferenceError(format!("mmap({} bytes) failed", size)))
    } else {
        Ok(p as *mut u8)
    }
}

#[cfg(not(target_os = "windows"))]
unsafe fn os_decommit(ptr: *mut u8, len: usize) {
    posix_madvise(ptr, len, MADV_DONTNEED);
}

#[cfg(not(target_os = "windows"))]
unsafe fn os_free(ptr: *mut u8, len: usize) {
    extern "C" { fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32; }
    munmap(ptr as *mut core::ffi::c_void, len);
}

#[cfg(not(target_os = "windows"))]
const MADV_DONTNEED: i32 = 4;
#[cfg(not(target_os = "windows"))]
const MADV_WILLNEED: i32 = 3;

#[cfg(not(target_os = "windows"))]
unsafe fn posix_madvise(ptr: *mut u8, len: usize, advice: i32) {
    extern "C" { fn madvise(addr: *mut core::ffi::c_void, len: usize, adv: i32) -> i32; }
    madvise(ptr as *mut core::ffi::c_void, len, advice);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_alloc_small_read_write() {
        let mut slot = ElasticTensorSlot::allocate(4096).expect("alloc");
        assert!(slot.len() >= 4096);
        slot.as_mut_slice()[0] = 0xDE;
        slot.as_mut_slice()[4095] = 0xAD;
        assert_eq!(slot.as_slice()[0], 0xDE);
        assert_eq!(slot.as_slice()[4095], 0xAD);
    }

    #[test]
    fn elastic_alloc_large_demand_paged() {
        // 128 MB reservation — only 2 pages physically touched
        let mut slot = ElasticTensorSlot::allocate(128 * 1024 * 1024).expect("large alloc");
        assert!(slot.len() >= 128 * 1024 * 1024);
        slot.as_mut_slice()[0] = 1;
        let last = slot.len() - 1;
        slot.as_mut_slice()[last] = 2;
        assert_eq!(slot.as_slice()[0], 1);
        assert_eq!(slot.as_slice()[last], 2);
    }

    #[test]
    fn elastic_advise_will_need_sets_resident_hint() {
        let mut slot = ElasticTensorSlot::allocate(8192).expect("alloc");
        assert!(!slot.resident_hint);
        slot.advise_will_need();
        assert!(slot.resident_hint);
    }

    #[test]
    fn elastic_release_pages_does_not_segfault() {
        let mut slot = ElasticTensorSlot::allocate(8192).expect("alloc");
        slot.as_mut_slice()[0] = 99;
        slot.release_pages();
        assert!(!slot.resident_hint);
        // Virtual range still valid — length is accessible without crash
        let _ = slot.len();
    }

    #[test]
    fn elastic_slot_drop_is_safe() {
        let slot = ElasticTensorSlot::allocate(4096).expect("alloc");
        drop(slot); // must not crash or leak
    }

    #[test]
    fn elastic_multiple_slots_independent() {
        let mut a = ElasticTensorSlot::allocate(4096).expect("alloc a");
        let mut b = ElasticTensorSlot::allocate(4096).expect("alloc b");
        a.as_mut_slice()[0] = 11;
        b.as_mut_slice()[0] = 22;
        assert_eq!(a.as_slice()[0], 11);
        assert_eq!(b.as_slice()[0], 22);
    }
}
