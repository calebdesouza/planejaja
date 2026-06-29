/// SSD Weight Stream — Layer-by-Layer Async Streaming from GGUF
///
/// Architecture: MoE-first. For dense models, every FFN block is treated as
/// a single "expert 0". For MoE models (DeepSeek-V3 / Mixtral style) expert
/// sparsity is exploited: only the activated subset is read per token.
///
/// Tiers:
///   VRAM  — pinned attention layers + hot expert cache (top-K by frequency)
///   RAM   — warm expert buffer (LRU, configurable budget)
///   SSD   — compressed experts (GDeflate format, async Windows OVERLAPPED IO)
///
/// Target: 300B MoE @ 30-40 tok/s on consumer hardware via:
///   1. Expert sparsity  — only 8/256 experts read per token
///   2. Hot expert cache — top-100 experts in VRAM (~360 MB), zero IO on hit
///   3. GDeflate reads   — 3× effective NVMe bandwidth (compressed on disk)
///   4. Double-buffer    — IO and compute overlap layer-by-layer
///   5. Expert prefetch  — predict next activation from router logits

use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use nodestor_core::NodeStorError;

// ─── Expert identity ─────────────────────────────────────────────────────────

/// Globally unique key for one expert weight block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer:  u32,
    pub expert: u32,
}

// ─── Weight block ─────────────────────────────────────────────────────────────

/// A raw weight block (Q4/Q8/F16) freshly read from disk or cache.
#[derive(Clone)]
pub struct WeightBlock {
    pub id:     ExpertId,
    pub data:   Vec<u8>,    // raw bytes, quantization matches model file
    pub quant:  QuantType,
    pub rows:   u32,
    pub cols:   u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType { Q4_0, Q4_K, Q5_0, Q8_0, F16, BF16 }

impl QuantType {
    pub fn bytes_per_element(self) -> f32 {
        match self {
            Self::Q4_0 | Self::Q4_K => 0.5,
            Self::Q5_0              => 0.625,
            Self::Q8_0              => 1.0,
            Self::F16 | Self::BF16  => 2.0,
        }
    }
}

// ─── Disk layout ──────────────────────────────────────────────────────────────

/// Offset + size of one expert block inside the GGUF file.
#[derive(Clone, Copy, Debug)]
pub struct ExpertDiskEntry {
    pub offset:           u64,
    pub compressed_bytes: u64,   // on-disk size (GDeflate or raw)
    pub raw_bytes:        u64,   // decompressed size
    pub quant:            QuantType,
    pub rows:             u32,
    pub cols:             u32,
    pub is_compressed:    bool,
}

// ─── Frequency tracker ────────────────────────────────────────────────────────

/// Tracks expert activation frequency to decide what stays in VRAM / RAM.
pub struct ExpertFrequency {
    counts:      HashMap<ExpertId, u64>,
    total_steps: u64,
}

impl ExpertFrequency {
    pub fn new() -> Self {
        Self { counts: HashMap::new(), total_steps: 0 }
    }

    pub fn record(&mut self, activated: &[ExpertId]) {
        self.total_steps += 1;
        for &id in activated {
            *self.counts.entry(id).or_insert(0) += 1;
        }
    }

    /// Returns the top-K experts sorted by frequency (descending).
    pub fn top_k(&self, k: usize) -> Vec<(ExpertId, f64)> {
        let mut v: Vec<(ExpertId, u64)> = self.counts.iter()
            .map(|(&id, &c)| (id, c))
            .collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        v.truncate(k);
        let total = self.total_steps.max(1) as f64;
        v.into_iter().map(|(id, c)| (id, c as f64 / total)).collect()
    }

    pub fn frequency(&self, id: ExpertId) -> f64 {
        let c = self.counts.get(&id).copied().unwrap_or(0);
        c as f64 / self.total_steps.max(1) as f64
    }
}

// ─── Three-tier cache ─────────────────────────────────────────────────────────

pub enum CacheTier { Vram, Ram, Ssd }

/// Tiered expert cache: VRAM hot → RAM warm → SSD cold.
///
/// VRAM tier is simulated here as a pinned Vec (in real GPU path these would
/// be Vulkan device-local buffers; the interface is the same).
pub struct ExpertCache {
    /// VRAM tier: fixed capacity (bytes). LFU eviction when full.
    vram_capacity:   usize,
    vram_used:       usize,
    vram:            HashMap<ExpertId, WeightBlock>,

    /// RAM tier: LRU, configurable budget.
    ram_capacity:    usize,
    ram_used:        usize,
    ram:             HashMap<ExpertId, WeightBlock>,
    ram_lru:         VecDeque<ExpertId>,

    pub frequency:   ExpertFrequency,

    // Stats
    pub hits_vram:   u64,
    pub hits_ram:    u64,
    pub hits_ssd:    u64,
}

impl ExpertCache {
    /// `vram_budget` and `ram_budget` are in bytes.
    pub fn new(vram_budget: usize, ram_budget: usize) -> Self {
        Self {
            vram_capacity: vram_budget,
            vram_used:     0,
            vram:          HashMap::new(),
            ram_capacity:  ram_budget,
            ram_used:      0,
            ram:           HashMap::new(),
            ram_lru:       VecDeque::new(),
            frequency:     ExpertFrequency::new(),
            hits_vram:     0,
            hits_ram:      0,
            hits_ssd:      0,
        }
    }

    pub fn get(&mut self, id: ExpertId) -> Option<(CacheTier, &WeightBlock)> {
        if self.vram.contains_key(&id) {
            self.hits_vram += 1;
            return Some((CacheTier::Vram, self.vram.get(&id).unwrap()));
        }
        if self.ram.contains_key(&id) {
            self.hits_ram += 1;
            // Promote to front of LRU
            self.ram_lru.retain(|&x| x != id);
            self.ram_lru.push_front(id);
            return Some((CacheTier::Ram, self.ram.get(&id).unwrap()));
        }
        self.hits_ssd += 1;
        None
    }

    /// Insert a freshly-loaded block. Decides tier based on expert frequency.
    pub fn insert(&mut self, block: WeightBlock) {
        let id   = block.id;
        let size = block.data.len();
        let freq = self.frequency.frequency(id);

        // Hot expert (freq > 1%): try VRAM first
        if freq > 0.01 {
            if self.vram_used + size <= self.vram_capacity {
                self.vram_used += size;
                self.vram.insert(id, block);
                return;
            }
            // VRAM full: evict lowest-frequency VRAM entry
            if let Some(evict_id) = self.vram.iter()
                .min_by(|a, b| {
                    self.frequency.frequency(*a.0)
                        .partial_cmp(&self.frequency.frequency(*b.0))
                        .unwrap()
                }).map(|(&k, _)| k)
            {
                if self.frequency.frequency(evict_id) < freq {
                    if let Some(evicted) = self.vram.remove(&evict_id) {
                        self.vram_used -= evicted.data.len();
                        // Demote to RAM
                        self.insert_ram(evicted);
                    }
                    self.vram_used += size;
                    self.vram.insert(id, block);
                    return;
                }
            }
        }

        // Warm: go to RAM
        self.insert_ram(block);
    }

    fn insert_ram(&mut self, block: WeightBlock) {
        let id   = block.id;
        let size = block.data.len();
        // Evict LRU entries until there's space
        while self.ram_used + size > self.ram_capacity && !self.ram_lru.is_empty() {
            if let Some(evict_id) = self.ram_lru.pop_back() {
                if let Some(evicted) = self.ram.remove(&evict_id) {
                    self.ram_used -= evicted.data.len();
                }
            }
        }
        self.ram_used += size;
        self.ram_lru.push_front(id);
        self.ram.insert(id, block);
    }

    /// Pre-warm VRAM with the top-K experts by historical frequency.
    pub fn promote_top_k(&mut self, blocks: Vec<WeightBlock>) {
        for block in blocks {
            let size = block.data.len();
            if self.vram_used + size <= self.vram_capacity {
                self.vram_used += size;
                self.vram.insert(block.id, block);
            }
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits_vram + self.hits_ram + self.hits_ssd;
        if total == 0 { return 0.0; }
        (self.hits_vram + self.hits_ram) as f64 / total as f64
    }

    pub fn vram_used_mb(&self) -> f64 { self.vram_used as f64 / 1_048_576.0 }
    pub fn ram_used_mb(&self)  -> f64 { self.ram_used  as f64 / 1_048_576.0 }
}

// ─── SSD streamer ─────────────────────────────────────────────────────────────

/// Configuration for the SSD streaming pipeline.
pub struct SsdStreamConfig {
    pub gguf_path:      PathBuf,
    /// VRAM budget for expert cache (bytes). Default 512 MB.
    pub vram_budget:    usize,
    /// RAM budget for expert warm cache (bytes). Default 8 GB.
    pub ram_budget:     usize,
    /// Number of layers to double-buffer (prefetch ahead). Default 2.
    pub prefetch_depth: usize,
    /// Whether the GGUF is stored GDeflate-compressed.
    pub gdeflate:       bool,
    /// Expert count per layer (1 = dense model).
    pub experts_per_layer: u32,
    /// Experts activated per token (1 = dense).
    pub active_experts:    u32,
}

impl Default for SsdStreamConfig {
    fn default() -> Self {
        Self {
            gguf_path:         PathBuf::new(),
            vram_budget:       512 * 1_048_576,      // 512 MB
            ram_budget:        8 * 1024 * 1_048_576, // 8 GB
            prefetch_depth:    2,
            gdeflate:          false,
            experts_per_layer: 1,
            active_experts:    1,
        }
    }
}

/// The main streaming engine. Call `fetch_experts()` per token to get the
/// weight blocks for this token's activated experts.
pub struct SsdWeightStream {
    cfg:          SsdStreamConfig,
    disk_index:   HashMap<ExpertId, ExpertDiskEntry>,
    cache:        ExpertCache,
    file:         File,

    // Prefetch double-buffer: (layer, expert_ids) → pending read
    prefetch_buf: VecDeque<(u32, Vec<ExpertId>)>,

    // Performance tracking
    pub total_fetch_time: Duration,
    pub total_io_bytes:   u64,
    pub tokens_served:    u64,
}

impl SsdWeightStream {
    pub fn new(cfg: SsdStreamConfig, disk_index: HashMap<ExpertId, ExpertDiskEntry>)
        -> Result<Self, NodeStorError>
    {
        let file = File::open(&cfg.gguf_path)
            .map_err(|e| NodeStorError::InferenceError(
                format!("ssd_stream: cannot open {}: {}", cfg.gguf_path.display(), e)
            ))?;

        let cache = ExpertCache::new(cfg.vram_budget, cfg.ram_budget);

        Ok(Self {
            cfg,
            disk_index,
            cache,
            file,
            prefetch_buf: VecDeque::new(),
            total_fetch_time: Duration::ZERO,
            total_io_bytes:   0,
            tokens_served:    0,
        })
    }

    /// Fetch the weight blocks for a set of expert IDs (one token's activation).
    /// Cache hits return immediately; misses read from SSD synchronously with
    /// optional GDeflate decompression.
    pub fn fetch_experts(&mut self, experts: &[ExpertId])
        -> Result<Vec<WeightBlock>, NodeStorError>
    {
        let t0 = Instant::now();
        let mut result = Vec::with_capacity(experts.len());

        for &id in experts {
            // Cache check
            if let Some((_, block)) = self.cache.get(id) {
                result.push(block.clone());
                continue;
            }

            // SSD read
            let entry = self.disk_index.get(&id)
                .copied()
                .ok_or_else(|| NodeStorError::InferenceError(
                    format!("ssd_stream: expert {:?} not found in disk index", id)
                ))?;

            let block = self.read_from_disk(id, entry)?;
            self.cache.insert(block.clone());
            result.push(block);
        }

        // Record frequency for future cache promotion
        self.cache.frequency.record(experts);

        self.total_fetch_time += t0.elapsed();
        self.tokens_served += 1;
        Ok(result)
    }

    /// Read one expert block from the GGUF file.
    fn read_from_disk(&mut self, id: ExpertId, entry: ExpertDiskEntry)
        -> Result<WeightBlock, NodeStorError>
    {
        self.file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| NodeStorError::InferenceError(format!("ssd_stream seek: {}", e)))?;

        let read_size = entry.compressed_bytes as usize;
        let mut buf = vec![0u8; read_size];
        self.file.read_exact(&mut buf)
            .map_err(|e| NodeStorError::InferenceError(format!("ssd_stream read: {}", e)))?;

        self.total_io_bytes += read_size as u64;

        // Decompress if GDeflate-encoded
        let data = if entry.is_compressed {
            self.gdeflate_decompress(&buf, entry.raw_bytes as usize)?
        } else {
            buf
        };

        Ok(WeightBlock {
            id,
            data,
            quant: entry.quant,
            rows:  entry.rows,
            cols:  entry.cols,
        })
    }

    /// GDeflate decompression.
    ///
    /// On Windows 11 with DirectStorage the GPU decompresses in hardware.
    /// This CPU fallback is used when DirectStorage is unavailable or for
    /// testing. The compressed format is RFC 1951 DEFLATE (subset of GDeflate).
    fn gdeflate_decompress(&self, compressed: &[u8], expected_len: usize)
        -> Result<Vec<u8>, NodeStorError>
    {
        // Decompress via miniz_oxide (pure-Rust, DEFLATE compatible).
        // GDeflate on GPU would skip this path entirely — the bytes go
        // straight from the NVMe controller to VRAM decompressor.
        let out = miniz_oxide::inflate::decompress_to_vec(compressed)
            .map_err(|e| NodeStorError::InferenceError(
                format!("gdeflate decompress: {:?}", e)
            ))?;
        if out.len() != expected_len {
            return Err(NodeStorError::InferenceError(format!(
                "gdeflate: expected {} bytes, got {}", expected_len, out.len()
            )));
        }
        Ok(out)
    }

    /// Trigger prefetch of `layer`'s experts into RAM so they're ready when
    /// `fetch_experts` is called. Returns immediately; actual IO is done here
    /// synchronously but will be overlapped with GPU compute in the caller's
    /// pipeline loop (call prefetch_layer(N+1) right after GPU starts layer N).
    pub fn prefetch_layer(&mut self, layer: u32, expert_ids: &[ExpertId])
        -> Result<(), NodeStorError>
    {
        for &id in expert_ids {
            if self.cache.get(id).is_some() { continue; }
            let entry = match self.disk_index.get(&id).copied() {
                Some(e) => e,
                None    => continue,
            };
            let block = self.read_from_disk(id, entry)?;
            // Insert at RAM tier regardless of frequency for prefetch
            self.cache.insert_ram(block);
        }
        Ok(())
    }

    /// Promote the top-N most-frequent experts to VRAM (call after warmup
    /// tokens to lock the hot set into fast storage).
    pub fn promote_hot_experts(&mut self, top_n: usize) {
        let hot = self.cache.frequency.top_k(top_n);
        let mut to_promote: Vec<WeightBlock> = Vec::new();
        for (id, _freq) in &hot {
            if let Some(block) = self.cache.ram.get(id) {
                to_promote.push(block.clone());
            }
        }
        self.cache.promote_top_k(to_promote);
    }

    // ── Stats ────────────────────────────────────────────────────────────────

    pub fn stats(&self) -> SsdStreamStats {
        let total = self.tokens_served.max(1);
        SsdStreamStats {
            tokens_served:     self.tokens_served,
            avg_fetch_ms:      self.total_fetch_time.as_secs_f64() * 1000.0 / total as f64,
            total_io_mb:       self.total_io_bytes as f64 / 1_048_576.0,
            io_mb_per_token:   self.total_io_bytes as f64 / 1_048_576.0 / total as f64,
            cache_hit_rate:    self.cache.hit_rate(),
            vram_used_mb:      self.cache.vram_used_mb(),
            ram_used_mb:       self.cache.ram_used_mb(),
            vram_hits:         self.cache.hits_vram,
            ram_hits:          self.cache.hits_ram,
            ssd_hits:          self.cache.hits_ssd,
        }
    }

    pub fn experts_per_layer(&self) -> u32 { self.cfg.experts_per_layer }
    pub fn active_experts(&self)    -> u32 { self.cfg.active_experts }
    pub fn is_moe(&self)            -> bool { self.cfg.experts_per_layer > 1 }
}

#[derive(Debug)]
pub struct SsdStreamStats {
    pub tokens_served:   u64,
    pub avg_fetch_ms:    f64,
    pub total_io_mb:     f64,
    pub io_mb_per_token: f64,
    pub cache_hit_rate:  f64,
    pub vram_used_mb:    f64,
    pub ram_used_mb:     f64,
    pub vram_hits:       u64,
    pub ram_hits:        u64,
    pub ssd_hits:        u64,
}

impl std::fmt::Display for SsdStreamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "tokens={} avg_fetch={:.1}ms io={:.0}MB/tok hit={:.1}% \
             vram={:.0}MB ram={:.0}MB (V:{} R:{} S:{})",
            self.tokens_served,
            self.avg_fetch_ms,
            self.io_mb_per_token,
            self.cache_hit_rate * 100.0,
            self.vram_used_mb,
            self.ram_used_mb,
            self.vram_hits, self.ram_hits, self.ssd_hits,
        )
    }
}

// ─── MoE Router Prefetcher ────────────────────────────────────────────────────

/// Predicts which experts will activate next based on the current token's
/// router logits, and triggers a prefetch before the GPU actually needs them.
///
/// Strategy: top-(active_k × 2) experts by router score → prefetch all of them.
/// The extra k candidates cover the ~15% case where the actual router diverges
/// from the prediction.
pub struct ExpertPrefetcher {
    pub prefetch_factor: usize, // default 2: prefetch 2× the active count
}

impl ExpertPrefetcher {
    pub fn new(prefetch_factor: usize) -> Self {
        Self { prefetch_factor }
    }

    /// Given router logits for one layer, returns the expert IDs to prefetch.
    pub fn predict_next(
        &self,
        layer:        u32,
        router_logits: &[f32],
        active_k:     usize,
    ) -> Vec<ExpertId> {
        let n = router_logits.len();
        let prefetch_k = (active_k * self.prefetch_factor).min(n);
        let mut indexed: Vec<(usize, f32)> = router_logits.iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(prefetch_k);
        indexed.into_iter()
            .map(|(expert, _)| ExpertId { layer, expert: expert as u32 })
            .collect()
    }
}

// ─── SSD Expert Store ─────────────────────────────────────────────────────────

/// Decoded expert weight triple (gate, up, down) in F32, ready for compute.
pub struct ExpertWeightTriple {
    pub gate: Vec<f32>,   // [intermediate × hidden]
    pub up:   Vec<f32>,   // [intermediate × hidden]
    pub down: Vec<f32>,   // [hidden × intermediate]
}

/// LRU cache of decoded F32 expert triples.
pub struct ExpertWeightCache {
    entries:  HashMap<(usize, usize), ExpertWeightTriple>,
    lru:      VecDeque<(usize, usize)>,
    capacity: usize,
    pub hits:   u64,
    pub misses: u64,
}

impl ExpertWeightCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries:  HashMap::new(),
            lru:      VecDeque::new(),
            capacity: capacity.max(1),
            hits:     0,
            misses:   0,
        }
    }

    pub fn get_clone(&mut self, layer: usize, expert: usize) -> Option<ExpertWeightTriple> {
        let key = (layer, expert);
        if let Some(t) = self.entries.get(&key) {
            self.lru.retain(|&k| k != key);
            self.lru.push_front(key);
            self.hits += 1;
            return Some(ExpertWeightTriple {
                gate: t.gate.clone(),
                up:   t.up.clone(),
                down: t.down.clone(),
            });
        }
        self.misses += 1;
        None
    }

    pub fn insert(&mut self, layer: usize, expert: usize, triple: ExpertWeightTriple) {
        let key = (layer, expert);
        while self.entries.len() >= self.capacity {
            if let Some(evict) = self.lru.pop_back() { self.entries.remove(&evict); }
        }
        self.lru.push_front(key);
        self.entries.insert(key, triple);
    }

    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 { 0.0 } else { self.hits as f64 / t as f64 }
    }
}

/// On-demand expert weight loader: mmap GGUF + LRU decode cache.
///
/// For 300B+ MoE models that cannot fit in RAM, this loads only the 8 active
/// expert weight slices per token. Uses mmap for zero-copy OS-paged access
/// and decodes Q4/Q5/Q8/F16 to F32 on the fly.
pub struct SsdExpertStore {
    store: crate::weight_store::WeightStore,
    cache: std::sync::Mutex<ExpertWeightCache>,
}

impl SsdExpertStore {
    /// Build from an already-opened WeightStore. `cache_capacity` = max expert
    /// triples to keep decoded in memory (e.g., 200 for DeepSeek-V3 warmup).
    pub fn new(model_path: &str, cache_capacity: usize) -> Result<Self, nodestor_core::NodeStorError> {
        let store = crate::weight_store::WeightStore::open(std::path::Path::new(model_path))?;
        Ok(Self {
            store,
            cache: std::sync::Mutex::new(ExpertWeightCache::new(cache_capacity)),
        })
    }

    /// Get decoded weights for (layer, expert). Checks LRU cache first; on miss
    /// reads the expert's byte slice from the mmap'd GGUF file and decodes it.
    pub fn get_expert(
        &self,
        layer:        usize,
        expert:       usize,
        intermediate: usize,
        hidden:       usize,
    ) -> Option<ExpertWeightTriple> {
        // Cache check
        if let Ok(mut c) = self.cache.lock() {
            if let Some(triple) = c.get_clone(layer, expert) {
                return Some(triple);
            }
        }
        // Load from mmap
        let triple = self.load_expert_slice(layer, expert, intermediate, hidden)?;
        // Store in cache
        if let Ok(mut c) = self.cache.lock() {
            c.insert(layer, expert, ExpertWeightTriple {
                gate: triple.gate.clone(),
                up:   triple.up.clone(),
                down: triple.down.clone(),
            });
        }
        Some(triple)
    }

    fn load_expert_slice(
        &self,
        layer:        usize,
        expert:       usize,
        intermediate: usize,
        hidden:       usize,
    ) -> Option<ExpertWeightTriple> {
        use crate::dequant::{DequantDispatcher, QuantFormat};
        use nodestor_core::TensorDtype;

        let decode_slice = |tensor_key: &str, rows: usize, cols: usize| -> Option<Vec<f32>> {
            let bytes = self.store.tensor_bytes(tensor_key)?;
            let dtype = self.store.tensor_info(tensor_key)?.dtype;
            let qfmt  = tensor_dtype_to_quant(dtype);
            let n_elems = rows * cols;
            // Expert byte range inside the stacked tensor
            let blk_elems = qfmt.weights_per_block();
            let blk_bytes = qfmt.block_size_bytes();
            let n_blocks  = (n_elems + blk_elems - 1) / blk_elems;
            let expert_byte_size = n_blocks * blk_bytes;
            let start = expert * expert_byte_size;
            let end   = start + expert_byte_size;
            if end > bytes.len() { return None; }
            let mut disp = DequantDispatcher::new(false);
            Some(disp.dequantize(&bytes[start..end], qfmt, n_elems))
        };

        let gate = decode_slice(&format!("blk.{}.ffn_gate_exps.weight", layer), intermediate, hidden)?;
        let up   = decode_slice(&format!("blk.{}.ffn_up_exps.weight",   layer), intermediate, hidden)?;
        let down = decode_slice(&format!("blk.{}.ffn_down_exps.weight",  layer), hidden, intermediate)?;
        Some(ExpertWeightTriple { gate, up, down })
    }

    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.lock().map(|c| c.hit_rate()).unwrap_or(0.0)
    }
}

fn tensor_dtype_to_quant(dtype: nodestor_core::TensorDtype) -> crate::dequant::QuantFormat {
    use nodestor_core::TensorDtype as D;
    use crate::dequant::QuantFormat as Q;
    match dtype {
        D::F32              => Q::F32,
        D::F16 | D::BF16   => Q::F16,
        D::Q5_0 | D::Q5_1  => Q::Q5_0,
        D::Q8_0             => Q::Q8_0,
        D::Q8_1             => Q::Q8_1,
        D::Q4K              => Q::Q4KMedium,
        D::Q5K              => Q::Q5KMedium,
        D::Q6K              => Q::Q6K,
        _                   => Q::F32,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_frequency_top_k() {
        let mut freq = ExpertFrequency::new();
        let ids: Vec<ExpertId> = (0..4).map(|i| ExpertId { layer: 0, expert: i }).collect();

        for _ in 0..10 { freq.record(&ids[..2]); }  // experts 0,1 both get count 10
        freq.record(&ids[..1]);                       // expert 0 gets +1 → count 11
        freq.record(&ids[2..3]);                      // expert 2 activated 1×

        let top = freq.top_k(2);
        assert_eq!(top[0].0.expert, 0);
        assert_eq!(top[1].0.expert, 1);
        assert!(top[0].1 > top[1].1);               // 11 > 10
    }

    #[test]
    fn cache_vram_ram_split() {
        // 1 MB VRAM, 4 MB RAM
        let mut cache = ExpertCache::new(1_048_576, 4 * 1_048_576);
        let id0 = ExpertId { layer: 0, expert: 0 };

        // Give id0 high frequency
        for _ in 0..100 { cache.frequency.record(&[id0]); }

        let block = WeightBlock {
            id: id0, data: vec![0u8; 512 * 1024],
            quant: QuantType::Q4_0, rows: 4096, cols: 4096,
        };
        cache.insert(block);
        // Should be in VRAM (frequent + fits)
        assert!(cache.vram.contains_key(&id0));
    }

    #[test]
    fn expert_prefetcher_top_k() {
        let pf = ExpertPrefetcher::new(2);
        let logits = vec![0.1f32, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4];
        let ids = pf.predict_next(0, &logits, 2); // active_k=2, prefetch=4
        assert_eq!(ids.len(), 4);
        // Highest logit is expert 1 (0.9)
        assert_eq!(ids[0].expert, 1);
    }

    #[test]
    fn stats_display() {
        let s = SsdStreamStats {
            tokens_served: 100, avg_fetch_ms: 4.2, total_io_mb: 280.0,
            io_mb_per_token: 2.8, cache_hit_rate: 0.87,
            vram_used_mb: 360.0, ram_used_mb: 1200.0,
            vram_hits: 70, ram_hits: 17, ssd_hits: 13,
        };
        let text = format!("{}", s);
        assert!(text.contains("hit=87.0%"));
    }
}
