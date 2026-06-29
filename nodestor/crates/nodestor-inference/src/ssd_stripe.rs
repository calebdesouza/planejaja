/// SsdStripe — Leituras Paralelas em Múltiplos SSDs
///
/// Distribui blocos de experts entre N SSDs em round-robin.
/// Leituras emitidas em paralelo via rayon::scope.
///
/// Com 4× NVMe PCIe 4.0 (4 × 7 GB/s = 28 GB/s brutos):
///   GDeflate 3× → 84 GB/s efetivos para a GPU
///   300B Q4 comprimido (58 GB) → 0.69 s por passagem completa
///   MoE 300B (28 MB ativos/token) → <2 ms de IO por token
///
/// Interface idêntica a um único SsdWeightStream — o chamador
/// não precisa saber quantos SSDs existem.

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use nodestor_core::NodeStorError;
use crate::ssd_stream::{ExpertId, ExpertDiskEntry, WeightBlock, QuantType, ExpertCache};

// ─── Shard ────────────────────────────────────────────────────────────────────

/// One SSD shard: a file handle + its slice of the disk index.
struct Shard {
    file:  Mutex<File>,
    index: HashMap<ExpertId, ExpertDiskEntry>,
}

impl Shard {
    fn open(path: &PathBuf, index: HashMap<ExpertId, ExpertDiskEntry>)
        -> Result<Self, NodeStorError>
    {
        let file = File::open(path)
            .map_err(|e| NodeStorError::InferenceError(
                format!("ssd_stripe: open {}: {}", path.display(), e)
            ))?;
        Ok(Self { file: Mutex::new(file), index })
    }

    fn read_block(&self, id: ExpertId) -> Result<Option<WeightBlock>, NodeStorError> {
        let entry = match self.index.get(&id) {
            Some(e) => *e,
            None    => return Ok(None),
        };

        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| NodeStorError::InferenceError(format!("stripe seek: {}", e)))?;

        let mut buf = vec![0u8; entry.compressed_bytes as usize];
        file.read_exact(&mut buf)
            .map_err(|e| NodeStorError::InferenceError(format!("stripe read: {}", e)))?;

        let data = if entry.is_compressed {
            decompress_gdeflate(&buf, entry.raw_bytes as usize)?
        } else {
            buf
        };

        Ok(Some(WeightBlock {
            id,
            data,
            quant: entry.quant,
            rows:  entry.rows,
            cols:  entry.cols,
        }))
    }
}

// ─── GDeflate decompressor ────────────────────────────────────────────────────

fn decompress_gdeflate(compressed: &[u8], expected: usize)
    -> Result<Vec<u8>, NodeStorError>
{
    let out = miniz_oxide::inflate::decompress_to_vec(compressed)
        .map_err(|e| NodeStorError::InferenceError(
            format!("gdeflate: {:?}", e)
        ))?;
    if out.len() != expected {
        return Err(NodeStorError::InferenceError(format!(
            "gdeflate size mismatch: got {} expected {}", out.len(), expected
        )));
    }
    Ok(out)
}

// ─── Stripe config ────────────────────────────────────────────────────────────

pub struct StripeConfig {
    /// One path per SSD. Round-robin block assignment.
    pub shard_paths: Vec<PathBuf>,
    /// Per-shard disk index (block_id → offset+size on that shard).
    pub shard_indices: Vec<HashMap<ExpertId, ExpertDiskEntry>>,
    /// Shared expert cache (VRAM + RAM tiers).
    pub cache: Arc<Mutex<ExpertCache>>,
    /// Use rayon parallel reads. Set false for single-threaded testing.
    pub parallel: bool,
}

// ─── SsdStripe ────────────────────────────────────────────────────────────────

pub struct SsdStripe {
    shards:   Vec<Arc<Shard>>,
    cache:    Arc<Mutex<ExpertCache>>,
    parallel: bool,

    // Stats
    pub total_io_bytes:   u64,
    pub total_fetch_time: Duration,
    pub tokens_served:    u64,
    pub cache_hits:       u64,
    pub ssd_reads:        u64,
}

impl SsdStripe {
    pub fn new(cfg: StripeConfig) -> Result<Self, NodeStorError> {
        if cfg.shard_paths.len() != cfg.shard_indices.len() {
            return Err(NodeStorError::InferenceError(
                "ssd_stripe: shard_paths and shard_indices must have the same length".into()
            ));
        }
        if cfg.shard_paths.is_empty() {
            return Err(NodeStorError::InferenceError(
                "ssd_stripe: at least one shard required".into()
            ));
        }

        let shards = cfg.shard_paths.iter().zip(cfg.shard_indices.into_iter())
            .map(|(path, index)| Shard::open(path, index).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            shards,
            cache:    cfg.cache,
            parallel: cfg.parallel,
            total_io_bytes:   0,
            total_fetch_time: Duration::ZERO,
            tokens_served:    0,
            cache_hits:       0,
            ssd_reads:        0,
        })
    }

    /// Fetch a batch of expert weight blocks for one token.
    ///
    /// Cache hits returned immediately.
    /// Cache misses dispatched in parallel across shards, then merged.
    pub fn fetch_experts(&mut self, experts: &[ExpertId])
        -> Result<Vec<WeightBlock>, NodeStorError>
    {
        let t0 = Instant::now();

        // Split into cache hits and misses
        let mut result: Vec<WeightBlock> = Vec::with_capacity(experts.len());
        let mut misses: Vec<ExpertId>    = Vec::new();

        {
            let mut cache = self.cache.lock().unwrap();
            for &id in experts {
                if let Some((_, block)) = cache.get(id) {
                    result.push(block.clone());
                    self.cache_hits += 1;
                } else {
                    misses.push(id);
                }
            }
        }

        if !misses.is_empty() {
            let loaded = if self.parallel {
                self.parallel_read(&misses)?
            } else {
                self.serial_read(&misses)?
            };

            for block in loaded {
                self.total_io_bytes += block.data.len() as u64;
                self.ssd_reads += 1;
                let mut cache = self.cache.lock().unwrap();
                cache.insert(block.clone());
                result.push(block);
            }
        }

        self.total_fetch_time += t0.elapsed();
        self.tokens_served += 1;
        Ok(result)
    }

    /// Read misses from their respective shards in parallel (rayon).
    fn parallel_read(&self, misses: &[ExpertId])
        -> Result<Vec<WeightBlock>, NodeStorError>
    {
        use rayon::prelude::*;

        let n_shards = self.shards.len();
        let shards   = &self.shards;

        // Route each miss to the correct shard
        let routed: Vec<(usize, ExpertId)> = misses.iter()
            .map(|&id| {
                let shard_idx = self.shard_for(id);
                (shard_idx, id)
            }).collect();

        // Parallel read — one thread per (shard, expert) pair
        let results: Vec<Result<Option<WeightBlock>, NodeStorError>> =
            routed.par_iter()
                  .map(|&(shard_idx, id)| shards[shard_idx].read_block(id))
                  .collect();

        let mut blocks = Vec::with_capacity(misses.len());
        for (i, res) in results.into_iter().enumerate() {
            match res? {
                Some(b) => blocks.push(b),
                None    => return Err(NodeStorError::InferenceError(
                    format!("ssd_stripe: expert {:?} not found in any shard", misses[i])
                )),
            }
        }
        Ok(blocks)
    }

    /// Sequential fallback (single-threaded, for tests / single-SSD).
    fn serial_read(&self, misses: &[ExpertId])
        -> Result<Vec<WeightBlock>, NodeStorError>
    {
        misses.iter().map(|&id| {
            let shard_idx = self.shard_for(id);
            self.shards[shard_idx].read_block(id)?.ok_or_else(||
                NodeStorError::InferenceError(
                    format!("ssd_stripe: expert {:?} not found", id)
                ))
        }).collect()
    }

    /// Deterministic round-robin shard assignment.
    /// Block N always lives on shard N % n_shards — no lookup table needed.
    fn shard_for(&self, id: ExpertId) -> usize {
        // Mix layer and expert indices for even distribution
        let key = (id.layer as u64).wrapping_mul(1000003)
            .wrapping_add(id.expert as u64);
        (key % self.shards.len() as u64) as usize
    }

    /// Estimate aggregate bandwidth in GB/s across all shards.
    pub fn bandwidth_gb_s(&self, shard_bw_gb_s: f64) -> f64 {
        self.shards.len() as f64 * shard_bw_gb_s
    }

    pub fn n_shards(&self) -> usize { self.shards.len() }

    pub fn stats(&self) -> StripeStats {
        let n = self.tokens_served.max(1) as f64;
        StripeStats {
            n_shards:        self.shards.len(),
            tokens_served:   self.tokens_served,
            avg_fetch_ms:    self.total_fetch_time.as_secs_f64() * 1000.0 / n,
            total_io_mb:     self.total_io_bytes as f64 / 1_048_576.0,
            io_mb_per_token: self.total_io_bytes as f64 / 1_048_576.0 / n,
            cache_hit_rate:  {
                let total = self.cache_hits + self.ssd_reads;
                if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
            },
        }
    }
}

#[derive(Debug)]
pub struct StripeStats {
    pub n_shards:        usize,
    pub tokens_served:   u64,
    pub avg_fetch_ms:    f64,
    pub total_io_mb:     f64,
    pub io_mb_per_token: f64,
    pub cache_hit_rate:  f64,
}

impl std::fmt::Display for StripeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "shards={} tokens={} avg={:.1}ms io={:.1}MB/tok hit={:.1}%",
            self.n_shards, self.tokens_served,
            self.avg_fetch_ms, self.io_mb_per_token,
            self.cache_hit_rate * 100.0,
        )
    }
}

// ─── Bandwidth estimator ──────────────────────────────────────────────────────

/// Theoretical throughput given stripe configuration and model profile.
#[derive(Debug)]
pub struct BandwidthEstimate {
    pub raw_gb_s:       f64,   // n_shards × shard_bw_gb_s
    pub effective_gb_s: f64,   // raw × gdeflate_ratio
    pub io_ms_per_token: f64,  // active_mb_per_token / effective_gb_s × 1000
    pub est_tps:        f64,   // 1000 / io_ms_per_token (IO-bound ceiling)
}

impl BandwidthEstimate {
    pub fn compute(
        n_shards:            usize,
        shard_bw_gb_s:       f64,
        gdeflate_ratio:      f64,
        active_mb_per_token: f64, // compressed MB read per token (after cache)
    ) -> Self {
        let raw       = n_shards as f64 * shard_bw_gb_s;
        let effective = raw * gdeflate_ratio;
        let io_ms     = active_mb_per_token / (effective * 1024.0) * 1000.0;
        let tps       = if io_ms > 0.0 { 1000.0 / io_ms } else { f64::INFINITY };
        Self {
            raw_gb_s:        raw,
            effective_gb_s:  effective,
            io_ms_per_token: io_ms,
            est_tps:         tps,
        }
    }
}

impl std::fmt::Display for BandwidthEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "raw={:.0}GB/s effective={:.0}GB/s io={:.2}ms/tok ceiling={:.0}tok/s",
            self.raw_gb_s, self.effective_gb_s,
            self.io_ms_per_token, self.est_tps,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssd_stream::ExpertCache;
    use tempfile::tempdir;

    fn make_expert_file(
        dir: &std::path::Path,
        name: &str,
        experts: &[(ExpertId, Vec<u8>)],
    ) -> (PathBuf, HashMap<ExpertId, ExpertDiskEntry>) {
        use std::io::Write;
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        let mut index = HashMap::new();
        let mut offset = 0u64;

        for (id, data) in experts {
            file.write_all(data).unwrap();
            index.insert(*id, ExpertDiskEntry {
                offset,
                compressed_bytes: data.len() as u64,
                raw_bytes:        data.len() as u64,
                quant:            QuantType::Q8_0,
                rows:             8,
                cols:             8,
                is_compressed:    false,
            });
            offset += data.len() as u64;
        }
        (path, index)
    }

    #[test]
    fn single_shard_fetch() {
        let dir = tempdir().unwrap();
        let id0 = ExpertId { layer: 0, expert: 0 };
        let id1 = ExpertId { layer: 0, expert: 1 };
        let (path, index) = make_expert_file(dir.path(), "shard0.bin", &[
            (id0, vec![0xAA; 64]),
            (id1, vec![0xBB; 64]),
        ]);

        let cache = Arc::new(Mutex::new(ExpertCache::new(
            8 * 1024 * 1024,
            64 * 1024 * 1024,
        )));

        let mut stripe = SsdStripe::new(StripeConfig {
            shard_paths:   vec![path],
            shard_indices: vec![index],
            cache,
            parallel: false,
        }).unwrap();

        let blocks = stripe.fetch_experts(&[id0, id1]).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].data, vec![0xAA; 64]);
        assert_eq!(blocks[1].data, vec![0xBB; 64]);
        assert_eq!(stripe.ssd_reads, 2);
    }

    #[test]
    fn two_shard_parallel() {
        let dir = tempdir().unwrap();
        // id0 → shard 0, id1 → shard 1 (by round-robin hash)
        let id0 = ExpertId { layer: 0, expert: 0 };
        let id1 = ExpertId { layer: 0, expert: 1 };

        let (p0, idx0) = make_expert_file(dir.path(), "s0.bin", &[(id0, vec![11u8; 32])]);
        let (p1, idx1) = make_expert_file(dir.path(), "s1.bin", &[(id1, vec![22u8; 32])]);

        let cache = Arc::new(Mutex::new(ExpertCache::new(
            8 * 1024 * 1024,
            64 * 1024 * 1024,
        )));

        let mut stripe = SsdStripe::new(StripeConfig {
            shard_paths:   vec![p0, p1],
            shard_indices: vec![idx0, idx1],
            cache,
            parallel: true,
        }).unwrap();

        let blocks = stripe.fetch_experts(&[id0, id1]).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn cache_hit_avoids_ssd() {
        let dir = tempdir().unwrap();
        let id0 = ExpertId { layer: 0, expert: 0 };
        let (path, index) = make_expert_file(dir.path(), "s.bin", &[(id0, vec![99u8; 64])]);

        let cache = Arc::new(Mutex::new(ExpertCache::new(
            8 * 1024 * 1024,
            64 * 1024 * 1024,
        )));

        let mut stripe = SsdStripe::new(StripeConfig {
            shard_paths:   vec![path],
            shard_indices: vec![index],
            cache,
            parallel: false,
        }).unwrap();

        // First fetch: SSD miss
        stripe.fetch_experts(&[id0]).unwrap();
        assert_eq!(stripe.ssd_reads, 1);
        assert_eq!(stripe.cache_hits, 0);

        // Second fetch: cache hit
        stripe.fetch_experts(&[id0]).unwrap();
        assert_eq!(stripe.ssd_reads, 1);  // no new SSD read
        assert_eq!(stripe.cache_hits, 1);
    }

    #[test]
    fn bandwidth_estimate_4_shards() {
        // 4× PCIe 4.0 NVMe, GDeflate 3×, MoE 300B active 28 MB/token compressed
        let est = BandwidthEstimate::compute(4, 7.0, 3.0, 28.0 / 1024.0);
        println!("4-shard 300B MoE: {}", est);
        // 4×7×3 = 84 GB/s effective, 28/1024 MB / 84 GB/s = 0.32ms → >3000 tok/s ceiling
        assert!(est.effective_gb_s > 80.0);
        assert!(est.est_tps > 1000.0, "Expected >1000 tok/s IO ceiling, got {:.0}", est.est_tps);
    }

    #[test]
    fn shard_assignment_distributes_evenly() {
        let n = 4;
        let mut counts = vec![0usize; n];
        // Create a dummy stripe just to test routing
        let dir = tempdir().unwrap();
        let id_any = ExpertId { layer: 0, expert: 0 };
        let (path, index) = make_expert_file(dir.path(), "d.bin", &[(id_any, vec![0u8; 1])]);
        let cache = Arc::new(Mutex::new(ExpertCache::new(1024, 1024)));
        let paths  = vec![path.clone(); n];
        let indices = vec![index; n];
        let stripe = SsdStripe::new(StripeConfig {
            shard_paths: paths, shard_indices: indices, cache, parallel: false,
        }).unwrap();

        for layer in 0..8u32 {
            for expert in 0..32u32 {
                let id = ExpertId { layer, expert };
                counts[stripe.shard_for(id)] += 1;
            }
        }
        // All shards should receive some blocks (rough balance)
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "Shard {} got 0 blocks", i);
        }
        println!("Shard distribution: {:?}", counts);
    }

    #[test]
    fn stats_display() {
        let s = StripeStats {
            n_shards: 4, tokens_served: 1000,
            avg_fetch_ms: 1.3, total_io_mb: 28000.0,
            io_mb_per_token: 28.0, cache_hit_rate: 0.72,
        };
        let t = format!("{}", s);
        assert!(t.contains("shards=4"));
        assert!(t.contains("hit=72.0%"));
    }
}
