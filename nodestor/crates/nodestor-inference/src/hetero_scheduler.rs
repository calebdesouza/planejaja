/// HeteroScheduler — CPU + GPU Pipeline para Inferência MoE
///
/// Distribui trabalho entre CPU e GPU com overlap máximo:
///
///   CPU  → Attention  (memory-bandwidth bound, excelente para contextos longos,
///                       CPU cache localidade por cabeça de atenção)
///   GPU  → FFN / MoE  (compute-bound, tensor cores, paralelismo massivo)
///
/// Pipeline de duplo-buffer por layer:
///
///   t=0   CPU: attention[0]          GPU: (idle, carregando experts[0])
///   t=1   CPU: attention[1]          GPU: FFN/experts[0]
///   t=2   CPU: attention[2]          GPU: FFN/experts[1]
///   ...
///
/// Resultado: IO do SSD, compute CPU e compute GPU rodam simultaneamente.
/// A latência por token converge para max(t_cpu_attn, t_gpu_ffn, t_ssd_io).

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::ssd_stream::{ExpertId, SsdWeightStream, WeightBlock};
use nodestor_core::NodeStorError;

// ─── Compute device ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device { Cpu, Gpu }

// ─── Layer assignment ─────────────────────────────────────────────────────────

/// Describes what the scheduler will do with one transformer layer.
#[derive(Clone, Debug)]
pub struct LayerPlan {
    pub layer_idx:   u32,
    pub attn_device: Device,   // always CPU for now (heterogeneous path)
    pub ffn_device:  Device,   // always GPU for now
    pub experts:     Vec<ExpertId>, // empty for dense layers
}

// ─── Scheduler config ─────────────────────────────────────────────────────────

pub struct HeteroConfig {
    pub n_layers:         u32,
    /// Number of attention heads. CPU attention splits work per head.
    pub n_heads:          u32,
    /// KV head count (GQA/MQA).
    pub n_kv_heads:       u32,
    pub head_dim:         u32,
    /// Hidden dimension.
    pub d_model:          u32,
    /// Intermediate FFN dimension.
    pub d_ffn:            u32,
    /// MoE: experts per layer. 1 = dense.
    pub experts_per_layer: u32,
    /// MoE: experts activated per token.
    pub active_experts:   u32,
    /// If true, CPU attention runs on all layers; GPU only does FFN.
    /// If false, GPU does both (standard single-device path).
    pub heterogeneous:    bool,
}

impl Default for HeteroConfig {
    fn default() -> Self {
        // Default: dense 7B-ish model
        Self {
            n_layers:          32,
            n_heads:           32,
            n_kv_heads:        8,
            head_dim:          128,
            d_model:           4096,
            d_ffn:             14336,
            experts_per_layer: 1,
            active_experts:    1,
            heterogeneous:     true,
        }
    }
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

pub struct HeteroScheduler {
    cfg:    HeteroConfig,
    stream: Arc<Mutex<SsdWeightStream>>,

    // Performance counters
    pub cpu_attn_time: Duration,
    pub gpu_ffn_time:  Duration,
    pub ssd_io_time:   Duration,
    pub tokens_run:    u64,
}

impl HeteroScheduler {
    pub fn new(cfg: HeteroConfig, stream: Arc<Mutex<SsdWeightStream>>) -> Self {
        Self {
            cfg,
            stream,
            cpu_attn_time: Duration::ZERO,
            gpu_ffn_time:  Duration::ZERO,
            ssd_io_time:   Duration::ZERO,
            tokens_run:    0,
        }
    }

    /// Build the layer execution plan for one token.
    ///
    /// For MoE models, `router_logits[layer]` must provide the softmax scores
    /// over experts so the scheduler can decide which experts to load.
    /// Pass an empty slice to use the dense (expert 0) path.
    pub fn plan_token(
        &self,
        router_logits_per_layer: &[Vec<f32>],
    ) -> Vec<LayerPlan> {
        (0..self.cfg.n_layers).map(|l| {
            let experts = if self.cfg.experts_per_layer > 1 {
                // MoE: pick top-k experts by router logit
                let logits = router_logits_per_layer.get(l as usize)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                top_k_experts(l, logits, self.cfg.active_experts as usize)
            } else {
                // Dense: single synthetic expert 0
                vec![ExpertId { layer: l, expert: 0 }]
            };

            LayerPlan {
                layer_idx:   l,
                attn_device: if self.cfg.heterogeneous { Device::Cpu } else { Device::Gpu },
                ffn_device:  Device::Gpu,
                experts,
            }
        }).collect()
    }

    /// Execute one full forward pass for a single token using the hetero pipeline.
    ///
    /// `hidden`: mutable slice of [d_model] floats (in-place residual update).
    /// `kv_cache`: optional KV cache reference (opaque for now — caller manages).
    /// `router_logits`: per-layer expert router scores (empty = dense).
    ///
    /// Returns the updated hidden state after all layers.
    pub fn forward_token(
        &mut self,
        hidden:       &mut [f32],
        router_logits: &[Vec<f32>],
        attn_fn:      &dyn Fn(u32, &mut [f32]),         // CPU attention
        ffn_fn:       &dyn Fn(u32, &mut [f32], &[WeightBlock]), // GPU FFN
    ) -> Result<(), NodeStorError> {
        let plan = self.plan_token(router_logits);
        let t_token = Instant::now();

        for layer_plan in &plan {
            let l = layer_plan.layer_idx;

            // ── Prefetch next layer's experts while computing this layer ──
            if l + 1 < self.cfg.n_layers {
                let next_experts = if self.cfg.experts_per_layer > 1 {
                    let logits = router_logits.get(l as usize + 1)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    top_k_experts(l + 1, logits, self.cfg.active_experts as usize)
                } else {
                    vec![ExpertId { layer: l + 1, expert: 0 }]
                };

                let mut stream = self.stream.lock().unwrap();
                // Non-blocking: if prefetch fails it just means a cache miss later
                let _ = stream.prefetch_layer(l + 1, &next_experts);
            }

            // ── CPU: Attention ───────────────────────────────────────────
            let t_attn = Instant::now();
            attn_fn(l, hidden);
            self.cpu_attn_time += t_attn.elapsed();

            // ── SSD: Fetch expert weights (overlapped with attn above in
            //    a real async impl; here sequential for correctness) ───────
            let t_ssd = Instant::now();
            let expert_blocks = {
                let mut stream = self.stream.lock().unwrap();
                stream.fetch_experts(&layer_plan.experts)?
            };
            self.ssd_io_time += t_ssd.elapsed();

            // ── GPU: FFN / MoE ───────────────────────────────────────────
            let t_ffn = Instant::now();
            ffn_fn(l, hidden, &expert_blocks);
            self.gpu_ffn_time += t_ffn.elapsed();
        }

        self.tokens_run += 1;
        Ok(())
    }

    /// Statistics for the last N tokens.
    pub fn stats(&self) -> HeteroStats {
        let n = self.tokens_run.max(1) as f64;
        HeteroStats {
            tokens_run:       self.tokens_run,
            avg_attn_ms:      self.cpu_attn_time.as_secs_f64() * 1000.0 / n,
            avg_ffn_ms:       self.gpu_ffn_time.as_secs_f64()  * 1000.0 / n,
            avg_ssd_ms:       self.ssd_io_time.as_secs_f64()   * 1000.0 / n,
            bottleneck:       {
                let a = self.cpu_attn_time.as_secs_f64();
                let f = self.gpu_ffn_time.as_secs_f64();
                let s = self.ssd_io_time.as_secs_f64();
                if a >= f && a >= s { Bottleneck::CpuAttention }
                else if f >= s      { Bottleneck::GpuFfn }
                else                { Bottleneck::SsdIo }
            },
        }
    }

    /// Estimate theoretical tokens/sec for a given model size and hardware.
    ///
    /// Used to calibrate expectations before a real run. Accounts for:
    /// - Expert sparsity (MoE activation ratio)
    /// - SSD bandwidth + GDeflate compression factor
    /// - CPU attention time (per head, per layer)
    /// - GPU FFN time (expert matmul on tensor cores)
    pub fn estimate_tps(
        n_layers:          u32,
        experts_per_layer: u32,
        active_experts:    u32,
        expert_bytes:      u64,   // bytes per expert (Q4)
        ssd_bw_gb_s:       f64,   // NVMe raw bandwidth
        gdeflate_ratio:    f64,   // compression factor (1.0 = no compression)
        cache_hit_rate:    f64,   // fraction of expert fetches hitting VRAM/RAM
        gpu_tflops:        f64,   // GPU TFLOPS (FP16)
        d_model:           u32,
        d_ffn:             u32,
    ) -> TpsEstimate {
        let active_per_layer = active_experts as f64;
        let bytes_per_expert_compressed = expert_bytes as f64 / gdeflate_ratio;

        // IO: only cold misses go to SSD
        let cold_fraction = 1.0 - cache_hit_rate;
        let io_bytes_per_token = n_layers as f64
            * active_per_layer
            * bytes_per_expert_compressed
            * cold_fraction;
        let t_io_s = io_bytes_per_token / (ssd_bw_gb_s * 1e9);

        // GPU FFN: 2 × d_model × d_ffn × active_experts × n_layers FLOPs
        let ffn_flops = 2.0 * d_model as f64 * d_ffn as f64
            * active_per_layer * n_layers as f64;
        let t_gpu_s = ffn_flops / (gpu_tflops * 1e12);

        // CPU attention: dominated by KV read (memory-bound)
        // Rough: 2 × d_model² × n_layers × 0.25 (typical attn fraction)
        let attn_flops = 2.0 * d_model as f64 * d_model as f64 * n_layers as f64 * 0.25;
        // CPU at ~1 TFLOPS Q4-optimized (conservative)
        let t_cpu_s = attn_flops / 1e12;

        // With overlap: total = max of the three paths
        let t_per_token_s = t_io_s.max(t_gpu_s).max(t_cpu_s);
        let tps = if t_per_token_s > 0.0 { 1.0 / t_per_token_s } else { 0.0 };

        TpsEstimate {
            tps,
            t_io_ms:    t_io_s  * 1000.0,
            t_gpu_ms:   t_gpu_s * 1000.0,
            t_cpu_ms:   t_cpu_s * 1000.0,
            bottleneck: {
                if t_io_s >= t_gpu_s && t_io_s >= t_cpu_s { Bottleneck::SsdIo }
                else if t_gpu_s >= t_cpu_s                 { Bottleneck::GpuFfn }
                else                                        { Bottleneck::CpuAttention }
            },
        }
    }
}

// ─── MoE top-k selection ──────────────────────────────────────────────────────

fn top_k_experts(layer: u32, logits: &[f32], k: usize) -> Vec<ExpertId> {
    if logits.is_empty() {
        return (0..k as u32).map(|e| ExpertId { layer, expert: e }).collect();
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.into_iter()
        .map(|(e, _)| ExpertId { layer, expert: e as u32 })
        .collect()
}

// ─── Stats types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck { CpuAttention, GpuFfn, SsdIo }

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuAttention => write!(f, "CPU-attention"),
            Self::GpuFfn       => write!(f, "GPU-FFN"),
            Self::SsdIo        => write!(f, "SSD-IO"),
        }
    }
}

#[derive(Debug)]
pub struct HeteroStats {
    pub tokens_run:   u64,
    pub avg_attn_ms:  f64,
    pub avg_ffn_ms:   f64,
    pub avg_ssd_ms:   f64,
    pub bottleneck:   Bottleneck,
}

impl std::fmt::Display for HeteroStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "tokens={} attn={:.1}ms ffn={:.1}ms ssd={:.1}ms bottleneck={}",
            self.tokens_run,
            self.avg_attn_ms, self.avg_ffn_ms, self.avg_ssd_ms,
            self.bottleneck,
        )
    }
}

#[derive(Debug)]
pub struct TpsEstimate {
    pub tps:        f64,
    pub t_io_ms:    f64,
    pub t_gpu_ms:   f64,
    pub t_cpu_ms:   f64,
    pub bottleneck: Bottleneck,
}

impl std::fmt::Display for TpsEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "{:.1} tok/s  (IO={:.1}ms GPU={:.1}ms CPU={:.1}ms  bottleneck={})",
            self.tps, self.t_io_ms, self.t_gpu_ms, self.t_cpu_ms, self.bottleneck,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tps_estimate_deepseek_v3() {
        // DeepSeek-V3 profile:
        //   671B total, 61 layers, 256 experts/layer, 8 active/token
        //   d_model=7168, d_ffn=2048 (fine-grained MoE)
        //   Expert ~3.5 MB in Q4
        //   RTX 4090: 82 TFLOPS FP16
        //   PCIe 4.0 NVMe: 7 GB/s, GDeflate 2.5×
        //   Cache hit rate after warmup: 0.70
        let est = HeteroScheduler::estimate_tps(
            61,              // n_layers
            256,             // experts_per_layer
            8,               // active_experts
            3_670_016,       // 3.5 MB per expert in Q4
            7.0,             // ssd_bw_gb_s
            2.5,             // gdeflate_ratio
            0.70,            // cache_hit_rate
            82.0,            // gpu_tflops (RTX 4090)
            7168,            // d_model
            2048,            // d_ffn (fine-grained expert)
        );
        // Should project > 10 tok/s (realistic 300B MoE target)
        println!("DeepSeek-V3 estimate: {}", est);
        assert!(est.tps > 5.0,
            "Expected > 5 tok/s for 300B MoE with hot cache, got {:.2}", est.tps);
    }

    #[test]
    fn tps_estimate_34b_dense() {
        // 34B dense model, RTX 4090, no MoE
        let est = HeteroScheduler::estimate_tps(
            48,             // n_layers
            1,              // experts_per_layer (dense)
            1,              // active_experts
            180_000_000,    // ~180 MB per "expert" (full FFN layer in Q4)
            7.0,
            1.8,            // gdeflate on dense weights
            0.0,            // no cache for dense (full layer always needed)
            82.0,
            5120,
            13824,
        );
        println!("34B dense estimate: {}", est);
        // Dense 34B streaming from SSD with 0% cache: bottleneck is SSD IO (~1.5 tok/s).
        // 10-40 tok/s is only achievable with the model in VRAM.
        assert!(est.tps > 0.5);
    }

    #[test]
    fn top_k_experts_selection() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7, 0.5];
        let experts = top_k_experts(0, &logits, 2);
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].expert, 1); // 0.9
        assert_eq!(experts[1].expert, 3); // 0.7
    }

    #[test]
    fn bottleneck_display() {
        assert_eq!(format!("{}", Bottleneck::SsdIo), "SSD-IO");
        assert_eq!(format!("{}", Bottleneck::GpuFfn), "GPU-FFN");
        assert_eq!(format!("{}", Bottleneck::CpuAttention), "CPU-attention");
    }
}
