//! GPU-accelerated forward pass para modelos Transformer (Qwen2/Llama).
//!
//! Estratégia híbrida: matmuls pesados (W_q, W_k, W_v, W_o, FFN gate/up/down, lm_head)
//! rodam nos compute shaders Vulkan. KV cache, RoPE e atenção GQA rodam na CPU — são
//! operações pequenas (< 4 KB por camada para tokens únicos) que não justificam o
//! overhead de dispatch Vulkan. O ganho vem do gargalo real: leitura de pesos da VRAM
//! a 256 GB/s vs RAM DDR4 a ~40 GB/s.
//!
//! Para especulação: `gpu_forward_verify` processa N rascunhos em sequência na GPU,
//! mantendo os pesos carregados no cache L2/SRAM entre os passos — exatamente o
//! "N tokens por weight-load" prometido pela teoria.

use std::sync::OnceLock;
use nodestor_vulkan::{VulkanEngine, WeightBank};
use crate::cpu_reference::{CpuModelConfig, CpuKvCache};

static WAVEFRONT: OnceLock<crate::wavefront_scheduler::WavefrontScheduler> = OnceLock::new();

#[inline]
fn global_wavefront() -> &'static crate::wavefront_scheduler::WavefrontScheduler {
    WAVEFRONT.get_or_init(crate::wavefront_scheduler::WavefrontScheduler::default)
}

/// y[o] = Σ_i W[o·in_dim + i] · x[i], executed in wavefront slices for TDR immunity.
/// Borrows w and x directly — no intermediate weight allocation.
fn wavefront_matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    global_wavefront().dispatch_sliced(out_dim, |start, end| {
        for o in start..end {
            let base = o * in_dim;
            if base + in_dim > w.len() { continue; }
            out[o] = w[base..base + in_dim].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        }
    });
    out
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ─── CPU micro-kernels (inline duplicates of cpu_reference privates) ─────────
// Used by the CPU arm of per-layer heterogeneous dispatch so that staging_wb
// (HOST_VISIBLE / mmap SSD) serves layers that did not fit in DEVICE_LOCAL.

#[inline]
fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w.get(i).copied().unwrap_or(1.0)).collect()
}

#[inline]
fn cpu_silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

/// y[o] = Σ_i W[o·in + i] · x[i] — rayon-parallel, AVX2 auto-vectorised.
fn cpu_matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    (0..out_dim).into_par_iter().map(|o| {
        let base = o * in_dim;
        if base + in_dim > w.len() { return 0.0f32; }
        w[base..base + in_dim].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
    }).collect()
}

/// Permuta bias NEOX→INTERLEAVED in-place (Q/K heads).
#[inline]
fn cpu_add_bias_neox(vec: &mut [f32], bias: &[f32], n_heads: usize, hd: usize) {
    let half = hd / 2;
    for h in 0..n_heads {
        let off = h * hd;
        for j in 0..half {
            if off + 2 * j + 1 < vec.len() && off + half + j < bias.len() {
                vec[off + 2 * j]     += bias[off + j];
                vec[off + 2 * j + 1] += bias[off + half + j];
            }
        }
    }
}

/// SwiGLU FFN: gate=SiLU(Wg·x)*Wu·x, out=Wd·mid — reads from `wb` (staging or GPU).
fn cpu_swiglu_ffn(
    normed: &[f32],
    wb: &nodestor_vulkan::WeightBank,
    layer: usize,
    inter: usize,
    hidden: usize,
) -> Vec<f32> {
    let wg = wb.get(&format!("blk.{layer}.ffn_gate.weight")).map(|b| b.as_f32_slice());
    let wu = wb.get(&format!("blk.{layer}.ffn_up.weight")).map(|b| b.as_f32_slice());
    let wd = wb.get(&format!("blk.{layer}.ffn_down.weight")).map(|b| b.as_f32_slice());
    match (wg, wu, wd) {
        (Some(wg), Some(wu), Some(wd)) => {
            let gate = cpu_matvec(wg, normed, inter, hidden);
            let up   = cpu_matvec(wu, normed, inter, hidden);
            let mut mid = vec![0.0f32; inter];
            for i in 0..inter { mid[i] = cpu_silu(gate[i]) * up[i]; }
            cpu_matvec(wd, &mid, hidden, inter)
        }
        _ => vec![0.0f32; hidden],
    }
}

// ─── Hidden-state envelope for heterogeneous ping-pong ───────────────────────

/// Wraps the transformer hidden state as it moves between VRAM and host RAM.
/// Transfers occur only at physical compute boundaries (GPU→CPU or CPU→GPU layer
/// transitions), not inside a layer — achieving zero-copy within each arm.
enum HiddenState {
    Cpu(Vec<f32>),
    Gpu(nodestor_vulkan::GpuBuffer),
}

impl HiddenState {
    /// Returns host copy, downloading from VRAM only when currently on GPU.
    fn to_cpu(self, engine: &VulkanEngine) -> Option<Vec<f32>> {
        match self {
            HiddenState::Cpu(v) => Some(v),
            HiddenState::Gpu(b) => engine.download_f32(&b).ok(),
        }
    }

    /// Returns GPU buffer, uploading from host only when currently on CPU.
    fn to_gpu(self, engine: &VulkanEngine) -> Option<nodestor_vulkan::GpuBuffer> {
        match self {
            HiddenState::Gpu(b) => Some(b),
            HiddenState::Cpu(v) => engine.upload_f32(&v).ok(),
        }
    }

    /// APEX: Orthogonal projection (POD) applied in the native representation.
    ///
    /// h' = h − intensity · (⟨h, d⟩ / ‖d‖²) · d
    ///
    /// GPU arm: download → subtract → upload (via apply_steering_gpu).
    /// CPU arm: project_out_direction_saturating in-place on Vec<f32> — no allocation
    ///          beyond the existing buffer, no round-trip through Vulkan.
    fn apply_apex(
        self,
        engine: &VulkanEngine,
        dir: &[f32],
        intensity: f32,
        d_sq: f32,
        layer: usize,
    ) -> Option<HiddenState> {
        if d_sq < 1e-12 { return Some(self); }
        match self {
            HiddenState::Gpu(buf) => {
                apply_steering_gpu(buf, engine, dir, intensity, d_sq, layer)
                    .map(HiddenState::Gpu)
            }
            HiddenState::Cpu(mut v) => {
                crate::refusal_mapper::project_out_direction_saturating(
                    &mut v, dir, intensity, 4.0,
                );
                crate::observability::emit(layer, &v);
                Some(HiddenState::Cpu(v))
            }
        }
    }
}

/// Permuta bias de NEOX-order para INTERLEAVED-order para soma correta com
/// vetores Q/K que já foram permutados pelo llama.cpp no GGUF.
fn bias_neox_to_interleaved(bias: &[f32], n_heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_heads * hd];
    for h in 0..n_heads {
        for j in 0..hd / 2 {
            out[h * hd + 2 * j]     = bias[h * hd + j];
            out[h * hd + 2 * j + 1] = bias[h * hd + hd / 2 + j];
        }
    }
    out
}

/// RoPE INTERLEAVED (NEOX-style) idêntico ao cpu_reference::rope_neox.
fn rope_neox(v: &mut [f32], n_heads: usize, hd: usize, pos: usize, base: f32) {
    for h in 0..n_heads {
        for i in 0..hd / 2 {
            let freq = 1.0 / base.powf(2.0 * i as f32 / hd as f32);
            let theta = pos as f32 * freq;
            let (s, c) = theta.sin_cos();
            let idx = h * hd + 2 * i;
            let v0 = v[idx];
            let v1 = v[idx + 1];
            v[idx]     = v0 * c - v1 * s;
            v[idx + 1] = v0 * s + v1 * c;
        }
    }
}

/// GQA attention na CPU: O(klen × nq × hd) — pequeno para tokens únicos.
/// `k_cache[p]` = [nkv × hd], `v_cache[p]` = [nkv × hd].
fn gqa_attention(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    klen: usize,
    nq: usize,
    group: usize, // nq / nkv
    hd: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * hd];
    for h in 0..nq {
        let kv_h = h / group;
        let qh = &q[h * hd..(h + 1) * hd];

        // scores = Q·K^T × scale, mascarado causalmente (klen inclui posição atual)
        let mut scores: Vec<f32> = (0..klen)
            .map(|p| {
                let kp = &k_cache[p][kv_h * hd..(kv_h + 1) * hd];
                qh.iter().zip(kp.iter()).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();

        // softmax estável
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max_s).exp(); *s }).sum();
        if sum > 0.0 { scores.iter_mut().for_each(|s| *s /= sum); }

        // weighted sum de V
        for i in 0..hd {
            out[h * hd + i] = (0..klen)
                .map(|p| scores[p] * v_cache[p][kv_h * hd + i])
                .sum();
        }
    }
    out
}

// ─── Forward step único (GPU matmuls + CPU KV/attn) ─────────────────────────

/// Aplica POD (Projeção Ortogonal Dinâmica) in-place num buffer GPU.
/// Faz download → subtract projection → upload. Retorna None se dimensões incompatíveis.
#[inline]
fn apply_steering_gpu(
    x: nodestor_vulkan::GpuBuffer,
    engine: &VulkanEngine,
    dir: &[f32],
    intensity: f32,
    d_sq: f32,
    layer: usize,
) -> Option<nodestor_vulkan::GpuBuffer> {
    if d_sq < 1e-12 { return Some(x); }
    let mut x_data = engine.download_f32(&x).ok()?;
    if x_data.len() < dir.len() { return Some(x); }
    let h_dot: f32 = x_data.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
    if !h_dot.is_finite() { return Some(x); }
    let scale = intensity * h_dot / d_sq;
    if !scale.is_finite() { return Some(x); }
    x_data.iter_mut().zip(dir.iter()).for_each(|(xi, di)| *xi -= scale * di);
    // Emit to observability sink when verbose-neurons mode is active (zero cost otherwise).
    crate::observability::emit(layer, &x_data);
    engine.upload_f32(&x_data).ok()
}

/// Forward de UM token na posição `pos`, usando GPU para os matmuls pesados.
/// `steering`: `Some((direction, intensity, d_sq))` para POD após cada residual.
/// Retorna logits[vocab] ou `None` em caso de erro (fallback: caller usa CPU).
pub fn gpu_forward_step(
    token: u32,
    pos: usize,
    cfg: &CpuModelConfig,
    engine: &VulkanEngine,
    wb: &WeightBank,
    staging_wb: &WeightBank,
    kv: &mut CpuKvCache,
    steering: Option<(&[f32], f32, f32)>,
) -> Option<Vec<f32>> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nq = cfg.n_heads;
    let nkv = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    // Embedding: always from staging_wb (HOST_VISIBLE / mmap SSD).
    // The gpu_weight_bank buffers are DEVICE_LOCAL — as_f32_slice() returns &[] on them.
    let emb = {
        let tok_embd = staging_wb.get("token_embd.weight")?;
        let s = tok_embd.as_f32_slice();
        let start = token as usize * h;
        if start + h > s.len() { return None; }
        s[start..start + h].to_vec()
    };

    // Hidden state starts on CPU; first GPU layer will upload it transparently.
    let mut hs = HiddenState::Cpu(emb);

    // Per-layer heterogeneous dispatch.
    // GPU layers: weights in DEVICE_LOCAL → Vulkan compute shaders (256 GB/s VRAM BW).
    // CPU layers: weights in staging_wb   → rayon SIMD micro-kernels (SSD mmap, 7 GB/s).
    // Transfer at physical boundaries only (GPU→CPU or CPU→GPU), never inside a layer.
    // APEX mathematical interventions (POD orthogonal projection) fire at BOTH residual
    // points in every layer regardless of which compute device ran that layer.
    for layer in 0..cfg.n_layers {
        let layer_on_gpu = wb.get(&format!("blk.{layer}.attn_q.weight"))
            .map_or(false, |t| t.is_on_gpu());

        if layer_on_gpu {
            // ── GPU path ─────────────────────────────────────────────────────
            // Batch A: RmsNorm + Q + K + V → 1 queue_wait_idle (was 4).
            let x = hs.to_gpu(engine)?;
            let attn_norm = wb.get(&format!("blk.{layer}.attn_norm.weight"))?;
            let wq = wb.get(&format!("blk.{layer}.attn_q.weight"))?;
            let wk = wb.get(&format!("blk.{layer}.attn_k.weight"))?;
            let wv = wb.get(&format!("blk.{layer}.attn_v.weight"))?;
            let (x_norm, q_gpu, k_gpu, v_gpu) = engine.batch_attn_prep(
                &x, attn_norm, wq, wk, wv,
                h as u32, q_dim as u32, kv_dim as u32, cfg.eps,
            ).ok()?;

            let mut q_vec = q_gpu.as_f32_slice().to_vec();
            let mut k_vec = k_gpu.as_f32_slice().to_vec();
            let mut v_vec = v_gpu.as_f32_slice().to_vec();
            if q_vec.is_empty() || k_vec.is_empty() || v_vec.is_empty() { return None; }
            let _ = x_norm; // keep alive until after barrier flush

            if let Some(bq) = wb.get(&format!("blk.{layer}.attn_q.bias")) {
                let p = bias_neox_to_interleaved(bq.as_f32_slice(), nq, hd);
                for i in 0..q_vec.len().min(p.len()) { q_vec[i] += p[i]; }
            }
            if let Some(bk) = wb.get(&format!("blk.{layer}.attn_k.bias")) {
                let p = bias_neox_to_interleaved(bk.as_f32_slice(), nkv, hd);
                for i in 0..k_vec.len().min(p.len()) { k_vec[i] += p[i]; }
            }
            if let Some(bv) = wb.get(&format!("blk.{layer}.attn_v.bias")) {
                let bs = bv.as_f32_slice();
                for i in 0..v_vec.len().min(bs.len()) { v_vec[i] += bs[i]; }
            }

            rope_neox(&mut q_vec, nq, hd, pos, cfg.rope_base);
            rope_neox(&mut k_vec, nkv, hd, pos, cfg.rope_base);
            kv.push_kv(layer, k_vec, v_vec);
            let (k_cache, v_cache) = kv.layer_kv(layer);
            let attn_out = gqa_attention(&q_vec, k_cache, v_cache, k_cache.len(), nq, group, hd, scale);

            // Batch B: matmul(wo, attn) + add(x, proj) → 1 queue_wait_idle (was 2).
            let attn_gpu = engine.upload_f32(&attn_out).ok()?;
            let wo = wb.get(&format!("blk.{layer}.attn_output.weight"))?;
            let x = engine.batch_attn_out(wo, &attn_gpu, &x, h as u32, q_dim as u32).ok()?;

            // APEX post-attention (GPU arm: download → POD → upload)
            let mut hs_post_attn = HiddenState::Gpu(x);
            if let Some((dir, intensity, d_sq)) = steering {
                hs_post_attn = hs_post_attn.apply_apex(engine, dir, intensity, d_sq, layer)?;
            }

            // Batch C: RmsNorm + gate + up + SiLU + mul + down → 1 queue_wait_idle (was 6).
            let x = hs_post_attn.to_gpu(engine)?;
            let ffn_norm = wb.get(&format!("blk.{layer}.ffn_norm.weight"))?;

            // batch_ffn now includes the residual add (x + ffn(x_norm)) — saves 1 sync/layer.
            let x = if let (Some(wg), Some(wu), Some(wd)) = (
                wb.get(&format!("blk.{layer}.ffn_gate.weight")),
                wb.get(&format!("blk.{layer}.ffn_up.weight")),
                wb.get(&format!("blk.{layer}.ffn_down.weight")),
            ) {
                engine.batch_ffn(&x, ffn_norm, wg, wu, wd, h as u32, cfg.intermediate as u32, cfg.eps).ok()?
            } else if let Some(moe_cfg) = &cfg.moe {
                let x_norm2 = engine.rmsnorm(&x, ffn_norm, 1, h as u32, cfg.eps).ok()?;
                let normed_cpu = x_norm2.as_f32_slice().to_vec();
                let moe_out = crate::moe_kernel::moe_ffn_step(&normed_cpu, moe_cfg, wb, layer)?;
                crate::observability::emit_expert_selection(
                    layer, &moe_out.routing.expert_indices, moe_out.routing.entropy,
                );
                let moe_gpu = engine.upload_f32(&moe_out.hidden).ok()?;
                engine.add(&x, &moe_gpu, h as u32).ok()?
            } else { return None; };

            // APEX post-FFN (GPU arm)
            let mut hs_post_ffn = HiddenState::Gpu(x);
            if let Some((dir, intensity, d_sq)) = steering {
                hs_post_ffn = hs_post_ffn.apply_apex(engine, dir, intensity, d_sq, layer)?;
            }
            hs = hs_post_ffn;

        } else {
            // ── CPU path via staging_wb (HOST_VISIBLE / mmap SSD) ────────────
            // to_cpu() downloads from VRAM only when hs is GPU (boundary crossing).
            let mut x = hs.to_cpu(engine)?;

            let attn_norm = staging_wb.get(&format!("blk.{layer}.attn_norm.weight"))
                .map(|b| b.as_f32_slice().to_vec())
                .unwrap_or_else(|| vec![1.0f32; h]);
            let normed = cpu_rmsnorm(&x, &attn_norm, cfg.eps);

            let wq_b = staging_wb.get(&format!("blk.{layer}.attn_q.weight"))?;
            let wk_b = staging_wb.get(&format!("blk.{layer}.attn_k.weight"))?;
            let wv_b = staging_wb.get(&format!("blk.{layer}.attn_v.weight"))?;

            let mut q_vec = wavefront_matvec(wq_b.as_f32_slice(), &normed, q_dim, h);
            let mut k_vec = wavefront_matvec(wk_b.as_f32_slice(), &normed, kv_dim, h);
            let mut v_vec = wavefront_matvec(wv_b.as_f32_slice(), &normed, kv_dim, h);

            if let Some(bq) = staging_wb.get(&format!("blk.{layer}.attn_q.bias")) {
                cpu_add_bias_neox(&mut q_vec, bq.as_f32_slice(), nq, hd);
            }
            if let Some(bk) = staging_wb.get(&format!("blk.{layer}.attn_k.bias")) {
                cpu_add_bias_neox(&mut k_vec, bk.as_f32_slice(), nkv, hd);
            }
            if let Some(bv) = staging_wb.get(&format!("blk.{layer}.attn_v.bias")) {
                let bs = bv.as_f32_slice();
                for i in 0..v_vec.len().min(bs.len()) { v_vec[i] += bs[i]; }
            }

            rope_neox(&mut q_vec, nq, hd, pos, cfg.rope_base);
            rope_neox(&mut k_vec, nkv, hd, pos, cfg.rope_base);
            kv.push_kv(layer, k_vec, v_vec);
            let (k_cache, v_cache) = kv.layer_kv(layer);
            let attn_out = gqa_attention(&q_vec, k_cache, v_cache, k_cache.len(), nq, group, hd, scale);

            let wo_b = staging_wb.get(&format!("blk.{layer}.attn_output.weight"))?;
            let attn_proj = wavefront_matvec(wo_b.as_f32_slice(), &attn_out, h, q_dim);
            for i in 0..h { x[i] += attn_proj[i]; }

            // APEX post-attention (CPU arm: in-place POD projection, no Vulkan round-trip)
            if let Some((dir, intensity, d_sq)) = steering {
                if d_sq > 1e-12 {
                    crate::refusal_mapper::project_out_direction_saturating(
                        &mut x, dir, intensity, 4.0,
                    );
                    crate::observability::emit(layer, &x);
                }
            }

            let ffn_norm = staging_wb.get(&format!("blk.{layer}.ffn_norm.weight"))
                .map(|b| b.as_f32_slice().to_vec())
                .unwrap_or_else(|| vec![1.0f32; h]);
            let normed2 = cpu_rmsnorm(&x, &ffn_norm, cfg.eps);

            let ffn_out = if let Some(moe_cfg) = &cfg.moe {
                match crate::moe_kernel::moe_ffn_step(&normed2, moe_cfg, staging_wb, layer) {
                    Some(moe_out) => {
                        crate::observability::emit_expert_selection(
                            layer, &moe_out.routing.expert_indices, moe_out.routing.entropy,
                        );
                        moe_out.hidden
                    }
                    None => cpu_swiglu_ffn(&normed2, staging_wb, layer, cfg.intermediate, h),
                }
            } else {
                cpu_swiglu_ffn(&normed2, staging_wb, layer, cfg.intermediate, h)
            };
            for i in 0..h { x[i] += ffn_out[i]; }

            // APEX post-FFN (CPU arm)
            if let Some((dir, intensity, d_sq)) = steering {
                if d_sq > 1e-12 {
                    crate::refusal_mapper::project_out_direction_saturating(
                        &mut x, dir, intensity, 4.0,
                    );
                    crate::observability::emit(layer, &x);
                }
            }
            hs = HiddenState::Cpu(x);
        }
    }

    // Release batch descriptor pool — O(1) reset instead of N individual frees.
    engine.reset_batch_pool();

    // Final norm + LM head.
    // Prefer GPU when hidden state already resides there (zero transfer cost).
    // Fall back to CPU lm_head when last layers ran on CPU (model too large for VRAM).
    match hs {
        HiddenState::Gpu(x) => {
            let out_norm = wb.get("output_norm.weight")?;
            let x_final = engine.rmsnorm(&x, out_norm, 1, h as u32, cfg.eps).ok()?;
            let lm_head = wb.get("output.weight").or_else(|| wb.get("token_embd.weight"))?;
            let logits_gpu = engine.matmul(lm_head, &x_final, cfg.vocab as u32, h as u32, 1).ok()?;
            engine.download_f32(&logits_gpu).ok()
        }
        HiddenState::Cpu(x) => {
            let out_norm = staging_wb.get("output_norm.weight")
                .map(|b| b.as_f32_slice().to_vec())
                .unwrap_or_else(|| vec![1.0f32; h]);
            let x_normed = cpu_rmsnorm(&x, &out_norm, cfg.eps);
            let lm_b = staging_wb.get("output.weight")
                .or_else(|| staging_wb.get("token_embd.weight"))?;
            Some(wavefront_matvec(lm_b.as_f32_slice(), &x_normed, cfg.vocab, h))
        }
    }
}

// ─── Helpers para batched forward ────────────────────────────────────────────

/// Transpõe [rows × cols] → [cols × rows] no CPU.
#[inline]
fn transpose_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// GQA attention com K/V de três fontes: cache persistente, batch anterior, token atual.
fn gqa_attention_combined(
    q: &[f32],
    k_persist: &[Vec<f32>],
    k_batch: &[Vec<f32>],
    v_persist: &[Vec<f32>],
    v_batch: &[Vec<f32>],
    k_cur: &[f32],
    v_cur: &[f32],
    nq: usize,
    group: usize,
    hd: usize,
    scale: f32,
) -> Vec<f32> {
    let base = k_persist.len();
    let batch_prev = k_batch.len();
    let klen = base + batch_prev + 1;
    let mut out = vec![0.0f32; nq * hd];
    for head in 0..nq {
        let kv_h = head / group;
        let qh = &q[head * hd..(head + 1) * hd];
        let mut scores = Vec::with_capacity(klen);
        for p in 0..base {
            let kp = &k_persist[p][kv_h * hd..(kv_h + 1) * hd];
            scores.push(qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        for p in 0..batch_prev {
            let kp = &k_batch[p][kv_h * hd..(kv_h + 1) * hd];
            scores.push(qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        {
            let kp = &k_cur[kv_h * hd..(kv_h + 1) * hd];
            scores.push(qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max_s).exp(); *s }).sum();
        if sum > 0.0 { scores.iter_mut().for_each(|s| *s /= sum); }
        for d in 0..hd {
            let mut acc = 0.0f32;
            for p in 0..base    { acc += scores[p]              * v_persist[p][kv_h * hd + d]; }
            for p in 0..batch_prev { acc += scores[base + p]   * v_batch[p][kv_h * hd + d]; }
            acc += scores[base + batch_prev] * v_cur[kv_h * hd + d];
            out[head * hd + d] = acc;
        }
    }
    out
}

// ─── Forward batched (especulação real: B tokens, 1 weight-load) ──────────────

/// Processa B tokens em paralelo numa única passagem pelo transformer.
/// Layout GPU: [B × h] row-major para rmsnorm; transpõe para [h × B] antes de matmul.
/// Retorna até B vetores de logits (vec vazio = falha total, usa fallback sequencial).
pub fn gpu_forward_batch(
    tokens: &[u32],
    start_pos: usize,
    cfg: &CpuModelConfig,
    engine: &VulkanEngine,
    wb: &WeightBank,
    staging_wb: &WeightBank,
    kv: &mut CpuKvCache,
    steering: Option<(&[f32], f32, f32)>,
) -> Vec<Vec<f32>> {
    let b = tokens.len();
    if b == 0 { return vec![]; }
    if b == 1 {
        return match gpu_forward_step(tokens[0], start_pos, cfg, engine, wb, staging_wb, kv, steering) {
            Some(l) => vec![l],
            None    => vec![],
        };
    }

    let h    = cfg.hidden;
    let hd   = cfg.head_dim;
    let nq   = cfg.n_heads;
    let nkv  = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let inter  = cfg.intermediate;
    let scale  = 1.0 / (hd as f32).sqrt();

    // ── 1. Embedding lookup: x_batch [B × h] ──────────────────────────────────
    // Guard: all compute layers must be in DEVICE_LOCAL before batch forward.
    let all_compute_on_gpu = (0..cfg.n_layers).all(|l| {
        wb.get(&format!("blk.{l}.attn_q.weight")).map_or(false, |t| t.is_on_gpu())
    });
    if !all_compute_on_gpu { return vec![]; }
    // Read embeddings from staging (HOST_VISIBLE) — GpuOnly buffers return &[] from as_f32_slice().
    let mut x_data = vec![0.0f32; b * h];
    {
        let tok_embd_cpu = match staging_wb.get("token_embd.weight") { Some(e) => e, None => return vec![] };
        let embd = tok_embd_cpu.as_f32_slice();
        for (bi, &tok) in tokens.iter().enumerate() {
            let s = tok as usize * h;
            if s + h > embd.len() { return vec![]; }
            x_data[bi * h..(bi + 1) * h].copy_from_slice(&embd[s..s + h]);
        }
    }
    let mut x = match engine.upload_f32(&x_data) { Ok(b) => b, Err(_) => return vec![] };

    // Macros auxiliares com escopo na função (macro_rules! é visível daqui em diante)
    macro_rules! wb_get {
        ($key:expr) => {
            match wb.get(&$key) { Some(w) => w, None => return vec![] }
        };
    }
    macro_rules! gpu {
        ($e:expr) => {
            match $e { Ok(v) => v, Err(_) => return vec![] }
        };
    }

    // ── 2. Camadas transformer ─────────────────────────────────────────────────
    for layer in 0..cfg.n_layers {

        // — Attention norm: rmsnorm normaliza B linhas de h elementos cada —
        let attn_norm = wb_get!(format!("blk.{layer}.attn_norm.weight"));
        let x_norm = gpu!(engine.rmsnorm(&x, attn_norm, b as u32, h as u32, cfg.eps));

        // Download x_norm [B × h], transpor para [h × B] para matmul
        let xn_data = match engine.download_f32(&x_norm) {
            Ok(d) if d.len() == b * h => d,
            _ => return vec![],
        };
        let xn_T = transpose_f32(&xn_data, b, h);
        let xn_T_gpu = gpu!(engine.upload_f32(&xn_T));

        // — Q/K/V: W[out × h] × xn_T[h × B] = out[out × B] —
        let wq = wb_get!(format!("blk.{layer}.attn_q.weight"));
        let wk = wb_get!(format!("blk.{layer}.attn_k.weight"));
        let wv = wb_get!(format!("blk.{layer}.attn_v.weight"));

        // matmul_to_host: output HOST_VISIBLE, lê direto sem staging
        let q_buf = gpu!(engine.matmul_to_host(wq, &xn_T_gpu, q_dim as u32, h as u32, b as u32));
        let k_buf = gpu!(engine.matmul_to_host(wk, &xn_T_gpu, kv_dim as u32, h as u32, b as u32));
        let v_buf = gpu!(engine.matmul_to_host(wv, &xn_T_gpu, kv_dim as u32, h as u32, b as u32));

        let q_raw = q_buf.as_f32_slice();
        let k_raw = k_buf.as_f32_slice();
        let v_raw = v_buf.as_f32_slice();
        if q_raw.len() != q_dim * b || k_raw.len() != kv_dim * b || v_raw.len() != kv_dim * b {
            return vec![];
        }

        // Transpor [out × B] → [B × out] para attention
        let mut q_BT = transpose_f32(q_raw, q_dim, b);  // [B × q_dim]
        let mut k_BT = transpose_f32(k_raw, kv_dim, b); // [B × kv_dim]
        let mut v_BT = transpose_f32(v_raw, kv_dim, b); // [B × kv_dim]

        // Bias Q/K/V (aplicado por token)
        if let Some(bq) = wb.get(&format!("blk.{layer}.attn_q.bias")) {
            let permuted = bias_neox_to_interleaved(bq.as_f32_slice(), nq, hd);
            for bi in 0..b {
                let s = &mut q_BT[bi * q_dim..(bi + 1) * q_dim];
                for (xi, &pi) in s.iter_mut().zip(permuted.iter()) { *xi += pi; }
            }
        }
        if let Some(bk) = wb.get(&format!("blk.{layer}.attn_k.bias")) {
            let permuted = bias_neox_to_interleaved(bk.as_f32_slice(), nkv, hd);
            for bi in 0..b {
                let s = &mut k_BT[bi * kv_dim..(bi + 1) * kv_dim];
                for (xi, &pi) in s.iter_mut().zip(permuted.iter()) { *xi += pi; }
            }
        }
        if let Some(bv) = wb.get(&format!("blk.{layer}.attn_v.bias")) {
            let bs = bv.as_f32_slice();
            for bi in 0..b {
                let s = &mut v_BT[bi * kv_dim..(bi + 1) * kv_dim];
                for (xi, &pi) in s.iter_mut().zip(bs.iter()) { *xi += pi; }
            }
        }

        // RoPE por token na sua posição
        for bi in 0..b {
            rope_neox(&mut q_BT[bi * q_dim..(bi + 1) * q_dim],   nq,  hd, start_pos + bi, cfg.rope_base);
            rope_neox(&mut k_BT[bi * kv_dim..(bi + 1) * kv_dim], nkv, hd, start_pos + bi, cfg.rope_base);
        }

        // — GQA attention causal com KV combinado —
        // k_BT[bi] pode atender a: cache persistente + batch anterior (0..bi) + si mesmo
        let base_klen = { let (ck, _) = kv.layer_kv(layer); ck.len() };
        let mut batch_k: Vec<Vec<f32>> = Vec::with_capacity(b);
        let mut batch_v: Vec<Vec<f32>> = Vec::with_capacity(b);
        let mut attn_out = vec![0.0f32; b * q_dim];

        for bi in 0..b {
            let k_i = k_BT[bi * kv_dim..(bi + 1) * kv_dim].to_vec();
            let v_i = v_BT[bi * kv_dim..(bi + 1) * kv_dim].to_vec();
            let q_i = &q_BT[bi * q_dim..(bi + 1) * q_dim];
            let (cached_k, cached_v) = kv.layer_kv(layer);
            let a = gqa_attention_combined(
                q_i,
                cached_k, &batch_k,
                cached_v, &batch_v,
                &k_i, &v_i,
                nq, group, hd, scale,
            );
            attn_out[bi * q_dim..(bi + 1) * q_dim].copy_from_slice(&a);
            batch_k.push(k_i);
            batch_v.push(v_i);
        }
        // Flush K/V do batch para o cache persistente
        for bi in 0..b {
            kv.push_kv(layer, batch_k[bi].clone(), batch_v[bi].clone());
        }

        // — Output projection: wo[h × q_dim] × attn_T[q_dim × B] = attn_proj[h × B] —
        // attn_out é [B × q_dim] → transpor para [q_dim × B]
        let attn_T = transpose_f32(&attn_out, b, q_dim);
        let attn_T_gpu = gpu!(engine.upload_f32(&attn_T));
        let wo = wb_get!(format!("blk.{layer}.attn_output.weight"));
        // matmul_to_host: attn_proj[h × B] HOST_VISIBLE
        let attn_proj_buf = gpu!(engine.matmul_to_host(wo, &attn_T_gpu, h as u32, q_dim as u32, b as u32));
        let attn_proj_raw = attn_proj_buf.as_f32_slice();
        if attn_proj_raw.len() != h * b { return vec![]; }

        // Transpor attn_proj [h × B] → [B × h] para residual add
        let mut attn_proj_bh = transpose_f32(attn_proj_raw, h, b);

        // Steering pós-atenção: aplicado por token no CPU
        if let Some((dir, intensity, d_sq)) = steering {
            if d_sq > 1e-12 {
                // Download x [B × h], somar com attn_proj, aplicar POD, re-upload
                let x_dl = match engine.download_f32(&x) {
                    Ok(d) if d.len() == b * h => d,
                    _ => return vec![],
                };
                let mut x_new = vec![0.0f32; b * h];
                let n_dir = h.min(dir.len());
                for bi in 0..b {
                    for j in 0..h { x_new[bi * h + j] = x_dl[bi * h + j] + attn_proj_bh[bi * h + j]; }
                    let slice = &mut x_new[bi * h..bi * h + n_dir];
                    let h_dot: f32 = slice.iter().zip(dir.iter()).map(|(a, d)| a * d).sum();
                    let sc = intensity * h_dot / d_sq;
                    for i in 0..n_dir { slice[i] -= sc * dir[i]; }
                }
                x = gpu!(engine.upload_f32(&x_new));
            } else {
                let ap_gpu = gpu!(engine.upload_f32(&attn_proj_bh));
                x = gpu!(engine.add(&x, &ap_gpu, (h * b) as u32));
            }
        } else {
            let ap_gpu = gpu!(engine.upload_f32(&attn_proj_bh));
            x = gpu!(engine.add(&x, &ap_gpu, (h * b) as u32));
        }

        // — FFN norm —
        let ffn_norm = wb_get!(format!("blk.{layer}.ffn_norm.weight"));
        let x_norm2 = gpu!(engine.rmsnorm(&x, ffn_norm, b as u32, h as u32, cfg.eps));
        let xn2_data = match engine.download_f32(&x_norm2) {
            Ok(d) if d.len() == b * h => d,
            _ => return vec![],
        };
        let xn2_T = transpose_f32(&xn2_data, b, h);
        let xn2_T_gpu = gpu!(engine.upload_f32(&xn2_T));

        // — SwiGLU: W[inter × h] × xn2_T[h × B] = [inter × B] —
        let wgate = wb_get!(format!("blk.{layer}.ffn_gate.weight"));
        let wup   = wb_get!(format!("blk.{layer}.ffn_up.weight"));
        let wdown = wb_get!(format!("blk.{layer}.ffn_down.weight"));

        let gate     = gpu!(engine.matmul(wgate, &xn2_T_gpu, inter as u32, h as u32, b as u32));
        let up       = gpu!(engine.matmul(wup,   &xn2_T_gpu, inter as u32, h as u32, b as u32));
        let gate_act = gpu!(engine.silu(&gate, (inter * b) as u32));
        let swiglu   = gpu!(engine.mul(&gate_act, &up, (inter * b) as u32));
        // wdown[h × inter] × swiglu[inter × B] = ffn_out[h × B]
        let ffn_buf  = gpu!(engine.matmul_to_host(wdown, &swiglu, h as u32, inter as u32, b as u32));
        let ffn_raw  = ffn_buf.as_f32_slice();
        if ffn_raw.len() != h * b { return vec![]; }

        // ffn_out [h × B] → [B × h] para residual
        let ffn_bh = transpose_f32(ffn_raw, h, b);

        // Steering pós-FFN + residual
        if let Some((dir, intensity, d_sq)) = steering {
            if d_sq > 1e-12 {
                let x_dl = match engine.download_f32(&x) {
                    Ok(d) if d.len() == b * h => d,
                    _ => return vec![],
                };
                let mut x_new = vec![0.0f32; b * h];
                let n_dir = h.min(dir.len());
                for bi in 0..b {
                    for j in 0..h { x_new[bi * h + j] = x_dl[bi * h + j] + ffn_bh[bi * h + j]; }
                    let slice = &mut x_new[bi * h..bi * h + n_dir];
                    let h_dot: f32 = slice.iter().zip(dir.iter()).map(|(a, d)| a * d).sum();
                    let sc = intensity * h_dot / d_sq;
                    for i in 0..n_dir { slice[i] -= sc * dir[i]; }
                }
                x = gpu!(engine.upload_f32(&x_new));
            } else {
                let ffn_gpu2 = gpu!(engine.upload_f32(&ffn_bh));
                x = gpu!(engine.add(&x, &ffn_gpu2, (h * b) as u32));
            }
        } else {
            let ffn_gpu2 = gpu!(engine.upload_f32(&ffn_bh));
            x = gpu!(engine.add(&x, &ffn_gpu2, (h * b) as u32));
        }
    }

    // ── 3. Final norm + LM head ────────────────────────────────────────────────
    let out_norm = match wb.get("output_norm.weight") { Some(w) => w, None => return vec![] };
    let x_final = gpu!(engine.rmsnorm(&x, out_norm, b as u32, h as u32, cfg.eps));
    let xf_data = match engine.download_f32(&x_final) {
        Ok(d) if d.len() == b * h => d,
        _ => return vec![],
    };
    let xf_T = transpose_f32(&xf_data, b, h);
    let xf_T_gpu = match engine.upload_f32(&xf_T) { Ok(b) => b, Err(_) => return vec![] };

    let lm_head = match wb.get("output.weight").or_else(|| wb.get("token_embd.weight")) {
        Some(w) => w,
        None    => return vec![],
    };
    // logits [vocab × B]
    let logits_buf = match engine.matmul_to_host(lm_head, &xf_T_gpu, cfg.vocab as u32, h as u32, b as u32) {
        Ok(buf) => buf,
        Err(_)  => return vec![],
    };
    let logits_raw = logits_buf.as_f32_slice();
    if logits_raw.len() != cfg.vocab * b { return vec![]; }

    // [vocab × B] → B vetores de [vocab]
    let logits_BV = transpose_f32(logits_raw, cfg.vocab, b);
    (0..b).map(|bi| {
        let slice = &logits_BV[bi * cfg.vocab..(bi + 1) * cfg.vocab];
        // Clamping NaN/Inf: previne colapso semântico após overflow numérico em camadas deep.
        // Substitui NaN → -1e9 (token effetivamente impossível) e Inf → ±30.0 (range safe).
        slice.iter().map(|&v| {
            if v.is_nan()       { -1e9_f32 }
            else if v > 30.0    { 30.0_f32 }
            else if v < -30.0   { -30.0_f32 }
            else                { v }
        }).collect()
    }).collect()
}

// ─── Forward verify (especulação em lote) ────────────────────────────────────

/// Verifica N rascunhos especulativos usando GPU.
/// Tenta `gpu_forward_batch` (1 weight-load para N tokens); fallback sequencial se falhar.
/// `steering`: `Some((direction, intensity, d_sq))` para POD após cada residual.
pub fn gpu_forward_verify(
    drafts: &[u32],
    start_pos: usize,
    cfg: &CpuModelConfig,
    engine: &VulkanEngine,
    wb: &WeightBank,
    staging_wb: &WeightBank,
    kv: &mut CpuKvCache,
    steering: Option<(&[f32], f32, f32)>,
) -> Vec<Vec<f32>> {
    if drafts.is_empty() { return vec![]; }

    // Tenta batched: 1 weight-load para len(drafts) tokens
    let batched = gpu_forward_batch(drafts, start_pos, cfg, engine, wb, staging_wb, kv, steering);
    if !batched.is_empty() { return batched; }

    // Fallback sequencial (rollback do KV cache se batch inseriu parcialmente)
    kv.truncate(start_pos);
    let mut out = Vec::with_capacity(drafts.len());
    for (i, &d) in drafts.iter().enumerate() {
        match gpu_forward_step(d, start_pos + i, cfg, engine, wb, staging_wb, kv, steering) {
            Some(logits) => out.push(logits),
            None => break,
        }
    }
    out
}
