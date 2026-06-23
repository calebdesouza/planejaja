//! Forward de REFERÊNCIA em CPU — inferência Llama numericamente correta.
//!
//! Objetivo: COERÊNCIA. Implementa o forward autoregressivo completo com as
//! convenções exatas do GGUF/llama.cpp para que modelos reais produzam TEXTO
//! coerente:
//! - Atenção causal multi-head com KV de TODO o contexto (não seq_len=1).
//! - GQA (n_kv_heads < n_heads).
//! - RoPE estilo NeoX (pares i ↔ i+head_dim/2), não interleaved.
//! - Layout de pesos GGUF `[out, in]` (row-major): y[o] = Σ_i W[o·in+i]·x[i].
//! - RMSNorm pré-norma, SwiGLU (SiLU(gate)·up), residuais.
//!
//! É o caminho de CORREÇÃO (CPU). A velocidade extrema vem depois, do compute
//! GPU; a especulação (COBER) multiplica por cima — ambos exigem este forward
//! correto como base.

use nodestor_vulkan::WeightBank;

/// Dimensões e hiperparâmetros do modelo (extraídos do GGUF/GraphInterpreter).
pub struct CpuModelConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rope_base: f32,
    pub eps: f32,
}

#[inline]
fn weight<'a>(wb: &'a WeightBank, key: &str) -> Option<&'a [f32]> {
    wb.get(key).map(|b| b.as_f32_slice())
}

/// y[o] = Σ_i W[o·in_dim + i] · x[i]   (W em layout GGUF [out_dim, in_dim]).
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let base = o * in_dim;
        if base + in_dim > w.len() { break; }
        let row = &w[base..base + in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// RMSNorm: y[i] = x[i] / sqrt(mean(x²) + eps) · w[i].
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x { ss += v * v; }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let wi = w.get(i).copied().unwrap_or(1.0);
        y[i] = x[i] * inv * wi;
    }
    y
}

#[inline]
fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

/// Softmax estável in-place.
fn softmax(s: &mut [f32]) {
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in s.iter_mut() { *v = (*v - m).exp(); sum += *v; }
    if sum > 0.0 { for v in s.iter_mut() { *v /= sum; } }
}

/// RoPE estilo INTERLEAVED (llama.cpp `NORM`) aplicado in-place a [n_heads · head_dim].
/// Modelos Llama convertidos para GGUF têm Q/K PERMUTADOS para esta convenção, em que
/// pares ADJACENTES (2i, 2i+1) são rotacionados por θ = pos · base^(-2i/hd):
///   x'[2i]   = x[2i]·cos − x[2i+1]·sin
///   x'[2i+1] = x[2i]·sin + x[2i+1]·cos
fn rope_neox(vec: &mut [f32], n_heads: usize, hd: usize, pos: usize, base: f32) {
    let half = hd / 2;
    for h in 0..n_heads {
        let off = h * hd;
        for i in 0..half {
            let freq = base.powf(-2.0 * i as f32 / hd as f32);
            let ang = pos as f32 * freq;
            let (sin, cos) = ang.sin_cos();
            let a = vec[off + 2 * i];
            let b = vec[off + 2 * i + 1];
            vec[off + 2 * i] = a * cos - b * sin;
            vec[off + 2 * i + 1] = a * sin + b * cos;
        }
    }
}

/// Forward autoregressivo correto sobre toda a sequência `tokens`.
/// Retorna os logits [vocab] da ÚLTIMA posição (predição do próximo token).
pub fn forward_last_logits(tokens: &[u32], cfg: &CpuModelConfig, wb: &WeightBank) -> Option<Vec<f32>> {
    let seq = tokens.len();
    if seq == 0 { return None; }
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nq = cfg.n_heads;
    let nkv = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    // Embeddings: x[t] = linha do token em token_embd.weight [vocab, hidden].
    let tok_embd = weight(wb, "token_embd.weight")?;
    let mut x: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
        let s = (tok as usize) * h;
        tok_embd.get(s..s + h).map(|r| r.to_vec()).unwrap_or_else(|| vec![0.0; h])
    }).collect();

    for layer in 0..cfg.n_layers {
        let attn_norm = weight(wb, &format!("blk.{}.attn_norm.weight", layer))?;
        let wq = weight(wb, &format!("blk.{}.attn_q.weight", layer))?;
        let wk = weight(wb, &format!("blk.{}.attn_k.weight", layer))?;
        let wv = weight(wb, &format!("blk.{}.attn_v.weight", layer))?;
        let wo = weight(wb, &format!("blk.{}.attn_output.weight", layer))?;
        let ffn_norm = weight(wb, &format!("blk.{}.ffn_norm.weight", layer))?;
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer))?;
        let wup = weight(wb, &format!("blk.{}.ffn_up.weight", layer))?;
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer))?;

        // Q, K, V (com RoPE NeoX) por posição.
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(seq);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(seq);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for t in 0..seq {
            let normed = rmsnorm(&x[t], attn_norm, cfg.eps);
            let mut q = matvec(wq, &normed, q_dim, h);
            let mut k = matvec(wk, &normed, kv_dim, h);
            let v = matvec(wv, &normed, kv_dim, h);
            rope_neox(&mut q, nq, hd, t, cfg.rope_base);
            rope_neox(&mut k, nkv, hd, t, cfg.rope_base);
            qs.push(q); ks.push(k); vs.push(v);
        }

        // Atenção causal multi-head com GQA.
        let mut attn = vec![vec![0.0f32; q_dim]; seq];
        for t in 0..seq {
            for head in 0..nq {
                let kvh = head / group;
                let mut scores = vec![0.0f32; t + 1];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qs[t][head * hd + d] * ks[s][kvh * hd + d];
                    }
                    scores[s] = dot * scale;
                }
                softmax(&mut scores);
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for s in 0..=t {
                        acc += scores[s] * vs[s][kvh * hd + d];
                    }
                    attn[t][head * hd + d] = acc;
                }
            }
        }

        // Output proj + residual; depois FFN (SwiGLU) + residual.
        for t in 0..seq {
            let proj = matvec(wo, &attn[t], h, q_dim);
            for i in 0..h { x[t][i] += proj[i]; }

            let normed2 = rmsnorm(&x[t], ffn_norm, cfg.eps);
            let gate = matvec(wgate, &normed2, cfg.intermediate, h);
            let up = matvec(wup, &normed2, cfg.intermediate, h);
            let mut swiglu = vec![0.0f32; cfg.intermediate];
            for i in 0..cfg.intermediate { swiglu[i] = silu(gate[i]) * up[i]; }
            let down = matvec(wdown, &swiglu, h, cfg.intermediate);
            for i in 0..h { x[t][i] += down[i]; }
        }
    }

    // Norma final + LM head na ÚLTIMA posição. `output.weight` ausente ⇒ pesos
    // amarrados (tied) ao token_embd.
    let final_norm = weight(wb, "output_norm.weight")?;
    let normed = rmsnorm(&x[seq - 1], final_norm, cfg.eps);
    let lm_head = weight(wb, "output.weight").unwrap_or(tok_embd);
    Some(matvec(lm_head, &normed, cfg.vocab, h))
}
