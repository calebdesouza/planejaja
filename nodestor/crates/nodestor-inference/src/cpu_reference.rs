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

/// KV cache em CPU: por camada, K e V acumulados (uma entrada por posição já vista).
/// Permite o forward INCREMENTAL — processa só o token novo e atende sobre todo o
/// histórico cacheado, em vez de recomputar a sequência inteira a cada passo.
pub struct CpuKvCache {
    k: Vec<Vec<Vec<f32>>>, // [layer][pos][kv_dim]
    v: Vec<Vec<Vec<f32>>>,
}

impl CpuKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers] }
    }
    /// Número de posições já cacheadas (idêntico em todas as camadas).
    pub fn len(&self) -> usize { self.k.first().map(|l| l.len()).unwrap_or(0) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Reverte o cache para `len` posições (rollback de rascunhos REJEITADOS na
    /// decodificação especulativa). Mantém o histórico aceito intacto.
    pub fn truncate(&mut self, len: usize) {
        for l in self.k.iter_mut() { l.truncate(len); }
        for l in self.v.iter_mut() { l.truncate(len); }
    }
}

/// Índice do maior elemento (argmax) — escolha greedy (lossless por construção).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best as u32
}

/// Forward INCREMENTAL de UM token na posição `pos`, usando/atualizando o KV cache.
/// Retorna os logits [vocab] para predizer o PRÓXIMO token. Custo O(seq) por passo
/// (atende ao histórico cacheado) em vez de O(seq²) do recompute total.
fn step_impl(token: u32, pos: usize, cfg: &CpuModelConfig, wb: &WeightBank, cache: &mut CpuKvCache, n_run: usize) -> Option<Vec<f32>> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nq = cfg.n_heads;
    let nkv = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let tok_embd = weight(wb, "token_embd.weight")?;
    let s = (token as usize) * h;
    let mut x: Vec<f32> = tok_embd.get(s..s + h).map(|r| r.to_vec()).unwrap_or_else(|| vec![0.0; h]);

    for layer in 0..n_run.min(cfg.n_layers) {
        let attn_norm = weight(wb, &format!("blk.{}.attn_norm.weight", layer))?;
        let wq = weight(wb, &format!("blk.{}.attn_q.weight", layer))?;
        let wk = weight(wb, &format!("blk.{}.attn_k.weight", layer))?;
        let wv = weight(wb, &format!("blk.{}.attn_v.weight", layer))?;
        let wo = weight(wb, &format!("blk.{}.attn_output.weight", layer))?;
        let ffn_norm = weight(wb, &format!("blk.{}.ffn_norm.weight", layer))?;
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer))?;
        let wup = weight(wb, &format!("blk.{}.ffn_up.weight", layer))?;
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer))?;

        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let vv = matvec(wv, &normed, kv_dim, h);
        rope_neox(&mut q, nq, hd, pos, cfg.rope_base);
        rope_neox(&mut k, nkv, hd, pos, cfg.rope_base);

        // Anexa K,V desta posição ao cache da camada.
        cache.k[layer].push(k);
        cache.v[layer].push(vv);
        let klen = cache.k[layer].len();

        // Atenção: Q atual atende a TODAS as posições cacheadas (causal por construção).
        let mut attn_out = vec![0.0f32; q_dim];
        for head in 0..nq {
            let kvh = head / group;
            let mut scores = vec![0.0f32; klen];
            for sp in 0..klen {
                let kc = &cache.k[layer][sp];
                let mut dot = 0.0f32;
                for d in 0..hd { dot += q[head * hd + d] * kc[kvh * hd + d]; }
                scores[sp] = dot * scale;
            }
            softmax(&mut scores);
            for d in 0..hd {
                let mut acc = 0.0f32;
                for sp in 0..klen { acc += scores[sp] * cache.v[layer][sp][kvh * hd + d]; }
                attn_out[head * hd + d] = acc;
            }
        }

        let proj = matvec(wo, &attn_out, h, q_dim);
        for i in 0..h { x[i] += proj[i]; }

        let normed2 = rmsnorm(&x, ffn_norm, cfg.eps);
        let gate = matvec(wgate, &normed2, cfg.intermediate, h);
        let up = matvec(wup, &normed2, cfg.intermediate, h);
        let mut swiglu = vec![0.0f32; cfg.intermediate];
        for i in 0..cfg.intermediate { swiglu[i] = silu(gate[i]) * up[i]; }
        let down = matvec(wdown, &swiglu, h, cfg.intermediate);
        for i in 0..h { x[i] += down[i]; }
    }

    let final_norm = weight(wb, "output_norm.weight")?;
    let normed = rmsnorm(&x, final_norm, cfg.eps);
    let lm_head = weight(wb, "output.weight").unwrap_or(tok_embd);
    Some(matvec(lm_head, &normed, cfg.vocab, h))
}

/// Forward incremental COMPLETO (todas as camadas) — o modelo ALVO.
pub fn forward_step(token: u32, pos: usize, cfg: &CpuModelConfig, wb: &WeightBank, cache: &mut CpuKvCache) -> Option<Vec<f32>> {
    step_impl(token, pos, cfg, wb, cache, cfg.n_layers)
}

/// Forward incremental PARCIAL (primeiras `n_run` camadas + norm + lm_head) — o
/// RASCUNHADOR self-speculative (early-exit): o próprio modelo, truncado, prevê o
/// futuro de forma barata. Sem pesos extras nem treino; usa as features do modelo.
pub fn forward_step_partial(token: u32, pos: usize, cfg: &CpuModelConfig, wb: &WeightBank, cache: &mut CpuKvCache, n_run: usize) -> Option<Vec<f32>> {
    step_impl(token, pos, cfg, wb, cache, n_run.max(1).min(cfg.n_layers))
}

/// VERIFICAÇÃO BATCHED de K rascunhos (especulação SEM cabeça / n-gram).
///
/// Processa cada rascunho `drafts[i]` na posição `start_pos+i`, ESTENDENDO o cache,
/// e retorna os K vetores de logits (logits[i] prediz a posição start_pos+i+1).
///
/// Na GPU isto é UM ÚNICO forward pass com tree-attention causal — carrega os pesos
/// do modelo da VRAM UMA vez e avalia os K tokens em paralelo. É exatamente daí que
/// vem "1 weight-load = K tokens" (a vitória contra o gargalo de banda). Na CPU
/// (compute-bound) é a soma de K passos; o chamador aceita o prefixo concordante e
/// faz rollback (`cache.truncate`) dos rejeitados.
pub fn forward_verify(drafts: &[u32], start_pos: usize, cfg: &CpuModelConfig, wb: &WeightBank, cache: &mut CpuKvCache) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(drafts.len());
    for (i, &d) in drafts.iter().enumerate() {
        match forward_step(d, start_pos + i, cfg, wb, cache) {
            Some(l) => out.push(l),
            None => break,
        }
    }
    out
}

// ─── Projeção Ortogonal Dinâmica no Loop de Inferência ───────────────────────

/// Forward incremental com Projeção Ortogonal Dinâmica (POD) camada por camada.
///
/// Após cada bloco Attention+MLP completo, projeta o hidden state do stream
/// residual **fora** da direção especificada antes de passar para a próxima camada:
///
///   h_clean = h − intensity · (⟨h, d⟩ / ⟨d, d⟩) · d
///
/// onde:
///   - `h`          = hidden state pós-FFN (stream residual na saída da camada L)
///   - `d`          = direction vector no espaço hidden_dim (normalizado ou não)
///   - `⟨h, d⟩`     = produto interno entre h e d
///   - `⟨d, d⟩`     = ‖d‖² (corrige a escala de d automaticamente)
///   - `intensity`  = escala da intervenção (1.0 = projeção pura; 0.0 = passivo)
///
/// A operação é **lossless no espaço ortogonal**: componentes de h perpendiculares
/// a d são preservadas com fidelidade numérica total (FP32 acumulado).
pub fn forward_step_with_steering(
    token: u32,
    pos: usize,
    cfg: &CpuModelConfig,
    wb: &WeightBank,
    cache: &mut CpuKvCache,
    direction: &[f32],
    intensity: f32,
) -> Option<Vec<f32>> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nq = cfg.n_heads;
    let nkv = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let tok_embd = weight(wb, "token_embd.weight")?;
    let s = (token as usize) * h;
    let mut x: Vec<f32> = tok_embd.get(s..s + h).map(|r| r.to_vec()).unwrap_or_else(|| vec![0.0; h]);

    // Pré-computa ‖d‖² (constante entre camadas — só precisa ser calculado uma vez)
    let n_dir = h.min(direction.len());
    let d_sq: f32 = direction[..n_dir].iter().map(|&v| v * v).sum();
    let apply_steering = d_sq > 1e-12 && intensity.abs() > 1e-9;

    for layer in 0..cfg.n_layers {
        let attn_norm = weight(wb, &format!("blk.{}.attn_norm.weight", layer))?;
        let wq  = weight(wb, &format!("blk.{}.attn_q.weight", layer))?;
        let wk  = weight(wb, &format!("blk.{}.attn_k.weight", layer))?;
        let wv  = weight(wb, &format!("blk.{}.attn_v.weight", layer))?;
        let wo  = weight(wb, &format!("blk.{}.attn_output.weight", layer))?;
        let ffn_norm = weight(wb, &format!("blk.{}.ffn_norm.weight", layer))?;
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer))?;
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer))?;
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer))?;

        // ── Atenção ──────────────────────────────────────────────────────────
        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let vv    = matvec(wv, &normed, kv_dim, h);
        rope_neox(&mut q, nq, hd, pos, cfg.rope_base);
        rope_neox(&mut k, nkv, hd, pos, cfg.rope_base);

        cache.k[layer].push(k);
        cache.v[layer].push(vv);
        let klen = cache.k[layer].len();

        let mut attn_out = vec![0.0f32; q_dim];
        for head in 0..nq {
            let kvh = head / group;
            let mut scores = vec![0.0f32; klen];
            for sp in 0..klen {
                let kc = &cache.k[layer][sp];
                let mut dot = 0.0f32;
                for d in 0..hd { dot += q[head * hd + d] * kc[kvh * hd + d]; }
                scores[sp] = dot * scale;
            }
            softmax(&mut scores);
            for d in 0..hd {
                let mut acc = 0.0f32;
                for sp in 0..klen { acc += scores[sp] * cache.v[layer][sp][kvh * hd + d]; }
                attn_out[head * hd + d] = acc;
            }
        }
        let proj = matvec(wo, &attn_out, h, q_dim);
        for i in 0..h { x[i] += proj[i]; }

        // ── FFN SwiGLU ───────────────────────────────────────────────────────
        let normed2 = rmsnorm(&x, ffn_norm, cfg.eps);
        let gate = matvec(wgate, &normed2, cfg.intermediate, h);
        let up   = matvec(wup,   &normed2, cfg.intermediate, h);
        let mut swiglu = vec![0.0f32; cfg.intermediate];
        for i in 0..cfg.intermediate { swiglu[i] = silu(gate[i]) * up[i]; }
        let down = matvec(wdown, &swiglu, h, cfg.intermediate);
        for i in 0..h { x[i] += down[i]; }

        // ── Projeção Ortogonal Dinâmica (POD) ────────────────────────────────
        // INTERCEPÇÃO após bloco Attn+MLP completo (pós-residual), antes da
        // próxima camada. O stream residual x é modificado in-place:
        //   x ← x − intensity · (⟨x, d⟩ / ⟨d, d⟩) · d
        if apply_steering {
            let h_dot_d: f32 = x[..n_dir].iter()
                .zip(direction[..n_dir].iter())
                .map(|(&xi, &di)| xi * di)
                .sum();
            let proj_scale = intensity * h_dot_d / d_sq;
            for i in 0..n_dir { x[i] -= proj_scale * direction[i]; }
        }
    }

    let final_norm = weight(wb, "output_norm.weight")?;
    let normed = rmsnorm(&x, final_norm, cfg.eps);
    let lm_head = weight(wb, "output.weight").unwrap_or(tok_embd);
    Some(matvec(lm_head, &normed, cfg.vocab, h))
}

/// Verificação batched de K rascunhos com POD aplicada em cada passo.
/// Equivalente a `forward_verify` mas usando `forward_step_with_steering`.
pub fn forward_verify_with_steering(
    drafts: &[u32],
    start_pos: usize,
    cfg: &CpuModelConfig,
    wb: &WeightBank,
    cache: &mut CpuKvCache,
    direction: &[f32],
    intensity: f32,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(drafts.len());
    for (i, &d) in drafts.iter().enumerate() {
        match forward_step_with_steering(d, start_pos + i, cfg, wb, cache, direction, intensity) {
            Some(l) => out.push(l),
            None => break,
        }
    }
    out
}

/// Extrai o hidden state normalizado da última camada (após RMSNorm final, antes do LM Head).
///
/// Retorna um vetor de dimensão `hidden_dim`, representando o estado interno do modelo
/// para o token de entrada na posição `pos`. Usado exclusivamente para **calibração**:
/// coletar representações de amostras positivas e negativas para calcular a direção
/// contrastiva via `refusal_mapper::calibrate_direction_from_hidden_states`.
///
/// O vetor retornado está no mesmo espaço que `direction` em `forward_step_with_steering`.
pub fn extract_final_hidden(
    token: u32,
    pos: usize,
    cfg: &CpuModelConfig,
    wb: &WeightBank,
    cache: &mut CpuKvCache,
) -> Option<Vec<f32>> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nq = cfg.n_heads;
    let nkv = cfg.n_kv_heads.max(1);
    let group = (nq / nkv).max(1);
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    let tok_embd = weight(wb, "token_embd.weight")?;
    let s = (token as usize) * h;
    let mut x: Vec<f32> = tok_embd.get(s..s + h).map(|r| r.to_vec()).unwrap_or_else(|| vec![0.0; h]);

    for layer in 0..cfg.n_layers {
        let attn_norm = weight(wb, &format!("blk.{}.attn_norm.weight", layer))?;
        let wq  = weight(wb, &format!("blk.{}.attn_q.weight", layer))?;
        let wk  = weight(wb, &format!("blk.{}.attn_k.weight", layer))?;
        let wv  = weight(wb, &format!("blk.{}.attn_v.weight", layer))?;
        let wo  = weight(wb, &format!("blk.{}.attn_output.weight", layer))?;
        let ffn_norm = weight(wb, &format!("blk.{}.ffn_norm.weight", layer))?;
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer))?;
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer))?;
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer))?;

        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let vv    = matvec(wv, &normed, kv_dim, h);
        rope_neox(&mut q, nq, hd, pos, cfg.rope_base);
        rope_neox(&mut k, nkv, hd, pos, cfg.rope_base);

        cache.k[layer].push(k);
        cache.v[layer].push(vv);
        let klen = cache.k[layer].len();

        let mut attn_out = vec![0.0f32; q_dim];
        for head in 0..nq {
            let kvh = head / group;
            let mut scores = vec![0.0f32; klen];
            for sp in 0..klen {
                let kc = &cache.k[layer][sp];
                let mut dot = 0.0f32;
                for d in 0..hd { dot += q[head * hd + d] * kc[kvh * hd + d]; }
                scores[sp] = dot * scale;
            }
            softmax(&mut scores);
            for d in 0..hd {
                let mut acc = 0.0f32;
                for sp in 0..klen { acc += scores[sp] * cache.v[layer][sp][kvh * hd + d]; }
                attn_out[head * hd + d] = acc;
            }
        }
        let proj = matvec(wo, &attn_out, h, q_dim);
        for i in 0..h { x[i] += proj[i]; }

        let normed2 = rmsnorm(&x, ffn_norm, cfg.eps);
        let gate = matvec(wgate, &normed2, cfg.intermediate, h);
        let up   = matvec(wup,   &normed2, cfg.intermediate, h);
        let mut swiglu = vec![0.0f32; cfg.intermediate];
        for i in 0..cfg.intermediate { swiglu[i] = silu(gate[i]) * up[i]; }
        let down = matvec(wdown, &swiglu, h, cfg.intermediate);
        for i in 0..h { x[i] += down[i]; }
    }

    // Retorna o hidden state após RMSNorm final (espaço de representação normalizado,
    // antes da projeção na cabeça de linguagem). É este espaço que a calibração usa.
    let final_norm = weight(wb, "output_norm.weight")?;
    Some(rmsnorm(&x, final_norm, cfg.eps))
}
