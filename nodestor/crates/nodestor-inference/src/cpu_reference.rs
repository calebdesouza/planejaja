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
use rayon::prelude::*;

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
    /// MoE: configuração de roteamento (None = modelo denso padrão).
    pub moe: Option<crate::moe_kernel::MoeConfig>,
}

#[inline]
fn weight<'a>(wb: &'a WeightBank, key: &str) -> Option<&'a [f32]> {
    wb.get(key).map(|b| b.as_f32_slice())
}

/// y[o] = Σ_i W[o·in_dim + i] · x[i]   (W em layout GGUF [out_dim, in_dim]).
/// Parallelized with rayon; inner loop uses iter().zip() for AVX2 auto-vectorization.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim).into_par_iter().map(|o| {
        let base = o * in_dim;
        if base + in_dim > w.len() { return 0.0f32; }
        w[base..base + in_dim].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
    }).collect()
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
    let pos_f = pos as f32;
    // Precompute (sin, cos) once per dimension — same across all heads.
    // Eliminates n_heads repetitions of powf (expensive transcendental).
    let sincos: Vec<(f32, f32)> = (0..half).map(|i| {
        (pos_f * base.powf(-2.0 * i as f32 / hd as f32)).sin_cos()
    }).collect();
    for h in 0..n_heads {
        let off = h * hd;
        for i in 0..half {
            let (sin, cos) = sincos[i];
            let a = vec[off + 2 * i];
            let b = vec[off + 2 * i + 1];
            vec[off + 2 * i]     = a * cos - b * sin;
            vec[off + 2 * i + 1] = a * sin + b * cos;
        }
    }
}

/// Adiciona bias que está em layout NEOX a um vetor Q/K em layout INTERLEAVED.
///
/// O llama.cpp permuta os pesos Q/K de NEOX→INTERLEAVED durante a conversão GGUF,
/// mas NÃO permuta os biases. Portanto o bias b[j] (NEOX: j < hd/2 = componente real,
/// j ≥ hd/2 = componente imaginária) precisa ser remapeado para o layout interleaved
/// (posição 2*j = real, 2*j+1 = imaginária) antes de ser somado ao vetor.
fn add_bias_neox_to_interleaved(vec: &mut [f32], bias: &[f32], n_heads: usize, hd: usize) {
    let half = hd / 2;
    for h in 0..n_heads {
        let off = h * hd;
        for j in 0..half {
            if off + 2 * j + 1 < vec.len() && off + half + j < bias.len() {
                vec[off + 2 * j]     += bias[off + j];           // real
                vec[off + 2 * j + 1] += bias[off + half + j];    // imaginary
            }
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
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer));
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer));
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer));

        // Q, K, V (com RoPE NeoX) por posição.
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(seq);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(seq);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for t in 0..seq {
            let normed = rmsnorm(&x[t], attn_norm, cfg.eps);
            let mut q = matvec(wq, &normed, q_dim, h);
            let mut k = matvec(wk, &normed, kv_dim, h);
            let mut v = matvec(wv, &normed, kv_dim, h);
            // Q/K: bias em NEOX layout → converter para interleaved antes de somar
            if let Some(b) = weight(wb, &format!("blk.{layer}.attn_q.bias")) { add_bias_neox_to_interleaved(&mut q, b, nq, hd); }
            if let Some(b) = weight(wb, &format!("blk.{layer}.attn_k.bias")) { add_bias_neox_to_interleaved(&mut k, b, nkv, hd); }
            // V: não sofre RoPE, bias se soma diretamente
            if let Some(b) = weight(wb, &format!("blk.{layer}.attn_v.bias")) { for i in 0..v.len().min(b.len()) { v[i] += b[i]; } }
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

        // Output proj + residual; FFN (denso ou MoE) + residual.
        for t in 0..seq {
            let proj = matvec(wo, &attn[t], h, q_dim);
            for i in 0..h { x[t][i] += proj[i]; }

            let normed2 = rmsnorm(&x[t], ffn_norm, cfg.eps);
            let ffn_out: Vec<f32> = if let Some(moe_cfg) = &cfg.moe {
                match crate::moe_kernel::moe_ffn_step(&normed2, moe_cfg, wb, layer) {
                    Some(moe_out) => {
                        crate::observability::emit_expert_selection(
                            layer,
                            &moe_out.routing.expert_indices,
                            moe_out.routing.entropy,
                        );
                        moe_out.hidden
                    }
                    None => {
                        if let (Some(wg), Some(wu), Some(wd)) = (wgate, wup, wdown) {
                            let gate = matvec(wg, &normed2, cfg.intermediate, h);
                            let up   = matvec(wu, &normed2, cfg.intermediate, h);
                            let mut mid = vec![0.0f32; cfg.intermediate];
                            for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
                            matvec(wd, &mid, h, cfg.intermediate)
                        } else { vec![0.0f32; h] }
                    }
                }
            } else {
                let wg = wgate?; let wu = wup?; let wd = wdown?;
                let gate = matvec(wg, &normed2, cfg.intermediate, h);
                let up   = matvec(wu, &normed2, cfg.intermediate, h);
                let mut mid = vec![0.0f32; cfg.intermediate];
                for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
                matvec(wd, &mid, h, cfg.intermediate)
            };
            for i in 0..h { x[t][i] += ffn_out[i]; }
        }
        crate::observability::emit(layer, &x[seq - 1]);
    }

    // Norma final + LM head na ÚLTIMA posição. `output.weight` ausente ⇒ pesos
    // amarrados (tied) ao token_embd.
    let final_norm = weight(wb, "output_norm.weight")?;
    let normed = rmsnorm(&x[seq - 1], final_norm, cfg.eps);
    let lm_head = weight(wb, "output.weight").unwrap_or(tok_embd);
    let logits = matvec(lm_head, &normed, cfg.vocab, h);
    // Emit logits (sentinel layer = usize::MAX) for entropy + top-k in TUI.
    crate::observability::emit(usize::MAX, &logits);
    Some(logits)
}

/// KV cache em CPU: por camada, K e V acumulados (uma entrada por posição já vista).
/// Permite o forward INCREMENTAL — processa só o token novo e atende sobre todo o
/// histórico cacheado, em vez de recomputar a sequência inteira a cada passo.
pub struct CpuKvCache {
    k: Vec<Vec<Vec<f32>>>, // [layer][pos][kv_dim]
    v: Vec<Vec<Vec<f32>>>,
    /// JANELA DESLIZANTE: se `Some(w)`, o cache nunca passa de `w` posições — as
    /// mais antigas são despejadas. Memória LIMITADA, nunca OOM, com contexto de
    /// qualquer tamanho. `None` = ilimitado (cresce com a RAM).
    max_window: Option<usize>,
}

impl CpuKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers], max_window: None }
    }
    /// Cache com JANELA DESLIZANTE de `window` posições (sliding-window attention,
    /// estilo Mistral). O token atual atende às últimas `window` posições; as que
    /// saem podem ir para o vector DB (LanceDB) para recuperação híbrida posterior
    /// — é a base do "contexto infinito com VRAM constante".
    pub fn with_window(n_layers: usize, window: usize) -> Self {
        Self { k: vec![Vec::new(); n_layers], v: vec![Vec::new(); n_layers], max_window: Some(window.max(1)) }
    }
    /// Número de posições já cacheadas (idêntico em todas as camadas).
    pub fn len(&self) -> usize { self.k.first().map(|l| l.len()).unwrap_or(0) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Despeja as posições mais antigas até caber na janela. Retorna quantas saíram
    /// (para o chamador indexá-las no vector DB, se desejar). No-op se `max_window`
    /// for `None` ou se já couber.
    pub fn enforce_window(&mut self) -> usize {
        let mut evicted = 0;
        if let Some(w) = self.max_window {
            while self.len() > w {
                for l in self.k.iter_mut() { if !l.is_empty() { l.remove(0); } }
                for l in self.v.iter_mut() { if !l.is_empty() { l.remove(0); } }
                evicted += 1;
            }
        }
        evicted
    }

    /// Reverte o cache para `len` posições (rollback de rascunhos REJEITADOS na
    /// decodificação especulativa). Mantém o histórico aceito intacto.
    pub fn truncate(&mut self, len: usize) {
        for l in self.k.iter_mut() { l.truncate(len); }
        for l in self.v.iter_mut() { l.truncate(len); }
    }

    /// Anexa K e V de uma posição à camada `layer`.
    pub fn push_kv(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        self.k[layer].push(k);
        self.v[layer].push(v);
    }

    /// Retorna slices imutáveis de K e V para a camada `layer`.
    pub fn layer_kv(&self, layer: usize) -> (&[Vec<f32>], &[Vec<f32>]) {
        (&self.k[layer], &self.v[layer])
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
        // Pesos densos são opcionais: modelos MoE não os têm (experts são carregados em moe_ffn_step)
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer));
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer));
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer));

        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let mut vv = matvec(wv, &normed, kv_dim, h);
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_q.bias")) { add_bias_neox_to_interleaved(&mut q, b, nq, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_k.bias")) { add_bias_neox_to_interleaved(&mut k, b, nkv, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_v.bias")) { for i in 0..vv.len().min(b.len()) { vv[i] += b[i]; } }
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

        // Dispatch: MoE (DeepSeek/Mixtral/Qwen) ou FFN denso (Llama/Mistral denso)
        let ffn_out: Vec<f32> = if let Some(moe_cfg) = &cfg.moe {
            match crate::moe_kernel::moe_ffn_step(&normed2, moe_cfg, wb, layer) {
                Some(moe_out) => {
                    // Emite expert loads para observabilidade e aquecimento de cache SSD
                    crate::observability::emit_expert_selection(
                        layer,
                        &moe_out.routing.expert_indices,
                        moe_out.routing.entropy,
                    );
                    moe_out.hidden
                }
                // Fallback silencioso: sem pesos MoE → FFN denso com o que tiver
                None => {
                    if let (Some(wg), Some(wu), Some(wd)) = (wgate, wup, wdown) {
                        let gate = matvec(wg, &normed2, cfg.intermediate, h);
                        let up   = matvec(wu, &normed2, cfg.intermediate, h);
                        let mut mid = vec![0.0f32; cfg.intermediate];
                        for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
                        matvec(wd, &mid, h, cfg.intermediate)
                    } else { vec![0.0f32; h] }
                }
            }
        } else {
            // Caminho denso padrão (Llama, Mistral, Phi, etc.)
            let wg = wgate?; let wu = wup?; let wd = wdown?;
            let gate = matvec(wg, &normed2, cfg.intermediate, h);
            let up   = matvec(wu, &normed2, cfg.intermediate, h);
            let mut mid = vec![0.0f32; cfg.intermediate];
            for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
            matvec(wd, &mid, h, cfg.intermediate)
        };

        for i in 0..h { x[i] += ffn_out[i]; }
        crate::observability::emit(layer, &x);
    }

    let final_norm = weight(wb, "output_norm.weight")?;
    let normed = rmsnorm(&x, final_norm, cfg.eps);
    let lm_head = weight(wb, "output.weight").unwrap_or(tok_embd);
    let logits = matvec(lm_head, &normed, cfg.vocab, h);
    crate::observability::emit(usize::MAX, &logits);
    Some(logits)
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
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer));
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer));
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer));

        // ── Atenção ──────────────────────────────────────────────────────────
        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let mut vv = matvec(wv, &normed, kv_dim, h);
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_q.bias")) { add_bias_neox_to_interleaved(&mut q, b, nq, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_k.bias")) { add_bias_neox_to_interleaved(&mut k, b, nkv, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_v.bias")) { for i in 0..vv.len().min(b.len()) { vv[i] += b[i]; } }
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

        // ── FFN: dispatch MoE ou denso ────────────────────────────────────────
        let normed2 = rmsnorm(&x, ffn_norm, cfg.eps);
        let ffn_out: Vec<f32> = if let Some(moe_cfg) = &cfg.moe {
            match crate::moe_kernel::moe_ffn_step(&normed2, moe_cfg, wb, layer) {
                Some(moe_out) => {
                    crate::observability::emit_expert_selection(
                        layer,
                        &moe_out.routing.expert_indices,
                        moe_out.routing.entropy,
                    );
                    moe_out.hidden
                }
                None => {
                    if let (Some(wg), Some(wu), Some(wd)) = (wgate, wup, wdown) {
                        let gate = matvec(wg, &normed2, cfg.intermediate, h);
                        let up   = matvec(wu, &normed2, cfg.intermediate, h);
                        let mut mid = vec![0.0f32; cfg.intermediate];
                        for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
                        matvec(wd, &mid, h, cfg.intermediate)
                    } else { vec![0.0f32; h] }
                }
            }
        } else {
            let wg = wgate?; let wu = wup?; let wd = wdown?;
            let gate = matvec(wg, &normed2, cfg.intermediate, h);
            let up   = matvec(wu, &normed2, cfg.intermediate, h);
            let mut mid = vec![0.0f32; cfg.intermediate];
            for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
            matvec(wd, &mid, h, cfg.intermediate)
        };
        for i in 0..h { x[i] += ffn_out[i]; }

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
        let wgate = weight(wb, &format!("blk.{}.ffn_gate.weight", layer));
        let wup   = weight(wb, &format!("blk.{}.ffn_up.weight", layer));
        let wdown = weight(wb, &format!("blk.{}.ffn_down.weight", layer));

        let normed = rmsnorm(&x, attn_norm, cfg.eps);
        let mut q = matvec(wq, &normed, q_dim, h);
        let mut k = matvec(wk, &normed, kv_dim, h);
        let mut vv = matvec(wv, &normed, kv_dim, h);
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_q.bias")) { add_bias_neox_to_interleaved(&mut q, b, nq, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_k.bias")) { add_bias_neox_to_interleaved(&mut k, b, nkv, hd); }
        if let Some(b) = weight(wb, &format!("blk.{layer}.attn_v.bias")) { for i in 0..vv.len().min(b.len()) { vv[i] += b[i]; } }
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
        let ffn_out: Vec<f32> = if let Some(moe_cfg) = &cfg.moe {
            match crate::moe_kernel::moe_ffn_step(&normed2, moe_cfg, wb, layer) {
                Some(moe_out) => moe_out.hidden,
                None => {
                    if let (Some(wg), Some(wu), Some(wd)) = (wgate, wup, wdown) {
                        let gate = matvec(wg, &normed2, cfg.intermediate, h);
                        let up   = matvec(wu, &normed2, cfg.intermediate, h);
                        let mut mid = vec![0.0f32; cfg.intermediate];
                        for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
                        matvec(wd, &mid, h, cfg.intermediate)
                    } else { vec![0.0f32; h] }
                }
            }
        } else {
            let wg = wgate?; let wu = wup?; let wd = wdown?;
            let gate = matvec(wg, &normed2, cfg.intermediate, h);
            let up   = matvec(wu, &normed2, cfg.intermediate, h);
            let mut mid = vec![0.0f32; cfg.intermediate];
            for i in 0..cfg.intermediate { mid[i] = silu(gate[i]) * up[i]; }
            matvec(wd, &mid, h, cfg.intermediate)
        };
        for i in 0..h { x[i] += ffn_out[i]; }
    }

    // Retorna o hidden state após RMSNorm final (espaço de representação normalizado,
    // antes da projeção na cabeça de linguagem). É este espaço que a calibração usa.
    let final_norm = weight(wb, "output_norm.weight")?;
    Some(rmsnorm(&x, final_norm, cfg.eps))
}

/// Cria um WeightBank mínimo sintético para testes de forward:
/// 1 camada, hidden=4, n_heads=2, head_dim=2, intermediate=8, vocab=16.
/// Todos os tensores são identidade ou constante não-zero para produzir logits finitos.
#[cfg(test)]
pub fn make_test_weight_bank() -> (nodestor_vulkan::WeightBank, CpuModelConfig) {
    use nodestor_vulkan::{WeightBank, VulkanEngine};
    let cfg = CpuModelConfig {
        n_layers: 1,
        hidden: 4,
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 2,
        intermediate: 8,
        vocab: 16,
        rope_base: 10000.0,
        eps: 1e-5,
        moe: None,
    };
    let engine = VulkanEngine::new_simulation();
    let mut wb = WeightBank::new();
    let h = cfg.hidden;
    let voc = cfg.vocab;
    let inter = cfg.intermediate;
    let q_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    let upload_f32 = |wb: &mut WeightBank, name: &str, data: Vec<f32>| {
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        if let Ok(buf) = engine.upload(raw) { wb.insert(name.to_string(), buf); }
    };

    // token_embd.weight [vocab, hidden] — identidade bloco 4×4, repetida
    let mut embd = vec![0.0f32; voc * h];
    for i in 0..voc.min(h) { embd[i * h + i] = 1.0; }
    upload_f32(&mut wb, "token_embd.weight", embd.clone());
    upload_f32(&mut wb, "output.weight", embd);

    // Normas (todos uns — RMSNorm com weight=1 é passthrough)
    upload_f32(&mut wb, "blk.0.attn_norm.weight", vec![1.0; h]);
    upload_f32(&mut wb, "blk.0.ffn_norm.weight", vec![1.0; h]);
    upload_f32(&mut wb, "output_norm.weight", vec![1.0; h]);

    // Projeções Q,K,V — identidade (q_dim×h, kv_dim×h, kv_dim×h)
    let eye_qh: Vec<f32> = (0..q_dim).flat_map(|r| (0..h).map(move |c| if r == c { 1.0 } else { 0.0 })).collect();
    let eye_kvh: Vec<f32> = (0..kv_dim).flat_map(|r| (0..h).map(move |c| if r == c { 1.0 } else { 0.0 })).collect();
    upload_f32(&mut wb, "blk.0.attn_q.weight", eye_qh.clone());
    upload_f32(&mut wb, "blk.0.attn_k.weight", eye_kvh.clone());
    upload_f32(&mut wb, "blk.0.attn_v.weight", eye_kvh);
    // attn_output: h×q_dim identidade
    let eye_hq: Vec<f32> = (0..h).flat_map(|r| (0..q_dim).map(move |c| if r == c { 1.0 } else { 0.0 })).collect();
    upload_f32(&mut wb, "blk.0.attn_output.weight", eye_hq);

    // FFN: gate/up [inter×h], down [h×inter]
    let ffn_up: Vec<f32> = (0..inter).flat_map(|r| (0..h).map(move |c| if r % h == c { 0.5 } else { 0.0 })).collect();
    let ffn_down: Vec<f32> = (0..h).flat_map(|r| (0..inter).map(move |c| if c % h == r { 0.5 } else { 0.0 })).collect();
    upload_f32(&mut wb, "blk.0.ffn_gate.weight", ffn_up.clone());
    upload_f32(&mut wb, "blk.0.ffn_up.weight", ffn_up);
    upload_f32(&mut wb, "blk.0.ffn_down.weight", ffn_down);

    (wb, cfg)
}

#[cfg(test)]
mod kv_cache_consistency_tests {
    use super::{CpuKvCache, forward_last_logits, forward_step, argmax, make_test_weight_bank};

    /// Prova que forward_step incremental (KV-cache reuse) produz os MESMOS logits
    /// que forward_last_logits (full-recompute) para a mesma sequência de tokens.
    ///
    /// Este é o teorema central de correção da inferência incremental:
    ///   argmax(forward_last_logits([t0,t1,t2])) == argmax(step_last após prefill [t0,t1] + step t2)
    #[test]
    fn test_incremental_matches_full_recompute() {
        let (wb, cfg) = make_test_weight_bank();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        // Caminho 1: full recompute — recalcula a sequência inteira no último passo.
        let full_logits = forward_last_logits(&tokens, &cfg, &wb)
            .expect("forward_last_logits deve produzir logits");
        let full_tok = argmax(&full_logits);

        // Caminho 2: incremental com KV-cache — processa token por token.
        let mut kv = CpuKvCache::new(cfg.n_layers);
        let mut last_logits: Option<Vec<f32>> = None;
        for (pos, &tok) in tokens.iter().enumerate() {
            last_logits = forward_step(tok, pos, &cfg, &wb, &mut kv);
        }
        let inc_tok = argmax(last_logits.as_ref().expect("forward_step deve produzir logits"));

        assert_eq!(full_tok, inc_tok,
            "incremental e full-recompute devem escolher o mesmo próximo token \
             (full={full_tok}, incremental={inc_tok})");
    }

    /// Prova que o cache cresce exatamente uma posição por chamada a forward_step.
    #[test]
    fn test_kv_cache_grows_one_per_step() {
        let (wb, cfg) = make_test_weight_bank();
        let mut kv = CpuKvCache::new(cfg.n_layers);
        assert_eq!(kv.len(), 0);
        for pos in 0..5 {
            forward_step(pos as u32, pos, &cfg, &wb, &mut kv);
            assert_eq!(kv.len(), pos + 1, "cache deve ter exatamente pos+1 entradas após o passo {pos}");
        }
    }

    /// Prova que argmax dos logits é determinístico para a mesma entrada.
    #[test]
    fn test_incremental_is_deterministic() {
        let (wb, cfg) = make_test_weight_bank();
        let tokens: Vec<u32> = vec![0, 1, 2];

        let run = || {
            let mut kv = CpuKvCache::new(cfg.n_layers);
            let mut last: Option<Vec<f32>> = None;
            for (p, &t) in tokens.iter().enumerate() { last = forward_step(t, p, &cfg, &wb, &mut kv); }
            argmax(last.as_ref().unwrap())
        };
        assert_eq!(run(), run(), "forward incremental deve ser determinístico");
    }

    /// Prova que truncate() (rollback de rascunhos rejeitados em COBER) restaura o
    /// cache ao estado de antes da especulação.
    #[test]
    fn test_kv_cache_truncate_rollback() {
        let (wb, cfg) = make_test_weight_bank();
        let mut kv = CpuKvCache::new(cfg.n_layers);

        // Prefill: 3 tokens aceitos
        for (p, &t) in [1u32, 2, 3].iter().enumerate() {
            forward_step(t, p, &cfg, &wb, &mut kv);
        }
        let accepted_len = kv.len(); // = 3

        // Especulação: 2 rascunhos inseridos no cache
        for (p, &t) in [4u32, 5].iter().enumerate() {
            forward_step(t, accepted_len + p, &cfg, &wb, &mut kv);
        }
        assert_eq!(kv.len(), 5, "cache deve ter 5 entradas após prefill+rascunho");

        // Rollback: rejeita os 2 rascunhos
        kv.truncate(accepted_len);
        assert_eq!(kv.len(), accepted_len, "truncate deve restaurar ao comprimento aceito");
    }

    /// Prova que forward_verify (verificação batched de rascunhos) produz os mesmos
    /// logits que processar os mesmos tokens um a um com forward_step.
    ///
    /// Esta é a garantia de LOSSLESSNESS do COBER: a verificação batched é
    /// matematicamente equivalente à verificação sequencial, então aceitar o
    /// prefixo concordante não altera a distribuição.
    #[test]
    fn test_forward_verify_matches_sequential_steps() {
        use super::{forward_verify, forward_step, make_test_weight_bank, argmax, CpuKvCache};

        let (wb, cfg) = make_test_weight_bank();
        let prefill: Vec<u32> = vec![1, 2];
        let drafts: Vec<u32> = vec![3, 4, 5];

        // Prefill comum: ambos os caminhos partem do mesmo estado de cache.
        let mut kv_seq = CpuKvCache::new(cfg.n_layers);
        for (p, &t) in prefill.iter().enumerate() { forward_step(t, p, &cfg, &wb, &mut kv_seq); }
        let mut kv_bat = CpuKvCache::new(cfg.n_layers);
        for (p, &t) in prefill.iter().enumerate() { forward_step(t, p, &cfg, &wb, &mut kv_bat); }

        // Caminho sequencial: processa cada rascunho um a um.
        let mut seq_toks: Vec<u32> = Vec::new();
        let start = prefill.len();
        for (i, &d) in drafts.iter().enumerate() {
            let logits = forward_step(d, start + i, &cfg, &wb, &mut kv_seq).unwrap();
            seq_toks.push(argmax(&logits));
        }

        // Caminho batched: forward_verify.
        let bat_logits = forward_verify(&drafts, start, &cfg, &wb, &mut kv_bat);
        let bat_toks: Vec<u32> = bat_logits.iter().map(|l| argmax(l)).collect();

        assert_eq!(seq_toks, bat_toks,
            "forward_verify deve produzir os mesmos tokens que passos sequenciais \
             (losslessness da verificação batched): seq={seq_toks:?} bat={bat_toks:?}");
        assert_eq!(kv_seq.len(), kv_bat.len(),
            "ambos os caminhos devem deixar o cache com o mesmo comprimento");
    }
}

#[cfg(test)]
mod sliding_window_tests {
    use super::CpuKvCache;

    /// Empurra 1 posição (marcada por `marker`) em todas as camadas — como o forward.
    fn push_pos(c: &mut CpuKvCache, marker: f32) {
        for l in c.k.iter_mut() { l.push(vec![marker]); }
        for l in c.v.iter_mut() { l.push(vec![marker]); }
    }

    #[test]
    fn test_sliding_window_bounds_memory_and_evicts_oldest() {
        let window = 4;
        let mut c = CpuKvCache::with_window(2, window);
        // Empurra 10 posições (0..10) — equivalente a um contexto bem maior que a janela.
        for i in 0..10 {
            push_pos(&mut c, i as f32);
            c.enforce_window();
            assert!(c.len() <= window, "cache NUNCA passa da janela (i={}, len={})", i, c.len());
        }
        // Memória LIMITADA: exatamente `window` posições, independente do contexto.
        assert_eq!(c.len(), window);
        // Mantém as MAIS RECENTES (6,7,8,9), despejou as antigas (0..6).
        assert_eq!(c.k[0][window - 1][0], 9.0, "última = mais recente");
        assert_eq!(c.k[0][0][0], 6.0, "primeira na janela = 10−window = 6");
    }

    #[test]
    fn test_unlimited_cache_grows_freely() {
        let mut c = CpuKvCache::new(1); // sem janela (None)
        for i in 0..50 { push_pos(&mut c, i as f32); c.enforce_window(); }
        assert_eq!(c.len(), 50, "sem janela, enforce_window é no-op e o cache cresce");
    }
}
