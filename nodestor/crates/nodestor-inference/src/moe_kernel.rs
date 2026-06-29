/// MoE Kernel — Roteamento e Computação de Experts
///
/// Implementa o forward FFN para Mixture-of-Experts com a matemática
/// exata de DeepSeek-V2/V3, Mixtral, Qwen-MoE e derivados.
///
/// Algoritmo central (Token-Choice Top-K):
///   s_i  = activation(W_router · x)   [n_experts]       raw scores
///   k*   = top_k(s_i + bias_i)        [top_k indices]   bias=0 para Mixtral
///   w_i  = s_i / Σ_{j ∈ k*} s_j      [top_k weights]   renormalização
///   y    = Σ_{i ∈ k*} w_i · FFN_i(x) + FFN_shared(x)
///
/// Referências:
///   DeepSeek-V3 Technical Report (2024) — routing sem perda auxiliar
///   Mixtral of Experts (2024, Mistral AI) — top-2 softmax router
///   Switch Transformers (Fedus et al., 2022) — capacidade por expert

use rayon::prelude::*;
use nodestor_vulkan::WeightBank;

// ─── Configuração ─────────────────────────────────────────────────────────────

/// Tipo de ativação do roteador.
/// Softmax: probabilidades somam 1 (Mixtral, Qwen-MoE, DeepSeek-V2)
/// Sigmoid: cada expert é independente, scores renormalizados (DeepSeek-V3)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouterKind { Softmax, Sigmoid }

/// Parâmetros MoE extraídos do GGUF / GraphInterpreter.
#[derive(Clone)]
pub struct MoeConfig {
    pub hidden:              usize,
    pub intermediate:        usize,
    pub n_experts:           usize,
    pub top_k:               usize,
    pub n_shared_experts:    usize,
    pub shared_intermediate: usize,
    pub router_kind:         RouterKind,
    pub norm_top_k_weights:  bool,
    /// On-demand SSD expert loader for models too large to fit in WeightBank.
    /// None → all expert weights are in WeightBank (small models / full-RAM).
    pub ssd_store: Option<std::sync::Arc<crate::ssd_stream::SsdExpertStore>>,
}

impl std::fmt::Debug for MoeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeConfig")
            .field("n_experts",  &self.n_experts)
            .field("top_k",      &self.top_k)
            .field("router",     &self.router_kind)
            .field("ssd_store",  &self.ssd_store.is_some())
            .finish()
    }
}

impl MoeConfig {
    /// Detecta configuração MoE diretamente do WeightBank (nomes de tensor GGUF).
    pub fn from_weight_bank(wb: &WeightBank, hidden: usize, intermediate: usize) -> Option<Self> {
        // Verifica se é MoE: tensor de router deve existir na camada 0
        let router_key = "blk.0.ffn_gate_inp.weight";
        if wb.get(router_key).is_none() { return None; }

        // Conta experts: tamanho do router = n_experts × hidden
        let router_len = wb.get(router_key)?.as_f32_slice().len();
        let n_experts = router_len / hidden.max(1);
        if n_experts == 0 { return None; }

        // Detecta shared experts
        let n_shared = if wb.get("blk.0.ffn_gate_shexp.weight").is_some() { 1 } else { 0 };
        let shared_int = if n_shared > 0 {
            // Shared expert intermediate dim = tamanho gate_shexp / hidden
            wb.get("blk.0.ffn_gate_shexp.weight")
              .map(|t| t.as_f32_slice().len() / hidden.max(1))
              .unwrap_or(intermediate * n_shared)
        } else { 0 };

        // Heurística para top_k: DeepSeek-V3 usa top-8 com 256 experts;
        // modelos com ≤8 experts tipicamente usam top-2.
        let top_k = if n_experts > 32 { 8 } else if n_experts > 8 { 4 } else { 2 };

        // DeepSeek-V3 usa sigmoid; modelos com bias explícito ou indicador
        let router_kind = if n_experts >= 64 { RouterKind::Sigmoid } else { RouterKind::Softmax };

        Some(MoeConfig {
            hidden, intermediate, n_experts, top_k,
            n_shared_experts: n_shared, shared_intermediate: shared_int,
            router_kind, norm_top_k_weights: true,
            ssd_store: None,
        })
    }

    /// Attach a SsdExpertStore for on-demand SSD loading of expert weights.
    /// Call this after `from_weight_bank()` for models too large to fit in RAM.
    pub fn with_ssd_store(mut self, store: std::sync::Arc<crate::ssd_stream::SsdExpertStore>) -> Self {
        self.ssd_store = Some(store);
        self
    }
}

// ─── Extração de pesos por expert ─────────────────────────────────────────────

/// Extrai os pesos de um único expert de uma matriz de experts empilhados.
///
/// Layout GGUF: [n_experts · out_dim, in_dim] (row-major)
/// Expert e ocupa as linhas [e·out_dim .. (e+1)·out_dim].
#[inline]
fn expert_slice(stacked: &[f32], e: usize, out_dim: usize, in_dim: usize) -> &[f32] {
    let stride = out_dim * in_dim;
    let start  = e * stride;
    let end    = start + stride;
    if end <= stacked.len() { &stacked[start..end] } else { &[] }
}

// ─── Operações matemáticas ────────────────────────────────────────────────────

/// y[o] = Σ_i W[o·in + i] · x[i]  (matvec, row-major, AVX2 via auto-vectorize)
#[inline]
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim).into_par_iter().map(|o| {
        let base = o * in_dim;
        if base + in_dim > w.len() { return 0.0f32; }
        w[base..base+in_dim].iter().zip(x.iter()).map(|(&a,&b)| a*b).sum()
    }).collect()
}

/// SiLU: x / (1 + exp(-x))  — derivada contínua, melhor que ReLU para LLMs.
#[inline] fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Softmax numericamente estável in-place.
fn softmax_vec(v: &mut Vec<f32>) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 1e-12 { for x in v.iter_mut() { *x /= sum; } }
}

/// Sigmoid element-wise.
#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ─── FFN de um único expert ───────────────────────────────────────────────────

/// SwiGLU FFN para um expert: down(SiLU(gate(x)) ⊙ up(x))
/// Retorna vetor [hidden].
fn expert_swiglu(
    x:          &[f32],
    wgate:      &[f32],
    wup:        &[f32],
    wdown:      &[f32],
    inter:      usize,
    hidden:     usize,
) -> Vec<f32> {
    // Gate e Up projetam hidden → inter; Down projeta inter → hidden.
    // Paralelizamos gate e up em paralelo (rayon interno via matvec).
    let gate_out = matvec(wgate, x, inter, hidden);
    let up_out   = matvec(wup,   x, inter, hidden);
    // SwiGLU element-wise
    let mut mid = vec![0.0f32; inter];
    for i in 0..inter { mid[i] = silu(gate_out[i]) * up_out[i]; }
    matvec(wdown, &mid, hidden, inter)
}

// ─── Roteador ─────────────────────────────────────────────────────────────────

/// Resultado do roteador: índices selecionados + pesos normalizados + entropia.
pub struct RouterOutput {
    /// Índices dos top-K experts selecionados (comprimento = top_k).
    pub expert_indices: Vec<usize>,
    /// Pesos dos experts selecionados (somam 1.0 se norm_top_k_weights).
    pub expert_weights: Vec<f32>,
    /// Entropia do router H = -Σ s_i log(s_i), em nats.
    /// Alta entropia → routing uniforme (saudável).
    /// Baixa entropia → colapso de routing (problema de treino).
    pub entropy: f32,
    /// Pontuação máxima (confiança do router no expert mais ativado).
    pub max_score: f32,
}

/// Computa o roteamento: scores → top-K seleção → normalização.
///
/// `raw_logits`: W_router · x  [n_experts] — saída bruta da projeção linear.
/// `bias`:       vetor de bias por expert (DeepSeek-V3 aux-free), zeros se ausente.
pub fn route(
    raw_logits: Vec<f32>,
    bias:       Option<&[f32]>,
    cfg:        &MoeConfig,
) -> RouterOutput {
    let n = raw_logits.len();

    // 1. Ativar scores
    let mut scores: Vec<f32> = match cfg.router_kind {
        RouterKind::Softmax => {
            let mut s = raw_logits;
            softmax_vec(&mut s);
            s
        }
        RouterKind::Sigmoid => raw_logits.iter().map(|&v| sigmoid(v)).collect(),
    };

    // 2. Entropia (antes de adicionar bias — sobre a distribuição real)
    let entropy = scores.iter().filter(|&&s| s > 1e-12)
        .map(|&s| -s * s.ln())
        .sum::<f32>();
    let max_score = scores.iter().cloned().fold(0.0f32, f32::max);

    // 3. Adicionar bias APENAS para seleção (DeepSeek-V3 aux-free)
    //    Os pesos finais usam `scores` sem bias.
    let selection_scores: Vec<f32> = if let Some(b) = bias {
        scores.iter().zip(b.iter()).map(|(&s, &bi)| s + bi).collect()
    } else {
        scores.clone()
    };

    // 4. Top-K por índice de selection_score (não precisa de sort completo — O(n·k))
    let top_k = cfg.top_k.min(n);
    let mut expert_indices = Vec::with_capacity(top_k);

    // Partial top-K: O(n * k) — correto e sem alocação de sort completo
    for _ in 0..top_k {
        let mut best = f32::NEG_INFINITY;
        let mut best_i = 0usize;
        for i in 0..n {
            if expert_indices.contains(&i) { continue; }
            if selection_scores[i] > best { best = selection_scores[i]; best_i = i; }
        }
        expert_indices.push(best_i);
    }

    // 5. Pesos: scores sem bias dos experts selecionados, renormalizados
    let mut expert_weights: Vec<f32> = expert_indices.iter().map(|&i| scores[i]).collect();
    if cfg.norm_top_k_weights {
        let sum: f32 = expert_weights.iter().sum();
        if sum > 1e-12 { for w in expert_weights.iter_mut() { *w /= sum; } }
    }

    RouterOutput { expert_indices, expert_weights, entropy, max_score }
}

// ─── FFN MoE principal ────────────────────────────────────────────────────────

/// Saída do forward MoE: vetor residual [hidden] + estatísticas para observabilidade.
pub struct MoeOutput {
    /// Vetor de saída do bloco FFN, a ser somado ao residual stream.
    pub hidden: Vec<f32>,
    /// Informações de roteamento (experts selecionados, pesos, entropia).
    pub routing: RouterOutput,
    /// Quanto cada expert contribuiu em norma L2 (para TUI heatmap).
    pub expert_l2: Vec<f32>,
}

/// Forward completo do bloco MoE-FFN para UM token.
///
/// `normed_x`:  hidden state pós-RMSNorm [hidden]
/// `cfg`:       hiperparâmetros MoE
/// `wb`:        WeightBank com todos os pesos do modelo
/// `layer`:     índice da camada (para construir nomes de tensor)
///
/// Retorna `None` se algum tensor crítico estiver ausente.
pub fn moe_ffn_step(
    normed_x:   &[f32],
    cfg:        &MoeConfig,
    wb:         &WeightBank,
    layer:      usize,
) -> Option<MoeOutput> {
    let h    = cfg.hidden;
    let inter = cfg.intermediate;
    let n_exp = cfg.n_experts;

    // ── Router ───────────────────────────────────────────────────────────────
    let router_w = wb.get(&format!("blk.{layer}.ffn_gate_inp.weight"))?.as_f32_slice();
    let raw_logits = matvec(router_w, normed_x, n_exp, h);

    let bias_data: Option<Vec<f32>> = wb
        .get(&format!("blk.{layer}.exp_probs_b"))
        .map(|t| t.as_f32_slice().to_vec());
    let bias_ref = bias_data.as_deref();

    let routing = route(raw_logits, bias_ref, cfg);

    // ── Expert weights: WeightBank (in-memory) or SSD on-demand ──────────────
    let wb_gate = wb.get(&format!("blk.{layer}.ffn_gate_exps.weight"));
    let wb_up   = wb.get(&format!("blk.{layer}.ffn_up_exps.weight"));
    let wb_down = wb.get(&format!("blk.{layer}.ffn_down_exps.weight"));
    let in_memory = wb_gate.is_some() && wb_up.is_some() && wb_down.is_some();

    // ── Computação paralela dos top-K experts ─────────────────────────────────
    let expert_outputs: Vec<(Vec<f32>, f32)> = if in_memory {
        // Fast path: slices directly from in-memory WeightBank (zero copy until SwiGLU).
        let gate_exps = wb_gate.unwrap().as_f32_slice();
        let up_exps   = wb_up.unwrap().as_f32_slice();
        let down_exps = wb_down.unwrap().as_f32_slice();
        routing.expert_indices
            .par_iter()
            .zip(routing.expert_weights.par_iter())
            .map(|(&e, &w)| {
                let wg = expert_slice(gate_exps, e, inter, h);
                let wu = expert_slice(up_exps,   e, inter, h);
                let wd = expert_slice(down_exps,  e, h, inter);
                (expert_swiglu(normed_x, wg, wu, wd, inter, h), w)
            })
            .collect()
    } else {
        // SSD streaming path: pre-fetch each active expert, then parallel SwiGLU.
        let ssd = cfg.ssd_store.as_ref()?;
        let triples: Option<Vec<_>> = routing.expert_indices.iter()
            .map(|&e| ssd.get_expert(layer, e, inter, h))
            .collect();
        let triples = triples?;
        triples.par_iter()
            .zip(routing.expert_weights.par_iter())
            .map(|(t, &w)| {
                (expert_swiglu(normed_x, &t.gate, &t.up, &t.down, inter, h), w)
            })
            .collect()
    };

    // ── Soma ponderada dos experts ────────────────────────────────────────────
    let mut hidden = vec![0.0f32; h];
    let mut expert_l2 = vec![0.0f32; n_exp];

    for (i, (expert_out, w)) in expert_outputs.into_iter().enumerate() {
        let e_idx = routing.expert_indices[i];
        let l2: f32 = expert_out.iter().map(|&v| v * v).sum::<f32>().sqrt();
        expert_l2[e_idx] = l2;
        for j in 0..h { hidden[j] += w * expert_out[j]; }
    }

    // ── Expert compartilhado (DeepSeek-V2/V3) — sempre computado ─────────────
    if cfg.n_shared_experts > 0 {
        let sint = cfg.shared_intermediate;
        // Suporte a shared expert único (n_shared_experts = 1) e múltiplos
        for s in 0..cfg.n_shared_experts {
            let (sg_key, su_key, sd_key) = if cfg.n_shared_experts == 1 {
                (
                    format!("blk.{layer}.ffn_gate_shexp.weight"),
                    format!("blk.{layer}.ffn_up_shexp.weight"),
                    format!("blk.{layer}.ffn_down_shexp.weight"),
                )
            } else {
                (
                    format!("blk.{layer}.ffn_gate_shexp.{s}.weight"),
                    format!("blk.{layer}.ffn_up_shexp.{s}.weight"),
                    format!("blk.{layer}.ffn_down_shexp.{s}.weight"),
                )
            };
            if let (Some(sg), Some(su), Some(sd)) = (
                wb.get(&sg_key), wb.get(&su_key), wb.get(&sd_key)
            ) {
                let shared_out = expert_swiglu(
                    normed_x,
                    sg.as_f32_slice(), su.as_f32_slice(), sd.as_f32_slice(),
                    sint, h,
                );
                for j in 0..h { hidden[j] += shared_out[j]; }
            }
        }
    }

    Some(MoeOutput { hidden, routing, expert_l2 })
}

// ─── Estatísticas de carga ────────────────────────────────────────────────────

/// Acumulador de frequência de seleção por expert ao longo de N tokens.
/// Alimenta o expert cache do SsdWeightStream (experts quentes → VRAM).
pub struct ExpertLoadTracker {
    pub counts:      Vec<u64>,
    pub tokens_seen: u64,
}

impl ExpertLoadTracker {
    pub fn new(n_experts: usize) -> Self {
        Self { counts: vec![0; n_experts], tokens_seen: 0 }
    }

    pub fn record(&mut self, selected: &[usize]) {
        for &e in selected { if e < self.counts.len() { self.counts[e] += 1; } }
        self.tokens_seen += 1;
    }

    /// Top-N experts mais frequentes (para pre-warming do cache SSD).
    pub fn top_n(&self, n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, u64)> = self.counts.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Taxa de ativação: fração de tokens em que cada expert foi selecionado.
    pub fn activation_rate(&self) -> Vec<f32> {
        let t = self.tokens_seen.max(1) as f32;
        self.counts.iter().map(|&c| c as f32 / t).collect()
    }

    /// Entropia de carga: H = -Σ p_i log p_i. Máximo = log(n_experts).
    /// Mede quão uniforme é o uso dos experts.
    pub fn load_entropy(&self) -> f32 {
        let rates = self.activation_rate();
        rates.iter().filter(|&&r| r > 1e-12).map(|&r| -r * r.ln()).sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_ref(v: &[f32]) -> Vec<f32> {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    #[test]
    fn router_softmax_top2_correct() {
        // 4 experts, top-2, nenhum bias
        let cfg = MoeConfig {
            hidden: 4, intermediate: 8, n_experts: 4, top_k: 2,
            n_shared_experts: 0, shared_intermediate: 0,
            router_kind: RouterKind::Softmax, norm_top_k_weights: true,
            ssd_store: None,
        };
        // Expert 2 com logit mais alto, depois expert 0
        let logits = vec![1.0f32, 0.1, 3.0, 0.5];
        let out = route(logits, None, &cfg);
        assert_eq!(out.expert_indices[0], 2, "expert mais ativado deve ser índice 2");
        assert_eq!(out.expert_indices[1], 0, "segundo deve ser índice 0");
        // Pesos somam 1
        let sum: f32 = out.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "pesos devem somar 1, got {}", sum);
        assert!(out.entropy > 0.0, "entropia deve ser positiva");
    }

    #[test]
    fn router_sigmoid_top2() {
        let cfg = MoeConfig {
            hidden: 4, intermediate: 8, n_experts: 4, top_k: 2,
            n_shared_experts: 0, shared_intermediate: 0,
            router_kind: RouterKind::Sigmoid, norm_top_k_weights: true,
            ssd_store: None,
        };
        let logits = vec![0.0f32, 2.0, -1.0, 1.5];
        let out = route(logits, None, &cfg);
        assert_eq!(out.expert_indices[0], 1); // sigmoid(2.0) ≈ 0.88
        assert_eq!(out.expert_indices[1], 3); // sigmoid(1.5) ≈ 0.82
        let sum: f32 = out.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn router_bias_shifts_selection() {
        // Sem bias: expert 1 vence. Com bias que promove expert 0: expert 0 vence.
        let cfg = MoeConfig {
            hidden: 4, intermediate: 8, n_experts: 4, top_k: 1,
            n_shared_experts: 0, shared_intermediate: 0,
            router_kind: RouterKind::Softmax, norm_top_k_weights: true,
            ssd_store: None,
        };
        let logits = vec![1.0f32, 2.0, 0.5, 0.1];
        // Sem bias: expert 1 selecionado
        let out_no_bias = route(logits.clone(), None, &cfg);
        assert_eq!(out_no_bias.expert_indices[0], 1);

        // Com bias forte no expert 0
        let bias = vec![5.0f32, 0.0, 0.0, 0.0];
        let out_biased = route(logits, Some(&bias), &cfg);
        assert_eq!(out_biased.expert_indices[0], 0, "bias deve forçar seleção do expert 0");
        // Mas o PESO ainda usa o score sem bias (expert 0 tem score softmax menor)
        // O peso de expert 0 = softmax(1.0) / softmax(1.0) = 1.0 (único selecionado)
        assert!((out_biased.expert_weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn expert_slice_indexing() {
        // 3 experts, out_dim=2, in_dim=3 → stacked = [6×3] = 18 elementos
        let stacked: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let e0 = expert_slice(&stacked, 0, 2, 3);
        let e1 = expert_slice(&stacked, 1, 2, 3);
        let e2 = expert_slice(&stacked, 2, 2, 3);
        assert_eq!(e0, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(e1, &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        assert_eq!(e2, &[12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn expert_swiglu_shape() {
        // Verifica que a saída tem dimensão correta e não é zero
        let hidden = 4;
        let inter  = 8;
        let x: Vec<f32>  = (0..hidden).map(|i| (i+1) as f32 * 0.1).collect();
        let wg: Vec<f32> = vec![0.1f32; inter * hidden];
        let wu: Vec<f32> = vec![0.2f32; inter * hidden];
        let wd: Vec<f32> = vec![0.05f32; hidden * inter];
        let out = expert_swiglu(&x, &wg, &wu, &wd, inter, hidden);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().any(|&v| v.abs() > 1e-6), "saída não deve ser zero");
    }

    #[test]
    fn expert_load_tracker_top_n() {
        let mut tracker = ExpertLoadTracker::new(8);
        // Expert 3 aparece 10 vezes, expert 7 aparece 5 vezes, outros 1 vez
        for _ in 0..10 { tracker.record(&[3]); }
        for _ in 0..5  { tracker.record(&[7]); }
        for e in 0..8usize { if e != 3 && e != 7 { tracker.record(&[e]); } }
        let top2 = tracker.top_n(2);
        assert_eq!(top2[0], 3);
        assert_eq!(top2[1], 7);
        assert!(tracker.load_entropy() > 0.0);
    }

    #[test]
    fn top_k_all_experts_equal_logits() {
        // Com logits iguais, todos os experts têm igual probabilidade
        let cfg = MoeConfig {
            hidden: 4, intermediate: 8, n_experts: 8, top_k: 4,
            n_shared_experts: 0, shared_intermediate: 0,
            router_kind: RouterKind::Softmax, norm_top_k_weights: true,
            ssd_store: None,
        };
        let logits = vec![1.0f32; 8];
        let out = route(logits, None, &cfg);
        assert_eq!(out.expert_indices.len(), 4);
        // Pesos devem ser iguais (0.25 cada)
        for &w in &out.expert_weights {
            assert!((w - 0.25).abs() < 1e-5, "pesos iguais com logits iguais, got {}", w);
        }
        // Entropia máxima: log(8) = 2.079
        assert!(out.entropy > 2.0, "entropia deve ser próxima de ln(8)={:.3}", 8.0f32.ln());
    }

    #[test]
    fn top_k_weight_sum_invariant() {
        // Propriedade invariante: pesos sempre somam 1 com norm_top_k_weights=true
        let cfg = MoeConfig {
            hidden: 4, intermediate: 8, n_experts: 16, top_k: 3,
            n_shared_experts: 0, shared_intermediate: 0,
            router_kind: RouterKind::Softmax, norm_top_k_weights: true,
            ssd_store: None,
        };
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Testa com 20 vetores pseudo-aleatórios diferentes
        for seed in 0u64..20 {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            let logits: Vec<f32> = (0..16).map(|i| {
                let mut hh = DefaultHasher::new();
                (seed * 100 + i as u64).hash(&mut hh);
                ((hh.finish() % 1000) as f32 / 100.0) - 5.0
            }).collect();
            let out = route(logits, None, &cfg);
            let sum: f32 = out.expert_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "seed={}: soma={}", seed, sum);
        }
    }
}
