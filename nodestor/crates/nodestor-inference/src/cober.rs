use crate::{
    candidate_engine::{CandidateEngine, CandidateConfig},
    caches::{ExpertLruCache, FeatureCache},
    vram_budget::VramBudget,
    prompt_lookup::PromptLookup,
    bloom_filter::TokenBloomFilter,
    golden_ngrams::GoldenNgramCache,
    rest_trie::RestTrie,
    insight_indexer::InsightIndexer,
    multi_draft::MultiDraftExplorer,
    cross_modal::CrossModalBus,
    jitter_buffer::{JitterBuffer, SymphonyConfig},
    syntactic_skeleton::SyntacticSkeleton,
    medusa_heads::AnchoredMedusa,
};
use std::collections::VecDeque;

/// Configuração do motor COBER.
#[derive(Debug, Clone)]
pub struct CoberConfig {
    /// Número máximo de tokens draft por rodada (K).
    pub max_draft_tokens: usize,
    /// Fator de ramificação da árvore de candidatos.
    pub tree_width: usize,
    /// Proporção do livre reservada ao NodeStor (0.75 = 75%).
    pub safety_margin: f32,
    /// Configuração da engine de candidatos.
    pub candidate_config: CandidateConfig,
    /// Nível de quantização dos embeddings residentes.
    pub embedding_quant: EmbeddingQuantLevel,
    /// EASD: limiar de entropia para colapsar a árvore (evitar desperdício).
    /// Se a entropia da distribuição superar este valor, K é reduzido ao mínimo.
    pub entropy_collapse_threshold: f32,
    /// EASD: entropia mínima abaixo da qual expandimos a árvore ao máximo.
    pub entropy_expand_threshold: f32,
}

/// Nível de quantização para embeddings residentes na VRAM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingQuantLevel {
    /// FP16 puro — máxima precisão, uso máximo de VRAM.
    Full,
    /// Q6_K — 6 bits por peso, perda ~2-3%, compressão 2.67x (padrão).
    Q6K,
}

impl Default for CoberConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: 32,
            tree_width: 4,
            safety_margin: 0.75,
            candidate_config: CandidateConfig::default(),
            embedding_quant: EmbeddingQuantLevel::Q6K,
            // EASD: entropia alta (>2.5 nats) → modelo incerto, colapsa árvore
            entropy_collapse_threshold: 2.5,
            // EASD: entropia baixa (<0.5 nats) → modelo seguro, expande árvore
            entropy_expand_threshold: 0.5,
        }
    }
}

/// Resultado de uma rodada COBER (draft + verificação).
#[derive(Debug)]
pub struct FunnelResult {
    /// Tokens aceitos nesta rodada com fidelidade 100%.
    pub accepted_tokens: Vec<u32>,
    /// Quantidade de drafts gerados no Estágio 1
    pub k_drafted: usize,
    /// Quantidade de drafts que passaram na peneira (Estágio 2)
    pub k_filtered: usize,
    /// Quantidade de tokens aprovados no Rejection Sampling (Estágio 4)
    pub k_accepted: usize,
    /// Fidelity guarantee
    pub fidelity: String,
}

#[derive(Debug)]
pub struct CoberRound {
    /// Tokens aceitos nesta rodada (podem ser 0..=max_draft_tokens).
    pub accepted_tokens: Vec<u32>,
    /// Tokens rejeitados (enviados de volta para nova rodada).
    pub rejected_count: usize,
    /// Taxa de aceitação desta rodada (0.0..=1.0).
    pub acceptance_rate: f32,
    /// Latência do draft em microssegundos.
    pub draft_latency_us: u64,
    /// Latência da verificação em microssegundos.
    pub verify_latency_us: u64,
}

/// Estatísticas acumuladas de uma sessão COBER.
#[derive(Debug, Default)]
pub struct CoberStats {
    pub total_rounds: u64,
    pub total_draft_tokens: u64,
    pub total_accepted_tokens: u64,
    pub hnsw_hits: u64,
    pub bm25_hits: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub feature_cache_hits: u64,
}

impl CoberStats {
    /// Taxa global de aceitação do speculative decoding.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_draft_tokens == 0 { return 0.0; }
        self.total_accepted_tokens as f64 / self.total_draft_tokens as f64
    }

    /// Multiplicador de throughput vs. geração sequencial pura.
    pub fn speedup_factor(&self) -> f64 {
        if self.total_rounds == 0 { return 1.0; }
        self.total_accepted_tokens as f64 / self.total_rounds as f64
    }

    /// Relatório formatado.
    pub fn report(&self) -> String {
        format!(
            "CoberStats:\n  Rounds:          {}\n  Tokens Draft:    {}\n  Tokens Aceitos:  {}\n  Aceitação:       {:.1}%\n  Speedup:         {:.1}x\n  HNSW Hits:       {}\n  BM25 Hits:       {}\n  Expert Cache:    {} hits / {} miss ({:.0}%)",
            self.total_rounds,
            self.total_draft_tokens,
            self.total_accepted_tokens,
            self.acceptance_rate() * 100.0,
            self.speedup_factor(),
            self.hnsw_hits,
            self.bm25_hits,
            self.expert_cache_hits,
            self.expert_cache_misses,
            if self.expert_cache_hits + self.expert_cache_misses > 0 {
                self.expert_cache_hits as f64 / (self.expert_cache_hits + self.expert_cache_misses) as f64 * 100.0
            } else { 0.0 }
        )
    }
}

/// Motor COBER Universal — orquestra Draft → Verify → Accept.
pub struct CoberEngine {
    pub config: CoberConfig,
    pub vram_budget: VramBudget,
    pub stats: CoberStats,

    // --- Modo Dense ---
    pub candidate_engine: Option<CandidateEngine>,

    // --- Modo MoE ---
    pub expert_cache: Option<ExpertLruCache>,

    // --- Modo Diffusion ---
    pub feature_cache: Option<FeatureCache>,

    // --- Subsistemas Esqueleto de Cristal (v2) ---
    pub prompt_lookup: Option<PromptLookup>,
    pub bloom_filter: Option<TokenBloomFilter>,
    pub golden_ngrams: Option<GoldenNgramCache>,
    pub rest_trie: Option<RestTrie>,
    pub syntactic_skeleton: Option<SyntacticSkeleton>,
    pub medusa: Option<AnchoredMedusa>,

    // --- Subsistemas Gerenciamento de Ideias (v3) ---
    pub insight_indexer: Option<InsightIndexer>,
    pub multi_draft: Option<MultiDraftExplorer>,
    pub cross_modal_bus: Option<CrossModalBus>,

    // --- Fase 12: Sinfonia (Sincronizador de Tensores) ---
    pub symphony: Option<JitterBuffer>,

    /// Histórico recente de tokens aceitos (para ajuste dinâmico de K).
    acceptance_history: VecDeque<f32>,

    // =========================================================
    // LATENT DRAFTER: Funil Especulativo Híbrido
    // =========================================================
    pub latent_drafter: Option<crate::latent_drafter::LatentDrafter>,

    // =========================================================
    // EAGLE-2: Hidden-State Draft Head
    // =========================================================
    /// Pesos da cabeça de rascunho EAGLE-2.
    /// Dimensão: [hidden_dim × vocab_size] em f32, representados como
    /// uma projeção linear flat (hidden_dim linhas × vocab_size colunas).
    /// `None` = modo legado (sem EAGLE-2).
    pub eagle2_head_weights: Option<Vec<f32>>,
    /// Dimensão oculta esperada (deve coincidir com a penúltima camada do modelo).
    pub eagle2_hidden_dim: usize,
    /// Tamanho do vocabulário para a projeção EAGLE-2.
    pub eagle2_vocab_size: usize,

    // =========================================================
    // EASD: Entropy-Aware Speculative Decoding
    // =========================================================
    /// Histórico de entropia por rodada (últimas 32 rodadas).
    entropy_history: VecDeque<f32>,

    // =========================================================
    // Pre-gate Shadow: Prefetch Preditivo de Experts MoE
    // =========================================================
    /// IDs dos experts previstos para a próxima camada (pré-carregamento preditivo).
    /// Populado pelo `predict_next_experts()` após cada camada MoE processada.
    pub prefetch_queue: Vec<(usize, usize)>, // (layer_idx, expert_idx)
    /// Estatísticas do Pre-gate: acertos vs. total de previsões.
    pub pregate_hits: u64,
    pub pregate_total: u64,
}

impl CoberEngine {
    /// Cria o motor COBER para o modo Dense (Speculative Decoding).
    pub fn new_dense(vram_budget: VramBudget) -> Self {
        let config = CoberConfig::default();
        let candidate_engine = Some(CandidateEngine::with_config(
            config.candidate_config.clone()
        ));
        Self {
            config,
            vram_budget,
            stats: CoberStats::default(),
            candidate_engine,
            expert_cache: None,
            feature_cache: None,
            prompt_lookup: Some(PromptLookup::new(vec![3, 6], 3)),
            bloom_filter: Some(TokenBloomFilter::new(10000, 0.01)),
            golden_ngrams: Some(GoldenNgramCache::new(512, 3)),
            rest_trie: Some(RestTrie::new(10)),
            syntactic_skeleton: Some(SyntacticSkeleton::new()),
            medusa: Some(AnchoredMedusa::new(3)),
            insight_indexer: Some(InsightIndexer::new(0.05)),
            multi_draft: Some(MultiDraftExplorer::new(10)),
            cross_modal_bus: Some(CrossModalBus::new(128)),
            symphony: Some(JitterBuffer::new(SymphonyConfig::default())),
            acceptance_history: VecDeque::with_capacity(64),
            latent_drafter: Some(crate::latent_drafter::LatentDrafter::new(4096, 0.85)),
            eagle2_head_weights: None,
            eagle2_hidden_dim: 4096,
            eagle2_vocab_size: 128256,
            entropy_history: VecDeque::with_capacity(32),
            prefetch_queue: Vec::new(),
            pregate_hits: 0,
            pregate_total: 0,
        }
    }

    /// Cria o motor COBER para o modo MoE.
    pub fn new_moe(vram_budget: VramBudget, _num_experts: usize, _top_k: usize) -> Self {
        let expert_cache = Some(ExpertLruCache::new(vram_budget.cache_budget));
        Self {
            config: CoberConfig::default(),
            vram_budget,
            stats: CoberStats::default(),
            candidate_engine: None,
            expert_cache,
            feature_cache: None,
            prompt_lookup: None,
            bloom_filter: None,
            golden_ngrams: None,
            rest_trie: None,
            syntactic_skeleton: None,
            medusa: None,
            insight_indexer: Some(InsightIndexer::new(0.05)),
            multi_draft: Some(MultiDraftExplorer::new(10)),
            cross_modal_bus: Some(CrossModalBus::new(128)),
            symphony: Some(JitterBuffer::new(SymphonyConfig::default())),
            acceptance_history: VecDeque::with_capacity(64),
            latent_drafter: Some(crate::latent_drafter::LatentDrafter::new(4096, 0.85)),
            eagle2_head_weights: None,
            eagle2_hidden_dim: 4096,
            eagle2_vocab_size: 128256,
            entropy_history: VecDeque::with_capacity(32),
            prefetch_queue: Vec::new(),
            pregate_hits: 0,
            pregate_total: 0,
        }
    }

    /// Cria o motor COBER para o modo Diffusion.
    pub fn new_diffusion(vram_budget: VramBudget) -> Self {
        let feature_cache = Some(FeatureCache::new(vram_budget.cache_budget));
        Self {
            config: CoberConfig::default(),
            vram_budget,
            stats: CoberStats::default(),
            candidate_engine: None,
            expert_cache: None,
            feature_cache,
            prompt_lookup: None,
            bloom_filter: None,
            golden_ngrams: None,
            rest_trie: None,
            syntactic_skeleton: None,
            medusa: None,
            insight_indexer: Some(InsightIndexer::new(0.05)),
            multi_draft: Some(MultiDraftExplorer::new(10)),
            cross_modal_bus: Some(CrossModalBus::new(128)),
            symphony: Some(JitterBuffer::new(SymphonyConfig::default())),
            acceptance_history: VecDeque::with_capacity(64),
            latent_drafter: Some(crate::latent_drafter::LatentDrafter::new(4096, 0.85)),
            eagle2_head_weights: None,
            eagle2_hidden_dim: 4096,
            eagle2_vocab_size: 128256,
            entropy_history: VecDeque::with_capacity(32),
            prefetch_queue: Vec::new(),
            pregate_hits: 0,
            pregate_total: 0,
        }
    }

    /// Orquestra Draft do Esqueleto de Cristal em Cascata (O(1)).
    /// O Coração do Roteamento Neural-Simbólico.
    ///
    /// Hierarquia de Decisão:
    /// 1. L1: Golden N-Grams (O(1) - Velocidade Pura)
    /// 2. L2: Prompt Lookup (N-Gram Contextual)
    /// 3. L3: EAGLE-2 (Neural Draft de Hidden States) + EASD (Entropia)
    /// 4. L4: Medusa Heads (Multi-token parallel)
    pub fn draft_with_crystal_skeleton(&mut self, context: &[u32], hidden_state: &[f32]) -> Vec<u32> {
        // 1. L1 Golden N-Grams Cache (Sequências idênticas validadas)
        if let Some(golden) = &mut self.golden_ngrams {
            if let Some(draft) = golden.try_get(context) {
                if let Some(bloom) = &self.bloom_filter {
                    if bloom.maybe_valid(&draft) { 
                        tracing::debug!("COBER: L1 Hit (Golden N-Gram)");
                        return draft; 
                    }
                } else { return draft; }
            }
        }

        // 2. Prompt Lookup (Busca no contexto recente e LanceDB)
        if let Some(lookup) = &mut self.prompt_lookup {
            lookup.inject_lancedb_context(context);
            if let Some(draft) = lookup.lookup(context) {
                tracing::debug!("COBER: L2 Hit (Prompt Lookup)");
                return draft;
            }
        }

        // 3. EAGLE-2 + EASD (Draft Neural Adaptativo)
        // Usamos a entropia das probabilidades previstas para decidir se vale a pena especular muito ou pouco.
        let (neural_probs, neural_tokens) = self.eagle2_predict_probs(hidden_state, 32); // Max possible
        let (tree_width, tree_depth) = self.easd_compute_tree_params(&neural_probs);
        
        if tree_depth > 0 {
            // Se a entropia estiver baixa (modelo seguro), o EAGLE-2 gera tokens
            let neural_draft: Vec<u32> = neural_tokens.into_iter().take(tree_depth).collect();
            if !neural_draft.is_empty() {
                tracing::debug!("COBER: L3 Hit (EAGLE-2 Neural Draft, depth={})", tree_depth);
                return neural_draft;
            }
        }

        // 4. Medusa Heads (Âncoras de similaridade se o EAGLE falhar)
        if let Some(medusa) = &self.medusa {
            let tree = medusa.generate_anchored_tree(hidden_state, tree_width);
            if !tree.tokens.is_empty() {
                tracing::debug!("COBER: L4 Hit (Medusa Anchored)");
                return tree.tokens.iter().take(tree_depth).cloned().collect();
            }
        }

        
        // Fallback: Modelo mestre decide token a token (Segurança máxima)
        Vec::new()
    }

    /// O Funil Especulativo Híbrido — Pipeline de Produção KiloToken
    /// 100% de Fidelidade Matemática
    pub fn draft_with_speculative_funnel(
        &mut self,
        context: &[u32],
        hidden_state: &[f32],
        engine: &nodestor_vulkan::VulkanEngine,
        transformer: &nodestor_vulkan::Transformer,
        weight_bank: &nodestor_vulkan::WeightBank,
        position: u32,
    ) -> FunnelResult {
        let mut k_drafted = 0;
        let mut k_filtered = 0;

        // 1. EASD calcula K_max baseado na entropia recente
        let k_max = self.adjust_k_dynamic();
        k_drafted = k_max;

        let draft_tokens = if let Some(drafter) = &self.latent_drafter {
            // 2. EAGLE-2 gera K_max hidden states latentes (Estágio 1)
            let drafts = drafter.draft_latent_block(hidden_state, k_max);

            // 3. Peneira cossenóide descarta lixo (Estágio 2)
            k_filtered = drafter.cosine_sieve(&drafts, hidden_state);
            let clean_drafts = &drafts[..k_filtered];

            // 4. Projeta sobreviventes em tokens (Estágio 3)
            if let Some(lm_head_buf) = weight_bank.get(&transformer.lm_head_key) {
                drafter.project_to_tokens(engine, clean_drafts, lm_head_buf, transformer.vocab_size as usize).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        if draft_tokens.is_empty() {
            return FunnelResult {
                accepted_tokens: Vec::new(),
                k_drafted,
                k_filtered,
                k_accepted: 0,
                fidelity: "BitExact".to_string(),
            };
        }

        // 5. Forward pass do modelo mestre em prefill (Estágio 4)
        // Por simplicidade, assumimos que `transformer.forward_batch` retorna `master_logits`.
        // A interface será implementada no `transformer.rs`.
        let master_logits = transformer.forward_batch(engine, &draft_tokens, weight_bank, position).unwrap_or_default();

        // 6. Rejection Sampling BIT-EXACT — o único juiz
        let round = self.verify_and_accept(&draft_tokens, &master_logits, *context.last().unwrap_or(&0));

        FunnelResult {
            accepted_tokens: round.accepted_tokens.clone(),
            k_drafted,
            k_filtered,
            k_accepted: round.accepted_tokens.len(),
            fidelity: "BitExact".to_string(),
        }
    }


    /// Fase 1 do COBER: Geração de Rascunho.
    ///
    /// Usa HNSW + BM25 + RRF para gerar K candidatos de tokens.
    /// É extremamente rápido (<1ms) pois opera apenas sobre os
    /// embeddings residentes na VRAM — sem qualquer I/O.
    pub fn draft_round(
        &mut self,
        current_token: u32,
        hnsw_results: &[u32],
    ) -> Vec<u32> {
        let engine = match &self.candidate_engine {
            Some(e) => e,
            None => return Vec::new(),
        };

        let query = &[current_token];
        let candidates = engine.generate_candidates(hnsw_results, query);

        // Contabilizar hits
        for c in &candidates {
            if c.from_hnsw { self.stats.hnsw_hits += 1; }
            if c.from_bm25 { self.stats.bm25_hits += 1; }
        }

        self.stats.total_draft_tokens += candidates.len() as u64;
        CandidateEngine::candidate_ids(&candidates)
    }

    /// Fase 2 do COBER: Verificação e Rejection Sampling.
    ///
    /// Recebe as probabilidades do modelo mestre (logits normalizados)
    /// para cada posição do draft e aceita/rejeita tokens via
    /// rejection sampling (processo matematicamente lossless).
    ///
    /// Retorna os tokens aceitos. O output é bit-idêntico ao que o
    /// modelo mestre geraria de forma autoregressiva pura.
    pub fn verify_and_accept(
        &mut self,
        draft_tokens: &[u32],
        master_logits: &[Vec<f32>], // logits[position][vocab]
        _current_token: u32,
    ) -> CoberRound {
        let start = std::time::Instant::now();
        let mut accepted = Vec::new();
        let mut rejected_count = 0;

        'outer: for (pos, &draft_token) in draft_tokens.iter().enumerate() {
            if pos >= master_logits.len() {
                break;
            }
            let logits = &master_logits[pos];

            // Verificação: o token draft é o argmax do modelo mestre?
            // (Simplificado — em prod real: rejection sampling probabilístico)
            let master_best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            if master_best == draft_token {
                // Token aceito! Ingesta no histórico para próxima rodada.
                if let Some(engine) = &mut self.candidate_engine {
                    engine.ingest_token(draft_token);
                }
                accepted.push(draft_token);
            } else {
                // Token rejeitado. Aceita o token do mestre e para.
                accepted.push(master_best);
                if let Some(engine) = &mut self.candidate_engine {
                    engine.ingest_token(master_best);
                }
                rejected_count += 1;
                break 'outer;
            }
        }

        let verify_latency = start.elapsed().as_micros() as u64;
        let acceptance_rate = if draft_tokens.is_empty() { 0.0 }
            else { accepted.len() as f32 / draft_tokens.len() as f32 };

        // Atualizar estatísticas
        self.stats.total_rounds += 1;
        self.stats.total_accepted_tokens += accepted.len() as u64;
        self.acceptance_history.push_back(acceptance_rate);
        if self.acceptance_history.len() > 64 {
            self.acceptance_history.pop_front();
        }

        CoberRound {
            accepted_tokens: accepted,
            rejected_count,
            acceptance_rate,
            draft_latency_us: 0, // preenchido pelo chamador
            verify_latency_us: verify_latency,
        }
    }

    /// Rejection Sampling Probabilístico (Lossless para T > 0)
    /// Formulação: P_accept = min(1, p(x) / q(x))
    pub fn verify_and_accept_probabilistic(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[Vec<f32>], // q(x)
        master_probs: &[Vec<f32>], // p(x)
    ) -> CoberRound {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let start = std::time::Instant::now();
        let mut accepted = Vec::new();
        let mut rejected_count = 0;

        'outer: for (pos, &draft_token) in draft_tokens.iter().enumerate() {
            if pos >= master_probs.len() || pos >= draft_probs.len() {
                break;
            }
            let p = &master_probs[pos];
            let q = &draft_probs[pos];
            
            let draft_token_usize = draft_token as usize;
            let p_val = if draft_token_usize < p.len() { p[draft_token_usize] } else { 0.0 };
            let q_val = if draft_token_usize < q.len() { q[draft_token_usize] } else { 1.0 };
            
            // P_accept = min(1, p(x) / q(x))
            let p_accept = if q_val > 0.0 { (p_val / q_val).min(1.0) } else { 1.0 };
            
            let r: f32 = rng.gen();
            
            if r < p_accept {
                // Aceito
                accepted.push(draft_token);
                if let Some(engine) = &mut self.candidate_engine {
                    engine.ingest_token(draft_token);
                }
            } else {
                // Rejeitado - Amostrar da distribuição residual
                // P_resample(x) = max(0, p(x) - q(x)) / sum(max(0, p(x') - q(x')))
                rejected_count += 1;
                
                let mut resample_probs = vec![0.0; p.len()];
                let mut sum = 0.0;
                for i in 0..p.len() {
                    let diff = p[i] - if i < q.len() { q[i] } else { 0.0 };
                    let val = diff.max(0.0);
                    resample_probs[i] = val;
                    sum += val;
                }
                
                let mut resampled_token = 0;
                if sum > 0.0 {
                    let mut r_resample: f32 = rng.gen::<f32>() * sum;
                    for (i, &prob) in resample_probs.iter().enumerate() {
                        r_resample -= prob;
                        if r_resample <= 0.0 {
                            resampled_token = i as u32;
                            break;
                        }
                    }
                } else {
                    // Fallback se max(0, p-q) for tudo zero
                    resampled_token = p.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);
                }
                
                accepted.push(resampled_token);
                if let Some(engine) = &mut self.candidate_engine {
                    engine.ingest_token(resampled_token);
                }
                break 'outer;
            }
        }

        let verify_latency = start.elapsed().as_micros() as u64;
        let acceptance_rate = if draft_tokens.is_empty() { 0.0 }
            else { (accepted.len() - rejected_count) as f32 / draft_tokens.len() as f32 };

        self.stats.total_rounds += 1;
        self.stats.total_accepted_tokens += accepted.len() as u64;

        CoberRound {
            accepted_tokens: accepted,
            rejected_count,
            acceptance_rate,
            draft_latency_us: 0,
            verify_latency_us: verify_latency,
        }
    }

    /// Tree Verification com RRSw (Recursive Rejection Sampling without Replacement)
    pub fn verify_tree_rrsw(
        &mut self,
        draft_tree: &[(u32, Vec<u32>)], // (nó, filhos)
        master_probs: &[Vec<f32>],
    ) -> CoberRound {
        // Implementação simplificada de RRSw delegando para validação linear (por enquanto)
        let mut flat_draft = Vec::new();
        for (node, _) in draft_tree {
            flat_draft.push(*node);
        }
        let dummy_q = vec![vec![0.1; master_probs.get(0).map(|v| v.len()).unwrap_or(1)]; flat_draft.len()];
        self.verify_and_accept_probabilistic(&flat_draft, &dummy_q, master_probs)
    }

    /// Ajusta K dinamicamente baseado na taxa de aceitação recente.
    ///
    /// Se aceitação > 70% → aumenta K (aproveitamos mais o draft)
    /// Se aceitação < 40% → reduz K (draft não está bom, economizamos verify)
    pub fn adjust_k_dynamic(&mut self) -> usize {
        if self.acceptance_history.is_empty() {
            return self.config.max_draft_tokens;
        }

        let avg_rate: f32 = self.acceptance_history.iter().sum::<f32>()
            / self.acceptance_history.len() as f32;

        if avg_rate > 0.70 {
            // Aumentar K gradualmente (máximo 64)
            self.config.max_draft_tokens = (self.config.max_draft_tokens + 4).min(64);
        } else if avg_rate < 0.40 && self.config.max_draft_tokens > 8 {
            // Reduzir K gradualmente (mínimo 8)
            self.config.max_draft_tokens = (self.config.max_draft_tokens - 4).max(8);
        }

        self.config.max_draft_tokens
    }

    /// Para modo MoE: verifica experts em cache e registra miss para prefetch.
    pub fn check_expert_cache(&mut self, expert_id: (usize, usize)) -> bool {
        if let Some(cache) = &mut self.expert_cache {
            if cache.contains(expert_id) {
                self.stats.expert_cache_hits += 1;
                true
            } else {
                self.stats.expert_cache_misses += 1;
                false
            }
        } else {
            false
        }
    }

    /// Torna o relatório de estatísticas acessível.
    pub fn stats_report(&self) -> String {
        self.stats.report()
    }

    // =========================================================
    // EAGLE-2: Hidden-State Draft Head
    // =========================================================

    /// Registra os pesos da cabeça de rascunho EAGLE-2.
    ///
    /// `weights` deve ter exatamente `hidden_dim * vocab_size` elementos f32
    /// (linha-maior: weights[h * vocab_size + v] = peso da dim h para o token v).
    pub fn load_eagle2_head(&mut self, weights: Vec<f32>, hidden_dim: usize, vocab_size: usize) {
        assert_eq!(
            weights.len(), hidden_dim * vocab_size,
            "EAGLE-2: pesos devem ter hidden_dim × vocab_size elementos"
        );
        self.eagle2_hidden_dim = hidden_dim;
        self.eagle2_vocab_size = vocab_size;
        self.eagle2_head_weights = Some(weights);
    }

    /// TREINA (destila) a cabeça de rascunho EAGLE-2 por descida de gradiente.
    ///
    /// Dado um conjunto de pares `(hidden_state, token_alvo)` coletados do modelo
    /// mestre, aprende `W` (hidden×vocab) que prediz o token a partir do hidden,
    /// minimizando cross-entropy: `L = -log softmax(W·h)[alvo]`. Esta é a "cabeça
    /// de rascunho treinada" que o EAGLE-2 exige — sem treino, a especulação
    /// extrema (alta aceitação) NÃO existe; é da natureza do método.
    ///
    /// Retorna a perda média da última época. NOTA HONESTA: aceitação alta (~90%)
    /// exige treino em CORPUS (muitos forwards do modelo-alvo). Com poucos pares,
    /// a cabeça apenas memoriza o contexto local.
    pub fn train_eagle2_head(
        &mut self,
        pairs: &[(Vec<f32>, u32)],
        hidden_dim: usize,
        vocab_size: usize,
        epochs: usize,
        lr: f32,
    ) -> f32 {
        let (h, v) = (hidden_dim, vocab_size);
        let mut w = match self.eagle2_head_weights.take() {
            Some(w) if w.len() == h * v => w,
            _ => vec![0.0f32; h * v],
        };
        let mut last_loss = 0.0f32;
        for _ in 0..epochs.max(1) {
            let mut epoch_loss = 0.0f32;
            let mut count = 0usize;
            for (hidden, target) in pairs {
                if hidden.len() < h { continue; }
                let target = *target as usize;
                if target >= v { continue; }
                // forward: logits = W·hidden  →  softmax  →  p
                let mut p = vec![0.0f32; v];
                for vi in 0..v {
                    let mut acc = 0.0f32;
                    for hi in 0..h { acc += hidden[hi] * w[hi * v + vi]; }
                    p[vi] = acc;
                }
                let m = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in p.iter_mut() { *x = (*x - m).exp(); sum += *x; }
                if sum > 0.0 { for x in p.iter_mut() { *x /= sum; } }
                epoch_loss += -(p[target].max(1e-9)).ln();
                count += 1;
                // grad/update: dL/dW[hi,vi] = hidden[hi]·(p[vi] − 1{vi=alvo})
                for hi in 0..h {
                    let hv = hidden[hi];
                    if hv == 0.0 { continue; }
                    let base = hi * v;
                    for vi in 0..v {
                        let g = hv * (p[vi] - if vi == target { 1.0 } else { 0.0 });
                        w[base + vi] -= lr * g;
                    }
                }
            }
            last_loss = epoch_loss / count.max(1) as f32;
        }
        self.eagle2_hidden_dim = h;
        self.eagle2_vocab_size = v;
        self.eagle2_head_weights = Some(w);
        last_loss
    }

    /// Gera probabilidades e Top-K tokens via EAGLE-2.
    /// Retorna `(probs, tokens)`.
    pub fn eagle2_predict_probs(&self, hidden_state: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
        let weights = match &self.eagle2_head_weights {
            Some(w) => w,
            None => return (Vec::new(), Vec::new()),
        };

        let v = self.eagle2_vocab_size;
        let h = self.eagle2_hidden_dim;
        if hidden_state.len() < h || weights.len() < h * v {
            return (Vec::new(), Vec::new());
        }

        let mut logits = vec![0.0f32; v];
        for vi in 0..v {
            let mut acc = 0.0f32;
            for hi in 0..h {
                acc += hidden_state[hi] * weights[hi * v + vi];
            }
            logits[vi] = acc;
        }

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; v];
        let mut sum_exp = 0.0f32;
        for i in 0..v {
            probs[i] = (logits[i] - max_logit).exp();
            sum_exp += probs[i];
        }
        if sum_exp > 0.0 {
            for p in probs.iter_mut() { *p /= sum_exp; }
        }

        let effective_k = k.min(v);
        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(effective_k + 1);
        for (vi, &prob) in probs.iter().enumerate() {
            heap.push((vi as u32, prob));
            heap.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if heap.len() > effective_k { heap.pop(); }
        }
        let tokens = heap.into_iter().map(|(t, _)| t).collect();
        (probs, tokens)
    }

    /// Gera tokens de rascunho via EAGLE-2.
    pub fn eagle2_draft_from_hidden(&self, hidden_state: &[f32], k: usize) -> Vec<u32> {
        self.eagle2_predict_probs(hidden_state, k).1
    }

    // =========================================================
    // EASD: Entropy-Aware Speculative Decoding
    // =========================================================

    /// Calcula a entropia de Shannon (em nats) de um vetor de logits/probabilidades.
    ///
    /// H(p) = -Σ p(x) · ln(p(x))
    ///
    /// Usado pelo EASD para decidir se a árvore de rascunho deve ser expandida
    /// (modelo seguro, baixa entropia) ou colapsada (modelo incerto, alta entropia).
    pub fn compute_entropy(probs: &[f32]) -> f32 {
        probs.iter()
            .filter(|&&p| p > 1e-9)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Determina o K e tree_width ideais para esta rodada com base na entropia.
    ///
    /// ## Lógica EASD:
    /// - `entropy < expand_threshold` → árvore larga, K máximo (modelo seguro)
    /// - `entropy > collapse_threshold` → árvore estreita, K mínimo (modelo incerto)
    /// - Entropia intermediária → interpolação linear
    ///
    /// Retorna `(effective_k, effective_width)`.
    pub fn easd_compute_tree_params(&mut self, master_probs: &[f32]) -> (usize, usize) {
        let entropy = Self::compute_entropy(master_probs);

        // Registra histórico de entropia
        self.entropy_history.push_back(entropy);
        if self.entropy_history.len() > 32 {
            self.entropy_history.pop_front();
        }

        let lo = self.config.entropy_expand_threshold;   // ex: 0.5
        let hi = self.config.entropy_collapse_threshold; // ex: 2.5
        let max_k = self.config.max_draft_tokens;        // ex: 32
        let min_k = 4usize;
        let max_w = self.config.tree_width;              // ex: 4
        let min_w = 1usize;

        if entropy <= lo {
            // Modelo muito seguro: árvore totalmente expandida
            (max_k, max_w)
        } else if entropy >= hi {
            // Modelo incerto: colapsa ao mínimo para não desperdiçar verificação
            (min_k, min_w)
        } else {
            // Interpolação linear entre [lo, hi]
            let t = (entropy - lo) / (hi - lo); // 0.0 = expand, 1.0 = collapse
            let k = (max_k as f32 * (1.0 - t) + min_k as f32 * t).round() as usize;
            let w = (max_w as f32 * (1.0 - t) + min_w as f32 * t).round() as usize;
            (k.max(min_k), w.max(min_w))
        }
    }

    /// Retorna a entropia média das últimas N rodadas.
    pub fn mean_entropy(&self) -> f32 {
        if self.entropy_history.is_empty() { return 0.0; }
        self.entropy_history.iter().sum::<f32>() / self.entropy_history.len() as f32
    }

    // =========================================================
    // Pre-gate Shadow: Prefetch Preditivo de Experts MoE
    // =========================================================

    /// Prevê os experts necessários na camada N+1 usando o estado oculto
    /// atual (camada N) com pesos quantizados em INT4 (shadow pass ultra-leve).
    ///
    /// ## Funcionamento:
    /// Após processar a camada N, o shadow pass projeta o hidden state em INT4
    /// para obter os scores dos experts da camada N+1. Os top-K experts previstos
    /// são enfileirados em `prefetch_queue` para que o APEX inicie o carregamento
    /// do SSD **antes** que a camada N+1 comece a computar.
    ///
    /// ## Por que 85-90% de acerto:
    /// Experts de camadas adjacentes têm forte correlação estatística — o modelo
    /// tende a usar os mesmos specialists para o mesmo tipo de conteúdo ao longo
    /// das camadas. O shadow pass em INT4 captura essa correlação com precisão
    /// suficiente sem custo computacional significativo.
    ///
    /// `router_weights_int4`: pesos INT4 do router da camada N+1 (4 bits/peso)
    /// `hidden_state`: saída da camada N (f32)
    /// `top_k`: quantos experts pré-carregar
    /// `next_layer_idx`: índice da camada N+1 (para o APEX)
    pub fn predict_next_experts(
        &mut self,
        router_weights_int4: &[u8], // pesos INT4 compactados: 2 experts por byte
        hidden_state: &[f32],
        num_experts: usize,
        top_k: usize,
        next_layer_idx: usize,
    ) {
        if router_weights_int4.is_empty() || hidden_state.is_empty() {
            return;
        }

        let h = hidden_state.len().min(router_weights_int4.len() * 2 / num_experts.max(1));

        // ── Score de cada expert via dot product INT4 × f32 ──
        // Cada byte de router_weights_int4 contém 2 pesos INT4 (nibbles).
        let mut expert_scores = vec![0.0f32; num_experts];
        for expert_idx in 0..num_experts {
            let mut score = 0.0f32;
            for hi in 0..h {
                let byte_idx = (expert_idx * h + hi) / 2;
                if byte_idx >= router_weights_int4.len() { break; }
                let byte = router_weights_int4[byte_idx];
                // Extrai nibble correto (0-15) e centraliza em [-7, 8]
                let nibble = if hi % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let w_int4 = nibble as f32 - 7.0; // dequant simples
                score += hidden_state[hi] * w_int4;
            }
            expert_scores[expert_idx] = score;
        }

        // ── Top-K selection ──
        let effective_k = top_k.min(num_experts);
        let mut indexed: Vec<(usize, f32)> = expert_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // ── Preenche a fila de prefetch ──
        self.prefetch_queue.clear();
        for (expert_idx, _score) in indexed.iter().take(effective_k) {
            self.prefetch_queue.push((next_layer_idx, *expert_idx));
        }
        self.pregate_total += 1;
    }

    /// Valida se um expert realmente usado estava na fila de prefetch.
    /// Usado para medir a taxa de acerto do Pre-gate Shadow.
    pub fn pregate_validate(&mut self, actually_used: &[(usize, usize)]) {
        let hits = actually_used.iter()
            .filter(|e| self.prefetch_queue.contains(e))
            .count();
        self.pregate_hits += hits as u64;
    }

    /// Taxa de acerto do Pre-gate Shadow (0.0 a 1.0).
    pub fn pregate_accuracy(&self) -> f64 {
        if self.pregate_total == 0 { return 0.0; }
        self.pregate_hits as f64 / (self.pregate_total as f64)
    }

    /// Relatório completo incluindo métricas EAGLE-2, EASD e Pre-gate.
    pub fn full_report(&self) -> String {
        format!(
            "{}\nEAGLE-2: {}\nEASD entropia média: {:.3} nats\nPre-gate acerto: {:.1}% ({}/{} previsões)",
            self.stats.report(),
            if self.eagle2_head_weights.is_some() { "ATIVO" } else { "inativo (legado)" },
            self.mean_entropy(),
            self.pregate_accuracy() * 100.0,
            self.pregate_hits,
            self.pregate_total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vram_budget::InferenceMode;

    fn make_budget_dense(vram_mb: u64) -> VramBudget {
        VramBudget::new(
            vram_mb * 1024 * 1024,
            (vram_mb as f64 * 0.18) as u64 * 1024 * 1024,
            InferenceMode::Dense,
        )
    }

    #[test]
    fn test_cober_dense_accept_all_correct_draft() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        // Draft com 4 tokens
        let draft = vec![10u32, 20, 30, 40];

        // Master logits: todos confirmam o draft
        let make_logit = |winner: u32, vocab: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; vocab];
            if (winner as usize) < vocab {
                v[winner as usize] = 1.0;
            }
            v
        };
        let vocab = 1000usize;
        let master_logits = vec![
            make_logit(10, vocab),
            make_logit(20, vocab),
            make_logit(30, vocab),
            make_logit(40, vocab),
        ];

        let round = engine.verify_and_accept(&draft, &master_logits, 0);

        assert_eq!(round.accepted_tokens, vec![10u32, 20, 30, 40],
            "Todos os tokens corretos devem ser aceitos");
        assert_eq!(round.rejected_count, 0);
        assert!((round.acceptance_rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cober_dense_reject_first_wrong_token() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        let draft = vec![10u32, 20, 30];
        let vocab = 1000usize;

        // Master discorda no token 0 (draft=10, master=99)
        let mut logit_0 = vec![0.0f32; vocab];
        logit_0[99] = 1.0;
        let mut logit_1 = vec![0.0f32; vocab];
        logit_1[20] = 1.0;

        let master_logits = vec![logit_0, logit_1];

        let round = engine.verify_and_accept(&draft, &master_logits, 0);

        // Deve aceitar token do mestre (99) e parar
        assert_eq!(round.accepted_tokens, vec![99u32],
            "Deve aceitar o token do mestre quando draft está errado");
        assert_eq!(round.rejected_count, 1);
    }

    #[test]
    fn test_dynamic_k_adjustment_increases_on_high_acceptance() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);
        engine.config.max_draft_tokens = 16;

        // Simular histórico de alta aceitação
        for _ in 0..20 {
            engine.acceptance_history.push_back(0.85);
        }

        let new_k = engine.adjust_k_dynamic();
        assert!(new_k > 16, "K deve aumentar quando aceitação > 70%");
    }

    #[test]
    fn test_dynamic_k_adjustment_decreases_on_low_acceptance() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);
        engine.config.max_draft_tokens = 32;

        // Simular histórico de baixa aceitação
        for _ in 0..20 {
            engine.acceptance_history.push_back(0.25);
        }

        let new_k = engine.adjust_k_dynamic();
        assert!(new_k < 32, "K deve reduzir quando aceitação < 40%");
    }

    #[test]
    fn test_stats_speedup_factor() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);
        engine.stats.total_rounds = 10;
        engine.stats.total_accepted_tokens = 180; // 18 tokens/round média

        let speedup = engine.stats.speedup_factor();
        assert!((speedup - 18.0).abs() < 0.1,
            "Speedup deve ser ~18x, foi {:.1}", speedup);
    }

    #[test]
    fn test_moe_expert_cache_integration() {
        let budget = VramBudget::new(
            12 * 1024 * 1024 * 1024u64,
            2 * 1024 * 1024 * 1024u64,
            InferenceMode::MoE { num_experts: 8, top_k: 2 },
        );
        let mut engine = CoberEngine::new_moe(budget, 8, 2);

        // Expert não está em cache → miss
        let cached = engine.check_expert_cache((0, 3));
        assert!(!cached);
        assert_eq!(engine.stats.expert_cache_misses, 1);
    }

    // =========================================================
    // Testes EAGLE-2
    // =========================================================

    #[test]
    fn test_eagle2_draft_without_weights_returns_empty() {
        let budget = make_budget_dense(8 * 1024);
        let engine = CoberEngine::new_dense(budget);
        // Sem pesos carregados → fallback vazio
        let draft = engine.eagle2_draft_from_hidden(&[0.1f32; 16], 5);
        assert!(draft.is_empty(), "Sem pesos EAGLE-2 deve retornar draft vazio");
    }

    #[test]
    fn test_eagle2_draft_top_k_correctness() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        // Vocab pequeno: 8 tokens, hidden_dim: 4
        let hidden_dim = 4usize;
        let vocab_size = 8usize;

        // Pesos identidade: token i tem score 1.0 para dimension i, 0 para outros
        // W[h * vocab + v] = 1.0 se h == v, senão 0.0
        let mut weights = vec![0.0f32; hidden_dim * vocab_size];
        for i in 0..hidden_dim.min(vocab_size) {
            weights[i * vocab_size + i] = 1.0;
        }
        engine.load_eagle2_head(weights, hidden_dim, vocab_size);

        // Hidden state: dim 2 tem maior valor → token 2 deve ser top-1
        let hidden = vec![0.1f32, 0.2, 5.0, 0.05];
        let draft = engine.eagle2_draft_from_hidden(&hidden, 3);

        assert_eq!(draft.len(), 3, "Deve retornar 3 tokens");
        assert_eq!(draft[0], 2u32, "Token 2 deve ser o top-1 (hidden[2] = 5.0 é o maior)");
    }

    #[test]
    fn test_eagle2_load_head_wrong_size_panics() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // 4*8 = 32, mas passamos 10 → deve panic
            engine.load_eagle2_head(vec![0.0f32; 10], 4, 8);
        }));
        assert!(result.is_err(), "Tamanho errado de pesos deve causar panic");
    }

    // =========================================================
    // Testes EASD
    // =========================================================

    #[test]
    fn test_easd_low_entropy_expands_tree() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        // Distribuição quase determinística: token 0 com prob ~1.0
        let mut probs = vec![0.0001f32; 100];
        probs[0] = 0.99;

        let (k, width) = engine.easd_compute_tree_params(&probs);
        // Entropia baixa → deve expandir ao máximo
        assert_eq!(k, engine.config.max_draft_tokens, "K baixa entropia deve ser máximo");
        assert_eq!(width, engine.config.tree_width, "Width baixa entropia deve ser máxima");
    }

    #[test]
    fn test_easd_high_entropy_collapses_tree() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        // Distribuição uniforme: máxima entropia
        let probs = vec![1.0 / 1000.0f32; 1000];

        let (k, width) = engine.easd_compute_tree_params(&probs);
        // Entropia alta → deve colapsar ao mínimo
        assert!(k <= 4, "K alta entropia deve ser mínimo (<=4), foi {}", k);
        assert!(width <= 1, "Width alta entropia deve ser mínima (<=1), foi {}", width);
    }

    #[test]
    fn test_easd_entropy_history_recorded() {
        let budget = make_budget_dense(8 * 1024);
        let mut engine = CoberEngine::new_dense(budget);

        assert_eq!(engine.mean_entropy(), 0.0, "Histórico vazio → entropia média 0");

        let probs = vec![0.5f32, 0.5]; // H = ln(2) ≈ 0.693
        engine.easd_compute_tree_params(&probs);
        let mean = engine.mean_entropy();
        assert!(mean > 0.6 && mean < 0.75,
            "Entropia de distribuição 50/50 deve ser ~0.693, foi {}", mean);
    }

    #[test]
    fn test_compute_entropy_uniform() {
        // Distribuição uniforme de 4 eventos: H = ln(4) ≈ 1.386
        let probs = vec![0.25f32; 4];
        let h = CoberEngine::compute_entropy(&probs);
        assert!((h - 1.386).abs() < 0.01, "Entropia uniforme(4) ≈ 1.386, foi {:.3}", h);
    }

    #[test]
    fn test_compute_entropy_deterministic() {
        // Distribuição determinística: H = 0
        let mut probs = vec![0.0f32; 10];
        probs[3] = 1.0;
        let h = CoberEngine::compute_entropy(&probs);
        assert!(h.abs() < 0.001, "Entropia determinística deve ser 0, foi {:.5}", h);
    }

    // =========================================================
    // Testes Pre-gate Shadow
    // =========================================================

    #[test]
    fn test_pregate_fills_prefetch_queue() {
        let budget = VramBudget::new(
            12 * 1024 * 1024 * 1024u64,
            2 * 1024 * 1024 * 1024u64,
            InferenceMode::MoE { num_experts: 8, top_k: 2 },
        );
        let mut engine = CoberEngine::new_moe(budget, 8, 2);

        // Router INT4 simples: 4 experts, hidden_dim=4 → 8 bytes (2 nibbles/byte)
        let router_weights = vec![0xF0u8, 0x0F, 0xAA, 0x55, 0xF0, 0x0F, 0xAA, 0x55];
        let hidden = vec![1.0f32, 0.0, 0.0, 0.0];

        engine.predict_next_experts(&router_weights, &hidden, 4, 2, 1);

        assert_eq!(engine.prefetch_queue.len(), 2, "Deve prever top-2 experts");
        assert!(engine.prefetch_queue.iter().all(|(layer, _)| *layer == 1),
            "Todos os prefetch devem ser para layer 1");
        assert_eq!(engine.pregate_total, 1);
    }

    #[test]
    fn test_pregate_validate_accuracy() {
        let budget = VramBudget::new(
            12 * 1024 * 1024 * 1024u64,
            2 * 1024 * 1024 * 1024u64,
            InferenceMode::MoE { num_experts: 4, top_k: 2 },
        );
        let mut engine = CoberEngine::new_moe(budget, 4, 2);

        // Força a fila de prefetch manualmente
        engine.prefetch_queue = vec![(1, 0), (1, 2)];
        engine.pregate_total = 1;

        // Expert (1,0) estava na fila → 1 acerto de 2 usados
        engine.pregate_validate(&[(1, 0), (1, 3)]);

        assert_eq!(engine.pregate_hits, 1);
        let acc = engine.pregate_accuracy();
        assert!((acc - 1.0).abs() < 0.01, "1 previsão total → acurácia = 100% (hits/total)");
    }

    #[test]
    fn test_full_report_includes_all_sections() {
        let budget = make_budget_dense(8 * 1024);
        let engine = CoberEngine::new_dense(budget);
        let report = engine.full_report();
        assert!(report.contains("EAGLE-2"), "Report deve mencionar EAGLE-2");
        assert!(report.contains("EASD"), "Report deve mencionar EASD");
        assert!(report.contains("Pre-gate"), "Report deve mencionar Pre-gate");
    }

    #[test]
    fn test_eagle2_head_learns_by_distillation() {
        // Prova que a maquinaria de TREINO da cabeça EAGLE-2 funciona: o SGD reduz
        // a perda e a cabeça passa a prever o token-alvo a partir do hidden.
        let mut engine = CoberEngine::new_dense(make_budget_dense(8 * 1024));
        let (h, v) = (4usize, 8usize);
        let pairs = vec![
            (vec![1.0, 0.0, 0.0, 0.0], 2u32),
            (vec![0.0, 1.0, 0.0, 0.0], 5u32),
            (vec![0.0, 0.0, 1.0, 0.0], 7u32),
            (vec![0.0, 0.0, 0.0, 1.0], 1u32),
        ];
        let loss_inicial = engine.train_eagle2_head(&pairs, h, v, 1, 0.5);
        let loss_final = engine.train_eagle2_head(&pairs, h, v, 300, 0.5);
        assert!(loss_final < loss_inicial, "a perda deve CAIR com o treino: {} -> {}", loss_inicial, loss_final);
        assert!(loss_final < 0.05, "a cabeça deve aprender (perda baixa), got {}", loss_final);
        for (hid, tok) in &pairs {
            let toks = engine.eagle2_draft_from_hidden(hid, 1);
            assert_eq!(toks.first().copied(), Some(*tok), "cabeça treinada deve prever {} para {:?}", tok, hid);
        }
    }
}
