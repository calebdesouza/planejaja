/// COBER — Motor de Candidatos: HNSW + BM25 + RRF Fusion.
///
/// Gera até `max_candidates` tokens candidatos com qualidade superior ao N-gram:
/// - HNSW (LanceDB): busca semântica no espaço de embeddings
/// - BM25: busca léxica exata no histórico de tokens
/// - RRF: fusão dos dois rankings num único score

use std::collections::{HashMap, VecDeque};

/// Candidato gerado pelo motor.
#[derive(Debug, Clone)]
pub struct TokenCandidate {
    /// ID do token candidato.
    pub token_id: u32,
    /// Score final RRF (maior = melhor).
    pub rrf_score: f32,
    /// Se surgiu via HNSW (semântico).
    pub from_hnsw: bool,
    /// Se surgiu via BM25 (léxico).
    pub from_bm25: bool,
}

/// Configuração do motor de candidatos.
#[derive(Debug, Clone)]
pub struct CandidateConfig {
    /// Peso do ranking HNSW no RRF (padrão: 0.6).
    pub hnsw_weight: f32,
    /// Peso do ranking BM25 no RRF (padrão: 0.4).
    pub bm25_weight: f32,
    /// Constante de suavização RRF (padrão: 60.0).
    pub rrf_k: f32,
    /// Número máximo de candidatos a retornar.
    pub max_candidates: usize,
    /// Janela de histórico de tokens para BM25.
    pub history_window: usize,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            hnsw_weight: 0.6,
            bm25_weight: 0.4,
            rrf_k: 60.0,
            max_candidates: 32,
            history_window: 2048,
        }
    }
}

/// Motor de candidatos anti-alucinação.
pub struct CandidateEngine {
    pub config: CandidateConfig,
    pub bm25: Bm25Index,
    pub token_history: VecDeque<u32>,
}

impl CandidateEngine {
    /// Cria um novo motor com configuração padrão.
    pub fn new() -> Self {
        Self::with_config(CandidateConfig::default())
    }

    /// Cria um novo motor com configuração personalizada.
    pub fn with_config(config: CandidateConfig) -> Self {
        Self {
            bm25: Bm25Index::new(config.history_window),
            token_history: VecDeque::with_capacity(config.history_window),
            config,
        }
    }

    /// Ingere um token gerado/verificado pelo modelo mestre.
    pub fn ingest_token(&mut self, token_id: u32) {
        if self.token_history.len() >= self.config.history_window {
            self.token_history.pop_front();
        }
        self.token_history.push_back(token_id);
        self.bm25.add_token(token_id);
    }

    /// Ingere múltiplos tokens de contexto (prompt).
    pub fn ingest_context(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.ingest_token(t);
        }
    }

    /// Gera candidatos usando HNSW + BM25 + RRF.
    ///
    /// `hnsw_results`: lista ordenada (melhor primeiro) de token_ids do HNSW.
    /// `query_tokens`: tokens para query BM25.
    pub fn generate_candidates(
        &self,
        hnsw_results: &[u32],
        query_tokens: &[u32],
    ) -> Vec<TokenCandidate> {
        // BM25 retorna tokens rankeados por frequência/relevância
        let bm25_results = self.bm25.query(query_tokens, self.config.max_candidates * 2);

        // Construir mapa de ranks HNSW: token → rank (0-indexed)
        let mut hnsw_ranks: HashMap<u32, usize> = HashMap::new();
        for (rank, &token) in hnsw_results.iter().enumerate() {
            hnsw_ranks.insert(token, rank);
        }

        // Construir mapa de ranks BM25
        let mut bm25_ranks: HashMap<u32, usize> = HashMap::new();
        for (rank, &token) in bm25_results.iter().enumerate() {
            bm25_ranks.insert(token, rank);
        }

        // União de todos os tokens candidatos
        let mut all_tokens: std::collections::HashSet<u32> = HashSet::new();
        all_tokens.extend(hnsw_results.iter().copied());
        all_tokens.extend(bm25_results.iter().copied());

        // Calcular RRF score para cada token
        let k = self.config.rrf_k;
        let hw = self.config.hnsw_weight;
        let bw = self.config.bm25_weight;

        let mut candidates: Vec<TokenCandidate> = all_tokens
            .into_iter()
            .map(|token| {
                let hnsw_score = hnsw_ranks
                    .get(&token)
                    .map(|&r| hw / (k + r as f32))
                    .unwrap_or(0.0);
                let bm25_score = bm25_ranks
                    .get(&token)
                    .map(|&r| bw / (k + r as f32))
                    .unwrap_or(0.0);
                let rrf_score = hnsw_score + bm25_score;

                TokenCandidate {
                    token_id: token,
                    rrf_score,
                    from_hnsw: hnsw_ranks.contains_key(&token),
                    from_bm25: bm25_ranks.contains_key(&token),
                }
            })
            .collect();

        // Ordenar por RRF score (maior primeiro)
        candidates.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);
        candidates
    }

    /// Retorna os IDs dos candidatos como slice.
    pub fn candidate_ids(candidates: &[TokenCandidate]) -> Vec<u32> {
        candidates.iter().map(|c| c.token_id).collect()
    }
}

impl Default for CandidateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─── BM25 In-Memory ──────────────────────────────────────────────────────────

use std::collections::HashSet;

/// Índice BM25 leve in-memory sobre histórico de tokens.
///
/// BM25 (Best Matching 25): ranqueia tokens por frequência relativa
/// no histórico de contexto, garantindo que termos exatos como
/// variáveis, nomes e códigos apareçam corretamente.
pub struct Bm25Index {
    /// Frequência de cada token no histórico.
    term_freq: HashMap<u32, u32>,
    /// Tokens em ordem histórica (para sliding window).
    history: VecDeque<u32>,
    /// Tamanho da janela do histórico.
    window_size: usize,
    /// Parâmetros BM25.
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new(window_size: usize) -> Self {
        Self {
            term_freq: HashMap::new(),
            history: VecDeque::with_capacity(window_size),
            window_size,
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Adiciona um token ao índice.
    pub fn add_token(&mut self, token_id: u32) {
        if self.history.len() >= self.window_size {
            if let Some(old) = self.history.pop_front() {
                if let Some(freq) = self.term_freq.get_mut(&old) {
                    *freq = freq.saturating_sub(1);
                    if *freq == 0 {
                        self.term_freq.remove(&old);
                    }
                }
            }
        }
        self.history.push_back(token_id);
        *self.term_freq.entry(token_id).or_insert(0) += 1;
    }

    /// Ranqueia tokens pelo score BM25 dado um array de query tokens.
    pub fn query(&self, query_tokens: &[u32], top_k: usize) -> Vec<u32> {
        if query_tokens.is_empty() || self.history.is_empty() {
            return Vec::new();
        }

        let avg_doc_len = self.history.len() as f32;
        let doc_len = self.history.len() as f32;

        // Calcular score BM25 para cada token único no histórico
        let mut scores: Vec<(u32, f32)> = self
            .term_freq
            .iter()
            .filter_map(|(&token, &freq)| {
                // Verifica se o token é relevante para a query
                if query_tokens.contains(&token) || freq > 2 {
                    let tf = freq as f32;
                    let normalized_tf = tf * (self.k1 + 1.0)
                        / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / avg_doc_len));
                    // IDF simplificado: log(1 + 1/freq) — tokens raros têm IDF maior
                    let idf = (1.0 + 1.0 / tf).ln() + 1.0;
                    Some((token, normalized_tf * idf))
                } else {
                    None
                }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores.into_iter().map(|(t, _)| t).collect()
    }

    /// Número de tokens únicos no índice.
    pub fn unique_terms(&self) -> usize {
        self.term_freq.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_ranks_frequent_tokens_higher() {
        let mut idx = Bm25Index::new(1024);
        // Token 42 aparece 5x, token 99 aparece 1x
        for _ in 0..5 { idx.add_token(42); }
        idx.add_token(99);

        let results = idx.query(&[42, 99], 10);
        assert!(!results.is_empty(), "BM25 deve retornar resultados");
        // Token 42 deve estar antes de token 99 (mais frequente = mais relevante)
        let pos_42 = results.iter().position(|&t| t == 42);
        let pos_99 = results.iter().position(|&t| t == 99);
        if let (Some(p42), Some(p99)) = (pos_42, pos_99) {
            assert!(p42 < p99, "Token 42 (5x) deve rankear antes de 99 (1x)");
        }
    }

    #[test]
    fn test_rrf_fusion_combines_both_sources() {
        let engine = CandidateEngine::new();
        let hnsw = vec![100u32, 200, 300];
        let query = &[100u32];
        // Manualmente popular BM25
        let mut bm25 = Bm25Index::new(100);
        for _ in 0..3 { bm25.add_token(200); }
        bm25.add_token(400);

        // Test básico: RRF deve retornar candidatos de ambas as fontes
        let engine2 = CandidateEngine {
            config: CandidateConfig::default(),
            bm25,
            token_history: VecDeque::new(),
        };
        let candidates = engine2.generate_candidates(&hnsw, query);
        assert!(!candidates.is_empty(), "RRF deve gerar candidatos");
        assert!(candidates.len() <= 32, "Deve respeitar max_candidates=32");
        // Scores devem estar em ordem decrescente
        for w in candidates.windows(2) {
            assert!(w[0].rrf_score >= w[1].rrf_score,
                "Candidatos devem estar ordenados por score decrescente");
        }
    }

    #[test]
    fn test_ingest_and_sliding_window() {
        let mut engine = CandidateEngine::with_config(CandidateConfig {
            history_window: 5,
            ..Default::default()
        });
        for i in 0..10u32 {
            engine.ingest_token(i);
        }
        // Janela de 5: deve ter apenas tokens 5..9
        assert_eq!(engine.token_history.len(), 5);
        assert!(!engine.token_history.contains(&0), "Token 0 deveria ter saído da janela");
        assert!(engine.token_history.contains(&9), "Token 9 deve estar na janela");
    }
}
