use std::collections::HashMap;

/// NodeStor COBER v2 - Subsistema 11: Multi-Draft Paralelo
/// ("Explosão de Caminhos" / Navegador de Grafos de Possibilidades)
///
/// Inspirado no SpecInfer (ASPLOS'24) e SpecHub (EMNLP'24).
/// Em vez de sugerir apenas 1 caminho, projeta N caminhos lógicos diferentes
/// ao mesmo tempo. A GPU não está mais escolhendo "a próxima palavra",
/// ela está escolhendo o "melhor futuro".
///
/// A IA descarta 9 caminhos comuns e escolhe o 10º que é uma conexão
/// entre dois fatos que nenhum humano percebeu antes.
/// A escrita é apenas o relatório final dessa escolha.

/// Um caminho lógico completo com sua pontuação
#[derive(Debug, Clone)]
pub struct DraftPath {
    /// ID do caminho
    pub id: usize,
    /// Tokens deste caminho candidato
    pub tokens: Vec<u32>,
    /// Fonte de onde veio (qual subsistema gerou)
    pub source: DraftSource,
    /// Confiança estimada baseada na fonte (0.0 - 1.0)
    pub confidence: f32,
    /// Embedding semântico do caminho (para deduplicação)
    pub semantic_embedding: Vec<f32>,
    /// Pontuação de "novidade" — caminhos que NÃO existem ainda são premiados
    pub novelty_score: f32,
}

/// Origem do rascunho (para tracking e aprendizado)
#[derive(Debug, Clone, PartialEq)]
pub enum DraftSource {
    /// Veio do cache L1 (memória muscular)
    GoldenCache,
    /// Veio do Prompt Lookup (N-Gram match)
    PromptLookup,
    /// Veio da REST Trie (Retrieval-Based)
    RestTrie,
    /// Veio das Medusa Heads (Ancorada)
    AnchoredMedusa,
    /// Veio do Esqueleto Sintático (Modo Criativo)
    SyntacticSkeleton,
    /// Veio do Insight Indexer (Caminho já descoberto)
    InsightRecall,
    /// Veio do Cross-Domain (Cruzamento Inédito)
    CrossDomain,
}

/// Resultado da explosão multi-draft após verificação GPU
#[derive(Debug)]
pub struct MultiDraftResult {
    /// O caminho vencedor (validado pelo modelo mestre)
    pub winner: DraftPath,
    /// Índice do vencedor na lista original
    pub winner_index: usize,
    /// Todos os caminhos descartados com razão
    pub rejected_paths: Vec<(DraftPath, String)>,
    /// Se houve "insight inédito" (cruzamento cross-domain)
    pub is_novel_insight: bool,
    /// Taxa de aceitação desta rodada multi-draft
    pub acceptance_rate: f32,
}

/// O Explorador de Caminhos Multi-Draft
pub struct MultiDraftExplorer {
    /// Número máximo de caminhos paralelos
    pub max_parallel_paths: usize,
    /// Limiar de novidade: caminhos com novelty_score > threshold são premiados
    pub novelty_threshold: f32,
    /// Limiar de deduplicação cosseno: caminhos com sim > threshold são fundidos
    pub dedup_threshold: f32,
    /// Estatísticas acumuladas
    pub stats: MultiDraftStats,
}

#[derive(Debug, Default)]
pub struct MultiDraftStats {
    pub total_rounds: u64,
    pub total_paths_explored: u64,
    pub novel_insights_found: u64,
    pub golden_wins: u64,
    pub prompt_lookup_wins: u64,
    pub rest_trie_wins: u64,
    pub medusa_wins: u64,
    pub skeleton_wins: u64,
    pub insight_wins: u64,
    pub cross_domain_wins: u64,
}

impl MultiDraftExplorer {
    pub fn new(max_parallel_paths: usize) -> Self {
        Self {
            max_parallel_paths,
            novelty_threshold: 0.3,
            dedup_threshold: 0.95,
            stats: MultiDraftStats::default(),
        }
    }

    /// Coleta caminhos de todos os subsistemas do Esqueleto de Cristal
    /// e os organiza para verificação paralela na GPU.
    pub fn collect_paths(&self, raw_paths: Vec<DraftPath>) -> Vec<DraftPath> {
        if raw_paths.is_empty() {
            return Vec::new();
        }

        let mut paths = raw_paths;

        // 1. Deduplicação: remove caminhos quase idênticos
        paths = self.deduplicate(paths);

        // 2. Ordenar por confiança ponderada por novidade
        paths.sort_by(|a, b| {
            let score_a = a.confidence + a.novelty_score * 0.5;
            let score_b = b.confidence + b.novelty_score * 0.5;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Truncar ao máximo de caminhos paralelos
        paths.truncate(self.max_parallel_paths);

        paths
    }

    /// Seleciona o vencedor após verificação GPU.
    /// `verification_scores[i]` = quantos tokens do caminho i foram aceitos.
    pub fn select_winner(
        &mut self,
        paths: Vec<DraftPath>,
        verification_scores: &[usize],
    ) -> MultiDraftResult {
        self.stats.total_rounds += 1;
        self.stats.total_paths_explored += paths.len() as u64;

        if paths.is_empty() || verification_scores.is_empty() {
            return MultiDraftResult {
                winner: DraftPath {
                    id: 0,
                    tokens: vec![],
                    source: DraftSource::PromptLookup,
                    confidence: 0.0,
                    semantic_embedding: vec![],
                    novelty_score: 0.0,
                },
                winner_index: 0,
                rejected_paths: vec![],
                is_novel_insight: false,
                acceptance_rate: 0.0,
            };
        }

        // Encontrar o caminho com mais tokens aceitos
        let winner_index = verification_scores
            .iter()
            .enumerate()
            .max_by_key(|(_, &score)| score)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let is_novel = paths[winner_index].novelty_score > self.novelty_threshold;
        if is_novel {
            self.stats.novel_insights_found += 1;
        }

        // Rastrear qual subsistema venceu
        match &paths[winner_index].source {
            DraftSource::GoldenCache => self.stats.golden_wins += 1,
            DraftSource::PromptLookup => self.stats.prompt_lookup_wins += 1,
            DraftSource::RestTrie => self.stats.rest_trie_wins += 1,
            DraftSource::AnchoredMedusa => self.stats.medusa_wins += 1,
            DraftSource::SyntacticSkeleton => self.stats.skeleton_wins += 1,
            DraftSource::InsightRecall => self.stats.insight_wins += 1,
            DraftSource::CrossDomain => self.stats.cross_domain_wins += 1,
        }

        let total_possible: usize = paths.iter().map(|p| p.tokens.len()).sum();
        let total_accepted: usize = verification_scores.iter().sum();
        let acceptance_rate = if total_possible > 0 {
            total_accepted as f32 / total_possible as f32
        } else {
            0.0
        };

        // Separar vencedor dos rejeitados
        let mut rejected_paths = Vec::new();
        let mut winner_path = None;

        for (i, path) in paths.into_iter().enumerate() {
            if i == winner_index {
                winner_path = Some(path);
            } else {
                let reason = if verification_scores.get(i).copied().unwrap_or(0) == 0 {
                    "Todos tokens rejeitados pelo mestre".to_string()
                } else {
                    format!("Aceitos apenas {}/{} tokens", 
                        verification_scores.get(i).unwrap_or(&0),
                        path.tokens.len()
                    )
                };
                rejected_paths.push((path, reason));
            }
        }

        MultiDraftResult {
            winner: winner_path.unwrap(),
            winner_index,
            rejected_paths,
            is_novel_insight: is_novel,
            acceptance_rate,
        }
    }

    /// Deduplicação via similaridade cosseno dos embeddings semânticos
    fn deduplicate(&self, paths: Vec<DraftPath>) -> Vec<DraftPath> {
        if paths.len() <= 1 {
            return paths;
        }

        let mut unique: Vec<DraftPath> = Vec::new();
        'outer: for path in paths {
            for existing in &unique {
                if Self::cosine_sim(&path.semantic_embedding, &existing.semantic_embedding) 
                    > self.dedup_threshold 
                {
                    continue 'outer;
                }
            }
            unique.push(path);
        }
        unique
    }

    /// Calcula a "Reorientação da Bússola" quando o modelo mestre rejeita.
    /// Em vez de simplesmente corrigir a palavra, injeta a direção semântica
    /// do token correto do mestre para que o N-Gram de Ouro aprenda o novo rumo.
    pub fn compute_compass_redirect(
        rejected_embedding: &[f32],
        master_correction_embedding: &[f32],
    ) -> Vec<f32> {
        // O vetor de redireção é a diferença entre onde estávamos indo
        // e onde o mestre nos mandou ir. Isso é a "costura invisível" —
        // parece que a IA "teve um insight genial".
        let len = rejected_embedding.len().min(master_correction_embedding.len());
        let mut redirect = vec![0.0f32; len];
        for i in 0..len {
            redirect[i] = master_correction_embedding[i] - rejected_embedding[i];
        }
        redirect
    }

    /// Relatório de desempenho multi-draft
    pub fn stats_report(&self) -> String {
        format!(
            "MultiDraft Stats:\n\
             Rounds: {}\n\
             Paths Explored: {}\n\
             Novel Insights: {}\n\
             Winner Sources: Golden={} PL={} REST={} Medusa={} Skeleton={} Insight={} Cross={}",
            self.stats.total_rounds,
            self.stats.total_paths_explored,
            self.stats.novel_insights_found,
            self.stats.golden_wins,
            self.stats.prompt_lookup_wins,
            self.stats.rest_trie_wins,
            self.stats.medusa_wins,
            self.stats.skeleton_wins,
            self.stats.insight_wins,
            self.stats.cross_domain_wins,
        )
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() { return 0.0; }
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let d = na.sqrt() * nb.sqrt();
        if d < 1e-10 { 0.0 } else { dot / d }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path(id: usize, tokens: Vec<u32>, source: DraftSource, conf: f32, novelty: f32, embed: Vec<f32>) -> DraftPath {
        DraftPath {
            id, tokens, source, confidence: conf, semantic_embedding: embed, novelty_score: novelty
        }
    }

    #[test]
    fn test_multi_draft_explosion_and_selection() {
        let mut explorer = MultiDraftExplorer::new(10);

        let paths = vec![
            make_path(0, vec![1, 2, 3], DraftSource::GoldenCache, 0.95, 0.0, vec![1.0, 0.0]),
            make_path(1, vec![4, 5, 6, 7], DraftSource::RestTrie, 0.70, 0.1, vec![0.0, 1.0]),
            make_path(2, vec![8, 9], DraftSource::CrossDomain, 0.50, 0.9, vec![0.5, 0.5]),
        ];

        let collected = explorer.collect_paths(paths);
        assert_eq!(collected.len(), 3);
        // Golden deve liderar por confiança pura
        assert_eq!(collected[0].source, DraftSource::GoldenCache);

        // GPU verifica: Caminho 2 (Cross-Domain) acertou 2 tokens, 
        // Caminho 0 (Golden) acertou 3, Caminho 1 (REST) acertou 1
        let verification_scores = vec![3, 1, 2];
        let result = explorer.select_winner(collected, &verification_scores);

        // Golden ganhou com 3 tokens aceitos
        assert_eq!(result.winner.source, DraftSource::GoldenCache);
        assert_eq!(result.rejected_paths.len(), 2);
        assert_eq!(explorer.stats.golden_wins, 1);
    }

    #[test]
    fn test_novel_insight_detection() {
        let mut explorer = MultiDraftExplorer::new(5);
        explorer.novelty_threshold = 0.3;

        // Use orthogonal 2d embeddings so dedup doesn't merge them
        let paths = vec![
            make_path(0, vec![1, 2], DraftSource::PromptLookup, 0.9, 0.0, vec![1.0, 0.0]),
            make_path(1, vec![3, 4, 5], DraftSource::CrossDomain, 0.6, 0.8, vec![0.0, 1.0]),
        ];

        let collected = explorer.collect_paths(paths);
        assert_eq!(collected.len(), 2);
        // After sort: CrossDomain(score=0.6+0.8*0.5=1.0) at idx 0, PromptLookup(0.9) at idx 1
        // Give CrossDomain (idx 0) more accepted tokens
        let result = explorer.select_winner(collected, &[3, 1]);

        assert!(result.is_novel_insight);
        assert_eq!(explorer.stats.novel_insights_found, 1);
        assert_eq!(explorer.stats.cross_domain_wins, 1);
    }

    #[test]
    fn test_deduplication() {
        let explorer = MultiDraftExplorer::new(10);

        let paths = vec![
            make_path(0, vec![1, 2], DraftSource::GoldenCache, 0.9, 0.0, vec![1.0, 0.0]),
            make_path(1, vec![1, 2], DraftSource::PromptLookup, 0.8, 0.0, vec![1.0, 0.001]),
            make_path(2, vec![3, 4], DraftSource::RestTrie, 0.7, 0.0, vec![0.0, 1.0]),
        ];

        let deduped = explorer.deduplicate(paths);
        assert_eq!(deduped.len(), 2); // path 0 e 1 são quase iguais
    }

    #[test]
    fn test_compass_redirect() {
        let rejected = vec![1.0, 0.0, 0.0];
        let master = vec![0.0, 1.0, 0.0];

        let redirect = MultiDraftExplorer::compute_compass_redirect(&rejected, &master);
        assert_eq!(redirect, vec![-1.0, 1.0, 0.0]);
    }
}
