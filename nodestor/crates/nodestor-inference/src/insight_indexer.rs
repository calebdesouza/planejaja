use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, Mutex};
use nodestor_streaming::apex::AmbientTask;

/// NodeStor COBER v2 - Subsistema 10: Indexador de Insights
/// (Grafo de Sinapses Persistentes)
///
/// Em vez de apenas guardar o texto que a IA escreveu, o COBER v2 indexa
/// o CAMINHO LÓGICO que levou àquela ideia: a Pergunta, a Lógica de
/// Raciocínio, e o Resultado Final.
///
/// Na próxima vez que qualquer ideia passar "perto" dessa vizinhança semântica,
/// o sistema "sente o cheiro" da solução anterior e a puxa como um rascunho
/// de altíssima fidelidade — é como "migalhas de pão de ouro".
///
/// Academicamente: Contextual Memory Intelligence (CMI) + Persistent
/// Synapse Graph (Hebbian Strengthening).

/// Um "Insight" é um caminho lógico completo: Pergunta → Raciocínio → Resposta.
#[derive(Debug, Clone)]
pub struct Insight {
    /// ID único monotônico do insight
    pub id: u64,
    /// Hash semântico da pergunta original
    pub query_hash: u64,
    /// Embedding vetorial do raciocínio (do hidden state final)
    pub reasoning_embedding: Vec<f32>,
    /// Os tokens de resposta que foram validados pelo modelo mestre
    pub verified_tokens: Vec<u32>,
    /// Força da sinapse (Hebbian: quanto mais acessado, mais forte)
    pub strength: f32,
    /// Quantas vezes este insight foi reutilizado (hit count)
    pub hit_count: u64,
    ///Modalidade: T=Texto, I=Imagem, A=Audio, V=Video, X=CrossModal
    pub modality: Modality,
    /// Tags semânticas (ex: "Física", "Biologia", "Contrato")
    pub semantic_tags: Vec<String>,
    /// Links para insights relacionados (Grafo de Sinapses)
    pub synapse_links: Vec<u64>,
}

/// Modalidade do insight para cruzamento cross-modal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    CrossModal, // insight nasceu do cruzamento de 2+ modalidades
}

/// Uma Sinapse conecta dois insights com uma força e direção
#[derive(Debug, Clone)]
pub struct Synapse {
    pub from_insight: u64,
    pub to_insight: u64,
    /// Força Hebbiana: cresce quando ambos são ativados juntos
    pub weight: f32,
    /// Tipo de relação
    pub relation: SynapseType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SynapseType {
    /// A levou a B (causalidade)
    Causal,
    /// A e B se reforçam mutuamente
    Reinforcing,
    /// A contradiz B (útil para detectar inconsistências)
    Contradicting,
    /// A é uma evolução de B
    Evolution,
    /// A cruza domínios com B (ex: Física + Biologia)
    CrossDomain,
}

/// O "Mapa de Buracos": regiões do espaço semântico onde
/// a taxa de erro da IA é alta → candidatas a "Pensamento Profundo"
#[derive(Debug, Clone)]
pub struct KnowledgeGap {
    /// Centróide do "buraco" no espaço vetorial
    pub centroid: Vec<f32>,
    /// Taxa de rejeição do modelo mestre nesta região
    pub rejection_rate: f32,
    /// Número de tentativas de acesso
    pub access_count: u64,
    /// Tags semânticas da região
    pub domain_tags: Vec<String>,
}

/// O Indexador de Insights: Biblioteca de Evolução da IA
pub struct InsightIndexer {
    /// Todos os insights organizados por ID
    pub insights: HashMap<u64, Insight>,
    /// Grafo de sinapses: (from, to) → Synapse
    pub synapses: Vec<Synapse>,
    /// Mapa de buracos do conhecimento
    pub knowledge_gaps: Vec<KnowledgeGap>,
    /// Índice de busca rápida: hash do contexto → lista de insights relevantes
    pub context_index: HashMap<u64, Vec<u64>>,
    /// Contador monotônico de IDs
    next_id: u64,
    /// Limiar de força para poda de sinapses fracas (Hebbian decay)
    pub prune_threshold: f32,
}

impl InsightIndexer {
    pub fn new(prune_threshold: f32) -> Self {
        Self {
            insights: HashMap::new(),
            synapses: Vec::new(),
            knowledge_gaps: Vec::new(),
            context_index: HashMap::new(),
            next_id: 0,
            prune_threshold,
        }
    }

    /// Registra um novo insight quando o modelo mestre valida um caminho complexo.
    pub fn record_insight(
        &mut self,
        query_context: &[u32],
        reasoning_embedding: Vec<f32>,
        verified_tokens: Vec<u32>,
        modality: Modality,
        semantic_tags: Vec<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let query_hash = Self::hash_context(query_context);

        let insight = Insight {
            id,
            query_hash,
            reasoning_embedding,
            verified_tokens,
            strength: 1.0,
            hit_count: 0,
            modality,
            semantic_tags,
            synapse_links: Vec::new(),
        };

        // Indexar no contexto para busca rápida
        self.context_index
            .entry(query_hash)
            .or_insert_with(Vec::new)
            .push(id);

        self.insights.insert(id, insight);
        id
    }

    /// Busca insights próximos semanticamente usando distância cosseno.
    /// Retorna (insight_id, similaridade) ordenado por relevância.
    pub fn search_nearby(
        &mut self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let mut scores: Vec<(u64, f32)> = self.insights
            .iter()
            .map(|(&id, insight)| {
                let sim = Self::cosine_similarity(query_embedding, &insight.reasoning_embedding);
                (id, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        // Reforço Hebbiano: fortalecer insights acessados
        for (id, _sim) in &scores {
            if let Some(insight) = self.insights.get_mut(id) {
                insight.hit_count += 1;
                insight.strength = (insight.strength + 0.1).min(10.0); // Cap em 10
            }
        }

        scores
    }

    /// Busca insights por contexto hash (busca exata, O(1))
    pub fn search_by_context(&mut self, context: &[u32]) -> Vec<&Insight> {
        let hash = Self::hash_context(context);
        if let Some(ids) = self.context_index.get(&hash) {
            ids.iter()
                .filter_map(|id| self.insights.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Cria uma sinapse entre dois insights (Grafo de Sinapses)
    pub fn connect_insights(
        &mut self,
        from_id: u64,
        to_id: u64,
        relation: SynapseType,
    ) {
        // Verifica se ambos existem
        if !self.insights.contains_key(&from_id) || !self.insights.contains_key(&to_id) {
            return;
        }

        let synapse = Synapse {
            from_insight: from_id,
            to_insight: to_id,
            weight: 1.0,
            relation,
        };

        self.synapses.push(synapse);

        // Atualiza links no insight
        if let Some(insight) = self.insights.get_mut(&from_id) {
            if !insight.synapse_links.contains(&to_id) {
                insight.synapse_links.push(to_id);
            }
        }
        if let Some(insight) = self.insights.get_mut(&to_id) {
            if !insight.synapse_links.contains(&from_id) {
                insight.synapse_links.push(from_id);
            }
        }
    }

    /// Cruzamento Cross-Domain: encontra insights de domínios diferentes
    /// que podem gerar uma síntese inédita.
    pub fn find_cross_domain_candidates(
        &self,
        query_embedding: &[f32],
        exclude_domain: &str,
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let mut candidates: Vec<(u64, f32)> = self.insights
            .iter()
            .filter(|(_, insight)| {
                // Exclui o domínio de origem para fomentar cruzamento
                !insight.semantic_tags.iter().any(|t| t == exclude_domain)
            })
            .map(|(&id, insight)| {
                let sim = Self::cosine_similarity(query_embedding, &insight.reasoning_embedding);
                (id, sim)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);
        candidates
    }

    /// Registra uma rejeição do modelo mestre para mapear "buracos" no conhecimento.
    pub fn record_rejection(
        &mut self,
        query_embedding: Vec<f32>,
        domain_tags: Vec<String>,
    ) {
        // Verifica se já existe um gap próximo
        let mut found_gap = false;
        for gap in &mut self.knowledge_gaps {
            let sim = Self::cosine_similarity(&query_embedding, &gap.centroid);
            if sim > 0.85 {
                // Mesma região: atualizar estatísticas
                gap.access_count += 1;
                gap.rejection_rate = (gap.rejection_rate * (gap.access_count - 1) as f32
                    + 1.0)
                    / gap.access_count as f32;
                found_gap = true;
                break;
            }
        }

        if !found_gap {
            self.knowledge_gaps.push(KnowledgeGap {
                centroid: query_embedding,
                rejection_rate: 1.0,
                access_count: 1,
                domain_tags,
            });
        }
    }

    /// Reforço Hebbiano: fortalece sinapses entre insights ativados juntos.
    /// "Neurons that fire together, wire together."
    pub fn hebbian_strengthen(&mut self, activated_ids: &[u64]) {
        for synapse in &mut self.synapses {
            if activated_ids.contains(&synapse.from_insight)
                && activated_ids.contains(&synapse.to_insight)
            {
                synapse.weight = (synapse.weight + 0.15).min(10.0);
            }
        }
    }

    /// Poda Hebbiana: remove sinapses fracas que decairam abaixo do limiar.
    pub fn hebbian_prune(&mut self) {
        self.synapses
            .retain(|s| s.weight > self.prune_threshold);
    }

    /// Decaimento temporal: todas as sinapses perdem um pouco de força a cada época.
    pub fn hebbian_decay(&mut self, decay_factor: f32) {
        for synapse in &mut self.synapses {
            synapse.weight *= decay_factor;
        }
    }

    /// Retorna os buracos de conhecimento mais críticos para "Pensamento Profundo".
    pub fn critical_gaps(&self, top_k: usize) -> Vec<&KnowledgeGap> {
        let mut gaps: Vec<&KnowledgeGap> = self.knowledge_gaps.iter().collect();
        gaps.sort_by(|a, b| {
            b.rejection_rate
                .partial_cmp(&a.rejection_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gaps.truncate(top_k);
        gaps
    }

    /// Exporta todos os insights para persistência no LanceDB (Fase L3).
    pub fn export_for_lancedb(&self) -> Vec<(&Insight, Vec<u64>)> {
        self.insights
            .values()
            .map(|insight| (insight, insight.synapse_links.clone()))
            .collect()
    }

    // --- Utilitários internos ---

    fn hash_context(context: &[u32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        hasher.finish()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-10 { 0.0 } else { dot / denom }
    }
}

/// Tarefa Ambient AI para poda do grafo de sinapses sem bloquear a inferência principal.
pub struct AmbientPruneTask {
    pub indexer: Arc<Mutex<InsightIndexer>>,
    pub decay_factor: f32,
}

impl AmbientTask for AmbientPruneTask {
    fn execute(&mut self) {
        if let Ok(mut ix) = self.indexer.lock() {
            tracing::info!("[Ambient AI] Executando poda Hebbiana em background...");
            ix.hebbian_decay(self.decay_factor);
            ix.hebbian_prune();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insight_record_and_search() {
        let mut indexer = InsightIndexer::new(0.1);

        // Insight 1: Física → "Efeito Fotoelétrico"
        let id1 = indexer.record_insight(
            &[1, 2, 3],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![100, 101, 102],
            Modality::Text,
            vec!["Física".to_string()],
        );

        // Insight 2: Biologia → "Fotossíntese"
        let id2 = indexer.record_insight(
            &[4, 5, 6],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![200, 201, 202],
            Modality::Text,
            vec!["Biologia".to_string()],
        );

        // Insight 3: Cruzamento Físico-Biológico
        let id3 = indexer.record_insight(
            &[7, 8, 9],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![300, 301],
            Modality::CrossModal,
            vec!["Física".to_string(), "Biologia".to_string()],
        );

        assert_eq!(indexer.insights.len(), 3);

        // Buscar perto de Física (embedding ~[0.8, 0.2, 0, 0])
        let nearby = indexer.search_nearby(&[0.8, 0.2, 0.0, 0.0], 2);
        assert!(!nearby.is_empty());
        // O insight de Física deve ser o mais próximo
        assert_eq!(nearby[0].0, id1);

        // Cross-domain: excluir Física, pegar Biologia
        let cross = indexer.find_cross_domain_candidates(
            &[0.5, 0.5, 0.0, 0.0],
            "Física",
            2,
        );
        assert!(!cross.is_empty());
        // Deve retornar Biologia (id2), não o cruzamento (tem tag Física)
        assert_eq!(cross[0].0, id2);
    }

    #[test]
    fn test_synapse_graph_and_hebbian() {
        let mut indexer = InsightIndexer::new(0.1);

        let id1 = indexer.record_insight(&[1, 2], vec![1.0], vec![10], Modality::Text, vec![]);
        let id2 = indexer.record_insight(&[3, 4], vec![0.5], vec![20], Modality::Text, vec![]);
        let id3 = indexer.record_insight(&[5, 6], vec![0.0], vec![30], Modality::Text, vec![]);

        // Conectar id1 → id2 (causal), id2 → id3 (reforço)
        indexer.connect_insights(id1, id2, SynapseType::Causal);
        indexer.connect_insights(id2, id3, SynapseType::Reinforcing);

        assert_eq!(indexer.synapses.len(), 2);

        // Hebbian: ativar id1 e id2 juntos → peso cresce
        indexer.hebbian_strengthen(&[id1, id2]);
        assert!(indexer.synapses[0].weight > 1.0); // Causal fortaleceu
        assert!((indexer.synapses[1].weight - 1.0).abs() < 0.01); // Reinforcing não mudou

        // Decay
        indexer.hebbian_decay(0.5);
        assert!(indexer.synapses[0].weight < 1.0);

        // Prune (threshold 0.1)
        indexer.hebbian_prune();
        assert_eq!(indexer.synapses.len(), 2); // Ambos > 0.1
    }

    #[test]
    fn test_knowledge_gaps() {
        let mut indexer = InsightIndexer::new(0.1);

        // Registrar 3 rejeições na mesma região
        indexer.record_rejection(vec![0.9, 0.1], vec!["Matemática".to_string()]);
        indexer.record_rejection(vec![0.91, 0.09], vec!["Matemática".to_string()]);
        indexer.record_rejection(vec![0.92, 0.08], vec!["Matemática".to_string()]);

        // Outra região
        indexer.record_rejection(vec![0.1, 0.9], vec!["Arte".to_string()]);

        let gaps = indexer.critical_gaps(5);
        // A região de Matemática deve ter a maior taxa de rejeição
        assert!(gaps[0].rejection_rate >= 0.9);
        assert!(gaps[0].access_count >= 2);
    }
}
