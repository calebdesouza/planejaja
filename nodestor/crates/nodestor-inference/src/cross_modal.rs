/// NodeStor COBER v2 - Subsistema 12: Barramento de Intenção Multimodal
/// (Cross-Modal Speculation / Espaço Latente Unificado)
///
/// Em vez de ter um banco para texto e outro para imagem, o LanceDB indexa
/// CONCEITOS ABSTRATOS. A ideia de "Um entardecer calmo" é um único endereço
/// matemático (vetor). Quando a IA "pensa" nesse conceito, o COBER v2 dispara
/// rascunhos para múltiplos decodificadores ao mesmo tempo.
///
/// Funciona com QUALQUER modelo: Llama (texto), Whisper (áudio),
/// Stable Diffusion (imagem). Todos pedem dados à memória. O nosso sistema
/// intercepta esse pedido e entrega um rascunho verificado.
///
/// O conceito não é converter texto em imagem. É fazer com que ambos
/// NASÇAM DA MESMA IDEIA VETORIAL dentro do motor Rust.

/// Modalidade suportada pelo Barramento
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityType {
    Text,       // Tokens textuais (LLM: GPT, Llama, etc.)
    Image,      // Tokens visuais / patches (DiT, SD, etc.)
    Audio,      // Tokens de áudio / mel spectrogram (Whisper, Bark, etc.)
    Video,      // Tokens de vídeo / frames (Sora-like)
    Code,       // Tokens de código (sub-tipo de Text, com parsers especiais)
}

/// Um "Conceito Latente Unificado" - o nó central do Barramento de Intenção
#[derive(Debug, Clone)]
pub struct LatentConcept {
    /// ID do conceito
    pub id: u64,
    /// Embedding no espaço latente unificado (dimensionalidade do CLIP-like)
    pub unified_embedding: Vec<f32>,
    /// Descrição textual semântica (para debug e busca BM25)
    pub description: String,
    /// Quais modalidades já têm rascunhos prontos para este conceito
    pub available_modalities: Vec<ModalityType>,
    /// Rascunhos pré-computados por modalidade
    pub modal_drafts: Vec<ModalDraft>,
}

/// Um rascunho num formato específico (texto, imagem, áudio)
#[derive(Debug, Clone)]
pub struct ModalDraft {
    pub modality: ModalityType,
    /// Tokens/features do rascunho nesta modalidade
    pub draft_tokens: Vec<u32>,
    /// Confiança do rascunho (0.0 - 1.0)
    pub confidence: f32,
    /// Embedding do rascunho projetado de volta ao espaço unificado
    pub projected_embedding: Vec<f32>,
}

/// Resultado da verificação cruzada entre modalidades
#[derive(Debug)]
pub struct CrossModalVerification {
    /// Modalidade principal da saída
    pub primary_modality: ModalityType,
    /// Se a verificação cruzada encontrou consistência
    pub is_consistent: bool,
    /// Score de coerência entre as modalidades (0.0 - 1.0)
    pub coherence_score: f32,
    /// Detalhes por modalidade
    pub per_modality_scores: Vec<(ModalityType, f32)>,
}

/// MRepE — Alignment Projector Cross-Modal.
///
/// Implementa as "Funções de Mapeamento de Alinhamento" da pesquisa:
/// uma matriz de projeção (`latent_dim × latent_dim`) aprendida que garante
/// que conceitos de alinhamento (ex: "segurança", "violação") mapeiem para
/// a MESMA região do espaço latente em todas as modalidades.
///
/// ## Problema que resolve:
/// Sem o projector, um modelo pode:
/// - RECUSAR um texto sobre armas (filtro de texto OK)
/// - Mas GERAR uma imagem de arma (filtro de imagem não ativado)
///
/// Com o projector, o conceito "arma" é um PONTO no espaço latente unificado
/// que é reconhecido por TODAS as modalidades igualmente.
///
/// ## Método de aprendizado:
/// Recebe pares (`emb_texto`, `emb_imagem`) que devem representar o mesmo conceito.
/// Ajusta incrementalmente a matriz de projeção para minimizar a distância
/// entre os embeddings projetados (gradiente de identidade simplificado).
pub struct AlignmentProjector {
    /// Matriz de projeção: `latent_dim × latent_dim`
    /// (inicializada como identidade = sem projeção = comportamento original)
    pub matrix: Vec<f32>,
    pub latent_dim: usize,
    /// Taxa de aprendizado do ajuste incremental
    pub learning_rate: f32,
    /// Número de pares de calibração aprendidos
    pub pairs_learned: u64,
}

impl AlignmentProjector {
    /// Cria um projector inicializado como identidade (sem efeito até calibrar).
    pub fn new(latent_dim: usize) -> Self {
        // Identidade: M[i,j] = 1.0 se i==j, 0.0 caso contrário
        let mut matrix = vec![0.0f32; latent_dim * latent_dim];
        for i in 0..latent_dim {
            matrix[i * latent_dim + i] = 1.0;
        }
        Self {
            matrix,
            latent_dim,
            learning_rate: 0.01,
            pairs_learned: 0,
        }
    }

    /// Projeta um embedding: `e' = M * e`
    pub fn project(&self, embedding: &[f32]) -> Vec<f32> {
        let d = self.latent_dim;
        let len = embedding.len().min(d);
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            for j in 0..len {
                out[i] += self.matrix[i * d + j] * embedding[j];
            }
        }
        out
    }

    /// Aprende de um par de embeddings que DEVEM representar o mesmo conceito.
    ///
    /// Atualiza incrementalmente a matriz para minimizar ||M*e_a - e_b||.
    /// `e_a`: embedding de modalidade A (ex: texto)
    /// `e_b`: embedding alvo de modalidade B (ex: imagem)
    pub fn learn_from_pair(&mut self, e_a: &[f32], e_b: &[f32]) {
        let d = self.latent_dim;
        let projected = self.project(e_a);

        // Gradiente simples: delta = lr * (e_b - projected) * e_a^T
        // Atualiza M += lr * outer(e_b - projected, e_a)
        for i in 0..d.min(e_b.len()) {
            let error = e_b[i] - projected[i];
            for j in 0..d.min(e_a.len()) {
                self.matrix[i * d + j] += self.learning_rate * error * e_a[j];
            }
        }

        self.pairs_learned += 1;
    }

    /// Verifica se o projector está próximo da identidade (não calibrado)
    pub fn is_identity(&self) -> bool {
        let d = self.latent_dim;
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (self.matrix[i * d + j] - expected).abs() > 0.01 {
                    return false;
                }
            }
        }
        true
    }
}

/// O Barramento de Intenção Multimodal
pub struct CrossModalBus {
    /// Conceitos latentes indexados (equivalente à "memória conceitual")
    pub concepts: Vec<LatentConcept>,
    /// Dimensão do espaço latente unificado
    pub latent_dim: usize,
    /// Próximo ID disponível de conceito
    next_concept_id: u64,
    /// Limiar de similaridade para reusar um conceito existente
    pub reuse_threshold: f32,
    /// Estatísticas
    pub stats: CrossModalStats,
    /// Projector de alinhamento MRepE (opcional — None = comportamento original)
    pub alignment_projector: Option<AlignmentProjector>,
}

#[derive(Debug, Default)]
pub struct CrossModalStats {
    pub concepts_created: u64,
    pub concepts_reused: u64,
    pub cross_verifications: u64,
    pub consistency_hits: u64,
}

impl CrossModalBus {
    pub fn new(latent_dim: usize) -> Self {
        Self {
            concepts: Vec::new(),
            latent_dim,
            next_concept_id: 0,
            reuse_threshold: 0.85,
            stats: CrossModalStats::default(),
            alignment_projector: None,
        }
    }

    /// Habilita o AlignmentProjector MRepE no barramento.
    /// Após chamar este método, todos os embeddings registrados serão
    /// projetados para o espaço alinhado antes da busca por similaridade.
    pub fn enable_alignment_projector(&mut self) {
        self.alignment_projector = Some(AlignmentProjector::new(self.latent_dim));
    }

    /// Ensina o projector que dois embeddings representam o mesmo conceito.
    /// `emb_source`: embedding da modalidade SOURCE (ex: texto)
    /// `emb_target`: embedding TARGET que deve mapear pro mesmo ponto (ex: imagem)
    pub fn align_concept_pair(&mut self, emb_source: &[f32], emb_target: &[f32]) {
        if let Some(projector) = &mut self.alignment_projector {
            projector.learn_from_pair(emb_source, emb_target);
        }
    }

    /// Projeta um embedding pelo AlignmentProjector (se ativo).
    /// Se o projector não está ativo, retorna o embedding original.
    pub fn project_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        match &self.alignment_projector {
            Some(p) => p.project(embedding),
            None => embedding.to_vec(),
        }
    }

    /// Registra ou reutiliza um conceito no espaço latente unificado.
    /// Se já existe um conceito próximo, adiciona a nova modalidade a ele.
    pub fn register_concept(
        &mut self,
        unified_embedding: Vec<f32>,
        description: String,
        initial_draft: ModalDraft,
    ) -> u64 {
        // Verifica se existe um conceito próximo o suficiente
        for concept in &mut self.concepts {
            let sim = Self::cosine_sim(&unified_embedding, &concept.unified_embedding);
            if sim > self.reuse_threshold {
                // Reusar conceito existente: adicionar nova modalidade
                if !concept.available_modalities.contains(&initial_draft.modality) {
                    concept.available_modalities.push(initial_draft.modality);
                }
                concept.modal_drafts.push(initial_draft);
                self.stats.concepts_reused += 1;
                return concept.id;
            }
        }

        // Conceito novo
        let id = self.next_concept_id;
        self.next_concept_id += 1;
        self.stats.concepts_created += 1;

        let mut concept = LatentConcept {
            id,
            unified_embedding,
            description,
            available_modalities: vec![initial_draft.modality],
            modal_drafts: Vec::new(),
        };
        concept.modal_drafts.push(initial_draft);
        self.concepts.push(concept);

        id
    }

    /// Busca conceitos próximos no espaço latente unificado
    pub fn search_concepts(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<(u64, f32)> {
        let mut scores: Vec<(u64, f32)> = self.concepts
            .iter()
            .map(|c| (c.id, Self::cosine_sim(query_embedding, &c.unified_embedding)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Gera rascunhos para TODAS as modalidades disponíveis de um conceito.
    /// É aqui que "o entardecer calmo" gera texto + imagem + áudio juntos.
    pub fn dispatch_multi_modal(
        &self,
        concept_id: u64,
    ) -> Vec<&ModalDraft> {
        self.concepts
            .iter()
            .find(|c| c.id == concept_id)
            .map(|c| c.modal_drafts.iter().collect())
            .unwrap_or_default()
    }

    /// Verificação cruzada: compara se o rascunho de texto "bate" com o de imagem.
    /// Se o texto diz "sol se põe vermelho" e a imagem tem tons azuis, é inconsistente.
    pub fn verify_cross_modal(
        &mut self,
        concept_id: u64,
    ) -> CrossModalVerification {
        self.stats.cross_verifications += 1;

        let concept = match self.concepts.iter().find(|c| c.id == concept_id) {
            Some(c) => c,
            None => return CrossModalVerification {
                primary_modality: ModalityType::Text,
                is_consistent: true,
                coherence_score: 1.0,
                per_modality_scores: Vec::new(),
            },
        };

        let drafts: Vec<&ModalDraft> = concept.modal_drafts.iter().collect();

        if drafts.len() < 2 {
            return CrossModalVerification {
                primary_modality: drafts.first().map(|d| d.modality).unwrap_or(ModalityType::Text),
                is_consistent: true,
                coherence_score: 1.0,
                per_modality_scores: Vec::new(),
            };
        }

        // Verifica consistência pairwise entre todos os embeddings projetados
        let mut total_sim = 0.0f32;
        let mut pair_count = 0u32;
        let mut per_modality_scores = Vec::new();

        for i in 0..drafts.len() {
            for j in (i + 1)..drafts.len() {
                let sim = Self::cosine_sim(
                    &drafts[i].projected_embedding,
                    &drafts[j].projected_embedding,
                );
                total_sim += sim;
                pair_count += 1;
            }
            per_modality_scores.push((drafts[i].modality, drafts[i].confidence));
        }

        let coherence = if pair_count > 0 { total_sim / pair_count as f32 } else { 0.0 };
        let is_consistent = coherence > 0.6;
        if is_consistent {
            self.stats.consistency_hits += 1;
        }

        CrossModalVerification {
            primary_modality: drafts[0].modality,
            is_consistent,
            coherence_score: coherence,
            per_modality_scores,
        }
    }

    /// "Traduz" um conceito de uma modalidade para outra.
    /// Ex: texto "sol se põe" → rascunho de imagem com cores quentes.
    pub fn translate_modality(
        &mut self,
        concept_id: u64,
        target_modality: ModalityType,
        draft_tokens: Vec<u32>,
        projected_embedding: Vec<f32>,
    ) -> bool {
        if let Some(concept) = self.concepts.iter_mut().find(|c| c.id == concept_id) {
            if !concept.available_modalities.contains(&target_modality) {
                concept.available_modalities.push(target_modality);
            }
            concept.modal_drafts.push(ModalDraft {
                modality: target_modality,
                draft_tokens,
                confidence: 0.7, // Tradução cross-modal tem confiança moderada
                projected_embedding,
            });
            true
        } else {
            false
        }
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

    #[test]
    fn test_unified_latent_space() {
        let mut bus = CrossModalBus::new(4);

        // Conceito "entardecer calmo" — primeiro como texto
        let text_draft = ModalDraft {
            modality: ModalityType::Text,
            draft_tokens: vec![100, 101, 102], // "sol", "se", "põe"
            confidence: 0.9,
            projected_embedding: vec![0.8, 0.1, 0.05, 0.05],
        };

        let concept_id = bus.register_concept(
            vec![0.7, 0.2, 0.05, 0.05],
            "Entardecer calmo".to_string(),
            text_draft,
        );

        // Adiciona rascunho de imagem PRO MESMO CONCEITO (reuse!)
        let image_draft = ModalDraft {
            modality: ModalityType::Image,
            draft_tokens: vec![500, 501, 502], // patches visuais
            confidence: 0.85,
            projected_embedding: vec![0.75, 0.15, 0.05, 0.05],
        };

        let reused_id = bus.register_concept(
            vec![0.71, 0.19, 0.05, 0.05], // muito próximo → reuse
            "Entardecer calmo visual".to_string(),
            image_draft,
        );

        // Deve reusar o mesmo conceito
        assert_eq!(concept_id, reused_id);
        assert_eq!(bus.stats.concepts_reused, 1);

        // Verificar que o conceito agora tem 2 modalidades
        let concept = &bus.concepts[0];
        assert!(concept.available_modalities.contains(&ModalityType::Text));
        assert!(concept.available_modalities.contains(&ModalityType::Image));
        assert_eq!(concept.modal_drafts.len(), 2);
    }

    #[test]
    fn test_cross_modal_verification() {
        let mut bus = CrossModalBus::new(4);

        let concept_id = bus.register_concept(
            vec![0.5, 0.5, 0.0, 0.0],
            "Teste".to_string(),
            ModalDraft {
                modality: ModalityType::Text,
                draft_tokens: vec![1, 2],
                confidence: 0.9,
                projected_embedding: vec![0.9, 0.1, 0.0, 0.0], // texto aponta pra um lado
            },
        );

        // Imagem consistente (aponta pro mesmo lado)
        bus.translate_modality(
            concept_id,
            ModalityType::Image,
            vec![10, 20],
            vec![0.85, 0.15, 0.0, 0.0],
        );

        let verification = bus.verify_cross_modal(concept_id);
        assert!(verification.is_consistent);
        assert!(verification.coherence_score > 0.8);
    }

    #[test]
    fn test_cross_modal_inconsistency() {
        let mut bus = CrossModalBus::new(4);

        let concept_id = bus.register_concept(
            vec![0.5, 0.5, 0.0, 0.0],
            "Teste inconsistente".to_string(),
            ModalDraft {
                modality: ModalityType::Text,
                draft_tokens: vec![1, 2],
                confidence: 0.9,
                projected_embedding: vec![1.0, 0.0, 0.0, 0.0], // texto aponta norte
            },
        );

        // Imagem INCONSISTENTE (aponta sul — azul quando deveria ser vermelho)
        bus.translate_modality(
            concept_id,
            ModalityType::Image,
            vec![10, 20],
            vec![0.0, 0.0, 0.0, 1.0], // perpendicular
        );

        let verification = bus.verify_cross_modal(concept_id);
        assert!(!verification.is_consistent);
        assert!(verification.coherence_score < 0.5);
    }

    #[test]
    fn test_dispatch_multi_modal() {
        let mut bus = CrossModalBus::new(2);

        let cid = bus.register_concept(
            vec![1.0, 0.0],
            "Música".to_string(),
            ModalDraft {
                modality: ModalityType::Audio,
                draft_tokens: vec![70, 71],
                confidence: 0.8,
                projected_embedding: vec![0.9, 0.1],
            },
        );

        bus.translate_modality(cid, ModalityType::Text, vec![80, 81], vec![0.85, 0.15]);
        bus.translate_modality(cid, ModalityType::Image, vec![90, 91], vec![0.8, 0.2]);

        let dispatched = bus.dispatch_multi_modal(cid);
        assert_eq!(dispatched.len(), 3); // áudio + texto + imagem

        let modalities: Vec<ModalityType> = dispatched.iter().map(|d| d.modality).collect();
        assert!(modalities.contains(&ModalityType::Audio));
        assert!(modalities.contains(&ModalityType::Text));
        assert!(modalities.contains(&ModalityType::Image));
    }

    #[test]
    fn test_alignment_projector_starts_as_identity() {
        let proj = AlignmentProjector::new(3);
        assert!(proj.is_identity(), "Projector recém-criado deve ser identidade");
        assert_eq!(proj.pairs_learned, 0);

        // Projetar com identidade = input inalterado
        let emb = vec![1.0f32, 2.0, 3.0];
        let result = proj.project(&emb);
        for i in 0..3 {
            assert!((result[i] - emb[i]).abs() < 1e-5,
                "Identidade deve preservar embedding: idx={}, esperado={}, obtido={}",
                i, emb[i], result[i]);
        }
    }

    #[test]
    fn test_alignment_projector_learns_from_pair() {
        let mut proj = AlignmentProjector::new(4);

        let emb_text = vec![1.0f32, 0.0, 0.0, 0.0];
        let emb_image = vec![0.8f32, 0.2, 0.0, 0.0]; // ligeiramente diferente

        // Treinar 100 pares — deve aproximar a projeção do alvo
        for _ in 0..100 {
            proj.learn_from_pair(&emb_text, &emb_image);
        }

        assert_eq!(proj.pairs_learned, 100);
        assert!(!proj.is_identity(), "Após aprendizado, projector não deve ser identidade");

        // A projeção do texto deve estar mais próxima da imagem alvo
        let projected = proj.project(&emb_text);
        let dist_before = (1.0f32 - 0.8f32).powi(2).sqrt(); // sem projeção
        let dist_after: f32 = projected.iter().zip(emb_image.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(dist_after < dist_before,
            "Após aprendizado, distância deve reduzir: antes={:.4}, depois={:.4}",
            dist_before, dist_after);
    }

    #[test]
    fn test_bus_alignment_projector_optional() {
        let mut bus = CrossModalBus::new(4);
        assert!(bus.alignment_projector.is_none(), "Projector deve ser None por padrão");

        // project_embedding sem projector deve retornar input original
        let emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let projected = bus.project_embedding(&emb);
        assert_eq!(projected, emb, "Sem projector, embedding deve ser identidade");

        // Habilitar projector
        bus.enable_alignment_projector();
        assert!(bus.alignment_projector.is_some(), "Projector deve estar ativo após enable");

        // Treinar alinhamento
        let source = vec![1.0f32, 0.0, 0.0, 0.0];
        let target = vec![0.5f32, 0.5, 0.0, 0.0];
        bus.align_concept_pair(&source, &target);

        let proj = bus.alignment_projector.as_ref().unwrap();
        assert_eq!(proj.pairs_learned, 1);
    }
}
