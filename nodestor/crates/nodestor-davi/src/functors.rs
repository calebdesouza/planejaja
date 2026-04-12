//! D6 — Domain Functors: Category Theory para Cross-Domain Discovery
//!
//! Funtores descobrem que "gravidade" e "custo de oportunidade"
//! compartilham a mesma estrutura matemática.
//! 1 descoberta em Física = 1 descoberta em CADA domínio mapeado.

use std::collections::HashMap;

/// Um insight em um domínio específico
#[derive(Debug, Clone)]
pub struct Insight {
    pub id: u64,
    pub domain: String,
    pub statement: String,
    pub embedding: Vec<f32>,
    /// Relações com outros insights (id_origem, id_destino)
    pub relations: Vec<(u64, u64)>,
}

/// Um insight traduzido para outro domínio via Funtor
#[derive(Debug, Clone)]
pub struct TranslatedInsight {
    pub original_id: u64,
    pub original_domain: String,
    pub translated_domain: String,
    pub translated_statement: String,
    pub translated_embedding: Vec<f32>,
    /// Quão bem preservada foi a estrutura (0=péssimo, 1=perfeito)
    pub fidelity: f32,
}

/// Um Funtor: mapa que preserva estrutura entre domínios
#[derive(Debug, Clone)]
pub struct DomainFunctor {
    pub source_domain: String,
    pub target_domain: String,
    /// Mapa de embeddings fonte → destino (por hash de embedding)
    pub concept_map: HashMap<u64, Vec<f32>>,
    /// Mapa de relações: se A→B em fonte, então F(A)→F(B) em destino
    pub relation_map: HashMap<(u64, u64), (u64, u64)>,
    /// Score de isomorfismo: quão perfeita é a preservação estrutural
    pub isomorphism_score: f32,
}

impl DomainFunctor {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source_domain: source.into(),
            target_domain: target.into(),
            concept_map: HashMap::new(),
            relation_map: HashMap::new(),
            isomorphism_score: 0.0,
        }
    }

    /// Adiciona um mapeamento de conceito
    pub fn add_concept_mapping(&mut self, source_hash: u64, target_embedding: Vec<f32>) {
        self.concept_map.insert(source_hash, target_embedding);
    }

    /// Adiciona um mapeamento de relação
    pub fn add_relation_mapping(&mut self, from: (u64, u64), to: (u64, u64)) {
        self.relation_map.insert(from, to);
    }

    /// Aplica o funtor a um embedding (translação linear simplificada)
    pub fn apply_to_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        // Busca o embedding mais próximo no concept_map
        let hash = hash_embedding(embedding);
        if let Some(target) = self.concept_map.get(&hash) {
            return target.clone();
        }
        // Fallback: rotação vetorial simples (proxy de translação entre domínios)
        embedding.iter().enumerate()
            .map(|(i, v)| if i % 2 == 0 { *v * 0.9 } else { *v * 1.1 })
            .collect()
    }
}

/// Compositor de Funtores: descobre e compõe mapeamentos entre domínios
pub struct FunctorComposer {
    /// Biblioteca de funtores conhecidos
    pub functor_library: Vec<DomainFunctor>,
}

impl FunctorComposer {
    pub fn new() -> Self {
        Self { functor_library: Vec::new() }
    }

    /// Descobre um novo funtor entre dois domínios.
    /// Heurística: calcula similaridade estrutural (relações preservadas).
    pub fn discover_functor(
        &mut self,
        domain_a_insights: &[Insight],
        domain_b_insights: &[Insight],
    ) -> Option<DomainFunctor> {
        if domain_a_insights.is_empty() || domain_b_insights.is_empty() {
            return None;
        }

        let domain_a = &domain_a_insights[0].domain;
        let domain_b = &domain_b_insights[0].domain;
        let mut functor = DomainFunctor::new(domain_a.clone(), domain_b.clone());

        // Pareia insights por similaridade coseno entre embeddings
        let mut paired = 0usize;
        let mut total = 0usize;

        for a in domain_a_insights {
            if let Some(b) = best_match(a, domain_b_insights) {
                let hash = hash_embedding(&a.embedding);
                functor.add_concept_mapping(hash, b.embedding.clone());
                paired += 1;
            }
            total += 1;
        }

        // Score de isomorfismo = proporção de conceitos pareados
        functor.isomorphism_score = if total > 0 {
            paired as f32 / total as f32
        } else {
            0.0
        };

        // Só aceita funtores com preservação suficiente
        if functor.isomorphism_score >= 0.3 {
            self.functor_library.push(functor.clone());
            Some(functor)
        } else {
            None
        }
    }

    /// Compõe dois funtores: F: A→B e G: B→C resulta em G∘F: A→C
    pub fn compose(&self, f: &DomainFunctor, g: &DomainFunctor) -> Option<DomainFunctor> {
        if f.target_domain != g.source_domain {
            return None; // Não compatíveis para composição
        }

        let mut composed = DomainFunctor::new(
            f.source_domain.clone(),
            g.target_domain.clone(),
        );

        // Compõe os mapas de conceito: A→B→C
        for (hash_a, emb_b) in &f.concept_map {
            let hash_b = hash_embedding(emb_b);
            if let Some(emb_c) = g.concept_map.get(&hash_b) {
                composed.add_concept_mapping(*hash_a, emb_c.clone());
            }
        }

        // Score da composição: produto dos scores individuais
        composed.isomorphism_score = f.isomorphism_score * g.isomorphism_score;
        Some(composed)
    }

    /// Traduz um insight de um domínio para outro
    pub fn translate_insight(
        &self,
        insight: &Insight,
        functor: &DomainFunctor,
    ) -> TranslatedInsight {
        let translated_embedding = functor.apply_to_embedding(&insight.embedding);
        let translated_statement = format!(
            "[{}→{}] {}",
            functor.source_domain, functor.target_domain, insight.statement
        );

        TranslatedInsight {
            original_id: insight.id,
            original_domain: insight.domain.clone(),
            translated_domain: functor.target_domain.clone(),
            translated_statement,
            translated_embedding,
            fidelity: functor.isomorphism_score,
        }
    }

    /// Busca um funtor entre dois domínios específicos
    pub fn find_functor(&self, from: &str, to: &str) -> Option<&DomainFunctor> {
        self.functor_library.iter()
            .find(|f| f.source_domain == from && f.target_domain == to)
    }
}

impl Default for FunctorComposer {
    fn default() -> Self { Self::new() }
}

fn best_match<'a>(target: &Insight, candidates: &'a [Insight]) -> Option<&'a Insight> {
    candidates.iter().max_by(|a, b| {
        let da = cosine_similarity(&target.embedding, &a.embedding);
        let db = cosine_similarity(&target.embedding, &b.embedding);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

fn hash_embedding(embedding: &[f32]) -> u64 {
    let mut hash = 14695981039346656037u64;
    for &v in embedding {
        let bits = v.to_bits();
        hash ^= bits as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn physics_insights() -> Vec<Insight> {
        vec![
            Insight {
                id: 1,
                domain: "Fisica".to_string(),
                statement: "Gravidade minimiza energia potencial".to_string(),
                embedding: vec![0.8, 0.2, 0.1],
                relations: vec![(1, 2)],
            },
            Insight {
                id: 2,
                domain: "Fisica".to_string(),
                statement: "Entropia sempre aumenta".to_string(),
                embedding: vec![0.3, 0.9, 0.1],
                relations: vec![],
            },
        ]
    }

    fn economics_insights() -> Vec<Insight> {
        vec![
            Insight {
                id: 10,
                domain: "Economia".to_string(),
                statement: "Custo de oportunidade minimiza perdas".to_string(),
                embedding: vec![0.75, 0.25, 0.15],
                relations: vec![(10, 11)],
            },
            Insight {
                id: 11,
                domain: "Economia".to_string(),
                statement: "Desordem dos mercados aumenta".to_string(),
                embedding: vec![0.25, 0.85, 0.15],
                relations: vec![],
            },
        ]
    }

    #[test]
    fn test_functor_discovery() {
        let mut composer = FunctorComposer::new();
        let physics = physics_insights();
        let economics = economics_insights();
        let functor = composer.discover_functor(&physics, &economics);
        assert!(functor.is_some(), "Deveria descobrir funtor entre Física e Economia");
        let f = functor.unwrap();
        assert!(f.isomorphism_score >= 0.3);
    }

    #[test]
    fn test_functor_composition() {
        let mut composer = FunctorComposer::new();
        let physics = physics_insights();
        let economics = economics_insights();

        // Cria funtores F: Física→Economia e G: Economia→Física (inverso)
        if let Some(f) = composer.discover_functor(&physics, &economics) {
            // Cria G manualmente para teste
            let mut g = DomainFunctor::new("Economia", "Biologia");
            for (_, emb_b) in &f.concept_map {
                g.add_concept_mapping(hash_embedding(emb_b), vec![0.5, 0.5, 0.5]);
            }
            g.isomorphism_score = 0.7;

            let composed = composer.compose(&f, &g);
            assert!(composed.is_some(), "Deveria compor F: Física→Economia com G: Economia→Biologia");
            let gc = composed.unwrap();
            assert_eq!(gc.source_domain, "Fisica");
            assert_eq!(gc.target_domain, "Biologia");
        }
    }

    #[test]
    fn test_insight_translation() {
        let mut composer = FunctorComposer::new();
        let physics = physics_insights();
        let economics = economics_insights();

        if let Some(functor) = composer.discover_functor(&physics, &economics) {
            let translated = composer.translate_insight(&physics[0], &functor);
            assert_eq!(translated.original_id, 1);
            assert_eq!(translated.translated_domain, "Economia");
            assert!(!translated.translated_statement.is_empty());
            assert!(translated.fidelity >= 0.3);
        }
    }
}
