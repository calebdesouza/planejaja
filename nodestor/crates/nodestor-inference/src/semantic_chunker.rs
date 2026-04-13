//! PROBES V2 â€” Semantic Chunker (PrincÃ­pio 7: Dados com Significado)
//!
//! Chunking que usa SAE features para detectar fronteiras semÃ¢nticas naturais.
//! Em vez de cortar a cada N tokens (que destrÃ³i o contexto), corta onde
//! as features SAE mudam significativamente â€” fronteiras conceptuais reais.
//!
//! ## TÃ©cnicas implementadas:
//! - **Semantic Boundary Detection**: SAE detecta quando o "assunto muda"
//! - **Contextual Enrichment**: Cada chunk sabe "onde estÃ¡ no documento"
//! - **Parent-Child Hierarchy**: Chunk pequeno para busca, contexto pai para geraÃ§Ã£o
//! - **SAE Feature Tags**: Chunks auto-rotulados com significado semÃ¢ntico
//!
//! ## Em produÃ§Ã£o (com modelo real):
//! O `sae.encode()` usaria pesos treinados em ativaÃ§Ãµes reais.
//! Em simulaÃ§Ã£o: usa heurÃ­stica de variÃ¢ncia para detectar mudanÃ§a semÃ¢ntica.

use crate::sae_engine::SAEEngine;
use std::collections::HashMap;

/// Um chunk enriquecido com metadados semÃ¢nticos e features SAE.
#[derive(Debug, Clone)]
pub struct EnrichedChunk {
    /// Texto bruto do chunk
    pub text: String,
    /// ID Ãºnico do chunk (hash do texto + posiÃ§Ã£o)
    pub id: u64,
    /// Embedding vetorial mÃ©dio das features SAE ativas
    pub embedding: Vec<f32>,
    /// Features SAE ativas: (feature_idx, intensidade)
    pub sae_features: Vec<(usize, f32)>,
    /// Tags semÃ¢nticas derivadas das features (ex: "seguranÃ§a", "autenticaÃ§Ã£o")
    pub semantic_tags: Vec<String>,
    /// Documento pai (filename ou URL)
    pub parent_doc: String,
    /// CabeÃ§alho de seÃ§Ã£o (se detectado)
    pub section_header: Option<String>,
    /// Contexto de localizaÃ§Ã£o: "onde estou neste documento?"
    pub location_context: String,
    /// IDs de chunks vizinhos (predecessor + sucessor)
    pub neighbor_chunk_ids: Vec<u64>,
    /// Cluster topolÃ³gico (preenchido pelo Topology apÃ³s indexaÃ§Ã£o)
    pub topology_cluster: Option<usize>,
    /// PosiÃ§Ã£o no documento (0-indexed)
    pub position: usize,
}

/// Chunker semÃ¢ntico baseado em SAE features.
pub struct SemanticChunker {
    pub sae: SAEEngine,
    /// Threshold de distÃ¢ncia cosseno para detectar fronteira semÃ¢ntica.
    /// Se a distÃ¢ncia entre dois segmentos consecutivos > threshold â†’ novo chunk.
    pub boundary_threshold: f32,
    /// NÃºmero mÃ¡ximo de caracteres por chunk (fallback se nÃ£o detectar fronteira)
    pub max_chunk_chars: usize,
    /// NÃºmero mÃ­nimo de caracteres para formar um chunk vÃ¡lido
    pub min_chunk_chars: usize,
    /// Map de feature_idx â†’ tag legÃ­vel (configurÃ¡vel)
    pub feature_tag_map: HashMap<usize, String>,
}

impl SemanticChunker {
    /// Cria um chunker com configuraÃ§Ã£o padrÃ£o.
    pub fn new(hidden_dim: usize, dict_size: usize, max_chunk_chars: usize) -> Self {
        let mut chunker = Self {
            sae: SAEEngine::new(hidden_dim, dict_size, 0.3),
            boundary_threshold: 0.35,
            max_chunk_chars,
            min_chunk_chars: 50,
            feature_tag_map: HashMap::new(),
        };
        chunker.load_default_feature_tags();
        chunker
    }

    /// Chunking semÃ¢ntico completo de um documento.
    ///
    /// # Algoritmo:
    /// 1. Divide por delimitadores naturais (parÃ¡grafos, funÃ§Ãµes, seÃ§Ãµes)
    /// 2. Para cada segmento, extrai features SAE
    /// 3. Compara features consecutivas: mudanÃ§a grande â†’ fronteira
    /// 4. Agrega segmentos entre fronteiras em chunks
    /// 5. Enriquece cada chunk com contexto e tags
    pub fn chunk_document(&mut self, text: &str, doc_id: &str) -> Vec<EnrichedChunk> {
        let segments = self.split_into_segments(text);

        if segments.is_empty() {
            return Vec::new();
        }

        // Extrai features para cada segmento (prÃ©-computa antes do segundo passo)
        let segment_features: Vec<Vec<f32>> = {
            let segs: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();
            segs.into_iter().map(|seg| self.text_to_features(seg)).collect()
        };

        // Detecta fronteiras semÃ¢nticas
        let boundaries = self.detect_boundaries(&segment_features);

        // Agrupa segmentos entre fronteiras
        let raw_chunks = self.group_by_boundaries(&segments, &boundaries);
        let raw_len = raw_chunks.len();

        // Enriquece cada chunk com loop explÃ­cito (evita closure com &mut self)
        let mut chunks: Vec<EnrichedChunk> = Vec::new();
        for (pos, chunk_text) in raw_chunks.iter().enumerate() {
            if chunk_text.len() < self.min_chunk_chars {
                continue;
            }
            let features = self.text_to_features(chunk_text);
            let sae_features = self.extract_active_features(&features);
            let semantic_tags = self.features_to_tags(&sae_features);
            let embedding = self.features_to_embedding(&features);
            let header = self.detect_section_header(chunk_text);
            let id = self.hash_chunk(doc_id, pos, chunk_text);

            chunks.push(EnrichedChunk {
                text: chunk_text.clone(),
                id,
                embedding,
                sae_features,
                semantic_tags,
                parent_doc: doc_id.to_string(),
                section_header: header,
                location_context: format!(
                    "Documento: '{}' | Chunk {} de {}",
                    doc_id, pos + 1, raw_len
                ),
                neighbor_chunk_ids: Vec::new(),
                topology_cluster: None,
                position: pos,
            });
        }

        // Liga chunks vizinhos
        Self::link_neighbors(&mut chunks);

        chunks
    }

    /// Retorna o contexto pai completo dado um chunk filho.
    /// Em produÃ§Ã£o: buscaria a seÃ§Ã£o pai no banco de documentos.
    pub fn get_parent_context(&self, chunk: &EnrichedChunk, full_doc: &str) -> String {
        // Encontra a seÃ§Ã£o do documento que contÃ©m este chunk
        let chunk_start = full_doc.find(&chunk.text).unwrap_or(0);
        let context_start = chunk_start.saturating_sub(200);
        let context_end = (chunk_start + chunk.text.len() + 200).min(full_doc.len());
        full_doc[context_start..context_end].to_string()
    }

    /// DistÃ¢ncia cosseno entre dois vetores de features.
    pub fn semantic_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 1.0;
        }
        let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
        let na: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 {
            return 1.0;
        }
        1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    // --- UtilitÃ¡rios internos ---

    fn split_into_segments<'a>(&self, text: &'a str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current = String::new();

        for line in text.lines() {
            let trimmed = line.trim();

            // Delimitadores fortes: linhas em branco apÃ³s parÃ¡grafo, cabeÃ§alhos, funÃ§Ãµes
            let is_strong_boundary = trimmed.is_empty()
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("## ")
                || trimmed.starts_with("# ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("function ");

            if is_strong_boundary && !current.trim().is_empty() {
                if current.len() >= self.min_chunk_chars {
                    segments.push(current.trim().to_string());
                }
                current = String::new();
            }

            current.push_str(line);
            current.push('\n');

            // ForÃ§a quebra se excedeu tamanho mÃ¡ximo
            if current.len() >= self.max_chunk_chars {
                if !current.trim().is_empty() {
                    segments.push(current.trim().to_string());
                }
                current = String::new();
            }
        }

        if !current.trim().is_empty() && current.len() >= self.min_chunk_chars {
            segments.push(current.trim().to_string());
        }

        segments
    }

    fn text_to_features(&mut self, text: &str) -> Vec<f32> {
        // Em produÃ§Ã£o: tokeniza e faz forward pass para obter hidden state
        // Em simulaÃ§Ã£o: usa heurÃ­stica baseada em caracterÃ­sticas do texto
        let len = self.sae.hidden_dim;
        let mut pseudo_hidden = vec![0.0f32; len];

        // Simula features baseadas em caracterÃ­sticas lÃ©xicas
        let char_bytes = text.as_bytes();
        for (i, &b) in char_bytes.iter().take(len).enumerate() {
            pseudo_hidden[i % len] += b as f32 / 255.0;
        }

        // Normaliza
        let norm: f32 = pseudo_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut pseudo_hidden {
                *v /= norm;
            }
        }

        self.sae.encode(&pseudo_hidden)
    }

    fn detect_boundaries(&self, features: &[Vec<f32>]) -> Vec<bool> {
        if features.len() <= 1 {
            return vec![true; features.len()];
        }

        let mut boundaries = vec![true]; // Primeiro segmento sempre inicia chunk

        for window in features.windows(2) {
            let dist = self.semantic_distance(&window[0], &window[1]);
            boundaries.push(dist > self.boundary_threshold);
        }

        boundaries
    }

    fn group_by_boundaries(&self, segments: &[String], boundaries: &[bool]) -> Vec<String> {
        let mut groups: Vec<String> = Vec::new();
        let mut current = String::new();

        for (i, segment) in segments.iter().enumerate() {
            if i > 0 && boundaries.get(i).copied().unwrap_or(false) && !current.is_empty() {
                groups.push(current.trim().to_string());
                current = String::new();
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(segment);
        }

        if !current.trim().is_empty() {
            groups.push(current.trim().to_string());
        }

        groups
    }

    fn extract_active_features(&self, features: &[f32]) -> Vec<(usize, f32)> {
        features.iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| (i, v))
            .take(32) // Top-32 features ativas
            .collect()
    }

    fn features_to_tags(&self, active_features: &[(usize, f32)]) -> Vec<String> {
        let mut tags: Vec<String> = active_features.iter()
            .filter_map(|(idx, _)| self.feature_tag_map.get(idx).cloned())
            .collect();
        tags.dedup();
        tags
    }

    fn features_to_embedding(&self, features: &[f32]) -> Vec<f32> {
        // Embedding compacto: mÃ©dia das features ativas (dimensÃ£o reduzida)
        let active: Vec<f32> = features.iter().filter(|&&v| v > 0.0).copied().collect();
        if active.is_empty() {
            return vec![0.0; 64.min(features.len())];
        }
        // Pega as primeiras 64 features como embedding compacto
        features[..64.min(features.len())].to_vec()
    }

    fn detect_section_header(&self, text: &str) -> Option<String> {
        let first_line = text.lines().next().unwrap_or("").trim();
        if first_line.starts_with('#')
            || first_line.starts_with("fn ")
            || first_line.starts_with("pub fn ")
            || first_line.starts_with("class ")
            || first_line.starts_with("def ")
        {
            Some(first_line.to_string())
        } else {
            None
        }
    }

    fn hash_chunk(&self, doc_id: &str, pos: usize, text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        doc_id.hash(&mut h);
        pos.hash(&mut h);
        text.len().hash(&mut h);
        h.finish()
    }

    fn link_neighbors(chunks: &mut Vec<EnrichedChunk>) {
        let ids: Vec<u64> = chunks.iter().map(|c| c.id).collect();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            if i > 0 {
                chunk.neighbor_chunk_ids.push(ids[i - 1]);
            }
            if i + 1 < ids.len() {
                chunk.neighbor_chunk_ids.push(ids[i + 1]);
            }
        }
    }

    fn load_default_feature_tags(&mut self) {
        // Bootstrap: features associadas a conceitos comuns
        // Em produÃ§Ã£o: carregado de um arquivo de configuraÃ§Ã£o calibrado
        let defaults: &[(usize, &str)] = &[
            (0, "sintaxe"),
            (10, "seguranÃ§a"),
            (20, "autenticaÃ§Ã£o"),
            (30, "criptografia"),
            (40, "banco_de_dados"),
            (50, "api"),
            (60, "configuraÃ§Ã£o"),
            (70, "teste"),
            (80, "documentaÃ§Ã£o"),
            (90, "algoritmo"),
            (100, "estrutura_de_dados"),
            (110, "rede"),
            (120, "erro"),
            (130, "performance"),
            (140, "paralelismo"),
        ];
        for &(idx, tag) in defaults {
            self.feature_tag_map.insert(idx, tag.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunker() -> SemanticChunker {
        SemanticChunker::new(64, 256, 500)
    }

    #[test]
    fn test_chunk_simple_document() {
        let mut chunker = make_chunker();
        let doc = "# MÃ³dulo de AutenticaÃ§Ã£o\n\
                   Gerencia tokens JWT e validaÃ§Ã£o de usuÃ¡rios.\n\n\
                   pub fn validate_token(token: &str) -> bool {\n\
                       // Verifica assinatura e expiraÃ§Ã£o\n\
                       true\n\
                   }\n\n\
                   pub fn create_token(user_id: u64) -> String {\n\
                       // Cria novo JWT\n\
                       String::new()\n\
                   }";

        let chunks = chunker.chunk_document(doc, "auth.rs");

        assert!(!chunks.is_empty(), "Documento deve gerar pelo menos 1 chunk");
        assert!(chunks.iter().all(|c| !c.text.is_empty()), "Todos os chunks devem ter texto");
        assert!(chunks.iter().all(|c| c.parent_doc == "auth.rs"),
            "Todos os chunks devem ter parent_doc correto");
    }

    #[test]
    fn test_chunks_have_embeddings() {
        let mut chunker = make_chunker();
        let doc = "Esta Ã© uma funÃ§Ã£o de autenticaÃ§Ã£o importante.\n\
                   Ela valida tokens JWT de forma segura.\n\n\
                   Outra seÃ§Ã£o completamente diferente sobre banco de dados.";

        let chunks = chunker.chunk_document(doc, "test.rs");

        for chunk in &chunks {
            assert!(!chunk.embedding.is_empty(),
                "Chunk deve ter embedding: '{}'", &chunk.text[..chunk.text.len().min(30)]);
        }
    }

    #[test]
    fn test_chunks_have_unique_ids() {
        let mut chunker = make_chunker();
        let doc = "SeÃ§Ã£o 1:\nConteÃºdo da primeira seÃ§Ã£o com texto suficiente.\n\n\
                   SeÃ§Ã£o 2:\nConteÃºdo da segunda seÃ§Ã£o completamente diferente.\n\n\
                   SeÃ§Ã£o 3:\nConteÃºdo da terceira seÃ§Ã£o tambÃ©m diferente e Ãºnico.";

        let chunks = chunker.chunk_document(doc, "multi_section.md");

        if chunks.len() > 1 {
            let ids: Vec<u64> = chunks.iter().map(|c| c.id).collect();
            let unique_ids: std::collections::HashSet<u64> = ids.iter().copied().collect();
            assert_eq!(ids.len(), unique_ids.len(), "Todos os IDs de chunk devem ser Ãºnicos");
        }
    }

    #[test]
    fn test_chunk_location_context() {
        let mut chunker = make_chunker();
        let doc = "Primeira parte do documento com conteÃºdo relevante aqui.\n\n\
                   Segunda parte com conteÃºdo muito diferente da primeira parte.";

        let chunks = chunker.chunk_document(doc, "doc.md");

        for chunk in &chunks {
            assert!(chunk.location_context.contains("doc.md"),
                "Location context deve mencionar o documento");
        }
    }

    #[test]
    fn test_neighbor_linking() {
        let mut chunker = make_chunker();
        let doc = "FunÃ§Ã£o authenticate():\n\
                   Verifica credenciais do usuÃ¡rio de forma segura e confiÃ¡vel.\n\n\
                   pub fn get_user_from_db(id: u64) -> Option<User> {\n\
                       // consulta banco de dados\n\
                       None\n\
                   }\n\n\
                   pub fn hash_password(password: &str) -> String {\n\
                       // bcrypt hash seguro\n\
                       String::new()\n\
                   }";

        let chunks = chunker.chunk_document(doc, "auth_module.rs");

        if chunks.len() >= 2 {
            // O segundo chunk deve ter o ID do primeiro como vizinho
            assert!(!chunks[1].neighbor_chunk_ids.is_empty(),
                "Chunk 2 deve ter vizinhos");
            assert!(chunks[1].neighbor_chunk_ids.contains(&chunks[0].id),
                "Chunk 2 deve ter chunk 1 como vizinho");
        }
    }

    #[test]
    fn test_empty_document_returns_empty() {
        let mut chunker = make_chunker();
        let chunks = chunker.chunk_document("", "empty.txt");
        assert!(chunks.is_empty(), "Documento vazio deve retornar vazio");
    }

    #[test]
    fn test_semantic_distance_identical() {
        let mut chunker = make_chunker();
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let dist = chunker.semantic_distance(&a, &b);
        assert!(dist < 0.01, "Vetores idÃªnticos devem ter distÃ¢ncia â‰ˆ 0: got {:.3}", dist);
    }

    #[test]
    fn test_semantic_distance_orthogonal() {
        let mut chunker = make_chunker();
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let dist = chunker.semantic_distance(&a, &b);
        assert!(dist > 0.9, "Vetores ortogonais devem ter distÃ¢ncia â‰ˆ 1: got {:.3}", dist);
    }

    #[test]
    fn test_parent_context_extraction() {
        let mut chunker = make_chunker();
        let full_doc = "IntroduÃ§Ã£o ao sistema.\n\nFunÃ§Ã£o principal aqui.\n\nConclcusÃ£o.";
        let chunk = EnrichedChunk {
            text: "FunÃ§Ã£o principal aqui.".to_string(),
            id: 1,
            embedding: vec![],
            sae_features: vec![],
            semantic_tags: vec![],
            parent_doc: "doc.md".to_string(),
            section_header: None,
            location_context: String::new(),
            neighbor_chunk_ids: vec![],
            topology_cluster: None,
            position: 1,
        };

        let ctx = chunker.get_parent_context(&chunk, full_doc);
        assert!(!ctx.is_empty(), "Contexto pai nÃ£o deve ser vazio");
        assert!(ctx.contains("FunÃ§Ã£o principal"), "Contexto deve conter o chunk");
    }
}

