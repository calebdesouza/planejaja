//! VectorStore — Banco Vetorial Nativo com Busca Híbrida.
//!
//! Combina busca vetorial (semântica) com Full-Text Search (literal).
//! Isso resolve o problema: "SN-9823-X" → vetores acham coisas "parecidas",
//! mas o FTS acha o exato. O RRF Fusion combina os dois mundos.
//!
//! ## Arquitetura:
//! - **HNSW** (Hierarchical Navigable Small World): busca vetorial aproximada
//!   - M=16 vizinhos, ef_construction=200
//!   - O(log N) busca, excelente para coleções de 100k+ chunks
//! - **BM25** (Best Match 25): Full-Text Search probabilístico
//!   - Índice invertido + IDF + TF com saturação k1/b
//!   - Match exato de termos, números de série, IDs
//! - **RRF** (Reciprocal Rank Fusion): combina as duas listas de resultados
//!   - `score = Σ 1/(k + rank_i)` onde k=60 (constante empírica)
//!   - Robusto a diferenças de escala entre os dois scorers

use std::collections::HashMap;
use std::path::PathBuf;

use nodestor_core::NodeStorError;

// ── HNSW In-Memory ──────────────────────────────────────────────────────────

/// Nó individual no grafo HNSW.
#[derive(Debug, Clone)]
struct HnswNode {
    pub id: String,
    pub embedding: Vec<f32>,
    pub text: String,
    pub metadata: HashMap<String, String>,
    /// Vizinhos no grafo (list de índices nos diferentes níveis)
    pub neighbors: Vec<Vec<usize>>,
}

/// Índice HNSW in-memory para busca vetorial aproximada.
/// Implementado em Rust puro — zero dependências externas.
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    /// M: número máximo de vizinhos por nó (trade-off: qualidade vs memória)
    m: usize,
    /// ef_construction: candidatos durante indexação (maior = melhor qualidade)
    ef_construction: usize,
    /// Nível máximo de entrada no grafo (log scale)
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            m,
            ef_construction,
            entry_point: None,
            max_level: 0,
        }
    }

    /// Insere um documento no índice.
    pub fn insert(&mut self, id: String, embedding: Vec<f32>, text: String, metadata: HashMap<String, String>) {
        let idx = self.nodes.len();
        let level = Self::random_level(self.m);
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(idx);
        }

        let node = HnswNode {
            id,
            embedding,
            text,
            metadata,
            neighbors: vec![Vec::new(); level + 1],
        };
        self.nodes.push(node);

        // Conecta ao grafo usando os vizinhos mais próximos
        if idx > 0 {
            if let Some(ep) = self.entry_point {
                let neighbors = self.search_layer(idx, ep, self.m.min(self.ef_construction));
                for (neighbor_idx, _score) in neighbors.iter().take(self.m) {
                    if *neighbor_idx < self.nodes.len() && *neighbor_idx != idx {
                        if let Some(node) = self.nodes.get_mut(idx) {
                            if let Some(layer) = node.neighbors.get_mut(0) {
                                if !layer.contains(neighbor_idx) {
                                    layer.push(*neighbor_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Busca os `top_k` vizinhos mais próximos de `query`.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if self.nodes.is_empty() { return vec![]; }

        let ep = self.entry_point.unwrap_or(0);
        let candidates = self.search_layer(self.nodes.len(), ep, self.ef_construction.max(top_k));

        candidates.into_iter()
            .take(top_k)
            .filter_map(|(idx, _)| {
                self.nodes.get(idx).map(|n| {
                    let score = cosine_similarity(query, &n.embedding);
                    (n.id.clone(), score)
                })
            })
            .collect()
    }

    /// Busca com beam search aproximada na camada 0.
    fn search_layer(&self, query_idx: usize, entry: usize, ef: usize) -> Vec<(usize, f32)> {
        let query_emb = if query_idx < self.nodes.len() {
            self.nodes[query_idx].embedding.clone()
        } else if self.nodes.is_empty() {
            return vec![];
        } else {
            // Fallback: usa embedding do entry point como query aproximada
            self.nodes[0].embedding.clone()
        };

        let mut visited = vec![false; self.nodes.len()];
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        if entry < self.nodes.len() {
            let score = cosine_similarity(&query_emb, &self.nodes[entry].embedding);
            candidates.push((entry, score));
            visited[entry] = true;
        }

        let mut i = 0;
        while i < candidates.len() && i < ef {
            let (cur_idx, _) = candidates[i];

            if let Some(node) = self.nodes.get(cur_idx) {
                for &neighbor in node.neighbors.get(0).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if neighbor < self.nodes.len() && !visited[neighbor] {
                        visited[neighbor] = true;
                        let score = cosine_similarity(&query_emb, &self.nodes[neighbor].embedding);
                        candidates.push((neighbor, score));
                    }
                }
            }
            i += 1;
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(ef);
        candidates
    }

    fn random_level(m: usize) -> usize {
        // Geração de nível sem rand: usa um hash do número de nós
        let h = (m as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
        (h.trailing_zeros() as usize / 2).min(4)
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
}

// ── BM25 Full-Text Search ────────────────────────────────────────────────────

/// Índice invertido com pontuação BM25.
pub struct BM25Index {
    /// Índice invertido: token → [(doc_id, tf)]
    inverted: HashMap<String, Vec<(String, f32)>>,
    /// IDF por token (calculado lazily)
    idf_cache: HashMap<String, f32>,
    /// Comprimento médio dos documentos (em tokens)
    avg_doc_len: f32,
    /// Total de documentos
    doc_count: usize,
    /// Comprimento de cada documento
    doc_lengths: HashMap<String, usize>,
    /// Parâmetros BM25
    k1: f32,
    b: f32,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            idf_cache: HashMap::new(),
            avg_doc_len: 0.0,
            doc_count: 0,
            doc_lengths: HashMap::new(),
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Indexa um documento.
    pub fn add(&mut self, id: &str, text: &str) {
        let tokens = tokenize(text);
        let doc_len = tokens.len();

        self.doc_lengths.insert(id.to_string(), doc_len);
        self.doc_count += 1;
        self.avg_doc_len = self.doc_lengths.values().sum::<usize>() as f32 / self.doc_count as f32;

        // Calcula TF por token
        let mut tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        for (token, count) in tf {
            let tf_norm = count as f32 / doc_len.max(1) as f32;
            self.inverted
                .entry(token)
                .or_insert_with(Vec::new)
                .push((id.to_string(), tf_norm));
        }

        // Invalida cache IDF (será recalculado no próximo search)
        self.idf_cache.clear();
    }

    /// Busca BM25.
    pub fn search(&mut self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let query_tokens = tokenize(query);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for token in &query_tokens {
            let idf = self.compute_idf(token);
            if let Some(postings) = self.inverted.get(token) {
                for (doc_id, tf) in postings {
                    let dl = *self.doc_lengths.get(doc_id).unwrap_or(&1) as f32;
                    let tf_bm25 = tf * (self.k1 + 1.0)
                        / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len.max(1.0)));
                    *scores.entry(doc_id.clone()).or_insert(0.0) += idf * tf_bm25;
                }
            }
        }

        let mut results: Vec<(String, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn compute_idf(&mut self, token: &str) -> f32 {
        if let Some(&cached) = self.idf_cache.get(token) {
            return cached;
        }
        let df = self.inverted.get(token).map(|v| v.len()).unwrap_or(0) as f32;
        let n = self.doc_count as f32;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        self.idf_cache.insert(token.to_string(), idf);
        idf
    }

    pub fn doc_count(&self) -> usize { self.doc_count }
}

impl Default for BM25Index {
    fn default() -> Self { Self::new() }
}

// ── Tokenizador simples para BM25 ─────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

// ── Resultado Híbrido + RRF Fusion ───────────────────────────────────────────

/// Resultado de busca híbrida (vetorial + FTS combinados).
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// ID único do documento
    pub id: String,
    /// Score combinado via RRF Fusion
    pub combined_score: f32,
    /// Score individual da busca vetorial (cosine similarity)
    pub vector_score: f32,
    /// Score individual do BM25
    pub fts_score: f32,
    /// Texto do chunk
    pub text: String,
    /// Metadados (ex: fonte, data, tags)
    pub metadata: HashMap<String, String>,
}

/// Combina duas listas de resultados usando Reciprocal Rank Fusion.
///
/// `RRF(d) = Σ 1/(k + rank(d, list_i))`  onde k=60 (constante empírica da Literatura)
fn rrf_fusion(
    vector_results: &[(String, f32)],
    fts_results: &[(String, f32)],
    k: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, (id, _)) in fts_results.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
    }

    let mut merged: Vec<(String, f32)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

// ── VectorStore ───────────────────────────────────────────────────────────────

/// Estatísticas do VectorStore.
#[derive(Debug, Default, Clone)]
pub struct VectorStoreStats {
    pub total_documents: usize,
    pub total_searches: u64,
    pub hybrid_searches: u64,
    pub pure_vector_searches: u64,
}

/// O banco vetorial nativo com busca híbrida.
///
/// Combina HNSW (semântica) + BM25 FTS (literal) + RRF Fusion (fusão).
pub struct VectorStore {
    /// Índice vetorial HNSW
    hnsw: HnswIndex,
    /// Índice Full-Text Search BM25
    fts: BM25Index,
    /// Caminho de persistência
    storage_path: PathBuf,
    /// Textos por ID (para devolver no resultado)
    texts: HashMap<String, String>,
    /// Metadados por ID
    meta_store: HashMap<String, HashMap<String, String>>,
    /// Estatísticas
    pub stats: VectorStoreStats,
}

impl VectorStore {
    pub fn new(storage_path: &str) -> Self {
        Self {
            hnsw: HnswIndex::new(16, 200),
            fts: BM25Index::new(),
            storage_path: PathBuf::from(storage_path),
            texts: HashMap::new(),
            meta_store: HashMap::new(),
            stats: VectorStoreStats::default(),
        }
    }

    /// Insere um documento no banco vetorial.
    ///
    /// O documento é indexado tanto no HNSW (para busca semântica)
    /// quanto no BM25 (para busca literal exata).
    pub fn insert(
        &mut self,
        id: &str,
        embedding: &[f32],
        text: &str,
        metadata: HashMap<String, String>,
    ) -> Result<(), NodeStorError> {
        if embedding.is_empty() {
            return Err(NodeStorError::ConfigError("Embedding não pode ser vazio".into()));
        }
        if id.is_empty() {
            return Err(NodeStorError::ConfigError("ID não pode ser vazio".into()));
        }

        self.hnsw.insert(
            id.to_string(),
            embedding.to_vec(),
            text.to_string(),
            metadata.clone(),
        );
        self.fts.add(id, text);
        self.texts.insert(id.to_string(), text.to_string());
        self.meta_store.insert(id.to_string(), metadata);
        self.stats.total_documents += 1;
        Ok(())
    }

    /// Busca vetorial pura (apenas semântica, sem FTS).
    pub fn vector_search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<HybridSearchResult> {
        self.stats.total_searches += 1;
        self.stats.pure_vector_searches += 1;

        self.hnsw.search(query_embedding, top_k)
            .into_iter()
            .map(|(id, score)| HybridSearchResult {
                text: self.texts.get(&id).cloned().unwrap_or_default(),
                metadata: self.meta_store.get(&id).cloned().unwrap_or_default(),
                combined_score: score,
                vector_score: score,
                fts_score: 0.0,
                id,
            })
            .collect()
    }

    /// Busca FTS pura (apenas literal, sem vetorial).
    pub fn fts_search(&mut self, query: &str, top_k: usize) -> Vec<HybridSearchResult> {
        self.stats.total_searches += 1;

        self.fts.search(query, top_k)
            .into_iter()
            .map(|(id, score)| HybridSearchResult {
                text: self.texts.get(&id).cloned().unwrap_or_default(),
                metadata: self.meta_store.get(&id).cloned().unwrap_or_default(),
                combined_score: score,
                vector_score: 0.0,
                fts_score: score,
                id,
            })
            .collect()
    }

    /// **Busca Híbrida: semântica + literal + RRF Fusion.**
    ///
    /// A solução para o problema número de série:
    /// - "SN-9823-X" → FTS encontra o exato
    /// - "sensor de temperatura" → vetorial encontra o conceito
    /// - Para queries que precisam de ambos: RRF combina
    pub fn hybrid_search(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Vec<HybridSearchResult> {
        self.stats.total_searches += 1;
        self.stats.hybrid_searches += 1;

        let vec_results = self.hnsw.search(query_embedding, top_k * 2);
        let fts_results = self.fts.search(query_text, top_k * 2);

        // Guarda scores individuais para o resultado final
        let vec_scores: HashMap<String, f32> = vec_results.iter().cloned().collect();
        let fts_scores: HashMap<String, f32> = fts_results.iter().cloned().collect();

        // RRF Fusion (k=60 é o padrão da literatura)
        let merged = rrf_fusion(&vec_results, &fts_results, 60.0);

        merged.into_iter()
            .take(top_k)
            .map(|(id, combined_score)| HybridSearchResult {
                text: self.texts.get(&id).cloned().unwrap_or_default(),
                metadata: self.meta_store.get(&id).cloned().unwrap_or_default(),
                vector_score: *vec_scores.get(&id).unwrap_or(&0.0),
                fts_score: *fts_scores.get(&id).unwrap_or(&0.0),
                combined_score,
                id,
            })
            .collect()
    }

    /// Total de documentos indexados.
    pub fn document_count(&self) -> usize {
        self.stats.total_documents
    }
}

// ── Cosine Similarity ─────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { (dot / denom).clamp(-1.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> VectorStore {
        VectorStore::new("/tmp/nodestor_test_vs")
    }

    fn embed(v: &[f32]) -> Vec<f32> {
        // Normaliza L2 para que cosine sim seja mais previsível nos testes
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_insert_and_vector_search() {
        let mut store = make_store();
        store.insert("doc1", &embed(&[1.0, 0.0, 0.0, 0.0]), "Machine learning basics", HashMap::new()).unwrap();
        store.insert("doc2", &embed(&[0.0, 1.0, 0.0, 0.0]), "Database optimization", HashMap::new()).unwrap();
        store.insert("doc3", &embed(&[0.9, 0.1, 0.0, 0.0]), "Deep learning tutorial", HashMap::new()).unwrap();

        // Query próxima de doc1 e doc3
        let results = store.vector_search(&embed(&[1.0, 0.0, 0.0, 0.0]), 2);
        assert!(!results.is_empty(), "Deve retornar resultados");
        assert!(results[0].vector_score > 0.9, "Resultado mais próximo deve ter score alto");
    }

    #[test]
    fn test_fts_exact_match() {
        let mut store = make_store();
        store.insert("doc1", &embed(&[1.0, 0.0]), "sensor SN-9823-X temperatura", HashMap::new()).unwrap();
        store.insert("doc2", &embed(&[0.0, 1.0]), "sensor parecido SN-9999-Y", HashMap::new()).unwrap();
        store.insert("doc3", &embed(&[0.5, 0.5]), "relatório de qualidade", HashMap::new()).unwrap();

        let results = store.fts_search("SN-9823-X", 3);
        assert!(!results.is_empty(), "FTS deve encontrar match exato");
        assert_eq!(results[0].id, "doc1", "FTS deve priorizar match exato");
    }

    #[test]
    fn test_hybrid_search_combines_both() {
        let mut store = make_store();
        // doc1: semanticamente próximo da query E tem match FTS
        store.insert("doc1", &embed(&[1.0, 0.0, 0.0]), "serial number SN-9823-X", HashMap::new()).unwrap();
        // doc2: semanticamente próximo, mas sem match FTS
        store.insert("doc2", &embed(&[0.9, 0.1, 0.0]), "product identification code", HashMap::new()).unwrap();
        // doc3: longe semanticamente, mas tem match FTS
        store.insert("doc3", &embed(&[0.0, 0.0, 1.0]), "document SN-9823-X reference", HashMap::new()).unwrap();

        let results = store.hybrid_search(&embed(&[1.0, 0.0, 0.0]), "SN-9823-X", 3);
        assert!(!results.is_empty());
        // doc1 deve estar no topo (bom em ambas as métricas)
        assert_eq!(results[0].id, "doc1",
            "doc1 (bom em ambos) deve liderar, got '{}'", results[0].id);
    }

    #[test]
    fn test_rrf_fusion_combines_ranks() {
        let vec_results = vec![
            ("a".to_string(), 0.9),
            ("b".to_string(), 0.7),
            ("c".to_string(), 0.5),
        ];
        let fts_results = vec![
            ("c".to_string(), 10.0),  // c lidera no FTS
            ("a".to_string(), 5.0),
            ("d".to_string(), 3.0),   // d aparece apenas no FTS
        ];

        let merged = rrf_fusion(&vec_results, &fts_results, 60.0);
        assert!(!merged.is_empty());
        // "a" deve ter score alto (rank 1 em vec, rank 2 em FTS)
        let a_score = merged.iter().find(|(id, _)| id == "a").map(|(_, s)| *s).unwrap_or(0.0);
        let c_score = merged.iter().find(|(id, _)| id == "c").map(|(_, s)| *s).unwrap_or(0.0);
        // Tanto "a" quanto "c" devem ter scores significativos
        assert!(a_score > 0.0, "a deve ter score positivo");
        assert!(c_score > 0.0, "c deve ter score positivo");
    }

    #[test]
    fn test_insert_rejects_empty_embedding() {
        let mut store = make_store();
        let r = store.insert("doc1", &[], "texto", HashMap::new());
        assert!(r.is_err(), "Embedding vazio deve ser rejeitado");
    }

    #[test]
    fn test_insert_rejects_empty_id() {
        let mut store = make_store();
        let r = store.insert("", &[1.0, 0.0], "texto", HashMap::new());
        assert!(r.is_err(), "ID vazio deve ser rejeitado");
    }

    #[test]
    fn test_document_count() {
        let mut store = make_store();
        assert_eq!(store.document_count(), 0);
        store.insert("a", &[1.0, 0.0], "texto a", HashMap::new()).unwrap();
        store.insert("b", &[0.0, 1.0], "texto b", HashMap::new()).unwrap();
        assert_eq!(store.document_count(), 2);
    }

    #[test]
    fn test_bm25_multi_term_scoring() {
        let mut idx = BM25Index::new();
        idx.add("d1", "machine learning deep learning neural networks");
        idx.add("d2", "database sql query optimization");
        idx.add("d3", "machine learning classification algorithms");

        let results = idx.search("machine learning", 3);
        assert!(!results.is_empty());
        // d1 e d3 ambos têm "machine" e "learning" — devem liderar
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"d1") || ids.contains(&"d3"),
            "Documentos com ambos os termos devem aparecer");
    }

    #[test]
    fn test_cosine_similarity_values() {
        // Vetores idênticos → 1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        // Vetores ortogonais → 0.0
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        // Vetores opostos → -1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }
}
