//! Busca vetorial para integração com a base de conhecimento (RAG).
//!
//! O motor de inferência usa este módulo para buscar contextos relevantes (RAG)
//! e embuti-los no prompt dinamicamente antes da geração.
//!
//! ## Arquitetura ("o devido lugar" do banco vetorial)
//! `VectorSearch` é a interface RAG que o pipeline consome. Internamente ela é
//! servida pelo [`crate::vector_store::VectorStore`] nativo — um banco híbrido
//! **HNSW (semântico) + BM25 (literal) + RRF Fusion**, em Rust puro, sempre ativo
//! e sem dependência pesada. Isso dá RAG funcional out-of-the-box em qualquer
//! máquina. O recurso opcional `lancedb_native` adiciona persistência em disco
//! (LanceDB) para coleções em escala de datacenter, sem mudar esta interface.

use nodestor_core::NodeStorError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::vector_store::VectorStore;

/// Resultado de uma busca vetorial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// ID do documento ou fragmento na base de dados.
    pub id: String,
    /// Score de similaridade (Cosseno para vetorial; RRF para híbrida).
    pub score: f32,
    /// Texto ou payload de metadados associado.
    pub payload: Option<String>,
}

/// Interface assíncrona para o motor de busca vetorial integrado (RAG).
pub struct VectorSearch {
    #[cfg(feature = "lancedb_native")]
    client: std::sync::Arc<tokio::sync::Mutex<Option<lancedb::Connection>>>,
    collection_name: String,
    #[allow(dead_code)]
    db_path: String,
    /// Banco vetorial híbrido em-processo (HNSW + BM25 + RRF). Sempre ativo.
    store: std::sync::Mutex<VectorStore>,
    /// Dimensão dos embeddings produzidos por [`VectorSearch::embed_text`].
    embed_dim: usize,
}

impl VectorSearch {
    pub fn new(collection_name: &str, db_path: &str) -> Self {
        Self {
            #[cfg(feature = "lancedb_native")]
            client: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            collection_name: collection_name.to_string(),
            db_path: db_path.to_string(),
            store: std::sync::Mutex::new(VectorStore::new(db_path)),
            embed_dim: 128,
        }
    }

    /// Embedding determinístico texto→vetor via "hashing trick" (feature hashing).
    ///
    /// Cada token (palavra + trigrama de caractere) é hasheado para uma dimensão
    /// com sinal; o vetor resultante é L2-normalizado. É determinístico, não exige
    /// modelo externo e captura similaridade lexical — suficiente para RAG factual.
    /// Pode ser trocado por um encoder neural real mantendo esta mesma assinatura.
    pub fn embed_text(text: &str, dim: usize) -> Vec<f32> {
        let dim = dim.max(1);
        let mut v = vec![0.0f32; dim];
        let lower = text.to_lowercase();

        let mut add_feature = |s: &str| {
            let h = fnv1a64(s.as_bytes());
            let idx = (h % dim as u64) as usize;
            let sign = if (h >> 8) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        };

        // Palavras (≥ 2 caracteres alfanuméricos)
        for w in lower.split(|c: char| !c.is_alphanumeric()).filter(|s| s.len() >= 2) {
            add_feature(w);
        }
        // Trigramas de caractere (robustez a variações morfológicas/typos)
        let chars: Vec<char> = lower.chars().filter(|c| !c.is_whitespace()).collect();
        for win in chars.windows(3) {
            let tri: String = win.iter().collect();
            add_feature(&tri);
        }

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() { *x /= norm; }
        }
        v
    }

    /// Dimensão de embedding configurada para este índice.
    pub fn embed_dim(&self) -> usize { self.embed_dim }

    /// Número de documentos atualmente indexados.
    pub fn document_count(&self) -> usize {
        self.store.lock().map(|s| s.document_count()).unwrap_or(0)
    }

    #[cfg(feature = "lancedb_native")]
    #[allow(dead_code)]
    async fn get_table(&self) -> Result<lancedb::Table, NodeStorError> {
        let mut guard = self.client.lock().await;
        if guard.is_none() {
            let conn = lancedb::connect(&self.db_path).execute().await
                .map_err(|e| NodeStorError::ConfigError(format!("LanceDB falhou: {}", e)))?;
            *guard = Some(conn);
        }
        let conn = guard.as_ref().unwrap();
        conn.open_table(&self.collection_name).execute().await
            .map_err(|e| NodeStorError::ConfigError(format!("Tabela vetorial não encontrada: {}", e)))
    }

    /// Insere um registro já embeddado (ex.: indexação de Eviction do KV cache).
    pub async fn insert_document(
        &self,
        id: String,
        embedding: Vec<f32>,
        payload: Option<String>,
    ) -> Result<(), NodeStorError> {
        let text = payload.unwrap_or_default();
        if let Ok(mut store) = self.store.lock() {
            store.insert(&id, &embedding, &text, HashMap::new())?;
        }
        tracing::debug!("VectorSearch: registro '{}' indexado em '{}'", id, self.collection_name);
        Ok(())
    }

    /// Busca por similaridade vetorial pura (KNN) sobre o índice HNSW.
    pub async fn search_knn(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, NodeStorError> {
        if query_embedding.is_empty() {
            return Err(NodeStorError::ConfigError("Embedding vazio".into()));
        }

        let results = match self.store.lock() {
            Ok(mut store) => store.vector_search(query_embedding, top_k),
            Err(_) => Vec::new(),
        };

        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            score: r.vector_score,
            payload: Some(r.text),
        }).collect())
    }

    /// Busca híbrida a partir de TEXTO: embeda a query e funde HNSW + BM25 via RRF.
    /// É o caminho recomendado para RAG (resolve tanto conceito quanto match exato
    /// de IDs/números de série).
    pub async fn search_text(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, NodeStorError> {
        let emb = Self::embed_text(query_text, self.embed_dim);
        let results = match self.store.lock() {
            Ok(mut store) => store.hybrid_search(&emb, query_text, top_k),
            Err(_) => Vec::new(),
        };
        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            score: r.combined_score,
            payload: Some(r.text),
        }).collect())
    }

    /// Indexa um documento inteiro: embeda o conteúdo e insere no HNSW + BM25.
    pub async fn add_document(
        &self,
        path: &str,
        content: &str,
    ) -> Result<(), NodeStorError> {
        let emb = Self::embed_text(content, self.embed_dim);
        if let Ok(mut store) = self.store.lock() {
            store.insert(path, &emb, content, HashMap::new())?;
        }
        tracing::debug!("VectorSearch: documento indexado (HNSW+BM25): {}", path);
        Ok(())
    }

    /// Insere/atualiza um fragmento de conhecimento com embedding explícito.
    pub async fn upsert(
        &self,
        id: &str,
        embedding: &[f32],
        payload: &str,
    ) -> Result<(), NodeStorError> {
        if let Ok(mut store) = self.store.lock() {
            store.insert(id, embedding, payload, HashMap::new())?;
        }
        tracing::debug!("VectorSearch upsert: '{}' em '{}' (dim={})", id, self.collection_name, embedding.len());
        Ok(())
    }
}

/// FNV-1a 64-bit — hash determinístico e rápido para o feature hashing.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_search_text_returns_relevant_doc() {
        let vs = VectorSearch::new("kb", "/tmp/nodestor_vs_search_test");
        vs.add_document("doc_ml", "machine learning and neural networks for deep models").await.unwrap();
        vs.add_document("doc_db", "database indexing and SQL query optimization").await.unwrap();
        vs.add_document("doc_cook", "a recipe for chocolate cake with sugar and flour").await.unwrap();

        let results = vs.search_text("how do neural networks learn", 2).await.unwrap();
        assert!(!results.is_empty(), "deve retornar resultados");
        assert_eq!(results[0].id, "doc_ml", "o doc de ML deve liderar para uma query de ML, veio '{}'", results[0].id);
    }

    #[tokio::test]
    async fn test_fts_exact_id_match_via_hybrid() {
        let vs = VectorSearch::new("kb", "/tmp/nodestor_vs_fts_test");
        vs.add_document("a", "the sensor reading was nominal today").await.unwrap();
        vs.add_document("b", "part SN-9823-X failed inspection").await.unwrap();
        let results = vs.search_text("SN-9823-X", 3).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "b", "match exato de ID deve liderar via BM25+RRF");
    }

    #[tokio::test]
    async fn test_infinite_context_recall_of_evicted_fact() {
        // CONTEXTO INFINITO: histórico longo, janela pequena. Os blocos antigos
        // "saem da janela deslizante" e são INDEXADOS aqui; uma pergunta MUITO depois
        // os RECUPERA via HNSW+BM25, mesmo já não estando na atenção ativa.
        let vs = VectorSearch::new("history", "/tmp/nodestor_infctx_test");
        // Turnos antigos, já despejados da janela ativa:
        vs.add_document("turn_01", "Decision: the fiscal API endpoint is /v2/nfe with a 30 second timeout.").await.unwrap();
        vs.add_document("turn_02", "The user's cat is named Pixel and their favorite color is teal.").await.unwrap();
        vs.add_document("turn_03", "Deployment runs Docker on port 8080 behind nginx with TLS.").await.unwrap();
        vs.add_document("turn_04", "Lunch was pizza; the standup moved to 3pm on Fridays.").await.unwrap();
        vs.add_document("turn_05", "We chose Rust for the engine and Python for the CLI wrapper.").await.unwrap();

        // Pergunta posterior sobre um fato ANTIGO (fora da janela): o loop o traz de volta.
        let results = vs.search_text("What did we decide about the fiscal API endpoint?", 1).await.unwrap();
        assert!(!results.is_empty(), "deve recuperar o bloco despejado");
        assert_eq!(results[0].id, "turn_01", "recupera o turno da decisão da API fiscal");
        assert!(results[0].payload.as_deref().unwrap_or("").contains("/v2/nfe"),
            "o fato exato volta no payload recuperado");
    }

    #[tokio::test]
    async fn test_search_knn_with_embedding() {
        let vs = VectorSearch::new("kb", "/tmp/nodestor_vs_knn_test");
        vs.add_document("x", "vulkan gpu compute shaders and pipelines").await.unwrap();
        vs.add_document("y", "italian pasta and tomato sauce").await.unwrap();
        // Embeda a mesma intenção da query e busca por vetor puro.
        let q = VectorSearch::embed_text("gpu shader pipeline", vs.embed_dim());
        let results = vs.search_knn(&q, 1).await.unwrap();
        assert_eq!(results[0].id, "x", "kNN vetorial deve achar o doc de GPU");
    }

    #[test]
    fn test_embed_text_deterministic_and_normalized() {
        let a = VectorSearch::embed_text("hello world", 128);
        let b = VectorSearch::embed_text("hello world", 128);
        assert_eq!(a, b, "embedding deve ser determinístico");
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4 || norm == 0.0, "embedding deve ser L2-normalizado, norm={}", norm);
    }

    #[tokio::test]
    async fn test_empty_query_embedding_rejected() {
        let vs = VectorSearch::new("kb", "/tmp/nodestor_vs_empty_test");
        assert!(vs.search_knn(&[], 3).await.is_err(), "embedding vazio deve ser rejeitado");
    }
}
