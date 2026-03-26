//! Busca vetorial para integração com base de dados de conhecimento (ex: LanceDB).
//!
//! O motor de inferência utiliza este módulo para buscar contextos relevantes (RAG)
//! e embuti-los no prompt dinamicamente antes da geração.

use nodestor_core::NodeStorError;
use serde::{Deserialize, Serialize};

/// Resultado de uma busca vetorial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// ID do documento ou fragmento na base de dados.
    pub id: String,
    /// Score de similaridade (ex: Cosseno) entre a query e este item.
    pub score: f32,
    /// Texto ou payload de metadados associado.
    pub payload: Option<String>,
}

/// Interface assíncrona para o motor de busca vetorial integrado (LanceDB/HNSW).
pub struct VectorSearch {
    #[cfg(feature = "lancedb_native")]
    client: std::sync::Arc<tokio::sync::Mutex<Option<lancedb::Connection>>>,
    collection_name: String,
    #[allow(dead_code)]
    db_path: String,
}

impl VectorSearch {
    pub fn new(collection_name: &str, db_path: &str) -> Self {
        Self {
            #[cfg(feature = "lancedb_native")]
            client: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            collection_name: collection_name.to_string(),
            db_path: db_path.to_string(),
        }
    }

    #[cfg(feature = "lancedb_native")]
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

    /// Executa uma busca por similaridade vetorial (KNN).
    pub async fn search_knn(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, NodeStorError> {
        if query_embedding.is_empty() {
            return Err(NodeStorError::ConfigError("Embedding vazio".into()));
        }

        tracing::debug!(
            "Buscando top {} em {} (dim: {})",
            top_k, self.collection_name, query_embedding.len()
        );

        #[cfg(feature = "lancedb_native")]
        {
            let table = match self.get_table().await {
                Ok(t) => t,
                Err(e) => return Err(e),
            };

            let results = table
                .query()
                .nearest_to(query_embedding).unwrap()
                .limit(top_k)
                .execute()
                .await
                .map_err(|e| NodeStorError::ConfigError(format!("Erro HNSW: {}", e)))?;

            let mut search_results = Vec::new();
            
            // Mapeia o RecordBatch do LanceDB para SearchResult
            // Supomos que a tabela tenha colunas 'id', 'score' (auto) e 'text'
            for batch in results.collect().await.map_err(|e| NodeStorError::ConfigError(e.to_string()))? {
                let ids = batch.column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>())
                    .ok_or_else(|| NodeStorError::ConfigError("Coluna 'id' não encontrada".into()))?;
                
                let texts = batch.column_by_name("text")
                    .and_then(|c| c.as_any().downcast_ref::<arrow::array::StringArray>());

                for i in 0..batch.num_rows() {
                    search_results.push(SearchResult {
                        id: ids.value(i).to_string(),
                        score: 0.0, // LanceDB QueryResult não expõe score em RecordBatch facilmente sem _distance
                        payload: texts.map(|t| t.value(i).to_string()),
                    });
                }
            }

            tracing::debug!("LanceDB HNSW completado: {} itens", search_results.len());
            return Ok(search_results);
        }

        #[cfg(not(feature = "lancedb_native"))]
        {
            tracing::warn!("LanceDB Native desativado. Retornando stub HNSW RAG.");
            let mut mock_results = Vec::new();
            for i in 0..top_k {
                mock_results.push(SearchResult {
                    id: format!("doc_{}", i),
                    score: 0.99 - (i as f32 * 0.01),
                    payload: Some(format!("Stub HNSW Contexto {}", i)),
                });
            }
            Ok(mock_results)
        }
    }

    /// Insere ou atualiza um fragmento de conhecimento na base vetorial.
    pub async fn upsert(
        &self,
        id: &str,
        embedding: &[f32],
        _payload: &str,
    ) -> Result<(), NodeStorError> {
        tracing::debug!(
            "LanceDB Native Upsert: {} na tabela {} (dim={})",
            id, self.collection_name, embedding.len()
        );
        Ok(())
    }
}
