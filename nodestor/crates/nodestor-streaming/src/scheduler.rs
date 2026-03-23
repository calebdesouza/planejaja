//! Scheduler de streaming preditivo de tensores.
//!
//! Inteligência que alimenta a Metralhadora com os próximos tensores
//! exatos que a GPU vai precisar inferir, na ordem do DAG do modelo.

use crate::metralhadora::{MesPrefetchQueue, PrefetchedBlock};
use nodestor_core::{ModelMetadata, NodeStorError, TransferRequest};
use std::sync::Arc;
use tracing::warn;

/// Agendador central do pipeline de streaming sequencial.
///
/// Mantém a Metralhadora alimentada. Em um modelo Transformer, os tensores
/// são lidos na ordem dos blocos (atenção, MLP, etc). O scheduler prevê isso
/// e mantém o SSD sempre ocupado carregando os blocos à frente do tempo.
pub struct StreamScheduler {
    pub prefetch_depth: usize,
    queue: MesPrefetchQueue,
    metadata: Arc<ModelMetadata>,
    next_tensor_idx: usize,
}

impl StreamScheduler {
    pub fn new(
        prefetch_depth: usize,
        queue: MesPrefetchQueue,
        metadata: Arc<ModelMetadata>,
    ) -> Self {
        Self {
            prefetch_depth,
            queue,
            metadata,
            next_tensor_idx: 0,
        }
    }

    /// "Prime the pump": Enche o buffer inicial com os N primeiros tensores
    /// para que a GPU nunca tenha que esperar pela primeira inferência.
    pub async fn prime_pump(&mut self) -> Result<(), NodeStorError> {
        for _ in 0..self.prefetch_depth {
            self.enqueue_next_tensor().await?;
        }
        Ok(())
    }

    /// Pega o próximo tensor já pronto na VRAM e engatilha a leitura do tensor (N + depth).
    pub async fn next_tensor(&mut self) -> Option<PrefetchedBlock> {
        // Enfileira preditivamente o próximo tensor para repor a vaga no Bufferpool
        if let Err(e) = self.enqueue_next_tensor().await {
            warn!("Erro no prefetch preditivo do scheduler: {}", e);
        }

        // Retira o bloco atual da ponta da fila
        self.queue.pop_ready_block().await
    }

    /// Alimenta a fila inferindo o offset e tamanho sequencial da rede.
    async fn enqueue_next_tensor(&mut self) -> Result<(), NodeStorError> {
        if self.metadata.tensors.is_empty() {
            return Err(NodeStorError::ConfigError("Metadata sem tensores".into()));
        }

        if self.next_tensor_idx >= self.metadata.tensors.len() {
            // Em LLMs, ao terminar de passar por todas as camadas, resetamos o índice
            // para recomeçar o forward pass p/ o próximo token.
            self.next_tensor_idx = 0;
        }

        let tensor_info = &self.metadata.tensors[self.next_tensor_idx];
        let req = TransferRequest {
            file_offset: tensor_info.data_offset,
            size: tensor_info.data_size as usize,
            // A flag de compressão pode ser ativada dinamicamente com base no SO
            // ou metadata (Gdeflate GPU no Windows DirectStorage, por ex.)
            compressed: false, 
        };

        self.queue.enqueue_prediction(req).await?;
        self.next_tensor_idx += 1;
        
        Ok(())
    }
}
