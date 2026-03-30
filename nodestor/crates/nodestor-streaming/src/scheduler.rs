//! Scheduler de streaming preditivo de tensores com pipeline Multi-dimensional (Burst/Speculative).
//!
//! Inteligência que alimenta a Metralhadora lançando bateladas em paralelo
//! baseadas no ExecutionPlan gerado em LayerGraph.

use crate::metralhadora::{MesPrefetchQueue, PrefetchedBlock};
use crate::layer_graph::{ExecutionPlan, LayerGraph};
use nodestor_core::{ModelMetadata, NodeStorError, TransferRequest};
use std::sync::Arc;
use tracing::warn;

/// O BurstScheduler enfileira grupos inteiros de tensores simultaneamente
/// maximizando o paralelismo do SSD NVMe em canais flash.
pub struct BurstScheduler {
    pub prefetch_depth: usize,
    queue: MesPrefetchQueue,
    plan: ExecutionPlan,
    next_group_idx: usize,
}

impl BurstScheduler {
    pub fn new(
        prefetch_depth: usize,
        queue: MesPrefetchQueue,
        metadata: Arc<ModelMetadata>,
    ) -> Self {
        // Gera o plano em tempo O(T) onde T é o nro de tensores (~300 em LLaMA)
        let plan = LayerGraph::build(&metadata);
        
        Self {
            prefetch_depth,
            queue,
            plan,
            next_group_idx: 0,
        }
    }

    /// Executa o cold start inundando a VRAM com as primeiras camadas
    pub async fn prime_pump(&mut self) -> Result<(), NodeStorError> {
        for _ in 0..self.prefetch_depth {
            self.enqueue_next_group().await?;
        }
        Ok(())
    }

    /// Pega o próximo bloco pronto e enfileira (prefetch) outro grupo inteiro dinamicamente
    pub async fn next_tensor(&mut self) -> Option<PrefetchedBlock> {
        // Note que o next_tensor em Burst continuará entregando "1 bloco por vez" para o Caller,
        // mas a magia é que nos bastidores a fila é inundada em rajadas (bursts).
        
        let block = self.queue.pop_ready_block().await;
        
        if let Some(_b) = &block {
            // Verifica na fila se estamos precisando repor. 
            // O ideal para Burst é repormos por grupos!
            // Para simplificar, nós disparamos a recarga logo após obtermos um certo checkpoint.
            // Para manter a esteira em movimento, disparamos a enfileirada:
            if let Err(e) = self.enqueue_next_group().await {
                 warn!("Erro no prefetch em modo burst do scheduler: {}", e);
            }
        }
        
        block
    }
    
    /// Este método também pode ser chamado para puxar um Grupo Inteiro
    pub async fn next_group_blocks(&mut self, tensors_in_group: usize) -> Vec<PrefetchedBlock> {
        let mut blocks = Vec::with_capacity(tensors_in_group);
        
        for _ in 0..tensors_in_group {
            if let Some(b) = self.queue.pop_ready_block().await {
                blocks.push(b);
            }
        }
        
        // Repõe grupo preditivamente
        if let Err(e) = self.enqueue_next_group().await {
             warn!("Erro no prefetch de grupo: {}", e);
        }
        
        blocks
    }

    /// Dispara N requisições de uma vez para saturar o barramento PCIe e canais flash.
    pub async fn enqueue_next_group(&mut self) -> Result<(), NodeStorError> {
        if self.plan.groups.is_empty() {
            return Err(NodeStorError::ConfigError("Execution plan sem grupos".into()));
        }

        // Loop causal: ao terminar um pass, começamos o pass do próximo token
        if self.next_group_idx >= self.plan.groups.len() {
            self.next_group_idx = 0;
        }

        let group = &self.plan.groups[self.next_group_idx];
        
        // Categoria 1: Para backends que suportam BATCH REAL (io_uring)
        // enfileirar como uma batelada vetorial seria o ideal.
        // Aqui nós faremos enfileiramento linear para a metralhadora que internamente 
        // agora saberá fazer fetch_batch quando agrupados (ainda enviaremos 1 a 1 para o Request TX).
        let mut requests = Vec::with_capacity(group.tensors.len());
        for slice in &group.tensors {
            let req = TransferRequest {
                file_offset: slice.offset,
                size: slice.size as usize,
                compressed: false, // Pode ser inferido de heurísticas mais tarde
            };
            requests.push(req);
        }

        self.queue.enqueue_burst(requests).await?;
        self.next_group_idx += 1;
        Ok(())
    }
}
