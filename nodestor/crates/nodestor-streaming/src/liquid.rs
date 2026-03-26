use std::sync::Arc;
use nodestor_core::{DataTransport, NodeStorError, LiquidTransferRequest, TransferResult};
use nodestor_vulkan::VulkanEngine;
use tokio::sync::mpsc;
use tracing::{info, debug};

pub struct LiquidChunk {
    pub chunk_index: usize,
    pub data: Vec<u8>,
    pub transfer_us: u64,
}

/// Orquestrador do Liquid Streaming Pipeline.
///
/// Gerencia a "Corrida de Latência" e o pipeline de micro-fatias.
pub struct LiquidOrchestrator {
    vulkan: Arc<VulkanEngine>,
    transport_a: Arc<dyn DataTransport + Send + Sync>,
    transport_b: Option<Arc<dyn DataTransport + Send + Sync>>,
}

impl LiquidOrchestrator {
    pub fn new(
        vulkan: Arc<VulkanEngine>,
        transport_a: Arc<dyn DataTransport + Send + Sync>,
        transport_b: Option<Arc<dyn DataTransport + Send + Sync>>,
    ) -> Self {
        Self { vulkan, transport_a, transport_b }
    }

    /// Executa o streaming líquido com Corrida de Latência.
    pub async fn stream_liquid(
        &self,
        request: LiquidTransferRequest,
    ) -> Result<(), NodeStorError> {
        info!("🌊 Liquid Streaming: Iniciando corrida p/ '{}'", request.tensor_name);
        
        let (tx, mut rx) = mpsc::channel::<TransferResult>(10);
        let request_arc = Arc::new(request);
        
        let tx_clone = tx.clone();
        let t_a = self.transport_a.clone();
        let r_a = request_arc.clone();
        
        // Competidor A: Win32 Overlapped V2 / Native Fast Path
        tokio::spawn(async move {
            let callback = Box::new(move |res| {
                let _ = tx_clone.blocking_send(res);
            });
            let _ = t_a.transfer_liquid(&r_a, callback);
        });

        // Competidor B: DirectStorage (se disponível)
        if let Some(t_b) = &self.transport_b {
            let tx_clone_b = tx.clone();
            let t_b_clone = t_b.clone();
            let r_b = request_arc.clone();
            tokio::spawn(async move {
                let callback = Box::new(move |res| {
                    let _ = tx_clone_b.blocking_send(res);
                });
                let _ = t_b_clone.transfer_liquid(&r_b, callback);
            });
        }

        // Processador de fatias - o primeiro que chegar ganha
        let mut chunks_processed = 0;
        let total_chunks = (request_arc.total_size + request_arc.chunk_size - 1) / request_arc.chunk_size;

        while let Some(res) = rx.recv().await {
            debug!("   Líquido: Fatia de {} bytes recebida.", res.data.len());
            
            // Dispatch para GPU imediatamente
            let output_elements = res.data.len() / 4; // F32
            let (_gpu_buf, fence) = self.vulkan.decompress_liquid(&res.data, output_elements)?;
            
            // Incrementamos
            chunks_processed += 1;
            if chunks_processed >= total_chunks {
                // TODO: Limpar Fences remanescentes em produção
                unsafe {
                    let device = self.vulkan.ctx.device().unwrap();
                    device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
                }
                break;
            }
        }

        info!("🏆 Liquid Streaming: {} finalizado com sucesso.", request_arc.tensor_name);
        Ok(())
    }
}
