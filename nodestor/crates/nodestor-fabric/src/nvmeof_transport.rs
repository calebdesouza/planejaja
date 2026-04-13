use nodestor_core::{DataTransport, NodeStorError, TransferRequest, TransferResult, LiquidTransferRequest, TransportBackend};
use crate::fabric_server::FabricServer;
use crate::rdma_sim::RdmaChannel;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

/// Implementa o trait `DataTransport` sobre o protocolo NVMe-oF.
///
/// Permite que o NodeStor use um servidor remoto de SSDs NVMe como se fosse
/// um disco local — mas com performance de rede (100 GbE / InfiniBand).
///
/// ## Integração com o Pipeline
/// ```text
/// InferencePipeline
///     └─ DataTransport (trait)
///           └─ NvmeOfTransport  (este módulo)
///                 ├─ RDMA Channel → Servidor NVMe-oF
///                 └─ FabricServer (simulado) → SSDs NVMe
/// ```
pub struct NvmeOfTransport {
    server: Arc<FabricServer>,
    rdma: RdmaChannel,
    model_id: String,
}

impl NvmeOfTransport {
    /// Cria um transporte NVMe-oF conectado a um servidor remoto.
    pub fn new(server: Arc<FabricServer>, model_id: &str, use_infiniband: bool) -> Self {
        let rdma = if use_infiniband {
            RdmaChannel::new_infiniband(&server.bind_addr)
        } else {
            RdmaChannel::new_100gbe(&server.bind_addr)
        };
        
        Self { server, rdma, model_id: model_id.to_string() }
    }
}

impl DataTransport for NvmeOfTransport {
    fn transfer(
        &self,
        tensor_name: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError> {
        let start = Instant::now();
        
        // Busca o tensor no servidor NVMe-oF
        let response = self.server
            .serve_tensor(&self.model_id, tensor_name)
            .map_err(|e| NodeStorError::TransferFailed(e.to_string()))?;
        
        // RDMA DMA para "VRAM local"
        let rdma_result = self.rdma
            .rdma_read(&response.data, tensor_name)
            .map_err(|e| NodeStorError::TransferFailed(e.to_string()))?;
        
        // Respeita o offset e tamanho solicitados
        let offset = request.file_offset as usize;
        let size = request.size.min(rdma_result.data.len().saturating_sub(offset));
        let data = rdma_result.data[offset..offset + size].to_vec();
        
        let duration_us = start.elapsed().as_micros() as u64;
        
        debug!(
            "NvmeOfTransport: {} bytes de '{}' em {}µs via RDMA",
            data.len(), tensor_name, duration_us
        );
        
        Ok(TransferResult::new(data, duration_us))
    }

    fn transfer_liquid(
        &self,
        request: &LiquidTransferRequest,
        callback: Box<dyn Fn(TransferResult) + Send + Sync>,
    ) -> Result<(), NodeStorError> {
        use rayon::prelude::*;
        
        let num_chunks = (request.total_size + request.chunk_size - 1) / request.chunk_size;
        
        // Busca o tensor completo uma vez (RDMA é mais eficiente em blocos grandes)
        let response = self.server
            .serve_tensor(&self.model_id, &request.tensor_name)
            .map_err(|e| NodeStorError::TransferFailed(e.to_string()))?;
        
        let data = response.data;
        
        // Distribui os chunks em paralelo via rayon (CPU-free para o usuário)
        (0..num_chunks).into_par_iter().for_each(|i| {
            let offset = i * request.chunk_size;
            let size = request.chunk_size.min(data.len().saturating_sub(offset));
            if size == 0 { return; }
            let chunk = data[offset..offset + size].to_vec();
            let result = TransferResult::new(chunk, 0);
            callback(result);
        });
        
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "NvmeOfTransport (RDMA)"
    }

    fn backend_type(&self) -> TransportBackend {
        // NvmeOf é o mais alto na hierarquia — trata como NvidiaGds em termos de prioridade
        TransportBackend::NvidiaGds
    }

    fn theoretical_max_throughput_bps(&self) -> u64 {
        // 100 GbE = 12.5 GB/s, InfiniBand 200G = 25 GB/s
        // Declaramos o máximo teórico do link
        25_000_000_000 // 25 GB/s (InfiniBand 200G)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric_server::FabricServer;

    fn make_transport(model_id: &str) -> NvmeOfTransport {
        let server = Arc::new(FabricServer::new("0.0.0.0:4420"));
        server.register_tensor(model_id, "matrix.weight", vec![0xDDu8; 4 * 1024 * 1024], 0);
        server.register_tensor(model_id, "bias.weight", vec![0xEEu8; 1024], 4 * 1024 * 1024);
        NvmeOfTransport::new(server, model_id, false)
    }

    #[test]
    fn test_nvmeof_transfer_full() {
        let transport = make_transport("gpt4-local");
        let req = TransferRequest { file_offset: 0, size: 4 * 1024 * 1024, compressed: false };
        
        let result = transport.transfer("matrix.weight", &req).unwrap();
        
        assert_eq!(result.data.len(), 4 * 1024 * 1024);
        assert_eq!(result.data[0], 0xDD);
        assert!(result.throughput_gbs() >= 0.0);
        
        println!(
            "✅ NvmeOfTransport: {} MB em {}µs → {:.2} GB/s",
            result.data.len() / (1024 * 1024),
            result.duration_us,
            result.throughput_gbs()
        );
    }

    #[test]
    fn test_nvmeof_transfer_with_offset() {
        let transport = make_transport("phi3");
        
        // Pega só 512 bytes a partir do offset 1024
        let req = TransferRequest { file_offset: 1024, size: 512, compressed: false };
        let result = transport.transfer("matrix.weight", &req).unwrap();
        
        assert_eq!(result.data.len(), 512);
        assert_eq!(result.data[0], 0xDD); // Dados ainda são 0xDD após o offset
        
        println!("✅ NvmeOfTransport com offset: 512 bytes @ offset 1024 ✓");
    }

    #[test]
    fn test_nvmeof_backend_metadata() {
        let transport = make_transport("test");
        assert_eq!(transport.backend_name(), "NvmeOfTransport (RDMA)");
        assert_eq!(transport.backend_type(), TransportBackend::NvidiaGds);
        assert_eq!(transport.theoretical_max_throughput_bps(), 25_000_000_000);
    }

    #[test]
    fn test_nvmeof_liquid_transfer() {
        let transport = make_transport("streaming-model");
        
        let req = LiquidTransferRequest {
            file_path: "matrix.weight".to_string(),
            file_offset: 0,
            tensor_name: "matrix.weight".to_string(),
            total_size: 4 * 1024 * 1024,
            chunk_size: 512 * 1024, // 512KB chunks
            compression: nodestor_core::CompressionHint::None,
            look_ahead_hint: true,
        };
        
        let chunks_received = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let chunks_clone = std::sync::Arc::clone(&chunks_received);
        
        transport.transfer_liquid(&req, Box::new(move |result| {
            chunks_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(result.data[0], 0xDD, "Chunk com dados corrompidos!");
        })).unwrap();
        
        let num_chunks = chunks_received.load(std::sync::atomic::Ordering::Relaxed);
        // 4MB / 512KB = 8 chunks
        assert_eq!(num_chunks, 8, "Deve ter 8 chunks de 512KB");
        
        println!("✅ NvmeOfTransport Liquid: {} chunks de 512KB transferidos via RDMA", num_chunks);
    }

    #[test]
    fn test_nvmeof_tensor_not_found() {
        let transport = make_transport("model");
        let req = TransferRequest { file_offset: 0, size: 100, compressed: false };
        
        let result = transport.transfer("nonexistent.weight", &req);
        assert!(result.is_err(), "Deve retornar erro para tensor inexistente");
    }
}
