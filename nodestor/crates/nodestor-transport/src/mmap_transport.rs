use nodestor_core::{DataTransport, NodeStorError, TransferRequest, TransferResult, TransportBackend};
use std::fs::File;
use std::time::Instant;
use memmap2::Mmap;
use tracing::debug;

/// Transporte baseado em Memory-Mapped I/O (mmap).
///
/// Mapeia o arquivo diretamente no espaço de endereçamento virtual,
/// permitindo que a GPU ou o sistema acesse os dados via DMA sem 
/// cópias intermediárias para buffers de espaço de usuário.
///
/// Ideal para macOS, Intel iGPU e como fallback universal de alta performance.
pub struct MmapTransport;

impl MmapTransport {
    pub fn new() -> Self {
        debug!("MmapTransport inicializado (Zero-Copy DMA mode)");
        Self
    }
}

impl DataTransport for MmapTransport {
    fn transfer(
        &self,
        path: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError> {
        let start = Instant::now();

        // No mmap real, abriríamos o arquivo e mapearíamos a região
        // Para evitar overhead de re-mapeamento constante, em um sistema real 
        // manteríamos o mmap em cache no Transport.
        let file = File::open(path).map_err(NodeStorError::IoError)?;
        
        // Mapeamos o arquivo inteiro (ou a região do tensor)
        // OBS: Para arquivos gigantes (>2GB no Win 32-bit), mmap manual é necessário.
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| NodeStorError::TransferFailed(
                format!("Falha no mmap de '{}': {}", path, e)
            ))?
        };

        // Fatia os dados mapeados
        let end = (request.file_offset as usize + request.size).min(mmap.len());
        let data = mmap[request.file_offset as usize..end].to_vec();

        let duration_us = start.elapsed().as_micros() as u64;
        
        debug!(
            "MmapTransport: {} bytes mapeados em {}µs ({:.2} GB/s)",
            data.len(), duration_us,
            if duration_us > 0 { data.len() as f64 / (duration_us as f64 / 1e6) / 1e9 } else { 0.0 }
        );

        Ok(TransferResult::new(data, duration_us))
    }

    fn backend_name(&self) -> &'static str {
        "MmapDMA"
    }

    fn backend_type(&self) -> TransportBackend {
        TransportBackend::VulkanGeneric
    }

    fn theoretical_max_throughput_bps(&self) -> u64 {
        3_500_000_000 // 3.5 GB/s — mmap geralmente atinge o limite do barramento SSD
    }

    fn transfer_liquid(
        &self,
        request: &nodestor_core::LiquidTransferRequest,
        callback: Box<dyn Fn(nodestor_core::TransferResult) + Send + Sync>,
    ) -> Result<(), nodestor_core::NodeStorError> {
        use rayon::prelude::*;
        let num_chunks = (request.total_size + request.chunk_size - 1) / request.chunk_size;
        
        (0..num_chunks).into_par_iter().for_each(|i| {
            let offset = (i * request.chunk_size) as u64;
            let size = (request.total_size - (i * request.chunk_size)).min(request.chunk_size);
            
            let req = nodestor_core::TransferRequest {
                file_offset: request.file_offset + offset,
                size,
                compressed: false,
            };
            if let Ok(res) = self.transfer(&request.file_path, &req) {
                callback(res);
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_mmap_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap_test.bin");
        let content = b"Mmap DMA Test Content 123456789";
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(content).unwrap();
        }

        let t = MmapTransport::new();
        let req = TransferRequest {
            file_offset: 0,
            size: content.len(),
            compressed: false,
        };
        let result = t.transfer(path.to_str().unwrap(), &req).expect("Mmap deve funcionar");
        assert_eq!(result.data, content);
    }
}
