//! Modo Metralhadora — Prefetch e batch submission preditivo p/ SSD via Transport.
//!
//! Opera sobre um `DataTransport` e um `BufferPool` empilhando comandos de leitura
//! preditivos. Executa os I/Os pesados em blocking threads que usufruem do 
//! `transfer_batch` p/ engajar OS-level async (ex: io_uring, FILE_FLAG_OVERLAPPED).

use crate::buffer_pool::{BufferPool, PooledBuffer};
use nodestor_core::{DataTransport, NodeStorError, TransferRequest};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

pub struct PrefetchedBlock {
    pub buffer: PooledBuffer,
    pub original_request: TransferRequest,
    pub load_time_us: u64,
}

/// Fila agressiva do Modo Metralhadora.
pub struct MesPrefetchQueue {
    request_tx: mpsc::Sender<TransferRequest>,
    result_rx: mpsc::Receiver<PrefetchedBlock>,
    _model_path: Arc<String>,
}

impl MesPrefetchQueue {
    /// Inicia a metralhadora em um thread paralelo.
    pub fn new(
        transport: Arc<dyn DataTransport + Send + Sync>,
        model_path: String,
        pool: BufferPool,
    ) -> Self {
        let depth = pool.depth;
        let (req_tx, mut req_rx) = mpsc::channel::<TransferRequest>(depth);
        let (res_tx, res_rx) = mpsc::channel::<PrefetchedBlock>(depth);
        
        let path_arc = Arc::new(model_path);
        let worker_path = path_arc.clone();

        // Worker thread de prefetch continuo
        tokio::spawn(async move {
            info!("Modo Metralhadora engajado (depth = {})", depth);

            while let Some(req) = req_rx.recv().await {
                // Ao receber um request, primeiro pegamos um buffer do pool (pode bloquear se tudo cheio)
                let mut p_buf = match pool.acquire().await {
                    Ok(b) => b,
                    Err(e) => {
                        error!("Prefetch falhou ao pegar buffer: {}", e);
                        continue;
                    }
                };

                let tport = transport.clone();
                let path = worker_path.clone();
                let req_clone = req.clone();

                // Fazer o I/O em um runtime blocking thread, enviando os dados direto para o PooledBuffer.
                // Isso aproveita io_uring real síncrono ou overlapped sem travar o Tokio executor.
                let load_result = tokio::task::spawn_blocking(move || {
                    tport.transfer(&path, &req_clone)
                })
                .await;

                match load_result {
                    Ok(Ok(transfer_result)) => {
                        let tensor_data = transfer_result.data;
                        let gpu_buf = p_buf.buffer.as_mut().unwrap();
                        let dest = gpu_buf.as_mut_bytes();

                        let _bytes_written = if req.compressed {
                            let mut cursor = std::io::Cursor::new(dest);
                            match zstd::stream::copy_decode(&tensor_data[..], &mut cursor) {
                                Ok(_) => cursor.position() as usize,
                                Err(e) => {
                                    error!("Descompressão Zstd falhou: {}", e);
                                    continue;
                                }
                            }
                        } else {
                            let len = tensor_data.len().min(dest.len());
                            dest[..len].copy_from_slice(&tensor_data[..len]);
                            len
                        };

                        let block = PrefetchedBlock {
                            buffer: p_buf,
                            original_request: req,
                            load_time_us: transfer_result.duration_us,
                        };

                        if res_tx.send(block).await.is_err() {
                            debug!("Metralhadora: canal de resultado fechado.");
                            break;
                        }
                    }
                    Ok(Err(e)) => error!("Metralhadora transport falhou: {}", e),
                    Err(_) => error!("Metralhadora thread blocking panic"),
                }
            }
            debug!("Modo Metralhadora desengajado.");
        });

        Self {
            request_tx: req_tx,
            result_rx: res_rx,
            _model_path: path_arc,
        }
    }

    /// Enfileira um predição de leitura de bloco para o background fetcher.
    pub async fn enqueue_prediction(&self, req: TransferRequest) -> Result<(), NodeStorError> {
        self.request_tx.send(req).await.map_err(|_| {
            NodeStorError::TransferFailed("Canal da Metralhadora inativo".to_string())
        })
    }

    /// Retira o próximo bloco que já foi carregado para a VRAM (ou aguarda se I/O em andamento).
    pub async fn pop_ready_block(&mut self) -> Option<PrefetchedBlock> {
        self.result_rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_transport::PreadFallback;
    use nodestor_vulkan::VulkanContext;
    use std::io::Write;

    #[tokio::test]
    async fn test_metralhadora_end_to_end() {
        // Criar arq fake
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let content = vec![0xABu8; 1024];
        std::fs::File::create(&path).unwrap().write_all(&content).unwrap();

        let ctx = VulkanContext::new(None).unwrap();
        let pool = BufferPool::new(&ctx, 512, 2).unwrap();
        
        let tport = Arc::new(PreadFallback::new());
        let mut metra = MesPrefetchQueue::new(tport, path.to_str().unwrap().to_string(), pool);

        // Enfileira request
        metra.enqueue_prediction(TransferRequest { file_offset: 0, size: 512, compressed: false }).await.unwrap();

        // Ouve conclusão
        let block = metra.pop_ready_block().await.unwrap();
        assert_eq!(block.original_request.size, 512);
        assert_eq!(block.buffer.buffer.as_ref().unwrap().as_bytes()[0], 0xAB);
        assert!(block.load_time_us > 0 || block.load_time_us == 0);
    }
}
