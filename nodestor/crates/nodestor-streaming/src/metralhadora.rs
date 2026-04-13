//! Modo Metralhadora — Prefetch e batch submission preditivo p/ SSD via Transport.
//!
//! Opera sobre um `DataTransport` e um `BufferPool` empilhando comandos de leitura
//! preditivos. Executa os I/Os pesados em blocking threads que usufruem do 
//! `transfer_batch` p/ engajar OS-level async (ex: io_uring, FILE_FLAG_OVERLAPPED)
//! com submissões NVMe paralelas para ataques multi-canal (Parallel Flash Sharding).

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

/// Fila agressiva do Modo Metralhadora (agora suporta Submissões em Lote).
pub struct MesPrefetchQueue {
    request_tx: mpsc::Sender<Vec<TransferRequest>>,
    result_rx: mpsc::Receiver<PrefetchedBlock>,
    _model_path: Arc<String>,
}

impl MesPrefetchQueue {
    /// Inicia a metralhadora em um thread paralelo, pronta para rajadas de I/O.
    pub fn new(
        transport: Arc<dyn DataTransport + Send + Sync>,
        model_path: String,
        pool: BufferPool,
    ) -> Self {
        // Agora aguardamos Vetores inteiros (rajadas/batches) em vez de requests individuais
        let depth = pool.depth;
        let (req_tx, mut req_rx) = mpsc::channel::<Vec<TransferRequest>>(depth);
        let (res_tx, res_rx) = mpsc::channel::<PrefetchedBlock>(depth * 2); 
        
        let path_arc = Arc::new(model_path);
        let worker_path = path_arc.clone();

        // Worker thread de prefetch continuo e paralelo
        tokio::spawn(async move {
            info!("Modo Metralhadora Multi-Direcional engajado (depth = {})", depth);

            while let Some(req_batch) = req_rx.recv().await {
                
                // Em um batch agressivo, o primeiro passo é agarrar os buffers suficientes na VRAM
                let mut p_bufs = Vec::with_capacity(req_batch.len());
                for _ in 0..req_batch.len() {
                    match pool.acquire().await {
                        Ok(b) => p_bufs.push(b),
                        Err(e) => {
                            error!("Prefetch Burst secou buffers: {}", e);
                            break; // Pega o que conseguir
                        }
                    }
                }

                if p_bufs.is_empty() {
                    continue; // Sem buffers, descartamos o batch atual
                }

                // Ajusta as requisições para a quantidade exata de buffers reais pegos
                let num_valid = p_bufs.len();
                let valid_reqs = req_batch[0..num_valid].to_vec();

                let tport = transport.clone();
                let path = worker_path.clone();

                // Fazer o I/O da batelada inteira de uma vez chamando `transfer_batch`. 
                // Isso força o O.S e o kernel NVMe a dividir entre múltiplos canais paralelos (Parallel Flash).
                let load_results = tokio::task::spawn_blocking(move || {
                    tport.transfer_batch(&path, &valid_reqs)
                })
                .await;

                match load_results {
                    Ok(Ok(transfer_results)) => {
                        // Repassa os resultados do batch iterativamente para o channel de retorno single-element
                        for (i, transfer_result) in transfer_results.into_iter().enumerate() {
                            let tensor_data = transfer_result.data;
                            let original_req = req_batch[i].clone();
                            let mut p_buf = p_bufs.remove(0); // Puxa na ordem do O(1)
                            
                            let gpu_buf = p_buf.buffer.as_mut().unwrap();
                            let dest = gpu_buf.as_mut_bytes();

                            let _bytes_written = if original_req.compressed {
                                let mut cursor = std::io::Cursor::new(dest);
                                match zstd::stream::copy_decode(&tensor_data[..], &mut cursor) {
                                    Ok(_) => cursor.position() as usize,
                                    Err(e) => {
                                        error!("Descompressão Zstd falhou (burst): {}", e);
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
                                original_request: original_req,
                                load_time_us: transfer_result.duration_us,
                            };

                            if res_tx.send(block).await.is_err() {
                                debug!("Metralhadora: canal de resultado fechado durante rajada.");
                                break;
                            }
                        }
                    }
                    Ok(Err(e)) => error!("Metralhadora burst transport falhou: {}", e),
                    Err(_) => error!("Metralhadora burst thread blocking panic"),
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

    /// Enfileira uma bateria completa de predições. Modo Burst O(1).
    pub async fn enqueue_burst(&self, reqs: Vec<TransferRequest>) -> Result<(), NodeStorError> {
        self.request_tx.send(reqs).await.map_err(|_| {
            NodeStorError::TransferFailed("Canal da Metralhadora inativo".to_string())
        })
    }

    /// Modificação para prefetch linear para testes e fallback.
    pub async fn enqueue_prediction(&self, req: TransferRequest) -> Result<(), NodeStorError> {
        self.enqueue_burst(vec![req]).await
    }

    /// Retira o próximo bloco iterativamente independentemente de quão grande foi o burst.
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
    #[ignore = "Requer VulkanContext real (GPU) — executar em CI com GPU"]
    async fn test_metralhadora_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let content = vec![0xABu8; 1024];
        std::fs::File::create(&path).unwrap().write_all(&content).unwrap();

        let ctx = VulkanContext::new(None).unwrap();
        let pool = BufferPool::new(&ctx, 512, 4).unwrap();
        
        let tport = Arc::new(PreadFallback::new());
        let mut metra = MesPrefetchQueue::new(tport, path.to_str().unwrap().to_string(), pool);

        // Envío em batch (Burst)
        let batch = vec![
            TransferRequest { file_offset: 0, size: 512, compressed: false },
            TransferRequest { file_offset: 512, size: 512, compressed: false },
        ];
        
        metra.enqueue_burst(batch).await.unwrap();

        // Recolher itens da batelada individualmente
        let block1 = metra.pop_ready_block().await.unwrap();
        let block2 = metra.pop_ready_block().await.unwrap();
        
        assert_eq!(block1.original_request.file_offset, 0);
        assert_eq!(block2.original_request.file_offset, 512);
        assert_eq!(block1.buffer.buffer.as_ref().unwrap().as_bytes()[0], 0xAB);
    }
}
