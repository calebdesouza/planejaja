//! Double/Triple buffering pool para GpuBuffers.
//!
//! Gerencia alocações reutilizáveis de memória na GPU para minimizar o overhead
//! de alocação durante o streaming. É seguro para concorrência via Tokio.

use nodestor_core::NodeStorError;
use nodestor_vulkan::{GpuBuffer, VulkanContext};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Uma guarda que devolve o buffer ao pool automaticamente quando sai de escopo.
pub struct PooledBuffer {
    /// O buffer real (Option para permitir a extração temporária no drop)
    pub buffer: Option<GpuBuffer>,
    /// Canal de retorno para o pool
    return_tx: mpsc::Sender<GpuBuffer>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            // Tenta devolver ao pool de forma síncrona/não-bloqueante
            let _ = self.return_tx.try_send(buf);
        }
    }
}

/// Pool de buffers VRAM pré-alocados.
#[derive(Clone)]
pub struct BufferPool {
    pool_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<GpuBuffer>>>,
    return_tx: mpsc::Sender<GpuBuffer>,
    pub buffer_size: usize,
    pub depth: usize,
}

impl BufferPool {
    /// Cria um novo pool alocando `depth` buffers de tamanho `buffer_size` no VulkanContext.
    pub fn new(
        ctx: &VulkanContext,
        buffer_size: usize,
        depth: usize,
    ) -> Result<Self, NodeStorError> {
        let (tx, rx) = mpsc::channel(depth);

        for _ in 0..depth {
            let buf = ctx.alloc_gpu_buffer(buffer_size)?;
            // Canal preenchido em bloco (sem falhas pois capa = depth)
            tx.try_send(buf).map_err(|_| {
                NodeStorError::ConfigError("Falha ao povoar BufferPool".to_string())
            })?;
        }

        Ok(Self {
            pool_rx: Arc::new(tokio::sync::Mutex::new(rx)),
            return_tx: tx,
            buffer_size,
            depth,
        })
    }

    /// Adquire um buffer livre do pool. 
    /// Se todos estiverem em uso, suspende a task (await) até que um seja liberado.
    pub async fn acquire(&self) -> Result<PooledBuffer, NodeStorError> {
        let mut rx = self.pool_rx.lock().await;
        let buf = rx.recv().await.ok_or_else(|| {
            NodeStorError::ConfigError("BufferPool foi fechado inesperadamente".to_string())
        })?;

        Ok(PooledBuffer {
            buffer: Some(buf),
            return_tx: self.return_tx.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pool_acquire_and_release() {
        let ctx = VulkanContext::new(None).unwrap();
        let pool = BufferPool::new(&ctx, 1024, 2).unwrap();

        {
            let buf1 = pool.acquire().await.unwrap();
            assert_eq!(buf1.buffer.as_ref().unwrap().size, 1024);

            let _buf2 = pool.acquire().await.unwrap();
            // Aqui o pool está vazio...
        }
        // Mas ambos saíram de escopo e realizaram o drop.
        
        // Devemos conseguir adquirir novamente sem bloquear (a timeout garante que não trave para sempre)
        let _buf1_again = tokio::time::timeout(std::time::Duration::from_millis(100), pool.acquire())
            .await
            .expect("Timeout ao adquirir buffer após devolução")
            .unwrap();
    }
}
