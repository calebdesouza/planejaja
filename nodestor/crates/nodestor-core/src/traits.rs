use crate::{error::NodeStorError, types::*};

/// Trait central para transporte de dados SSD → GPU.
///
/// Cada backend (io_uring, DirectStorage, GDS, fallback) implementa este trait.
/// A factory em `nodestor-transport` seleciona automaticamente o melhor backend.
pub trait DataTransport: Send + Sync {
    /// Transfere um bloco de dados do arquivo aberto para um buffer in-memory.
    /// Em backends de alta performance, a transferência é feita diretamente para a GPU.
    fn transfer(
        &self,
        path: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError>;

    /// Envia dados brutos do host (RAM) para um buffer na VRAM (SSD-Offloading).
    /// Usado para o Zero-Loss Context Paging.
    fn write_to_vram_buffer(&self, _data: &[u8]) -> Result<(), NodeStorError> {
        Ok(())
    }

    /// Enfileira múltiplas transferências (Modo Metralhadora).
    /// Implementação default: executa sequencialmente.
    fn transfer_batch(
        &self,
        path: &str,
        requests: &[TransferRequest],
    ) -> Result<Vec<TransferResult>, NodeStorError> {
        requests.iter().map(|r| self.transfer(path, r)).collect()
    }

    /// Nome do backend para logging e diagnóstico.
    fn backend_name(&self) -> &'static str;

    /// Tipo de transporte implementado.
    fn backend_type(&self) -> TransportBackend;

    /// Throughput máximo teórico em bytes/segundo.
    fn theoretical_max_throughput_bps(&self) -> u64;

    /// Inicia uma transferência líquida (micro-fatias em paralelo).
    fn transfer_liquid(
        &self,
        request: &LiquidTransferRequest,
        callback: Box<dyn Fn(TransferResult) + Send + Sync>,
    ) -> Result<(), NodeStorError>;
}

/// Trait para parsers de formatos de modelo.
pub trait ModelParser: Send + Sync {
    /// Analisa o arquivo e retorna os metadados completos do modelo.
    ///
    /// IMPORTANTE: Esta operação NUNCA deve carregar os tensores em memória.
    /// Apenas lê o header e constrói o índice de tensores com seus offsets.
    fn parse(&self, path: &str) -> Result<ModelMetadata, NodeStorError>;

    /// Retorna true se este parser suporta o arquivo dado.
    fn can_parse(&self, path: &str) -> bool;

    /// Nome do formato.
    fn format_name(&self) -> &'static str;
}
