use thiserror::Error;

/// Erros centrais do NodeStor.
#[derive(Error, Debug)]
pub enum NodeStorError {
    #[error("Hardware não suportado: {0}")]
    UnsupportedHardware(String),

    #[error("Formato de modelo inválido: {0}")]
    InvalidModelFormat(String),

    #[error("Falha na transferência de dados: {0}")]
    TransferFailed(String),

    #[error("Vulkan error: {0}")]
    VulkanError(String),

    #[error("GDeflate error: {0}")]
    GDeflateError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Modelo não encontrado: {0}")]
    ModelNotFound(String),

    #[error("VRAM insuficiente: necessário {needed} bytes, disponível {available} bytes")]
    InsufficientVram { needed: u64, available: u64 },

    #[error("Operação não suportada no hardware atual: {0}")]
    NotSupported(String),

    #[error("Erro de configuração: {0}")]
    ConfigError(String),

    #[error("Timeout após {ms}ms")]
    Timeout { ms: u64 },
}
