//! Erros específicos do módulo Vulkan.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Falha ao criar instância Vulkan: {0}")]
    InstanceCreation(String),

    #[error("Nenhum dispositivo físico compatível encontrado")]
    NoCompatibleDevice,

    #[error("Falha ao criar dispositivo lógico: {0}")]
    DeviceCreation(String),

    #[error("Falha ao alocar buffer de GPU: {0}")]
    AllocationFailed(String),

    #[error("Falha ao executar compute shader: {0}")]
    DispatchFailed(String),

    #[error("Shader SPIR-V inválido: {0}")]
    InvalidShader(String),

    #[error("Extensão Vulkan não disponível: {0}")]
    MissingExtension(String),

    #[error("Operação não suportada no hardware atual: {0}")]
    Unsupported(String),
}

impl From<VulkanError> for nodestor_core::NodeStorError {
    fn from(e: VulkanError) -> Self {
        nodestor_core::NodeStorError::VulkanError(e.to_string())
    }
}
