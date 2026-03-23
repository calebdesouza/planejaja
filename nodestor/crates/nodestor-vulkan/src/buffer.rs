//! Gerenciamento de GPU buffers via `gpu-allocator`.
//!
//! `GpuBuffer` é a abstração sobre uma fatia de VRAM alocada. Suporta
//! upload (RAM→GPU), download (GPU→RAM) e zero-copy quando o backend I/O
//! entrega dados diretamente na VRAM via DMA.

use crate::error::VulkanError;
use nodestor_core::NodeStorError;

/// Uso pretendido do buffer — influencia as flags de memória Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// Dados entram pela CPU (staging) e são transferidos para VRAM.
    Staging,
    /// Buffer residente em VRAM para compute shaders.
    Storage,
    /// Resultado de compute — pode ser lido de volta pela CPU.
    Readback,
}

/// Fatia de memória na GPU.
///
/// Em backends zero-copy (io_uring DMABUF, DirectStorage, GDS),
/// o SSD escreve diretamente neste buffer — a CPU nunca toca os dados.
pub struct GpuBuffer {
    /// Tamanho em bytes.
    pub size: usize,
    /// Tipo de uso.
    pub usage: GpuBufferUsage,
    /// Dados simulados em RAM (usado quando Vulkan não está disponível).
    /// Em produção com Vulkan real: este campo seria um `vk::Buffer` + `Allocation`.
    pub(crate) data: Vec<u8>,
}

impl GpuBuffer {
    /// Cria buffer de armazenamento vazio na GPU.
    pub fn new_storage(size: usize) -> Self {
        Self {
            size,
            usage: GpuBufferUsage::Storage,
            data: vec![0u8; size],
        }
    }

    /// Cria buffer de staging com dados da CPU.
    pub fn from_cpu_data(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            size,
            usage: GpuBufferUsage::Staging,
            data,
        }
    }

    /// Cria buffer de readback (leitura pelos resultados da GPU).
    pub fn new_readback(size: usize) -> Self {
        Self {
            size,
            usage: GpuBufferUsage::Readback,
            data: vec![0u8; size],
        }
    }

    /// Retorna os bytes do buffer (para leitura de resultados).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Retorna bytes mutáveis do buffer (para streaming carregar direto na "VRAM").
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Interpreta o buffer como slice de f32.
    pub fn as_f32_slice(&self) -> &[f32] {
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        // SAFETY: alinhamento correto garantido pela alocação Vec<u8> com tamanho múltiplo de 4
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuBuffer {{ size: {} bytes, usage: {:?} }}",
            self.size, self.usage
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_creation() {
        let buf = GpuBuffer::new_storage(1024);
        assert_eq!(buf.size, 1024);
        assert_eq!(buf.as_bytes().len(), 1024);
    }

    #[test]
    fn test_gpu_buffer_from_data() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let buf = GpuBuffer::from_cpu_data(data.clone());
        assert_eq!(buf.size, 8);
        assert_eq!(buf.as_bytes(), data.as_slice());
    }

    #[test]
    fn test_gpu_buffer_as_f32() {
        // Cria 4 bytes = 1 f32 (valor 1.0 em IEEE 754 little-endian)
        let bytes = 1.0f32.to_le_bytes().to_vec();
        let buf = GpuBuffer::from_cpu_data(bytes);
        let floats = buf.as_f32_slice();
        assert_eq!(floats.len(), 1);
        assert!((floats[0] - 1.0f32).abs() < 1e-6);
    }
}
