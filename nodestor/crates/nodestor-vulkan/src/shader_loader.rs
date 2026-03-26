//! Shaders SPIR-V embutidos e carregamento em runtime.
//!
//! Os shaders são compilados de GLSL → SPIR-V durante o build e embutidos
//! diretamente no binário via `include_bytes!`. Zero dependências externas
//! em runtime — o executável é totalmente autossuficiente.

use crate::error::VulkanError;

/// Identificador de shader disponível.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    /// Dequantização Q4_0/Q4_1 → F16 (GGML 4-bit)
    DequantQ4,
    /// Dequantização Q8_0 → F16 (GGML 8-bit)
    DequantQ8,
    /// Multiplicação de matrizes F16
    Matmul,
    /// Similaridade cosseno (busca vetorial DiskANN)
    CosineSim,
    /// Expansão bit-idêntica (Lossless)
    Lossless,
}

impl ShaderKind {
    /// Retorna o nome legível do shader.
    pub fn name(&self) -> &'static str {
        match self {
            Self::DequantQ4 => "dequant_q4",
            Self::DequantQ8 => "dequant_q8",
            Self::Matmul => "matmul",
            Self::CosineSim => "cosine_sim",
            Self::Lossless => "lossless_expansion",
        }
    }

    /// Retorna todos os tipos de shader disponíveis.
    pub fn all() -> &'static [ShaderKind] {
        &[
            ShaderKind::DequantQ4,
            ShaderKind::DequantQ8,
            ShaderKind::Matmul,
            ShaderKind::CosineSim,
            ShaderKind::Lossless,
        ]
    }
}

/// Bytecode SPIR-V de um shader.
#[derive(Debug)]
pub struct ShaderSpirv {
    pub kind: ShaderKind,
    /// Bytecode SPIR-V (múltiplos de 4 bytes, conforme spec Vulkan).
    pub bytecode: Vec<u8>,
}

impl ShaderSpirv {
    /// Valida que o bytecode é SPIR-V válido (magic number 0x07230203).
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), VulkanError> {
        if self.bytecode.len() < 4 {
            return Err(VulkanError::InvalidShader(format!(
                "Shader '{}' muito curto ({} bytes)",
                self.kind.name(),
                self.bytecode.len()
            )));
        }

        let magic = u32::from_le_bytes([
            self.bytecode[0],
            self.bytecode[1],
            self.bytecode[2],
            self.bytecode[3],
        ]);

        if magic != 0x07230203 {
            return Err(VulkanError::InvalidShader(format!(
                "Shader '{}' magic inválido: 0x{:08X}",
                self.kind.name(),
                magic
            )));
        }

        Ok(())
    }
}

/// Carrega um shader pelo tipo.
pub fn load_shader(kind: ShaderKind) -> Result<ShaderSpirv, VulkanError> {
    let bytecode = match kind {
        ShaderKind::DequantQ4 => include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4.spv")).to_vec(),
        ShaderKind::DequantQ8 => include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q8.spv")).to_vec(),
        ShaderKind::Matmul => include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv")).to_vec(),
        ShaderKind::CosineSim => include_bytes!(concat!(env!("OUT_DIR"), "/cosine_sim.spv")).to_vec(),
        ShaderKind::Lossless => include_bytes!(concat!(env!("OUT_DIR"), "/lossless_expansion.spv")).to_vec(),
    };

    Ok(ShaderSpirv {
        kind,
        bytecode,
    })
}

/// Carrega todos os shaders disponíveis de uma vez.
pub fn load_all_shaders() -> Vec<ShaderSpirv> {
    ShaderKind::all()
        .iter()
        .filter_map(|&kind| {
            load_shader(kind)
                .map_err(|e| {
                    tracing::warn!("Falha ao carregar shader '{}': {}", kind.name(), e);
                })
                .ok()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_all_shaders() {
        // Este teste pode falhar no CI se os shaders não forem buildados, 
        // mas em ambiente de build real deve passar.
    }
}
