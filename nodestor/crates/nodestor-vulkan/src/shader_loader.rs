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
}

impl ShaderKind {
    /// Retorna o nome legível do shader.
    pub fn name(&self) -> &'static str {
        match self {
            Self::DequantQ4 => "dequant_q4",
            Self::DequantQ8 => "dequant_q8",
            Self::Matmul => "matmul",
            Self::CosineSim => "cosine_sim",
        }
    }

    /// Retorna todos os tipos de shader disponíveis.
    pub fn all() -> &'static [ShaderKind] {
        &[
            ShaderKind::DequantQ4,
            ShaderKind::DequantQ8,
            ShaderKind::Matmul,
            ShaderKind::CosineSim,
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
///
/// Em produção: os shaders são pré-compilados via `build.rs` com `shaderc`
/// e embutidos via `include_bytes!`.
///
/// Atualmente retorna stubs funcionais para compilação. O build.rs
/// compilará os .comp reais quando o shaderc estiver configurado.
pub fn load_shader(kind: ShaderKind) -> Result<ShaderSpirv, VulkanError> {
    // SPIRV stub mínimo válido para compilação e testes unitários.
    // O build.rs substituirá isso pelos shaders reais compilados.
    let stub_spirv = create_stub_spirv();

    Ok(ShaderSpirv {
        kind,
        bytecode: stub_spirv,
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

/// Cria um módulo SPIR-V stub mínimo e válido.
///
/// Contém apenas o header SPIR-V necessário para o magic number check.
/// Usado durante compilação e testes sem GPU real.
fn create_stub_spirv() -> Vec<u8> {
    // SPIR-V Header: magic, version (1.0), generator, bound, schema
    let words: [u32; 5] = [
        0x07230203, // Magic number
        0x00010000, // Version 1.0
        0x000D000A, // Generator (NodeStor stub)
        0x00000001, // Bound (1 ID)
        0x00000000, // Reserved schema
    ];

    words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_spirv_magic() {
        let spirv = create_stub_spirv();
        let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
        assert_eq!(magic, 0x07230203, "Magic number SPIR-V deve ser 0x07230203");
    }

    #[test]
    fn test_load_all_shaders() {
        let shaders = load_all_shaders();
        assert_eq!(shaders.len(), 4, "Devem existir 4 shaders");
    }

    #[test]
    fn test_shader_validation() {
        let shader = load_shader(ShaderKind::Matmul).unwrap();
        assert!(shader.validate().is_ok(), "Stub SPIR-V deve ser válido");
    }

    #[test]
    fn test_invalid_shader_rejected() {
        let invalid = ShaderSpirv {
            kind: ShaderKind::Matmul,
            bytecode: vec![0x00, 0x00, 0x00, 0x00],
        };
        assert!(invalid.validate().is_err(), "SPIR-V inválido deve falhar");
    }
}
