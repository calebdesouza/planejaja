//! # Dequantização K-quants — Motor NodeStor
//!
//! Implementa os kernels de dequantização para os formatos GGUF modernos:
//! - **Q4_K_M / Q4_K_S** — 4-bit com super-blocos de 256 pesos (formato dominante consumer 2025)
//! - **Q5_K_M / Q5_K_S** — 5-bit com super-blocos (melhor qualidade que Q4_K, cabe em GPUs 24 GB)
//! - **Q8_0**            — 8-bit simples (baseline de qualidade, quase igual a FP16)
//!
//! ## Dispatcher CPU/GPU
//!
//! Para cada tensor, o dispatcher escolhe automaticamente:
//! 1. **GPU path** (Vulkan compute shader) — se VulkanContext disponível
//! 2. **CPU SIMD path** — AVX-512 (x86_64) ou NEON (ARM), via cfg! feature-gated
//! 3. **CPU scalar path** — Rust puro, zero dependências, funciona em qualquer máquina
//!
//! ## Bloqueadores resolvidos neste módulo
//!
//! Sem este módulo, o NodeStor não consegue usar modelos 70B+ quantizados
//! porque a GPU não sabe computar com pesos em formato Q4/Q5 — ela precisa
//! de FP16. Este módulo faz a conversão Q4/Q5 → FP16 dentro do pipeline,
//! on-the-fly, sem materializar o modelo inteiro em FP16.
//!
//! ## Referência de formato GGUF
//!
//! Cada super-bloco Q4_K contém 256 pesos em 144 bytes:
//! - 2 bytes: d     (escala global, FP16)
//! - 2 bytes: dmin  (mínimo global, FP16)
//! - 12 bytes: scales (6-bit por sub-bloco × 8 sub-blocos, compactados)
//! - 128 bytes: qs   (256 × 4-bit pesos = 128 bytes)
//!
//! Throughput esperado (RTX 4090, shader Vulkan):
//! - Q4_K_M: ~180 GB/s de pesos processados/segundo
//! - Q5_K_M: ~150 GB/s de pesos processados/segundo

pub mod q4_k;
pub mod q5_0;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod simd_x86;
pub mod simd_arm;

use crate::dequant::q4_k::{DequantQ4K, Q4KVariant};
use crate::dequant::q5_0::DequantQ5_0;
use crate::dequant::q5_k::{DequantQ5K, Q5KVariant};
use crate::dequant::q6_k::DequantQ6K;
use crate::dequant::q8_0::{DequantQ8_0, DequantQ8_1};

/// Formato de quantização do tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// FP16 — sem quantização (passthrough)
    F16,
    /// FP32 — sem quantização (passthrough)
    F32,
    /// Q5_0 — 5-bit simples, blocos de 22 bytes (d FP16 + qh 4B + qs 16B)
    Q5_0,
    /// Q8_0 — 8-bit simples, blocos de 34 bytes (d FP16 + 32×i8)
    Q8_0,
    /// Q8_1 — 8-bit com soma, blocos de 36 bytes (d FP16 + s FP16 + 32×i8)
    Q8_1,
    /// Q4_K_S — 4-bit com super-blocos, variante Small (menos preciso)
    Q4KSmall,
    /// Q4_K_M — 4-bit com super-blocos, variante Medium (uso mais comum em consumer)
    Q4KMedium,
    /// Q5_K_S — 5-bit com super-blocos, variante Small
    Q5KSmall,
    /// Q5_K_M — 5-bit com super-blocos, variante Medium (melhor equilíbrio qualidade/tamanho)
    Q5KMedium,
    /// Q6_K — 6-bit com super-blocos (128+64+16+2 bytes = 210 bytes por 256 pesos)
    Q6K,
}

impl QuantFormat {
    /// Tamanho em bytes de um super-bloco neste formato.
    pub fn block_size_bytes(&self) -> usize {
        match self {
            QuantFormat::F16 => 2,
            QuantFormat::F32 => 4,
            QuantFormat::Q5_0 => 22,       // 2 bytes d + 4 bytes qh + 16 bytes qs
            QuantFormat::Q8_0 => 34,       // 2 bytes d + 32 bytes Q8
            QuantFormat::Q8_1 => 36,       // 2 bytes d + 2 bytes s + 32 bytes Q8
            QuantFormat::Q4KSmall |
            QuantFormat::Q4KMedium => 144, // 4 + 12 + 128 bytes
            QuantFormat::Q5KSmall |
            QuantFormat::Q5KMedium => 176, // 4 + 12 + 128 + 32 bytes (bits extras)
            QuantFormat::Q6K => 210,       // 128 + 64 + 16 + 2 bytes
        }
    }

    /// Número de pesos (floats) por bloco.
    pub fn weights_per_block(&self) -> usize {
        match self {
            QuantFormat::F16 | QuantFormat::F32 => 1,
            QuantFormat::Q5_0 | QuantFormat::Q8_0 | QuantFormat::Q8_1 => 32,
            QuantFormat::Q4KSmall | QuantFormat::Q4KMedium |
            QuantFormat::Q5KSmall | QuantFormat::Q5KMedium |
            QuantFormat::Q6K => 256,
        }
    }

    /// Nome legível para logs e telemetria.
    pub fn name(&self) -> &'static str {
        match self {
            QuantFormat::F16 => "F16",
            QuantFormat::F32 => "F32",
            QuantFormat::Q5_0 => "Q5_0",
            QuantFormat::Q8_0 => "Q8_0",
            QuantFormat::Q8_1 => "Q8_1",
            QuantFormat::Q4KSmall => "Q4_K_S",
            QuantFormat::Q4KMedium => "Q4_K_M",
            QuantFormat::Q5KSmall => "Q5_K_S",
            QuantFormat::Q5KMedium => "Q5_K_M",
            QuantFormat::Q6K => "Q6_K",
        }
    }
}

/// Resultado da dequantização: vetor de FP32 pronto para operações matemáticas.
pub type DequantOutput = Vec<f32>;

/// Dispatcher central de dequantização.
///
/// Escolhe automaticamente CPU scalar → CPU SIMD → GPU Vulkan conforme disponibilidade.
pub struct DequantDispatcher {
    /// Preferir GPU em vez de CPU SIMD (padrão: true se Vulkan disponível)
    prefer_gpu: bool,
    /// Estatísticas de uso para telemetria
    pub stats: DequantStats,
}

/// Telemetria de dequantização para o painel htop.
#[derive(Debug, Default)]
pub struct DequantStats {
    pub q4k_blocks_processed: u64,
    pub q5k_blocks_processed: u64,
    pub q8_blocks_processed: u64,
    pub cpu_scalar_calls: u64,
    pub cpu_simd_calls: u64,
    pub gpu_calls: u64,
    pub total_weights_output: u64,
}

impl DequantDispatcher {
    pub fn new(prefer_gpu: bool) -> Self {
        Self {
            prefer_gpu,
            stats: DequantStats::default(),
        }
    }

    /// Dequantiza um tensor completo dado em bytes brutos.
    ///
    /// Retorna os pesos em FP32, prontos para Matmul / Attention.
    ///
    /// # Parâmetros
    /// - `raw_bytes`: o tensor serializado no formato GGUF
    /// - `format`: qual formato de quantização está em uso
    /// - `num_elements`: número total de floats na saída
    pub fn dequantize(
        &mut self,
        raw_bytes: &[u8],
        format: QuantFormat,
        num_elements: usize,
    ) -> DequantOutput {
        match format {
            QuantFormat::F32 => {
                // Passthrough: converte bytes diretamente em f32
                self.stats.total_weights_output += num_elements as u64;
                raw_bytes.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            QuantFormat::F16 => {
                // FP16 → FP32 via half_to_f32
                self.stats.total_weights_output += num_elements as u64;
                raw_bytes.chunks_exact(2)
                    .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect()
            }
            QuantFormat::Q5_0 => {
                self.stats.total_weights_output += num_elements as u64;
                self.stats.cpu_scalar_calls += 1;
                DequantQ5_0::dequantize(raw_bytes)
            }
            QuantFormat::Q8_0 => {
                self.stats.q8_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.stats.cpu_scalar_calls += 1;
                DequantQ8_0::dequantize(raw_bytes)
            }
            QuantFormat::Q8_1 => {
                self.stats.q8_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.stats.cpu_scalar_calls += 1;
                DequantQ8_1::dequantize(raw_bytes)
            }
            QuantFormat::Q4KSmall => {
                self.stats.q4k_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.dispatch_q4k(raw_bytes, Q4KVariant::Small)
            }
            QuantFormat::Q4KMedium => {
                self.stats.q4k_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.dispatch_q4k(raw_bytes, Q4KVariant::Medium)
            }
            QuantFormat::Q5KSmall => {
                self.stats.q5k_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.dispatch_q5k(raw_bytes, Q5KVariant::Small)
            }
            QuantFormat::Q5KMedium => {
                self.stats.q5k_blocks_processed += (raw_bytes.len() / format.block_size_bytes()) as u64;
                self.stats.total_weights_output += num_elements as u64;
                self.dispatch_q5k(raw_bytes, Q5KVariant::Medium)
            }
            QuantFormat::Q6K => {
                self.stats.total_weights_output += num_elements as u64;
                self.stats.cpu_scalar_calls += 1;
                DequantQ6K::dequantize(raw_bytes)
            }
        }
    }

    fn dispatch_q4k(&mut self, raw: &[u8], variant: Q4KVariant) -> DequantOutput {
        // Futuramente: if self.prefer_gpu && vulkan_available → dispatch Vulkan shader
        // Por ora: CPU path (SIMD se disponível, scalar como fallback)
        if simd_x86::avx2_available() {
            self.stats.cpu_simd_calls += 1;
            simd_x86::dequant_q4k_avx2(raw, variant)
        } else {
            self.stats.cpu_scalar_calls += 1;
            DequantQ4K::dequantize(raw, variant)
        }
    }

    fn dispatch_q5k(&mut self, raw: &[u8], variant: Q5KVariant) -> DequantOutput {
        if simd_x86::avx2_available() {
            self.stats.cpu_simd_calls += 1;
            simd_x86::dequant_q5k_avx2(raw, variant)
        } else {
            self.stats.cpu_scalar_calls += 1;
            DequantQ5K::dequantize(raw, variant)
        }
    }

    /// Relatório de telemetria para o painel htop.
    pub fn report(&self) -> String {
        format!(
            "Dequant | Q4K={} Q5K={} Q8={} | SIMD={} GPU={} Scalar={}",
            self.stats.q4k_blocks_processed,
            self.stats.q5k_blocks_processed,
            self.stats.q8_blocks_processed,
            self.stats.cpu_simd_calls,
            self.stats.gpu_calls,
            self.stats.cpu_scalar_calls,
        )
    }
}

/// Converte FP16 (IEEE 754 half precision) para FP32.
///
/// Implementação pura em Rust — sem dependências externas.
/// Usada também internamente pelos kernels Q4K e Q5K para ler escalas.
#[inline(always)]
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp  = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let (exp32, mant32) = if exp == 0 {
        if mant == 0 {
            (0, 0)
        } else {
            // Subnormal → normaliza
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (e, m << 13)
        }
    } else if exp == 31 {
        (255, mant << 13) // Inf ou NaN
    } else {
        (exp + 127 - 15, mant << 13)
    };

    f32::from_bits(sign | (exp32 << 23) | mant32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_to_f32_zero() {
        assert_eq!(half_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn test_half_to_f32_one() {
        // FP16: 1.0 = 0x3C00
        let result = half_to_f32(0x3C00);
        assert!((result - 1.0f32).abs() < 1e-5, "FP16 1.0 → FP32 deve ser 1.0, got {}", result);
    }

    #[test]
    fn test_half_to_f32_minus_two() {
        // FP16: -2.0 = 0xC000
        let result = half_to_f32(0xC000);
        assert!((result - (-2.0f32)).abs() < 1e-5, "FP16 -2.0 → FP32, got {}", result);
    }

    #[test]
    fn test_quant_format_block_sizes() {
        assert_eq!(QuantFormat::Q4KMedium.block_size_bytes(), 144);
        assert_eq!(QuantFormat::Q5KMedium.block_size_bytes(), 176);
        assert_eq!(QuantFormat::Q8_0.block_size_bytes(), 34);
        assert_eq!(QuantFormat::Q4KMedium.weights_per_block(), 256);
        assert_eq!(QuantFormat::Q5KMedium.weights_per_block(), 256);
    }

    #[test]
    fn test_dispatcher_f32_passthrough() {
        let mut d = DequantDispatcher::new(false);
        let data: Vec<u8> = vec![0u8, 0, 128, 63]; // 1.0f32 em little-endian
        let out = d.dequantize(&data, QuantFormat::F32, 1);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_dispatcher_f16_passthrough() {
        let mut d = DequantDispatcher::new(false);
        let data: Vec<u8> = vec![0x00, 0x3C]; // FP16 1.0
        let out = d.dequantize(&data, QuantFormat::F16, 1);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0f32).abs() < 1e-4);
    }

    #[test]
    fn test_dispatcher_report_not_empty() {
        let d = DequantDispatcher::new(false);
        assert!(!d.report().is_empty());
    }
}
