//! # Q4_K — Dequantização de 4-bit com super-blocos (formato GGUF)
//!
//! ## Estrutura do super-bloco Q4_K (256 pesos, 144 bytes)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ SUPER-BLOCO Q4_K (144 bytes = 256 pesos)                  │
//! ├───────┬───────┬─────────────────┬────────────────────────-┤
//! │  d    │ dmin  │   scales[12]    │        qs[128]          │
//! │ 2 B   │ 2 B   │   12 bytes      │       128 bytes         │
//! │ FP16  │ FP16  │ 6-bit/sub-bloco │ 256 pesos × 4-bit      │
//! └───────┴───────┴─────────────────┴────────────────────────-┘
//! ```
//!
//! ## Fórmula de dequantização
//!
//! Para cada sub-bloco i (0..8), grupo de 32 pesos:
//! ```text
//! scale_i = d    × get_scale(scales, i)     → escala positiva
//! min_i   = dmin × get_min(scales, i)       → offset (mínimo)
//! weight_j = scale_i × q_j - min_i          → peso final FP32
//! ```
//!
//! onde q_j ∈ [0, 15] é o nibble de 4 bits do peso.
//!
//! ## Variantes
//! - **Q4_K_S** (Small): menos bits de precisão nas escalas dos sub-blocos
//! - **Q4_K_M** (Medium): escalas quantizadas com 6 bits cada → melhor qualidade

use super::half_to_f32;

/// Constantes do formato Q4_K.
const QK_K: usize = 256;         // pesos por super-bloco
const K_SCALE_SIZE: usize = 12;  // bytes para escalas e mínimos
const BLOCK_SIZE: usize = 4 + K_SCALE_SIZE + QK_K / 2; // = 144 bytes

/// Variante do formato Q4_K.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4KVariant {
    Small,
    Medium,
}

/// Kernel de dequantização Q4_K em CPU puro (Rust escalar).
///
/// Funciona em qualquer plataforma sem dependências.
/// Para performance máxima, o dispatcher usa `simd_x86::dequant_q4k_avx2`
/// quando AVX2 está disponível.
pub struct DequantQ4K;

impl DequantQ4K {
    /// Dequantiza um slice de bytes brutos Q4_K em FP32.
    ///
    /// # Pânico
    /// Panica se `raw.len()` não é múltiplo de `BLOCK_SIZE` (144).
    pub fn dequantize(raw: &[u8], _variant: Q4KVariant) -> Vec<f32> {
        assert_eq!(
            raw.len() % BLOCK_SIZE, 0,
            "Q4_K: tamanho inválido {} (esperado múltiplo de {})",
            raw.len(), BLOCK_SIZE
        );

        let num_blocks = raw.len() / BLOCK_SIZE;
        let mut output = Vec::with_capacity(num_blocks * QK_K);

        for block in raw.chunks_exact(BLOCK_SIZE) {
            Self::dequantize_block(block, &mut output);
        }

        output
    }

    /// Dequantiza um único super-bloco de 144 bytes e acumula em `out`.
    #[inline]
    fn dequantize_block(block: &[u8], out: &mut Vec<f32>) {
        // Layout: [d: FP16][dmin: FP16][scales: 12 bytes][qs: 128 bytes]
        let d    = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];   // 12 bytes de escalas
        let qs     = &block[16..144]; // 128 bytes de pesos 4-bit

        // Itera sobre 4 grupos de 64 pesos (2 sub-blocos por grupo)
        let mut is = 0usize; // índice de sub-bloco nos scales (0..8)
        let mut q_offset = 0usize;

        for _chunk in 0..4 {
            // Sub-bloco par (i=is): next 32 pesos, nibbles baixos
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            // Sub-bloco ímpar (i=is+1): next 32 pesos, nibbles altos
            let (sc1, m1b) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1b as f32;

            // 32 pesos: nibble baixo (bits 0-3)
            for l in 0..32 {
                let q = (qs[q_offset + l] & 0x0F) as f32;
                out.push(d1 * q - m1);
            }

            // 32 pesos: nibble alto (bits 4-7)
            for l in 0..32 {
                let q = (qs[q_offset + l] >> 4) as f32;
                out.push(d2 * q - m2);
            }

            q_offset += 32;
            is += 2;
        }
    }
}

/// Extrai a escala (d_scale) e o mínimo (d_min) para o sub-bloco `j`
/// a partir dos 12 bytes de escalas compactadas.
///
/// Cada escala e mínimo ocupa 6 bits, compactados em 12 bytes para 8 sub-blocos.
#[inline(always)]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = scales[j]     & 0x3F;
        let m = scales[j + 4] & 0x3F;
        (d, m)
    } else {
        // Bits extras nos bytes superiores dos primeiros 4 e do byte j+4
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constrói um super-bloco Q4_K com valores conhecidos para validar.
    fn make_test_block(d_fp16: u16, dmin_fp16: u16, scale: u8, q_nibble: u8) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_SIZE];

        // d e dmin em FP16 little-endian
        block[0] = (d_fp16 & 0xFF) as u8;
        block[1] = (d_fp16 >> 8) as u8;
        block[2] = (dmin_fp16 & 0xFF) as u8;
        block[3] = (dmin_fp16 >> 8) as u8;

        // Escalas: preenche os 12 bytes com scale para que
        // get_scale_min_k4 retorne (scale, 0) para os sub-blocos 0..3.
        // bytes 4..8: sc para sub-blocos 0..3 (& 0x3F)
        // bytes 8..12: min para sub-blocos 0..3 (& 0x3F) = 0
        for i in 0..4 {
            block[4 + i] = scale & 0x3F; // d_scale
            block[8 + i] = 0x00;          // d_min = 0
        }

        // Pesos: todos o mesmo nibble (q_nibble | q_nibble << 4)
        let byte_val = (q_nibble & 0xF) | ((q_nibble & 0xF) << 4);
        for i in 16..144 {
            block[i] = byte_val;
        }

        block
    }

    #[test]
    fn test_q4k_zero_weights() {
        // d=1.0 (0x3C00 FP16), dmin=0.0, scale=1, q=0 → todos os pesos = 1.0 * 1 * 0 - 0 = 0.0
        let block = make_test_block(0x3C00, 0x0000, 1, 0);
        let out = DequantQ4K::dequantize(&block, Q4KVariant::Medium);

        assert_eq!(out.len(), 256);
        for w in &out {
            assert!(w.abs() < 1e-5, "Esperado 0.0, got {}", w);
        }
    }

    #[test]
    fn test_q4k_uniform_weights() {
        // d=1.0 (0x3C00), dmin=0.0, scale=4, q=7
        // Sub-blocos 0..3: sc = block[4+i] & 0x3F = 4, min = 0
        // Peso esperado para nibble=7: 1.0 * 4 * 7 - 0 = 28.0
        // Sub-blocos 4..7: usam lógica de extração diferente — podem ter scale diferente
        // Aqui apenas verificamos que a saída é finita e que os primeiros 128 pesos
        // (sub-blocos 0..3) têm o valor correto.
        let block = make_test_block(0x3C00, 0x0000, 4, 7);
        let out = DequantQ4K::dequantize(&block, Q4KVariant::Medium);

        assert_eq!(out.len(), 256);
        // Os primeiros 128 pesos (sub-blocos 0..3) devem ser 28.0
        for w in &out[..128] {
            assert!((w - 28.0f32).abs() < 1e-3, "Primo sub-bloco: Esperado 28.0, got {}", w);
        }
        // Todos os 256 devem ser finitos
        for w in &out {
            assert!(w.is_finite(), "Peso não finito: {}", w);
        }
    }

    #[test]
    fn test_q4k_block_size_check() {
        let block = make_test_block(0x3C00, 0x0000, 1, 5);
        assert_eq!(block.len(), BLOCK_SIZE);
        assert_eq!(block.len(), 144);
    }

    #[test]
    fn test_q4k_multiple_blocks() {
        let block = make_test_block(0x3C00, 0x0000, 2, 3);
        let two_blocks: Vec<u8> = block.iter().chain(block.iter()).cloned().collect();

        let out = DequantQ4K::dequantize(&two_blocks, Q4KVariant::Medium);
        assert_eq!(out.len(), 512); // 2 × 256 pesos
    }

    #[test]
    fn test_get_scale_min_k4_first_four() {
        // Sub-blocos 0..3: extraídos diretamente de scales[j] e scales[j+4]
        let scales = [0x15u8, 0x20, 0x3F, 0x0A, 0x01, 0x02, 0x08, 0x10, 0, 0, 0, 0];
        let (d, m) = get_scale_min_k4(0, &scales);
        assert_eq!(d, 0x15 & 0x3F);
        assert_eq!(m, 0x01 & 0x3F);
    }

    #[test]
    fn test_q4k_output_range() {
        // Qualquer saída deve estar em um range razoável para escala FP16
        let block = make_test_block(0x3C00, 0x3C00, 15, 15);
        let out = DequantQ4K::dequantize(&block, Q4KVariant::Medium);
        assert_eq!(out.len(), 256);
        // Todos os valores devem ser finitos
        for w in &out {
            assert!(w.is_finite(), "Peso não finito: {}", w);
        }
    }
}
