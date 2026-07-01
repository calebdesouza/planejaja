//! Q8_0 e Q8_1 — Dequantização 8-bit (baseline de qualidade).
//!
//! **Q8_0**: blocos de 34 bytes = 2 (escala FP16) + 32 (int8 pesos)
//! **Q8_1**: blocos de 36 bytes = 2 (d FP16) + 2 (s FP16, soma) + 32 (int8 pesos)
//! Perda de qualidade vs FP16: <0.05% — praticamente imperceptível.

use super::half_to_f32;

const Q8_0_BLOCK: usize = 34; // 2 bytes d + 32 bytes qs
const Q8_1_BLOCK: usize = 36; // 2 bytes d + 2 bytes s + 32 bytes qs
const WEIGHTS_PER_BLOCK: usize = 32;

pub struct DequantQ8_0;

impl DequantQ8_0 {
    pub fn dequantize(raw: &[u8]) -> Vec<f32> {
        assert_eq!(raw.len() % Q8_0_BLOCK, 0, "Q8_0: tamanho inválido {} (deve ser múltiplo de {})", raw.len(), Q8_0_BLOCK);

        let mut output = Vec::with_capacity((raw.len() / Q8_0_BLOCK) * WEIGHTS_PER_BLOCK);

        for block in raw.chunks_exact(Q8_0_BLOCK) {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            for &q in &block[2..34] {
                output.push(d * (q as i8) as f32);
            }
        }

        output
    }
}

/// Q8_1: escala d FP16 + soma s FP16 (ignorada no dequant) + 32 pesos int8.
pub struct DequantQ8_1;

impl DequantQ8_1 {
    pub fn dequantize(raw: &[u8]) -> Vec<f32> {
        assert_eq!(raw.len() % Q8_1_BLOCK, 0, "Q8_1: tamanho inválido {} (deve ser múltiplo de {})", raw.len(), Q8_1_BLOCK);

        let mut output = Vec::with_capacity((raw.len() / Q8_1_BLOCK) * WEIGHTS_PER_BLOCK);

        for block in raw.chunks_exact(Q8_1_BLOCK) {
            // bytes 0-1: d (escala FP16)
            // bytes 2-3: s (soma × d, usada apenas em GEMV — ignoramos aqui)
            // bytes 4-35: 32 pesos int8
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            for &q in &block[4..36] {
                output.push(d * (q as i8) as f32);
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_zero() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0 FP16
        // todos os pesos = 0 → saída = 0.0
        let out = DequantQ8_0::dequantize(&block);
        assert_eq!(out.len(), 32);
        for w in &out { assert!(w.abs() < 1e-6); }
    }

    #[test]
    fn test_q8_uniform() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        for i in 2..34 { block[i] = 5; } // q = 5 → peso = 5.0
        let out = DequantQ8_0::dequantize(&block);
        for w in &out { assert!((w - 5.0).abs() < 1e-5); }
    }

    #[test]
    fn test_q8_signed() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0xFF; // -1 em i8 (0xFF = 255 unsigned = -1 signed)
        let out = DequantQ8_0::dequantize(&block);
        assert!((out[0] - (-1.0f32)).abs() < 1e-5, "Expected -1.0, got {}", out[0]);
    }
}
