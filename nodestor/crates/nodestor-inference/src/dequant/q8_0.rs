//! Q8_0 — Dequantização 8-bit simples (baseline de qualidade).
//!
//! Formato: blocos de 32 pesos × 1 byte = 32 bytes + 2 bytes de escala (FP16) = 34 bytes.
//! Perda de qualidade vs FP16: <0.05% — praticamente imperceptível.

use super::half_to_f32;

const BLOCK_SIZE: usize = 34; // 2 (escala FP16) + 32 (int8)
const WEIGHTS_PER_BLOCK: usize = 32;

pub struct DequantQ8_0;

impl DequantQ8_0 {
    pub fn dequantize(raw: &[u8]) -> Vec<f32> {
        assert_eq!(raw.len() % BLOCK_SIZE, 0, "Q8_0: tamanho inválido {}", raw.len());

        let mut output = Vec::with_capacity((raw.len() / BLOCK_SIZE) * WEIGHTS_PER_BLOCK);

        for block in raw.chunks_exact(BLOCK_SIZE) {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            for &q in &block[2..34] {
                // q é int8 com sinal (interpretado como i8)
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
