//! # Q5_0 — Dequantização de 5-bit simples (formato GGUF)
//!
//! ## Estrutura do bloco Q5_0 (32 pesos, 22 bytes)
//!
//! ```text
//! ┌──────────┬───────────┬──────────────────────┐
//! │  d [2B]  │  qh [4B]  │       qs [16B]        │
//! │  FP16    │ 5°bit×32  │  nibbles 4-bit × 32   │
//! └──────────┴───────────┴──────────────────────┘
//! ```
//!
//! ## Fórmula
//!
//! Para cada par (j=0..15):
//!   x0 = (qs[j] & 0x0F) | (bit_j(qh) << 4) − 16
//!   x1 = (qs[j] >> 4)   | (bit_{j+16}(qh) << 4) − 16
//!   weight_{j}    = d × x0
//!   weight_{j+16} = d × x1
//!
//! onde qh é um uint32 LE e bit_i(qh) = (qh >> i) & 1.

use super::half_to_f32;

const QK5_0: usize = 32;
const BLOCK_SIZE: usize = 2 + 4 + QK5_0 / 2; // = 22 bytes

pub struct DequantQ5_0;

impl DequantQ5_0 {
    pub fn dequantize(raw: &[u8]) -> Vec<f32> {
        assert_eq!(
            raw.len() % BLOCK_SIZE, 0,
            "Q5_0: tamanho inválido {} (esperado múltiplo de {})",
            raw.len(), BLOCK_SIZE
        );
        let num_blocks = raw.len() / BLOCK_SIZE;
        let mut result = vec![0.0f32; num_blocks * QK5_0];

        for (b, block) in raw.chunks_exact(BLOCK_SIZE).enumerate() {
            let d  = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let qs = &block[6..22];
            let out = &mut result[b * QK5_0..(b + 1) * QK5_0];

            // GGML Q5_0 "split" layout: first 16 weights use lower nibbles + qh bits 0..15,
            // next 16 weights use upper nibbles + qh bits 16..31.
            for j in 0..QK5_0 / 2 {
                let xh_0 = ((qh >> j)        & 0x1) as i32;
                let xh_1 = ((qh >> (j + 16)) & 0x1) as i32;
                out[j]             = ((((qs[j] & 0x0F) as i32) | (xh_0 << 4)) - 16) as f32 * d;
                out[j + QK5_0 / 2] = ((((qs[j] >> 4)   as i32) | (xh_1 << 4)) - 16) as f32 * d;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_0_block_size() {
        assert_eq!(BLOCK_SIZE, 22);
    }

    #[test]
    fn test_q5_0_all_zero_quants() {
        // d=1.0 FP16=0x3C00, qh=0 (no 5th bits), qs=0 (all nibbles=0)
        // x0 = (0 | 0) - 16 = -16; weight = 1.0 * -16 = -16.0
        // x1 = (0 | 0) - 16 = -16; weight = 1.0 * -16 = -16.0
        let mut block = vec![0u8; BLOCK_SIZE];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0 FP16
        let out = DequantQ5_0::dequantize(&block);
        assert_eq!(out.len(), QK5_0);
        for &w in &out {
            assert!((w + 16.0).abs() < 1e-4, "Expected -16.0, got {}", w);
        }
    }

    #[test]
    fn test_q5_0_midpoint_gives_zero() {
        // q = (0xF nibble) | (1 << 4) = 15 | 16 = 31 - 16 = 15 → not zero
        // Midpoint: q = 16 → qs nibble = 0, qh bit = 1 → (0 | 16) - 16 = 0 → weight = 0
        let mut block = vec![0u8; BLOCK_SIZE];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        // Set all qh bits = 1 (lower 16 bits for first group, upper 16 for second)
        block[2] = 0xFF; block[3] = 0xFF; block[4] = 0xFF; block[5] = 0xFF;
        // qs = 0 (nibbles = 0) → q = (0 | 1<<4) - 16 = 16 - 16 = 0
        let out = DequantQ5_0::dequantize(&block);
        assert_eq!(out.len(), QK5_0);
        for &w in &out {
            assert!(w.abs() < 1e-4, "Expected 0.0, got {}", w);
        }
    }

    #[test]
    fn test_q5_0_output_is_finite() {
        // Use a realistic block with mixed values
        let mut block = vec![0u8; BLOCK_SIZE];
        block[0] = 0x89; block[1] = 0x3A; // d = small positive FP16
        block[2] = 0xAA; block[3] = 0x55; block[4] = 0xAA; block[5] = 0x55;
        for i in 6..22 { block[i] = 0x5A; }
        let out = DequantQ5_0::dequantize(&block);
        assert_eq!(out.len(), QK5_0);
        for &w in &out {
            assert!(w.is_finite(), "Expected finite, got {}", w);
        }
    }
}
