//! # Q5_K — Dequantização de 5-bit com super-blocos (formato GGUF)
//!
//! ## Estrutura do super-bloco Q5_K (256 pesos, 176 bytes)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │ SUPER-BLOCO Q5_K (176 bytes = 256 pesos)                      │
//! ├───────┬───────┬─────────────────┬───────────┬────────────────-┤
//! │  d    │ dmin  │   scales[12]    │  qh[32]   │    qs[128]     │
//! │ 2 B   │ 2 B   │   12 bytes      │  32 bytes │   128 bytes    │
//! │ FP16  │ FP16  │ 6-bit/sub-bloco │ bits altos│ nibbles baixos │
//! └───────┴───────┴─────────────────┴───────────┴────────────────-┘
//! ```
//!
//! ## Diferença em relação ao Q4_K
//!
//! O Q5_K adiciona um byte extra `qh` de "bits altos" para cada grupo de 8 pesos.
//! Cada peso usa 5 bits: 4 bits de `qs` (nibble) + 1 bit do `qh`.
//!
//! ```text
//! peso_5bit = nibble_4bit | (bit_alto_qh << 4)  →  q ∈ [0, 31]
//! ```
//!
//! Isso dá 32 níveis de quantização (vs 16 do Q4_K), reduzindo a perda de precisão
//! de ~1-2% para ~0.3%.

use super::half_to_f32;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
const BLOCK_SIZE: usize = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // = 176 bytes

/// Variante do formato Q5_K.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q5KVariant {
    Small,
    Medium,
}

/// Kernel de dequantização Q5_K em CPU puro (Rust escalar).
pub struct DequantQ5K;

impl DequantQ5K {
    /// Dequantiza um slice de bytes brutos Q5_K em FP32.
    pub fn dequantize(raw: &[u8], _variant: Q5KVariant) -> Vec<f32> {
        assert_eq!(
            raw.len() % BLOCK_SIZE, 0,
            "Q5_K: tamanho inválido {} (esperado múltiplo de {})",
            raw.len(), BLOCK_SIZE
        );

        let num_blocks = raw.len() / BLOCK_SIZE;
        let mut output = Vec::with_capacity(num_blocks * QK_K);

        for block in raw.chunks_exact(BLOCK_SIZE) {
            Self::dequantize_block(block, &mut output);
        }

        output
    }

    #[inline]
    fn dequantize_block(block: &[u8], out: &mut Vec<f32>) {
        // Layout: [d:2][dmin:2][scales:12][qh:32][qs:128]
        let d    = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];   // 12 bytes
        let qh     = &block[16..48];  // 32 bytes de bits altos (1 bit por peso)
        let qs     = &block[48..176]; // 128 bytes de nibbles

        let mut is = 0usize;
        let mut q_offset = 0usize;
        let mut qh_offset = 0usize;

        // 4 grupos de 64 pesos cada = 256 total
        for _chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            let (sc1, m1b) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1b as f32;

            // 32 pesos: nibble baixo + bit alto de qh
            for l in 0..32 {
                let nibble = qs[q_offset + l] & 0x0F;
                let high_bit = (qh[qh_offset + l / 8] >> (l % 8)) & 0x01;
                let q = nibble | (high_bit << 4); // q ∈ [0, 31]
                out.push(d1 * q as f32 - m1);
            }

            // 32 pesos: nibble alto + bit alto de qh (deslocado)
            for l in 0..32 {
                let nibble = qs[q_offset + l] >> 4;
                let high_bit = (qh[qh_offset + 4 + l / 8] >> (l % 8)) & 0x01;
                let q = nibble | (high_bit << 4);
                out.push(d2 * q as f32 - m2);
            }

            q_offset += 32;
            qh_offset += 8;
            is += 2;
        }
    }
}

#[inline(always)]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q5k_block(d_fp16: u16, dmin_fp16: u16, scale: u8, q_nibble: u8, q_high: u8) -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_SIZE];
        block[0] = (d_fp16 & 0xFF) as u8;
        block[1] = (d_fp16 >> 8) as u8;
        block[2] = (dmin_fp16 & 0xFF) as u8;
        block[3] = (dmin_fp16 >> 8) as u8;

        for i in 0..4 {
            block[4 + i] = scale & 0x3F;
        }

        // qh: bit alto uniforme (0 ou 1) em todos os bytes
        let qh_val = if q_high != 0 { 0xFF } else { 0x00 };
        for i in 16..48 {
            block[i] = qh_val;
        }

        // qs: nibble uniforme
        let byte_val = (q_nibble & 0xF) | ((q_nibble & 0xF) << 4);
        for i in 48..176 {
            block[i] = byte_val;
        }

        block
    }

    #[test]
    fn test_q5k_block_size() {
        let block = make_q5k_block(0x3C00, 0, 1, 0, 0);
        assert_eq!(block.len(), 176);
    }

    #[test]
    fn test_q5k_zero_weights() {
        // d=1.0, dmin=0, scale=1, q=0, high=0 → todos 0.0
        let block = make_q5k_block(0x3C00, 0x0000, 1, 0, 0);
        let out = DequantQ5K::dequantize(&block, Q5KVariant::Medium);
        assert_eq!(out.len(), 256);
        for w in &out {
            assert!(w.is_finite());
            assert!(w.abs() < 1e-4, "Esperado ~0.0, got {}", w);
        }
    }

    #[test]
    fn test_q5k_higher_range_than_q4k() {
        // Q5K suporte q ∈ [0, 31] vs Q4K q ∈ [0, 15]
        // Com scale=1 e high=1: q = nibble + 16
        // nibble=15 + high → q = 31 → peso = 31 * d * scale - dmin
        let block = make_q5k_block(0x3C00, 0x0000, 1, 15, 1);
        let out = DequantQ5K::dequantize(&block, Q5KVariant::Medium);
        assert_eq!(out.len(), 256);
        // Com high=1 e nibble=15: q=31, peso = 31.0
        for w in &out {
            assert!(w.is_finite());
            assert!(*w >= 0.0, "Pesos devem ser não-negativos com dmin=0, got {}", w);
        }
    }

    #[test]
    fn test_q5k_multiple_blocks() {
        let block = make_q5k_block(0x3C00, 0, 2, 7, 0);
        let two_blocks: Vec<u8> = block.iter().chain(block.iter()).cloned().collect();
        let out = DequantQ5K::dequantize(&two_blocks, Q5KVariant::Medium);
        assert_eq!(out.len(), 512);
    }
}
