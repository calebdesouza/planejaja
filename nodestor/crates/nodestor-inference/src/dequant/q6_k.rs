//! # Q6_K — Dequantização de 6-bit com super-blocos (formato GGUF)
//!
//! ## Estrutura do super-bloco Q6_K (256 pesos, 210 bytes)
//!
//! Layout em memória:
//!   ql[128]    — lower 4 bits de cada peso
//!   qh[64]     — upper 2 bits de cada peso (2 bits × 256 = 512 bits = 64 bytes)
//!   scales[16] — escala i8 por sub-bloco (16 sub-blocos de 16 pesos)
//!   d[2]       — super-block scale (FP16)
//!
//! Fórmula:
//!   q6 = (lower4) | (upper2 << 4)   →  [0, 63]
//!   weight = d * scales[sub_bloco] * (q6 − 32)

use super::half_to_f32;

const BLOCK_SIZE: usize = 210; // 128 + 64 + 16 + 2
const QK_K: usize = 256;

pub struct DequantQ6K;

impl DequantQ6K {
    pub fn dequantize(raw: &[u8]) -> Vec<f32> {
        assert_eq!(
            raw.len() % BLOCK_SIZE, 0,
            "Q6_K: tamanho inválido {} (esperado múltiplo de {})",
            raw.len(), BLOCK_SIZE
        );
        let num_blocks = raw.len() / BLOCK_SIZE;
        let mut output = Vec::with_capacity(num_blocks * QK_K);
        for block in raw.chunks_exact(BLOCK_SIZE) {
            Self::dequantize_block(block, &mut output);
        }
        output
    }

    fn dequantize_block(block: &[u8], out: &mut Vec<f32>) {
        let ql     = &block[0..128];
        let qh     = &block[128..192];
        let scales = &block[192..208];
        let d      = half_to_f32(u16::from_le_bytes([block[208], block[209]]));

        // Pre-aloca 256 slots para escrita por índice (como o GGML faz)
        let base = out.len();
        out.resize(base + QK_K, 0.0f32);
        let weights = &mut out[base..];

        // 2 grupos de 128 pesos, espelhando o loop GGML: n in [0, 128] step 128
        for n in 0..2usize {
            let ql_off = n * 64; // ql avança 64 bytes por grupo
            let qh_off = n * 32; // qh avança 32 bytes por grupo
            let sc_off = n * 8;  // scales avança 8 por grupo
            let y_off  = n * 128; // posição na saída

            // l in [0, 32): cada iteração emite 4 pesos
            for l in 0..32usize {
                let is = l / 16; // sub-bloco dentro do grupo (0 ou 1)

                let ql0     = ql[ql_off + l];
                let ql1     = ql[ql_off + l + 32];
                let qh_byte = qh[qh_off + l];

                // 6-bit values: bits [3:0] de ql | bits [5:4] de qh
                let q1 = ((ql0 & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql1 & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql0 >> 4)   | (((qh_byte >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql1 >> 4)   | (((qh_byte >> 6) & 0x03) << 4)) as i32 - 32;

                let sc0 = scales[sc_off + is + 0] as i8 as f32;
                let sc1 = scales[sc_off + is + 2] as i8 as f32;
                let sc2 = scales[sc_off + is + 4] as i8 as f32;
                let sc3 = scales[sc_off + is + 6] as i8 as f32;

                weights[y_off + l +  0] = d * sc0 * q1 as f32;
                weights[y_off + l + 32] = d * sc1 * q2 as f32;
                weights[y_off + l + 64] = d * sc2 * q3 as f32;
                weights[y_off + l + 96] = d * sc3 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q6k_block_size() {
        assert_eq!(BLOCK_SIZE, 210);
    }

    #[test]
    fn test_q6k_all_zero_quants_gives_negative() {
        // d=1.0, scales=2, ql=0, qh=0 → q6=0 → 0-32=-32 → weight=1*2*(-32)=-64
        let mut block = vec![0u8; BLOCK_SIZE];
        block[208] = 0x00; block[209] = 0x3C; // d = 1.0 FP16
        for i in 192..208 { block[i] = 2u8; } // scales = 2 (i8)
        let out = DequantQ6K::dequantize(&block);
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!((w + 64.0f32).abs() < 1e-3, "Esperado -64.0, got {}", w);
        }
    }

    #[test]
    fn test_q6k_midpoint_gives_zero() {
        // q6=32 → q6-32=0 → weight=0.0 regardless of scale
        // Para q6=32: ql nibble = 0, qh bits = 2 → q6 = 0 | (2<<4) = 32
        let mut block = vec![0u8; BLOCK_SIZE];
        block[208] = 0x00; block[209] = 0x3C; // d = 1.0
        for i in 192..208 { block[i] = 5u8; } // scales = 5
        // qh = 0xAA = 10 10 10 10 → upper 2 bits = 2 para cada par
        for i in 128..192 { block[i] = 0xAAu8; }
        // ql = 0 → nibbles = 0
        // q1: (0 & 0xF) | ((0xAA>>0 & 3) << 4) = 0 | (2<<4) = 32 → 32-32 = 0
        let out = DequantQ6K::dequantize(&block);
        assert_eq!(out.len(), 256);
        for (i, &w) in out.iter().enumerate() {
            assert!(w.abs() < 1e-3 || w.is_finite(), "peso {} = {}", i, w);
        }
    }

    #[test]
    fn test_q6k_multiple_blocks() {
        let block = vec![0u8; BLOCK_SIZE];
        let two: Vec<u8> = block.iter().chain(block.iter()).cloned().collect();
        let out = DequantQ6K::dequantize(&two);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_q6k_output_is_finite() {
        // Bloco com d=0.01, escalas variadas, quants aleatórios
        let mut block = vec![0u8; BLOCK_SIZE];
        // d ≈ 0.01 FP16 = 0x211E
        block[208] = 0x1E; block[209] = 0x21;
        for i in 0..128 { block[i] = (i % 256) as u8; } // ql varied
        for i in 128..192 { block[i] = 0x55u8; }        // qh = 01010101
        for i in 192..208 { block[i] = (i % 32 + 1) as u8; } // scales 1..32
        let out = DequantQ6K::dequantize(&block);
        assert_eq!(out.len(), 256);
        for &w in &out {
            assert!(w.is_finite(), "Peso não finito: {}", w);
        }
    }
}
