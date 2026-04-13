//! # CPU SIMD — Dequantização ARM NEON
//!
//! Kernels para Apple Silicon (M1/M2/M3/M4), AWS Graviton, Snapdragon, Ampere.
//! ARM NEON processa 4 floats por instrução (128-bit lanes).
//!
//! Em Apple M1/M2, o NEON tem throughput de ~130-180 GFLOPS para FP32,
//! tornando a dequantização de modelos 70B viável sem GPU dedicada.

use super::q4_k::{DequantQ4K, Q4KVariant};
use super::q5_k::{DequantQ5K, Q5KVariant};

/// Verifica se NEON está disponível (sempre true em ARM64/AArch64).
#[inline]
pub fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    { true }
    #[cfg(not(target_arch = "aarch64"))]
    { false }
}

/// Dequantiza Q4_K usando ARM NEON se disponível.
pub fn dequant_q4k_neon(raw: &[u8], variant: Q4KVariant) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dequant_q4k_neon_inner(raw, variant) };
    }
    #[allow(unreachable_code)]
    DequantQ4K::dequantize(raw, variant)
}

/// Dequantiza Q5_K usando ARM NEON se disponível.
pub fn dequant_q5k_neon(raw: &[u8], variant: Q5KVariant) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dequant_q5k_neon_inner(raw, variant) };
    }
    #[allow(unreachable_code)]
    DequantQ5K::dequantize(raw, variant)
}

/// Converte N × FP16 para FP32 usando NEON `vcvt_f32_f16`.
///
/// Em Apple M1: esta instrução é executada em 1 ciclo para 4 valores simultâneos.
pub fn convert_f16_to_f32_neon(f16_data: &[u16]) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { convert_f16_neon_inner(f16_data) };
    }
    #[allow(unreachable_code)]
    f16_data.iter().map(|&h| super::half_to_f32(h)).collect()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_q4k_neon_inner(raw: &[u8], _variant: Q4KVariant) -> Vec<f32> {
    use std::arch::aarch64::*;

    const BLOCK_SIZE: usize = 144;
    const QK_K: usize = 256;

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d    = super::half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = super::half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs     = &block[16..144];

        // NEON: vetores de 4 floats (128-bit)
        let d_v    = vdupq_n_f32(d);
        let dmin_v = vdupq_n_f32(dmin);

        let mut is = 0usize;
        let mut q_offset = 0usize;

        for _chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);

            let scale0_v = vmulq_n_f32(d_v, sc0 as f32);
            let scale1_v = vmulq_n_f32(d_v, sc1 as f32);
            let min0_v   = vmulq_n_f32(dmin_v, m0 as f32);
            let min1_v   = vmulq_n_f32(dmin_v, m1 as f32);

            // 32 pesos: nibbles baixos, em chunks de 4
            for l in (0..32).step_by(4) {
                // Carrega 4 bytes
                let b = [
                    qs[q_offset + l]     & 0x0F,
                    qs[q_offset + l + 1] & 0x0F,
                    qs[q_offset + l + 2] & 0x0F,
                    qs[q_offset + l + 3] & 0x0F,
                ];
                // u8 → u32 → f32 via NEON
                let u8x4   = vld1_u8(b.as_ptr());   // 4 × u8 em 64-bit
                let u16x4  = vmovl_u8(u8x4);        // u8 → u16
                let u32x4  = vmovl_u16(vget_low_u16(u16x4)); // u16 → u32
                let f32x4  = vcvtq_f32_u32(u32x4);  // u32 → f32
                // peso = scale * q - min
                let result = vsubq_f32(vmulq_f32(scale0_v, f32x4), min0_v);
                let mut tmp = [0f32; 4];
                vst1q_f32(tmp.as_mut_ptr(), result);
                output.extend_from_slice(&tmp);
            }

            // 32 pesos: nibbles altos
            for l in (0..32).step_by(4) {
                let b = [
                    qs[q_offset + l]     >> 4,
                    qs[q_offset + l + 1] >> 4,
                    qs[q_offset + l + 2] >> 4,
                    qs[q_offset + l + 3] >> 4,
                ];
                let u8x4  = vld1_u8(b.as_ptr());
                let u16x4 = vmovl_u8(u8x4);
                let u32x4 = vmovl_u16(vget_low_u16(u16x4));
                let f32x4 = vcvtq_f32_u32(u32x4);
                let result = vsubq_f32(vmulq_f32(scale1_v, f32x4), min1_v);
                let mut tmp = [0f32; 4];
                vst1q_f32(tmp.as_mut_ptr(), result);
                output.extend_from_slice(&tmp);
            }

            q_offset += 32;
            is += 2;
        }
    }

    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_q5k_neon_inner(raw: &[u8], variant: Q5KVariant) -> Vec<f32> {
    // Para Q5_K, usa o caminho escalar por agora (bits extras do qh complicam NEON)
    // Em produção: usar vorr_u8 para combinar nibble + high bit
    DequantQ5K::dequantize(raw, variant)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn convert_f16_neon_inner(data: &[u16]) -> Vec<f32> {
    use std::arch::aarch64::*;

    let mut out = Vec::with_capacity(data.len());
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Carrega 4 × FP16 como u16x4
        let half4 = vld1_u16(chunk.as_ptr()) as *const f16;
        // vcvt_f32_f16: converte 4 × FP16 → 4 × FP32 em 1 ciclo
        let f32x4 = vcvt_f32_f16(vld1_f16(chunk.as_ptr() as *const f16));
        let mut tmp = [0f32; 4];
        vst1q_f32(tmp.as_mut_ptr(), vcombine_f32(f32x4, vdup_n_f32(0.0)));
        out.extend_from_slice(&tmp[..4]);
        let _ = half4; // silence warning
    }

    for &h in remainder {
        out.push(super::half_to_f32(h));
    }

    out
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

    #[test]
    fn test_neon_detection() {
        // neon_available() deve retornar true em ARM64, false em x86
        #[cfg(target_arch = "aarch64")]
        assert!(neon_available());
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!neon_available());
    }

    #[test]
    fn test_dequant_q4k_neon_matches_scalar() {
        use super::super::q4_k::DequantQ4K;

        let mut block = vec![0u8; 144];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0 FP16
        for i in 4..8 { block[i] = 1; }
        for i in 16..144 { block[i] = 0x33; }

        let scalar = DequantQ4K::dequantize(&block, Q4KVariant::Medium);
        let neon   = dequant_q4k_neon(&block, Q4KVariant::Medium);

        // Em x86, neon cai no escalar — logo ambos são idênticos
        assert_eq!(scalar.len(), neon.len());
        for (s, n) in scalar.iter().zip(neon.iter()) {
            assert!((s - n).abs() < 1e-4, "Diferença: scalar={} neon={}", s, n);
        }
    }

    #[test]
    fn test_f16_neon_fallback() {
        // Em x86 usa o caminho escalar
        let result = convert_f16_to_f32_neon(&[0x3C00]);
        assert!((result[0] - 1.0f32).abs() < 1e-4);
    }
}
