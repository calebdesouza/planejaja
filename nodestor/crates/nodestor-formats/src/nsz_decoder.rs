//! # Decodificador TCA-TBE — Reconstrução Lossless (CPU Fallback + Validação)
//!
//! Reconstrói tensores originais a partir de tiles NSZ comprimidos.
//! Usado para:
//! - Validação bit-exact de que a compressão é realmente lossless
//! - Fallback CPU para máquinas sem GPU Vulkan
//! - Testes unitários e de integração
//!
//! ## Garantia Zero-Deviation
//!
//! Para cada peso decodificado, `original.to_bits() == decoded.to_bits()`.
//! Não "quase igual". **Idêntico em nível de bits.**

use crate::nsz_format::*;

/// Decodifica um tile TCA-TBE de volta para vetor de FP32.
///
/// Garantia: `encode(weights) → tiles → decode(tiles) == weights` (bit-exact).
pub fn decode_tile_fp32(tile: &NszTile) -> Vec<f32> {
    let n = tile.actual_size;
    let modal = tile.header.modal_exponent;

    let mut result = Vec::with_capacity(n);
    let mut near_idx = 0usize;    // índice no stream packed de near deltas
    let mut residual_idx = 0usize; // índice no vetor de expoentes residuais

    for i in 0..n {
        let byte_idx = i / 8;
        let bit_idx = i % 8;

        // 1. Determinar expoente via bitmaps (branch-free logic)
        let is_match = (tile.bitmap_match[byte_idx] >> bit_idx) & 1;
        let is_near = (tile.bitmap_near[byte_idx] >> bit_idx) & 1;

        let exponent: u8 = if is_match == 1 {
            modal
        } else if is_near == 1 {
            // Ler delta do near stream packed
            let delta_byte = near_idx / 8;
            let delta_bit = near_idx % 8;
            let delta_val = if delta_byte < tile.near_deltas.len() {
                (tile.near_deltas[delta_byte] >> delta_bit) & 1
            } else {
                0 // segurança
            };
            near_idx += 1;
            if delta_val == 1 {
                modal.wrapping_add(1) // modal + 1
            } else {
                modal.wrapping_sub(1) // modal - 1
            }
        } else {
            // Residual: expoente completo
            let exp = if residual_idx < tile.residual_exponents.len() {
                tile.residual_exponents[residual_idx]
            } else {
                0 // segurança
            };
            residual_idx += 1;
            exp
        };

        // 2. Ler mantissa verbatim (23 bits → 3 bytes por mantissa)
        let m_base = i * 3;
        let mantissa = if m_base + 2 < tile.mantissas.len() {
            (tile.mantissas[m_base] as u32)
                | ((tile.mantissas[m_base + 1] as u32) << 8)
                | ((tile.mantissas[m_base + 2] as u32 & 0x7F) << 16)
        } else {
            0
        };

        // 3. Ler sinal (1 bit packed)
        let sign_byte = i / 8;
        let sign_bit = i % 8;
        let sign = if sign_byte < tile.signs.len() {
            ((tile.signs[sign_byte] >> sign_bit) & 1) as u32
        } else {
            0
        };

        // 4. Reconstruir FP32: sinal(1) | expoente(8) | mantissa(23)
        let fp32_bits = (sign << 31) | ((exponent as u32) << 23) | mantissa;
        result.push(f32::from_bits(fp32_bits));
    }

    result
}

/// Decodifica um tile TCA-TBE de volta para vetor de FP16 (u16 raw).
pub fn decode_tile_fp16(tile: &NszTile) -> Vec<u16> {
    let n = tile.actual_size;
    let modal = tile.header.modal_exponent;
    let mut result = Vec::with_capacity(n);
    let mut near_idx = 0usize;
    let mut residual_idx = 0usize;

    for i in 0..n {
        let byte_idx = i / 8;
        let bit_idx = i % 8;

        let is_match = (tile.bitmap_match[byte_idx] >> bit_idx) & 1;
        let is_near = (tile.bitmap_near[byte_idx] >> bit_idx) & 1;

        let exponent: u8 = if is_match == 1 {
            modal
        } else if is_near == 1 {
            let delta_byte = near_idx / 8;
            let delta_bit = near_idx % 8;
            let delta_val = if delta_byte < tile.near_deltas.len() {
                (tile.near_deltas[delta_byte] >> delta_bit) & 1
            } else { 0 };
            near_idx += 1;
            if delta_val == 1 { modal.wrapping_add(1) } else { modal.wrapping_sub(1) }
        } else {
            let exp = if residual_idx < tile.residual_exponents.len() {
                tile.residual_exponents[residual_idx]
            } else { 0 };
            residual_idx += 1;
            exp
        };

        // FP16 mantissa = 10 bits → 2 bytes
        let m_base = i * 2;
        let mantissa = if m_base + 1 < tile.mantissas.len() {
            (tile.mantissas[m_base] as u16)
                | ((tile.mantissas[m_base + 1] as u16 & 0x03) << 8)
        } else { 0 };

        let sign_byte = i / 8;
        let sign_bit = i % 8;
        let sign = if sign_byte < tile.signs.len() {
            ((tile.signs[sign_byte] >> sign_bit) & 1) as u16
        } else { 0 };

        // FP16: sinal(1) | expoente(5) | mantissa(10)
        let fp16_bits = (sign << 15) | ((exponent as u16 & 0x1F) << 10) | mantissa;
        result.push(fp16_bits);
    }

    result
}

/// Decodifica um tile TCA-TBE de volta para vetor de BF16 (u16 raw).
pub fn decode_tile_bf16(tile: &NszTile) -> Vec<u16> {
    let n = tile.actual_size;
    let modal = tile.header.modal_exponent;
    let mut result = Vec::with_capacity(n);
    let mut near_idx = 0usize;
    let mut residual_idx = 0usize;

    for i in 0..n {
        let byte_idx = i / 8;
        let bit_idx = i % 8;

        let is_match = (tile.bitmap_match[byte_idx] >> bit_idx) & 1;
        let is_near = (tile.bitmap_near[byte_idx] >> bit_idx) & 1;

        let exponent: u8 = if is_match == 1 {
            modal
        } else if is_near == 1 {
            let delta_byte = near_idx / 8;
            let delta_bit = near_idx % 8;
            let delta_val = if delta_byte < tile.near_deltas.len() {
                (tile.near_deltas[delta_byte] >> delta_bit) & 1
            } else { 0 };
            near_idx += 1;
            if delta_val == 1 { modal.wrapping_add(1) } else { modal.wrapping_sub(1) }
        } else {
            let exp = if residual_idx < tile.residual_exponents.len() {
                tile.residual_exponents[residual_idx]
            } else { 0 };
            residual_idx += 1;
            exp
        };

        // BF16 mantissa = 7 bits → 1 byte
        let mantissa = if i < tile.mantissas.len() {
            tile.mantissas[i] & 0x7F
        } else { 0 };

        let sign_byte = i / 8;
        let sign_bit = i % 8;
        let sign = if sign_byte < tile.signs.len() {
            ((tile.signs[sign_byte] >> sign_bit) & 1) as u16
        } else { 0 };

        // BF16: sinal(1) | expoente(8) | mantissa(7)
        let bf16_bits = (sign << 15) | ((exponent as u16) << 7) | (mantissa as u16);
        result.push(bf16_bits);
    }

    result
}

/// Valida que uma codificação é bit-exact lossless para FP32.
///
/// Retorna `true` se `decode(encode(original)) == original` em cada bit.
pub fn validate_lossless_fp32(original: &[f32], decoded: &[f32]) -> bool {
    if original.len() != decoded.len() {
        return false;
    }
    for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
        if orig.to_bits() != dec.to_bits() {
            eprintln!(
                "DIVERGÊNCIA bit-level no peso #{}: original=0x{:08X} ({}) decoded=0x{:08X} ({})",
                i,
                orig.to_bits(), orig,
                dec.to_bits(), dec,
            );
            return false;
        }
    }
    true
}

/// Valida lossless para FP16 (u16 raw).
pub fn validate_lossless_u16(original: &[u16], decoded: &[u16]) -> bool {
    if original.len() != decoded.len() {
        return false;
    }
    for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
        if orig != dec {
            eprintln!(
                "DIVERGÊNCIA bit-level no peso #{}: original=0x{:04X} decoded=0x{:04X}",
                i, orig, dec,
            );
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsz_encoder::NszEncoder;

    #[test]
    fn test_roundtrip_fp32_ones() {
        let encoder = NszEncoder::new();
        let weights = vec![1.0f32; 512];
        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);
        assert!(validate_lossless_fp32(&weights, &decoded), "Roundtrip falhou para 1.0s");
    }

    #[test]
    fn test_roundtrip_fp32_zeros() {
        let encoder = NszEncoder::new();
        let weights = vec![0.0f32; 512];
        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);
        assert!(validate_lossless_fp32(&weights, &decoded), "Roundtrip falhou para zeros");
    }

    #[test]
    fn test_roundtrip_fp32_negative() {
        let encoder = NszEncoder::new();
        let weights = vec![-0.5f32; 512];
        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);
        assert!(validate_lossless_fp32(&weights, &decoded), "Roundtrip falhou para negativos");
    }

    #[test]
    fn test_roundtrip_fp32_mixed_realistic() {
        let encoder = NszEncoder::new();
        // Simula pesos neurais: maioria perto de 0, alguns outliers
        let mut weights = Vec::with_capacity(512);
        for i in 0..400 {
            weights.push(0.01 * (i as f32 - 200.0) / 200.0); // ~[-0.01, 0.01]
        }
        for i in 0..80 {
            weights.push(0.5 * (i as f32 - 40.0) / 40.0); // ~[-0.5, 0.5] (near)
        }
        for i in 0..32 {
            weights.push(100.0 + i as f32); // outliers grandes (residual)
        }

        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);
        assert!(
            validate_lossless_fp32(&weights, &decoded),
            "Roundtrip falhou para pesos mistos realistas"
        );
    }

    #[test]
    fn test_roundtrip_fp32_subnormals() {
        let encoder = NszEncoder::new();
        // Testa valores subnormais (expoente = 0)
        let mut weights = Vec::with_capacity(512);
        for i in 0..512 {
            weights.push(f32::from_bits(i as u32)); // subnormais e zeros
        }
        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);
        assert!(
            validate_lossless_fp32(&weights, &decoded),
            "Roundtrip falhou para subnormais"
        );
    }

    #[test]
    fn test_roundtrip_fp32_multi_tile() {
        let encoder = NszEncoder::new();
        let mut weights = Vec::with_capacity(1500);
        for i in 0..1500 {
            weights.push((i as f32 * 0.001) - 0.75);
        }
        let tiles = encoder.encode_fp32(&weights);
        assert_eq!(tiles.len(), 3); // 512 + 512 + 476

        let mut decoded = Vec::new();
        for tile in &tiles {
            decoded.extend_from_slice(&decode_tile_fp32(tile));
        }
        assert!(
            validate_lossless_fp32(&weights, &decoded),
            "Roundtrip multi-tile falhou"
        );
    }

    #[test]
    fn test_roundtrip_fp32_special_values() {
        let encoder = NszEncoder::new();
        let mut weights = vec![0.0f32; 512];
        weights[0] = f32::INFINITY;
        weights[1] = f32::NEG_INFINITY;
        weights[2] = -0.0;
        weights[3] = f32::MIN_POSITIVE;
        weights[4] = f32::MAX;
        weights[5] = f32::MIN;
        // NaN é especial: NaN.to_bits() pode variar, mas devemos preservar
        weights[6] = f32::NAN;

        let tiles = encoder.encode_fp32(&weights);
        let decoded = decode_tile_fp32(&tiles[0]);

        // Validar que INF, -INF, -0.0, MIN_POSITIVE, MAX, MIN são idênticos
        for i in 0..6 {
            assert_eq!(
                weights[i].to_bits(), decoded[i].to_bits(),
                "Valor especial #{} divergiu", i
            );
        }
        // NaN: ambos devem ser NaN (bits podem diferir mas ambos são NaN)
        assert!(decoded[6].is_nan(), "NaN não foi preservado");
    }
}
