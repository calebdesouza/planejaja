//! # Codificador TCA-TBE — Compressão Lossless Offline
//!
//! Codifica tensores FP32/FP16/BF16 no formato NSZ (NodeStor Zip) usando
//! Tensor-Core-Aware Triple Bitmap Encoding.
//!
//! ## Processo de Codificação (por tile de 512 pesos):
//!
//! 1. Extrair expoente, mantissa e sinal de cada peso
//! 2. Encontrar o expoente modal (mais frequente)
//! 3. Classificar cada peso em 3 categorias:
//!    - **Match**: expoente == modal → bit=1 no bitmap_match
//!    - **Near**: |expoente - modal| == 1 → bit=1 no bitmap_near + delta packed
//!    - **Residual**: outlier → expoente completo no residual buffer
//! 4. Copiar mantissas e sinais verbatim (lossless obrigatório)
//!
//! ## Uso
//!
//! ```ignore
//! let encoder = NszEncoder::new();
//! let stats = encoder.encode_model("model.gguf", "model.nsz")?;
//! println!("Compressão: {:.1}%", stats.savings_percent);
//! ```

use crate::nsz_format::*;
use nodestor_core::NodeStorError;

/// Estatísticas de uma operação de codificação.
#[derive(Debug, Clone)]
pub struct NszEncodeStats {
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub num_tensors: u32,
    pub num_tiles: u32,
    pub savings_percent: f32,
    pub avg_modal_concentration: f32,
    pub avg_near_concentration: f32,
    pub total_residuals: u64,
}

/// Codificador NSZ para tensores de qualquer formato flutuante.
pub struct NszEncoder {
    /// Tamanho do tile em pesos (padrão: 512)
    pub tile_size: usize,
}

impl NszEncoder {
    pub fn new() -> Self {
        Self { tile_size: TILE_SIZE }
    }

    /// Codifica um tensor FP32 inteiro em uma sequência de tiles TCA-TBE.
    pub fn encode_fp32(&self, weights: &[f32]) -> Vec<NszTile> {
        let mut tiles = Vec::new();
        for chunk in weights.chunks(self.tile_size) {
            tiles.push(self.encode_tile_fp32(chunk));
        }
        tiles
    }

    /// Codifica um tensor FP16 (u16 raw) em tiles TCA-TBE.
    pub fn encode_fp16(&self, weights: &[u16]) -> Vec<NszTile> {
        let mut tiles = Vec::new();
        for chunk in weights.chunks(self.tile_size) {
            tiles.push(self.encode_tile_fp16(chunk));
        }
        tiles
    }

    /// Codifica um tensor BF16 (u16 raw) em tiles TCA-TBE.
    pub fn encode_bf16(&self, weights: &[u16]) -> Vec<NszTile> {
        let mut tiles = Vec::new();
        for chunk in weights.chunks(self.tile_size) {
            tiles.push(self.encode_tile_bf16(chunk));
        }
        tiles
    }

    /// Codifica um único tile de pesos FP32.
    fn encode_tile_fp32(&self, chunk: &[f32]) -> NszTile {
        let n = chunk.len();
        // 1. Extrair componentes IEEE-754 FP32
        let mut exponents = Vec::with_capacity(n);
        let mut mantissas_raw = Vec::with_capacity(n);
        let mut signs = Vec::with_capacity(n);

        for &w in chunk {
            let bits = w.to_bits();
            signs.push(((bits >> 31) & 1) as u8);
            exponents.push(((bits >> 23) & 0xFF) as u8);
            // Mantissa = 23 bits, stored as 3 bytes (24 bits, MSB padding)
            let mantissa = bits & 0x7FFFFF;
            mantissas_raw.push(mantissa);
        }

        // 2. Encontrar expoente modal
        let modal = find_modal_exp(&exponents);

        // 3. Classificar e construir bitmaps
        let bitmap_bytes = (n + 7) / 8;
        let mut bitmap_match = vec![0u8; bitmap_bytes];
        let mut bitmap_near = vec![0u8; bitmap_bytes];
        let mut near_deltas_bits = Vec::new();
        let mut residual_exponents = Vec::new();

        for (i, &exp) in exponents.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            if exp == modal {
                bitmap_match[byte_idx] |= 1 << bit_idx;
            } else if exp == modal.wrapping_add(1) || (modal > 0 && exp == modal - 1) {
                bitmap_near[byte_idx] |= 1 << bit_idx;
                // Delta: 0 = modal-1, 1 = modal+1
                near_deltas_bits.push(if exp > modal { 1u8 } else { 0u8 });
            } else {
                residual_exponents.push(exp);
            }
        }

        // Pack near deltas into bytes (8 deltas per byte)
        let near_deltas = pack_bits(&near_deltas_bits);

        // 4. Pack mantissas (23 bits cada para FP32 → 3 bytes por mantissa)
        let mut mantissa_bytes = Vec::with_capacity(n * 3);
        for &m in &mantissas_raw {
            mantissa_bytes.push((m & 0xFF) as u8);
            mantissa_bytes.push(((m >> 8) & 0xFF) as u8);
            mantissa_bytes.push(((m >> 16) & 0x7F) as u8);
        }

        // 5. Pack sinais (1 bit por peso)
        let sign_bytes = pack_bits(&signs);

        let header = TileHeader {
            modal_exponent: modal,
            num_residuals: residual_exponents.len() as u16,
            dtype: OriginalDtype::FP32 as u8,
            num_near: near_deltas_bits.len() as u16,
            _reserved: [0, 0],
        };

        NszTile {
            header,
            bitmap_match,
            bitmap_near,
            near_deltas,
            residual_exponents,
            mantissas: mantissa_bytes,
            signs: sign_bytes,
            actual_size: n,
        }
    }

    /// Codifica um único tile de pesos FP16.
    fn encode_tile_fp16(&self, chunk: &[u16]) -> NszTile {
        let n = chunk.len();
        let mut exponents = Vec::with_capacity(n);
        let mut mantissas_raw = Vec::with_capacity(n);
        let mut signs = Vec::with_capacity(n);

        for &w in chunk {
            signs.push(((w >> 15) & 1) as u8);
            exponents.push(((w >> 10) & 0x1F) as u8); // 5 bits
            mantissas_raw.push((w & 0x3FF) as u16); // 10 bits
        }

        let modal = find_modal_exp(&exponents);

        let bitmap_bytes = (n + 7) / 8;
        let mut bitmap_match = vec![0u8; bitmap_bytes];
        let mut bitmap_near = vec![0u8; bitmap_bytes];
        let mut near_deltas_bits = Vec::new();
        let mut residual_exponents = Vec::new();

        for (i, &exp) in exponents.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if exp == modal {
                bitmap_match[byte_idx] |= 1 << bit_idx;
            } else if exp == modal.wrapping_add(1) || (modal > 0 && exp == modal - 1) {
                bitmap_near[byte_idx] |= 1 << bit_idx;
                near_deltas_bits.push(if exp > modal { 1u8 } else { 0u8 });
            } else {
                residual_exponents.push(exp);
            }
        }

        let near_deltas = pack_bits(&near_deltas_bits);

        // FP16 mantissa = 10 bits → 2 bytes por mantissa
        let mut mantissa_bytes = Vec::with_capacity(n * 2);
        for &m in &mantissas_raw {
            mantissa_bytes.push((m & 0xFF) as u8);
            mantissa_bytes.push(((m >> 8) & 0x03) as u8);
        }

        let sign_bytes = pack_bits(&signs);

        NszTile {
            header: TileHeader {
                modal_exponent: modal,
                num_residuals: residual_exponents.len() as u16,
                dtype: OriginalDtype::FP16 as u8,
                num_near: near_deltas_bits.len() as u16,
                _reserved: [0, 0],
            },
            bitmap_match,
            bitmap_near,
            near_deltas,
            residual_exponents,
            mantissas: mantissa_bytes,
            signs: sign_bytes,
            actual_size: n,
        }
    }

    /// Codifica um único tile de pesos BF16.
    fn encode_tile_bf16(&self, chunk: &[u16]) -> NszTile {
        let n = chunk.len();
        let mut exponents = Vec::with_capacity(n);
        let mut mantissas_raw = Vec::with_capacity(n);
        let mut signs = Vec::with_capacity(n);

        for &w in chunk {
            signs.push(((w >> 15) & 1) as u8);
            exponents.push(((w >> 7) & 0xFF) as u8); // 8 bits (igual FP32)
            mantissas_raw.push((w & 0x7F) as u8); // 7 bits
        }

        let modal = find_modal_exp(&exponents);

        let bitmap_bytes = (n + 7) / 8;
        let mut bitmap_match = vec![0u8; bitmap_bytes];
        let mut bitmap_near = vec![0u8; bitmap_bytes];
        let mut near_deltas_bits = Vec::new();
        let mut residual_exponents = Vec::new();

        for (i, &exp) in exponents.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if exp == modal {
                bitmap_match[byte_idx] |= 1 << bit_idx;
            } else if exp == modal.wrapping_add(1) || (modal > 0 && exp == modal - 1) {
                bitmap_near[byte_idx] |= 1 << bit_idx;
                near_deltas_bits.push(if exp > modal { 1u8 } else { 0u8 });
            } else {
                residual_exponents.push(exp);
            }
        }

        let near_deltas = pack_bits(&near_deltas_bits);

        // BF16 mantissa = 7 bits → 1 byte por mantissa (com 1 bit padding)
        let mantissa_bytes: Vec<u8> = mantissas_raw;

        let sign_bytes = pack_bits(&signs);

        NszTile {
            header: TileHeader {
                modal_exponent: modal,
                num_residuals: residual_exponents.len() as u16,
                dtype: OriginalDtype::BF16 as u8,
                num_near: near_deltas_bits.len() as u16,
                _reserved: [0, 0],
            },
            bitmap_match,
            bitmap_near,
            near_deltas,
            residual_exponents,
            mantissas: mantissa_bytes,
            signs: sign_bytes,
            actual_size: n,
        }
    }

    /// Codifica tiles e serializa em bytes prontos para disco.
    pub fn encode_tensor_to_bytes(&self, weights: &[f32], name: &str) -> (NszTensorEntry, Vec<u8>) {
        let tiles = self.encode_fp32(weights);
        let mut data = Vec::new();
        for tile in &tiles {
            data.extend_from_slice(&tile.to_bytes());
        }

        let entry = NszTensorEntry {
            name: name.to_string(),
            dtype: OriginalDtype::FP32,
            original_size: (weights.len() * 4) as u64,
            compressed_offset: 0, // será ajustado pelo writer
            compressed_size: data.len() as u64,
            num_tiles: tiles.len() as u32,
            num_weights: weights.len() as u64,
            shape: [weights.len() as u32, 0, 0, 0],
        };

        (entry, data)
    }
}

/// Encontra o expoente mais frequente num slice.
fn find_modal_exp(exponents: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &e in exponents {
        hist[e as usize] += 1;
    }
    let mut modal = 0u8;
    let mut max_count = 0;
    for (i, &c) in hist.iter().enumerate() {
        if c > max_count {
            max_count = c;
            modal = i as u8;
        }
    }
    modal
}

/// Empacota um vetor de bits (0 ou 1) em bytes.
fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let num_bytes = (bits.len() + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_all_same_fp32() {
        let encoder = NszEncoder::new();
        let weights = vec![1.0f32; 512];
        let tiles = encoder.encode_fp32(&weights);
        assert_eq!(tiles.len(), 1);
        let tile = &tiles[0];
        // Todos têm o mesmo expoente (127 para 1.0)
        assert_eq!(tile.header.modal_exponent, 127);
        assert_eq!(tile.header.num_residuals, 0);
        assert_eq!(tile.header.num_near, 0);
        // Todos os bits no bitmap_match devem estar set
        for &byte in &tile.bitmap_match {
            assert_eq!(byte, 0xFF);
        }
    }

    #[test]
    fn test_encode_mixed_fp32() {
        let encoder = NszEncoder::new();
        let mut weights = vec![1.0f32; 400]; // exp=127
        weights.extend_from_slice(&[2.0f32; 80]); // exp=128 (near: modal+1)
        weights.extend_from_slice(&[100.0f32; 32]); // exp=133 (residual)
        let tiles = encoder.encode_fp32(&weights);
        assert_eq!(tiles.len(), 1);
        let tile = &tiles[0];
        assert_eq!(tile.header.modal_exponent, 127);
        assert_eq!(tile.header.num_near, 80);
        assert_eq!(tile.header.num_residuals, 32);
    }

    #[test]
    fn test_pack_bits() {
        let bits = vec![1, 0, 1, 1, 0, 0, 0, 1, 1];
        let packed = pack_bits(&bits);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0b10001101); // bits 0-7
        assert_eq!(packed[1], 0b00000001); // bit 8
    }

    #[test]
    fn test_encode_compression_ratio() {
        let encoder = NszEncoder::new();
        // 512 pesos × 4 bytes = 2048 bytes original
        let weights = vec![0.02f32; 512];
        let tiles = encoder.encode_fp32(&weights);
        let compressed = tiles[0].to_bytes();
        // Com pesos idênticos, compressão deve ser significativa
        assert!(
            compressed.len() < 2048,
            "Compressed ({}) deve ser menor que original (2048)",
            compressed.len()
        );
    }

    #[test]
    fn test_encode_tensor_to_bytes() {
        let encoder = NszEncoder::new();
        let weights = vec![1.0f32; 1024];
        let (entry, data) = encoder.encode_tensor_to_bytes(&weights, "test.weight");
        assert_eq!(entry.name, "test.weight");
        assert_eq!(entry.num_weights, 1024);
        assert_eq!(entry.num_tiles, 2); // 1024 / 512 = 2
        assert!(data.len() > 0);
    }

    #[test]
    fn test_find_modal() {
        let exps = vec![127u8, 127, 127, 128, 126, 127, 127];
        assert_eq!(find_modal_exp(&exps), 127);
    }
}
