//! Crate para compressão GDeflate no NodeStor
//! Utiliza a libdeflate via FFI para máxima performance no processo de orquestração.

pub mod tile;

use nodestor_core::NodeStorError;
use gdeflate::{Compressor, Decompressor, CompressionLevel, OwnedCompressionResult};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GDeflateError {
    #[error("GDeflate Compression Error")]
    CompressionFailed,
    #[error("GDeflate Decompression Error")]
    DecompressionFailed,
}

impl From<GDeflateError> for NodeStorError {
    fn from(err: GDeflateError) -> Self {
        NodeStorError::GDeflateError(err.to_string())
    }
}

/// Comprime os dados no formato GDeflate.
/// Divide os dados em tiles de 64 KB (65536 bytes) para permitir decodificação paralela na GPU.
pub fn compress_gdeflate(data: &[u8]) -> Result<OwnedCompressionResult, GDeflateError> {
    let mut compressor = Compressor::new(CompressionLevel::Level3)
        .map_err(|_| GDeflateError::CompressionFailed)?;
        
    let result = compressor.compress(data, 65536)
        .map_err(|_| GDeflateError::CompressionFailed)?;
        
    Ok(result)
}

/// Decomprime os dados GDeflate via CPU.
/// Utilizado como fallback e para validação cross-reference.
pub fn decompress_gdeflate_cpu(result: &OwnedCompressionResult) -> Result<Vec<u8>, GDeflateError> {
    let mut decompressor = Decompressor::new()
        .map_err(|_| GDeflateError::DecompressionFailed)?;
        
    let decompressed = decompressor.decompress(result)
        .map_err(|_| GDeflateError::DecompressionFailed)?;
        
    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let original_data = b"Hello, GDeflate Universal! NodeStor-G is here.";
        
        let compressed = compress_gdeflate(original_data).expect("Failed to compress");
        assert!(!compressed.bytes.is_empty());
        assert_eq!(compressed.tile_size, 65536);
        assert_eq!(compressed.tiles.len(), 1);
        
        let decompressed = decompress_gdeflate_cpu(&compressed).expect("Failed to decompress");
        assert_eq!(original_data, decompressed.as_slice());
    }
}
