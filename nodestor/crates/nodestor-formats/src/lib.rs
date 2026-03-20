//! nodestor-formats — Parsers para formatos de modelo GGUF e Safetensors.
//!
//! Todos os parsers são zero-allocation para dados de tensores: lemos apenas headers
//! e construímos um índice de offsets. Os tensores em si são transferidos via transport.

mod gguf;
mod safetensors;

pub use gguf::GgufParser;
pub use safetensors::SafetensorsParser;

use nodestor_core::{ModelParser, NodeStorError};

/// Seleciona automaticamente o parser correto baseado na extensão do arquivo.
pub fn detect_parser(path: &str) -> Result<Box<dyn ModelParser>, NodeStorError> {
    let lower = path.to_lowercase();
    if lower.ends_with(".gguf") {
        Ok(Box::new(GgufParser::new()))
    } else if lower.ends_with(".safetensors") {
        Ok(Box::new(SafetensorsParser::new()))
    } else {
        // Tenta detectar pelo magic bytes
        detect_by_magic(path)
    }
}

fn detect_by_magic(path: &str) -> Result<Box<dyn ModelParser>, NodeStorError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|_| NodeStorError::ModelNotFound(path.to_string()))?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).map_err(|e| NodeStorError::IoError(e))?;

    // GGUF magic: bytes b"GGUF"
    if &magic[..4] == b"GGUF" {
        return Ok(Box::new(GgufParser::new()));
    }
    // Safetensors: starts with 8-byte little-endian length then JSON
    // Heurística: primeiro byte é um int64 pequeno (< 1MB para o header JSON)
    let header_len = u64::from_le_bytes(magic);
    if header_len < 10_000_000 && header_len > 2 {
        return Ok(Box::new(SafetensorsParser::new()));
    }

    Err(NodeStorError::InvalidModelFormat(
        format!("Formato não reconhecido para: {}", path)
    ))
}
