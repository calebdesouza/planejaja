//! Gerenciamento de Tiles para GDeflate.
//! 
//! O NodeStor divide a compressão GDeflate em tiles (páginas) de 64 KB,
//! permitindo submissão paralela e granular ao shader de descompressão na GPU.
//! Este módulo gerencia o empacotamento do header com os offsets e o bitstream bruto.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use nodestor_core::NodeStorError;
use gdeflate::OwnedCompressionResult;
use std::io::{Cursor, Write};
use crate::GDeflateError;

/// Tamanho do tile escolhido para máxima saturação da GPU (compatível com NVidia e specs GDeflate)
pub const TILE_SIZE: usize = 65536; // 64 KB

/// Representa a página GDeflate serializada para viagem em I/O.
pub struct SerializedGDeflateStream {
    /// O buffer completo que pode ser persistido em arquivo e lido via mmap.
    /// Possui o header (tamanhos de cada tile) seguido dos bytes comprimidos sequenciais.
    pub raw_bytes: Vec<u8>,
}

pub struct TileInfo {
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub offset_in_stream: u64, // Onde o tile começa no raw_bytes (pós-header)
}

impl SerializedGDeflateStream {
    /// Cria a estrutura serializada agregando o resultado do `compressor.compress()`.
    pub fn from_compression_result(result: &OwnedCompressionResult) -> Self {
        let mut raw_bytes = Vec::new();

        // Número de tiles (u32)
        let num_tiles = result.tiles.len() as u32;
        raw_bytes.write_u32::<LittleEndian>(num_tiles).unwrap();
        
        // Tamanho do tile (u32)
        raw_bytes.write_u32::<LittleEndian>(result.tile_size).unwrap();

        // Para cada tile, gravar seu tamanho comprimido (u32) e descomprimido (u32)
        for tile in &result.tiles {
            raw_bytes.write_u32::<LittleEndian>(tile.compressed_size).unwrap();
            raw_bytes.write_u32::<LittleEndian>(tile.uncompressed_size).unwrap();
        }

        // Fazer append do bitstream GDeflate real (todos os tiles concatenados)
        raw_bytes.write_all(&result.bytes).unwrap();

        Self { raw_bytes }
    }

    /// Faz o parse do cabeçalho de um stream carregado na memória (ex: via Mmap)
    pub fn parse_header(data: &[u8]) -> Result<(Vec<TileInfo>, u32, usize), NodeStorError> {
        if data.len() < 8 {
            return Err(GDeflateError::DecompressionFailed.into());
        }
        
        let mut cursor = Cursor::new(data);
        let num_tiles = cursor.read_u32::<LittleEndian>().unwrap();
        let tile_size = cursor.read_u32::<LittleEndian>().unwrap();

        if data.len() < 8 + (num_tiles as usize) * 8 {
            return Err(GDeflateError::DecompressionFailed.into());
        }

        let mut tiles = Vec::with_capacity(num_tiles as usize);
        let mut current_offset = 8 + (num_tiles as usize) * 8; // após o header

        for _ in 0..num_tiles {
            let comp_size = cursor.read_u32::<LittleEndian>().unwrap();
            let uncomp_size = cursor.read_u32::<LittleEndian>().unwrap();
            
            tiles.push(TileInfo {
                compressed_size: comp_size,
                uncompressed_size: uncomp_size,
                offset_in_stream: current_offset as u64,
            });
            
            current_offset += comp_size as usize;
        }

        let header_bytes = 8 + (num_tiles as usize) * 8;
        Ok((tiles, tile_size, header_bytes))
    }
}
