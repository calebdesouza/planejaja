//! # Formato NSZ (NodeStor Zip) — Compressão Lossless Neural via TCA-TBE
//!
//! Formato de arquivo binário para tensores comprimidos sem perdas, baseado na
//! codificação Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE).
//!
//! ## Princípio Matemático
//!
//! Explora a concentração entrópica dos expoentes IEEE-754:
//! - Expoentes consomem ~2-3 bits de informação real (de 8 possíveis em BF16/FP32)
//! - Triple Bitmap codifica expoentes em formato fixo, branch-free na GPU
//! - Mantissas e sinais são preservados verbatim (100% lossless)
//!
//! ## Layout do Tile TCA-TBE
//!
//! Cada tile contém TILE_SIZE pesos (padrão: 512, alinhado a warps de 32/64):
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │ TileHeader (8 bytes)                             │
//! │   modal_exponent: u8                             │
//! │   num_residuals: u16                             │
//! │   original_dtype: u8 (0=FP32, 1=FP16, 2=BF16)   │
//! │   reserved: [u8; 4]                              │
//! ├──────────────────────────────────────────────────┤
//! │ Bitmap Match  (TILE_SIZE/8 bytes = 64 bytes)     │
//! │   bit=1 se E[i] == modal_exponent                │
//! ├──────────────────────────────────────────────────┤
//! │ Bitmap Near   (TILE_SIZE/8 bytes = 64 bytes)     │
//! │   bit=1 se |E[i] - modal| == 1                   │
//! ├──────────────────────────────────────────────────┤
//! │ Near Deltas   (num_near bytes, packed ±1)        │
//! │   Para cada bit=1 em Bitmap Near:                │
//! │     0 = modal-1, 1 = modal+1                     │
//! ├──────────────────────────────────────────────────┤
//! │ Residual Exponents (num_residuals bytes)         │
//! │   Expoente completo para os outliers             │
//! ├──────────────────────────────────────────────────┤
//! │ Mantissas (verbatim, TILE_SIZE * mbits / 8)      │
//! │   Copiadas bit-a-bit do original (lossless)      │
//! ├──────────────────────────────────────────────────┤
//! │ Sinais (TILE_SIZE / 8 bytes = 64 bytes)          │
//! │   1 bit por peso                                 │
//! ├──────────────────────────────────────────────────┤
//! │ Padding para alinhamento 64 bytes                │
//! └──────────────────────────────────────────────────┘
//! ```

use nodestor_core::NodeStorError;

/// Tamanho de um tile em número de pesos. 512 = 8 warps × 64 threads (AMD wavefront).
/// Também funciona perfeitamente com warps de 32 (NVIDIA: 16 warps).
pub const TILE_SIZE: usize = 512;

/// Alinhamento em bytes para coalescência de memória GPU.
pub const ALIGNMENT: usize = 64;

/// Magic number do formato NSZ: "NSZ1" em ASCII.
pub const NSZ_MAGIC: [u8; 4] = [b'N', b'S', b'Z', b'1'];

/// Versão atual do formato.
pub const NSZ_VERSION: u16 = 1;

/// Tipo de dado original do tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OriginalDtype {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
}

impl OriginalDtype {
    pub fn bits_per_weight(&self) -> u32 {
        match self {
            OriginalDtype::FP32 => 32,
            OriginalDtype::FP16 => 16,
            OriginalDtype::BF16 => 16,
        }
    }

    pub fn mantissa_bits(&self) -> u32 {
        match self {
            OriginalDtype::FP32 => 23,
            OriginalDtype::FP16 => 10,
            OriginalDtype::BF16 => 7,
        }
    }

    pub fn exponent_bits(&self) -> u32 {
        match self {
            OriginalDtype::FP32 => 8,
            OriginalDtype::FP16 => 5,
            OriginalDtype::BF16 => 8,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::FP32),
            1 => Some(Self::FP16),
            2 => Some(Self::BF16),
            _ => None,
        }
    }
}

/// Header principal do arquivo NSZ.
#[derive(Debug, Clone)]
pub struct NszFileHeader {
    /// Magic: "NSZ1"
    pub magic: [u8; 4],
    /// Versão do formato
    pub version: u16,
    /// Flags: bits reservados
    pub flags: u16,
    /// Número de tensores no arquivo
    pub num_tensors: u32,
    /// Tamanho total original (descomprimido) em bytes
    pub original_total_bytes: u64,
    /// Tamanho total comprimido em bytes
    pub compressed_total_bytes: u64,
}

impl NszFileHeader {
    pub const SIZE: usize = 4 + 2 + 2 + 4 + 8 + 8; // 28 bytes

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.flags.to_le_bytes());
        buf.extend_from_slice(&self.num_tensors.to_le_bytes());
        buf.extend_from_slice(&self.original_total_bytes.to_le_bytes());
        buf.extend_from_slice(&self.compressed_total_bytes.to_le_bytes());
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, NodeStorError> {
        if data.len() < Self::SIZE {
            return Err(NodeStorError::InvalidModelFormat("NSZ header too short".into()));
        }
        let magic = [data[0], data[1], data[2], data[3]];
        if magic != NSZ_MAGIC {
            return Err(NodeStorError::InvalidModelFormat(
                format!("Invalid NSZ magic: {:?}", magic),
            ));
        }
        Ok(Self {
            magic,
            version: u16::from_le_bytes([data[4], data[5]]),
            flags: u16::from_le_bytes([data[6], data[7]]),
            num_tensors: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            original_total_bytes: u64::from_le_bytes(data[12..20].try_into().unwrap()),
            compressed_total_bytes: u64::from_le_bytes(data[20..28].try_into().unwrap()),
        })
    }
}

/// Entrada no índice de tensores do arquivo NSZ.
#[derive(Debug, Clone)]
pub struct NszTensorEntry {
    /// Nome do tensor (max 64 bytes, zero-padded)
    pub name: String,
    /// Tipo de dado original
    pub dtype: OriginalDtype,
    /// Tamanho original em bytes
    pub original_size: u64,
    /// Offset do primeiro tile comprimido no arquivo
    pub compressed_offset: u64,
    /// Tamanho comprimido total em bytes
    pub compressed_size: u64,
    /// Número de tiles
    pub num_tiles: u32,
    /// Número total de pesos (floats/halfs)
    pub num_weights: u64,
    /// Shape original [dim0, dim1, dim2, dim3] (0 para dims não usadas)
    pub shape: [u32; 4],
}

impl NszTensorEntry {
    pub const SIZE: usize = 64 + 1 + 8 + 8 + 8 + 4 + 8 + 16; // 117 bytes, pad to 128

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        // Nome: 64 bytes zero-padded
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len().min(64);
        buf.extend_from_slice(&name_bytes[..name_len]);
        buf.resize(64, 0u8); // pad
        buf.push(self.dtype as u8);
        buf.extend_from_slice(&self.original_size.to_le_bytes());
        buf.extend_from_slice(&self.compressed_offset.to_le_bytes());
        buf.extend_from_slice(&self.compressed_size.to_le_bytes());
        buf.extend_from_slice(&self.num_tiles.to_le_bytes());
        buf.extend_from_slice(&self.num_weights.to_le_bytes());
        for &d in &self.shape {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        // Pad to 128 bytes
        buf.resize(128, 0u8);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, NodeStorError> {
        if data.len() < 128 {
            return Err(NodeStorError::InvalidModelFormat("NSZ tensor entry too short".into()));
        }
        let name_end = data[..64].iter().position(|&b| b == 0).unwrap_or(64);
        let name = String::from_utf8_lossy(&data[..name_end]).to_string();
        let dtype = OriginalDtype::from_u8(data[64])
            .ok_or_else(|| NodeStorError::InvalidModelFormat("Invalid dtype".into()))?;
        Ok(Self {
            name,
            dtype,
            original_size: u64::from_le_bytes(data[65..73].try_into().unwrap()),
            compressed_offset: u64::from_le_bytes(data[73..81].try_into().unwrap()),
            compressed_size: u64::from_le_bytes(data[81..89].try_into().unwrap()),
            num_tiles: u32::from_le_bytes(data[89..93].try_into().unwrap()),
            num_weights: u64::from_le_bytes(data[93..101].try_into().unwrap()),
            shape: [
                u32::from_le_bytes(data[101..105].try_into().unwrap()),
                u32::from_le_bytes(data[105..109].try_into().unwrap()),
                u32::from_le_bytes(data[109..113].try_into().unwrap()),
                u32::from_le_bytes(data[113..117].try_into().unwrap()),
            ],
        })
    }
}

/// Header de um tile TCA-TBE individual.
#[derive(Debug, Clone, Copy)]
pub struct TileHeader {
    /// Expoente mais frequente neste tile
    pub modal_exponent: u8,
    /// Número de pesos cujo expoente não é modal nem near (outliers completos)
    pub num_residuals: u16,
    /// Tipo de dado original
    pub dtype: u8,
    /// Número de pesos "near" (expoente a ±1 do modal)
    pub num_near: u16,
    /// Reservado
    pub _reserved: [u8; 2],
}

impl TileHeader {
    pub const SIZE: usize = 8;

    pub fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0] = self.modal_exponent;
        buf[1..3].copy_from_slice(&self.num_residuals.to_le_bytes());
        buf[3] = self.dtype;
        buf[4..6].copy_from_slice(&self.num_near.to_le_bytes());
        buf[6..8].copy_from_slice(&self._reserved);
        buf
    }

    pub fn from_bytes(data: &[u8; 8]) -> Self {
        Self {
            modal_exponent: data[0],
            num_residuals: u16::from_le_bytes([data[1], data[2]]),
            dtype: data[3],
            num_near: u16::from_le_bytes([data[4], data[5]]),
            _reserved: [data[6], data[7]],
        }
    }
}

/// Um tile TCA-TBE completo com todos os seus dados.
#[derive(Debug, Clone)]
pub struct NszTile {
    pub header: TileHeader,
    /// Bitmap: bit=1 se expoente do peso == modal (TILE_SIZE/8 bytes)
    pub bitmap_match: Vec<u8>,
    /// Bitmap: bit=1 se |expoente - modal| == 1 (TILE_SIZE/8 bytes)
    pub bitmap_near: Vec<u8>,
    /// Deltas para os near: 0 = modal-1, 1 = modal+1 (packed bits)
    pub near_deltas: Vec<u8>,
    /// Expoentes completos dos outliers
    pub residual_exponents: Vec<u8>,
    /// Mantissas verbatim (lossless)
    pub mantissas: Vec<u8>,
    /// Sinais (1 bit por peso, packed)
    pub signs: Vec<u8>,
    /// Número real de pesos neste tile (pode ser < TILE_SIZE no último tile)
    pub actual_size: usize,
}

impl NszTile {
    /// Serializa o tile para bytes (formato em disco).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.header.to_bytes());
        buf.extend_from_slice(&self.bitmap_match);
        buf.extend_from_slice(&self.bitmap_near);
        buf.extend_from_slice(&self.near_deltas);
        buf.extend_from_slice(&self.residual_exponents);
        buf.extend_from_slice(&self.mantissas);
        buf.extend_from_slice(&self.signs);
        // Pad to ALIGNMENT
        let remainder = buf.len() % ALIGNMENT;
        if remainder != 0 {
            buf.resize(buf.len() + (ALIGNMENT - remainder), 0u8);
        }
        buf
    }

    /// Tamanho em disco após serialização (com padding).
    pub fn serialized_size(&self) -> usize {
        let raw = TileHeader::SIZE
            + self.bitmap_match.len()
            + self.bitmap_near.len()
            + self.near_deltas.len()
            + self.residual_exponents.len()
            + self.mantissas.len()
            + self.signs.len();
        // Alinha a 64 bytes
        let remainder = raw % ALIGNMENT;
        if remainder == 0 { raw } else { raw + ALIGNMENT - remainder }
    }
}

/// Alinha um valor para cima ao múltiplo de `align` mais próximo.
pub fn align_up(value: usize, align: usize) -> usize {
    let rem = value % align;
    if rem == 0 { value } else { value + align - rem }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsz_magic() {
        assert_eq!(&NSZ_MAGIC, b"NSZ1");
    }

    #[test]
    fn test_file_header_roundtrip() {
        let header = NszFileHeader {
            magic: NSZ_MAGIC,
            version: NSZ_VERSION,
            flags: 0,
            num_tensors: 21,
            original_total_bytes: 1_000_000,
            compressed_total_bytes: 730_000,
        };
        let bytes = header.to_bytes();
        let parsed = NszFileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.num_tensors, 21);
        assert_eq!(parsed.original_total_bytes, 1_000_000);
        assert_eq!(parsed.compressed_total_bytes, 730_000);
    }

    #[test]
    fn test_tensor_entry_roundtrip() {
        let entry = NszTensorEntry {
            name: "blk.0.attn_q.weight".into(),
            dtype: OriginalDtype::FP32,
            original_size: 16384,
            compressed_offset: 256,
            compressed_size: 12000,
            num_tiles: 32,
            num_weights: 4096,
            shape: [64, 64, 0, 0],
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), 128);
        let parsed = NszTensorEntry::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.name, "blk.0.attn_q.weight");
        assert_eq!(parsed.num_weights, 4096);
        assert_eq!(parsed.shape, [64, 64, 0, 0]);
    }

    #[test]
    fn test_tile_header_roundtrip() {
        let h = TileHeader {
            modal_exponent: 120,
            num_residuals: 15,
            dtype: 0,
            num_near: 87,
            _reserved: [0, 0],
        };
        let bytes = h.to_bytes();
        let parsed = TileHeader::from_bytes(&bytes);
        assert_eq!(parsed.modal_exponent, 120);
        assert_eq!(parsed.num_residuals, 15);
        assert_eq!(parsed.num_near, 87);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(128, 64), 128);
    }

    #[test]
    fn test_dtype_properties() {
        assert_eq!(OriginalDtype::FP32.mantissa_bits(), 23);
        assert_eq!(OriginalDtype::FP16.mantissa_bits(), 10);
        assert_eq!(OriginalDtype::BF16.mantissa_bits(), 7);
        assert_eq!(OriginalDtype::FP32.exponent_bits(), 8);
        assert_eq!(OriginalDtype::FP16.exponent_bits(), 5);
    }
}
