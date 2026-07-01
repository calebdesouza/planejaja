use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use nodestor_core::{
    ModelFormat, ModelMetadata, ModelParser, NodeStorError, TensorDtype, TensorInfo,
};
use tracing::{debug, info};

/// Magic number do formato GGUF.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" em little-endian

/// Suporte a GGUF versão 2 e 3.
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

/// Parser completo para o formato GGUF.
///
/// Referência da especificação:
/// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
pub struct GgufParser;

impl GgufParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GgufParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelParser for GgufParser {
    fn parse(&self, path: &str) -> Result<ModelMetadata, NodeStorError> {
        info!("Parsing GGUF: {}", path);

        let file_size = std::fs::metadata(path)
            .map_err(|_| NodeStorError::ModelNotFound(path.to_string()))?
            .len();

        let file = std::fs::File::open(path)
            .map_err(|_| NodeStorError::ModelNotFound(path.to_string()))?;

        let mut reader = std::io::BufReader::new(file);
        parse_gguf_inner(&mut reader, path, file_size)
    }

    fn can_parse(&self, path: &str) -> bool {
        path.to_lowercase().ends_with(".gguf")
    }

    fn format_name(&self) -> &'static str {
        "GGUF"
    }
}

fn parse_gguf_inner<R: Read + Seek>(
    reader: &mut R,
    _path: &str,
    file_size: u64,
) -> Result<ModelMetadata, NodeStorError> {
    // ── Magic ───────────────────────────────────────────
    let magic = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| NodeStorError::InvalidModelFormat(format!("Não foi possível ler magic: {e}")))?;

    if magic != GGUF_MAGIC {
        return Err(NodeStorError::InvalidModelFormat(format!(
            "Magic inválido: 0x{:08X} (esperado 0x{:08X})",
            magic, GGUF_MAGIC
        )));
    }

    // ── Version ─────────────────────────────────────────
    let version = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| NodeStorError::InvalidModelFormat(format!("Não foi possível ler versão: {e}")))?;

    if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
        return Err(NodeStorError::InvalidModelFormat(format!(
            "Versão GGUF {} não suportada (suportado: {}-{})",
            version, GGUF_VERSION_MIN, GGUF_VERSION_MAX
        )));
    }

    debug!("GGUF version: {}", version);

    // ── Tensor count e metadata count ───────────────────
    let tensor_count = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| NodeStorError::InvalidModelFormat(e.to_string()))?;

    let metadata_kv_count = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| NodeStorError::InvalidModelFormat(e.to_string()))?;

    debug!("Tensores: {}, Metadata KVs: {}", tensor_count, metadata_kv_count);

    // ── Metadata KV pairs ───────────────────────────────
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    for _ in 0..metadata_kv_count {
        if let Ok((key, val)) = read_kv_pair(reader) {
            metadata.insert(key, val);
        }
    }

    let model_name = metadata.get("general.name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let architecture = metadata.get("general.architecture")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // ── Tensor infos ─────────────────────────────────────
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for i in 0..tensor_count {
        let tensor = read_tensor_info(reader, i)
            .map_err(|e| NodeStorError::InvalidModelFormat(
                format!("Erro lendo tensor {}: {}", i, e)
            ))?;
        tensors.push(tensor);
    }

    // O offset dos dados começa alinhado em 32 bytes após o header
    let current_pos = reader
        .seek(SeekFrom::Current(0))
        .map_err(|e| NodeStorError::IoError(e))?;
    let data_offset = align_to(current_pos, 32);

    // Ajusta os offsets dos tensores (relativos ao data_offset)
    let tensors: Vec<TensorInfo> = tensors.into_iter().map(|mut t| {
        t.data_offset += data_offset;
        t
    }).collect();

    let param_count: Option<u64> = if !tensors.is_empty() {
        Some(tensors.iter().map(|t| t.num_elements()).sum())
    } else {
        None
    };

    info!(
        "GGUF parseado: {} tensores, modelo: {:?}, arquitetura: {:?}",
        tensors.len(), model_name, architecture
    );

    Ok(ModelMetadata {
        format: ModelFormat::Gguf,
        model_name,
        architecture,
        param_count,
        tensors,
        data_offset,
        file_size,
        extra: serde_json::Value::Object(
            metadata.into_iter().collect()
        ),
    })
}

fn read_gguf_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    if len > 1_000_000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("String GGUF muito longa: {} bytes", len),
        ));
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_kv_pair<R: Read>(reader: &mut R) -> std::io::Result<(String, serde_json::Value)> {
    let key = read_gguf_string(reader)?;
    let value_type = reader.read_u32::<LittleEndian>()?;

    let value = read_gguf_value(reader, value_type)?;
    Ok((key, value))
}

fn read_gguf_value<R: Read>(reader: &mut R, value_type: u32) -> std::io::Result<serde_json::Value> {
    // GGUF value types:
    // 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
    // 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
    match value_type {
        0 => Ok(serde_json::Value::Number(reader.read_u8()?.into())),
        1 => Ok(serde_json::Value::Number(reader.read_i8()?.into())),
        2 => Ok(serde_json::Value::Number(reader.read_u16::<LittleEndian>()?.into())),
        3 => Ok(serde_json::Value::Number(reader.read_i16::<LittleEndian>()?.into())),
        4 => Ok(serde_json::Value::Number(reader.read_u32::<LittleEndian>()?.into())),
        5 => Ok(serde_json::Value::Number(reader.read_i32::<LittleEndian>()?.into())),
        6 => {
            let v = reader.read_f32::<LittleEndian>()?;
            Ok(serde_json::json!(v))
        }
        7 => {
            let b = reader.read_u8()?;
            Ok(serde_json::Value::Bool(b != 0))
        }
        8 => {
            let s = read_gguf_string(reader)?;
            Ok(serde_json::Value::String(s))
        }
        9 => {
            // Array: type (u32) + count (u64) + items
            let arr_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            // Limite de segurança ALTO: o vocab do tokenizer (tokenizer.ggml.tokens)
            // pode ter 256k+ itens; truncar quebra a tokenização. Cobrimos vocabs
            // reais sem materializar arrays patologicamente gigantes.
            let cap = count.min(2_000_000) as usize;
            let mut arr = Vec::with_capacity(cap.min(65_536));
            for _ in 0..cap {
                arr.push(read_gguf_value(reader, arr_type)?);
            }
            for _ in (cap as u64)..count {
                let _ = read_gguf_value(reader, arr_type)?;
            }
            Ok(serde_json::Value::Array(arr))
        }
        10 => Ok(serde_json::json!(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(serde_json::json!(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(serde_json::json!(reader.read_f64::<LittleEndian>()?)),
        unknown => {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tipo GGUF desconhecido: {}", unknown),
            ))
        }
    }
}

fn read_tensor_info<R: Read>(reader: &mut R, index: u64) -> std::io::Result<TensorInfo> {
    let name = read_gguf_string(reader)
        .map_err(|e| std::io::Error::new(e.kind(), format!("Tensor {}: nome inválido: {}", index, e)))?;

    let n_dims = reader.read_u32::<LittleEndian>()?;
    if n_dims > 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Tensor '{}': número de dimensões inválido: {}", name, n_dims),
        ));
    }

    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    let dtype_id = reader.read_u32::<LittleEndian>()?;
    let dtype = ggml_dtype(dtype_id);

    let data_offset = reader.read_u64::<LittleEndian>()?;

    // Calcula tamanho baseado nas dimensões e tipo
    let num_elements: u64 = shape.iter().product();
    let data_size = calculate_tensor_size(num_elements, dtype_id);

    Ok(TensorInfo {
        name,
        shape,
        dtype,
        data_offset,
        data_size,
    })
}

fn ggml_dtype(dtype_id: u32) -> TensorDtype {
    match dtype_id {
        0  => TensorDtype::F32,
        1  => TensorDtype::F16,
        2  => TensorDtype::Q4_0,
        3  => TensorDtype::Q4_1,
        6  => TensorDtype::Q5_0,
        7  => TensorDtype::Q5_1,
        8  => TensorDtype::Q8_0,
        9  => TensorDtype::Q8_1,   // Q8_1: blocos 36 bytes (d FP16 + s FP16 + 32×i8)
        10 => TensorDtype::F32,    // Q2_K — fallback F32 (não implementado)
        11 => TensorDtype::F32,    // Q3_K — fallback F32 (não implementado)
        12 => TensorDtype::Q4K,    // Q4_K — implementado em dequant/q4_k.rs
        13 => TensorDtype::Q5K,    // Q5_K — implementado em dequant/q5_k.rs
        14 => TensorDtype::Q6K,    // Q6_K — implementado em dequant/q6_k.rs
        15 => TensorDtype::F32,    // Q8_K — fallback F32 (não implementado)
        16 => TensorDtype::I8,
        17 => TensorDtype::I16,
        18 => TensorDtype::I32,
        30 => TensorDtype::BF16,
        31 => TensorDtype::I64,
        _  => TensorDtype::F32, // Fallback para tipos desconhecidos
    }
}

fn calculate_tensor_size(num_elements: u64, dtype_id: u32) -> u64 {
    match dtype_id {
        0 => num_elements * 4,          // F32
        1 => num_elements * 2,          // F16
        30 => num_elements * 2,         // BF16
        31 => num_elements * 8,         // I64
        16 => num_elements,             // I8
        17 => num_elements * 2,         // I16
        18 => num_elements * 4,         // I32
        2 => (num_elements + 31) / 32 * 18,       // Q4_0:  18 bytes/bloco de 32
        3 => (num_elements + 31) / 32 * 20,       // Q4_1:  20 bytes/bloco
        6 => (num_elements + 31) / 32 * 22,       // Q5_0:  22 bytes/bloco
        7 => (num_elements + 31) / 32 * 24,       // Q5_1:  24 bytes/bloco
        8  => (num_elements + 31) / 32 * 34,      // Q8_0:  34 bytes/bloco (2 FP16 + 32×i8)
        9  => (num_elements * 36 + 31) / 32,      // Q8_1: 36 bytes/32 pesos (d+s FP16 + 32×i8)
        12 => (num_elements * 144 + 255) / 256,   // Q4_K: 144 bytes/256 pesos
        13 => (num_elements * 176 + 255) / 256,   // Q5_K: 176 bytes/256 pesos
        14 => (num_elements * 210 + 255) / 256,   // Q6_K: 210 bytes/256 pesos (não dequantizado)
        15 => (num_elements * 9 + 7) / 8,         // Q8_K: ~Q8_0 size (fallback)
        _  => num_elements * 4,                   // Fallback: assume F32
    }
}

fn align_to(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cria um arquivo GGUF mínimo válido para testes.
    fn create_test_gguf(path: &str) {
        use byteorder::WriteBytesExt;
        use std::io::Write;

        let mut f = std::fs::File::create(path).unwrap();

        // Magic "GGUF" = 0x46554747
        f.write_u32::<LittleEndian>(0x46554747).unwrap();
        // Version 3
        f.write_u32::<LittleEndian>(3).unwrap();
        // 2 tensors
        f.write_u64::<LittleEndian>(2).unwrap();
        // 3 metadata KVs
        f.write_u64::<LittleEndian>(3).unwrap();

        // KV 1: general.architecture = "test_model"
        write_gguf_string(&mut f, "general.architecture");
        f.write_u32::<LittleEndian>(8).unwrap(); // STRING type
        write_gguf_string(&mut f, "test_model");

        // KV 2: general.name = "Test Model"
        write_gguf_string(&mut f, "general.name");
        f.write_u32::<LittleEndian>(8).unwrap();
        write_gguf_string(&mut f, "Test Model");

        // KV 3: llm.context_length = 4096 (UINT32)
        write_gguf_string(&mut f, "llm.context_length");
        f.write_u32::<LittleEndian>(4).unwrap(); // UINT32 type
        f.write_u32::<LittleEndian>(4096).unwrap();

        // Tensor 1: "token.embd" — shape [100, 64] — F32
        write_gguf_string(&mut f, "token.embd");
        f.write_u32::<LittleEndian>(2).unwrap(); // 2 dimensions
        f.write_u64::<LittleEndian>(100).unwrap();
        f.write_u64::<LittleEndian>(64).unwrap();
        f.write_u32::<LittleEndian>(0).unwrap(); // F32
        f.write_u64::<LittleEndian>(0).unwrap(); // offset 0

        // Tensor 2: "output.weight" — shape [64] — F16
        write_gguf_string(&mut f, "output.weight");
        f.write_u32::<LittleEndian>(1).unwrap(); // 1 dimension
        f.write_u64::<LittleEndian>(64).unwrap();
        f.write_u32::<LittleEndian>(1).unwrap(); // F16
        f.write_u64::<LittleEndian>(25600).unwrap(); // offset = 100*64*4 bytes

        // Dados fictícios dos tensores (zeros)
        let tensor_data = vec![0u8; 26000];
        f.write_all(&tensor_data).unwrap();
    }

    fn write_gguf_string<W: std::io::Write>(w: &mut W, s: &str) {
        use byteorder::WriteBytesExt;
        w.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }

    #[test]
    fn test_parse_valid_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        let path_str = path.to_str().unwrap();
        create_test_gguf(path_str);

        let parser = GgufParser::new();
        let meta = parser.parse(path_str).expect("GGUF deve ser parseado com sucesso");

        assert_eq!(meta.format, ModelFormat::Gguf);
        assert_eq!(meta.model_name.as_deref(), Some("Test Model"));
        assert_eq!(meta.architecture.as_deref(), Some("test_model"));
        assert_eq!(meta.tensor_count(), 2);
        assert_eq!(meta.tensors[0].name, "token.embd");
        assert_eq!(meta.tensors[0].shape, vec![100, 64]);
        assert_eq!(meta.tensors[0].dtype, TensorDtype::F32);
        assert_eq!(meta.tensors[1].name, "output.weight");
        assert_eq!(meta.tensors[1].dtype, TensorDtype::F16);
    }

    #[test]
    fn test_parse_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"NOTGGUF!").unwrap();

        let parser = GgufParser::new();
        let result = parser.parse(path.to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Magic inválido") || err.contains("magic"), "Erro deve mencionar 'magic': {}", err);
    }

    #[test]
    fn test_can_parse_extensions() {
        let parser = GgufParser::new();
        assert!(parser.can_parse("model.gguf"));
        assert!(parser.can_parse("path/to/llama-7b-q4.gguf"));
        assert!(!parser.can_parse("model.safetensors"));
        assert!(!parser.can_parse("model.bin"));
    }

    #[test]
    fn test_tensor_num_elements() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        create_test_gguf(path.to_str().unwrap());

        let parser = GgufParser::new();
        let meta = parser.parse(path.to_str().unwrap()).unwrap();

        // Tensor 1: shape [100, 64] = 6400 elementos
        assert_eq!(meta.tensors[0].num_elements(), 6400);
        // Tensor 2: shape [64] = 64 elementos
        assert_eq!(meta.tensors[1].num_elements(), 64);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 32), 0);
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(33, 32), 64);
        assert_eq!(align_to(64, 32), 64);
        assert_eq!(align_to(65, 32), 96);
    }
}
