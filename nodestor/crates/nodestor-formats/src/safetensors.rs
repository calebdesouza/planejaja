use nodestor_core::{
    ModelFormat, ModelMetadata, ModelParser, NodeStorError, TensorDtype, TensorInfo,
};
use tracing::info;

/// Parser para o formato Safetensors (HuggingFace).
///
/// Especificação: https://huggingface.co/docs/safetensors
/// Estrutura: [8 bytes: header_len u64 LE] [header_len bytes: JSON] [dados binários]
pub struct SafetensorsParser;

impl SafetensorsParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SafetensorsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelParser for SafetensorsParser {
    fn parse(&self, path: &str) -> Result<ModelMetadata, NodeStorError> {
        info!("Parsing Safetensors: {}", path);

        let file_size = std::fs::metadata(path)
            .map_err(|_| NodeStorError::ModelNotFound(path.to_string()))?
            .len();

        let mut file = std::fs::File::open(path)
            .map_err(|_| NodeStorError::ModelNotFound(path.to_string()))?;

        // Lê os 8 bytes do tamanho do header
        let mut len_buf = [0u8; 8];
        use std::io::Read;
        file.read_exact(&mut len_buf)
            .map_err(|e| NodeStorError::InvalidModelFormat(
                format!("Não foi possível ler header length: {}", e)
            ))?;

        let header_len = u64::from_le_bytes(len_buf);

        if header_len > 100_000_000 {
            return Err(NodeStorError::InvalidModelFormat(
                format!("Header JSON muito grande: {} bytes", header_len)
            ));
        }

        // Lê o JSON do header
        let mut header_json = vec![0u8; header_len as usize];
        file.read_exact(&mut header_json)
            .map_err(|e| NodeStorError::InvalidModelFormat(
                format!("Não foi possível ler header JSON: {}", e)
            ))?;

        let data_offset = 8 + header_len;

        // Parseia o JSON
        let header: serde_json::Value = serde_json::from_slice(&header_json)
            .map_err(|e| NodeStorError::InvalidModelFormat(
                format!("Header JSON inválido: {}", e)
            ))?;

        let header_obj = header.as_object()
            .ok_or_else(|| NodeStorError::InvalidModelFormat(
                "Header JSON não é um objeto".to_string()
            ))?;

        let mut tensors = Vec::new();
        let mut model_metadata_extra = serde_json::Map::new();

        for (key, value) in header_obj {
            // A chave "__metadata__" contém informações do modelo
            if key == "__metadata__" {
                if let Some(obj) = value.as_object() {
                    model_metadata_extra.extend(obj.clone());
                }
                continue;
            }

            // Cada outro campo é um tensor
            let tensor = parse_safetensors_tensor(key, value, data_offset)?;
            tensors.push(tensor);
        }

        // Ordena tensores por offset para streaming sequencial eficiente
        tensors.sort_by_key(|t| t.data_offset);

        let model_name = model_metadata_extra.get("model_name")
            .or(model_metadata_extra.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let architecture = model_metadata_extra.get("model_type")
            .or(model_metadata_extra.get("architecture"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let param_count: Option<u64> = if !tensors.is_empty() {
            Some(tensors.iter().map(|t| t.num_elements()).sum())
        } else {
            None
        };

        info!(
            "Safetensors parseado: {} tensores",
            tensors.len()
        );

        Ok(ModelMetadata {
            format: ModelFormat::Safetensors,
            model_name,
            architecture,
            param_count,
            tensors,
            data_offset,
            file_size,
            extra: serde_json::Value::Object(model_metadata_extra),
        })
    }

    fn can_parse(&self, path: &str) -> bool {
        path.to_lowercase().ends_with(".safetensors")
    }

    fn format_name(&self) -> &'static str {
        "Safetensors"
    }
}

fn parse_safetensors_tensor(
    name: &str,
    value: &serde_json::Value,
    file_data_offset: u64,
) -> Result<TensorInfo, NodeStorError> {
    let obj = value.as_object()
        .ok_or_else(|| NodeStorError::InvalidModelFormat(
            format!("Tensor '{}' não é um objeto JSON", name)
        ))?;

    // dtype
    let dtype_str = obj.get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| NodeStorError::InvalidModelFormat(
            format!("Tensor '{}': dtype ausente", name)
        ))?;
    let dtype = parse_safetensors_dtype(dtype_str)?;

    // shape
    let shape_arr = obj.get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| NodeStorError::InvalidModelFormat(
            format!("Tensor '{}': shape ausente ou inválido", name)
        ))?;
    let shape: Vec<u64> = shape_arr.iter()
        .filter_map(|v| v.as_u64())
        .collect();

    // data_offsets [start, end]
    let offsets = obj.get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| NodeStorError::InvalidModelFormat(
            format!("Tensor '{}': data_offsets ausente", name)
        ))?;

    if offsets.len() < 2 {
        return Err(NodeStorError::InvalidModelFormat(
            format!("Tensor '{}': data_offsets deve ter 2 elementos", name)
        ));
    }

    let start = offsets[0].as_u64().unwrap_or(0);
    let end = offsets[1].as_u64().unwrap_or(0);
    let data_size = end.saturating_sub(start);

    Ok(TensorInfo {
        name: name.to_string(),
        shape,
        dtype,
        data_offset: file_data_offset + start, // Offset absoluto no arquivo
        data_size,
    })
}

fn parse_safetensors_dtype(s: &str) -> Result<TensorDtype, NodeStorError> {
    match s {
        "F32" | "float32" => Ok(TensorDtype::F32),
        "F16" | "float16" => Ok(TensorDtype::F16),
        "BF16" | "bfloat16" => Ok(TensorDtype::BF16),
        "I8" | "int8" => Ok(TensorDtype::I8),
        "I16" | "int16" => Ok(TensorDtype::I16),
        "I32" | "int32" => Ok(TensorDtype::I32),
        "I64" | "int64" => Ok(TensorDtype::I64),
        "F64" | "float64" => Ok(TensorDtype::F64),
        "BOOL" | "bool" => Ok(TensorDtype::Bool),
        unknown => Err(NodeStorError::InvalidModelFormat(
            format!("Dtype Safetensors desconhecido: '{}'", unknown)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_safetensors(path: &str) {
        let header = serde_json::json!({
            "__metadata__": {
                "model_name": "Test Model ST",
                "model_type": "test_architecture"
            },
            "embedding.weight": {
                "dtype": "F32",
                "shape": [100, 32],
                "data_offsets": [0, 12800]
            },
            "output.bias": {
                "dtype": "F16",
                "shape": [32],
                "data_offsets": [12800, 12864]
            }
        });

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut file = std::fs::File::create(path).unwrap();
        use std::io::Write;
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(&header_bytes).unwrap();
        // Dados fictícios
        file.write_all(&vec![0u8; 12864]).unwrap();
    }

    #[test]
    fn test_parse_valid_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let path_str = path.to_str().unwrap();
        create_test_safetensors(path_str);

        let parser = SafetensorsParser::new();
        let meta = parser.parse(path_str).expect("Deve parsear com sucesso");

        assert_eq!(meta.format, ModelFormat::Safetensors);
        assert_eq!(meta.model_name.as_deref(), Some("Test Model ST"));
        assert_eq!(meta.architecture.as_deref(), Some("test_architecture"));
        assert_eq!(meta.tensor_count(), 2);
    }

    #[test]
    fn test_tensor_offsets_are_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        create_test_safetensors(path.to_str().unwrap());

        let parser = SafetensorsParser::new();
        let meta = parser.parse(path.to_str().unwrap()).unwrap();

        // data_offset deve ser absoluto (data_offset + offset_relativo)
        // data_offset = 8 + len(header_json)
        let expected_base = meta.data_offset;
        assert!(expected_base > 8, "data_offset deve ser > 8 (header length field)");

        // O tensor deve ter offset >= data_offset
        for tensor in &meta.tensors {
            assert!(tensor.data_offset >= expected_base,
                "Tensor '{}' offset {} deve ser >= {}", tensor.name, tensor.data_offset, expected_base);
        }
    }

    #[test]
    fn test_can_parse_safetensors() {
        let parser = SafetensorsParser::new();
        assert!(parser.can_parse("model.safetensors"));
        assert!(!parser.can_parse("model.gguf"));
        assert!(!parser.can_parse("model.bin"));
    }
}
