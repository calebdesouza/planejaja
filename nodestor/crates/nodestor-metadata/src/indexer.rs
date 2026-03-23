//! Indexador de metadados de tensores do modelo local.
//!
//! Permite à engine localizar com latência zero o byte-offset exato de qualquer
//! bloco do modelo dentro do disco rígido. Necessário para a Metralhadora (Streaming).

use nodestor_core::{ModelMetadata, TensorInfo};
use std::collections::HashMap;

/// Indexador O(1) para mapas de tensores carregados via GGUF ou Safetensors.
pub struct TensorIndexer {
    /// Lista linear ordenada (mantém a ordem original do arquivo)
    pub ordered_tensors: Vec<TensorInfo>,
    /// Mapa de hash para lookup imediato pelo nome da variável
    name_index: HashMap<String, usize>,
    /// Tamanho total contabilizado em bytes
    pub total_bytes: u64,
}

impl TensorIndexer {
    pub fn new() -> Self {
        Self {
            ordered_tensors: Vec::new(),
            name_index: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Popula o indexador diretamente do metadata parseado.
    pub fn build_from(&mut self, metadata: &ModelMetadata) {
        self.ordered_tensors = metadata.tensors.clone();
        self.name_index.clear();
        self.total_bytes = 0;

        for (i, tensor) in self.ordered_tensors.iter().enumerate() {
            self.name_index.insert(tensor.name.clone(), i);
            self.total_bytes += tensor.data_size;
        }
        
        tracing::info!(
            "TensorIndexer montado: {} tensores iteráveis, {} MB mapeados",
            self.ordered_tensors.len(),
            self.total_bytes / 1024 / 1024
        );
    }

    /// Busca um tensor pelo seu nome exato em tempo O(1).
    pub fn get_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.name_index.get(name).map(|&idx| &self.ordered_tensors[idx])
    }

    /// Retorna uma sub-lista de tensores cujo nome contenha o padrão.
    /// Ex: `find_by_pattern("attn_k")`
    pub fn find_by_pattern(&self, pattern: &str) -> Vec<&TensorInfo> {
        self.ordered_tensors
            .iter()
            .filter(|t| t.name.contains(pattern))
            .collect()
    }
}

impl Default for TensorIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::{ModelFormat, TensorDtype};

    #[test]
    fn test_indexer_build_and_search() {
        let t1 = TensorInfo {
            name: "layer1.weight".into(),
            shape: vec![128, 128],
            dtype: TensorDtype::F16,
            data_offset: 0,
            data_size: 32768,
        };
        let t2 = TensorInfo {
            name: "layer1.bias".into(),
            shape: vec![128],
            dtype: TensorDtype::F32,
            data_offset: 32768,
            data_size: 512,
        };

        let metadata = ModelMetadata {
            format: ModelFormat::Safetensors,
            model_name: None,
            architecture: None,
            param_count: Some(16512),
            tensors: vec![t1, t2],
            data_offset: 0,
            file_size: 33280,
            extra: serde_json::Value::Null,
        };

        let mut indexer = TensorIndexer::new();
        indexer.build_from(&metadata);

        assert_eq!(indexer.ordered_tensors.len(), 2);
        assert_eq!(indexer.total_bytes, 33280);
        
        let bias = indexer.get_by_name("layer1.bias").unwrap();
        assert_eq!(bias.data_size, 512);

        let by_pattern = indexer.find_by_pattern("weight");
        assert_eq!(by_pattern.len(), 1);
        assert_eq!(by_pattern[0].name, "layer1.weight");
    }
}
