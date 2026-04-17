use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::{Mmap, MmapOptions};
use nodestor_core::{ModelMetadata, NodeStorError, TensorInfo, ModelParser};
use nodestor_formats::GgufParser;
use nodestor_vulkan::{VulkanEngine, GpuBuffer};
use tracing::info;

/// Gerencia o mapeamento de memória zero-copy do arquivo GGUF inteiro.
/// Expõe fatias de memória alinhadas para envio direto ao DMA da GPU.
pub struct WeightStore {
    mmap: Mmap,
    metadata: ModelMetadata,
    tensor_index: HashMap<String, TensorInfo>,
}

impl WeightStore {
    /// Abre o modelo GGUF usando mapeamento de memória.
    pub fn open(path: &Path) -> Result<Self, NodeStorError> {
        let path_str = path.to_str().ok_or_else(|| {
            NodeStorError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Caminho não é UTF-8 válido",
            ))
        })?;

        // Usa o GgufParser para ler o cabeçalho e pegar a metadata
        let parser = GgufParser::new();
        if !parser.can_parse(path_str) {
            return Err(NodeStorError::InvalidModelFormat(
                "Arquivo não é GGUF ou não é suportado pelo parser".into(),
            ));
        }

        let metadata = parser.parse(path_str)?;

        // Mapeia o arquivo inteiro em memória
        let file = File::open(path).map_err(NodeStorError::IoError)?;
        
        // Mmap pode falhar se arquivo vazio (tratado via struct metadata)
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(NodeStorError::IoError)? };

        // Indexar por nome O(1) fetch
        let mut tensor_index = HashMap::new();
        for t in &metadata.tensors {
            tensor_index.insert(t.name.clone(), t.clone());
        }

        Ok(Self {
            mmap,
            metadata,
            tensor_index,
        })
    }

    /// Retorna metadata do modelo lido.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Obtém `TensorInfo` para o nome dado.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name)
    }

    /// Retorna lista de nomes de tensores disponíveis.
    pub fn list_tensors(&self) -> Vec<&str> {
        self.tensor_index.keys().map(|k| k.as_str()).collect()
    }

    /// Busca o slice contendo os bytes crus do tensor, alinhado.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        let start = info.data_offset as usize;
        let end = start + info.data_size as usize;
        
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }
}

/// Gerencia os pesos de um modelo já carregados na VRAM da GPU.
pub struct ModelWeights {
    pub buffers: HashMap<String, GpuBuffer>,
}

impl ModelWeights {
    /// Faz o upload de todos os tensores do WeightStore para a GPU.
    pub fn from_weight_store(store: &WeightStore, engine: &VulkanEngine) -> Result<Self, NodeStorError> {
        let mut buffers = HashMap::new();
        let tensors = store.list_tensors();
        
        info!("Iniciando upload de {} tensores para a GPU...", tensors.len());
        
        for name in tensors {
            if let Some(bytes) = store.tensor_bytes(name) {
                // Upload para a GPU via VulkanEngine (Staging -> Device Local)
                let buffer = engine.upload(bytes)?;
                buffers.insert(name.to_string(), buffer);
            } else {
                return Err(NodeStorError::IoError(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("Tensor {} fora dos limites do arquivo", name),
                )));
            }
        }
        
        info!("Upload de pesos concluído com sucesso.");
        Ok(Self { buffers })
    }

    /// Retorna uma referência a um GpuBuffer específico na VRAM
    pub fn get(&self, name: &str) -> Option<&GpuBuffer> {
        self.buffers.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Write, Seek};
    use byteorder::{LittleEndian, WriteBytesExt};

    fn write_gguf_string<W: std::io::Write>(w: &mut W, s: &str) {
        w.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }

    fn create_test_gguf(path: &str) {
        let mut f = std::fs::File::create(path).unwrap();

        // Magic "GGUF" = 0x46554747
        f.write_u32::<LittleEndian>(0x46554747).unwrap();
        // Version 3
        f.write_u32::<LittleEndian>(3).unwrap();
        // 1 tensors
        f.write_u64::<LittleEndian>(1).unwrap();
        // 0 metadata KVs
        f.write_u64::<LittleEndian>(0).unwrap();

        // Tensor 1: "token.embd" — shape [100, 64] — F32
        write_gguf_string(&mut f, "token.embd");
        f.write_u32::<LittleEndian>(2).unwrap(); // 2 dimensions
        f.write_u64::<LittleEndian>(100).unwrap();
        f.write_u64::<LittleEndian>(64).unwrap();
        f.write_u32::<LittleEndian>(0).unwrap(); // F32
        f.write_u64::<LittleEndian>(0).unwrap(); // offset 0

        let current_pos = f.stream_position().unwrap();
        let target_pos = (current_pos + 31) & !31; // align to 32
        let padding = target_pos - current_pos;
        if padding > 0 {
            f.write_all(&vec![0u8; padding as usize]).unwrap();
        }
        
        let mut tensor_data = vec![0u8; 25600];
        tensor_data[0] = 0xAB;
        tensor_data[1] = 0xCD;
        f.write_all(&tensor_data).unwrap();
    }


    #[test]
    fn test_weight_store_opens_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_store.gguf");
        create_test_gguf(path.to_str().unwrap());

        let store = WeightStore::open(&path).expect("Abrir store");
        
        assert_eq!(store.list_tensors().len(), 1);
        
        let info = store.tensor_info("token.embd").unwrap();
        assert_eq!(info.dtype, nodestor_core::TensorDtype::F32);
        
        let slice = store.tensor_bytes("token.embd").unwrap();
        assert_eq!(slice.len(), 25600);
        assert_eq!(slice[0], 0xAB);
        assert_eq!(slice[1], 0xCD);
    }
}
