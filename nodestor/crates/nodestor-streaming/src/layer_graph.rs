use nodestor_core::{ModelMetadata, TensorInfo};

/// Referência a um tensor no disco.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

impl From<&TensorInfo> for TensorSlice {
    fn from(t: &TensorInfo) -> Self {
        Self {
            name: t.name.clone(),
            offset: t.data_offset,
            size: t.data_size,
        }
    }
}

/// Tipo do grupo lógico.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupType {
    Embedding,
    Attention,
    Mlp,
    Output,
    Other,
}

/// Grupo lógico de tensores que são consumidos juntos.
#[derive(Debug, Clone)]
pub struct TensorGroup {
    pub layer_idx: usize,
    pub group_type: GroupType,
    pub tensors: Vec<TensorSlice>,
    pub total_bytes: u64,
}

impl TensorGroup {
    pub fn new(layer_idx: usize, group_type: GroupType) -> Self {
        Self {
            layer_idx,
            group_type,
            tensors: Vec::new(),
            total_bytes: 0,
        }
    }

    pub fn add(&mut self, tensor: TensorSlice) {
        self.total_bytes += tensor.size;
        self.tensors.push(tensor);
    }
}

/// Ordem determinística de leitura dos blocos de tensores.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub groups: Vec<TensorGroup>,
    pub num_layers: usize,
    pub layer_byte_budget: u64,
    pub total_forward_pass_bytes: u64,
}

pub struct LayerGraph;

impl LayerGraph {
    /// Inspeciona o metadata e agrupa os tensores seguindo a topologia de um Transformer.
    /// Caso o modelo possua nomenclatura desconhecida (não-LLaMA/padrão), 
    /// fallback para grupo único linear.
    pub fn build(metadata: &ModelMetadata) -> ExecutionPlan {
        let mut embedding_group = TensorGroup::new(0, GroupType::Embedding);
        let mut output_group = TensorGroup::new(0, GroupType::Output);
        
        // Estrutura temporária: vetor de (Camada -> (Atenção, Mlp, Outros))
        let mut layers: std::collections::BTreeMap<usize, (TensorGroup, TensorGroup, TensorGroup)> = std::collections::BTreeMap::new();
        
        let mut total_bytes = 0;
        let mut has_recognized_pattern = false;

        for t in &metadata.tensors {
            let slice = TensorSlice::from(t);
            total_bytes += slice.size;

            let name_lc = t.name.to_lowercase();
            
            // Heurística de extração de camada (blk.X. ou layers.X.)
            let mut layer_idx_opt = None;
            let parts: Vec<&str> = name_lc.split('.').collect();
            
            for i in 0..parts.len() {
                if (parts[i] == "blk" || parts[i] == "layers") && i + 1 < parts.len() {
                    if let Ok(idx) = parts[i+1].parse::<usize>() {
                        layer_idx_opt = Some(idx);
                        has_recognized_pattern = true;
                        break;
                    }
                }
            }

            if let Some(idx) = layer_idx_opt {
                let entry = layers.entry(idx).or_insert_with(|| {
                    (
                        TensorGroup::new(idx, GroupType::Attention),
                        TensorGroup::new(idx, GroupType::Mlp),
                        TensorGroup::new(idx, GroupType::Other),
                    )
                });

                if name_lc.contains("attn") || name_lc.contains("self_attn") {
                    entry.0.add(slice);
                } else if name_lc.contains("ffn") || name_lc.contains("mlp") {
                    entry.1.add(slice);
                } else {
                    entry.2.add(slice);
                }
            } else {
                // Não possui identificador de camada. Provavelmente é Embed ou Output.
                if name_lc.contains("embd") || name_lc.contains("embed") || name_lc.contains("token") {
                    embedding_group.add(slice);
                } else if name_lc.contains("output") || name_lc.contains("head") || name_lc.contains("norm") {
                    output_group.add(slice);
                } else {
                    // Outro (cairá no embedding pra iniciar)
                    embedding_group.add(slice);
                }
            }
        }

        // Se o modelo é totalmente fora do padrão, agrupa tudo de forma linear em fallback
        if !has_recognized_pattern {
            let mut fallback_group = TensorGroup::new(0, GroupType::Other);
            for t in &metadata.tensors {
                fallback_group.add(TensorSlice::from(t));
            }
            return ExecutionPlan {
                groups: vec![fallback_group],
                num_layers: 1,
                layer_byte_budget: total_bytes,
                total_forward_pass_bytes: total_bytes,
            };
        }

        // Montagem ordenada do Plano de Execução (Grafo Causal)
        let mut groups = Vec::new();
        
        if !embedding_group.tensors.is_empty() {
            groups.push(embedding_group);
        }

        let num_layers = layers.len();
        
        for (_idx, (attn, mlp, other)) in layers.into_iter() {
            if !other.tensors.is_empty() {
                groups.push(other);
            }
            if !attn.tensors.is_empty() {
                groups.push(attn);
            }
            if !mlp.tensors.is_empty() {
                groups.push(mlp);
            }
        }

        if !output_group.tensors.is_empty() {
            groups.push(output_group);
        }

        let layer_byte_budget = if num_layers > 0 {
            total_bytes / num_layers as u64
        } else {
            total_bytes
        };

        ExecutionPlan {
            groups,
            num_layers,
            layer_byte_budget,
            total_forward_pass_bytes: total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::{ModelFormat, TensorDtype};

    fn tensor_info(name: &str, offset: u64, size: u64) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            shape: vec![],
            dtype: TensorDtype::F32,
            data_offset: offset,
            data_size: size,
        }
    }

    fn make_metadata(tensors: Vec<TensorInfo>) -> ModelMetadata {
        ModelMetadata {
            format: ModelFormat::Gguf,
            model_name: None,
            architecture: None,
            param_count: None,
            tensors,
            data_offset: 0,
            file_size: 0,
            extra: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_layer_graph_from_llama_names() {
        let tensors = vec![
            tensor_info("token_embd.weight", 0, 32768),
            tensor_info("blk.0.attn_k.weight", 32768, 16384),
            tensor_info("blk.0.attn_v.weight", 49152, 16384),
            tensor_info("blk.0.ffn_gate.weight", 65536, 32768),
            tensor_info("blk.1.attn_k.weight", 98304, 16384),
            tensor_info("blk.1.attn_v.weight", 114688, 16384),
            tensor_info("blk.1.ffn_gate.weight", 131072, 32768),
            tensor_info("output.weight", 163840, 32768),
        ];

        let plan = LayerGraph::build(&make_metadata(tensors));
        
        assert_eq!(plan.num_layers, 2);
        assert_eq!(plan.groups.len(), 6); // Embed + 2*(Attn+Mlp) + Output
        assert!(matches!(plan.groups[0].group_type, GroupType::Embedding));
        assert!(matches!(plan.groups[1].group_type, GroupType::Attention)); // blk 0 attn
        assert!(matches!(plan.groups[2].group_type, GroupType::Mlp));       // blk 0 mlp
        assert_eq!(plan.groups[1].tensors.len(), 2);
    }
}
