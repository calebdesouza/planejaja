use std::collections::HashMap;
use std::path::Path;
use nodestor_core::NodeStorError;

pub struct LoraAdapter {
    pub name: String,
    pub alpha: f32,
    pub scaling: f32,
    /// Tensor Name -> (A matrix, B matrix)
    pub weights: HashMap<String, LoraWeight>,
}

pub struct LoraWeight {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraAdapter {
    pub fn load<P: AsRef<Path>>(path: P, alpha: f32, rank: usize) -> Result<Self, NodeStorError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        // Simulação do carregamento via nodestor_formats::safetensors
        // Na prática, leríamos `adapter_model.safetensors` do disco
        
        let mut weights = HashMap::new();
        
        // Exemplo: Simulação para um tensor k_proj
        weights.insert("attn.k_proj".into(), LoraWeight {
            a: vec![0.01; rank * 4096],     // 4096 in_features -> rank
            b: vec![-0.01; 4096 * rank],    // rank -> 4096 out_features
            rank,
            in_features: 4096,
            out_features: 4096,
        });

        Ok(Self {
            name: path_str,
            alpha,
            scaling: alpha / (rank as f32),
            weights,
        })
    }

    /// W_new = W_base + (B * A) * scaling
    pub fn apply_to_f32(&self, tensor_name: &str, base_weight: &mut [f32]) -> bool {
        if let Some(lora) = self.weights.get(tensor_name) {
            let scale = self.scaling;
            let m = lora.out_features;
            let k = lora.rank;
            let n = lora.in_features;

            // B * A
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for r in 0..k {
                        sum += lora.b[i * k + r] * lora.a[r * n + j];
                    }
                    base_weight[i * n + j] += sum * scale;
                }
            }
            true
        } else {
            false
        }
    }
}
