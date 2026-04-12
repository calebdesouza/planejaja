use std::f32;

/// PROBES V2 — SAE Engine (Sparse Autoencoder)
///
/// Implementa a microscopia latente decompondo _hidden states_ residuais em 
/// representações de Features monosemânticas de altíssima dimensão 
/// usando ativação funcional JumpReLU para impor a esparsidade desejada.
///
/// Utilizado para inspecionar, em tempo pseudo-real, os conceitos acionados por
/// uma inferência ativa limitando o custo computacional com matrizes lineares leves.

pub struct SAEEngine {
    pub hidden_dim: usize,         // Dimensão do espaço latente do LLM (ex: 4096)
    pub dict_size: usize,          // Expansão do espaço de features (ex: 32768)
    pub encoder_weights: Vec<f32>, // Shape: (F x D)
    pub decoder_weights: Vec<f32>, // Shape: (D x F) usualmente atrelados (tied)
    pub encoder_bias: Vec<f32>,    // Shape: (F)
    pub threshold: f32,            // JumpReLU activation threshold
}

impl SAEEngine {
    /// Inicializa um Sparse Autoencoder com as devidas dimensões. 
    /// Em ambiente de produção o init importaria os Tensores pré-treinados 
    /// como os proveídos no Gemma Scope ou Llama SAE.
    pub fn new(hidden_dim: usize, dict_size: usize, threshold: f32) -> Self {
        Self {
            hidden_dim,
            dict_size,
            // Mock de pesos pré-treinados:
            encoder_weights: vec![0.01; hidden_dim * dict_size], 
            decoder_weights: vec![0.01; hidden_dim * dict_size],
            encoder_bias: vec![0.0; dict_size],
            threshold,
        }
    }

    /// Mapeia _Forward_: Espaço Latente → Espaço Monosemântico de Features.
    /// f = JumpReLU(W_enc * h + b_enc)
    pub fn encode(&self, hidden_states: &[f32]) -> Vec<f32> {
        assert_eq!(hidden_states.len(), self.hidden_dim, "Dimensão residual incompatível");
        let mut features = vec![0.0; self.dict_size];

        // Transição CPU-Bound iterativa (idealmente em blocos WGPU)
        for i in 0..self.dict_size {
            let mut dot = self.encoder_bias[i];
            for j in 0..self.hidden_dim {
                // Layout planar: ROW_MAJOR array -> row `i`, col `j`
                dot += hidden_states[j] * self.encoder_weights[i * self.hidden_dim + j];
            }
            
            // JumpReLU: Se ativado além da margem de ruído, mantem probabilidade causal 
            // Senão silêncio total, impondo _sparsity_ estanque.
            if dot > self.threshold {
                features[i] = dot;
            } else {
                features[i] = 0.0;
            }
        }
        features
    }

    /// Reconstrói as características latentes ativas de volta ao Residual Stream.
    /// Serve para monitorar erro de reconstrução e injetar Steering Vectors de alta dimensionalidade de volta as trilhas baixas da IA.
    /// h_hat = W_dec * f 
    pub fn decode(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), self.dict_size, "Dimensão de features incompatível");
        let mut reconstructed = vec![0.0; self.hidden_dim];

        // Otimização basica: Apenas roda dot para features não zeradas (graças a Sparsity)
        let active_features: Vec<(usize, f32)> = features
            .iter()
            .enumerate()
            .filter(|(_, &f)| f > 0.0)
            .map(|(i, &f)| (i, f))
            .collect();

        for i in 0..self.hidden_dim {
            let mut dot = 0.0;
            for &(j, val) in &active_features {
                dot += val * self.decoder_weights[i * self.dict_size + j];
            }
            reconstructed[i] = dot;
        }

        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sae_engine_sparsity_and_reconstruction() {
        let h_dim = 128;
        let dict = 1024;
        let threshold = 0.5; // JumpReLU corta as baixas

        let mut sae = SAEEngine::new(h_dim, dict, threshold);

        // Simulamos pesos para ter pelo menos UMA feature reativa
        sae.encoder_weights[5 * h_dim + 10] = 10.0; // W_enc[5, 10] = 10.0
        sae.decoder_weights[10 * dict + 5] = 1.0;   // W_dec[10, 5] = 1.0

        let mut h = vec![0.0; h_dim];
        h[10] = 1.0; // Input artificial onde o componente isolado garante o firing

        let latents = sae.encode(&h);
        
        // Assegurar esparsidade: maioria esmagadora deve estar zerada
        let non_zeros = latents.iter().filter(|&&v| v > 0.0).count();
        assert!(non_zeros < dict / 10, "A esparsidade falhou: muitos latentes ativados.");
        assert_eq!(latents[5], 10.0, "O SAE não recuperou o feature esperado da ativação forte.");

        let h_rcns = sae.decode(&latents);
        
        // Como MOCK_DEC recupera para 10.0 * 1.0, a coordenada em dim-10 ressurge.
        assert!(h_rcns[10] > 0.0, "O resíduo original apagou na reconstrução de volta ao pipeline.");
    }
}
