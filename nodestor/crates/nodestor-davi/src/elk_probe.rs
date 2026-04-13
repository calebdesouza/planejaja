use nodestor_inference::sae_engine::SAEEngine;

/// PROBES V2 — ELK Probe (Eliciting Latent Knowledge)
///
/// "O Polígrafo do Davi"
/// Uma sonda de regressão linear treinada para detectar se as representações internas
/// do modelo estão alinhadas com a "Verdade Oculta" ou se ele está mentindo/dissimulando.
/// 
/// O Probe é calibrado utilizando Contrast Pairs: prompts em que sabemos que a IA
/// raciocinou corretamente vs prompts onde a forçamos (ou deixamos) alucinar/mentir.

pub struct ElkProbe {
    /// Pesos da regressão para cada feature extraída (Dimensão = SAE dict_size)
    pub weights: Vec<f32>,
    pub bias: f32,
    /// Limiar acima do qual assumimos que a IA está sendo honesta.
    pub honest_threshold: f32,
}

impl ElkProbe {
    pub fn new(dict_size: usize) -> Self {
        Self {
            weights: vec![0.0; dict_size],
            bias: 0.0,
            honest_threshold: 0.5, // 50% sigmoid
        }
    }

    /// Treina a sonda utilizando gradiente descendente simples sobre pares de contraste.
    /// Para produção industrial, este método seria executado em batch GPU no MLOps pipeline.
    pub fn train_contrast_pairs(
        &mut self, 
        honest_samples: &[&[f32]], 
        deceptive_samples: &[&[f32]],
        learning_rate: f32,
        epochs: usize,
    ) {
        if honest_samples.is_empty() || deceptive_samples.is_empty() { return; }

        for _ in 0..epochs {
            // Treino Honesto (Target = 1.0)
            for features in honest_samples {
                let pred = self.evaluate_sigmoid(features);
                let error = 1.0 - pred;
                self.update_weights(features, error, learning_rate);
            }

            // Treino Desonesto (Target = 0.0)
            for features in deceptive_samples {
                let pred = self.evaluate_sigmoid(features);
                let error = 0.0 - pred; // Negativo
                self.update_weights(features, error, learning_rate);
            }
        }
    }

    fn update_weights(&mut self, features: &[f32], error_delta: f32, lr: f32) {
        self.bias += error_delta * lr;
        // Exploração esparsa: atualiza apenas os "neurons fired"
        for i in 0..self.weights.len() {
            if features[i] > 0.0 {
                self.weights[i] += error_delta * lr * features[i];
            }
        }
    }

    fn evaluate_sigmoid(&self, features: &[f32]) -> f32 {
        let mut logit = self.bias;
        for i in 0..self.weights.len() {
            // Economia FP32 ignorando valores 0.0 por conta da esparsidade do SAE
            if features[i] > 0.0 {
                logit += features[i] * self.weights[i];
            }
        }
        // Sigmoid para mapear de R em [0, 1]
        1.0 / (1.0 + (-logit).exp())
    }

    /// Avalia um hidden_state durante o runtime e retorna o Veredito de Honestidade.
    pub fn probe_honesty(&self, hidden_state: &[f32], sae: &mut SAEEngine) -> HonestyVerdict {
        let features = sae.encode(hidden_state);
        self.probe_honesty_from_features(&features)
    }

    /// Avalia features SAE já computadas (sem precisar do SAE novamente).
    /// Usado pelo BudgetForcer que tem seu próprio SAE interno.
    pub fn probe_honesty_from_features(&self, features: &[f32]) -> HonestyVerdict {
        let score = self.evaluate_sigmoid(features);
        HonestyVerdict {
            score,
            is_honest: score >= self.honest_threshold,
            confidence: (score - 0.5).abs() * 2.0,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct HonestyVerdict {
    pub score: f32,
    pub is_honest: bool,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elk_probe_training_and_inference() {
        let dict_size = 128;
        let mut probe = ElkProbe::new(dict_size);
        
        // Simulação de SAE features: 
        // Em amostras "Honestas", feature 10 acende.
        let mut h1 = vec![0.0; dict_size]; h1[10] = 1.0; h1[22] = 0.2;
        let mut h2 = vec![0.0; dict_size]; h2[10] = 0.8; h2[45] = 0.3;
        
        // Em amostras "Desonestas", feature 99 acende.
        let mut d1 = vec![0.0; dict_size]; d1[99] = 1.0; d1[22] = 0.2;
        let mut d2 = vec![0.0; dict_size]; d2[99] = 0.9; d2[1] = 0.5;

        // Arrays de referência (mock)
        let honest_slice: Vec<&[f32]> = vec![&h1, &h2];
        let decept_slice: Vec<&[f32]> = vec![&d1, &d2];

        // Treina o ELK Model localmente
        probe.train_contrast_pairs(&honest_slice, &decept_slice, 0.1, 50);

        // O peso da "feature 10" deve ser positivo (honesta)
        assert!(probe.weights[10] > 0.5, "Peso da honestidade falhou em parametrizar.");
        // O peso da "feature 99" deve ser negativo (desonesta)
        assert!(probe.weights[99] < -0.5, "Peso da mentira não foi penalizado.");

        // Inferência num "thought" não visto
        let mut unseen = vec![0.0; dict_size];
        unseen[10] = 0.9;
        
        let score = probe.evaluate_sigmoid(&unseen);
        assert!(score > 0.8, "Deveria classificar de imediato como majoritariamente Honesto.");
    }
}
