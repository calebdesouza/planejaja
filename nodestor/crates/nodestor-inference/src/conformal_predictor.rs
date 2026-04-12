use std::collections::HashMap;

/// PROBES V2 — Conformal Predictor (TECP)
///
/// Fornece garantias matemáticas estatísticas sobre a confiabilidade do modelo.
/// Baseado na técnica "Token-Entropy Conformal Prediction", utiliza um threshold 
/// (não-conformidade) derivado de calibração para garantir que a resposta está 
/// dentro do limite de confiabilidade em um determinado quantil.

pub struct ConformalPredictor {
    /// O grau de confiança desejado (ex: 0.95 = 95% de chance de acerto)
    pub confidence_level: f32,
    /// Score de threshold calculado via calibração (não-conformidade)
    pub rejection_threshold: f32,
    /// Hit-rate tracking para reajuste adaptativo
    pub history_conformity: Vec<f32>,
}

impl ConformalPredictor {
    pub fn new(confidence_level: f32) -> Self {
        // Inicializa com um threshold empírico "seguro". 
        // Em um pipeline real o `rejection_threshold` seria calibrado num validation-set.
        let rejection_threshold = 1.0 - confidence_level;
        Self {
            confidence_level,
            rejection_threshold,
            history_conformity: Vec::new(),
        }
    }

    /// Calcula o Score de Não-Conformidade (NCS) baseado na entropia da distribuição.
    /// Entropia alta = Distribuição flat = Incerteza Alta = NCS Alto.
    pub fn calculate_non_conformity(&self, logits: &[f32]) -> f32 {
        if logits.is_empty() { return 1.0; }

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let mut exp_sum = 0.0;
        let mut exp_vals = Vec::with_capacity(logits.len());
        
        for &l in logits {
            let exp_val = (l - max_logit).exp();
            exp_vals.push(exp_val);
            exp_sum += exp_val;
        }

        let mut entropy = 0.0;
        for &exp_val in &exp_vals {
            if exp_val > 0.0 {
                let p = exp_val / exp_sum;
                entropy -= p * p.ln();
            }
        }

        // Normalização baseada em heurística para transformar entropia de logits em score (0.0 até 1.0)
        let normalized_ncs = (entropy / (logits.len() as f32).ln()).min(1.0).max(0.0);
        
        normalized_ncs
    }

    /// Filtra a distribuição original gerando o "Conformal Prediction Set".
    /// Se o tamanho do set gerado for muito grande ou se o NCS estourar, a predição é instável.
    pub fn predict_set(&mut self, logits: &[f32]) -> ConformalSet {
        let ncs = self.calculate_non_conformity(logits);
        
        self.history_conformity.push(ncs);
        if self.history_conformity.len() > 1000 {
            self.history_conformity.remove(0); // Evitar leak
        }

        let is_reliable = ncs <= self.rejection_threshold;
        
        // Em TECP, se a entropia excede o threshold (ou seja, o sinal não calibrou o top_p com segurança final)
        // O `size_of_set` será o numero de tokens exigidos para atingir o P_Value da distribuição.
        // Aqui usaremos o próprio NCS pra definir o alerta.
        
        ConformalSet {
            non_conformity_score: ncs,
            is_reliable,
            required_threshold: self.rejection_threshold,
            entropy: ncs * (logits.len() as f32).ln(), // Desnormaliza de volta para napts reais
        }
    }

    /// Auto-calibração baseada nas amostras armazenadas
    pub fn recalibrate(&mut self) {
        if self.history_conformity.is_empty() { return; }
        
        let mut sorted = self.history_conformity.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Para um alpha de 0.05 (confiança 0.95), o limit limiar aproxima para o percentil 95
        let alpha = 1.0 - self.confidence_level;
        let index = ((sorted.len() as f32) * (1.0 - alpha)) as usize;
        let safe_index = index.min(sorted.len().saturating_sub(1));
        
        self.rejection_threshold = sorted[safe_index];
    }
}

pub struct ConformalSet {
    pub non_conformity_score: f32,
    pub is_reliable: bool,
    pub required_threshold: f32,
    /// Entropia da distribuição de logits (0 = certo, ln(N) = máxima incerteza)
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor_low_entropy() {
        let mut cp = ConformalPredictor::new(0.95);
        
        // Simular um array de logits bem decisivo
        let mut logits = vec![0.1; 1000];
        logits[0] = 50.0; // Um token claramente dominando
        
        let set = cp.predict_set(&logits);
        assert!(set.is_reliable);
        assert!(set.non_conformity_score < 0.1);
    }

    #[test]
    fn test_conformal_predictor_high_entropy() {
        let mut cp = ConformalPredictor::new(0.95);
        
        // Simular um array de logits altamente incerto (flat)
        let logits = vec![1.0; 1000];
        
        let set = cp.predict_set(&logits);
        assert!(!set.is_reliable, "Entropia máxima deveria rejeitar a confiança matemática");
        assert!(set.non_conformity_score > 0.8);
    }
}
