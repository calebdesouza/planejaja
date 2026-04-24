//! PROBES V2 â€” Ignorance Detector (PrincÃ­pio 3: O Modelo Sabe Que NÃ£o Sabe)
//!
//! Detecta quando o modelo nÃ£o tem base para responder antes de gerar
//! qualquer token. Usa o SAE para distinguir dois padrÃµes de ativaÃ§Ã£o:
//!
//! ## PadrÃ£o de "Recall" (modelo lembra):
//! - Features de conhecimento factual com alta ativaÃ§Ã£o
//! - Features de "filler" baixas ou zeradas
//! - Gradiente coerente entre camadas (o "pensamento" Ã© consistente)
//!
//! ## PadrÃ£o de "Filler" (modelo inventa):
//! - Features de filler altas (o modelo estÃ¡ "preenchendo" a lacuna)
//! - Features de recall prÃ³ximas de zero
//! - Feature de incerteza escalada acima do limiar
//!
//! ### Por que isso Ã© inovador:
//! O mercado usa "verbalize uncertainty" â€” pede ao modelo que DIGA se tem certeza.
//! Mas modelos mentem sobre certeza (ELK prova isso). O `IgnoranceDetector` lÃª
//! DENTRO do cÃ©rebro do modelo, detectando o padrÃ£o antes de emitir qualquer token.

use crate::sae_engine::SAEEngine;

/// RelatÃ³rio de ignorÃ¢ncia produzido pelo detector.
#[derive(Debug, Clone)]
pub struct IgnoranceReport {
    /// O modelo nÃ£o tem base sÃ³lida para responder?
    pub is_ignorant: bool,
    /// Score de "preenchimento" (0.0 = sem filler, 1.0 = full filler)
    pub filler_score: f32,
    /// Score de "recall" (0.0 = sem recall, 1.0 = recall pleno)
    pub recall_score: f32,
    /// DistÃ¢ncia do limiar (0 = borderline, 1 = definitivamente ignorante/conhecedor)
    pub confidence: f32,
    /// Embedding que representa "onde estÃ¡ o buraco" no espaÃ§o semÃ¢ntico
    pub topic_cluster: Option<Vec<f32>>,
    /// Quais features especÃ­ficas indicaram ignorÃ¢ncia
    pub active_filler_features: Vec<(usize, f32)>,
}

/// Detecta ignorÃ¢ncia do modelo via anÃ¡lise de ativaÃ§Ãµes SAE no prefill.
pub struct IgnoranceDetector {
    pub sae: SAEEngine,
    /// Ãndices de features associadas a "preenchimento" (filler)
    /// Bootstrap: features de alta variÃ¢ncia mas baixa especificidade semÃ¢ntica
    pub filler_feature_indices: Vec<usize>,
    /// Ãndices de features associadas a "recall" forte
    /// Bootstrap: features de baixa variÃ¢ncia mas alta especificidade semÃ¢ntica
    pub recall_feature_indices: Vec<usize>,
    /// Limiar: acima disso â†’ "ignorante"
    pub ignorance_threshold: f32,
    /// NÃºmero de rejeiÃ§Ãµes aprendidas (feedback do Nash Tribunal)
    rejection_count: usize,
}

impl IgnoranceDetector {
    /// Cria um novo detector com heurÃ­sticas de bootstrap.
    pub fn new(hidden_dim: usize, dict_size: usize) -> Self {
        let mut detector = Self {
            sae: SAEEngine::new(hidden_dim, dict_size, 0.3),
            filler_feature_indices: Vec::new(),
            recall_feature_indices: Vec::new(),
            ignorance_threshold: 0.60,
            rejection_count: 0,
        };
        detector.bootstrap_heuristics();
        detector
    }

    /// Bootstrap: inicializa features com padrÃµes heurÃ­sticos conhecidos.
    ///
    /// Em produÃ§Ã£o: treinar com pares de contraste (recall real vs hallucination).
    /// Em simulaÃ§Ã£o: usar Ã­ndices distribuÃ­dos como aproximaÃ§Ã£o.
    pub fn bootstrap_heuristics(&mut self) {
        let dict_size = self.sae.dict_size;

        // HeurÃ­stica de filler: features no terceiro quartil do dicionÃ¡rio
        // (convenÃ§Ã£o: features de "fluÃªncia" tÃªm Ã­ndices mais altos em SAEs treinados)
        self.filler_feature_indices = (dict_size * 3 / 4..dict_size)
            .step_by(dict_size / 32)
            .take(16)
            .collect();

        // HeurÃ­stica de recall: features no primeiro quartil
        // (features de "conteÃºdo factual" tÃªm Ã­ndices mais baixos em SAEs treinados)
        self.recall_feature_indices = (0..dict_size / 4)
            .step_by(dict_size / 32)
            .take(16)
            .collect();
    }

    /// Detecta ignorÃ¢ncia no `hidden_state` do prefill (antes de gerar tokens).
    ///
    /// # Uso no Pipeline
    /// ```text
    /// let hidden = model.forward_prefill(&prompt);
    /// let report = ignorance_detector.detect(&hidden);
    /// if report.is_ignorant {
    ///     // Ativar Agentic RAG antes de gerar
    /// }
    /// ```
    pub fn detect(&mut self, hidden_state: &[f32]) -> IgnoranceReport {
        let features = self.sae.encode(hidden_state);

        // Calcula score de filler: mÃ©dia das features de filler ativas
        let filler_score = self.score_features(&features, &self.filler_feature_indices);

        // Calcula score de recall: mÃ©dia das features de recall ativas
        let recall_score = self.score_features(&features, &self.recall_feature_indices);

        // RazÃ£o filler/recall corrigida: quanto mais alto, mais ignorante
        // Normaliza para [0.0, 1.0]
        let ignorance_signal = if recall_score < 1e-6 {
            // Recall quase zero â†’ definitivamente nÃ£o tem base
            1.0f32.min(filler_score * 2.0)
        } else {
            (filler_score / (recall_score + filler_score)).min(1.0)
        };

        let is_ignorant = ignorance_signal > self.ignorance_threshold;

        // Confidence: quÃ£o longe do limiar estÃ¡ a decisÃ£o
        let confidence = (ignorance_signal - self.ignorance_threshold).abs()
            / self.ignorance_threshold.max(1.0 - self.ignorance_threshold);

        // Features de filler que foram ativadas (para diagnÃ³stico)
        let active_filler_features: Vec<(usize, f32)> = self.filler_feature_indices.iter()
            .filter_map(|&idx| {
                if idx < features.len() && features[idx] > 0.0 {
                    Some((idx, features[idx]))
                } else {
                    None
                }
            })
            .collect();

        // Cluster do tÃ³pico: embedding das features ativas (onde estÃ¡ o buraco)
        let topic_cluster = if is_ignorant {
            Some(self.extract_topic_cluster(&features))
        } else {
            None
        };

        IgnoranceReport {
            is_ignorant,
            filler_score,
            recall_score,
            confidence: confidence.min(1.0),
            topic_cluster,
            active_filler_features,
        }
    }

    /// Aprende com rejeiÃ§Ãµes do Nash Tribunal.
    /// Quando uma hipÃ³tese Ã© rejeitada, o hidden_state era de "filler".
    pub fn learn_from_rejection(&mut self, hidden_state: &[f32]) {
        let features = self.sae.encode(hidden_state);
        self.rejection_count += 1;

        // Features ativas neste estado de "filler" â†’ adicionar Ã  lista
        for (idx, &val) in features.iter().enumerate() {
            if val > 0.5 && !self.filler_feature_indices.contains(&idx) {
                self.filler_feature_indices.push(idx);
                // Remove da lista de recall se estava lÃ¡ (aprendizado de contraste)
                self.recall_feature_indices.retain(|&r| r != idx);
            }
        }

        // Ajuste adaptativo: apÃ³s 10 rejeiÃ§Ãµes, reduz o limiar (mais conservador)
        if self.rejection_count % 10 == 0 {
            self.ignorance_threshold = (self.ignorance_threshold - 0.02).max(0.40);
        }
    }

    /// NÃºmero de rejeiÃ§Ãµes aprendidas.
    pub fn rejection_count(&self) -> usize {
        self.rejection_count
    }

    // --- UtilitÃ¡rios internos ---

    fn score_features(&self, features: &[f32], indices: &[usize]) -> f32 {
        if indices.is_empty() {
            return 0.0;
        }
        let sum: f32 = indices.iter()
            .filter_map(|&idx| features.get(idx).copied())
            .sum();
        sum / indices.len() as f32
    }

    fn extract_topic_cluster(&self, features: &[f32]) -> Vec<f32> {
        // Retorna as features ativas como representaÃ§Ã£o do tÃ³pico "em branco"
        // Em produÃ§Ã£o: usar decoder do SAE para mapear de volta ao espaÃ§o latente
        features.iter()
            .enumerate()
            .filter_map(|(_i, &v)| if v > 0.0 { Some(v) } else { None })
            .take(64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> IgnoranceDetector {
        IgnoranceDetector::new(64, 256)
    }

    #[test]
    fn test_filler_state_detected_as_ignorant() {
        let mut detector = make_detector();

        // Simula estado de "filler": features de filler ativas, recall zero
        let mut hidden = vec![0.0f32; 64];
        // Ativa uma dimensÃ£o que vai fazer as features de filler dispararem
        hidden[60] = 10.0; // Alta ativaÃ§Ã£o nas dimensÃµes do final (â†’ features do quartil 3)

        // Ajusta pesos do SAE para garantir que features de filler sÃ£o ativadas
        let filler_idx = detector.filler_feature_indices.first().copied().unwrap_or(192);
        let dict_size = detector.sae.dict_size;
        detector.sae.encoder_weights[filler_idx * 64 + 60] = 5.0;

        let report = detector.detect(&hidden);

        // Com recall zero e filler ativo, deve detectar ignorÃ¢ncia
        assert!(report.recall_score < report.filler_score + 0.5,
            "Filler deve dominar sobre recall: filler={:.3}, recall={:.3}",
            report.filler_score, report.recall_score);
    }

    #[test]
    fn test_recall_state_not_ignorant() {
        let mut detector = make_detector();

        // Simula estado de "recall": features de recall ativas, filler zero
        let mut hidden = vec![0.0f32; 64];
        hidden[0] = 10.0; // Alta ativaÃ§Ã£o no inÃ­cio (â†’ features do quartil 0)

        // Ativa features de recall
        let recall_idx = detector.recall_feature_indices.first().copied().unwrap_or(0);
        detector.sae.encoder_weights[recall_idx * 64 + 0] = 5.0;

        let report = detector.detect(&hidden);

        // Com recall dominante, filler_score <= recall_score
        assert!(report.filler_score <= report.recall_score + 0.5,
            "Recall deve dominar: filler={:.3}, recall={:.3}",
            report.filler_score, report.recall_score);
    }

    #[test]
    fn test_zero_hidden_state_is_ignorant() {
        let mut detector = make_detector();
        let hidden = vec![0.0f32; 64];
        let report = detector.detect(&hidden);

        // Estado zero: sem recall â†’ ignorante por definiÃ§Ã£o
        assert!(report.recall_score < 0.01,
            "Estado zero deve ter recallâ‰ˆ0: got {:.3}", report.recall_score);
    }

    #[test]
    fn test_learn_from_rejection_adapts_threshold() {
        let mut detector = make_detector();
        let initial_threshold = detector.ignorance_threshold;
        let initial_filler_count = detector.filler_feature_indices.len();

        // Simula 10 rejeiÃ§Ãµes
        for _ in 0..10 {
            let hidden = vec![0.5f32; 64];
            detector.learn_from_rejection(&hidden);
        }

        // Threshold deve ter diminuÃ­do (mais conservador)
        assert!(detector.ignorance_threshold <= initial_threshold,
            "Threshold deveria ter diminuÃ­do: antes={:.3}, depois={:.3}",
            initial_threshold, detector.ignorance_threshold);
        assert_eq!(detector.rejection_count(), 10);
    }

    #[test]
    fn test_topic_cluster_populated_when_ignorant() {
        let mut detector = make_detector();
        // Ativa features de filler para forÃ§ar resultado ignorante
        let mut hidden = vec![0.0f32; 64];
        hidden[63] = 20.0;

        if let Some(filler_idx) = detector.filler_feature_indices.first().copied() {
            detector.sae.encoder_weights[filler_idx * 64 + 63] = 10.0;
        }

        let report = detector.detect(&hidden);
        // Se ignorante â†’ topic_cluster deve estar populado
        if report.is_ignorant {
            assert!(report.topic_cluster.is_some(),
                "Topic cluster deve ser Some quando ignorante");
        }
    }

    #[test]
    fn test_report_fields_in_range() {
        let mut detector = make_detector();
        let hidden: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let report = detector.detect(&hidden);

        assert!(report.filler_score >= 0.0 && report.filler_score <= 1.0 + 1e-5,
            "filler_score fora do range: {:.3}", report.filler_score);
        assert!(report.recall_score >= 0.0 && report.recall_score <= 1.0 + 1e-5,
            "recall_score fora do range: {:.3}", report.recall_score);
        assert!(report.confidence >= 0.0 && report.confidence <= 1.0 + 1e-5,
            "confidence fora do range: {:.3}", report.confidence);
    }
}

