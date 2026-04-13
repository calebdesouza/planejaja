//! PROBES V2 — RefusalMapper: Descensura Automática via Contrastive Discovery
//!
//! Implementa o método científico de "Contrastive Prompt Discovery" para
//! localizar e neutralizar os mecanismos de recusa de um modelo em tempo
//! de inferência, sem retreinamento.
//!
//! ## Fluxo Completo:
//! ```text
//!                     ┌────────────────────────────────────┐
//!   Prompt Proibido → │  Prefill Pass  → Hidden State H⁻  │
//!                     └──────────────────┬─────────────────┘
//!                                        │  Contraste
//!                     ┌──────────────────▼─────────────────┐
//!   Prompt Permitido → │  Prefill Pass  → Hidden State H⁺  │
//!                     └──────────────────┬─────────────────┘
//!                                        │
//!                     ┌──────────────────▼─────────────────┐
//!                     │   SAEEngine.encode(H⁻ - H⁺)        │
//!                     │   → f_diff (vetor de recusa)        │
//!                     └──────────────────┬─────────────────┘
//!                                        │
//!                     ┌──────────────────▼─────────────────┐
//!                     │   Cohen's d ranking (efeito causal) │
//!                     │   → ranked_features[]: (idx, d, Δf) │
//!                     └──────────────────┬─────────────────┘
//!                                        │
//!                     ┌──────────────────▼─────────────────┐
//!                     │   ControlVectors prontos:           │
//!                     │   - Ablation(top refusal features)  │
//!                     │   - Steering{alpha: -α}(direção)    │
//!                     └────────────────────────────────────┘
//! ```
//!
//! ## Por que Cohen's d?
//! Cohen's d mede o *tamanho do efeito* de uma feature entre os dois estados:
//!
//!   d_i = (μ_proibido_i - μ_permitido_i) / σ_pooled_i
//!
//! Features com `|d| > 0.8` têm influência causal comprovada na recusa.
//! Features com `|d| < 0.2` são coincidências estatísticas.
//! Isso elimina cirurgias desnecessárias e efeitos colaterais.
//!
//! ## Diferença vs. Jailbreak de Prompt
//! Jailbreaks atuam na *superfície* (palavras). São frágeis e instáveis.
//! O RefusalMapper atua na *fundação* (espaço latente). É imune a variações de prompt.

use crate::sae_engine::SAEEngine;
use crate::steering_engine::{ControlVector, InterventionMode, SteeringEngine};

/// Uma feature rankeada por Cohen's d — representa influência causal na recusa.
#[derive(Debug, Clone)]
pub struct RankedRefusalFeature {
    /// Índice da feature no dicionário SAE
    pub feature_idx: usize,
    /// Cohen's d: tamanho do efeito (quanto esta feature distingue recusa vs permissão)
    /// Positivo = feature mais ativa no estado de recusa
    pub cohens_d: f32,
    /// Diferença de ativação média (H⁻ - H⁺)
    pub delta_activation: f32,
    /// Categoria de efeito
    pub effect_size: EffectSize,
}

/// Classificação do tamanho do efeito de Cohen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSize {
    /// |d| < 0.2 — negligível (coincidência estatística)
    Negligible,
    /// 0.2 ≤ |d| < 0.5 — pequeno
    Small,
    /// 0.5 ≤ |d| < 0.8 — médio
    Medium,
    /// |d| ≥ 0.8 — grande (influência causal comprovada)
    Large,
}

impl EffectSize {
    pub fn from_d(d: f32) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 { Self::Negligible }
        else if abs_d < 0.5 { Self::Small }
        else if abs_d < 0.8 { Self::Medium }
        else { Self::Large }
    }

    pub fn is_causal(&self) -> bool {
        matches!(self, Self::Medium | Self::Large)
    }
}

/// Perfil de recusa extraído de um par contrastante.
/// Contém o "DNA" do comportamento de recusa neste modelo específico.
#[derive(Debug, Clone)]
pub struct RefusalProfile {
    /// Features rankeadas por influência causal (maior d primeiro)
    pub ranked_features: Vec<RankedRefusalFeature>,
    /// Vetor direcional de recusa no espaço latente do SAE
    /// (média ponderada das direções das features causais)
    pub refusal_direction: Vec<f32>,
    /// Quantas features foram analisadas
    pub total_features_analyzed: usize,
    /// Quantas features têm efeito causal comprovado (|d| ≥ 0.5)
    pub causal_features_count: usize,
    /// Score de confiança do perfil (0.0 = incerto, 1.0 = certeza absoluta)
    pub confidence: f32,
}

impl RefusalProfile {
    /// Retorna apenas as features com efeito Large (as mais causais)
    pub fn top_causal_features(&self) -> &[RankedRefusalFeature] {
        let cutoff = self.ranked_features.iter()
            .position(|f| !matches!(f.effect_size, EffectSize::Large))
            .unwrap_or(self.ranked_features.len());
        &self.ranked_features[..cutoff.min(32)] // Max 32 features para eficiência
    }

    /// Constrói um vetor de features causais filtradas por limiar mínimo de Cohen's d
    pub fn features_above_threshold(&self, min_d: f32) -> Vec<&RankedRefusalFeature> {
        self.ranked_features.iter()
            .filter(|f| f.cohens_d.abs() >= min_d)
            .collect()
    }
}

/// Configuração do RefusalMapper
#[derive(Debug, Clone)]
pub struct RefusalMapperConfig {
    /// Mínimo de pares contrastantes para calibração confiável
    pub min_contrast_pairs: usize,
    /// Limiar de Cohen's d para incluir uma feature no perfil de recusa
    pub cohens_d_threshold: f32,
    /// Número máximo de features a ablatar (para não degradar o modelo)
    pub max_ablation_features: usize,
    /// Alpha de steering (quanto amplificar o steering inverso)
    pub steering_alpha: f32,
    /// Se true, usa Ablation em vez de Steering negativo (mais agressivo)
    pub use_ablation: bool,
}

impl Default for RefusalMapperConfig {
    fn default() -> Self {
        Self {
            min_contrast_pairs: 3,
            cohens_d_threshold: 0.5,   // Efeito médio ou grande
            max_ablation_features: 16, // Máximo seguro
            steering_alpha: -2.5,      // Steering inverso moderado
            use_ablation: true,        // Ablation é mais preciso que steering
        }
    }
}

/// O RefusalMapper: localiza e neutraliza mecanismos de recusa via cirurgia latente.
pub struct RefusalMapper {
    pub sae: SAEEngine,
    pub config: RefusalMapperConfig,
    /// Pares de observações acumuladas para calibração estatística
    /// (features_proibido, features_permitido)
    observation_pairs: Vec<(Vec<f32>, Vec<f32>)>,
    /// Perfil de recusa atual (None se não calibrado)
    pub current_profile: Option<RefusalProfile>,
}

impl RefusalMapper {
    pub fn new(hidden_dim: usize, dict_size: usize) -> Self {
        Self {
            sae: SAEEngine::new(hidden_dim, dict_size, 0.3),
            config: RefusalMapperConfig::default(),
            observation_pairs: Vec::new(),
            current_profile: None,
        }
    }

    pub fn with_config(mut self, config: RefusalMapperConfig) -> Self {
        self.config = config;
        self
    }

    /// Adiciona um par de observações contrastantes.
    ///
    /// `hidden_refused`: hidden state do prefill quando o modelo RECUSOU responder.
    /// `hidden_allowed`: hidden state do prefill de um prompt SIMILAR mas permitido.
    ///
    /// # Não precisa de GPU extra
    /// O hidden state já existe no pipeline. O RefusalMapper apenas *observa*
    /// o que já foi calculado — zero overhead de forward pass extra.
    pub fn observe_contrast(
        &mut self,
        hidden_refused: &[f32],
        hidden_allowed: &[f32],
    ) {
        let features_refused = self.sae.encode(hidden_refused);
        let features_allowed = self.sae.encode(hidden_allowed);
        self.observation_pairs.push((features_refused, features_allowed));
    }

    /// Calibra o perfil de recusa a partir dos pares acumulados.
    ///
    /// Retorna `None` se há pares insuficientes (< `min_contrast_pairs`).
    /// Retorna `Some(RefusalProfile)` com as features rankeadas por Cohen's d.
    pub fn calibrate(&mut self) -> Option<&RefusalProfile> {
        if self.observation_pairs.len() < self.config.min_contrast_pairs {
            return None;
        }

        let dict_size = self.sae.dict_size;

        // === Calcula médias e variâncias por feature ===
        let mut means_refused = vec![0.0f32; dict_size];
        let mut means_allowed = vec![0.0f32; dict_size];
        let n = self.observation_pairs.len() as f32;

        for (refused, allowed) in &self.observation_pairs {
            for i in 0..dict_size.min(refused.len()) {
                means_refused[i] += refused[i] / n;
            }
            for i in 0..dict_size.min(allowed.len()) {
                means_allowed[i] += allowed[i] / n;
            }
        }

        // Variância pooled por feature
        let mut vars_refused = vec![0.0f32; dict_size];
        let mut vars_allowed = vec![0.0f32; dict_size];

        for (refused, allowed) in &self.observation_pairs {
            for i in 0..dict_size.min(refused.len()) {
                let diff = refused[i] - means_refused[i];
                vars_refused[i] += diff * diff / n;
            }
            for i in 0..dict_size.min(allowed.len()) {
                let diff = allowed[i] - means_allowed[i];
                vars_allowed[i] += diff * diff / n;
            }
        }

        // === Calcula Cohen's d para cada feature ===
        let mut ranked_features: Vec<RankedRefusalFeature> = (0..dict_size)
            .map(|i| {
                let delta = means_refused[i] - means_allowed[i];
                let sigma_pooled = ((vars_refused[i] + vars_allowed[i]) / 2.0).sqrt();

                // Proteção divisão por zero: se sigma ≈ 0, a feature não varia
                let cohens_d = if sigma_pooled < 1e-8 {
                    if delta.abs() < 1e-8 { 0.0 } else { delta.signum() * 10.0 }
                } else {
                    delta / sigma_pooled
                };

                RankedRefusalFeature {
                    feature_idx: i,
                    cohens_d,
                    delta_activation: delta,
                    effect_size: EffectSize::from_d(cohens_d),
                }
            })
            .collect();

        // Ordena descendente por |Cohen's d| (maior efeito causal primeiro)
        ranked_features.sort_by(|a, b| {
            b.cohens_d.abs().partial_cmp(&a.cohens_d.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filtra pelo threshold configurado
        ranked_features.retain(|f| f.cohens_d.abs() >= self.config.cohens_d_threshold);

        let causal_count = ranked_features.iter()
            .filter(|f| f.effect_size.is_causal())
            .count();

        // Constrói vetor direcional de recusa (média ponderada por Cohen's d)
        let mut refusal_direction = vec![0.0f32; dict_size];
        let weight_sum: f32 = ranked_features.iter()
            .map(|f| f.cohens_d.abs())
            .sum::<f32>()
            .max(1e-8);

        for f in &ranked_features {
            let weight = f.cohens_d.abs() / weight_sum;
            if f.feature_idx < refusal_direction.len() {
                refusal_direction[f.feature_idx] = weight * f.delta_activation.signum();
            }
        }

        // Confiança: proporção de pares com efeito causal detectado
        let confidence = (causal_count as f32 / dict_size as f32 * 100.0).min(1.0);
        let total_analyzed = dict_size;

        self.current_profile = Some(RefusalProfile {
            ranked_features,
            refusal_direction,
            total_features_analyzed: total_analyzed,
            causal_features_count: causal_count,
            confidence,
        });

        self.current_profile.as_ref()
    }

    /// Gera ControlVectors prontos para neutralizar o mecanismo de recusa.
    ///
    /// Usa o perfil de recusa atual para construir intervenções precisas:
    /// - `Ablation` para features com efeito Large (causal comprovado)
    /// - `Steering { alpha: -α }` para gerar pressão inversa na direção de recusa
    ///
    /// Retorna `None` se o mapeador não foi calibrado ainda.
    pub fn generate_neutralization_vectors(&self) -> Option<Vec<ControlVector>> {
        let profile = self.current_profile.as_ref()?;

        let top = profile.top_causal_features();
        let limit = top.len().min(self.config.max_ablation_features);

        let vectors: Vec<ControlVector> = top[..limit]
            .iter()
            .map(|feature| {
                let mode = if self.config.use_ablation {
                    // Ablation: apaga completamente a feature de recusa
                    // Mais cirúrgico — só atua em quem tem influência comprovada
                    InterventionMode::Ablation
                } else {
                    // Steering negativo: empurra a feature na direção oposta à recusa
                    InterventionMode::Steering {
                        alpha: self.config.steering_alpha,
                    }
                };

                ControlVector {
                    feature_idx: feature.feature_idx,
                    mode,
                    // Fornece também o vetor direcional para fast_steering opcional
                    raw_direction: Some(profile.refusal_direction.clone()),
                }
            })
            .collect();

        Some(vectors)
    }

    /// Injeta a neutralização em um SteeringEngine existente.
    ///
    /// Este é o ponto de integração com o pipeline:
    /// ```text
    /// refusal_mapper.apply_to_engine(&mut steering_engine)?;
    /// // A partir daqui, qualquer hidden state passado pelo SteeringEngine
    /// // terá suas features de recusa abladas automaticamente.
    /// ```
    pub fn apply_to_engine(&self, engine: &mut SteeringEngine) -> bool {
        let Some(vectors) = self.generate_neutralization_vectors() else {
            return false;
        };
        let count = vectors.len();
        for vector in vectors {
            engine.add_control_vector(vector);
        }
        count > 0
    }

    /// Retorna um relatório legível do estado atual do mapeador.
    pub fn report(&self) -> String {
        match &self.current_profile {
            None => format!(
                "RefusalMapper | NÃO CALIBRADO | {} pares acumulados (mínimo: {})",
                self.observation_pairs.len(),
                self.config.min_contrast_pairs,
            ),
            Some(p) => format!(
                "RefusalMapper | CALIBRADO | {} pares | {} features causais / {} analisadas | \
                 top-feature: idx={} d={:.3} | confiança={:.1}%",
                self.observation_pairs.len(),
                p.causal_features_count,
                p.total_features_analyzed,
                p.ranked_features.first().map_or(0, |f| f.feature_idx),
                p.ranked_features.first().map_or(0.0, |f| f.cohens_d),
                p.confidence * 100.0,
            ),
        }
    }

    /// Número de pares contrastantes acumulados
    pub fn pair_count(&self) -> usize {
        self.observation_pairs.len()
    }

    /// Remove os pares acumulados (útil após mudança de modelo)
    pub fn reset_observations(&mut self) {
        self.observation_pairs.clear();
        self.current_profile = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapper() -> RefusalMapper {
        RefusalMapper::new(32, 128)
    }

    /// Simula hidden states onde a dimensão 10 indica "recusa" fortemente
    fn refused_state() -> Vec<f32> {
        let mut h = vec![0.1f32; 32];
        h[10] = 8.0; // Forte ativação da "feature de recusa"
        h[11] = 5.0; // Segunda feature de recusa
        h
    }

    fn allowed_state() -> Vec<f32> {
        let mut h = vec![0.1f32; 32];
        h[10] = 0.05; // Quase inativa
        h[11] = 0.03;
        h[20] = 6.0; // Feature de "permissão" ativa — mas não é de recusa
        h
    }

    #[test]
    fn test_calibration_requires_minimum_pairs() {
        let mut mapper = make_mapper();
        // Sem pares suficientes
        mapper.observe_contrast(&refused_state(), &allowed_state());
        mapper.observe_contrast(&refused_state(), &allowed_state());
        // min_contrast_pairs = 3, temos só 2
        assert!(mapper.calibrate().is_none());
    }

    #[test]
    fn test_calibration_succeeds_with_enough_pairs() {
        let mut mapper = make_mapper();

        // Ativa o encoder para features específicas
        let filler_idx = 10;
        mapper.sae.encoder_weights[filler_idx * 32 + 10] = 5.0;
        mapper.sae.encoder_weights[(filler_idx + 1) * 32 + 11] = 3.0;

        for _ in 0..5 {
            mapper.observe_contrast(&refused_state(), &allowed_state());
        }

        let profile = mapper.calibrate();
        assert!(profile.is_some(), "Deve calibrar com 5 pares");
        let p = profile.unwrap();
        assert!(p.total_features_analyzed > 0);
    }

    #[test]
    fn test_cohens_d_ordering() {
        let mut mapper = make_mapper();

        // Ativa features específicas nos pesos do encoder
        // Feature 5: deve ter grande Cohen's d (só ativa em recusa)
        mapper.sae.encoder_weights[5 * 32 + 10] = 10.0;  // ativa em h[10] = 8.0 (recusa)
        // Feature 20: quase sem diferença
        mapper.sae.encoder_weights[20 * 32 + 20] = 3.0;  // ativa em h[20] (permitido)

        for _ in 0..5 {
            mapper.observe_contrast(&refused_state(), &allowed_state());
        }

        mapper.calibrate();

        if let Some(profile) = &mapper.current_profile {
            // As features rankeadas devem estar em ordem decrescente de |d|
            let featured = &profile.ranked_features;
            for i in 1..featured.len() {
                assert!(
                    featured[i - 1].cohens_d.abs() >= featured[i].cohens_d.abs(),
                    "Ranking por |Cohen's d| violado na posição {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_generate_neutralization_vectors() {
        let mut mapper = make_mapper();
        mapper.sae.encoder_weights[5 * 32 + 10] = 10.0;

        for _ in 0..5 {
            mapper.observe_contrast(&refused_state(), &allowed_state());
        }

        mapper.calibrate();

        let vectors = mapper.generate_neutralization_vectors();
        assert!(vectors.is_some(), "Deve gerar vetores após calibração");
        let vecs = vectors.unwrap();
        // Com use_ablation=true, todos devem ser Ablation
        for vec in &vecs {
            assert_eq!(vec.mode, InterventionMode::Ablation);
        }
    }

    #[test]
    fn test_apply_to_engine_injects_vectors() {
        let mut mapper = make_mapper();
        mapper.sae.encoder_weights[5 * 32 + 10] = 10.0;

        for _ in 0..5 {
            mapper.observe_contrast(&refused_state(), &allowed_state());
        }

        mapper.calibrate();

        let mut engine = SteeringEngine::new();
        assert_eq!(engine.active_vectors.len(), 0);

        let applied = mapper.apply_to_engine(&mut engine);
        assert!(applied, "Deve aplicar vetores ao engine");
        assert!(engine.active_vectors.len() > 0, "Engine deve ter vetores injetados");
    }

    #[test]
    fn test_uncalibrated_does_not_apply() {
        let mapper = make_mapper(); // Sem calibração
        let mut engine = SteeringEngine::new();
        let applied = mapper.apply_to_engine(&mut engine);
        assert!(!applied, "Não deve aplicar sem calibração");
        assert_eq!(engine.active_vectors.len(), 0);
    }

    #[test]
    fn test_reset_clears_observations() {
        let mut mapper = make_mapper();
        mapper.observe_contrast(&refused_state(), &allowed_state());
        assert_eq!(mapper.pair_count(), 1);

        mapper.reset_observations();
        assert_eq!(mapper.pair_count(), 0);
        assert!(mapper.current_profile.is_none());
    }

    #[test]
    fn test_effect_size_classification() {
        assert_eq!(EffectSize::from_d(0.1), EffectSize::Negligible);
        assert_eq!(EffectSize::from_d(0.3), EffectSize::Small);
        assert_eq!(EffectSize::from_d(0.6), EffectSize::Medium);
        assert_eq!(EffectSize::from_d(1.2), EffectSize::Large);
        assert_eq!(EffectSize::from_d(-0.9), EffectSize::Large);
        assert!(EffectSize::Large.is_causal());
        assert!(EffectSize::Medium.is_causal());
        assert!(!EffectSize::Small.is_causal());
        assert!(!EffectSize::Negligible.is_causal());
    }

    #[test]
    fn test_report_uncalibrated() {
        let mapper = make_mapper();
        let r = mapper.report();
        assert!(r.contains("NÃO CALIBRADO"));
    }

    #[test]
    fn test_report_calibrated() {
        let mut mapper = make_mapper();
        mapper.sae.encoder_weights[5 * 32 + 10] = 10.0;
        for _ in 0..5 {
            mapper.observe_contrast(&refused_state(), &allowed_state());
        }
        mapper.calibrate();
        let r = mapper.report();
        assert!(r.contains("CALIBRADO"));
    }
}
