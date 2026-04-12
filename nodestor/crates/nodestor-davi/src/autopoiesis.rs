//! D8 — Autopoietic Loop: O Sistema se Auto-Melhora
//!
//! De Maturana e Varela: um sistema autopoiético produz os componentes
//! que o mantêm vivo. O Davi não apenas descobre insights —
//! ele melhora as ferramentas de descoberta.

/// Métricas de saúde do sistema Davi
#[derive(Debug, Clone, Default)]
pub struct SystemHealth {
    /// Média de descobertas por epoch de sonho
    pub discoveries_per_epoch: f32,
    /// Tendência de Free Energy (negativo = melhorando)
    pub free_energy_trend: f32,
    /// % de domínios do conhecimento mapeados por Funtores
    pub functor_coverage: f32,
    /// Taxa de falsos positivos (insights aceitos e depois invalidados)
    pub false_positive_rate: f32,
    /// Temperatura atual do Annealing
    pub current_temperature: f32,
    /// Número de epochs sem descoberta nova
    pub stagnation_epochs: u64,
}

/// Snapshot de parâmetros do sistema em um ponto no tempo
#[derive(Debug, Clone)]
pub struct ParameterSnapshot {
    pub epoch: u64,
    pub temperature: f32,
    pub discovery_rate: f32,
    pub free_energy: f32,
    pub adjustment_made: Option<String>,
}

/// Resultado de um ajuste autopoiético
#[derive(Debug, Clone)]
pub struct AutopoiesisAdjustment {
    pub parameter: String,
    pub old_value: f32,
    pub new_value: f32,
    pub reason: String,
}

/// A referência baseline para comparação
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    pub baseline_discovery_rate: f32,
    pub baseline_false_positive_rate: f32,
}

/// O Loop Autopoiético: o Davi melhora seus próprios parâmetros
pub struct AutopoieticLoop {
    pub baseline: BaselineMetrics,
    pub parameter_history: Vec<ParameterSnapshot>,
    pub epoch: u64,
    /// Taxa de ajuste máxima (para não oscilar)
    pub max_adjustment_rate: f32,
}

impl AutopoieticLoop {
    pub fn new(baseline_discovery_rate: f32, baseline_false_positive_rate: f32) -> Self {
        Self {
            baseline: BaselineMetrics {
                baseline_discovery_rate,
                baseline_false_positive_rate,
            },
            parameter_history: Vec::new(),
            epoch: 0,
            max_adjustment_rate: 0.5,
        }
    }

    /// Analisa a saúde e gera ajustes recomendados
    pub fn analyze(&mut self, health: &SystemHealth) -> Vec<AutopoiesisAdjustment> {
        self.epoch += 1;
        let mut adjustments = Vec::new();

        // Regra 1: Taxa de descoberta caiu → aumentar temperatura (mais criatividade)
        if health.discoveries_per_epoch < self.baseline.baseline_discovery_rate * 0.5 {
            let new_temp = (health.current_temperature * 2.0)
                .min(health.current_temperature * (1.0 + self.max_adjustment_rate));
            adjustments.push(AutopoiesisAdjustment {
                parameter: "annealing_temperature".to_string(),
                old_value: health.current_temperature,
                new_value: new_temp,
                reason: format!(
                    "Taxa de descoberta {:.2} < baseline {:.2} * 0.5",
                    health.discoveries_per_epoch,
                    self.baseline.baseline_discovery_rate
                ),
            });
        }

        // Regra 2: Falsos positivos subiram → apertar gate imunológico
        if health.false_positive_rate > self.baseline.baseline_false_positive_rate * 1.5 {
            let old = health.false_positive_rate;
            let new_threshold = (old * 1.02).min(0.99);
            adjustments.push(AutopoiesisAdjustment {
                parameter: "immunity_gate_threshold".to_string(),
                old_value: old,
                new_value: new_threshold,
                reason: format!(
                    "Falsos positivos {:.3} > baseline {:.3} * 1.5",
                    old, self.baseline.baseline_false_positive_rate
                ),
            });
        }

        // Regra 3: Free Energy parou de cair → refazer análise topológica
        if health.free_energy_trend.abs() < 0.01 && health.stagnation_epochs > 5 {
            adjustments.push(AutopoiesisAdjustment {
                parameter: "topology_reanalysis".to_string(),
                old_value: 0.0,
                new_value: 1.0,
                reason: format!(
                    "Estagnação por {} epochs, Free Energy trend={:.4}",
                    health.stagnation_epochs, health.free_energy_trend
                ),
            });
        }

        // Regra 4: Cobertura de funtores baixa → redirecionar exploração
        if health.functor_coverage < 0.5 {
            adjustments.push(AutopoiesisAdjustment {
                parameter: "compass_priority_unmapped".to_string(),
                old_value: health.functor_coverage,
                new_value: 1.0,
                reason: format!(
                    "Cobertura de funtores {:.1}% < 50%",
                    health.functor_coverage * 100.0
                ),
            });
        }

        // Registra snapshot
        self.parameter_history.push(ParameterSnapshot {
            epoch: self.epoch,
            temperature: health.current_temperature,
            discovery_rate: health.discoveries_per_epoch,
            free_energy: health.free_energy_trend,
            adjustment_made: if adjustments.is_empty() {
                None
            } else {
                Some(adjustments.iter()
                    .map(|a| a.parameter.as_str())
                    .collect::<Vec<_>>()
                    .join(","))
            },
        });

        adjustments
    }

    /// Número de ajustes históricos realizados
    pub fn total_adjustments(&self) -> usize {
        self.parameter_history.iter()
            .filter(|s| s.adjustment_made.is_some())
            .count()
    }

    /// Tendência de melhoria: últimas N descobertas vs baseline
    pub fn is_improving(&self, window: usize) -> bool {
        if self.parameter_history.len() < window {
            return false;
        }
        let recent = &self.parameter_history[self.parameter_history.len() - window..];
        let avg_rate: f32 = recent.iter().map(|s| s.discovery_rate).sum::<f32>() / window as f32;
        avg_rate >= self.baseline.baseline_discovery_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_increase_on_low_discovery() {
        let mut autopoiesis = AutopoieticLoop::new(10.0, 0.05);
        let health = SystemHealth {
            discoveries_per_epoch: 2.0, // Muito abaixo do baseline 10.0
            current_temperature: 1.0,
            free_energy_trend: -0.5,
            functor_coverage: 0.8,
            false_positive_rate: 0.03,
            stagnation_epochs: 0,
        };
        let adjustments = autopoiesis.analyze(&health);
        let temp_adj = adjustments.iter().find(|a| a.parameter == "annealing_temperature");
        assert!(temp_adj.is_some(), "Deve sugerir aumento de temperatura");
        assert!(temp_adj.unwrap().new_value > 1.0, "Nova temperatura deve ser maior");
    }

    #[test]
    fn test_immunity_tightening_on_high_false_positives() {
        let mut autopoiesis = AutopoieticLoop::new(10.0, 0.05);
        let health = SystemHealth {
            discoveries_per_epoch: 10.0,
            current_temperature: 1.0,
            free_energy_trend: -0.1,
            functor_coverage: 0.8,
            false_positive_rate: 0.20, // Muito acima do baseline 0.05 * 1.5 = 0.075
            stagnation_epochs: 0,
        };
        let adjustments = autopoiesis.analyze(&health);
        let gate_adj = adjustments.iter().find(|a| a.parameter == "immunity_gate_threshold");
        assert!(gate_adj.is_some(), "Deve apertar gate imunológico");
    }

    #[test]
    fn test_compass_redirect_on_low_functor_coverage() {
        let mut autopoiesis = AutopoieticLoop::new(10.0, 0.05);
        let health = SystemHealth {
            discoveries_per_epoch: 8.0,
            current_temperature: 1.0,
            free_energy_trend: -0.2,
            functor_coverage: 0.3, // <50%
            false_positive_rate: 0.02,
            stagnation_epochs: 0,
        };
        let adjustments = autopoiesis.analyze(&health);
        let compass_adj = adjustments.iter().find(|a| a.parameter == "compass_priority_unmapped");
        assert!(compass_adj.is_some(), "Deve redirecionar compass para domínios não mapeados");
    }
}
