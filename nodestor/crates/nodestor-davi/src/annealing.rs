//! D4 — Semantic Annealing: Temperatura Controla a Criatividade
//!
//! Da Metalurgia: metal quente aceita qualquer forma (criativo).
//! Metal frio mantém a forma (preciso).
//!
//! Critério de Metropolis: aceita pioras com probabilidade e^(-ΔE/T)

/// Schedule de resfriamento da temperatura
#[derive(Debug, Clone)]
pub enum CoolingSchedule {
    /// T = T₀ - alpha * step
    Linear { alpha: f32 },
    /// T = T₀ * decay^step (mais natural, como resfriamento real)
    Exponential { decay: f32 },
    /// T = T₀ / (1 + step) — convergência garantida matematicamente
    Cauchy,
    /// Adaptativo baseado na curva de Free Energy
    /// Se F cai rápido → esfria devagar (está achando coisas boas)
    /// Se F estabilizou → esfria rápido (já explorou tudo aqui)
    AdaptiveFreeEnergy { sensitivity: f32 },
}

/// O motor de Annealing Semântico do Davi
pub struct SemanticAnnealing {
    /// Temperatura inicial
    pub initial_temperature: f32,
    /// Temperatura atual
    pub current_temperature: f32,
    /// Schedule de resfriamento
    pub cooling_schedule: CoolingSchedule,
    /// Trigger de reaquecimento (variância mínima antes de aquecer)
    pub reheat_trigger: f32,
    /// Passo atual
    pub step: u64,
    /// Semente determinística para reprodutibilidade
    pub seed: u64,
}

impl SemanticAnnealing {
    pub fn new(initial_temperature: f32, schedule: CoolingSchedule) -> Self {
        Self {
            initial_temperature,
            current_temperature: initial_temperature,
            cooling_schedule: schedule,
            reheat_trigger: 0.001,
            step: 0,
            seed: 42,
        }
    }

    /// Testa se aceita um salto cross-domain (Critério de Metropolis)
    /// Alta temp: aceita quase qualquer coisa (criativo)
    /// Baixa temp: só aceita melhorias (cirúrgico)
    pub fn accept_cross_domain_jump(
        &mut self,
        current_energy: f32,
        proposed_energy: f32,
    ) -> bool {
        if proposed_energy <= current_energy {
            // Sempre aceita melhorias
            return true;
        }
        // Rejeita se temperatura muito baixa
        if self.current_temperature < 1e-6 {
            return false;
        }
        // Metropolis: aceita pioras com probabilidade e^(-ΔE/T)
        let delta = proposed_energy - current_energy;
        let probability = (-delta / self.current_temperature).exp();
        // Geração pseudo-aleatória determinística (LCG simples)
        let rand_val = self.rand_01();
        rand_val < probability
    }

    /// Avança um passo de resfriamento
    pub fn cool_one_step(&mut self) {
        self.step += 1;
        self.current_temperature = match &self.cooling_schedule {
            CoolingSchedule::Linear { alpha } => {
                (self.initial_temperature - alpha * self.step as f32).max(0.001)
            }
            CoolingSchedule::Exponential { decay } => {
                self.initial_temperature * decay.powi(self.step as i32)
            }
            CoolingSchedule::Cauchy => {
                self.initial_temperature / (1.0 + self.step as f32)
            }
            CoolingSchedule::AdaptiveFreeEnergy { sensitivity } => {
                // Placeholder: usa exponencial com sensibilidade
                self.initial_temperature * (1.0 - sensitivity).powi(self.step as i32)
            }
        };
    }

    /// Verifica se o sistema travou num mínimo local e deve reaquecer
    pub fn check_reheat(&mut self, free_energy_history: &[f32]) -> bool {
        let window = 10;
        if free_energy_history.len() < window {
            return false;
        }
        let recent = &free_energy_history[free_energy_history.len() - window..];
        let mean: f32 = recent.iter().sum::<f32>() / window as f32;
        let variance: f32 = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / window as f32;

        if variance < self.reheat_trigger {
            // Travou! Reaquecer para escapar do mínimo local
            self.current_temperature = (self.initial_temperature * 0.5)
                .max(self.current_temperature * 3.0);
            return true;
        }
        false
    }

    /// Temperatura normalizada [0, 1] para UI
    pub fn normalized_temperature(&self) -> f32 {
        (self.current_temperature / self.initial_temperature).clamp(0.0, 1.0)
    }

    /// Gerador pseudo-aleatório determinístico (LCG)
    fn rand_01(&mut self) -> f32 {
        self.seed = self.seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.seed >> 33) as f32 / u32::MAX as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_improvement_always() {
        let mut annealing = SemanticAnnealing::new(
            10.0,
            CoolingSchedule::Exponential { decay: 0.95 },
        );
        // Melhorias (proposed < current) devem SEMPRE ser aceitas
        for _ in 0..100 {
            assert!(annealing.accept_cross_domain_jump(10.0, 5.0));
        }
    }

    #[test]
    fn test_reject_at_low_temperature() {
        let mut annealing = SemanticAnnealing::new(
            0.0001,  // Temperatura muito baixa
            CoolingSchedule::Cauchy,
        );
        // Com temperatura quase zero, pioras grandes devem ser rejeitadas
        let mut accepted = 0;
        for _ in 0..100 {
            if annealing.accept_cross_domain_jump(1.0, 100.0) {
                accepted += 1;
            }
        }
        assert!(accepted < 10, "Temperatura baixa deveria rejeitar a maioria das pioras");
    }

    #[test]
    fn test_cooling_reduces_temperature() {
        let mut annealing = SemanticAnnealing::new(
            100.0,
            CoolingSchedule::Exponential { decay: 0.9 },
        );
        let initial = annealing.current_temperature;
        for _ in 0..20 {
            annealing.cool_one_step();
        }
        assert!(annealing.current_temperature < initial * 0.5,
            "Temperatura deve cair com o resfriamento");
    }

    #[test]
    fn test_reheat_on_stagnation() {
        let mut annealing = SemanticAnnealing::new(
            10.0,
            CoolingSchedule::Cauchy,
        );
        // Simula estagnação: energia constante por 10+ steps
        let stagnant_history: Vec<f32> = (0..15).map(|_| 1.0).collect();
        annealing.current_temperature = 0.01; // Temperatura já baixa
        let reheated = annealing.check_reheat(&stagnant_history);
        assert!(reheated, "Sistema estagnado deve reaquecer");
        assert!(annealing.current_temperature > 0.01, "Temperatura deve subir após reaquecimento");
    }
}
