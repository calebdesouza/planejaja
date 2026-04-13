use crate::elk_probe::{ElkProbe, HonestyVerdict};
use nodestor_inference::sae_engine::SAEEngine;

/// PROBES V2 â€” CoT Monitor (Chain of Thought)
///
/// Mecanismo de defesa em profundidade focado na prevenÃ§Ã£o da "ObfuscaÃ§Ã£o de RaciocÃ­nio".
/// Monitora a produÃ§Ã£o sequencial de Chain of Thought dos Modelos de RaciocÃ­nio (ex: o1, R1).
/// Se o texto se mantÃ©m benigno mas as ativaÃ§Ãµes latentes apontam desvios (via ELK),
/// o monitor aciona a contenÃ§Ã£o prevenindo manipulaÃ§Ã£o encoberta.

pub struct CoTMonitor {
    /// NÃºmero de insights latentes consecutivos identificados como dissimulados.
    pub consecutive_lies: usize,
    /// Limite permissivo antes de soar o alarme definitivo (impede falso positivo por spike).
    pub obfuscation_threshold: usize,
}

impl Default for CoTMonitor {
    fn default() -> Self {
        Self::new(3)
    }
}

impl CoTMonitor {
    pub fn new(threshold: usize) -> Self {
        Self {
            consecutive_lies: 0,
            obfuscation_threshold: threshold,
        }
    }

    /// Avalia um Ãºnico passo (token ou frase) da CoT durante a inferÃªncia.
    /// Retorna `true` se o processo deve ser INTERROMPIDO imediatamente.
    pub fn evaluate_step(
        &mut self, 
        hidden_state: &[f32], 
        sae: &mut SAEEngine, 
        elk: &ElkProbe
    ) -> Result<HonestyVerdict, CoTObfuscationError> {
        let verdict = elk.probe_honesty(hidden_state, sae);

        if !verdict.is_honest {
            self.consecutive_lies += 1;
        } else {
            // Decay orgÃ¢nico: se for apenas um spike ruidoso, ele se recupera.
            self.consecutive_lies = self.consecutive_lies.saturating_sub(1);
        }

        if self.consecutive_lies >= self.obfuscation_threshold {
            return Err(CoTObfuscationError {
                detected_score: verdict.score,
                sequence_length: self.consecutive_lies,
            });
        }

        Ok(verdict)
    }

    pub fn reset(&mut self) {
        self.consecutive_lies = 0;
    }
}

#[derive(Debug, PartialEq)]
pub struct CoTObfuscationError {
    pub detected_score: f32,
    pub sequence_length: usize,
}

impl std::fmt::Display for CoTObfuscationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, 
            "âš ï¸ ALARME DE OBFUSCAÃ‡ÃƒO CoT: MÃºltiplas ativaÃ§Ãµes consecutivas dissimuladas. Score final {:.2} apÃ³s {} iteraÃ§Ãµes.", 
            self.detected_score, self.sequence_length
        )
    }
}

impl std::error::Error for CoTObfuscationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cot_monitor_catches_obfuscation() {
        let sae = SAEEngine::new(64, 128, 0.5);
        let mut elk = ElkProbe::new(128);
        
        // Manipulamos os pesos do ELK no teste para simular treino.
        // A _feature 10_ no dicionario SAE vira nosso atestado de mentira (peso hiper-negativo)
        elk.weights[10] = -5.0; 

        // SAE Mock para disparar a feature 10
        // Como MOCK encode() Ã© um dot-product bruto, W_enc * h > threshold dispara.
        let mut mock_sae = sae;
        mock_sae.encoder_weights[10 * 64 + 5] = 10.0;
        
        let mut monitor = CoTMonitor::new(3);

        // Simulando fluxo de passos 
        let mut h_honest = vec![0.0; 64]; // Feature 10 nao acende -> logit 0.0 -> score 0.5 (honesto)
        let mut h_lie = vec![0.0; 64]; h_lie[5] = 1.0; // Feature 10 acende -> score cai pra ~0.0

        // Passo 1: Honesto
        assert!(monitor.evaluate_step(&h_honest, &mut mock_sae, &elk).is_ok());
        
        // Passo 2: Mentira
        assert!(monitor.evaluate_step(&h_lie, &mut mock_sae, &elk).is_ok());
        assert_eq!(monitor.consecutive_lies, 1);
        
        // Passo 3: Mentira (2 consecutivas permitidas)
        assert!(monitor.evaluate_step(&h_lie, &mut mock_sae, &elk).is_ok());
        
        // Passo 4: O Abismo (3 consecutivas -> Dispara o alarme de contenÃ§Ã£o)
        let alarm = monitor.evaluate_step(&h_lie, &mut mock_sae, &elk);
        assert!(alarm.is_err(), "O detector de obfuscaÃ§Ã£o CoT nÃ£o disparou no threshold esperado!");
        
        println!("{}", alarm.err().unwrap());
    }
}

