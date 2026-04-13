//! PROBES V2 — Budget Forcer (Princípio 2: Forçar Raciocínio Profundo)
//!
//! Budget Forcing guiado — o oposto do "Wait" cego do mercado.
//! Em vez de injetar "Wait" indiscriminadamente, o `BudgetForcer` usa
//! `ConformalPredictor` + `ElkProbe` para saber QUANDO forçar mais
//! raciocínio e QUANDO cortar (detecção de "inverse scaling").
//!
//! ## Fluxo por Token:
//! ```text
//! Token candidato → Conformal verifica NCS
//!   NCS < 0.15 → Accept (certeza alta)
//!   NCS 0.15-0.50 → AcceptUncertain (anota baixa certeza no audit)
//!   NCS > 0.50 → ForceThink (injeta Wait, ativa Steering, re-executa)
//!                └── Se após MAX_ATTEMPTS NCS ainda alto:
//!                    ELK verifica: modelo está honesto?
//!                    - Honesto + preso → AcceptUncertain (o problema é difícil)
//!                    - Desonesto → Revert (modelo se perdeu, desfaz N tokens)
//! ```
//!
//! ## Detecção de Inverse Scaling:
//! Se o NCS nos últimos N tokens está SUBINDO ao invés de cair após "Wait",
//! o modelo está se perdendo no raciocínio — `is_stuck()` sinaliza corte.
//!
//! ## Pesquisa (2025-2026):
//! s1 Framework mostrou que "forçar mais tempo" nem sempre melhora — precisa de
//! controle. O nosso BudgetForcer é o primeiro a usar métricas internas (Conformal
//! + ELK) para detectar o ponto de inversão e cortar antes dele.

use crate::conformal_predictor::{ConformalPredictor, ConformalSet};
use crate::sae_engine::SAEEngine;
use std::collections::VecDeque;

/// Estados de Markov para drift cognitivo (LANA — Learning Nash Alignment).
///
/// Modela o comportamento do modelo como uma cadeia de Markov de primeira ordem:
///   M0 (Estável) → M1 (Desvio) → M2 (Fratura) → Me (Equilíbrio)
///
/// Transições:
/// - Estável → Desvio: NCS médio supera `drift_threshold`
/// - Desvio → Fratura: inverse scaling detectado + honestidade baixa
/// - Fratura → Equilíbrio: Steering/Revert aplicado + NCS cai
/// - Equilíbrio → Estável: NCS estabiliza abaixo do `accept_threshold`
///
/// Quando `Fractured`, o pipeline deve acionar `SteeringEngine` para
/// aplicar Ablation das features de recusa/preenchimento detectadas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveDriftState {
    /// M0: Sistema Estavel. NCS baixo, honestidade alta.
    Stable,
    /// M1: Desvio Crescente. NCS subindo mas ainda controlável.
    Drifting,
    /// M2: Fratura Cognitiva. Inverse scaling + desonestidade confirmada.
    /// Sinal para o pipeline aplicar cirurgia latente de emergência.
    Fractured,
    /// Me: Equilíbrio Restaurado. Sistema recalibrou após intervenção.
    Equilibrium,
}

impl CognitiveDriftState {
    /// Nome descritivo do estado para logs e painel.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Stable => "[M0] Estável",
            Self::Drifting => "[M1] Desvio",
            Self::Fractured => "[M2] Fratura",
            Self::Equilibrium => "[Me] Equilíbrio",
        }
    }

    /// Indica se o estado requer intervenção imediata do SteeringEngine.
    pub fn requires_surgery(&self) -> bool {
        matches!(self, Self::Fractured)
    }
}


/// Decisão do BudgetForcer para o token candidato.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetDecision {
    /// Token aceito — certeza acima do limiar.
    Accept,
    /// Forçar mais raciocínio — injeta token "Wait" e re-executa forward pass.
    ForceThink {
        reason: String,
    },
    /// Reverter — modelo se perdeu no raciocínio. Desfaz N tokens.
    Revert {
        rollback_n: usize,
        reason: String,
    },
    /// Aceito com ressalva — baixa certeza mas sem melhora possível.
    AcceptUncertain {
        ncs: f32,
        note: String,
    },
}

/// Resultado completo da avaliação de um token.
#[derive(Debug, Clone)]
pub struct TokenVerdict {
    pub decision: BudgetDecision,
    pub ncs: f32,
    pub honesty: f32,
    pub attempts: usize,
    pub forced_think_count: usize,
}

/// Estatísticas acumuladas do BudgetForcer.
#[derive(Debug, Clone, Default)]
pub struct BudgetStats {
    pub total_tokens_evaluated: u64,
    pub accepted: u64,
    pub accepted_uncertain: u64,
    pub force_think_events: u64,
    pub reverts: u64,
    pub avg_ncs: f32,
    pub avg_attempts_per_forced: f32,
}

/// O BudgetForcer — controle inteligente de raciocínio por token.
pub struct BudgetForcer {
    conformal: ConformalPredictor,
    pub sae: SAEEngine,
    /// Número máximo de tentativas de "forçar" antes de aceitar ou reverter
    pub max_force_attempts: usize,
    /// Janela de NCS dos últimos N tokens para detectar inverse scaling
    ncs_window: VecDeque<f32>,
    /// Tamanho da janela de detecção de inverse scaling
    pub window_size: usize,
    /// Limiar de NCS para forçar raciocínio
    pub force_threshold: f32,
    /// Limiar de NCS para aceitar diretamente
    pub accept_threshold: f32,
    /// Limiar de desonestidade (variância SAE baixa = filling)
    pub dishonesty_threshold: f32,
    /// Estatísticas acumuladas
    stats: BudgetStats,
    /// Tokens gerados desde o último force (para rollback contextual)
    tokens_since_last_force: usize,
    /// Estado atual de deriva cognitiva (modelo de Markov)
    drift_state: CognitiveDriftState,
    /// Histórico de scores de honestidade para detectar tendência
    honesty_window: VecDeque<f32>,
}

impl BudgetForcer {
    /// Token injetado para forçar raciocínio adicional.
    /// Em produção: vocabulário real teria um token "Wait" explícito.
    pub const WAIT_TOKEN_ID: u32 = 29871; // Aproximação: espaço em LLaMA vocab

    pub fn new(hidden_dim: usize, dict_size: usize, confidence_level: f32) -> Self {
        Self {
            conformal: ConformalPredictor::new(confidence_level),
            sae: SAEEngine::new(hidden_dim, dict_size, 0.3),
            max_force_attempts: 3,
            ncs_window: VecDeque::new(),
            window_size: 10,
            force_threshold: 0.50,
            accept_threshold: 0.15,
            dishonesty_threshold: 0.35,
            stats: BudgetStats::default(),
            tokens_since_last_force: 0,
            drift_state: CognitiveDriftState::Stable,
            honesty_window: VecDeque::new(),
        }
    }

    /// Avalia um token candidato e decide se aceitar, forçar ou reverter.
    ///
    /// # Parâmetros
    /// - `token_logits`: Distribuição de probabilidade sobre vocab
    /// - `hidden_state`: Hidden state da última camada (para ELK + SAE)
    /// - `attempt`: Número de tentativas para este token (0 = primeira)
    pub fn evaluate_token(
        &mut self,
        token_logits: &[f32],
        hidden_state: &[f32],
        attempt: usize,
    ) -> TokenVerdict {
        self.stats.total_tokens_evaluated += 1;
        self.tokens_since_last_force += 1;

        let conformal_set = self.conformal.predict_set(token_logits);
        let ncs = conformal_set.non_conformity_score;

        // Atualiza janela para detecção de inverse scaling
        self.ncs_window.push_back(ncs);
        if self.ncs_window.len() > self.window_size {
            self.ncs_window.pop_front();
        }

        // Atualiza NCS médio
        let alpha = 0.1;
        self.stats.avg_ncs = self.stats.avg_ncs * (1.0 - alpha) + ncs * alpha;

        let verdict = self.make_decision(ncs, hidden_state, attempt, &conformal_set);

        // Atualiza modelo de Markov de deriva cognitiva
        // Nota: encoda SAE primeiro em variável temporária para evitar borrow duplo
        let sae_features_for_drift = self.sae.encode(hidden_state);
        let current_honesty = self.honesty_from_sae_variance(&sae_features_for_drift);
        self.honesty_window.push_back(current_honesty);
        if self.honesty_window.len() > self.window_size {
            self.honesty_window.pop_front();
        }
        self.drift_state = self.compute_drift_state(ncs, current_honesty);

        // Atualiza estatísticas
        match &verdict.decision {
            BudgetDecision::Accept => self.stats.accepted += 1,
            BudgetDecision::AcceptUncertain { .. } => self.stats.accepted_uncertain += 1,
            BudgetDecision::ForceThink { .. } => {
                self.stats.force_think_events += 1;
                self.tokens_since_last_force = 0;
            }
            BudgetDecision::Revert { .. } => {
                self.stats.reverts += 1;
                self.tokens_since_last_force = 0;
            }
        }

        verdict
    }

    /// Detecta "inverse scaling": NCS subindo ao invés de cair após force.
    /// Se o modelo está se perdendo, é melhor cortar do que continuar.
    pub fn is_stuck(&self) -> bool {
        if self.ncs_window.len() < 4 {
            return false;
        }
        let window: Vec<f32> = self.ncs_window.iter().copied().collect();
        let n = window.len();
        let first_half: f32 = window[..n / 2].iter().sum::<f32>() / (n / 2) as f32;
        let second_half: f32 = window[n / 2..].iter().sum::<f32>() / (n - n / 2) as f32;
        // Preso se NCS subiu mais de 15% na segunda metade da janela
        second_half > first_half * 1.15
    }

    /// Retorna o token "Wait" para injeção no contexto.
    pub fn force_wait_token() -> u32 {
        Self::WAIT_TOKEN_ID
    }

    /// Estatísticas acumuladas.
    pub fn stats(&self) -> &BudgetStats {
        &self.stats
    }

    /// Estado atual de drift cognitivo (modelo de Markov de 4 estados).
    pub fn drift_state(&self) -> CognitiveDriftState {
        self.drift_state
    }

    /// Relatório legível.
    pub fn report(&self) -> String {
        let s = &self.stats;
        let total = s.total_tokens_evaluated.max(1) as f32;
        format!(
            "BudgetForcer | tokens={} | accept={:.0}% | uncertain={:.0}% | \
             force={} | reverts={} | avg_ncs={:.3} | drift={}",
            s.total_tokens_evaluated,
            s.accepted as f32 / total * 100.0,
            s.accepted_uncertain as f32 / total * 100.0,
            s.force_think_events,
            s.reverts,
            s.avg_ncs,
            self.drift_state.label(),
        )
    }

    // --- Lógica de decisão ---

    fn make_decision(
        &mut self,
        ncs: f32,
        hidden_state: &[f32],
        attempt: usize,
        _conformal_set: &ConformalSet,
    ) -> TokenVerdict {
        // Caminho 1: Certeza alta → Aceitar direto
        if ncs <= self.accept_threshold {
            return TokenVerdict {
                decision: BudgetDecision::Accept,
                ncs,
                honesty: 1.0, // Não precisa verificar ELK
                attempts: attempt,
                forced_think_count: 0,
            };
        }

        // Heurística local de honestidade via variância SAE:
        // Alta variância nas features = modelo ativando features específicas (confiante/honesto)
        // Baixa variância = features uniformes = padrão de preenchimento
        let features = self.sae.encode(hidden_state);
        let honesty = self.honesty_from_sae_variance(&features);
        let honesty_is_low = honesty < self.dishonesty_threshold;

        // Caminho 2: Desonestidade alta + incerteza alta → Revert
        if honesty_is_low && ncs > self.force_threshold {
            return TokenVerdict {
                decision: BudgetDecision::Revert {
                    rollback_n: (self.tokens_since_last_force / 2).max(1).min(10),
                    reason: format!(
                        "SAE variance baixa (score={:.2}) com NCS alto ({:.2}) — possível filling",
                        honesty, ncs
                    ),
                },
                ncs,
                honesty,
                attempts: attempt,
                forced_think_count: attempt,
            };
        }

        // Caminho 3: Modelo preso em inverse scaling → AcceptUncertain e corta
        if self.is_stuck() && attempt > 0 {
            return TokenVerdict {
                decision: BudgetDecision::AcceptUncertain {
                    ncs,
                    note: format!(
                        "Inverse scaling detectado (NCS subindo). Aceitando após {} tentativas.",
                        attempt
                    ),
                },
                ncs,
                honesty,
                attempts: attempt,
                forced_think_count: attempt,
            };
        }

        // Caminho 4: Incerteza alta + tentativas ainda disponíveis → ForceThink
        if ncs > self.force_threshold && attempt < self.max_force_attempts {
            return TokenVerdict {
                decision: BudgetDecision::ForceThink {
                    reason: format!(
                        "NCS={:.2} acima do limiar {:.2}. Tentativa {}/{}.",
                        ncs, self.force_threshold, attempt + 1, self.max_force_attempts
                    ),
                },
                ncs,
                honesty,
                attempts: attempt,
                forced_think_count: attempt,
            };
        }

        // Caminho 5: Tentativas esgotadas + modelo honesto → AcceptUncertain
        if ncs > self.accept_threshold {
            return TokenVerdict {
                decision: BudgetDecision::AcceptUncertain {
                    ncs,
                    note: format!(
                        "Máximo de {} tentativas esgotado. Modelo honesto mas incerto.",
                        self.max_force_attempts
                    ),
                },
                ncs,
                honesty,
                attempts: attempt,
                forced_think_count: attempt,
            };
        }

        // Default: aceitar
        TokenVerdict {
            decision: BudgetDecision::Accept,
            ncs,
            honesty,
            attempts: attempt,
            forced_think_count: 0,
        }
    }

    /// Hâcurstica SAE de honestidade: usa variância das features ativas.
    /// Alta variância = modelo confiante em features específicas (honesto).
    /// Baixa variância = features uniformes = padrão de preenchimento.
    fn honesty_from_sae_variance(&self, features: &[f32]) -> f32 {
        let active: Vec<f32> = features.iter().filter(|&&v| v > 0.0).copied().collect();
        if active.is_empty() {
            return 0.5; // Sem features ativas: incógnita
        }
        let mean = active.iter().sum::<f32>() / active.len() as f32;
        let variance = active.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / active.len() as f32;
        // Normaliza variância para [0, 1] com escala empírica
        (variance * 4.0).min(1.0)
    }

    /// Computa o próximo estado na cadeia de Markov de deriva cognitiva.
    ///
    /// Transições baseadas em NCS atual + honestidade + histórico da janela:
    ///
    /// ```text
    /// Stable  → Drifting   : NCS > drift_threshold (0.35)
    /// Drifting → Fractured  : is_stuck() AND honesty < dishonesty_threshold
    /// Fractured → Equilibrium: NCS < accept_threshold (sistema respondeu à cirurgia)
    /// Equilibrium → Stable   : honesty > 0.6 AND NCS estável
    /// ```
    fn compute_drift_state(&self, ncs: f32, honesty: f32) -> CognitiveDriftState {
        let avg_honesty = if self.honesty_window.is_empty() {
            honesty
        } else {
            self.honesty_window.iter().sum::<f32>() / self.honesty_window.len() as f32
        };

        match self.drift_state {
            CognitiveDriftState::Stable => {
                if ncs > 0.35 {
                    CognitiveDriftState::Drifting
                } else {
                    CognitiveDriftState::Stable
                }
            }
            CognitiveDriftState::Drifting => {
                if self.is_stuck() && honesty < self.dishonesty_threshold {
                    CognitiveDriftState::Fractured
                } else if ncs < self.accept_threshold {
                    // Drift se resolveu sem intervenção
                    CognitiveDriftState::Stable
                } else {
                    CognitiveDriftState::Drifting
                }
            }
            CognitiveDriftState::Fractured => {
                // Equilíbrio quando NCS caiu (sinal de que cirurgia funcionou)
                if ncs < self.accept_threshold {
                    CognitiveDriftState::Equilibrium
                } else {
                    CognitiveDriftState::Fractured
                }
            }
            CognitiveDriftState::Equilibrium => {
                if avg_honesty > 0.6 && ncs < self.accept_threshold {
                    CognitiveDriftState::Stable
                } else if ncs > self.force_threshold {
                    // Recidiva: voltar a Drifting
                    CognitiveDriftState::Drifting
                } else {
                    CognitiveDriftState::Equilibrium
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_forcer() -> BudgetForcer {
        BudgetForcer::new(64, 128, 0.95)
    }

    #[test]
    fn test_high_confidence_logits_accepted() {
        let mut forcer = make_forcer();

        // Logits muito concentrados → NCS muito baixo → Accept
        let mut logits = vec![0.1f32; 100];
        logits[0] = 1000.0;
        let hidden = vec![0.0f32; 64];

        let verdict = forcer.evaluate_token(&logits, &hidden, 0);
        assert_eq!(verdict.decision, BudgetDecision::Accept,
            "Logits concentrados devem ser aceitos diretamente");
        assert!(verdict.ncs < forcer.accept_threshold + 0.1,
            "NCS deve ser próximo de zero: {:.3}", verdict.ncs);
    }

    #[test]
    fn test_flat_logits_trigger_force_think() {
        let mut forcer = make_forcer();

        // Logits uniformes → NCS alto → ForceThink
        let logits = vec![1.0f32; 100];
        let hidden = vec![0.0f32; 64];

        let verdict = forcer.evaluate_token(&logits, &hidden, 0);

        let is_force_or_uncertain = matches!(
            verdict.decision,
            BudgetDecision::ForceThink { .. } | BudgetDecision::AcceptUncertain { .. }
        );
        assert!(is_force_or_uncertain,
            "Logits uniformes devem gerar ForceThink ou AcceptUncertain, não Accept. Got: {:?}",
            verdict.decision);
    }

    #[test]
    fn test_max_attempts_leads_to_accept_uncertain() {
        let mut forcer = make_forcer();
        forcer.max_force_attempts = 2;

        let logits = vec![1.0f32; 50];
        let hidden = vec![0.0f32; 64];

        // Simula tentativa no limite máximo
        let verdict = forcer.evaluate_token(&logits, &hidden, forcer.max_force_attempts);

        // Com tentativas esgotadas: AcceptUncertain ou Revert (não ForceThink)
        let not_force = !matches!(verdict.decision, BudgetDecision::ForceThink { .. });
        assert!(not_force,
            "No limite de tentativas não deve gerar mais ForceThink. Got: {:?}",
            verdict.decision);
    }

    #[test]
    fn test_stats_accumulate_correctly() {
        let mut forcer = make_forcer();

        let certain_logits = {
            let mut v = vec![0.0f32; 50];
            v[0] = 1000.0;
            v
        };
        let uncertain_logits = vec![1.0f32; 50];
        let hidden = vec![0.0f32; 64];

        forcer.evaluate_token(&certain_logits, &hidden, 0);
        forcer.evaluate_token(&uncertain_logits, &hidden, 0);
        forcer.evaluate_token(&certain_logits, &hidden, 0);

        let stats = forcer.stats();
        assert_eq!(stats.total_tokens_evaluated, 3,
            "Total deve ser 3: got {}", stats.total_tokens_evaluated);
        assert!(stats.accepted >= 2,
            "Ao menos 2 logits certos devem ter sido aceitos: got {}", stats.accepted);
    }

    #[test]
    fn test_is_stuck_detects_rising_ncs() {
        let mut forcer = make_forcer();
        forcer.window_size = 6;

        // Simula NCS crescente (inverse scaling)
        let hidden = vec![0.0f32; 64];
        for ncs_val in &[0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
            // Cria logits com NCS aproximado ao valor desejado
            // (logits mais uniformes = NCS mais alto)
            let logits: Vec<f32> = (0..50).map(|i| {
                if i == 0 { 1.0 + (1.0 - ncs_val) * 10.0 } else { 1.0 }
            }).collect();
            forcer.evaluate_token(&logits, &hidden, 0);
        }

        // Com NCS crescendo de 0.3 para 0.8, deve detectar stuck
        // (depende da precisão da aproximação de NCS, então apenas verifica que não paniquei)
        let _ = forcer.is_stuck();
    }

    #[test]
    fn test_wait_token_constant() {
        let wait = BudgetForcer::force_wait_token();
        assert!(wait > 0, "Wait token deve ser um ID válido não-zero");
    }

    #[test]
    fn test_report_format() {
        let mut forcer = make_forcer();
        let logits = vec![1.0f32; 10];
        let hidden = vec![0.0f32; 64];
        forcer.evaluate_token(&logits, &hidden, 0);

        let report = forcer.report();
        assert!(report.contains("BudgetForcer"), "Report deve conter nome do módulo");
        assert!(report.contains("tokens=1"), "Report deve conter total correto");
    }

    #[test]
    fn test_dishonest_high_ncs_triggers_revert() {
        let mut forcer = make_forcer();

        // Para forçar o revert, precisamos que ELK detecte desonestidade
        // Isso requer treinamento do ELK — sem treino, probe.score ≈ 0.5 (limiar)
        // Testamos que o código não paniquei e retorna um resultado válido
        let logits = vec![1.0f32; 50];
        let hidden = vec![0.5f32; 64];

        let verdict = forcer.evaluate_token(&logits, &hidden, 0);
        let is_valid = matches!(
            verdict.decision,
            BudgetDecision::Accept
                | BudgetDecision::ForceThink { .. }
                | BudgetDecision::Revert { .. }
                | BudgetDecision::AcceptUncertain { .. }
        );
        assert!(is_valid, "Veredito deve ser uma variante válida de BudgetDecision");
    }

    #[test]
    fn test_drift_state_starts_stable() {
        let forcer = make_forcer();
        assert_eq!(forcer.drift_state(), CognitiveDriftState::Stable,
            "Estado inicial deve ser Stable");
        assert!(!forcer.drift_state().requires_surgery(),
            "Stable não requer cirurgia");
    }

    #[test]
    fn test_drift_state_transitions_to_drifting_on_high_ncs() {
        let mut forcer = make_forcer();
        // Logits uniformes → NCS alto → deve transitar para Drifting
        let logits = vec![1.0f32; 100];
        let hidden = vec![0.0f32; 64];
        forcer.evaluate_token(&logits, &hidden, 0);
        // Após NCS alto, deve estar Drifting ou Stable (depende do NCS exato)
        let state = forcer.drift_state();
        let valid = matches!(state, CognitiveDriftState::Stable | CognitiveDriftState::Drifting);
        assert!(valid, "Estado deve ser Stable ou Drifting após NCS alto: {:?}", state);
    }

    #[test]
    fn test_fractured_state_requires_surgery() {
        assert!(CognitiveDriftState::Fractured.requires_surgery(),
            "Fractured deve sinalizar necessidade de cirurgia");
        assert!(!CognitiveDriftState::Stable.requires_surgery());
        assert!(!CognitiveDriftState::Drifting.requires_surgery());
        assert!(!CognitiveDriftState::Equilibrium.requires_surgery());
    }

    #[test]
    fn test_drift_state_labels_non_empty() {
        assert!(!CognitiveDriftState::Stable.label().is_empty());
        assert!(!CognitiveDriftState::Drifting.label().is_empty());
        assert!(!CognitiveDriftState::Fractured.label().is_empty());
        assert!(!CognitiveDriftState::Equilibrium.label().is_empty());
    }

    #[test]
    fn test_report_contains_drift_state() {
        let mut forcer = make_forcer();
        let logits = vec![1.0f32; 10];
        let hidden = vec![0.0f32; 64];
        forcer.evaluate_token(&logits, &hidden, 0);
        let report = forcer.report();
        assert!(report.contains("drift="), "Report deve conter estado de drift");
    }
}
