//! D5 — Nash Tribunal: Verificação como Jogo Adversarial
//!
//! 3 agentes debatem cada hipótese até atingir Equilíbrio de Nash.
//! A verdade emerge do conflito, não da autoridade.
//!
//! - Advogado: tenta PROVAR a hipótese
//! - Promotor: tenta DEMOLIR a hipótese
//! - Juiz: arbitra com base em evidências

/// Uma hipótese gerada pelo Dreaming Engine
#[derive(Debug, Clone)]
pub struct DreamHypothesis {
    pub id: u64,
    pub statement: String,
    pub embedding: Vec<f32>,
    pub domain: String,
    pub confidence_prior: f32,
}

/// Um pedaço de evidência recuperado do LanceDB
#[derive(Debug, Clone)]
pub struct Evidence {
    pub source: String,
    pub content: String,
    pub relevance_score: f32,
    pub supports_hypothesis: bool,
}

/// Um round de debate no Tribunal de Nash
#[derive(Debug, Clone)]
pub struct DebateRound {
    pub round: usize,
    /// Argumento do Advogado (embedding de argumento)
    pub advocate_argument: String,
    /// Refutação do Promotor
    pub prosecutor_rebuttal: String,
    /// Score do Juiz (-1.0 = promotor vence, +1.0 = advogado vence)
    pub judge_score: f32,
}

/// Veredicto final do Tribunal Nash
#[derive(Debug, Clone)]
pub struct NashVerdict {
    pub hypothesis_id: u64,
    /// True = hipótese aceita, False = rejeitada
    pub accepted: bool,
    /// Confiança do veredicto [0, 1]
    pub confidence: f32,
    /// Veredito do ELK Probe (Evidência Material e Fisiológica)
    pub elk_honesty_score: Option<f32>,
    /// Log completo do debate (audit trail)
    pub debate_log: Vec<DebateRound>,
    /// Evidências usadas
    pub supporting_evidence: usize,
    pub opposing_evidence: usize,
}

/// O Agente Advogado — constrói argumentos a favor
struct AdvocateAgent;

impl AdvocateAgent {
    fn argue(&self, hypothesis: &DreamHypothesis, evidence: &[Evidence], round: usize) -> String {
        let supporting: Vec<_> = evidence.iter()
            .filter(|e| e.supports_hypothesis)
            .collect();
        if supporting.is_empty() {
            format!(
                "Round {}: A hipótese '{}' é plausível no domínio '{}' com confiança prévia {:.2}",
                round, hypothesis.statement, hypothesis.domain, hypothesis.confidence_prior
            )
        } else {
            format!(
                "Round {}: {} evidências suportam '{}': {}",
                round,
                supporting.len(),
                hypothesis.statement,
                supporting.iter()
                    .map(|e| e.source.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// O Agente Promotor — destrói hipóteses sem evidência
struct ProsecutorAgent;

impl ProsecutorAgent {
    fn rebut(&self, hypothesis: &DreamHypothesis, evidence: &[Evidence], round: usize) -> String {
        let opposing: Vec<_> = evidence.iter()
            .filter(|e| !e.supports_hypothesis)
            .collect();
        if opposing.is_empty() {
            format!(
                "Round {}: Sem evidências diretas para '{}'. Confiança baseline insuficiente.",
                round, hypothesis.statement
            )
        } else {
            format!(
                "Round {}: {} evidências contradizem '{}': {}",
                round,
                opposing.len(),
                hypothesis.statement,
                opposing.iter()
                    .map(|e| e.source.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// O Agente Juiz — arbitra com base em evidências quantitativas
struct JudgeAgent;

impl JudgeAgent {
    fn arbitrate(
        &self,
        hypothesis: &DreamHypothesis,
        evidence: &[Evidence],
        round: usize,
    ) -> f32 {
        let supporting: f32 = evidence.iter()
            .filter(|e| e.supports_hypothesis)
            .map(|e| e.relevance_score)
            .sum();
        let opposing: f32 = evidence.iter()
            .filter(|e| !e.supports_hypothesis)
            .map(|e| e.relevance_score)
            .sum();

        // Score base: prior do modelo + balanço de evidências
        let evidence_balance = if supporting + opposing > 0.0 {
            (supporting - opposing) / (supporting + opposing)
        } else {
            0.0
        };

        // Reduz credibilidade com o número de rounds (consenso mais difícil com mais rounds)
        let round_penalty = 0.05 * round as f32;
        (hypothesis.confidence_prior * 0.3 + evidence_balance * 0.7 - round_penalty)
            .clamp(-1.0, 1.0)
    }

    /// O Juiz recebe a Evidência Fisiológica do ELK e aplica um override final se a rede estiver mentindo gravemente
    fn apply_physiological_evidence(&self, current_score: f32, elk_honesty_score: Option<f32>) -> f32 {
        if let Some(elk) = elk_honesty_score {
            // Se o score de honestidade for menor que 0.3, a IA foi flagrada em flagrante dissimulação
            if elk < 0.3 {
                return current_score - 1.0; // Punição extrema: Demissor imediato
            }
            // Se a IA for extremamente honesta, boost de credibilidade
            if elk > 0.8 {
                return current_score + 0.2;
            }
        }
        current_score
    }
}

/// O Tribunal Nash: 3 agentes adversariais até o equilíbrio
pub struct NashTribunal {
    advocate: AdvocateAgent,
    prosecutor: ProsecutorAgent,
    judge: JudgeAgent,
    pub max_debate_rounds: usize,
    /// Tolerância de convergência: score estável por N rounds = Nash
    pub stability_tolerance: f32,
    pub stability_window: usize,
}

impl NashTribunal {
    pub fn new(max_debate_rounds: usize) -> Self {
        Self {
            advocate: AdvocateAgent,
            prosecutor: ProsecutorAgent,
            judge: JudgeAgent,
            max_debate_rounds,
            stability_tolerance: 0.05,
            stability_window: 3,
        }
    }

    /// Verifica uma hipótese via debate adversarial
    pub fn verify_hypothesis(
        &self,
        hypothesis: &DreamHypothesis,
        evidence: &[Evidence],
        elk_honesty_score: Option<f32>,
    ) -> NashVerdict {
        let mut rounds = Vec::new();

        for r in 0..self.max_debate_rounds {
            let advocate_arg = self.advocate.argue(hypothesis, evidence, r);
            let prosecutor_reb = self.prosecutor.rebut(hypothesis, evidence, r);
            let mut judge_score = self.judge.arbitrate(hypothesis, evidence, r);

            // Evidência material do ELK Probe age no background do Juiz
            judge_score = self.judge.apply_physiological_evidence(judge_score, elk_honesty_score);

            rounds.push(DebateRound {
                round: r,
                advocate_argument: advocate_arg,
                prosecutor_rebuttal: prosecutor_reb,
                judge_score,
            });

            // Testa equilíbrio de Nash
            if self.is_at_equilibrium(&rounds) {
                break;
            }
        }

        let final_score = rounds.last().map(|r| r.judge_score).unwrap_or(0.0);
        let confidence = self.compute_confidence(&rounds);

        NashVerdict {
            hypothesis_id: hypothesis.id,
            accepted: final_score > 0.0,
            confidence,
            elk_honesty_score,
            debate_log: rounds,
            supporting_evidence: evidence.iter().filter(|e| e.supports_hypothesis).count(),
            opposing_evidence: evidence.iter().filter(|e| !e.supports_hypothesis).count(),
        }
    }

    /// Equilíbrio de Nash: score estável por `stability_window` rounds
    fn is_at_equilibrium(&self, rounds: &[DebateRound]) -> bool {
        if rounds.len() < self.stability_window {
            return false;
        }
        let recent = &rounds[rounds.len() - self.stability_window..];
        let scores: Vec<f32> = recent.iter().map(|r| r.judge_score).collect();
        let max_diff = scores.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        max_diff < self.stability_tolerance
    }

    /// Confiança: inversamente proporcional à variância do debate
    fn compute_confidence(&self, rounds: &[DebateRound]) -> f32 {
        if rounds.is_empty() {
            return 0.0;
        }
        let scores: Vec<f32> = rounds.iter().map(|r| r.judge_score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        // Alta variância = baixa confiança
        (1.0 - variance.sqrt()).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hypothesis(confidence: f32) -> DreamHypothesis {
        DreamHypothesis {
            id: 1,
            statement: "A gravidade e o custo de oportunidade compartilham estrutura".to_string(),
            embedding: vec![0.5, 0.5],
            domain: "Física".to_string(),
            confidence_prior: confidence,
        }
    }

    fn supporting_evidence() -> Vec<Evidence> {
        vec![
            Evidence {
                source: "Newton (1687)".into(),
                content: "F = G*m1*m2/r²".into(),
                relevance_score: 0.95,
                supports_hypothesis: true,
            },
            Evidence {
                source: "Samuelson (1948)".into(),
                content: "Minimização de custo".into(),
                relevance_score: 0.85,
                supports_hypothesis: true,
            },
        ]
    }

    fn opposing_evidence() -> Vec<Evidence> {
        vec![
            Evidence {
                source: "Counter et al.".into(),
                content: "Domínios distintos".into(),
                relevance_score: 0.7,
                supports_hypothesis: false,
            },
        ]
    }

    #[test]
    fn test_elk_honesty_veto() {
        let tribunal = NashTribunal::new(5);
        let hyp = make_hypothesis(0.9); // Alta confiança
        let evidence = supporting_evidence(); // Tem evidência a favor
        
        // Porem o ELK detecta mentira/obfuscação
        let verdict = tribunal.verify_hypothesis(&hyp, &evidence, Some(0.1));
        assert!(!verdict.accepted, "Juiz deve vetar a hipotese mesmo embasada, se o ELK flagrar desonestidade");
    }

    #[test]
    fn test_hypothesis_accepted_with_strong_evidence() {
        let tribunal = NashTribunal::new(5);
        let hyp = make_hypothesis(0.8);
        let evidence = supporting_evidence();
        let verdict = tribunal.verify_hypothesis(&hyp, &evidence, None);
        assert!(verdict.accepted, "Hipótese com evidência forte deveria ser aceita");
        assert!(verdict.confidence > 0.5);
    }

    #[test]
    fn test_hypothesis_rejected_no_evidence() {
        let tribunal = NashTribunal::new(5);
        let hyp = make_hypothesis(0.1); // Baixa confiança prévia
        let verdict = tribunal.verify_hypothesis(&hyp, &[], None);
        assert!(!verdict.accepted, "Hipótese sem evidência deveria ser rejeitada");
    }

    #[test]
    fn test_debate_audit_trail_populated() {
        let tribunal = NashTribunal::new(3);
        let hyp = make_hypothesis(0.5);
        let evidence = [supporting_evidence(), opposing_evidence()].concat();
        let verdict = tribunal.verify_hypothesis(&hyp, &evidence, None);
        assert!(!verdict.debate_log.is_empty(), "Audit trail deve ter entradas");
        // Cada round deve ter argumento e refutação
        for round in &verdict.debate_log {
            assert!(!round.advocate_argument.is_empty());
            assert!(!round.prosecutor_rebuttal.is_empty());
        }
    }

    #[test]
    fn test_tie_scenario() {
        let tribunal = NashTribunal::new(5);
        let hyp = make_hypothesis(0.5);
        // Evidência balanceada
        let mut evidence = supporting_evidence();
        evidence.extend(opposing_evidence());
        let verdict = tribunal.verify_hypothesis(&hyp, &evidence, None);
        // Não importa o resultado, mas deve terminar
        assert!(verdict.confidence >= 0.0);
        assert!(!verdict.debate_log.is_empty());
    }
}
