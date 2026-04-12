//! D11 — Intent Compiler: Linguagem Natural → Plano Davi
//!
//! Converte intenções humanas em sequências de módulos Davi.
//! "Explore física quântica" → [Topology, FreeEnergy, Annealing(T=8), Nash, Stigmergy]

/// Um passo individual no plano Davi
#[derive(Debug, Clone, PartialEq)]
pub enum DaviStep {
    /// Análise topológica do espaço de conhecimento
    TopologyAnalysis { filtration_scale: f32 },
    /// Consulta ao Free Energy para priorização
    FreeEnergyPrioritization { precision: f32 },
    /// Salto semântico com temperatura específica
    AnnealingJump { temperature: f32 },
    /// Geração de hipótese sobre um domínio
    HypothesisGeneration { domain: String },
    /// Verificação via Nash Tribunal
    NashVerification { max_rounds: usize },
    /// Tradução cross-domain via Funtor
    FunctorTranslation { source_domain: String, target_domain: String },
    /// Depósito de feromônio no Swarm
    PheromoneDeposit { intensity: f32 },
    /// Loop autopoiético de otimização
    AutopoiesisOptimize,
    /// Ciclo completo de sonho
    FullDreamCycle,
}

/// Um plano de execução compilado do Davi
#[derive(Debug, Clone)]
pub struct DaviPlan {
    /// Intenção original em linguagem natural
    pub original_intent: String,
    /// Sequência de passos a executar
    pub steps: Vec<DaviStep>,
    /// Custo estimado em ms
    pub estimated_cost_ms: u64,
    /// Domínio(s) de interesse detectados
    pub detected_domains: Vec<String>,
    /// Confiança na interpretação (0-1)
    pub interpretation_confidence: f32,
}

impl DaviPlan {
    pub fn describe(&self) -> String {
        let steps_desc: Vec<String> = self.steps.iter()
            .enumerate()
            .map(|(i, s)| format!("  {}. {:?}", i + 1, s))
            .collect();
        format!(
            "Intent: '{}'\nDomains: {:?}\nSteps ({}):\n{}\nCost: ~{}ms | Confidence: {:.0}%",
            self.original_intent,
            self.detected_domains,
            self.steps.len(),
            steps_desc.join("\n"),
            self.estimated_cost_ms,
            self.interpretation_confidence * 100.0,
        )
    }
}

/// O compilador de intenção: converte linguagem natural em plano Davi
pub struct IntentCompiler {
    /// Vocabulário de domínios suportados
    domain_keywords: Vec<(Vec<&'static str>, &'static str)>,
    /// Verbos de ação que mapeiam para estratégias
    action_keywords: Vec<(&'static str, IntentType)>,
}

#[derive(Debug, Clone, PartialEq)]
enum IntentType {
    Explore,
    Verify,
    Connect,
    Optimize,
    Summarize,
}

impl IntentCompiler {
    pub fn new() -> Self {
        Self {
            domain_keywords: vec![
                (vec!["física", "fisica", "physics", "quântica", "quantum", "gravitação", "gravidade"], "Fisica"),
                (vec!["biologia", "biology", "genética", "genetica", "célula", "celula", "dna"], "Biologia"),
                (vec!["economia", "economia", "economics", "mercado", "custo", "finanças"], "Economia"),
                (vec!["matemática", "matematica", "math", "topologia", "álgebra", "calculo"], "Matematica"),
                (vec!["filosofia", "philosophy", "ética", "etica", "consciência"], "Filosofia"),
                (vec!["computação", "computacao", "software", "algoritmo", "ia", "inteligência"], "Computacao"),
            ],
            action_keywords: vec![
                ("explore", IntentType::Explore),
                ("explorar", IntentType::Explore),
                ("explorar", IntentType::Explore),
                ("descobrir", IntentType::Explore),
                ("discover", IntentType::Explore),
                ("verify", IntentType::Verify),
                ("verificar", IntentType::Verify),
                ("confirmar", IntentType::Verify),
                ("connect", IntentType::Connect),
                ("conectar", IntentType::Connect),
                ("relacionar", IntentType::Connect),
                ("optimize", IntentType::Optimize),
                ("otimizar", IntentType::Optimize),
                ("melhorar", IntentType::Optimize),
                ("summarize", IntentType::Summarize),
                ("resumir", IntentType::Summarize),
                ("sumarizar", IntentType::Summarize),
            ],
        }
    }

    /// Compila uma intenção em linguagem natural para um DaviPlan
    pub fn compile(&self, intent: &str) -> DaviPlan {
        let lower = intent.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Detecta domínios
        let mut detected_domains = Vec::new();
        for (keywords, domain) in &self.domain_keywords {
            if keywords.iter().any(|k| lower.contains(k)) {
                detected_domains.push(domain.to_string());
            }
        }

        // Detecta intenção/ação
        let intent_type = self.action_keywords.iter()
            .find(|(kw, _)| words.contains(kw))
            .map(|(_, t)| t.clone())
            .unwrap_or(IntentType::Explore); // Default: explorar

        // Detecta temperatura de criatividade
        let temperature = if lower.contains("criativo") || lower.contains("creative") || lower.contains("inovador") {
            8.0
        } else if lower.contains("preciso") || lower.contains("careful") || lower.contains("rigoroso") {
            0.5
        } else {
            3.0 // Balanceado
        };

        // Detecta cross-domain
        let wants_cross_domain = lower.contains("analog") || lower.contains("relacion") ||
            lower.contains("conect") || detected_domains.len() > 1;

        // Compila o plano baseado na intenção
        let steps = self.compile_steps(
            &intent_type,
            &detected_domains,
            temperature,
            wants_cross_domain,
        );

        let estimated_cost_ms = steps.iter().map(|s| match s {
            DaviStep::TopologyAnalysis { .. } => 50,
            DaviStep::FreeEnergyPrioritization { .. } => 10,
            DaviStep::AnnealingJump { .. } => 5,
            DaviStep::HypothesisGeneration { .. } => 20,
            DaviStep::NashVerification { max_rounds } => 30 * (*max_rounds as u64),
            DaviStep::FunctorTranslation { .. } => 40,
            DaviStep::PheromoneDeposit { .. } => 5,
            DaviStep::AutopoiesisOptimize => 15,
            DaviStep::FullDreamCycle => 200,
        }).sum();

        let confidence = if detected_domains.is_empty() { 0.5 } else { 0.85 };

        DaviPlan {
            original_intent: intent.to_string(),
            steps,
            estimated_cost_ms,
            detected_domains,
            interpretation_confidence: confidence,
        }
    }

    fn compile_steps(
        &self,
        intent: &IntentType,
        domains: &[String],
        temperature: f32,
        cross_domain: bool,
    ) -> Vec<DaviStep> {
        let mut steps = Vec::new();
        let domain = domains.first().cloned().unwrap_or_else(|| "Geral".to_string());

        match intent {
            IntentType::Explore => {
                steps.push(DaviStep::TopologyAnalysis { filtration_scale: 2.0 });
                steps.push(DaviStep::FreeEnergyPrioritization { precision: 0.5 });
                steps.push(DaviStep::AnnealingJump { temperature });
                steps.push(DaviStep::HypothesisGeneration { domain: domain.clone() });
                steps.push(DaviStep::NashVerification { max_rounds: 3 });
                steps.push(DaviStep::PheromoneDeposit { intensity: 1.0 });
            }
            IntentType::Verify => {
                steps.push(DaviStep::HypothesisGeneration { domain: domain.clone() });
                steps.push(DaviStep::NashVerification { max_rounds: 5 });
                steps.push(DaviStep::AutopoiesisOptimize);
            }
            IntentType::Connect => {
                steps.push(DaviStep::TopologyAnalysis { filtration_scale: 3.0 });
                if cross_domain && domains.len() >= 2 {
                    steps.push(DaviStep::FunctorTranslation {
                        source_domain: domains[0].clone(),
                        target_domain: domains[1].clone(),
                    });
                }
                steps.push(DaviStep::AnnealingJump { temperature: temperature * 1.5 });
                steps.push(DaviStep::PheromoneDeposit { intensity: 2.0 });
            }
            IntentType::Optimize => {
                steps.push(DaviStep::FreeEnergyPrioritization { precision: 1.0 });
                steps.push(DaviStep::AutopoiesisOptimize);
                steps.push(DaviStep::AnnealingJump { temperature: temperature * 0.5 });
            }
            IntentType::Summarize => {
                steps.push(DaviStep::TopologyAnalysis { filtration_scale: 1.0 });
                steps.push(DaviStep::FreeEnergyPrioritization { precision: 0.8 });
                steps.push(DaviStep::NashVerification { max_rounds: 2 });
            }
        }

        if cross_domain && !matches!(intent, IntentType::Connect) && domains.len() >= 2 {
            steps.push(DaviStep::FunctorTranslation {
                source_domain: domains[0].clone(),
                target_domain: domains[1].clone(),
            });
        }

        steps
    }
}

impl Default for IntentCompiler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_explore_physics() {
        let compiler = IntentCompiler::new();
        let plan = compiler.compile("explore física quântica");
        assert!(plan.detected_domains.contains(&"Fisica".to_string()));
        assert!(!plan.steps.is_empty());
        assert!(plan.steps.iter().any(|s| matches!(s, DaviStep::TopologyAnalysis { .. })));
        assert!(plan.steps.iter().any(|s| matches!(s, DaviStep::NashVerification { .. })));
        assert!(plan.estimated_cost_ms > 0);
    }

    #[test]
    fn test_compile_verify_hypothesis() {
        let compiler = IntentCompiler::new();
        let plan = compiler.compile("verificar hipótese sobre biologia");
        assert!(plan.detected_domains.contains(&"Biologia".to_string()));
        // Plano de verificação deve ter Nash Tribunal
        assert!(plan.steps.iter().any(|s| matches!(s, DaviStep::NashVerification { max_rounds: 5 })));
    }
}
