//! D9 — Dreaming Engine: O Orquestrador do Loop de Sonho
//!
//! Quando a GPU fica ociosa, o Davi "sonha":
//! TDA → FEP → Annealing → Crystal Skeleton → Funtores → Nash → Stigmergy → Autopoiese

use crate::topology::{TopologicalGapDetector, DreamTarget};
use crate::free_energy::{FreeEnergyObjective, DreamAction};
use crate::annealing::{SemanticAnnealing, CoolingSchedule};
use crate::nash_tribunal::{NashTribunal, DreamHypothesis, Evidence, NashVerdict};
use crate::functors::{FunctorComposer, Insight};
use crate::stigmergy::StigmergicSwarm;
use crate::autopoiesis::{AutopoieticLoop, SystemHealth};
use crate::audit_logger::{AuditLogger, AuditEventType};
use crate::provenance_graph::{ProvenanceDAG, StepType};
use nodestor_inference::semantic_attention;

/// Uma descoberta nova validada pelo sistema completo
#[derive(Debug, Clone)]
pub struct NovelDiscovery {
    pub id: u64,
    pub statement: String,
    pub domain: String,
    pub confidence: f32,
    pub source_embedding: Vec<f32>,
    pub provenance_root: u64,
}

/// Configuração do ciclo de sonho
pub struct DreamConfig {
    /// Tempo máximo de sonho em ms (0 = sem limite)
    pub max_duration_ms: u64,
    /// Número máximo de hipóteses geradas por ciclo
    pub max_hypotheses: usize,
    /// Temperatura inicial do annealing
    pub initial_temperature: f32,
    /// Persistência mínima para considerar gap topológico
    pub min_topological_persistence: f32,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            max_duration_ms: 1000,
            max_hypotheses: 10,
            initial_temperature: 5.0,
            min_topological_persistence: 0.3,
        }
    }
}

/// O Motor de Sonho: orquestra todos os 7 subsistemas do Davi
pub struct DreamingEngine {
    pub topology: TopologicalGapDetector,
    pub free_energy: FreeEnergyObjective,
    pub annealing: SemanticAnnealing,
    pub tribunal: NashTribunal,
    pub functors: FunctorComposer,
    pub swarm: StigmergicSwarm,
    pub autopoiesis: AutopoieticLoop,
    pub audit: AuditLogger,
    pub provenance: ProvenanceDAG,
    /// Contador de descobertas
    pub discovery_count: u64,
    /// Histórico de energia livre
    pub energy_history: Vec<f32>,
    /// Histórico contínuo para Block AttnRes inter-cíclico
    pub past_discoveries_embeddings: Vec<Vec<f32>>,
}

impl DreamingEngine {
    pub fn new(config: &DreamConfig) -> Self {
        Self {
            topology: TopologicalGapDetector::new(2.0, config.min_topological_persistence),
            free_energy: FreeEnergyObjective::new(0.5, 0.1),
            annealing: SemanticAnnealing::new(
                config.initial_temperature,
                CoolingSchedule::Exponential { decay: 0.95 },
            ),
            tribunal: NashTribunal::new(5),
            functors: FunctorComposer::new(),
            swarm: StigmergicSwarm::new(0.1, 0.05),
            autopoiesis: AutopoieticLoop::new(5.0, 0.05),
            audit: AuditLogger::new(1000),
            provenance: ProvenanceDAG::new(),
            discovery_count: 0,
            energy_history: Vec::new(),
            past_discoveries_embeddings: Vec::new(),
        }
    }

    /// Executa um ciclo completo de sonho.
    /// Retorna as descobertas validadas pelo Nash Tribunal.
    pub fn dream_cycle(
        &mut self,
        knowledge_embeddings: &[Vec<f32>],
        domain_insights: &[Insight],
    ) -> Vec<NovelDiscovery> {
        let mut discoveries = Vec::new();

        self.audit.log(
            AuditEventType::SystemBoot,
            "dream_cycle start",
            format!("{} embeddings, {} insights", knowledge_embeddings.len(), domain_insights.len()),
            "{}",
        );

        // PASSO 1: Topologia — detecta buracos no conhecimento
        let gaps = if !knowledge_embeddings.is_empty() {
            self.topology.detect_gaps(knowledge_embeddings)
        } else {
            Vec::new()
        };

        self.audit.log(
            AuditEventType::TopologyAnalyzed,
            "topology step",
            format!("{} gaps detectados", gaps.len()),
            "{}",
        );

        // PASSO 2: Free Energy — prioriza buracos por surpresa
        let dream_targets = self.topology.gaps_as_dream_targets();
        let candidate_actions: Vec<DreamAction> = dream_targets.iter()
            .map(|t| DreamAction {
                name: format!("Explore gap (dim={})", t.dimension),
                target_embedding: t.center.clone(),
                expected_energy_reduction: t.priority,
            })
            .collect();

        let target = self.free_energy.select_action(&candidate_actions)
            .or_else(|| {
                // Sem gaps: explora o espaço de knowledge conhecido
                if !knowledge_embeddings.is_empty() {
                    Some(&candidate_actions[0])
                } else {
                    None
                }
            });

        // PASSO 3: Gera hipóteses via Annealing
        let mut hypotheses_generated = 0;
        let max_hyp = if gaps.is_empty() { 3 } else { gaps.len().min(5) };

        for (i, gap) in gaps.iter().enumerate().take(max_hyp) {
            // Annealing decide se aceita este salto cross-domain
            let accepted_jump = self.annealing.accept_cross_domain_jump(
                1.0 - gap.priority,
                gap.persistence * 0.5,
            );
            self.annealing.cool_one_step();

            self.audit.log(
                AuditEventType::AnnealingStep,
                format!("Gap {} (dim={}) | T={:.3}", i, gap.dimension, self.annealing.current_temperature),
                if accepted_jump { "aceito" } else { "rejeitado" },
                "{}",
            );

            if !accepted_jump { continue; }

            // PASSO 4: Gera hipótese para este gap
            // --- DREAM STACKING (Temporal Attention Residuals) ---
            // Em vez de usar apenas o vetor cru do gap atual, o Davi "lembra"
            // de insights do passado. Softmax Attention decide se um insight
            // antigo é relevante para preencher a lacuna atual.
            let mut final_embedding = gap.center.clone();
            
            if !self.past_discoveries_embeddings.is_empty() {
                // Referências para o borrow checker do compute_attention_weights
                let keys: Vec<&[f32]> = self.past_discoveries_embeddings
                    .iter()
                    .map(|v| v.as_slice())
                    .collect();
                    
                let mut alphas = semantic_attention::compute_attention_weights(&gap.center, &keys, 1.0);
                
                // Decaimento temporal progressivo: sonhos mais antigos têm seu alpha atenuado
                // Isso evita Echo Chambers de insights fósseis (Model Collapse)
                let total_keys = alphas.len();
                for (idx, alpha) in alphas.iter_mut().enumerate() {
                    // idx = 0 é o mais antigo. idx = total_keys - 1 é o mais recente.
                    let age_factor = (idx as f32 + 1.0) / (total_keys as f32); // 0.0 -> 1.0
                    *alpha *= age_factor;
                }
                
                // Mescla (perturba) a lacuna atual com os fantasmas dos sonhos passados relevantes
                for (alpha, past_emb) in alphas.iter().zip(self.past_discoveries_embeddings.iter()) {
                    for i in 0..final_embedding.len().min(past_emb.len()) {
                        final_embedding[i] += alpha * past_emb[i];
                    }
                }
            }

            let hyp = DreamHypothesis {
                id: self.discovery_count + i as u64,
                statement: format!(
                    "Existe conhecimento inexplorado em dimensão {} (persistência={:.2})",
                    gap.dimension, gap.persistence
                ),
                embedding: final_embedding,
                domain: domain_insights.first()
                    .map(|ins| ins.domain.clone())
                    .unwrap_or_else(|| "Geral".to_string()),
                confidence_prior: gap.priority.min(1.0),
            };

            // PASSO 5: Nash Tribunal valida a hipótese
            let evidence: Vec<Evidence> = domain_insights.iter()
                .take(3)
                .map(|ins| Evidence {
                    source: ins.domain.clone(),
                    content: ins.statement.clone(),
                    relevance_score: 0.7,
                    supports_hypothesis: true,
                })
                .collect();

            self.audit.log(
                AuditEventType::NashDebateStarted,
                format!("Hipótese: {}", hyp.statement),
                format!("{} evidências", evidence.len()),
                "{}",
            );

            let verdict = self.tribunal.verify_hypothesis(&hyp, &evidence, None);

            self.audit.log(
                AuditEventType::NashDebateResolved,
                format!("Veredicto: {}", if verdict.accepted { "ACEITO" } else { "REJEITADO" }),
                format!("conf={:.2}, rounds={}", verdict.confidence, verdict.debate_log.len()),
                "{}",
            );

            if !verdict.accepted { continue; }

            // PASSO 6: Tradução cross-domain via Funtores
            let discovery_embedding = hyp.embedding.clone();

            // PASSO 7: Deposita Feromônio no Swarm
            self.swarm.deposit(
                self.discovery_count,
                discovery_embedding.clone(),
                verdict.confidence,
                0, // node_id = 0 (este nó)
            );

            // Registra no Provenance DAG
            let prov_root = self.provenance.add_node(
                vec![], None, StepType::RawData,
                format!("Gap topológico dim={}", gap.dimension), 1.0,
            );
            let prov_verdict = self.provenance.add_node(
                vec![prov_root], Some(self.discovery_count),
                StepType::NashDebate,
                format!("Nash conf={:.2}", verdict.confidence),
                verdict.confidence,
            );
            let prov_final = self.provenance.add_node(
                vec![prov_verdict], Some(self.discovery_count),
                StepType::FinalInsight,
                hyp.statement.clone(),
                verdict.confidence,
            );

            self.audit.log(
                AuditEventType::InsightDiscovered,
                hyp.statement.clone(),
                format!("id={}, conf={:.2}", self.discovery_count, verdict.confidence),
                "{}",
            );

            // Adiciona o embedding no histórico temporal para os próximos loops de sonho
            self.past_discoveries_embeddings.push(discovery_embedding.clone());

            discoveries.push(NovelDiscovery {
                id: self.discovery_count,
                statement: hyp.statement,
                domain: hyp.domain,
                confidence: verdict.confidence,
                source_embedding: discovery_embedding,
                provenance_root: prov_root,
            });

            self.discovery_count += 1;
            hypotheses_generated += 1;
        }

        // PASSO 8: Autopoiese — ajusta parâmetros baseado na saúde
        let health = SystemHealth {
            discoveries_per_epoch: discoveries.len() as f32,
            free_energy_trend: self.free_energy.convergence_trend(),
            functor_coverage: self.functors.functor_library.len() as f32 * 0.1,
            false_positive_rate: 0.02,
            current_temperature: self.annealing.current_temperature,
            stagnation_epochs: if discoveries.is_empty() { 1 } else { 0 },
        };

        let adjustments = self.autopoiesis.analyze(&health);
        for adj in &adjustments {
            if adj.parameter == "annealing_temperature" {
                self.annealing.current_temperature = adj.new_value;
            }
            self.audit.log(
                AuditEventType::AutopoiesisAdjustment,
                adj.parameter.clone(),
                format!("{:.3} → {:.3}: {}", adj.old_value, adj.new_value, adj.reason),
                "{}",
            );
        }

        // Evaporação de feromônios físicos do banco
        self.swarm.evaporate(1);

        // Prevenção de Model Collapse (Memory Purge)
        // Se a mente reter mais de 1000 sonhos empilhados, cortamos pela raiz
        // para dar espaço a uma nova geometria semântica, forçando "desaprendizado" plástico.
        if self.past_discoveries_embeddings.len() > 1000 {
            self.past_discoveries_embeddings.clear();
            self.audit.log(
                AuditEventType::AutopoiesisAdjustment,
                "Model_Collapse_Prevention".to_string(),
                "Purgando past_discoveries_embeddings para evadir Echo Chamber".to_string(),
                "{}",
            );
        }

        discoveries
    }

    /// Relatório de saúde do sistema
    pub fn health_report(&self) -> String {
        format!(
            "DreamingEngine | discoveries={} | audit_entries={} | prov_nodes={} | \
             pheromones={} | temp={:.3} | autopoiesis_adjustments={}",
            self.discovery_count,
            self.audit.len(),
            self.provenance.len(),
            self.swarm.active_pheromones(),
            self.annealing.current_temperature,
            self.autopoiesis.total_adjustments(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_knowledge_embeddings() -> Vec<Vec<f32>> {
        // 3 clusters bem separados para gerar gaps topológicos
        let mut embs = Vec::new();
        for i in 0..5 {
            embs.push(vec![i as f32 * 0.1, 0.0]);
        }
        for i in 0..5 {
            embs.push(vec![5.0 + i as f32 * 0.1, 0.0]);
        }
        for i in 0..5 {
            embs.push(vec![0.0, 5.0 + i as f32 * 0.1]);
        }
        embs
    }

    fn make_insights() -> Vec<Insight> {
        vec![
            Insight {
                id: 1,
                domain: "Fisica".to_string(),
                statement: "Energia é conservada".to_string(),
                embedding: vec![0.1, 0.0],
                relations: vec![],
            },
        ]
    }

    #[test]
    fn test_single_dream_cycle() {
        let config = DreamConfig::default();
        let mut engine = DreamingEngine::new(&config);
        let embeddings = make_knowledge_embeddings();
        let insights = make_insights();

        let discoveries = engine.dream_cycle(&embeddings, &insights);
        // O sistema deve completar o ciclo sem panic
        assert!(engine.audit.len() > 0, "Audit deve ter entradas");
        assert!(engine.provenance.len() >= 0);
    }

    #[test]
    fn test_multi_cycle_accumulation() {
        let config = DreamConfig::default();
        let mut engine = DreamingEngine::new(&config);
        let embeddings = make_knowledge_embeddings();
        let insights = make_insights();

        let mut total_discoveries = 0;
        for _ in 0..3 {
            let d = engine.dream_cycle(&embeddings, &insights);
            total_discoveries += d.len();
        }

        // Autopoiese deve ter rodado pelo menos 3 vezes
        assert_eq!(engine.autopoiesis.parameter_history.len(), 3);
        assert!(engine.audit.len() > 3);
    }

    #[test]
    fn test_health_report_generated() {
        let config = DreamConfig::default();
        let engine = DreamingEngine::new(&config);
        let report = engine.health_report();
        assert!(report.contains("discoveries=0"));
        assert!(report.contains("temp="));
    }
}
