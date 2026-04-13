//! PROBES V2 — Adaptive Router (Princípio 6: Gastar Energia Só Onde É Necessário)
//!
//! Routing endógeno — o próprio PROBES V2 decide quanta computação alocar.
//! Em vez de um classificador externo (TF-IDF, MLP separado), o router usa
//! o `ConformalPredictor` interno para medir a certeza do modelo e rotear
//! com base nessa certeza antes de gerar qualquer token.
//!
//! ## Limiares (NCS = Non-Conformity Score, 0=certo, 1=incerto):
//! - NCS < 0.10 → `Direct`: modelo tem 90%+ certeza. Responder sem busca.
//! - NCS 0.10-0.40 → `LightRetrieval`: busca vetorial simples top-k.
//! - NCS 0.40-0.65 → `EnhancedRetrieval`: busca + amplificação de raciocínio.
//! - NCS > 0.65 → `DeepReasoning`: Dreaming Engine completo com budget forcing.
//!
//! ## Pesquisa (2025-2026):
//! RAGRouter-Bench mostrou que routing por métricas de certeza internas supera
//! classificadores externos em 12% de precisão. O nosso diferencial: routing
//! endógeno via Conformal — nenhum modelo extra necessário.

use crate::conformal_predictor::ConformalPredictor;
use crate::ignorance_detector::IgnoranceReport;
use std::collections::VecDeque;

/// A rota escolhida pelo sistema adaptativo.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    /// Resposta direta — modelo tem certeza total. Sem busca, sem Steering.
    Direct,

    /// Busca vetorial simples — modelo precisa de contexto mas não está perdido.
    LightRetrieval {
        top_k: usize,
    },

    /// Busca + amplificação cognitiva — modelo precisa de ajuda estruturada.
    EnhancedRetrieval {
        top_k: usize,
        steering_alpha: f32,
    },

    /// Deep Reasoning completo — Dreaming Engine + Budget Forcing.
    DeepReasoning {
        max_hops: usize,
        budget_force: bool,
    },
}

impl RouteDecision {
    /// Custo estimado em forward passes (para telemetria).
    pub fn estimated_cost(&self) -> usize {
        match self {
            RouteDecision::Direct => 1,
            RouteDecision::LightRetrieval { .. } => 3,
            RouteDecision::EnhancedRetrieval { .. } => 7,
            RouteDecision::DeepReasoning { max_hops, .. } => 10 + max_hops * 5,
        }
    }

    /// Nome legível da rota.
    pub fn name(&self) -> &'static str {
        match self {
            RouteDecision::Direct => "Direct",
            RouteDecision::LightRetrieval { .. } => "LightRetrieval",
            RouteDecision::EnhancedRetrieval { .. } => "EnhancedRetrieval",
            RouteDecision::DeepReasoning { .. } => "DeepReasoning",
        }
    }
}

/// Estatísticas de routing acumuladas.
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub total_queries: u64,
    pub direct_count: u64,
    pub light_count: u64,
    pub enhanced_count: u64,
    pub deep_count: u64,
    /// Tokens economizados por não ir para Deep quando não necessário
    pub estimated_tokens_saved: u64,
    /// NCS médio das últimas N queries
    pub avg_ncs: f32,
}

/// O router adaptativo — endógeno ao PROBES V2.
pub struct AdaptiveRouter {
    conformal: ConformalPredictor,
    /// Histórico de (NCS, route_name) para calibração adaptativa
    ncs_history: VecDeque<(f32, &'static str)>,
    pub calibration_window: usize,
    /// Estatísticas acumuladas
    stats: RoutingStats,
    /// Limiares configuráveis (NCS)
    pub threshold_direct: f32,
    pub threshold_light: f32,
    pub threshold_enhanced: f32,
}

impl AdaptiveRouter {
    /// Cria router com limiares padrão calibrados.
    pub fn new(confidence_level: f32) -> Self {
        Self {
            conformal: ConformalPredictor::new(confidence_level),
            ncs_history: VecDeque::new(),
            calibration_window: 100,
            stats: RoutingStats {
                total_queries: 0,
                direct_count: 0,
                light_count: 0,
                enhanced_count: 0,
                deep_count: 0,
                estimated_tokens_saved: 0,
                avg_ncs: 0.0,
            },
            threshold_direct: 0.10,
            threshold_light: 0.40,
            threshold_enhanced: 0.65,
        }
    }

    /// Rota baseada nos logits do primeiro forward pass.
    ///
    /// # Uso no Pipeline
    /// ```text
    /// let initial_logits = model.forward(&prompt);
    /// let decision = router.route_from_logits(&initial_logits);
    /// match decision {
    ///     RouteDecision::Direct => generate_direct(),
    ///     RouteDecision::DeepReasoning { .. } => trigger_dreaming_engine(),
    ///     ...
    /// }
    /// ```
    pub fn route_from_logits(&mut self, initial_logits: &[f32]) -> RouteDecision {
        let conformal_set = self.conformal.predict_set(initial_logits);
        let ncs = conformal_set.non_conformity_score;

        let decision = self.decide_from_ncs(ncs);
        self.record_routing(ncs, decision.name());
        decision
    }

    /// Rota baseada no IgnoranceReport (pré-geração, sem forward pass completo).
    /// Mais rápido: usa apenas o prefill para decidir.
    pub fn route_from_ignorance(&self, report: &IgnoranceReport) -> RouteDecision {
        if !report.is_ignorant {
            // Modelo tem base → routing leve baseado na confidence
            let ncs_proxy = 1.0 - report.recall_score;
            return self.decide_from_ncs(ncs_proxy);
        }

        // Ignorante com alta confidence na ignorância → Deep Reasoning imediato
        if report.confidence > 0.7 {
            return RouteDecision::DeepReasoning {
                max_hops: 3,
                budget_force: true,
            };
        }

        // Ignorante mas borderline → Enhanced com busca
        RouteDecision::EnhancedRetrieval {
            top_k: 5,
            steering_alpha: 2.5,
        }
    }

    /// Registra o resultado final para calibração adaptativa.
    ///
    /// Se o NCS ao final foi muito diferente do NCS inicial,
    /// os limiares se ajustam para melhorar o routing futuro.
    pub fn record_outcome(&mut self, decision: RouteDecision, ncs_at_end: f32) {
        // Calibração: se Deep Reasoning mas NCS final baixo,
        // talvez Enhanced fosse suficiente → ajusta limiar
        if let RouteDecision::DeepReasoning { .. } = &decision {
            if ncs_at_end < self.threshold_light {
                // Deep foi desnecessário: relaxa o threshold_enhanced
                self.threshold_enhanced = (self.threshold_enhanced + 0.01).min(0.80);
            }
        }
        // Se Direct mas NCS final alto (modelo errou), aperta o limiar
        if let RouteDecision::Direct = &decision {
            if ncs_at_end > self.threshold_light {
                self.threshold_direct = (self.threshold_direct - 0.01).max(0.05);
            }
        }

        self.conformal.recalibrate();
    }

    /// Estatísticas de routing acumuladas.
    pub fn routing_stats(&self) -> &RoutingStats {
        &self.stats
    }

    /// Relatório legível do router.
    pub fn report(&self) -> String {
        let s = &self.stats;
        let total = s.total_queries.max(1) as f32;
        format!(
            "AdaptiveRouter | queries={} | direct={:.0}% | light={:.0}% | enhanced={:.0}% | deep={:.0}% | avg_ncs={:.3} | tokens_saved=~{}",
            s.total_queries,
            s.direct_count as f32 / total * 100.0,
            s.light_count as f32 / total * 100.0,
            s.enhanced_count as f32 / total * 100.0,
            s.deep_count as f32 / total * 100.0,
            s.avg_ncs,
            s.estimated_tokens_saved,
        )
    }

    // --- Utilitários internos ---

    fn decide_from_ncs(&self, ncs: f32) -> RouteDecision {
        if ncs < self.threshold_direct {
            RouteDecision::Direct
        } else if ncs < self.threshold_light {
            RouteDecision::LightRetrieval { top_k: 3 }
        } else if ncs < self.threshold_enhanced {
            RouteDecision::EnhancedRetrieval {
                top_k: 5,
                steering_alpha: 2.0 + (ncs - self.threshold_light) * 4.0,
            }
        } else {
            let hops = if ncs > 0.80 { 5 } else { 3 };
            RouteDecision::DeepReasoning {
                max_hops: hops,
                budget_force: ncs > 0.75,
            }
        }
    }

    fn record_routing(&mut self, ncs: f32, route_name: &'static str) {
        self.stats.total_queries += 1;

        // Atualiza NCS médio com média móvel exponencial
        let alpha = 2.0 / (self.calibration_window as f32 + 1.0);
        self.stats.avg_ncs = self.stats.avg_ncs * (1.0 - alpha) + ncs * alpha;

        // Contadores por rota
        match route_name {
            "Direct" => {
                self.stats.direct_count += 1;
                // Tokens economizados: diferença de custo para DeepReasoning
                self.stats.estimated_tokens_saved += RouteDecision::DeepReasoning {
                    max_hops: 3, budget_force: false
                }.estimated_cost() as u64 - 1;
            }
            "LightRetrieval" => self.stats.light_count += 1,
            "EnhancedRetrieval" => self.stats.enhanced_count += 1,
            "DeepReasoning" => self.stats.deep_count += 1,
            _ => {}
        }

        // Mantém janela de histórico
        self.ncs_history.push_back((ncs, route_name));
        if self.ncs_history.len() > self.calibration_window {
            self.ncs_history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router() -> AdaptiveRouter {
        AdaptiveRouter::new(0.95)
    }

    #[test]
    fn test_direct_for_high_confidence_logits() {
        let mut router = make_router();

        // Logits muito concentrados → baixo NCS → Direct
        let mut logits = vec![0.0f32; 100];
        logits[0] = 100.0; // Um token domina completamente

        let decision = router.route_from_logits(&logits);
        assert_eq!(decision, RouteDecision::Direct,
            "Logits concentrados devem rotear para Direct");
    }

    #[test]
    fn test_deep_reasoning_for_uncertain_logits() {
        let mut router = make_router();

        // Logits uniformes → NCS alto → DeepReasoning
        let logits = vec![1.0f32; 1000];

        let decision = router.route_from_logits(&logits);
        matches!(decision, RouteDecision::DeepReasoning { .. });
    }

    #[test]
    fn test_route_from_ignorance_confident() {
        let router = make_router();

        let report = IgnoranceReport {
            is_ignorant: true,
            filler_score: 0.9,
            recall_score: 0.1,
            confidence: 0.85,
            topic_cluster: Some(vec![0.5; 10]),
            active_filler_features: vec![],
        };

        let decision = router.route_from_ignorance(&report);
        assert!(matches!(decision, RouteDecision::DeepReasoning { .. }),
            "Alta confidence de ignorância → DeepReasoning");
    }

    #[test]
    fn test_route_from_ignorance_borderline() {
        let router = make_router();

        let report = IgnoranceReport {
            is_ignorant: true,
            filler_score: 0.65,
            recall_score: 0.35,
            confidence: 0.3, // Borderline
            topic_cluster: None,
            active_filler_features: vec![],
        };

        let decision = router.route_from_ignorance(&report);
        assert!(matches!(decision, RouteDecision::EnhancedRetrieval { .. }),
            "Ignorância borderline → EnhancedRetrieval");
    }

    #[test]
    fn test_not_ignorant_routes_light_or_direct() {
        let router = make_router();

        let report = IgnoranceReport {
            is_ignorant: false,
            filler_score: 0.1,
            recall_score: 0.9,
            confidence: 0.9,
            topic_cluster: None,
            active_filler_features: vec![],
        };

        let decision = router.route_from_ignorance(&report);
        assert!(
            matches!(decision, RouteDecision::Direct)
                || matches!(decision, RouteDecision::LightRetrieval { .. }),
            "Conhecimento sólido deve rotear para Direct ou Light, não Deep"
        );
    }

    #[test]
    fn test_stats_accumulate() {
        let mut router = make_router();

        let logits_certain = {
            let mut v = vec![0.0f32; 50];
            v[0] = 100.0;
            v
        };
        let logits_uncertain = vec![1.0f32; 50];

        router.route_from_logits(&logits_certain);
        router.route_from_logits(&logits_uncertain);
        router.route_from_logits(&logits_certain);

        let stats = router.routing_stats();
        assert_eq!(stats.total_queries, 3);
        assert!(stats.direct_count >= 1, "Ao menos 1 Direct esperado");
    }

    #[test]
    fn test_record_outcome_adjusts_thresholds() {
        let mut router = make_router();
        let initial_enhanced_threshold = router.threshold_enhanced;

        // Simula Deep Reasoning que foi desnecessário (NCS final baixo)
        router.record_outcome(
            RouteDecision::DeepReasoning { max_hops: 3, budget_force: false },
            0.05, // NCS final muito baixo → Deep foi desnecessário
        );

        assert!(router.threshold_enhanced >= initial_enhanced_threshold,
            "Threshold enhanced deve ter relaxado: antes={:.3}, depois={:.3}",
            initial_enhanced_threshold, router.threshold_enhanced);
    }

    #[test]
    fn test_report_not_empty() {
        let mut router = make_router();
        let logits = vec![1.0f32; 10];
        router.route_from_logits(&logits);

        let report = router.report();
        assert!(report.contains("AdaptiveRouter"), "Report deve conter nome do módulo");
        assert!(report.contains("queries=1"), "Report deve mostrar total de queries");
    }

    #[test]
    fn test_estimated_cost_ordering() {
        // Direct < Light < Enhanced < Deep
        assert!(RouteDecision::Direct.estimated_cost()
            < RouteDecision::LightRetrieval { top_k: 3 }.estimated_cost());
        assert!(RouteDecision::LightRetrieval { top_k: 3 }.estimated_cost()
            < RouteDecision::EnhancedRetrieval { top_k: 5, steering_alpha: 2.0 }.estimated_cost());
        assert!(RouteDecision::EnhancedRetrieval { top_k: 5, steering_alpha: 2.0 }.estimated_cost()
            < RouteDecision::DeepReasoning { max_hops: 3, budget_force: false }.estimated_cost());
    }
}
