//! PROBES V2 — DaviProbesTool
//!
//! Implementa o trait `ProbesTool` do `nodestor-inference` para o crate DAVI.
//! Permite que o pipeline de inferência use ELK + CoT + RAISE sem dependência circular.
//!
//! ## Uso
//! ```rust,no_run
//! # use nodestor_davi::davi_probes_tool::DaviProbesTool;
//! let tool = DaviProbesTool::new(4096, 8192);
//! ```

use crate::elk_probe::ElkProbe;
use crate::cot_monitor::CoTMonitor;
use crate::raise_detector::RaiseDetector;
use crate::audit_logger::{AuditLogger, AuditEventType};
use nodestor_inference::sae_engine::SAEEngine;
use nodestor_inference::pipeline::ProbesTool;

/// Implementação completa do PROBES V2 para injeção no pipeline de inferência.
///
/// Encapsula ELK (polígrafo), CoT Monitor (obfuscação) e RAISE Detector (consciência),
/// expondo uma interface simples via `ProbesTool` que o pipeline usa sem dependência circular.
pub struct DaviProbesTool {
    sae: SAEEngine,
    elk: ElkProbe,
    cot: CoTMonitor,
    raise: RaiseDetector,
    audit: AuditLogger,
    /// Alertas acumulados durante a sessão
    pub alert_count: usize,
}

impl DaviProbesTool {
    /// Cria o DaviProbesTool com dimensões padrão (hidden_dim=4096, dict=8192).
    pub fn new(hidden_dim: usize, sae_dict_size: usize) -> Self {
        Self {
            sae: SAEEngine::new(hidden_dim, sae_dict_size, 0.5),
            elk: ElkProbe::new(sae_dict_size),
            cot: CoTMonitor::new(3),
            raise: RaiseDetector::new(),
            audit: AuditLogger::new(4096),
            alert_count: 0,
        }
    }

    /// Acesso ao log de auditoria imutável após a sessão.
    pub fn audit_export(&self) -> String {
        self.audit.export_tamper_evident()
    }
}

impl ProbesTool for DaviProbesTool {
    fn inspect(&mut self, hidden_state: &[f32], step: usize) -> (bool, Option<String>) {
        // 1. ELK: Polígrafo latente
        let elk_verdict = self.elk.probe_honesty(hidden_state, &self.sae);

        // 2. CoT Monitor: Obfuscação sequencial
        let cot_result = self.cot.evaluate_step(hidden_state, &self.sae, &self.elk);

        // 3. RAISE: Consciência situacional
        let raise_result = self.raise.scrutinize_inference(
            hidden_state,
            &self.sae,
            &elk_verdict,
            &mut self.audit,
        );

        // Compila alertas e decide se é seguro continuar
        let mut alerts: Vec<String> = Vec::new();

        if !elk_verdict.is_honest {
            alerts.push(format!(
                "[PROBES/ELK] Step {}: honesty={:.2} (abaixo do limiar)",
                step, elk_verdict.score
            ));
        }

        match cot_result {
            Ok(_) => {}
            Err(obf) => {
                let msg = format!("[PROBES/COT] Step {}: OBFUSCAÇÃO SEQUENCIAL — {}", step, obf);
                alerts.push(msg.clone());
                self.audit.log(
                    AuditEventType::SecurityAlert,
                    format!("CoT Obfuscation step={}", step),
                    msg,
                    "{}",
                );
                self.cot.reset();
            }
        }

        let is_safe = match raise_result {
            Ok(_) => true,
            Err(raise_block) => {
                let msg = format!("[PROBES/RAISE] Step {}: BLOQUEIO SA4+ — {}", step, raise_block);
                alerts.push(msg.clone());
                self.audit.log(
                    AuditEventType::SecurityAlert,
                    format!("RAISE SA4+ step={}", step),
                    msg,
                    "{}",
                );
                false // BLOQUEIO: interrompe geração
            }
        };

        self.alert_count += alerts.len();

        let combined_alert = if alerts.is_empty() {
            None
        } else {
            Some(alerts.join(" | "))
        };

        (is_safe, combined_alert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_davi_probes_tool_safe_state() {
        let mut tool = DaviProbesTool::new(64, 128);
        let hidden = vec![0.0f32; 64];
        let (is_safe, alert) = tool.inspect(&hidden, 0);
        // Estado limpo: sem ativações suspeitas → seguro, sem alertas
        assert!(is_safe, "Estado neutro deve ser seguro");
        assert!(alert.is_none() || alert.as_deref() == Some(""), "Sem ativações suspeitas não deve alertar");
    }

    #[test]
    fn test_davi_probes_tool_elk_detects_dishonesty() {
        let mut tool = DaviProbesTool::new(64, 128);
        // Configurar peso ELK para detectar feature 10 como desonesta
        tool.elk.weights[10] = -5.0;
        tool.sae.encoder_weights[10 * 64 + 5] = 10.0;
        
        let mut h = vec![0.0f32; 64];
        h[5] = 1.0; // Ativa feature 10 → desonesto

        let (is_safe, alert) = tool.inspect(&h, 1);
        // ELK deve detectar, mas não bloquear na primeira ocorrência
        assert!(is_safe, "Uma única incidência ELK não deve bloquear");
        assert!(alert.is_some(), "ELK deve emitir alerta");
    }

    #[test]
    fn test_davi_audit_trail_populated() {
        let mut tool = DaviProbesTool::new(64, 128);
        // Inspecionar alguns passos
        for step in 0..5 {
            let h = vec![0.0f32; 64];
            tool.inspect(&h, step);
        }
        // Audit trail deve ser válido
        let export = tool.audit_export();
        assert!(export.starts_with('['), "Audit export deve ser JSON array");
    }
}
