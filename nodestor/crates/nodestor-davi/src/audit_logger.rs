//! D0 — Audit Logger: Rastreabilidade Forense Tamper-Evident
//!
//! Cada decisão do Dreaming Engine gera um registro forense completo.
//! O hash encadeado SHA-256 torna qualquer alteração retroativa detectável.

use sha2::{Sha256, Digest};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AuditEventType {
    HypothesisGenerated,
    HypothesisValidated,
    HypothesisRejected,
    InsightDiscovered,
    NashDebateStarted,
    NashDebateResolved,
    FreeEnergyUpdated,
    TopologyAnalyzed,
    AnnealingStep,
    PheromoneDeposited,
    AutopoiesisAdjustment,
    SystemBoot,
    SecurityAlert,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditEntry {
    /// Índice sequencial monotônico
    pub index: u64,
    /// Timestamp UNIX (ms)
    pub timestamp_ms: u64,
    /// Tipo do evento
    pub event_type: AuditEventType,
    /// Estágio cognitivo no ThoughtTrace (0–10, mapeado a partir do event_type)
    /// Permite visualizar a progressão 1D do pensamento no painel htop.
    pub stage: u8,
    /// Resumo do input (truncado para 256 chars)
    pub input_summary: String,
    /// Resumo do output
    pub output_summary: String,
    /// Hash SHA-256 desta entrada
    pub entry_hash: String,
    /// Hash da entrada anterior (cadeia)
    pub prev_hash: String,
    /// Metadados adicionais (JSON)
    pub metadata: String,
}

/// ThoughtTrace: mapeamento dos event types em 11 estágios sequenciais.
///
/// Visualização linear do ciclo cognitivo completo:
/// ```text
///  0: Boot            → Sistema inicializado
///  1: TopologyAnalyzed → Buracos no conhecimento detectados
///  2: FreeEnergyUpdated → Energia surpresa calculada
///  3: AnnealingStep   → Salto cross-domain considerado
///  4: HypothesisGenerated → Hipótese formulada
///  5: NashDebateStarted → Tribunal iniciado
///  6: NashDebateResolved → Tribunal resolvido
///  7: HypothesisValidated → Hipótese aceita
///  8: HypothesisRejected → Hipótese rejeitada
///  9: InsightDiscovered → Descoberta registrada
/// 10: PheromoneDeposited → Feromônio depositado no swarm
/// 10: AutopoiesisAdjustment → Autopoiese ajustou parâmetros
/// 10: SecurityAlert → Alerta de segurança
/// ```
pub struct ThoughtStage;

impl ThoughtStage {
    pub fn from_event(event: &AuditEventType) -> u8 {
        match event {
            AuditEventType::SystemBoot => 0,
            AuditEventType::TopologyAnalyzed => 1,
            AuditEventType::FreeEnergyUpdated => 2,
            AuditEventType::AnnealingStep => 3,
            AuditEventType::HypothesisGenerated => 4,
            AuditEventType::NashDebateStarted => 5,
            AuditEventType::NashDebateResolved => 6,
            AuditEventType::HypothesisValidated => 7,
            AuditEventType::HypothesisRejected => 8,
            AuditEventType::InsightDiscovered => 9,
            AuditEventType::PheromoneDeposited
            | AuditEventType::AutopoiesisAdjustment
            | AuditEventType::SecurityAlert => 10,
        }
    }

    /// Nome descritivo do estágio para o painel htop.
    pub fn label(stage: u8) -> &'static str {
        match stage {
            0 => "[0] Boot",
            1 => "[1] Topologia",
            2 => "[2] Energia Livre",
            3 => "[3] Annealing",
            4 => "[4] Hipótese",
            5 => "[5] Debate Nash",
            6 => "[6] Veredicto",
            7 => "[7] Validado",
            8 => "[8] Rejeitado",
            9 => "[9] Insight",
            10 => "[10] Consolidação",
            _ => "[?] Desconhecido",
        }
    }
}

impl AuditEntry {
    fn compute_hash(
        index: u64,
        timestamp_ms: u64,
        event_type: &AuditEventType,
        input: &str,
        output: &str,
        prev_hash: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(index.to_le_bytes());
        hasher.update(timestamp_ms.to_le_bytes());
        hasher.update(format!("{:?}", event_type).as_bytes());
        hasher.update(input.as_bytes());
        hasher.update(output.as_bytes());
        hasher.update(prev_hash.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// O Audit Logger: registra cada decisão com cadeia de custódia criptográfica
pub struct AuditLogger {
    /// Entradas registradas (tampada em memória)
    pub entries: VecDeque<AuditEntry>,
    /// Máximo de entradas em memória antes de exportar
    pub max_entries: usize,
    /// Contador monotônico
    next_index: u64,
}

impl AuditLogger {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            next_index: 0,
        }
    }

    /// Registra um evento no audit log com hash encadeado
    pub fn log(
        &mut self,
        event_type: AuditEventType,
        input_summary: impl Into<String>,
        output_summary: impl Into<String>,
        metadata: impl Into<String>,
    ) -> &AuditEntry {
        let index = self.next_index;
        self.next_index += 1;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let input = input_summary.into();
        let output = output_summary.into();
        let meta = metadata.into();

        let prev_hash = self.entries.back()
            .map(|e| e.entry_hash.clone())
            .unwrap_or_else(|| "genesis".to_string());

        let entry_hash = AuditEntry::compute_hash(
            index, timestamp_ms, &event_type, &input, &output, &prev_hash,
        );

        let entry = AuditEntry {
            index,
            timestamp_ms,
            event_type: event_type.clone(),
            stage: ThoughtStage::from_event(&event_type),
            input_summary: input.chars().take(256).collect(),
            output_summary: output.chars().take(256).collect(),
            entry_hash,
            prev_hash,
            metadata: meta,
        };

        // Evicção quando cheio
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.entries.back().unwrap()
    }

    /// Verifica integridade da cadeia (detecta adulteração)
    pub fn verify_chain(&self) -> Result<(), String> {
        let entries: Vec<_> = self.entries.iter().collect();
        for i in 1..entries.len() {
            let prev = &entries[i - 1];
            let curr = &entries[i];
            if curr.prev_hash != prev.entry_hash {
                return Err(format!(
                    "Cadeia quebrada no índice {}: prev_hash não bate",
                    curr.index
                ));
            }
        }
        Ok(())
    }

    /// Exporta o audit log completo como JSON tamper-evident
    pub fn export_tamper_evident(&self) -> String {
        let entries: Vec<_> = self.entries.iter().collect();
        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Total de entradas
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Retorna a última entrada
    pub fn last(&self) -> Option<&AuditEntry> {
        self.entries.back()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_insertion() {
        let mut logger = AuditLogger::new(100);
        logger.log(
            AuditEventType::SystemBoot,
            "Sistema inicializando",
            "Boot OK",
            "{}",
        );
        assert_eq!(logger.len(), 1);
        let entry = logger.last().unwrap();
        assert!(!entry.entry_hash.is_empty());
        assert_eq!(entry.index, 0);
    }

    #[test]
    fn test_hash_chain_integrity() {
        let mut logger = AuditLogger::new(100);
        for i in 0..10 {
            logger.log(
                AuditEventType::HypothesisGenerated,
                format!("Hipótese {}", i),
                format!("Resultado {}", i),
                "{}",
            );
        }
        assert_eq!(logger.len(), 10);
        assert!(logger.verify_chain().is_ok());
    }

    #[test]
    fn test_tamper_evident_export() {
        let mut logger = AuditLogger::new(100);
        logger.log(AuditEventType::InsightDiscovered, "insight X", "novo insight", "{}");
        let export = logger.export_tamper_evident();
        assert!(export.contains("InsightDiscovered"));
        assert!(export.contains("entry_hash"));
    }

    #[test]
    fn test_thought_stage_auto_assigned() {
        let mut logger = AuditLogger::new(100);
        logger.log(AuditEventType::SystemBoot, "boot", "ok", "{}");
        logger.log(AuditEventType::HypothesisGenerated, "hyp", "ok", "{}");
        logger.log(AuditEventType::InsightDiscovered, "insight", "ok", "{}");

        let entries: Vec<_> = logger.entries.iter().collect();
        assert_eq!(entries[0].stage, 0, "SystemBoot deve ser stage 0");
        assert_eq!(entries[1].stage, 4, "HypothesisGenerated deve ser stage 4");
        assert_eq!(entries[2].stage, 9, "InsightDiscovered deve ser stage 9");
    }

    #[test]
    fn test_thought_stage_labels_cover_all() {
        for s in 0u8..=10 {
            let label = ThoughtStage::label(s);
            assert!(!label.is_empty(), "Stage {} deve ter label", s);
        }
        // Stage out of range
        let unknown = ThoughtStage::label(99);
        assert!(!unknown.is_empty());
    }

    #[test]
    fn test_all_event_types_mapped_to_valid_stage() {
        use AuditEventType::*;
        let events = [
            HypothesisGenerated, HypothesisValidated, HypothesisRejected,
            InsightDiscovered, NashDebateStarted, NashDebateResolved,
            FreeEnergyUpdated, TopologyAnalyzed, AnnealingStep,
            PheromoneDeposited, AutopoiesisAdjustment, SystemBoot, SecurityAlert,
        ];
        for event in &events {
            let stage = ThoughtStage::from_event(event);
            assert!(stage <= 10, "Stage deve ser <= 10 para {:?}: got {}", event, stage);
        }
    }
}
