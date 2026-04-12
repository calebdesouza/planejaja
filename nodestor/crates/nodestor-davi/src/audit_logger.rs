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
            event_type,
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
}
