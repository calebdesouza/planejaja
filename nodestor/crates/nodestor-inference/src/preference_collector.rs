use crate::dataset_curator::{PreferencePair, PreferencePairSource};

#[derive(Debug, Clone)]
pub struct PreferenceConfig {
    pub store_path: String,
    pub consolidation_threshold: usize,
    pub max_pairs: usize,
    pub sae_dim: usize,
}

impl Default for PreferenceConfig {
    fn default() -> Self {
        Self {
            store_path: "tests/fixtures/pref_default".to_string(),
            consolidation_threshold: 10,
            max_pairs: 1000,
            sae_dim: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsolidationSignal {
    pub pairs_ready: usize,
}

#[derive(Debug, Default)]
pub struct CollectorStats {
    pub corrections_captured: usize,
    pub auto_captured: usize,
    pub signals_emitted: usize,
}

pub struct PreferenceCollector {
    config: PreferenceConfig,
    pending: Vec<PreferencePair>,
    pub stats: CollectorStats,
}

impl PreferenceCollector {
    pub fn new(config: PreferenceConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            stats: CollectorStats::default(),
        }
    }

    pub fn record_correction(
        &mut self,
        prompt: &str,
        rejected: &str,
        chosen: &str,
        _ai_pathway: Vec<f32>,
        context_embedding: Vec<f32>,
    ) -> Result<Option<ConsolidationSignal>, String> {
        let pair = PreferencePair {
            prompt: prompt.to_string(),
            chosen_response: chosen.to_string(),
            rejected_response: rejected.to_string(),
            rejected_pathway: context_embedding.clone(),
            chosen_pathway: Some(context_embedding.clone()),
            context_embedding,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            source: PreferencePairSource::UserCorrection,
        };
        self.pending.push(pair);
        self.stats.corrections_captured += 1;

        if self.pending.len() >= self.config.consolidation_threshold {
            self.stats.signals_emitted += 1;
            Ok(Some(ConsolidationSignal { pairs_ready: self.pending.len() }))
        } else {
            Ok(None)
        }
    }

    pub fn record_auto(
        &mut self,
        prompt: &str,
        rejected: &str,
        chosen: &str,
        _ai_pathway: Vec<f32>,
        context_embedding: Vec<f32>,
        _confidence_scores: Vec<f32>,
    ) -> Result<Option<ConsolidationSignal>, String> {
        let pair = PreferencePair {
            prompt: prompt.to_string(),
            chosen_response: chosen.to_string(),
            rejected_response: rejected.to_string(),
            rejected_pathway: vec![],
            chosen_pathway: None,
            context_embedding,
            created_at: 0,
            source: PreferencePairSource::AutoCapture,
        };
        self.pending.push(pair);
        self.stats.auto_captured += 1;
        Ok(None)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn is_consolidation_ready(&self) -> bool {
        self.pending.len() >= self.config.consolidation_threshold
    }

    pub fn drain_for_training(&mut self) -> Vec<PreferencePair> {
        std::mem::take(&mut self.pending)
    }
}
