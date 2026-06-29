use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CuratorConfig {
    pub contradiction_threshold: f32,
    pub redundancy_threshold: f32,
    pub min_quality_score: f32,
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.85,
            redundancy_threshold: 0.95,
            min_quality_score: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CurationDocument {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreferencePairSource {
    UserCorrection,
    AutoCapture,
    SyntheticAugmentation,
}

#[derive(Debug, Clone)]
pub struct PreferencePair {
    pub prompt: String,
    pub chosen_response: String,
    pub rejected_response: String,
    pub rejected_pathway: Vec<f32>,
    pub chosen_pathway: Option<Vec<f32>>,
    pub context_embedding: Vec<f32>,
    pub created_at: u64,
    pub source: PreferencePairSource,
}

#[derive(Debug)]
pub struct ContradictionPair {
    pub doc_a: String,
    pub doc_b: String,
    pub similarity: f32,
}

#[derive(Debug)]
pub struct CurationReport {
    pub total_documents: usize,
    pub contradictions_found: Vec<ContradictionPair>,
    pub redundant_pairs: Vec<(String, String)>,
    pub low_quality_ids: Vec<String>,
    pub curated_ids: Vec<String>,
    pub quality_rate: f32,
    pub pairs_generated: Vec<PreferencePair>,
}

impl CurationReport {
    pub fn dpo_pair_count(&self) -> usize {
        self.pairs_generated.len()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { 0.0 } else { dot / (na * nb) }
}

fn quality_score(doc: &CurationDocument) -> f32 {
    let words: Vec<&str> = doc.text.split_whitespace().collect();
    if words.len() < 3 { return 0.1; }
    let fillers = ["eu", "acho", "talvez", "né", "assim"];
    let filler_ratio = words.iter().filter(|w| fillers.contains(&w.to_lowercase().as_str())).count() as f32 / words.len() as f32;
    (1.0 - filler_ratio * 5.0).max(0.0).min(1.0)
}

pub struct DatasetCurator {
    config: CuratorConfig,
}

impl DatasetCurator {
    pub fn new(config: CuratorConfig) -> Self {
        Self { config }
    }

    pub fn curate(&self, docs: &[CurationDocument]) -> CurationReport {
        let mut contradictions_found = Vec::new();
        let mut redundant_pairs = Vec::new();
        let mut low_quality_ids = Vec::new();
        let mut curated_ids = Vec::new();

        // Quality filter
        for doc in docs {
            let score = quality_score(doc);
            if score < self.config.min_quality_score {
                low_quality_ids.push(doc.id.clone());
            } else {
                curated_ids.push(doc.id.clone());
            }
        }

        // Contradiction and redundancy detection
        for i in 0..docs.len() {
            for j in (i + 1)..docs.len() {
                let sim = cosine_sim(&docs[i].embedding, &docs[j].embedding);
                let a_neg = docs[i].text.contains("não") || docs[i].text.contains("never");
                let b_neg = docs[j].text.contains("não") || docs[j].text.contains("never");
                if sim >= self.config.contradiction_threshold && (a_neg ^ b_neg) {
                    contradictions_found.push(ContradictionPair {
                        doc_a: docs[i].id.clone(),
                        doc_b: docs[j].id.clone(),
                        similarity: sim,
                    });
                } else if sim >= self.config.redundancy_threshold {
                    redundant_pairs.push((docs[i].id.clone(), docs[j].id.clone()));
                }
            }
        }

        let total_documents = docs.len();
        let quality_rate = if total_documents == 0 {
            1.0
        } else {
            curated_ids.len() as f32 / total_documents as f32
        };

        CurationReport {
            total_documents,
            contradictions_found,
            redundant_pairs,
            low_quality_ids,
            curated_ids,
            quality_rate,
            pairs_generated: Vec::new(),
        }
    }

    pub fn record_user_correction(
        &self,
        prompt: &str,
        rejected: &str,
        chosen: &str,
        context_embedding: Vec<f32>,
    ) -> PreferencePair {
        PreferencePair {
            prompt: prompt.to_string(),
            chosen_response: chosen.to_string(),
            rejected_response: rejected.to_string(),
            rejected_pathway: vec![0.0; 32],
            chosen_pathway: Some(vec![0.0; 32]),
            context_embedding,
            created_at: 0,
            source: PreferencePairSource::UserCorrection,
        }
    }
}
