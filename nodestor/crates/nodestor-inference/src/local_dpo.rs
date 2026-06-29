use crate::dataset_curator::PreferencePair;
use std::io::{Write, Read};

#[derive(Debug, Clone)]
pub struct DPOConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub beta: f32,
    pub replay_fraction: f32,
    pub convergence_threshold: f32,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 4,
            learning_rate: 1e-4,
            beta: 0.1,
            replay_fraction: 0.2,
            convergence_threshold: 0.001,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoRALayer {
    pub a_weights: Vec<f32>,
    pub b_weights: Vec<f32>,
    pub rank: usize,
    pub hidden: usize,
}

#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub rank: usize,
    pub version: u32,
    pub layers: Vec<LoRALayer>,
}

impl LoRAAdapter {
    pub fn new(n_layers: usize, hidden: usize, rank: usize) -> Self {
        let layers = (0..n_layers).map(|_| LoRALayer {
            a_weights: vec![0.0f32; rank * hidden],
            b_weights: vec![0.0f32; hidden * rank],
            rank,
            hidden,
        }).collect();
        Self { rank, version: 1, layers }
    }

    pub fn size_mb(&self) -> f32 {
        let total_params: usize = self.layers.iter()
            .map(|l| l.a_weights.len() + l.b_weights.len())
            .sum();
        (total_params * 4) as f32 / 1_048_576.0
    }

    pub fn trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.a_weights.len() + l.b_weights.len()).sum()
    }

    pub fn export(&self, path: &str) -> Result<(), String> {
        let mut buf = Vec::new();
        // Magic bytes
        buf.extend_from_slice(b"LORA");
        // Version (u32 LE)
        buf.extend_from_slice(&self.version.to_le_bytes());
        // Rank (u32 LE)
        buf.extend_from_slice(&(self.rank as u32).to_le_bytes());
        // n_layers (u32 LE)
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            buf.extend_from_slice(&(layer.hidden as u32).to_le_bytes());
            for w in &layer.a_weights { buf.extend_from_slice(&w.to_le_bytes()); }
            for w in &layer.b_weights { buf.extend_from_slice(&w.to_le_bytes()); }
        }
        std::fs::write(path, &buf).map_err(|e| e.to_string())
    }

    pub fn import(path: &str) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        if raw.len() < 16 || &raw[0..4] != b"LORA" {
            return Err("Invalid LORA magic".to_string());
        }
        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let rank    = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let n_layers = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        let mut pos = 16usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            if pos + 4 > raw.len() { return Err("Truncated LORA".to_string()); }
            let hidden = u32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let a_len = rank * hidden;
            let b_len = hidden * rank;
            let mut a_weights = Vec::with_capacity(a_len);
            for _ in 0..a_len {
                if pos + 4 > raw.len() { return Err("Truncated A".to_string()); }
                a_weights.push(f32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
                pos += 4;
            }
            let mut b_weights = Vec::with_capacity(b_len);
            for _ in 0..b_len {
                if pos + 4 > raw.len() { return Err("Truncated B".to_string()); }
                b_weights.push(f32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
                pos += 4;
            }
            layers.push(LoRALayer { a_weights, b_weights, rank, hidden });
        }
        Ok(Self { rank, version, layers })
    }
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub epochs_run: usize,
    pub final_loss: f32,
    pub pairs_used: usize,
    pub converged: bool,
    pub loss_history: Vec<f32>,
}

#[derive(Debug)]
pub enum DPOState {
    Idle,
    Training { epoch: usize },
    Completed { epochs: usize, final_loss: f32 },
    Converged { epoch: usize, loss: f32 },
}

pub struct LocalDPO {
    pub model_name: String,
    pub lora: LoRAAdapter,
    pub config: DPOConfig,
    pub state: DPOState,
    replay_buffer: Vec<PreferencePair>,
}

impl LocalDPO {
    pub fn new(model_name: &str, n_layers: usize, hidden: usize, rank: usize, config: DPOConfig) -> Self {
        Self {
            model_name: model_name.to_string(),
            lora: LoRAAdapter::new(n_layers, hidden, rank),
            config,
            state: DPOState::Idle,
            replay_buffer: Vec::new(),
        }
    }

    pub fn dpo_loss_pair(&self, pair: &PreferencePair) -> f32 {
        // Simplified DPO loss: -log_sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        // With mock logits based on text length
        let len_chosen = pair.chosen_response.len() as f32;
        let len_rejected = pair.rejected_response.len() as f32;
        let log_ratio = (len_chosen / (len_chosen + 1.0)).ln() - (len_rejected / (len_rejected + 1.0)).ln();
        let logit = self.config.beta * log_ratio;
        // -log_sigmoid(logit) = log(1 + exp(-logit))
        (1.0 + (-logit).exp()).ln().max(0.0)
    }

    pub fn train_session(&mut self, pairs: Vec<PreferencePair>) -> TrainingResult {
        self.state = DPOState::Training { epoch: 0 };

        let mut all_pairs = pairs.clone();
        // Add replay
        let replay_n = (self.replay_buffer.len() as f32 * self.config.replay_fraction) as usize;
        all_pairs.extend(self.replay_buffer.iter().take(replay_n).cloned());

        let pairs_used = all_pairs.len();
        let mut loss_history = Vec::new();
        let mut converged = false;

        let mut prev_loss = f32::MAX;
        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0f32;
            let n_batches = (all_pairs.len() / self.config.batch_size.max(1)).max(1);
            for batch_idx in 0..n_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(all_pairs.len());
                let batch = &all_pairs[start..end];
                let batch_loss: f32 = batch.iter().map(|p| self.dpo_loss_pair(p)).sum::<f32>() / batch.len() as f32;
                // Simulate gradient step: A weights drift slightly
                let lr = self.config.learning_rate;
                for layer in &mut self.lora.layers {
                    for w in &mut layer.a_weights {
                        *w -= lr * batch_loss * 0.001;
                    }
                }
                epoch_loss += batch_loss;
            }
            epoch_loss /= n_batches as f32;
            loss_history.push(epoch_loss);

            let delta = (prev_loss - epoch_loss).abs();
            if delta < self.config.convergence_threshold && epoch > 0 {
                converged = true;
                self.state = DPOState::Converged { epoch, loss: epoch_loss };
                break;
            }
            prev_loss = epoch_loss;
        }

        if !converged {
            let final_loss = *loss_history.last().unwrap_or(&0.0);
            self.state = DPOState::Completed { epochs: loss_history.len(), final_loss };
        }

        self.lora.version += 1;

        // Replenish replay buffer
        self.replay_buffer.extend(pairs);
        if self.replay_buffer.len() > 1000 {
            self.replay_buffer.drain(..self.replay_buffer.len() - 1000);
        }

        let final_loss = *loss_history.last().unwrap_or(&0.0);
        TrainingResult {
            epochs_run: loss_history.len().max(1),
            final_loss,
            pairs_used,
            converged,
            loss_history,
        }
    }
}
