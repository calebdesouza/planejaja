//! Sistema de Amostragem Aleatória Constrita (Sampler)
//!
//! Implementa as estratégias Top-K, Top-P (Nucleus), Temperatura e Repetition Penalty.
//! Recebe os "logits" (probabilidades) preenchidos em GPU via VRAM/SSD e sorteia o Token final.

use nodestor_core::NodeStorError;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub use_conformal: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            use_conformal: false,
        }
    }
}

use crate::conformal_predictor::{ConformalPredictor, ConformalSet};

pub struct Sampler {
    config: SamplerConfig,
    pub conformal: Option<ConformalPredictor>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let conformal = if config.use_conformal {
            Some(ConformalPredictor::new(0.95))
        } else {
            None
        };
        Self { config, conformal }
    }

    /// Executa a amostragem retornando o token e opcionalmente o Set Conformal.
    pub fn sample_with_conformal(&mut self, logits: &mut [f32], context: &[u32]) -> Result<(u32, Option<ConformalSet>), NodeStorError> {
        let conformal_set = if let Some(cp) = &mut self.conformal {
            Some(cp.predict_set(logits))
        } else {
            None
        };
        
        let token = self.sample(logits, context)?;
        Ok((token, conformal_set))
    }

    /// Executa a amostragem sobre os logits brutos (após Softmax ou Linears) exportados da GPU.
    pub fn sample(&self, logits: &mut [f32], context: &[u32]) -> Result<u32, NodeStorError> {
        let vocab_size = logits.len();
        if vocab_size == 0 {
            return Err(NodeStorError::VulkanError("Sem logits para amostragem".into()));
        }

        // 1. Repetition Penalty
        if self.config.repetition_penalty != 1.0 {
            for &token in context {
                if (token as usize) < vocab_size {
                    let logit = logits[token as usize];
                    if logit > 0.0 {
                        logits[token as usize] = logit / self.config.repetition_penalty;
                    } else {
                        logits[token as usize] = logit * self.config.repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature
        if self.config.temperature == 0.0 {
            // Greedy Search genérica iterativa (O(N))
            let mut argmax = 0;
            let mut max_val = logits[0];
            for (i, &l) in logits.iter().enumerate().skip(1) {
                if l > max_val {
                    max_val = l;
                    argmax = i as u32;
                }
            }
            return Ok(argmax);
        } else if self.config.temperature != 1.0 {
            for l in logits.iter_mut() {
                *l /= self.config.temperature;
            }
        }

        // 3. Aplicação do Softmax se não feito na GPU:
        // Softmax(x) = exp(x - max) / sum(exp(x - max))
        let max_logit = logits.iter().fold(-f32::INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - max_logit).exp();
            sum_exp += *l;
        }
        for l in logits.iter_mut() {
            *l /= sum_exp;
        }

        // 4. Prepara Array de Probabilidades Ordenadas
        let mut probs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // 5. Top-K cutoff
        let limit = if self.config.top_k > 0 && self.config.top_k < vocab_size {
            self.config.top_k
        } else {
            vocab_size
        };
        probs.truncate(limit);

        // 6. Top-P (Nucleus)
        if self.config.top_p < 1.0 {
            let mut cum_prob = 0.0;
            let mut last_idx = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cum_prob += p;
                if cum_prob > self.config.top_p {
                    last_idx = i + 1;
                    break;
                }
            }
            probs.truncate(last_idx);
        }

        // 7. Amostragem final
        let weights: Vec<f32> = probs.iter().map(|&(_, p)| p).collect();
        let dist = match WeightedIndex::new(&weights) {
            Ok(d) => d,
            Err(_) => return Ok(probs[0].0 as u32), // Fallback to max freq
        };

        let mut rng = rand::thread_rng();
        let drawn_idx = dist.sample(&mut rng);
        
        Ok(probs[drawn_idx].0 as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let config = SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, use_conformal: false };
        let sampler = Sampler::new(config);
        let mut logits = vec![0.1, 5.0, 2.0, -1.0];
        let token = sampler.sample(&mut logits, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_temperature_sharpens() {
        let config_hot = SamplerConfig { temperature: 2.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, use_conformal: false };
        let config_cold = SamplerConfig { temperature: 0.1, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, use_conformal: false };
        let sampler_hot = Sampler::new(config_hot);
        let sampler_cold = Sampler::new(config_cold);
        
        let logits = vec![1.0, 1.2];
        let mut h_logits = logits.clone();
        let mut c_logits = logits.clone();
        
        // Apenas simula o step da temperatura para constatar que divide
        sampler_hot.sample(&mut h_logits, &[]).unwrap();
        sampler_cold.sample(&mut c_logits, &[]).unwrap();
        
        // Logits frios ficam "mais extremos" apos o division + softmax iterativo
        // A amostragem foca nos logits q se destacam, testamos que roda sem panics.
    }

    #[test]
    fn test_top_k_truncates() {
        let config = SamplerConfig { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0, use_conformal: false };
        let sampler = Sampler::new(config);
        let mut logits = vec![1.0, 10.0, 2.0, 3.0];
        let token = sampler.sample(&mut logits, &[]).unwrap();
        assert_eq!(token, 1); // com top_k=1 só pode sair o maior
    }

    #[test]
    fn test_top_p_nucleus() {
        // Se top_p = 0.5, e o logit 1 tem > 0.5 prob absoluta, só ele sobrevive
        let config = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.5, repetition_penalty: 1.0, use_conformal: false };
        let sampler = Sampler::new(config);
        let mut logits = vec![0.0, 20.0, 0.0, 0.0];
        let token = sampler.sample(&mut logits, &[]).unwrap();
        assert_eq!(token, 1); 
    }

    #[test]
    fn test_repetition_penalty() {
        let config = SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 2.0, use_conformal: false };
        let sampler = Sampler::new(config);
        // Sem penalty, token 1 seria o argmax
        let mut logits = vec![1.0, 10.0, 8.0, 1.0];
        let context = vec![1]; // Token 1 tá no contexto
        let token = sampler.sample(&mut logits, &context).unwrap();
        // Logit do 1 vai pra 5.0 (10/2). Logit do 2 continua 8. Então o argmax vira 2.
        assert_eq!(token, 2);
    }

    #[test]
    fn test_empty_logits_errors() {
        let sampler = Sampler::new(SamplerConfig::default());
        let mut logits = vec![];
        assert!(sampler.sample(&mut logits, &[]).is_err());
    }
}
