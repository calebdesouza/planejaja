use nodestor_core::NodeStorError;
use nodestor_vulkan::VulkanEngine;
use nodestor_vulkan::GpuBuffer;

use crate::hamiltonian_dynamics::HamiltonianLatentDynamics;
use crate::hnsw_index::HnswIndex;
use crate::lsh_buckets::LshVocabIndex;
use std::sync::Arc;

/// Motor do Funil Especulativo Híbrido.
/// Responsável por gerar blocos latentes (Hamiltoniano), peneirar com cosseno e projetar.
pub struct LatentDrafter {
    pub hidden_dim: usize,
    pub cosine_threshold: f32, // τ_min para a peneira
    pub dynamics: HamiltonianLatentDynamics,
    pub hnsw_index: Option<Arc<HnswIndex>>,
    pub lsh_index: Option<Arc<LshVocabIndex>>,
}

impl LatentDrafter {
    pub fn new(hidden_dim: usize, cosine_threshold: f32) -> Self {
        Self {
            hidden_dim,
            cosine_threshold,
            // Hamiltoniano: dt=0.05, budget=1.0. Em produção afinaríamos esses valores
            dynamics: HamiltonianLatentDynamics::new(hidden_dim, 0.05, 1.0),
            hnsw_index: None,
            lsh_index: None,
        }
    }

    /// Adiciona os índices HNSW e LSH para projeção O(1) da LM Head
    pub fn with_indices(mut self, hnsw: Arc<HnswIndex>, lsh: Arc<LshVocabIndex>) -> Self {
        self.hnsw_index = Some(hnsw);
        self.lsh_index = Some(lsh);
        self
    }

    /// Estágio 1: Gera K hidden states consecutivos no espaço latente via Hamiltoniano
    pub fn draft_latent_block(&self, seed_hidden: &[f32], max_depth: usize) -> Vec<Vec<f32>> {
        // Inicializa o momentum semântico (p_0)
        let seed_p = self.dynamics.init_momentum(seed_hidden, 0.01);
        
        // Evolui usando Störmer-Verlet (Leapfrog), sem acúmulo de erro exponencial (zero drift)
        let (drafts, _drift, _ok) = self.dynamics.evolve(seed_hidden, &seed_p, max_depth);
        
        drafts
    }

    /// Estágio 2: Peneira cossenóide
    /// Retorna quantos drafts sobreviveram (comprimento do prefixo contíguo válido)
    pub fn cosine_sieve(&self, drafts: &[Vec<f32>], reference: &[f32]) -> usize {
        let mut k_filtered = 0;
        let current_ref = reference;

        for draft in drafts {
            let cos_sim = Self::cosine_similarity(draft, current_ref);
            if cos_sim < self.cosine_threshold {
                break; // Corta no primeiro que diverge
            }
            k_filtered += 1;
            // O próximo reference para o cosseno deveria ser o hidden state real gerado
            // Como não temos, usamos o próprio draft validado iterativamente como heurística,
            // ou comparamos todos com o reference raiz ajustado.
            // Para simplificar: comparamos com o seed inicial. Na prática o τ lida com a divergência angular aceitável da bacia.
        }
        k_filtered
    }

    /// Estágio 3: Projeta os sobreviventes em tokens usando a LM Head real
    pub fn project_to_tokens(
        &self,
        engine: &VulkanEngine,
        accepted_states: &[Vec<f32>],
        lm_head_buffer: &GpuBuffer,
        vocab_size: usize,
    ) -> Result<Vec<u32>, NodeStorError> {
        if accepted_states.is_empty() {
            return Ok(Vec::new());
        }

        // Fase 3 Quântico-Latente: O(1) LM Head Projection
        if let (Some(hnsw), Some(lsh)) = (&self.hnsw_index, &self.lsh_index) {
            let mut tokens = Vec::with_capacity(accepted_states.len());
            for state in accepted_states {
                // Lookup O(1) via LSH para achar o sub-espaço (bucket) semântico
                let candidates: Vec<usize> = lsh.lookup(state);
                
                // Em produção real, calcularíamos o argmax explícito só nos `candidates` (ex: 30-80 dot products vs 128k).
                // Mas aqui usamos a topologia HNSW em O(log V) para demonstrar a busca indexada pura.
                let top1: Vec<(usize, f32)> = hnsw.search(state, 1);
                if let Some(&(tok, _)) = top1.first() {
                    tokens.push(tok as u32);
                } else {
                    tokens.push(0); // Fallback
                }
            }
            return Ok(tokens);
        }

        let k_filtered = accepted_states.len();
        
        // Achata os vetores
        let mut flat_states = Vec::with_capacity(k_filtered * self.hidden_dim);
        for state in accepted_states {
            flat_states.extend_from_slice(state);
        }

        // Upload batch states
        let state_bytes = unsafe {
            std::slice::from_raw_parts(
                flat_states.as_ptr() as *const u8,
                flat_states.len() * 4,
            )
        };
        let batch_buf = engine.upload(state_bytes)?;
        
        // Executa Batch Matmul [K x H] x [H x V] -> [K x V]
        let computed_buf = engine.matmul(&batch_buf, lm_head_buffer, k_filtered as u32, self.hidden_dim as u32, vocab_size as u32)
            .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

        // Download dos logits
        let flat_logits = engine.download_f32(&computed_buf)?;
        
        // Argmax para cada token no batch
        let mut tokens = Vec::with_capacity(k_filtered);
        for i in 0..k_filtered {
            let start = i * vocab_size;
            let end = start + vocab_size;
            let token_logits = &flat_logits[start..end];
            
            // Argmax
            let mut best_tok = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in token_logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_tok = idx as u32;
                }
            }
            tokens.push(best_tok);
        }

        Ok(tokens)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }

    /// PRM (Process Reward Model): Avalia um estado latente para o MCTS
    /// Retorna uma probabilidade/recompensa entre 0 e 1 de que o estado leva à solução correta.
    /// Em vez de chamar a rede PRM inteira, no NodeStor avaliamos a "surpresa" (entropia)
    /// ou a estabilidade do estado no índice HNSW (clustering semântico).
    pub fn evaluate_state(&self, state: &[f32]) -> f32 {
        if let Some(hnsw) = &self.hnsw_index {
            // Se o estado está muito próximo do centroide de um cluster forte, recompensa alta
            let top_k: Vec<(usize, f32)> = hnsw.search(state, 5);
            let mut avg_dist = 0.0;
            for &(_, dist) in &top_k {
                avg_dist += dist;
            }
            if !top_k.is_empty() {
                avg_dist /= top_k.len() as f32;
                // Distância menor -> similaridade maior -> recompensa maior
                // Assume que dist é distância L2. Recompensa cai exponencialmente
                return (- (avg_dist as f32)).exp();
            }
        }
        
        // Fallback: recompensa base neutra/otimista
        0.5
    }
}
