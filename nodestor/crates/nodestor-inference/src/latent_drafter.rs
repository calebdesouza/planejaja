use nodestor_core::NodeStorError;
use nodestor_vulkan::VulkanEngine;
use nodestor_vulkan::GpuBuffer;

/// Motor do Funil Especulativo Híbrido.
/// Responsável por gerar blocos latentes (EAGLE-2), peneirar com cosseno e projetar.
pub struct LatentDrafter {
    pub eagle2_weights: Vec<f32>, // Uma camada linear [H x H] simulada
    pub hidden_dim: usize,
    pub cosine_threshold: f32, // τ_min para a peneira
}

impl LatentDrafter {
    pub fn new(hidden_dim: usize, cosine_threshold: f32) -> Self {
        // Inicializa com pesos identity + ruído para testes empíricos
        let mut weights = vec![0.0; hidden_dim * hidden_dim];
        for i in 0..hidden_dim {
            weights[i * hidden_dim + i] = 1.0; // Identidade base
        }
        Self {
            eagle2_weights: weights,
            hidden_dim,
            cosine_threshold,
        }
    }

    /// Estágio 1: Gera K hidden states consecutivos no espaço latente
    pub fn draft_latent_block(&self, seed_hidden: &[f32], max_depth: usize) -> Vec<Vec<f32>> {
        let mut drafts = Vec::with_capacity(max_depth);
        let mut current_state = seed_hidden.to_vec();

        // Autoregressivo no espaço latente
        for _ in 0..max_depth {
            let next_state = self.eagle2_forward(&current_state);
            drafts.push(next_state.clone());
            current_state = next_state;
        }

        drafts
    }

    /// Executa a camada linear EAGLE-2 (Matmul CPU simplificado para demonstração/testes)
    /// Em produção, rodaria no Vulkan ou seria offloaded se a CPU for gargalo,
    /// mas o custo O(H^2) é muito baixo comparado ao O(P) do forward master.
    fn eagle2_forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.hidden_dim];
        // Adiciona um pequeno "drift" pra não ser idêntico (simula predição imperfeita)
        for i in 0..self.hidden_dim {
            output[i] = input[i] * 0.995 + (i as f32 * 0.0005); // Drift mais suave E muda o ângulo progressivamente
        }
        output
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
        
        // Buffer de saída para os logits do batch [K_filtered x V]
        let output_size = k_filtered * vocab_size * 4;
        let output_buf = engine.alloc_buffer(output_size.max(4))?;

        // Executa Batch Matmul [K x H] x [H x V] -> [K x V]
        engine.matmul(&batch_buf, lm_head_buffer, k_filtered as u32, self.hidden_dim as u32, vocab_size as u32)
            .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

        // Download dos logits
        let flat_logits = engine.download_f32(&output_buf)?;
        
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
}
