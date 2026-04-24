
/// Modo de inferência detectado automaticamente pelo LayerGraph.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceMode {
    /// Modelo denso (LLaMA, Gemma, GPT). Usa Speculative Decoding.
    Dense,
    /// Mixture of Experts (Mixtral, DeepSeek, Qwen-MoE). Usa Expert Prefetch.
    MoE { num_experts: usize, top_k: usize },
    /// Modelo de difusão (SDXL, FLUX). Usa Streaming por Denoising Step.
    Diffusion { num_steps: usize },
}

impl Default for InferenceMode {
    fn default() -> Self {
        InferenceMode::Dense
    }
}

/// Orçamento de VRAM calculado em runtime sobre o LIVRE real.
///
/// Fórmula:
/// ```text
/// VRAM_LIVRE  = VRAM_TOTAL - VRAM_USADA_PELO_SO
/// TETO        = VRAM_LIVRE * 0.75  (margem de segurança)
/// ```
///
/// Proporções por modo:
/// - Dense:     primary=60% (Embedding+LMHead+HNSW+BM25),  kv=40%
/// - MoE:       primary=30% (Router+SharedAttn),  cache=40%, kv=30%
/// - Diffusion: primary=20% (Timestep Embed),  cache=50%, kv=30%
#[derive(Debug, Clone)]
pub struct VramBudget {
    /// VRAM total detectada pelo scanner (bytes).
    pub total_vram: u64,
    /// VRAM atualmente usada pelo SO + driver + DWM (bytes, runtime query).
    pub used_by_system: u64,
    /// VRAM livre real (total - usada pelo sistema).
    pub free_vram: u64,
    /// Teto do NodeStor: 75% do livre.
    pub nodestor_ceiling: u64,
    /// Budget para componentes primários (Embedding, Router, Timestep).
    pub primary_budget: u64,
    /// Budget para caches (Expert LRU, Feature Cache).
    pub cache_budget: u64,
    /// Budget para KV Cache / Latents.
    pub kv_budget: u64,
    /// Modo de inferência que define as proporções.
    pub mode: InferenceMode,
}

impl VramBudget {
    /// Cria um VramBudget com valores explícitos (para testes).
    pub fn new(
        total_vram: u64,
        used_by_system: u64,
        mode: InferenceMode,
    ) -> Self {
        let free_vram = total_vram.saturating_sub(used_by_system);
        let nodestor_ceiling = (free_vram as f64 * 0.75) as u64;
        let (p_ratio, c_ratio, kv_ratio) = Self::ratios_for_mode(&mode);

        Self {
            total_vram,
            used_by_system,
            free_vram,
            nodestor_ceiling,
            primary_budget: (nodestor_ceiling as f64 * p_ratio) as u64,
            cache_budget: (nodestor_ceiling as f64 * c_ratio) as u64,
            kv_budget: (nodestor_ceiling as f64 * kv_ratio) as u64,
            mode,
        }
    }

    /// Detecta automaticamente a VRAM livre via Vulkan Budget Extension.
    /// Se a extensão não estiver disponível, usa heurística (total - 20%).
    pub fn detect(total_vram: u64, mode: InferenceMode) -> Self {
        // Em runtime real usaríamos vkGetPhysicalDeviceMemoryBudgetPropertiesEXT.
        // Por segurança, estimamos 18% de uso pelo sistema como baseline.
        let estimated_system_use = (total_vram as f64 * 0.18) as u64;
        Self::new(total_vram, estimated_system_use, mode)
    }

    /// Fallback: estima o budget sem query Vulkan.
    pub fn estimate(total_vram: u64) -> Self {
        let estimated_system_use = (total_vram as f64 * 0.20) as u64;
        Self::new(total_vram, estimated_system_use, InferenceMode::Dense)
    }

    /// Retorna as proporções (primary, cache, kv) para cada modo.
    fn ratios_for_mode(mode: &InferenceMode) -> (f64, f64, f64) {
        match mode {
            InferenceMode::Dense => (0.60, 0.0, 0.40),
            InferenceMode::MoE { .. } => (0.30, 0.40, 0.30),
            InferenceMode::Diffusion { .. } => (0.20, 0.50, 0.30),
        }
    }

    /// Verifica se `bytes` cabe no budget primário.
    pub fn can_fit_primary(&self, bytes: u64) -> bool {
        bytes <= self.primary_budget
    }

    /// Verifica se `bytes` cabe no budget de KV Cache.
    pub fn can_fit_kv(&self, bytes: u64) -> bool {
        bytes <= self.kv_budget
    }

    /// Atualiza o budget com uma nova leitura de uso do sistema.
    pub fn refresh(&mut self, used_by_system: u64) {
        self.used_by_system = used_by_system;
        self.free_vram = self.total_vram.saturating_sub(used_by_system);
        self.nodestor_ceiling = (self.free_vram as f64 * 0.75) as u64;
        let (p_ratio, c_ratio, kv_ratio) = Self::ratios_for_mode(&self.mode);
        self.primary_budget = (self.nodestor_ceiling as f64 * p_ratio) as u64;
        self.cache_budget = (self.nodestor_ceiling as f64 * c_ratio) as u64;
        self.kv_budget = (self.nodestor_ceiling as f64 * kv_ratio) as u64;
    }

    /// Relatório legível do orçamento.
    pub fn report(&self) -> String {
        format!(
            "VramBudget [{:?}]\n  Total:   {:>8} MB\n  Sistema: {:>8} MB\n  Livre:   {:>8} MB\n  Teto 75%:{:>8} MB\n  Primary: {:>8} MB\n  Cache:   {:>8} MB\n  KV:      {:>8} MB",
            self.mode,
            self.total_vram / 1024 / 1024,
            self.used_by_system / 1024 / 1024,
            self.free_vram / 1024 / 1024,
            self.nodestor_ceiling / 1024 / 1024,
            self.primary_budget / 1024 / 1024,
            self.cache_budget / 1024 / 1024,
            self.kv_budget / 1024 / 1024,
        )
    }

    /// Calcula quantos bytes um embedding Q6_K ocupa.
    /// Q6_K: 6 bits por peso. Para dim D e vocab size V:
    /// bytes = V * D * 6 / 8 (arredondado para cima)
    pub fn q6k_embedding_bytes(vocab_size: u64, embed_dim: u64) -> u64 {
        (vocab_size * embed_dim * 6 + 7) / 8
    }

    /// Verifica se o Draft (Embedding Q6_K + LM Head Q6_K) cabe no primary budget.
    pub fn can_fit_draft_q6k(&self, vocab_size: u64, embed_dim: u64) -> bool {
        let emb_bytes = Self::q6k_embedding_bytes(vocab_size, embed_dim);
        let lmhead_bytes = emb_bytes; // LM Head tem mesma forma que Embedding
        let total = emb_bytes + lmhead_bytes;
        self.can_fit_primary(total)
    }
}

impl std::fmt::Display for VramBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_dense_8gb() {
        // GPU 8 GB, SO usa ~1.5 GB
        let budget = VramBudget::new(
            8 * 1024 * 1024 * 1024,
            1536 * 1024 * 1024,
            InferenceMode::Dense,
        );

        // Livre = 8 - 1.5 = 6.5 GB
        let free_gb = budget.free_vram as f64 / 1024.0 / 1024.0 / 1024.0;
        assert!((free_gb - 6.5).abs() < 0.1, "Livre deveria ser ~6.5 GB, foi {:.2}", free_gb);

        // Teto = 6.5 * 0.75 ≈ 4.87 GB
        let ceiling_gb = budget.nodestor_ceiling as f64 / 1024.0 / 1024.0 / 1024.0;
        assert!((ceiling_gb - 4.875).abs() < 0.1, "Teto ~4.87 GB, foi {:.2}", ceiling_gb);

        // Primary (Dense) = 60% do teto ≈ 2.93 GB
        assert!(budget.primary_budget > 0);
        assert!(budget.primary_budget < budget.nodestor_ceiling);

        println!("{}", budget.report());
    }

    #[test]
    fn test_budget_moe_12gb() {
        let budget = VramBudget::new(
            12 * 1024 * 1024 * 1024,
            1200 * 1024 * 1024,
            InferenceMode::MoE { num_experts: 8, top_k: 2 },
        );

        // MoE: primary=30%, cache=40%, kv=30%
        let total_allocated = budget.primary_budget + budget.cache_budget + budget.kv_budget;
        assert!(total_allocated <= budget.nodestor_ceiling,
            "Budgets não devem exceder teto: {} <= {}", total_allocated, budget.nodestor_ceiling);

        println!("{}", budget.report());
    }

    #[test]
    fn test_never_exceed_ceiling() {
        for total_gb in [4u64, 8, 12, 16, 24] {
            let budget = VramBudget::estimate(total_gb * 1024 * 1024 * 1024);
            let sum = budget.primary_budget + budget.cache_budget + budget.kv_budget;
            assert!(sum <= budget.nodestor_ceiling,
                "{}GB GPU: budgets ({} MB) excedem teto ({} MB)",
                total_gb,
                sum / 1024 / 1024,
                budget.nodestor_ceiling / 1024 / 1024
            );
        }
    }

    #[test]
    fn test_refresh_reduces_budget() {
        let mut budget = VramBudget::new(
            8 * 1024 * 1024 * 1024,
            1024 * 1024 * 1024,
            InferenceMode::Dense,
        );
        let initial_ceiling = budget.nodestor_ceiling;

        // Simula mais uso pelo sistema (Chrome abriu, etc.)
        budget.refresh(3 * 1024 * 1024 * 1024);
        assert!(budget.nodestor_ceiling < initial_ceiling,
            "Teto deve reduzir quando sistema usa mais VRAM");
    }

    #[test]
    fn test_q6k_size_calculation() {
        // LLaMA 3 8B: vocab=128256, embed_dim=4096
        let bytes = VramBudget::q6k_embedding_bytes(128256, 4096);
        let mb = bytes / 1024 / 1024;
        // FP16 seria: 128256 * 4096 * 2 = ~1 GB
        // Q6_K deveria ser ~375 MB (2.67x menor)
        assert!(mb < 400, "Q6_K embedding deve ser < 400 MB para 8B, foi {} MB", mb);
        assert!(mb > 200, "Q6_K embedding deve ser > 200 MB para 8B, foi {} MB", mb);
        println!("LLaMA 3 8B Embedding Q6_K: {} MB", mb);
    }

    #[test]
    fn test_can_fit_draft_llama_8b_on_8gb() {
        // RTX 4060, SO usa ~1.5 GB
        let budget = VramBudget::new(
            8 * 1024 * 1024 * 1024,
            1536 * 1024 * 1024,
            InferenceMode::Dense,
        );
        // LLaMA 3 8B: vocab=128256, embed_dim=4096
        assert!(
            budget.can_fit_draft_q6k(128256, 4096),
            "Draft Q6_K do LLaMA 8B deve caber em 8 GB GPU"
        );
    }
}
