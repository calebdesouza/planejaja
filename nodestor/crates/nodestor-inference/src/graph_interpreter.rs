//! Graph Interpreter — O Fim da Manutenção Manual.
//!
//! Lê qualquer GGUF e monta automaticamente a "Lista de Execução"
//! de operadores para o `OperatorRegistry`.
//!
//! ## Como funciona:
//! ```text
//! 1. Parser GGUF existente lê os metadados e lista os tensores
//! 2. GraphInterpreter detecta a arquitetura pelos NOMES dos tensores:
//!    "blk.0.attn_q.weight" → Llama
//!    "encoder.layer.0.attention.self.query.weight" → BERT
//! 3. Monta a sequência exata de TensorOps para aquela arquitetura
//! 4. O OperatorRegistry executa os mesmos kernels Vulkan do LLM
//! ```
//!
//! ## Quando sair o "Embedding-Modelo-X" em 2027:
//! Se usarem Matmul + LayerNorm + Attention (todo transformer usa),
//! o NodeStor roda no dia 1, sem uma linha de Rust nova.

use nodestor_core::{ModelMetadata, NodeStorError};
use nodestor_vulkan::operator_registry::{TensorOp, ActivationKind};
use std::collections::HashMap;

/// Arquitetura de modelo detectada automaticamente a partir do GGUF.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    /// Família Llama: `RmsNorm + SiLU/SwiGLU + RoPE + GQA`
    /// Exemplos: Llama 3, Mistral, Qwen, Phi-3, DeepSeek
    Llama {
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        intermediate_size: u32,
        head_dim: u32,
        vocab_size: u32,
        rope_base: f32,
        is_moe: bool,
        num_experts: u32,
    },
    /// Família BERT: `LayerNorm + GELU + Absolute Position + Bidirecional`
    /// Exemplos: BERT, RoBERTa, DistilBERT
    Bert {
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        intermediate_size: u32,
        vocab_size: u32,
    },
    /// BGE/E5 (BERT para embeddings): mesma estrutura + MeanPool final
    /// Exemplos: BGE-small/base/large, E5-small/base/large, Nomic-embed
    BertEmbedding {
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        intermediate_size: u32,
        vocab_size: u32,
        output_dim: u32,
    },
    /// Família GPT-2: `LayerNorm + GELU + Absolute Position + Causal`
    /// Exemplos: GPT-2, CodeGPT
    Gpt2 {
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        intermediate_size: u32,
        vocab_size: u32,
    },
    /// Arquitetura desconhecida — tenta inferir dos tensores
    Unknown {
        tensor_names: Vec<String>,
        estimated_layers: u32,
    },
}

impl ModelArchitecture {
    /// Nome legível da arquitetura.
    pub fn name(&self) -> &'static str {
        match self {
            ModelArchitecture::Llama { .. }          => "Llama",
            ModelArchitecture::Bert { .. }           => "BERT",
            ModelArchitecture::BertEmbedding { .. }  => "BERT-Embedding",
            ModelArchitecture::Gpt2 { .. }           => "GPT-2",
            ModelArchitecture::Unknown { .. }        => "Unknown",
        }
    }

    /// `true` se o modelo é de embedding (e.g. BGE, E5).
    pub fn is_embedding_model(&self) -> bool {
        matches!(self, ModelArchitecture::BertEmbedding { .. })
    }
}

/// Tipo de saída do modelo.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Gera tokens (LLM) — Llama, GPT-2
    Generative,
    /// Gera vetores (RAG) — BGE, E5, Nomic
    Embedding,
    /// Gera probabilidades de classes — BERT classification heads
    Classifier,
}

/// O grafo de execução montado pelo `GraphInterpreter`.
/// Contém tudo o que o `OperatorRegistry` precisa para rodar o modelo.
#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    /// Arquitetura detectada
    pub architecture: ModelArchitecture,
    /// Lista ordenada de operações (a "receita" do modelo)
    pub ops: Vec<TensorOp>,
    /// Mapeamento: nome do tensor GGUF → chave no BufferPool
    pub tensor_binding: HashMap<String, String>,
    /// Tipo de modelo: generativo, embedding ou classificador
    pub model_type: ModelType,
    /// Dimensão do embedding de saída (apenas para modelos de embedding)
    pub embedding_dim: Option<u32>,
    /// Nome completo do modelo (do metadado GGUF, se disponível)
    pub model_name: String,
}

impl ExecutionGraph {
    /// Número de operadores no grafo.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Número de camadas Transformer no modelo.
    pub fn num_layers(&self) -> u32 {
        match &self.architecture {
            ModelArchitecture::Llama { num_layers, .. } => *num_layers,
            ModelArchitecture::Bert { num_layers, .. } => *num_layers,
            ModelArchitecture::BertEmbedding { num_layers, .. } => *num_layers,
            ModelArchitecture::Gpt2 { num_layers, .. } => *num_layers,
            ModelArchitecture::Unknown { estimated_layers, .. } => *estimated_layers,
        }
    }

    /// Dimensão do espaço de embedding interno.
    pub fn hidden_dim(&self) -> u32 {
        match &self.architecture {
            ModelArchitecture::Llama { hidden_dim, .. } => *hidden_dim,
            ModelArchitecture::Bert { hidden_dim, .. } => *hidden_dim,
            ModelArchitecture::BertEmbedding { hidden_dim, .. } => *hidden_dim,
            ModelArchitecture::Gpt2 { hidden_dim, .. } => *hidden_dim,
            ModelArchitecture::Unknown { .. } => 0,
        }
    }
}

/// O Interpretador de Grafo Dinâmico.
///
/// A MÁGICA: em vez de codificar "modelos", codificamos "operadores".
/// Este interpretador instrui o motor sobre como montar qualquer modelo
/// a partir de peças de LEGO (Matmul, Attention, LayerNorm...).
pub struct GraphInterpreter;

impl GraphInterpreter {
    /// Analisa os metadados de um GGUF e retorna o `ExecutionGraph` pronto.
    ///
    /// Chame com o `ModelMetadata` retornado pelo parser GGUF existente
    /// em `nodestor-formats`.
    pub fn interpret(metadata: &ModelMetadata) -> Result<ExecutionGraph, NodeStorError> {
        let architecture = Self::detect_architecture(metadata)?;
        let model_type   = Self::infer_model_type(&architecture, metadata);
        let embedding_dim = Self::infer_embedding_dim(&architecture);
        let tensor_binding = Self::map_tensors(&architecture, metadata);
        let ops = Self::build_execution_plan(&architecture)?;
        let model_name = metadata.model_name.clone()
            .unwrap_or_else(|| "unknown".to_string());

        Ok(ExecutionGraph {
            architecture,
            ops,
            tensor_binding,
            model_type,
            embedding_dim,
            model_name,
        })
    }

    // ─── Detecção de Arquitetura ─────────────────────────────────────────────

    /// Detecta a arquitetura baseando-se nos nomes dos tensores no GGUF.
    ///
    /// Heurísticas (validadas com modelos reais da HuggingFace GGUF Hub):
    /// - `blk.0.attn_q.weight`           → Llama-family (llama.cpp naming)
    /// - `encoder.layer.0.attention...`  → BERT (HF naming)
    /// - `encoder.layer.0.attention...` + sem `cls.predictions` → BGE/E5
    /// - `h.0.attn.c_attn.weight`        → GPT-2 (HF naming)
    pub fn detect_architecture(metadata: &ModelMetadata) -> Result<ModelArchitecture, NodeStorError> {
        let tensor_names: Vec<&str> = metadata.tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        // 1. Llama-family (llama.cpp GGUF naming convention)
        let has_blk_attn = tensor_names.iter().any(|n| n.contains("blk.") && n.contains("attn_q"));
        let has_blk_ffn  = tensor_names.iter().any(|n| n.contains("blk.") && n.contains("ffn_gate"));

        if has_blk_attn || has_blk_ffn {
            let num_layers = Self::count_layers_by_prefix(&tensor_names, "blk.");
            let (hidden_dim, num_heads, num_kv_heads, intermediate_size) =
                Self::extract_llama_dims(metadata);

            let is_moe = tensor_names.iter().any(|n| n.contains("ffn_gate_exps") || n.contains("ffn_gate_inp"));
            let num_experts = if is_moe {
                Self::try_u32_from_extra(&metadata.extra, &["llama.expert_count", "model.num_local_experts"]).unwrap_or(8)
            } else {
                0
            };

            let rope_base = Self::try_f32_from_extra(&metadata.extra, &[
                "llama.rope.freq_base", "qwen2.rope.freq_base", "gemma.rope.freq_base",
                "phi3.rope.freq_base", "model.rope_theta", "rope_theta",
            ]).unwrap_or(10000.0);

            return Ok(ModelArchitecture::Llama {
                num_layers,
                hidden_dim,
                num_heads,
                num_kv_heads,
                intermediate_size,
                head_dim: if num_heads > 0 { hidden_dim / num_heads } else { 64 },
                vocab_size: Self::extract_vocab_size(metadata),
                rope_base,
                is_moe,
                num_experts,
            });
        }

        // 2. BERT / BGE-family (HuggingFace GGUF naming)
        let has_encoder_layer = tensor_names.iter()
            .any(|n| n.contains("encoder.layer.") && n.contains("attention.self.query"));

        if has_encoder_layer {
            let num_layers = Self::count_layers_by_prefix(&tensor_names, "encoder.layer.");
            let (hidden_dim, num_heads, intermediate_size) =
                Self::extract_bert_dims(metadata);

            // BGE/E5: sem head de classificação "cls.predictions" ou "classifier"
            let is_embedding_only = !tensor_names.iter()
                .any(|n| n.contains("cls.predictions") || n.contains("classifier"));

            let vocab_size = Self::extract_vocab_size(metadata);

            if is_embedding_only {
                return Ok(ModelArchitecture::BertEmbedding {
                    num_layers,
                    hidden_dim,
                    num_heads,
                    intermediate_size,
                    vocab_size,
                    output_dim: hidden_dim, // BGE output_dim = hidden_dim
                });
            } else {
                return Ok(ModelArchitecture::Bert {
                    num_layers,
                    hidden_dim,
                    num_heads,
                    intermediate_size,
                    vocab_size,
                });
            }
        }

        // 3. GPT-2 naming
        let has_gpt2 = tensor_names.iter()
            .any(|n| n.starts_with("h.") && n.contains("attn.c_attn"));

        if has_gpt2 {
            let num_layers = Self::count_layers_by_prefix(&tensor_names, "h.");
            let (hidden_dim, num_heads, intermediate_size) = Self::extract_bert_dims(metadata);
            return Ok(ModelArchitecture::Gpt2 {
                num_layers,
                hidden_dim,
                num_heads,
                intermediate_size,
                vocab_size: Self::extract_vocab_size(metadata),
            });
        }

        // 4. Fallback: arquitetura desconhecida, tenta inferir
        let estimated_layers = Self::count_layers_by_prefix(&tensor_names, "layer.");
        Ok(ModelArchitecture::Unknown {
            tensor_names: tensor_names.iter().take(20).map(|s| s.to_string()).collect(),
            estimated_layers: estimated_layers.max(1),
        })
    }

    // ─── Construção do Plano de Execução ─────────────────────────────────────

    /// Monta a sequência de `TensorOp` para a arquitetura detectada.
    pub fn build_execution_plan(arch: &ModelArchitecture) -> Result<Vec<TensorOp>, NodeStorError> {
        match arch {
            ModelArchitecture::Llama {
                num_layers, hidden_dim, num_heads, num_kv_heads,
                intermediate_size, head_dim, vocab_size, rope_base, is_moe, num_experts,
            } => Ok(Self::build_llama_plan(
                *num_layers, *hidden_dim, *num_heads, *num_kv_heads,
                *intermediate_size, *head_dim, *vocab_size, *rope_base, *is_moe, *num_experts,
            )),

            ModelArchitecture::Bert {
                num_layers, hidden_dim, num_heads, intermediate_size, vocab_size,
            } => Ok(Self::build_bert_plan(*num_layers, *hidden_dim, *num_heads, *intermediate_size, *vocab_size)),

            ModelArchitecture::BertEmbedding {
                num_layers, hidden_dim, num_heads, intermediate_size, vocab_size, ..
            } => Ok(Self::build_bge_plan(*num_layers, *hidden_dim, *num_heads, *intermediate_size, *vocab_size)),

            ModelArchitecture::Gpt2 {
                num_layers, hidden_dim, num_heads, intermediate_size, vocab_size,
            } => Ok(Self::build_gpt2_plan(*num_layers, *hidden_dim, *num_heads, *intermediate_size, *vocab_size)),

            ModelArchitecture::Unknown { estimated_layers, .. } => {
                // Melhor esforço: usa Llama-like com defaults conservadores
                Ok(Self::build_llama_plan(*estimated_layers, 4096, 32, 32, 11008, 128, 32000, 10000.0, false, 0))
            }
        }
    }

    /// Plano Llama: `[Embed] → N×[RmsNorm+QKV+RoPE+Attn+RmsNorm+FFN(SiLU)] → [RmsNorm+LMHead+Softmax]`
    fn build_llama_plan(
        num_layers: u32, hidden_dim: u32, num_heads: u32, num_kv_heads: u32,
        intermediate_size: u32, head_dim: u32, vocab_size: u32, rope_base: f32,
        is_moe: bool, num_experts: u32,
    ) -> Vec<TensorOp> {
        let mut ops = Vec::new();

        // Embedding inicial
        ops.push(TensorOp::EmbeddingLookup { vocab_size, hidden_dim });

        // N camadas Transformer Llama
        for layer_idx in 0..num_layers {
            // Attention sublayer
            ops.push(TensorOp::RmsNorm { hidden_size: hidden_dim, eps: 1e-5 });
            // Q, K, V projections (3 matmuls)
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: num_heads * head_dim, quantized: false });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: num_kv_heads * head_dim, quantized: false });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: num_kv_heads * head_dim, quantized: false });
            // RoPE
            ops.push(TensorOp::RoPE { head_dim, freq_base: rope_base, start_pos: layer_idx, seq_len: 1 });
            // Multi-head attention
            ops.push(TensorOp::Attention { num_heads, num_kv_heads, head_dim, seq_len: 1 });
            // Output projection
            ops.push(TensorOp::Matmul { m: 1, k: num_heads * head_dim, n: hidden_dim, quantized: false });
            // Residual
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });

            // FFN sublayer (SwiGLU = gate_proj × up_proj → SiLU → down_proj)
            ops.push(TensorOp::RmsNorm { hidden_size: hidden_dim, eps: 1e-5 });
            if is_moe {
                // Ao invés do FFN denso, despacha a operação de roteamento MoE (RaBitQ 1-bit logic)
                ops.push(TensorOp::MoERouting { num_experts, top_k: 2, hard_k1: false });
            } else {
                ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: intermediate_size, quantized: false });
                ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: intermediate_size, quantized: false });
                ops.push(TensorOp::Activation { kind: ActivationKind::SiLU, elements: intermediate_size });
                ops.push(TensorOp::Matmul { m: 1, k: intermediate_size, n: hidden_dim, quantized: false });
            }
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
        }

        // Final norm + LM head + Softmax
        ops.push(TensorOp::RmsNorm { hidden_size: hidden_dim, eps: 1e-5 });
        ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: vocab_size, quantized: false });
        ops.push(TensorOp::Softmax { seq_len: 1, dim: vocab_size });

        ops
    }

    /// Plano BERT: `[Embed+Pos] → N×[LayerNorm+Attn+Add+LayerNorm+FFN(GELU)+Add] → [Pooler]`
    fn build_bert_plan(
        num_layers: u32, hidden_dim: u32, num_heads: u32,
        intermediate_size: u32, vocab_size: u32,
    ) -> Vec<TensorOp> {
        let head_dim = hidden_dim / num_heads.max(1);
        let mut ops = Vec::new();

        ops.push(TensorOp::EmbeddingLookup { vocab_size, hidden_dim });

        for _ in 0..num_layers {
            // Self-Attention sublayer
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-12 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false }); // Q
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false }); // K
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false }); // V
            ops.push(TensorOp::Attention { num_heads, num_kv_heads: num_heads, head_dim, seq_len: 128 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false }); // Output
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-12 });

            // FFN sublayer (BERT usa GELU, não SiLU)
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: intermediate_size, quantized: false });
            ops.push(TensorOp::Activation { kind: ActivationKind::GELU, elements: intermediate_size });
            ops.push(TensorOp::Matmul { m: 1, k: intermediate_size, n: hidden_dim, quantized: false });
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
        }

        // BERT final: Softmax para classificação (pode ser omitido para feature extraction)
        ops.push(TensorOp::Softmax { seq_len: 1, dim: hidden_dim });
        ops
    }

    /// Plano BGE/E5: mesma estrutura BERT + MeanPool final (em vez de Softmax).
    /// O MeanPool é o que transforma a sequência em embedding de busca.
    fn build_bge_plan(
        num_layers: u32, hidden_dim: u32, num_heads: u32,
        intermediate_size: u32, vocab_size: u32,
    ) -> Vec<TensorOp> {
        let head_dim = hidden_dim / num_heads.max(1);
        let mut ops = Vec::new();

        ops.push(TensorOp::EmbeddingLookup { vocab_size, hidden_dim });

        for _ in 0..num_layers {
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-12 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false });
            ops.push(TensorOp::Attention { num_heads, num_kv_heads: num_heads, head_dim, seq_len: 128 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false });
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-12 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: intermediate_size, quantized: false });
            ops.push(TensorOp::Activation { kind: ActivationKind::GELU, elements: intermediate_size });
            ops.push(TensorOp::Matmul { m: 1, k: intermediate_size, n: hidden_dim, quantized: false });
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
        }

        // === A DIFERENÇA DO BGE: MeanPool em vez de Softmax ===
        // Transforma hidden states [T, D] → embedding [D]
        ops.push(TensorOp::MeanPool { seq_len: 128, hidden_dim });
        // Normalização L2 (implícita — a maioria dos benchmarks espera vetores L2-normalizados)

        ops
    }

    /// Plano GPT-2: `[Embed+Pos] → N×[LayerNorm+Attn+Add+LayerNorm+FFN(GELU)+Add] → [LMHead+Softmax]`
    fn build_gpt2_plan(
        num_layers: u32, hidden_dim: u32, num_heads: u32,
        intermediate_size: u32, vocab_size: u32,
    ) -> Vec<TensorOp> {
        let head_dim = hidden_dim / num_heads.max(1);
        let mut ops = Vec::new();

        ops.push(TensorOp::EmbeddingLookup { vocab_size, hidden_dim });

        for _ in 0..num_layers {
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-5 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim * 3, quantized: false }); // QKV combined
            ops.push(TensorOp::Attention { num_heads, num_kv_heads: num_heads, head_dim, seq_len: 1 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: hidden_dim, quantized: false });
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
            ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-5 });
            ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: intermediate_size, quantized: false });
            ops.push(TensorOp::Activation { kind: ActivationKind::GELU, elements: intermediate_size });
            ops.push(TensorOp::Matmul { m: 1, k: intermediate_size, n: hidden_dim, quantized: false });
            ops.push(TensorOp::ResidualAdd { elements: hidden_dim });
        }

        ops.push(TensorOp::LayerNorm { hidden_size: hidden_dim, eps: 1e-5 });
        ops.push(TensorOp::Matmul { m: 1, k: hidden_dim, n: vocab_size, quantized: false });
        ops.push(TensorOp::Softmax { seq_len: 1, dim: vocab_size });
        ops
    }

    // ─── Utilitários de Extração de Metadados ────────────────────────────────

    fn count_layers_by_prefix(names: &[&str], prefix: &str) -> u32 {
        let mut max_layer = 0u32;
        for name in names {
            if name.starts_with(prefix) {
                // Extrai o número após o prefix
                let rest = &name[prefix.len()..];
                if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
                    if let Ok(n) = rest[..end].parse::<u32>() {
                        max_layer = max_layer.max(n + 1);
                    }
                }
            }
        }
        max_layer.max(1)
    }

    fn extract_llama_dims(metadata: &ModelMetadata) -> (u32, u32, u32, u32) {
        // Tenta extrair dos metadados GGUF via `extra` JSON.
        // Cobre: llama (Llama/Mistral/Phi), qwen2 (Qwen2.5), gemma, phi, falcon, gpt2, etc.
        let hidden = Self::try_u32_from_extra(&metadata.extra, &[
            "llama.embedding_length", "qwen2.embedding_length", "gemma.embedding_length",
            "phi2.embedding_length", "phi3.embedding_length", "falcon.embedding_length",
            "gpt2.embedding_length", "model.hidden_size", "hidden_size",
        ]).unwrap_or(4096);

        let heads = Self::try_u32_from_extra(&metadata.extra, &[
            "llama.attention.head_count", "qwen2.attention.head_count",
            "gemma.attention.head_count", "phi2.attention.head_count",
            "phi3.attention.head_count", "falcon.attention.head_count",
            "gpt2.attention.head_count", "model.num_attention_heads", "num_attention_heads",
        ]).unwrap_or(32);

        let kv_heads = Self::try_u32_from_extra(&metadata.extra, &[
            "llama.attention.head_count_kv", "qwen2.attention.head_count_kv",
            "gemma.attention.head_count_kv", "phi2.attention.head_count_kv",
            "phi3.attention.head_count_kv", "falcon.attention.head_count_kv",
            "gpt2.attention.head_count_kv", "model.num_key_value_heads", "num_key_value_heads",
        ]).unwrap_or(heads);

        let intermediate = Self::try_u32_from_extra(&metadata.extra, &[
            "llama.feed_forward_length", "qwen2.feed_forward_length",
            "gemma.feed_forward_length", "phi2.feed_forward_length",
            "phi3.feed_forward_length", "falcon.feed_forward_length",
            "gpt2.feed_forward_length", "model.intermediate_size", "intermediate_size",
        ]).unwrap_or(11008);

        (hidden, heads, kv_heads, intermediate)
    }

    fn extract_bert_dims(metadata: &ModelMetadata) -> (u32, u32, u32) {
        let hidden = Self::try_u32_from_extra(&metadata.extra, &[
            "bert.embedding_length", "hidden_size", "model.hidden_size"
        ]).unwrap_or(768);

        let heads = Self::try_u32_from_extra(&metadata.extra, &[
            "bert.attention.head_count", "num_attention_heads", "model.num_attention_heads"
        ]).unwrap_or(12);

        let intermediate = Self::try_u32_from_extra(&metadata.extra, &[
            "bert.feed_forward_length", "intermediate_size", "model.intermediate_size"
        ]).unwrap_or(3072);

        (hidden, heads, intermediate)
    }

    fn extract_vocab_size(metadata: &ModelMetadata) -> u32 {
        Self::try_u32_from_extra(&metadata.extra, &[
            "llama.vocab_size", "qwen2.vocab_size", "gemma.vocab_size",
            "phi2.vocab_size", "phi3.vocab_size", "falcon.vocab_size",
            "gpt2.vocab_size", "bert.vocab_size", "vocab_size", "tokenizer.ggml.tokens",
        ]).unwrap_or(32000)
    }

    fn try_u32_from_extra(extra: &serde_json::Value, keys: &[&str]) -> Option<u32> {
        for key in keys {
            if let Some(v) = extra.get(key) {
                if let Some(n) = v.as_u64() {
                    return Some(n as u32);
                }
                if let Some(s) = v.as_str() {
                    if let Ok(n) = s.parse::<u32>() {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    fn try_f32_from_extra(extra: &serde_json::Value, keys: &[&str]) -> Option<f32> {
        for key in keys {
            if let Some(v) = extra.get(key) {
                if let Some(n) = v.as_f64() {
                    return Some(n as f32);
                }
                if let Some(s) = v.as_str() {
                    if let Ok(n) = s.parse::<f32>() {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    fn infer_model_type(arch: &ModelArchitecture, _metadata: &ModelMetadata) -> ModelType {
        match arch {
            ModelArchitecture::BertEmbedding { .. } => ModelType::Embedding,
            ModelArchitecture::Llama { .. } | ModelArchitecture::Gpt2 { .. } => ModelType::Generative,
            ModelArchitecture::Bert { .. } => ModelType::Classifier,
            ModelArchitecture::Unknown { .. } => ModelType::Generative,
        }
    }

    fn infer_embedding_dim(arch: &ModelArchitecture) -> Option<u32> {
        match arch {
            ModelArchitecture::BertEmbedding { output_dim, .. } => Some(*output_dim),
            _ => None,
        }
    }

    fn map_tensors(arch: &ModelArchitecture, metadata: &ModelMetadata) -> HashMap<String, String> {
        let mut binding = HashMap::new();
        // Mapeamento genérico: nome do tensor GGUF → chave no BufferPool
        for tensor in &metadata.tensors {
            let key = match arch {
                ModelArchitecture::Llama { .. } => {
                    // blk.0.attn_q.weight → layer_0_q_weight
                    tensor.name.replace('.', "_")
                }
                ModelArchitecture::Bert { .. } | ModelArchitecture::BertEmbedding { .. } => {
                    tensor.name.replace('.', "_")
                }
                _ => tensor.name.replace('.', "_"),
            };
            binding.insert(tensor.name.clone(), key);
        }
        binding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::{ModelMetadata, ModelFormat, TensorInfo, TensorDtype};

    fn make_tensor(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape: vec![128, 128],
            dtype: TensorDtype::F32,
            data_offset: 0,
            data_size: 65536,
        }
    }

    fn make_llama_metadata() -> ModelMetadata {
        ModelMetadata {
            format: ModelFormat::Gguf,
            model_name: Some("llama3-8b".to_string()),
            architecture: Some("llama".to_string()),
            param_count: Some(8_000_000_000),
            tensors: vec![
                make_tensor("blk.0.attn_q.weight"),
                make_tensor("blk.0.attn_k.weight"),
                make_tensor("blk.0.attn_v.weight"),
                make_tensor("blk.0.ffn_gate.weight"),
                make_tensor("blk.1.attn_q.weight"),
                make_tensor("blk.1.ffn_gate.weight"),
                make_tensor("output.weight"),
            ],
            data_offset: 0,
            file_size: 1024 * 1024,
            extra: serde_json::json!({
                "llama.embedding_length": 4096,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.feed_forward_length": 14336,
                "llama.vocab_size": 128256
            }),
        }
    }

    fn make_bge_metadata() -> ModelMetadata {
        ModelMetadata {
            format: ModelFormat::Gguf,
            model_name: Some("bge-small-en-v1.5".to_string()),
            architecture: Some("bert".to_string()),
            param_count: Some(33_000_000),
            tensors: vec![
                make_tensor("encoder.layer.0.attention.self.query.weight"),
                make_tensor("encoder.layer.0.attention.self.key.weight"),
                make_tensor("encoder.layer.0.attention.self.value.weight"),
                make_tensor("encoder.layer.0.intermediate.dense.weight"),
                make_tensor("encoder.layer.1.attention.self.query.weight"),
                make_tensor("encoder.layer.1.intermediate.dense.weight"),
            ],
            data_offset: 0,
            file_size: 33 * 1024 * 1024,
            extra: serde_json::json!({
                "hidden_size": 384,
                "num_attention_heads": 12,
                "intermediate_size": 1536
            }),
        }
    }

    fn make_bert_metadata() -> ModelMetadata {
        let mut m = make_bge_metadata();
        m.tensors.push(make_tensor("cls.predictions.bias"));
        m
    }

    #[test]
    fn test_detect_llama_architecture() {
        let meta = make_llama_metadata();
        let arch = GraphInterpreter::detect_architecture(&meta).unwrap();
        assert!(matches!(arch, ModelArchitecture::Llama { .. }),
            "Deve detectar Llama, got {:?}", arch.name());

        if let ModelArchitecture::Llama { num_layers, hidden_dim, .. } = arch {
            assert_eq!(num_layers, 2, "Deve detectar 2 camadas (blk.0 e blk.1)");
            assert_eq!(hidden_dim, 4096, "hidden_dim deve ser 4096");
        }
    }

    #[test]
    fn test_detect_qwen2_architecture_dims() {
        // Qwen2.5-0.5B usa chaves qwen2.* no GGUF — devem ser lidas corretamente
        let meta = ModelMetadata {
            format: ModelFormat::Gguf,
            model_name: Some("qwen2.5-0.5b".to_string()),
            architecture: Some("qwen2".to_string()),
            param_count: Some(500_000_000),
            tensors: vec![
                make_tensor("blk.0.attn_q.weight"),
                make_tensor("blk.0.attn_k.weight"),
                make_tensor("blk.0.attn_v.weight"),
                make_tensor("blk.0.ffn_gate.weight"),
                make_tensor("blk.1.attn_q.weight"),
                make_tensor("blk.1.ffn_gate.weight"),
                make_tensor("token_embd.weight"),
            ],
            data_offset: 0,
            file_size: 500 * 1024 * 1024,
            extra: serde_json::json!({
                "qwen2.embedding_length": 896,
                "qwen2.attention.head_count": 14,
                "qwen2.attention.head_count_kv": 2,
                "qwen2.feed_forward_length": 4864,
                "qwen2.vocab_size": 151936,
                "qwen2.rope.freq_base": 1000000.0
            }),
        };
        let arch = GraphInterpreter::detect_architecture(&meta).unwrap();
        assert!(matches!(arch, ModelArchitecture::Llama { .. }), "Qwen2 deve detectar como Llama-family");
        if let ModelArchitecture::Llama { hidden_dim, num_heads, num_kv_heads, intermediate_size, vocab_size, rope_base, .. } = arch {
            assert_eq!(hidden_dim, 896, "hidden_dim deve ser 896 (não 4096)");
            assert_eq!(num_heads, 14, "num_heads deve ser 14");
            assert_eq!(num_kv_heads, 2, "num_kv_heads deve ser 2");
            assert_eq!(intermediate_size, 4864, "intermediate deve ser 4864");
            assert_eq!(vocab_size, 151936, "vocab deve ser 151936");
            assert!((rope_base - 1_000_000.0f32).abs() < 1.0, "rope_base deve ser 1M, got {}", rope_base);
        }
    }

    #[test]
    fn test_detect_bge_architecture() {
        let meta = make_bge_metadata();
        let arch = GraphInterpreter::detect_architecture(&meta).unwrap();
        assert!(matches!(arch, ModelArchitecture::BertEmbedding { .. }),
            "BGE (sem cls.predictions) deve ser BertEmbedding, got {}", arch.name());
        assert!(arch.is_embedding_model(), "BGE deve ser reconhecido como modelo de embedding");
    }

    #[test]
    fn test_detect_bert_vs_bge() {
        // Com cls.predictions → BERT classificador
        let bert_meta = make_bert_metadata();
        let bert_arch = GraphInterpreter::detect_architecture(&bert_meta).unwrap();
        assert!(matches!(bert_arch, ModelArchitecture::Bert { .. }),
            "BERT com cls.predictions deve ser Bert, got {}", bert_arch.name());
        assert!(!bert_arch.is_embedding_model(), "BERT classificador não é embedding model");
    }

    #[test]
    fn test_llama_plan_has_rope_and_silu() {
        let meta = make_llama_metadata();
        let arch = GraphInterpreter::detect_architecture(&meta).unwrap();
        let ops = GraphInterpreter::build_execution_plan(&arch).unwrap();

        assert!(ops.iter().any(|op| matches!(op, TensorOp::RoPE { .. })),
            "Llama deve ter RoPE");
        assert!(ops.iter().any(|op| matches!(op, TensorOp::Activation { kind: ActivationKind::SiLU, .. })),
            "Llama deve ter SiLU");
        assert!(!ops.iter().any(|op| matches!(op, TensorOp::MeanPool { .. })),
            "Llama NÃO deve ter MeanPool");
    }

    #[test]
    fn test_bge_plan_has_mean_pool_and_gelu() {
        let meta = make_bge_metadata();
        let arch = GraphInterpreter::detect_architecture(&meta).unwrap();
        let ops = GraphInterpreter::build_execution_plan(&arch).unwrap();

        assert!(ops.iter().any(|op| matches!(op, TensorOp::MeanPool { .. })),
            "BGE deve terminar com MeanPool");
        assert!(ops.iter().any(|op| matches!(op, TensorOp::Activation { kind: ActivationKind::GELU, .. })),
            "BGE deve usar GELU (não SiLU)");
        assert!(!ops.iter().any(|op| matches!(op, TensorOp::RoPE { .. })),
            "BGE NÃO deve ter RoPE");
        assert!(ops.iter().any(|op| matches!(op, TensorOp::LayerNorm { .. })),
            "BGE deve ter LayerNorm (não RmsNorm)");
    }

    #[test]
    fn test_interpret_returns_correct_model_type() {
        let bge = make_bge_metadata();
        let graph = GraphInterpreter::interpret(&bge).unwrap();
        assert_eq!(graph.model_type, ModelType::Embedding);
        assert!(graph.embedding_dim.is_some());

        let llama = make_llama_metadata();
        let graph = GraphInterpreter::interpret(&llama).unwrap();
        assert_eq!(graph.model_type, ModelType::Generative);
        assert!(graph.embedding_dim.is_none());
    }

    #[test]
    fn test_execution_graph_metadata() {
        let meta = make_llama_metadata();
        let graph = GraphInterpreter::interpret(&meta).unwrap();
        assert_eq!(graph.model_name, "llama3-8b");
        assert_eq!(graph.num_layers(), 2);
        assert_eq!(graph.hidden_dim(), 4096);
        assert!(graph.op_count() > 0, "Grafo deve ter operadores");
    }

    #[test]
    fn test_tensor_binding_populated() {
        let meta = make_llama_metadata();
        let graph = GraphInterpreter::interpret(&meta).unwrap();
        // Cada tensor do metadata deve ter uma binding
        assert_eq!(
            graph.tensor_binding.len(),
            meta.tensors.len(),
            "Cada tensor deve ter uma binding no BufferPool"
        );
    }
}
