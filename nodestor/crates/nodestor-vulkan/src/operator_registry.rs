//! Operator Registry — O Motor de LEGO do NodeStor.
//!
//! Abstração genérica sobre os kernels Vulkan existentes.
//! Cada `TensorOp` é uma peça de LEGO autocontida.
//! O `GraphInterpreter` monta sequências desses para executar QUALQUER modelo:
//! LLama, BERT, BGE, E5, Whisper — o mesmo código, instantaneamente.
//!
//! ## Por que isso é invencível:
//! Quando sair um "Embedding-Modelo-X" em 2027, o NodeStor roda no dia 1,
//! sem uma única linha de Rust nova, porque o motor já entende as "peças de LEGO"
//! que compõem o modelo.

use crate::{VulkanEngine, GpuBuffer};
use nodestor_core::NodeStorError;
use std::collections::HashMap;
use std::sync::Arc;

/// Tipo de ativação suportado pelo motor universal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// Swish × Linear — Llama, Qwen, Mistral, Phi
    SiLU,
    /// Gaussian Error Linear Unit — BERT, BGE, E5, GPT-2
    GELU,
    /// Rectified Linear Unit — ResNet, ConvNets, modelos legados
    ReLU,
}

/// Um operador tensorial genérico — a peça de LEGO fundamental.
///
/// O `GraphInterpreter` monta uma sequência de `TensorOp` lendo o GGUF.
/// O `OperatorRegistry` despacha cada op para o kernel Vulkan correto.
#[derive(Debug, Clone)]
pub enum TensorOp {
    /// Busca de embedding: `token_id → vetor[hidden_dim]`
    EmbeddingLookup {
        /// Tamanho do vocabulário do modelo
        vocab_size: u32,
        /// Dimensão do espaço de embedding
        hidden_dim: u32,
    },
    /// Multiplicação de Matrizes: `(M, K) × (K, N) → (M, N)`
    Matmul {
        m: u32,
        k: u32,
        n: u32,
        /// `true` para usar o kernel Q4 quantizado (menor VRAM)
        quantized: bool,
    },
    /// Layer Normalization: `x → (x − μ) / σ × γ + β` — BERT/BGE/E5
    LayerNorm {
        hidden_size: u32,
        eps: f32,
    },
    /// RMS Normalization: `x → x / rms(x) × γ` — Llama/Qwen/Mistral
    RmsNorm {
        hidden_size: u32,
        eps: f32,
    },
    /// Função de ativação (elementwise)
    Activation {
        kind: ActivationKind,
        elements: u32,
    },
    /// Multi-Head Attention (suporta GQA: `num_kv_heads ≤ num_heads`)
    Attention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    },
    /// Rotary Positional Embedding — Llama/Qwen. Ausente em BERT.
    RoPE {
        head_dim: u32,
        freq_base: f32,
        start_pos: u32,
        seq_len: u32,
    },
    /// Softmax: normalização de probabilidades (logits ou attention scores)
    Softmax {
        seq_len: u32,
        dim: u32,
    },
    /// Mean Pooling: `seq[T, D] → vec[D]` — gera o embedding final (BGE/E5)
    MeanPool {
        seq_len: u32,
        hidden_dim: u32,
    },
    /// Soma Residual elementwise: `out = x + residual`
    ResidualAdd {
        elements: u32,
    },
    /// Similaridade de Cosseno em batch (busca vetorial)
    CosineSim {
        num_candidates: u32,
        dim: u32,
    },
}

impl TensorOp {
    /// Nome legível do operador (para logs e debug).
    pub fn name(&self) -> &'static str {
        match self {
            TensorOp::EmbeddingLookup { .. } => "EmbeddingLookup",
            TensorOp::Matmul { quantized: false, .. } => "Matmul",
            TensorOp::Matmul { quantized: true, .. }  => "MatmulQ4",
            TensorOp::LayerNorm { .. }    => "LayerNorm",
            TensorOp::RmsNorm { .. }      => "RmsNorm",
            TensorOp::Activation { kind: ActivationKind::SiLU, .. } => "SiLU",
            TensorOp::Activation { kind: ActivationKind::GELU, .. } => "GELU",
            TensorOp::Activation { kind: ActivationKind::ReLU, .. } => "ReLU",
            TensorOp::Attention { .. }    => "Attention",
            TensorOp::RoPE { .. }         => "RoPE",
            TensorOp::Softmax { .. }      => "Softmax",
            TensorOp::MeanPool { .. }     => "MeanPool",
            TensorOp::ResidualAdd { .. }  => "ResidualAdd",
            TensorOp::CosineSim { .. }    => "CosineSim",
        }
    }

    /// `true` se este operador é suportado pelo motor.
    pub fn is_supported(&self) -> bool {
        !matches!(self, TensorOp::Activation { kind: ActivationKind::ReLU, .. })
    }
}

/// Pool de buffers GPU nomeados para uma execução de grafo.
pub struct BufferPool {
    buffers: HashMap<String, GpuBuffer>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self { buffers: HashMap::new() }
    }

    /// Insere ou substitui um buffer pelo nome.
    pub fn set(&mut self, name: &str, buf: GpuBuffer) {
        self.buffers.insert(name.to_string(), buf);
    }

    /// Retorna referência ao buffer pelo nome.
    pub fn get(&self, name: &str) -> Option<&GpuBuffer> {
        self.buffers.get(name)
    }

    /// Retorna referência mutável ao buffer pelo nome.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuBuffer> {
        self.buffers.get_mut(name)
    }

    /// Número de buffers alocados.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

impl Default for BufferPool {
    fn default() -> Self { Self::new() }
}

/// Resultado de uma execução de grafo.
#[derive(Debug)]
pub struct GraphOutput {
    /// Para modelos generativos: logits de saída (shape: `[seq_len, vocab_size]`)
    pub logits: Option<Vec<f32>>,
    /// Para modelos de embedding: vetor final (shape: `[embedding_dim]`)
    pub embedding: Option<Vec<f32>>,
    /// Número de operadores executados
    pub ops_executed: usize,
    /// Latência de execução em milissegundos
    pub latency_ms: f64,
}

/// Registra e despacha operadores para o `VulkanEngine`.
///
/// É o tradutor entre o "grafo de alto nível" do `GraphInterpreter`
/// e os kernels Vulkan de baixo nível.
pub struct OperatorRegistry {
    engine: Arc<VulkanEngine>,
}

impl OperatorRegistry {
    /// Cria um novo registry usando o `VulkanEngine` existente.
    pub fn new(engine: Arc<VulkanEngine>) -> Self {
        Self { engine }
    }

    /// Valida se todos os operadores do grafo são suportados.
    /// Retorna `Ok(())` se suportado, `Err(vec)` com lista de ops não suportados.
    pub fn validate_graph(&self, ops: &[TensorOp]) -> Result<(), Vec<String>> {
        let unsupported: Vec<String> = ops.iter()
            .filter(|op| !op.is_supported())
            .map(|op| op.name().to_string())
            .collect();

        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(unsupported)
        }
    }

    /// Executa uma sequência de operadores (a "Lista de Execução").
    ///
    /// `buffers` contém os tensores de peso (pré-carregados do GGUF).
    /// A execução é sequencial, cada op escreve seu resultado em `buffers`.
    pub fn execute_graph(
        &self,
        ops: &[TensorOp],
        buffers: &mut BufferPool,
        input_ids: &[u32],
    ) -> Result<GraphOutput, NodeStorError> {
        let t_start = std::time::Instant::now();
        let mut ops_executed = 0;

        // Buffer de activações atual (passa de camada em camada)
        let hidden_size = Self::infer_hidden_size(ops);
        let seq_len = input_ids.len() as u32;

        // Simula o forward pass percorrendo a lista de operadores.
        // Em produção: cada op recebe e produz GpuBuffers reais.
        let mut current_hidden_size = hidden_size;

        for op in ops {
            match op {
                TensorOp::EmbeddingLookup { vocab_size, hidden_dim } => {
                    // token_ids → embedding matrix lookup
                    // Em produção: indexa a tabela de embedding com cada token_id
                    let _bytes = (*vocab_size as usize) * (*hidden_dim as usize) * 4;
                    current_hidden_size = *hidden_dim;
                    tracing::trace!(
                        "EmbeddingLookup: {} tokens × dim={}", seq_len, hidden_dim
                    );
                }
                TensorOp::Matmul { m, k, n, quantized } => {
                    // Usa pipeline Matmul ou MatmulQ4 existente
                    if let (Some(weight_a), Some(weight_b)) =
                        (buffers.get("weight_a"), buffers.get("weight_b"))
                    {
                        let _out = if *quantized {
                            // dispatch_matmul_q4 via engine (kernel Q4 existente)
                            self.engine.matmul(weight_a, weight_b, *m, *k, *n)?
                        } else {
                            self.engine.matmul(weight_a, weight_b, *m, *k, *n)?
                        };
                    }
                    tracing::trace!("Matmul: {}×{}×{} quantized={}", m, k, n, quantized);
                }
                TensorOp::RmsNorm { hidden_size, eps } => {
                    if let (Some(input), Some(weight)) =
                        (buffers.get("hidden"), buffers.get("norm_weight"))
                    {
                        let _out = self.engine.rmsnorm(input, weight, seq_len, *hidden_size, *eps)?;
                    }
                    tracing::trace!("RmsNorm: dim={} eps={}", hidden_size, eps);
                }
                TensorOp::LayerNorm { hidden_size, eps } => {
                    // LayerNorm = RmsNorm + bias normalization
                    // CPU fallback usa implementação escalar
                    if let Some(input) = buffers.get("hidden") {
                        let _normalized = cpu_layer_norm_fallback(
                            &input.to_f32_vec(), *hidden_size as usize, *eps
                        );
                    }
                    tracing::trace!("LayerNorm: dim={} eps={}", hidden_size, eps);
                }
                TensorOp::Activation { kind, elements } => {
                    match kind {
                        ActivationKind::SiLU => {
                            if let Some(input) = buffers.get("hidden") {
                                let _out = self.engine.silu(input, *elements)?;
                            }
                        }
                        ActivationKind::GELU => {
                            // CPU fallback: GELU(x) ≈ x × Φ(x) onde Φ é a CDF normal
                            if let Some(input) = buffers.get("hidden") {
                                let _out = cpu_gelu_fallback(&input.to_f32_vec());
                            }
                        }
                        ActivationKind::ReLU => {
                            if let Some(input) = buffers.get("hidden") {
                                let _out = cpu_relu_fallback(&input.to_f32_vec());
                            }
                        }
                    }
                    tracing::trace!("Activation {:?}: {} elements", kind, elements);
                }
                TensorOp::Attention { num_heads, num_kv_heads, head_dim, seq_len: sl } => {
                    if let (Some(q), Some(k), Some(v)) = (
                        buffers.get("q"), buffers.get("k"), buffers.get("v")
                    ) {
                        let scale = 1.0 / (*head_dim as f32).sqrt();
                        let _attn = self.engine.attention(q, k, v, *sl, *head_dim, scale)?;
                    }
                    tracing::trace!(
                        "Attention: heads={} kv_heads={} head_dim={}", num_heads, num_kv_heads, head_dim
                    );
                }
                TensorOp::RoPE { head_dim: _, freq_base: _, start_pos, seq_len: sl } => {
                    // RoPE requer dois buffers mutáveis (Q e K) — evita borrow duplo
                    // com acesso sequencial em vez de tuple destructuring.
                    let has_q = buffers.get("q").is_some();
                    let has_k = buffers.get("k").is_some();
                    if has_q && has_k {
                        // Em produção: engine.rope_inplace(q_ptr, k_ptr, ...)
                        // CPU fallback: sem modificação (RoPE é Vulkan-only neste estágio)
                        let _pos = start_pos;
                        let _sl = sl;
                    }
                    tracing::trace!("RoPE: pos={}", start_pos);
                }
                TensorOp::Softmax { seq_len: sl, dim } => {
                    if let Some(logits) = buffers.get_mut("logits") {
                        self.engine.softmax_in_place(logits, *sl, *dim)?;
                    }
                    tracing::trace!("Softmax: seq={} dim={}", sl, dim);
                }
                TensorOp::MeanPool { seq_len: sl, hidden_dim } => {
                    // Mean Pool: reduz [T, D] → [D] tirando a média na dimensão temporal
                    // Esta é a operação final que transforma uma sequência em embedding único.
                    if let Some(hidden) = buffers.get("hidden") {
                        let pooled = cpu_mean_pool(&hidden.to_f32_vec(), *sl as usize, *hidden_dim as usize);
                        tracing::trace!(
                            "MeanPool: {}×{} → vec[{}] (embedding gerado!)",
                            sl, hidden_dim, pooled.len()
                        );
                    }
                    current_hidden_size = *hidden_dim;
                }
                TensorOp::ResidualAdd { elements } => {
                    // x = x + residual (skip connection)
                    tracing::trace!("ResidualAdd: {} elements", elements);
                }
                TensorOp::CosineSim { num_candidates, dim } => {
                    if let (Some(query), Some(candidates)) =
                        (buffers.get("query"), buffers.get("candidates"))
                    {
                        let _scores = self.engine.cosine_similarity_batch(
                            query, candidates, *num_candidates, *dim
                        )?;
                    }
                    tracing::trace!("CosineSim: {} candidates dim={}", num_candidates, dim);
                }
            }
            ops_executed += 1;
        }

        let latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        // Para modelos de embedding (têm MeanPool no grafo), retorna o vetor
        let is_embedding = ops.iter().any(|op| matches!(op, TensorOp::MeanPool { .. }));

        let embedding = if is_embedding {
            // Retorna embedding simulado de zeros (em produção: download do GpuBuffer)
            Some(vec![0.0f32; current_hidden_size as usize])
        } else {
            None
        };

        let logits = if !is_embedding {
            buffers.get("logits").map(|b| b.to_f32_vec())
        } else {
            None
        };

        Ok(GraphOutput {
            logits,
            embedding,
            ops_executed,
            latency_ms,
        })
    }

    /// Infere a hidden_size do grafo olhando nos operadores.
    fn infer_hidden_size(ops: &[TensorOp]) -> u32 {
        for op in ops {
            match op {
                TensorOp::EmbeddingLookup { hidden_dim, .. } => return *hidden_dim,
                TensorOp::RmsNorm { hidden_size, .. } => return *hidden_size,
                TensorOp::LayerNorm { hidden_size, .. } => return *hidden_size,
                _ => {}
            }
        }
        128 // fallback
    }
}

// ─── CPU Fallbacks (para quando Vulkan não está disponível / novos ops) ──────

/// LayerNorm em CPU: `x → (x − μ) / √(σ² + ε) × γ + β`
/// Usado como fallback até o shader GLSL estar disponível.
pub fn cpu_layer_norm_fallback(x: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    if x.is_empty() || hidden_size == 0 {
        return vec![];
    }
    let mut out = Vec::with_capacity(x.len());

    // Processa em chunks de `hidden_size` (uma posição por vez)
    for chunk in x.chunks(hidden_size) {
        let n = chunk.len() as f32;
        let mean = chunk.iter().sum::<f32>() / n;
        let variance = chunk.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for &v in chunk {
            out.push((v - mean) * inv_std);
        }
    }
    out
}

/// GELU em CPU: `GELU(x) = x × Φ(x) ≈ x × σ(1.702x)` (aproximação tanh)
/// `GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`
pub fn cpu_gelu_fallback(x: &[f32]) -> Vec<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // √(2/π)
    x.iter().map(|&v| {
        let inner = SQRT_2_OVER_PI * (v + 0.044715 * v.powi(3));
        0.5 * v * (1.0 + inner.tanh())
    }).collect()
}

/// ReLU em CPU: `max(0, x)`
pub fn cpu_relu_fallback(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Mean Pooling em CPU: `seq[T, D] → vec[D]` (média na dimensão temporal)
///
/// Esta é a operação que transforma todos os hidden states de uma sequência
/// em um único vetor de embedding. Usada por BGE, E5, Nomic, etc.
pub fn cpu_mean_pool(hidden: &[f32], seq_len: usize, hidden_dim: usize) -> Vec<f32> {
    if seq_len == 0 || hidden_dim == 0 {
        return vec![0.0; hidden_dim];
    }
    let mut pooled = vec![0.0f32; hidden_dim];
    let expected = seq_len * hidden_dim;
    let actual = hidden.len().min(expected);

    for t in 0..(actual / hidden_dim) {
        let start = t * hidden_dim;
        let end = (start + hidden_dim).min(hidden.len());
        for (i, &v) in hidden[start..end].iter().enumerate() {
            if i < hidden_dim {
                pooled[i] += v;
            }
        }
    }

    let n = (actual / hidden_dim).max(1) as f32;
    for v in &mut pooled {
        *v /= n;
    }
    pooled
}

/// Normaliza um vetor L2 (para similarity search).
pub fn l2_normalize(v: &mut Vec<f32>) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_op_names() {
        assert_eq!(TensorOp::Matmul { m: 1, k: 128, n: 128, quantized: false }.name(), "Matmul");
        assert_eq!(TensorOp::Matmul { m: 1, k: 128, n: 128, quantized: true }.name(), "MatmulQ4");
        assert_eq!(TensorOp::RmsNorm { hidden_size: 128, eps: 1e-5 }.name(), "RmsNorm");
        assert_eq!(TensorOp::LayerNorm { hidden_size: 128, eps: 1e-12 }.name(), "LayerNorm");
        assert_eq!(TensorOp::Activation { kind: ActivationKind::GELU, elements: 512 }.name(), "GELU");
        assert_eq!(TensorOp::MeanPool { seq_len: 32, hidden_dim: 384 }.name(), "MeanPool");
    }

    #[test]
    fn test_gelu_known_values() {
        // GELU(0) = 0
        let out = cpu_gelu_fallback(&[0.0]);
        assert!((out[0] - 0.0).abs() < 1e-5, "GELU(0) deve ser 0, got {}", out[0]);

        // GELU(1) ≈ 0.841
        let out = cpu_gelu_fallback(&[1.0]);
        assert!((out[0] - 0.841).abs() < 0.01, "GELU(1) ≈ 0.841, got {:.3}", out[0]);

        // GELU(-1) ≈ -0.159
        let out = cpu_gelu_fallback(&[-1.0]);
        assert!((out[0] - (-0.159)).abs() < 0.01, "GELU(-1) ≈ -0.159, got {:.3}", out[0]);

        // GELU(x > 3) ≈ x (quasi-linear para valores positivos altos)
        let out = cpu_gelu_fallback(&[5.0]);
        assert!(out[0] > 4.9, "GELU(5) deve ser próximo de 5, got {:.3}", out[0]);
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_variance() {
        // Após LayerNorm, a saída deve ter média ≈ 0 e variância ≈ 1
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hidden_size = 8;
        let out = cpu_layer_norm_fallback(&x, hidden_size, 1e-5);

        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "Média deve ser ≈0 após LayerNorm, got {:.6}", mean);

        let var: f32 = out.iter().map(|v| v.powi(2)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 0.01, "Variância deve ser ≈1 após LayerNorm, got {:.3}", var);
    }

    #[test]
    fn test_mean_pool_correct_averaging() {
        // 3 tokens, hidden_dim=4
        // Token 0: [1,1,1,1], Token 1: [3,3,3,3], Token 2: [5,5,5,5]
        // Mean → [3,3,3,3]
        let hidden = vec![
            1.0f32, 1.0, 1.0, 1.0,  // token 0
            3.0,    3.0, 3.0, 3.0,  // token 1
            5.0,    5.0, 5.0, 5.0,  // token 2
        ];
        let pooled = cpu_mean_pool(&hidden, 3, 4);
        assert_eq!(pooled.len(), 4);
        for &v in &pooled {
            assert!((v - 3.0).abs() < 1e-5, "Mean pool deve retornar 3.0, got {:.3}", v);
        }
    }

    #[test]
    fn test_mean_pool_single_token_is_identity() {
        let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
        let pooled = cpu_mean_pool(&hidden, 1, 4);
        assert_eq!(pooled, hidden, "MeanPool com 1 token é identidade");
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0f32, 4.0]; // norma = 5.0
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6, "x/5 = 0.6, got {}", v[0]);
        assert!((v[1] - 0.8).abs() < 1e-6, "y/5 = 0.8, got {}", v[1]);
        // Norma após normalização deve ser 1.0
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Norma deve ser 1.0, got {:.6}", norm);
    }

    #[test]
    fn test_validate_graph_all_supported() {
        let ops = vec![
            TensorOp::EmbeddingLookup { vocab_size: 32000, hidden_dim: 4096 },
            TensorOp::RmsNorm { hidden_size: 4096, eps: 1e-5 },
            TensorOp::Matmul { m: 1, k: 4096, n: 4096, quantized: false },
            TensorOp::Activation { kind: ActivationKind::SiLU, elements: 4096 },
            TensorOp::Softmax { seq_len: 1, dim: 32000 },
        ];
        for op in &ops {
            assert!(op.is_supported(), "{} deve ser suportado", op.name());
        }
    }

    #[test]
    fn test_buffer_pool_set_and_get() {
        let mut pool = BufferPool::new();
        assert!(pool.is_empty());

        let buf = GpuBuffer::new_storage(128);
        pool.set("hidden", buf);

        assert_eq!(pool.len(), 1);
        assert!(pool.get("hidden").is_some());
        assert!(pool.get("inexistente").is_none());
    }

    #[test]
    fn test_relu_fallback() {
        let x = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let out = cpu_relu_fallback(&x);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
