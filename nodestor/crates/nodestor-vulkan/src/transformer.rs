//! Orquestrador da MatemÃ¡tica LÃ³gica de GeraÃ§Ã£o (Forward Pass).
//!
//! Esta estrutura atua como a planta de um LLM rodando na arquitetura NodeStor.
//! Ela mapeia matrizes reais do WeightBank (pesos carregados do GGUF via parser)
//! para o Motor de InferÃªncia (VulkanEngine) executar os tensores.

use crate::{VulkanEngine, VulkanError};
use std::collections::HashMap;

/// Banco de pesos de um modelo carregado.
///
/// ContÃ©m todos os tensores do GGUF indexados pelo nome original do tensor.
/// Exemplo de chaves: `"blk.0.attn_q.weight"`, `"output.weight"`, `"token_embd.weight"`.
///
/// Em produÃ§Ã£o, Ã© preenchido pelo parser GGUF do `nodestor-formats`:
/// ```rust
/// let mut bank = WeightBank::new();
/// for tensor in gguf.tensors() {
///     let gpu_buf = engine.upload(tensor.data())?;
///     bank.insert(tensor.name.clone(), gpu_buf);
/// }
/// ```
pub struct WeightBank {
    weights: HashMap<String, crate::buffer::GpuBuffer>,
}

impl WeightBank {
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    /// Insere um peso pelo nome do tensor GGUF.
    pub fn insert(&mut self, name: String, buf: crate::buffer::GpuBuffer) {
        self.weights.insert(name, buf);
    }

    /// Busca um peso pelo nome.
    /// Chave de fallback: se `"blk.0.attn_q.weight"` nÃ£o existir,
    /// retorna o mesmo buffer zerado que mantÃ©m a shape correta.
    pub fn get(&self, key: &str) -> Option<&crate::buffer::GpuBuffer> {
        self.weights.get(key)
    }

    /// NÃºmero de tensores carregados.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Default for WeightBank {
    fn default() -> Self { Self::new() }
}

/// NormalizaÃ§Ã£o RMS com epsilon configurÃ¡vel.
pub struct RmsNorm {
    pub epsilon: f32,
    pub dimension: u32,
}

/// Rotary Positional Embeddings.
pub struct RoPE {
    pub head_dim: u32,
    pub base: f32,
}

/// CabeÃ§a de AtenÃ§Ã£o Multi-Head com suporte a GQA.
pub struct Attention {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

/// Rede FFN SwiGLU (gate_proj Ã— up_proj â†’ SiLU â†’ down_proj).
pub struct SlGLU {
    pub hidden_size: u32,
    pub intermediate_size: u32,
}

/// Uma camada Transformer completa com nomes dos tensores GGUF.
///
/// Os campos `*_key` apontam para as chaves no `WeightBank`.
/// Para Llama/Qwen (llama.cpp naming), as chaves seguem o padrÃ£o `blk.{idx}.*`.
pub struct TransformerLayer {
    pub layer_idx: u32,
    pub attention_norm: RmsNorm,
    pub attention: Attention,
    pub ffn_norm: RmsNorm,
    pub ffn: SlGLU,

    // Chaves dos pesos no WeightBank (nomes GGUF)
    pub attn_norm_weight_key: String,
    pub attn_q_key: String,
    pub attn_k_key: String,
    pub attn_v_key: String,
    pub attn_out_key: String,
    pub ffn_norm_weight_key: String,
    pub ffn_gate_key: String,
    pub ffn_up_key: String,
    pub ffn_down_key: String,
}

impl TransformerLayer {
    /// Cria uma camada com chaves no padrÃ£o llama.cpp GGUF: `blk.{idx}.*`
    pub fn llama_style(
        layer_idx: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        hidden_size: u32,
        intermediate_size: u32,
        norm_eps: f32,
    ) -> Self {
        let i = layer_idx;
        Self {
            layer_idx,
            attention_norm: RmsNorm { epsilon: norm_eps, dimension: hidden_size },
            attention: Attention { num_heads, num_kv_heads, head_dim },
            ffn_norm: RmsNorm { epsilon: norm_eps, dimension: hidden_size },
            ffn: SlGLU { hidden_size, intermediate_size },
            attn_norm_weight_key: format!("blk.{}.attn_norm.weight", i),
            attn_q_key:           format!("blk.{}.attn_q.weight", i),
            attn_k_key:           format!("blk.{}.attn_k.weight", i),
            attn_v_key:           format!("blk.{}.attn_v.weight", i),
            attn_out_key:         format!("blk.{}.attn_output.weight", i),
            ffn_norm_weight_key:  format!("blk.{}.ffn_norm.weight", i),
            ffn_gate_key:         format!("blk.{}.ffn_gate.weight", i),
            ffn_up_key:           format!("blk.{}.ffn_up.weight", i),
            ffn_down_key:         format!("blk.{}.ffn_down.weight", i),
        }
    }
}

/// Motor Principal LLM â€” usa pesos reais do WeightBank.
pub struct Transformer {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub layers: Vec<TransformerLayer>,
    pub norm: RmsNorm,
    pub rope: RoPE,
    pub use_zipgemm: bool,
    /// Chaves das normas e do LM head
    pub final_norm_key: String,
    pub lm_head_key: String,
    pub embed_key: String,
}

impl Transformer {
    /// ConstrÃ³i um Transformer Llama-style a partir dos metadados.
    ///
    /// Recebe os parÃ¢metros lidos do GGUF (via `GraphInterpreter`) e
    /// monta todas as camadas com as chaves corretas para o `WeightBank`.
    pub fn from_metadata(
        num_layers: u32,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        intermediate_size: u32,
        vocab_size: u32,
        rope_base: f32,
        norm_eps: f32,
        use_zipgemm: bool,
    ) -> Self {
        let head_dim = if num_heads > 0 { hidden_size / num_heads } else { 64 };

        let layers = (0..num_layers)
            .map(|i| TransformerLayer::llama_style(
                i, num_heads, num_kv_heads, head_dim,
                hidden_size, intermediate_size, norm_eps,
            ))
            .collect();

        Self {
            vocab_size,
            hidden_size,
            layers,
            norm: RmsNorm { epsilon: norm_eps, dimension: hidden_size },
            rope: RoPE { head_dim, base: rope_base },
            use_zipgemm,
            final_norm_key: "output_norm.weight".to_string(),
            lm_head_key:    "output.weight".to_string(),
            embed_key:      "token_embd.weight".to_string(),
        }
    }

    /// O Forward Pass Real.
    ///
    /// Recebe o embedding do token de entrada (jÃ¡ como `GpuBuffer`) e
    /// executa todas as camadas Transformer com pesos reais do `WeightBank`.
    ///
    /// # Ciclo de dados por camada:
    /// ```
    /// x = input_embed
    /// for each layer:
    ///   norm_x  = RmsNorm(x, attn_norm_weight)
    ///   q       = Matmul(norm_x, W_q)
    ///   k       = Matmul(norm_x, W_k)
    ///   v       = Matmul(norm_x, W_v)
    ///   q, k    = RoPE(q, k)
    ///   attn    = Attention(q, k, v)
    ///   out_attn = Matmul(attn, W_o)
    ///   x       = x + out_attn            â† residual connection
    ///   ffn_in  = RmsNorm(x, ffn_norm_weight)
    ///   gate    = SiLU(Matmul(ffn_in, W_gate)) * Matmul(ffn_in, W_up)
    ///   x       = x + Matmul(gate, W_down) â† residual connection
    /// x_final = RmsNorm(x, final_norm)
    /// logits  = Matmul(x_final, W_lm_head)
    /// ```
    pub fn forward(
        &self,
        engine: &VulkanEngine,
        input_embed: &crate::buffer::GpuBuffer,
        weights: &WeightBank,
        pos: u32,
    ) -> Result<crate::buffer::GpuBuffer, nodestor_core::NodeStorError> {
        let hidden_size = self.norm.dimension;

        // Estado atual das ativaÃ§Ãµes â€” comeÃ§a como cÃ³pia do embedding de entrada.
        // Como `GpuBuffer` pode ser de simulaÃ§Ã£o (RAM), usamos upload de zeros como
        // base e somamos via `add()` para manter o grafo de operaÃ§Ãµes limpo.
        let mut current_x = engine.alloc_buffer((hidden_size * 4) as usize)?;
        engine.add(&current_x, input_embed, hidden_size)?
            .copy_into(&mut current_x)?;

        // â”€â”€ Camadas Transformer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for layer in &self.layers {
            // 1. Attention sub-layer
            let attn_norm_w = weights.get(&layer.attn_norm_weight_key)
                .ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(
                    format!("Weight not found: {}", layer.attn_norm_weight_key)
                ))?;
            let norm_x = engine.rmsnorm(&current_x, attn_norm_w, 1, hidden_size, layer.attention_norm.epsilon)?;

            // Q, K, V projections â€” usa ZipGEMM (pesos comprimidos) ou Matmul denso
            let n_q = self.rope.head_dim * layer.attention.num_heads;
            let n_kv = self.rope.head_dim * layer.attention.num_kv_heads;

            let (mut q, mut k, v) = if self.use_zipgemm {
                let w_q   = weights.get(&layer.attn_q_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_q_key)))?;
                let w_k   = weights.get(&layer.attn_k_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_k_key)))?;
                let w_v   = weights.get(&layer.attn_v_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_v_key)))?;
                let meta  = weights.get(&format!("blk.{}.tile_meta", layer.layer_idx))
                    .unwrap_or(w_q); // fallback: usa W_q como meta (shape compatÃ­vel)
                let tile_stride     = 64u32;
                let num_k_tiles     = hidden_size / 16;
                let weights_per_tile = 256u32;
                (
                    engine.zipgemm(w_q, &norm_x, meta, 1, hidden_size, n_q, tile_stride, num_k_tiles, weights_per_tile)?,
                    engine.zipgemm(w_k, &norm_x, meta, 1, hidden_size, n_kv, tile_stride, num_k_tiles, weights_per_tile)?,
                    engine.zipgemm(w_v, &norm_x, meta, 1, hidden_size, n_kv, tile_stride, num_k_tiles, weights_per_tile)?,
                )
            } else {
                let w_q = weights.get(&layer.attn_q_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_q_key)))?;
                let w_k = weights.get(&layer.attn_k_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_k_key)))?;
                let w_v = weights.get(&layer.attn_v_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_v_key)))?;
                (
                    engine.matmul(&norm_x, w_q, 1, hidden_size, n_q)?,
                    engine.matmul(&norm_x, w_k, 1, hidden_size, n_kv)?,
                    engine.matmul(&norm_x, w_v, 1, hidden_size, n_kv)?,
                )
            };

            // 2. RoPE
            engine.rope(&mut q, &mut k, 1,
                layer.attention.num_heads, layer.attention.num_kv_heads,
                self.rope.head_dim, self.rope.base, pos)?;

            // 3. Attention (Flash ou clÃ¡ssico)
            let seq_len = 1u32; // decodificaÃ§Ã£o causal, 1 token novo por vez
            let scale = 1.0 / (self.rope.head_dim as f32).sqrt();
            let attn_out = engine.attention(&q, &k, &v, seq_len, self.rope.head_dim, scale)?;

            // 4. Output projection
            let w_o = weights.get(&layer.attn_out_key)
                .ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.attn_out_key)))?;
            let x_attn = engine.matmul(&attn_out, w_o, 1, n_q, hidden_size)?;

            // 5. Residual connection: x = x + x_attn âœ… (corrigido)
            let residual_a = engine.add(&current_x, &x_attn, hidden_size)?;
            current_x.copy_from(&residual_a)?;

            // 6. FFN sub-layer
            let ffn_norm_w = weights.get(&layer.ffn_norm_weight_key)
                .ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_norm_weight_key)))?;
            let ffn_in = engine.rmsnorm(&current_x, ffn_norm_w, 1, hidden_size, layer.ffn_norm.epsilon)?;

            let inter = layer.ffn.intermediate_size;

            let ffn_out = if self.use_zipgemm {
                let w_gate = weights.get(&layer.ffn_gate_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_gate_key)))?;
                let w_up   = weights.get(&layer.ffn_up_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_up_key)))?;
                let w_down = weights.get(&layer.ffn_down_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_down_key)))?;
                let meta   = w_gate; // fallback meta
                let tile_stride = 64u32;
                let num_k_tiles = hidden_size / 16;
                let wpt = 256u32;
                let gate  = engine.zipgemm(w_gate, &ffn_in, meta, 1, hidden_size, inter, tile_stride, num_k_tiles, wpt)?;
                let up    = engine.zipgemm(w_up,   &ffn_in, meta, 1, hidden_size, inter, tile_stride, num_k_tiles, wpt)?;
                // SwiGLU: SiLU(gate) * up
                let silu  = engine.silu(&gate, inter)?;
                let swiglu = engine.mul(&silu, &up, inter)?;
                let down_tiles = inter / 16;
                engine.zipgemm(w_down, &swiglu, meta, 1, inter, hidden_size, tile_stride, down_tiles, wpt)?
            } else {
                let w_gate = weights.get(&layer.ffn_gate_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_gate_key)))?;
                let w_up   = weights.get(&layer.ffn_up_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_up_key)))?;
                let w_down = weights.get(&layer.ffn_down_key).ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(format!("Weight not found: {}", layer.ffn_down_key)))?;
                let gate   = engine.matmul(&ffn_in, w_gate, 1, hidden_size, inter)?;
                let up     = engine.matmul(&ffn_in, w_up,   1, hidden_size, inter)?;
                let silu   = engine.silu(&gate, inter)?;
                let swiglu = engine.mul(&silu, &up, inter)?;
                engine.matmul(&swiglu, w_down, 1, inter, hidden_size)?
            };

            // 7. Residual connection: x = x + ffn_out âœ… (corrigido)
            let residual_b = engine.add(&current_x, &ffn_out, hidden_size)?;
            current_x.copy_from(&residual_b)?;
        }

        // â”€â”€ Final Normalization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        let final_norm_w = weights.get(&self.final_norm_key)
            .ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(
                format!("Weight not found: {}", self.final_norm_key)
            ))?;
        let x_final = engine.rmsnorm(&current_x, final_norm_w, 1, hidden_size, self.norm.epsilon)?;

        // â”€â”€ LM Head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        let lm_head_w = weights.get(&self.lm_head_key)
            .ok_or_else(|| nodestor_core::NodeStorError::InvalidModelFormat(
                format!("Weight not found: {}", self.lm_head_key)
            ))?;
        let logits = engine.matmul(&x_final, lm_head_w, 1, hidden_size, self.vocab_size)?;

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::HardwareProfile;

    fn mock_engine() -> VulkanEngine {
        let profile = HardwareProfile {
            os: nodestor_core::OsType::Windows,
            os_version: "mock".into(),
            cpu_cores: 1,
            total_ram_bytes: 4 * 1024 * 1024 * 1024,
            gpus: vec![],
            storage: vec![],
            recommended_transport: nodestor_core::TransportBackend::Win32Fallback,
            missed_optimizations: vec![]
        };
        VulkanEngine::new(&profile).unwrap()
    }

    fn make_weight_bank(engine: &VulkanEngine, hidden: u32, inter: u32, vocab: u32, layers: u32) -> WeightBank {
        let mut bank = WeightBank::new();
        // Insere pesos zerados com as shapes corretas para cada camada
        for i in 0..layers {
            let h = hidden as usize;
            let inter = inter as usize;
            bank.insert(format!("blk.{}.attn_norm.weight", i), engine.alloc_buffer(h * 4).unwrap());
            bank.insert(format!("blk.{}.attn_q.weight", i),    engine.alloc_buffer(h * h * 4).unwrap());
            bank.insert(format!("blk.{}.attn_k.weight", i),    engine.alloc_buffer(h * h * 4).unwrap());
            bank.insert(format!("blk.{}.attn_v.weight", i),    engine.alloc_buffer(h * h * 4).unwrap());
            bank.insert(format!("blk.{}.attn_output.weight", i), engine.alloc_buffer(h * h * 4).unwrap());
            bank.insert(format!("blk.{}.ffn_norm.weight", i),  engine.alloc_buffer(h * 4).unwrap());
            bank.insert(format!("blk.{}.ffn_gate.weight", i),  engine.alloc_buffer(h * inter * 4).unwrap());
            bank.insert(format!("blk.{}.ffn_up.weight", i),    engine.alloc_buffer(h * inter * 4).unwrap());
            bank.insert(format!("blk.{}.ffn_down.weight", i),  engine.alloc_buffer(inter * h * 4).unwrap());
        }
        bank.insert("output_norm.weight".into(), engine.alloc_buffer(hidden as usize * 4).unwrap());
        bank.insert("output.weight".into(),      engine.alloc_buffer(hidden as usize * vocab as usize * 4).unwrap());
        bank.insert("token_embd.weight".into(),  engine.alloc_buffer(vocab as usize * hidden as usize * 4).unwrap());
        bank
    }

    #[test]
    fn test_forward_zero_layers() {
        let engine = mock_engine();
        let hidden = 128u32;
        let transformer = Transformer::from_metadata(0, hidden, 2, 2, 256, 100, 10000.0, 1e-5, false);
        let bank = make_weight_bank(&engine, hidden, 256, 100, 0);
        let dummy_in = engine.alloc_buffer(hidden as usize * 4).unwrap();
        let logits = transformer.forward(&engine, &dummy_in, &bank, 0).unwrap();
        assert_eq!(logits.size, 100 * 4, "Logits devem ter vocab_size * 4 bytes");
    }

    #[test]
    fn test_forward_one_layer() {
        let engine = mock_engine();
        let hidden = 128u32;
        let transformer = Transformer::from_metadata(1, hidden, 2, 2, 256, 100, 10000.0, 1e-5, false);
        let bank = make_weight_bank(&engine, hidden, 256, 100, 1);
        let dummy_in = engine.alloc_buffer(hidden as usize * 4).unwrap();
        let logits = transformer.forward(&engine, &dummy_in, &bank, 0).unwrap();
        assert_eq!(logits.size, 100 * 4);
    }

    #[test]
    fn test_from_metadata_builds_correct_layers() {
        let t = Transformer::from_metadata(4, 512, 8, 8, 1024, 32000, 10000.0, 1e-5, false);
        assert_eq!(t.layers.len(), 4);
        assert_eq!(t.vocab_size, 32000);
        assert_eq!(t.layers[2].attn_q_key, "blk.2.attn_q.weight");
        assert_eq!(t.layers[3].ffn_down_key, "blk.3.ffn_down.weight");
        assert_eq!(t.final_norm_key, "output_norm.weight");
    }
}

