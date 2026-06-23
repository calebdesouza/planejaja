//! LoRA Core — Adaptadores de Baixo Posto para Edge Fine-Tuning.
//!
//! Implementa a fórmula delta sem alocação extra no inner loop:
//!   Δ(x) = (x · A) · B · (alpha / r)    A ∈ ℝ^{in×r}, B ∈ ℝ^{r×out}
//!   y    = W_base · x + Δ(x)
//!
//! Formato binário `.lora` (portável):
//!   [magic: u32=0x4C4F5241]["LORA"][version: u32=1]
//!   [hidden_dim: u32][global_rank: u32][global_alpha: f32][n_layers: u32]
//!   Por camada:
//!     [name_len: u32][name: UTF-8][in_features: u32][out_features: u32][rank: u32]
//!     [a: f32 × (in × rank)][b: f32 × (rank × out)]

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use nodestor_core::NodeStorError;

const LORA_MAGIC: u32 = 0x4C4F_5241; // "LORA"
const LORA_VERSION: u32 = 1;

// ─── LoraLayer ───────────────────────────────────────────────────────────────

/// Adaptador LoRA para uma projeção linear específica.
///
/// A ∈ ℝ^{in_features × rank}  — layout row-major: `a[i * rank + r]`
/// B ∈ ℝ^{rank × out_features} — layout row-major: `b[r * out_features + o]`
///
/// Invariante: B = 0 no início do treino (delta inicial nulo, conforme o artigo LoRA).
pub struct LoraLayer {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub alpha: f32,
}

impl LoraLayer {
    /// Cria um novo adaptador com B=0 e A inicializado com distribuição normal
    /// truncada (std = 1/√rank). Não depende de `rand` — usa hash deterministico.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        let std = 1.0_f32 / (rank as f32).sqrt();
        let mut a = Vec::with_capacity(in_features * rank);
        for i in 0..(in_features * rank) {
            // LCG de 64 bits (Knuth) — reproduzível, zero dependências externas
            let x = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
                >> 33;
            let f = (x as f32 / u32::MAX as f32 - 0.5) * 2.0 * std;
            a.push(f);
        }
        let b = vec![0.0f32; rank * out_features];
        Self { a, b, rank, in_features, out_features, alpha }
    }

    /// Escala LoRA: alpha / rank.
    #[inline]
    pub fn scale(&self) -> f32 {
        if self.rank == 0 { 0.0 } else { self.alpha / self.rank as f32 }
    }

    /// Calcula apenas o delta: Δ(x) = (x · A) · B · scale.
    ///
    /// Complexidade: O(in × rank + rank × out).
    /// Zero alocação extra além do buffer temporário `tmp[rank]`.
    pub fn apply_delta(&self, x: &[f32]) -> Vec<f32> {
        let n_in = x.len().min(self.in_features);
        let scale = self.scale();

        // tmp[r] = Σ_i A[i, r] · x[i]
        let mut tmp = vec![0.0f32; self.rank];
        for i in 0..n_in {
            let base = i * self.rank;
            let a_row = match self.a.get(base..base + self.rank) {
                Some(s) => s,
                None => break,
            };
            for r in 0..self.rank {
                tmp[r] += a_row[r] * x[i];
            }
        }

        // out[o] = Σ_r B[r, o] · tmp[r] · scale
        let mut out = vec![0.0f32; self.out_features];
        for r in 0..self.rank {
            let base = r * self.out_features;
            let b_row = match self.b.get(base..base + self.out_features) {
                Some(s) => s,
                None => break,
            };
            for o in 0..self.out_features {
                out[o] += b_row[o] * tmp[r] * scale;
            }
        }
        out
    }

    /// Forward completo: y = W_base · x + Δ(x).
    /// `w_base`: layout GGUF [out_features, in_features] row-major.
    pub fn forward(&self, x: &[f32], w_base: &[f32]) -> Vec<f32> {
        let n_in = x.len().min(self.in_features);
        let mut y = vec![0.0f32; self.out_features];
        for o in 0..self.out_features {
            let base = o * self.in_features;
            if let Some(row) = w_base.get(base..base + n_in) {
                for i in 0..n_in {
                    y[o] += row[i] * x[i];
                }
            }
        }
        let delta = self.apply_delta(x);
        for o in 0..self.out_features {
            y[o] += delta[o];
        }
        y
    }
}

// ─── LoraBank ────────────────────────────────────────────────────────────────

/// Banco de adaptadores LoRA — checkpoint completo com metadados de arquitetura.
pub struct LoraBank {
    pub layers: HashMap<String, LoraLayer>,
    /// hidden_dim do modelo para o qual este adaptador foi treinado.
    /// Usado para validação de compatibilidade e prevenção do "Vetor Fantasma".
    pub hidden_dim: usize,
    pub rank: usize,
    pub alpha: f32,
}

impl LoraBank {
    pub fn new(hidden_dim: usize, rank: usize, alpha: f32) -> Self {
        Self { layers: HashMap::new(), hidden_dim, rank, alpha }
    }

    pub fn insert(&mut self, name: String, layer: LoraLayer) {
        self.layers.insert(name, layer);
    }

    pub fn get(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut LoraLayer> {
        self.layers.get_mut(name)
    }

    /// Valida compatibilidade de arquitetura pelo `hidden_dim`.
    ///
    /// Previne o "Vetor Fantasma" — tentativa de usar um adaptador de 70B
    /// (hidden_dim=8192) em um modelo de 8B (hidden_dim=4096), que causaria
    /// produto escalar entre vetores de tamanhos diferentes (Segmentation Fault
    /// ou corrupção silenciosa de memória em código unsafe).
    pub fn check_compatible(&self, model_hidden: usize) -> Result<(), NodeStorError> {
        if self.hidden_dim != model_hidden {
            return Err(NodeStorError::InferenceError(format!(
                "Incompatibilidade de arquitetura LoRA: adaptador treinado para \
                 hidden_dim={} mas o modelo carregado tem hidden_dim={}. \
                 Use um adaptador treinado para este modelo específico.",
                self.hidden_dim, model_hidden
            )));
        }
        Ok(())
    }

    /// Fusão linear de N adaptadores em memória: `self += other × weight`.
    ///
    /// Permite combinar múltiplos `.lora` sem modificar os pesos base.
    /// Camadas presentes apenas em `other` são inseridas com `weight` aplicado.
    pub fn merge_with(&mut self, other: &LoraBank, weight: f32) {
        for (name, other_layer) in &other.layers {
            if let Some(self_layer) = self.layers.get_mut(name) {
                if self_layer.a.len() == other_layer.a.len() {
                    for (s, &o) in self_layer.a.iter_mut().zip(other_layer.a.iter()) {
                        *s += o * weight;
                    }
                }
                if self_layer.b.len() == other_layer.b.len() {
                    for (s, &o) in self_layer.b.iter_mut().zip(other_layer.b.iter()) {
                        *s += o * weight;
                    }
                }
            } else {
                self.layers.insert(name.clone(), LoraLayer {
                    a: other_layer.a.iter().map(|&v| v * weight).collect(),
                    b: other_layer.b.iter().map(|&v| v * weight).collect(),
                    rank: other_layer.rank,
                    in_features: other_layer.in_features,
                    out_features: other_layer.out_features,
                    alpha: other_layer.alpha,
                });
            }
        }
    }

    // ── Serialização binária ──────────────────────────────────────────────────

    /// Salva o banco em formato `.lora` binário portável.
    pub fn save(&self, path: &Path) -> Result<(), NodeStorError> {
        let mut buf: Vec<u8> = Vec::new();
        let w_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let w_f32 = |buf: &mut Vec<u8>, v: f32| buf.extend_from_slice(&v.to_le_bytes());

        w_u32(&mut buf, LORA_MAGIC);
        w_u32(&mut buf, LORA_VERSION);
        w_u32(&mut buf, self.hidden_dim as u32);
        w_u32(&mut buf, self.rank as u32);
        w_f32(&mut buf, self.alpha);
        w_u32(&mut buf, self.layers.len() as u32);

        for (name, layer) in &self.layers {
            let name_bytes = name.as_bytes();
            w_u32(&mut buf, name_bytes.len() as u32);
            buf.extend_from_slice(name_bytes);
            w_u32(&mut buf, layer.in_features as u32);
            w_u32(&mut buf, layer.out_features as u32);
            w_u32(&mut buf, layer.rank as u32);
            for &v in &layer.a { w_f32(&mut buf, v); }
            for &v in &layer.b { w_f32(&mut buf, v); }
        }

        std::fs::write(path, &buf)
            .map_err(|e| NodeStorError::InferenceError(format!("Falha ao salvar .lora '{}': {}", path.display(), e)))
    }

    /// Carrega um banco de adaptadores de um arquivo `.lora`.
    /// Falha com mensagem clara para arquivos corrompidos ou versão incompatível.
    pub fn load(path: &Path) -> Result<Self, NodeStorError> {
        let data = std::fs::read(path)
            .map_err(|e| NodeStorError::InferenceError(format!("Falha ao ler .lora '{}': {}", path.display(), e)))?;

        let mut pos = 0usize;

        macro_rules! rd_u32 {
            () => {{
                if pos + 4 > data.len() {
                    return Err(NodeStorError::InferenceError("Arquivo .lora truncado (u32)".into()));
                }
                let v = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
                pos += 4;
                v
            }};
        }
        macro_rules! rd_f32 {
            () => {{ f32::from_bits(rd_u32!()) }};
        }

        let magic = rd_u32!();
        if magic != LORA_MAGIC {
            return Err(NodeStorError::InferenceError(format!(
                "Arquivo não é um .lora válido (magic=0x{:08X}, esperado=0x{:08X}). \
                 Verifique se o arquivo não está corrompido.",
                magic, LORA_MAGIC
            )));
        }
        let version = rd_u32!();
        if version != LORA_VERSION {
            return Err(NodeStorError::InferenceError(format!(
                "Versão .lora não suportada: {} (suportado: {})", version, LORA_VERSION
            )));
        }

        let hidden_dim = rd_u32!() as usize;
        let rank = rd_u32!() as usize;
        let alpha = rd_f32!();
        let n_layers = rd_u32!() as usize;

        let mut layers = HashMap::new();
        for _ in 0..n_layers {
            let name_len = rd_u32!() as usize;
            if pos + name_len > data.len() {
                return Err(NodeStorError::InferenceError("Nome de camada truncado".into()));
            }
            let name = String::from_utf8(data[pos..pos + name_len].to_vec())
                .map_err(|_| NodeStorError::InferenceError("Nome de camada inválido (UTF-8)".into()))?;
            pos += name_len;

            let in_features = rd_u32!() as usize;
            let out_features = rd_u32!() as usize;
            let layer_rank = rd_u32!() as usize;
            let a_len = in_features * layer_rank;
            let b_len = layer_rank * out_features;

            if pos + (a_len + b_len) * 4 > data.len() {
                return Err(NodeStorError::InferenceError(format!(
                    "Dados da camada '{}' truncados: esperado {} bytes, disponível {}",
                    name, (a_len + b_len) * 4, data.len() - pos
                )));
            }

            let mut a = Vec::with_capacity(a_len);
            for _ in 0..a_len { a.push(rd_f32!()); }
            let mut b = Vec::with_capacity(b_len);
            for _ in 0..b_len { b.push(rd_f32!()); }

            layers.insert(name, LoraLayer { a, b, rank: layer_rank, in_features, out_features, alpha });
        }

        Ok(Self { layers, hidden_dim, rank, alpha })
    }
}

// ─── KDynamicScheduler ───────────────────────────────────────────────────────

/// Controlador de K-dinâmico para decodificação especulativa adaptativa.
///
/// Ajusta o tamanho do bloco especulativo K com base na taxa de aceitação
/// exponencialmente ponderada (EMA). Degrade graciosamente para K=1 (autoregressivo
/// padrão) quando a taxa de aceitação colapsa — nunca trava o pipeline.
///
/// Cenário crítico coberto: steering POD com `intensity` alta → distribuição do
/// modelo mestre diverge do rascunhador → taxa de aceitação cai para 0 →
/// K reduz para 1 → throughput degrade mas o pipeline NÃO TRAVA.
pub struct KDynamicScheduler {
    pub k_current: usize,
    k_min: usize,
    k_max: usize,
    acceptance_ema: f32,
    ema_alpha: f32,
    threshold_low: f32,
    threshold_high: f32,
}

impl KDynamicScheduler {
    pub fn new(k_initial: usize, k_min: usize, k_max: usize) -> Self {
        let k_min = k_min.max(1);
        Self {
            k_current: k_initial.max(k_min).min(k_max),
            k_min,
            k_max,
            acceptance_ema: 0.8,
            ema_alpha: 0.25,
            threshold_low: 0.3,
            threshold_high: 0.7,
        }
    }

    /// Atualiza K com base em `accepted` de `total` rascunhos e retorna o novo K.
    pub fn step(&mut self, accepted: usize, total: usize) -> usize {
        if total == 0 { return self.k_current; }
        let rate = accepted as f32 / total as f32;
        self.acceptance_ema = self.ema_alpha * rate + (1.0 - self.ema_alpha) * self.acceptance_ema;

        if self.acceptance_ema < self.threshold_low && self.k_current > self.k_min {
            self.k_current = (self.k_current / 2).max(self.k_min);
        } else if self.acceptance_ema > self.threshold_high && self.k_current < self.k_max {
            self.k_current = (self.k_current + 1).min(self.k_max);
        }
        self.k_current
    }

    #[inline]
    pub fn k(&self) -> usize { self.k_current }
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(in_f: usize, out_f: usize, r: usize) -> LoraLayer {
        LoraLayer::new(in_f, out_f, r, r as f32)
    }

    // ── Álgebra do delta ─────────────────────────────────────────────────────

    #[test]
    fn test_lora_delta_zero_when_b_is_zero() {
        // B=0 no início → delta deve ser exatamente zero em todos os elementos
        let layer = make_layer(8, 4, 2);
        let x = vec![1.0f32; 8];
        let delta = layer.apply_delta(&x);
        assert_eq!(delta.len(), 4);
        for (i, &d) in delta.iter().enumerate() {
            assert!(d.abs() < 1e-8, "delta[{}]={} deve ser 0 quando B=0", i, d);
        }
    }

    #[test]
    fn test_lora_forward_equals_base_plus_delta() {
        // y_forward deve coincidir com y_base + y_delta calculados separadamente
        let mut layer = make_layer(4, 4, 2);
        // Activa B para produzir delta não-trivial
        for (i, v) in layer.b.iter_mut().enumerate() {
            *v = 0.05 * (i as f32 + 1.0);
        }
        let x = vec![0.5f32; 4];
        let w_base: Vec<f32> = (0..16).map(|i| 0.01 * i as f32).collect();

        let y_forward = layer.forward(&x, &w_base);
        let delta = layer.apply_delta(&x);

        let mut y_manual = vec![0.0f32; 4];
        for o in 0..4 {
            for i in 0..4 { y_manual[o] += w_base[o * 4 + i] * x[i]; }
            y_manual[o] += delta[o];
        }

        for i in 0..4 {
            assert!((y_forward[i] - y_manual[i]).abs() < 1e-6,
                    "forward[{}]: {} ≠ {}", i, y_forward[i], y_manual[i]);
        }
    }

    // ── Validação de dimensões ("Vetor Fantasma") ─────────────────────────────

    #[test]
    fn test_lora_dimension_mismatch_returns_error() {
        // Adaptador treinado para modelo 70B (8192) → modelo 8B (4096) = erro claro
        let bank = LoraBank::new(8192, 8, 16.0);
        let result = bank.check_compatible(4096);
        assert!(result.is_err(), "deve detectar incompatibilidade de arquitetura");
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("8192") && msg.contains("4096"),
                "mensagem deve citar as duas dimensões: {}", msg);
    }

    #[test]
    fn test_lora_compatible_bank_succeeds() {
        let bank = LoraBank::new(4096, 8, 16.0);
        assert!(bank.check_compatible(4096).is_ok());
    }

    // ── Serialização binária ─────────────────────────────────────────────────

    #[test]
    fn test_lora_save_load_round_trip() {
        let mut bank = LoraBank::new(64, 4, 8.0);
        let mut layer = LoraLayer::new(64, 32, 4, 8.0);
        layer.a[0] = 0.12345;
        layer.b[7] = -0.98765;
        bank.insert("blk.0.attn_q.weight".to_string(), layer);

        let tmp = std::env::temp_dir().join("test_lora_core_rt.lora");
        bank.save(&tmp).expect("salvar deve funcionar");

        let loaded = LoraBank::load(&tmp).expect("carregar deve funcionar");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(loaded.hidden_dim, 64);
        assert_eq!(loaded.rank, 4);
        let lyr = loaded.layers.get("blk.0.attn_q.weight").expect("camada deve existir");
        assert_eq!(lyr.in_features, 64);
        assert_eq!(lyr.out_features, 32);
        assert!((lyr.a[0] - 0.12345).abs() < 1e-7, "a[0] round-trip");
        assert!((lyr.b[7] - (-0.98765)).abs() < 1e-7, "b[7] round-trip");
    }

    #[test]
    fn test_lora_load_invalid_magic_returns_error() {
        let tmp = std::env::temp_dir().join("test_lora_bad_magic.lora");
        std::fs::write(&tmp, b"\x00\x00\x00\x00garbage data").ok();
        let result = LoraBank::load(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert!(result.is_err(), "magic inválido deve retornar erro");
        // Usa Display (não Debug) para evitar exigir Debug nos tipos internos
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("magic") || msg.contains("Magic") || msg.contains("lora"),
                "mensagem deve explicar o erro: {}", msg);
    }

    // ── K-Dinâmico: resilência sob steering pesado ───────────────────────────

    #[test]
    fn test_speculation_rollback_under_heavy_steering() {
        // Cenário: steering com --intensity muito alta → taxa de aceitação = 0%
        // Invariante: K reduz para k_min=1 sem deadlock, depois se recupera
        let mut sched = KDynamicScheduler::new(8, 1, 16);

        // Simula 20 rounds de 100% rejeição (divergência máxima por steering)
        for round in 0..20 {
            let k_before = sched.k();
            let k_after = sched.step(0, k_before);
            assert!(k_after >= 1,
                "round {}: K={} deve ser ≥ 1 (deadlock prevention)", round, k_after);
        }
        assert_eq!(sched.k(), 1,
            "K deve colapsar para 1 após rejeição persistente (fallback autoregressivo)");

        // Recuperação: após aceitar 100% dos rascunhos, K deve subir
        for _ in 0..30 {
            let k = sched.k();
            sched.step(k, k); // 100% acceptance
        }
        assert!(sched.k() > 1,
            "K deve aumentar após taxa de aceitação alta (K={} após recuperação)", sched.k());
    }
}
