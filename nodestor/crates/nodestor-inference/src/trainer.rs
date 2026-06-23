//! Treinador Local de Micro-Adaptadores LoRA (Edge Fine-Tuning CPU).
//!
//! Implementa o loop de retropropagação restrito à projeção de saída (lm_head LoRA):
//! - Gradiente exato: ∂L/∂B e ∂L/∂A calculados analiticamente
//! - AdamW com acumulação de gradientes (controle rígido de VRAM via CPU)
//! - Cross-Entropy Loss com softmax numericamente estável
//! - Gradient clipping anti-NaN (norma-L2 global)
//! - Pesos base 100% congelados — só os adaptadores A, B são atualizados
//! - Reader de datasets JSONL locais (formato SFT e CLM)
//!
//! # Derivação do gradiente para a camada de saída LoRA
//!
//! Dado o forward:  y = W_base · h + Δ(h),   Δ(h) = B · (A · h) · scale
//! Loss:            L = −log(softmax(y)[target])
//! grad_y:          ∂L/∂y = softmax(y) − one_hot(target)
//! grad_B[r, o]:    ∂L/∂B[r,o] = (A·h)[r] · (∂L/∂y)[o] · scale
//! grad_A[i, r]:    ∂L/∂A[i,r] = h[i] · (Bᵀ · ∂L/∂y)[r] · scale

use std::io::BufRead;
use std::path::Path;
use nodestor_core::NodeStorError;
use crate::lora_core::LoraLayer;

// ─── Configuração ────────────────────────────────────────────────────────────

/// Parâmetros do loop de treinamento local.
pub struct TrainingConfig {
    pub learning_rate: f32,
    /// Passos de acumulação antes de aplicar gradientes (simula batch maior em VRAM limitada)
    pub grad_accum_steps: usize,
    /// Norma-L2 máxima dos gradientes (clipping anti-NaN)
    pub max_grad_norm: f32,
    pub weight_decay: f32,
    /// β₁ do AdamW (momentum de primeira ordem)
    pub beta1: f32,
    /// β₂ do AdamW (momentum de segunda ordem)
    pub beta2: f32,
    pub eps: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            grad_accum_steps: 4,
            max_grad_norm: 1.0,
            weight_decay: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Resultado de um passo de treinamento.
pub struct TrainStepResult {
    pub loss: f32,
    pub grad_norm_a: f32,
    pub grad_norm_b: f32,
    pub step: u32,
}

// ─── AdamW ───────────────────────────────────────────────────────────────────

/// Estado do otimizador AdamW para um vetor de parâmetros.
pub struct AdamWState {
    m: Vec<f32>,   // primeiro momento (média)
    v: Vec<f32>,   // segundo momento (variância)
    pub step: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    lr: f32,
    weight_decay: f32,
}

impl AdamWState {
    pub fn new(n_params: usize, cfg: &TrainingConfig) -> Self {
        Self {
            m: vec![0.0f32; n_params],
            v: vec![0.0f32; n_params],
            step: 0,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            lr: cfg.learning_rate,
            weight_decay: cfg.weight_decay,
        }
    }

    /// Aplica uma atualização AdamW in-place ao vetor de parâmetros `params`.
    /// `grad` deve ter o mesmo comprimento que `params`.
    pub fn update(&mut self, params: &mut [f32], grad: &[f32]) {
        self.step += 1;
        let t = self.step as f32;
        let bias1 = 1.0 - self.beta1.powf(t);
        let bias2 = 1.0 - self.beta2.powf(t);

        for i in 0..params.len().min(grad.len()) {
            let g = grad[i];
            // Guard: ignora gradientes NaN/Inf em vez de contaminar o estado
            if !g.is_finite() { continue; }

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;

            // Weight decay decoupled (AdamW vs Adam)
            params[i] *= 1.0 - self.lr * self.weight_decay;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── Loss e Gradiente ────────────────────────────────────────────────────────

/// Cross-Entropy Loss com softmax numericamente estável.
///
/// Retorna `(loss, grad_logits)` onde:
///   `grad_logits[i] = softmax(logits)[i] − one_hot(target)[i]`
pub fn cross_entropy_loss(logits: &[f32], target_id: usize) -> (f32, Vec<f32>) {
    let n = logits.len();
    if n == 0 { return (0.0, vec![]); }

    // Softmax numericamente estável (subtrai o máximo)
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() { *p /= sum; }
    }

    let target = target_id.min(n - 1);
    let loss = -(probs[target].max(1e-12)).ln();

    // Gradiente: softmax(y) − one_hot(target)
    let mut grad = probs;
    grad[target] -= 1.0;

    (loss, grad)
}

/// Clipa gradientes pela norma-L2 global.
/// Se `‖g‖ > max_norm`, escala `g ← g * (max_norm / ‖g‖)`.
/// Retorna a norma antes do clipping.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let norm_sq: f32 = grads.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();
    if norm.is_finite() && norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
    // Se norm é NaN/Inf → zera todos os gradientes (guarda anti-NaN)
    if !norm.is_finite() {
        for g in grads.iter_mut() { *g = 0.0; }
    }
    norm
}

// ─── LocalTrainer ────────────────────────────────────────────────────────────

/// Treinador local de LoRA para a camada de saída (lm_head).
///
/// Pesos base (`w_base`) são passados como slice imutável — nunca modificados.
/// Apenas as matrizes A e B do `LoraLayer` são atualizadas.
pub struct LocalTrainer {
    pub config: TrainingConfig,
    opt_a: AdamWState,
    opt_b: AdamWState,
    grad_a_accum: Vec<f32>,
    grad_b_accum: Vec<f32>,
    accum_count: usize,
}

impl LocalTrainer {
    pub fn new(config: TrainingConfig, lora: &LoraLayer) -> Self {
        let a_len = lora.a.len();
        let b_len = lora.b.len();
        Self {
            opt_a: AdamWState::new(a_len, &config),
            opt_b: AdamWState::new(b_len, &config),
            grad_a_accum: vec![0.0f32; a_len],
            grad_b_accum: vec![0.0f32; b_len],
            accum_count: 0,
            config,
        }
    }

    /// Executa um passo de treinamento sobre um único exemplo.
    ///
    /// Parâmetros:
    /// - `lora`: adaptador a ser atualizado (A e B modificados in-place)
    /// - `hidden`: hidden state final do modelo [in_features] — CONGELADO
    /// - `w_base`: pesos da projeção de saída [out_features, in_features] — CONGELADO
    /// - `target_id`: token alvo (índice no vocabulário)
    ///
    /// Retorna `None` se os dados forem inválidos (hidden vazio, target out of bounds).
    pub async fn train_local_step(
        &mut self,
        lora: &mut LoraLayer,
        hidden: &[f32],
        w_base: &[f32],
        target_id: usize,
    ) -> Option<TrainStepResult> {
        if hidden.is_empty() || lora.out_features == 0 { return None; }
        if target_id >= lora.out_features { return None; }

        // ── 1. Forward pass (pesos base + LoRA delta) ────────────────────────
        let logits = lora.forward(hidden, w_base);

        // ── 2. Cross-entropy loss + gradiente dos logits ─────────────────────
        let (loss, grad_y) = cross_entropy_loss(&logits, target_id);

        // ── 3. Backprop: gradientes de B e A ─────────────────────────────────
        // tmp[r] = (A · hidden)[r]  (reutiliza o intermédio do forward)
        let n_in = hidden.len().min(lora.in_features);
        let scale = lora.scale();
        let mut tmp_ax = vec![0.0f32; lora.rank];
        for i in 0..n_in {
            let base = i * lora.rank;
            if let Some(row) = lora.a.get(base..base + lora.rank) {
                for r in 0..lora.rank {
                    tmp_ax[r] += row[r] * hidden[i];
                }
            }
        }

        // grad_B[r, o] = tmp_ax[r] · grad_y[o] · scale
        for r in 0..lora.rank {
            let base_b = r * lora.out_features;
            if base_b + lora.out_features > self.grad_b_accum.len() { break; }
            for o in 0..lora.out_features {
                self.grad_b_accum[base_b + o] += tmp_ax[r] * grad_y[o] * scale;
            }
        }

        // btg[r] = (Bᵀ · grad_y)[r] = Σ_o B[r, o] · grad_y[o]
        let mut btg = vec![0.0f32; lora.rank];
        for r in 0..lora.rank {
            let base_b = r * lora.out_features;
            if let Some(b_row) = lora.b.get(base_b..base_b + lora.out_features) {
                for o in 0..lora.out_features {
                    btg[r] += b_row[o] * grad_y[o];
                }
            }
        }

        // grad_A[i, r] = hidden[i] · btg[r] · scale
        for i in 0..n_in {
            let base_a = i * lora.rank;
            if base_a + lora.rank > self.grad_a_accum.len() { break; }
            for r in 0..lora.rank {
                self.grad_a_accum[base_a + r] += hidden[i] * btg[r] * scale;
            }
        }

        self.accum_count += 1;

        // ── 4. Aplica gradientes acumulados (a cada `grad_accum_steps` passos) ─
        if self.accum_count >= self.config.grad_accum_steps {
            // Normaliza pelo número de passos acumulados
            let n = self.accum_count as f32;
            for g in self.grad_a_accum.iter_mut() { *g /= n; }
            for g in self.grad_b_accum.iter_mut() { *g /= n; }

            // Gradient clipping
            let norm_a = clip_grad_norm(&mut self.grad_a_accum, self.config.max_grad_norm);
            let norm_b = clip_grad_norm(&mut self.grad_b_accum, self.config.max_grad_norm);

            // AdamW update (base permanece congelada)
            self.opt_a.update(&mut lora.a, &self.grad_a_accum);
            self.opt_b.update(&mut lora.b, &self.grad_b_accum);

            // Zera acumuladores
            for g in self.grad_a_accum.iter_mut() { *g = 0.0; }
            for g in self.grad_b_accum.iter_mut() { *g = 0.0; }
            self.accum_count = 0;

            return Some(TrainStepResult {
                loss,
                grad_norm_a: norm_a,
                grad_norm_b: norm_b,
                step: self.opt_b.step,
            });
        }

        Some(TrainStepResult {
            loss,
            grad_norm_a: 0.0,
            grad_norm_b: 0.0,
            step: self.opt_b.step,
        })
    }
}

// ─── Dataset JSONL ───────────────────────────────────────────────────────────

/// Par de treinamento lido de um arquivo JSONL.
pub struct TrainSample {
    pub input: String,
    pub output: String,
}

/// Lê pares de treinamento de um arquivo JSONL.
///
/// Formatos suportados:
/// - `{"input": "...", "output": "..."}` — SFT (Supervised Fine-Tuning)
/// - `{"text": "..."}` — CLM (Causal Language Modeling, input=output=text)
/// - `{"prompt": "...", "completion": "..."}` — formato OpenAI
pub fn read_jsonl_dataset(path: &Path) -> Result<Vec<TrainSample>, NodeStorError> {
    let file = std::fs::File::open(path)
        .map_err(|e| NodeStorError::InferenceError(format!("Falha ao abrir dataset '{}': {}", path.display(), e)))?;

    let mut samples = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| NodeStorError::InferenceError(format!("Linha {}: {}", lineno + 1, e)))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| NodeStorError::InferenceError(format!("JSON inválido na linha {}: {}", lineno + 1, e)))?;

        let sample = if let (Some(input), Some(output)) = (
            v.get("input").and_then(|x| x.as_str()),
            v.get("output").and_then(|x| x.as_str()),
        ) {
            TrainSample { input: input.to_string(), output: output.to_string() }
        } else if let (Some(prompt), Some(completion)) = (
            v.get("prompt").and_then(|x| x.as_str()),
            v.get("completion").and_then(|x| x.as_str()),
        ) {
            TrainSample { input: prompt.to_string(), output: completion.to_string() }
        } else if let Some(text) = v.get("text").and_then(|x| x.as_str()) {
            TrainSample { input: text.to_string(), output: text.to_string() }
        } else {
            return Err(NodeStorError::InferenceError(format!(
                "Linha {}: formato não reconhecido (esperado: input/output, prompt/completion, ou text)",
                lineno + 1
            )));
        };

        samples.push(sample);
    }

    if samples.is_empty() {
        return Err(NodeStorError::InferenceError(format!(
            "Dataset '{}' está vazio ou não contém amostras válidas", path.display()
        )));
    }

    Ok(samples)
}

// ─── Testes ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora_core::LoraLayer;

    /// Gera um hidden state e w_base sintéticos para testes.
    /// w_base: identidade escalonada para que y = scale * hidden inicialmente.
    fn make_test_setup(hidden_dim: usize, vocab: usize, rank: usize) -> (LoraLayer, Vec<f32>, Vec<f32>) {
        let lora = LoraLayer::new(hidden_dim, vocab, rank, rank as f32);
        let hidden: Vec<f32> = (0..hidden_dim).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        // w_base: diagonal escalonada (identidade se hidden_dim==vocab)
        let mut w_base = vec![0.0f32; vocab * hidden_dim];
        for o in 0..vocab.min(hidden_dim) {
            w_base[o * hidden_dim + o] = 1.0;
        }
        (lora, hidden, w_base)
    }

    // ── Convergência ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_loss_decreases_over_epochs() {
        // Verifica que a perda cai após passos de otimização com gradiente exato.
        // Setup mínimo: vocab=16, hidden=8, rank=2, target=3
        let (mut lora, hidden, w_base) = make_test_setup(8, 16, 2);
        let cfg = TrainingConfig {
            learning_rate: 0.01,
            grad_accum_steps: 1, // aplica a cada passo
            max_grad_norm: 1.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut trainer = LocalTrainer::new(cfg, &lora);

        // Mede a perda inicial (antes de qualquer update)
        let (loss_0, _) = cross_entropy_loss(&lora.forward(&hidden, &w_base), 3);

        // 20 passos de gradient descent
        for _ in 0..20 {
            trainer.train_local_step(&mut lora, &hidden, &w_base, 3).await;
        }

        let (loss_final, _) = cross_entropy_loss(&lora.forward(&hidden, &w_base), 3);

        assert!(loss_final < loss_0,
            "perda deve diminuir: loss_0={:.4} > loss_final={:.4}", loss_0, loss_final);
    }

    // ── Pesos base congelados ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_frozen_base_weights() {
        // Garante que w_base permanece 100% intocado após múltiplos steps.
        let (mut lora, hidden, w_base) = make_test_setup(8, 16, 2);
        let w_base_snapshot = w_base.clone();

        let cfg = TrainingConfig { learning_rate: 0.1, grad_accum_steps: 1, ..Default::default() };
        let mut trainer = LocalTrainer::new(cfg, &lora);

        for _ in 0..10 {
            trainer.train_local_step(&mut lora, &hidden, &w_base, 5).await;
        }

        // w_base deve ser idêntico ao snapshot inicial
        for (i, (&original, &current)) in w_base_snapshot.iter().zip(w_base.iter()).enumerate() {
            assert_eq!(original, current,
                "w_base[{}] foi modificado: {} → {}", i, original, current);
        }

        // LoRA B deve ter mudado (A pode ou não, dependendo do gradient)
        let b_all_zero = lora.b.iter().all(|&v| v == 0.0);
        assert!(!b_all_zero, "B deve ter sido atualizado pelo gradient descent");
    }

    // ── Gradient clipping anti-NaN ───────────────────────────────────────────

    #[test]
    fn test_gradient_clamping_prevents_nan() {
        // Injeta gradientes com NaN e Inf — clip_grad_norm deve neutralizá-los
        let mut grads = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 1.0, 2.0];
        let original_len = grads.len();
        clip_grad_norm(&mut grads, 1.0);

        // Após clipping com NaN na norma → todos os gradientes devem ser finitos
        for (i, &g) in grads.iter().enumerate() {
            assert!(g.is_finite() || g == 0.0,
                "grad[{}]={} deve ser finito após clipping", i, g);
        }
        assert_eq!(grads.len(), original_len, "comprimento deve ser preservado");
    }

    #[test]
    fn test_gradient_clipping_scales_large_norm() {
        // Verifica que gradientes com norma > max_norm são escalados corretamente
        let mut grads = vec![3.0f32, 4.0]; // norma = 5.0
        let norm_before = clip_grad_norm(&mut grads, 1.0);
        assert!((norm_before - 5.0).abs() < 1e-5, "norma antes: {}", norm_before);

        let norm_after: f32 = grads.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((norm_after - 1.0).abs() < 1e-5,
            "norma após clipping deve ser ≈ 1.0, obtido {}", norm_after);
    }

    // ── Cross-entropy ────────────────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_loss_correct_gradient() {
        // Para logits uniformes [1, 1, 1, 1], softmax = [0.25, 0.25, 0.25, 0.25]
        // grad[target] deve ser ≈ 0.25 - 1.0 = -0.75; grad[outros] ≈ 0.25
        let logits = vec![1.0f32; 4];
        let (loss, grad) = cross_entropy_loss(&logits, 2);
        assert!(loss > 0.0, "perda deve ser positiva");
        assert!((grad[2] - (-0.75)).abs() < 0.01, "grad[target]={:.4}", grad[2]);
        assert!((grad[0] - 0.25).abs() < 0.01, "grad[0]={:.4}", grad[0]);
    }

    // ── JSONL reader ─────────────────────────────────────────────────────────

    #[test]
    fn test_jsonl_reader_sft_format() {
        let tmp = std::env::temp_dir().join("test_trainer_sft.jsonl");
        std::fs::write(&tmp,
            "{\"input\": \"Olá\", \"output\": \"Mundo\"}\n\
             {\"prompt\": \"2+2\", \"completion\": \"4\"}\n\
             {\"text\": \"auto\"}\n"
        ).unwrap();
        let samples = read_jsonl_dataset(&tmp).expect("deve ler JSONL");
        std::fs::remove_file(&tmp).ok();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].input, "Olá");
        assert_eq!(samples[0].output, "Mundo");
        assert_eq!(samples[1].input, "2+2");
        assert_eq!(samples[2].input, "auto");
        assert_eq!(samples[2].output, "auto");
    }

    #[test]
    fn test_jsonl_reader_empty_file_returns_error() {
        let tmp = std::env::temp_dir().join("test_trainer_empty.jsonl");
        std::fs::write(&tmp, "# comentário\n\n").unwrap();
        let result = read_jsonl_dataset(&tmp);
        std::fs::remove_file(&tmp).ok();
        assert!(result.is_err(), "arquivo vazio deve retornar erro");
    }
}
