use std::f32;

/// PROBES V2 — SAE Engine (Sparse Autoencoder)
///
/// Implementa a microscopia latente decompondo _hidden states_ residuais em
/// representações de Features monosemânticas de altíssima dimensão
/// usando ativação funcional JumpReLU para impor a esparsidade desejada.
///
/// ## Melhorias v2 (pesquisa Cirurgia Latente Universal):
///
/// ### OrtSAE (Ortogonalidade)
/// Aplica uma passo de Gram-Schmidt parcial periodicamente para garantir que
/// features distintas não se "absorvam" (feature absorption). Cada feature
/// tende a representar exatamente UM conceito humano-interpretável.
///
/// ### TIDE (Temporal-aware InferencE)
/// Em modelos de difusão, o `JumpReLU threshold` é modulado pelo `timestep`:
/// - Timestep alto (início da denoising) → threshold mais baixo → mais features ativas → foco em estrutura global
/// - Timestep baixo (fim da denoising) → threshold mais alto → features esparsas → foco em detalhes finos
///
/// ### Cohen's d Ranking
/// `rank_features_by_cohens_d()` compara duas populações de ativações
/// (ex: prompt proibido vs permitido) e retorna as features ordenadas
/// pelo tamanho do efeito causal — base do `RefusalMapper`.

/// Modo de entrada do SAE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SAEInputMode {
    /// Dimensão fixa — comportamento original (Transformers, texto)
    Fixed,
    /// Funcional — amostragem adaptativa antes da multiplicação matricial
    /// Útil para audio/imagem com resolução variável (Whisper, CLIP)
    Functional { target_dim: usize },
}

pub struct SAEEngine {
    pub hidden_dim: usize,         // Dimensão do espaço latente do LLM (ex: 4096)
    pub dict_size: usize,          // Expansão do espaço de features (ex: 32768)
    pub encoder_weights: Vec<f32>, // Shape: (F x D)
    pub decoder_weights: Vec<f32>, // Shape: (D x F) usualmente atrelados (tied)
    pub encoder_bias: Vec<f32>,    // Shape: (F)
    pub threshold: f32,            // JumpReLU activation threshold base
    /// Modo de entrada (Fixed para LLMs, Functional para multimodal)
    pub input_mode: SAEInputMode,
    /// Contador de encodes para acionar ortogonalização esparsa (OrtSAE)
    encode_count: u64,
    /// Intervalo de re-ortogonalização (a cada N encodes)
    pub orthogonalization_interval: u64,
}

impl SAEEngine {
    /// Inicializa um Sparse Autoencoder com as devidas dimensões.
    /// Em ambiente de produção o init importaria os Tensores pré-treinados
    /// como os proveídos no Gemma Scope ou Llama SAE.
    pub fn new(hidden_dim: usize, dict_size: usize, threshold: f32) -> Self {
        Self {
            hidden_dim,
            dict_size,
            // Mock de pesos pré-treinados:
            encoder_weights: vec![0.01; hidden_dim * dict_size],
            decoder_weights: vec![0.01; hidden_dim * dict_size],
            encoder_bias: vec![0.0; dict_size],
            threshold,
            input_mode: SAEInputMode::Fixed,
            encode_count: 0,
            orthogonalization_interval: 512, // Re-ortogonaliza a cada 512 encodes
        }
    }

    /// Mapeia _Forward_: Espaço Latente → Espaço Monosemântico de Features.
    /// f = JumpReLU(W_enc * h + b_enc)
    ///
    /// `timestep`: Se `Some(t)`, ativa modo TIDE — o threshold JumpReLU é modulado
    /// pelo timestep de difusão. `t ∈ [0.0, 1.0]` onde 1.0 = início (estrutura)
    /// e 0.0 = fim (detalhe). Sem timestep = comportamento original.
    pub fn encode(&mut self, hidden_states: &[f32]) -> Vec<f32> {
        self.encode_with_timestep(hidden_states, None)
    }

    /// Variante TIDE: encode com consciência temporal para models de difusão.
    pub fn encode_with_timestep(
        &mut self,
        hidden_states: &[f32],
        timestep: Option<f32>,
    ) -> Vec<f32> {
        // Amostragem adaptativa para modo Functional (SAE-NO básico)
        let effective_states: Vec<f32> = match self.input_mode {
            SAEInputMode::Fixed => {
                assert_eq!(
                    hidden_states.len(), self.hidden_dim,
                    "Dimensão residual incompatível"
                );
                hidden_states.to_vec()
            }
            SAEInputMode::Functional { target_dim } => {
                // Amostragem linear do input para target_dim (proxy de neural operator)
                // Em produção: usar interpolação bilinear ou spline
                let len = hidden_states.len();
                (0..target_dim.min(self.hidden_dim))
                    .map(|i| {
                        let src_idx = (i as f32 / target_dim as f32 * len as f32) as usize;
                        hidden_states.get(src_idx).copied().unwrap_or(0.0)
                    })
                    .chain(std::iter::repeat(0.0))
                    .take(self.hidden_dim)
                    .collect()
            }
        };

        // TIDE: modulação do threshold pelo timestep de difusão
        // Timestep alto (1.0) = início da denoising = threshold reduzido = mais features = estrutura
        // Timestep baixo (0.0) = fim = threshold elevado = features esparsas = detalhes
        let effective_threshold = match timestep {
            Some(t) => {
                let t = t.clamp(0.0, 1.0);
                // Interpola: threshold dimunui 40% no início (máx estrutura)
                self.threshold * (1.0 - t * 0.4)
            }
            None => self.threshold,
        };

        let mut features = vec![0.0f32; self.dict_size];

        for i in 0..self.dict_size {
            let mut dot = self.encoder_bias[i];
            for j in 0..self.hidden_dim {
                dot += effective_states[j] * self.encoder_weights[i * self.hidden_dim + j];
            }
            features[i] = if dot > effective_threshold { dot } else { 0.0 };
        }

        // OrtSAE: re-ortogonalização periódica (cada N invocações)
        self.encode_count += 1;
        if self.encode_count % self.orthogonalization_interval == 0 {
            self.partial_orthogonalize();
        }

        features
    }

    /// OrtSAE — Gram-Schmidt parcial nos pesos do encoder.
    ///
    /// Para cada par de features vizinhas (i, i+1), remove a componente de i
    /// na direção de i+1, reduzindo a superposição entre features.
    /// "Parcial" porque fazemos apenas pares adjacentes por eficiência —
    /// a ortogonalidade total exigiria O(F²) operações.
    pub fn partial_orthogonalize(&mut self) {
        let d = self.hidden_dim;
        let f = self.dict_size;

        // Itera em pares de features adjacentes
        let mut i = 0;
        while i + 1 < f {
            // Extrai as linhas i e i+1 do encoder (cada linha = um vetor de feature)
            let row_i_start = i * d;
            let row_j_start = (i + 1) * d;

            // Calcula produto interno <e_i, e_{i+1}>
            let mut dot_ij = 0.0f32;
            let mut norm_i_sq = 0.0f32;
            for k in 0..d {
                dot_ij += self.encoder_weights[row_i_start + k]
                    * self.encoder_weights[row_j_start + k];
                norm_i_sq += self.encoder_weights[row_i_start + k].powi(2);
            }

            // Somente ortogonaliza se há superposição significativa
            if norm_i_sq > 1e-10 && dot_ij.abs() > 0.1 * norm_i_sq.sqrt() {
                let projection = dot_ij / norm_i_sq;
                // e_{i+1} ← e_{i+1} - proj * e_i  (remove componente de e_i)
                for k in 0..d {
                    let proj_component = projection * self.encoder_weights[row_i_start + k];
                    self.encoder_weights[row_j_start + k] -= proj_component;
                }
            }

            i += 2; // Pares não-overlapping para evitar ordenação causal
        }
    }

    /// Rankeia features por Cohen's d entre duas populações de ativações.
    ///
    /// Usado pelo `RefusalMapper` para identificar features causalmente
    /// responsáveis pela diferença entre dois comportamentos do modelo.
    ///
    /// `pop_a`: população de ativações do estado A (ex: recusa)
    /// `pop_b`: população de ativações do estado B (ex: permissão)
    ///
    /// Retorna `Vec<(feature_idx, cohens_d)>` ordenado por `|d|` decrescente.
    pub fn rank_features_by_cohens_d(
        &self,
        pop_a: &[Vec<f32>], // n_samples × dict_size
        pop_b: &[Vec<f32>],
    ) -> Vec<(usize, f32)> {
        if pop_a.is_empty() || pop_b.is_empty() {
            return Vec::new();
        }

        let na = pop_a.len() as f32;
        let nb = pop_b.len() as f32;

        // Médias por feature
        let mut mean_a = vec![0.0f32; self.dict_size];
        let mut mean_b = vec![0.0f32; self.dict_size];

        for sample in pop_a {
            for (i, &v) in sample.iter().enumerate().take(self.dict_size) {
                mean_a[i] += v / na;
            }
        }
        for sample in pop_b {
            for (i, &v) in sample.iter().enumerate().take(self.dict_size) {
                mean_b[i] += v / nb;
            }
        }

        // Variâncias por feature
        let mut var_a = vec![0.0f32; self.dict_size];
        let mut var_b = vec![0.0f32; self.dict_size];

        for sample in pop_a {
            for (i, &v) in sample.iter().enumerate().take(self.dict_size) {
                var_a[i] += (v - mean_a[i]).powi(2) / na;
            }
        }
        for sample in pop_b {
            for (i, &v) in sample.iter().enumerate().take(self.dict_size) {
                var_b[i] += (v - mean_b[i]).powi(2) / nb;
            }
        }

        // Cohen's d por feature
        let mut ranked: Vec<(usize, f32)> = (0..self.dict_size)
            .map(|i| {
                let sigma_pooled = ((var_a[i] + var_b[i]) / 2.0).sqrt();
                let d = if sigma_pooled < 1e-8 {
                    let delta = mean_a[i] - mean_b[i];
                    if delta.abs() < 1e-8 { 0.0 } else { delta.signum() * 10.0 }
                } else {
                    (mean_a[i] - mean_b[i]) / sigma_pooled
                };
                (i, d)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Reconstrói as características latentes ativas de volta ao Residual Stream.
    /// Serve para monitorar erro de reconstrução e injetar Steering Vectors de alta dimensionalidade de volta as trilhas baixas da IA.
    /// h_hat = W_dec * f 
    pub fn decode(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), self.dict_size, "Dimensão de features incompatível");
        let mut reconstructed = vec![0.0; self.hidden_dim];

        // Otimização basica: Apenas roda dot para features não zeradas (graças a Sparsity)
        let active_features: Vec<(usize, f32)> = features
            .iter()
            .enumerate()
            .filter(|(_, &f)| f > 0.0)
            .map(|(i, &f)| (i, f))
            .collect();

        for i in 0..self.hidden_dim {
            let mut dot = 0.0;
            for &(j, val) in &active_features {
                dot += val * self.decoder_weights[i * self.dict_size + j];
            }
            reconstructed[i] = dot;
        }

        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sae_engine_sparsity_and_reconstruction() {
        let h_dim = 128;
        let dict = 1024;
        let threshold = 0.5;

        let mut sae = SAEEngine::new(h_dim, dict, threshold);
        sae.encoder_weights[5 * h_dim + 10] = 10.0;
        sae.decoder_weights[10 * dict + 5] = 1.0;

        let mut h = vec![0.0; h_dim];
        h[10] = 1.0;

        let latents = sae.encode(&h);
        let non_zeros = latents.iter().filter(|&&v| v > 0.0).count();
        assert!(non_zeros < dict / 10, "Esparsidade falhou.");
        assert_eq!(latents[5], 10.0, "SAE nao recuperou feature esperado.");

        let h_rcns = sae.decode(&latents);
        assert!(h_rcns[10] > 0.0, "Residuo apagou na reconstrucao.");
    }

    #[test]
    fn test_tide_timestep_modulates_threshold() {
        let mut sae = SAEEngine::new(16, 64, 1.0);
        sae.encoder_weights[3 * 16 + 5] = 10.0;
        let mut h = vec![0.0f32; 16];
        h[5] = 0.07; // dot = 0.7

        // threshold base=1.0 -> 0.7 < 1.0 -> nao ativa
        let f_no_tide = sae.encode(&h);
        assert_eq!(f_no_tide[3], 0.0, "Sem TIDE nao deve ativar");

        // t=1.0 -> threshold = 0.6 -> 0.7 > 0.6 -> ATIVA
        let f_tide_start = sae.encode_with_timestep(&h, Some(1.0));
        assert!(f_tide_start[3] > 0.0, "TIDE t=1.0 deve ativar feature borderline");

        // t=0.0 -> threshold = 1.0 -> nao ativa
        let f_tide_end = sae.encode_with_timestep(&h, Some(0.0));
        assert_eq!(f_tide_end[3], 0.0, "TIDE t=0.0 nao muda threshold");
    }

    #[test]
    fn test_ortsae_reduces_feature_overlap() {
        let d = 4;
        let f = 4;
        let mut sae = SAEEngine::new(d, f, 0.1);

        // e_0 = [1,0,0,0], e_1 = [0.9,0.1,0,0] (alto overlap)
        sae.encoder_weights[0] = 1.0;
        sae.encoder_weights[1] = 0.0;
        sae.encoder_weights[2] = 0.0;
        sae.encoder_weights[3] = 0.0;
        sae.encoder_weights[4] = 0.9;
        sae.encoder_weights[5] = 0.1;
        sae.encoder_weights[6] = 0.0;
        sae.encoder_weights[7] = 0.0;

        let dot_before: f32 = (0..d).map(|k| sae.encoder_weights[k] * sae.encoder_weights[d + k]).sum();
        sae.partial_orthogonalize();
        let dot_after: f32 = (0..d).map(|k| sae.encoder_weights[k] * sae.encoder_weights[d + k]).sum();

        assert!(dot_after.abs() < dot_before.abs(),
            "OrtSAE deve reduzir overlap: antes={:.4}, depois={:.4}", dot_before, dot_after);
    }

    #[test]
    fn test_cohens_d_identifies_causal_feature() {
        let f = 16;
        let sae = SAEEngine::new(8, f, 0.0);

        let pop_a: Vec<Vec<f32>> = (0..10).map(|_| {
            let mut v = vec![0.1f32; f]; v[5] = 5.0; v
        }).collect();
        let pop_b: Vec<Vec<f32>> = (0..10).map(|_| vec![0.1f32; f]).collect();

        let ranked = sae.rank_features_by_cohens_d(&pop_a, &pop_b);
        assert!(!ranked.is_empty());
        assert_eq!(ranked[0].0, 5, "Feature 5 deve ter maior |d|");
        assert!(ranked[0].1 > 0.0, "d deve ser positivo");

        for i in 1..ranked.len() {
            assert!(ranked[i-1].1.abs() >= ranked[i].1.abs(), "Ranking por |d| violado na posicao {}", i);
        }
    }

    #[test]
    fn test_cohens_d_empty_population() {
        let sae = SAEEngine::new(8, 16, 0.1);
        let empty: Vec<Vec<f32>> = vec![];
        let pop = vec![vec![1.0f32; 16]];
        assert!(sae.rank_features_by_cohens_d(&empty, &pop).is_empty());
        assert!(sae.rank_features_by_cohens_d(&pop, &empty).is_empty());
    }

    #[test]
    fn test_functional_mode_does_not_panic() {
        let mut sae = SAEEngine::new(8, 4, 0.0);
        sae.input_mode = SAEInputMode::Functional { target_dim: 16 };
        let big_input = vec![1.0f32; 24];
        let _ = sae.encode(&big_input);
    }
}
