//! D10 — Latent Jump: Saltos no Espaço Latente
//!
//! Permite ao Dreaming Engine "saltar" de uma região conhecida
//! para uma região inexplorada do espaço semântico.
//! Implementa perturbação Gaussiana + Annealing para controle de alcance.

/// Resultado de um salto latente
#[derive(Debug, Clone)]
pub struct JumpResult {
    /// Ponto de partida
    pub origin: Vec<f32>,
    /// Ponto de chegada após o salto
    pub destination: Vec<f32>,
    /// Distância percorrida
    pub distance: f32,
    /// Temperatura usada (controla alcance)
    pub temperature: f32,
}

/// Motor de saltos no espaço latente
pub struct LatentJump {
    /// Semente determinística (reprodutibilidade de saltos)
    seed: u64,
    /// Escala base dos saltos (amplificada pela temperatura)
    pub base_scale: f32,
}

impl LatentJump {
    pub fn new(seed: u64, base_scale: f32) -> Self {
        Self { seed, base_scale }
    }

    /// Executa um único salto no espaço latente
    /// Temperatura alta → salto longo e caótico
    /// Temperatura baixa → perturbação mínima, exploração local
    pub fn jump(&mut self, embedding: &[f32], temperature: f32) -> JumpResult {
        let scale = self.base_scale * temperature;
        let mut destination = Vec::with_capacity(embedding.len());

        for &dim_val in embedding {
            // Perturbação Gaussiana via Box-Muller (aproximado com LCG)
            let r1 = self.rand_f32();
            let r2 = self.rand_f32();
            let gaussian = (-2.0 * r1.ln().max(-50.0)).sqrt() * (2.0 * std::f32::consts::PI * r2).cos();
            destination.push(dim_val + scale * gaussian);
        }

        let dist = euclidean_distance(embedding, &destination);
        JumpResult {
            origin: embedding.to_vec(),
            destination,
            distance: dist,
            temperature,
        }
    }

    /// Executa uma cadeia de saltos (random walk guiado por temperatura)
    pub fn jump_chain(&mut self, start: &[f32], n_steps: usize, temperature: f32) -> Vec<JumpResult> {
        let mut results = Vec::with_capacity(n_steps);
        let mut current = start.to_vec();

        for _ in 0..n_steps {
            let result = self.jump(&current, temperature);
            current = result.destination.clone();
            results.push(result);
        }

        results
    }

    /// Salto direcionado: move-se na direção de um alvo com perturbação
    pub fn directed_jump(
        &mut self,
        origin: &[f32],
        target: &[f32],
        temperature: f32,
    ) -> JumpResult {
        let scale = self.base_scale * temperature;
        let len = origin.len().min(target.len());
        let mut destination = Vec::with_capacity(len);

        for i in 0..len {
            // Move-se 70% para o alvo + 30% de ruído
            let toward_target = origin[i] + 0.7 * (target[i] - origin[i]);
            let noise = (self.rand_f32() - 0.5) * 2.0 * scale * 0.3;
            destination.push(toward_target + noise);
        }

        let dist = euclidean_distance(origin, &destination);
        JumpResult {
            origin: origin.to_vec(),
            destination,
            distance: dist,
            temperature,
        }
    }

    /// Normaliza um embedding (para manter escala após saltos)
    pub fn normalize(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return embedding.to_vec();
        }
        embedding.iter().map(|x| x / norm).collect()
    }

    fn rand_f32(&mut self) -> f32 {
        self.seed = self.seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (self.seed >> 33) as f32 / u32::MAX as f32;
        // Evita 0.0 exato (ln(0) = -inf)
        (v + 1e-10).min(1.0)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_jump() {
        let mut lj = LatentJump::new(42, 0.1);
        let start = vec![0.0f32; 16];
        let result = lj.jump(&start, 1.0);
        assert_eq!(result.origin.len(), 16);
        assert_eq!(result.destination.len(), 16);
        assert!(result.distance > 0.0, "Deve se mover do ponto de origem");
    }

    #[test]
    fn test_jump_chain() {
        let mut lj = LatentJump::new(123, 0.05);
        let start = vec![0.5f32; 8];
        let chain = lj.jump_chain(&start, 10, 0.5);
        assert_eq!(chain.len(), 10);
        // Cada resultado deve conectar ao anterior
        for i in 1..chain.len() {
            // Origin do salto i deve ser destination do salto i-1
            let prev_dest = &chain[i - 1].destination;
            let curr_orig = &chain[i].origin;
            for (a, b) in prev_dest.iter().zip(curr_orig.iter()) {
                assert!((a - b).abs() < 1e-5, "Cadeia deve ser contígua");
            }
        }
    }
}
