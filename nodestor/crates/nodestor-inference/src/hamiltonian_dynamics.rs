//! Hamiltonian Dynamics — Integrador Simplético para Rascunho Latente.
//!
//! Substitui o `eagle2_forward()` (drift linear acumulativo) por um
//! integrador Störmer-Verlet (Leapfrog) que preserva a energia semântica
//! ao longo de milhares de passos, eliminando o "drift" caótico.
//!
//! # Mecânica Hamiltoniana no Espaço Latente
//!
//! O "estado" é (q, p) onde:
//!   q = posição semântica (o hidden state)
//!   p = momento semântico (a "velocidade" do pensamento)
//!
//! A energia total H(q,p) = T(p) + V(q) é CONSERVADA pelo integrador
//! simplético, impedindo drift exponencial (Teorema de Liouville).
//!
//! # Integrador Störmer-Verlet (Leapfrog)
//!
//! O Leapfrog é simplético (preserva volume no espaço de fases):
//!   p_{n+1/2} = p_n     - (dt/2) * ∇V(q_n)
//!   q_{n+1}   = q_n     + dt * p_{n+1/2}
//!   p_{n+1}   = p_{n+1/2} - (dt/2) * ∇V(q_{n+1})
//!
//! Propriedade chave: |H(q_n, p_n) - H(q_0, p_0)| = O(dt²) ∀n.

/// Sistema Hamiltoniano para o espaço latente do LLM.
pub struct HamiltonianLatentDynamics {
    hidden_dim: usize,
    /// Passo temporal. Valores típicos: 0.01 - 0.1.
    /// Menor dt = maior precisão, mais passos necessários.
    dt: f32,
    /// Pesos do potencial V(q). Interpretação: "curvatura" do espaço semântico.
    /// V(q) = 0.5 * sum(potential_weights[i] * q[i]²) (potencial harmônico).
    potential_weights: Vec<f32>,
    /// Energia máxima permitida de desvio antes de forçar fallback.
    energy_budget: f32,
}

impl HamiltonianLatentDynamics {
    /// Cria um novo sistema Hamiltoniano para o espaço latente.
    ///
    /// `hidden_dim`: dimensão do espaço (ex: 4096 para Llama)
    /// `dt`: passo temporal (0.01 para alta precisão, 0.1 para velocidade)
    /// `energy_budget`: |ΔH| máximo tolerado antes de forçar fallback
    pub fn new(hidden_dim: usize, dt: f32, energy_budget: f32) -> Self {
        // Potencial harmônico uniforme (todas dimensões igualmente "rígidas")
        // Em produção, esses pesos seriam aprendidos do modelo.
        let potential_weights = vec![1.0; hidden_dim];

        Self { hidden_dim, dt, potential_weights, energy_budget }
    }

    /// Cria com pesos de potencial customizados (aprendidos do modelo).
    pub fn with_potential(hidden_dim: usize, dt: f32, energy_budget: f32,
                          potential_weights: Vec<f32>) -> Self {
        assert_eq!(potential_weights.len(), hidden_dim);
        Self { hidden_dim, dt, potential_weights, energy_budget }
    }

    /// Calcula a energia cinética T(p) = ||p||² / 2.
    pub fn kinetic_energy(&self, p: &[f32]) -> f32 {
        p.iter().map(|&pi| pi * pi).sum::<f32>() * 0.5
    }

    /// Calcula a energia potencial V(q) = 0.5 * Σ w_i * q_i².
    /// Potencial harmônico: a "paisagem energética" do espaço semântico.
    pub fn potential_energy(&self, q: &[f32]) -> f32 {
        q.iter().zip(self.potential_weights.iter())
            .map(|(&qi, &wi)| wi * qi * qi)
            .sum::<f32>() * 0.5
    }

    /// Calcula a energia total H(q, p) = T(p) + V(q).
    /// Se H cresce significativamente, o sistema está divergindo.
    pub fn total_energy(&self, q: &[f32], p: &[f32]) -> f32 {
        self.kinetic_energy(p) + self.potential_energy(q)
    }

    /// Calcula o gradiente do potencial: ∇V(q) = [w_0*q_0, w_1*q_1, ...].
    fn grad_potential(&self, q: &[f32]) -> Vec<f32> {
        q.iter().zip(self.potential_weights.iter())
            .map(|(&qi, &wi)| wi * qi)
            .collect()
    }

    /// Integrador Störmer-Verlet (Leapfrog) — um passo.
    ///
    /// Preserva o volume no espaço de fases (Teorema de Liouville).
    /// Erro de energia: O(dt²) por passo, NÃO acumula ao longo do tempo.
    pub fn leapfrog_step(&self, q: &mut [f32], p: &mut [f32]) {
        let half_dt = self.dt * 0.5;
        let grad_v = self.grad_potential(q);

        // Meio passo do momento: p_{n+1/2} = p_n - (dt/2) * ∇V(q_n)
        for i in 0..self.hidden_dim {
            p[i] -= half_dt * grad_v[i];
        }

        // Passo completo da posição: q_{n+1} = q_n + dt * p_{n+1/2}
        for i in 0..self.hidden_dim {
            q[i] += self.dt * p[i];
        }

        // Segundo meio passo do momento: p_{n+1} = p_{n+1/2} - (dt/2) * ∇V(q_{n+1})
        let grad_v_new = self.grad_potential(q);
        for i in 0..self.hidden_dim {
            p[i] -= half_dt * grad_v_new[i];
        }
    }

    /// Gera K estados latentes com conservação de energia garantida.
    ///
    /// Retorna:
    /// - `states`: K hidden states evoluídos no espaço latente
    /// - `energy_drift`: |H_final - H_initial| (deve ser ~0 para integradores simpléticos)
    /// - `ok`: true se a energia se manteve dentro do budget
    pub fn evolve(&self, seed_q: &[f32], seed_p: &[f32], steps: usize)
        -> (Vec<Vec<f32>>, f32, bool) {
        let mut q = seed_q.to_vec();
        let mut p = seed_p.to_vec();
        let h_initial = self.total_energy(&q, &p);

        let mut states = Vec::with_capacity(steps);

        for _ in 0..steps {
            self.leapfrog_step(&mut q, &mut p);
            states.push(q.clone());
        }

        let h_final = self.total_energy(&q, &p);
        let drift = (h_final - h_initial).abs();
        let ok = drift < self.energy_budget;

        (states, drift, ok)
    }

    /// Inicializa o momento semântico a partir do hidden state.
    /// Usa uma heurística: p_0 = α * q_0 onde α é pequeno,
    /// representando um "empurrão" na direção do pensamento atual.
    pub fn init_momentum(&self, seed_q: &[f32], alpha: f32) -> Vec<f32> {
        seed_q.iter().map(|&qi| qi * alpha).collect()
    }

    /// Retorna o energy budget configurado.
    pub fn energy_budget(&self) -> f32 {
        self.energy_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conservation_2000_steps() {
        let dim = 128;
        let dynamics = HamiltonianLatentDynamics::new(dim, 0.01, 1.0);

        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let p: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.07).cos() * 0.3).collect();

        let h0 = dynamics.total_energy(&q, &p);
        let (_, drift, ok) = dynamics.evolve(&q, &p, 2000);

        assert!(drift < 1e-2,
            "Energy drift after 2000 steps = {:.6}, must be < 0.01", drift);
        assert!(ok, "Energy must stay within budget");
    }

    #[test]
    fn test_leapfrog_reversibility() {
        let dim = 64;
        let dynamics = HamiltonianLatentDynamics::new(dim, 0.01, 1.0);

        let q0: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
        let p0: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.03).cos()).collect();

        // Forward 100 steps
        let mut q = q0.clone();
        let mut p = p0.clone();
        for _ in 0..100 { dynamics.leapfrog_step(&mut q, &mut p); }

        // Reverse: negate momentum and run 100 more steps
        for pi in p.iter_mut() { *pi = -*pi; }
        for _ in 0..100 { dynamics.leapfrog_step(&mut q, &mut p); }
        for pi in p.iter_mut() { *pi = -*pi; }

        // Should return close to initial state
        let max_diff: f32 = q.iter().zip(q0.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3,
            "Leapfrog should be reversible, max_diff = {:.6}", max_diff);
    }

    #[test]
    fn test_hamiltonian_vs_linear_drift() {
        let dim = 128;
        let dynamics = HamiltonianLatentDynamics::new(dim, 0.05, 10.0);

        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let p = dynamics.init_momentum(&q, 0.01);

        let (states, _, _) = dynamics.evolve(&q, &p, 500);

        // Cosine similarity between initial and final state
        let cos_sim = cosine_sim(&q, states.last().unwrap());
        assert!(cos_sim > 0.5,
            "Hamiltonian should maintain similarity, got cos_sim = {:.4}", cos_sim);
    }

    #[test]
    fn test_energy_budget_exceeded_returns_false() {
        let dim = 64;
        // Very tight budget with large dt → should exceed
        let dynamics = HamiltonianLatentDynamics::new(dim, 1.0, 1e-10);

        let q: Vec<f32> = vec![1.0; dim];
        let p: Vec<f32> = vec![1.0; dim];

        let (_, _, ok) = dynamics.evolve(&q, &p, 100);
        assert!(!ok, "Large dt with tight budget should trigger energy violation");
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 { 0.0 } else { dot / (na * nb) }
    }
}
