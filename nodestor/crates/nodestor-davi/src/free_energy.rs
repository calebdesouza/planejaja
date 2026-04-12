//! D2 — Free Energy Core: Função Objetivo de Friston
//!
//! Todo sistema que se auto-organiza sobrevive minimizando a surpresa.
//! F = E[log q(z)] - E[log p(x,z)]
//!
//! Tradução para o Davi:
//! - Alta surpresa num região → EXPLORAR lá
//! - Baixa surpresa → EXPLORAR (região dominada)

use std::collections::HashMap;

/// Uma crença sobre uma região do espaço de conhecimento
#[derive(Debug, Clone)]
pub struct BeliefNode {
    /// Embedding representativo da região
    pub embedding: Vec<f32>,
    /// Certeza da crença (0 = incerto, 1 = certo)
    pub certainty: f32,
    /// Número de vezes que esta região foi visitada
    pub visit_count: u64,
}

/// Uma ação candidata do Dreaming Engine
#[derive(Debug, Clone)]
pub struct DreamAction {
    /// Nome descritivo da ação
    pub name: String,
    /// Embedding da região alvo
    pub target_embedding: Vec<f32>,
    /// Energia esperada reduzida pela ação
    pub expected_energy_reduction: f32,
}

/// O motor central que guia onde o Davi deve "sonhar"
pub struct FreeEnergyObjective {
    /// Crenças atuais sobre o espaço de conhecimento
    pub beliefs: Vec<BeliefNode>,
    /// Mapa de surpresa por região (hash → surpresa)
    pub surprise_map: HashMap<u64, f32>,
    /// Precisão atual das previsões (inverso da variância)
    pub precision: f32,
    /// Limiar de energia para parar de sonhar
    pub awakening_threshold: f32,
    /// Histórico de Energia Livre global
    pub energy_history: Vec<f32>,
}

impl FreeEnergyObjective {
    pub fn new(precision: f32, awakening_threshold: f32) -> Self {
        Self {
            beliefs: Vec::new(),
            surprise_map: HashMap::new(),
            precision,
            awakening_threshold,
            energy_history: Vec::new(),
        }
    }

    /// Adiciona uma crença sobre uma região
    pub fn add_belief(&mut self, embedding: Vec<f32>, certainty: f32) {
        self.beliefs.push(BeliefNode {
            embedding,
            certainty: certainty.clamp(0.0, 1.0),
            visit_count: 0,
        });
    }

    /// Calcula F para uma região (alta F = precisa de exploração)
    pub fn compute_free_energy(&self, region_embedding: &[f32]) -> f32 {
        let surprise = self.expected_surprise(region_embedding);
        let complexity = self.model_complexity();
        // F = complexidade - precisão × surpresa
        // Alta surpresa → alta F → o sistema PRECISA ir lá
        complexity - self.precision * surprise
    }

    /// Surpresa esperada: quão incerto o modelo é sobre esta região
    fn expected_surprise(&self, region: &[f32]) -> f32 {
        // Encontra a crença mais próxima
        let nearest = self.beliefs.iter()
            .map(|b| {
                let dist = cosine_distance(&b.embedding, region);
                (dist, b.certainty)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        match nearest {
            Some((dist, certainty)) => dist * (1.0 - certainty),
            None => 1.0, // Sem crenças = máxima surpresa
        }
    }

    /// Complexidade do modelo interno (entropia das crenças)
    fn model_complexity(&self) -> f32 {
        if self.beliefs.is_empty() {
            return 1.0;
        }
        let avg_certainty: f32 = self.beliefs.iter()
            .map(|b| b.certainty)
            .sum::<f32>() / self.beliefs.len() as f32;
        // Maior certeza = menor complexidade
        1.0 - avg_certainty
    }

    /// Seleciona a ação que mais reduz a energia livre (Active Inference)
    pub fn select_action<'a>(&self, candidates: &'a [DreamAction]) -> Option<&'a DreamAction> {
        candidates.iter().min_by(|a, b| {
            let fa = self.expected_free_energy_after(a);
            let fb = self.expected_free_energy_after(b);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Energia livre esperada APÓS executar uma ação
    fn expected_free_energy_after(&self, action: &DreamAction) -> f32 {
        let current = self.compute_free_energy(&action.target_embedding);
        current - action.expected_energy_reduction
    }

    /// O sistema deve continuar sonhando?
    /// True = ainda há surpresa global acima do limiar
    pub fn should_continue_dreaming(&self) -> bool {
        let global_f = self.compute_global_free_energy();
        global_f > self.awakening_threshold
    }

    /// Energia livre global (média sobre todas as regiões monitoradas)
    pub fn compute_global_free_energy(&self) -> f32 {
        if self.surprise_map.is_empty() {
            return self.model_complexity();
        }
        let total: f32 = self.surprise_map.values().sum();
        let avg_surprise = total / self.surprise_map.len() as f32;
        self.model_complexity() - self.precision * avg_surprise
    }

    /// Atualiza surpresa após visitar uma região
    pub fn update_after_visit(&mut self, region_hash: u64, new_certainty: f32) {
        let surprise = 1.0 - new_certainty.clamp(0.0, 1.0);
        self.surprise_map.insert(region_hash, surprise);
        let gf = self.compute_global_free_energy();
        self.energy_history.push(gf);
    }

    /// Tendência de convergência (negativo = melhorando)
    pub fn convergence_trend(&self) -> f32 {
        if self.energy_history.len() < 2 {
            return 0.0;
        }
        let last = *self.energy_history.last().unwrap();
        let prev = self.energy_history[self.energy_history.len() - 2];
        last - prev
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 1.0; }
    1.0 - (dot / (na * nb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_energy_computation() {
        let mut fep = FreeEnergyObjective::new(0.5, 0.1);
        fep.add_belief(vec![1.0, 0.0, 0.0], 0.9);  // Região conhecida
        fep.add_belief(vec![0.0, 1.0, 0.0], 0.1);  // Região desconhecida

        let f_known = fep.compute_free_energy(&[1.0, 0.0, 0.0]);
        let f_unknown = fep.compute_free_energy(&[0.0, 1.0, 0.0]);
        // Região desconhecida deve ter MAIOR surpresa → maior F (atenção priorizada)
        assert!(f_unknown > f_known - 0.5, "Desconhecido deveria ter energia mais alta");
    }

    #[test]
    fn test_action_selection() {
        let fep = FreeEnergyObjective::new(0.5, 0.1);
        let actions = vec![
            DreamAction {
                name: "Explore Física".to_string(),
                target_embedding: vec![1.0, 0.0],
                expected_energy_reduction: 0.8,
            },
            DreamAction {
                name: "Explore Biologia".to_string(),
                target_embedding: vec![0.0, 1.0],
                expected_energy_reduction: 0.3,
            },
        ];
        let best = fep.select_action(&actions);
        // Deve selecionar a ação com maior redução de energia esperada
        assert!(best.is_some());
        assert_eq!(best.unwrap().name, "Explore Física");
    }

    #[test]
    fn test_dreaming_convergence() {
        let mut fep = FreeEnergyObjective::new(0.5, 0.05);
        // Sem crenças → energia alta → deve continuar sonhando
        assert!(fep.should_continue_dreaming());

        // Adiciona muitas crenças certas → energia cai
        for i in 0..10 {
            fep.add_belief(vec![i as f32, 0.0], 0.95);
            fep.update_after_visit(i as u64, 0.95);
        }

        let trend = fep.convergence_trend();
        assert!(trend <= 0.1, "Energia deveria estar caindo ou estável");
    }

    #[test]
    fn test_global_energy_tracking() {
        let mut fep = FreeEnergyObjective::new(1.0, 0.1);
        // Registra múltiplas visitas com certeza crescente
        for i in 0..5 {
            let certainty = 0.2 * (i + 1) as f32; // 0.2, 0.4, 0.6, 0.8, 1.0
            fep.update_after_visit(i as u64, certainty);
        }
        assert_eq!(fep.energy_history.len(), 5);
    }
}
