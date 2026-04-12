//! D7 — Stigmergic Swarm: Feromônios Digitais para Inteligência de Enxame
//!
//! Formigas não falam umas com as outras. Elas deixam feromônio no chão.
//! Cada nó Davi deposita "feromônio" no LanceDB que outros nós "cheiram".
//! O consenso emerge do campo — Zero comunicação direta. O(1) por nó.

use std::collections::HashMap;

/// Um feromônio digital depositado por um nó Davi
#[derive(Debug, Clone)]
pub struct DigitalPheromone {
    /// ID do insight marcado
    pub insight_id: u64,
    /// Posição no espaço latente
    pub position: Vec<f32>,
    /// Intensidade atual do feromônio
    pub intensity: f32,
    /// Taxa de evaporação por epoch (ex: 0.05 = perde 5% por epoch)
    pub evaporation_rate: f32,
    /// Qual nó depositou este feromônio
    pub origin_node: u64,
    /// Hash de autenticidade (HMAC-like simples)
    pub signature: u64,
}

impl DigitalPheromone {
    fn compute_signature(insight_id: u64, node: u64, intensity: f32) -> u64 {
        let mut h = 14695981039346656037u64;
        h ^= insight_id;
        h = h.wrapping_mul(1099511628211);
        h ^= node;
        h = h.wrapping_mul(1099511628211);
        h ^= intensity.to_bits() as u64;
        h
    }
}

/// Uma "auto-estrada de insight": caminho intensamente marcado
#[derive(Debug, Clone)]
pub struct InsightHighway {
    /// Centróide do caminho
    pub centroid: Vec<f32>,
    /// Intensidade total do caminho
    pub total_intensity: f32,
    /// Número de feromônios que formam esta auto-estrada
    pub pheromone_count: usize,
    /// Domínio predominante
    pub dominant_domain: Option<String>,
}

/// O Swarm Estigmérgico: campo de feromônios distribuído
pub struct StigmergicSwarm {
    /// Todos os feromônios no campo
    pub pheromone_field: Vec<DigitalPheromone>,
    /// Limiar de ativação: feromônio precisa ter intensidade mínima
    pub activation_threshold: f32,
    /// Taxa global de evaporação (aplicada a todos por epoch)
    pub global_evaporation: f32,
    /// Raio de busca para sensado
    pub sense_radius: f32,
    /// Epoch atual do swarm
    pub epoch: u64,
}

impl StigmergicSwarm {
    pub fn new(activation_threshold: f32, global_evaporation: f32) -> Self {
        Self {
            pheromone_field: Vec::new(),
            activation_threshold,
            global_evaporation,
            sense_radius: 1.0,
            epoch: 0,
        }
    }

    /// Deposita feromônio: "Achei algo valioso aqui"
    pub fn deposit(&mut self, insight_id: u64, position: Vec<f32>, intensity: f32, node_id: u64) {
        let sig = DigitalPheromone::compute_signature(insight_id, node_id, intensity);

        // Se já existe feromônio nesta posição, reforça em vez de criar novo
        let existing = self.pheromone_field.iter_mut()
            .find(|p| p.insight_id == insight_id && p.origin_node == node_id);

        if let Some(ph) = existing {
            ph.intensity = (ph.intensity + intensity).min(10.0);
            ph.signature = sig;
        } else {
            self.pheromone_field.push(DigitalPheromone {
                insight_id,
                position,
                intensity,
                evaporation_rate: self.global_evaporation,
                origin_node: node_id,
                signature: sig,
            });
        }
    }

    /// Sente feromônios ao redor de uma posição
    pub fn sense(&self, current_pos: &[f32], radius: f32) -> Vec<&DigitalPheromone> {
        self.pheromone_field.iter()
            .filter(|p| {
                p.intensity >= self.activation_threshold
                    && euclidean_distance(current_pos, &p.position) <= radius
            })
            .collect()
    }

    /// Aplica evaporação: feromônios velhos e não-reforçados desaparecem
    pub fn evaporate(&mut self, epochs: u64) {
        self.epoch += epochs;
        self.pheromone_field.retain_mut(|p| {
            p.intensity *= (1.0 - p.evaporation_rate).powi(epochs as i32);
            p.intensity >= 0.001 // Remove feromônios praticamente evaporados
        });
    }

    /// Calcula as "auto-estradas de insight": regiões densamente marcadas
    pub fn compute_highways(&self) -> Vec<InsightHighway> {
        if self.pheromone_field.is_empty() {
            return Vec::new();
        }

        // Clustering simples: agrupa feromônios próximos
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; self.pheromone_field.len()];

        for i in 0..self.pheromone_field.len() {
            if assigned[i] { continue; }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..self.pheromone_field.len() {
                if assigned[j] { continue; }
                let d = euclidean_distance(
                    &self.pheromone_field[i].position,
                    &self.pheromone_field[j].position,
                );
                if d <= self.sense_radius {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }

        // Converte clusters em highways
        clusters.into_iter()
            .filter(|c| c.len() >= 2) // Pelo menos 2 feromônios para ser uma highway
            .map(|indices| {
                let phs: Vec<_> = indices.iter().map(|&i| &self.pheromone_field[i]).collect();
                let total_intensity: f32 = phs.iter().map(|p| p.intensity).sum();
                let dim = phs[0].position.len();
                let centroid: Vec<f32> = (0..dim)
                    .map(|d| phs.iter().map(|p| p.position[d]).sum::<f32>() / phs.len() as f32)
                    .collect();
                InsightHighway {
                    centroid,
                    total_intensity,
                    pheromone_count: phs.len(),
                    dominant_domain: None,
                }
            })
            .collect()
    }

    /// Total de feromônios ativos
    pub fn active_pheromones(&self) -> usize {
        self.pheromone_field.iter()
            .filter(|p| p.intensity >= self.activation_threshold)
            .count()
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
    fn test_pheromone_deposit_and_sense() {
        let mut swarm = StigmergicSwarm::new(0.1, 0.05);

        swarm.deposit(1, vec![0.0, 0.0], 1.0, 42);
        swarm.deposit(2, vec![0.1, 0.1], 0.8, 43);
        swarm.deposit(3, vec![5.0, 5.0], 1.0, 44); // Longe

        assert_eq!(swarm.pheromone_field.len(), 3);

        // Sente ao redor de [0,0] com raio 1.0
        let sensed = swarm.sense(&[0.0, 0.0], 1.0);
        assert_eq!(sensed.len(), 2, "Deve sentir 2 feromônios próximos");
        assert!(sensed.iter().all(|p| p.intensity >= 0.1));
    }

    #[test]
    fn test_evaporation_removes_weak_pheromones() {
        let mut swarm = StigmergicSwarm::new(0.1, 0.5); // 50% por epoch
        swarm.deposit(1, vec![0.0, 0.0], 0.01, 1); // Muito fraco — deve evaporar

        swarm.evaporate(5); // 5 epochs de 50%: 0.01 * 0.5^5 = 0.0003125 < 0.001
        assert_eq!(swarm.pheromone_field.len(), 0, "Feromônio fraco deve ser removido");
    }

    #[test]
    fn test_highways_emerge() {
        let mut swarm = StigmergicSwarm::new(0.1, 0.05);
        swarm.sense_radius = 0.5;

        // Cluster A: 3 feromônios próximos
        for i in 0..3 {
            swarm.deposit(i, vec![i as f32 * 0.1, 0.0], 1.0, i as u64);
        }
        // Cluster B: isolado
        swarm.deposit(10, vec![10.0, 10.0], 1.0, 10);

        let highways = swarm.compute_highways();
        assert!(!highways.is_empty(), "Deve detectar pelo menos uma highway");
        // A highway do cluster A deve ter 3 feromônios
        let biggest = highways.iter().max_by_key(|h| h.pheromone_count).unwrap();
        assert!(biggest.pheromone_count >= 2);
    }

    #[test]
    fn test_reinforcement_same_position() {
        let mut swarm = StigmergicSwarm::new(0.1, 0.05);
        swarm.deposit(1, vec![0.0, 0.0], 1.0, 42);
        swarm.deposit(1, vec![0.0, 0.0], 1.0, 42); // Mesmo nó, mesmo insight

        // Deve reforçar em vez de duplicar
        assert_eq!(swarm.pheromone_field.len(), 1);
        assert!(swarm.pheromone_field[0].intensity > 1.0);
    }
}
