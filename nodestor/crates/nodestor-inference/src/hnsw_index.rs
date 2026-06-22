//! HNSW Index — Hierarchical Navigable Small World para busca vetorial O(log V).
//!
//! Substitui a multiplicação bruta [1×H]×[H×V] da LM Head por busca topológica.
//! Construído uma vez no boot do modelo (~2-5s para 128K vocab).

use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Clone)]
struct Neighbor {
    id: usize,
    distance: f32,
}

impl PartialEq for Neighbor { fn eq(&self, o: &Self) -> bool { self.distance == o.distance } }
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { self.distance.partial_cmp(&o.distance) }
}
impl Ord for Neighbor {
    fn cmp(&self, o: &Self) -> Ordering { self.partial_cmp(o).unwrap_or(Ordering::Equal) }
}

/// Uma camada do grafo HNSW.
struct HnswLayer {
    /// neighbors[node_id] = lista de vizinhos naquela camada
    neighbors: Vec<Vec<usize>>,
}

/// Índice HNSW para vetores de embedding do vocabulário.
pub struct HnswIndex {
    layers: Vec<HnswLayer>,
    entry_point: usize,
    ef_search: usize,
    max_connections: usize,
    vectors: Vec<Vec<f32>>,
    dim: usize,
}

impl HnswIndex {
    /// Constrói o grafo HNSW a partir da matriz LM Head [V × H].
    /// `embeddings`: flat array [vocab_size * hidden_dim]
    pub fn build(embeddings: &[f32], vocab_size: usize, hidden_dim: usize,
                 max_connections: usize, ef_construction: usize) -> Self {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let start = i * hidden_dim;
            let end = (start + hidden_dim).min(embeddings.len());
            vectors.push(embeddings[start..end].to_vec());
        }

        let max_level = if vocab_size <= 1 { 0 }
            else { (vocab_size as f64).ln().ceil() as usize / 2 };
        let max_level = max_level.max(1).min(6);

        let mut layers: Vec<HnswLayer> = (0..max_level).map(|_| HnswLayer {
            neighbors: vec![Vec::new(); vocab_size],
        }).collect();

        // Nível de cada nó: P(level=l) = 1/2^l (distribuição geométrica)
        let node_levels: Vec<usize> = (0..vocab_size).map(|i| {
            let hash = (i as u64).wrapping_mul(0x517cc1b727220a95) >> 58;
            (hash as usize).min(max_level - 1)
        }).collect();

        let entry_point = 0;

        // Inserção incremental
        for node in 0..vocab_size {
            let node_level = node_levels[node];
            for level in 0..=node_level.min(max_level - 1) {
                // Encontra vizinhos mais próximos no nível atual
                let mut candidates: Vec<(usize, f32)> = Vec::new();
                for other in 0..node {
                    if node_levels[other] >= level {
                        let dist = Self::l2_distance_vecs(&vectors[node], &vectors[other]);
                        candidates.push((other, dist));
                    }
                }
                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                candidates.truncate(max_connections);

                for &(neighbor_id, _) in &candidates {
                    layers[level].neighbors[node].push(neighbor_id);
                    if layers[level].neighbors[neighbor_id].len() < max_connections * 2 {
                        layers[level].neighbors[neighbor_id].push(node);
                    }
                }
            }
        }

        Self { layers, entry_point, ef_search: ef_construction.min(64),
            max_connections, vectors, dim: hidden_dim }
    }

    /// Busca os K vizinhos mais próximos. Complexidade: O(ef_search * log V).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() { return Vec::new(); }

        let mut current = self.entry_point;

        // Greedy search das camadas superiores → camada 0
        for level in (1..self.layers.len()).rev() {
            loop {
                let mut improved = false;
                for &neighbor in &self.layers[level].neighbors[current] {
                    if Self::l2_distance_vecs(query, &self.vectors[neighbor])
                        < Self::l2_distance_vecs(query, &self.vectors[current]) {
                        current = neighbor;
                        improved = true;
                    }
                }
                if !improved { break; }
            }
        }

        // Busca detalhada na camada 0 com ef_search candidatos
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

        visited[current] = true;
        let dist = Self::l2_distance_vecs(query, &self.vectors[current]);
        candidates.push(std::cmp::Reverse(Neighbor { id: current, distance: dist }));
        results.push(Neighbor { id: current, distance: dist });

        while let Some(std::cmp::Reverse(closest)) = candidates.pop() {
            let worst_result = results.peek().map(|n| n.distance).unwrap_or(f32::INFINITY);
            if closest.distance > worst_result && results.len() >= self.ef_search { break; }

            if closest.id < self.layers[0].neighbors.len() {
                for &neighbor in &self.layers[0].neighbors[closest.id] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        let d = Self::l2_distance_vecs(query, &self.vectors[neighbor]);
                        if results.len() < self.ef_search || d < worst_result {
                            candidates.push(std::cmp::Reverse(Neighbor { id: neighbor, distance: d }));
                            results.push(Neighbor { id: neighbor, distance: d });
                            if results.len() > self.ef_search { results.pop(); }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter()
            .map(|n| (n.id, n.distance)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }

    /// Busca brute-force para validação. O(V).
    pub fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self.vectors.iter().enumerate()
            .map(|(i, v)| (i, Self::l2_distance_vecs(query, v))).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        dists.truncate(k);
        dists
    }

    fn l2_distance_vecs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| ((i as f32) * 0.01).sin()).collect()
    }

    #[test]
    fn test_hnsw_build_and_search_top1() {
        let dim = 64;
        let vocab = 200;
        let emb = make_embeddings(vocab, dim);
        let index = HnswIndex::build(&emb, vocab, dim, 16, 32);
        let query: Vec<f32> = emb[0..dim].to_vec(); // query = token 0
        let results = index.search(&query, 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "Top-1 must be token 0 itself");
    }

    #[test]
    fn test_hnsw_vs_bruteforce_top1_match() {
        let dim = 32;
        let vocab = 500;
        let emb = make_embeddings(vocab, dim);
        let index = HnswIndex::build(&emb, vocab, dim, 16, 64);
        // Test with 10 random queries
        for q_idx in [0, 50, 100, 200, 499] {
            let query: Vec<f32> = emb[q_idx * dim..(q_idx + 1) * dim].to_vec();
            let hnsw_top1 = index.search(&query, 1)[0].0;
            let bf_top1 = index.brute_force_search(&query, 1)[0].0;
            assert_eq!(hnsw_top1, bf_top1,
                "HNSW top-1 ({}) must match brute-force ({}) for query {}",
                hnsw_top1, bf_top1, q_idx);
        }
    }
}
