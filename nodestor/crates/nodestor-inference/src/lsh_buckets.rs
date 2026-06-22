//! LSH Buckets — Locality-Sensitive Hashing para colapso O(1) da LM Head.
//!
//! Mapeia vetores similares ao mesmo bucket via hiperplanos aleatórios.
//! Pré-computado no boot. Na inferência, o hash do hidden state cai
//! diretamente no bucket correto → O(1).

use std::collections::HashMap;

/// Índice LSH para vocabulário do modelo.
pub struct LshVocabIndex {
    num_tables: usize,
    num_hyperplanes: usize,
    dim: usize,
    /// Hiperplanos aleatórios [num_tables × num_hyperplanes × dim]
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// bucket_hash → lista de token_ids
    buckets: Vec<HashMap<u64, Vec<usize>>>,
}

impl LshVocabIndex {
    /// Constrói as tabelas LSH a partir dos embeddings do vocabulário.
    /// `num_tables`: L tabelas (6-10). Mais tabelas = maior recall.
    /// `num_hyperplanes`: K bits por hash (12-16). Mais bits = buckets menores.
    pub fn build(embeddings: &[f32], vocab_size: usize, hidden_dim: usize,
                 num_tables: usize, num_hyperplanes: usize) -> Self {
        // Gera hiperplanos pseudo-aleatórios (determinísticos via seed)
        let mut hyperplanes = Vec::with_capacity(num_tables);
        for t in 0..num_tables {
            let mut table_planes = Vec::with_capacity(num_hyperplanes);
            for h in 0..num_hyperplanes {
                let plane: Vec<f32> = (0..hidden_dim).map(|d| {
                    // Hash determinístico para reproduzibilidade
                    let seed = (t * 10007 + h * 1009 + d * 101 + 7) as f64;
                    (seed.sin() * 43758.5453).fract() as f32
                }).collect();
                table_planes.push(plane);
            }
            hyperplanes.push(table_planes);
        }

        // Insere cada token do vocabulário nos buckets
        let mut buckets: Vec<HashMap<u64, Vec<usize>>> = (0..num_tables)
            .map(|_| HashMap::new()).collect();

        for token_id in 0..vocab_size {
            let start = token_id * hidden_dim;
            let end = (start + hidden_dim).min(embeddings.len());
            let vec = &embeddings[start..end];

            for (t, table_planes) in hyperplanes.iter().enumerate() {
                let hash = Self::compute_hash(vec, table_planes);
                buckets[t].entry(hash).or_default().push(token_id);
            }
        }

        Self { num_tables, num_hyperplanes, dim: hidden_dim, hyperplanes, buckets }
    }

    /// Dado um hidden state, retorna os candidatos no bucket (tipicamente 30-80 tokens).
    /// Complexidade: O(L) para L tabelas hash + O(bucket_size) para dedup.
    pub fn lookup(&self, query: &[f32]) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::new();
        let mut seen = vec![false; 0]; // lazy init

        for (t, table_planes) in self.hyperplanes.iter().enumerate() {
            let hash = Self::compute_hash(query, table_planes);
            if let Some(bucket) = self.buckets[t].get(&hash) {
                // Lazy-init do seen array no primeiro acesso
                if seen.is_empty() && !bucket.is_empty() {
                    let max_id = self.buckets.iter()
                        .flat_map(|b| b.values().flat_map(|v| v.iter()))
                        .max().copied().unwrap_or(0);
                    seen = vec![false; max_id + 1];
                }
                for &id in bucket {
                    if id < seen.len() && !seen[id] {
                        seen[id] = true;
                        candidates.push(id);
                    }
                }
            }
        }
        candidates
    }

    /// Calcula o hash LSH: bit i = sign(dot(query, hyperplane_i)).
    fn compute_hash(vec: &[f32], planes: &[Vec<f32>]) -> u64 {
        let mut hash = 0u64;
        for (i, plane) in planes.iter().enumerate() {
            let dot: f32 = vec.iter().zip(plane.iter())
                .map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
        }
        hash
    }

    /// Número médio de candidatos por lookup (para diagnóstico).
    pub fn avg_bucket_size(&self) -> f64 {
        let total: usize = self.buckets.iter()
            .flat_map(|b| b.values())
            .map(|v| v.len())
            .sum();
        let count = self.buckets.iter().map(|b| b.len()).sum::<usize>().max(1);
        total as f64 / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| ((i as f32) * 0.01).sin()).collect()
    }

    #[test]
    fn test_lsh_build_and_lookup() {
        let dim = 64;
        let vocab = 200;
        let emb = make_embeddings(vocab, dim);
        let index = LshVocabIndex::build(&emb, vocab, dim, 6, 12);
        let query = &emb[0..dim]; // token 0
        let candidates = index.lookup(query);
        assert!(!candidates.is_empty(), "LSH must return candidates");
        assert!(candidates.contains(&0), "Token 0 must be in its own bucket");
    }

    #[test]
    fn test_lsh_contains_correct_token() {
        let dim = 32;
        let vocab = 500;
        let emb = make_embeddings(vocab, dim);
        let index = LshVocabIndex::build(&emb, vocab, dim, 8, 10);

        // Para cada token de teste, verifica que está no seu próprio bucket
        for q in [0, 100, 250, 499] {
            let query = &emb[q * dim..(q + 1) * dim];
            let candidates = index.lookup(query);
            assert!(candidates.contains(&q),
                "Token {} must be in its own LSH bucket (got {} candidates)",
                q, candidates.len());
        }
    }

    #[test]
    fn test_lsh_avg_bucket_reasonable() {
        let dim = 64;
        let vocab = 1000;
        let emb = make_embeddings(vocab, dim);
        let index = LshVocabIndex::build(&emb, vocab, dim, 6, 12);
        let avg = index.avg_bucket_size();
        assert!(avg > 0.0 && avg < vocab as f64,
            "Avg bucket size {} must be between 0 and vocab_size", avg);
    }
}
