use std::collections::HashMap;
use crate::buffer_pool::PooledBuffer;

/// Tracking das estatísticas do prefetcher especulativo.
#[derive(Debug, Default)]
pub struct SpecStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Cache LRU para grupos pré-carregados, operando na VRAM.
pub struct SpeculativeCache {
    cache: HashMap<usize, Vec<PooledBuffer>>,
    max_cached_groups: usize,
    pub stats: SpecStats,
}

impl SpeculativeCache {
    /// Inicializa com uma capacidade limite dinâmica de grupos alocados na GPU.
    pub fn new(max_cached_groups: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_cached_groups,
            stats: SpecStats::default(),
        }
    }

    /// Tenta adquirir os buffers já carregados para um group index.
    pub fn try_get(&mut self, group_idx: usize) -> Option<Vec<PooledBuffer>> {
        if let Some(bufs) = self.cache.remove(&group_idx) {
            self.stats.hits += 1;
            Some(bufs)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insere o grupo no cache. Se já estamos no limite `max_cached_groups`,
    /// remove a inserção causalmente mais antiga (menor `group_idx`).
    pub fn insert(&mut self, group_idx: usize, bufs: Vec<PooledBuffer>) {
        if self.cache.len() >= self.max_cached_groups {
            // A heurística causal é que grupos com ID menor foram processados primeiro,
            // ou seja, estão no passado e não serão visitados de novo na mesma geração iterativa.
            if let Some(&oldest) = self.cache.keys().min() {
                self.cache.remove(&oldest);
                self.stats.evictions += 1;
            }
        }
        self.cache.insert(group_idx, bufs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_cache_hit_miss_eviction() {
        let mut cache = SpeculativeCache::new(3);

        // Insere 3 grupos simulados vazios
        cache.insert(0, vec![]);
        cache.insert(1, vec![]);
        cache.insert(2, vec![]);

        // Hit no grupo 1
        assert!(cache.try_get(1).is_some());
        assert_eq!(cache.stats.hits, 1);

        // Miss no grupo 5
        assert!(cache.try_get(5).is_none());
        assert_eq!(cache.stats.misses, 1);

        // Ao inserir o grupo 3, deveria evictar o '0' como o mais antigo remanescente.
        cache.insert(3, vec![]);
        assert_eq!(cache.stats.evictions, 1);
        assert!(cache.try_get(0).is_none()); // Evictou
    }
}
