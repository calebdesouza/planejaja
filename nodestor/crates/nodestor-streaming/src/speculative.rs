use std::collections::{HashMap, VecDeque};
use crate::buffer_pool::PooledBuffer;

/// Tracking das estatísticas do prefetcher especulativo.
#[derive(Debug, Default)]
pub struct SpecStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Cache LRU para grupos e tensores pré-carregados, operando na VRAM / RAM.
pub struct SpeculativeCache {
    /// Lookup por índice do grupo (usado no pipeline assíncrono).
    pub by_group: HashMap<usize, Vec<PooledBuffer>>,
    
    /// Lookup direto por nome (usado pelo ApexOrchestrator).
    pub by_name: HashMap<String, Vec<u8>>,
    
    /// Ordem de inserção para política LRU/FIFO.
    eviction_queue: VecDeque<String>,
    
    pub max_cached_groups: usize,
    pub max_cached_bytes: usize,
    pub current_bytes: usize,
    
    pub stats: SpecStats,
}

impl SpeculativeCache {
    /// Inicializa com limites adaptativos dependendo do perfil de VRAM.
    pub fn new(max_cached_groups: usize) -> Self {
        Self {
            by_group: HashMap::new(),
            by_name: HashMap::new(),
            eviction_queue: VecDeque::new(),
            max_cached_groups,
            max_cached_bytes: 512 * 1024 * 1024, // 512 MB default
            current_bytes: 0,
            stats: SpecStats::default(),
        }
    }

    pub fn with_bytes_limit(mut self, max_bytes: usize) -> Self {
        self.max_cached_bytes = max_bytes;
        self
    }

    /// Adquire buffers de um grupo inteiro já pré-carregado.
    pub fn try_get_group(&mut self, group_idx: usize) -> Option<Vec<PooledBuffer>> {
        if let Some(bufs) = self.by_group.remove(&group_idx) {
            self.stats.hits += 1;
            Some(bufs)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Tenta adquirir um tensor isolado pelo seu nome (ApexOrchestrator).
    pub fn try_get_by_name(&mut self, name: &str) -> Option<Vec<u8>> {
        if let Some(data) = self.by_name.remove(name) {
            self.current_bytes -= data.len();
            // Remove from queue to maintain consistency
            self.eviction_queue.retain(|x| x != name);
            self.stats.hits += 1;
            Some(data)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insere o grupo no cache baseado em PolledBuffers (modo VRAM pipeline).
    pub fn insert_group(&mut self, group_idx: usize, bufs: Vec<PooledBuffer>) {
        if self.by_group.len() >= self.max_cached_groups {
            if let Some(&oldest) = self.by_group.keys().min() {
                self.by_group.remove(&oldest);
                self.stats.evictions += 1;
            }
        }
        self.by_group.insert(group_idx, bufs);
    }

    /// Insere bytes brutos por nome (modo APEX Burst Reader).
    pub fn insert_by_name(&mut self, name: &str, data: Vec<u8>) {
        let size = data.len();
        
        while self.current_bytes + size > self.max_cached_bytes && !self.eviction_queue.is_empty() {
            if let Some(oldest_name) = self.eviction_queue.pop_front() {
                if let Some(removed) = self.by_name.remove(&oldest_name) {
                    self.current_bytes -= removed.len();
                    self.stats.evictions += 1;
                }
            }
        }

        // Se mesmo limpando tudo ficar grande, ignora (evita panic e OOM)
        if size <= self.max_cached_bytes {
            self.current_bytes += size;
            self.by_name.insert(name.to_string(), data);
            self.eviction_queue.push_back(name.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_cache_hit_miss_eviction() {
        let mut cache = SpeculativeCache::new(3);

        cache.insert_group(0, vec![]);
        cache.insert_group(1, vec![]);
        cache.insert_group(2, vec![]);
        cache.insert_group(4, vec![]); // This will trigger eviction of 0

        assert!(cache.try_get_group(1).is_some());
        assert_eq!(cache.stats.hits, 1);

        assert!(cache.try_get_group(5).is_none());
        assert_eq!(cache.stats.misses, 1);

        cache.insert_group(3, vec![]);
        assert_eq!(cache.stats.evictions, 1);
        assert!(cache.try_get_group(0).is_none());
    }

    #[test]
    fn test_speculative_cache_by_name() {
        let mut cache = SpeculativeCache::new(2).with_bytes_limit(1024);
        
        let data1 = vec![1u8; 512];
        let data2 = vec![2u8; 512];
        
        cache.insert_by_name("tensorA", data1);
        cache.insert_by_name("tensorB", data2);
        
        assert_eq!(cache.current_bytes, 1024);
        
        // Overflow forced
        let data3 = vec![3u8; 512];
        cache.insert_by_name("tensorC", data3);
        
        // TensorA era o mais antigo, deve ter sido evictado
        assert!(cache.try_get_by_name("tensorA").is_none());
        assert_eq!(cache.stats.evictions, 1);
        
        // TensorC deve existir
        assert!(cache.try_get_by_name("tensorC").is_some());
        assert_eq!(cache.stats.hits, 1);
    }
}
