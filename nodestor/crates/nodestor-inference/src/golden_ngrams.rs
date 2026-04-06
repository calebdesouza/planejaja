use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// NodeStor COBER v2 - Subsistema 5: N-Grams de Ouro (Memória Muscular)
///
/// Mantém um cache rápido L1 das aceitações passadas (Tree Attention).
/// Quando o modelo começa a falar sobre um assunto, os "caminhos" gerados
/// são lembrados. Na próxima vez que o mesmo Hash de Contexto aparecer,
/// o N-Gram Ouro é injetado, possivelmente zerando a necessidade de verificação GPU.
pub struct GoldenNgramCache {
    /// Mapeia o Hash do Contexto para as continuações validadas.
    /// Ex: Hash("O Brasil ") -> ["é", "um", "país"]
    cache: HashMap<u64, Vec<u32>>,
    /// Fila LRU simples para controle de tamanho da memória muscular
    eviction_queue: VecDeque<u64>,
    /// Máximo de entradas antes de começar a evictar
    max_entries: usize,
    /// Quantidade de tokens de contexto usados para calcular o hash (safety context size)
    context_size: usize,
    
    // Estatísticas para telemetria
    pub hits: u64,
    pub misses: u64,
}

impl GoldenNgramCache {
    pub fn new(max_entries: usize, context_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            eviction_queue: VecDeque::new(),
            max_entries,
            context_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Calcula o Hash de Safety garantindo que a continuação só será
    /// sugerida se o contexto local (as últimas palavras) for idêntico.
    fn compute_context_hash(&self, context: &[u32]) -> Option<u64> {
        if context.len() < self.context_size {
            return None;
        }
        let recent_context = &context[context.len() - self.context_size..];
        let mut hasher = DefaultHasher::new();
        recent_context.hash(&mut hasher);
        Some(hasher.finish())
    }

    /// Insere uma sequência recém validada pela GPU na memória muscular.
    pub fn insert_verified(&mut self, context: &[u32], continuation: &[u32]) {
        if continuation.is_empty() {
            return;
        }

        if let Some(hash) = self.compute_context_hash(context) {
            // Políticas de Eviction e limites de N-Gram
            if self.cache.len() >= self.max_entries {
                if let Some(oldest) = self.eviction_queue.pop_front() {
                    self.cache.remove(&oldest);
                }
            }

            self.cache.insert(hash, continuation.to_vec());
            self.eviction_queue.push_back(hash);
        }
    }

    /// Tenta recuperar uma sugestão de Ouro com base no contexto recente.
    pub fn try_get(&mut self, context: &[u32]) -> Option<Vec<u32>> {
        if let Some(hash) = self.compute_context_hash(context) {
            if let Some(continuation) = self.cache.get(&hash) {
                self.hits += 1;
                return Some(continuation.clone());
            }
        }
        self.misses += 1;
        None
    }

    /// Integração com o banco vetor/SSD futuro para não perder a memória ao fechar.
    /// Exporta os hashes e vec de tokens para armazenar no LanceDB/RaBitQ.
    pub fn export_for_persistence(&self) -> Vec<(u64, Vec<u32>)> {
        self.cache.iter()
            .map(|(&k, v)| (k, v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_ngram_cache_insert_get() {
        let mut cache = GoldenNgramCache::new(100, 3);
        
        let context = vec![1, 2, 3, 4, 5]; // "hoje", "o", "dia", "está", "lindo"
        let continuation = vec![6, 7];      // "não", "acha?"

        cache.insert_verified(&context, &continuation);

        // Deve dar hit quando temos o mesmo contexto
        let hit = cache.try_get(&[99, 100, 3, 4, 5]).unwrap();
        assert_eq!(hit, vec![6, 7]);
        assert_eq!(cache.hits, 1);

        // Deve dar miss num contexto parecido mas onde a ultima palavra muda
        let miss = cache.try_get(&[99, 100, 3, 4, 8]);
        assert_eq!(miss, None);
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_golden_ngram_eviction() {
        // Cache minúsculo de apenas 2 entradas
        let mut cache = GoldenNgramCache::new(2, 2);

        cache.insert_verified(&[1, 2], &[90]);
        cache.insert_verified(&[3, 4], &[91]);
        cache.insert_verified(&[5, 6], &[92]); // Deve ejetar o [1, 2]

        assert!(cache.try_get(&[1, 2]).is_none());
        assert!(cache.try_get(&[3, 4]).is_some());
        assert!(cache.try_get(&[5, 6]).is_some());
    }
}
