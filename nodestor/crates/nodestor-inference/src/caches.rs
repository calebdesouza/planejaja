/// COBER — Expert LRU Cache para Modo MoE.
///
/// Mantém os experts mais recentemente usados na VRAM.
/// Aproveita a alta localidade temporal dos MoEs:
/// tokens consecutivos reusam os mesmos experts ~70% do tempo.

use std::collections::{HashMap, VecDeque};

/// Identificador único de um expert: (índice_da_camada, índice_do_expert).
pub type ExpertId = (usize, usize);

/// Expert carregado na VRAM (dado como blob de bytes quantizados).
#[derive(Debug, Clone)]
pub struct ExpertData {
    /// Pesos do expert (Q4/Q6K).
    pub weights: Vec<u8>,
    /// Tamanho em bytes.
    pub size_bytes: usize,
    /// Quantas vezes foi acessado nessa sessão.
    pub hit_count: u64,
}

/// Cache LRU de experts na VRAM.
///
/// Quando um expert é requisitado:
/// - Hit: retorna imediatamente (zero leitura de SSD)
/// - Miss: carrega do SSD, evicta o LRU se necessário
pub struct ExpertLruCache {
    /// Dados dos experts residentes.
    cache: HashMap<ExpertId, ExpertData>,
    /// Ordem de acesso para LRU (frente = mais recente).
    lru_order: VecDeque<ExpertId>,
    /// Orçamento de VRAM em bytes para o cache.
    pub budget_bytes: u64,
    /// Bytes atualmente usados.
    pub used_bytes: u64,
    /// Estatísticas de hit/miss.
    pub hits: u64,
    pub misses: u64,
}

impl ExpertLruCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            budget_bytes,
            used_bytes: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Verifica se um expert está em cache.
    pub fn contains(&self, id: ExpertId) -> bool {
        self.cache.contains_key(&id)
    }

    /// Obtém um expert do cache e atualiza o LRU.
    pub fn get(&mut self, id: ExpertId) -> Option<&ExpertData> {
        if self.cache.contains_key(&id) {
            self.hits += 1;
            // Atualizar LRU: mover para frente
            if let Some(pos) = self.lru_order.iter().position(|&x| x == id) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_front(id);
            // Incrementar hit_count
            if let Some(data) = self.cache.get_mut(&id) {
                data.hit_count += 1;
            }
            self.cache.get(&id)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insere um expert no cache (evicta LRU se necessário).
    pub fn insert(&mut self, id: ExpertId, data: ExpertData) {
        let size = data.size_bytes as u64;

        // Evictar até ter espaço suficiente
        while self.used_bytes + size > self.budget_bytes && !self.lru_order.is_empty() {
            if let Some(lru_id) = self.lru_order.pop_back() {
                if let Some(evicted) = self.cache.remove(&lru_id) {
                    self.used_bytes = self.used_bytes.saturating_sub(evicted.size_bytes as u64);
                }
            }
        }

        // Se ainda não cabe (expert maior que o budget), não inserir
        if size > self.budget_bytes {
            return;
        }

        self.used_bytes += size;
        self.lru_order.push_front(id);
        self.cache.insert(id, data);
    }

    /// Prediz quais experts serão necessários no próximo token.
    /// Baseado na localidade temporal: repete os experts do token atual.
    pub fn predict_next(&self, current_experts: &[ExpertId]) -> Vec<ExpertId> {
        // Heurística simples: os mesmos experts têm ~70% de chance de repetir.
        // Em implementação completa, usaríamos o Router para scoring.
        current_experts.to_vec()
    }

    /// Retorna a taxa de acerto do cache.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Número de experts em cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Uso atual em MB.
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / 1024.0 / 1024.0
    }

    /// Budget total em MB.
    pub fn budget_mb(&self) -> f64 {
        self.budget_bytes as f64 / 1024.0 / 1024.0
    }
}

/// Cache de features para modo Diffusion.
///
/// Entre passos consecutivos de denoising, features intermediárias
/// mudam pouco. Este cache detecta e reutiliza features estáveis,
/// reduzindo a quantidade de camadas que precisam ser recarregadas.
pub struct FeatureCache {
    /// Features por camada: layer_idx → ativação (bytes FP16).
    features: HashMap<usize, Vec<u8>>,
    /// Limiar de mudança: features com delta < threshold são reutilizadas.
    pub change_threshold: f32,
    /// Orçamento de bytes para o cache.
    pub budget_bytes: u64,
    pub used_bytes: u64,
    /// Estatísticas: camadas que reutilizaram feature vs recomputaram.
    pub layer_hits: u64,
    pub layer_misses: u64,
}

impl FeatureCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            features: HashMap::new(),
            change_threshold: 0.01, // 1% de mudança → reutiliza
            budget_bytes,
            used_bytes: 0,
            layer_hits: 0,
            layer_misses: 0,
        }
    }

    /// Verifica se a feature de uma camada está cacheada.
    pub fn has_feature(&self, layer_idx: usize) -> bool {
        self.features.contains_key(&layer_idx)
    }

    /// Obtém feature cacheada de uma camada.
    pub fn get_feature(&mut self, layer_idx: usize) -> Option<&Vec<u8>> {
        if self.features.contains_key(&layer_idx) {
            self.layer_hits += 1;
            self.features.get(&layer_idx)
        } else {
            self.layer_misses += 1;
            None
        }
    }

    /// Armazena a feature de uma camada.
    pub fn store_feature(&mut self, layer_idx: usize, feature: Vec<u8>) {
        let size = feature.len() as u64;
        if size > self.budget_bytes {
            return; // Feature maior que o budget total: ignorar
        }
        // Remove antiga se existir
        if let Some(old) = self.features.remove(&layer_idx) {
            self.used_bytes = self.used_bytes.saturating_sub(old.len() as u64);
        }
        // Insere nova se couber
        if self.used_bytes + size <= self.budget_bytes {
            self.used_bytes += size;
            self.features.insert(layer_idx, feature);
        }
    }

    /// Invalida o cache (início de novo passo de denoising major).
    pub fn invalidate(&mut self) {
        self.features.clear();
        self.used_bytes = 0;
    }

    /// Taxa de acerto por camada.
    pub fn hit_rate(&self) -> f64 {
        let total = self.layer_hits + self.layer_misses;
        if total == 0 { 0.0 } else { self.layer_hits as f64 / total as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_expert(size_kb: usize) -> ExpertData {
        ExpertData {
            weights: vec![0u8; size_kb * 1024],
            size_bytes: size_kb * 1024,
            hit_count: 0,
        }
    }

    #[test]
    fn test_expert_cache_hit_miss() {
        let mut cache = ExpertLruCache::new(10 * 1024 * 1024); // 10 MB budget
        cache.insert((0, 0), make_expert(1024)); // 1 MB

        assert!(cache.contains((0, 0)), "Expert (0,0) deve estar em cache");
        assert!(!cache.contains((0, 1)), "Expert (0,1) não deve estar em cache");

        let result = cache.get((0, 0));
        assert!(result.is_some());
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 0);

        let _ = cache.get((0, 1));
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_expert_cache_lru_eviction() {
        let mut cache = ExpertLruCache::new(3 * 1024 * 1024); // 3 MB budget

        cache.insert((0, 0), make_expert(1024)); // 1 MB
        cache.insert((0, 1), make_expert(1024)); // 1 MB
        cache.insert((0, 2), make_expert(1024)); // 1 MB — cache cheio

        // Acessar (0,0) para torná-lo recente
        let _ = cache.get((0, 0));

        // Inserir novo expert: deve evictar o LRU, que agora é (0,1)
        cache.insert((0, 3), make_expert(1024)); // 1 MB

        assert!(cache.contains((0, 0)), "(0,0) deve estar em cache (foi acessado recentemente)");
        assert!(cache.contains((0, 3)), "(0,3) deve estar em cache (recém inserido)");
        // (0,1) ou (0,2) devem ter sido evictados
        let evicted = !cache.contains((0, 1)) || !cache.contains((0, 2));
        assert!(evicted, "Um dos experts antigos deveria ter sido evictado");
    }

    #[test]
    fn test_expert_cache_hit_rate_calculation() {
        let mut cache = ExpertLruCache::new(50 * 1024 * 1024);
        cache.insert((0, 0), make_expert(1));
        cache.insert((0, 1), make_expert(1));

        // 2 hits, 1 miss
        let _ = cache.get((0, 0));
        let _ = cache.get((0, 1));
        let _ = cache.get((0, 2)); // miss

        let rate = cache.hit_rate();
        assert!((rate - 0.666).abs() < 0.01, "Hit rate deve ser ~66.6%, foi {:.3}", rate);
    }

    #[test]
    fn test_feature_cache_store_and_retrieve() {
        let mut cache = FeatureCache::new(100 * 1024 * 1024); // 100 MB

        let feature_data = vec![1u8, 2, 3, 4];
        cache.store_feature(5, feature_data.clone());

        assert!(cache.has_feature(5));
        let retrieved = cache.get_feature(5);
        assert_eq!(retrieved, Some(&feature_data));
        assert_eq!(cache.layer_hits, 1);
    }

    #[test]
    fn test_feature_cache_invalidate() {
        let mut cache = FeatureCache::new(100 * 1024 * 1024);
        cache.store_feature(0, vec![0u8; 100]);
        cache.store_feature(1, vec![0u8; 100]);
        assert!(cache.has_feature(0));

        cache.invalidate();
        assert!(!cache.has_feature(0), "Cache deve estar vazio após invalidar");
        assert_eq!(cache.used_bytes, 0);
    }
}
