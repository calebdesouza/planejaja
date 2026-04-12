use crate::semantic_attention::compute_attention_weights;
use std::collections::HashMap;

/// NodeStor COBER v2 — Semantic Paging (Atenção Residual)
///
/// Substitui o LRU cego. O motor decide quais camadas do
/// Transformer ou páginas de KV cache manter na VRAM baseado
/// na relevância semântica (softmax attention) do contexto atual.

#[derive(Debug, Clone)]
pub struct PagedLayer {
    pub id: usize,
    /// Chave semântica (representação latente do conteúdo desta camada)
    pub semantic_key: Vec<f32>,
    pub size_bytes: u64,
    /// Se true, está alocada na VRAM. Se false, no SSD.
    pub in_vram: bool,
    /// Se true, a camada é fundacional (ex: primeira/última) e imune à remoção.
    pub is_structural: bool,
}

pub struct SemanticPager {
    pub layers: HashMap<usize, PagedLayer>,
    pub vram_capacity_bytes: u64,
    pub current_vram_usage: u64,
}

impl SemanticPager {
    pub fn new(capacity: u64) -> Self {
        Self {
            layers: HashMap::new(),
            vram_capacity_bytes: capacity,
            current_vram_usage: 0,
        }
    }

    pub fn register_layer(&mut self, id: usize, semantic_key: Vec<f32>, size_bytes: u64, is_structural: bool) {
        self.layers.insert(id, PagedLayer {
            id,
            semantic_key,
            size_bytes,
            in_vram: false, // Inicia no SSD
            is_structural,
        });
    }

    /// Executa o Paging Preditivo: dado o token atual (query),
    /// calcula attention weights para todas as camadas.
    /// Mantém/Coloca na VRAM as com maiores pesos até encher o budget.
    pub fn execute_semantic_paging(&mut self, current_query: &[f32]) {
        if self.layers.is_empty() { return; }

        let mut layer_ids = Vec::with_capacity(self.layers.len());
        let mut keys = Vec::with_capacity(self.layers.len());

        for (id, layer) in &self.layers {
            layer_ids.push(*id);
            // Safe as we tie lifetimes of references to iteration.
            // Para poder coletar as keys num Vec<&[f32]>:
        }

        // Recupera os &[f32] numa pass separada para agradar o borrow checker
        for id in &layer_ids {
            keys.push(self.layers.get(id).unwrap().semantic_key.as_slice());
        }

        // Tenta um T menor para deixar softmax mais afiado/confiante
        let alphas = compute_attention_weights(current_query, &keys, 0.5);

        // Associa id -> alpha
        let mut scored_layers: Vec<(usize, f32)> = layer_ids.into_iter()
            .zip(alphas.into_iter())
            .collect();

        // Ordena descending por alpha
        scored_layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zera o rastreador de uso atual para recalcularmos do zero
        let mut new_vram_usage = 0u64;
        let capacity = self.vram_capacity_bytes;

        // FASE 1: Trava as camadas estruturais (Blindagem Psicológica de Afasia)
        for (id, layer) in self.layers.iter_mut() {
            if layer.is_structural {
                if new_vram_usage + layer.size_bytes <= capacity {
                    layer.in_vram = true;
                    new_vram_usage += layer.size_bytes;
                } else {
                    // Cuidado extremo se a VRAM não couber nem a estrutura básica
                    layer.in_vram = false;
                }
            }
        }

        // FASE 2: Distribui o budget restante sob o poder do Softmax
        for (id, _alpha) in scored_layers {
            let layer = self.layers.get_mut(&id).unwrap();
            
            if layer.is_structural {
                continue; // Já cuidamos dela na FASE 1
            }

            if new_vram_usage + layer.size_bytes <= capacity {
                // Cabe na VRAM
                layer.in_vram = true;
                new_vram_usage += layer.size_bytes;
            } else {
                // Não cabe mais, manda pro SSD
                layer.in_vram = false;
            }
        }
        
        self.current_vram_usage = new_vram_usage;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_paging() {
        // Teto de 200 bytes. Cada camada tem 100 bytes. Só cabem 2 na VRAM.
        let mut pager = SemanticPager::new(200);
        
        // Camada 1: Estrutural de Gramática (sempre fixada)
        pager.register_layer(1, vec![0.0, 0.0, 0.0], 100, true);
        // Camada 2: Cães
        pager.register_layer(2, vec![1.0, 0.0, 0.0], 100, false);
        // Camada 3: Pássaros
        pager.register_layer(3, vec![0.0, 0.0, 1.0], 100, false);

        // O prompt atual fala fortemente de Pássaros
        let query = vec![0.0, 0.1, 0.9]; 
        
        pager.execute_semantic_paging(&query);
        
        // Esperamos que 1 esteja na VRAM porque é estrutural (blindada).
        // A disputa pelas vagas restantes sobra para a Camada 3 (Pássaros) vencer a 2 (Cães)
        assert!(pager.layers.get(&1).unwrap().in_vram, "Camada estrutural blindada foi removida");
        assert!(!pager.layers.get(&2).unwrap().in_vram);
        assert!(pager.layers.get(&3).unwrap().in_vram);
        assert_eq!(pager.current_vram_usage, 200);
    }
}
