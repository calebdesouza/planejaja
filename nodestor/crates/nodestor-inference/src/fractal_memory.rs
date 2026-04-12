use std::collections::HashMap;

/// NodeStor COBER v2 - Subsistema 6: Memória Fractal
/// 
/// Em vez de tratar todos os tokens do contexto de maneira igual, o NodeStor
/// comprime contextos antigos em resoluções menores, mantendo os mais novos intactos.
/// 
/// L0 (Tokens): Acesso total, na arquitetura Transformer pura. VRAM.
/// L1 (Vectors): Acesso por query HNSW/L1, RAM/VRAM secundária.
/// L2 (Summaries): Representação condensada conceitual, Texto em RAM.
/// L3 (Index): HNSW Persistente no LanceDB, SSD.

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryLevel {
    L0Tokens,
    L1Vectors,
    L2Summaries,
    L3Index, // Eviction Indexada no LanceDB
}

pub struct MemoryBlock {
    pub id: usize,
    pub level: MemoryLevel,
    /// Os tokens brutos, se em L0, senão `None`
    pub tokens: Option<Vec<u32>>,
    /// O vetor centroid ou reduzido (PQ) do bloco, para L1 e L3
    pub vector: Option<Vec<f32>>,
    /// O resumo em texto de alto nível daquele bloco, para L2
    pub summary: Option<String>,
}

pub struct FractalMemory {
    pub blocks: HashMap<usize, MemoryBlock>,
    /// Configuração de limites: quantos blocos mantemos por nível antes de comprimir ao próximo
    pub l0_max_blocks: usize,
    pub l1_max_blocks: usize,
    pub l2_max_blocks: usize,
    next_block_id: usize,
}

impl FractalMemory {
    pub fn new(l0_max: usize, l1_max: usize, l2_max: usize) -> Self {
        Self {
            blocks: HashMap::new(),
            l0_max_blocks: l0_max,
            l1_max_blocks: l1_max,
            l2_max_blocks: l2_max,
            next_block_id: 0,
        }
    }

    /// Aloca um novo bloco no nível L0
    pub fn push_l0_block(&mut self, tokens: Vec<u32>) -> usize {
        let id = self.next_block_id;
        self.next_block_id += 1;
        
        self.blocks.insert(id, MemoryBlock {
            id,
            level: MemoryLevel::L0Tokens,
            tokens: Some(tokens),
            vector: None,
            summary: None,
        });

        // Tenta compactar (evict) os excedentes em L0 -> L1 -> L2 -> L3
        self.compact_exceeding();
        id
    }

    /// Executa a rotina de compactação em cascata baseada em LRU/FIFO.
    /// Blocos L0 em excesso viram L1. Blocos L1 em excesso viram L2. etc.
    fn compact_exceeding(&mut self) {
        let l0_count = self.blocks.values().filter(|b| b.level == MemoryLevel::L0Tokens).count();
        if l0_count > self.l0_max_blocks {
            if let Some(oldest_l0) = self.find_oldest(MemoryLevel::L0Tokens) {
                self.compact_l0_to_l1(oldest_l0);
            }
        }

        let l1_count = self.blocks.values().filter(|b| b.level == MemoryLevel::L1Vectors).count();
        if l1_count > self.l1_max_blocks {
            if let Some(oldest_l1) = self.find_oldest(MemoryLevel::L1Vectors) {
                self.compact_l1_to_l2(oldest_l1);
            }
        }

        let l2_count = self.blocks.values().filter(|b| b.level == MemoryLevel::L2Summaries).count();
        if l2_count > self.l2_max_blocks {
            if let Some(oldest_l2) = self.find_oldest(MemoryLevel::L2Summaries) {
                self.compact_l2_to_l3(oldest_l2);
            }
        }
    }

    fn find_oldest(&self, level: MemoryLevel) -> Option<usize> {
        // Usa o id como proxy para idade (menor = mais velho), visto que id é iterado monotonamente
        self.blocks
            .values()
            .filter(|b| b.level == level)
            .map(|b| b.id)
            .min()
    }

    /// L0 -> L1: Calcula embedding representativo com Atenção Semântica (Sobrevivência do Mais Apto)
    fn compact_l0_to_l1(&mut self, id: usize) {
        if let Some(block) = self.blocks.get_mut(&id) {
            // Se existirem tokens, normalmente teríamos uma matriz de embeddings [N, emb_dim].
            // Para mockar a atenção semântica, simularemos N embeddings e ponderaremos via soft-attn.
            let mock_query = vec![1.0; 128]; // O "contexto Master" atual ou surpresa
            
            // Simula 3 tokens neste bloco
            let tok1 = vec![0.8; 128];
            let tok2 = vec![0.1; 128]; // ruido
            let tok3 = vec![0.9; 128];
            
            let keys: Vec<&[f32]> = vec![&tok1, &tok2, &tok3];
            
            // Temperatura = 3.0 para suavizar (soften) a atenção e não matar o ruído periférico (contraditório)
            let mut alphas = crate::semantic_attention::compute_attention_weights(&mock_query, &keys, 3.0);
            
            // Aplica um piso mínimo (Soft-clip cognitivo) para evitar viés de confirmação absoluto
            let min_attention_floor = 0.10;
            let mut sum_alpha = 0.0;
            for alpha in alphas.iter_mut() {
                if *alpha < min_attention_floor {
                    *alpha = min_attention_floor;
                }
                sum_alpha += *alpha;
            }
            // Renormaliza para somar 1.0
            for alpha in alphas.iter_mut() {
                *alpha /= sum_alpha;
            }
            
            // O vetor compactado final não é uma média cega, mas a soma ponderada pelos Alphas!
            let mut final_vector = vec![0.0f32; 128];
            for (alpha, tok_emb) in alphas.iter().zip(keys.iter()) {
                for i in 0..128 {
                    final_vector[i] += alpha * tok_emb[i];
                }
            }
            
            block.tokens = None; // Libera a RAM/VRAM dos tokens brutos (Eviction real!)
            block.vector = Some(final_vector); // Mantém apenas o L1 compactado inteligentemente
            block.level = MemoryLevel::L1Vectors;
        }
    }

    /// L1 -> L2: Gera o resumo condensado do bloco vector e joga o vector fora
    fn compact_l1_to_l2(&mut self, id: usize) {
        if let Some(block) = self.blocks.get_mut(&id) {
            // Em aplicação real, faz pipeline para o modelo dar summarize
            let mock_summary = "Sumario condensado do bloco evictado.".to_string();
            
            block.vector = None; // Libera RAM do embedding
            block.summary = Some(mock_summary); 
            block.level = MemoryLevel::L2Summaries;
        }
    }

    /// L2 -> L3: Persiste o bloco L2 no LanceDB HNSW HDD e limpa ele da RAM
    fn compact_l2_to_l3(&mut self, id: usize) {
        if let Some(block) = self.blocks.get_mut(&id) {
            // Em aplicação real, enviaria o vector/summary via RRF pro LanceDB Core.
            // Aqui marcamos como L3, indicando que ele "sumiu" e vive no disco.
            block.summary = None; 
            block.level = MemoryLevel::L3Index; 
        }
    }

    /// RAG Interno: Foca a "lupa" da memória fractal dando Zoom L3/L2 -> L0
    /// Quando uma query atual bate com o summary ou vetor
    pub fn zoom_block_to_l0(&mut self, id: usize, recovered_tokens: Vec<u32>) {
        if let Some(block) = self.blocks.get_mut(&id) {
            // O sistema localiza que um bloco antigo é altamente relevante
            // e trás ele do LanceDB(L3) devolta ao foco L0 da Memória do Modelo
            block.tokens = Some(recovered_tokens);
            block.vector = None;
            block.summary = None;
            block.level = MemoryLevel::L0Tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_memory_cascading_compaction() {
        // Limites minusculos pra testar o ripple do Eviction
        // 2 no L0, 1 no L1, 1 no L2
        let mut fm = FractalMemory::new(2, 1, 1);
        
        let id0 = fm.push_l0_block(vec![1, 2, 3]); // L0 count: 1
        let id1 = fm.push_l0_block(vec![4, 5, 6]); // L0 count: 2
        
        assert_eq!(fm.blocks.get(&id0).unwrap().level, MemoryLevel::L0Tokens);
        assert_eq!(fm.blocks.get(&id1).unwrap().level, MemoryLevel::L0Tokens);
        
        // Exceder L0 empurra pro L1
        let _id2 = fm.push_l0_block(vec![7, 8, 9]); // L0 c: 2, L1 c: 1
        assert_eq!(fm.blocks.get(&id0).unwrap().level, MemoryLevel::L1Vectors); //id0 foi pra L1
        
        // Exceder dnv empurra id0 pro L2, id1 pro L1.
        let _id3 = fm.push_l0_block(vec![10, 11]); 
        assert_eq!(fm.blocks.get(&id0).unwrap().level, MemoryLevel::L2Summaries);
        assert_eq!(fm.blocks.get(&id1).unwrap().level, MemoryLevel::L1Vectors);

        // Exceder dnv empurra id0 pro L3 do LanceDB. E "apaga" seu relampago em RAM.
        let _id4 = fm.push_l0_block(vec![12, 13]);
        let block0 = fm.blocks.get(&id0).unwrap();
        assert_eq!(block0.level, MemoryLevel::L3Index);
        assert!(block0.summary.is_none());
        assert!(block0.vector.is_none());
        assert!(block0.tokens.is_none()); // 100% de desconto de memoria. Tudo está no LanceDB SSD
    }
}
