use std::collections::{HashMap, VecDeque};

/// Identificador de um usuário ou prompt simultâneo
pub type SessionId = usize;

/// ID físico de um bloco alocado na VRAM
pub type PhysicalBlockId = usize;

/// ID lógico (virtual) que o usuário enxerga para o seu contexto contíguo
pub type LogicalBlockId = usize;

/// Uma requisição do usuário na fila do escalonador
pub struct InferenceRequest {
    pub session_id: SessionId,
    pub token_ids: Vec<u32>,
    pub is_prefill: bool,
}

/// NodeStor COBER v2 - Subsistema PagedAttention / Multi-Tenant
/// 
/// Em vez de alocar tensores monolíticos, alocamos páginas fixas de VRAM (e SSD via KV Cache).
/// Múltiplas sessões compartilham o mesmo PagePool através de suas Virtual Paging Tables.
pub struct PagePool {
    pub total_blocks: usize,
    /// true se o bloco está livre, false se está ocupado
    pub free_bitmap: Vec<bool>,
    /// Cache de Prefixos: mapeia (Sequence Hash) -> PhysicalBlockId para reuso cross-session
    pub prefix_cache: HashMap<u64, PhysicalBlockId>,
}

impl PagePool {
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_blocks,
            free_bitmap: vec![true; total_blocks],
            prefix_cache: HashMap::new(),
        }
    }

    pub fn allocate_block(&mut self) -> Option<PhysicalBlockId> {
        if let Some(idx) = self.free_bitmap.iter().position(|&is_free| is_free) {
            self.free_bitmap[idx] = false;
            Some(idx)
        } else {
            None // Out of VRAM error: Trigger KVCache Eviction here
        }
    }

    pub fn free_block(&mut self, id: PhysicalBlockId) {
        if id < self.total_blocks {
            self.free_bitmap[id] = true;
        }
    }
}

/// Tabela de roteamento de um usuário virtual. Similar a Page Table da CPU.
pub struct SessionPageTable {
    pub logical_to_physical: HashMap<LogicalBlockId, PhysicalBlockId>,
    pub num_allocated: usize,
}

impl SessionPageTable {
    pub fn new() -> Self {
        Self {
            logical_to_physical: HashMap::new(),
            num_allocated: 0,
        }
    }
}

pub struct MultiTenantScheduler {
    pub page_pool: PagePool,
    pub session_tables: HashMap<SessionId, SessionPageTable>,
    pub queue: VecDeque<InferenceRequest>,
    pub context_size_per_block: usize,
}

impl MultiTenantScheduler {
    pub fn new(total_blocks: usize, context_size_per_block: usize) -> Self {
        Self {
            page_pool: PagePool::new(total_blocks),
            session_tables: HashMap::new(),
            queue: VecDeque::new(),
            context_size_per_block,
        }
    }

    pub fn submit_request(&mut self, request: InferenceRequest) {
        self.queue.push_back(request);
    }

    /// Aloca páginas físicas para todas as requests na fila (Continuous Batching)
    pub fn schedule_next_batch(&mut self) -> Vec<SessionId> {
        let mut active_sessions = Vec::new();
        let mut to_process = self.queue.len();

        for _ in 0..to_process {
            if let Some(mut req) = self.queue.pop_front() {
                let table = self.session_tables.entry(req.session_id).or_insert_with(SessionPageTable::new);

                // Quantos blocos a request precisa?
                let needed_blocks = (req.token_ids.len() + self.context_size_per_block - 1) / self.context_size_per_block;
                let mut success = true;

                for block_offset in 0..needed_blocks {
                    let logical_id = table.num_allocated + block_offset;
                    
                    if !table.logical_to_physical.contains_key(&logical_id) {
                        if let Some(phys_id) = self.page_pool.allocate_block() {
                            table.logical_to_physical.insert(logical_id, phys_id);
                        } else {
                            // Out of VRAM (GPU FULL), re-queueia e para o scheduling (ou faria Eviction)
                            success = false;
                            break;
                        }
                    }
                }

                if success {
                    table.num_allocated += needed_blocks;
                    req.token_ids.clear(); // Processado (mock)
                    active_sessions.push(req.session_id);
                } else {
                    self.queue.push_front(req); // Volta pra fila
                }
            }
        }

        active_sessions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_tenant_paging() {
        let mut scheduler = MultiTenantScheduler::new(5, 10); // 5 blocos total, 10 tokens por bloco

        // Sessao 1 pede 15 tokens (2 blocos)
        scheduler.submit_request(InferenceRequest {
            session_id: 1,
            token_ids: vec![1; 15],
            is_prefill: true,
        });

        // Sessao 2 pede 25 tokens (3 blocos)
        scheduler.submit_request(InferenceRequest {
            session_id: 2,
            token_ids: vec![2; 25],
            is_prefill: true,
        });

        let active = scheduler.schedule_next_batch();
        assert_eq!(active.len(), 2); // Ambas entraram (2 + 3 = 5 blocos consumidos certinho. 100% locação)

        // Verificação do esvaziamento da piscina
        assert!(!scheduler.page_pool.free_bitmap.contains(&true));

        // Sessão 3 pede 1 bloco. Deve falhar
        scheduler.submit_request(InferenceRequest {
            session_id: 3,
            token_ids: vec![3; 5],
            is_prefill: true,
        });

        let active_fail = scheduler.schedule_next_batch();
        assert_eq!(active_fail.len(), 0); // VRAM Cheia
    }
}
