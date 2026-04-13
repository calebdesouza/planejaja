//! # PagedAttention — Gestão de KV-Cache por Páginas Lógicas
//!
//! ## O problema que resolve
//!
//! O KV-cache clássico aloca um bloco contíguo de VRAM por request:
//! - Request A com 2048 tokens: reserve 2048 * 2 * D * L bytes na VRAM
//! - Request B com 8 tokens:    reserve 8 * 2 * D * L bytes na VRAM
//!
//! **Fragmentação:** A reserva de memória contigua leva a desperdício enorme quando
//! requests de tamanhos diferentes coexistem. vLLM demonstrou que isso desperdiça
//! 60-80% da VRAM em servidores multi-usuário.
//!
//! ## Solução: Blocos lógicos → páginas físicas não-contíguas
//!
//! ```text
//! Request A: bloco_lógico 0 → página_física #5
//!            bloco_lógico 1 → página_física #12
//!            bloco_lógico 2 → página_física #3
//!
//! Request B: bloco_lógico 0 → página_física #8
//!            (compartilhado com A se mesmo prefixo) ← RadixCache
//! ```
//!
//! Cada página física tem tamanho fixo de `BLOCK_SIZE` tokens.
//! A GPU recebe uma **block table** (array de índices) e sabe quais páginas
//! pertencem a qual request — sem precisar que elas sejam contíguas em memória.
//!
//! ## Throughput esperado
//!
//! Em servidor com 4 × RTX 4090 (96 GB VRAM):
//! - KV-cache clássico: ~30-50 usuários simultâneos
//! - PagedAttention:    ~200-400 usuários simultâneos (mesmo hardware!)
//!
//! A razão: fragmentação cai de ~75% para ~5%.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Tamanho de uma página em tokens.
/// 16 é o valor usado pelo vLLM e é o trade-off ótimo entre overhead de
/// tabela de páginas e granularidade de alocação.
pub const PAGE_SIZE: usize = 16;

/// Identificador único de uma página física de VRAM.
pub type PhysicalBlockId = u32;

/// Identificador único de um request de inferência.
pub type RequestId = u64;

/// Estado de uma página física.
#[derive(Debug, Clone, PartialEq)]
pub enum PageState {
    /// Livre para alocação.
    Free,
    /// Alocada para um request específico, com contagem de tokens usados.
    Allocated { request_id: RequestId, tokens_used: usize },
    /// Compartilhada (read-only) entre múltiplos requests via Copy-on-Write.
    Shared { ref_count: u32, owner: RequestId },
}

/// Mapa de blocos lógicos de um request para páginas físicas.
#[derive(Debug, Default, Clone)]
pub struct BlockTable {
    /// `logical_block_idx → physical_block_id`
    pub mapping: Vec<PhysicalBlockId>,
}

impl BlockTable {
    /// Retorna o número de blocões lógicos alocados.
    pub fn len(&self) -> usize { self.mapping.len() }
    pub fn is_empty(&self) -> bool { self.mapping.is_empty() }

    /// Retorna a página física para um bloco lógico.
    pub fn physical_block(&self, logical_idx: usize) -> Option<PhysicalBlockId> {
        self.mapping.get(logical_idx).copied()
    }
}

/// Estatísticas de uso do gerenciador de páginas.
#[derive(Debug, Default)]
pub struct PagedAttentionStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub allocations: u64,
    pub deallocations: u64,
    pub page_faults: u64,
    pub cow_copies: u64,  // Copy-on-Write copies (RadixCache)
}

/// Gerenciador de KV-Cache com paginação lógica/física.
///
/// Mantém um pool de páginas físicas e os block tables de cada request.
/// Integra com o `RadixCache` para reutilização de prefixos comuns.
pub struct PagedAttentionManager {
    /// Pool de páginas físicas. Index = PhysicalBlockId.
    pages: Vec<PageState>,
    /// Fila de páginas livres (LIFO para cache warmth)
    free_list: Vec<PhysicalBlockId>,
    /// Block table por request: request_id → BlockTable
    block_tables: HashMap<RequestId, BlockTable>,
    /// Dimensão da cabeça de atenção (head_dim × 2 para K e V)
    head_dim: usize,
    /// Número de cabeças de KV
    num_kv_heads: usize,
    /// Estatísticas
    pub stats: PagedAttentionStats,
}

impl PagedAttentionManager {
    /// Inicializa o gerenciador com `num_pages` páginas físicas.
    ///
    /// Memória total de KV-cache = `num_pages × PAGE_SIZE × num_kv_heads × head_dim × 2 × sizeof(f16)`.
    pub fn new(num_pages: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let free_list: Vec<PhysicalBlockId> = (0..num_pages as PhysicalBlockId).rev().collect();
        let pages = vec![PageState::Free; num_pages];

        info!(
            "PagedAttention: {} páginas × {} tokens = {} tokens máx | kv_heads={} head_dim={}",
            num_pages, PAGE_SIZE, num_pages * PAGE_SIZE, num_kv_heads, head_dim
        );

        Self {
            pages,
            free_list,
            block_tables: HashMap::new(),
            head_dim,
            num_kv_heads,
            stats: PagedAttentionStats {
                total_pages: num_pages,
                free_pages: num_pages,
                ..Default::default()
            },
        }
    }

    /// Registra um novo request no sistema.
    pub fn register_request(&mut self, request_id: RequestId) {
        self.block_tables.insert(request_id, BlockTable::default());
        debug!("PagedAttention: Request {} registrado", request_id);
    }

    /// Aloca páginas suficientes para N tokens novos em um request.
    ///
    /// Retorna `Err` se não houver páginas livres suficientes (VRAM cheia).
    pub fn allocate_tokens(
        &mut self,
        request_id: RequestId,
        num_new_tokens: usize,
    ) -> Result<(), String> {
        let table = self.block_tables
            .entry(request_id)
            .or_insert_with(BlockTable::default);

        // Quantas páginas novas precisamos?
        let current_tokens: usize = table.mapping.len() * PAGE_SIZE;
        let target_tokens = current_tokens + num_new_tokens;
        let pages_needed = target_tokens.div_ceil(PAGE_SIZE).saturating_sub(table.mapping.len());

        if pages_needed > self.free_list.len() {
            warn!(
                "PagedAttention: VRAM cheia! Precisamos {} páginas, só {} livres",
                pages_needed, self.free_list.len()
            );
            return Err(format!(
                "KV-cache cheio: {} páginas necessárias, {} disponíveis",
                pages_needed, self.free_list.len()
            ));
        }

        for _ in 0..pages_needed {
            let phys_id = self.free_list.pop().unwrap();
            self.pages[phys_id as usize] = PageState::Allocated {
                request_id,
                tokens_used: 0,
            };
            table.mapping.push(phys_id);
            self.stats.free_pages -= 1;
            self.stats.allocations += 1;

            debug!(
                "PagedAttention: Request {} alocou página física #{}",
                request_id, phys_id
            );
        }

        Ok(())
    }

    /// Libera todas as páginas de um request (fim de inferência).
    pub fn free_request(&mut self, request_id: RequestId) {
        if let Some(table) = self.block_tables.remove(&request_id) {
            for phys_id in table.mapping {
                match &self.pages[phys_id as usize] {
                    PageState::Shared { ref_count, .. } if *ref_count > 1 => {
                        // Decrementa ref_count — a página ainda é usada por outros
                        let rc = *ref_count - 1;
                        let owner = match &self.pages[phys_id as usize] {
                            PageState::Shared { owner, .. } => *owner,
                            _ => 0,
                        };
                        self.pages[phys_id as usize] = PageState::Shared {
                            ref_count: rc,
                            owner,
                        };
                        debug!("PagedAttention: Página #{} ref_count → {}", phys_id, rc);
                    }
                    _ => {
                        // Página exclusiva — devolve ao pool
                        self.pages[phys_id as usize] = PageState::Free;
                        self.free_list.push(phys_id);
                        self.stats.free_pages += 1;
                        self.stats.deallocations += 1;
                    }
                }
            }
            debug!("PagedAttention: Request {} liberado. Páginas livres: {}", request_id, self.stats.free_pages);
        }
    }

    /// Retorna o block table de um request para envio ao shader de attention.
    pub fn get_block_table(&self, request_id: RequestId) -> Option<&BlockTable> {
        self.block_tables.get(&request_id)
    }

    /// Número de tokens atualmente alocados para um request.
    pub fn allocated_tokens(&self, request_id: RequestId) -> usize {
        self.block_tables
            .get(&request_id)
            .map(|t| t.mapping.len() * PAGE_SIZE)
            .unwrap_or(0)
    }

    /// Número de páginas físicas livres no pool.
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Relatório de uso para o painel htop.
    pub fn report(&self) -> String {
        let used = self.stats.total_pages - self.free_list.len();
        let pct = (used * 100) / self.stats.total_pages.max(1);
        format!(
            "PagedKV | {}/{} páginas ({pct}%) | alloc={} free={} | requests={}",
            used,
            self.stats.total_pages,
            self.stats.allocations,
            self.stats.deallocations,
            self.block_tables.len(),
        )
    }

    /// Copy-on-Write: faz uma cópia privada de uma página compartilhada
    /// quando um request precisa escrever nela.
    pub fn copy_on_write(&mut self, request_id: RequestId, logical_block: usize) -> Result<PhysicalBlockId, String> {
        let old_phys = self.block_tables
            .get(&request_id)
            .and_then(|t| t.mapping.get(logical_block).copied())
            .ok_or("Bloco lógico não encontrado")?;

        // Se a página é exclusiva, não precisa copiar
        if matches!(&self.pages[old_phys as usize], PageState::Allocated { request_id: rid, .. } if *rid == request_id) {
            return Ok(old_phys);
        }

        // Aloca nova página privada
        let new_phys = self.free_list.pop()
            .ok_or("KV-cache cheio — sem páginas para CoW")?;

        self.pages[new_phys as usize] = PageState::Allocated { request_id, tokens_used: 0 };
        self.stats.free_pages -= 1;
        self.stats.cow_copies += 1;

        // Atualiza o block table deste request
        if let Some(table) = self.block_tables.get_mut(&request_id) {
            table.mapping[logical_block] = new_phys;
        }

        debug!(
            "PagedAttention CoW: Request {} bloco lógico {} → página física {} (era #{})",
            request_id, logical_block, new_phys, old_phys
        );

        Ok(new_phys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_alloc_basic() {
        let mut mgr = PagedAttentionManager::new(64, 8, 128);
        mgr.register_request(1);
        mgr.allocate_tokens(1, 32).unwrap(); // 32 tokens = 2 páginas (16 per page)
        assert_eq!(mgr.get_block_table(1).unwrap().len(), 2);
        assert_eq!(mgr.free_pages(), 62);
    }

    #[test]
    fn test_paged_free_returns_to_pool() {
        let mut mgr = PagedAttentionManager::new(64, 8, 128);
        mgr.register_request(1);
        mgr.allocate_tokens(1, 16).unwrap();
        assert_eq!(mgr.free_pages(), 63);
        mgr.free_request(1);
        assert_eq!(mgr.free_pages(), 64);
    }

    #[test]
    fn test_paged_vram_full_error() {
        let mut mgr = PagedAttentionManager::new(2, 8, 128); // só 2 páginas
        mgr.register_request(1);
        // 3 páginas * 16 tokens = 48 tokens → precisa de 3 páginas → falha
        let result = mgr.allocate_tokens(1, 48);
        assert!(result.is_err());
    }

    #[test]
    fn test_paged_multi_request() {
        let mut mgr = PagedAttentionManager::new(32, 4, 64);
        mgr.register_request(1);
        mgr.register_request(2);
        mgr.allocate_tokens(1, 16).unwrap();
        mgr.allocate_tokens(2, 32).unwrap(); // 2 páginas
        assert_eq!(mgr.free_pages(), 29); // 32 - 1 - 2 = 29
        mgr.free_request(1);
        assert_eq!(mgr.free_pages(), 30);
    }

    #[test]
    fn test_block_table_lookup() {
        let mut mgr = PagedAttentionManager::new(16, 4, 64);
        mgr.register_request(42);
        mgr.allocate_tokens(42, 16).unwrap(); // 1 página
        let table = mgr.get_block_table(42).unwrap();
        assert_eq!(table.len(), 1);
        assert!(table.physical_block(0).is_some());
    }

    #[test]
    fn test_report_format() {
        let mgr = PagedAttentionManager::new(100, 4, 128);
        let report = mgr.report();
        assert!(report.contains("PagedKV"));
        assert!(report.contains("100"));
    }

    #[test]
    fn test_copy_on_write_exclusive_page() {
        let mut mgr = PagedAttentionManager::new(16, 4, 64);
        mgr.register_request(1);
        mgr.allocate_tokens(1, 16).unwrap();
        // CoW numa página exclusiva deve retornar a mesma página (sem cópia)
        let phys_before = mgr.get_block_table(1).unwrap().mapping[0];
        let phys_after = mgr.copy_on_write(1, 0).unwrap();
        assert_eq!(phys_before, phys_after);
        assert_eq!(mgr.stats.cow_copies, 0);
    }
}
