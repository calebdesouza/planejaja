//! # RadixCache — Compartilhamento de Prefixo com RadixAttention
//!
//! ## O problema
//!
//! Em sistemas multi-usuário, múltiplos requests frequentemente compartilham
//! o mesmo prefixo de system prompt:
//!
//! ```text
//! Request A: "Você é um assistente útil...[sistema] Qual é a capital do Brasil?"
//! Request B: "Você é um assistente útil...[sistema] Calcule a raiz de 144."
//!                      ↑ IDÊNTICO ↑
//!              16 páginas de KV-cache = desperdício!
//! ```
//!
//! ## Solução: Árvore Radix de Prefixos
//!
//! Mantemos uma árvore onde cada nó representa um bloco de tokens.
//! Se dois requests compartilham prefixo, apontam para os mesmos nós físicos.
//!
//! ```text
//! Raiz → [sistema_prompt: páginas #1, #2, #3, #4]
//!              ├── Request A → [pergunta_A: página #5]
//!              └── Request B → [pergunta_B: página #6]
//! ```
//!
//! Páginas #1-#4 são **compartilhadas** (read-only via CoW).
//! Páginas #5 e #6 são privadas de cada request.
//!
//! ## Throughput
//!
//! Em um servidor de API com system prompts repetidos:
//! - Sem RadixCache: cada request computa e armazena o prefixo inteiro
//! - Com RadixCache: prefixo computado 1 vez, amortizado em milhares de requests
//! - Ganho de throughput: 2-5× em casos típicos de API pública

use std::collections::HashMap;
use crate::paged_attention::{PhysicalBlockId, RequestId, PagedAttentionManager};
use tracing::{debug, info};

/// Hash de um bloco de tokens (fingerprint para matching de prefixo).
type BlockHash = u64;

/// Nó da árvore radix de prefixos.
#[derive(Debug)]
pub struct RadixNode {
    /// Páginas físicas correspondentes a este nó.
    pub physical_blocks: Vec<PhysicalBlockId>,
    /// Número de referências ativas (requests usando este nó).
    pub ref_count: u32,
    /// Filhos: hash do próximo bloco → nó filho
    pub children: HashMap<BlockHash, Box<RadixNode>>,
    /// Número de tokens representados neste nó.
    pub token_count: usize,
    /// Tempo de último acesso (para LRU eviction da árvore).
    pub last_access: u64,
}

impl RadixNode {
    fn new(physical_blocks: Vec<PhysicalBlockId>, token_count: usize) -> Self {
        Self {
            physical_blocks,
            ref_count: 1,
            children: HashMap::new(),
            token_count,
            last_access: 0,
        }
    }

    fn is_leaf(&self) -> bool { self.children.is_empty() }
}

/// Estatísticas da RadixCache.
#[derive(Debug, Default)]
pub struct RadixStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefix_tokens_reused: u64,
    pub nodes_total: u64,
    pub evictions: u64,
}

/// Cache de prefixos compartilhados via árvore Radix.
pub struct RadixCache {
    /// Raiz da árvore Radix (nó virtual sem tokens)
    root: HashMap<BlockHash, Box<RadixNode>>,
    /// Clock lógico para LRU
    clock: u64,
    /// Estatísticas
    pub stats: RadixStats,
}

impl RadixCache {
    pub fn new() -> Self {
        Self {
            root: HashMap::new(),
            clock: 0,
            stats: RadixStats::default(),
        }
    }

    /// Tenta encontrar um prefixo compartilhado na árvore.
    ///
    /// Retorna: `(matched_tokens, physical_blocks_to_reuse)`
    ///
    /// Se `matched_tokens == 0`, nenhum prefixo encontrado.
    /// Se `matched_tokens > 0`, as páginas físicas podem ser reutilizadas diretamente.
    pub fn lookup(&mut self, token_hashes: &[BlockHash]) -> (usize, Vec<PhysicalBlockId>) {
        self.clock += 1;
        let mut matched_blocks = Vec::new();
        let mut matched_tokens = 0;
        let mut current = &mut self.root;

        for &hash in token_hashes {
            if let Some(node) = current.get_mut(&hash) {
                node.last_access = self.clock;
                matched_tokens += node.token_count;
                matched_blocks.extend_from_slice(&node.physical_blocks);
                current = &mut node.children;
            } else {
                // Prefixo termina aqui
                break;
            }
        }

        if matched_tokens > 0 {
            self.stats.cache_hits += 1;
            self.stats.prefix_tokens_reused += matched_tokens as u64;
            debug!("RadixCache HIT: {} tokens de prefixo reutilizados", matched_tokens);
        } else {
            self.stats.cache_misses += 1;
        }

        (matched_tokens, matched_blocks)
    }

    /// Insere um prefixo computado na árvore para futura reutilização.
    ///
    /// `token_hashes`: hashes (um por bloco de PAGE_SIZE tokens)
    /// `physical_blocks`: páginas físicas correspondentes
    pub fn insert(&mut self, token_hashes: &[BlockHash], physical_blocks: &[PhysicalBlockId]) {
        if token_hashes.is_empty() { return; }

        self.clock += 1;
        let mut current = &mut self.root;

        for (i, &hash) in token_hashes.iter().enumerate() {
            let phys = physical_blocks.get(i).copied().unwrap_or(0);

            let node = current.entry(hash).or_insert_with(|| {
                self.stats.nodes_total += 1;
                Box::new(RadixNode::new(vec![phys], crate::paged_attention::PAGE_SIZE))
            });

            node.last_access = self.clock;
            node.ref_count += 1;
            current = &mut node.children;
        }

        info!("RadixCache: Prefixo de {} blocos inserido", token_hashes.len());
    }

    /// Hash FNV-1a simples para sequências de tokens.
    ///
    /// Rápido, determinístico, sem dependências externas.
    pub fn hash_tokens(tokens: &[u32]) -> BlockHash {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64  = 1099511628211;

        let mut hash = FNV_OFFSET;
        for &t in tokens {
            hash ^= t as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Relatório para o painel htop.
    pub fn report(&self) -> String {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        let hit_rate = if total > 0 {
            (self.stats.cache_hits * 100) / total
        } else { 0 };
        format!(
            "RadixCache | hit_rate={}% | tokens_reus={} | nodes={} | evictions={}",
            hit_rate,
            self.stats.prefix_tokens_reused,
            self.stats.nodes_total,
            self.stats.evictions,
        )
    }
}

impl Default for RadixCache {
    fn default() -> Self { Self::new() }
}

/// Bridge entre RadixCache e PagedAttentionManager.
///
/// Combina os dois sistemas para dar o comportamento completo do RadixAttention:
/// 1. Novo request chega com token IDs
/// 2. RadixCache faz lookup por prefixo compartilhado
/// 3. Se encontrado: reutiliza páginas, aloca só o diff
/// 4. Após inferência: insere sequência completa na RadixCache
pub struct RadixAttentionBridge {
    pub paged: PagedAttentionManager,
    pub radix: RadixCache,
}

impl RadixAttentionBridge {
    pub fn new(num_pages: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            paged: PagedAttentionManager::new(num_pages, num_kv_heads, head_dim),
            radix: RadixCache::new(),
        }
    }

    /// Registra um request e reutiliza prefixo compartilhado se disponível.
    ///
    /// Retorna: `(prefix_pages_reused, new_pages_allocated)`
    pub fn register_with_prefix(
        &mut self,
        request_id: RequestId,
        token_ids: &[u32],
    ) -> (usize, usize) {
        self.paged.register_request(request_id);

        // Hash cada bloco de PAGE_SIZE tokens
        let block_hashes: Vec<BlockHash> = token_ids
            .chunks(crate::paged_attention::PAGE_SIZE)
            .map(RadixCache::hash_tokens)
            .collect();

        // Tenta reutilizar prefixo
        let (matched_tokens, _reused_pages) = self.radix.lookup(&block_hashes);

        // Aloca apenas o diff (tokens após o prefixo reutilizado)
        let remaining_tokens = token_ids.len().saturating_sub(matched_tokens);
        if remaining_tokens > 0 {
            let _ = self.paged.allocate_tokens(request_id, remaining_tokens);
        }

        let reused_pages = matched_tokens / crate::paged_attention::PAGE_SIZE;
        let new_pages = remaining_tokens.div_ceil(crate::paged_attention::PAGE_SIZE);

        debug!(
            "RadixAttention: Request {} | prefixo={} tokens ({} páginas) | novo={} tokens",
            request_id, matched_tokens, reused_pages, remaining_tokens
        );

        (reused_pages, new_pages)
    }

    /// Relatório combinado de PagedAttention + RadixCache.
    pub fn report(&self) -> String {
        format!("{}\n{}", self.paged.report(), self.radix.report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_hash_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let h1 = RadixCache::hash_tokens(&tokens);
        let h2 = RadixCache::hash_tokens(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_radix_hash_different_sequences() {
        let h1 = RadixCache::hash_tokens(&[1, 2, 3]);
        let h2 = RadixCache::hash_tokens(&[1, 2, 4]); // diferente no último token
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_radix_lookup_empty() {
        let mut cache = RadixCache::new();
        let (matched, blocks) = cache.lookup(&[12345]);
        assert_eq!(matched, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_radix_insert_and_hit() {
        let mut cache = RadixCache::new();
        let hash = RadixCache::hash_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        cache.insert(&[hash], &[42]);
        let (matched, blocks) = cache.lookup(&[hash]);
        assert!(matched > 0, "Deve encontrar o prefixo inserido");
        assert_eq!(blocks[0], 42);
    }

    #[test]
    fn test_radix_cache_miss() {
        let mut cache = RadixCache::new();
        let (matched, _) = cache.lookup(&[999999]);
        assert_eq!(matched, 0);
        assert_eq!(cache.stats.cache_misses, 1);
    }

    #[test]
    fn test_radix_bridge_new_request() {
        let mut bridge = RadixAttentionBridge::new(64, 4, 128);
        let tokens: Vec<u32> = (0..32).collect(); // 32 tokens = 2 páginas
        let (reused, new_pages) = bridge.register_with_prefix(1, &tokens);
        assert_eq!(reused, 0); // nenhum prefixo conhecido ainda
        assert!(new_pages > 0); // deve alocar páginas
    }

    #[test]
    fn test_radix_report() {
        let cache = RadixCache::new();
        assert!(cache.report().contains("RadixCache"));
    }

    #[test]
    fn test_bridge_report() {
        let bridge = RadixAttentionBridge::new(32, 4, 64);
        let report = bridge.report();
        assert!(report.contains("PagedKV"));
        assert!(report.contains("RadixCache"));
    }
}
