/// ContextSsd — Contexto Infinito via KV Cache no SSD + Índice Vetorial
///
/// Problema: modelos têm janelas de contexto finitas (8K–128K tokens).
/// Solução: KV cache antigo é comprimido e arquivado no SSD. Um índice
/// vetorial (cosine similarity por embeddings semânticos) recupera os
/// blocos de passado relevantes para cada novo token, reinjetando-os
/// na atenção como "memória de longo prazo" sem truncar o histórico.
///
/// Fluxo por token:
///   1. Forward pass emite hidden state h ∈ ℝ^d_model (embedding semântico)
///   2. KV states do token são adicionados ao buffer ativo em RAM
///   3. Quando RAM buffer excede `active_capacity`:
///      → bloco mais antigo é comprimido (LZ4) e escrito no SSD
///      → embedding do bloco é inserido no índice vetorial
///   4. Na atenção, consulta o índice com o hidden state atual
///      → top-K blocos semânticamente relevantes carregados do SSD
///      → reinjetados como KV extra no attention computation
///
/// Resultado: contexto efetivamente infinito com custo de IO proporcional
/// à entropia semântica do histórico (não ao seu tamanho bruto).

use std::{
    collections::VecDeque,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};
use nodestor_core::NodeStorError;
use crate::hnsw_index::HnswIndex;

// ─── KV Block ─────────────────────────────────────────────────────────────────

/// One block of archived KV states for a contiguous range of token positions.
#[derive(Clone, Debug)]
pub struct KvBlock {
    /// Token position range [start, end).
    pub pos_start:    u64,
    pub pos_end:      u64,
    /// Number of layers.
    pub n_layers:     u32,
    /// Number of KV heads per layer.
    pub n_kv_heads:   u32,
    /// Head dimension.
    pub head_dim:     u32,
    /// Raw KV data: [n_layers][n_kv_heads][seq_len][head_dim] × 2 (K and V), F16.
    pub data:         Vec<u8>,
    /// Semantic embedding of this block (mean-pooled last-layer hidden states).
    pub embedding:    Vec<f32>,
}

impl KvBlock {
    pub fn seq_len(&self) -> u64 { self.pos_end - self.pos_start }

    pub fn raw_bytes(&self) -> usize {
        let seq = self.seq_len() as usize;
        self.n_layers as usize
            * self.n_kv_heads as usize
            * seq
            * self.head_dim as usize
            * 2   // K and V
            * 2   // F16 = 2 bytes
    }
}

// ─── Disk archive entry ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct ArchiveEntry {
    offset:            u64,
    compressed_bytes:  u64,
    raw_bytes:         u64,
    pos_start:         u64,
    pos_end:           u64,
    embedding_dim:     u32,
    // Embedding stored inline in the index — not here
}

// ─── Flat vector index (cosine similarity) ────────────────────────────────────
//
// For contexts up to ~100K tokens (a few thousand archived blocks), brute-force
// cosine search is fast enough (<1ms). Beyond that, replace with an HNSW index:
// the interface is identical (insert / query), only the struct changes.

struct FlatVectorIndex {
    /// (block_id, embedding) pairs.
    entries: Vec<(u64, Vec<f32>)>,
    dim:     usize,
}

impl FlatVectorIndex {
    fn new(dim: usize) -> Self {
        Self { entries: Vec::new(), dim }
    }

    fn insert(&mut self, block_id: u64, embedding: Vec<f32>) {
        debug_assert_eq!(embedding.len(), self.dim);
        self.entries.push((block_id, embedding));
    }

    /// Returns the top-K block IDs by cosine similarity to `query`.
    fn query(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let q_norm = l2_norm(query);
        if q_norm < 1e-8 { return Vec::new(); }

        let mut scored: Vec<(u64, f32)> = self.entries.iter()
            .map(|(id, emb)| {
                let sim = dot(query, emb) / (q_norm * l2_norm(emb)).max(1e-8);
                (*id, sim)
            }).collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(top_k);
        scored
    }

    fn len(&self) -> usize { self.entries.len() }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ─── LZ4-style compressor (byte-level RLE for KV data) ───────────────────────
//
// Pure-Rust placeholder compression. KV cache data is float16 (semi-random),
// so typical LZ4 achieves 1.3–1.8× on it. For the GPU path, GDeflate is
// better but requires the DirectStorage API. This runs on CPU for the archive.

fn compress_kv(data: &[u8]) -> Vec<u8> {
    // Simple run-length encoding as placeholder.
    // Replace with miniz_oxide::deflate or lz4_flex in production.
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == b && run < 255 {
            run += 1;
        }
        if run > 2 {
            out.push(0xFF);       // escape
            out.push(run as u8);
            out.push(b);
            i += run;
        } else {
            if b == 0xFF {
                out.push(0xFF);   // escape literal
                out.push(1);
                out.push(b);
            } else {
                out.push(b);
            }
            i += 1;
        }
    }
    out
}

fn decompress_kv(compressed: &[u8], expected_raw: usize) -> Result<Vec<u8>, NodeStorError> {
    let mut out = Vec::with_capacity(expected_raw);
    let mut i = 0;
    while i < compressed.len() {
        let b = compressed[i];
        if b == 0xFF {
            i += 1;
            if i + 1 >= compressed.len() {
                return Err(NodeStorError::InferenceError(
                    "context_ssd: decompress truncated".into()
                ));
            }
            let run = compressed[i] as usize;
            let val = compressed[i + 1];
            for _ in 0..run { out.push(val); }
            i += 2;
        } else {
            out.push(b);
            i += 1;
        }
    }
    if out.len() != expected_raw {
        return Err(NodeStorError::InferenceError(format!(
            "context_ssd: decompress size mismatch: got {} expected {}",
            out.len(), expected_raw
        )));
    }
    Ok(out)
}

// ─── ContextSsd ───────────────────────────────────────────────────────────────

pub struct ContextSsdConfig {
    /// Path for the KV archive file.
    pub archive_path:    PathBuf,
    /// Max tokens to keep in RAM before offloading to SSD.
    pub active_capacity: u64,
    /// Block size: how many tokens per archived KV block.
    pub block_size:      u64,
    /// How many archived blocks to retrieve per attention query.
    pub top_k_retrieve:  usize,
    /// Embedding dimension (= d_model of the model).
    pub embedding_dim:   usize,
    /// Model KV dimensions.
    pub n_layers:        u32,
    pub n_kv_heads:      u32,
    pub head_dim:        u32,
}

impl Default for ContextSsdConfig {
    fn default() -> Self {
        Self {
            archive_path:    PathBuf::from("context_archive.kva"),
            active_capacity: 4096,
            block_size:      512,
            top_k_retrieve:  4,
            embedding_dim:   4096,
            n_layers:        32,
            n_kv_heads:      8,
            head_dim:        128,
        }
    }
}

pub struct ContextSsd {
    cfg:      ContextSsdConfig,
    /// Active KV in RAM: ring buffer of (position, hidden_state, kv_data).
    active:   VecDeque<ActiveToken>,
    /// HNSW vector index over archived blocks — O(log n) cosine search.
    index:    HnswIndex,
    /// Disk archive entries by block_id.
    archive:  Vec<ArchiveEntry>,
    /// Embeddings parallel to archive (block_id → mean embedding).
    archive_embeddings: Vec<Vec<f32>>,
    /// Archive file handle.
    file:     File,
    next_offset: u64,
    next_block_id: u64,

    // Stats
    pub tokens_active:   u64,
    pub blocks_archived: u64,
    pub retrievals:      u64,
    pub io_read_bytes:   u64,
    pub io_write_bytes:  u64,
}

struct ActiveToken {
    position:    u64,
    embedding:   Vec<f32>,   // last-layer hidden state (for indexing)
    kv_data:     Vec<u8>,    // raw F16 KV for all layers, this position
}

impl ContextSsd {
    pub fn new(cfg: ContextSsdConfig) -> Result<Self, NodeStorError> {
        let file = OpenOptions::new()
            .create(true).read(true).write(true)
            .open(&cfg.archive_path)
            .map_err(|e| NodeStorError::InferenceError(
                format!("context_ssd: open archive {}: {}", cfg.archive_path.display(), e)
            ))?;

        let dim = cfg.embedding_dim;
        Ok(Self {
            cfg,
            active:          VecDeque::new(),
            index:           HnswIndex::with_params(dim, 16, 200, 50),
            archive:         Vec::new(),
            archive_embeddings: Vec::new(),
            file,
            next_offset:     0,
            next_block_id:   0,
            tokens_active:   0,
            blocks_archived: 0,
            retrievals:      0,
            io_read_bytes:   0,
            io_write_bytes:  0,
        })
    }

    /// Push one new token's KV state into the active buffer.
    ///
    /// `position`: absolute token position in the sequence.
    /// `embedding`: last-layer hidden state (d_model floats).
    /// `kv_data`: raw KV bytes for all layers at this position (F16).
    pub fn push_token(
        &mut self,
        position:  u64,
        embedding: Vec<f32>,
        kv_data:   Vec<u8>,
    ) -> Result<(), NodeStorError> {
        self.active.push_back(ActiveToken { position, embedding, kv_data });
        self.tokens_active += 1;

        // Offload oldest block when active buffer exceeds capacity
        if self.active.len() as u64 >= self.cfg.active_capacity + self.cfg.block_size {
            self.offload_oldest_block()?;
        }
        Ok(())
    }

    /// Offload the oldest `block_size` tokens from RAM to SSD.
    fn offload_oldest_block(&mut self) -> Result<(), NodeStorError> {
        let bs = self.cfg.block_size as usize;
        if self.active.len() < bs { return Ok(()); }

        // Collect tokens for this block
        let mut block_tokens: Vec<ActiveToken> = Vec::with_capacity(bs);
        for _ in 0..bs {
            if let Some(tok) = self.active.pop_front() {
                block_tokens.push(tok);
            }
        }

        let pos_start = block_tokens.first().map(|t| t.position).unwrap_or(0);
        let pos_end   = block_tokens.last().map(|t| t.position + 1).unwrap_or(0);

        // Mean-pool embeddings for the block's semantic vector
        let dim = self.cfg.embedding_dim;
        let mut mean_emb = vec![0.0f32; dim];
        for tok in &block_tokens {
            for (a, b) in mean_emb.iter_mut().zip(tok.embedding.iter()) {
                *a += b;
            }
        }
        let n = block_tokens.len() as f32;
        for x in &mut mean_emb { *x /= n; }

        // Concatenate raw KV data
        let raw_kv: Vec<u8> = block_tokens.into_iter()
            .flat_map(|t| t.kv_data.into_iter())
            .collect();
        let raw_bytes = raw_kv.len();

        // Compress and write to SSD
        let compressed = compress_kv(&raw_kv);
        let compressed_bytes = compressed.len();

        self.file.seek(SeekFrom::Start(self.next_offset))
            .map_err(|e| NodeStorError::InferenceError(format!("context_ssd seek: {}", e)))?;
        self.file.write_all(&compressed)
            .map_err(|e| NodeStorError::InferenceError(format!("context_ssd write: {}", e)))?;

        self.io_write_bytes += compressed_bytes as u64;

        let block_id = self.next_block_id;
        self.archive.push(ArchiveEntry {
            offset:           self.next_offset,
            compressed_bytes: compressed_bytes as u64,
            raw_bytes:        raw_bytes as u64,
            pos_start,
            pos_end,
            embedding_dim:    dim as u32,
        });

        self.next_offset    += compressed_bytes as u64;
        self.next_block_id  += 1;
        self.blocks_archived += 1;

        // Insert into HNSW index + parallel embedding store
        self.archive_embeddings.push(mean_emb.clone());
        self.index.insert(block_id, mean_emb);
        Ok(())
    }

    /// Retrieve the top-K archived KV blocks most semantically similar to `query`.
    ///
    /// Used during attention: the returned blocks are injected as extra KV pairs,
    /// extending the effective context window without truncation.
    pub fn retrieve_relevant(
        &mut self,
        query_embedding: &[f32],
    ) -> Result<Vec<KvBlock>, NodeStorError> {
        let k = self.cfg.top_k_retrieve;
        let matches = self.index.query(query_embedding, k);
        self.retrievals += 1;

        let mut blocks = Vec::with_capacity(matches.len());
        for (block_id, _score) in matches {
            let entry = match self.archive.get(block_id as usize) {
                Some(e) => *e,
                None    => continue,
            };

            self.file.seek(SeekFrom::Start(entry.offset))
                .map_err(|e| NodeStorError::InferenceError(
                    format!("context_ssd read seek: {}", e)
                ))?;

            let mut compressed = vec![0u8; entry.compressed_bytes as usize];
            self.file.read_exact(&mut compressed)
                .map_err(|e| NodeStorError::InferenceError(
                    format!("context_ssd read: {}", e)
                ))?;

            self.io_read_bytes += entry.compressed_bytes;

            let raw = decompress_kv(&compressed, entry.raw_bytes as usize)?;

            // Embedding is stored in the archive alongside the entry
            let embedding = self.archive_embeddings
                .get(block_id as usize)
                .cloned()
                .unwrap_or_default();

            let seq_len = (entry.pos_end - entry.pos_start) as u32;
            blocks.push(KvBlock {
                pos_start:  entry.pos_start,
                pos_end:    entry.pos_end,
                n_layers:   self.cfg.n_layers,
                n_kv_heads: self.cfg.n_kv_heads,
                head_dim:   self.cfg.head_dim,
                data:       raw,
                embedding,
            });
        }
        Ok(blocks)
    }

    /// Returns active KV data (in RAM) as a flat byte slice reference.
    pub fn active_kv_iter(&self) -> impl Iterator<Item = (u64, &[u8])> {
        self.active.iter().map(|t| (t.position, t.kv_data.as_slice()))
    }

    pub fn active_len(&self)   -> usize { self.active.len() }
    pub fn archived_blocks(&self) -> usize { self.archive.len() }
    pub fn total_tokens(&self) -> u64 {
        self.tokens_active
            + self.blocks_archived * self.cfg.block_size
    }

    pub fn stats(&self) -> ContextStats {
        ContextStats {
            active_tokens:   self.active.len() as u64,
            archived_blocks: self.blocks_archived,
            index_entries:   self.index.len() as u64,
            retrievals:      self.retrievals,
            io_read_mb:      self.io_read_bytes  as f64 / 1_048_576.0,
            io_write_mb:     self.io_write_bytes as f64 / 1_048_576.0,
        }
    }
}

#[derive(Debug)]
pub struct ContextStats {
    pub active_tokens:   u64,
    pub archived_blocks: u64,
    pub index_entries:   u64,
    pub retrievals:      u64,
    pub io_read_mb:      f64,
    pub io_write_mb:     f64,
}

impl std::fmt::Display for ContextStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "active={} archived_blocks={} idx={} retrievals={} r={:.1}MB w={:.1}MB",
            self.active_tokens, self.archived_blocks, self.index_entries,
            self.retrievals, self.io_read_mb, self.io_write_mb,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_cfg(dir: &std::path::Path, active_cap: u64, block_sz: u64) -> ContextSsdConfig {
        ContextSsdConfig {
            archive_path:    dir.join("test.kva"),
            active_capacity: active_cap,
            block_size:      block_sz,
            top_k_retrieve:  2,
            embedding_dim:   8,
            n_layers:        2,
            n_kv_heads:      2,
            head_dim:        4,
        }
    }

    #[test]
    fn push_and_offload() {
        let dir = tempdir().unwrap();
        let mut ctx = ContextSsd::new(make_cfg(dir.path(), 4, 2)).unwrap();

        // Push 6 tokens — should trigger 1 offload (block of 2)
        for i in 0u64..6 {
            let emb = vec![i as f32; 8];
            let kv  = vec![i as u8; 64]; // 2 layers × 2 heads × 4 dim × 2 (KV) × 2 (F16)
            ctx.push_token(i, emb, kv).unwrap();
        }

        assert!(ctx.archived_blocks() >= 1, "Expected at least 1 archived block");
        assert!(ctx.active_len() < 7);
    }

    #[test]
    fn retrieve_relevant_blocks() {
        let dir = tempdir().unwrap();
        let mut ctx = ContextSsd::new(make_cfg(dir.path(), 2, 2)).unwrap();

        // Token 0: embedding close to [1,0,0,...] (topic A)
        // Token 1: embedding close to [0,1,0,...] (topic B)
        let emb_a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let emb_b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        ctx.push_token(0, emb_a.clone(), vec![0u8; 64]).unwrap();
        ctx.push_token(1, emb_b.clone(), vec![1u8; 64]).unwrap();
        // These two form a block and get offloaded when 2 more arrive
        ctx.push_token(2, vec![0.5f32; 8], vec![2u8; 64]).unwrap();
        ctx.push_token(3, vec![0.5f32; 8], vec![3u8; 64]).unwrap();

        // Query with topic-A vector → should retrieve the block containing token 0
        let result = ctx.retrieve_relevant(&emb_a).unwrap();
        assert!(!result.is_empty(), "Should retrieve at least one block");
        // The returned block should contain position 0 or 1
        let any_match = result.iter().any(|b| b.pos_start <= 1);
        assert!(any_match, "Retrieved block should cover early positions");
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).flat_map(|i| vec![i as u8; 4]).collect();
        let compressed = compress_kv(&data);
        let recovered  = decompress_kv(&compressed, data.len()).unwrap();
        assert_eq!(recovered, data);
        // Should have compressed (runs of identical bytes)
        assert!(compressed.len() < data.len(), "Compression should reduce size");
    }

    #[test]
    fn hnsw_cosine_in_context() {
        use crate::hnsw_index::HnswIndex;
        let mut idx = HnswIndex::new(4);
        idx.insert(0, vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(1, vec![0.0, 1.0, 0.0, 0.0]);
        idx.insert(2, vec![0.0, 0.0, 1.0, 0.0]);

        let results = idx.query(&[0.9, 0.1, 0.0, 0.0], 1);
        assert_eq!(results[0].0, 0);

        let results = idx.query(&[0.1, 0.9, 0.0, 0.0], 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn stats_display() {
        let s = ContextStats {
            active_tokens: 512, archived_blocks: 8, index_entries: 8,
            retrievals: 32, io_read_mb: 12.4, io_write_mb: 3.1,
        };
        let txt = format!("{}", s);
        assert!(txt.contains("active=512"));
        assert!(txt.contains("archived_blocks=8"));
    }
}
