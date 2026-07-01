//! nodestor-metadata — Motor de metadados com LanceDB + DiskANN.
//!
//! ## Status
//! - [x] Estrutura do crate
//! - [x] `indexer.rs` — indexação de offsets de tensores
//! - [x] `search.rs` — interface RAG (VectorSearch) sobre o VectorStore nativo
//! - [x] `vector_store.rs` — banco vetorial híbrido (HNSW + BM25 + RRF), Rust puro
//! - [ ] `quantization.rs` — Binary quantization (1TB → MBs na RAM)

pub mod indexer;
pub mod search;
pub mod vector_store;

pub use vector_store::VectorStore;
