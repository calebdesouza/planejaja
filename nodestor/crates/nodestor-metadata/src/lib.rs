//! nodestor-metadata — Motor de metadados com LanceDB + DiskANN.
//!
//! ## Status
//! - [x] Estrutura do crate
//! - [ ] `indexer.rs` — indexação de offsets de tensores
//! - [ ] `search.rs` — busca vetorial (DiskANN/HNSW)
//! - [ ] `quantization.rs` — Binary quantization (1TB → MBs na RAM)

pub mod indexer;
pub mod search;
