//! nodestor-inference — Pipeline de inferência end-to-end.
//!
//! Orquestra: scanner → transport → streaming kernel → modelo → tokens.
//!
//! ## Status
//! - [x] Estrutura do crate
//! - [ ] `pipeline.rs` — orquestração do fluxo
//! - [ ] `prefetch.rs` — DiskANN pre-fetch integrado

pub mod pipeline;
