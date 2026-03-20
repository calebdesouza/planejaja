//! nodestor-core — Tipos centrais, traits e erros do NodeStor.
//! Todos os outros crates dependem deste.

pub mod error;
pub mod types;
pub mod traits;

pub use error::NodeStorError;
pub use types::*;
pub use traits::*;
