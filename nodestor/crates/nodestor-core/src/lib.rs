//! nodestor-core — Tipos centrais, traits e erros do NodeStor.
//! Todos os outros crates dependem deste.

pub mod error;
pub mod types;
pub mod traits;
pub mod telemetry;
pub mod config;


pub use error::NodeStorError;
pub use types::*;
pub use traits::*;
pub use telemetry::*;
pub use config::*;

