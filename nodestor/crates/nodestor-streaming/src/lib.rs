//! nodestor-streaming — Tensor Streaming Kernel.
//!
//! Implementa o pipeline assíncrono de streaming de tensores SSD → GPU:
//! - Double/Triple buffering de VRAM
//! - Modo Metralhadora (batch submission de SQEs)
//! - Pre-fetching preditivo integrado ao DiskANN
//!
//! ## Status
//! - [x] Estrutura do crate e dependências
//! - [ ] `scheduler.rs` — agendador de fatias de tensor
//! - [ ] `buffer_pool.rs` — pool de GPU buffers com double buffering
//! - [ ] `metralhadora.rs` — batch submission assíncrono

pub mod buffer_pool;
pub mod scheduler;
pub mod metralhadora;
pub mod liquid;
pub mod layer_graph;
pub mod speculative;
pub mod apex;
pub mod burst_reader;
pub use buffer_pool::{BufferPool, PooledBuffer};
pub use metralhadora::{MesPrefetchQueue, PrefetchedBlock};
pub use scheduler::BurstScheduler;
pub use apex::{ApexOrchestrator, ApexStreamStats, ApexStats, TransportRoute};
pub use burst_reader::BurstReader;
