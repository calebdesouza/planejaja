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
pub mod metralhadora;
pub mod scheduler;

pub use buffer_pool::{BufferPool, PooledBuffer};
pub use metralhadora::{MesPrefetchQueue, PrefetchedBlock};
pub use scheduler::StreamScheduler;
