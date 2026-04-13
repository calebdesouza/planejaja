//! nodestor-fabric — NVMe-oF (NVMe over Fabrics) com RDMA simulado
//!
//! ## O que é NVMe-oF?
//!
//! Permite que GPUs em máquinas clientes "puxem" tensores de IA de um servidor
//! central de SSDs NVMe pela rede, **sem que a CPU processe os dados**.
//!
//! ```text
//!  ┌─────────────────────────────────────────────────┐
//!  │  COFRE CENTRAL (Servidor NVMe-oF)               │
//!  │  ╔═══════════╗ ╔═══════════╗ ╔═══════════╗      │
//!  │  ║ SSD NVMe  ║ ║ SSD NVMe  ║ ║ SSD NVMe  ║      │
//!  │  ║ Gen 5     ║ ║ Gen 5     ║ ║ Gen 5     ║      │
//!  │  ╚═══════════╝ ╚═══════════╝ ╚═══════════╝      │
//!  │         ↕ NVMe-oF (TCP/RDMA)                    │
//!  └─────────────────────────────────────────────────┘
//!           ↕ 100 GbE / InfiniBand RoCEv2
//!  ┌─────────────────────────────────────────────────┐
//!  │  GPU CLIENTS (Workers)                          │
//!  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
//!  │  │ GPU Node │ │ GPU Node │ │ GPU Node │        │
//!  │  │ VRAM←DMA │ │ VRAM←DMA │ │ VRAM←DMA │        │
//!  │  └──────────┘ └──────────┘ └──────────┘        │
//!  └─────────────────────────────────────────────────┘
//! ```
//!
//! ## CPU Usage < 10%
//!
//! RDMA (Remote Direct Memory Access) permite que o NIC (placa de rede) mova
//! os dados **diretamente** da VRAM do servidor para a VRAM do cliente,
//! sem envolver as CPUs em nenhuma das máquinas.

pub mod fabric_client;
pub mod fabric_server;
pub mod rdma_sim;
pub mod nvmeof_transport;
pub mod fabric_registry;

pub use fabric_client::FabricClient;
pub use fabric_server::FabricServer;
pub use rdma_sim::{RdmaChannel, RdmaStats};
pub use nvmeof_transport::NvmeOfTransport;
pub use fabric_registry::FabricRegistry;
