use nodestor_core::{DataTransport, NodeStorError};
use nodestor_core::traits::DataTransport as _; // Ensure trait in scope

/// io_uring transport para Linux kernel 5.11+
/// Usa o crate `io-uring` para submissão assíncrona de I/O

use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use nodestor_core::{TransferRequest, TransferResult, TransportBackend};
use tracing::{debug, info};

pub struct IoUringTransport {
    _ring: io_uring::IoUring,
}

impl IoUringTransport {
    pub fn new() -> Result<Self, NodeStorError> {
        let ring = io_uring::IoUring::new(128)
            .map_err(|e| NodeStorError::NotSupported(
                format!("io_uring não disponível: {}", e)
            ))?;
        info!("io_uring transport inicializado com queue depth 128");
        Ok(Self { _ring: ring })
    }
}

impl DataTransport for IoUringTransport {
    fn transfer(
        &self,
        path: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError> {
        // Implementação simplificada: usa read padrão por enquanto
        // A implementação completa usaria io_uring SQE submission
        let start = Instant::now();

        let mut file = std::fs::File::open(path).map_err(NodeStorError::IoError)?;
        file.seek(SeekFrom::Start(request.file_offset))
            .map_err(NodeStorError::IoError)?;

        let mut data = vec![0u8; request.size];
        let n = file.read(&mut data).map_err(NodeStorError::IoError)?;
        data.truncate(n);

        let duration_us = start.elapsed().as_micros() as u64;
        debug!("io_uring transport: {} bytes em {}µs", n, duration_us);
        Ok(TransferResult::new(data, duration_us))
    }

    fn backend_name(&self) -> &'static str {
        "IoUringTransport"
    }

    fn backend_type(&self) -> TransportBackend {
        TransportBackend::IoUringStandard
    }

    fn theoretical_max_throughput_bps(&self) -> u64 {
        7_000_000_000 // 7 GB/s com io_uring
    }
}
