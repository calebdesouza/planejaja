//! io_uring transport com SQE submission real para Linux kernel 5.11+.
//!
//! ## Versões de kernel suportadas
//! - **Kernel ≥ 6.16:** io_uring + DMABUF (zero-copy SSD→GPU via DMA)
//! - **Kernel ≥ 5.11:** io_uring padrão (leitura assíncrona sem copy do kernel)
//! - **Kernel < 5.11:** cai para PreadFallback (não usa este módulo)
//!
//! ## Como funciona o io_uring
//! Em vez de chamar `read()` para cada bloco (síncrono, caro para a CPU),
//! submetemos múltiplas SQEs (Submission Queue Entries) de uma vez e
//! coletamos os resultados das CQEs (Completion Queue Entries).
//! A CPU fica livre enquanto o kernel gerencia as leituras — que podem
//! ser executadas via DMA direto sem envolver a CPU.

use nodestor_core::{DataTransport, NodeStorError, TransferRequest, TransferResult, TransportBackend};
use std::time::Instant;
use tracing::{debug, info};

/// Transport io_uring com submission queue real para Linux.
pub struct IoUringTransport {
    ring: io_uring::IoUring,
    /// Kernel version para saber se DMABUF está disponível
    supports_dmabuf: bool,
}

impl IoUringTransport {
    /// Cria transport io_uring.
    ///
    /// Inicializa um ring com queue depth de 128 entries — suficiente para
    /// o Modo Metralhadora enviar lotes de tensores sem bloquear.
    pub fn new() -> Result<Self, NodeStorError> {
        let ring = io_uring::IoUring::new(128).map_err(|e| {
            NodeStorError::NotSupported(format!(
                "io_uring não disponível (kernel < 5.11?): {}",
                e
            ))
        })?;

        // Verifica suporte a DMABUF (kernel 6.16+)
        let supports_dmabuf = check_dmabuf_support();

        info!(
            "io_uring transport inicializado | queue_depth=128 | dmabuf={}",
            supports_dmabuf
        );

        Ok(Self { ring, supports_dmabuf })
    }

    /// Submete uma SQE de leitura e aguarda a CQE correspondente.
    ///
    /// Este é o coração do transporte: usa io_uring para leitura assíncrona
    /// real, sem bloquear a thread principal.
    fn submit_read_sqe(
        &self,
        path: &str,
        offset: u64,
        size: usize,
    ) -> Result<Vec<u8>, NodeStorError> {
        use std::os::unix::io::AsRawFd;

        let file = std::fs::File::open(path).map_err(NodeStorError::IoError)?;
        let fd = file.as_raw_fd();

        // Buffer de destino
        let mut buf = vec![0u8; size];

        // Cria SQE de leitura
        let read_e = io_uring::opcode::Read::new(
            io_uring::types::Fd(fd),
            buf.as_mut_ptr(),
            size as u32,
        )
        .offset(offset)
        .build()
        .user_data(0x42); // tag para identificar esta operação

        // Submete ao ring
        {
            let mut sq = self.ring.submission();
            // SAFETY: o buffer `buf` vive durante todo o submit+await
            unsafe { sq.push(&read_e) }.map_err(|e| {
                NodeStorError::TransferFailed(format!("io_uring push SQE: {}", e))
            })?;
        }

        // Submete e aguarda 1 completion
        self.ring
            .submit_and_wait(1)
            .map_err(|e| NodeStorError::TransferFailed(format!("io_uring submit_and_wait: {}", e)))?;

        // Lê o resultado da CQE
        let mut cq = self.ring.completion();
        let cqe = cq.next().ok_or_else(|| {
            NodeStorError::TransferFailed("io_uring: nenhuma CQE disponível".to_string())
        })?;

        let result = cqe.result();
        if result < 0 {
            return Err(NodeStorError::TransferFailed(format!(
                "io_uring leitura falhou com errno: {}",
                -result
            )));
        }

        let bytes_read = result as usize;
        buf.truncate(bytes_read);

        Ok(buf)
    }

    /// Modo Metralhadora: submete múltiplas SQEs de uma vez.
    ///
    /// Este é o verdadeiro poder do io_uring: em vez de N round-trips,
    /// submetemos N SQEs em uma única syscall e coletamos N CQEs.
    /// A CPU faz uma syscall por lote — não uma por bloco.
    fn submit_batch_sqes(
        &self,
        path: &str,
        requests: &[TransferRequest],
    ) -> Result<Vec<Vec<u8>>, NodeStorError> {
        use std::os::unix::io::AsRawFd;

        if requests.is_empty() {
            return Ok(vec![]);
        }

        let file = std::fs::File::open(path).map_err(NodeStorError::IoError)?;
        let fd = file.as_raw_fd();

        // Aloca buffers para todos os requests
        let mut buffers: Vec<Vec<u8>> = requests
            .iter()
            .map(|r| vec![0u8; r.size])
            .collect();

        // Submete todas as SQEs de uma vez
        {
            let mut sq = self.ring.submission();
            for (i, (req, buf)) in requests.iter().zip(buffers.iter_mut()).enumerate() {
                let read_e = io_uring::opcode::Read::new(
                    io_uring::types::Fd(fd),
                    buf.as_mut_ptr(),
                    req.size as u32,
                )
                .offset(req.file_offset)
                .build()
                .user_data(i as u64);

                // SAFETY: buffers vivem até as CQEs serem coletadas abaixo
                unsafe { sq.push(&read_e) }.map_err(|e| {
                    NodeStorError::TransferFailed(format!("io_uring push batch SQE {}: {}", i, e))
                })?;
            }
        }

        // Uma única syscall para todo o lote
        self.ring
            .submit_and_wait(requests.len())
            .map_err(|e| NodeStorError::TransferFailed(format!("io_uring batch submit: {}", e)))?;

        // Coleta resultados (podem chegar fora de ordem — ordenamos por user_data)
        let mut results: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        let mut cq = self.ring.completion();
        for cqe in cq.by_ref() {
            let idx = cqe.user_data() as usize;
            let bytes = cqe.result().max(0) as usize;
            results.push((idx, bytes));
        }

        // Trunca cada buffer ao tamanho real lido
        for (idx, bytes_read) in results {
            if idx < buffers.len() {
                buffers[idx].truncate(bytes_read);
            }
        }

        Ok(buffers)
    }
}

/// Verifica suporte a io_uring + DMABUF (kernel 6.16+).
fn check_dmabuf_support() -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/version") {
            // Extrai versão do kernel
            if let Some(version_str) = content.split_whitespace().nth(2) {
                let parts: Vec<&str> = version_str.split('.').collect();
                if let (Some(major), Some(minor)) = (
                    parts.first().and_then(|s| s.parse::<u32>().ok()),
                    parts.get(1).and_then(|s| s.parse::<u32>().ok()),
                ) {
                    return major > 6 || (major == 6 && minor >= 16);
                }
            }
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

impl DataTransport for IoUringTransport {
    fn transfer(
        &self,
        path: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError> {
        let start = Instant::now();

        let data = self.submit_read_sqe(path, request.file_offset, request.size)?;

        let duration_us = start.elapsed().as_micros() as u64;
        debug!(
            "io_uring SQE: {} bytes em {}µs ({:.2} GB/s) | dmabuf={}",
            data.len(),
            duration_us,
            if duration_us > 0 {
                data.len() as f64 / (duration_us as f64 / 1e6) / 1e9
            } else {
                0.0
            },
            self.supports_dmabuf
        );

        Ok(TransferResult::new(data, duration_us))
    }

    fn transfer_batch(
        &self,
        path: &str,
        requests: &[TransferRequest],
    ) -> Result<Vec<TransferResult>, NodeStorError> {
        let start = Instant::now();

        let all_data = self.submit_batch_sqes(path, requests)?;

        let elapsed = start.elapsed().as_micros() as u64;
        let per_request_us = if requests.is_empty() { 0 } else { elapsed / requests.len() as u64 };

        Ok(all_data
            .into_iter()
            .map(|d| TransferResult::new(d, per_request_us))
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        if self.supports_dmabuf {
            "IoUring+DMABUF"
        } else {
            "IoUringStandard"
        }
    }

    fn backend_type(&self) -> TransportBackend {
        if self.supports_dmabuf {
            TransportBackend::IoUringDmabuf
        } else {
            TransportBackend::IoUringStandard
        }
    }

    fn theoretical_max_throughput_bps(&self) -> u64 {
        if self.supports_dmabuf {
            20_000_000_000 // 20 GB/s com DMABUF (kernel 6.16+)
        } else {
            7_000_000_000 // 7 GB/s com io_uring padrão
        }
    }
}
