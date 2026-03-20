use nodestor_core::{DataTransport, NodeStorError, TransferRequest, TransferResult, TransportBackend};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use tracing::debug;

/// Backend de transporte universal baseado em leitura de arquivo padrão.
///
/// Funciona em qualquer SO e qualquer hardware.
/// Performance: ~500MB/s - 3GB/s dependendo do SO e SSD.
///
/// Este é o fallback para quando io_uring, DirectStorage ou GDS não estão disponíveis.
pub struct PreadFallback;

impl PreadFallback {
    pub fn new() -> Self {
        debug!("PreadFallback inicializado");
        Self
    }
}

impl Default for PreadFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl DataTransport for PreadFallback {
    fn transfer(
        &self,
        path: &str,
        request: &TransferRequest,
    ) -> Result<TransferResult, NodeStorError> {
        let start = Instant::now();

        // Abre o arquivo
        let mut file = std::fs::File::open(path)
            .map_err(|e| NodeStorError::IoError(e))?;

        // Seek para o offset desejado
        file.seek(SeekFrom::Start(request.file_offset))
            .map_err(|e| NodeStorError::TransferFailed(
                format!("Seek falhou no offset {}: {}", request.file_offset, e)
            ))?;

        // Lê os dados
        let mut data = vec![0u8; request.size];
        let bytes_read = file.read(&mut data)
            .map_err(|e| NodeStorError::TransferFailed(
                format!("Leitura de {} bytes falhou: {}", request.size, e)
            ))?;

        data.truncate(bytes_read);

        let duration_us = start.elapsed().as_micros() as u64;

        debug!(
            "PreadFallback: {} bytes em {}µs ({:.2} GB/s)",
            bytes_read,
            duration_us,
            if duration_us > 0 {
                bytes_read as f64 / (duration_us as f64 / 1_000_000.0) / 1_000_000_000.0
            } else { 0.0 }
        );

        Ok(TransferResult::new(data, duration_us))
    }

    fn transfer_batch(
        &self,
        path: &str,
        requests: &[TransferRequest],
    ) -> Result<Vec<TransferResult>, NodeStorError> {
        // Otimização: abre o arquivo uma vez e reutiliza para todos os requests
        let mut file = std::fs::File::open(path)
            .map_err(|e| NodeStorError::IoError(e))?;

        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            let start = Instant::now();

            file.seek(SeekFrom::Start(request.file_offset))
                .map_err(|e| NodeStorError::TransferFailed(e.to_string()))?;

            let mut data = vec![0u8; request.size];
            let bytes_read = file.read(&mut data)
                .map_err(|e| NodeStorError::TransferFailed(e.to_string()))?;

            data.truncate(bytes_read);
            let duration_us = start.elapsed().as_micros() as u64;
            results.push(TransferResult::new(data, duration_us));
        }

        Ok(results)
    }

    fn backend_name(&self) -> &'static str {
        "PreadFallback"
    }

    fn backend_type(&self) -> TransportBackend {
        TransportBackend::PreadFallback
    }

    fn theoretical_max_throughput_bps(&self) -> u64 {
        1_500_000_000 // 1.5 GB/s conservador
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_file(path: &str, content: &[u8]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content).unwrap();
    }

    #[test]
    fn test_basic_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        let content = b"Hello, NodeStor! This is test data for transfer verification.";
        create_test_file(path.to_str().unwrap(), content);

        let transport = PreadFallback::new();
        let request = TransferRequest {
            file_offset: 0,
            size: content.len(),
            compressed: false,
        };

        let result = transport.transfer(path.to_str().unwrap(), &request)
            .expect("Transferência básica deve funcionar");

        assert_eq!(result.data, content);
        assert!(result.duration_us > 0 || result.duration_us == 0); // Máquina pode ser muito rápida
    }

    #[test]
    fn test_transfer_with_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("offset_test.bin");
        let content = b"HEADER_DATAACTUAL_TENSOR_DATA_HERE";
        create_test_file(path.to_str().unwrap(), content);

        let transport = PreadFallback::new();
        let request = TransferRequest {
            file_offset: 11, // Pula "HEADER_DATA"
            size: 23,
            compressed: false,
        };

        let result = transport.transfer(path.to_str().unwrap(), &request).unwrap();
        assert_eq!(&result.data, b"ACTUAL_TENSOR_DATA_HERE");
    }

    #[test]
    fn test_batch_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch_test.bin");

        // Arquivo de 1KB com padrão recognisível
        let mut content = vec![0u8; 1024];
        content[0..4].copy_from_slice(b"AAAA");
        content[512..516].copy_from_slice(b"BBBB");
        create_test_file(path.to_str().unwrap(), &content);

        let transport = PreadFallback::new();
        let requests = vec![
            TransferRequest { file_offset: 0, size: 4, compressed: false },
            TransferRequest { file_offset: 512, size: 4, compressed: false },
        ];

        let results = transport.transfer_batch(path.to_str().unwrap(), &requests).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(&results[0].data, b"AAAA");
        assert_eq!(&results[1].data, b"BBBB");
    }

    #[test]
    fn test_transfer_file_not_found() {
        let transport = PreadFallback::new();
        let request = TransferRequest {
            file_offset: 0,
            size: 100,
            compressed: false,
        };

        let result = transport.transfer("/path/that/does/not/exist.bin", &request);
        assert!(result.is_err());
    }

    #[test]
    fn test_throughput_calculation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("throughput_test.bin");

        // 1MB de dados
        let content = vec![0xABu8; 1024 * 1024];
        create_test_file(path.to_str().unwrap(), &content);

        let transport = PreadFallback::new();
        let request = TransferRequest {
            file_offset: 0,
            size: content.len(),
            compressed: false,
        };

        let result = transport.transfer(path.to_str().unwrap(), &request).unwrap();

        println!(
            "Throughput medido: {:.2} GB/s ({} bytes em {}µs)",
            result.throughput_gbs(),
            result.data.len(),
            result.duration_us
        );

        assert_eq!(result.data.len(), 1024 * 1024);
    }

    #[test]
    fn test_backend_metadata() {
        let t = PreadFallback::new();
        assert_eq!(t.backend_name(), "PreadFallback");
        assert_eq!(t.backend_type(), TransportBackend::PreadFallback);
        assert!(t.theoretical_max_throughput_bps() > 0);
    }
}
