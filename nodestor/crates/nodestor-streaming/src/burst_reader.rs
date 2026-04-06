//! Leitura Multi-Thread em Explosão (Burst Reader)
//!
//! Exposto sobre a fundação do Direct I/O, estende as capacidades NVMe para múltiplos
//! canais de I/O em paralelo. Quando um ExecutionPlan agrupa vários tensores num mesmo pass,
//! ele cria handlers `O_DIRECT`/`FILE_FLAG_NO_BUFFERING` separados em threads nativas (scopadas),
//! forçando o firmware do SSD M.2 a saturar as pistas do PCI-Express e as Queue Depths simultaneamente.
//! Zero overhead de Tokio ou Futures, apenas raw blocking I/O concorrente.

use nodestor_core::NodeStorError;
use nodestor_transport::DirectIOReader;
use std::thread;

pub struct BurstReader {
    pub path: String,
    pub num_threads: usize,
}

impl BurstReader {
    /// Inicializa o leitor de burst automático determinando o número de threads pelo hardware
    pub fn new(path: &str) -> Self {
        // Usa metade das threads para não estrangular outras atividades da CPU. Mínimo 2, Máximo 8.
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let max_threads = (num_cpus / 2).max(2).min(8);
        
        Self {
            path: path.to_string(),
            num_threads: max_threads,
        }
    }

    /// Executa leitura altamente paralela de múltiplos slices simultaneamente.
    ///
    /// Abre file handles isolados (para contornar locks de sincronização fd no OS) e dispara 
    /// reads em threads separadas, usando memory buffers mutáveis independentes.
    pub fn read_burst(
        &self,
        slices: &[(u64, usize)],
        buffers: &mut [&mut [u8]],
    ) -> Result<Vec<usize>, NodeStorError> {
        if slices.len() != buffers.len() {
            return Err(NodeStorError::TransferFailed(
                "BurstReader: Número de fatias (slices) deve ser igual ao de buffers".to_string(),
            ));
        }

        // Caminho eficiente para caso único (fallback transparente)
        if slices.len() <= 1 {
            let reader = DirectIOReader::open(&self.path)?;
            let mut results = Vec::new();
            for (i, &(offset, size)) in slices.iter().enumerate() {
                let bytes = reader.read_at(offset, size, buffers[i])?;
                results.push(bytes);
            }
            return Ok(results);
        }

        thread::scope(|s| {
            let path = &self.path;
            
            // `iter_mut()` fraciona os tempos de vida, permitindo envio para as threads scopadas 
            let handles: Vec<_> = slices
                .iter()
                .zip(buffers.iter_mut())
                .map(|(&(offset, size), buf)| {
                    s.spawn(move || -> Result<usize, NodeStorError> {
                        // Criação local do DirectIOReader força a abertura de um File Handle novo por thread.
                        let reader = DirectIOReader::open(path)?;
                        reader.read_at(offset, size, buf)
                    })
                })
                .collect();

            let mut results = Vec::with_capacity(slices.len());
            for h in handles {
                match h.join() {
                    Ok(res) => results.push(res?),
                    Err(_) => return Err(NodeStorError::TransferFailed("Burst thread panic".to_string())),
                }
            }

            Ok(results)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_burst_reader_parallel_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("burst_test.bin");
        // Arquivo de 32KB
        std::fs::File::create(&path).unwrap().write_all(&vec![0xAA; 32768]).unwrap();

        let reader = BurstReader::new(path.to_str().unwrap());
        
        let slices = vec![(0, 4096), (4096, 4096), (8192, 4096), (12288, 4096)];
        
        // Pinned memory emulação (alinhado idealmente, ms pro teste alocamos o u8 solto)
        let mut b1 = vec![0u8; 4096];
        let mut b2 = vec![0u8; 4096];
        let mut b3 = vec![0u8; 4096];
        let mut b4 = vec![0u8; 4096];
        let mut bufs: Vec<&mut [u8]> = vec![&mut b1, &mut b2, &mut b3, &mut b4];

        let results = reader.read_burst(&slices, &mut bufs).unwrap();

        assert_eq!(results, vec![4096, 4096, 4096, 4096]);
        assert_eq!(bufs[0][0], 0xAA);
        assert_eq!(bufs[3][4000], 0xAA);
    }
}
