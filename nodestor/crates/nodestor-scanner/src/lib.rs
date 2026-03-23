//! nodestor-scanner — Detecção automática de hardware e seleção do backend ideal.
//!
//! Percorre GPU, storage e SO de forma não-destrutiva e retorna um `HardwareProfile`
//! que o restante do sistema usa para escolher as rotas de dados mais eficientes.

mod gpu;
mod storage;
mod os_info;

use nodestor_core::{HardwareProfile, NodeStorError, TransportBackend};
use tracing::{debug, info, warn};

pub use gpu::detect_gpus;
pub use storage::detect_storage;
pub use os_info::detect_os;

/// Ponto de entrada principal: detecta todo o hardware e seleciona o melhor backend.
pub fn scan() -> Result<HardwareProfile, NodeStorError> {
    info!("NodeStor Scanner iniciando detecção de hardware...");

    let gpus = detect_gpus();
    debug!("GPUs detectadas: {}", gpus.len());

    let storage = detect_storage();
    debug!("Dispositivos de armazenamento detectados: {}", storage.len());

    let (os, os_version) = detect_os();
    debug!("SO detectado: {} ({})", os, os_version);

    let cpu_cores = num_cpus();
    let total_ram_bytes = total_ram();

    let recommended_transport = select_transport(&os, &gpus, &os_version);
    info!("Backend de transporte selecionado: {}", recommended_transport);

    Ok(HardwareProfile {
        gpus,
        storage,
        os,
        os_version,
        recommended_transport,
        cpu_cores,
        total_ram_bytes,
    })
}

fn select_transport(
    os: &nodestor_core::OsType,
    gpus: &[nodestor_core::GpuCapabilities],
    os_version: &str,
) -> TransportBackend {
    use nodestor_core::{GpuVendor, OsType};

    // NVIDIA GPU no Linux → tentar GDS (requer driver + cuFile)
    let has_nvidia = gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia);

    match os {
        OsType::Linux => {
            if has_nvidia && gds_available() {
                return TransportBackend::NvidiaGds;
            }
            let kernel = parse_kernel_version(os_version);
            if kernel >= (6, 16, 0) {
                warn!("io_uring + DMABUF requer kernel 6.16+. Versão: {}", os_version);
                TransportBackend::IoUringDmabuf
            } else if kernel >= (5, 11, 0) {
                TransportBackend::IoUringStandard
            } else {
                warn!("Kernel antigo ({}). Usando fallback pread.", os_version);
                TransportBackend::PreadFallback
            }
        }
        OsType::Windows => {
            if directstorage_available() {
                TransportBackend::DirectStorage
            } else {
                warn!("DirectStorage DLLs não encontradas. Use Win32 fallback.");
                warn!("Para melhor performance, baixe: https://github.com/microsoft/DirectStorage/releases");
                TransportBackend::Win32Fallback
            }
        }
        _ => {
            warn!("SO não suportado para transporte de alta performance. Usando pread.");
            TransportBackend::PreadFallback
        }
    }
}

/// Tenta detectar se cuFile/GDS está disponível.
fn gds_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Verifica presença da lib cuFile
        std::path::Path::new("/usr/lib/libcufile.so").exists()
            || std::path::Path::new("/usr/local/cuda/lib64/libcufile.so").exists()
    }
    #[cfg(not(target_os = "linux"))]
    false
}

/// Tenta detectar se DirectStorage DLLs estão disponíveis.
fn directstorage_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        let cwd = std::env::current_dir().unwrap_or_default();
        cwd.join("dstorage.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\dstorage.dll").exists()
    }
    #[cfg(not(target_os = "windows"))]
    false
}

fn parse_kernel_version(version_str: &str) -> (u32, u32, u32) {
    // Extrai "6.16.0" de strings como "Linux 6.16.0-generic"
    let nums: Vec<u32> = version_str
        .split_whitespace()
        .flat_map(|part| part.split('.'))
        .filter_map(|s| s.parse().ok())
        .take(3)
        .collect();

    match nums.as_slice() {
        [major, minor, patch, ..] => (*major, *minor, *patch),
        [major, minor] => (*major, *minor, 0),
        [major] => (*major, 0, 0),
        [] => (0, 0, 0),
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn total_ram() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_returns_profile() {
        let profile = scan().expect("Scanner deve funcionar em qualquer sistema");
        // SO deve ser detectado
        assert_ne!(profile.os_version, "", "Versão do SO não deve ser vazia");
        // CPU cores deve ser >= 1
        assert!(profile.cpu_cores >= 1, "Deve ter pelo menos 1 núcleo de CPU");
        // RAM deve ser > 0
        assert!(profile.total_ram_bytes > 0, "RAM total deve ser > 0");
        // Deve ter selecionado algum backend
        println!("Backend selecionado: {}", profile.recommended_transport);
    }

    #[test]
    fn test_parse_kernel_version() {
        assert_eq!(parse_kernel_version("Linux 6.16.2-generic"), (6, 16, 0));
        assert_eq!(parse_kernel_version("6.1.0"), (6, 1, 0));
        assert_eq!(parse_kernel_version("5.15.134.1-microsoft-standard-WSL2"), (5, 15, 134));
        assert_eq!(parse_kernel_version("unknown"), (0, 0, 0));
    }

    #[test]
    fn test_profile_methods() {
        let profile = scan().unwrap();
        let throughput = profile.estimated_transport_throughput();
        assert!(throughput > 0, "Throughput estimado deve ser > 0");
    }
}
