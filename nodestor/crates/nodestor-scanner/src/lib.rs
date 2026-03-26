//! nodestor-scanner — Detecção automática de hardware e seleção do backend ideal.
//!
//! Percorre GPU, storage e SO de forma não-destrutiva e retorna um `HardwareProfile`
//! que o restante do sistema usa para escolher as rotas de dados mais eficientes.

mod gpu;
mod storage;
mod os_info;

use nodestor_core::{HardwareProfile, NodeStorError, TransportBackend};
use tracing::{debug, info};

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

    let has_nvidia = gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia);
    let has_amd = gpus.iter().any(|g| g.vendor == GpuVendor::Amd);
    let has_vulkan = gpus.iter().any(|g| g.supports_vulkan_compute);

    match os {
        OsType::Linux => {
            // 1. NVIDIA + Linux + cuFile disponível -> NvidiaGds
            if has_nvidia && gds_available() {
                return TransportBackend::NvidiaGds;
            }
            // 2. AMD + Linux + ROCm disponível -> RocmDirectGma
            if has_amd && rocm_available() {
                return TransportBackend::RocmDirectGma;
            }
            
            let kernel = parse_kernel_version(os_version);
            // 4. Linux + kernel ≥ 6.16 -> IoUringDmabuf
            if kernel >= (6, 16, 0) {
                return TransportBackend::IoUringDmabuf;
            }
            // 5. Linux + kernel ≥ 5.11 -> IoUringStandard
            if kernel >= (5, 11, 0) {
                return TransportBackend::IoUringStandard;
            }
        }
        OsType::Windows => {
            // 3. Windows + DirectStorage DLLs presentes -> DirectStorage
            if directstorage_available() {
                return TransportBackend::DirectStorage;
            }
        }
        _ => {}
    }

    // 6. Qualquer (Intel, Mac, sem GPU, CPU-only) -> VulkanGeneric (DMA via Vulkan)
    if has_vulkan {
        TransportBackend::VulkanGeneric
    } else {
        TransportBackend::PreadFallback
    }
}

/// Tenta detectar se ROCm/DirectGMA está disponível.
fn rocm_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/dev/kfd").exists() && 
        std::path::Path::new("/opt/rocm").exists()
    }
    #[cfg(not(target_os = "linux"))]
    false
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
        // 1. Verifica no diretório do executável atual
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                if parent.join("dstorage.dll").exists() && parent.join("dstoragecore.dll").exists() {
                    return true;
                }
            }
        }

        // 2. Verifica no CWD (Diretório de trabalho)
        let cwd = std::env::current_dir().unwrap_or_default();
        if cwd.join("dstorage.dll").exists() && cwd.join("dstoragecore.dll").exists() {
            return true;
        }

        // 3. Verifica no System32 (Onde o Windows instala componentes globais)
        let system32 = std::path::Path::new("C:\\Windows\\System32");
        if system32.join("dstorage.dll").exists() {
            return true;
        }

        // 4. Fallback: Tenta ver se está no PATH (se pode carregar via OS)
        // Em um sistema real, poderíamos usar LoadLibraryExA com flags específicas,
        // mas para o Scanner, a presença do arquivo no System32 ou CWD é o sinal mais forte.
        false
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
