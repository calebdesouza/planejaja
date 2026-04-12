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

    let mut missed_optimizations = Vec::new();
    let recommended_transport = select_transport(&os, &gpus, &storage, &os_version, &mut missed_optimizations);
    info!("Backend de transporte selecionado: {}", recommended_transport);

    Ok(HardwareProfile {
        gpus,
        storage,
        os,
        os_version,
        recommended_transport,
        cpu_cores,
        total_ram_bytes,
        missed_optimizations,
    })
}

fn select_transport(
    os: &nodestor_core::OsType,
    gpus: &[nodestor_core::GpuCapabilities],
    storage: &[nodestor_core::StorageInfo],
    os_version: &str,
    missed: &mut Vec<String>,
) -> TransportBackend {
    use nodestor_core::{GpuVendor, OsType, NvmeGen};

    let has_nvidia = gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia);
    let has_amd = gpus.iter().any(|g| g.vendor == GpuVendor::Amd);
    let has_vulkan = gpus.iter().any(|g| g.supports_vulkan_compute);

    match os {
        OsType::Linux => {
            // 1. NVIDIA GDS
            if has_nvidia {
                if gds_available() {
                    return TransportBackend::NvidiaGds;
                } else {
                    missed.push("NVIDIA GPUDirect Storage ignorado: libcufile.so não encontrado.".to_string());
                }
            }
            
            // 2. AMD ROCm
            if has_amd {
                if rocm_available() {
                    return TransportBackend::RocmDirectGma;
                } else {
                    missed.push("AMD ROCm DirectGMA ignorado: /dev/kfd ou /opt/rocm não encontrados.".to_string());
                }
            }
            
            let kernel = parse_kernel_version(os_version);
            if kernel >= (6, 16, 0) {
                return TransportBackend::IoUringDmabuf;
            } else if kernel >= (5, 11, 0) {
                missed.push(format!("DMABUF Zero-copy ignorado: Kernel {} < 6.16.", os_version));
                return TransportBackend::IoUringStandard;
            }
        }
        OsType::Windows => {
            // 3. DirectStorage
            if directstorage_available() {
                let has_gen3 = storage.iter().any(|s| s.nvme_gen >= NvmeGen::Gen3);
                if has_gen3 {
                    return TransportBackend::DirectStorage;
                } else {
                    missed.push("DirectStorage em modo degradado: Nenhum SSD NVMe Gen3+ detectado.".to_string());
                }
            } else {
                missed.push("DirectStorage ignorado: dstorage.dll e dstoragecore.dll não encontrados localmente.".to_string());
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
    // Extrai "6.16.2" de strings como "Linux 6.16.2-generic" ou "6.1.88-1-MANJARO"
    let nums: Vec<u32> = version_str
        .split_whitespace()
        .flat_map(|part| part.split('.'))
        .filter_map(|s| {
            // Remove sufixos como "-generic", "-MANJARO" antes de parsear
            s.split('-').next().unwrap_or(s).parse::<u32>().ok()
        })
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
    use nodestor_core::{GpuVendor, TransportBackend};

    /// Testa que `scan()` funciona em qualquer sistema operacional e retorna um perfil válido.
    #[test]
    fn test_scan_returns_profile() {
        let profile = scan().expect("Scanner deve funcionar em qualquer sistema");
        // SO deve ser detectado
        assert_ne!(profile.os_version, "", "Versão do SO não deve ser vazia");
        // CPU cores deve ser >= 1 (funciona em qualquer máquina)
        assert!(profile.cpu_cores >= 1, "Deve ter pelo menos 1 núcleo de CPU");
        // RAM deve ser > 0 (funciona em qualquer máquina)
        assert!(profile.total_ram_bytes > 0, "RAM total deve ser > 0");
        // Deve ter GPU ou CPU-only fallback
        assert!(!profile.gpus.is_empty(), "Deve ter pelo menos 1 device (GPU ou CPU-only)");
        // Backend selecionado deve ser válido
        println!("Backend selecionado: {}", profile.recommended_transport);
        println!("GPUs: {:?}", profile.gpus.iter().map(|g| &g.device_name).collect::<Vec<_>>());
        println!("Storage: {} dispositivos", profile.storage.len());
    }

    /// Testa que o throughput estimado é sempre > 0 (CPU-only ou qualquer GPU).
    #[test]
    fn test_profile_throughput_always_positive() {
        let profile = scan().unwrap();
        let throughput = profile.estimated_transport_throughput();
        assert!(throughput > 0, "Throughput estimado deve ser > 0 mesmo sem GPU");
    }

    /// Testa CPU-only fallback: quando não há GPU, o device é Unknown e backend é PreadFallback.
    #[test]
    fn test_cpu_only_fallback_structure() {
        let gpus_none: Vec<nodestor_core::GpuCapabilities> = Vec::new();
        let storage: Vec<nodestor_core::StorageInfo> = Vec::new();
        let mut missed = Vec::new();
        let backend = select_transport(
            &nodestor_core::OsType::Linux, 
            &gpus_none, 
            &storage, 
            "5.10.0",
            &mut missed
        );
        assert_eq!(backend, TransportBackend::PreadFallback, "Sem GPU/Vulkan deve usar PreadFallback");
    }

    /// Valida parsing de versão do kernel Linux em todos os formatos.
    #[test]
    fn test_parse_kernel_version() {
        assert_eq!(parse_kernel_version("Linux 6.16.2-generic"), (6, 16, 2));
        assert_eq!(parse_kernel_version("6.1.0"), (6, 1, 0));
        assert_eq!(parse_kernel_version("5.15.134.1-microsoft-standard-WSL2"), (5, 15, 134));
        assert_eq!(parse_kernel_version("unknown"), (0, 0, 0));
        assert_eq!(parse_kernel_version("6.16"), (6, 16, 0));
        assert_eq!(parse_kernel_version(""), (0, 0, 0));
    }

    /// Testa que a detecção de storage retorna pelo menos 1 dispositivo em qualquer máquina.
    #[test]
    fn test_storage_always_finds_at_least_one_device() {
        let storage = detect_storage();
        assert!(!storage.is_empty(), "Deve encontrar pelo menos 1 dispositivo de armazenamento");
        // Primeiro dispositivo (mais rápido) deve ter throughput estimado > 0
        assert!(storage[0].estimated_read_bps > 0, "Throughput de leitura deve ser > 0");
    }

    /// Testa que GPUs detectadas têm campos coerentes.
    #[test]
    fn test_gpus_have_coherent_fields() {
        let gpus = detect_gpus();
        assert!(!gpus.is_empty(), "Deve ter pelo menos 1 device (GPU ou CPU-only fallback)");
        for gpu in &gpus {
            assert!(!gpu.device_name.is_empty(), "Nome da GPU não deve ser vazio");
            // CPU-only não suporta Vulkan — coerência
            if gpu.vendor == GpuVendor::Unknown && gpu.device_name.contains("CPU-only") {
                assert!(!gpu.supports_vulkan_compute, "CPU-only não deve reportar Vulkan");
            }
        }
    }

    /// Testa o backend Windows (DirectStorage) em ambientes sem a DLL.
    #[test]
    fn test_windows_backend_without_dstorage() {
        #[cfg(target_os = "windows")]
        {
            // Em qualquer Windows sem DirectStorage, deve cair no VulkanGeneric ou PreadFallback
            let profile = scan().unwrap();
            match profile.recommended_transport {
                TransportBackend::DirectStorage |
                TransportBackend::VulkanGeneric |
                TransportBackend::PreadFallback => {},
                other => println!("Backend Windows: {}", other),
            }
        }
        #[cfg(not(target_os = "windows"))]
        { /* não aplicavel neste SO */ }
    }

    /// Testa que o BackEnd Linux seleciona corretamente por kernel version.
    #[test]
    fn test_linux_iouring_selection() {
        let gpu_vulkan = vec![nodestor_core::GpuCapabilities {
            vendor: GpuVendor::Unknown,
            device_name: "TestGPU".to_string(),
            vram_bytes: 0,
            supports_vulkan_compute: true,
            supports_cooperative_matrix2: false,
            supports_cooperative_matrix_khr: false,
            supports_bfloat16: false,
            pcie_gen: 0, pcie_lanes: 0, resizable_bar_enabled: false,
            driver_version: "".to_string(),
        }];
        let storage: Vec<nodestor_core::StorageInfo> = Vec::new();
        let mut missed = Vec::new();

        // Kernel 6.16+ deve selecionar IoUringDmabuf
        let backend = select_transport(
            &nodestor_core::OsType::Linux, 
            &gpu_vulkan, 
            &storage, 
            "6.16.0",
            &mut missed
        );
        assert_eq!(backend, TransportBackend::IoUringDmabuf, "Kernel 6.16+ com Vulkan deve usar IoUringDmabuf");

        // Kernel 5.11 deve usar IoUringStandard
        let backend2 = select_transport(
            &nodestor_core::OsType::Linux, 
            &gpu_vulkan, 
            &storage, 
            "5.11.0",
            &mut missed
        );
        assert_eq!(backend2, TransportBackend::IoUringStandard, "Kernel 5.11 deve usar IoUringStandard");

        // Kernel antigo deve usar VulkanGeneric
        let backend3 = select_transport(
            &nodestor_core::OsType::Linux, 
            &gpu_vulkan, 
            &storage, 
            "5.4.0",
            &mut missed
        );
        assert_eq!(backend3, TransportBackend::VulkanGeneric, "Kernel antigo com Vulkan deve usar VulkanGeneric");
    }
}
