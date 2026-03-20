use nodestor_core::{GpuCapabilities, GpuVendor};
use tracing::debug;

/// Detecta GPUs disponíveis no sistema.
/// Tenta via sysinfo; em um sistema completo isso seria via Vulkan instance query.
pub fn detect_gpus() -> Vec<GpuCapabilities> {
    use sysinfo::{Components, System};

    let mut gpus = Vec::new();

    // Tenta detectar via sysinfo (informação básica)
    // Em produção completa: usar ash para query Vulkan physical devices
    let sys = System::new_all();

    // sysinfo não expõe GPUs diretamente; usamos heurística por componentes
    let components = Components::new_with_refreshed_list();
    for component in &components {
        let label = component.label().to_lowercase();
        if label.contains("gpu") || label.contains("vga") {
            debug!("Componente GPU detectado: {}", component.label());
        }
    }

    // Detecção via variáveis de ambiente (útil em CI/CD e containers)
    if let Ok(cuda_visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        if !cuda_visible.is_empty() && cuda_visible != "-1" {
            debug!("CUDA_VISIBLE_DEVICES={}", cuda_visible);
        }
    }

    // Heurística de detecção por driver files
    let nvidia = detect_nvidia_gpu();
    let amd = detect_amd_gpu();
    let intel = detect_intel_gpu();

    if let Some(gpu) = nvidia {
        gpus.push(gpu);
    }
    if let Some(gpu) = amd {
        gpus.push(gpu);
    }
    if let Some(gpu) = intel {
        gpus.push(gpu);
    }

    // Se nenhuma GPU foi detectada, adiciona um placeholder CPU (para testes)
    if gpus.is_empty() {
        debug!("Nenhuma GPU detectada. Usando CPU fallback.");
        gpus.push(GpuCapabilities {
            vendor: GpuVendor::Unknown,
            device_name: "CPU Fallback (sem GPU dedicada)".to_string(),
            vram_bytes: 0,
            supports_vulkan_compute: false,
            supports_cooperative_matrix2: false,
            supports_cooperative_matrix_khr: false,
            supports_bfloat16: false,
        });
    }

    gpus
}

fn detect_nvidia_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/proc/driver/nvidia/version").exists() {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Nvidia,
                device_name: read_nvidia_name().unwrap_or_else(|| "NVIDIA GPU".to_string()),
                vram_bytes: read_nvidia_vram(),
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: check_driver_version_nvidia_cm2(),
                supports_cooperative_matrix_khr: true,
                supports_bfloat16: true,
            });
        }
    }
    #[cfg(target_os = "windows")]
    {
        // Verifica se nvapi.dll existe (indica driver NVIDIA instalado)
        if std::path::Path::new("C:\\Windows\\System32\\nvapi64.dll").exists() {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Nvidia,
                device_name: "NVIDIA GPU (Windows)".to_string(),
                vram_bytes: 0, // Requer nvapi para leitura real
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: true, // Driver 575+ no Windows
                supports_cooperative_matrix_khr: true,
                supports_bfloat16: true,
            });
        }
    }
    None
}

fn detect_amd_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/sys/class/drm").exists() {
            let entries = std::fs::read_dir("/sys/class/drm").ok()?;
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") && !name.contains('-') {
                    let vendor_path = entry.path().join("device/vendor");
                    if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                        if vendor.trim() == "0x1002" {
                            // AMD PCI vendor ID
                            return Some(GpuCapabilities {
                                vendor: GpuVendor::Amd,
                                device_name: "AMD GPU".to_string(),
                                vram_bytes: 0,
                                supports_vulkan_compute: true,
                                supports_cooperative_matrix2: false,
                                supports_cooperative_matrix_khr: true,
                                supports_bfloat16: true,
                            });
                        }
                    }
                }
            }
        }
    }
    None
}

fn detect_intel_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/dri/renderD128").exists() {
            // Intel iGPU é o mais comum em /dev/dri
            return Some(GpuCapabilities {
                vendor: GpuVendor::Intel,
                device_name: "Intel GPU (Integrated)".to_string(),
                vram_bytes: 0, // Memória compartilhada com sistema
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
            });
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_nvidia_name() -> Option<String> {
    std::fs::read_to_string("/proc/driver/nvidia/gpus/")
        .ok()
        .map(|_| "NVIDIA GPU".to_string())
}

#[cfg(not(target_os = "linux"))]
fn read_nvidia_name() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn read_nvidia_vram() -> u64 {
    // Tenta ler VRAM via /proc/driver/nvidia/gpus/*
    if let Ok(dir) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in dir.flatten() {
            let info_path = entry.path().join("information");
            if let Ok(content) = std::fs::read_to_string(info_path) {
                for line in content.lines() {
                    if line.starts_with("Video Memory:") {
                        let parts: Vec<&str> = line.split(':').collect();
                        if let Some(size_str) = parts.get(1) {
                            if let Ok(mb) = size_str.trim().trim_end_matches(" MiB").parse::<u64>() {
                                return mb * 1024 * 1024;
                            }
                        }
                    }
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn read_nvidia_vram() -> u64 {
    0
}

#[cfg(target_os = "linux")]
fn check_driver_version_nvidia_cm2() -> bool {
    // cooperative_matrix2 requer driver 575+
    if let Ok(content) = std::fs::read_to_string("/proc/driver/nvidia/version") {
        if let Some(line) = content.lines().next() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == "Kernel" && i + 1 < parts.len() {
                    if let Ok(version) = parts[i + 1].parse::<f64>() {
                        return version >= 575.0;
                    }
                }
            }
        }
    }
    false
}

#[cfg(not(target_os = "linux"))]
fn check_driver_version_nvidia_cm2() -> bool {
    false
}
