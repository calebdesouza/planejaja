use nodestor_core::{GpuCapabilities, GpuVendor};
use tracing::debug;

/// Detecta GPUs disponíveis no sistema.
pub fn detect_gpus() -> Vec<GpuCapabilities> {
    // 1. Tenta via Vulkan Engine (Passe 1 do Scanner)
    let mut gpus = nodestor_vulkan::VulkanEngine::probe_gpus();
    
    if !gpus.is_empty() {
        debug!("Vulkan probe detectou {} GPU(s)", gpus.len());
        return gpus;
    }

    // 2. Fallback para heurísticas de SO/System info
    debug!("Vulkan probe não retornou GPUs. Usando heurísticas de fallback.");
    
    let nvidia = detect_nvidia_gpu();
    let amd = detect_amd_gpu();
    let intel = detect_intel_gpu();

    if let Some(gpu) = nvidia { gpus.push(gpu); }
    if let Some(gpu) = amd { gpus.push(gpu); }
    if let Some(gpu) = intel { gpus.push(gpu); }

    if gpus.is_empty() {
        debug!("Nenhuma GPU detectada. Usando CPU fallback.");
        gpus.push(GpuCapabilities::default());
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
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: false,
                driver_version: "Detectado via /proc".to_string(),
            });
        }
    }
    #[cfg(target_os = "windows")]
    {
        if std::path::Path::new("C:\\Windows\\System32\\nvapi64.dll").exists() {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Nvidia,
                device_name: "NVIDIA GPU (Windows)".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: true,
                supports_cooperative_matrix_khr: true,
                supports_bfloat16: true,
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: false,
                driver_version: "Detectado via nvapi".to_string(),
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
                            return Some(GpuCapabilities {
                                vendor: GpuVendor::Amd,
                                device_name: "AMD GPU".to_string(),
                                vram_bytes: 0,
                                supports_vulkan_compute: true,
                                supports_cooperative_matrix2: false,
                                supports_cooperative_matrix_khr: true,
                                supports_bfloat16: true,
                                pcie_gen: 0,
                                pcie_lanes: 0,
                                resizable_bar_enabled: false,
                                driver_version: "Detectado via sysfs".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if std::path::Path::new("C:\\Windows\\System32\\amdvlk64.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\atidxx64.dll").exists()
        {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Amd,
                device_name: "AMD Radeon GPU (Windows)".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: true,
                supports_bfloat16: true,
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: false,
                driver_version: "Detectado via amdvlk".to_string(),
            });
        }
    }
    None
}

fn detect_intel_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/dri/renderD128").exists() {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Intel,
                device_name: "Intel GPU (Integrated)".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: false,
                driver_version: "Detectado via dri".to_string(),
            });
        }
    }
    #[cfg(target_os = "windows")]
    {
        if std::path::Path::new("C:\\Windows\\System32\\igvk64.dll").exists() {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Intel,
                device_name: "Intel HD/Iris Graphics (Windows)".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: false,
                driver_version: "Detectado via igvk".to_string(),
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

#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
fn read_nvidia_name() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn read_nvidia_vram() -> u64 {
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

#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
fn read_nvidia_vram() -> u64 {
    0
}

#[cfg(target_os = "linux")]
fn check_driver_version_nvidia_cm2() -> bool {
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

#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
fn check_driver_version_nvidia_cm2() -> bool {
    false
}
