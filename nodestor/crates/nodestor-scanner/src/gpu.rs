use nodestor_core::{GpuCapabilities, GpuVendor};
use tracing::debug;

/// Detecta GPUs disponíveis no sistema.
/// Estratégia em 3 camadas:
/// 1. Vulkan (prova real de compute — funciona em NVIDIA, AMD, Intel, Qualcomm)
/// 2. Heurísticas de SO (fallback: arquivos de sistema, variáveis de ambiente)
/// 3. CPU-only fallback (funciona em qualquer máquina, inclusive headless)
pub fn detect_gpus() -> Vec<GpuCapabilities> {
    // 1. Prioridade máxima: Vulkan (detecção real de capacidades)
    let vulkan_gpus = nodestor_vulkan::VulkanEngine::probe_gpus();
    if !vulkan_gpus.is_empty() {
        debug!("Vulkan probe detectou {} GPU(s)", vulkan_gpus.len());
        return vulkan_gpus;
    }

    // 2. Heurísticas de SO
    debug!("Vulkan probe não retornou GPUs. Usando heurísticas de SO.");
    let mut gpus = Vec::new();

    if let Some(gpu) = detect_nvidia_gpu() { gpus.push(gpu); }
    if let Some(gpu) = detect_amd_gpu() { gpus.push(gpu); }
    if let Some(gpu) = detect_intel_gpu() { gpus.push(gpu); }
    if let Some(gpu) = detect_qualcomm_gpu() { gpus.push(gpu); }
    if let Some(gpu) = detect_apple_gpu() { gpus.push(gpu); }

    if !gpus.is_empty() {
        debug!("Heurísticas detectaram {} GPU(s)", gpus.len());
        return gpus;
    }

    // 3. CPU-only: garante que o sistema funciona em QUALQUER máquina
    debug!("Nenhuma GPU detectada. Usando CPU-only fallback — sistema totalmente funcional.");
    vec![cpu_only_fallback()]
}

/// Returns a CPU-only capability profile for headless/server environments.
fn cpu_only_fallback() -> GpuCapabilities {
    GpuCapabilities {
        vendor: GpuVendor::Unknown,
        device_name: "CPU-only (sem GPU)".to_string(),
        vram_bytes: 0,
        supports_vulkan_compute: false,
        supports_cooperative_matrix2: false,
        supports_cooperative_matrix_khr: false,
        supports_bfloat16: false,
        pcie_gen: 0,
        pcie_lanes: 0,
        resizable_bar_enabled: false,
        driver_version: "N/A".to_string(),
    }
}

// ─── NVIDIA ───────────────────────────────────────────────────────────────────

fn detect_nvidia_gpu() -> Option<GpuCapabilities> {
    // Linux: /proc/driver/nvidia (driver carregado)
    #[cfg(target_os = "linux")]
    if std::path::Path::new("/proc/driver/nvidia/version").exists() {
        let name = read_nvidia_name_linux().unwrap_or_else(|| "NVIDIA GPU".to_string());
        let vram = read_nvidia_vram_linux();
        let driver = read_nvidia_driver_version_linux();
        let cm2 = check_nvidia_driver_version_for_cm2(&driver);
        return Some(GpuCapabilities {
            vendor: GpuVendor::Nvidia,
            device_name: name,
            vram_bytes: vram,
            supports_vulkan_compute: true,
            supports_cooperative_matrix2: cm2,
            supports_cooperative_matrix_khr: true,
            supports_bfloat16: true,
            pcie_gen: 0,
            pcie_lanes: 0,
            resizable_bar_enabled: check_resizable_bar_linux("NVIDIA"),
            driver_version: driver,
        });
    }

    // Windows: nvapi64.dll (NVIDIA API nesse nível indica driver instalado)
    #[cfg(target_os = "windows")]
    if std::path::Path::new("C:\\Windows\\System32\\nvapi64.dll").exists()
        || std::path::Path::new("C:\\Windows\\SysWOW64\\nvapi.dll").exists()
    {
        let vram = read_nvidia_vram_windows();
        return Some(GpuCapabilities {
            vendor: GpuVendor::Nvidia,
            device_name: "NVIDIA GPU (Windows)".to_string(),
            vram_bytes: vram,
            supports_vulkan_compute: true,
            supports_cooperative_matrix2: true, // Assume RTX+ default no Windows
            supports_cooperative_matrix_khr: true,
            supports_bfloat16: true,
            pcie_gen: 0,
            pcie_lanes: 0,
            resizable_bar_enabled: false,
            driver_version: read_nvidia_driver_version_windows(),
        });
    }

    None
}

// ─── AMD ──────────────────────────────────────────────────────────────────────

fn detect_amd_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        // Verifica vendor via sysfs DRM
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") && !name.contains('-') {
                    let vendor_path = entry.path().join("device/vendor");
                    if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                        if vendor.trim() == "0x1002" {
                            // AMD Vendor ID
                            let device_id = read_sysfs_device_id(entry.path().join("device/device").to_str().unwrap_or(""));
                            let vram = read_amd_vram_linux(entry.path().to_str().unwrap_or(""));
                            return Some(GpuCapabilities {
                                vendor: GpuVendor::Amd,
                                device_name: format!("AMD Radeon GPU ({})", device_id),
                                vram_bytes: vram,
                                supports_vulkan_compute: true,
                                supports_cooperative_matrix2: false, // AMD RDNA3+ only
                                supports_cooperative_matrix_khr: true,
                                supports_bfloat16: true, // RDNA2+
                                pcie_gen: 0,
                                pcie_lanes: 0,
                                resizable_bar_enabled: check_resizable_bar_linux("AMD"),
                                driver_version: "AMDGPU via sysfs".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // amdvlk64 = Vulkan driver AMD open-source; atidxx = driver clássico
        let has_amd = std::path::Path::new("C:\\Windows\\System32\\amdvlk64.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\atidxx64.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\amdgfxInfo64.dll").exists();
        if has_amd {
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
                driver_version: "AMD via DLL".to_string(),
            });
        }
    }

    None
}

// ─── INTEL ────────────────────────────────────────────────────────────────────

fn detect_intel_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        // Intel Arc/Xe usa renderD128+ via DRM
        for render_node in &["/dev/dri/renderD128", "/dev/dri/renderD129"] {
            if std::path::Path::new(render_node).exists() {
                // Verifica se é Intel via vendor 0x8086
                if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                    for entry in entries.flatten() {
                        let vendor_path = entry.path().join("device/vendor");
                        if let Ok(v) = std::fs::read_to_string(&vendor_path) {
                            if v.trim() == "0x8086" {
                                return Some(GpuCapabilities {
                                    vendor: GpuVendor::Intel,
                                    device_name: "Intel Graphics/Arc (Linux)".to_string(),
                                    vram_bytes: 0,
                                    supports_vulkan_compute: true,
                                    supports_cooperative_matrix2: false,
                                    supports_cooperative_matrix_khr: true,
                                    supports_bfloat16: false,
                                    pcie_gen: 0,
                                    pcie_lanes: 0,
                                    resizable_bar_enabled: false,
                                    driver_version: "Intel i915/xe via DRI".to_string(),
                                });
                            }
                        }
                    }
                }
                // Fallback: renderD128 existe mas não conseguiu verify vendor
                return Some(GpuCapabilities {
                    vendor: GpuVendor::Intel,
                    device_name: "Intel GPU (Integrated)".to_string(),
                    vram_bytes: 0,
                    supports_vulkan_compute: true,
                    supports_cooperative_matrix2: false,
                    supports_cooperative_matrix_khr: false,
                    supports_bfloat16: false,
                    pcie_gen: 0, pcie_lanes: 0, resizable_bar_enabled: false,
                    driver_version: "Intel via DRI".to_string(),
                });
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // igvk64 = Intel Vulkan; igd10iumd64 = Intel UHD driver DX12
        let has_intel = std::path::Path::new("C:\\Windows\\System32\\igvk64.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\igd10iumd64.dll").exists()
            || std::path::Path::new("C:\\Windows\\System32\\igvulkan64.dll").exists();
        if has_intel {
            return Some(GpuCapabilities {
                vendor: GpuVendor::Intel,
                device_name: "Intel HD/Iris/Arc Graphics (Windows)".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: true,
                supports_bfloat16: false,
                pcie_gen: 0, pcie_lanes: 0, resizable_bar_enabled: false,
                driver_version: "Intel via igvk".to_string(),
            });
        }
    }

    None
}

// ─── QUALCOMM (Mobile / NPU) ──────────────────────────────────────────────────

fn detect_qualcomm_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "linux")]
    {
        // Snapdragon Adreno: vendor 0x5143 no sysfs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let vendor_path = entry.path().join("device/vendor");
                if let Ok(v) = std::fs::read_to_string(&vendor_path) {
                    if v.trim() == "0x5143" {
                        return Some(GpuCapabilities {
                            vendor: GpuVendor::Unknown, // TODO: Add Qualcomm variant
                            device_name: "Qualcomm Adreno GPU (Linux/Android)".to_string(),
                            vram_bytes: 0,
                            supports_vulkan_compute: true,
                            supports_cooperative_matrix2: false,
                            supports_cooperative_matrix_khr: false,
                            supports_bfloat16: false,
                            pcie_gen: 0, pcie_lanes: 0, resizable_bar_enabled: false,
                            driver_version: "Qualcomm Adreno via sysfs".to_string(),
                        });
                    }
                }
            }
        }
    }
    None
}

// ─── APPLE SILICON ────────────────────────────────────────────────────────────

fn detect_apple_gpu() -> Option<GpuCapabilities> {
    #[cfg(target_os = "macos")]
    {
        // Em macOS, Metal é o backend. Para Vulkan, MoltenVK faz a ponte.
        // sysctl -n hw.model retorna "MacBookPro18,1" etc.
        let has_moltenvk = std::path::Path::new("/usr/local/lib/libMoltenVK.dylib").exists()
            || std::path::Path::new("/usr/local/lib/libvulkan.dylib").exists();
        
        let model = std::process::Command::new("sysctl")
            .args(["-n", "hw.model"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".to_string());

        return Some(GpuCapabilities {
            vendor: GpuVendor::Unknown,
            device_name: format!("Apple GPU / {} (macOS)", model),
            vram_bytes: detect_apple_unified_memory(),
            supports_vulkan_compute: has_moltenvk,
            supports_cooperative_matrix2: false,
            supports_cooperative_matrix_khr: false,
            supports_bfloat16: true, // Apple Silicon tem bfloat16 nativo
            pcie_gen: 0, pcie_lanes: 0, resizable_bar_enabled: false,
            driver_version: "Apple Metal / MoltenVK".to_string(),
        });
    }
    #[cfg(not(target_os = "macos"))]
    None
}

// ─── Helpers plataforma-específica ────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn read_nvidia_name_linux() -> Option<String> {
    // Tenta ler o nome via /proc/driver/nvidia/gpus/<id>/information
    if let Ok(dir) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in dir.flatten() {
            let info_path = entry.path().join("information");
            if let Ok(content) = std::fs::read_to_string(info_path) {
                for line in content.lines() {
                    if line.starts_with("Model:") {
                        return Some(line.trim_start_matches("Model:").trim().to_string());
                    }
                }
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn read_nvidia_name_linux() -> Option<String> { None }

#[cfg(target_os = "linux")]
fn read_nvidia_vram_linux() -> u64 {
    if let Ok(dir) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in dir.flatten() {
            let info_path = entry.path().join("information");
            if let Ok(content) = std::fs::read_to_string(info_path) {
                for line in content.lines() {
                    if line.starts_with("Video Memory:") {
                        let size_str = line
                            .trim_start_matches("Video Memory:")
                            .trim()
                            .trim_end_matches("MiB");
                        if let Ok(mb) = size_str.trim().parse::<u64>() {
                            return mb * 1024 * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn read_nvidia_vram_linux() -> u64 { 0 }

#[cfg(target_os = "linux")]
fn read_nvidia_driver_version_linux() -> String {
    std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|s| s.lines().next().map(String::from))
        .unwrap_or_else(|| "Unknown".to_string())
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn read_nvidia_driver_version_linux() -> String { "Unknown".to_string() }

fn check_nvidia_driver_version_for_cm2(driver_str: &str) -> bool {
    // Cooperative Matrix 2 requer driver >= 575.0
    for part in driver_str.split_whitespace() {
        if let Ok(v) = part.parse::<f64>() {
            if v >= 100.0 { // Número de versão plausível
                return v >= 575.0;
            }
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn read_nvidia_vram_windows() -> u64 {
    // Tenta ler VRAM via WMIC (sempre disponível no Windows 10+)
    let output = std::process::Command::new("wmic")
        .args(["path", "Win32_VideoController", "get", "AdapterRAM", "/value"])
        .output()
        .ok();
    
    if let Some(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if line.starts_with("AdapterRAM=") {
                let val_str = line.trim_start_matches("AdapterRAM=").trim();
                if let Ok(bytes) = val_str.parse::<u64>() {
                    if bytes > 0 { return bytes; }
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
fn read_nvidia_vram_windows() -> u64 { 0 }

#[cfg(target_os = "windows")]
fn read_nvidia_driver_version_windows() -> String {
    // Versão do driver NVIDIA via WMIC
    let output = std::process::Command::new("wmic")
        .args(["path", "Win32_VideoController", "get", "DriverVersion", "/value"])
        .output()
        .ok();
    
    if let Some(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if line.starts_with("DriverVersion=") {
                return line.trim_start_matches("DriverVersion=").trim().to_string();
            }
        }
    }
    "Unknown".to_string()
}

#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
fn read_nvidia_driver_version_windows() -> String { "Unknown".to_string() }

#[cfg(target_os = "linux")]
fn read_sysfs_device_id(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn read_sysfs_device_id(_path: &str) -> String { "unknown".to_string() }

#[cfg(target_os = "linux")]
fn read_amd_vram_linux(card_path: &str) -> u64 {
    // AMDGPU expõe VRAM via mem_info_vram_total
    let vram_path = format!("{}/device/mem_info_vram_total", card_path);
    std::fs::read_to_string(&vram_path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn read_amd_vram_linux(_card_path: &str) -> u64 { 0 }

#[cfg(target_os = "linux")]
fn check_resizable_bar_linux(vendor_hint: &str) -> bool {
    // Verifica se BAR resizeable está ativo para o vendor
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        for entry in entries.flatten() {
            let resource_path = entry.path().join("resource");
            if let Ok(content) = std::fs::read_to_string(resource_path) {
                // Heurística: se o BAR0 (linha 1) tem tamanho >= 4GB, está ativo
                let first_bar = content.lines().nth(1).unwrap_or("");
                let parts: Vec<&str> = first_bar.split_whitespace().collect();
                if parts.len() >= 2 {
                    let start = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).unwrap_or(0);
                    let end = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap_or(0);
                    let size = end.saturating_sub(start);
                    if size >= 4 * 1024 * 1024 * 1024 {
                        let _ = vendor_hint; // Usado conceptualmente
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn check_resizable_bar_linux(_vendor_hint: &str) -> bool { false }

#[cfg(target_os = "macos")]
fn detect_apple_unified_memory() -> u64 {
    // sysctl hw.memsize retorna a RAM total (= VRAM unificada no Apple Silicon)
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn detect_apple_unified_memory() -> u64 { 0 }
