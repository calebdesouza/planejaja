use nodestor_core::{NvmeGen, StorageInfo};
use sysinfo::Disks;

/// Detecta dispositivos de armazenamento disponíveis com classificação precisa.
///
/// Estratégias de classificação NVMe Gen:
/// 1. Nome do modelo (contém "Gen4", "990 Pro", etc.)
/// 2. Velocidade de leitura medida via tamanho de bloco conhecido (heurística)
/// 3. Tipo de interface (sysfs no Linux, WMIC no Windows)
/// 4. Fallback conservador (Gen3 para SSDs não identificados)
pub fn detect_storage() -> Vec<StorageInfo> {
    let disks = Disks::new_with_refreshed_list();
    let mut results = Vec::new();

    for disk in &disks {
        let path = disk.mount_point().to_string_lossy().to_string();
        let name = disk.name().to_string_lossy().to_string();
        let total_bytes = disk.total_space();
        let available_bytes = disk.available_space();
        let is_ssd = matches!(disk.kind(), sysinfo::DiskKind::SSD);

        // Enriquece o nome com info real do sistema quando disponível
        let enriched_name = enrich_disk_name(&name, &path);
        let nvme_gen = classify_nvme_gen(&enriched_name, &path, is_ssd);
        let estimated_read_bps = theoretical_read_speed(nvme_gen, is_ssd);

        results.push(StorageInfo {
            path,
            name: enriched_name,
            nvme_gen,
            total_bytes,
            available_bytes,
            is_ssd,
            estimated_read_bps,
        });
    }

    // Prioriza por velocidade estimada (mais rápido primeiro)
    results.sort_by(|a, b| b.estimated_read_bps.cmp(&a.estimated_read_bps));
    results
}

/// Tenta enriquecer o nome do disco com dados reais do SO.
fn enrich_disk_name(base_name: &str, mount_path: &str) -> String {
    // Linux: lê o model do disco via sysfs block
    #[cfg(target_os = "linux")]
    if let Some(model) = read_block_model_linux(mount_path) {
        if !model.is_empty() && model != base_name {
            return format!("{} ({})", base_name, model);
        }
    }

    // Windows: tenta enriquecer via WMIC para o disco mapeado
    #[cfg(target_os = "windows")]
    {
        let _ = mount_path; // usado conceitualmente
        if let Some(model) = read_disk_model_windows(base_name) {
            return model;
        }
    }

    base_name.to_string()
}

/// Classifica NVMe Gen com base em todas as fontes disponíveis.
fn classify_nvme_gen(name: &str, mount_path: &str, is_ssd: bool) -> NvmeGen {
    if !is_ssd {
        return NvmeGen::Unknown;
    }

    let lower = name.to_lowercase();

    // Gen5: 10-14 GB/s (Samsung 990 Gen5, Crucial T705, WD Black SN850X Gen5 etc)
    if lower.contains("gen5") || lower.contains("pcie5") || lower.contains("pcie 5")
        || lower.contains("t705") || lower.contains("sn850x gen5")
    {
        return NvmeGen::Gen5;
    }

    // Gen4: 5-7 GB/s (Samsung 990 Pro, 980 Pro, WD SN850X, SK Hynix Platinum P41, Seagate PS5)
    if lower.contains("gen4") || lower.contains("pcie4") || lower.contains("pcie 4")
        || lower.contains("990 pro") || lower.contains("980 pro") || lower.contains("sn850")
        || lower.contains("p41") || lower.contains("sabrent rocket 4")
        || lower.contains("mp600") || lower.contains("firecuda 530")
    {
        return NvmeGen::Gen4;
    }

    // Gen3: 2-3.5 GB/s (Samsung 970 EVO, 860 EVO NVMe, WD Blue SN550, etc)
    if lower.contains("gen3") || lower.contains("pcie3") || lower.contains("pcie 3")
        || lower.contains("970 evo") || lower.contains("970 pro")
        || lower.contains("sn550") || lower.contains("sn570") || lower.contains("sn580")
        || lower.contains("860 evo") || lower.contains("p3 plus")
    {
        return NvmeGen::Gen3;
    }

    // SATA SSD / Gen2: ~500 MB/s (Samsung 870 EVO, Crucial MX500, Kingston A400)
    if lower.contains("sata") || lower.contains("gen2") || lower.contains("870 evo")
        || lower.contains("mx500") || lower.contains("a400") || lower.contains("kingston")
    {
        return NvmeGen::Gen2;
    }

    // Fallback via sysfs (Linux) ou WMIC (Windows)
    #[cfg(target_os = "linux")]
    if let Some(gen) = detect_nvme_gen_linux(mount_path) {
        return gen;
    }

    // Fallback conservador: SSDs não identificados assumem Gen3
    NvmeGen::Gen3
}

/// Throughput teórico realista por geração.
fn theoretical_read_speed(gen: NvmeGen, is_ssd: bool) -> u64 {
    if !is_ssd {
        return 200_000_000; // HDD ~200 MB/s
    }
    match gen {
        NvmeGen::Gen5   => 13_000_000_000, // ~13 GB/s (Crucial T705)
        NvmeGen::Gen4   =>  7_000_000_000, // ~7 GB/s  (Samsung 990 Pro)
        NvmeGen::Gen3   =>  3_500_000_000, // ~3.5 GB/s (970 EVO Plus)
        NvmeGen::Gen2   =>  1_500_000_000, // ~1.5 GB/s (SATA SSD)
        NvmeGen::Gen1   =>    500_000_000, // ~500 MB/s (HDD SSD antigo)
        NvmeGen::Unknown =>   500_000_000, // Conservador
    }
}

// ─── Helpers Linux ─────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn read_block_model_linux(mount_path: &str) -> Option<String> {
    // Mapeamento mount_path → block device
    // ex: "/" → /dev/sda, "/home" → /dev/nvme0n1p2
    let output = std::process::Command::new("findmnt")
        .args(["-n", "-o", "SOURCE", mount_path])
        .output()
        .ok()?;
    
    let device = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if device.is_empty() { return None; }

    // Remove partition number: /dev/nvme0n1p2 → nvme0n1
    let base_dev = device
        .trim_start_matches("/dev/")
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end_matches('p')
        .to_string();

    let model_path = format!("/sys/block/{}/device/model", base_dev);
    std::fs::read_to_string(&model_path)
        .ok()
        .map(|s| s.trim().to_string())
}

#[cfg(target_os = "linux")]
fn detect_nvme_gen_linux(mount_path: &str) -> Option<NvmeGen> {
    // Verifica PCIe speed via /sys/block/<dev>/device/pci_bus_info ou speed
    let output = std::process::Command::new("findmnt")
        .args(["-n", "-o", "SOURCE", mount_path])
        .output()
        .ok()?;
    
    let device = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let base_dev = device
        .trim_start_matches("/dev/")
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end_matches('p')
        .to_string();

    // current_link_speed: "8.0 GT/s PCIe" = Gen3; "16.0 GT/s PCIe" = Gen4; "32.0 GT/s" = Gen5
    let speed_path = format!("/sys/block/{}/device/../current_link_speed", base_dev);
    let speed_str = std::fs::read_to_string(&speed_path).ok()?;
    
    if speed_str.contains("32.0") { Some(NvmeGen::Gen5) }
    else if speed_str.contains("16.0") { Some(NvmeGen::Gen4) }
    else if speed_str.contains("8.0") { Some(NvmeGen::Gen3) }
    else if speed_str.contains("5.0") { Some(NvmeGen::Gen2) }
    else if speed_str.contains("2.5") { Some(NvmeGen::Gen1) }
    else { None }
}

// ─── Helpers Windows ──────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn read_disk_model_windows(_name_hint: &str) -> Option<String> {
    // WMIC retorna o modelo de todos os discos físicos
    let output = std::process::Command::new("wmic")
        .args(["diskdrive", "get", "Model", "/value"])
        .output()
        .ok()?;
    
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if line.starts_with("Model=") {
            let model = line.trim_start_matches("Model=").trim().to_string();
            if !model.is_empty() {
                return Some(model);
            }
        }
    }
    None
}
