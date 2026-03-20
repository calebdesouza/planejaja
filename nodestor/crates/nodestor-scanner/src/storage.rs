use nodestor_core::{NvmeGen, StorageInfo};
use sysinfo::{Disks, System};

/// Detecta dispositivos de armazenamento disponíveis.
pub fn detect_storage() -> Vec<StorageInfo> {
    let disks = Disks::new_with_refreshed_list();
    let mut results = Vec::new();

    for disk in &disks {
        let path = disk.mount_point().to_string_lossy().to_string();
        let name = disk.name().to_string_lossy().to_string();
        let total_bytes = disk.total_space();
        let available_bytes = disk.available_space();

        // sysinfo indica se é SSD via is_removable + kind
        let is_ssd = matches!(disk.kind(), sysinfo::DiskKind::SSD);

        // Estima geração NVMe baseado no nome e velocidade
        let nvme_gen = estimate_nvme_gen(&name, is_ssd);
        let estimated_read_bps = estimate_read_speed(nvme_gen, is_ssd);

        results.push(StorageInfo {
            path,
            name,
            nvme_gen,
            total_bytes,
            available_bytes,
            is_ssd,
            estimated_read_bps,
        });
    }

    // Ordenar por espaço disponível (mais espaço primeiro)
    results.sort_by(|a, b| b.available_bytes.cmp(&a.available_bytes));
    results
}

fn estimate_nvme_gen(name: &str, is_ssd: bool) -> NvmeGen {
    if !is_ssd {
        return NvmeGen::Unknown;
    }
    let lower = name.to_lowercase();
    // Heurística por nome do modelo
    if lower.contains("gen5") || lower.contains("pcie5") {
        NvmeGen::Gen5
    } else if lower.contains("gen4") || lower.contains("pcie4") || lower.contains("990") || lower.contains("980 pro") {
        NvmeGen::Gen4
    } else if lower.contains("gen3") || lower.contains("pcie3") || lower.contains("970") || lower.contains("860") {
        NvmeGen::Gen3
    } else if lower.contains("gen2") || lower.contains("sata") {
        NvmeGen::Gen2
    } else if is_ssd {
        // SSD sem identificação clara: assume Gen3 (mais comum)
        NvmeGen::Gen3
    } else {
        NvmeGen::Unknown
    }
}

fn estimate_read_speed(gen: NvmeGen, is_ssd: bool) -> u64 {
    if !is_ssd {
        return 200_000_000; // HDD ~200 MB/s
    }
    match gen {
        NvmeGen::Gen5 => 14_000_000_000,
        NvmeGen::Gen4 => 7_000_000_000,
        NvmeGen::Gen3 => 3_500_000_000,
        NvmeGen::Gen2 => 1_500_000_000,
        NvmeGen::Gen1 => 500_000_000,
        NvmeGen::Unknown => 500_000_000, // Conservador
    }
}
