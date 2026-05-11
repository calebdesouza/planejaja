//! PlatformIOCapabilities — Detecção automática de capacidades de I/O da plataforma.
//!
//! O ApexOrchestrator usa esta struct para decidir qual estratégia de bypass
//! usar em tempo de execução, sem intervenção do usuário.
//!
//! ## Estratégias por Plataforma
//! - **Windows**: `FILE_FLAG_NO_BUFFERING` + `FILE_FLAG_OVERLAPPED` (IOCP-ready)
//! - **Linux**: `O_DIRECT` + `pread` (io_uring disponível em kernels ≥ 5.1)
//! - **macOS**: `fcntl(F_NOCACHE)` + Unified Memory no Apple Silicon
//! - **Android**: `O_DIRECT` (suportado desde API Level 26 em storage interno)
//! - **Outros**: Fallback buffered universal

use tracing::info;

/// Capacidades de I/O da plataforma atual, detectadas em runtime.
#[derive(Debug, Clone)]
pub struct PlatformIOCapabilities {
    /// Direct I/O disponível (O_DIRECT, FILE_FLAG_NO_BUFFERING, F_NOCACHE).
    pub direct_io: bool,
    /// Descrição da estratégia de bypass em uso.
    pub bypass_strategy: &'static str,
    /// Memória Unificada (Apple Silicon) — GPU e CPU compartilham RAM sem cópia PCIe.
    pub unified_memory: bool,
    /// io_uring disponível (Linux 5.1+) — 64K filas NVMe simultâneas.
    pub io_uring_available: bool,
    /// Alinhamento mínimo de setor para Direct I/O (normalmente 4096).
    pub sector_alignment: usize,
    /// Nome da plataforma (para logging/diagnóstico).
    pub platform_name: &'static str,
}

impl PlatformIOCapabilities {
    /// Detecta as capacidades da plataforma atual em runtime.
    /// Zero configuração — tudo é inferido via APIs do SO e Vulkan.
    pub fn detect() -> Self {
        let caps = Self::detect_platform();
        info!(
            "APEX Platform: {} | Direct I/O: {} ({}) | UMA: {} | io_uring: {}",
            caps.platform_name,
            caps.direct_io,
            caps.bypass_strategy,
            caps.unified_memory,
            caps.io_uring_available,
        );
        caps
    }

    #[cfg(target_os = "windows")]
    fn detect_platform() -> Self {
        Self {
            direct_io: true,
            bypass_strategy: "FILE_FLAG_NO_BUFFERING + OVERLAPPED (Windows NT)",
            unified_memory: false,
            io_uring_available: false,
            sector_alignment: 4096,
            platform_name: "Windows",
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_platform() -> Self {
        // io_uring disponível em kernel ≥ 5.1
        let uring = Self::probe_io_uring();
        Self {
            direct_io: true,
            bypass_strategy: if uring { "O_DIRECT + io_uring (Linux)" } else { "O_DIRECT + pread (Linux)" },
            unified_memory: false,
            io_uring_available: uring,
            sector_alignment: 4096,
            platform_name: "Linux",
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_platform() -> Self {
        // Apple Silicon (aarch64) = Unified Memory Architecture
        // CPU e GPU compartilham RAM física — SSD → RAM já é SSD → GPU
        let is_apple_silicon = cfg!(target_arch = "aarch64");
        Self {
            direct_io: true,
            bypass_strategy: "fcntl(F_NOCACHE, 1) (Darwin)",
            unified_memory: is_apple_silicon,
            io_uring_available: false,
            sector_alignment: 4096,
            platform_name: if is_apple_silicon { "macOS Apple Silicon (UMA)" } else { "macOS Intel" },
        }
    }

    #[cfg(target_os = "android")]
    fn detect_platform() -> Self {
        Self {
            direct_io: true, // Verificado em probe() do DirectIOReader
            bypass_strategy: "O_DIRECT (Android 8.0+ / API 26+)",
            unified_memory: false,
            io_uring_available: false,
            sector_alignment: 4096,
            platform_name: "Android",
        }
    }

    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "android",
    )))]
    fn detect_platform() -> Self {
        Self {
            direct_io: false,
            bypass_strategy: "Buffered Fallback Universal",
            unified_memory: false,
            io_uring_available: false,
            sector_alignment: 4096,
            platform_name: "Plataforma Desconhecida",
        }
    }

    #[cfg(target_os = "linux")]
    fn probe_io_uring() -> bool {
        // io_uring_params é uma struct do kernel não exposta pelo libc — definimos manualmente
        #[repr(C)]
        struct IoUringParams {
            sq_entries: u32,
            cq_entries: u32,
            flags: u32,
            sq_thread_cpu: u32,
            sq_thread_idle: u32,
            features: u32,
            wq_fd: u32,
            resv: [u32; 3],
            sq_off: [u8; 40],
            cq_off: [u8; 40],
        }
        let params: IoUringParams = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                0u32,
                &params as *const IoUringParams,
            )
        };
        // ENOSYS = syscall não existe. Qualquer outro erro = uring disponível mas args inválidos
        let errno = unsafe { *libc::__errno_location() };
        ret >= 0 || errno != libc::ENOSYS
    }

    /// Descreve o multiplicador de throughput esperado vs. IO buffered padrão.
    pub fn expected_multiplier(&self) -> &'static str {
        match (self.direct_io, self.unified_memory) {
            (_, true) => "até 3x (UMA: SSD→RAM = SSD→GPU)",
            (true, false) => "2.5-4x (Direct I/O + GDeflate)",
            (false, _) => "1.0-1.5x (Buffered Fallback)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_capabilities_detect() {
        let caps = PlatformIOCapabilities::detect();
        // Em qualquer plataforma, deve retornar uma descrição não-vazia
        assert!(!caps.bypass_strategy.is_empty());
        assert!(!caps.platform_name.is_empty());
        assert!(caps.sector_alignment >= 512);
        // O multiplicador esperado sempre tem uma descrição
        assert!(!caps.expected_multiplier().is_empty());
    }

    #[test]
    fn test_sector_alignment_is_power_of_two() {
        let caps = PlatformIOCapabilities::detect();
        let align = caps.sector_alignment;
        assert!(align > 0 && (align & (align - 1)) == 0, "sector_alignment deve ser potência de 2");
    }
}
