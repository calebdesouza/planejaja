//! nodestor-transport — Abstração de I/O para transporte SSD → GPU.
//!
//! Implementa o trait `DataTransport` para múltiplos backends,
//! selecionados automaticamente com base no hardware detectado.

mod fallback;
#[cfg(target_os = "linux")]
mod io_uring_transport;

pub use fallback::PreadFallback;
use nodestor_core::{DataTransport, HardwareProfile, NodeStorError, TransportBackend};
use tracing::info;

/// Cria o melhor transporte disponível para este HardwareProfile.
pub fn create_transport(profile: &HardwareProfile) -> Box<dyn DataTransport> {
    info!("Criando transporte: {}", profile.recommended_transport);

    match profile.recommended_transport {
        #[cfg(target_os = "linux")]
        TransportBackend::IoUringDmabuf | TransportBackend::IoUringStandard => {
            match io_uring_transport::IoUringTransport::new() {
                Ok(t) => {
                    info!("io_uring transport inicializado com sucesso");
                    Box::new(t)
                }
                Err(e) => {
                    tracing::warn!("io_uring falhou ({}), usando fallback pread", e);
                    Box::new(PreadFallback::new())
                }
            }
        }
        // DirectStorage requer DLLs externas — usando fallback por ora
        // (a implementação completa requer direct-storage-rs + DLLs da Microsoft)
        TransportBackend::DirectStorage => {
            info!("DirectStorage selecionado (requires DLLs). Usando Win32 fallback.");
            Box::new(PreadFallback::new())
        }
        // Para todos os outros casos (incluindo NvidiaGds, Win32Fallback, PreadFallback)
        _ => Box::new(PreadFallback::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::{OsType, TransportBackend};

    fn make_minimal_profile(transport: TransportBackend) -> HardwareProfile {
        HardwareProfile {
            gpus: vec![],
            storage: vec![],
            os: OsType::Windows,
            os_version: "Windows 11".to_string(),
            recommended_transport: transport,
            cpu_cores: 4,
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn test_create_fallback_transport() {
        let profile = make_minimal_profile(TransportBackend::PreadFallback);
        let transport = create_transport(&profile);
        assert_eq!(transport.backend_type(), TransportBackend::PreadFallback);
    }

    #[test]
    fn test_create_transport_for_directstorage_falls_to_fallback() {
        let profile = make_minimal_profile(TransportBackend::DirectStorage);
        let transport = create_transport(&profile);
        // Sem DLLs, cai para PreadFallback
        assert_eq!(transport.backend_name(), "PreadFallback");
    }
}
