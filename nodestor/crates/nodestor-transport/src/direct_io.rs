//! Direct I/O — Bypass do Page Cache do Sistema Operacional.
//!
//! ## O Problema
//! Quando lemos um arquivo com `File::open()` normal, o OS copia os bytes para o
//! **Page Cache** (cache de arquivos em RAM do kernel). Isso:
//! 1. Consome RAM silenciosamente
//! 2. Adiciona uma cópia extra: `SSD → RAM(kernel) → RAM(app) → VRAM`
//!
//! ## A Solução: Direct I/O
//! Com `FILE_FLAG_NO_BUFFERING` (Windows) ou `O_DIRECT` (Linux), ordenamos ao OS
//! para **pular o Page Cache** e entregar os bytes direto no nosso buffer:
//! `SSD DMA Engine → Pinned Memory → GPU DMA Engine → VRAM`
//!
//! A CPU não toca nos bytes. Nenhuma cópia intermediária.
//!
//! ## Disponibilidade Universal
//! - Windows: `FILE_FLAG_NO_BUFFERING` — disponível desde Windows XP
//! - Linux: `O_DIRECT` — disponível desde kernel 2.6
//! - Não requer privilégios de administrador
//! - Não requer BIOS especial
//! - Não requer drivers adicionais
//!
//! Se não suportado (e.g., sistemas de arquivos em rede), fallback automático
//! para leitura buffered normal.

use nodestor_core::NodeStorError;
use tracing::{debug, warn};

/// Configuração de alinhamento para Direct I/O.
/// O SSD exige que o buffer e o offset sejam alinhados ao tamanho do setor.
#[derive(Debug, Clone, Copy)]
pub struct DirectIOAlignment {
    /// Tamanho do setor do dispositivo (512 ou 4096 bytes geralmente).
    pub sector_size: usize,
}

impl Default for DirectIOAlignment {
    fn default() -> Self {
        Self { sector_size: 4096 } // 4096 é seguro para NVMe e SATA
    }
}

impl DirectIOAlignment {
    /// Arredonda `size` para cima para o múltiplo mais próximo do setor.
    pub fn align_size(&self, size: usize) -> usize {
        (size + self.sector_size - 1) & !(self.sector_size - 1)
    }

    /// Arredonda `offset` para baixo para múltiplo do setor.
    pub fn align_offset(&self, offset: u64) -> u64 {
        offset & !(self.sector_size as u64 - 1)
    }
}

/// Leitor Direct I/O com bypass do Page Cache.
///
/// Modos de operação:
/// - **Modo Direto**: `O_DIRECT` / `FILE_FLAG_NO_BUFFERING` — SSD → Pinned Memory (zero cache)
/// - **Modo Buffered Fallback**: `File::open()` normal — se Direct I/O não suportado
pub struct DirectIOReader {
    path: String,
    alignment: DirectIOAlignment,
    use_direct: bool,
    file_size: u64,
}

fn io_err(msg: impl Into<String>) -> NodeStorError {
    NodeStorError::TransferFailed(msg.into())
}

impl DirectIOReader {
    /// Abre o arquivo para Direct I/O.
    /// Se o filesystem não suportar Direct I/O (e.g., NFS, tmpfs),
    /// automaticamente usa fallback buffered.
    pub fn open(path: &str) -> Result<Self, NodeStorError> {
        let file_size = std::fs::metadata(path)
            .map_err(|e| io_err(format!("Não foi possível abrir {}: {}", path, e)))?
            .len();

        let (use_direct, alignment) = Self::probe_direct(path);

        if use_direct {
            debug!("DirectIOReader: {} — Direct I/O ativo (bypass Page Cache, setor {}B)",
                path, alignment.sector_size);
        } else {
            warn!("DirectIOReader: {} — Fallback para I/O buffered (Direct I/O não disponível)", path);
        }

        Ok(Self {
            path: path.to_string(),
            alignment,
            use_direct,
            file_size,
        })
    }

    fn probe_direct(path: &str) -> (bool, DirectIOAlignment) {
        let _ = path;
        #[cfg(target_os = "windows")]
        return (true, DirectIOAlignment { sector_size: 4096 });

        #[cfg(target_os = "linux")]
        return Self::probe_linux(path);

        #[cfg(target_os = "macos")]
        return (true, DirectIOAlignment { sector_size: 4096 });

        #[cfg(target_os = "android")]
        return Self::probe_android(path);

        #[cfg(not(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "android",
        )))]
        (false, DirectIOAlignment::default())
    }

    #[cfg(target_os = "linux")]
    fn probe_linux(path: &str) -> (bool, DirectIOAlignment) {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(_) => (true, DirectIOAlignment { sector_size: 4096 }),
            Err(e) => {
                warn!("O_DIRECT não suportado em {}: {} — usando buffered", path, e);
                (false, DirectIOAlignment::default())
            }
        }
    }

    /// Lê `size` bytes a partir de `offset` no arquivo.
    ///
    /// **Modo Direto**: O SSD faz DMA direto para `dst_ptr` sem passar pelo kernel.
    /// Se `dst_ptr` for um ponteiro Pinned Memory (`GpuBuffer::mapped_mut_ptr()`),
    /// o dado viaja: `SSD → RAM Pinned → GPU` sem nunca ser copiado pela CPU.
    pub fn read_at(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        if self.use_direct {
            self.read_direct(offset, size, dst)
        } else {
            self.read_buffered(offset, size, dst)
        }
    }

    /// Retorna um Vec<u8> com os bytes lidos de forma otimizada.
    pub fn read_chunk(&self, offset: u64, size: usize) -> Result<Vec<u8>, NodeStorError> {
        let aligned_size = self.alignment.align_size(size);
        let mut buf = vec![0u8; aligned_size];
        let read = self.read_at(offset, size, &mut buf)?;
        buf.truncate(read);
        Ok(buf)
    }

    /// Tamanho total do arquivo.
    pub fn file_size(&self) -> u64 { self.file_size }

    /// `true` se Direct I/O está ativo (sem Page Cache).
    pub fn is_direct(&self) -> bool { self.use_direct }

    // Direct I/O real via FILE_FLAG_NO_BUFFERING + FILE_FLAG_OVERLAPPED.
    // Usamos std::os::windows::fs::OpenOptionsExt::custom_flags() ao invés de
    // CreateFileW direto — é idiomático em Rust e evita problemas de tipos do windows-sys.
    // O SSD faz DMA direto para dst sem passar pelo Page Cache do Windows NT.
    #[cfg(target_os = "windows")]
    fn read_direct(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{GetLastError, ERROR_IO_PENDING, TRUE};
        use windows_sys::Win32::Storage::FileSystem::ReadFile;
        use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};

        // u32 literals: evita newtypes do windows-sys que causam mismatched types
        const FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FLAG_OVERLAPPED: u32   = 0x4000_0000;

        let aligned_offset = self.alignment.align_offset(offset);
        let prefix_skip    = (offset - aligned_offset) as usize;
        let aligned_size   = self.alignment.align_size(prefix_skip + size.min(dst.len()));

        // Abre com NO_BUFFERING: bypass do Page Cache do Windows NT
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FLAG_NO_BUFFERING | FLAG_OVERLAPPED)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(e) => {
                warn!("DirectIO Windows: custom_flags falhou ({}), usando buffered", e);
                return self.read_buffered(offset, size, dst);
            }
        };

        let h = file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;

        // Closure que executa ReadFile com OVERLAPPED (I/O assíncrono real)
        let do_read = |buf_ptr: *mut u8, buf_size: usize| -> Result<usize, NodeStorError> {
            let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
            ov.Anonymous.Anonymous.Offset     = aligned_offset as u32;
            ov.Anonymous.Anonymous.OffsetHigh = (aligned_offset >> 32) as u32;
            let mut n: u32 = 0;

            let ok = unsafe { ReadFile(h, buf_ptr as *mut _, buf_size as u32, &mut n, &mut ov) };
            if ok == 0 {
                let err = unsafe { GetLastError() };
                if err == ERROR_IO_PENDING {
                    // DMA em andamento — espera conclusão sem bloquear outros threads
                    let done = unsafe { GetOverlappedResult(h, &ov, &mut n, TRUE) };
                    if done == 0 {
                        return Err(io_err(format!(
                            "GetOverlappedResult falhou: err={}",
                            unsafe { GetLastError() }
                        )));
                    }
                } else {
                    return Err(io_err(format!("ReadFile(OVERLAPPED) err={}", err)));
                }
            }
            Ok(n as usize)
        };

        // Se dst não está alinhado a 4096B, usa buffer temporário alinhado
        let needs_intermediate = (dst.as_ptr() as usize) % self.alignment.sector_size != 0
            || dst.len() < aligned_size;

        if needs_intermediate {
            let layout = std::alloc::Layout::from_size_align(
                aligned_size, self.alignment.sector_size
            ).map_err(|e| io_err(e.to_string()))?;
            let tmp = unsafe { std::alloc::alloc(layout) };
            if tmp.is_null() { return Err(io_err("OOM: Direct I/O Windows buffer")); }
            match do_read(tmp, aligned_size) {
                Ok(n) => {
                    let copy_len = (n.saturating_sub(prefix_skip)).min(size).min(dst.len());
                    unsafe {
                        std::ptr::copy_nonoverlapping(tmp.add(prefix_skip), dst.as_mut_ptr(), copy_len);
                        std::alloc::dealloc(tmp, layout);
                    }
                    Ok(copy_len)
                }
                Err(e) => { unsafe { std::alloc::dealloc(tmp, layout) }; Err(e) }
            }
        } else {
            // Pinned Memory do Vulkan: DMA direto — zero cópias, zero CPU
            do_read(dst.as_mut_ptr(), aligned_size)
                .map(|n| n.saturating_sub(prefix_skip).min(size))
        }
    }

    // ─── Plataforma: Linux ────────────────────────────────────────────────────

    #[cfg(target_os = "linux")]
    fn read_direct(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        let aligned_offset = self.alignment.align_offset(offset);
        let prefix_skip = (offset - aligned_offset) as usize;

        let use_intermediate = (dst.as_ptr() as usize) % self.alignment.sector_size != 0;

        if use_intermediate {
            // dst não alinhado: lê para buffer alinhado e copia.
            // Em produção, Pinned Memory já é alinhada — este branch é raro.
            let _aligned_size = self.alignment.align_size(prefix_skip + size);
            let layout = std::alloc::Layout::from_size_align(_aligned_size, self.alignment.sector_size)
                .map_err(|e| io_err(e.to_string()))?;
            unsafe {
                let aligned_buf = std::alloc::alloc(layout);
                if aligned_buf.is_null() {
                    return Err(io_err("OOM ao alocar buffer alinhado para Direct I/O"));
                }
                let result = self.pread_direct(aligned_offset, _aligned_size, aligned_buf);
                let copy_len = size.min(dst.len());
                std::ptr::copy_nonoverlapping(aligned_buf.add(prefix_skip), dst.as_mut_ptr(), copy_len);
                std::alloc::dealloc(aligned_buf, layout);
                result.map(|_| copy_len)
            }
        } else {
            // dst já alinhado (Pinned Memory via Vulkan): pread direto!
            // Zero cópias intermediárias — SSD DMA → dst (Pinned RAM)
            self.pread_direct(aligned_offset, size.min(dst.len()), dst.as_mut_ptr())
                .map(|n| n.min(size))
        }
    }

    #[cfg(target_os = "linux")]
    fn pread_direct(&self, offset: u64, size: usize, ptr: *mut u8) -> Result<usize, NodeStorError> {
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&self.path)
            .map_err(|e: std::io::Error| io_err(e.to_string()))?;

        let n = unsafe {
            libc::pread(
                file.as_raw_fd(),
                ptr as *mut libc::c_void,
                size,
                offset as libc::off_t,
            )
        };
        if n < 0 {
            Err(io_err(format!("pread(O_DIRECT) falhou: errno={}", unsafe { *libc::__errno_location() })))
        } else {
            Ok(n as usize)
        }
    }

    // ─── Plataforma: Outros (macOS, etc.) ────────────────────────────────────

    // ─── Plataforma: macOS ────────────────────────────────────────────────────
    // fcntl(F_NOCACHE, 1): instrui o kernel Darwin a NÃO usar o Unified Buffer Cache.
    // No Apple Silicon (UMA), CPU e GPU compartilham a mesma memória física —
    // o dado que o SSD escreve na RAM já está na GPU sem cópia PCIe.
    #[cfg(target_os = "macos")]
    fn read_direct(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::io::AsRawFd;

        let file = std::fs::File::open(&self.path)
            .map_err(|e| io_err(e.to_string()))?;

        // F_NOCACHE = 48 em macOS: desativa o page cache para este file descriptor.
        // O dado vai direto do SSD para o nosso buffer sem passar pelo cache do kernel.
        let nocache_ret = unsafe { libc::fcntl(file.as_raw_fd(), 48 /* F_NOCACHE */, 1i32) };
        if nocache_ret < 0 {
            // fcntl pode falhar em FS de rede (NFS, SMB) — fallback gracioso
            debug!("macOS F_NOCACHE indisponível, usando buffered read");
            return self.read_buffered(offset, size, dst);
        }

        let mut f = file;
        f.seek(SeekFrom::Start(offset)).map_err(|e| io_err(e.to_string()))?;
        let len = size.min(dst.len());
        f.read(&mut dst[..len]).map_err(|e| io_err(e.to_string()))
    }

    // ─── Plataforma: Android ─────────────────────────────────────────────────
    // Android (Linux) suporta O_DIRECT em storage interno (API Level 26+).
    // `probe_android` testa se O_DIRECT funciona antes de usar.
    #[cfg(target_os = "android")]
    fn probe_android(path: &str) -> (bool, DirectIOAlignment) {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(_) => {
                debug!("Android: O_DIRECT suportado em {}", path);
                (true, DirectIOAlignment { sector_size: 4096 })
            }
            Err(e) => {
                warn!("Android: O_DIRECT não suportado ({}), usando buffered", e);
                (false, DirectIOAlignment::default())
            }
        }
    }

    #[cfg(target_os = "android")]
    fn read_direct(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        // Android usa O_DIRECT idêntico ao Linux — reutiliza a implementação pread
        let aligned_offset = self.alignment.align_offset(offset);
        let prefix_skip = (offset - aligned_offset) as usize;

        let use_intermediate = (dst.as_ptr() as usize) % self.alignment.sector_size != 0;

        if use_intermediate {
            let aligned_size = self.alignment.align_size(prefix_skip + size);
            let layout = std::alloc::Layout::from_size_align(aligned_size, self.alignment.sector_size)
                .map_err(|e| io_err(e.to_string()))?;
            unsafe {
                let buf = std::alloc::alloc(layout);
                if buf.is_null() { return Err(io_err("OOM: Android Direct I/O")); }
                let r = self.pread_direct(aligned_offset, aligned_size, buf);
                let copy_len = size.min(dst.len());
                std::ptr::copy_nonoverlapping(buf.add(prefix_skip), dst.as_mut_ptr(), copy_len);
                std::alloc::dealloc(buf, layout);
                r.map(|_| copy_len)
            }
        } else {
            self.pread_direct(aligned_offset, size.min(dst.len()), dst.as_mut_ptr())
                .map(|n| n.min(size))
        }
    }

    // ─── Plataforma: Outros ───────────────────────────────────────────────────
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "android",
    )))]
    fn read_direct(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        self.read_buffered(offset, size, dst)
    }

    // ─── Fallback Universal ───────────────────────────────────────────────────

    fn read_buffered(&self, offset: u64, size: usize, dst: &mut [u8]) -> Result<usize, NodeStorError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&self.path)
            .map_err(|e: std::io::Error| io_err(e.to_string()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e: std::io::Error| io_err(e.to_string()))?;
        let len = size.min(dst.len());
        file.read(&mut dst[..len])
            .map_err(|e: std::io::Error| io_err(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_size() {
        let a = DirectIOAlignment { sector_size: 4096 };
        assert_eq!(a.align_size(1), 4096);
        assert_eq!(a.align_size(4096), 4096);
        assert_eq!(a.align_size(4097), 8192);
        assert_eq!(a.align_size(8192), 8192);
    }

    #[test]
    fn test_alignment_offset() {
        let a = DirectIOAlignment { sector_size: 4096 };
        assert_eq!(a.align_offset(0), 0);
        assert_eq!(a.align_offset(4096), 4096);
        assert_eq!(a.align_offset(4097), 4096);
        assert_eq!(a.align_offset(8191), 4096);
        assert_eq!(a.align_offset(8192), 8192);
    }

    #[test]
    fn test_reader_not_found() {
        let result = DirectIOReader::open("/nonexistent_nodestor_test_12345.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_temp_file_buffered() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_directio.bin");
        let data = vec![0xABu8; 8192];
        std::fs::File::create(&path).unwrap().write_all(&data).unwrap();

        let reader = DirectIOReader::open(path.to_str().unwrap()).unwrap();
        assert_eq!(reader.file_size(), 8192);

        let chunk = reader.read_chunk(0, 4096).unwrap();
        assert_eq!(chunk.len(), 4096);
        assert!(chunk.iter().all(|&b| b == 0xAB));
    }
}
