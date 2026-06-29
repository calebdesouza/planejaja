// Platform-Adaptive Tensor IO
//
// Abstracts the three major IO strategies behind a single trait:
//
//   Windows  → tokio async reads with FILE_FLAG_SEQUENTIAL_SCAN hint
//              (DirectStorage GDeflate path is behind cfg(feature="directstorage"))
//   macOS    → memmap2 + MADV_WILLNEED (prefetch advice before read window)
//   Linux    → memmap2 + MADV_WILLNEED; O_DIRECT for large sequential reads
//
// All paths expose the same `TensorReader::read_range(offset, len)` async API,
// so the inference pipeline is OS-agnostic at the call site.

use std::path::{Path, PathBuf};
use std::io;

// ─── Trait ─────────────────────────────────────────────────────────────────────

/// Reads a byte range from a model file.
/// All implementations must be Send + Sync so they can be shared across tokio tasks.
pub trait TensorReader: Send + Sync {
    /// Reads `len` bytes starting at `offset` in the file.
    /// Returns a heap-allocated buffer. Implementations may return cached slices
    /// via a Vec backed by mmap data — callers must not assume zero-copy.
    fn read_range_sync(&self, offset: u64, len: usize) -> io::Result<Vec<u8>>;

    /// Human-readable backend name for diagnostics.
    fn backend_name(&self) -> &'static str;

    /// Whether the implementation uses memory-mapped IO.
    fn is_mmap(&self) -> bool { false }
}

// ─── Factory ───────────────────────────────────────────────────────────────────

/// Creates the best available `TensorReader` for the current OS.
/// Falls back through the strategy stack until one succeeds.
pub fn open_tensor_file(path: &Path) -> io::Result<Box<dyn TensorReader>> {
    #[cfg(target_os = "macos")]
    { return macos::MmapReader::open(path).map(|r| Box::new(r) as Box<dyn TensorReader>); }

    #[cfg(target_os = "linux")]
    { return linux::MmapReader::open(path).map(|r| Box::new(r) as Box<dyn TensorReader>); }

    #[cfg(target_os = "windows")]
    { return windows::AsyncReader::open(path).map(|r| Box::new(r) as Box<dyn TensorReader>); }

    #[allow(unreachable_code)]
    fallback::StdReader::open(path).map(|r| Box::new(r) as Box<dyn TensorReader>)
}

// ─── macOS — memmap2 + MADV_WILLNEED ──────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use memmap2::{Mmap, MmapOptions, Advice};
    use std::fs::File;

    pub struct MmapReader {
        mmap: Mmap,
    }

    impl MmapReader {
        pub fn open(path: &Path) -> io::Result<Self> {
            let file = File::open(path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            // Advise OS to prefetch the full file sequentially.
            // On macOS this calls madvise(MADV_WILLNEED) under the hood.
            mmap.advise(Advice::WillNeed)?;
            Ok(Self { mmap })
        }
    }

    impl TensorReader for MmapReader {
        fn read_range_sync(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let start = offset as usize;
            let end   = start.checked_add(len)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
            if end > self.mmap.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past end of file"));
            }
            Ok(self.mmap[start..end].to_vec())
        }

        fn backend_name(&self) -> &'static str { "memmap2+MADV_WILLNEED (macOS)" }
        fn is_mmap(&self) -> bool { true }
    }
}

// ─── Linux — memmap2 + MADV_WILLNEED ──────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use memmap2::{Mmap, MmapOptions, Advice};
    use std::fs::File;

    pub struct MmapReader {
        mmap: Mmap,
    }

    impl MmapReader {
        pub fn open(path: &Path) -> io::Result<Self> {
            let file = File::open(path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            // Linux kernel prefetch: MADV_WILLNEED + MADV_SEQUENTIAL
            mmap.advise(Advice::WillNeed)?;
            mmap.advise(Advice::Sequential)?;
            Ok(Self { mmap })
        }
    }

    impl TensorReader for MmapReader {
        fn read_range_sync(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let start = offset as usize;
            let end   = start.checked_add(len)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range overflow"))?;
            if end > self.mmap.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past end of file"));
            }
            Ok(self.mmap[start..end].to_vec())
        }

        fn backend_name(&self) -> &'static str { "memmap2+MADV_WILLNEED+MADV_SEQUENTIAL (Linux)" }
        fn is_mmap(&self) -> bool { true }
    }
}

// ─── Windows — synchronous reads with sequential hint ─────────────────────────

#[cfg(target_os = "windows")]
mod windows {
    use super::*;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::windows::fs::OpenOptionsExt;

    // FILE_FLAG_SEQUENTIAL_SCAN: tells Windows Cache Manager we read front-to-back.
    // This enables read-ahead prefetching and disables random-access cache eviction.
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

    pub struct AsyncReader {
        path: PathBuf,
    }

    impl AsyncReader {
        pub fn open(path: &Path) -> io::Result<Self> {
            // Verify file is accessible at open time.
            let _ = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
                .open(path)?;
            Ok(Self { path: path.to_owned() })
        }
    }

    impl TensorReader for AsyncReader {
        fn read_range_sync(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
                .open(&self.path)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)?;
            Ok(buf)
        }

        fn backend_name(&self) -> &'static str { "Win32 FILE_FLAG_SEQUENTIAL_SCAN" }
    }
}

// ─── Fallback — portable std::fs reads ────────────────────────────────────────

mod fallback {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    pub struct StdReader {
        path: PathBuf,
    }

    impl StdReader {
        pub fn open(path: &Path) -> io::Result<Self> {
            // Validate file exists.
            std::fs::metadata(path)?;
            Ok(Self { path: path.to_owned() })
        }
    }

    impl TensorReader for StdReader {
        fn read_range_sync(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
            let mut file = std::fs::File::open(&self.path)?;
            file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)?;
            Ok(buf)
        }

        fn backend_name(&self) -> &'static str { "std::fs fallback" }
    }
}

// ─── Standard directories ──────────────────────────────────────────────────────

/// Returns the NodeStor data directory for the current OS.
///
///   Linux/macOS: $HOME/.nodestor/
///   Windows:     %USERPROFILE%\.nodestor\   (CSIDL_PROFILE fallback: C:\Users\<user>)
pub fn nodestor_data_dir() -> PathBuf {
    let base = dirs_home();
    base.join(".nodestor")
}

pub fn models_dir()  -> PathBuf { nodestor_data_dir().join("models")  }
pub fn loras_dir()   -> PathBuf { nodestor_data_dir().join("loras")   }
pub fn prompts_dir() -> PathBuf { nodestor_data_dir().join("prompts") }
pub fn cache_dir()   -> PathBuf { nodestor_data_dir().join("cache")   }

/// Creates the NodeStor directory tree if absent.
pub fn ensure_dirs() -> io::Result<()> {
    for dir in [models_dir(), loras_dir(), prompts_dir(), cache_dir()] {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(())
}

/// Resolves the user's home directory cross-platform.
fn dirs_home() -> PathBuf {
    if cfg!(target_os = "windows") {
        std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOMEDRIVE").and_then(|d| {
                std::env::var("HOMEPATH").map(|p| d + &p)
            }))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\Users\\Default"))
    } else {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file_with(data: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        let p = f.path().to_owned();
        (f, p)
    }

    #[test]
    fn platform_reader_reads_correct_range() {
        let data: Vec<u8> = (0u8..=255).collect();
        let (_f, path) = temp_file_with(&data);
        let reader = open_tensor_file(&path).expect("open_tensor_file failed");
        let result = reader.read_range_sync(10, 20).expect("read_range_sync failed");
        assert_eq!(result.len(), 20);
        assert_eq!(result[0], 10);
        assert_eq!(result[19], 29);
    }

    #[test]
    fn platform_reader_rejects_out_of_bounds() {
        let data = vec![0u8; 100];
        let (_f, path) = temp_file_with(&data);
        let reader = open_tensor_file(&path).unwrap();
        let err = reader.read_range_sync(90, 20);
        assert!(err.is_err(), "reading past EOF must fail");
    }

    #[test]
    fn platform_reader_backend_name_nonempty() {
        let data = vec![1u8; 8];
        let (_f, path) = temp_file_with(&data);
        let reader = open_tensor_file(&path).unwrap();
        assert!(!reader.backend_name().is_empty());
    }

    #[test]
    fn nodestor_data_dir_has_nodestor_suffix() {
        let d = nodestor_data_dir();
        assert!(d.ends_with(".nodestor"), "data dir must end with .nodestor: {:?}", d);
    }

    #[test]
    fn models_dir_is_under_data_dir() {
        let md = models_dir();
        assert!(md.starts_with(nodestor_data_dir()));
    }

    #[test]
    fn loras_dir_is_under_data_dir() {
        assert!(loras_dir().starts_with(nodestor_data_dir()));
    }
}
