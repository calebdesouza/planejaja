use nodestor_core::OsType;

/// Detecta o SO atual e retorna (tipo, versão).
pub fn detect_os() -> (OsType, String) {
    #[cfg(target_os = "windows")]
    {
        let version = read_windows_version();
        (OsType::Windows, version)
    }
    #[cfg(target_os = "linux")]
    {
        let version = read_linux_version();
        (OsType::Linux, version)
    }
    #[cfg(target_os = "macos")]
    {
        (OsType::MacOs, "macOS".to_string())
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        (OsType::Unknown, "Unknown OS".to_string())
    }
}

#[cfg(target_os = "linux")]
fn read_linux_version() -> String {
    // Lê /proc/version para obter versão do kernel
    std::fs::read_to_string("/proc/version")
        .unwrap_or_else(|_| "Linux unknown".to_string())
        .lines()
        .next()
        .unwrap_or("Linux unknown")
        .to_string()
}

#[cfg(target_os = "windows")]
fn read_windows_version() -> String {
    use std::process::Command;
    let output = Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .unwrap_or_else(|_| std::process::Output {
            status: std::process::ExitStatus::default(),
            stdout: b"Windows unknown".to_vec(),
            stderr: vec![],
        });
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_string()
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn read_unknown_version() -> String {
    "Unknown".to_string()
}
