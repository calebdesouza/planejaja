use std::path::PathBuf;
use nodestor_core::NodeStorError;

/// Returns the NodeStor user data directory using OS-native conventions.
///
/// Windows : %USERPROFILE%\.nodestor\
/// Linux   : $HOME/.nodestor/
/// macOS   : $HOME/.nodestor/
///
/// Falls back to the current working directory if the home variable is unset.
pub fn nodestor_home() -> PathBuf {
    home_dir().join(".nodestor")
}

/// Returns the model storage directory, creating it if absent.
pub fn models_dir() -> Result<PathBuf, NodeStorError> {
    let p = nodestor_home().join("models");
    std::fs::create_dir_all(&p).map_err(|e| {
        NodeStorError::InferenceError(format!("models_dir create_dir_all '{}': {}", p.display(), e))
    })?;
    Ok(p)
}

/// Returns the LoRA storage directory, creating it if absent.
pub fn loras_dir() -> Result<PathBuf, NodeStorError> {
    let p = nodestor_home().join("loras");
    std::fs::create_dir_all(&p).map_err(|e| {
        NodeStorError::InferenceError(format!("loras_dir create_dir_all '{}': {}", p.display(), e))
    })?;
    Ok(p)
}

/// Returns the system-prompt storage directory, creating it if absent.
pub fn prompts_dir() -> Result<PathBuf, NodeStorError> {
    let p = nodestor_home().join("prompts");
    std::fs::create_dir_all(&p).map_err(|e| {
        NodeStorError::InferenceError(format!("prompts_dir create_dir_all '{}': {}", p.display(), e))
    })?;
    Ok(p)
}

/// Resolves `path` against `nodestor_home()` if it is a bare filename with no
/// directory component; otherwise returns the path unchanged. This lets callers
/// accept both `"model.gguf"` and `"/abs/path/model.gguf"` transparently.
pub fn resolve_model_path(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.components().count() == 1 && !p.has_root() {
        models_dir().unwrap_or_else(|_| nodestor_home().join("models")).join(p)
    } else {
        p
    }
}

/// OS home directory — platform-agnostic, no external crates.
fn home_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOMEDRIVE").and_then(|d| std::env::var("HOMEPATH").map(|p| d + &p)))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodestor_home_is_valid_path() {
        let p = nodestor_home();
        // Must have at least ".nodestor" as last component
        assert_eq!(p.file_name().and_then(|n| n.to_str()), Some(".nodestor"),
            "nodestor_home() must end in '.nodestor': {}", p.display());
    }

    #[test]
    fn nodestor_home_uses_pathbuf_not_string_concat() {
        let p = nodestor_home();
        // Verify it's a real PathBuf path (no double slashes, correct separator)
        let s = p.to_string_lossy();
        assert!(!s.contains("//"), "path must not contain double slashes: {}", s);
    }

    #[test]
    fn models_dir_creates_and_returns_models_subdir() {
        let p = models_dir().expect("models_dir must succeed");
        assert!(p.ends_with("models"), "models_dir must end in 'models': {}", p.display());
        assert!(p.exists(), "models dir must exist after creation: {}", p.display());
    }

    #[test]
    fn loras_dir_creates_and_returns_loras_subdir() {
        let p = loras_dir().expect("loras_dir must succeed");
        assert!(p.ends_with("loras"), "loras_dir must end in 'loras': {}", p.display());
        assert!(p.exists(), "loras dir must exist after creation: {}", p.display());
    }

    #[test]
    fn prompts_dir_creates_and_returns_prompts_subdir() {
        let p = prompts_dir().expect("prompts_dir must succeed");
        assert!(p.ends_with("prompts"), "prompts_dir must end in 'prompts': {}", p.display());
        assert!(p.exists());
    }

    #[test]
    fn resolve_model_path_absolute_is_unchanged() {
        #[cfg(target_os = "windows")]
        let abs = PathBuf::from(r"C:\models\llama.gguf");
        #[cfg(not(target_os = "windows"))]
        let abs = PathBuf::from("/models/llama.gguf");

        let resolved = resolve_model_path(abs.to_str().unwrap());
        assert_eq!(resolved, abs, "absolute path must pass through unchanged");
    }

    #[test]
    fn resolve_model_path_bare_name_prefixed_with_models_dir() {
        let resolved = resolve_model_path("model.gguf");
        let s = resolved.to_string_lossy();
        assert!(s.contains(".nodestor"), "bare filename must be rooted in nodestor home: {}", s);
        assert!(s.ends_with("model.gguf"), "filename must be preserved: {}", s);
    }

    #[test]
    fn home_dir_returns_non_empty_path() {
        let h = super::home_dir();
        let s = h.to_string_lossy();
        assert!(!s.is_empty(), "home_dir() must not return empty path");
    }
}
