use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStorConfig {
    pub inference: InferenceSection,
    pub server: ServerSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSection {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_true")]
    pub speculative: bool,
    #[serde(default = "default_false")]
    pub uncensored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSection {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub api_keys: Vec<String>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 4096 }
fn default_true() -> bool { true }
fn default_false() -> bool { false }
fn default_port() -> u16 { 8080 }

impl Default for NodeStorConfig {
    fn default() -> Self {
        Self {
            inference: InferenceSection {
                temperature: default_temperature(),
                max_tokens: default_max_tokens(),
                speculative: default_true(),
                uncensored: default_false(),
            },
            server: ServerSection {
                port: default_port(),
                api_keys: vec![],
            },
        }
    }
}

impl NodeStorConfig {
    pub fn load_or_default() -> Self {
        if let Some(mut proj_dirs) = dirs::home_dir() {
            proj_dirs.push(".nodestor");
            let config_file = proj_dirs.join("config.toml");
            
            if config_file.exists() {
                if let Ok(content) = std::fs::read_to_string(&config_file) {
                    if let Ok(config) = toml::from_str(&content) {
                        return config;
                    }
                }
            } else {
                // Cria o diretório e o arquivo de config padrão se não existir
                let _ = std::fs::create_dir_all(&proj_dirs);
                let default_config = Self::default();
                if let Ok(toml_string) = toml::to_string(&default_config) {
                    let _ = std::fs::write(&config_file, toml_string);
                }
                return default_config;
            }
        }
        Self::default()
    }
}
