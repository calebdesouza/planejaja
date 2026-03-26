use std::path::PathBuf;
use walkdir::WalkDir;

/// Encontra modelos (.gguf ou .safetensors) em locais comuns.
pub fn find_models_on_ssd() -> Vec<PathBuf> {
    let mut models = Vec::new();
    let mut paths_to_search = Vec::new();

    // 1. Diretório Atual
    if let Ok(p) = std::env::current_dir() {
        paths_to_search.push(p);
    }

    // 2. Diretórios Comuns (Home, Downloads)
    if let Some(home) = dirs::home_dir() {
        paths_to_search.push(home.join("Downloads"));
        paths_to_search.push(home.join("Documents"));
        paths_to_search.push(home.join(".cache").join("huggingface"));
    }

    for root in paths_to_search {
        if !root.exists() { continue; }
        
        let walker = WalkDir::new(root)
            .max_depth(3) // Evita varredura infinita
            .follow_links(false);

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "gguf" || ext == "safetensors" {
                        models.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    models.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    models.reverse(); // Maiores primeiro
    models
}

/// Menu interativo para selecionar um modelo encontrado.
pub fn interactive_model_picker() -> anyhow::Result<Option<String>> {
    use dialoguer::{theme::ColorfulTheme, Select};

    println!("\n🔍 Varrendo SSD em busca de modelos AI...");
    let models = find_models_on_ssd();

    if models.is_empty() {
        println!("⚠️ Nenhum modelo (.gguf ou .safetensors) encontrado.");
        return Ok(None);
    }

    let mut options: Vec<String> = models.iter()
        .map(|p| format!("{} ({:.2} GB)", p.file_name().unwrap_or_default().to_string_lossy(), std::fs::metadata(p).unwrap().len() as f64 / 1e9))
        .collect();
    
    options.push("[DIGITAR CAMINHO MANUAL]".to_string());
    options.push("[CANCELAR]".to_string());

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Selecione o Modelo para o Motor")
        .items(&options)
        .default(0)
        .interact_opt()?;

    match selection {
        Some(i) if i < models.len() => Ok(Some(models[i].to_string_lossy().into_owned())),
        Some(i) if i == models.len() => {
            let path: String = dialoguer::Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Caminho completo do arquivo (vazio para voltar)")
                .allow_empty(true)
                .interact_text()?;
            
            if path.trim().is_empty() {
                Ok(None)
            } else {
                Ok(Some(path))
            }
        },
        _ => Ok(None),
    }
}
