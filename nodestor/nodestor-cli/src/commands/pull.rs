use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use futures::StreamExt;

pub async fn cmd_pull(model_id: &str, filename: &str) -> Result<()> {
    println!("\n📥 NodeStor — HuggingFace Model Pull\n{}", "─".repeat(50));
    
    let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, filename);
    
    println!("📡 Conectando ao repositório: {}", model_id);
    println!("📄 Solicitando arquivo: {}", filename);
    
    let client = Client::new();
    let res = client.get(&url).send().await?;
    
    if !res.status().is_success() {
        return Err(anyhow::anyhow!("Falha no download. O arquivo ou modelo não foi encontrado. (Status: {})", res.status()));
    }

    let total_size = res.content_length().unwrap_or(0);
    
    // Configura diretório de download local
    let mut model_dir = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Não foi possível encontrar o diretório home"))?;
    model_dir.push(".nodestor");
    model_dir.push("models");
    tokio::fs::create_dir_all(&model_dir).await?;
    
    let dest_path = model_dir.join(filename);
    
    println!("💾 Salvando em: {}", dest_path.display());
    
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));

    let mut file = File::create(&dest_path).await?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk).await?;
        let new = std::cmp::min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }
    
    pb.finish_with_message("✅ Download completo!");
    println!("\n✅ Modelo salvo com sucesso em: {}", dest_path.display());

    Ok(())
}
