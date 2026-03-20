use clap::{Parser, Subcommand};
use anyhow::Result;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "nodestor",
    about = "NodeStor — Motor de transporte de dados para IA\nStreaming de tensores SSD→GPU de alta performance",
    version,
    long_about = None
)]
struct Cli {
    /// Nível de log (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", global = true)]
    log: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detecta e mostra informações do hardware
    Scan,
    /// Analisa um arquivo de modelo (GGUF ou Safetensors)
    Inspect {
        /// Caminho do arquivo de modelo
        path: String,
        /// Mostra lista completa de tensores
        #[arg(long)]
        tensors: bool,
    },
    /// Executa benchmark de throughput do transporte
    Bench {
        /// Arquivo para benchmarkar (qualquer arquivo grande)
        path: String,
        /// Tamanho do bloco de leitura em MB
        #[arg(long, default_value = "64")]
        block_mb: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Configura logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cli.log))
        )
        .with_target(false)
        .compact()
        .init();

    match cli.command {
        Commands::Scan => cmd_scan(),
        Commands::Inspect { path, tensors } => cmd_inspect(&path, tensors),
        Commands::Bench { path, block_mb } => cmd_bench(&path, block_mb),
    }
}

fn cmd_scan() -> Result<()> {
    println!("\n🔍 NodeStor — Scanner de Hardware\n{}", "─".repeat(50));

    let profile = nodestor_scanner::scan()?;

    // SO
    println!("💻 Sistema Operacional");
    println!("   SO: {} ({})", profile.os, profile.os_version);
    println!("   CPU: {} núcleos", profile.cpu_cores);
    println!("   RAM: {:.1} GB", profile.total_ram_bytes as f64 / 1e9);

    // GPUs
    println!("\n🎮 GPUs Detectadas");
    if profile.gpus.is_empty() {
        println!("   Nenhuma GPU dedicada encontrada");
    } else {
        for (i, gpu) in profile.gpus.iter().enumerate() {
            println!("   [{}] {} ({})", i, gpu.device_name, gpu.vendor);
            if gpu.vram_bytes > 0 {
                println!("       VRAM: {:.1} GB", gpu.vram_bytes as f64 / 1e9);
            }
            println!("       Vulkan Compute: {}", if gpu.supports_vulkan_compute { "✅" } else { "❌" });
            println!("       Cooperative Matrix2: {}", if gpu.supports_cooperative_matrix2 { "✅" } else { "❌" });
            println!("       BFloat16: {}", if gpu.supports_bfloat16 { "✅" } else { "❌" });
        }
    }

    // Storage
    println!("\n💾 Armazenamento Detectado");
    if profile.storage.is_empty() {
        println!("   Nenhum dispositivo detectado");
    } else {
        for storage in profile.storage.iter().take(5) {
            println!(
                "   {} — {} ({}) — Disponível: {:.1} GB",
                storage.path,
                storage.name.chars().take(30).collect::<String>(),
                storage.nvme_gen,
                storage.available_bytes as f64 / 1e9
            );
            if storage.estimated_read_bps > 0 {
                println!(
                    "       Velocidade estimada: {:.1} GB/s",
                    storage.estimated_read_bps as f64 / 1e9
                );
            }
        }
    }

    // Recomendação
    println!("\n⚡ Backend de Transporte Recomendado");
    println!("   {}", profile.recommended_transport);
    println!(
        "   Throughput máximo estimado: {:.1} GB/s",
        profile.estimated_transport_throughput() as f64 / 1e9
    );

    println!("\n✅ Scan concluído!\n");
    Ok(())
}

fn cmd_inspect(path: &str, show_tensors: bool) -> Result<()> {
    println!("\n📦 NodeStor — Inspeção de Modelo\n{}", "─".repeat(50));

    let parser = nodestor_formats::detect_parser(path)?;
    let meta = parser.parse(path)?;

    println!("📄 Arquivo: {}", path);
    println!("📋 Formato: {}", parser.format_name());
    if let Some(name) = &meta.model_name {
        println!("🤖 Modelo: {}", name);
    }
    if let Some(arch) = &meta.architecture {
        println!("🏗️  Arquitetura: {}", arch);
    }
    if let Some(params) = meta.param_count {
        println!("🔢 Parâmetros: {:.1}B", params as f64 / 1e9);
    }
    println!("📊 Tensores: {}", meta.tensor_count());
    println!("💽 Tamanho total: {:.2} GB", meta.file_size as f64 / 1e9);
    println!("📍 Data offset: {} bytes", meta.data_offset);
    println!("🗜️  Dados de tensores: {:.2} GB", meta.total_tensor_size() as f64 / 1e9);

    if show_tensors {
        println!("\n📝 Lista de Tensores ({}):", meta.tensor_count());
        println!("{:<40} {:>12} {:>10} {:>8}", "Nome", "Elementos", "Tipo", "Tamanho");
        println!("{}", "─".repeat(75));
        for tensor in meta.tensors.iter().take(50) {
            let shape_str = tensor.shape.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("×");
            println!(
                "{:<40} {:>12} {:>10} {:>6.1} MB",
                tensor.name.chars().take(40).collect::<String>(),
                shape_str,
                tensor.dtype.name(),
                tensor.data_size as f64 / 1e6
            );
        }
        if meta.tensor_count() > 50 {
            println!("  ... e mais {} tensores", meta.tensor_count() - 50);
        }
    }

    println!("\n✅ Inspeção concluída!\n");
    Ok(())
}

fn cmd_bench(path: &str, block_mb: usize) -> Result<()> {
    use nodestor_core::TransferRequest;
    use std::time::Instant;

    println!("\n⚡ NodeStor — Benchmark de Throughput\n{}", "─".repeat(50));

    let profile = nodestor_scanner::scan()?;
    let transport = nodestor_transport::create_transport(&profile);

    let file_size = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("Arquivo não encontrado: {}", e))?
        .len();

    let block_size = block_mb * 1024 * 1024;
    let num_blocks = (file_size / block_size as u64).max(1) as usize;

    println!("📄 Arquivo: {} ({:.2} GB)", path, file_size as f64 / 1e9);
    println!("🧱 Bloco: {} MB", block_mb);
    println!("🔢 Blocos: {}", num_blocks);
    println!("⚡ Backend: {}", transport.backend_name());
    println!("\n📊 Executando benchmark...");

    let mut total_bytes = 0u64;
    let total_start = Instant::now();

    for i in 0..num_blocks {
        let offset = (i as u64) * block_size as u64;
        let size = ((file_size - offset) as usize).min(block_size);

        let request = TransferRequest {
            file_offset: offset,
            size,
            compressed: false,
        };

        let result = transport.transfer(path, &request)
            .map_err(|e| anyhow::anyhow!("Transferência falhou: {}", e))?;

        total_bytes += result.data.len() as u64;
    }

    let total_elapsed = total_start.elapsed();
    let throughput_gbs = total_bytes as f64 / total_elapsed.as_secs_f64() / 1e9;

    println!("\n📈 Resultado:");
    println!("   Total lido: {:.2} GB", total_bytes as f64 / 1e9);
    println!("   Tempo total: {:.2}s", total_elapsed.as_secs_f64());
    println!("   Throughput: {:.2} GB/s", throughput_gbs);
    println!("   Throughput: {:.0} MB/s", throughput_gbs * 1000.0);

    let theoretical = transport.theoretical_max_throughput_bps() as f64 / 1e9;
    let efficiency = (throughput_gbs / theoretical * 100.0).min(100.0);
    println!("   Eficiência vs máximo teórico: {:.1}%", efficiency);

    println!("\n✅ Benchmark concluído!\n");
    Ok(())
}
