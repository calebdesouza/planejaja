use std::env;
use std::time::Instant;
use nodestor_inference::pipeline::{InferencePipeline, InferenceConfig};
use nodestor_core::NodeStorError;

#[tokio::main]
async fn main() -> Result<(), NodeStorError> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       NodeStor: Ignição GGUF (Real Hardware Inference) 🚀         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Apontamos para o micro modelo FP32
    let model_path = "micro_modelo_ignition.gguf";

    println!("📥 Inicializando Pipeline com modelo: {}", model_path);
    let t_init = Instant::now();

    // 2. Cria a Pipeline V2 configurada para alta voltagem (bypass page cache via APEX)
    let config = InferenceConfig {
        model_path: model_path.to_string(),
        prefetch_depth: 3,  // Triple Buffered
        buffer_size: 1024 * 1024 * 64, // 64 MB blocks para NVME queues
    };

    let mut pipeline = InferencePipeline::init(config).expect("🧨 Falha catastrófica no boot do Vulkan ou APEX");

    let init_ms = t_init.elapsed().as_millis();
    println!("✅ Boot Concluído em {} ms", init_ms);
    println!("🎮 GPU Ativa: {}", pipeline.profile.primary_gpu().map(|g| g.device_name.as_str()).unwrap_or("Fallback CPU"));
    println!("📦 Tensores Encontrados: {}", pipeline.metadata.tensors.len());
    
    // 3. Prompt de teste
    let prompt = "A capital do Brasil é ";
    let tokens_to_generate = 10;
    
    println!("\n🧠 [PROMPT]: \"{}\"\n", prompt);
    println!("⚡ Iniciando Mecanismo de Atenção (Kernel Direct I/O via Pinned DMA)...");

    let t_gen = Instant::now();

    // 4. Primeiro Forward Pass real da história do projeto
    let (text, stats) = pipeline.generate(prompt, tokens_to_generate).await.unwrap();

    let total_time = t_gen.elapsed().as_millis();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  [RESPOSTA]: {}", text);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ 📊 ESTATÍSTICAS DA PRIMEIRA RESPIRADA");
    println!("║ Prompt Tokens: {}", stats.prompt_tokens);
    println!("║ Gerados      : {}", stats.generated_tokens);
    println!("║ Throughput   : {:.2} tokens/sec", stats.tokens_per_second);
    println!("║ Tempo Total  : {} ms", total_time);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
