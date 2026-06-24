use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};
use std::time::Instant;

#[tokio::test]
#[ignore = "requires real GGUF fixture at tests/fixtures/qwen2.5-0.5b-q4_k_m.gguf"]
async fn test_real_qwen_model() {
    let model_path = "tests/fixtures/qwen2.5-0.5b-q4_k_m.gguf";

    let config = InferenceConfig {
        model_path: model_path.to_string(),
        prefetch_depth: 2,
        buffer_size: 16 * 1024 * 1024,
    };

    let start_init = Instant::now();
    let pipeline = InferencePipeline::init(config).expect("Falha ao inicializar o pipeline");
    println!("Pipeline inicializado em {:.2}s", start_init.elapsed().as_secs_f64());

    let prompt = "A inteligência artificial é";
    let start_gen = Instant::now();
    let (text, stats) = pipeline.generate(prompt, 20, None, 0.7).await.expect("Falha na geração");

    println!("Output: {}", text);
    println!("TPS: {:.2}", stats.tokens_per_second);
    println!("Tempo: {}ms", stats.total_time_ms);

    assert!(stats.generated_tokens > 0, "Deveria ter gerado pelo menos 1 token");
    assert!(!text.is_empty(), "O texto gerado não pode ser vazio");
    let elapsed = start_gen.elapsed().as_secs_f64();
    assert!(elapsed < 30.0, "Geração de 20 tokens não deve levar mais de 30s");
}
