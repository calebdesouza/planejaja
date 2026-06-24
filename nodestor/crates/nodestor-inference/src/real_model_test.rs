#[cfg(test)]
mod tests {
    use crate::pipeline::{InferenceConfig, InferencePipeline};
    use std::time::Instant;

    #[tokio::test]
    async fn test_real_qwen_model() {
        let model_path = "c:/Users/Adm/Nova pasta/planejaja/nodestor/nodestor-integration-tests/tests/fixtures/qwen2.5-0.5b-q4_k_m.gguf";
        
        println!("Iniciando teste de integração real com o modelo Qwen 0.5B...");
        
        let config = InferenceConfig {
            model_path: model_path.to_string(),
            prefetch_depth: 2,
            buffer_size: 16 * 1024 * 1024, // 16MB buffer para não estourar RAM no teste
        };

        println!("Inicializando Pipeline...");
        let start_init = Instant::now();
        let mut pipeline = InferencePipeline::init(config).expect("Falha ao inicializar o pipeline");
        println!("Pipeline inicializado em {:.2}s", start_init.elapsed().as_secs_f64());

        println!("Gerando tokens com modelo real...");
        let prompt = "A inteligência artificial é";
        let (text, stats) = pipeline.generate(prompt, 10, None, 0.7).await.expect("Falha na geração");
        
        println!("--- RESULTADO DA GERAÇÃO ---");
        println!("Prompt: {}", prompt);
        println!("Output: {}", text);
        println!("----------------------------");
        
        println!("Métricas de Auditoria:");
        println!("- Tokens Gerados: {}", stats.generated_tokens);
        println!("- Tokens por Segundo (TPS): {:.2}", stats.tokens_per_second);
        println!("- Tempo Total: {}ms", stats.total_time_ms);
        
        assert!(stats.generated_tokens > 0, "Deveria ter gerado pelo menos 1 token");
        assert!(!text.is_empty(), "O texto gerado não pode ser vazio");
    }
}
