//! nodestor-inference — Pipeline de inferência end-to-end.
//!
//! Orquestra: scanner → transport → streaming kernel → modelo → tokens.
//!
//! ## Motor COBER Neural Engine
//! - [`vram_budget`] — Orçamento VRAM baseado no livre real
//! - [`candidate_engine`] — HNSW + BM25 + RRF anti-alucinação
//! - [`cober`] — Draft → Verify → Accept (Rejection Sampling lossless)
//! - [`caches`] — Expert LRU Cache (MoE) + Feature Cache (Diffusion)

pub mod pipeline;
pub mod kv_cache;
pub mod tokenizer;
pub mod sampler;
pub mod vram_budget;
pub mod candidate_engine;
pub mod cober;
pub mod caches;
pub mod prompt_lookup;
pub mod bloom_filter;
pub mod golden_ngrams;
pub mod rest_trie;
pub mod syntactic_skeleton;
pub mod medusa_heads;
pub mod fractal_memory;
pub mod multi_tenant;
pub mod insight_indexer;
pub mod multi_draft;
pub mod cross_modal;
pub mod jitter_buffer;
pub mod vulkan_adaptive;
pub mod bench_cober_v2;


#[cfg(test)]
mod tests {
    use super::pipeline::{InferencePipeline, InferenceConfig};
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_full_7_layer_pipeline_orchestration() {
        // SETUP: Criação do ambiente de teste (Camada 3/HAL)
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("llama3_fake.gguf");
        let mut file = std::fs::File::create(&model_path).unwrap();
        // Simula um cabeçalho GGUF v3 real com 1 tensor (Camada 1/6)
        use byteorder::{LittleEndian, WriteBytesExt};
        file.write_u32::<LittleEndian>(0x46554747).unwrap(); // "GGUF"
        file.write_u32::<LittleEndian>(3).unwrap();          // v3
        file.write_u64::<LittleEndian>(1).unwrap();          // 1 Tensor
        file.write_u64::<LittleEndian>(0).unwrap();          // 0 KVs
        
        // Tensor definition: name (len + data), n_dims, dims, type, offset
        let t_name = "test_tensor";
        file.write_u64::<LittleEndian>(t_name.len() as u64).unwrap();
        file.write_all(t_name.as_bytes()).unwrap();
        file.write_u32::<LittleEndian>(1).unwrap();          // 1 dim
        file.write_u64::<LittleEndian>(128).unwrap();        // 128 elms
        file.write_u32::<LittleEndian>(0).unwrap();          // F32
        file.write_u64::<LittleEndian>(0).unwrap();          // offset 0
        
        file.write_all(&[0u8; 1024]).unwrap(); // Dummy tensor data (padding)

        // CONFIG: Parametrização da Engine (Camada 7/API)
        let config = InferenceConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            prefetch_depth: 2,
            buffer_size: 1024,
        };

        // BOOT: Inicialização do Cérebro (Camada 1, 3, 6)
        let pipeline = InferencePipeline::init(config).expect("Falha ao inicializar 7 camadas");
        
        // EXEC: Geração de Tokens com RAG e Streaming (Camada 2, 4, 5, 6)
        let (output, stats) = pipeline.generate("Olá NodeStor!", 5).await
            .expect("Falha na geração via 7 camadas");

        // PROOF: Verificação de métricas e vitalidade
        assert_eq!(stats.generated_tokens, 5, "Deveria gerar exatamente max_tokens solicitados");
        assert!(stats.tokens_per_second >= 0.0, "TPS deve ser mensurável");
        assert!(stats.total_time_ms > 0, "Geração deve levar algum tempo real (ms)");
        assert!(!output.is_empty(), "A resposta não deve ser vazia e os tokens devem ser parseados");
        assert!(output.contains("<unk>") || output.contains("Hello"), "Output deve usar o vocab dummy");
        
        println!("Super-Teste Concluído: {} tokens a {:.2} t/s", 
            stats.generated_tokens, stats.tokens_per_second);
    }
}
