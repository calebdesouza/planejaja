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
pub mod cpu_reference;
pub mod gpu_forward;
pub mod kv_cache;
pub mod tokenizer;
pub mod sampler;
pub mod cpu_backend;
pub mod lora_adapter;
pub mod clip_encoder;
pub mod whisper_pipeline;
pub mod diffusion_pipeline;


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
pub mod semantic_attention;
pub mod semantic_paging;
pub mod bench_cober_v2;
pub mod bench_master;
pub mod conformal_predictor;
pub mod sae_engine;
pub mod steering_engine;
// --- Deep Reasoning: 5 módulos novos (Princípios 2, 3, 6, 7, PCM) ---
pub mod ignorance_detector;
pub mod adaptive_router;
pub mod semantic_chunker;
pub mod budget_forcer;
pub mod persistent_memory;
// --- PROBES V2 Expansão: Cirurgia Latente Universal ---
pub mod refusal_mapper;

// --- Edge Training: LoRA Core + Trainer Local ---
pub mod lora_core;
pub mod trainer;

// --- System Prompt Builder: Editor Dinâmico Empresarial ---
pub mod system_prompt_builder;

// --- K-Quants: Dequantização Universal (Crítico para modelos 70B+) ---
pub mod dequant;

// --- GraphInterpreter: Detecção automática de arquitetura do modelo ---
pub mod graph_interpreter;

// --- WeightStore: Cache de tensores carregados do GGUF ---
pub mod weight_store;

// --- PagedAttention + RadixAttention: Multi-usuário eficiente ---
pub mod paged_attention;
pub mod radix_cache;

// --- Lossless Compression Mechanics ---
pub mod entropy_analyzer;

// --- Deep Research Engine: Auto-Loop + Tool Registry ---
pub mod tool_registry;
pub mod agent_loop;

// --- Edge RLHF: Wake-Sleep Cycle (Vigília→Curadoria→Sono) ---
pub mod background_indexer;
pub mod dataset_curator;
pub mod preference_collector;
pub mod local_dpo;

// --- Mechanistic Interpretability & Observability ---
pub mod observability;

// --- SSD Weight Stream: MoE-aware tiered expert cache + GDeflate streaming ---
pub mod ssd_stream;

// --- HeteroScheduler: CPU (attention) + GPU (FFN/MoE) pipeline ---
pub mod hetero_scheduler;

// --- ContextSsd: infinite context via KV archive on SSD + vector index ---
pub mod context_ssd;

// --- SsdStripe: multi-SSD parallel reads (N × 7 GB/s + GDeflate) ---
pub mod ssd_stripe;

// --- MoE Kernel: DeepSeek/Mixtral/Qwen expert routing + parallel FFN ---
pub mod moe_kernel;

// --- Sparse Activation: dense-to-MoE math optimization (60-80% neuron skip) ---
pub mod sparse_activation;

// --- SpecStream V2: hyperbolic tree speculation, dynamic K budget ---
pub mod spec_stream_v2;

// --- Quant-Invariant Bandwidth: GDeflate math + MoE early-abort provisioner ---
pub mod quant_invariant;

// --- Platform IO: memmap2/MADV_WILLNEED (macOS/Linux) + Win32 sequential scan ---
pub mod platform_io;

// --- Timeline Semaphore: CPU attention ∥ GPU FFN pipeline overlap ---
pub mod timeline_semaphore;

#[cfg(test)]
mod tests {
    use super::pipeline::{InferencePipeline, InferenceConfig};
    use std::io::Write;
    use tempfile::tempdir;

    /// Teste de orquestração completa das 7 camadas do pipeline.
    /// REQUER: GPU com suporte Vulkan disponível.
    /// Para executar explicitamente: cargo test -- --ignored test_full_7_layer_pipeline
    #[tokio::test]
    #[ignore = "Requer GPU Vulkan real — execução em ambiente de integração/CI com GPU"]
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
        let mut pipeline = InferencePipeline::init(config).expect("Falha ao inicializar 7 camadas");
        
        // EXEC: Geração de Tokens com RAG e Streaming (Camada 2, 4, 5, 6)
        let (output, stats) = pipeline.generate("Olá NodeStor!", 5, None, 0.7).await
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

pub mod latent_drafter;
pub mod prod_empirical_test;
pub mod mamba_layer;
pub mod mcts_engine;
pub mod hnsw_index;
pub mod lsh_buckets;
pub mod hamiltonian_dynamics;

// --- Resilient Apex Hardware-Mapped Engine ---
// Wavefront Timeslicing: sub-4ms GPU dispatch slices with cooperative yield
pub mod wavefront_scheduler;
// Elastic Memory: VirtualAlloc/mmap demand-paged tensor slots (no hard page-lock)
pub mod elastic_memory;
// APEX Fused Kernel: dequant + POD geometry without intermediate VRAM allocations
pub mod apex_fused_kernel;
// Paths: platform-agnostic %USERPROFILE%/.nodestor/ / $HOME/.nodestor/ layout
pub mod paths;
