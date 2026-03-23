//! NodeStor HTTP REST Server.
//!
//! Exposes the high-performance Tensor Streaming backend via simple JSON APIs.

use axum::{
    routing::{get, post},
    Json, Router,
};
use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{error, info};

#[derive(Deserialize)]
struct InferenceRequest {
    model_path: String,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_prefetch_depth")]
    prefetch_depth: usize,
}

fn default_max_tokens() -> usize { 100 }
fn default_prefetch_depth() -> usize { 4 }

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
    generated_tokens: usize,
    prompt_tokens: usize,
    tokens_per_second: f64,
    total_time_ms: u128,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/scan", get(scan_hardware))
        .route("/infer", post(infer_handler));

    let addr = "0.0.0.0:8080";
    info!("NodeStor Zero-Copy Server iniciado em http://{} 🚀", addr);
    info!("Endpoints: GET /health | GET /scan | POST /infer");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Dispara o "Modo Metralhadora" e executa a matemática Vulkan em altíssima velocidade.
async fn infer_handler(Json(req): Json<InferenceRequest>) -> Json<Value> {
    info!("Recebido /infer -> modelo: {}, tokens: {}", req.model_path, req.max_tokens);

    let config = InferenceConfig {
        model_path: req.model_path.clone(),
        prefetch_depth: req.prefetch_depth,
        buffer_size: 64 * 1024 * 1024, // 64 MB blocks per double-buffer
    };

    // Spin-up do motor L3 (Scanner + Vulkan + Transport + Metadata)
    let pipeline = match InferencePipeline::init(config) {
        Ok(p) => p,
        Err(e) => {
            error!("Falha ao inicializar o motor Vulkan/Transport: {}", e);
            return Json(json!({ "error": e.to_string() }));
        }
    };

    // Run the streaming compute loop
    match pipeline.generate(&req.prompt, req.max_tokens).await {
        Ok((text, stats)) => Json(json!(InferenceResponse {
            text,
            generated_tokens: stats.generated_tokens,
            prompt_tokens: stats.prompt_tokens,
            tokens_per_second: stats.tokens_per_second,
            total_time_ms: stats.total_time_ms,
        })),
        Err(e) => {
            error!("Erro durante o streaming da inferência: {}", e);
            Json(json!({ "error": e.to_string() }))
        }
    }
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "service": "NodeStor",
        "gpu_accelerated": "vulkan_native"
    }))
}

async fn scan_hardware() -> Json<Value> {
    match nodestor_scanner::scan() {
        Ok(profile) => Json(json!({
            "os": format!("{}", profile.os),
            "cpu_cores": profile.cpu_cores,
            "ram_gb": profile.total_ram_bytes as f64 / 1e9,
            "gpus": profile.gpus.iter().map(|g| json!({
                "name": g.device_name,
                "vendor": format!("{}", g.vendor),
                "vram_gb": g.vram_bytes as f64 / 1e9,
                "vulkan": g.supports_vulkan_compute,
            })).collect::<Vec<_>>(),
            "recommended_transport": format!("{}", profile.recommended_transport),
            "estimated_throughput_gbs": profile.estimated_transport_throughput() as f64 / 1e9,
        })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}
