//! Inline OpenAI-compatible HTTP server for the `nodestor serve` command.
//!
//! Endpoints:
//!   GET  /health                   — liveness check
//!   GET  /v1/models                — list loaded model
//!   POST /v1/chat/completions      — OpenAI ChatCompletion (stream or batch)
//!   POST /v1/completions           — legacy text completion
//!   GET  /stream                   — raw SSE token stream (used by `nodestor chat`)
//!
//! Compatible with: Claude Code, OpenAI SDK, LangChain, Ollama drop-in mode.

use axum::{
    body::Body,
    extract::{Query, State},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::{Stream, StreamExt};
use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

pub struct ServerState {
    pub pipeline: Arc<InferencePipeline>,
    pub model_id: String,
}

// ── Request / Response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct StreamQuery {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32  { 0.7 }

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => out.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n", msg.content
            )),
            "user" => out.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n", msg.content
            )),
            "assistant" => out.push_str(&format!(
                "<|im_start|>assistant\n{}<|im_end|>\n", msg.content
            )),
            _ => {}
        }
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "engine": "nodestor" }))
}

async fn list_models(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": unix_now(),
            "owned_by": "nodestor"
        }]
    }))
}

async fn stream_raw(
    State(state): State<Arc<ServerState>>,
    Query(params): Query<StreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let pipeline = state.pipeline.clone();
    let prompt   = params.prompt;
    let max_tok  = params.max_tokens;
    let temp     = params.temperature;

    let stream = async_stream::stream! {
        let mut gen = pipeline.generate_stream(prompt, max_tok, temp).await;
        while let Some(item) = gen.next().await {
            match item {
                Ok(token) => {
                    yield Ok(Event::default().data(token));
                }
                Err(_) => break,
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let prompt   = messages_to_prompt(&req.messages);
    let max_tok  = req.max_tokens;
    let temp     = req.temperature;
    let model_id = state.model_id.clone();
    let comp_id  = format!("chatcmpl-{}", Uuid::new_v4());

    if req.stream {
        // SSE streaming path
        let pipeline = state.pipeline.clone();
        let stream = async_stream::stream! {
            // Role delta (first chunk)
            let first = StreamChunk {
                id: comp_id.clone(),
                object: "chat.completion.chunk".into(),
                created: unix_now(),
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta { role: Some("assistant".into()), content: None },
                    finish_reason: None,
                }],
            };
            if let Ok(json) = serde_json::to_string(&first) {
                yield Ok::<_, Infallible>(Event::default().data(json));
            }

            let mut gen = pipeline.generate_stream(prompt, max_tok, temp).await;
            while let Some(item) = gen.next().await {
                match item {
                    Ok(token) => {
                        let chunk = StreamChunk {
                            id: comp_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created: unix_now(),
                            model: model_id.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: Delta { role: None, content: Some(token) },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(json) = serde_json::to_string(&chunk) {
                            yield Ok(Event::default().data(json));
                        }
                    }
                    Err(_) => break,
                }
            }

            // Stop chunk
            let stop = StreamChunk {
                id: comp_id.clone(),
                object: "chat.completion.chunk".into(),
                created: unix_now(),
                model: model_id.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta { role: None, content: None },
                    finish_reason: Some("stop".into()),
                }],
            };
            if let Ok(json) = serde_json::to_string(&stop) {
                yield Ok(Event::default().data(json));
            }
            yield Ok(Event::default().data("[DONE]"));
        };

        Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .text("ping"),
            )
            .into_response()
    } else {
        // Batch (collect all tokens)
        let pipeline = state.pipeline.clone();
        let mut gen = pipeline.generate_stream(prompt.clone(), max_tok, temp).await;
        let mut text = String::new();
        let mut n_completion = 0usize;
        while let Some(Ok(tok)) = gen.next().await {
            text.push_str(&tok);
            n_completion += 1;
        }
        let n_prompt = prompt.split_whitespace().count();

        let resp = ChatCompletionResponse {
            id: comp_id,
            object: "chat.completion".into(),
            created: unix_now(),
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage { role: "assistant".into(), content: text },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: n_prompt,
                completion_tokens: n_completion,
                total_tokens: n_prompt + n_completion,
            },
        };
        Json(resp).into_response()
    }
}

async fn text_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let model_id = state.model_id.clone();
    let comp_id  = format!("cmpl-{}", Uuid::new_v4());

    if req.stream {
        let pipeline = state.pipeline.clone();
        let stream = async_stream::stream! {
            let mut gen = pipeline.generate_stream(req.prompt, req.max_tokens, req.temperature).await;
            while let Some(item) = gen.next().await {
                match item {
                    Ok(token) => {
                        let chunk = json!({
                            "id": comp_id,
                            "object": "text_completion",
                            "choices": [{"text": token, "index": 0, "finish_reason": null}]
                        });
                        yield Ok::<_, Infallible>(Event::default().data(chunk.to_string()));
                    }
                    Err(_) => break,
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(stream).into_response()
    } else {
        let pipeline = state.pipeline.clone();
        let mut gen = pipeline.generate_stream(req.prompt.clone(), req.max_tokens, req.temperature).await;
        let mut text = String::new();
        while let Some(Ok(tok)) = gen.next().await { text.push_str(&tok); }

        Json(json!({
            "id": comp_id,
            "object": "text_completion",
            "created": unix_now(),
            "model": model_id,
            "choices": [{ "text": text, "index": 0, "finish_reason": "stop" }]
        })).into_response()
    }
}

// ── Ollama-compatible endpoints ───────────────────────────────────────────────

#[derive(Deserialize)]
struct OllamaGenerateReq {
    #[allow(dead_code)]
    model: String,
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    num_predict: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Deserialize)]
struct OllamaChatReq {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    num_predict: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

async fn ollama_version() -> impl IntoResponse {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

async fn ollama_tags(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    // Scan ~/.nodestor/models/ for GGUF files in addition to the loaded model
    let mut models = vec![json!({
        "name": format!("{}:latest", state.model_id),
        "model": format!("{}:latest", state.model_id),
        "modified_at": "2025-01-01T00:00:00Z",
        "size": 0,
        "digest": "nodestor",
        "details": { "format": "gguf", "family": "unknown" }
    })];

    if let Some(mut home) = dirs::home_dir() {
        home.push(".nodestor"); home.push("models");
        if let Ok(rd) = std::fs::read_dir(&home) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "gguf").unwrap_or(false) {
                    let name = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    if name != state.model_id {
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        models.push(json!({
                            "name": format!("{}:latest", name),
                            "model": format!("{}:latest", name),
                            "size": size,
                            "details": { "format": "gguf" }
                        }));
                    }
                }
            }
        }
    }

    Json(json!({ "models": models }))
}

async fn ollama_generate(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<OllamaGenerateReq>,
) -> impl IntoResponse {
    let model_id = state.model_id.clone();
    let pipeline = state.pipeline.clone();

    if req.stream {
        let prompt   = req.prompt.clone();
        let max_tok  = req.num_predict;
        let temp     = req.temperature;
        let stream = async_stream::stream! {
            let mut gen = pipeline.generate_stream(prompt, max_tok, temp).await;
            while let Some(Ok(tok)) = gen.next().await {
                let chunk = serde_json::to_string(&json!({
                    "model": model_id,
                    "response": tok,
                    "done": false
                })).unwrap_or_default();
                yield Ok::<_, Infallible>(axum::body::Bytes::from(format!("{}\n", chunk)));
            }
            let done = serde_json::to_string(&json!({
                "model": model_id,
                "response": "",
                "done": true
            })).unwrap_or_default();
            yield Ok(axum::body::Bytes::from(format!("{}\n", done)));
        };
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(axum::body::Body::from_stream(stream))
            .unwrap()
            .into_response()
    } else {
        let mut gen = pipeline.generate_stream(req.prompt, req.num_predict, req.temperature).await;
        let mut text = String::new();
        while let Some(Ok(tok)) = gen.next().await { text.push_str(&tok); }
        Json(json!({
            "model": state.model_id,
            "response": text,
            "done": true,
            "context": [],
        })).into_response()
    }
}

async fn ollama_chat(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<OllamaChatReq>,
) -> impl IntoResponse {
    let prompt = messages_to_prompt(&req.messages);
    let model_id = state.model_id.clone();

    if req.stream {
        let pipeline = state.pipeline.clone();
        let max_tok = req.num_predict;
        let temp = req.temperature;
        let stream = async_stream::stream! {
            let mut gen = pipeline.generate_stream(prompt, max_tok, temp).await;
            while let Some(Ok(tok)) = gen.next().await {
                let chunk = serde_json::to_string(&json!({
                    "model": model_id,
                    "message": { "role": "assistant", "content": tok },
                    "done": false
                })).unwrap_or_default();
                yield Ok::<_, Infallible>(axum::body::Bytes::from(format!("{}\n", chunk)));
            }
            let done = serde_json::to_string(&json!({
                "model": model_id,
                "message": { "role": "assistant", "content": "" },
                "done": true
            })).unwrap_or_default();
            yield Ok(axum::body::Bytes::from(format!("{}\n", done)));
        };
        axum::response::Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(axum::body::Body::from_stream(stream))
            .unwrap()
            .into_response()
    } else {
        let pipeline = state.pipeline.clone();
        let mut gen = pipeline.generate_stream(prompt, req.num_predict, req.temperature).await;
        let mut text = String::new();
        while let Some(Ok(tok)) = gen.next().await { text.push_str(&tok); }
        Json(json!({
            "model": state.model_id,
            "message": { "role": "assistant", "content": text },
            "done": true
        })).into_response()
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the OpenAI-compatible HTTP server on the specified port.
/// Blocks until Ctrl+C or fatal error.
pub async fn cmd_serve(model: &str, port: u16) -> anyhow::Result<()> {
    use tracing::info;

    let model_id = std::path::Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "nodestor-model".to_string());

    info!("NodeStor Engine initializing: {}", model);
    let config = InferenceConfig {
        model_path: model.to_string(),
        prefetch_depth: 4,
        buffer_size: 64 * 1024 * 1024,
    };
    let pipeline = Arc::new(
        InferencePipeline::init(config)
            .map_err(|e| anyhow::anyhow!("Engine init failed: {}", e))?
    );
    info!("Engine ready. Model: {}", model_id);

    let state = Arc::new(ServerState {
        pipeline,
        model_id: model_id.clone(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let router = Router::new()
        // OpenAI-compatible
        .route("/health",              get(health))
        .route("/v1/models",           get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions",      post(text_completions))
        .route("/stream",              get(stream_raw))
        // Ollama-compatible (OpenWebUI, LM Studio, etc.)
        .route("/api/version",         get(ollama_version))
        .route("/api/tags",            get(ollama_tags))
        .route("/api/generate",        post(ollama_generate))
        .route("/api/chat",            post(ollama_chat))
        .layer(cors)
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("NodeStor listening on http://0.0.0.0:{port}  (model: {model_id})");
    println!("  OpenAI : POST /v1/chat/completions  GET /v1/models");
    println!("  Ollama : POST /api/generate  POST /api/chat  GET /api/tags");
    println!();
    println!("Connect from:");
    println!("  LM Studio / Cursor   → OpenAI Base URL: http://localhost:{port}/v1");
    println!("  OpenWebUI            → Ollama URL:      http://localhost:{port}");
    println!("  Claude Code          → ANTHROPIC_BASE_URL=http://localhost:{port}/v1");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

/// Connection info for external tools — printed by `nodestor connect`.
pub fn print_connection_info(port: u16, model_id: &str) {
    let w = 65;
    let sep = "─".repeat(w - 2);
    println!("┌{}┐", sep);
    println!("│  NodeStor — External Tool Integration                       │");
    println!("├{}┤", sep);
    println!("│                                                             │");
    println!("│  Claude Code (MCP / Base URL override):                     │");
    println!("│    ANTHROPIC_BASE_URL=http://localhost:{port:<5}               │");
    println!("│    ANTHROPIC_MODEL={model_id:<40} │");
    println!("│                                                             │");
    println!("│  OpenAI SDK (Python / Node):                                │");
    println!("│    openai.base_url = 'http://localhost:{port}/v1'              │");
    println!("│    openai.api_key  = 'nodestor'  # any non-empty value       │");
    println!("│                                                             │");
    println!("│  LangChain / LlamaIndex:                                    │");
    println!("│    openai_api_base = 'http://localhost:{port}/v1'              │");
    println!("│                                                             │");
    println!("│  Ngrok tunnel (expose to internet):                         │");
    println!("│    ngrok http {port:<5}                                        │");
    println!("│                                                             │");
    println!("└{}┘", sep);
}
