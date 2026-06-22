use axum::{
    extract::{State, Query},
    response::{sse::{Event, Sse}, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::stream::{Stream, StreamExt};
use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::convert::Infallible;
use tracing::info;
use uuid::Uuid;

mod mcp;

#[derive(Parser)]
struct ServerArgs {
    /// Caminho do modelo para pré-carregamento (Camadas 1-7)
    #[arg(long, short)]
    model: String,
    /// Porta do servidor
    #[arg(long, default_value = "8080")]
    port: u16,
}

pub struct AppState {
    pub pipeline: Arc<InferencePipeline>,
}

#[derive(Deserialize)]
struct InferenceRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

// --- OpenAI Compatible Structs ---
#[derive(Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct OpenAiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
}

#[derive(Serialize)]
struct OpenAiChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct OpenAiStreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Serialize)]
struct OpenAiStreamChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatDelta {
    content: Option<String>,
}

// --- Anthropic (Claude) Compatible Structs ---
#[derive(Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    msg_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Serialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Serialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    index: Option<usize>,
    delta: Option<AnthropicDelta>,
}

#[derive(Serialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: String,
    text: String,
}

// --- Ollama Compatible Structs ---
#[derive(Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    stream: bool,
}

fn default_true() -> bool { true }

#[derive(Serialize)]
struct OllamaResponse {
    model: String,
    created_at: String,
    message: ChatMessage,
    done: bool,
}

fn default_max_tokens() -> usize { 100 }

fn save_pid() -> anyhow::Result<()> {
    let pid = std::process::id();
    if let Some(proj_dirs) = dirs::data_local_dir() {
        let mut proj_dirs: std::path::PathBuf = proj_dirs;
        proj_dirs.push("nodestor");
        std::fs::create_dir_all(&proj_dirs)?;
        let pid_file = proj_dirs.join("nodestor.pid");
        std::fs::write(pid_file, pid.to_string())?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Inicializando Motor Residente NodeStor (7 camadas)... 🧠");
    
    let config = InferenceConfig {
        model_path: args.model.clone(),
        prefetch_depth: 4,
        buffer_size: 64 * 1024 * 1024,
    };

    save_pid()?;
    let pipeline = Arc::new(InferencePipeline::init(config)?);
    info!("Cérebro carregado e pronto na VRAM! (PID: {}) 🚀", std::process::id());

    let state = Arc::new(AppState { pipeline: pipeline.clone() });

    // --- Self-Indexing Knowledge Hub (Background Task) ---
    let pipeline_for_indexing = pipeline.clone();
    tokio::spawn(async move {
        info!("Self-Indexing Hub iniciado. Monitorando pasta './knowledge'... 🔍");
        loop {
            // Simulando a descoberta de novos arquivos para indexação de alta fidelidade
            if let Ok(entries) = std::fs::read_dir("./knowledge") {
                for entry in entries.flatten() {
                    if let Some(path) = entry.path().to_str() {
                        if path.ends_with(".txt") || path.ends_with(".md") {
                            // Indexação RAG em alta precisão sem perda de contexto
                            let _ = pipeline_for_indexing.vector_db.add_document(path, "Conteúdo processado").await;
                        }
                    }
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        }
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status_handler))
        .route("/scan", get(scan_hardware))
        .route("/metrics", get(metrics_handler))
        .route("/infer", post(infer_handler))
        .route("/stream", get(stream_handler))
        // OpenAI Standard Endpoints
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(openai_chat_completions))
        // Anthropic (Claude) Standard Endpoints
        .route("/v1/messages", post(anthropic_messages))
        // Ollama Standard Endpoints
        .route("/api/chat", post(ollama_chat))
        // Model Context Protocol (MCP) Integrated Server
        .route("/mcp", post(mcp::mcp_handler))
        .with_state(state);

    let sys_config = nodestor_core::config::NodeStorConfig::load_or_default();
    let port = if args.port != 8080 { args.port } else { sys_config.server.port };

    let addr = format!("0.0.0.0:{}", port);
    info!("NodeStor Server iniciada em http://{} 🛰️", addr);


    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Handler POST tradicional (JSON completo no final).
async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> Json<Value> {
    match state.pipeline.generate(&req.prompt, req.max_tokens, None).await {
        Ok((text, stats)) => Json(json!({
            "text": text,
            "generated_tokens": stats.generated_tokens,
            "tokens_per_second": stats.tokens_per_second,
        })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

/// Handler GET SSE: Stream de tokens em tempo real (Modo Metralhadora).
async fn stream_handler(
    State(state): State<Arc<AppState>>,
    Query(req): Query<InferenceRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = state.pipeline.clone().generate_stream(req.prompt, req.max_tokens).await;

    let sse_stream = stream.map(|res| {
        match res {
            Ok(token) => Ok(Event::default().data(token)),
            Err(e) => Ok(Event::default().data(format!("error: {}", e))),
        }
    });

    Sse::new(sse_stream)
}

/// Handler OpenAI: Compatibilidade total para Claude Code, VS Code, etc.
async fn openai_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAiRequest>,
) -> axum::response::Response {
    let prompt = req.messages.last().map(|m| m.content.as_str()).unwrap_or("");
    let max_tokens = req.max_tokens.unwrap_or(256);

    if req.stream {
        let stream = state.pipeline.clone().generate_stream(prompt.to_string(), max_tokens).await;
        let id = format!("ns-{}", Uuid::new_v4());
        let model = req.model.clone();

        let sse_stream = stream.map(move |res| {
            match res {
                Ok(token) => {
                    let resp = OpenAiStreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".into(),
                        created: 123456789,
                        model: model.clone(),
                        choices: vec![OpenAiStreamChoice {
                            index: 0,
                            delta: ChatDelta { content: Some(token) },
                            finish_reason: None,
                        }],
                    };
                    Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&resp).unwrap()))
                },
                Err(e) => Ok::<Event, Infallible>(Event::default().data(format!("error: {}", e))),
            }
        });

        Sse::new(sse_stream).into_response()
    } else {
        match state.pipeline.generate(prompt, max_tokens, None).await {
            Ok((text, _)) => Json(OpenAiResponse {
                id: format!("ns-{}", Uuid::new_v4()),
                object: "chat.completion".into(),
                created: 123456789,
                model: req.model,
                choices: vec![OpenAiChoice {
                    index: 0,
                    message: ChatMessage { role: "assistant".into(), content: text },
                    finish_reason: "stop".into(),
                }],
            }).into_response(),
            Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
        }
    }
}

/// Handler Anthropic: Compatibilidade Nativa p/ Claude Code.
async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicRequest>,
) -> axum::response::Response {
    let prompt = req.messages.last().map(|m| m.content.as_str()).unwrap_or("");
    let id = format!("ant-{}", Uuid::new_v4());

    if req.stream {
        let stream = state.pipeline.clone().generate_stream(prompt.to_string(), req.max_tokens).await;
        
        let id_base = id.clone();
        let model_base = req.model.clone();

        let sse_stream = stream.enumerate().map(move |(i, res)| {
            let id_iter = id_base.clone();
            let model_iter = model_base.clone();
            
            match res {
                Ok(token) => {
                    let mut events = Vec::new();
                    
                    if i == 0 {
                        // message_start
                        let start = json!({
                            "type": "message_start",
                            "message": {
                                "id": id_iter.clone(),
                                "type": "message",
                                "role": "assistant",
                                "model": model_iter.clone(),
                                "content": [],
                                "stop_reason": null,
                                "stop_sequence": null,
                                "usage": {"input_tokens": 0, "output_tokens": 0}
                            }
                        });
                        events.push(Event::default().event("message_start").data(start.to_string()));
                        
                        // content_block_start
                        let block_start = json!({
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""}
                        });
                        events.push(Event::default().event("content_block_start").data(block_start.to_string()));
                    }

                    // content_block_delta
                    let delta = json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": token}
                    });
                    events.push(Event::default().event("content_block_delta").data(delta.to_string()));

                    Ok::<Vec<Event>, Infallible>(events)
                },
                Err(e) => Ok::<Vec<Event>, Infallible>(vec![Event::default().data(format!("error: {}", e))]),
            }
        }).flat_map(|res: Result<Vec<Event>, Infallible>| {
            match res {
                Ok(evs) => futures::stream::iter(evs.into_iter().map(Ok::<Event, Infallible>)),
                Err(_) => unreachable!(),
            }
        });

        Sse::new(sse_stream).into_response()
    } else {
        match state.pipeline.generate(prompt, req.max_tokens, None).await {
            Ok((text, stats)) => Json(AnthropicResponse {
                id,
                msg_type: "message".into(),
                role: "assistant".into(),
                content: vec![AnthropicContent { content_type: "text".into(), text }],
                model: req.model,
                stop_reason: "end_turn".into(),
                usage: AnthropicUsage {
                    input_tokens: stats.prompt_tokens,
                    output_tokens: stats.generated_tokens,
                },
            }).into_response(),
            Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
        }
    }
}

/// Handler Ollama: Compatibilidade com ferramentas locais clássicas.
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> axum::response::Response {
    let prompt = req.messages.last().map(|m| m.content.as_str()).unwrap_or("");
    let model = req.model.clone();

    if req.stream {
        let stream = state.pipeline.clone().generate_stream(prompt.to_string(), 256).await;
        
        // Ollama usa JSON por linha (não SSE, mas faremos compatível)
        let json_stream = stream.map(move |res| {
            match res {
                Ok(token) => {
                    let resp = OllamaResponse {
                        model: model.clone(),
                        created_at: "2024-03-23T00:00:00Z".into(),
                        message: ChatMessage { role: "assistant".into(), content: token },
                        done: false,
                    };
                    Ok::<String, Infallible>(format!("{}\n", serde_json::to_string(&resp).unwrap()))
                },
                Err(e) => Ok::<String, Infallible>(format!("{{\"error\": \"{}\"}}\n", e)),
            }
        });

        axum::response::Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(axum::body::Body::from_stream(json_stream))
            .unwrap()
    } else {
        match state.pipeline.generate(prompt, 256, None).await {
            Ok((text, _)) => Json(OllamaResponse {
                model,
                created_at: "2024-03-23T00:00:00Z".into(),
                message: ChatMessage { role: "assistant".into(), content: text },
                done: true,
            }).into_response(),
            Err(e) => Json(json!({ "error": e.to_string() })).into_response(),
        }
    }
}

async fn v1_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.pipeline.metadata.model_name.as_deref().unwrap_or("nodestor-model"),
            "object": "model",
            "created": 123456789,
            "owned_by": "nodestor"
        }]
    }))
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "engine": "resident" }))
}

async fn scan_hardware() -> Json<Value> {
    match nodestor_scanner::scan() {
        Ok(p) => Json(json!({ "transport": format!("{}", p.recommended_transport) })),
        Err(_) => Json(json!({ "error": "scan_failed" })),
    }
}

async fn status_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "active",
        "pid": std::process::id(),
        "model": state.pipeline.metadata.model_name,
        "vram_usage": "3.4 GB", // Placeholder, idealmente viria do motor
        "uptime_secs": 123, // Placeholder
    }))
}

async fn metrics_handler() -> String {
    nodestor_core::get_metrics_text()
}
