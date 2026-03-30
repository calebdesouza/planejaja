use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use axum::{extract::State, Json};
use crate::AppState;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Serialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Value,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

pub async fn mcp_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<McpRequest>,
) -> Json<McpResponse> {
    let result = match req.method.as_str() {
        "resources/list" => Some(json!({
            "resources": [
                {
                    "uri": "nodestor://hardware/scan",
                    "name": "Scanner Result",
                    "description": "Detecção de hardware industrial via Vulkan",
                    "mimeType": "application/json"
                },
                {
                    "uri": "nodestor://engine/metrics",
                    "name": "Performance Metrics",
                    "description": "TTFT, Throughput e VRAM em tempo real",
                    "mimeType": "text/plain"
                }
            ]
        })),
        "tools/list" => Some(json!({
            "tools": [
                {
                    "name": "search_knowledge_base",
                    "description": "Busca vetorial no LanceDB para contexto RAG",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "k": { "type": "integer", "default": 5 }
                        },
                        "required": ["query"]
                    }
                }
            ]
        })),
        "tools/call" => {
            if let Some(params) = req.params {
                if params["name"] == "search_knowledge_base" {
                    let query = params["arguments"]["query"].as_str().unwrap_or("");
                    // Aqui faríamos a ponte real com o LanceDB via pipeline
                    Some(json!({
                        "content": [
                            {
                                "type": "text",
                                "text": format!("Resultados da busca para '{}': [Fragmento de Conhecimento Local]", query)
                            }
                        ]
                    }))
                } else {
                    None
                }
            } else {
                None
            }
        },
        _ => None,
    };

    let error = if result.is_none() { Some(json!({"code": -32601, "message": "Method not found"})) } else { None };
    Json(McpResponse {
        jsonrpc: "2.0".into(),
        id: req.id,
        result,
        error,
    })
}
