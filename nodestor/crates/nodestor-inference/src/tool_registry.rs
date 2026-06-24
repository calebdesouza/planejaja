//! Sistema de Registro e Invocação de Ferramentas por Tags
//!
//! O modelo emite `<call_tool_name>query</call_tool_name>` no texto gerado.
//! O pipeline intercepta, executa a ferramenta, injeta a resposta e retoma a
//! inferência com o novo contexto — sem interromper o streaming para o usuário.
//!
//! ## Fluxo
//! ```text
//! Model → "<call_vector_db>o que é RoPE?</call_vector_db>"
//!   → ToolRegistry::scan_for_call()     → Some(("call_vector_db", "o que é RoPE?"))
//!   → ToolRegistry::invoke()             → ToolResult { response: "RoPE é..." }
//!   → inject_tool_response()             → "<tool_response>RoPE é...</tool_response>"
//!   → re-inject into KV cache context   → model continues reasoning
//! ```
//!
//! ## Kit da Comunidade
//! Usuários criam `fisica_quantica.json` com definições de ferramentas e
//! system prompts especializados. Qualquer um pode baixar e usar:
//! `nodestor run --model modelo.gguf --tools-kit fisica_quantica.json`

use nodestor_core::NodeStorError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Definição de uma ferramenta — parte do Kit compartilhável
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Tag usada pelo modelo: `<tag>query</tag>`
    pub tag: String,
    /// Descrição injetada no System Prompt para o modelo saber usar
    pub description: String,
    /// Dica de quando usar esta ferramenta
    pub system_hint: String,
}

/// Kit de ferramentas portável — o formato `.json` / `.tools` da comunidade NodeStor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolKit {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: Option<String>,
    pub tools: Vec<ToolDef>,
    /// System Prompt especializado injetado junto com o kit (opcional)
    pub system_prompt: Option<String>,
}

impl ToolKit {
    /// Carrega um kit da comunidade a partir de arquivo `.json` / `.tools`
    pub fn load_from_file(path: &str) -> Result<Self, NodeStorError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| NodeStorError::ConfigError(format!("ToolKit '{}': {}", path, e)))?;
        serde_json::from_str(&raw)
            .map_err(|e| NodeStorError::ConfigError(format!("ToolKit JSON inválido: {}", e)))
    }

    /// Kit científico embutido — funciona sem arquivo externo
    pub fn builtin_science() -> Self {
        Self {
            name: "science".into(),
            version: "1.0".into(),
            description: "Pesquisa científica cross-domain: física, biologia, matemática".into(),
            author: Some("NodeStor Community".into()),
            tools: vec![
                ToolDef {
                    tag: "call_vector_db".into(),
                    description: "Busca semântica no banco vetorial (papers, fatos, memórias)".into(),
                    system_hint: "Use para recuperar informações factuais do banco de conhecimento antes de responder.".into(),
                },
                ToolDef {
                    tag: "call_dream".into(),
                    description: "Ativa o motor DAVI para gerar hipóteses cross-domain inéditas".into(),
                    system_hint: "Use para misturar conceitos de domínios opostos (ex: física quântica + genética).".into(),
                },
                ToolDef {
                    tag: "call_think".into(),
                    description: "Força um passo de reflexão antes de concluir".into(),
                    system_hint: "Use quando precisar revisar a cadeia de raciocínio antes de dar a resposta final.".into(),
                },
                ToolDef {
                    tag: "call_hypothesis".into(),
                    description: "Gera e valida uma hipótese científica via Nash Tribunal".into(),
                    system_hint: "Use para propor e testar formalmente uma hipótese contra evidências disponíveis.".into(),
                },
            ],
            system_prompt: Some(
                "Você é um cientista autônomo de elite com acesso a ferramentas de pesquisa.\n\
                 Quando precisar de dados externos, use: <call_vector_db>sua query</call_vector_db>\n\
                 Para cruzar conceitos de domínios diferentes: <call_dream>domínio1 + domínio2</call_dream>\n\
                 Para reflexão profunda: <call_think>seu raciocínio interno</call_think>\n\
                 Para validar hipóteses: <call_hypothesis>hipótese formal</call_hypothesis>\n\
                 Raciocine em ciclos. Cada descoberta alimenta a próxima pergunta.".into()
            ),
        }
    }

    /// Kit de programação — debugging, busca de docs, análise de código
    pub fn builtin_coding() -> Self {
        Self {
            name: "coding".into(),
            version: "1.0".into(),
            description: "Assistente de programação com busca de docs e análise".into(),
            author: Some("NodeStor Community".into()),
            tools: vec![
                ToolDef {
                    tag: "call_vector_db".into(),
                    description: "Busca na documentação e base de código indexada".into(),
                    system_hint: "Use para buscar exemplos de código, docs de API ou snippets anteriores.".into(),
                },
                ToolDef {
                    tag: "call_think".into(),
                    description: "Análise step-by-step do problema antes de codar".into(),
                    system_hint: "Use para decompor problemas complexos antes de escrever código.".into(),
                },
            ],
            system_prompt: Some(
                "Você é um engenheiro de software sênior. Antes de implementar, sempre analise:\n\
                 <call_think>decomposição do problema</call_think>\n\
                 Para buscar exemplos ou docs: <call_vector_db>query</call_vector_db>".into()
            ),
        }
    }

    /// Gera o bloco de System Prompt completo com as definições de todas as ferramentas
    pub fn build_system_block(&self) -> String {
        let mut s = String::new();
        if let Some(ref sp) = self.system_prompt {
            s.push_str(sp);
            s.push_str("\n\n");
        }
        s.push_str(&format!("=== Kit de Ferramentas: {} v{} ===\n", self.name, self.version));
        for tool in &self.tools {
            s.push_str(&format!("• <{}> — {}\n  Quando usar: {}\n",
                tool.tag, tool.description, tool.system_hint));
        }
        s
    }
}

/// Resultado de uma invocação de ferramenta
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tag: String,
    pub query: String,
    pub response: String,
    pub success: bool,
    pub latency_ms: u64,
}

impl ToolResult {
    /// Formata para injeção no contexto do modelo
    pub fn to_context_block(&self) -> String {
        format!(
            "<tool_response tool=\"{}\" success=\"{}\">\n{}\n</tool_response>",
            self.tag,
            self.success,
            self.response.trim()
        )
    }
}

/// Handler de ferramenta — closure que recebe a query e retorna a resposta
pub type ToolHandler = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Registro central de ferramentas — conecta tags a handlers reais
pub struct ToolRegistry {
    handlers: HashMap<String, ToolHandler>,
    pub kit: Option<ToolKit>,
    /// Total de invocações realizadas
    pub invocation_count: usize,
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { handlers: HashMap::new(), kit: None, invocation_count: 0 }
    }

    /// Registra um handler para uma tag específica
    pub fn register<F>(&mut self, tag: impl Into<String>, handler: F)
    where F: Fn(&str) -> String + Send + Sync + 'static
    {
        self.handlers.insert(tag.into(), Box::new(handler));
    }

    /// Atalho para registrar o handler do vector DB
    pub fn register_vector_db<F>(&mut self, handler: F)
    where F: Fn(&str) -> String + Send + Sync + 'static
    {
        self.register("call_vector_db", handler);
    }

    /// Atalho para o handler de dream (DAVI)
    pub fn register_dream<F>(&mut self, handler: F)
    where F: Fn(&str) -> String + Send + Sync + 'static
    {
        self.register("call_dream", handler);
    }

    /// Atalho para o handler de reflexão interna
    pub fn register_think<F>(&mut self, handler: F)
    where F: Fn(&str) -> String + Send + Sync + 'static
    {
        self.register("call_think", handler);
    }

    /// Carrega um kit e registra handlers padrão (pode ser sobrescrito depois)
    pub fn with_kit(mut self, kit: ToolKit) -> Self {
        self.kit = Some(kit);
        self
    }

    /// Escaneia texto gerado em busca do PRIMEIRO `<call_tag>query</call_tag>`
    /// Retorna (tag, query) se encontrado.
    pub fn scan_for_call(text: &str) -> Option<(String, String)> {
        let open_bracket = text.find('<')?;
        let rest = &text[open_bracket + 1..];
        let tag_end = rest.find('>')?;
        let tag_name = &rest[..tag_end];

        // Só aceita tags do formato call_* com caracteres válidos
        if !tag_name.starts_with("call_") { return None; }
        if !tag_name.chars().all(|c| c.is_alphanumeric() || c == '_') { return None; }

        let close_tag = format!("</{}>", tag_name);
        let content_start = open_bracket + 1 + tag_end + 1;
        if content_start >= text.len() { return None; }

        let close_pos = text[content_start..].find(&close_tag)?;
        let query = text[content_start..content_start + close_pos].trim().to_string();

        Some((tag_name.to_string(), query))
    }

    /// Executa a ferramenta com medição de latência
    pub fn invoke(&mut self, tag: &str, query: &str) -> ToolResult {
        let t0 = std::time::Instant::now();
        self.invocation_count += 1;

        match self.handlers.get(tag) {
            Some(h) => {
                let response = h(query);
                ToolResult {
                    tag: tag.to_string(),
                    query: query.to_string(),
                    response,
                    success: true,
                    latency_ms: t0.elapsed().as_millis() as u64,
                }
            }
            None => ToolResult {
                tag: tag.to_string(),
                query: query.to_string(),
                response: format!("[ferramenta '{}' não registrada — use `registry.register(\"{}\", handler)`]", tag, tag),
                success: false,
                latency_ms: 0,
            },
        }
    }

    /// Tenta escanear e invocar em um passo
    pub fn try_invoke_from_text(&mut self, text: &str) -> Option<ToolResult> {
        let (tag, query) = Self::scan_for_call(text)?;
        Some(self.invoke(&tag, &query))
    }

    /// Tags com handlers registrados
    pub fn registered_tags(&self) -> Vec<String> {
        self.handlers.keys().cloned().collect()
    }

    /// Verifica se tem handler para uma tag
    pub fn has_handler(&self, tag: &str) -> bool {
        self.handlers.contains_key(tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_for_call_valid() {
        let text = "Preciso verificar: <call_vector_db>RoPE rotary embedding</call_vector_db> antes de responder.";
        let result = ToolRegistry::scan_for_call(text);
        assert!(result.is_some());
        let (tag, query) = result.unwrap();
        assert_eq!(tag, "call_vector_db");
        assert_eq!(query, "RoPE rotary embedding");
    }

    #[test]
    fn test_scan_for_call_dream() {
        let text = "Vou explorar: <call_dream>física quântica + biologia molecular</call_dream>";
        let (tag, query) = ToolRegistry::scan_for_call(text).unwrap();
        assert_eq!(tag, "call_dream");
        assert!(query.contains("física quântica"));
    }

    #[test]
    fn test_scan_ignores_html_tags() {
        let text = "<h1>Título</h1> e depois <b>negrito</b>";
        assert!(ToolRegistry::scan_for_call(text).is_none(), "HTML tags não devem ser interceptadas");
    }

    #[test]
    fn test_scan_empty_returns_none() {
        assert!(ToolRegistry::scan_for_call("resposta simples sem tools").is_none());
        assert!(ToolRegistry::scan_for_call("").is_none());
    }

    #[test]
    fn test_invoke_registered_handler() {
        let mut reg = ToolRegistry::new();
        reg.register("call_vector_db", |q| format!("resultado para: {}", q));
        let result = reg.invoke("call_vector_db", "entropia neural");
        assert!(result.success);
        assert!(result.response.contains("entropia neural"));
    }

    #[test]
    fn test_invoke_unregistered_returns_error_message() {
        let mut reg = ToolRegistry::new();
        let result = reg.invoke("call_inexistente", "query");
        assert!(!result.success);
        assert!(result.response.contains("não registrada"));
    }

    #[test]
    fn test_tool_result_context_block() {
        let r = ToolResult {
            tag: "call_vector_db".into(),
            query: "RoPE".into(),
            response: "RoPE é Rotary Position Embedding".into(),
            success: true,
            latency_ms: 12,
        };
        let block = r.to_context_block();
        assert!(block.contains("tool_response"));
        assert!(block.contains("success=\"true\""));
        assert!(block.contains("RoPE é Rotary"));
    }

    #[test]
    fn test_toolkit_builtin_science_system_block() {
        let kit = ToolKit::builtin_science();
        let block = kit.build_system_block();
        assert!(block.contains("call_vector_db"));
        assert!(block.contains("call_dream"));
        assert!(block.contains("call_think"));
        assert!(block.contains("call_hypothesis"));
    }

    #[test]
    fn test_toolkit_load_from_invalid_path() {
        let result = ToolKit::load_from_file("/nao/existe.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_toolkit_load_from_json_string() {
        let kit = ToolKit {
            name: "test".into(),
            version: "1.0".into(),
            description: "kit de teste".into(),
            author: None,
            tools: vec![ToolDef {
                tag: "call_test".into(),
                description: "teste".into(),
                system_hint: "use em testes".into(),
            }],
            system_prompt: None,
        };
        let json = serde_json::to_string(&kit).unwrap();
        let tmp = std::env::temp_dir().join("nodestor_toolkit_test.json");
        std::fs::write(&tmp, &json).unwrap();
        let loaded = ToolKit::load_from_file(tmp.to_str().unwrap()).unwrap();
        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.tools.len(), 1);
        assert_eq!(loaded.tools[0].tag, "call_test");
    }

    #[test]
    fn test_invocation_counter() {
        let mut reg = ToolRegistry::new();
        reg.register("call_a", |_| "ok".into());
        reg.invoke("call_a", "q1");
        reg.invoke("call_a", "q2");
        reg.invoke("call_b", "q3"); // não registrado mas conta
        assert_eq!(reg.invocation_count, 3);
    }
}
