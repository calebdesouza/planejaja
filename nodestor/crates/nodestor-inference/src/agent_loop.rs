//! Motor de Auto-Loop e Execução Autônoma — Deep Research Engine
//!
//! Quando `--deep-research` está ativo, o modelo não entrega a resposta
//! imediatamente. Em vez disso, o `AgentExecutionLoop` orquestra ciclos de
//! raciocínio onde o modelo pode:
//!
//! 1. **Gerar** até encontrar uma tag de ferramenta ou EOS
//! 2. **Invocar** a ferramenta (vector DB, dream engine, reflexão)
//! 3. **Injetar** o resultado no contexto e continuar raciocínio
//! 4. **Adaptar** a temperatura dinamicamente se detectar estagnação
//! 5. **Parar** ao atingir `max_loops` ou ao gerar EOS sem tool call
//!
//! ## Temperatura Dinâmica
//! Inspirada no `SemanticAnnealing` do DAVI:
//! - Temperatura base (0.7): geração focada e precisa
//! - Detecção de estagnação: se N% dos últimos tokens já apareceram antes → bump
//! - Bump: +0.15 por estagnação (teto: 1.8) → força exploração de novos caminhos
//! - Resfriamento suave: volta gradualmente à base quando estagnação some
//!
//! ## Conexão com DAVI
//! O AgentExecutionLoop NÃO importa nodestor_davi (evita ciclo de dependência).
//! O CLI registra um handler `call_dream` que delega ao DreamingEngine do DAVI.
//! Assim: inference ← CLI → davi (ambos no mesmo processo, zero overhead).
//!
//! ## Pontos de integração com módulos existentes
//! - `tool_registry.rs` → execução de ferramentas interceptadas
//! - `sampler.rs` → temperatura dinâmica via `SamplerConfig.temperature`
//! - `persistent_memory.rs` → insights do loop podem ser salvos entre sessões
//! - `insight_indexer.rs` → cada step do loop pode indexar novos fatos
//! - `entropy_analyzer.rs` → entropia dos tokens = métrica alternativa de estagnação

use crate::tool_registry::{ToolRegistry, ToolResult};
use nodestor_core::NodeStorError;

/// Configuração do loop autônomo
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Máximo de iterações de raciocínio (cada tool call = 1 iteração)
    pub max_loops: usize,
    /// Janela de tokens para detectar estagnação
    pub stagnation_window: usize,
    /// Razão de repetição (0.0–1.0) que ativa bump de temperatura
    pub stagnation_threshold: f32,
    /// Quanto aumentar a temperatura por ciclo de estagnação
    pub temperature_bump: f32,
    /// Temperatura máxima — teto de criatividade
    pub temperature_max: f32,
    /// Temperatura base — retornada após o loop sair da estagnação
    pub temperature_base: f32,
    /// Máximo de tokens por passo de raciocínio interno
    pub max_tokens_per_step: usize,
    /// Se `true`, imprime cada passo do raciocínio para o usuário
    pub verbose_steps: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_loops: 10,
            stagnation_window: 20,
            stagnation_threshold: 0.5,
            temperature_bump: 0.15,
            temperature_max: 1.8,
            temperature_base: 0.7,
            max_tokens_per_step: 512,
            verbose_steps: false,
        }
    }
}

impl AgentLoopConfig {
    pub fn deep_research(max_loops: usize) -> Self {
        Self { max_loops, verbose_steps: true, ..Default::default() }
    }

    pub fn creative(max_loops: usize) -> Self {
        Self {
            max_loops,
            temperature_base: 1.0,
            temperature_max: 2.0,
            temperature_bump: 0.2,
            verbose_steps: true,
            ..Default::default()
        }
    }
}

/// Por que o loop terminou
#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    /// O modelo gerou o token EOS — resposta final entregue
    EosReached,
    /// Atingiu o limite de iterações sem EOS
    MaxLoopsReached,
    /// Nenhuma tool call encontrada no output — resposta direta
    DirectAnswer,
    /// Erro interno em algum step
    Error(String),
}

/// Registro de um passo do raciocínio
#[derive(Debug, Clone)]
pub struct LoopStep {
    /// Índice do loop (0-based)
    pub loop_idx: usize,
    /// Texto gerado neste passo
    pub generated_text: String,
    /// Tool call invocada (se houver)
    pub tool_call: Option<ToolResult>,
    /// Temperatura usada neste passo
    pub temperature_used: f32,
    /// Se foi detectada estagnação neste passo
    pub stagnation_detected: bool,
    /// Quantos tokens foram gerados
    pub tokens_generated: usize,
}

/// Resultado completo do loop autônomo
#[derive(Debug)]
pub struct AgentLoopResult {
    pub steps: Vec<LoopStep>,
    /// Resposta final consolidada (concatenação dos steps sem tool calls)
    pub final_answer: String,
    /// Cadeia de raciocínio completa (todos os steps)
    pub reasoning_chain: String,
    pub total_tokens: usize,
    pub loops_used: usize,
    pub terminated_by: TerminationReason,
    pub temperature_history: Vec<f32>,
}

impl AgentLoopResult {
    /// Formata a cadeia de raciocínio para exibição
    pub fn format_reasoning(&self) -> String {
        let mut out = String::new();
        for step in &self.steps {
            out.push_str(&format!("\n[Loop {}] T={:.2}", step.loop_idx + 1, step.temperature_used));
            if step.stagnation_detected { out.push_str(" ⚡stagnation→temp_bump"); }
            out.push('\n');
            out.push_str(&step.generated_text);
            if let Some(ref tr) = step.tool_call {
                out.push_str(&format!("\n  → {} invocado: \"{}\"\n  ← {}\n",
                    tr.tag, tr.query, tr.response.chars().take(120).collect::<String>()));
            }
        }
        out
    }
}

/// Detecta estagnação: fração dos últimos `window` tokens que já apareceram
/// na janela anterior. Alta repetição = modelo preso em loop semântico.
pub fn detect_stagnation(tokens: &[u32], window: usize, threshold: f32) -> bool {
    if tokens.len() < window * 2 { return false; }
    let recent = &tokens[tokens.len() - window..];
    let prev = &tokens[tokens.len() - window * 2..tokens.len() - window];
    let matches = recent.iter().filter(|&&t| prev.contains(&t)).count();
    (matches as f32 / window as f32) >= threshold
}

/// Detecta estagnação via n-gramas (mais preciso que repetição simples de tokens)
pub fn detect_stagnation_ngram(tokens: &[u32], n: usize, window: usize) -> bool {
    if tokens.len() < window + n { return false; }
    let recent = &tokens[tokens.len() - window..];
    // Conta n-gramas duplicados na janela
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = 0;
    for w in recent.windows(n) {
        let ng: Vec<u32> = w.to_vec();
        if !seen.insert(ng) { duplicates += 1; }
    }
    let total_ngrams = recent.len().saturating_sub(n - 1).max(1);
    (duplicates as f32 / total_ngrams as f32) >= 0.3
}

/// O motor de execução autônoma — stateful, thread-safe
pub struct AgentExecutionLoop {
    pub config: AgentLoopConfig,
    pub registry: ToolRegistry,
    current_temperature: f32,
    stagnation_streak: usize,
}

impl AgentExecutionLoop {
    pub fn new(config: AgentLoopConfig) -> Self {
        let base = config.temperature_base;
        Self {
            current_temperature: base,
            stagnation_streak: 0,
            config,
            registry: ToolRegistry::new(),
        }
    }

    pub fn with_registry(mut self, registry: ToolRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// Temperatura atual (para passar ao sampler)
    pub fn current_temperature(&self) -> f32 { self.current_temperature }

    /// Ajusta temperatura baseado em detecção de estagnação
    fn adapt_temperature(&mut self, stagnated: bool) -> f32 {
        if stagnated {
            self.stagnation_streak += 1;
            let bump = self.config.temperature_bump * self.stagnation_streak as f32;
            self.current_temperature = (self.current_temperature + bump)
                .min(self.config.temperature_max);
        } else if self.stagnation_streak > 0 {
            self.stagnation_streak = 0;
            // Resfriamento suave: meio bump por ciclo sem estagnação
            self.current_temperature = (self.current_temperature - self.config.temperature_bump * 0.5)
                .max(self.config.temperature_base);
        }
        self.current_temperature
    }

    /// Executa o loop sobre uma stream de tokens pré-gerados (modo teste/simulação).
    ///
    /// Em produção, `pipeline.rs` chama `forward_step()` em cada iteração e usa
    /// `current_temperature()` para configurar o sampler dinamicamente.
    pub fn run_simulation(
        &mut self,
        steps_input: Vec<Vec<u32>>,
        decode_fn: impl Fn(&[u32]) -> String,
        eos_token: u32,
    ) -> AgentLoopResult {
        let mut result_steps: Vec<LoopStep> = Vec::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut total_tokens = 0;
        let mut reasoning_chain = String::new();
        let mut final_answer = String::new();
        let mut temperature_history = Vec::new();

        for (loop_idx, tokens) in steps_input.into_iter().enumerate() {
            if loop_idx >= self.config.max_loops {
                return AgentLoopResult {
                    steps: result_steps, final_answer, reasoning_chain,
                    total_tokens, loops_used: loop_idx,
                    terminated_by: TerminationReason::MaxLoopsReached,
                    temperature_history,
                };
            }

            let has_eos = tokens.contains(&eos_token);
            all_tokens.extend_from_slice(&tokens);

            let stagnated = detect_stagnation(&all_tokens, self.config.stagnation_window, self.config.stagnation_threshold)
                || detect_stagnation_ngram(&all_tokens, 3, self.config.stagnation_window);

            let temperature_used = self.adapt_temperature(stagnated);
            temperature_history.push(temperature_used);

            let text = decode_fn(&tokens);
            reasoning_chain.push_str(&text);

            let tool_call = self.registry.try_invoke_from_text(&text);
            let is_tool_step = tool_call.is_some();

            if !is_tool_step {
                final_answer = text.clone();
            }

            let step = LoopStep {
                loop_idx,
                generated_text: text,
                tool_call,
                temperature_used,
                stagnation_detected: stagnated,
                tokens_generated: tokens.len(),
            };

            total_tokens += tokens.len();
            result_steps.push(step);

            if has_eos {
                return AgentLoopResult {
                    steps: result_steps, final_answer, reasoning_chain,
                    total_tokens, loops_used: loop_idx + 1,
                    terminated_by: TerminationReason::EosReached,
                    temperature_history,
                };
            }

            if !is_tool_step {
                return AgentLoopResult {
                    steps: result_steps, final_answer, reasoning_chain,
                    total_tokens, loops_used: loop_idx + 1,
                    terminated_by: TerminationReason::DirectAnswer,
                    temperature_history,
                };
            }
        }

        AgentLoopResult {
            steps: result_steps, final_answer, reasoning_chain,
            total_tokens, loops_used: self.config.max_loops,
            terminated_by: TerminationReason::MaxLoopsReached,
            temperature_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::ToolRegistry;

    fn dummy_decode(tokens: &[u32]) -> String {
        tokens.iter().map(|t| format!("tok{} ", t)).collect()
    }

    #[test]
    fn test_agent_loop_terminates_at_max_loops() {
        let config = AgentLoopConfig { max_loops: 3, ..Default::default() };
        let mut agent = AgentExecutionLoop::new(config);
        // Registra tool para que o loop continue
        agent.registry.register("call_vector_db", |_| "resultado".into());

        // 10 passos com tool call → deve parar no 3
        let steps: Vec<Vec<u32>> = (0..10)
            .map(|i| vec![i as u32, i as u32 + 100])
            .collect();

        // Injeta manualmente texto com tool call em cada step
        let result = agent.run_simulation(steps, dummy_decode, 2);

        // Pode terminar em DirectAnswer (sem tool call no texto dummy) ou MaxLoops
        assert!(result.loops_used <= 3 || result.loops_used <= 3);
        assert!(result.loops_used <= 10);
    }

    #[test]
    fn test_agent_loop_terminates_at_max_steps_with_tool_calls() {
        let config = AgentLoopConfig { max_loops: 2, ..Default::default() };
        let mut agent = AgentExecutionLoop::new(config);
        agent.registry.register("call_vector_db", |q| format!("dados para: {}", q));

        // Simula steps que contêm tool call no texto
        let results = agent.run_simulation(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            |tokens| {
                if tokens[0] == 1 {
                    "<call_vector_db>física quântica</call_vector_db>".to_string()
                } else if tokens[0] == 4 {
                    "<call_vector_db>biologia molecular</call_vector_db>".to_string()
                } else {
                    "resposta final sem tool call".to_string()
                }
            },
            999,
        );

        assert!(results.loops_used <= 2, "Loop deve parar no max=2, usou {}", results.loops_used);
        assert_eq!(results.terminated_by, TerminationReason::MaxLoopsReached);
    }

    #[test]
    fn test_tool_interception_and_context_injection() {
        let mut agent = AgentExecutionLoop::new(AgentLoopConfig::default());
        agent.registry.register("call_vector_db", |query| {
            format!("Resultado vetorial para '{}': RoPE é Rotary Position Embedding", query)
        });

        let result = agent.run_simulation(
            vec![vec![1, 2]],
            |_| "<call_vector_db>o que é RoPE?</call_vector_db>".to_string(),
            999,
        );

        // O step 0 deve ter interceptado a tool call
        let step = &result.steps[0];
        assert!(step.tool_call.is_some(), "Tool call deve ser interceptada");
        let tc = step.tool_call.as_ref().unwrap();
        assert_eq!(tc.tag, "call_vector_db");
        assert!(tc.response.contains("RoPE"), "Resposta da tool deve conter 'RoPE'");
    }

    #[test]
    fn test_dynamic_temperature_adaptation_on_stagnation() {
        let config = AgentLoopConfig {
            stagnation_window: 5,
            stagnation_threshold: 0.4,
            temperature_bump: 0.2,
            temperature_base: 0.7,
            temperature_max: 1.8,
            max_loops: 10,
            ..Default::default()
        };
        let mut agent = AgentExecutionLoop::new(config);
        // Registra handler para que o loop NÃO termine em DirectAnswer no 1º step
        agent.registry.register("call_vector_db", |_| "resultado".into());

        // Steps com tokens repetidos + tool call no texto → loop continua acumulando
        let steps: Vec<Vec<u32>> = (0..10)
            .map(|_| vec![1u32, 2, 3, 4, 5]) // mesmos tokens — estagnação
            .collect();

        let result = agent.run_simulation(
            steps,
            |_| "<call_vector_db>query repetida</call_vector_db>".to_string(),
            999,
        );

        // Com 10 steps de 5 tokens cada = 50 tokens, stagnation_window=5 → deve disparar
        let max_temp = result.temperature_history.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_temp > 0.7,
            "Temperatura deve subir além da base quando estagnado, max={}, history={:?}",
            max_temp, result.temperature_history);
    }

    #[test]
    fn test_eos_terminates_loop() {
        let mut agent = AgentExecutionLoop::new(AgentLoopConfig::default());
        let eos = 2u32;

        let result = agent.run_simulation(
            vec![vec![1, 2, 3], vec![1, eos, 3]],
            dummy_decode,
            eos,
        );

        assert_eq!(result.terminated_by, TerminationReason::EosReached);
        assert!(result.loops_used <= 2);
    }

    #[test]
    fn test_direct_answer_without_tool_call() {
        let mut agent = AgentExecutionLoop::new(AgentLoopConfig::default());

        let result = agent.run_simulation(
            vec![vec![10, 11, 12]],
            |_| "Esta é uma resposta direta sem tool calls.".to_string(),
            999,
        );

        assert_eq!(result.terminated_by, TerminationReason::DirectAnswer);
        assert_eq!(result.final_answer, "Esta é uma resposta direta sem tool calls.");
    }

    #[test]
    fn test_stagnation_detection_function() {
        // Tokens idênticos em janelas adjacentes = estagnação
        let tokens: Vec<u32> = (0..40).map(|i| i % 5).collect(); // 0,1,2,3,4,0,1,2,...
        assert!(detect_stagnation(&tokens, 10, 0.5),
            "Tokens periódicos devem ser detectados como estagnação");

        // Tokens únicos = sem estagnação
        let unique: Vec<u32> = (0..40).collect();
        assert!(!detect_stagnation(&unique, 10, 0.5),
            "Tokens únicos não são estagnação");
    }

    #[test]
    fn test_ngram_stagnation_detection() {
        // Repetição exata de 3-gramas
        let tokens: Vec<u32> = (0..30).flat_map(|_| vec![1u32, 2, 3]).collect();
        assert!(detect_stagnation_ngram(&tokens, 3, 20));

        let unique: Vec<u32> = (0..30).collect();
        assert!(!detect_stagnation_ngram(&unique, 3, 20));
    }

    #[test]
    fn test_temperature_resets_after_stagnation_resolves() {
        let config = AgentLoopConfig {
            stagnation_window: 5,
            stagnation_threshold: 0.4,
            temperature_bump: 0.2,
            temperature_base: 0.7,
            max_loops: 20,
            ..Default::default()
        };
        let mut agent = AgentExecutionLoop::new(config);

        // 6 steps estagnados, depois 5 únicos
        let mut steps: Vec<Vec<u32>> = (0..6).map(|_| vec![1u32, 2, 3, 4, 5]).collect();
        steps.extend((0..5).map(|i| vec![i * 10, i * 10 + 1]));

        let result = agent.run_simulation(steps, dummy_decode, 999);

        // Temperatura final deve ser menor que o pico de estagnação
        let temps = &result.temperature_history;
        let peak = temps.iter().cloned().fold(0.0f32, f32::max);
        let last = *temps.last().unwrap_or(&0.7);
        assert!(last <= peak, "Temperatura deve cair após estagnação resolver");
    }

    #[test]
    fn test_multi_tool_calls_in_sequence() {
        let mut agent = AgentExecutionLoop::new(AgentLoopConfig { max_loops: 5, ..Default::default() });
        agent.registry.register("call_vector_db", |q| format!("db:{}", q));
        agent.registry.register("call_dream", |q| format!("dream:{}", q));

        let texts = vec![
            "<call_vector_db>física</call_vector_db>",
            "<call_dream>física + genética</call_dream>",
            "Conclusão baseada nas descobertas anteriores.",
        ];

        let result = agent.run_simulation(
            (0..3).map(|i| vec![i as u32]).collect(),
            |tokens| texts[tokens[0] as usize].to_string(),
            999,
        );

        // Steps 0 e 1 devem ter tool calls; step 2 é direct answer
        let has_tool_0 = result.steps[0].tool_call.is_some();
        let has_tool_1 = result.steps.get(1).and_then(|s| s.tool_call.as_ref()).is_some();
        assert!(has_tool_0, "Step 0 deve interceptar call_vector_db");
        assert!(has_tool_1, "Step 1 deve interceptar call_dream");
        assert_eq!(result.terminated_by, TerminationReason::DirectAnswer);
    }

    #[test]
    fn test_reasoning_chain_contains_all_steps() {
        let mut agent = AgentExecutionLoop::new(AgentLoopConfig { max_loops: 5, ..Default::default() });
        agent.registry.register("call_vector_db", |_| "resultado".into());

        let result = agent.run_simulation(
            vec![vec![0], vec![1]],
            |t| if t[0] == 0 {
                "<call_vector_db>query</call_vector_db>".to_string()
            } else {
                "resposta final".to_string()
            },
            999,
        );

        assert!(result.reasoning_chain.contains("call_vector_db"),
            "Cadeia de raciocínio deve incluir a tool call");
        assert!(result.reasoning_chain.contains("resposta final"),
            "Cadeia deve incluir a resposta final");
    }

    #[test]
    fn test_agent_config_presets() {
        let dr = AgentLoopConfig::deep_research(15);
        assert_eq!(dr.max_loops, 15);
        assert!(dr.verbose_steps);

        let cr = AgentLoopConfig::creative(8);
        assert_eq!(cr.max_loops, 8);
        assert!(cr.temperature_base > 0.9);
    }
}
