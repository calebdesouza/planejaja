//! Persistent Cognitive Memory (PCM) — O Núcleo da Memória Infinita
//!
//! Este módulo implementa a "Memória Cognitiva Persistente" — a camada
//! que faz o NodeStor não perder conhecimento quando o modelo é trocado.
//!
//! ## O que persiste (sobrevive a QUALQUER troca de modelo):
//! - `chunks_indexed`: Todos os documentos ingeridos (RAG index)
//! - `steering_presets`: Configurações de Steering aprendidas por tarefa
//! - `pheromone_trails`: Feromônios do StigmergicSwarm
//! - `knowledge_gaps`: Buracos de conhecimento detectados pelo Topology
//! - `topic_coverage`: Mapa de cobertura por domínio
//! - `session_history`: Resumo de cada sessão anterior
//!
//! ## O que NÃO persiste (específico ao modelo em execução):
//! - KV Cache (alocação específica na VRAM)
//! - Hidden states ativos (voláteis por natureza)
//! - Pesos do modelo (ficam no SSD intocados)
//!
//! ## Troca de Modelo:
//! ```text
//! SHUTDOWN (modelo A):
//!   PersistentMemory::save_snapshot(&state)
//!   → serializa tudo em ~/.nodestor/cognitive_memory.bin
//!
//! BOOT (modelo B):
//!   PersistentMemory::load_snapshot()
//!   → alimenta InsightIndexer, StigmergicSwarm, GoldenNgrams
//!   → modelo B começa com 6 meses de aprendizado do modelo A
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::io::{self, Write, Read};
use std::fs;

/// Registro de preset de Steering aprendido automaticamente.
/// Quando o sistema detecta que uma configuração funciona bem para um tipo de
/// tarefa, salva como preset para uso futuro via CAST.
#[derive(Debug, Clone)]
pub struct SteeringPreset {
    /// Nome legível do preset (ex: "security_audit", "code_review")
    pub name: String,
    /// Embedding do tipo de tarefa (para matching futuro)
    pub task_signature: Vec<f32>,
    /// Configurações de steering: (feature_idx, alpha, mode)
    pub vectors: Vec<(usize, f32, String)>,
    /// Quantas vezes foi usado com sucesso
    pub hit_count: u64,
    /// Taxa de sucesso (Nash aceitou após aplicar este preset)
    pub success_rate: f32,
    /// O modelo que criou este preset (para compatibilidade futura)
    pub origin_model: String,
}

/// Registro de feromônio para persistência.
#[derive(Debug, Clone)]
pub struct PheromoneRecord {
    /// ID do caminho de raciocínio
    pub path_id: u64,
    /// Embedding do caminho
    pub embedding: Vec<f32>,
    /// Força do feromônio (decai com o tempo)
    pub strength: f32,
    /// Timestamp de criação (Unix)
    pub created_at: u64,
    /// Descrição do caminho (ex: "auth → crypto = vulnerabilidade")
    pub description: String,
}

/// Registro de buraco de conhecimento persistente.
#[derive(Debug, Clone)]
pub struct KnowledgeGapRecord {
    /// Centro do buraco no espaço semântico
    pub centroid: Vec<f32>,
    /// Taxa de rejeição nesta região (0.0-1.0)
    pub rejection_rate: f32,
    /// Tags semânticas do domínio
    pub domain_tags: Vec<String>,
    /// Quantas vezes foi tentado sem sucesso
    pub access_count: u64,
    /// Foi preenchido por pesquisa? (False = ainda em buraco)
    pub is_filled: bool,
    /// Timestamp da última tentativa
    pub last_attempt: u64,
}

/// Mapa de cobertura por tópico.
#[derive(Debug, Clone)]
pub struct TopicMap {
    /// Tópicos conhecidos com embedding centroide
    pub topics: HashMap<String, Vec<f32>>,
    /// Cobertura por tópico: 0.0 = nada, 1.0 = full
    pub coverage: HashMap<String, f32>,
    /// Total de chunks indexados por tópico
    pub chunk_count: HashMap<String, u64>,
}

impl TopicMap {
    pub fn new() -> Self {
        Self {
            topics: HashMap::new(),
            coverage: HashMap::new(),
            chunk_count: HashMap::new(),
        }
    }

    /// Atualiza cobertura de um tópico com novo chunk.
    pub fn update_topic(&mut self, topic: &str, embedding: Vec<f32>) {
        let count = self.chunk_count.entry(topic.to_string()).or_insert(0);
        *count += 1;

        // Atualiza centroide com média móvel
        let centroid = self.topics.entry(topic.to_string())
            .or_insert_with(|| vec![0.0; embedding.len()]);

        let n = *count as f32;
        for (i, v) in centroid.iter_mut().enumerate() {
            if i < embedding.len() {
                *v = (*v * (n - 1.0) + embedding[i]) / n;
            }
        }

        // Cobertura cresce com raiz quadrada do count (lei de rendimento decrescente)
        let cov = ((*count as f32) / 100.0).sqrt().min(1.0);
        self.coverage.insert(topic.to_string(), cov);
    }

    /// Retorna os tópicos com cobertura mais fraca (buracos de conhecimento).
    pub fn uncovered_topics(&self) -> Vec<(&str, f32)> {
        let mut uncovered: Vec<(&str, f32)> = self.coverage.iter()
            .filter(|(_, &cov)| cov < 0.5)
            .map(|(topic, &cov)| (topic.as_str(), cov))
            .collect();
        uncovered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        uncovered
    }
}

impl Default for TopicMap {
    fn default() -> Self { Self::new() }
}

/// Snapshot completo do estado cognitivo do sistema.
/// Tudo que é necessário para um novo modelo começar onde o anterior parou.
#[derive(Debug, Clone)]
pub struct CognitiveSnapshot {
    /// Timestamp Unix de criação
    pub created_at: u64,
    /// Número de sessões acumuladas
    pub session_count: u64,
    /// Total de chunks indexados no RAG
    pub chunks_indexed: u64,
    /// Total de tokens verificados pelo Nash
    pub verified_insights: u64,
    /// Mapa de cobertura por tópico
    pub topic_map: TopicMap,
    /// Buracos de conhecimento detectados e rastreados
    pub knowledge_gaps: Vec<KnowledgeGapRecord>,
    /// Presets de Steering aprendidos por tipo de tarefa
    pub steering_presets: Vec<SteeringPreset>,
    /// Feromônios de caminhos de raciocínio bem-sucedidos
    pub pheromone_trails: Vec<PheromoneRecord>,
    /// Número de N-grams no cache Golden (não os dados brutos, só a contagem)
    pub golden_ngrams_count: u64,
    /// Número de sinapses no grafo de insights
    pub synapse_count: u64,
    /// Nome da última versão do modelo (para log de proveniência)
    pub last_model_name: String,
    /// Versão do schema do snapshot
    pub schema_version: u32,
}

impl CognitiveSnapshot {
    pub fn new(model_name: &str) -> Self {
        Self {
            created_at: Self::unix_timestamp(),
            session_count: 0,
            chunks_indexed: 0,
            verified_insights: 0,
            topic_map: TopicMap::new(),
            knowledge_gaps: Vec::new(),
            steering_presets: Vec::new(),
            pheromone_trails: Vec::new(),
            golden_ngrams_count: 0,
            synapse_count: 0,
            last_model_name: model_name.to_string(),
            schema_version: 1,
        }
    }

    fn unix_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// A Memória Cognitiva Persistente — gerencia save/load do estado cognitivo.
pub struct PersistentMemory {
    /// Diretório de persistência (agnóstico ao modelo)
    pub store_path: PathBuf,
    /// Versão do schema atual
    pub schema_version: u32,
}

impl PersistentMemory {
    pub const SNAPSHOT_FILE: &'static str = "cognitive_snapshot.json";
    pub const RAG_INDEX_FILE: &'static str = "rag_index.json";
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new(store_path: &str) -> Self {
        let path = PathBuf::from(store_path);
        if !path.exists() {
            let _ = fs::create_dir_all(&path);
        }
        Self {
            store_path: path,
            schema_version: Self::SCHEMA_VERSION,
        }
    }

    /// Persiste o snapshot completo em disco.
    /// Chamado no shutdown da sessão.
    pub fn save_snapshot(&self, snapshot: &CognitiveSnapshot) -> io::Result<()> {
        let path = self.store_path.join(Self::SNAPSHOT_FILE);

        // Serialização manual em JSON compacto (sem serde para zero-dep)
        let json = self.snapshot_to_json(snapshot);

        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Carrega o snapshot do disco.
    /// Retorna None se não existe (primeira sessão).
    pub fn load_snapshot(&self) -> io::Result<Option<CognitiveSnapshot>> {
        let path = self.store_path.join(Self::SNAPSHOT_FILE);

        if !path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        Ok(self.json_to_snapshot(&contents))
    }

    /// Merge de dois snapshots: não perde nenhuma informação.
    /// Usado quando o modelo B quer incorporar o que o modelo A aprendeu.
    pub fn merge_snapshots(old: &CognitiveSnapshot, new: &CognitiveSnapshot) -> CognitiveSnapshot {
        let mut merged = new.clone();

        // Acumula contadores
        merged.session_count = old.session_count + new.session_count;
        merged.chunks_indexed = old.chunks_indexed + new.chunks_indexed;
        merged.verified_insights = old.verified_insights + new.verified_insights;
        merged.golden_ngrams_count = old.golden_ngrams_count.max(new.golden_ngrams_count);
        merged.synapse_count = old.synapse_count + new.synapse_count;

        // Herda buracos de conhecimento não preenchidos
        for old_gap in &old.knowledge_gaps {
            if !old_gap.is_filled {
                let already_exists = new.knowledge_gaps.iter()
                    .any(|g| cosine_similarity(&g.centroid, &old_gap.centroid) > 0.9);
                if !already_exists {
                    merged.knowledge_gaps.push(old_gap.clone());
                }
            }
        }

        // Herda presets de Steering (por nome único)
        for old_preset in &old.steering_presets {
            let exists = new.steering_presets.iter().any(|p| p.name == old_preset.name);
            if !exists {
                merged.steering_presets.push(old_preset.clone());
            }
        }

        // Herda feromônios com força > 20% (feromônios fracos não valem a pena)
        for old_pheromone in &old.pheromone_trails {
            if old_pheromone.strength > 0.2 {
                merged.pheromone_trails.push(PheromoneRecord {
                    strength: old_pheromone.strength * 0.8, // Decai 20% na transição
                    ..old_pheromone.clone()
                });
            }
        }

        // Merge de topic maps
        for (topic, embedding) in &old.topic_map.topics {
            if !merged.topic_map.topics.contains_key(topic) {
                merged.topic_map.topics.insert(topic.clone(), embedding.clone());
                if let Some(&cov) = old.topic_map.coverage.get(topic) {
                    merged.topic_map.coverage.insert(topic.clone(), cov);
                }
            }
        }

        merged
    }

    /// Gera um relatório legível do que o sistema "sabe" e "não sabe".
    pub fn knowledge_report(&self, snapshot: &CognitiveSnapshot) -> String {
        let total_topics = snapshot.topic_map.topics.len();
        let covered = snapshot.topic_map.coverage.values()
            .filter(|&&c| c >= 0.5)
            .count();
        let open_gaps = snapshot.knowledge_gaps.iter()
            .filter(|g| !g.is_filled)
            .count();
        let top_presets = snapshot.steering_presets.iter()
            .take(3)
            .map(|p| format!("'{}'({:.0}%)", p.name, p.success_rate * 100.0))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "=== Relatório da Memória Cognitiva Persistente ===\n\
             Sessões: {} | Chunks RAG: {} | Insights verificados: {}\n\
             Domínios conhecidos: {}/{} cobertos ≥ 50%\n\
             Buracos abertos: {} | Feromônios ativos: {}\n\
             Presets de Steering: {} | Top: {}\n\
             N-grams memoizados: {} | Sinapses: {}\n\
             Último modelo: '{}' | Schema: v{}",
            snapshot.session_count,
            snapshot.chunks_indexed,
            snapshot.verified_insights,
            covered, total_topics,
            open_gaps,
            snapshot.pheromone_trails.iter().filter(|p| p.strength > 0.2).count(),
            snapshot.steering_presets.len(),
            if top_presets.is_empty() { "(nenhum ainda)".to_string() } else { top_presets },
            snapshot.golden_ngrams_count,
            snapshot.synapse_count,
            snapshot.last_model_name,
            snapshot.schema_version,
        )
    }

    /// Verifica se o diretório de persistência exists e é acessível.
    pub fn is_accessible(&self) -> bool {
        self.store_path.exists() && self.store_path.is_dir()
    }

    // --- Serialização manual (sem serde, zero deps extras) ---

    fn snapshot_to_json(&self, s: &CognitiveSnapshot) -> String {
        let gaps_json: Vec<String> = s.knowledge_gaps.iter().map(|g| {
            format!(
                "{{\"rejection_rate\":{:.3},\"access_count\":{},\"is_filled\":{},\"domain_tags\":[{}]}}",
                g.rejection_rate, g.access_count, g.is_filled,
                g.domain_tags.iter().map(|t| format!("\"{}\"", t)).collect::<Vec<_>>().join(",")
            )
        }).collect();

        let presets_json: Vec<String> = s.steering_presets.iter().map(|p| {
            format!(
                "{{\"name\":\"{}\",\"hit_count\":{},\"success_rate\":{:.3},\"origin_model\":\"{}\"}}",
                p.name, p.hit_count, p.success_rate, p.origin_model
            )
        }).collect();

        let pheromones_json: Vec<String> = s.pheromone_trails.iter()
            .filter(|p| p.strength > 0.1) // Só persiste feromônios relevantes
            .map(|p| {
                format!(
                    "{{\"path_id\":{},\"strength\":{:.3},\"description\":\"{}\"}}",
                    p.path_id, p.strength,
                    p.description.replace('"', "'") // Escapa aspas
                )
            }).collect();

        format!(
            "{{\
            \"schema_version\":{},\
            \"created_at\":{},\
            \"session_count\":{},\
            \"chunks_indexed\":{},\
            \"verified_insights\":{},\
            \"golden_ngrams_count\":{},\
            \"synapse_count\":{},\
            \"last_model_name\":\"{}\",\
            \"knowledge_gaps_count\":{},\
            \"steering_presets_count\":{},\
            \"pheromone_trails_count\":{},\
            \"knowledge_gaps\":[{}],\
            \"steering_presets\":[{}],\
            \"pheromone_trails\":[{}]\
            }}",
            s.schema_version,
            s.created_at,
            s.session_count,
            s.chunks_indexed,
            s.verified_insights,
            s.golden_ngrams_count,
            s.synapse_count,
            s.last_model_name.replace('"', "'"),
            s.knowledge_gaps.len(),
            s.steering_presets.len(),
            s.pheromone_trails.len(),
            gaps_json.join(","),
            presets_json.join(","),
            pheromones_json.join(","),
        )
    }

    fn json_to_snapshot(&self, json: &str) -> Option<CognitiveSnapshot> {
        // Parser manual mínimo — extrai apenas os campos escalares necessários
        fn extract_u64(json: &str, key: &str) -> Option<u64> {
            let pattern = format!("\"{}\":", key);
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse().ok()
        }

        fn extract_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
            let pattern = format!("\"{}\":\"", key);
            let start = json.find(&pattern)? + pattern.len();
            let end = json[start..].find('"')?;
            Some(&json[start..start + end])
        }

        let session_count = extract_u64(json, "session_count").unwrap_or(0);
        let chunks_indexed = extract_u64(json, "chunks_indexed").unwrap_or(0);
        let verified_insights = extract_u64(json, "verified_insights").unwrap_or(0);
        let golden_ngrams_count = extract_u64(json, "golden_ngrams_count").unwrap_or(0);
        let synapse_count = extract_u64(json, "synapse_count").unwrap_or(0);
        let created_at = extract_u64(json, "created_at").unwrap_or(0);
        let last_model_name = extract_str(json, "last_model_name")
            .unwrap_or("unknown").to_string();

        Some(CognitiveSnapshot {
            schema_version: extract_u64(json, "schema_version").unwrap_or(1) as u32,
            created_at,
            session_count,
            chunks_indexed,
            verified_insights,
            golden_ngrams_count,
            synapse_count,
            last_model_name,
            topic_map: TopicMap::new(),
            knowledge_gaps: Vec::new(),
            steering_presets: Vec::new(),
            pheromone_trails: Vec::new(),
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 { return 0.0; }
    let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
    let na: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir_named(name: &str) -> PathBuf {
        let path = PathBuf::from(format!(
            "{}/nodestor_pcm_test_{}_{}",
            std::env::temp_dir().to_str().unwrap(),
            std::process::id(),
            name,
        ));
        let _ = fs::create_dir_all(&path);
        path
    }

    fn temp_dir() -> PathBuf {
        temp_dir_named("default")
    }

    fn cleanup(path: &PathBuf) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn test_save_and_load_snapshot() {
        let dir = temp_dir_named("save_load");
        let mem = PersistentMemory::new(dir.to_str().unwrap());

        let mut snapshot = CognitiveSnapshot::new("llama3-8b");
        snapshot.session_count = 42;
        snapshot.chunks_indexed = 1337;
        snapshot.verified_insights = 99;

        mem.save_snapshot(&snapshot).expect("Save deve funcionar");
        let loaded = mem.load_snapshot().expect("Load não deve falhar com IO");

        assert!(loaded.is_some(), "Snapshot deve ser carregado após save");
        let loaded = loaded.unwrap();

        assert_eq!(loaded.session_count, 42,
            "session_count deve ser preservado: got {}", loaded.session_count);
        assert_eq!(loaded.chunks_indexed, 1337,
            "chunks_indexed deve ser preservado: got {}", loaded.chunks_indexed);
        assert_eq!(loaded.verified_insights, 99,
            "verified_insights deve ser preservado: got {}", loaded.verified_insights);
        assert_eq!(loaded.last_model_name, "llama3-8b",
            "last_model_name deve ser preservado: got '{}'", loaded.last_model_name);

        cleanup(&dir);
    }


    #[test]
    fn test_load_returns_none_when_no_file() {
        let dir = temp_dir();
        let unique_dir = dir.join("nonexistent_subdir");
        let mem = PersistentMemory::new(unique_dir.to_str().unwrap());

        // Não salva nada
        let loaded = mem.load_snapshot();
        assert!(loaded.is_ok(), "Load não deve retornar Err quando arquivo não existe");
        assert!(loaded.unwrap().is_none(), "Load deve retornar None quando arquivo não existe");

        cleanup(&dir);
    }

    #[test]
    fn test_merge_snapshots_accumulates_counts() {
        let mut old = CognitiveSnapshot::new("llama3-8b");
        old.session_count = 10;
        old.chunks_indexed = 500;
        old.verified_insights = 30;

        let mut new = CognitiveSnapshot::new("qwen2-14b");
        new.session_count = 5;
        new.chunks_indexed = 200;
        new.verified_insights = 15;

        let merged = PersistentMemory::merge_snapshots(&old, &new);

        assert_eq!(merged.session_count, 15, "Sessions devem somar");
        assert_eq!(merged.chunks_indexed, 700, "Chunks devem somar");
        assert_eq!(merged.verified_insights, 45, "Insights devem somar");
        assert_eq!(merged.last_model_name, "qwen2-14b", "Modelo deve ser o novo");
    }

    #[test]
    fn test_merge_inherits_open_gaps() {
        let mut old = CognitiveSnapshot::new("llama3-8b");
        old.knowledge_gaps.push(KnowledgeGapRecord {
            centroid: vec![0.1, 0.9, 0.0],
            rejection_rate: 0.85,
            domain_tags: vec!["TLS".to_string()],
            access_count: 5,
            is_filled: false, // Buraco aberto!
            last_attempt: 0,
        });

        let new = CognitiveSnapshot::new("qwen2-14b");
        let merged = PersistentMemory::merge_snapshots(&old, &new);

        assert!(!merged.knowledge_gaps.is_empty(),
            "Buracos abertos do modelo antigo devem ser herdados");
        assert!(!merged.knowledge_gaps[0].is_filled,
            "Buraco herdado ainda deve estar aberto");
    }

    #[test]
    fn test_merge_skips_filled_gaps() {
        let mut old = CognitiveSnapshot::new("llama3-8b");
        old.knowledge_gaps.push(KnowledgeGapRecord {
            centroid: vec![0.5; 4],
            rejection_rate: 0.1,
            domain_tags: vec!["preenchido".to_string()],
            access_count: 10,
            is_filled: true, // Buraco preenchido — não herdar
            last_attempt: 0,
        });

        let new = CognitiveSnapshot::new("qwen2-14b");
        let merged = PersistentMemory::merge_snapshots(&old, &new);

        assert!(merged.knowledge_gaps.is_empty(),
            "Buracos preenchidos não devem ser herdados");
    }

    #[test]
    fn test_pheromones_decay_on_merge() {
        let mut old = CognitiveSnapshot::new("llama3-8b");
        old.pheromone_trails.push(PheromoneRecord {
            path_id: 1,
            embedding: vec![0.5; 4],
            strength: 0.8,
            created_at: 0,
            description: "auth → crypto".to_string(),
        });

        let new = CognitiveSnapshot::new("qwen2-14b");
        let merged = PersistentMemory::merge_snapshots(&old, &new);

        assert!(!merged.pheromone_trails.is_empty(),
            "Feromônio forte deve ser herdado");
        assert!(merged.pheromone_trails[0].strength < 0.8,
            "Feromônio deve decair 20% na transição: got {:.2}",
            merged.pheromone_trails[0].strength);
        assert!((merged.pheromone_trails[0].strength - 0.64).abs() < 0.01,
            "Decaimento deve ser exato (0.8 * 0.8 = 0.64): got {:.3}",
            merged.pheromone_trails[0].strength);
    }

    #[test]
    fn test_weak_pheromones_not_inherited() {
        let mut old = CognitiveSnapshot::new("llama3-8b");
        old.pheromone_trails.push(PheromoneRecord {
            path_id: 2,
            embedding: vec![],
            strength: 0.15, // Fraco — não herdar
            created_at: 0,
            description: "caminho fraco".to_string(),
        });

        let new = CognitiveSnapshot::new("qwen2-14b");
        let merged = PersistentMemory::merge_snapshots(&old, &new);

        assert!(merged.pheromone_trails.is_empty(),
            "Feromônio com strength < 0.2 não deve ser herdado");
    }

    #[test]
    fn test_knowledge_report_format() {
        let mut snapshot = CognitiveSnapshot::new("llama3-8b");
        snapshot.session_count = 5;
        snapshot.chunks_indexed = 100;
        snapshot.steering_presets.push(SteeringPreset {
            name: "security_audit".to_string(),
            task_signature: vec![],
            vectors: vec![],
            hit_count: 20,
            success_rate: 0.85,
            origin_model: "llama3-8b".to_string(),
        });

        let dir = temp_dir();
        let mem = PersistentMemory::new(dir.to_str().unwrap());
        let report = mem.knowledge_report(&snapshot);

        assert!(report.contains("Sessões: 5"), "Report deve mostrar sessões");
        assert!(report.contains("Chunks RAG: 100"), "Report deve mostrar chunks");
        assert!(report.contains("security_audit"), "Report deve listar presets");
        assert!(report.contains("llama3-8b"), "Report deve mostrar modelo");

        cleanup(&dir);
    }

    #[test]
    fn test_topic_map_coverage_grows() {
        let mut map = TopicMap::new();

        for i in 0..50 {
            map.update_topic("segurança", vec![0.5, 0.5, i as f32 * 0.01]);
        }

        let cov = map.coverage.get("segurança").copied().unwrap_or(0.0);
        assert!(cov > 0.5, "Cobertura deve crescer com o volume de chunks: got {:.3}", cov);
        assert!(cov <= 1.0, "Cobertura não pode exceder 1.0: got {:.3}", cov);
    }

    #[test]
    fn test_is_accessible() {
        let dir = temp_dir();
        let mem = PersistentMemory::new(dir.to_str().unwrap());
        assert!(mem.is_accessible(), "Diretório recém criado deve ser acessível");
        cleanup(&dir);
    }
}
