//! D1 — Provenance Graph: DNA do Pensamento
//!
//! Grafo Acíclico Dirigido (DAG) que rastreia a linhagem completa
//! de cada insight: de quais dados nasceu, quais transformações existiram,
//! quem validou. O usuário vê o CAMINHO do pensamento.

use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum StepType {
    /// Dado bruto do LanceDB
    RawData,
    /// Resultado de busca por similaridade
    SimilaritySearch,
    /// Hipótese gerada pelo Dreaming Engine
    HypothesisGeneration,
    /// Debate do Nash Tribunal
    NashDebate,
    /// Tradução via Funtor de Category Theory
    FunctorTranslation,
    /// Salto via Annealing (conexão criativa)
    AnnealingJump,
    /// Validação final
    Validation,
    /// Insight final descoberto
    FinalInsight,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProvenanceNode {
    /// ID único deste nó no grafo
    pub id: u64,
    /// IDs dos nós pai (de onde este nó derivou)
    pub parent_ids: Vec<u64>,
    /// ID do insight associado (se houver)
    pub insight_id: Option<u64>,
    /// Tipo de transformação que gerou este nó
    pub step_type: StepType,
    /// Descrição humano-legível
    pub description: String,
    /// Confiança/weight desta etapa (0.0–1.0)
    pub confidence: f32,
}

/// O Grafo de Proveniência: o DNA completo de cada pensamento
pub struct ProvenanceDAG {
    /// Todos os nós do grafo
    pub nodes: HashMap<u64, ProvenanceNode>,
    /// Contador monotônico de IDs
    next_id: u64,
}

impl ProvenanceDAG {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Adiciona um nó ao grafo, retornando seu ID
    pub fn add_node(
        &mut self,
        parent_ids: Vec<u64>,
        insight_id: Option<u64>,
        step_type: StepType,
        description: impl Into<String>,
        confidence: f32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, ProvenanceNode {
            id,
            parent_ids,
            insight_id,
            step_type,
            description: description.into(),
            confidence: confidence.clamp(0.0, 1.0),
        });
        id
    }

    /// Traça a linhagem completa de um nó (caminho até as raízes)
    pub fn trace_lineage(&self, node_id: u64) -> Vec<&ProvenanceNode> {
        let mut result = Vec::new();
        let mut queue = vec![node_id];
        let mut visited = std::collections::HashSet::new();

        while let Some(id) = queue.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);

            if let Some(node) = self.nodes.get(&id) {
                result.push(node);
                for &pid in &node.parent_ids {
                    queue.push(pid);
                }
            }
        }

        // Ordena por ID para leitura linear
        result.sort_by_key(|n| n.id);
        result
    }

    /// Exporta o grafo no formato DOT (para visualização com Graphviz)
    pub fn export_dot(&self) -> String {
        let mut dot = String::from("digraph Provenance {\n  rankdir=TB;\n");
        for node in self.nodes.values() {
            let label = format!(
                "[{:?}]\\n{}\\nconf={:.2}",
                node.step_type, node.description, node.confidence
            );
            dot.push_str(&format!(
                "  {} [label=\"{}\"];\n",
                node.id, label
            ));
            for &pid in &node.parent_ids {
                dot.push_str(&format!("  {} -> {};\n", pid, node.id));
            }
        }
        dot.push('}');
        dot
    }

    /// Encontra todos os nós finais (sem filhos)
    pub fn leaf_nodes(&self) -> Vec<&ProvenanceNode> {
        let all_parents: std::collections::HashSet<u64> = self.nodes.values()
            .flat_map(|n| n.parent_ids.iter().copied())
            .collect();
        self.nodes.values()
            .filter(|n| !all_parents.contains(&n.id))
            .collect()
    }

    /// Total de nós
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for ProvenanceDAG {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_insertion() {
        let mut dag = ProvenanceDAG::new();
        let root = dag.add_node(vec![], None, StepType::RawData, "Dado bruto: Física", 1.0);
        let child = dag.add_node(vec![root], Some(42), StepType::HypothesisGeneration, "Hipótese sobre gravidade", 0.85);
        assert_eq!(dag.len(), 2);
        assert_eq!(dag.nodes[&child].parent_ids, vec![root]);
    }

    #[test]
    fn test_trace_lineage_full() {
        let mut dag = ProvenanceDAG::new();
        let n0 = dag.add_node(vec![], None, StepType::RawData, "Dado A", 1.0);
        let n1 = dag.add_node(vec![], None, StepType::RawData, "Dado B", 1.0);
        let n2 = dag.add_node(vec![n0, n1], None, StepType::SimilaritySearch, "Merge A+B", 0.9);
        let n3 = dag.add_node(vec![n2], Some(1), StepType::FinalInsight, "Insight final", 0.8);

        let lineage = dag.trace_lineage(n3);
        assert_eq!(lineage.len(), 4); // n0, n1, n2, n3
    }

    #[test]
    fn test_export_dot() {
        let mut dag = ProvenanceDAG::new();
        let n0 = dag.add_node(vec![], None, StepType::RawData, "Raiz", 1.0);
        dag.add_node(vec![n0], None, StepType::FinalInsight, "Fim", 0.9);
        let dot = dag.export_dot();
        assert!(dot.contains("digraph Provenance"));
        assert!(dot.contains("->"));
    }
}
