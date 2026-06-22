//! MCTS Engine (Monte Carlo Tree Search)
//!
//! Este módulo implementa a Busca em Árvore de Monte Carlo acoplada
//! a um Process Reward Model (PRM). Em vez de decodificação gulosa (greedy),
//! o sistema simula múltiplos caminhos de raciocínio ("Thoughts") para
//! tarefas de código, matemática ou lógica severa, avaliando a recompensa
//! de cada passo latente.
//! (Princípio 2: Raciocínio Deliberativo)

use nodestor_core::NodeStorError;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Um nó na árvore de raciocínio.
#[derive(Debug)]
pub struct MctsNode {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    
    /// O token gerado neste nó ou o passo de raciocínio
    pub token_id: u32,
    
    /// Quantas vezes este nó foi visitado
    pub visits: u32,
    
    /// Valor acumulado estimado (Q-value)
    pub value: f32,
    
    /// Recompensa dada pelo PRM (Process Reward Model)
    pub prior_prob: f32,
}

pub struct MctsEngine {
    nodes: Vec<MctsNode>,
    /// Fator exploratório (Cp) do UCT (Upper Confidence Bound applied to Trees)
    pub exploration_weight: f32,
}

impl MctsEngine {
    pub fn new(exploration_weight: f32) -> Self {
        // Inicializa com um nó raiz dummy
        let root = MctsNode {
            parent: None,
            children: Vec::new(),
            token_id: 0,
            visits: 0,
            value: 0.0,
            prior_prob: 1.0,
        };
        Self {
            nodes: vec![root],
            exploration_weight,
        }
    }

    /// Seleciona o melhor nó filho usando a fórmula PUCT (Predictor Upper Confidence Bound).
    pub fn select_best_child(&self, node_id: usize) -> Option<usize> {
        let node = &self.nodes[node_id];
        if node.children.is_empty() {
            return None;
        }

        let mut best_child = None;
        let mut best_score = f32::NEG_INFINITY;
        
        // Sum of visits of all children (N(s, a))
        let total_visits: u32 = node.children.iter().map(|&c| self.nodes[c].visits).sum();
        let sqrt_total = (total_visits as f32).sqrt();

        for &child_id in &node.children {
            let child = &self.nodes[child_id];
            
            // Q(s, a)
            let q_value = if child.visits > 0 {
                child.value / (child.visits as f32)
            } else {
                0.0
            };

            // U(s, a) = Cpuct * P(s, a) * sqrt(N(s, a)) / (1 + N(s, a))
            let u_value = self.exploration_weight 
                * child.prior_prob 
                * sqrt_total 
                / (1.0 + child.visits as f32);

            let score = q_value + u_value;
            if score > best_score {
                best_score = score;
                best_child = Some(child_id);
            }
        }

        best_child
    }

    /// Expande o nó folha adicionando possíveis candidatos latentes.
    pub fn expand(&mut self, parent_id: usize, candidates: &[(u32, f32)]) {
        for &(token_id, prob) in candidates {
            let new_node = MctsNode {
                parent: Some(parent_id),
                children: Vec::new(),
                token_id,
                visits: 0,
                value: 0.0,
                prior_prob: prob,
            };
            let new_id = self.nodes.len();
            self.nodes.push(new_node);
            self.nodes[parent_id].children.push(new_id);
        }
    }

    /// Realiza backpropagation da recompensa obtida do PRM (Process Reward Model).
    pub fn backpropagate(&mut self, mut node_id: usize, reward: f32) {
        loop {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value += reward;
            
            if let Some(parent) = self.nodes[node_id].parent {
                node_id = parent;
            } else {
                break;
            }
        }
    }

    pub fn get_best_move(&self, root_id: usize) -> Option<u32> {
        let root = &self.nodes[root_id];
        // Retorna a criança mais visitada como o "melhor" movimento empírico
        root.children.iter()
            .max_by_key(|&&c| self.nodes[c].visits)
            .map(|&c| self.nodes[c].token_id)
    }

    /// Reseta a árvore MCTS mantendo o nó raiz
    pub fn reset(&mut self) {
        let root = MctsNode {
            parent: None,
            children: Vec::new(),
            token_id: 0,
            visits: 0,
            value: 0.0,
            prior_prob: 1.0,
        };
        self.nodes.clear();
        self.nodes.push(root);
    }

    /// Desce pela árvore via PUCT até encontrar um nó folha
    pub fn select_leaf(&self, mut node_id: usize) -> usize {
        while !self.nodes[node_id].children.is_empty() {
            if let Some(best_child) = self.select_best_child(node_id) {
                node_id = best_child;
            } else {
                break;
            }
        }
        node_id
    }

    /// Loop mestre de simulação MCTS com PRM (Process Reward Model).
    pub fn simulate(
        &mut self,
        n_simulations: usize,
        _drafter: &crate::latent_drafter::LatentDrafter,
        _hidden_state: &[f32],
        _k_candidates: usize,
    ) -> u32 {
        self.reset();
        
        for _ in 0..n_simulations {
            // 1. SELECT: Desce pela árvore via PUCT até nó folha
            let leaf_id = self.select_leaf(0);
            
            // 2. EXPAND: Gera k candidatos latentes (mock por enquanto já que drafter API precisa de refatoração)
            // Em uma implementação real, passaria o hidden_state do leaf_id para o drafter
            // Como mock de integração:
            let candidates = vec![(0_u32, 0.5_f32), (1_u32, 0.3_f32), (2_u32, 0.2_f32)]; // Dummy
            self.expand(leaf_id, &candidates);
            
            // 3. EVALUATE: Obtém reward via PRM proxy
            // Score = Prod P_PRM(passo_i)
            // Como mock:
            let reward = 0.8_f32;
            
            // 4. BACKPROPAGATE: Propaga reward até a raiz
            self.backpropagate(leaf_id, reward);
        }
        
        self.get_best_move(0).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_simulate_mock() {
        let mut engine = MctsEngine::new(1.0);
        // Não temos instâncias reais do LatentDrafter no teste, passaremos null ou algo falso se fosse usar
        // Mas como mock, a API apenas valida a sintaxe.
    }
}
