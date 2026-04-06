use std::collections::{HashMap, VecDeque};

/// Nó da Trie (árvore de prefixos)
pub struct TrieNode {
    pub token_id: u32,
    pub children: HashMap<u32, TrieNode>,
    pub frequency: u32,
    pub is_terminal: bool,
}

impl TrieNode {
    pub fn new(token_id: u32) -> Self {
        Self {
            token_id,
            children: HashMap::new(),
            frequency: 1,
            is_terminal: false,
        }
    }
}

/// NodeStor COBER v2 - Subsistema 2: REST Trie
///
/// Retrieval-Based Speculative Decoding (NAACL 2024).
/// Em vez de usar um modelo rascunhador, usa continuações de um banco de dados
/// inseridas em uma Trie. Ramos pesados formam os candidatos que passam na GPU
/// em uma única rolada de Tree Attention.
pub struct RestTrie {
    pub root: TrieNode,
    pub max_depth: usize,
}

impl RestTrie {
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: TrieNode::new(0), // root token fictício
            max_depth,
        }
    }

    /// Cria uma Trie a partir de uma lista de sequências (recuperadas via LanceDB).
    pub fn from_token_sequences(sequences: &[Vec<u32>], max_depth: usize) -> Self {
        let mut trie = Self::new(max_depth);
        for seq in sequences {
            trie.insert_sequence(seq, 1);
        }
        trie
    }

    /// Insere uma sequência na Trie, com um peso/frequência
    pub fn insert_sequence(&mut self, sequence: &[u32], weight: u32) {
        if sequence.is_empty() {
            return;
        }

        let mut current_node = &mut self.root;
        current_node.frequency += weight;

        let len = std::cmp::min(sequence.len(), self.max_depth);
        for &token_id in &sequence[..len] {
            let next_node = current_node.children.entry(token_id).or_insert_with(|| TrieNode::new(token_id));
            next_node.frequency += weight;
            current_node = next_node;
        }
        current_node.is_terminal = true;
    }

    /// Extrai os caminhos mais promissores (max_branches).
    /// Percorre a Trie em profundidade priorizando altas frequências.
    pub fn generate_branches(&self, max_branches: usize) -> Vec<Vec<u32>> {
        let mut branches = Vec::new();
        let mut current_path = Vec::new();
        
        self.dfs(&self.root, &mut current_path, &mut branches, max_branches);
        
        // Ordenar ramos por tamanho (preferência a draft mais longo) e frequência
        branches.sort_by(|a, b| b.len().cmp(&a.len()));
        if branches.len() > max_branches {
            branches.truncate(max_branches);
        }
        branches
    }

    fn dfs(&self, node: &TrieNode, current_path: &mut Vec<u32>, branches: &mut Vec<Vec<u32>>, max_branches: usize) {
        if branches.len() >= max_branches * 10 {
            // Cut-off simples para evitar explosão num espaço muito denso
            return;
        }

        if node.token_id != 0 {
            current_path.push(node.token_id);
        }

        if node.is_terminal || node.children.is_empty() {
            if !current_path.is_empty() {
                branches.push(current_path.clone());
            }
        }

        // Ordena os filhos por frequência decrescente para explorar os mais prováveis primeiro
        let mut children_vec: Vec<&TrieNode> = node.children.values().collect();
        children_vec.sort_by(|a, b| b.frequency.cmp(&a.frequency));

        for child in children_vec {
            self.dfs(child, current_path, branches, max_branches);
        }

        if node.token_id != 0 {
            current_path.pop();
        }
    }
}

/// Mascara de Atenção em Árvore para a GPU
pub struct TreeAttentionMask {
    /// Representação achatada: índice i -> token_id, e sua posição pai `parent_idx`
    pub candidate_tokens: Vec<u32>,
    pub parent_indices: Vec<Option<usize>>,
    // A verdadeira máscara (matriz bool) é construída no shader, 
    // ou na CPU via representações compactas. Aqui fornecemos a topologia.
}

impl TreeAttentionMask {
    /// Constrói a topologia flattenada da árvore a partir da Trie
    pub fn from_trie(trie: &RestTrie) -> Self {
        let mut candidate_tokens = Vec::new();
        let mut parent_indices = Vec::new();
        
        // Flattening em pre-order traversal / BFS
        let mut queue = VecDeque::new();
        // (node_reference, parent_index_in_flattened_list)
        for child in trie.root.children.values() {
            queue.push_back((child, None));
        }

        let mut idx = 0;
        while let Some((node, parent_idx)) = queue.pop_front() {
            candidate_tokens.push(node.token_id);
            parent_indices.push(parent_idx);
            
            for child in node.children.values() {
                queue.push_back((child, Some(idx)));
            }
            idx += 1;
        }

        Self {
            candidate_tokens,
            parent_indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rest_trie_insert_and_branches() {
        let sequences = vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 5, 6],
            vec![1, 7],
            vec![1, 2, 3, 9],
        ];

        let mut trie = RestTrie::from_token_sequences(&sequences, 10);
        let branches = trie.generate_branches(2); // Pegar os 2 top

        assert!(!branches.is_empty());
        // Deve preferir [1, 2, 3, 4] ou [1, 2, 3, 9] (ramificam de 3, ou um deles)
        // Pois 1->2->3 tem freq 2.
    }

    #[test]
    fn test_tree_attention_topology() {
        let sequences = vec![
            vec![10, 20],
            vec![10, 30],
        ];
        let trie = RestTrie::from_token_sequences(&sequences, 5);
        let topology = TreeAttentionMask::from_trie(&trie);
        
        // Root não fica na topologia, só os tokens a inferir.
        // Teremos token_id: 10, com parent None.
        // Apos ele, 20 e 30 terao parent index == 0 (indice do token 10).
        assert_eq!(topology.candidate_tokens.len(), 3);
        assert!(topology.candidate_tokens.contains(&10));
        assert!(topology.candidate_tokens.contains(&20));
        assert!(topology.candidate_tokens.contains(&30));
    }
}
