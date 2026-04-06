use std::cmp::min;
use tracing::debug;

/// NodeStor COBER v2 - Subsistema 1: Prompt Lookup (N-Gram Scanner)
///
/// Ref: "Prompt Lookup Decoding" (2023).
/// Esta estrutura intercepta o contexto (prompt original + gerados + blocos LanceDB)
/// e, em tempo linear, encontra repetições do sufixo atual. Se encontrar uma repetição,
/// especulativamente copia as palavras seguintes sem consumir GPU.
pub struct PromptLookup {
    /// Tamanhos de n-grama para tentar o match, ordenados por preferência (do maior pro menor)
    /// Exemplo: [6, 5, 4, 3, 2]
    ngram_sizes: Vec<usize>,
    /// Quantos tokens tentar copiar do match (tamanho do draft)
    max_copy_tokens: usize,
    /// Pool de contexto: guarda todos os tokens passados e injetados
    context_pool: Vec<u32>,
}

impl PromptLookup {
    pub fn new(ngram_sizes: Vec<usize>, max_copy_tokens: usize) -> Self {
        Self {
            ngram_sizes,
            max_copy_tokens,
            context_pool: Vec::new(),
        }
    }

    /// Adiciona os tokens mais recentes do prompt ou da geração ao pool
    pub fn append_context(&mut self, tokens: &[u32]) {
        self.context_pool.extend_from_slice(tokens);
    }

    /// Expande o pool com blocos recuperados do LanceDB (RAG / Smart Page Fault)
    /// Separamos esses blocos com um token nulo ou especial (0) para não cruzar fronteiras
    pub fn inject_lancedb_context(&mut self, recovered_tokens: &[u32]) {
        if !self.context_pool.is_empty() {
            // Token boundary arbitrário para não fazer match vazando entre contextos distintos
            self.context_pool.push(u32::MAX); 
        }
        self.context_pool.extend_from_slice(recovered_tokens);
    }

    /// Limpa o contexto
    pub fn clear(&mut self) {
        self.context_pool.clear();
    }

    /// Retorna um rascunho (draft) copiando o que vem depois do match no contexto.
    /// `last_tokens` são os tokens mais recentes que a rede acabou de gerar/processar.
    pub fn lookup(&self, last_tokens: &[u32]) -> Option<Vec<u32>> {
        if self.context_pool.is_empty() || last_tokens.is_empty() {
            return None;
        }

        // Tenta os n-grams do maior para o menor. Match maior = maior confiança.
        for &ngram_size in &self.ngram_sizes {
            if last_tokens.len() < ngram_size || self.context_pool.len() < ngram_size {
                continue;
            }

            let pattern = &last_tokens[last_tokens.len() - ngram_size..];
            
            // Busca a última ocorrência no contexto que NÃO SEJA a própria ponta do contexto
            // (Assumimos que last_tokens já pode estar no final do context_pool em alguns fluxos, 
            // mas queremos encontrar uma ocorrência ANTERIOR)
            let search_space = self.context_pool.len().saturating_sub(pattern.len());
            
            // Busca de trás pra frente (para pegar a referência mais recente possível)
            for i in (0..search_space).rev() {
                if &self.context_pool[i..i + pattern.len()] == pattern {
                    // Match encontrado!
                    let next_idx = i + pattern.len();
                    
                    // Avalia quantos tokens podemos copiar
                    let mut copy_len = 0;
                    while copy_len < self.max_copy_tokens && (next_idx + copy_len) < self.context_pool.len() {
                        let t = self.context_pool[next_idx + copy_len];
                        if t == u32::MAX { // Bateu numa fronteira de documento
                            break;
                        }
                        copy_len += 1;
                    }

                    if copy_len > 0 {
                        let draft = self.context_pool[next_idx..next_idx + copy_len].to_vec();
                        debug!("Prompt Lookup: Match N-Gram (tamanho {}) -> Copiou {} tokens", ngram_size, draft.len());
                        return Some(draft);
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_simple_match() {
        let mut pl = PromptLookup::new(vec![3, 2], 5);
        // "O rato roeu a roupa do rei de roma. O rato roeu a" -> draft: " roupa do rei"
        // 1=O, 2=rato, 3=roeu, 4=a, 5=roupa, 6=do, 7=rei, 8=de, 9=roma, 10=.
        pl.append_context(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4]);

        // last_tokens (N-gram size 3): [2, 3, 4] ("rato roeu a")
        let draft = pl.lookup(&[2, 3, 4]).expect("Deveria achar match");
        
        // Espera-se "[5, 6, 7, 8, 9]"
        assert_eq!(draft, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_prompt_lookup_fallback() {
        let mut pl = PromptLookup::new(vec![4, 2], 3);
        // N-Gram 4 não vai encontrar, mas N-gram 2 vai.
        pl.append_context(&[10, 20, 30, 40, 50, 60, 70, 80]);
        
        // last_tokens: [99, 99, 20, 30] -> pattern de 2 [20, 30] existe na pos 1.
        let draft = pl.lookup(&[99, 99, 20, 30]).unwrap();
        // Depois de 20, 30 vem 40, 50, 60 (max 3 tokens)
        assert_eq!(draft, vec![40, 50, 60]);
    }

    #[test]
    fn test_prompt_lookup_boundary() {
        let mut pl = PromptLookup::new(vec![2], 10);
        pl.inject_lancedb_context(&[100, 200, 300]);
        pl.inject_lancedb_context(&[400, 500, 600]);

        // last_tokens: [100, 200]
        let draft = pl.lookup(&[100, 200]).unwrap();
        // Deveria devolver apenas [300], pois logo depois tem boundary u32::MAX
        assert_eq!(draft, vec![300]);
    }
}
