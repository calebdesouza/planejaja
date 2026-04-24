use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// NodeStor COBER v2 - Subsistema 4: Filtro de Bloom (Sistema Imunológico)
///
/// Estrutura probabilística de memória ultra-rápida e compacta.
/// Usada para "rejeição precoce" de candidatos gerados pelos rascunhadores antes
/// que a GPU tenha que gastar tempo validando. Remove aberrações e lixo sintático.
pub struct TokenBloomFilter {
    /// Array de bits, condensado em u64 para eficiência de memória e acesso
    bits: Vec<u64>,
    /// Número de funções de hash a serem utilizadas (k)
    num_hashes: usize,
    /// Tamanho total em bits (m)
    size_bits: usize,
}

impl TokenBloomFilter {
    /// Inicializa o filtro baseando-se no número de entradas esperadas e a chance
    /// de falso positivo aceitável (ex: 0.01 para 1%).
    pub fn new(expected_entries: usize, false_positive_rate: f64) -> Self {
        // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
        let size_bits = (-(expected_entries as f64) * false_positive_rate.ln() / (2.0f64.ln().powi(2))).ceil() as usize;
        // k = round((m / n) * log(2));
        let num_hashes = ((size_bits as f64 / expected_entries as f64) * 2.0f64.ln()).round() as usize;

        let num_hashes = std::cmp::max(1, num_hashes);
        let blocks = (size_bits + 63) / 64; // Arredenda pra cima pra caber em u64

        Self {
            bits: vec![0; blocks],
            num_hashes,
            size_bits,
        }
    }

    /// Implementação auxiliar para simular `k` funções de hash independentes
    /// usando a técnica Double Hashing (Kirsch-Mitzenmacher): 
    /// h_i(x) = (h1(x) + i * h2(x)) % m
    fn get_hash_indices(&self, ngram: &[u32]) -> Vec<usize> {
        // Hash primário
        let mut hasher1 = DefaultHasher::new();
        hasher1.write_u64(0x12345678);
        ngram.hash(&mut hasher1);
        let h1 = hasher1.finish();

        // Hash secundário (seed diferente)
        let mut hasher2 = DefaultHasher::new();
        hasher2.write_u64(0x87654321);
        ngram.hash(&mut hasher2);
        let h2 = hasher2.finish();

        let mut indices = Vec::with_capacity(self.num_hashes);
        for i in 0..self.num_hashes {
            // (h1 + i * h2) % size_bits
            let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let idx = (combined % (self.size_bits as u64)) as usize;
            indices.push(idx);
        }
        indices
    }

    /// Alimenta o filtro com um padrão linguístico validado ("saudável")
    pub fn insert(&mut self, ngram: &[u32]) {
        let indices = self.get_hash_indices(ngram);
        for idx in indices {
            let block = idx / 64;
            let bit = idx % 64;
            self.bits[block] |= 1 << bit;
        }
    }

    /// Checa se um padrão é válido.
    /// Retorna `false` se DEFINITIVAMENTE não é válido.
    /// Retorna `true` se PROVAVELMENTE é válido (sujeito ao falso positivo).
    pub fn maybe_valid(&self, ngram: &[u32]) -> bool {
        let indices = self.get_hash_indices(ngram);
        for idx in indices {
            let block = idx / 64;
            let bit = idx % 64;
            if (self.bits[block] & (1 << bit)) == 0 {
                return false; // Falta um bit, definitivamente inválido
            }
        }
        true
    }

    /// Filtra uma lista de candidatos, podando aberrações antes de enviar à GPU
    pub fn filter_candidates(&self, candidates: &[Vec<u32>]) -> Vec<Vec<u32>> {
        candidates.iter()
            .filter(|cand| self.maybe_valid(cand))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_insert_and_check() {
        let mut bloom = TokenBloomFilter::new(1000, 0.01);
        
        // Insere 2 n-gramas "A casa caiu" e "O cachorro"
        let valid_ngram_1 = vec![1, 2, 3];
        let valid_ngram_2 = vec![4, 5];
        
        bloom.insert(&valid_ngram_1);
        bloom.insert(&valid_ngram_2);

        // Devem retornar true
        assert!(bloom.maybe_valid(&valid_ngram_1));
        assert!(bloom.maybe_valid(&valid_ngram_2));

        // N-gramas absurdos não devem passar (a menos de uma colisão de 1%, mas neste caso tão pequeno não deve).
        let invalid_ngram = vec![1, 2, 99];
        let invalid_ngram_2 = vec![99, 100];
        assert!(!bloom.maybe_valid(&invalid_ngram));
        assert!(!bloom.maybe_valid(&invalid_ngram_2));
    }

    #[test]
    fn test_bloom_filter_candidates() {
        let mut bloom = TokenBloomFilter::new(100, 0.05);
        bloom.insert(&[10, 20]);
        bloom.insert(&[30, 40]);

        let candidates = vec![
            vec![10, 20],   // Valido
            vec![10, 99],   // Invalido
            vec![30, 40],   // Valido
            vec![99, 88],   // Invalido
        ];

        let filtered = bloom.filter_candidates(&candidates);
        
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0], vec![10, 20]);
        assert_eq!(filtered[1], vec![30, 40]);
    }
}
