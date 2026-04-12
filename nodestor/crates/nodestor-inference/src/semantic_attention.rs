/// NodeStor COBER v2 — Semantic Attention Utilities
///
/// Baseado na inovação de Attention Residuals (arXiv:2603.15031).
/// Substitui o acúmulo uniforme (média, soma cega, LRU estrito)
/// por atenção seletiva baseada no contexto.

/// Calcula a similaridade do cosseno entre dois tensores 1D.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Computa o softmax In-Place sobre um slice de logits.
pub fn inplace_softmax(logits: &mut [f32]) {
    if logits.is_empty() { return; }
    
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let mut sum = 0.0;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    
    if sum > 0.0 {
        for x in logits.iter_mut() {
            *x /= sum;
        }
    }
}

/// Retorna os pesos de atenção (alphas) para um conjunto de keys 
/// vs uma query, usando scaled dot product + softmax.
/// T ideal para ser descarregado na GPU posteriormente.
pub fn compute_attention_weights(query: &[f32], keys: &[&[f32]], temperature: f32) -> Vec<f32> {
    let mut logits = Vec::with_capacity(keys.len());
    
    for key in keys {
        // cosine_similarity atua como um soft dot-product normalizado
        let sim = cosine_similarity(query, key);
        logits.push(sim / temperature);
    }
    
    inplace_softmax(&mut logits);
    logits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        inplace_softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }
    
    #[test]
    fn test_compute_attention() {
        let query = vec![1.0, 0.0, 0.0];
        let k1 = vec![1.0, 0.0, 0.0]; // muito similar
        let k2 = vec![0.0, 1.0, 0.0]; // ortogonal
        let k3 = vec![-1.0, 0.0, 0.0]; // oposto
        
        let keys: Vec<&[f32]> = vec![&k1, &k2, &k3];
        let alphas = compute_attention_weights(&query, &keys, 1.0);
        
        // alpha preferirá k1
        assert!(alphas[0] > alphas[1]);
        assert!(alphas[1] > alphas[2]);
        let sum: f32 = alphas.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
