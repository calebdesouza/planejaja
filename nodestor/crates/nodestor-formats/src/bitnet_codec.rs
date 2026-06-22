//! Codec BitNet b1.58 (Sistema Ternário)
//!
//! Compacta pesos {-1, 0, 1} usando aritmética base-3 para empacotar até 20 valores
//! ternários em um único `u32`. Na GPU, a extração reversa alimenta uma matriz de
//! multiplicação feita apenas de instruções ADD e SUB, obliterando a necessidade
//! dos caros FMAs (Fused Multiply-Add).

use nodestor_core::NodeStorError;

/// Empacota um array de valores `f32` (-1.0, 0.0, 1.0) em blocos de `u32`.
/// Assumimos que a quantização já ocorreu e os valores estão muito próximos a {-1, 0, 1}.
/// Cada u32 pode armazenar 20 valores (3^20 = 3,486,784,401 <= 4,294,967,295).
pub fn pack_ternary_weights(weights: &[f32]) -> Vec<u32> {
    let mut packed = Vec::with_capacity((weights.len() + 19) / 20);
    
    for chunk in weights.chunks(20) {
        let mut accum: u32 = 0;
        let mut multiplier: u32 = 1;
        
        for &w in chunk {
            // Mapeia: -1.0 -> 0, 0.0 -> 1, 1.0 -> 2
            let ternary_val = if w < -0.5 {
                0
            } else if w > 0.5 {
                2
            } else {
                1
            };
            
            accum += ternary_val * multiplier;
            multiplier *= 3;
        }
        
        packed.push(accum);
    }
    
    packed
}

/// Função utilitária para debug/validação, descompacta os pesos na CPU.
pub fn unpack_ternary_weights(packed: &[u32], original_len: usize) -> Result<Vec<f32>, NodeStorError> {
    let mut unpacked = Vec::with_capacity(original_len);
    
    for &block in packed {
        let mut accum = block;
        
        for _ in 0..20 {
            if unpacked.len() == original_len {
                break;
            }
            
            let val = accum % 3;
            accum /= 3;
            
            // Re-mapeia: 0 -> -1.0, 1 -> 0.0, 2 -> 1.0
            let f_val = match val {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            
            unpacked.push(f_val);
        }
    }
    
    Ok(unpacked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_ternary() {
        let original = vec![
            -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0,
             0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0,
             1.0, 1.0, 0.0 // Sobra 3 elementos no final
        ];
        
        let packed = pack_ternary_weights(&original);
        assert_eq!(packed.len(), 2);
        
        let unpacked = unpack_ternary_weights(&packed, original.len()).unwrap();
        
        for (i, (&a, &b)) in original.iter().zip(unpacked.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "Mismatch at index {}: original {}, unpacked {}", i, a, b);
        }
    }
}
