//! LFSR Seed Engine — Compressão Fractal via Complexidade de Kolmogorov.
//!
//! Baseado no framework SeedLM: dado um tensor de pesos treinados,
//! encontra a semente LFSR que minimiza ||tensor - lfsr_expand(seed)||.
//! Custo de armazenamento: O(1) por tensor (24 bytes + resíduos esparsos).

/// Resíduo esparso: posição + valor delta.
#[derive(Debug, Clone)]
pub struct SparseResidual {
    pub index: u32,
    pub delta: f32,
}

/// Semente LFSR que codifica um tensor inteiro.
#[derive(Debug, Clone)]
pub struct LfsrSeed {
    pub polynomial: u64,
    pub seed: u64,
    pub length: usize,
    pub scale: f32,
    pub bias: f32,
    pub residuals: Vec<SparseResidual>,
}

// Polinômios primitivos em GF(2). Cada polynomial codifica os taps:
// Para grau k, bit (k-1) DEVE estar setado para garantir período máximo.
const PRIMITIVE_POLYNOMIALS: &[(u32, u64)] = &[
    // Grau 16: x^16 + x^14 + x^13 + x^11 + 1 (taps: 15,13,12,10,0)
    (16, 0b1011_0100_0000_0001),  // bit 15 setado
    // Grau 32: x^32 + x^22 + x^2 + x + 1 (taps: 31,21,1,0)
    (32, 0x8000_0000_0040_0007u64 >> 32 | 0x8040_0007), // bit 31 setado
    // Simplificado: vamos usar um polinômio primitivo comprovado de grau 32:
    // x^32 + x^7 + x^6 + x^2 + 1 → taps em bits 31,6,5,1,0
    // (32, 0x8000_0063),
];

// Polinômio de grau 32 comprovado: x^31 + x^3 + 1 (Galois config)
// Representado como: bit 31 (MSB do grau) + bit 3 + bit 0
const POLY32: u64 = (1u64 << 31) | (1u64 << 3) | 1; // 0x80000009

pub struct SeedLMEncoder;

impl SeedLMEncoder {
    #[inline]
    fn lfsr_step(state: &mut u64, polynomial: u64, degree: u32) -> f32 {
        let feedback = (*state & polynomial).count_ones() & 1;
        let mask = if degree >= 64 { u64::MAX } else { (1u64 << degree) - 1 };
        *state = ((*state >> 1) | ((feedback as u64) << (degree.saturating_sub(1)))) & mask;
        (*state & mask) as f32 / (mask as f32 + 1.0)
    }

    pub fn expand_seed(seed: &LfsrSeed) -> Vec<f32> {
        let degree = Self::degree_for_polynomial(seed.polynomial);
        let mut state = seed.seed;
        let mut output = Vec::with_capacity(seed.length);
        for _ in 0..seed.length {
            let raw = Self::lfsr_step(&mut state, seed.polynomial, degree);
            output.push(raw * seed.scale + seed.bias);
        }
        for r in &seed.residuals {
            if (r.index as usize) < output.len() {
                output[r.index as usize] += r.delta;
            }
        }
        output
    }

    pub fn encode_tensor(tensor: &[f32], max_residual_frac: f32) -> LfsrSeed {
        if tensor.is_empty() {
            return LfsrSeed { polynomial: PRIMITIVE_POLYNOMIALS[1].1, seed: 0,
                length: 0, scale: 1.0, bias: 0.0, residuals: Vec::new() };
        }
        let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-10);
        let scale = range;
        let bias = min_val;
        let normalized: Vec<f32> = tensor.iter()
            .map(|&v| ((v - bias) / scale).clamp(0.0, 0.9999)).collect();

        let mut best_mse = f32::INFINITY;
        let mut best_poly = PRIMITIVE_POLYNOMIALS[1].1;
        let mut best_seed = 1u64;

        for degree in [16u32, 32] {
            let polynomial = if degree == 32 { POLY32 } else { PRIMITIVE_POLYNOMIALS[0].1 };
            if degree >= 64 { continue; } // Skip 64-bit polys for encoding (overflow)
            let num_seeds = 256u64.min(1u64 << degree.min(16));
            let step = ((1u64 << degree.min(32)) / num_seeds).max(1);
            for s in 0..num_seeds {
                let max_state = (1u64 << degree) - 1;
                let cs = (s * step + 1).min(max_state);
                if cs == 0 { continue; }
                let mut state = cs;
                let mut mse = 0.0f32;
                let sample = tensor.len().min(1024);
                for i in 0..sample {
                    let raw = Self::lfsr_step(&mut state, polynomial, degree);
                    let diff = raw - normalized[i];
                    mse += diff * diff;
                }
                mse /= sample as f32;
                if mse < best_mse { best_mse = mse; best_poly = polynomial; best_seed = cs; }
            }
        }

        let expanded = {
            let mut state = best_seed;
            let degree = Self::degree_for_polynomial(best_poly);
            (0..tensor.len()).map(|_| {
                let raw = Self::lfsr_step(&mut state, best_poly, degree);
                raw * scale + bias
            }).collect::<Vec<_>>()
        };

        let threshold = range * max_residual_frac;
        let residuals: Vec<SparseResidual> = tensor.iter().zip(expanded.iter()).enumerate()
            .filter_map(|(i, (&orig, &recon))| {
                let d = orig - recon;
                if d.abs() > threshold { Some(SparseResidual { index: i as u32, delta: d }) }
                else { None }
            }).collect();

        LfsrSeed { polynomial: best_poly, seed: best_seed, length: tensor.len(),
            scale, bias, residuals }
    }

    pub fn encode_tensor_lossless(tensor: &[f32]) -> LfsrSeed {
        let mut seed = Self::encode_tensor(tensor, -1.0);
        let expanded = Self::expand_seed(&LfsrSeed { residuals: Vec::new(), ..seed.clone() });
        seed.residuals = tensor.iter().zip(expanded.iter()).enumerate()
            .filter_map(|(i, (&o, &r))| {
                let d = o - r;
                if d.abs() > f32::EPSILON { Some(SparseResidual { index: i as u32, delta: d }) }
                else { None }
            }).collect();
        seed
    }

    fn degree_for_polynomial(polynomial: u64) -> u32 {
        if polynomial == 0 { 16 } else { 64 - polynomial.leading_zeros() }
    }

    /// Skip-ahead em GF(2): avança o LFSR `n` posições via matrix exponentiation.
    /// Essencial para paralelismo GPU: cada thread skip-aheads ao seu segmento.
    pub fn skip_ahead(seed: u64, polynomial: u64, n: u64) -> u64 {
        let degree = Self::degree_for_polynomial(polynomial) as usize;
        if degree >= 64 || degree == 0 { return seed; }
        let mask = (1u64 << degree) - 1;

        // Matriz companheira A do LFSR Fibonacci (shift-right):
        // A corresponde a: state' = (state >> 1) | (feedback << (degree-1))
        // Linha i representa o mapeamento para bit i do novo estado.
        // bit[i] do novo estado = bit[i+1] do estado antigo (para i < degree-1)
        // bit[degree-1] do novo estado = dot(state, polynomial) mod 2 (feedback)
        let mut matrix = vec![0u64; degree];
        for i in 0..degree - 1 {
            // bit i do novo estado vem do bit i+1 do antigo (shift right)
            matrix[i] = 1u64 << (i + 1);
        }
        // Último bit (MSB) = feedback = XOR dos taps definidos pelo polinômio
        matrix[degree - 1] = polynomial & mask;

        // Identidade
        let mut result = vec![0u64; degree];
        for i in 0..degree { result[i] = 1u64 << i; }

        let mut base = matrix;
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 { result = Self::gf2_mat_mul(&result, &base, degree); }
            base = Self::gf2_mat_mul(&base, &base, degree);
            exp >>= 1;
        }
        Self::gf2_mat_vec(&result, seed, degree) & mask
    }

    fn gf2_mat_mul(a: &[u64], b: &[u64], degree: usize) -> Vec<u64> {
        let mut c = vec![0u64; degree];
        for i in 0..degree { for j in 0..degree {
            if (a[i] >> j) & 1 == 1 { c[i] ^= b[j]; }
        }}
        c
    }

    fn gf2_mat_vec(m: &[u64], v: u64, degree: usize) -> u64 {
        let mut r = 0u64;
        for i in 0..degree {
            let bit = (m[i] & v).count_ones() & 1;
            r |= (bit as u64) << i;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_seed_reproducible() {
        let s = LfsrSeed { polynomial: PRIMITIVE_POLYNOMIALS[1].1, seed: 99999,
            length: 100, scale: 1.0, bias: 0.0, residuals: Vec::new() };
        assert_eq!(SeedLMEncoder::expand_seed(&s), SeedLMEncoder::expand_seed(&s));
    }

    #[test]
    fn test_encode_decode_roundtrip_lossless() {
        let tensor: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let seed = SeedLMEncoder::encode_tensor_lossless(&tensor);
        let recon = SeedLMEncoder::expand_seed(&seed);
        let max_err: f32 = tensor.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "Lossless error {} too high", max_err);
    }

    #[test]
    fn test_skip_ahead_matches_sequential() {
        let poly = POLY32;
        let degree = SeedLMEncoder::degree_for_polynomial(poly);
        let initial = 42u64;
        let skip_n = 500u64;
        let skipped = SeedLMEncoder::skip_ahead(initial, poly, skip_n);
        let mut state = initial;
        for _ in 0..skip_n { SeedLMEncoder::lfsr_step(&mut state, poly, degree); }
        assert_eq!(skipped, state, "skip_ahead must match sequential");
    }
}
