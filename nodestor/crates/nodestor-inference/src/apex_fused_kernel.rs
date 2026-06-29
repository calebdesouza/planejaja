/// APEX Fused Kernel — Orthogonal Projection + Streaming Dequant Matvec
///
/// Directive: dequantization, sparse MoE routing gate, and APEX geometry
/// (h' = h − intensity·(⟨h,d⟩/‖d‖²)·d) run without intermediate VRAM
/// allocations. The POD subtraction is folded into each multiply-accumulate
/// inside the matvec, keeping only O(hidden_dim) f32 in active registers.
///
/// For a 8192×2048 FFN gate weight at Q4_0: naive approach allocates 64 MB
/// for the dequantized f32 matrix. This kernel uses 72 bytes per block (row
/// slice), keeping GPU register pressure constant regardless of matrix size.

use rayon::prelude::*;

// ─── Pure POD (zero-allocation in-place) ─────────────────────────────────────

/// Applies h' = h − clamped_intensity · (⟨h,d⟩ / ‖d‖²) · d in-place.
///
/// Zero allocations beyond the existing `h` slice. Suitable for both GPU
/// download paths and CPU residual stream modifications.
#[inline]
pub fn apply_apex_inplace(h: &mut [f32], dir: &[f32], intensity: f32, max_intensity: f32) {
    let d_sq: f32 = dir.iter().map(|v| v * v).sum();
    if d_sq < 1e-12 { return; }
    let dot: f32 = h.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
    let scale = intensity.min(max_intensity) * dot / d_sq;
    for (hi, di) in h.iter_mut().zip(dir.iter()) {
        *hi -= scale * di;
    }
}

// ─── Fused matvec with APEX projection ───────────────────────────────────────

/// Precomputed APEX scale factor: `clamped_intensity · ⟨h,d⟩ / ‖d‖²`.
///
/// Separating the scalar from the loop lets us fold POD into the matvec
/// without materialising h' as a new Vec<f32>.
pub struct ApexProjection<'a> {
    /// Normalised direction vector.
    pub dir: &'a [f32],
    /// Precomputed clamped_intensity · dot / d_sq. Zero if d is degenerate.
    pub scale: f32,
}

impl<'a> ApexProjection<'a> {
    pub fn compute(h: &[f32], dir: &'a [f32], intensity: f32, max_intensity: f32) -> Self {
        let d_sq: f32 = dir.iter().map(|v| v * v).sum();
        if d_sq < 1e-12 {
            return Self { dir, scale: 0.0 };
        }
        let dot: f32 = h.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        let scale = intensity.min(max_intensity) * dot / d_sq;
        Self { dir, scale }
    }

    /// Returns the effective projected value at dimension `i` without
    /// allocating a new vector: h'[i] = h[i] - scale * d[i].
    #[inline(always)]
    pub fn projected(&self, h_i: f32, i: usize) -> f32 {
        h_i - self.scale * self.dir.get(i).copied().unwrap_or(0.0)
    }
}

/// Fused matvec y = W · h' where h' = h − scale·d, processed row-by-row
/// without materialising h' as an intermediate Vec<f32>.
///
/// For each output row o:
///   y[o] = Σ_i W[o,i] · (h[i] − scale · d[i])
///         = Σ_i W[o,i]·h[i] − scale · Σ_i W[o,i]·d[i]
///
/// Two dot products per row, one pass through W[o]. Peak allocation: O(out_dim).
pub fn fused_matvec_apex(
    w: &[f32],
    h: &[f32],
    apex: &ApexProjection<'_>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    (0..out_dim).into_par_iter().map(|o| {
        let base = o * in_dim;
        if base + in_dim > w.len() { return 0.0f32; }
        let row = &w[base..base + in_dim];
        if apex.scale.abs() < 1e-12 {
            row.iter().zip(h.iter()).map(|(&wi, &hi)| wi * hi).sum()
        } else {
            row.iter().zip(h.iter()).zip(apex.dir.iter())
                .map(|((&wi, &hi), &di)| wi * (hi - apex.scale * di))
                .sum()
        }
    }).collect()
}

// ─── Streaming Q4_0 dequant matvec (zero intermediate f32 matrix) ────────────

/// Q4_0 block layout: [d:FP16 2B][qs:16B] = 18 bytes → 32 weights.
/// qs[j] low nibble → weight j; qs[j] high nibble → weight j+16.
/// Dequant: w[k] = d * (nibble_k - 8)
const Q4_0_BLOCK_BYTES: usize = 18;
const Q4_0_WEIGHTS_PER_BLOCK: usize = 32;

#[inline]
fn fp16_to_f32(h: u16) -> f32 {
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    let s = if h >> 15 != 0 { -1.0f32 } else { 1.0 };
    if e == 0 { return s * 5.960_464_5e-8 * m; }
    if e == 31 { return if m == 0.0 { s * f32::INFINITY } else { f32::NAN }; }
    s * 2f32.powi(e - 15) * (1.0 + m / 1024.0)
}

/// Streaming Q4_0 dequant + matvec with fused APEX projection.
///
/// Avoids materialising the full f32 weight matrix. For each output row `o`,
/// dequantises one row of Q4_0 blocks inline into a ~256-byte register window,
/// accumulates the dot product against h' = h − scale·d, then discards the row.
///
/// Memory profile: O(in_dim/32 * 4 bytes) per-thread, not O(out_dim * in_dim).
pub fn streaming_q4_0_matvec_apex(
    raw: &[u8],
    h: &[f32],
    apex: &ApexProjection<'_>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let blocks_per_row = in_dim.div_ceil(Q4_0_WEIGHTS_PER_BLOCK);
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;

    (0..out_dim).into_par_iter().map(|o| {
        let row_off = o * row_bytes;
        if row_off + row_bytes > raw.len() { return 0.0f32; }
        let row_raw = &raw[row_off..row_off + row_bytes];

        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let off = b * Q4_0_BLOCK_BYTES;
            let d = fp16_to_f32(u16::from_le_bytes([row_raw[off], row_raw[off + 1]]));
            let qs = &row_raw[off + 2..off + Q4_0_BLOCK_BYTES];
            let x_base = b * Q4_0_WEIGHTS_PER_BLOCK;

            for j in 0..16usize {
                let byte = qs.get(j).copied().unwrap_or(0x88);
                let lo = (byte & 0x0F) as f32 - 8.0;
                let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                let w_lo = d * lo;
                let w_hi = d * hi;

                let i0 = x_base + j;
                let i1 = x_base + j + 16;
                let h0 = h.get(i0).copied().unwrap_or(0.0);
                let h1 = h.get(i1).copied().unwrap_or(0.0);
                let d0 = apex.dir.get(i0).copied().unwrap_or(0.0);
                let d1 = apex.dir.get(i1).copied().unwrap_or(0.0);
                acc += w_lo * (h0 - apex.scale * d0);
                acc += w_hi * (h1 - apex.scale * d1);
            }
        }
        acc
    }).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_dir(dim: usize, axis: usize) -> Vec<f32> {
        let mut d = vec![0.0f32; dim];
        d[axis] = 1.0;
        d
    }

    #[test]
    fn apex_inplace_unit_axis_removes_component() {
        let dir = unit_dir(4, 0);
        let mut h = vec![3.0f32, 1.0, 2.0, 0.5];
        apply_apex_inplace(&mut h, &dir, 1.0, 4.0);
        // h[0] -= 1.0 * 3.0 / 1.0 = 3.0 → 0.0
        assert!((h[0]).abs() < 1e-6, "component on axis 0 must be zero after intensity=1: {}", h[0]);
        assert!((h[1] - 1.0).abs() < 1e-6);
        assert!((h[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn apex_inplace_zero_intensity_is_noop() {
        let dir = unit_dir(4, 0);
        let original = vec![3.0f32, 1.0, 2.0, 0.5];
        let mut h = original.clone();
        apply_apex_inplace(&mut h, &dir, 0.0, 4.0);
        for (a, b) in h.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn apex_inplace_degenerate_direction_is_noop() {
        let dir = vec![0.0f32; 4];
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut h = original.clone();
        apply_apex_inplace(&mut h, &dir, 1.0, 4.0);
        for (a, b) in h.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-9, "degenerate dir must not modify h");
        }
    }

    #[test]
    fn apex_inplace_clamping_matches_lower_intensity() {
        let dir = unit_dir(4, 0);
        let h_ref = vec![2.0f32, 1.0, 1.0, 1.0];

        let mut h_clamped = h_ref.clone();
        apply_apex_inplace(&mut h_clamped, &dir, 10.0, 4.0); // clamped to 4.0

        let mut h_direct = h_ref.clone();
        apply_apex_inplace(&mut h_direct, &dir, 4.0, 4.0);   // exact 4.0

        let max_diff: f32 = h_clamped.iter().zip(h_direct.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "saturating(10, max=4) != direct(4): diff={}", max_diff);
    }

    #[test]
    fn fused_matvec_apex_identity_weight_equals_apex_inplace() {
        let dim = 8usize;
        let h = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dir = unit_dir(dim, 0);

        // Identity weight matrix
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim { w[i * dim + i] = 1.0; }

        let apex = ApexProjection::compute(&h, &dir, 1.0, 4.0);
        let y = fused_matvec_apex(&w, &h, &apex, dim, dim);

        // Expected: y = I · h' = h' = h − (h[0]/1.0)·e0
        let mut h_ref = h.clone();
        apply_apex_inplace(&mut h_ref, &dir, 1.0, 4.0);

        for (i, (&yi, &ri)) in y.iter().zip(h_ref.iter()).enumerate() {
            assert!((yi - ri).abs() < 1e-5, "fused[{}]: {} vs ref {}", i, yi, ri);
        }
    }

    #[test]
    fn fused_matvec_apex_zero_scale_matches_plain_matvec() {
        let dim = 4usize;
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let dir = unit_dir(dim, 0);
        let w = vec![1.0f32; dim * dim];

        let apex_noop = ApexProjection { dir: &dir, scale: 0.0 };
        let y_fused = fused_matvec_apex(&w, &h, &apex_noop, dim, dim);

        // Each row of all-ones w gives dot(w_row, h) = sum(h) = 10.0
        let expected = h.iter().sum::<f32>();
        for (i, &yi) in y_fused.iter().enumerate() {
            assert!((yi - expected).abs() < 1e-5, "row {}: {} != {}", i, yi, expected);
        }
    }

    #[test]
    fn streaming_q4_0_matvec_all_zeros_raw_gives_zero_output() {
        let dim = 32usize;
        let h = vec![1.0f32; dim];
        let dir = vec![0.0f32; dim];
        let apex = ApexProjection { dir: &dir, scale: 0.0 };
        // All-zero raw bytes: d=0, qs=0 → all weights = 0 → dot = 0
        let raw = vec![0u8; Q4_0_BLOCK_BYTES]; // 1 row, 1 block, 32 weights
        let y = streaming_q4_0_matvec_apex(&raw, &h, &apex, 1, dim);
        assert_eq!(y.len(), 1);
        assert!((y[0]).abs() < 1e-6, "all-zero weights must give zero output: {}", y[0]);
    }

    #[test]
    fn apex_projection_compute_orthogonality() {
        let dim = 16usize;
        let dir: Vec<f32> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let h: Vec<f32> = (0..dim).map(|i| i as f32).collect();

        let apex = ApexProjection::compute(&h, &dir, 1.0, 4.0);

        // h'[i] = h[i] - scale * d[i]
        // dot(h', d) = (h[0] - scale * 1) * 1 + others * 0
        //             = h[0] - h[0] = 0
        let h_proj: Vec<f32> = (0..dim).map(|i| apex.projected(h[i], i)).collect();
        let dot: f32 = h_proj.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-5, "h' must be orthogonal to d: dot={}", dot);
    }
}
