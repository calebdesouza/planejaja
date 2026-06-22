use nodestor_core::{TensorInfo, TensorDtype};
use rayon::prelude::*;
use std::sync::Arc;

/// CPU Fallback Engine para máquinas sem suporte à GPU.
/// Opera puramente em f32 via Rayon para multi-threading básico.
pub struct CpuBackend {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl CpuBackend {
    pub fn new() -> anyhow::Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get_physical())
            .build()?;
            
        Ok(Self {
            thread_pool: Arc::new(pool),
        })
    }

    /// Implementação ingênua O(N^3) de MatMul paralelizada em blocos
    pub fn matmul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        self.thread_pool.install(|| {
            c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
                let a_row = &a[i * k .. (i + 1) * k];
                
                // Transposição on-the-fly (lenta, mas funcional para fallback)
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += a_row[p] * b[p * n + j];
                    }
                    c_row[j] = sum;
                }
            });
        });
    }

    pub fn rms_norm(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        self.thread_pool.install(|| {
            let n = x.len();
            let sum_sq: f32 = x.par_iter().map(|&v| v * v).sum();
            let rms = (sum_sq / n as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            
            x.par_iter_mut().zip(weight.par_iter()).for_each(|(xi, &wi)| {
                *xi = (*xi * inv_rms) * wi;
            });
        });
    }

    pub fn rotary_embedding(&self, q: &mut [f32], k_buf: &mut [f32], pos: usize, head_dim: usize) {
        self.thread_pool.install(|| {
            let inv_freq: Vec<f32> = (0..head_dim/2).map(|i| {
                1.0 / 10000_f32.powf((2 * i) as f32 / head_dim as f32)
            }).collect();
            
            let apply_rope = |vec: &mut [f32]| {
                for i in 0..head_dim/2 {
                    let theta = pos as f32 * inv_freq[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();
                    
                    let v0 = vec[i * 2];
                    let v1 = vec[i * 2 + 1];
                    
                    vec[i * 2] = v0 * cos_theta - v1 * sin_theta;
                    vec[i * 2 + 1] = v0 * sin_theta + v1 * cos_theta;
                }
            };
            
            q.chunks_mut(head_dim).for_each(apply_rope);
            k_buf.chunks_mut(head_dim).for_each(apply_rope);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul() {
        let cpu = CpuBackend::new().unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![2.0, 0.0, 1.0, 2.0]; // 2x2
        let mut c = vec![0.0; 4];
        
        cpu.matmul_f32(&a, &b, &mut c, 2, 2, 2);
        
        assert_eq!(c, vec![4.0, 4.0, 10.0, 8.0]);
    }
}
