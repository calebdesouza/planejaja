//! MambaLayer (SSM - State Space Model)
//!
//! Integra o Selective Scan da arquitetura Mamba 2 ao NodeStor.
//! Diferente de blocos Transformer tradicionais, Mamba propaga um estado
//! oculto ao invés de usar KV Cache. Isso dá a ele inferência O(1) de memória
//! por token gerado, eliminando os gargalos mortais de VRAM.

use nodestor_core::NodeStorError;
use nodestor_vulkan::VulkanEngine;
use nodestor_vulkan::GpuBuffer;
use std::sync::Arc;

pub struct MambaLayer {
    engine: Arc<VulkanEngine>,
    pub d_inner: u32,
    pub d_state: u32,
    state_a: GpuBuffer,  // Ping
    state_b: GpuBuffer,  // Pong
    use_a: bool,         // Flag de alternância
}

impl MambaLayer {
    pub fn new(engine: Arc<VulkanEngine>, d_inner: u32, d_state: u32) -> Result<Self, NodeStorError> {
        let state_a = engine.alloc_buffer((d_inner * d_state * 4) as usize)?;
        let state_b = engine.alloc_buffer((d_inner * d_state * 4) as usize)?;
        
        Ok(Self {
            engine,
            d_inner,
            d_state,
            state_a,
            state_b,
            use_a: true,
        })
    }

    /// Executa um passo do Selective Scan.
    pub fn forward(
        &mut self,
        u: &GpuBuffer,
        delta: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        seq_len: u32,
    ) -> Result<GpuBuffer, NodeStorError> {
        // Vulkan shader current updates in-place, we pass the active buffer
        // Note: The shader would need an update to read from one and write to another to fully utilize ping-pong.
        // For now, we alternate to satisfy the API mutability requests.
        let active_state = if self.use_a { &self.state_a } else { &self.state_b };
        
        let out = self.engine.mamba_selective_scan(
            u,
            delta,
            a,
            b,
            c,
            active_state,
            seq_len,
            self.d_inner,
            self.d_state,
        )?;
        
        // Em um ping-pong real (com shader atualizado), inverteríamos use_a.
        // Como o shader faz in-place, inverte-lo sem cópia perderia o estado atual.
        // self.use_a = !self.use_a; 
        
        Ok(out)
    }
}
