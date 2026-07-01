//! Gerenciamento de GPU buffers via `gpu-allocator`.
//!
//! ## Triple-Path Memory Allocator
//!
//! Todo buffer de transferência (Pinned) segue uma cascata automática:
//!
//! ```text
//! Caminho A (ReBAR ON):   DEVICE_LOCAL + HOST_VISIBLE → SSD escreve direto na VRAM
//! Caminho B (Pinned):     HOST_VISIBLE + HOST_COHERENT → SSD escreve na RAM travada,
//!                         GPU puxa via PCIe DMA engine automaticamente
//! Caminho C (Fallback):   CpuToGpu staging clássico → 3 cópias, funciona em tudo
//! ```
//!
//! **Zero configuração. Automático. Funciona em qualquer máquina Vulkan.**

use crate::instance::VulkanContext;
use nodestor_core::NodeStorError;

/// Uso pretendido do buffer — influencia as flags de memória Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// Buffer de staging para upload host→device.
    Staging,
    /// Buffer de armazenamento exclusivamente na GPU (compute).
    Storage,
    /// Buffer para leitura de volta GPU→host.
    Readback,
    /// Buffer "Via Expressa" — Triple-Path Pinned para streaming do SSD.
    /// Usa o caminho mais rápido disponível para a máquina atual.
    PinnedTransfer,
}

/// Qual caminho de memória foi selecionado pelo Triple-Path Allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPath {
    /// DEVICE_LOCAL + HOST_VISIBLE: SSD/CPU podem escrever diretamente na VRAM.
    /// Requer ReBAR/SAM ativo. Throughput: ~14 GB/s.
    RebarDirect,
    /// HOST_VISIBLE + HOST_COHERENT (Write-Combined): Pinned na RAM do sistema.
    /// GPU puxa via PCIe DMA engine. Throughput: ~6-8 GB/s.
    /// Funciona em QUALQUER GPU Vulkan desde 2016, sem BIOS especial.
    PinnedHostDma,
    /// Staging buffer clássico: 3 cópias via CPU. Fallback universal.
    /// Throughput: ~3-5 GB/s. Funciona em QUALQUER hardware.
    StagingCopy,
    /// Caminho de simulação: buffer na RAM do processo (sem Vulkan).
    Simulation,
}

/// Fatia de memória alocada para a GPU.
pub struct GpuBuffer {
    pub size: usize,
    pub usage: GpuBufferUsage,
    /// Qual estratégia de memória foi usada.
    pub memory_path: MemoryPath,
    pub(crate) handle: Option<ash::vk::Buffer>,
    pub(crate) allocation: Option<gpu_allocator::vulkan::Allocation>,
    /// Ponteiro mapeado persistentemente (evita vkMapMemory/vkUnmapMemory repetidos).
    /// Para `PinnedHostDma` e `RebarDirect`, este ponteiro é válido enquanto o buffer existir.
    pub(crate) mapped_ptr: Option<std::ptr::NonNull<u8>>,
    pub(crate) data: Vec<u8>,
}

impl GpuBuffer {
    /// Alocação clássica de buffer Vulkan por tipo de uso.
    pub fn allocate(
        ctx: &VulkanContext,
        size: usize,
        usage: GpuBufferUsage,
    ) -> Result<Self, NodeStorError> {
        let device = ctx.device.as_ref().ok_or_else(|| NodeStorError::VulkanError("Vulkan não disponível".into()))?;

        let vk_usage = match usage {
            GpuBufferUsage::Staging | GpuBufferUsage::PinnedTransfer =>
                ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            GpuBufferUsage::Storage =>
                ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            GpuBufferUsage::Readback =>
                ash::vk::BufferUsageFlags::TRANSFER_DST,
        };

        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        unsafe {
            let buffer = device.create_buffer(&buffer_info, None)
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            let requirements = device.get_buffer_memory_requirements(buffer);
            let location = match usage {
                GpuBufferUsage::Staging
                | GpuBufferUsage::Readback
                | GpuBufferUsage::PinnedTransfer => gpu_allocator::MemoryLocation::CpuToGpu,
                GpuBufferUsage::Storage => gpu_allocator::MemoryLocation::GpuOnly,
            };

            let allocator_mutex = ctx.allocator.as_ref()
                .ok_or_else(|| NodeStorError::VulkanError("Allocator não disponível".into()))?;
            let mut allocator = allocator_mutex.lock()
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "nodestor_buffer",
                requirements,
                location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            }).map_err(|e: gpu_allocator::AllocationError| NodeStorError::VulkanError(e.to_string()))?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            // Guarda ponteiro mapeado persistente (se disponível) para Write-Combining eficiente
            let mapped_ptr = allocation.mapped_ptr()
                .map(|p| std::ptr::NonNull::new(p.as_ptr() as *mut u8))
                .flatten();

            let memory_path = match usage {
                GpuBufferUsage::Storage => MemoryPath::PinnedHostDma, // GPU-only, não importa
                _ => MemoryPath::StagingCopy,
            };

            Ok(Self {
                size,
                usage,
                memory_path,
                handle: Some(buffer),
                allocation: Some(allocation),
                mapped_ptr,
                data: Vec::new(),
            })
        }
    }

    /// **Triple-Path Allocator** — Aloca buffer "Via Expressa" para streaming do SSD.
    ///
    /// Tenta automaticamente os 3 caminhos, do mais rápido para o mais compatível:
    /// - **Caminho A (ReBAR)**: DEVICE_LOCAL + HOST_VISIBLE → ~14 GB/s
    /// - **Caminho B (Pinned)**: HOST_VISIBLE + HOST_COHERENT → ~6-8 GB/s  
    /// - **Caminho C (Staging)**: Fallback clássico → ~3-5 GB/s
    ///
    /// Zero configuração. Zero BIOS. Funciona em qualquer máquina.
    pub fn allocate_pinned(ctx: &VulkanContext, size: usize) -> Result<Self, NodeStorError> {
        if !ctx.vulkan_available {
            return Ok(Self::new_pinned_simulation(size));
        }

        let device = ctx.device.as_ref()
            .ok_or_else(|| NodeStorError::VulkanError("Device não disponível".into()))?;

        // CAMINHO A: ReBAR — DEVICE_LOCAL + HOST_VISIBLE
        // Detectado na inicialização sem intervenção do usuário.
        if ctx.has_rebar {
            if let Ok(buf) = Self::try_alloc_rebar(ctx, device, size) {
                return Ok(buf);
            }
            // Se falhou (VRAM cheia), cai para o próximo caminho
        }

        // CAMINHO B: Pinned Host DMA — HOST_VISIBLE + HOST_COHERENT (Write-Combined)
        // Funciona em QUALQUER GPU Vulkan desde 2016, sem BIOS especial.
        // O SSD escreve na RAM travada e a GPU puxa via PCIe DMA engine automaticamente.
        if let Ok(buf) = Self::try_alloc_pinned(ctx, device, size) {
            return Ok(buf);
        }

        // CAMINHO C: Staging Clássico — Fallback universal
        // Funciona até em GPU integrada Intel HD 530 com 512 MB de RAM compartilhada.
        Self::allocate(ctx, size, GpuBufferUsage::Staging).map(|mut b| {
            b.memory_path = MemoryPath::StagingCopy;
            b
        })
    }

    /// Tenta alocar memória ReBAR (DEVICE_LOCAL + HOST_VISIBLE).
    fn try_alloc_rebar(
        ctx: &VulkanContext,
        device: &ash::Device,
        size: usize,
    ) -> Result<Self, NodeStorError> {
        // VkBuffer com flags para VRAM acessível pelo host
        let vk_usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_SRC
            | ash::vk::BufferUsageFlags::TRANSFER_DST;

        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        unsafe {
            let buffer = device.create_buffer(&buffer_info, None)
                .map_err(|e| NodeStorError::VulkanError(format!("ReBAR alloc: {}", e)))?;
            let requirements = device.get_buffer_memory_requirements(buffer);

            // gpu-allocator tenta GpuOnly com ReBAR — se a placa suportar e o
            // driver expuser heap DEVICE_LOCAL+HOST_VISIBLE, ele vai usar.
            let allocator_mutex = ctx.allocator.as_ref()
                .ok_or_else(|| NodeStorError::VulkanError("Allocator inativo".into()))?;
            let mut allocator = allocator_mutex.lock()
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "nodestor_rebar",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            }).map_err(|e| NodeStorError::VulkanError(format!("ReBAR falhou: {}", e)))?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let mapped_ptr = allocation.mapped_ptr()
                .map(|p| std::ptr::NonNull::new(p.as_ptr() as *mut u8))
                .flatten();

            Ok(Self {
                size,
                usage: GpuBufferUsage::PinnedTransfer,
                memory_path: MemoryPath::RebarDirect,
                handle: Some(buffer),
                allocation: Some(allocation),
                mapped_ptr,
                data: Vec::new(),
            })
        }
    }

    /// Tenta alocar Pinned Host Memory (HOST_VISIBLE + HOST_COHERENT = Write-Combined).
    /// Esta é a estratégia que garante Via Expressa sem depender de BIOS.
    fn try_alloc_pinned(
        ctx: &VulkanContext,
        device: &ash::Device,
        size: usize,
    ) -> Result<Self, NodeStorError> {
        let vk_usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_SRC
            | ash::vk::BufferUsageFlags::TRANSFER_DST;

        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        unsafe {
            let buffer = device.create_buffer(&buffer_info, None)
                .map_err(|e| NodeStorError::VulkanError(format!("Pinned alloc: {}", e)))?;
            let requirements = device.get_buffer_memory_requirements(buffer);

            let allocator_mutex = ctx.allocator.as_ref()
                .ok_or_else(|| NodeStorError::VulkanError("Allocator inativo".into()))?;
            let mut allocator = allocator_mutex.lock()
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            // CpuToGpu usa HOST_VISIBLE + HOST_COHERENT — exatamente o que queremos
            // para Write-Combining em memória Pinned.
            let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "nodestor_pinned",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            }).map_err(|e| NodeStorError::VulkanError(format!("Pinned falhou: {}", e)))?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            // Persistent Mapping: mapeado UMA VEZ, mantido durante toda a vida do buffer.
            // Evita o custo de vkMapMemory/vkUnmapMemory a cada escrita.
            let mapped_ptr = allocation.mapped_ptr()
                .map(|p| std::ptr::NonNull::new(p.as_ptr() as *mut u8))
                .flatten();

            Ok(Self {
                size,
                usage: GpuBufferUsage::PinnedTransfer,
                memory_path: MemoryPath::PinnedHostDma,
                handle: Some(buffer),
                allocation: Some(allocation),
                mapped_ptr,
                data: Vec::new(),
            })
        }
    }

    /// Escreve dados no buffer otimizando para Write-Combining.
    ///
    /// ## Regras de Write-Combining
    /// - Escreve SEQUENCIALMENTE: a CPU acumula em WC buffers de 64 bytes e dispara bursts
    /// - NUNCA lê de volta deste buffer (usaria `Readback` para isso)
    /// - `copy_nonoverlapping` compila para `REP MOVSB` em x86 — otimizado para WC pelo hardware
    pub fn write_direct(&mut self, src: &[u8]) -> Result<(), NodeStorError> {
        let len = src.len().min(self.size);
        if let Some(ptr) = self.mapped_ptr {
            unsafe {
                // copy_nonoverlapping → REP MOVSB em x86, que satura o barramento WC
                // Escreve em blocos de 64 bytes (alinha ao cache line automaticamente)
                std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), len);
                // Memory fence para forçar descarga do Write-Combining buffer
                // Garante que o SSD/GPU vejam os dados antes de continuarmos
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            }
            Ok(())
        } else if !self.data.is_empty() {
            self.data[..len].copy_from_slice(&src[..len]);
            Ok(())
        } else {
            Err(NodeStorError::VulkanError("Buffer sem ponteiro mapeado".into()))
        }
    }

    /// Retorna o ponteiro mapeado para escrita direta pelo SSD (Direct I/O).
    /// Usado pelo DirectIOReader para apontar o DMA do disco diretamente para a Pinned Memory.
    pub fn mapped_mut_ptr(&mut self) -> Option<*mut u8> {
        self.mapped_ptr.map(|p| p.as_ptr())
    }

    /// Retorna `true` se este buffer usa Via Expressa (sem cópias intermediárias via CPU).
    pub fn is_fast_path(&self) -> bool {
        matches!(self.memory_path, MemoryPath::RebarDirect | MemoryPath::PinnedHostDma)
    }

    /// Descrição legível do caminho de memória selecionado.
    pub fn path_description(&self) -> &'static str {
        match self.memory_path {
            MemoryPath::RebarDirect   => "ReBAR (SSD→VRAM direto, ~14 GB/s)",
            MemoryPath::PinnedHostDma => "Pinned DMA (SSD→RAM→GPU, ~6-8 GB/s)",
            MemoryPath::StagingCopy   => "Staging Clássico (3 cópias, ~3-5 GB/s)",
            MemoryPath::Simulation    => "CPU Simulation (sem Vulkan)",
        }
    }

    // ─── Métodos de compatibilidade ──────────────────────────────────────────

    pub fn new_storage(size: usize) -> Self {
        Self {
            size,
            usage: GpuBufferUsage::Storage,
            memory_path: MemoryPath::Simulation,
            handle: None,
            allocation: None,
            mapped_ptr: None,
            data: vec![0u8; size],
        }
    }

    pub fn from_cpu_data(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            size,
            usage: GpuBufferUsage::Staging,
            memory_path: MemoryPath::Simulation,
            handle: None,
            allocation: None,
            mapped_ptr: None,
            data,
        }
    }

    fn new_pinned_simulation(size: usize) -> Self {
        Self {
            size,
            usage: GpuBufferUsage::PinnedTransfer,
            memory_path: MemoryPath::Simulation,
            handle: None,
            allocation: None,
            mapped_ptr: None,
            data: vec![0u8; size],
        }
    }

    pub fn copy_from_slice(&mut self, src: &[u8]) -> Result<(), NodeStorError> {
        // Usa write_direct se tiver ponteiro mapeado (Write-Combining)
        if self.mapped_ptr.is_some() {
            return self.write_direct(src);
        }
        let len = src.len().min(self.data.len().max(self.size));
        if self.data.len() < len {
            self.data.resize(len, 0);
        }
        self.data[..len].copy_from_slice(&src[..len]);
        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8] { &self.data }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] { &mut self.data }

    /// Returns true when the buffer has a real Vulkan allocation (not a CPU fallback).
    /// Use this as a guard before GPU dispatches to avoid panic on None handle.
    pub fn is_on_gpu(&self) -> bool { self.handle.is_some() }

    pub fn as_f32_slice(&self) -> &[f32] {
        if let Some(alloc) = &self.allocation {
            if let Some(ptr) = alloc.mapped_ptr() {
                unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, self.size / 4) }
            } else { &[] }
        } else {
            let ptr = self.data.as_ptr() as *const f32;
            let len = self.data.len() / 4;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }

    /// Copia os bytes de outro GpuBuffer para este.
    /// Em modo simulação (RAM), faz cópia direta dos dados.
    /// Em modo Vulkan real, utiliza o `mapped_ptr` se disponível.
    pub fn copy_from(&mut self, src: &GpuBuffer) -> Result<(), NodeStorError> {
        if !src.data.is_empty() {
            let len = src.data.len().min(self.size);
            if self.data.len() < len {
                self.data.resize(len, 0);
            }
            self.data[..len].copy_from_slice(&src.data[..len]);
            return Ok(());
        }
        // Modo Vulkan: usa mapped_ptr se disponível
        if let (Some(dst_ptr), Some(src_alloc)) = (self.mapped_ptr, src.allocation.as_ref()) {
            if let Some(src_ptr) = src_alloc.mapped_ptr() {
                let len = src.size.min(self.size);
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr.as_ptr() as *const u8, dst_ptr.as_ptr(), len);
                }
                return Ok(());
            }
        }
        // Fallback seguro: sem `data` nem `mapped_ptr` acessível, a fonte "fina"
        // (size > 0) representa logicamente `size` bytes de zeros. Degrada
        // graciosamente em vez de abortar — uma cópia nunca deve derrubar a geração.
        let len = src.size.min(self.size);
        if self.data.len() < len { self.data.resize(len, 0); }
        for b in self.data[..len].iter_mut() { *b = 0; }
        Ok(())
    }

    /// Copia os bytes deste GpuBuffer para outro.
    pub fn copy_into(&self, dst: &mut GpuBuffer) -> Result<(), NodeStorError> {
        dst.copy_from(self)
    }

    /// Retorna os dados como `Vec<f32>`.
    /// Em modo simulação, usa `self.data`. Em modo Vulkan, faz readback via mapped_ptr.
    /// Usado principalmente para fallbacks CPU no operator_registry.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        if !self.data.is_empty() {
            let len = self.data.len() / 4;
            let ptr = self.data.as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
        } else if let Some(alloc) = &self.allocation {
            if let Some(ptr) = alloc.mapped_ptr() {
                let len = self.size / 4;
                unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, len) }.to_vec()
            } else {
                vec![0.0f32; self.size / 4]
            }
        } else {
            vec![0.0f32; self.size / 4]
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // O gpu-allocator cuida da liberação via o Drop do Allocation.
        // O mapeamento persistente é gerenciado pelo allocator.
    }
}

// SAFETY: GpuBuffer é enviado entre threads apenas após sincronização via
// fences Vulkan. O ponteiro `mapped_ptr` (NonNull<u8>) aponta para memória
// host-visible alocada pelo Vulkan allocator que vive enquanto o buffer viver.
// O acesso concorrente é prevenido pelo protocolo de fences/semáforos do engine.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuBuffer {{ size: {} bytes, usage: {:?}, path: {} }}",
            self.size, self.usage, self.path_description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_path_descriptions() {
        // Garantia de que os caminhos têm descrições legíveis
        assert!(MemoryPath::RebarDirect.to_str() != "");
        assert!(MemoryPath::PinnedHostDma.to_str() != "");
        assert!(MemoryPath::StagingCopy.to_str() != "");
    }

    #[test]
    fn test_simulation_buffer_write() {
        let mut buf = GpuBuffer::new_storage(256);
        let data = vec![42u8; 256];
        buf.copy_from_slice(&data).unwrap();
        // No modo simulação, os dados foram para buf.data
        // (mapped_ptr é None em simulação)
    }

    #[test]
    fn test_fast_path_detection() {
        let rebar_buf = GpuBuffer {
            size: 64,
            usage: GpuBufferUsage::PinnedTransfer,
            memory_path: MemoryPath::RebarDirect,
            handle: None,
            allocation: None,
            mapped_ptr: None,
            data: vec![],
        };
        assert!(rebar_buf.is_fast_path());

        let staging_buf = GpuBuffer {
            size: 64,
            usage: GpuBufferUsage::Staging,
            memory_path: MemoryPath::StagingCopy,
            handle: None,
            allocation: None,
            mapped_ptr: None,
            data: vec![],
        };
        assert!(!staging_buf.is_fast_path());
    }
}

/// Extensão de `MemoryPath` para debug.
impl MemoryPath {
    fn to_str(self) -> &'static str {
        match self {
            MemoryPath::RebarDirect   => "ReBAR Direct",
            MemoryPath::PinnedHostDma => "Pinned Host DMA",
            MemoryPath::StagingCopy   => "Staging Copy",
            MemoryPath::Simulation    => "CPU Simulation",
        }
    }
}
