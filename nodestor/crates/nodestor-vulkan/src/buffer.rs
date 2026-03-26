//! Gerenciamento de GPU buffers via `gpu-allocator`.

use crate::instance::VulkanContext;
use nodestor_core::NodeStorError;

/// Uso pretendido do buffer — influencia as flags de memória Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBufferUsage {
    Staging,
    Storage,
    Readback,
}

/// Fatia de memória na GPU.
pub struct GpuBuffer {
    pub size: usize,
    pub usage: GpuBufferUsage,
    pub(crate) handle: Option<ash::vk::Buffer>,
    pub(crate) allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub(crate) data: Vec<u8>,
}

impl GpuBuffer {
    pub fn allocate(
        ctx: &VulkanContext,
        size: usize,
        usage: GpuBufferUsage,
    ) -> Result<Self, NodeStorError> {
        let device = ctx.device.as_ref().ok_or_else(|| NodeStorError::VulkanError("Vulkan não disponível".into()))?;
        
        let vk_usage = match usage {
            GpuBufferUsage::Staging => ash::vk::BufferUsageFlags::TRANSFER_SRC,
            GpuBufferUsage::Storage => ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::TRANSFER_DST,
            GpuBufferUsage::Readback => ash::vk::BufferUsageFlags::TRANSFER_DST,
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
                GpuBufferUsage::Staging | GpuBufferUsage::Readback => gpu_allocator::MemoryLocation::CpuToGpu,
                GpuBufferUsage::Storage => gpu_allocator::MemoryLocation::GpuOnly,
            };

            let allocator_mutex = ctx.allocator.as_ref().ok_or_else(|| NodeStorError::VulkanError("Allocator não disponível".into()))?;
            let mut allocator = allocator_mutex.lock().map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "nodestor_buffer",
                requirements,
                location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            }).map_err(|e: gpu_allocator::AllocationError| NodeStorError::VulkanError(e.to_string()))?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            Ok(Self {
                size,
                usage,
                handle: Some(buffer),
                allocation: Some(allocation),
                data: Vec::new(),
            })
        }
    }

    pub fn new_storage(size: usize) -> Self {
        Self { size, usage: GpuBufferUsage::Storage, handle: None, allocation: None, data: vec![0u8; size] }
    }

    pub fn from_cpu_data(data: Vec<u8>) -> Self {
        let size = data.len();
        Self { size, usage: GpuBufferUsage::Staging, handle: None, allocation: None, data }
    }

    pub fn copy_from_slice(&mut self, src: &[u8]) -> Result<(), NodeStorError> {
        if let Some(alloc) = &self.allocation {
            let mapped_ptr = alloc.mapped_ptr().ok_or_else(|| NodeStorError::VulkanError("Buffer não mapeado".into()))?;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), mapped_ptr.as_ptr() as *mut u8, src.len().min(self.size));
            }
            Ok(())
        } else {
            let len = src.len().min(self.data.len());
            self.data[..len].copy_from_slice(&src[..len]);
            Ok(())
        }
    }

    pub fn as_bytes(&self) -> &[u8] { &self.data }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] { &mut self.data }

    pub fn as_f32_slice(&self) -> &[f32] {
        if let Some(alloc) = &self.allocation {
            if let Some(ptr) = alloc.mapped_ptr() {
                unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, self.size / 4) }
            } else {
                &[]
            }
        } else {
            let ptr = self.data.as_ptr() as *const f32;
            let len = self.data.len() / 4;
            unsafe { std::slice::from_raw_parts(ptr, len) }
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuBuffer {{ size: {} bytes, usage: {:?} }}", self.size, self.usage)
    }
}
