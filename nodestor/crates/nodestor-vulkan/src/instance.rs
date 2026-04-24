use crate::buffer::{GpuBuffer, GpuBufferUsage, MemoryPath};
use crate::error::VulkanError;
use nodestor_core::{GpuCapabilities, NodeStorError};
use tracing::{info, warn};

/// Contexto Vulkan completo — inicializado uma vez, reutilizado durante toda a sessão.
pub struct VulkanContext {
    #[allow(dead_code)]
    pub(crate) entry: Option<ash::Entry>,
    #[allow(dead_code)]
    pub(crate) instance: Option<ash::Instance>,
    pub(crate) device: Option<ash::Device>,
    /// Fila de compute principal (matmul, rmsnorm, rope, attention, etc.)
    pub(crate) queue: Option<ash::vk::Queue>,
    pub(crate) queue_family_index: u32,
    /// Fila de transferência dedicada (async upload/download, sem bloquear compute).
    /// `None` se a GPU não expor fila dedicada — neste caso usa `queue` para tudo.
    pub(crate) transfer_queue: Option<ash::vk::Queue>,
    pub(crate) transfer_queue_family: u32,
    pub(crate) allocator: Option<std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>>,
    pub(crate) command_pool: Option<ash::vk::CommandPool>,
    pub(crate) descriptor_pool: Option<ash::vk::DescriptorPool>,
    pub(crate) capabilities: GpuCapabilities,
    pub(crate) device_name: String,
    pub(crate) vulkan_available: bool,
    /// `true` se a GPU expõe heap DEVICE_LOCAL + HOST_VISIBLE com ≥ 80% da VRAM total.
    /// Detectado automaticamente via VkPhysicalDeviceMemoryProperties — sem configuração.
    pub(crate) has_rebar: bool,
    /// `true` se há uma fila de transferência separada da fila de compute.
    /// Permite overlap DMA + Compute para máximo throughput.
    pub(crate) has_dedicated_transfer_queue: bool,
}

impl VulkanContext {
    pub fn device_name(&self) -> &str { &self.device_name }
    pub fn capabilities(&self) -> &GpuCapabilities { &self.capabilities }
    pub fn device(&self) -> Option<&ash::Device> { self.device.as_ref() }
    pub fn has_rebar(&self) -> bool { self.has_rebar }
    pub fn has_dedicated_transfer_queue(&self) -> bool { self.has_dedicated_transfer_queue }

    pub fn new(gpu_hint: Option<&GpuCapabilities>) -> Result<Self, VulkanError> {
        match try_init_vulkan(gpu_hint) {
            Ok(ctx) => {
                info!(
                    "Vulkan inicializado: {} | ReBAR={} | TransferQueue={} | Via Expressa: {}",
                    ctx.device_name,
                    ctx.has_rebar,
                    ctx.has_dedicated_transfer_queue,
                    if ctx.has_rebar { "Caminho A (~14 GB/s)" }
                    else { "Caminho B Pinned DMA (~6-8 GB/s)" }
                );
                Ok(ctx)
            }
            Err(e) => {
                warn!("Vulkan indisponível ({}), usando simulação CPU", e);
                Ok(Self::simulation_context())
            }
        }
    }

    /// Upload clássico: aloca staging buffer e copia dados para a GPU.
    pub fn upload_to_gpu(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        if !self.vulkan_available {
            return Ok(GpuBuffer::from_cpu_data(data.to_vec()));
        }
        let mut buf = GpuBuffer::allocate(self, data.len(), GpuBufferUsage::Staging)?;
        buf.copy_from_slice(data)?;
        Ok(buf)
    }

    /// Upload via Via Expressa: usa Triple-Path Allocator para máximo throughput.
    /// Retorna o buffer já com o dado escrito via Write-Combining.
    pub fn upload_pinned(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        if !self.vulkan_available {
            return Ok(GpuBuffer::from_cpu_data(data.to_vec()));
        }
        let mut buf = GpuBuffer::allocate_pinned(self, data.len())?;
        buf.write_direct(data)?;
        Ok(buf)
    }

    /// Aloca buffer de armazenamento na GPU (compute apenas, sem acesso host).
    pub fn alloc_gpu_buffer(&self, size: usize) -> Result<GpuBuffer, NodeStorError> {
        if !self.vulkan_available {
            return Ok(GpuBuffer::new_storage(size));
        }
        GpuBuffer::allocate(self, size, GpuBufferUsage::Storage)
    }

    /// Aloca buffer Pinned para streaming do SSD → GPU (Via Expressa).
    /// Usa o melhor caminho disponível sem configuração manual.
    pub fn alloc_pinned_buffer(&self, size: usize) -> Result<GpuBuffer, NodeStorError> {
        GpuBuffer::allocate_pinned(self, size)
    }

    pub fn download_from_gpu(&self, buffer: &GpuBuffer) -> Result<Vec<f32>, NodeStorError> {
        if !self.vulkan_available {
            return Ok(buffer.as_f32_slice().to_vec());
        }
        Ok(buffer.as_f32_slice().to_vec())
    }

    fn simulation_context() -> Self {
        Self {
            entry: None,
            instance: None,
            device: None,
            queue: None,
            queue_family_index: 0,
            transfer_queue: None,
            transfer_queue_family: 0,
            allocator: None,
            command_pool: None,
            descriptor_pool: None,
            vulkan_available: false,
            has_rebar: false,
            has_dedicated_transfer_queue: false,
            capabilities: GpuCapabilities::default(),
            device_name: "CPU Simulation".to_string(),
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        if let Some(device) = &self.device {
            unsafe {
                if let Some(pool) = self.command_pool {
                    device.destroy_command_pool(pool, None);
                }
                if let Some(pool) = self.descriptor_pool {
                    device.destroy_descriptor_pool(pool, None);
                }
                device.destroy_device(None);
            }
        }
        if let Some(instance) = &self.instance {
            unsafe { instance.destroy_instance(None); }
        }
    }
}

/// Detecta se a GPU tem ReBAR ativo via VkPhysicalDeviceMemoryProperties.
///
/// Algoritmo:
/// 1. Soma o tamanho de todos os heaps DEVICE_LOCAL → `total_vram`
/// 2. Procura tipos de memória com flags DEVICE_LOCAL | HOST_VISIBLE
/// 3. Se o heap desse tipo tem tamanho ≥ 80% de `total_vram` → ReBAR está ativo
///
/// Zero BIOS. Zero configuração. Detectado puramente via API Vulkan.
fn detect_rebar(mem_props: &ash::vk::PhysicalDeviceMemoryProperties) -> bool {
    let mut total_device_local: u64 = 0;
    for heap in mem_props.memory_heaps.iter() {
        if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total_device_local = total_device_local.saturating_add(heap.size);
        }
    }
    if total_device_local == 0 { return false; }

    for i in 0..mem_props.memory_type_count as usize {
        let mt = mem_props.memory_types[i];
        let flags = mt.property_flags;
        if flags.contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && flags.contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            let heap_size = mem_props.memory_heaps[mt.heap_index as usize].size;
            // ≥ 80% do total DEVICE_LOCAL = ReBAR pleno (não apenas o pequeno BAR de 256 MB)
            if heap_size >= total_device_local * 80 / 100 {
                return true;
            }
        }
    }
    false
}

/// Descobre a família de filas de transferência dedicada (separada do compute).
///
/// Uma fila dedicada de transferência permite que a GPU realize DMA (SSD→VRAM)
/// **em paralelo** com o processamento de shaders, sem serialização.
///
/// Fallback: retorna o índice da fila de compute se não há família dedicada.
fn find_transfer_queue_family(
    queue_props: &[ash::vk::QueueFamilyProperties],
    compute_family: u32,
) -> (u32, bool) {
    // Primeiro: procura família que tem TRANSFER mas NÃO tem COMPUTE (dedicada pura)
    for (i, props) in queue_props.iter().enumerate() {
        let i = i as u32;
        if i == compute_family { continue; }
        if props.queue_flags.contains(ash::vk::QueueFlags::TRANSFER)
            && !props.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
        {
            return (i, true); // Dedicada: máximo overlap
        }
    }

    // Segundo: procura qualquer família com TRANSFER diferente da de compute
    for (i, props) in queue_props.iter().enumerate() {
        let i = i as u32;
        if i == compute_family { continue; }
        if props.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) {
            return (i, true); // Separada mas não pura: ainda permite overlap parcial
        }
    }

    // Fallback: mesma família, mesmo queue (sem overlap — mas funciona em TUDO)
    (compute_family, false)
}

pub fn probe_physical_devices() -> Result<Vec<GpuCapabilities>, VulkanError> {
    unsafe {
        let entry = match ash::Entry::load() {
            Ok(e) => e,
            Err(_) => return Ok(Vec::new()),
        };
        let app_info = ash::vk::ApplicationInfo::default()
            .api_version(ash::vk::make_api_version(0, 1, 3, 0));
        let instance = match entry.create_instance(
            &ash::vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        ) {
            Ok(i) => i,
            Err(_) => return Ok(Vec::new()),
        };

        let pdevices = instance.enumerate_physical_devices()
            .map_err(|_| VulkanError::NoCompatibleDevice)?;
        let mut results = Vec::new();

        for pdevice in pdevices {
            let props = instance.get_physical_device_properties(pdevice);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy().into_owned();

            let mem_props = instance.get_physical_device_memory_properties(pdevice);
            let rebar_enabled = detect_rebar(&mem_props);

            let mut vram_bytes: u64 = 0;
            for heap in mem_props.memory_heaps.iter() {
                if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    vram_bytes = vram_bytes.saturating_add(heap.size);
                }
            }

            results.push(GpuCapabilities {
                vendor: match props.vendor_id {
                    0x10DE => nodestor_core::GpuVendor::Nvidia,
                    0x1002 => nodestor_core::GpuVendor::Amd,
                    0x8086 => nodestor_core::GpuVendor::Intel,
                    _ => nodestor_core::GpuVendor::Unknown,
                },
                device_name: name,
                vram_bytes,
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
                pcie_gen: 0,
                pcie_lanes: 0,
                resizable_bar_enabled: rebar_enabled,
                driver_version: format!("{}.{}.{}",
                    (props.driver_version >> 22) & 0x3FF,
                    (props.driver_version >> 14) & 0xFF,
                    props.driver_version & 0x3FFF,
                ),
            });
        }
        Ok(results)
    }
}

fn try_init_vulkan(gpu_hint: Option<&GpuCapabilities>) -> Result<VulkanContext, VulkanError> {
    unsafe {
        let entry = ash::Entry::load()
            .map_err(|e| VulkanError::InstanceCreation(e.to_string()))?;
        let app_info = ash::vk::ApplicationInfo::default()
            .api_version(ash::vk::make_api_version(0, 1, 3, 0));
        let instance = entry.create_instance(
            &ash::vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        ).map_err(|e| VulkanError::InstanceCreation(e.to_string()))?;

        let pdevices = instance.enumerate_physical_devices()
            .map_err(|_| VulkanError::NoCompatibleDevice)?;

        for pdevice in pdevices {
            let props = instance.get_physical_device_properties(pdevice);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();

            if gpu_hint.map_or(true, |h| name.contains(&h.device_name)) {
                let queue_props = instance.get_physical_device_queue_family_properties(pdevice);

                // Fila de Compute principal
                let compute_family = queue_props.iter().enumerate()
                    .find(|(_, p)| p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE))
                    .map(|(i, _)| i as u32)
                    .ok_or(VulkanError::NoCompatibleDevice)?;

                // Fila de Transfer dedicada (Async DMA)
                let (transfer_family, has_dedicated_transfer) =
                    find_transfer_queue_family(&queue_props, compute_family);

                // Cria o device com ambas as filas (se diferentes)
                let queue_create_infos = if has_dedicated_transfer && transfer_family != compute_family {
                    vec![
                        ash::vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(compute_family)
                            .queue_priorities(&[1.0]),
                        ash::vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(transfer_family)
                            .queue_priorities(&[1.0]),
                    ]
                } else {
                    vec![
                        ash::vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(compute_family)
                            .queue_priorities(&[1.0]),
                    ]
                };

                let device = instance.create_device(
                    pdevice,
                    &ash::vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_create_infos),
                    None,
                ).map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

                // Obtém handles das filas
                let compute_queue = device.get_device_queue(compute_family, 0);
                let transfer_queue = if has_dedicated_transfer && transfer_family != compute_family {
                    Some(device.get_device_queue(transfer_family, 0))
                } else {
                    None
                };

                // Detecta ReBAR via memória
                let mem_props = instance.get_physical_device_memory_properties(pdevice);
                let has_rebar = detect_rebar(&mem_props);

                // Calcula VRAM total
                let vram_bytes: u64 = mem_props.memory_heaps.iter()
                    .filter(|h| h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|h| h.size)
                    .sum();

                let allocator = gpu_allocator::vulkan::Allocator::new(
                    &gpu_allocator::vulkan::AllocatorCreateDesc {
                        instance: instance.clone(),
                        device: device.clone(),
                        physical_device: pdevice,
                        debug_settings: Default::default(),
                        buffer_device_address: false,
                        allocation_sizes: Default::default(),
                    }
                ).map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

                // Command Pool persistente com RESET_COMMAND_BUFFER
                let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(compute_family)
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                let command_pool = device.create_command_pool(&cmd_pool_info, None)
                    .map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

                // Descriptor Pool ampliado (256 sets, 2048 buffers)
                let pool_sizes = [
                    ash::vk::DescriptorPoolSize::default()
                        .ty(ash::vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(2048)
                ];
                let descriptor_pool_info = ash::vk::DescriptorPoolCreateInfo::default()
                    .max_sets(256)
                    .flags(ash::vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&pool_sizes);
                let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)
                    .map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;


                let driver_ver = format!("{}.{}.{}",
                    (props.driver_version >> 22) & 0x3FF,
                    (props.driver_version >> 14) & 0xFF,
                    props.driver_version & 0x3FFF,
                );

                return Ok(VulkanContext {
                    entry: Some(entry),
                    instance: Some(instance),
                    device: Some(device.clone()),
                    queue: Some(compute_queue),
                    queue_family_index: compute_family,
                    transfer_queue,
                    transfer_queue_family: transfer_family,
                    allocator: Some(std::sync::Arc::new(std::sync::Mutex::new(allocator))),
                    command_pool: Some(command_pool),
                    descriptor_pool: Some(descriptor_pool),
                    vulkan_available: true,
                    has_rebar,
                    has_dedicated_transfer_queue: has_dedicated_transfer,
                    capabilities: GpuCapabilities {
                        vendor: match props.vendor_id {
                            0x10DE => nodestor_core::GpuVendor::Nvidia,
                            0x1002 => nodestor_core::GpuVendor::Amd,
                            0x8086 => nodestor_core::GpuVendor::Intel,
                            _ => nodestor_core::GpuVendor::Unknown,
                        },
                        device_name: name.into_owned(),
                        vram_bytes,
                        supports_vulkan_compute: true,
                        supports_cooperative_matrix2: false,
                        supports_cooperative_matrix_khr: false,
                        supports_bfloat16: false,
                        pcie_gen: 0,
                        pcie_lanes: 0,
                        resizable_bar_enabled: has_rebar,
                        driver_version: driver_ver,
                    },
                    device_name: "NodeStor GPU Engine".to_string(),
                });
            }
        }
    }
    Err(VulkanError::NoCompatibleDevice)
}
