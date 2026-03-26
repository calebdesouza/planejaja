use crate::buffer::{GpuBuffer, GpuBufferUsage};
use crate::error::VulkanError;
use nodestor_core::{GpuCapabilities, NodeStorError};

pub struct VulkanContext {
    #[allow(dead_code)]
    pub(crate) entry: Option<ash::Entry>,
    #[allow(dead_code)]
    pub(crate) instance: Option<ash::Instance>,
    pub(crate) device: Option<ash::Device>,
    pub(crate) queue: Option<ash::vk::Queue>,
    pub(crate) queue_family_index: u32,
    pub(crate) allocator: Option<std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>>,
    pub(crate) capabilities: GpuCapabilities,
    pub(crate) device_name: String,
    pub(crate) vulkan_available: bool,
}

impl VulkanContext {
    pub fn device_name(&self) -> &str { &self.device_name }
    pub fn capabilities(&self) -> &GpuCapabilities { &self.capabilities }
    pub fn device(&self) -> Option<&ash::Device> { self.device.as_ref() }
    
    pub fn new(gpu_hint: Option<&GpuCapabilities>) -> Result<Self, VulkanError> {
        match try_init_vulkan(gpu_hint) {
            Ok(ctx) => Ok(ctx),
            Err(_) => Ok(Self::simulation_context()),
        }
    }

    pub fn upload_to_gpu(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        if !self.vulkan_available {
            return Ok(GpuBuffer::from_cpu_data(data.to_vec()));
        }
        let mut buf = GpuBuffer::allocate(self, data.len(), GpuBufferUsage::Staging)?;
        buf.copy_from_slice(data)?;
        Ok(buf)
    }

    pub fn alloc_gpu_buffer(&self, size: usize) -> Result<GpuBuffer, NodeStorError> {
        if !self.vulkan_available {
            return Ok(GpuBuffer::new_storage(size));
        }
        GpuBuffer::allocate(self, size, GpuBufferUsage::Storage)
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
            allocator: None,
            vulkan_available: false,
            capabilities: GpuCapabilities {
                vendor: nodestor_core::GpuVendor::Unknown,
                device_name: "CPU Simulation".to_string(),
                vram_bytes: 0,
                supports_vulkan_compute: false,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
            },
            device_name: "CPU Simulation".to_string(),
        }
    }
}

pub fn probe_physical_devices() -> Result<Vec<GpuCapabilities>, VulkanError> {
    unsafe {
        let entry = match ash::Entry::load() {
            Ok(e) => e,
            Err(_) => return Ok(Vec::new()),
        };
        let app_info = ash::vk::ApplicationInfo::default().api_version(ash::vk::make_api_version(0, 1, 3, 0));
        let instance = match entry.create_instance(&ash::vk::InstanceCreateInfo::default().application_info(&app_info), None) {
            Ok(i) => i,
            Err(_) => return Ok(Vec::new()),
        };

        let pdevices = instance.enumerate_physical_devices().map_err(|_| VulkanError::NoCompatibleDevice)?;
        let mut results = Vec::new();
        for pdevice in pdevices {
            let props = instance.get_physical_device_properties(pdevice);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            results.push(GpuCapabilities {
                vendor: nodestor_core::GpuVendor::Unknown,
                device_name: name,
                vram_bytes: 0, 
                supports_vulkan_compute: true,
                supports_cooperative_matrix2: false,
                supports_cooperative_matrix_khr: false,
                supports_bfloat16: false,
            });
        }
        Ok(results)
    }
}

fn try_init_vulkan(gpu_hint: Option<&GpuCapabilities>) -> Result<VulkanContext, VulkanError> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| VulkanError::InstanceCreation(e.to_string()))?;
        let app_info = ash::vk::ApplicationInfo::default().api_version(ash::vk::make_api_version(0, 1, 3, 0));
        let instance = entry.create_instance(&ash::vk::InstanceCreateInfo::default().application_info(&app_info), None)
            .map_err(|e| VulkanError::InstanceCreation(e.to_string()))?;

        let pdevices = instance.enumerate_physical_devices().map_err(|_| VulkanError::NoCompatibleDevice)?;
        for pdevice in pdevices {
            let props = instance.get_physical_device_properties(pdevice);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            if gpu_hint.map_or(true, |h| name.contains(&h.device_name)) {
                let queue_props = instance.get_physical_device_queue_family_properties(pdevice);
                let q_index = queue_props.iter().enumerate()
                    .find(|(_, p)| p.queue_flags.contains(ash::vk::QueueFlags::COMPUTE))
                    .map(|(i, _)| i as u32)
                    .ok_or(VulkanError::NoCompatibleDevice)?;

                let device = instance.create_device(pdevice, &ash::vk::DeviceCreateInfo::default().queue_create_infos(&[ash::vk::DeviceQueueCreateInfo::default().queue_family_index(q_index).queue_priorities(&[1.0])]), None)
                    .map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

                let allocator = gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                    instance: instance.clone(),
                    device: device.clone(),
                    physical_device: pdevice,
                    debug_settings: Default::default(),
                    buffer_device_address: false,
                    allocation_sizes: Default::default(),
                }).map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

                return Ok(VulkanContext {
                    entry: Some(entry),
                    instance: Some(instance),
                    device: Some(device.clone()),
                    queue: Some(device.get_device_queue(q_index, 0)),
                    queue_family_index: q_index,
                    allocator: Some(std::sync::Arc::new(std::sync::Mutex::new(allocator))),
                    vulkan_available: true,
                    capabilities: GpuCapabilities {
                        vendor: nodestor_core::GpuVendor::Unknown,
                        device_name: name.clone().into_owned(),
                        vram_bytes: 0,
                        supports_vulkan_compute: true,
                        supports_cooperative_matrix2: false,
                        supports_cooperative_matrix_khr: false,
                        supports_bfloat16: false,
                    },
                    device_name: name.into_owned(),
                });
            }
        }
    }
    Err(VulkanError::NoCompatibleDevice)
}
