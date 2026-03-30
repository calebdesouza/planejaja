//! Compute pipelines Vulkan para operações de tensor.

use crate::{
    buffer::GpuBuffer,
    error::VulkanError,
    instance::VulkanContext,
    shader_loader::{self, ShaderKind},
};
use nodestor_core::NodeStorError;
use std::collections::HashMap;

/// Tipo de pipeline disponível no motor Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineKind {
    DequantQ4,
    DequantQ8,
    Matmul,
    CosineSim,
    Lossless,
    GDeflate,
}

/// Compute pipeline encapsulando um shader e seus recursos.
pub struct ComputePipeline {
    #[allow(dead_code)]
    pub(crate) kind: PipelineKind,
    #[allow(dead_code)]
    pub(crate) shader_module: Option<ash::vk::ShaderModule>,
    pub(crate) descriptor_set_layout: Option<ash::vk::DescriptorSetLayout>,
    pub(crate) pipeline_layout: Option<ash::vk::PipelineLayout>,
    pub(crate) pipeline: Option<ash::vk::Pipeline>,
    vulkan_active: bool,
}

impl ComputePipeline {
    pub fn new_simulation(kind: PipelineKind) -> Self {
        Self {
            kind,
            shader_module: None,
            descriptor_set_layout: None,
            pipeline_layout: None,
            pipeline: None,
            vulkan_active: false,
        }
    }

    pub fn new_real(
        ctx: &VulkanContext,
        kind: PipelineKind,
        spirv_bytecode: &[u8],
    ) -> Result<Self, VulkanError> {
        let device = ctx.device.as_ref().ok_or(VulkanError::NoCompatibleDevice)?;
        
        let shader_code = unsafe {
            std::slice::from_raw_parts(
                spirv_bytecode.as_ptr() as *const u32,
                spirv_bytecode.len() / 4,
            )
        };
        let shader_info = ash::vk::ShaderModuleCreateInfo::default().code(shader_code);

        unsafe {
            let shader_module = device.create_shader_module(&shader_info, None)
                .map_err(|e: ash::vk::Result| VulkanError::InvalidShader(e.to_string()))?;

            let bindings = match kind {
                PipelineKind::Matmul | PipelineKind::CosineSim => vec![
                    ash::vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                ],
                _ => vec![
                    ash::vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                ],
            };

            let layout_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)
                .map_err(|e: ash::vk::Result| VulkanError::DeviceCreation(e.to_string()))?;

            let layouts = [descriptor_set_layout];
            let push_constant_size = if kind == PipelineKind::GDeflate { 16 } else { 12 };
            let push_constant_ranges = [ash::vk::PushConstantRange::default()
                .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_size)];
            
            let pipeline_layout_info = ash::vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_ranges);

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e: ash::vk::Result| VulkanError::DeviceCreation(e.to_string()))?;

            let shader_entry_name = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();
            let stage_info = ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(shader_entry_name);

            let pipeline_info = ash::vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipelines = device.create_compute_pipelines(ash::vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e): (Vec<ash::vk::Pipeline>, ash::vk::Result)| VulkanError::DeviceCreation(e.to_string()))?;

            Ok(Self {
                kind,
                shader_module: Some(shader_module),
                descriptor_set_layout: Some(descriptor_set_layout),
                pipeline_layout: Some(pipeline_layout),
                pipeline: Some(pipelines[0]),
                vulkan_active: true,
            })
        }
    }

    pub fn dispatch(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        element_count: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            let copy_len = input.size.min(output.size);
            output.as_mut_bytes()[..copy_len].copy_from_slice(&input.as_bytes()[..copy_len]);
            return Ok(());
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let pool_sizes = [ash::vk::DescriptorPoolSize::default().ty(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
            let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let cmd_pool = device.create_command_pool(&cmd_pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_bufs = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [element_count, 0, 0];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, (element_count + 31) / 32, 1, 1);
            device.end_command_buffer(cmd_buf).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.destroy_command_pool(cmd_pool, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
        }
        Ok(())
    }

    /// Despacha uma micro-fatia do stream "Líquido" de forma assíncrona.
    ///
    /// Retorna uma Fence que sinaliza quando a expansão na GPU terminou.
    pub fn dispatch_liquid(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        element_count: u32,
    ) -> Result<ash::vk::Fence, NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("Vulkan inativo para dispatch líquido".into()));
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            
            // Reutilizamos a lógica de Descriptor Pool/Set simplificada para o MVP
            let pool_sizes = [ash::vk::DescriptorPoolSize::default().ty(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
            let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let cmd_pool = device.create_command_pool(&cmd_pool_info, None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_bufs = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [element_count, 0, 0];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, (element_count + 255) / 256, 1, 1);
            device.end_command_buffer(cmd_buf).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let fence = device.create_fence(&ash::vk::FenceCreateInfo::default(), None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], fence).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            Ok(fence)
        }
    }
    pub fn dispatch_gdeflate(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        tile_offset: u32,
        compressed_size: u32,
        uncompressed_size: u32,
        output_offset: u32,
    ) -> Result<ash::vk::Fence, NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("Cannot run GDeflate CPU fallback via pipeline, use nodestor_gdeflate instead".into()));
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            
            let pool_sizes = [ash::vk::DescriptorPoolSize::default().ty(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
            let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let cmd_pool = device.create_command_pool(&cmd_pool_info, None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_bufs = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            // push_constants: 4 u32 (16 bytes) conform mapping no shader glsl
            let constants = [tile_offset, compressed_size, uncompressed_size, output_offset];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 16);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            // GDeflate usa 32 threads por bloco = 1 wavefront. No shader: layout(local_size_x = 32) in
            // Uma chamada de compute processa 1 tile inteiro de 64KB usando 32 sub-streams simultâneos.
            // Para N tiles, chamaríamos dispatch(N, 1, 1). Aqui despachamos 1 para a abstração tile-a-tile.
            device.cmd_dispatch(cmd_buf, 1, 1, 1);
            
            device.end_command_buffer(cmd_buf).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let fence = device.create_fence(&ash::vk::FenceCreateInfo::default(), None).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], fence).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            Ok(fence)
        }
    }

    pub fn dispatch_matmul(
        &self,
        ctx: &VulkanContext,
        a: &GpuBuffer,
        b: &GpuBuffer,
        output: &mut GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            cpu_matmul_f32(a.as_f32_slice(), b.as_f32_slice(), output, m as usize, k as usize, n as usize);
            return Ok(());
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let pool_sizes = [ash::vk::DescriptorPoolSize::default().ty(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
            let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_a = [ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(a.size as u64)];
            let b_b = [ash::vk::DescriptorBufferInfo::default().buffer(b.handle.unwrap()).offset(0).range(b.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_a),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_b),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let cmd_pool = device.create_command_pool(&cmd_pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_bufs = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [m, k, n];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, (n + 15) / 16, (m + 15) / 16, 1);
            device.end_command_buffer(cmd_buf).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.destroy_command_pool(cmd_pool, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
        }
        Ok(())
    }

    pub fn dispatch_cosine(
        &self,
        ctx: &VulkanContext,
        query: &GpuBuffer,
        candidates: &GpuBuffer,
        scores: &mut GpuBuffer,
        num_candidates: u32,
        dim: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            cpu_cosine_batch(query.as_f32_slice(), candidates.as_f32_slice(), scores, num_candidates as usize, dim as usize);
            return Ok(());
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let pool_sizes = [ash::vk::DescriptorPoolSize::default().ty(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(3)];
            let pool_info = ash::vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
            let descriptor_pool = device.create_descriptor_pool(&pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_q = [ash::vk::DescriptorBufferInfo::default().buffer(query.handle.unwrap()).offset(0).range(query.size as u64)];
            let b_c = [ash::vk::DescriptorBufferInfo::default().buffer(candidates.handle.unwrap()).offset(0).range(candidates.size as u64)];
            let b_s = [ash::vk::DescriptorBufferInfo::default().buffer(scores.handle.unwrap()).offset(0).range(scores.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_q),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_c),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_s),
            ], &[]);

            let cmd_pool_info = ash::vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
            let cmd_pool = device.create_command_pool(&cmd_pool_info, None).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_bufs = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [dim, num_candidates, 0];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, (num_candidates + 31) / 32, 1, 1);
            device.end_command_buffer(cmd_buf).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.destroy_command_pool(cmd_pool, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
        }
        Ok(())
    }
}

pub fn create_all_pipelines(
    ctx: &VulkanContext,
) -> Result<HashMap<PipelineKind, ComputePipeline>, VulkanError> {
    let mut map = HashMap::new();
    if !ctx.vulkan_available {
        map.insert(PipelineKind::DequantQ4, ComputePipeline::new_simulation(PipelineKind::DequantQ4));
        map.insert(PipelineKind::DequantQ8, ComputePipeline::new_simulation(PipelineKind::DequantQ8));
        map.insert(PipelineKind::Matmul, ComputePipeline::new_simulation(PipelineKind::Matmul));
        map.insert(PipelineKind::CosineSim, ComputePipeline::new_simulation(PipelineKind::CosineSim));
        map.insert(PipelineKind::Lossless, ComputePipeline::new_simulation(PipelineKind::Lossless));
        map.insert(PipelineKind::GDeflate, ComputePipeline::new_simulation(PipelineKind::GDeflate));
        return Ok(map);
    }

    let shaders = shader_loader::load_all_shaders();
    for shader in shaders {
        let kind = match shader.kind {
            ShaderKind::DequantQ4 => PipelineKind::DequantQ4,
            ShaderKind::DequantQ8 => PipelineKind::DequantQ8,
            ShaderKind::Matmul => PipelineKind::Matmul,
            ShaderKind::CosineSim => PipelineKind::CosineSim,
            ShaderKind::Lossless => PipelineKind::Lossless,
            ShaderKind::GDeflate => PipelineKind::GDeflate,
        };

        let pipeline = ComputePipeline::new_real(ctx, kind, &shader.bytecode)?;
        map.insert(kind, pipeline);
    }
    Ok(map)
}

fn cpu_matmul_f32(a: &[f32], b: &[f32], output: &mut GpuBuffer, m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            let bytes = sum.to_le_bytes();
            let out_idx = (i * n + j) * 4;
            output.as_mut_bytes()[out_idx..out_idx + 4].copy_from_slice(&bytes);
        }
    }
}

fn cpu_cosine_batch(query: &[f32], candidates: &[f32], scores: &mut GpuBuffer, num_candidates: usize, dim: usize) {
    let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    for c in 0..num_candidates {
        let cand = &candidates[c * dim..(c + 1) * dim];
        let dot: f32 = query.iter().zip(cand.iter()).map(|(a, b)| a * b).sum();
        let c_norm = cand.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = if q_norm > 0.0 && c_norm > 0.0 { dot / (q_norm * c_norm) } else { 0.0 };
        scores.as_mut_bytes()[c * 4..(c + 1) * 4].copy_from_slice(&sim.to_le_bytes());
    }
}
