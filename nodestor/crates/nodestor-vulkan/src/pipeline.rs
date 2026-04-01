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
    MatmulQ4,
    MatmulTensorCore,
    RmsNorm,
    RoPe,
    SiLu,
    Softmax,
    Attention,
    /// Cooperative Matrix (WMMA/Tensor Core via GL_KHR_cooperative_matrix).
    /// Disponibilizado automaticamente se o hardware suportar.
    /// Fallback transparente para `Matmul` se não suportado.
    CoopMatrix,
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
                PipelineKind::Matmul | PipelineKind::CosineSim | PipelineKind::MatmulQ4 | PipelineKind::MatmulTensorCore | PipelineKind::RmsNorm | PipelineKind::RoPe | PipelineKind::SiLu | PipelineKind::Softmax | PipelineKind::Attention => vec![
                    ash::vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
                    ash::vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(ash::vk::ShaderStageFlags::COMPUTE),
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
            let push_constant_size = match kind {
                PipelineKind::GDeflate => 16,
                PipelineKind::RoPe => 24,
                _ => 12,
            };
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
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

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

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
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

            // Cleanup local items
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
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
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

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

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
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
            
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

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

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
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

    /// **Fase 2 — GDeflate + Matmul num único CommandBuffer (fusão inline).**
    ///
    /// Elimina o segundo `vkQueueSubmit` e a cópia intermediária.
    /// Um Memory Barrier entre os dois shaders garante ordem correta sem stall.
    pub fn dispatch_fused_decompress_matmul(
        &self,
        ctx: &VulkanContext,
        gdeflate_pipeline: &ComputePipeline,
        matmul_pipeline: &ComputePipeline,
        compressed_input: &GpuBuffer,
        decompressed_buf: &mut GpuBuffer,
        matmul_b: &GpuBuffer,
        matmul_out: &mut GpuBuffer,
        decomp_tiles: u32,
        compressed_size: u32,
        uncompressed_size: u32,
        m: u32, k: u32, n: u32,
    ) -> Result<ash::vk::Fence, NodeStorError> {
        if !self.vulkan_active || !gdeflate_pipeline.vulkan_active || !matmul_pipeline.vulkan_active {
            return Ok(ash::vk::Fence::null());
        }

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            // DescriptorSet GDeflate
            let gd_layouts = [gdeflate_pipeline.descriptor_set_layout.unwrap()];
            let gd_sets = device.allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&gd_layouts)
            ).map_err(|e| NodeStorError::VulkanError(format!("FusedGD: {}", e)))?;
            let gd_set = gd_sets[0];
            let b_comp = [ash::vk::DescriptorBufferInfo::default().buffer(compressed_input.handle.unwrap()).offset(0).range(compressed_input.size as u64)];
            let b_decomp_w = [ash::vk::DescriptorBufferInfo::default().buffer(decompressed_buf.handle.unwrap()).offset(0).range(decompressed_buf.size as u64)];
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(gd_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_comp),
                ash::vk::WriteDescriptorSet::default().dst_set(gd_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_decomp_w),
            ], &[]);

            // DescriptorSet Matmul
            let mm_layouts = [matmul_pipeline.descriptor_set_layout.unwrap()];
            let mm_sets = device.allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool).set_layouts(&mm_layouts)
            ).map_err(|e| NodeStorError::VulkanError(format!("FusedMM: {}", e)))?;
            let mm_set = mm_sets[0];
            let b_a = [ash::vk::DescriptorBufferInfo::default().buffer(decompressed_buf.handle.unwrap()).offset(0).range(decompressed_buf.size as u64)];
            let b_b = [ash::vk::DescriptorBufferInfo::default().buffer(matmul_b.handle.unwrap()).offset(0).range(matmul_b.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(matmul_out.handle.unwrap()).offset(0).range(matmul_out.size as u64)];
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(mm_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_a),
                ash::vk::WriteDescriptorSet::default().dst_set(mm_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_b),
                ash::vk::WriteDescriptorSet::default().dst_set(mm_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            // CommandBuffer único GDeflate + MemBarrier + Matmul
            let cmd_bufs = device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)
            ).map_err(|e| NodeStorError::VulkanError(format!("FusedCmd: {}", e)))?;
            let cmd = cmd_bufs[0];

            device.begin_command_buffer(cmd, &ash::vk::CommandBufferBeginInfo::default())
                .map_err(|e| NodeStorError::VulkanError(format!("FusedBegin: {}", e)))?;

            // PASSO 1: GDeflate
            device.cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::COMPUTE, gdeflate_pipeline.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd, ash::vk::PipelineBindPoint::COMPUTE, gdeflate_pipeline.pipeline_layout.unwrap(), 0, &[gd_set], &[]);
            let gd_consts = [0u32, compressed_size, uncompressed_size, 0u32];
            device.cmd_push_constants(cmd, gdeflate_pipeline.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0,
                std::slice::from_raw_parts(gd_consts.as_ptr() as *const u8, 16));
            device.cmd_dispatch(cmd, decomp_tiles, 1, 1);

            // MEMORY BARRIER: writes do GDeflate visíveis ao Matmul (Shader Write → Shader Read)
            let barrier = ash::vk::MemoryBarrier2::default()
                .src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(ash::vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(ash::vk::AccessFlags2::SHADER_READ);
            device.cmd_pipeline_barrier2(cmd, &ash::vk::DependencyInfo::default()
                .memory_barriers(std::slice::from_ref(&barrier)));

            // PASSO 2: Matmul usa saída do GDeflate como entrada
            device.cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::COMPUTE, matmul_pipeline.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd, ash::vk::PipelineBindPoint::COMPUTE, matmul_pipeline.pipeline_layout.unwrap(), 0, &[mm_set], &[]);
            let mm_consts = [m, k, n];
            device.cmd_push_constants(cmd, matmul_pipeline.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0,
                std::slice::from_raw_parts(mm_consts.as_ptr() as *const u8, 12));
            device.cmd_dispatch(cmd, (n + 15) / 16, (m + 15) / 16, 1);

            device.end_command_buffer(cmd)
                .map_err(|e| NodeStorError::VulkanError(format!("FusedEnd: {}", e)))?;

            // UM ÚNICO vkQueueSubmit — zero stall entre decompress e compute
            let fence = device.create_fence(&ash::vk::FenceCreateInfo::default(), None)
                .map_err(|e| NodeStorError::VulkanError(format!("FusedFence: {}", e)))?;
            device.queue_submit(ctx.queue.unwrap(),
                &[ash::vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
                .map_err(|e| NodeStorError::VulkanError(format!("FusedSubmit: {}", e)))?;

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
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

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

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
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

            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
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
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

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

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
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

            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dispatch_rmsnorm(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &mut GpuBuffer,
        seq_len: u32,
        hidden_size: u32,
        eps: f32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_set = device.allocate_descriptor_sets(&alloc_info).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_w = [ash::vk::DescriptorBufferInfo::default().buffer(weight.handle.unwrap()).offset(0).range(weight.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_w),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let mut constants = [0u8; 12];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&hidden_size.to_le_bytes());
            constants[8..12].copy_from_slice(&eps.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            device.cmd_dispatch(cmd_buf, 1, seq_len, 1);
            device.end_command_buffer(cmd_buf).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dispatch_silu(&self, ctx: &VulkanContext, input: &GpuBuffer, output: &mut GpuBuffer, elements: u32) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [elements, 0, 0];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, (elements + 255) / 256, 1, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_softmax(&self, ctx: &VulkanContext, buffer: &mut GpuBuffer, seq_len: u32, vocab_size: u32) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(buffer.handle.unwrap()).offset(0).range(buffer.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let constants = [seq_len, vocab_size, 0];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            device.cmd_dispatch(cmd_buf, 1, seq_len, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_rope(
        &self,
        ctx: &VulkanContext,
        q: &mut GpuBuffer,
        k: &mut GpuBuffer,
        seq_len: u32,
        num_heads_q: u32,
        num_heads_k: u32,
        head_dim: u32,
        freq_base: f32,
        start_pos: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];

            let b_q = [ash::vk::DescriptorBufferInfo::default().buffer(q.handle.unwrap()).offset(0).range(q.size as u64)];
            let b_k = [ash::vk::DescriptorBufferInfo::default().buffer(k.handle.unwrap()).offset(0).range(k.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_q),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_k),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let mut constants = [0u8; 24];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&num_heads_q.to_le_bytes());
            constants[8..12].copy_from_slice(&num_heads_k.to_le_bytes());
            constants[12..16].copy_from_slice(&head_dim.to_le_bytes());
            constants[16..20].copy_from_slice(&freq_base.to_le_bytes());
            constants[20..24].copy_from_slice(&start_pos.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            let max_heads = if num_heads_q > num_heads_k { num_heads_q } else { num_heads_k };
            // Local size is 64 per threadgroup (handling 128 elements per head)
            device.cmd_dispatch(cmd_buf, max_heads, seq_len, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_attention(
        &self,
        ctx: &VulkanContext,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        out_attn: &mut GpuBuffer,
        seq_len: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];

            let b_q = [ash::vk::DescriptorBufferInfo::default().buffer(q.handle.unwrap()).offset(0).range(q.size as u64)];
            let b_k = [ash::vk::DescriptorBufferInfo::default().buffer(k.handle.unwrap()).offset(0).range(k.size as u64)];
            let b_v = [ash::vk::DescriptorBufferInfo::default().buffer(v.handle.unwrap()).offset(0).range(v.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(out_attn.handle.unwrap()).offset(0).range(out_attn.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_q),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_k),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_v),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(&ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let mut constants = [0u8; 12];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&head_dim.to_le_bytes());
            constants[8..12].copy_from_slice(&scale.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            device.cmd_dispatch(cmd_buf, 1, seq_len, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
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
        map.insert(PipelineKind::MatmulQ4, ComputePipeline::new_simulation(PipelineKind::MatmulQ4));
        map.insert(PipelineKind::MatmulTensorCore, ComputePipeline::new_simulation(PipelineKind::MatmulTensorCore));
        map.insert(PipelineKind::RmsNorm, ComputePipeline::new_simulation(PipelineKind::RmsNorm));
        map.insert(PipelineKind::RoPe, ComputePipeline::new_simulation(PipelineKind::RoPe));
        map.insert(PipelineKind::SiLu, ComputePipeline::new_simulation(PipelineKind::SiLu));
        map.insert(PipelineKind::Softmax, ComputePipeline::new_simulation(PipelineKind::Softmax));
        map.insert(PipelineKind::Attention, ComputePipeline::new_simulation(PipelineKind::Attention));
        // CoopMatrix: sempre criado como simulação quando Vulkan indisponível
        map.insert(PipelineKind::CoopMatrix, ComputePipeline::new_simulation(PipelineKind::CoopMatrix));
        return Ok(map);
    }

    // Tentativa de carregar Cooperative Matrix (Fase 5).
    // Detecta suporte ao carregar o shader — se o SPIR-V for inválido para este
    // driver (não suporta a extensão), o `new_real` retorna erro e usamos fallback.
    if ctx.capabilities.supports_cooperative_matrix_khr || ctx.capabilities.supports_cooperative_matrix2 {
        let coop_spirv = shader_loader::load_shader_by_kind(ShaderKind::CoopMatrix);
        if let Some(spirv) = coop_spirv {
            match ComputePipeline::new_real(ctx, PipelineKind::CoopMatrix, &spirv) {
                Ok(p) => {
                    tracing::info!("Cooperative Matrix: Ativo (Tensor Core via GL_KHR)");
                    map.insert(PipelineKind::CoopMatrix, p);
                }
                Err(e) => {
                    tracing::warn!("CoopMatrix SPIR-V rejeitado pelo driver: {} — usando Matmul clássico", e);
                    map.insert(PipelineKind::CoopMatrix, ComputePipeline::new_simulation(PipelineKind::CoopMatrix));
                }
            }
        } else {
            tracing::warn!("Shader CoopMatrix não encontrado — usando Matmul clássico");
            map.insert(PipelineKind::CoopMatrix, ComputePipeline::new_simulation(PipelineKind::CoopMatrix));
        }
    } else {
        map.insert(PipelineKind::CoopMatrix, ComputePipeline::new_simulation(PipelineKind::CoopMatrix));
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
            ShaderKind::MatmulQ4 => PipelineKind::MatmulQ4,
            ShaderKind::MatmulTensorCore => PipelineKind::MatmulTensorCore,
            ShaderKind::RmsNorm => PipelineKind::RmsNorm,
            ShaderKind::RoPe => PipelineKind::RoPe,
            ShaderKind::SiLu => PipelineKind::SiLu,
            ShaderKind::Softmax => PipelineKind::Softmax,
            ShaderKind::Attention => PipelineKind::Attention,
            // CoopMatrix é carregado separadamente antes deste loop — não aparece em `all()`
            // mas o match deve ser exaustivo; este arm nunca é atingido em prática.
            ShaderKind::CoopMatrix => PipelineKind::CoopMatrix,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul_identity() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0, 0.0, 0.0, 1.0]; // 2x2 identidade
        let mut out = GpuBuffer::new_storage(4 * 4);
        
        cpu_matmul_f32(&a, &b, &mut out, 2, 2, 2);
        
        let out_slice = out.as_f32_slice();
        assert_eq!(out_slice, &a); 
    }

    #[test]
    fn test_cpu_matmul_known_values() {
        // [[1,2],   [[5,6],      [[19, 22],
        //  [3,4]] x  [7,8]]  =>   [43, 50]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = GpuBuffer::new_storage(4 * 4);
        
        cpu_matmul_f32(&a, &b, &mut out, 2, 2, 2);
        
        let out_slice = out.as_f32_slice();
        assert_eq!(out_slice[0], 19.0);
        assert_eq!(out_slice[1], 22.0);
        assert_eq!(out_slice[2], 43.0);
        assert_eq!(out_slice[3], 50.0);
    }

    #[test]
    fn test_cpu_cosine_orthogonal() {
        let q = [1.0, 0.0];
        let max_cands = 2; // Teste multiplos candidadtos
        // Cand 0 = [0, 1] (ortogonal), Cand 1 = [1, 0] (idêntico)
        let cands = [0.0, 1.0, 1.0, 0.0]; 
        
        let mut scores = GpuBuffer::new_storage(max_cands * 4);
        cpu_cosine_batch(&q, &cands, &mut scores, max_cands, 2);
        
        let score_slice = scores.as_f32_slice();
        assert_eq!(score_slice[0], 0.0); // ortogonal = 0
        assert_eq!(score_slice[1], 1.0); // idêntico = 1
    }
}
