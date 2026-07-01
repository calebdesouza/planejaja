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
    /// Q6_K (6-bit K-quant) — para embeddings residentes na VRAM.
    DequantQ6K,
    /// Q5_0 (5-bit simples) — tensor dtype mais comum no Qwen2.5 GGUF.
    DequantQ5_0,
    Matmul,
    CosineSim,
    Lossless,
    GDeflate,
    MatmulQ4,
    MatmulQ4K,
    MatmulTensorCore,
    MatmulTernary,
    MambaSelectiveScan,
    RmsNorm,
    RoPe,
    SiLu,
    Softmax,
    Attention,
    CoopMatrix,
    ZipGEMM,
    FlashAttention,
    TreeAttention,
    CrossEntropyMaskedBack,
    OutProd,
    OptStepAdam,
    Add,
    Mul,
    TurboQuantAttention,
    MoERouting,
    FusedLayerNormGelu,
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

    pub fn is_gpu_active(&self) -> bool { self.vulkan_active }

    /// Despacha dequantização Q5_0 com parâmetros corretos para o shader.
    ///
    /// O shader `dequant_q5_0.comp` tem `local_size_x=256` e 16 invocações por bloco
    /// (uma por par j=0..15). Push constant = NUM_BLOCKS, não element_count.
    /// Em modo simulação: retorna Err (caller faz fallback CPU).
    pub fn dispatch_q5_0(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        num_blocks: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("dispatch_q5_0: simulation mode".into()));
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default()
                .buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default()
                .buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];

            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(0)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(1)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b_out),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds)
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default())
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);

            // Push constant = NUM_BLOCKS (u32), shader expects pc.NUM_BLOCKS
            let bytes = num_blocks.to_ne_bytes();
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(),
                ash::vk::ShaderStageFlags::COMPUTE, 0, &bytes);

            // local_size_x=256, 16 invocações/bloco → ceil(num_blocks*16 / 256) workgroups
            let workgroups = (num_blocks * 16 + 255) / 256;
            device.cmd_dispatch(cmd_buf, workgroups, 1, 1);
            device.end_command_buffer(cmd_buf)
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(),
                &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])],
                ash::vk::Fence::null())
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap())
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;

            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set])
                .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn new_real(
        ctx: &VulkanContext,
        kind: PipelineKind,
        spirv_bytecode: &[u8],
    ) -> Result<Self, VulkanError> {
        let device = ctx.device.as_ref().ok_or(VulkanError::NoCompatibleDevice)?;
        
        let mut aligned_bytecode = Vec::with_capacity(spirv_bytecode.len() / 4);
        let mut chunks = spirv_bytecode.chunks_exact(4);
        for chunk in &mut chunks {
            aligned_bytecode.push(u32::from_ne_bytes(chunk.try_into().unwrap()));
        }

        let shader_info = ash::vk::ShaderModuleCreateInfo::default().code(&aligned_bytecode);

        unsafe {
            let shader_module = device.create_shader_module(&shader_info, None)

                .map_err(|e: ash::vk::Result| VulkanError::InvalidShader(e.to_string()))?;

            let num_bindings = match kind {
                PipelineKind::Softmax => 1,
                PipelineKind::DequantQ4 | PipelineKind::DequantQ8 | PipelineKind::DequantQ6K
                | PipelineKind::DequantQ5_0
                | PipelineKind::Lossless | PipelineKind::GDeflate | PipelineKind::RoPe | PipelineKind::SiLu => 2,
                PipelineKind::Matmul | PipelineKind::CosineSim | PipelineKind::MatmulQ4
                | PipelineKind::MatmulQ4K
                | PipelineKind::MatmulTensorCore | PipelineKind::MatmulTernary | PipelineKind::RmsNorm | PipelineKind::CoopMatrix
                | PipelineKind::OutProd | PipelineKind::Add | PipelineKind::Mul => 3,
                PipelineKind::Attention | PipelineKind::TurboQuantAttention | PipelineKind::MoERouting 
                | PipelineKind::ZipGEMM | PipelineKind::FlashAttention | PipelineKind::TreeAttention 
                | PipelineKind::CrossEntropyMaskedBack | PipelineKind::OptStepAdam | PipelineKind::FusedLayerNormGelu => 4,
                PipelineKind::MambaSelectiveScan => 7,
            };

            let mut bindings = Vec::new();
            for i in 0..num_bindings {
                bindings.push(
                    ash::vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
                );
            }

            let layout_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)
                .map_err(|e: ash::vk::Result| VulkanError::DeviceCreation(e.to_string()))?;

            let layouts = [descriptor_set_layout];
            let push_constant_size = match kind {
                PipelineKind::GDeflate => 16,
                PipelineKind::RoPe => 24,
                PipelineKind::ZipGEMM => 24,
                PipelineKind::TreeAttention => 28,
                PipelineKind::FlashAttention | PipelineKind::TurboQuantAttention => 16,
                PipelineKind::MoERouting => 20,
                PipelineKind::OptStepAdam => 32,
                PipelineKind::FusedLayerNormGelu => 8,
                PipelineKind::MambaSelectiveScan => 12,
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

    /// Records a matmul dispatch into an EXISTING command buffer (no submit/wait).
    /// Caller must call queue_submit + queue_wait_idle when all ops are recorded.
    /// Appends the allocated DescriptorSet to `ds_collector` for later cleanup.
    pub fn record_matmul_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        output: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("record_matmul_into: simulation pipeline".into()));
        }
        unsafe {
            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let ds = descriptor_sets[0];
            ds_collector.push(ds);

            let b_a   = [ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(a.size as u64)];
            let b_b   = [ash::vk::DescriptorBufferInfo::default().buffer(b.handle.unwrap()).offset(0).range(b.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_a),
                ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_b),
                ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[ds], &[]);
            let constants = [m, k, n];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_dispatch(cmd_buf, (n + 15) / 16, (m + 15) / 16, 1);

            // Memory barrier: writes from this dispatch visible to the next.
            let buf_barrier = ash::vk::BufferMemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::HOST_READ)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .buffer(output.handle.unwrap())
                .offset(0)
                .size(ash::vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                cmd_buf,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::HOST,
                ash::vk::DependencyFlags::empty(),
                &[], &[buf_barrier], &[],
            );
        }
        Ok(())
    }

    /// Generic helper: records any 3-binding compute dispatch into an existing command buffer.
    /// Inserts a COMPUTE→(COMPUTE|HOST) memory barrier on `output_buf` after the dispatch.
    unsafe fn record_3bind_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        b0: &GpuBuffer, b1: &GpuBuffer, b2: &GpuBuffer,
        push_bytes: &[u8],
        dispatch: (u32, u32, u32),
        output_buf: &GpuBuffer,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        let layouts = [self.descriptor_set_layout.unwrap()];
        let ds = device.allocate_descriptor_sets(
            &ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)
        ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];
        ds_collector.push(ds);

        let i0 = [ash::vk::DescriptorBufferInfo::default().buffer(b0.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE)];
        let i1 = [ash::vk::DescriptorBufferInfo::default().buffer(b1.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE)];
        let i2 = [ash::vk::DescriptorBufferInfo::default().buffer(b2.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE)];
        device.update_descriptor_sets(&[
            ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i0),
            ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i1),
            ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i2),
        ], &[]);

        device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
        device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[ds], &[]);
        device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
        device.cmd_dispatch(cmd_buf, dispatch.0, dispatch.1, dispatch.2);

        let barrier = ash::vk::BufferMemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::HOST_READ)
            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .buffer(output_buf.handle.unwrap()).offset(0).size(ash::vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(
            cmd_buf,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::HOST,
            ash::vk::DependencyFlags::empty(),
            &[], &[barrier], &[],
        );
        Ok(())
    }

    /// Generic helper: records any 2-binding compute dispatch into an existing command buffer.
    unsafe fn record_2bind_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        b0: &GpuBuffer, b1: &GpuBuffer,
        push_bytes: &[u8],
        dispatch: (u32, u32, u32),
        output_buf: &GpuBuffer,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        let layouts = [self.descriptor_set_layout.unwrap()];
        let ds = device.allocate_descriptor_sets(
            &ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)
        ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];
        ds_collector.push(ds);

        let i0 = [ash::vk::DescriptorBufferInfo::default().buffer(b0.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE)];
        let i1 = [ash::vk::DescriptorBufferInfo::default().buffer(b1.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE)];
        device.update_descriptor_sets(&[
            ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i0),
            ash::vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&i1),
        ], &[]);

        device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
        device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[ds], &[]);
        device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
        device.cmd_dispatch(cmd_buf, dispatch.0, dispatch.1, dispatch.2);

        let barrier = ash::vk::BufferMemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::HOST_READ)
            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .buffer(output_buf.handle.unwrap()).offset(0).size(ash::vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(
            cmd_buf,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::HOST,
            ash::vk::DependencyFlags::empty(),
            &[], &[barrier], &[],
        );
        Ok(())
    }

    pub fn record_rmsnorm_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        input: &GpuBuffer, weight: &GpuBuffer, output: &GpuBuffer,
        seq_len: u32, hidden_size: u32, eps: f32,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("record_rmsnorm_into: simulation".into()));
        }
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&seq_len.to_le_bytes());
        push[4..8].copy_from_slice(&hidden_size.to_le_bytes());
        push[8..12].copy_from_slice(&eps.to_le_bytes());
        unsafe { self.record_3bind_into(device, descriptor_pool, cmd_buf, input, weight, output, &push, (1, seq_len, 1), output, ds_collector) }
    }

    pub fn record_add_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        a: &GpuBuffer, b: &GpuBuffer, output: &GpuBuffer,
        elems: u32,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("record_add_into: simulation".into()));
        }
        unsafe { self.record_3bind_into(device, descriptor_pool, cmd_buf, a, b, output, &elems.to_le_bytes(), ((elems + 255) / 256, 1, 1), output, ds_collector) }
    }

    pub fn record_mul_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        a: &GpuBuffer, b: &GpuBuffer, output: &GpuBuffer,
        elems: u32,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("record_mul_into: simulation".into()));
        }
        unsafe { self.record_3bind_into(device, descriptor_pool, cmd_buf, a, b, output, &elems.to_le_bytes(), ((elems + 255) / 256, 1, 1), output, ds_collector) }
    }

    pub fn record_silu_into(
        &self,
        device: &ash::Device,
        descriptor_pool: ash::vk::DescriptorPool,
        cmd_buf: ash::vk::CommandBuffer,
        input: &GpuBuffer, output: &GpuBuffer,
        elements: u32,
        ds_collector: &mut Vec<ash::vk::DescriptorSet>,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            return Err(NodeStorError::VulkanError("record_silu_into: simulation".into()));
        }
        let push = {
            let mut b = [0u8; 12];
            b[0..4].copy_from_slice(&elements.to_le_bytes());
            b
        };
        unsafe { self.record_2bind_into(device, descriptor_pool, cmd_buf, input, output, &push, ((elements + 255) / 256, 1, 1), output, ds_collector) }
    }

    /// Fused Q4_K dequant + matrix-vector dispatch.
    ///
    /// `weight` must contain Q4K quantized bytes (144 bytes / 256 elements per block).
    /// `input` is FP32 vector of length K.
    /// `output` is a HOST_VISIBLE staging buffer of size N×4 bytes.
    /// Push constants: [N, K, n_blocks_per_row].
    pub fn dispatch_matmul_q4k(
        &self,
        ctx: &VulkanContext,
        weight: &crate::buffer::GpuBuffer,
        input: &crate::buffer::GpuBuffer,
        output: &mut crate::buffer::GpuBuffer,
        n: u32, // output rows
        k: u32, // inner dim
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            // CPU fallback: dequant Q4K inline then multiply
            cpu_matmul_q4k_fallback(weight.as_f32_slice(), input.as_f32_slice(), output, n as usize, k as usize);
            return Ok(());
        }

        let n_blocks = (k + 255) / 256;

        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            let layouts = [self.descriptor_set_layout.unwrap()];
            let alloc_info = ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b_w   = [ash::vk::DescriptorBufferInfo::default().buffer(weight.handle.unwrap()).offset(0).range(weight.size as u64)];
            let b_in  = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];

            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_w),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);

            let constants = [n, k, n_blocks];
            let bytes = std::slice::from_raw_parts(constants.as_ptr() as *const u8, 12);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, bytes);

            let groups_x = (n + 63) / 64;
            device.cmd_dispatch(cmd_buf, groups_x, 1, 1);
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
        if !self.vulkan_active {
            cpu_rmsnorm(input, weight, output, seq_len, hidden_size, eps);
            return Ok(());
        }
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

    pub fn dispatch_matmul_ternary(
        &self,
        ctx: &VulkanContext,
        a: &GpuBuffer,
        b_packed: &GpuBuffer,
        c: &mut GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];
            
            let b_a = ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_b = ash::vk::DescriptorBufferInfo::default().buffer(b_packed.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_c = ash::vk::DescriptorBufferInfo::default().buffer(c.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_a)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_b)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_c)),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let mut constants = [0u8; 12];
            constants[0..4].copy_from_slice(&m.to_le_bytes());
            constants[4..8].copy_from_slice(&k.to_le_bytes());
            constants[8..12].copy_from_slice(&n.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            let group_x = (n + 31) / 32;
            let group_y = m;
            let group_z = 1;

            device.cmd_dispatch(cmd_buf, group_x, group_y, group_z);
            device.end_command_buffer(cmd_buf).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dispatch_mamba_selective_scan(
        &self,
        ctx: &VulkanContext,
        u: &GpuBuffer,
        delta: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        state: &GpuBuffer,
        y: &mut GpuBuffer,
        seq_len: u32,
        d_inner: u32,
        d_state: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).unwrap()[0];

            let b_u = ash::vk::DescriptorBufferInfo::default().buffer(u.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_delta = ash::vk::DescriptorBufferInfo::default().buffer(delta.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_a = ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_b = ash::vk::DescriptorBufferInfo::default().buffer(b.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_c = ash::vk::DescriptorBufferInfo::default().buffer(c.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_state = ash::vk::DescriptorBufferInfo::default().buffer(state.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);
            let b_y = ash::vk::DescriptorBufferInfo::default().buffer(y.handle.unwrap()).offset(0).range(ash::vk::WHOLE_SIZE);

            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_u)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_delta)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_a)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_b)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(4).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_c)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(5).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_state)),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(6).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&b_y)),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);

            let mut constants = [0u8; 12];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&d_inner.to_le_bytes());
            constants[8..12].copy_from_slice(&d_state.to_le_bytes());

            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            let group_x = (d_inner + 255) / 256;

            device.cmd_dispatch(cmd_buf, group_x, 1, 1);
            device.end_command_buffer(cmd_buf).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(ctx.queue.unwrap()).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn dispatch_silu(&self, ctx: &VulkanContext, input: &GpuBuffer, output: &mut GpuBuffer, elements: u32) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            cpu_silu(input, output, elements);
            return Ok(());
        }
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
        if !self.vulkan_active {
            cpu_rope(q, k, seq_len, num_heads_q, num_heads_k, head_dim, freq_base, start_pos);
            return Ok(());
        }
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
        // Simulação (sem GPU): atenção de referência em CPU. O caminho principal
        // continua sendo o shader Vulkan abaixo; isto só preenche numéricos reais
        // quando não há acelerador.
        if !self.vulkan_active {
            cpu_attention_sim(q, k, v, out_attn, seq_len, head_dim, scale);
            return Ok(());
        }
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

    pub fn dispatch_turbo_quant_attention(
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
        // Simulação (sem GPU): mesma atenção de referência em CPU. Em hardware real
        // o caminho abaixo usa o shader TurboQuant (Lloyd-Max dequant + softmax).
        if !self.vulkan_active {
            cpu_attention_sim(q, k, v, out_attn, seq_len, head_dim, scale);
            return Ok(());
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

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
            
            let mut constants = [0u8; 16];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&head_dim.to_le_bytes());
            constants[8..12].copy_from_slice(&scale.to_le_bytes());
            // O pad de 4 bytes finais já é 0u8 pela inicialização.
            
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

    pub fn dispatch_zipgemm(
        &self,
        ctx: &VulkanContext,
        _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>,
        compressed_weights: &GpuBuffer,
        activations: &GpuBuffer,
        output: &mut GpuBuffer,
        tile_meta: &GpuBuffer,
        m: u32, k: u32, n: u32,
        tile_stride: u32,
        num_k_tiles: u32,
        weights_per_tile: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active { return Ok(()); }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_sets = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let descriptor_set = descriptor_sets[0];

            let b0 = [ash::vk::DescriptorBufferInfo::default().buffer(compressed_weights.handle.unwrap()).offset(0).range(compressed_weights.size as u64)];
            let b1 = [ash::vk::DescriptorBufferInfo::default().buffer(activations.handle.unwrap()).offset(0).range(activations.size as u64)];
            let b2 = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];
            let b3 = [ash::vk::DescriptorBufferInfo::default().buffer(tile_meta.handle.unwrap()).offset(0).range(tile_meta.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b0),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b1),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b2),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b3),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            
            let mut constants = [0u8; 24];
            constants[0..4].copy_from_slice(&m.to_le_bytes());
            constants[4..8].copy_from_slice(&k.to_le_bytes());
            constants[8..12].copy_from_slice(&n.to_le_bytes());
            constants[12..16].copy_from_slice(&tile_stride.to_le_bytes());
            constants[16..20].copy_from_slice(&num_k_tiles.to_le_bytes());
            constants[20..24].copy_from_slice(&weights_per_tile.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            device.cmd_dispatch(cmd_buf, (n + 15) / 16, (m + 15) / 16, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    /// Despacha o kernel de roteamento MoE (Mixture of Experts).
    /// Calcula os top_k experts para cada token na sequência usando os pesos gate comprimidos.
    /// Despacha o kernel de roteamento MoE (Mixture of Experts).
    /// 
    /// `routing_mode`: 0 = top-k padrão (Selection Sort), 1 = HARD k=1 (Esparsidade Extrema).
    /// Modo k=1 elimina o "Paradoxo da Densidade Temporal": ao ativar apenas 1 expert
    /// por camada, a cascata de page faults no SSD é estancada, permitindo
    /// prefetch preditivo One-Token-Lag via APEX.
    pub fn dispatch_moe_routing(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        gate_weights: &GpuBuffer,
        topk_indices: &mut GpuBuffer,
        topk_scores: &mut GpuBuffer,
        seq_len: u32,
        hidden_dim: u32,
        num_experts: u32,
        top_k: u32,
        routing_mode: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            // Modo simulação: seleciona experts 0..top_k deterministicamente
            let total = (seq_len * top_k) as usize;
            for i in 0..total {
                let idx_bytes = ((i % num_experts as usize) as u32).to_le_bytes();
                let score_bytes = (0.25f32).to_le_bytes();
                let offset = i * 4;
                if offset + 4 <= topk_indices.size {
                    topk_indices.as_mut_bytes()[offset..offset+4].copy_from_slice(&idx_bytes);
                }
                if offset + 4 <= topk_scores.size {
                    topk_scores.as_mut_bytes()[offset..offset+4].copy_from_slice(&score_bytes);
                }
            }
            return Ok(());
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            // ── SSBOs intermediários com tamanho calculado em runtime (num_experts ilimitado)
            // binding 4: expert_scores  — seq_len × num_experts floats
            let scores_bytes = (seq_len * num_experts * 4) as usize;
            let mut expert_scores_buf = GpuBuffer::new_storage(scores_bytes);

            // binding 5: selection_bitmap — seq_len × ceil(num_experts/32) uint32s
            let bitmap_words = (num_experts + 31) / 32;
            let bitmap_bytes = (seq_len * bitmap_words * 4) as usize;
            let mut selection_bitmap_buf = GpuBuffer::new_storage(bitmap_bytes);

            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts)
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            let b0 = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b1 = [ash::vk::DescriptorBufferInfo::default().buffer(gate_weights.handle.unwrap()).offset(0).range(gate_weights.size as u64)];
            let b2 = [ash::vk::DescriptorBufferInfo::default().buffer(topk_indices.handle.unwrap()).offset(0).range(topk_indices.size as u64)];
            let b3 = [ash::vk::DescriptorBufferInfo::default().buffer(topk_scores.handle.unwrap()).offset(0).range(topk_scores.size as u64)];
            let b4 = [ash::vk::DescriptorBufferInfo::default().buffer(expert_scores_buf.handle.unwrap()).offset(0).range(scores_bytes as u64)];
            let b5 = [ash::vk::DescriptorBufferInfo::default().buffer(selection_bitmap_buf.handle.unwrap()).offset(0).range(bitmap_bytes as u64)];

            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b0),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b1),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b2),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b3),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(4).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b4),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(5).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b5),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);

            let mut constants = [0u8; 20];
            constants[0..4].copy_from_slice(&seq_len.to_le_bytes());
            constants[4..8].copy_from_slice(&hidden_dim.to_le_bytes());
            constants[8..12].copy_from_slice(&num_experts.to_le_bytes());
            constants[12..16].copy_from_slice(&top_k.to_le_bytes());
            constants[16..20].copy_from_slice(&routing_mode.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            // Pass 1: Roteamento / top-k
            device.cmd_dispatch(cmd_buf, (seq_len + 31) / 32, 1, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_fused_layernorm_gelu(
        &self,
        ctx: &VulkanContext,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        output: &mut GpuBuffer,
        hidden_dim: u32,
        eps: f32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            // Emulação de CPU omitida para simplificação
            return Ok(());
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();

            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts)
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            let b_in = [ash::vk::DescriptorBufferInfo::default().buffer(input.handle.unwrap()).offset(0).range(input.size as u64)];
            let b_g = [ash::vk::DescriptorBufferInfo::default().buffer(gamma.handle.unwrap()).offset(0).range(gamma.size as u64)];
            let b_b = [ash::vk::DescriptorBufferInfo::default().buffer(beta.handle.unwrap()).offset(0).range(beta.size as u64)];
            let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(output.handle.unwrap()).offset(0).range(output.size as u64)];

            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_in),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_g),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_b),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
            ], &[]);

            let cmd_buf = device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);

            let mut constants = [0u8; 8];
            constants[0..4].copy_from_slice(&hidden_dim.to_le_bytes());
            constants[4..8].copy_from_slice(&eps.to_le_bytes());
            
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);

            // Um workgroup por linha/token
            device.cmd_dispatch(cmd_buf, input.size as u32 / (hidden_dim * 4), 1, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_add(
        &self,
        ctx: &VulkanContext,
        _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>,
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &mut GpuBuffer,
        elems: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            cpu_elementwise_add(a, b, out, elems);
            return Ok(());
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            let b0 = [ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(a.size as u64)];
            let b1 = [ash::vk::DescriptorBufferInfo::default().buffer(b.handle.unwrap()).offset(0).range(b.size as u64)];
            let b2 = [ash::vk::DescriptorBufferInfo::default().buffer(out.handle.unwrap()).offset(0).range(out.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b0),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b1),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b2),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];
            
            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &elems.to_le_bytes());
            device.cmd_dispatch(cmd_buf, (elems + 255) / 256, 1, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_mul(
        &self,
        ctx: &VulkanContext,
        _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>,
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &mut GpuBuffer,
        elems: u32,
    ) -> Result<(), NodeStorError> {
        if !self.vulkan_active {
            cpu_elementwise_mul(a, b, out, elems);
            return Ok(());
        }
        unsafe {
            let device = ctx.device.as_ref().unwrap();
            let descriptor_pool = ctx.descriptor_pool.unwrap();
            let command_pool = ctx.command_pool.unwrap();
            
            let layouts = [self.descriptor_set_layout.unwrap()];
            let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

            let b0 = [ash::vk::DescriptorBufferInfo::default().buffer(a.handle.unwrap()).offset(0).range(a.size as u64)];
            let b1 = [ash::vk::DescriptorBufferInfo::default().buffer(b.handle.unwrap()).offset(0).range(b.size as u64)];
            let b2 = [ash::vk::DescriptorBufferInfo::default().buffer(out.handle.unwrap()).offset(0).range(out.size as u64)];
            
            device.update_descriptor_sets(&[
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b0),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b1),
                ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b2),
            ], &[]);

            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
            let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];
            
            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
            device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
            device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &elems.to_le_bytes());
            device.cmd_dispatch(cmd_buf, (elems + 255) / 256, 1, 1);
            device.end_command_buffer(cmd_buf).unwrap();

            device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
            device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
            device.free_command_buffers(command_pool, &[cmd_buf]);
            device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
        }
        Ok(())
    }

    pub fn dispatch_flash_attention(
       &self,
       ctx: &VulkanContext,
       _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>,
       q: &GpuBuffer,
       k: &GpuBuffer,
       v: &GpuBuffer,
       out: &mut GpuBuffer,
       seq: u32,
       hd: u32,
       sc: f32,
       causal: u32,
    ) -> Result<(), NodeStorError> {
       if !self.vulkan_active { return Ok(()); }
       unsafe {
           let device = ctx.device.as_ref().unwrap();
           let descriptor_pool = ctx.descriptor_pool.unwrap();
           let command_pool = ctx.command_pool.unwrap();
           
           let layouts = [self.descriptor_set_layout.unwrap()];
           let descriptor_set = device.allocate_descriptor_sets(&ash::vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];

           let b_q = [ash::vk::DescriptorBufferInfo::default().buffer(q.handle.unwrap()).offset(0).range(q.size as u64)];
           let b_k = [ash::vk::DescriptorBufferInfo::default().buffer(k.handle.unwrap()).offset(0).range(k.size as u64)];
           let b_v = [ash::vk::DescriptorBufferInfo::default().buffer(v.handle.unwrap()).offset(0).range(v.size as u64)];
           let b_out = [ash::vk::DescriptorBufferInfo::default().buffer(out.handle.unwrap()).offset(0).range(out.size as u64)];
           
           device.update_descriptor_sets(&[
               ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_q),
               ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_k),
               ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_v),
               ash::vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3).descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b_out),
           ], &[]);

           let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(ash::vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
           let cmd_buf = device.allocate_command_buffers(&alloc_cmds).map_err(|e| NodeStorError::VulkanError(e.to_string()))?[0];
           
           device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default()).unwrap();
           device.cmd_bind_pipeline(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline.unwrap());
           device.cmd_bind_descriptor_sets(cmd_buf, ash::vk::PipelineBindPoint::COMPUTE, self.pipeline_layout.unwrap(), 0, &[descriptor_set], &[]);
           
           let mut constants = [0u8; 16];
           constants[0..4].copy_from_slice(&seq.to_le_bytes());
           constants[4..8].copy_from_slice(&hd.to_le_bytes());
           constants[8..12].copy_from_slice(&sc.to_le_bytes());
           constants[12..16].copy_from_slice(&causal.to_le_bytes());
           
           device.cmd_push_constants(cmd_buf, self.pipeline_layout.unwrap(), ash::vk::ShaderStageFlags::COMPUTE, 0, &constants);
           device.cmd_dispatch(cmd_buf, (seq + 31) / 32, 1, 1);
           device.end_command_buffer(cmd_buf).unwrap();

           device.queue_submit(ctx.queue.unwrap(), &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])], ash::vk::Fence::null()).unwrap();
           device.queue_wait_idle(ctx.queue.unwrap()).unwrap();
           device.free_command_buffers(command_pool, &[cmd_buf]);
           device.free_descriptor_sets(descriptor_pool, &[descriptor_set]).unwrap();
       }
       Ok(())
    }

    pub fn dispatch_cross_entropy_masked_back(&self, _ctx: &VulkanContext, _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>, _logits: &GpuBuffer, _grad_logits: &mut GpuBuffer, _vocab_size: u32, _target: u32, _scale: f32, _cmd: ash::vk::CommandBuffer, _set: ash::vk::DescriptorSet) -> Result<(), NodeStorError> { Ok(()) }
    pub fn dispatch_out_prod(&self, _ctx: &VulkanContext, _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>, _a: &GpuBuffer, _b: &GpuBuffer, _out: &mut GpuBuffer, _r: u32, _c: u32, _scale: f32, _cmd: ash::vk::CommandBuffer, _set: ash::vk::DescriptorSet) -> Result<(), NodeStorError> { Ok(()) }
    pub fn dispatch_opt_step_adam(&self, _ctx: &VulkanContext, _recycler: &Option<std::sync::Mutex<crate::command_recycler::CommandRecycler>>, _w: &mut GpuBuffer, _g: &GpuBuffer, _m1: &mut GpuBuffer, _m2: &mut GpuBuffer, _elems: u32, _lr: f32, _b1: f32, _b2: f32, _eps: f32, _wd: f32, _s1: f32, _s2: f32, _cmd: ash::vk::CommandBuffer, _set: ash::vk::DescriptorSet) -> Result<(), NodeStorError> { Ok(()) }

}

pub fn create_all_pipelines(
    ctx: &VulkanContext,
) -> Result<HashMap<PipelineKind, ComputePipeline>, VulkanError> {
    if !ctx.vulkan_available {
        // Sem device Vulkan real (headless/driver ausente): usa o conjunto COMPLETO
        // de pipelines de simulação — inclui TurboQuantAttention, MoERouting,
        // FusedLayerNormGelu, Add, Mul, etc. O subconjunto antigo aqui estava
        // INCOMPLETO (faltava TQA), travando o forward de modelos reais.
        return Ok(create_simulation_pipelines());
    }
    let mut map = HashMap::new();

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
            ShaderKind::DequantQ6K => PipelineKind::DequantQ6K,
            ShaderKind::DequantQ5_0 => PipelineKind::DequantQ5_0,
            ShaderKind::Matmul => PipelineKind::Matmul,
            ShaderKind::CosineSim => PipelineKind::CosineSim,
            ShaderKind::Lossless => PipelineKind::Lossless,
            ShaderKind::GDeflate => PipelineKind::GDeflate,
            ShaderKind::MatmulQ4 => PipelineKind::MatmulQ4,
            ShaderKind::MatmulQ4K => PipelineKind::MatmulQ4K,
            ShaderKind::MatmulTensorCore => PipelineKind::MatmulTensorCore,
            ShaderKind::MatmulTernary => PipelineKind::MatmulTernary,
            ShaderKind::MambaSelectiveScan => PipelineKind::MambaSelectiveScan,
            ShaderKind::RmsNorm => PipelineKind::RmsNorm,
            ShaderKind::RoPe => PipelineKind::RoPe,
            ShaderKind::SiLu => PipelineKind::SiLu,
            ShaderKind::Softmax => PipelineKind::Softmax,
            ShaderKind::Attention => PipelineKind::Attention,
            ShaderKind::CoopMatrix => PipelineKind::CoopMatrix,
            ShaderKind::ZipGEMM => PipelineKind::ZipGEMM,
            ShaderKind::FlashAttention => PipelineKind::FlashAttention,
            ShaderKind::TreeAttention => PipelineKind::TreeAttention,
            ShaderKind::CrossEntropyMaskedBack => PipelineKind::CrossEntropyMaskedBack,
            ShaderKind::OutProd => PipelineKind::OutProd,
            ShaderKind::OptStepAdam => PipelineKind::OptStepAdam,
            ShaderKind::Add => PipelineKind::Add,
            ShaderKind::Mul => PipelineKind::Mul,
            ShaderKind::TurboQuantAttention => PipelineKind::TurboQuantAttention,
            ShaderKind::MoERouting => PipelineKind::MoERouting,
            _ => continue,
        };

        match ComputePipeline::new_real(ctx, kind, &shader.bytecode) {
            Ok(p) => { map.insert(kind, p); }
            Err(e) => {
                // Shaders obrigatórios devem ser GPU reais — propaga o erro para
                // VulkanEngine::new() cair em simulação CPU completa.
                // Shaders opcionais (TensorCore, ZipGEMM, etc.) usam simulação local.
                let is_required = matches!(kind,
                    PipelineKind::Matmul | PipelineKind::RmsNorm | PipelineKind::RoPe
                    | PipelineKind::SiLu | PipelineKind::Add | PipelineKind::Mul
                );
                if is_required {
                    return Err(e);
                }
                tracing::warn!("Shader {:?} rejeitado ({}): usando simulação para este op opcional", kind, e);
                map.insert(kind, ComputePipeline::new_simulation(kind));
            }
        }
    }

    Ok(map)
}

/// Atenção de referência em CPU para o caminho de SIMULAÇÃO (sem Vulkan).
///
/// Implementa atenção multi-head com suporte a GQA derivando o nº de cabeças dos
/// tamanhos dos buffers: `num_q_heads = (q_len/seq_len)/head_dim`. Para cada
/// cabeça/posição: scores = scale·(Q·Kᵀ) → softmax estável → saída = Σ p·V.
/// A saída tem o mesmo layout de Q (`seq_len · num_q_heads · head_dim`).
///
/// O foco do NodeStor é a GPU; esta rotina é o espelho fiel em CPU para máquinas
/// sem acelerador e para testes numéricos determinísticos.
fn cpu_attention_sim(
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    out_attn: &mut GpuBuffer,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
) {
    let seq = seq_len as usize;
    let hd = head_dim as usize;
    if seq == 0 || hd == 0 { return; }

    let qs = q.as_f32_slice();
    let ks = k.as_f32_slice();
    let vs = v.as_f32_slice();

    let per_pos_q = (qs.len() / seq).max(hd);   // num_q_heads · head_dim
    let per_pos_kv = (ks.len() / seq).max(hd);  // num_kv_heads · head_dim
    let num_q_heads = (per_pos_q / hd).max(1);
    let num_kv_heads = (per_pos_kv / hd).max(1);
    let group = (num_q_heads / num_kv_heads).max(1); // mapeamento GQA

    let n_out = out_attn.size / 4;
    let mut out_vals = vec![0.0f32; n_out];

    for qi in 0..seq {
        for h in 0..num_q_heads {
            let kvh = (h / group).min(num_kv_heads - 1);
            // 1) scores = scale · (Q · Kᵀ) sobre todas as posições de chave
            let q_base = qi * per_pos_q + h * hd;
            let mut scores = vec![0.0f32; seq];
            for kj in 0..seq {
                let k_base = kj * per_pos_kv + kvh * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += qs.get(q_base + d).copied().unwrap_or(0.0)
                         * ks.get(k_base + d).copied().unwrap_or(0.0);
                }
                scores[kj] = dot * scale;
            }
            // 2) softmax numericamente estável (subtrai o máximo)
            let maxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - maxs).exp(); sum += *s; }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            // 3) saída = Σ (p · V)
            let o_base = qi * per_pos_q + h * hd;
            for d in 0..hd {
                let mut acc = 0.0f32;
                for kj in 0..seq {
                    let v_base = kj * per_pos_kv + kvh * hd;
                    acc += scores[kj] * inv * vs.get(v_base + d).copied().unwrap_or(0.0);
                }
                if let Some(slot) = out_vals.get_mut(o_base + d) { *slot = acc; }
            }
        }
    }

    let bytes = out_attn.as_mut_bytes();
    for (i, val) in out_vals.iter().enumerate() {
        let o = i * 4;
        if o + 4 <= bytes.len() {
            bytes[o..o + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
}

/// Escreve um vetor f32 nos bytes de um GpuBuffer (modo simulação).
fn write_f32_to_buffer(vals: &[f32], buf: &mut GpuBuffer) {
    let bytes = buf.as_mut_bytes();
    for (i, v) in vals.iter().enumerate() {
        let o = i * 4;
        if o + 4 <= bytes.len() {
            bytes[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// RMSNorm de referência (CPU): por linha, `y[i] = x[i] / sqrt(mean(x²) + eps) · w[i]`.
fn cpu_rmsnorm(input: &GpuBuffer, weight: &GpuBuffer, output: &mut GpuBuffer, seq_len: u32, hidden: u32, eps: f32) {
    let x = input.as_f32_slice();
    let w = weight.as_f32_slice();
    let hidden = (hidden as usize).max(1);
    let seq = seq_len as usize;
    let n_out = output.size / 4;
    let mut out = vec![0.0f32; n_out];
    for s in 0..seq {
        let base = s * hidden;
        if base + hidden > x.len() { break; }
        let mut sum_sq = 0.0f32;
        for i in 0..hidden { sum_sq += x[base + i] * x[base + i]; }
        let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            // Sem pesos materializados (buffer "fino") → escala 1.0 (identidade).
            let wi = if w.is_empty() { 1.0 } else { w.get(i).copied().unwrap_or(1.0) };
            if base + i < out.len() { out[base + i] = x[base + i] * inv_rms * wi; }
        }
    }
    write_f32_to_buffer(&out, output);
}

/// SiLU/Swish de referência (CPU): `y[i] = x[i] · sigmoid(x[i]) = x[i] / (1 + e^-x[i])`.
fn cpu_silu(input: &GpuBuffer, output: &mut GpuBuffer, elements: u32) {
    let x = input.as_f32_slice();
    let n = (elements as usize).min(x.len()).min(output.size / 4);
    let mut out = vec![0.0f32; output.size / 4];
    for i in 0..n {
        let v = x[i];
        out[i] = v / (1.0 + (-v).exp());
    }
    write_f32_to_buffer(&out, output);
}

/// Soma elementwise de referência (CPU): `out[i] = a[i] + b[i]`.
fn cpu_elementwise_add(a: &GpuBuffer, b: &GpuBuffer, out: &mut GpuBuffer, elems: u32) {
    let av = a.as_f32_slice();
    let bv = b.as_f32_slice();
    let n = (elems as usize).min(out.size / 4);
    let mut o = vec![0.0f32; out.size / 4];
    for i in 0..n {
        o[i] = av.get(i).copied().unwrap_or(0.0) + bv.get(i).copied().unwrap_or(0.0);
    }
    write_f32_to_buffer(&o, out);
}

/// Multiplicação elementwise de referência (CPU): `out[i] = a[i] · b[i]` (SwiGLU).
fn cpu_elementwise_mul(a: &GpuBuffer, b: &GpuBuffer, out: &mut GpuBuffer, elems: u32) {
    let av = a.as_f32_slice();
    let bv = b.as_f32_slice();
    let n = (elems as usize).min(out.size / 4);
    let mut o = vec![0.0f32; out.size / 4];
    for i in 0..n {
        o[i] = av.get(i).copied().unwrap_or(0.0) * bv.get(i).copied().unwrap_or(0.0);
    }
    write_f32_to_buffer(&o, out);
}

/// Aplica RoPE (Rotary Position Embedding) interleaved in-place a um buffer
/// [seq_len, num_heads, head_dim]. Para cada par (2i, 2i+1): rotação por
/// θ = pos / freq_base^(2i/head_dim), com pos = start_pos + índice da posição.
fn rope_apply(buf: &mut GpuBuffer, seq_len: u32, num_heads: u32, head_dim: u32, freq_base: f32, start_pos: u32) {
    let hd = head_dim as usize;
    if hd < 2 { return; }
    let half = hd / 2;
    let mut data: Vec<f32> = buf.as_f32_slice().to_vec();
    for s in 0..seq_len as usize {
        let pos = (start_pos as usize + s) as f32;
        for h in 0..num_heads as usize {
            let base = (s * num_heads as usize + h) * hd;
            for i in 0..half {
                let theta = pos / freq_base.powf((2 * i) as f32 / hd as f32);
                let (sin, cos) = theta.sin_cos();
                let i0 = base + 2 * i;
                let i1 = base + 2 * i + 1;
                if i1 < data.len() {
                    let x0 = data[i0];
                    let x1 = data[i1];
                    data[i0] = x0 * cos - x1 * sin;
                    data[i1] = x0 * sin + x1 * cos;
                }
            }
        }
    }
    write_f32_to_buffer(&data, buf);
}

/// RoPE de referência (CPU) aplicado a Q e K.
fn cpu_rope(q: &mut GpuBuffer, k: &mut GpuBuffer, seq_len: u32, num_heads_q: u32, num_heads_k: u32, head_dim: u32, freq_base: f32, start_pos: u32) {
    rope_apply(q, seq_len, num_heads_q, head_dim, freq_base, start_pos);
    rope_apply(k, seq_len, num_heads_k, head_dim, freq_base, start_pos);
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

/// CPU fallback para Q4K matmul (simulation mode).
/// Nota: `weight_bytes` aqui é um slice de f32 da memória de simulação — no fallback
/// os dados foram escritos como raw bytes via from_cpu_data, então as_f32_slice()
/// retorna o mesmo slice que foi passado. Na prática, este fallback só é chamado
/// quando Vulkan não está disponível e o pipeline está em modo de simulação.
fn cpu_matmul_q4k_fallback(weight_bytes: &[f32], input: &[f32], output: &mut GpuBuffer, n: usize, _k: usize) {
    // Em modo simulação o gpu_bank TAMBÉM tem os bytes raw, não F32 dequantizados.
    // Como o cpu_reference.rs usa staging_bank (já em F32), este fallback retorna zeros
    // para não produzir resultados incorretos — a rota CPU usa staging_bank via cpu_reference.
    let out = output.as_mut_bytes();
    let _ = (weight_bytes, input, n);
    out.fill(0);
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

/// Cria um mapa de pipelines em modo simulação (sem Vulkan).
/// Usado por `VulkanEngine::new_simulation()` para garantir funcionamento
/// em ambientes sem GPU (CI/CD, Docker, benchmarks headless).
pub fn create_simulation_pipelines() -> HashMap<PipelineKind, ComputePipeline> {
    let mut map = HashMap::new();
    for kind in [
        PipelineKind::DequantQ4,
        PipelineKind::DequantQ8,
        PipelineKind::DequantQ6K,
        PipelineKind::DequantQ5_0,
        PipelineKind::Matmul,
        PipelineKind::CosineSim,
        PipelineKind::Lossless,
        PipelineKind::GDeflate,
        PipelineKind::MatmulQ4,
        PipelineKind::MatmulQ4K,
        PipelineKind::MatmulTensorCore,
        PipelineKind::MatmulTernary,
        PipelineKind::RmsNorm,
        PipelineKind::RoPe,
        PipelineKind::SiLu,
        PipelineKind::Softmax,
        PipelineKind::Attention,
        PipelineKind::CoopMatrix,
        PipelineKind::ZipGEMM,
        PipelineKind::FlashAttention,
        PipelineKind::TreeAttention,
        PipelineKind::CrossEntropyMaskedBack,
        PipelineKind::OutProd,
        PipelineKind::OptStepAdam,
        PipelineKind::Add,
        PipelineKind::Mul,
    ] {
        map.insert(kind, ComputePipeline::new_simulation(kind));
    }
    // Pipelines V3 — adicionados explicitamente na lista de simulação
    map.insert(PipelineKind::TurboQuantAttention, ComputePipeline::new_simulation(PipelineKind::TurboQuantAttention));
    map.insert(PipelineKind::MoERouting, ComputePipeline::new_simulation(PipelineKind::MoERouting));
    map.insert(PipelineKind::FusedLayerNormGelu, ComputePipeline::new_simulation(PipelineKind::FusedLayerNormGelu));
    map
}

#[cfg(test)]
mod attention_sim_tests {
    use crate::VulkanEngine;

    fn f2b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }

    /// Prova numérica: a atenção em simulação computa softmax(Q·Kᵀ·scale)·V de
    /// verdade — não é mais um no-op que devolve zeros.
    #[test]
    fn test_cpu_attention_matches_hand_computed() {
        let engine = VulkanEngine::new_simulation();
        // 1 cabeça, head_dim=2, seq_len=2, scale=1.0
        let q = engine.upload(&f2b(&[1.0, 0.0,  0.0, 1.0])).unwrap();
        let k = engine.upload(&f2b(&[1.0, 0.0,  0.0, 1.0])).unwrap();
        let v = engine.upload(&f2b(&[1.0, 2.0,  3.0, 4.0])).unwrap();

        let out = engine.attention(&q, &k, &v, 2, 2, 1.0).unwrap();
        let got = engine.download_f32(&out).unwrap();

        // pos0: scores=[1,0] → p=[0.7311,0.2689] → [1.5379, 2.5379]
        // pos1: scores=[0,1] → p=[0.2689,0.7311] → [2.4621, 3.4621]
        let expected = [1.5379f32, 2.5379, 2.4621, 3.4621];
        assert_eq!(got.len(), 4, "saída deve ter seq_len·head_dim = 4 floats");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-3, "idx {}: esperado {:.4}, obtido {:.4}", i, e, g);
        }
    }

    /// Com uma única posição de chave (seq_len=1), softmax de 1 elemento = 1.0,
    /// logo a saída da atenção deve ser exatamente V.
    #[test]
    fn test_cpu_attention_single_key_returns_v() {
        let engine = VulkanEngine::new_simulation();
        let q = engine.upload(&f2b(&[0.5, -0.3])).unwrap();
        let k = engine.upload(&f2b(&[9.0, 9.0])).unwrap();
        let v = engine.upload(&f2b(&[7.0, -2.0])).unwrap();
        let out = engine.attention(&q, &k, &v, 1, 2, 0.125).unwrap();
        let got = engine.download_f32(&out).unwrap();
        assert!((got[0] - 7.0).abs() < 1e-5, "saída[0] deve ser V[0]=7.0, foi {}", got[0]);
        assert!((got[1] + 2.0).abs() < 1e-5, "saída[1] deve ser V[1]=-2.0, foi {}", got[1]);
    }

    /// GQA: 2 cabeças de query compartilhando 1 cabeça de KV não deve quebrar
    /// (saída do tamanho de Q, finita).
    #[test]
    fn test_cpu_attention_gqa_shapes() {
        let engine = VulkanEngine::new_simulation();
        // q: 2 heads × head_dim 2 = 4 floats; k/v: 1 kv head × 2 = 2 floats; seq=1
        let q = engine.upload(&f2b(&[1.0, 0.0, 0.0, 1.0])).unwrap();
        let k = engine.upload(&f2b(&[1.0, 1.0])).unwrap();
        let v = engine.upload(&f2b(&[5.0, 6.0])).unwrap();
        let out = engine.attention(&q, &k, &v, 1, 2, 1.0).unwrap();
        let got = engine.download_f32(&out).unwrap();
        assert_eq!(got.len(), 4, "saída deve ter o layout de Q (2 heads × 2)");
        // seq=1 → softmax=1 → cada cabeça de query recebe V da única kv head
        for (i, g) in got.iter().enumerate() {
            assert!(g.is_finite(), "saída[{}] deve ser finita", i);
            let expected = if i % 2 == 0 { 5.0 } else { 6.0 };
            assert!((g - expected).abs() < 1e-5, "idx {}: esperado {}, foi {}", i, expected, g);
        }
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

/// Testes de FIDELIDADE MATEMÁTICA das operações-núcleo do Transformer.
///
/// Cada teste compara a saída em CPU (caminho de simulação) contra valores
/// calculados À MÃO a partir das fórmulas canônicas. Provam que rmsnorm, silu,
/// add, mul, rope e matmul computam exatamente o que a matemática exige — base
/// para 100% de fidelidade do forward pass quando há pesos reais.
#[cfg(test)]
mod math_fidelity_tests {
    use crate::VulkanEngine;

    fn f2b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-3 }

    #[test]
    fn test_rmsnorm_formula() {
        let engine = VulkanEngine::new_simulation();
        // x = [3, 4], hidden=2. mean(x²) = (9+16)/2 = 12.5; rms = 3.53553.
        // y = x/rms · w, com w = [2, 0.5] → [3/3.53553·2, 4/3.53553·0.5]
        //   = [1.69706, 0.56569]
        let x = engine.upload(&f2b(&[3.0, 4.0])).unwrap();
        let w = engine.upload(&f2b(&[2.0, 0.5])).unwrap();
        let out = engine.rmsnorm(&x, &w, 1, 2, 1e-9).unwrap();
        let y = engine.download_f32(&out).unwrap();
        assert!(approx(y[0], 1.69706), "rmsnorm[0]={}", y[0]);
        assert!(approx(y[1], 0.56569), "rmsnorm[1]={}", y[1]);
    }

    #[test]
    fn test_silu_formula() {
        let engine = VulkanEngine::new_simulation();
        // silu(x) = x·sigmoid(x): silu(0)=0, silu(1)=0.73106, silu(-1)=-0.26894
        let x = engine.upload(&f2b(&[0.0, 1.0, -1.0])).unwrap();
        let out = engine.silu(&x, 3).unwrap();
        let y = engine.download_f32(&out).unwrap();
        assert!(approx(y[0], 0.0), "silu(0)={}", y[0]);
        assert!(approx(y[1], 0.73106), "silu(1)={}", y[1]);
        assert!(approx(y[2], -0.26894), "silu(-1)={}", y[2]);
    }

    #[test]
    fn test_add_and_mul_elementwise() {
        let engine = VulkanEngine::new_simulation();
        let a = engine.upload(&f2b(&[1.0, 2.0, 3.0])).unwrap();
        let b = engine.upload(&f2b(&[10.0, 20.0, 30.0])).unwrap();
        let sum = engine.download_f32(&engine.add(&a, &b, 3).unwrap()).unwrap();
        assert_eq!(sum, vec![11.0, 22.0, 33.0]);

        let c = engine.upload(&f2b(&[2.0, 3.0, 4.0])).unwrap();
        let d = engine.upload(&f2b(&[5.0, 6.0, 7.0])).unwrap();
        let prod = engine.download_f32(&engine.mul(&c, &d, 3).unwrap()).unwrap();
        assert_eq!(prod, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_matmul_formula() {
        let engine = VulkanEngine::new_simulation();
        // A(2×2)=[[1,2],[3,4]] · B(2×2)=[[5,6],[7,8]] = [[19,22],[43,50]]
        let a = engine.upload(&f2b(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let b = engine.upload(&f2b(&[5.0, 6.0, 7.0, 8.0])).unwrap();
        let out = engine.matmul(&a, &b, 2, 2, 2).unwrap();
        let y = engine.download_f32(&out).unwrap();
        assert_eq!(y, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_rope_rotation() {
        let engine = VulkanEngine::new_simulation();
        // head_dim=2, pos=1, freq_base=10000 → θ = 1/10000^0 = 1 rad.
        // q=[1,0] → [cos(1), sin(1)] = [0.54030, 0.84147]
        let mut q = engine.upload(&f2b(&[1.0, 0.0])).unwrap();
        let mut k = engine.upload(&f2b(&[1.0, 0.0])).unwrap();
        engine.rope(&mut q, &mut k, 1, 1, 1, 2, 10000.0, 1).unwrap();
        let qv = engine.download_f32(&q).unwrap();
        assert!(approx(qv[0], 0.54030), "rope q[0]={}", qv[0]);
        assert!(approx(qv[1], 0.84147), "rope q[1]={}", qv[1]);
    }

    #[test]
    fn test_rope_pos_zero_is_identity() {
        let engine = VulkanEngine::new_simulation();
        // pos=0 → θ=0 → rotação identidade: q inalterado.
        let mut q = engine.upload(&f2b(&[0.7, -0.3, 0.1, 0.9])).unwrap();
        let mut k = engine.upload(&f2b(&[0.2, 0.4, 0.6, 0.8])).unwrap();
        engine.rope(&mut q, &mut k, 1, 2, 2, 2, 10000.0, 0).unwrap();
        let qv = engine.download_f32(&q).unwrap();
        assert!(approx(qv[0], 0.7) && approx(qv[1], -0.3) && approx(qv[2], 0.1) && approx(qv[3], 0.9),
            "pos=0 deve ser identidade, got {:?}", qv);
    }
}
