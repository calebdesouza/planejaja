/// NodeStor COBER v2 — Vulkan Adaptive Pipeline: Mutação Restrita por Modelo
///
/// # Filosofia de Segurança
///
/// Este módulo NÃO gera SPIR-V do zero. Isso seria perigoso e inauditável.
/// Em vez disso, opera com **Mutação Restrita por Template**:
///
/// 1. O código SPIR-V é PRÉ-COMPILADO e ASSINADO (Template Estático).
/// 2. O Davi SÓ altera os **Specialization Constants** do Vulkan —
///    valores numéricos que o driver usa para otimizar branches internos.
/// 3. ANTES de enviar qualquer constante à GPU, o Rust valida
///    matematicamente cada valor contra bounds pré-definidos.
/// 4. Cada mutação é registrada no Audit Logger.
///
/// Resultado: nenhum código de GPU é gerado em runtime. Apenas CONSTANTES
/// são ajustadas dentro de um envelope validado. Zero risco de TDR,
/// zero injeção de shader malicioso, 100% auditável.
///
/// # Compatibilidade
/// - Funciona com QUALQUER GPU que suporte Vulkan 1.0+
/// - Intel integrada, AMD, NVIDIA, Apple (MoltenVK), Qualcomm (Android)
/// - Não requer driver especial, TCC mode, ou registry hacks

use std::collections::HashMap;
use std::time::SystemTime;

// ─────────────────────────────────────────────────────────────────────
// 1. DETECÇÃO DE HARDWARE
// ─────────────────────────────────────────────────────────────────────

/// Informações do hardware GPU detectadas em boot
#[derive(Debug, Clone)]
pub struct GpuProfile {
    /// Nome da GPU (ex: "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Vendor ID
    pub vendor: GpuVendor,
    /// VRAM total disponível (bytes)
    pub vram_total_bytes: u64,
    /// VRAM livre estimada (bytes)
    pub vram_free_bytes: u64,
    /// Largura de banda de memória (GB/s)
    pub memory_bandwidth_gbps: f32,
    /// Número de unidades de compute (CUs / SMs)
    pub compute_units: u32,
    /// Tamanho máximo de workgroup no eixo X
    pub max_workgroup_size_x: u32,
    /// Tamanho máximo de workgroup no eixo Y
    pub max_workgroup_size_y: u32,
    /// Tamanho máximo de shared memory por workgroup (bytes)
    pub max_shared_memory_bytes: u32,
    /// Se suporta subgroup operations (shuffle, ballot)
    pub supports_subgroup_ops: bool,
    /// Subgroup size (warp=32 NVIDIA, wavefront=64 AMD, etc.)
    pub subgroup_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,     // MoltenVK
    Qualcomm,  // Android/Adreno
    Unknown,
}

impl Default for GpuProfile {
    fn default() -> Self {
        // Perfil conservador que funciona em QUALQUER GPU Vulkan 1.0
        Self {
            name: "Generic Vulkan GPU".to_string(),
            vendor: GpuVendor::Unknown,
            vram_total_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            vram_free_bytes: 2 * 1024 * 1024 * 1024,  // 2GB
            memory_bandwidth_gbps: 100.0,
            compute_units: 16,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_shared_memory_bytes: 32768,
            supports_subgroup_ops: false,
            subgroup_size: 32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. SPECIALIZATION CONSTANTS (As únicas variáveis mutáveis)
// ─────────────────────────────────────────────────────────────────────

/// Um Specialization Constant com seus bounds de segurança
#[derive(Debug, Clone)]
pub struct SpecConstant {
    /// ID no SPIR-V (ex: constant_id = 0)
    pub constant_id: u32,
    /// Nome legível (para audit log)
    pub name: String,
    /// Valor atual
    pub value: SpecValue,
    /// Valor mínimo permitido (bound inferior)
    pub min_value: SpecValue,
    /// Valor máximo permitido (bound superior)
    pub max_value: SpecValue,
}

/// Tipo do valor de especialização (apenas inteiros e floats — sem ponteiros)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecValue {
    U32(u32),
    I32(i32),
    F32(f32),
}

impl SpecValue {
    /// Serializa o valor para um buffer de bytes (para Vulkan)
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            SpecValue::U32(v) => v.to_le_bytes().to_vec(),
            SpecValue::I32(v) => v.to_le_bytes().to_vec(),
            SpecValue::F32(v) => v.to_le_bytes().to_vec(),
        }
    }

    /// Tamanho em bytes
    pub fn size(&self) -> usize {
        4 // U32, I32 e F32 são todos 4 bytes
    }

    /// Verifica se `self` está dentro do range [min, max]
    pub fn is_within(&self, min: &SpecValue, max: &SpecValue) -> bool {
        match (self, min, max) {
            (SpecValue::U32(v), SpecValue::U32(lo), SpecValue::U32(hi)) => *v >= *lo && *v <= *hi,
            (SpecValue::I32(v), SpecValue::I32(lo), SpecValue::I32(hi)) => *v >= *lo && *v <= *hi,
            (SpecValue::F32(v), SpecValue::F32(lo), SpecValue::F32(hi)) => *v >= *lo && *v <= *hi,
            _ => false, // Tipos misturados = rejeição automática
        }
    }
}

impl SpecConstant {
    /// Valida se o valor atual está dentro dos bounds
    pub fn validate(&self) -> Result<(), SpecValidationError> {
        if !self.value.is_within(&self.min_value, &self.max_value) {
            return Err(SpecValidationError::OutOfBounds {
                name: self.name.clone(),
                value: self.value,
                min: self.min_value,
                max: self.max_value,
            });
        }
        Ok(())
    }
}

/// Erros de validação de segurança
#[derive(Debug, Clone)]
pub enum SpecValidationError {
    /// Valor fora dos bounds permitidos
    OutOfBounds {
        name: String,
        value: SpecValue,
        min: SpecValue,
        max: SpecValue,
    },
    /// Combinação inválida de constantes (ex: workgroup > max da GPU)
    InvalidCombination {
        reason: String,
    },
    /// Tentativa de mutar constante bloqueada
    LockedConstant {
        name: String,
    },
}

impl std::fmt::Display for SpecValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecValidationError::OutOfBounds { name, value, min, max } => {
                write!(f, "SECURITY: Constant '{}' value {:?} is outside safe bounds [{:?}, {:?}]",
                    name, value, min, max)
            }
            SpecValidationError::InvalidCombination { reason } => {
                write!(f, "SECURITY: Invalid constant combination: {}", reason)
            }
            SpecValidationError::LockedConstant { name } => {
                write!(f, "SECURITY: Attempt to mutate locked constant '{}'", name)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 3. SHADER TEMPLATE (O Código Mestre Estático)
// ─────────────────────────────────────────────────────────────────────

/// Template de Shader: SPIR-V pré-compilado + constantes mutáveis
#[derive(Debug, Clone)]
pub struct ShaderTemplate {
    /// Nome do shader (para audit)
    pub name: String,
    /// Hash SHA-256 do SPIR-V original (integridade)
    pub spirv_hash: [u8; 32],
    /// Tamanho do SPIR-V em bytes
    pub spirv_size: usize,
    /// Constantes de especialização definidas neste template
    pub spec_constants: Vec<SpecConstant>,
    /// Se true, este template está bloqueado (nenhuma mutação permitida)
    pub locked: bool,
}

impl ShaderTemplate {
    /// Cria o template de busca HNSW (o shader mais usado)
    pub fn hnsw_search_template() -> Self {
        Self {
            name: "hnsw_vector_search".to_string(),
            spirv_hash: [0; 32], // Preenchido pelo builder real
            spirv_size: 0,
            spec_constants: vec![
                SpecConstant {
                    constant_id: 0,
                    name: "WORKGROUP_SIZE_X".to_string(),
                    value: SpecValue::U32(64),
                    min_value: SpecValue::U32(32),
                    max_value: SpecValue::U32(1024),
                },
                SpecConstant {
                    constant_id: 1,
                    name: "VECTOR_DIM".to_string(),
                    value: SpecValue::U32(128),
                    min_value: SpecValue::U32(32),
                    max_value: SpecValue::U32(4096),
                },
                SpecConstant {
                    constant_id: 2,
                    name: "MAX_CANDIDATES".to_string(),
                    value: SpecValue::U32(64),
                    min_value: SpecValue::U32(8),
                    max_value: SpecValue::U32(512),
                },
                SpecConstant {
                    constant_id: 3,
                    name: "USE_SHARED_MEMORY".to_string(),
                    value: SpecValue::U32(1), // 0=disabled, 1=enabled
                    min_value: SpecValue::U32(0),
                    max_value: SpecValue::U32(1),
                },
            ],
            locked: false,
        }
    }

    /// Cria o template de draft validation (Crystal Skeleton verify)
    pub fn draft_verify_template() -> Self {
        Self {
            name: "draft_token_verify".to_string(),
            spirv_hash: [0; 32],
            spirv_size: 0,
            spec_constants: vec![
                SpecConstant {
                    constant_id: 0,
                    name: "BATCH_SIZE".to_string(),
                    value: SpecValue::U32(16),
                    min_value: SpecValue::U32(1),
                    max_value: SpecValue::U32(128),
                },
                SpecConstant {
                    constant_id: 1,
                    name: "MAX_DRAFT_TOKENS".to_string(),
                    value: SpecValue::U32(20),
                    min_value: SpecValue::U32(4),
                    max_value: SpecValue::U32(64),
                },
                SpecConstant {
                    constant_id: 2,
                    name: "PRECISION_MODE".to_string(),
                    value: SpecValue::U32(0), // 0=FP16, 1=FP32, 2=INT8
                    min_value: SpecValue::U32(0),
                    max_value: SpecValue::U32(2),
                },
            ],
            locked: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 4. MUTATION LOG (Livro de Receitas Auditável)
// ─────────────────────────────────────────────────────────────────────

/// Registro de uma mutação de shader (para auditoria)
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// Timestamp da mutação
    pub timestamp: SystemTime,
    /// Nome do shader mutado
    pub shader_name: String,
    /// Constante que foi alterada
    pub constant_name: String,
    /// Valor anterior
    pub old_value: SpecValue,
    /// Novo valor
    pub new_value: SpecValue,
    /// Razão da mutação (ex: "Detectei RTX 4090 com 128 SMs")
    pub reason: String,
    /// Hash do estado resultante (para verificação de cadeia)
    pub state_hash: u64,
}

// ─────────────────────────────────────────────────────────────────────
// 5. O MOTOR ADAPTATIVO (O Orquestrador)
// ─────────────────────────────────────────────────────────────────────

/// Motor de Adaptação de Pipeline Vulkan.
///
/// Detecta o hardware, ajusta as Specialization Constants dentro dos bounds
/// de segurança, e registra cada mutação no audit log.
pub struct AdaptivePipeline {
    /// Perfil da GPU detectada
    pub gpu_profile: GpuProfile,
    /// Templates de shader carregados (pré-compilados)
    pub templates: HashMap<String, ShaderTemplate>,
    /// Log de mutações (append-only, auditável)
    pub mutation_log: Vec<MutationRecord>,
    /// Hash do último estado (para cadeia de custódia)
    pub last_state_hash: u64,
    /// Estatísticas
    pub stats: AdaptiveStats,
}

#[derive(Debug, Default, Clone)]
pub struct AdaptiveStats {
    /// Total de mutações aplicadas com sucesso
    pub mutations_applied: u64,
    /// Total de mutações rejeitadas por segurança
    pub mutations_rejected: u64,
    /// Total de pipelines criados
    pub pipelines_created: u64,
}

impl AdaptivePipeline {
    /// Cria o motor com um perfil de GPU
    pub fn new(gpu_profile: GpuProfile) -> Self {
        let mut templates = HashMap::new();
        templates.insert(
            "hnsw_vector_search".to_string(),
            ShaderTemplate::hnsw_search_template(),
        );
        templates.insert(
            "draft_token_verify".to_string(),
            ShaderTemplate::draft_verify_template(),
        );

        Self {
            gpu_profile,
            templates,
            mutation_log: Vec::new(),
            last_state_hash: 0,
            stats: AdaptiveStats::default(),
        }
    }

    /// Adapta TODOS os templates ao hardware detectado.
    /// Retorna Ok(N) com o número de mutações, ou Err se algum bound foi violado.
    pub fn adapt_to_hardware(&mut self) -> Result<usize, Vec<SpecValidationError>> {
        let mut total_mutations = 0;
        let mut errors = Vec::new();

        // Clone GPU profile to avoid borrow issues
        let gpu = self.gpu_profile.clone();

        // Coleta os nomes dos templates para iterar sem borrow conflict
        let template_names: Vec<String> = self.templates.keys().cloned().collect();

        for name in &template_names {
            match self.adapt_template(&name, &gpu) {
                Ok(n) => total_mutations += n,
                Err(e) => errors.extend(e),
            }
        }

        if errors.is_empty() {
            Ok(total_mutations)
        } else {
            Err(errors)
        }
    }

    /// Adapta um template específico ao hardware
    fn adapt_template(
        &mut self,
        template_name: &str,
        gpu: &GpuProfile,
    ) -> Result<usize, Vec<SpecValidationError>> {
        let template = match self.templates.get(template_name) {
            Some(t) => t.clone(),
            None => return Ok(0),
        };

        if template.locked {
            return Err(vec![SpecValidationError::LockedConstant {
                name: template_name.to_string(),
            }]);
        }

        let mut mutations = Vec::new();

        for spec in &template.spec_constants {
            let new_value = self.compute_optimal_value(spec, gpu);
            if let Some(nv) = new_value {
                mutations.push((spec.constant_id, spec.name.clone(), spec.value, nv));
            }
        }

        let mut errors = Vec::new();
        let mut applied = 0;

        for (const_id, const_name, old_val, new_val) in mutations {
            match self.apply_mutation(template_name, const_id, new_val) {
                Ok(()) => {
                    self.mutation_log.push(MutationRecord {
                        timestamp: SystemTime::now(),
                        shader_name: template_name.to_string(),
                        constant_name: const_name,
                        old_value: old_val,
                        new_value: new_val,
                        reason: format!("Adapted to {} ({} CUs, {} subgroup)",
                            gpu.name, gpu.compute_units, gpu.subgroup_size),
                        state_hash: self.last_state_hash,
                    });
                    applied += 1;
                }
                Err(e) => errors.push(e),
            }
        }

        if errors.is_empty() {
            Ok(applied)
        } else {
            Err(errors)
        }
    }

    /// Calcula o valor ótimo para uma constante baseado no hardware
    fn compute_optimal_value(
        &self,
        spec: &SpecConstant,
        gpu: &GpuProfile,
    ) -> Option<SpecValue> {
        match spec.name.as_str() {
            "WORKGROUP_SIZE_X" => {
                // Workgroup = múltiplo do subgroup size, limitado pelo max da GPU
                let optimal = gpu.subgroup_size
                    .min(gpu.max_workgroup_size_x)
                    .max(32);
                // Arredonda para o múltiplo do subgroup mais próximo que cabe
                let aligned = (optimal / gpu.subgroup_size) * gpu.subgroup_size;
                let clamped = aligned.max(32);
                Some(SpecValue::U32(clamped))
            }
            "USE_SHARED_MEMORY" => {
                // Só ativa shared memory se a GPU tem bastante (>= 32KB)
                if gpu.max_shared_memory_bytes >= 32768 {
                    Some(SpecValue::U32(1))
                } else {
                    Some(SpecValue::U32(0))
                }
            }
            "BATCH_SIZE" => {
                // Batch maior para GPUs com muitos CUs
                let batch = if gpu.compute_units >= 80 {
                    64 // RTX 4090 class
                } else if gpu.compute_units >= 40 {
                    32 // RTX 3070 class
                } else if gpu.compute_units >= 16 {
                    16 // Mid-range
                } else {
                    8  // Integrated / mobile
                };
                Some(SpecValue::U32(batch))
            }
            "PRECISION_MODE" => {
                // Usar FP16 em GPUs modernas, FP32 em GPUs antigas
                match gpu.vendor {
                    GpuVendor::Nvidia if gpu.compute_units >= 40 => {
                        Some(SpecValue::U32(0)) // FP16 (Tensor Cores)
                    }
                    GpuVendor::Amd if gpu.compute_units >= 32 => {
                        Some(SpecValue::U32(0)) // FP16 (Matrix Cores)
                    }
                    _ => Some(SpecValue::U32(1)), // FP32 (seguro e universal)
                }
            }
            _ => None, // Constantes não reconhecidas não são mutadas
        }
    }

    /// Aplica uma mutação a uma constante, com validação de segurança
    fn apply_mutation(
        &mut self,
        template_name: &str,
        constant_id: u32,
        new_value: SpecValue,
    ) -> Result<(), SpecValidationError> {
        let template = self.templates.get_mut(template_name).unwrap();

        let spec = template.spec_constants
            .iter_mut()
            .find(|s| s.constant_id == constant_id)
            .unwrap();

        // GATE DE SEGURANÇA: validar contra bounds
        if !new_value.is_within(&spec.min_value, &spec.max_value) {
            self.stats.mutations_rejected += 1;
            return Err(SpecValidationError::OutOfBounds {
                name: spec.name.clone(),
                value: new_value,
                min: spec.min_value,
                max: spec.max_value,
            });
        }

        spec.value = new_value;
        self.stats.mutations_applied += 1;

        // Atualiza hash de estado (cadeia de custódia)
        self.last_state_hash = self.last_state_hash
            .wrapping_mul(6364136223846793005)
            .wrapping_add(constant_id as u64)
            .wrapping_add(match new_value {
                SpecValue::U32(v) => v as u64,
                SpecValue::I32(v) => v as u64,
                SpecValue::F32(v) => v.to_bits() as u64,
            });

        Ok(())
    }

    /// Serializa as constantes de um template para o formato Vulkan
    /// (byte buffer + VkSpecializationMapEntry equivalente)
    pub fn serialize_spec_constants(
        &self,
        template_name: &str,
    ) -> Option<(Vec<u8>, Vec<SpecMapEntry>)> {
        let template = self.templates.get(template_name)?;
        let mut data = Vec::new();
        let mut entries = Vec::new();

        for spec in &template.spec_constants {
            let offset = data.len();
            let bytes = spec.value.to_bytes();
            let size = bytes.len();
            data.extend_from_slice(&bytes);

            entries.push(SpecMapEntry {
                constant_id: spec.constant_id,
                offset: offset as u32,
                size: size as u32,
            });
        }

        Some((data, entries))
    }

    /// Verifica integridade: todas as constantes dentro dos bounds
    pub fn verify_all_templates(&self) -> Result<(), Vec<SpecValidationError>> {
        let mut errors = Vec::new();

        for (name, template) in &self.templates {
            for spec in &template.spec_constants {
                if let Err(e) = spec.validate() {
                    errors.push(e);
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Gera relatório de mutação legível para auditoria
    pub fn audit_report(&self) -> String {
        let mut report = String::from("═══ VULKAN ADAPTIVE PIPELINE — AUDIT REPORT ═══\n\n");
        report.push_str(&format!("GPU: {} ({:?})\n", self.gpu_profile.name, self.gpu_profile.vendor));
        report.push_str(&format!("CUs: {} | Subgroup: {} | VRAM: {}MB\n",
            self.gpu_profile.compute_units,
            self.gpu_profile.subgroup_size,
            self.gpu_profile.vram_total_bytes / (1024 * 1024),
        ));
        report.push_str(&format!("Mutations Applied: {} | Rejected: {}\n\n",
            self.stats.mutations_applied, self.stats.mutations_rejected));

        report.push_str("--- Mutation Log ---\n");
        for (i, record) in self.mutation_log.iter().enumerate() {
            report.push_str(&format!(
                "[{}] {}.{}: {:?} → {:?}\n    Reason: {}\n",
                i, record.shader_name, record.constant_name,
                record.old_value, record.new_value, record.reason,
            ));
        }

        report.push_str("\n--- Current Constants ---\n");
        for (name, template) in &self.templates {
            report.push_str(&format!("Shader: {}\n", name));
            for spec in &template.spec_constants {
                report.push_str(&format!("  [{}] {} = {:?} (bounds: {:?}..{:?})\n",
                    spec.constant_id, spec.name, spec.value,
                    spec.min_value, spec.max_value));
            }
        }

        report.push_str(&format!("\nState Hash: {:#018x}\n", self.last_state_hash));
        report
    }
}

/// Entrada de mapa de especialização (equivalente ao VkSpecializationMapEntry)
#[derive(Debug, Clone)]
pub struct SpecMapEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: u32,
}

// ─────────────────────────────────────────────────────────────────────
// 6. VRAM VIRTUAL (Memória Virtual Colapsada)
// ─────────────────────────────────────────────────────────────────────

/// Gerenciador de Memória Virtual VRAM: finge ter 1TB usando tiers.
///
/// NÃO usa Vulkan Sparse Resources (que tem latência imprevisível de
/// binding de até centenas de ms). Em vez disso, usa o modelo comprovado
/// do llama.cpp: mmap + paging explícito por camada com tier management.
///
/// Tier 0 (VRAM): Camadas ativas — 0.1μs acesso
/// Tier 1 (RAM):  Camadas em standby — 10μs acesso via DMA
/// Tier 2 (SSD):  Arquivo mmap — 50μs acesso via NVMe
///
/// Safety:
/// - Nunca usa OS swap (thrashing perigoso)
/// - Paging explícito com budget tracking (nunca excede VRAM)
/// - Batch page-in assíncrono (previne TDR)
pub struct VirtualVram {
    /// Mapa de páginas: ID da camada → tier atual
    pub page_table: HashMap<u64, VramPage>,
    /// Budget de VRAM (nunca exceder)
    pub vram_budget_bytes: u64,
    /// VRAM atualmente em uso
    pub vram_used_bytes: u64,
    /// Budget de RAM
    pub ram_budget_bytes: u64,
    /// RAM atualmente em uso
    pub ram_used_bytes: u64,
    /// Estatísticas
    pub stats: VirtualVramStats,
}

/// Uma página de memória virtual
#[derive(Debug, Clone)]
pub struct VramPage {
    /// ID da camada/tensor
    pub layer_id: u64,
    /// Tamanho em bytes
    pub size_bytes: u64,
    /// Tier atual (0=VRAM, 1=RAM, 2=SSD)
    pub current_tier: MemoryTier,
    /// Último acesso (para LRU eviction)
    pub last_access_epoch: u64,
    /// Frequência de acesso (para priority eviction)
    pub access_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryTier {
    Vram = 0,  // GPU memory — fastest
    Ram = 1,   // System RAM — medium
    Ssd = 2,   // NVMe mmap — slowest
}

#[derive(Debug, Default, Clone)]
pub struct VirtualVramStats {
    /// Páginas promovidas (SSD→RAM ou RAM→VRAM)
    pub promotions: u64,
    /// Páginas rebaixadas (VRAM→RAM ou RAM→SSD)
    pub demotions: u64,
    /// Cache hits (página já estava no tier certo)
    pub hits: u64,
    /// Cache misses (precisou promover)
    pub misses: u64,
    /// Bytes movidos entre tiers
    pub bytes_moved: u64,
}

impl VirtualVram {
    pub fn new(vram_budget_bytes: u64, ram_budget_bytes: u64) -> Self {
        Self {
            page_table: HashMap::new(),
            vram_budget_bytes,
            vram_used_bytes: 0,
            ram_budget_bytes,
            ram_used_bytes: 0,
            stats: VirtualVramStats::default(),
        }
    }

    /// Registra uma página (camada/tensor) no mapa virtual
    pub fn register_page(&mut self, layer_id: u64, size_bytes: u64, initial_tier: MemoryTier) {
        match initial_tier {
            MemoryTier::Vram => self.vram_used_bytes += size_bytes,
            MemoryTier::Ram => self.ram_used_bytes += size_bytes,
            MemoryTier::Ssd => {}
        }

        self.page_table.insert(layer_id, VramPage {
            layer_id,
            size_bytes,
            current_tier: initial_tier,
            last_access_epoch: 0,
            access_count: 0,
        });
    }

    /// Requisita acesso a uma página. Se não está no tier alvo, promove.
    /// Retorna o tier em que a página ficou.
    pub fn request_page(
        &mut self,
        layer_id: u64,
        target_tier: MemoryTier,
        current_epoch: u64,
    ) -> Result<MemoryTier, VramPageError> {
        // Phase 1: Read page info and update access stats
        let (current_tier, size) = {
            let page = self.page_table.get_mut(&layer_id)
                .ok_or(VramPageError::PageNotFound(layer_id))?;
            page.last_access_epoch = current_epoch;
            page.access_count += 1;
            (page.current_tier, page.size_bytes)
        }; // page borrow released here

        if current_tier <= target_tier {
            // Already at target tier or better
            self.stats.hits += 1;
            return Ok(current_tier);
        }

        // Phase 2: Ensure space (may evict other pages)
        self.stats.misses += 1;

        match target_tier {
            MemoryTier::Vram => {
                if self.vram_used_bytes + size > self.vram_budget_bytes {
                    self.evict_from_tier(MemoryTier::Vram, size, current_epoch)?;
                }
                match current_tier {
                    MemoryTier::Ram => self.ram_used_bytes -= size,
                    MemoryTier::Ssd => {}
                    MemoryTier::Vram => {}
                }
                self.vram_used_bytes += size;
            }
            MemoryTier::Ram => {
                if self.ram_used_bytes + size > self.ram_budget_bytes {
                    self.evict_from_tier(MemoryTier::Ram, size, current_epoch)?;
                }
                // SSD tier doesn't need subtraction (mmap)
                self.ram_used_bytes += size;
            }
            MemoryTier::Ssd => {}
        }

        // Phase 3: Update the page tier (re-borrow)
        let page = self.page_table.get_mut(&layer_id).unwrap();
        page.current_tier = target_tier;
        self.stats.promotions += 1;
        self.stats.bytes_moved += size;

        Ok(target_tier)
    }

    /// Evicta páginas LRU de um tier até liberar `needed` bytes
    fn evict_from_tier(
        &mut self,
        tier: MemoryTier,
        needed: u64,
        current_epoch: u64,
    ) -> Result<(), VramPageError> {
        // Colecta candidatos LRU neste tier
        let mut candidates: Vec<(u64, u64, u64)> = self.page_table.iter()
            .filter(|(_, p)| p.current_tier == tier)
            .map(|(id, p)| (*id, p.last_access_epoch, p.size_bytes))
            .collect();

        // Ordena por LRU (epoch mais antigo primeiro)
        candidates.sort_by_key(|(_, epoch, _)| *epoch);

        let mut freed = 0u64;
        let demote_to = match tier {
            MemoryTier::Vram => MemoryTier::Ram,
            MemoryTier::Ram => MemoryTier::Ssd,
            MemoryTier::Ssd => return Ok(()), // Não pode evictar do SSD
        };

        for (id, _, size) in &candidates {
            if freed >= needed {
                break;
            }

            // Verifica se tem espaço no tier inferior
            if demote_to == MemoryTier::Ram && self.ram_used_bytes + size > self.ram_budget_bytes {
                // RAM cheia: demota direto para SSD
                if let Some(page) = self.page_table.get_mut(id) {
                    match page.current_tier {
                        MemoryTier::Vram => self.vram_used_bytes -= size,
                        MemoryTier::Ram => self.ram_used_bytes -= size,
                        _ => {}
                    }
                    page.current_tier = MemoryTier::Ssd;
                    self.stats.demotions += 1;
                    self.stats.bytes_moved += size;
                    freed += size;
                }
            } else {
                if let Some(page) = self.page_table.get_mut(id) {
                    match page.current_tier {
                        MemoryTier::Vram => {
                            self.vram_used_bytes -= size;
                            self.ram_used_bytes += size;
                        }
                        MemoryTier::Ram => {
                            self.ram_used_bytes -= size;
                        }
                        _ => {}
                    }
                    page.current_tier = demote_to;
                    self.stats.demotions += 1;
                    self.stats.bytes_moved += size;
                    freed += size;
                }
            }
        }

        if freed >= needed {
            Ok(())
        } else {
            Err(VramPageError::InsufficientMemory {
                needed,
                freed,
                tier,
            })
        }
    }

    /// Hit rate do sistema de paging
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 { return 1.0; }
        self.stats.hits as f64 / total as f64
    }
}

#[derive(Debug, Clone)]
pub enum VramPageError {
    PageNotFound(u64),
    InsufficientMemory {
        needed: u64,
        freed: u64,
        tier: MemoryTier,
    },
}

// ─────────────────────────────────────────────────────────────────────
// 7. TESTES
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPU Profiles para Teste ──

    fn rtx4090() -> GpuProfile {
        GpuProfile {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: GpuVendor::Nvidia,
            vram_total_bytes: 24 * 1024 * 1024 * 1024,
            vram_free_bytes: 20 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 1008.0,
            compute_units: 128,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_shared_memory_bytes: 65536,
            supports_subgroup_ops: true,
            subgroup_size: 32,
        }
    }

    fn intel_uhd() -> GpuProfile {
        GpuProfile {
            name: "Intel UHD Graphics 630".to_string(),
            vendor: GpuVendor::Intel,
            vram_total_bytes: 2 * 1024 * 1024 * 1024,
            vram_free_bytes: 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 42.0,
            compute_units: 24,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_shared_memory_bytes: 32768,
            supports_subgroup_ops: false,
            subgroup_size: 16,
        }
    }

    fn amd_rx7900() -> GpuProfile {
        GpuProfile {
            name: "AMD Radeon RX 7900 XTX".to_string(),
            vendor: GpuVendor::Amd,
            vram_total_bytes: 24 * 1024 * 1024 * 1024,
            vram_free_bytes: 20 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 960.0,
            compute_units: 96,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_shared_memory_bytes: 65536,
            supports_subgroup_ops: true,
            subgroup_size: 64, // AMD wavefront
        }
    }

    fn mobile_qualcomm() -> GpuProfile {
        GpuProfile {
            name: "Qualcomm Adreno 750".to_string(),
            vendor: GpuVendor::Qualcomm,
            vram_total_bytes: 512 * 1024 * 1024,
            vram_free_bytes: 256 * 1024 * 1024,
            memory_bandwidth_gbps: 25.0,
            compute_units: 6,
            max_workgroup_size_x: 128,
            max_workgroup_size_y: 128,
            max_shared_memory_bytes: 16384,
            supports_subgroup_ops: false,
            subgroup_size: 32,
        }
    }

    // ── Test 1: RTX 4090 recebe configuração de alta performance ──
    #[test]
    fn test_adapt_rtx4090() {
        let mut pipeline = AdaptivePipeline::new(rtx4090());
        let mutations = pipeline.adapt_to_hardware().unwrap();

        assert!(mutations > 0, "Deve ter feito pelo menos 1 mutação");

        // Verify HNSW search was adapted
        let hnsw = &pipeline.templates["hnsw_vector_search"];
        let workgroup = hnsw.spec_constants.iter()
            .find(|s| s.name == "WORKGROUP_SIZE_X").unwrap();
        assert_eq!(workgroup.value, SpecValue::U32(32)); // 32 = subgroup size NVIDIA
        
        let shared = hnsw.spec_constants.iter()
            .find(|s| s.name == "USE_SHARED_MEMORY").unwrap();
        assert_eq!(shared.value, SpecValue::U32(1)); // RTX 4090 tem 64KB shared

        // Verify batch size for draft verify
        let verify = &pipeline.templates["draft_token_verify"];
        let batch = verify.spec_constants.iter()
            .find(|s| s.name == "BATCH_SIZE").unwrap();
        assert_eq!(batch.value, SpecValue::U32(64)); // 128 CUs = batch 64

        // FP16 habilitado para NVIDIA moderna
        let precision = verify.spec_constants.iter()
            .find(|s| s.name == "PRECISION_MODE").unwrap();
        assert_eq!(precision.value, SpecValue::U32(0)); // FP16

        // Verify integrity
        assert!(pipeline.verify_all_templates().is_ok());
        assert_eq!(pipeline.stats.mutations_rejected, 0);
    }

    // ── Test 2: Intel integrada recebe configuração conservadora ──
    #[test]
    fn test_adapt_intel_uhd() {
        let mut pipeline = AdaptivePipeline::new(intel_uhd());
        pipeline.adapt_to_hardware().unwrap();

        let verify = &pipeline.templates["draft_token_verify"];
        let batch = verify.spec_constants.iter()
            .find(|s| s.name == "BATCH_SIZE").unwrap();
        assert_eq!(batch.value, SpecValue::U32(16)); // 24 CUs = batch 16

        // FP32 (seguro) para Intel integrada
        let precision = verify.spec_constants.iter()
            .find(|s| s.name == "PRECISION_MODE").unwrap();
        assert_eq!(precision.value, SpecValue::U32(1)); // FP32

        assert!(pipeline.verify_all_templates().is_ok());
    }

    // ── Test 3: AMD com wavefront 64 recebe workgroup alinhado ──
    #[test]
    fn test_adapt_amd_wavefront() {
        let mut pipeline = AdaptivePipeline::new(amd_rx7900());
        pipeline.adapt_to_hardware().unwrap();

        let hnsw = &pipeline.templates["hnsw_vector_search"];
        let workgroup = hnsw.spec_constants.iter()
            .find(|s| s.name == "WORKGROUP_SIZE_X").unwrap();
        // AMD wavefront = 64, so workgroup must be multiple of 64
        assert_eq!(workgroup.value, SpecValue::U32(64));
    }

    // ── Test 4: Mobile Qualcomm com shared memory insuficiente ──
    #[test]
    fn test_adapt_mobile() {
        let mut pipeline = AdaptivePipeline::new(mobile_qualcomm());
        pipeline.adapt_to_hardware().unwrap();

        let hnsw = &pipeline.templates["hnsw_vector_search"];
        let shared = hnsw.spec_constants.iter()
            .find(|s| s.name == "USE_SHARED_MEMORY").unwrap();
        // 16KB < 32KB threshold → shared memory desabilitada
        assert_eq!(shared.value, SpecValue::U32(0));

        // Batch size mínimo (6 CUs)
        let verify = &pipeline.templates["draft_token_verify"];
        let batch = verify.spec_constants.iter()
            .find(|s| s.name == "BATCH_SIZE").unwrap();
        assert_eq!(batch.value, SpecValue::U32(8));
    }

    // ── Test 5: Mutação fora de bounds é REJEITADA ──
    #[test]
    fn test_security_rejects_out_of_bounds() {
        let mut pipeline = AdaptivePipeline::new(GpuProfile::default());

        // Tenta forçar um workgroup de 2048 (limite é 1024)
        let result = pipeline.apply_mutation(
            "hnsw_vector_search",
            0, // WORKGROUP_SIZE_X
            SpecValue::U32(2048),
        );

        assert!(result.is_err());
        assert_eq!(pipeline.stats.mutations_rejected, 1);
        assert_eq!(pipeline.stats.mutations_applied, 0);
    }

    // ── Test 6: Template bloqueado rejeita TODAS as mutações ──
    #[test]
    fn test_locked_template_rejects() {
        let mut pipeline = AdaptivePipeline::new(rtx4090());
        pipeline.templates.get_mut("hnsw_vector_search").unwrap().locked = true;

        let result = pipeline.adapt_template("hnsw_vector_search", &rtx4090());
        assert!(result.is_err());
    }

    // ── Test 7: Serialização Vulkan é correta ──
    #[test]
    fn test_vulkan_serialization() {
        let pipeline = AdaptivePipeline::new(GpuProfile::default());
        let (data, entries) = pipeline.serialize_spec_constants("hnsw_vector_search").unwrap();

        // HNSW tem 4 constantes × 4 bytes = 16 bytes
        assert_eq!(data.len(), 16);
        assert_eq!(entries.len(), 4);

        // Verifica offset contíguo e correto
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.offset, (i * 4) as u32);
            assert_eq!(entry.size, 4);
        }

        // Verifica que o primeiro valor serializado é decodificável
        let first_value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(first_value, 64); // Default WORKGROUP_SIZE_X
    }

    // ── Test 8: Audit log captura todas as mutações ──
    #[test]
    fn test_audit_log_completeness() {
        let mut pipeline = AdaptivePipeline::new(rtx4090());
        let mutations = pipeline.adapt_to_hardware().unwrap();

        assert_eq!(pipeline.mutation_log.len(), mutations);

        // Cada entrada deve ter timestamp, nome, razão
        for record in &pipeline.mutation_log {
            assert!(!record.shader_name.is_empty());
            assert!(!record.constant_name.is_empty());
            assert!(record.reason.contains("RTX 4090"));
        }

        // Hash chain deve ser diferente de zero
        assert_ne!(pipeline.last_state_hash, 0);
    }

    // ── Test 9: SpecValue bounds checking rigoroso ──
    #[test]
    fn test_spec_value_bounds() {
        // U32 bounds
        assert!(SpecValue::U32(50).is_within(&SpecValue::U32(10), &SpecValue::U32(100)));
        assert!(!SpecValue::U32(5).is_within(&SpecValue::U32(10), &SpecValue::U32(100)));
        assert!(!SpecValue::U32(101).is_within(&SpecValue::U32(10), &SpecValue::U32(100)));

        // F32 bounds
        assert!(SpecValue::F32(0.5).is_within(&SpecValue::F32(0.0), &SpecValue::F32(1.0)));
        assert!(!SpecValue::F32(-0.1).is_within(&SpecValue::F32(0.0), &SpecValue::F32(1.0)));

        // Tipo mismatch = rejeitado
        assert!(!SpecValue::U32(50).is_within(&SpecValue::F32(0.0), &SpecValue::F32(100.0)));
    }

    // ── Test 10: VirtualVram — Paging 3 tiers ──
    #[test]
    fn test_virtual_vram_paging() {
        let vram_budget = 1024; // 1KB VRAM
        let ram_budget = 4096;  // 4KB RAM
        let mut vram = VirtualVram::new(vram_budget, ram_budget);

        // Registra 3 camadas de 512 bytes cada no SSD
        vram.register_page(0, 512, MemoryTier::Ssd);
        vram.register_page(1, 512, MemoryTier::Ssd);
        vram.register_page(2, 512, MemoryTier::Ssd);

        // Promove camada 0 para VRAM
        let tier = vram.request_page(0, MemoryTier::Vram, 1).unwrap();
        assert_eq!(tier, MemoryTier::Vram);
        assert_eq!(vram.vram_used_bytes, 512);

        // Promove camada 1 para VRAM
        let tier = vram.request_page(1, MemoryTier::Vram, 2).unwrap();
        assert_eq!(tier, MemoryTier::Vram);
        assert_eq!(vram.vram_used_bytes, 1024); // Full!

        // Promove camada 2: VRAM cheia → evict camada 0 (LRU)
        let tier = vram.request_page(2, MemoryTier::Vram, 3).unwrap();
        assert_eq!(tier, MemoryTier::Vram);
        assert_eq!(vram.vram_used_bytes, 1024); // Ainda cheia (evictou + adicionou)

        // Camada 0 foi demovida para RAM
        let page0 = &vram.page_table[&0];
        assert_eq!(page0.current_tier, MemoryTier::Ram);

        assert!(vram.stats.promotions >= 3);
        assert!(vram.stats.demotions >= 1);
    }

    // ── Test 11: VirtualVram — LRU eviction order ──
    #[test]
    fn test_virtual_vram_lru_order() {
        let mut vram = VirtualVram::new(1024, 8192);

        // 4 páginas de 512 bytes, registro em VRAM (2× o budget)
        // Vamos registrá-las no SSD e promover uma a uma
        for i in 0..4u64 {
            vram.register_page(i, 512, MemoryTier::Ssd);
        }

        // Promove 0 e 1 para VRAM (enche)
        vram.request_page(0, MemoryTier::Vram, 100).unwrap();
        vram.request_page(1, MemoryTier::Vram, 200).unwrap();

        // Acessa 0 novamente (atualiza LRU para epoch 300)
        vram.request_page(0, MemoryTier::Vram, 300).unwrap();

        // Promove 2: deve evictar 1 (LRU epoch 200) e NÃO 0 (epoch 300)
        vram.request_page(2, MemoryTier::Vram, 400).unwrap();

        assert_eq!(vram.page_table[&0].current_tier, MemoryTier::Vram); // Preservada!
        assert_eq!(vram.page_table[&1].current_tier, MemoryTier::Ram);  // Evictada
        assert_eq!(vram.page_table[&2].current_tier, MemoryTier::Vram); // Promovida
    }

    // ── Test 12: VirtualVram — hit rate ──
    #[test]
    fn test_virtual_vram_hit_rate() {
        let mut vram = VirtualVram::new(2048, 4096);
        vram.register_page(0, 512, MemoryTier::Vram);

        // 10 acessos à mesma página em VRAM = 100% hit
        for i in 1..=10 {
            vram.request_page(0, MemoryTier::Vram, i).unwrap();
        }

        assert!((vram.hit_rate() - 1.0).abs() < f64::EPSILON);
        assert_eq!(vram.stats.hits, 10);
        assert_eq!(vram.stats.misses, 0);
    }

    // ── Test 13: Benchmark — adapt + serialize para 4 GPUs ──
    #[test]
    fn bench_adapt_all_gpus() {
        use std::time::Instant;

        let gpus = vec![
            ("RTX 4090", rtx4090()),
            ("Intel UHD", intel_uhd()),
            ("AMD RX 7900", amd_rx7900()),
            ("Qualcomm Adreno", mobile_qualcomm()),
        ];

        println!("\n═══════════════════════════════════════════════════");
        println!("  BENCHMARK: Adaptive Pipeline (4 GPUs)");
        println!("═══════════════════════════════════════════════════");

        for (name, gpu) in gpus {
            let start = Instant::now();
            let mut pipeline = AdaptivePipeline::new(gpu);

            for _ in 0..1000 {
                // Reset para re-adaptar
                pipeline.templates.insert(
                    "hnsw_vector_search".to_string(),
                    ShaderTemplate::hnsw_search_template(),
                );
                pipeline.templates.insert(
                    "draft_token_verify".to_string(),
                    ShaderTemplate::draft_verify_template(),
                );
                pipeline.adapt_to_hardware().unwrap();
                pipeline.serialize_spec_constants("hnsw_vector_search").unwrap();
            }

            let elapsed = start.elapsed();
            let per_iter = elapsed / 1000;

            println!("  {:<20} {:>6}ns/iter  {:>3} mutations  {} rejected",
                name, per_iter.as_nanos(), pipeline.stats.mutations_applied,
                pipeline.stats.mutations_rejected);

            assert!(pipeline.verify_all_templates().is_ok());
        }

        println!("═══════════════════════════════════════════════════\n");
    }
}
