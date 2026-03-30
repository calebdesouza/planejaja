use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Hardware Detection Types
// ─────────────────────────────────────────────

/// GPU vendor detectado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    #[default]
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Geração do barramento NVMe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum NvmeGen {
    Gen1,
    Gen2,
    Gen3,
    Gen4,
    Gen5,
    #[default]
    Unknown,
}

impl std::fmt::Display for NvmeGen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvmeGen::Gen1 => write!(f, "Gen 1 (~500 MB/s)"),
            NvmeGen::Gen2 => write!(f, "Gen 2 (~1.5 GB/s)"),
            NvmeGen::Gen3 => write!(f, "Gen 3 (~3.5 GB/s)"),
            NvmeGen::Gen4 => write!(f, "Gen 4 (~7 GB/s)"),
            NvmeGen::Gen5 => write!(f, "Gen 5 (~14 GB/s)"),
            NvmeGen::Unknown => write!(f, "Desconhecido"),
        }
    }
}

/// Capacidades detectadas de uma GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// Vendor da GPU.
    pub vendor: GpuVendor,
    /// Nome do modelo da GPU (ex: "NVIDIA GeForce RTX 4090").
    pub device_name: String,
    /// VRAM total em bytes.
    pub vram_bytes: u64,
    /// Suporte a compute shaders via Vulkan.
    pub supports_vulkan_compute: bool,
    /// Suporte a VK_NV_cooperative_matrix2 (NVIDIA, driver 575+).
    pub supports_cooperative_matrix2: bool,
    /// Suporte a VK_KHR_cooperative_matrix (cross-vendor).
    pub supports_cooperative_matrix_khr: bool,
    /// Suporte a BF16 (importante para inferência moderna).
    pub supports_bfloat16: bool,
    /// Geração PCIe (ex: 3, 4, 5).
    pub pcie_gen: u32,
    /// Número de lanes PCIe (ex: 16).
    pub pcie_lanes: u32,
    /// Se Resizable BAR está ativado.
    pub resizable_bar_enabled: bool,
    /// Versão do driver instalada.
    pub driver_version: String,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            device_name: "CPU Fallback".to_string(),
            vram_bytes: 0,
            supports_vulkan_compute: false,
            supports_cooperative_matrix2: false,
            supports_cooperative_matrix_khr: false,
            supports_bfloat16: false,
            pcie_gen: 0,
            pcie_lanes: 0,
            resizable_bar_enabled: false,
            driver_version: "N/A".to_string(),
        }
    }
}

/// Informação de um dispositivo de armazenamento detectado.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Caminho do dispositivo ou ponto de montagem.
    pub path: String,
    /// Nome/modelo do disco.
    pub name: String,
    /// Geração NVMe estimada.
    pub nvme_gen: NvmeGen,
    /// Capacidade total em bytes.
    pub total_bytes: u64,
    /// Espaço disponível em bytes.
    pub available_bytes: u64,
    /// Se é um SSD (vs HDD).
    pub is_ssd: bool,
    /// Taxa de leitura sequencial estimada em bytes/s.
    pub estimated_read_bps: u64,
}

/// Sistema operacional do host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OsType {
    Windows,
    Linux,
    MacOs,
    Unknown,
}

impl std::fmt::Display for OsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OsType::Windows => write!(f, "Windows"),
            OsType::Linux => write!(f, "Linux"),
            OsType::MacOs => write!(f, "macOS"),
            OsType::Unknown => write!(f, "Desconhecido"),
        }
    }
}

/// Backend de transporte de dados selecionado pelo scanner.
///
/// Hierarquia de prioridades (do mais rápido ao mais compatível):
/// 1. `NvidiaGds`      — NVIDIA GDS 2.0 (Linux) — ~28 GB/s
/// 2. `RocmDirectGma`  — AMD ROCm (Linux)        — ~20 GB/s
/// 3. `DirectStorage`  — Windows DS 1.4           — ~14 GB/s
/// 4. `IoUringDmabuf`  — Linux kernel 6.16+       — ~20 GB/s
/// 5. `IoUringStandard`— Linux kernel 5.11+       — ~7 GB/s
/// 6. `Win32Fallback`  — Windows antigo / sem DS  — ~5 GB/s
/// 7. `VulkanGeneric`  — Mac/Intel/CPU            — ~3-6 GB/s
/// 8. `PreadFallback`  — Qualquer máquina (base)  — ~1.5 GB/s
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportBackend {
    /// NVIDIA GPUDirect Storage 2.0 via cuFile (Linux).
    NvidiaGds,
    /// AMD ROCm com DirectGMA (Linux).
    RocmDirectGma,
    /// Microsoft DirectStorage 1.4 (Windows, requer DLLs).
    DirectStorage,
    /// Linux io_uring + DMABUF zero-copy (kernel 6.16+).
    IoUringDmabuf,
    /// Linux io_uring padrão (kernel 5.11+).
    IoUringStandard,
    /// Windows Win32 Overlapped I/O com alinhamento de setor (sem DirectStorage).
    Win32Fallback,
    /// Vulkan Compute + mmap: macOS, Intel iGPU, máquinas sem GPU dedicada.
    VulkanGeneric,
    /// Fallback universal com pread64/ReadFile — funciona em qualquer hardware.
    PreadFallback,
}

impl std::fmt::Display for TransportBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportBackend::NvidiaGds      => write!(f, "NVIDIA GPUDirect Storage 2.0"),
            TransportBackend::RocmDirectGma  => write!(f, "AMD ROCm + DirectGMA"),
            TransportBackend::DirectStorage  => write!(f, "DirectStorage 1.4 (Windows)"),
            TransportBackend::IoUringDmabuf  => write!(f, "io_uring + DMABUF (kernel 6.16+)"),
            TransportBackend::IoUringStandard=> write!(f, "io_uring padrão (kernel 5.11+)"),
            TransportBackend::Win32Fallback  => write!(f, "Win32 Overlapped I/O (sem DirectStorage)"),
            TransportBackend::VulkanGeneric  => write!(f, "Vulkan Generic (Mac/Intel/Universal)"),
            TransportBackend::PreadFallback  => write!(f, "Fallback Universal (pread — qualquer máquina)"),
        }
    }
}

/// Perfil completo do hardware do sistema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Lista de GPUs detectadas.
    pub gpus: Vec<GpuCapabilities>,
    /// Lista de dispositivos de armazenamento detectados.
    pub storage: Vec<StorageInfo>,
    /// Sistema operacional.
    pub os: OsType,
    /// Versão do kernel (Linux) ou OS (Windows).
    pub os_version: String,
    /// Backend de transporte recomendado.
    pub recommended_transport: TransportBackend,
    /// Número de núcleos de CPU.
    pub cpu_cores: usize,
    /// RAM total do sistema em bytes.
    pub total_ram_bytes: u64,
    /// Motivos pelos quais backends mais rápidos foram ignorados.
    pub missed_optimizations: Vec<String>,
}

impl HardwareProfile {
    /// Retorna a GPU primária, se disponível.
    pub fn primary_gpu(&self) -> Option<&GpuCapabilities> {
        self.gpus.first()
    }

    /// Retorna o armazenamento com mais espaço disponível.
    pub fn best_storage(&self) -> Option<&StorageInfo> {
        self.storage.iter().max_by_key(|s| s.available_bytes)
    }

    /// Throughput máximo estimado do transporte selecionado em bytes/s.
    pub fn estimated_transport_throughput(&self) -> u64 {
        match self.recommended_transport {
            TransportBackend::NvidiaGds       => 28_000_000_000, // 28 GB/s
            TransportBackend::RocmDirectGma   => 20_000_000_000, // 20 GB/s
            TransportBackend::DirectStorage   => 14_000_000_000, // 14 GB/s
            TransportBackend::IoUringDmabuf   => 20_000_000_000, // 20 GB/s
            TransportBackend::IoUringStandard =>  7_000_000_000, //  7 GB/s
            TransportBackend::Win32Fallback   =>  5_000_000_000, //  5 GB/s
            TransportBackend::VulkanGeneric   =>  3_500_000_000, //  3.5 GB/s
            TransportBackend::PreadFallback   =>  1_500_000_000, //  1.5 GB/s
        }
    }
}

// ─────────────────────────────────────────────
// Model Format Types
// ─────────────────────────────────────────────

/// Formatos de modelo suportados.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Unknown,
}

/// Tipo de dado de um tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    F32,
    F16,
    BF16,
    /// GGML Q4_0 (4-bit quantizado)
    Q4_0,
    /// GGML Q4_1
    Q4_1,
    /// GGML Q5_0
    Q5_0,
    /// GGML Q5_1
    Q5_1,
    /// GGML Q8_0
    Q8_0,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bool,
}

impl TensorDtype {
    /// Retorna o tamanho em bytes por elemento (None para tipos quantizados).
    pub fn element_size(&self) -> Option<f32> {
        match self {
            TensorDtype::F64 | TensorDtype::I64 => Some(8.0),
            TensorDtype::F32 | TensorDtype::I32 => Some(4.0),
            TensorDtype::F16 | TensorDtype::BF16 | TensorDtype::I16 => Some(2.0),
            TensorDtype::I8 | TensorDtype::Bool => Some(1.0),
            TensorDtype::Q8_0 => Some(1.0625), // (256 * 8 + 4) / 256
            TensorDtype::Q4_0 | TensorDtype::Q4_1 => Some(0.5625),
            TensorDtype::Q5_0 | TensorDtype::Q5_1 => Some(0.6875),
        }
    }

    /// Retorna nome legível.
    pub fn name(&self) -> &'static str {
        match self {
            TensorDtype::F32 => "F32",
            TensorDtype::F16 => "F16",
            TensorDtype::BF16 => "BF16",
            TensorDtype::Q4_0 => "Q4_0",
            TensorDtype::Q4_1 => "Q4_1",
            TensorDtype::Q5_0 => "Q5_0",
            TensorDtype::Q5_1 => "Q5_1",
            TensorDtype::Q8_0 => "Q8_0",
            TensorDtype::I8 => "I8",
            TensorDtype::I16 => "I16",
            TensorDtype::I32 => "I32",
            TensorDtype::I64 => "I64",
            TensorDtype::F64 => "F64",
            TensorDtype::Bool => "Bool",
        }
    }
}

/// Metadados de um tensor dentro de um arquivo de modelo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Nome do tensor (ex: "token_embd.weight").
    pub name: String,
    /// Dimensões do tensor (ex: [32000, 4096]).
    pub shape: Vec<u64>,
    /// Tipo de dado.
    pub dtype: TensorDtype,
    /// Offset em bytes dentro do arquivo (após o header).
    pub data_offset: u64,
    /// Tamanho em bytes dos dados.
    pub data_size: u64,
}

impl TensorInfo {
    /// Número total de elementos.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Metadados completos de um modelo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Formato do modelo.
    pub format: ModelFormat,
    /// Nome do modelo (ex: "LLaMA-3.1-70B-Instruct").
    pub model_name: Option<String>,
    /// Arquitetura (ex: "llama").
    pub architecture: Option<String>,
    /// Número de parâmetros estimado.
    pub param_count: Option<u64>,
    /// Lista de tensores com seus offsets.
    pub tensors: Vec<TensorInfo>,
    /// Offset onde os dados de tensores começam no arquivo.
    pub data_offset: u64,
    /// Tamanho total do arquivo em bytes.
    pub file_size: u64,
    /// Metadados extras (KV pairs do GGUF ou similares).
    pub extra: serde_json::Value,
}

impl ModelMetadata {
    /// Tamanho total dos dados de tensores em bytes.
    pub fn total_tensor_size(&self) -> u64 {
        self.tensors.iter().map(|t| t.data_size).sum()
    }

    /// Número de tensores.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ─────────────────────────────────────────────
// Transfer Types
// ─────────────────────────────────────────────

/// Requisição de transferência de um bloco de dados.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    /// Offset em bytes no arquivo de onde ler.
    pub file_offset: u64,
    /// Quantidade de bytes a ler.
    pub size: usize,
    /// Se os dados estão comprimidos (Zstd/LZ4).
    pub compressed: bool,
}

/// Dica de compressão/formato para o motor de streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionHint {
    None,
    Lz4,
    Zstd,
    GgmlQ4,
    GgmlQ8,
    GDeflate,      // Microsoft DirectStorage lossless
    ZstdLossless,  // NodeStor custom lossless
}

impl CompressionHint {
    pub fn decompress_on_gpu(&self) -> bool {
        match self {
            Self::GgmlQ4 | Self::GgmlQ8 | Self::GDeflate | Self::ZstdLossless => true,
            _ => false,
        }
    }
}

/// Requisição para streaming 'Líquido' (Micro-Fatiado).
pub struct LiquidTransferRequest {
    pub file_path: String,
    pub file_offset: u64,
    pub tensor_name: String,
    pub total_size: usize,
    pub chunk_size: usize,
    pub compression: CompressionHint,
    /// Dica para o motor disparar a próxima carga antecipadamente.
    pub look_ahead_hint: bool,
}

/// Resultado de uma transferência.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Dados lidos (após descompressão se aplicável).
    pub data: Vec<u8>,
    /// Duração da operação em microssegundos.
    pub duration_us: u64,
    /// Throughput medido em bytes/segundo.
    pub throughput_bps: f64,
}

impl TransferResult {
    /// Cria um resultado de transferência calculando o throughput.
    pub fn new(data: Vec<u8>, duration_us: u64) -> Self {
        let throughput_bps = if duration_us > 0 {
            (data.len() as f64) / (duration_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        Self { data, duration_us, throughput_bps }
    }

    /// Throughput em GB/s.
    pub fn throughput_gbs(&self) -> f64 {
        self.throughput_bps / 1_000_000_000.0
    }
}
