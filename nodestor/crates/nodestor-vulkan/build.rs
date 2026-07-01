use std::env;
use std::fs;
use std::path::Path;
use naga::front::glsl;
use naga::back::spv;

fn main() {
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir);

    let shader_dir = Path::new("shaders");
    if shader_dir.exists() {
        for entry in fs::read_dir(shader_dir).expect("Falha ao ler diretório de shaders") {
            let entry = entry.expect("Erro na entrada do diretório");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("comp") {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                // matmul_coop.comp usa GL_KHR_cooperative_matrix — não suportado pelo Naga.
                // O shader é carregado opcionalmente em runtime via `load_shader_by_kind(CoopMatrix)`.
                // Quando glslc/glslangValidator estiver disponível, compilar manualmente e ativar
                // a linha em `load_shader_by_kind` para incluir o .spv no binário.
                if name == "matmul_coop" || name == "tree_attention" || name == "turbo_quant_attention" || name == "moe_routing" || name == "fused_layernorm_gelu" {
                    println!("cargo:info={} shader excluído da compilação Naga (requer glslc + driver moderno)", name);
                    continue;
                }
                compile_shader(&path, dest_path);
            }
        }
    }
}

fn compile_shader(path: &Path, dest_dir: &Path) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { println!("cargo:warning=Não foi possível ler shader {:?}: {}", path, e); return; }
    };
    let name = path.file_stem().unwrap().to_str().unwrap();

    match try_compile_shader(name, &source, dest_dir) {
        Ok(()) => println!("cargo:info=Compilado (Naga): {}.spv", name),
        Err(e) => {
            println!("cargo:warning=Shader '{}' falhou no Naga: {} — gravando stub SPIR-V", name, e);
            write_stub_spirv(name, dest_dir);
        }
    }
}

fn try_compile_shader(name: &str, source: &str, dest_dir: &Path) -> Result<(), String> {
    let mut frontend = glsl::Frontend::default();
    let options = glsl::Options {
        stage: naga::ShaderStage::Compute,
        defines: Default::default(),
    };

    let module = frontend.parse(&options, source)
        .map_err(|e| format!("parse error em '{}': {:?}", name, e))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("validação falhou em '{}': {:?}", name, e))?;

    let spv_options = spv::Options::default();
    let binary = spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|e| format!("SPIR-V write error em '{}': {:?}", name, e))?;

    let dest_path = dest_dir.join(format!("{}.spv", name));
    let bytes: Vec<u8> = binary.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(dest_path, bytes).map_err(|e| format!("write error: {}", e))?;
    Ok(())
}

/// Grava um módulo SPIR-V mínimo válido (compute shader que não faz nada).
/// Isso permite que `include_bytes!` compile, mas o pipeline falhará ao criar
/// e o VulkanEngine cairá no fallback gracioso existente.
fn write_stub_spirv(name: &str, dest_dir: &Path) {
    const STUB_GLSL: &str = "#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n";
    let mut frontend = glsl::Frontend::default();
    let options = glsl::Options { stage: naga::ShaderStage::Compute, defines: Default::default() };
    if let Ok(module) = frontend.parse(&options, STUB_GLSL) {
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        ).validate(&module);
        if let Ok(info) = info {
            let spv_options = spv::Options::default();
            if let Ok(binary) = spv::write_vec(&module, &info, &spv_options, None) {
                let bytes: Vec<u8> = binary.iter().flat_map(|w| w.to_le_bytes()).collect();
                let dest_path = dest_dir.join(format!("{}.spv", name));
                let _ = fs::write(dest_path, bytes);
                return;
            }
        }
    }
    // Fallback de último recurso: SPIR-V magic + header mínimo (4 words)
    let dest_path = dest_dir.join(format!("{}.spv", name));
    let magic: Vec<u8> = [0x03u32, 0x02, 0x23, 0x07, 0x00, 0x01, 0x00, 0x00,
                           0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x00, 0x00, 0x00, 0x00].iter().flat_map(|&b: &u8| [b]).collect();
    let _ = fs::write(dest_path, magic);
}
