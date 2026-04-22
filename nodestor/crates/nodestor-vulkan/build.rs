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
                if name == "matmul_coop" || name == "tree_attention" || name == "turbo_quant_attention" || name == "moe_routing" {
                    println!("cargo:info={} shader excluído da compilação Naga (requer glslc + driver moderno)", name);
                    continue;
                }
                compile_shader(&path, dest_path);
            }
        }
    }
}

fn compile_shader(path: &Path, dest_dir: &Path) {
    let source = fs::read_to_string(path).expect("Falha ao ler arquivo de shader");
    let name = path.file_stem().unwrap().to_str().unwrap();

    let mut frontend = glsl::Frontend::default();
    let options = glsl::Options {
        stage: naga::ShaderStage::Compute,
        defines: Default::default(),
    };
    
    let module = frontend.parse(&options, &source)
        .expect("Falha ao fazer o parser GLSL com Naga");

    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .expect("Shader inválido detectado pelo Naga");

    let spv_options = spv::Options::default();
    let binary = spv::write_vec(&module, &info, &spv_options, None)
        .expect("Falha ao converter para SPIR-V com Naga");

    let dest_path = dest_dir.join(format!("{}.spv", name));
    let bytes: Vec<u8> = binary.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(dest_path, bytes).expect("Falha ao gravar SPIR-V");
    println!("cargo:info=Compilado (Naga): {}.spv", name);
}
