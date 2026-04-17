//! # nsz_convert — Conversor NSZ Lossless (CLI)
//!
//! Comprime um modelo GGUF para o formato NSZ (NodeStor Zip) sem perda matemática.
//!
//! Uso: cargo run --bin nsz_convert -- modelo.gguf
//!
//! Cada tensor do GGUF é analisado (entropia α-estável), comprimido via TCA-TBE,
//! e validado bit-a-bit antes de ser gravado no arquivo .nsz final.

use nodestor_formats::nsz_format::*;
use nodestor_formats::nsz_encoder::NszEncoder;
use nodestor_formats::nsz_decoder::{decode_tile_fp32, validate_lossless_fp32};
use nodestor_inference::entropy_analyzer::analyze_fp32_tensor;
use std::io::Write;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    NodeStor NSZ Converter — Compressão Lossless Neural      ║");
    println!("║    TCA-TBE (Tensor-Core-Aware Triple Bitmap Encoding)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Pegar caminho do modelo dos argumentos
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // Fallback para o micro modelo de teste
        "micro_modelo_ignition.gguf".to_string()
    };

    let output_path = model_path.replace(".gguf", ".nsz");

    println!("📥 Modelo de entrada: {}", model_path);
    println!("📤 Saída NSZ:         {}", output_path);
    println!();

    // 1. Parse do GGUF para extrair tensores
    let start = std::time::Instant::now();
    let gguf_data = match std::fs::read(&model_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ Erro ao ler '{}': {}", model_path, e);
            std::process::exit(1);
        }
    };
    println!("📦 Arquivo lido: {} bytes ({:.2} MB)", gguf_data.len(), gguf_data.len() as f64 / 1e6);

    // Parse simples do header GGUF para encontrar tensores
    if gguf_data.len() < 24 || &gguf_data[0..4] != b"GGUF" {
        eprintln!("❌ Arquivo não é GGUF válido");
        std::process::exit(1);
    }

    let version = u32::from_le_bytes(gguf_data[4..8].try_into().unwrap());
    let n_tensors = u64::from_le_bytes(gguf_data[8..16].try_into().unwrap()) as usize;
    let n_kv = u64::from_le_bytes(gguf_data[16..24].try_into().unwrap()) as usize;

    println!("📋 GGUF v{} | {} tensores | {} metadados", version, n_tensors, n_kv);
    println!();

    // Extrair info dos tensores do GGUF (parser simplificado)
    let mut cursor = 24usize;

    // Pular KV pairs
    for _ in 0..n_kv {
        if cursor + 8 > gguf_data.len() { break; }
        let key_len = u64::from_le_bytes(gguf_data[cursor..cursor+8].try_into().unwrap()) as usize;
        cursor += 8 + key_len;
        if cursor + 4 > gguf_data.len() { break; }
        let vtype = u32::from_le_bytes(gguf_data[cursor..cursor+4].try_into().unwrap());
        cursor += 4;
        cursor += skip_gguf_value(&gguf_data, cursor, vtype);
    }

    // Ler tensor info table
    struct TensorMeta {
        name: String,
        n_dims: u32,
        shape: Vec<u64>,
        dtype: u32,
        offset: u64,
    }
    let mut tensor_metas = Vec::new();

    for _ in 0..n_tensors {
        if cursor + 8 > gguf_data.len() { break; }
        let name_len = u64::from_le_bytes(gguf_data[cursor..cursor+8].try_into().unwrap()) as usize;
        cursor += 8;
        let name = String::from_utf8_lossy(&gguf_data[cursor..cursor+name_len]).to_string();
        cursor += name_len;

        let n_dims = u32::from_le_bytes(gguf_data[cursor..cursor+4].try_into().unwrap());
        cursor += 4;

        let mut shape = Vec::new();
        for _ in 0..n_dims {
            shape.push(u64::from_le_bytes(gguf_data[cursor..cursor+8].try_into().unwrap()));
            cursor += 8;
        }

        let dtype = u32::from_le_bytes(gguf_data[cursor..cursor+4].try_into().unwrap());
        cursor += 4;

        let offset = u64::from_le_bytes(gguf_data[cursor..cursor+8].try_into().unwrap());
        cursor += 8;

        tensor_metas.push(TensorMeta { name, n_dims, shape, dtype, offset });
    }

    // data_offset: alinhado a 32 bytes após o cursor
    let data_offset = (cursor + 31) & !31;

    println!("📊 Análise de Entropia:");
    println!("   {:<35} {:>8} {:>8} {:>8} {:>8}", "Tensor", "H(E)", "Modal%", "±1%", "α");
    println!("   {}", "─".repeat(75));

    let encoder = NszEncoder::new();
    let mut all_entries: Vec<NszTensorEntry> = Vec::new();
    let mut all_tile_data: Vec<Vec<u8>> = Vec::new();
    let mut total_original = 0u64;
    let mut total_compressed = 0u64;
    let mut total_validated = 0usize;

    for meta in &tensor_metas {
        // Só processamos FP32 (dtype 0) nesta versão
        if meta.dtype != 0 {
            println!("   {:<35} [SKIP: dtype={}]", meta.name, meta.dtype);
            continue;
        }

        let num_elements: u64 = if meta.shape.is_empty() { 1 } else { meta.shape.iter().product() };
        let byte_size = (num_elements * 4) as usize;
        let start_offset = data_offset + meta.offset as usize;
        let end_offset = start_offset + byte_size;

        if end_offset > gguf_data.len() {
            println!("   {:<35} [SKIP: offset fora do arquivo]", meta.name);
            continue;
        }

        // Converter bytes para f32
        let raw = &gguf_data[start_offset..end_offset];
        let weights: Vec<f32> = raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Análise de entropia
        let report = analyze_fp32_tensor(&weights);
        println!(
            "   {:<35} {:>6.2}b {:>6.1}% {:>6.1}% {:>6.2}",
            meta.name,
            report.exponent_entropy_bits,
            report.modal_concentration * 100.0,
            report.near_modal_concentration * 100.0,
            report.alpha_estimate,
        );

        // Codificar via TCA-TBE
        let tiles = encoder.encode_fp32(&weights);
        let mut tile_bytes: Vec<u8> = Vec::new();
        for tile in &tiles {
            tile_bytes.extend_from_slice(&NszTile::to_bytes(tile));
        }

        // Validação lossless rigorosa
        let mut decoded = Vec::new();
        for tile in &tiles {
            decoded.extend_from_slice(&decode_tile_fp32(tile));
        }
        let is_lossless = validate_lossless_fp32(&weights, &decoded);
        if !is_lossless {
            eprintln!("   ⚠️  FALHA DE VALIDAÇÃO LOSSLESS em '{}'!", meta.name);
            std::process::exit(1);
        }
        total_validated += 1;

        let original_size = byte_size as u64;
        let compressed_size = tile_bytes.len() as u64;
        total_original += original_size;
        total_compressed += compressed_size;

        let shape_arr = [
            meta.shape.get(0).copied().unwrap_or(0) as u32,
            meta.shape.get(1).copied().unwrap_or(0) as u32,
            meta.shape.get(2).copied().unwrap_or(0) as u32,
            meta.shape.get(3).copied().unwrap_or(0) as u32,
        ];

        all_entries.push(NszTensorEntry {
            name: meta.name.clone(),
            dtype: OriginalDtype::FP32,
            original_size,
            compressed_offset: 0, // ajustado abaixo
            compressed_size,
            num_tiles: tiles.len() as u32,
            num_weights: num_elements,
            shape: shape_arr,
        });

        all_tile_data.push(tile_bytes);
    }

    println!();
    println!("   {} tensores validados bit-exact ✅", total_validated);
    println!();

    // 2. Montar e gravar arquivo NSZ
    let header = NszFileHeader {
        magic: NSZ_MAGIC,
        version: NSZ_VERSION,
        flags: 0,
        num_tensors: all_entries.len() as u32,
        original_total_bytes: total_original,
        compressed_total_bytes: total_compressed,
    };

    // Calcular offsets dos tiles no arquivo
    let header_size = NszFileHeader::SIZE;
    let index_size = all_entries.len() * 128; // 128 bytes por entrada
    let data_start = align_up(header_size + index_size, ALIGNMENT);

    let mut current_offset = data_start as u64;
    for (i, entry) in all_entries.iter_mut().enumerate() {
        entry.compressed_offset = current_offset;
        current_offset += all_tile_data[i].len() as u64;
    }

    // Gravar
    let mut file = std::fs::File::create(&output_path).expect("Falha ao criar arquivo NSZ");

    // Header
    file.write_all(&NszFileHeader::to_bytes(&header)).unwrap();

    // Index table
    for entry in &all_entries {
        file.write_all(&NszTensorEntry::to_bytes(entry)).unwrap();
    }

    // Padding para alinhamento
    let current_pos = header_size + index_size;
    if current_pos < data_start {
        file.write_all(&vec![0u8; data_start - current_pos]).unwrap();
    }

    // Tile data
    for tile_data in &all_tile_data {
        file.write_all(tile_data).unwrap();
    }

    let elapsed = start.elapsed();
    let savings = if total_original > 0 {
        (1.0 - total_compressed as f64 / total_original as f64) * 100.0
    } else {
        0.0
    };

    println!("🔧 Resultado da Compressão:");
    println!("   Original:    {:>10} bytes ({:.2} MB)", total_original, total_original as f64 / 1e6);
    println!("   Comprimido:  {:>10} bytes ({:.2} MB)", total_compressed, total_compressed as f64 / 1e6);
    println!("   Economia:    {:>9.1}%", savings);
    println!("   Tempo:       {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("✅ '{}' criado com sucesso!", output_path);
    println!("   Validação bit-exact: ✓ PASSA ({} tensores, 0 divergências)", total_validated);
    println!();
}

/// Pula um valor GGUF com base no tipo.
fn skip_gguf_value(data: &[u8], pos: usize, vtype: u32) -> usize {
    match vtype {
        0 => 1,   // UINT8
        1 => 1,   // INT8
        2 => 2,   // UINT16
        3 => 2,   // INT16
        4 => 4,   // UINT32
        5 => 4,   // INT32
        6 => 4,   // FLOAT32
        7 => 1,   // BOOL
        8 => {    // STRING
            if pos + 8 > data.len() { return 0; }
            let len = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap()) as usize;
            8 + len
        }
        9 => {    // ARRAY
            if pos + 12 > data.len() { return 0; }
            let elem_type = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            let count = u64::from_le_bytes(data[pos+4..pos+12].try_into().unwrap()) as usize;
            let mut offset = 12;
            for _ in 0..count {
                offset += skip_gguf_value(data, pos + offset, elem_type);
            }
            offset
        }
        10 => 8,  // UINT64
        11 => 8,  // INT64
        12 => 8,  // FLOAT64
        _ => 0,
    }
}
