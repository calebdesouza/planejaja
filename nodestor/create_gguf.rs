use std::io::{Write, Result};
use byteorder::{LittleEndian, WriteBytesExt};

fn main() -> Result<()> {
    let path = "test.gguf";
    let mut f = std::fs::File::create(path)?;

    // Magic "GGUF" = 0x46554747
    f.write_u32::<LittleEndian>(0x46554747)?;
    // Version 3
    f.write_u32::<LittleEndian>(3)?;
    // 1 tensor
    f.write_u64::<LittleEndian>(1)?;
    // 2 metadata KVs
    f.write_u64::<LittleEndian>(2)?;

    // KV 1: general.architecture = "llama"
    write_gguf_string(&mut f, "general.architecture");
    f.write_u32::<LittleEndian>(8)?; // STRING type
    write_gguf_string(&mut f, "llama");

    // KV 2: general.name = "Test Llama"
    write_gguf_string(&mut f, "general.name");
    f.write_u32::<LittleEndian>(8)?;
    write_gguf_string(&mut f, "Test Llama");

    // Tensor 1: "token.embd" — shape [16, 16] — F32
    write_gguf_string(&mut f, "token.embd");
    f.write_u32::<LittleEndian>(2)?; // 2 dimensions
    f.write_u64::<LittleEndian>(16)?;
    f.write_u64::<LittleEndian>(16)?;
    f.write_u32::<LittleEndian>(0)?; // F32
    f.write_u64::<LittleEndian>(0)?; // offset 0

    // Dados fictícios dos tensores (zeros)
    let tensor_data = vec![0u8; 16 * 16 * 4 + 64]; // plus padding
    f.write_all(&tensor_data)?;

    Ok(())
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) {
    w.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    w.write_all(s.as_bytes()).unwrap();
}
