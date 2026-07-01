//! Wrapper para o sistema de tokenização nativo (HuggingFace tokenizers).
//!
//! NodeStor delega BPE e decodificação universal para a biblioteca oficial,
//! mantendo o pipeline rápido e sem reinventar a roda complexa de acentos e ByteLevel.

use nodestor_core::NodeStorError;
use tokenizers::Tokenizer;
use std::str::FromStr;

pub struct TokenizerManager {
    inner: Tokenizer,
    /// Raw vocab from GGUF tokenizer.ggml.tokens — used as fallback when BPE decode fails.
    /// SmolLM2 and models without merges fall through to this for correct token→text mapping.
    raw_vocab: Vec<String>,
}

impl TokenizerManager {
    /// Carrega o tokenizer de uma string JSON (ex: tokenizer.json de um repo HuggingFace).
    pub fn from_string(json_content: &str) -> std::result::Result<Self, NodeStorError> {
        let inner = Tokenizer::from_str(json_content)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;

        Ok(Self { inner, raw_vocab: Vec::new() })
    }

    /// Constrói um tokenizer REAL a partir dos metadados embutidos no GGUF
    /// (`tokenizer.ggml.tokens` + `tokenizer.ggml.merges`).
    ///
    /// Monta um `tokenizer.json` BPE byte-level (estilo GPT-2/Llama-3/Qwen/SmolLM2)
    /// e o desserializa via a lib oficial — assim modelos reais decodificam para
    /// TEXTO coerente em vez do tokenizer dummy. `model_type` vem de
    /// `tokenizer.ggml.model` ("gpt2"/"llama"/...).
    pub fn from_gguf(
        tokens: &[String],
        merges: &[String],
        bos: Option<u32>,
        eos: Option<u32>,
        unk: Option<u32>,
    ) -> std::result::Result<Self, NodeStorError> {
        use std::collections::HashMap;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::AddedToken;

        if tokens.is_empty() {
            return Err(NodeStorError::InvalidModelFormat("GGUF sem vocab de tokenizer".into()));
        }

        let err = |m: String| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, m));

        // vocab: token_string → id (índice no array do GGUF)
        let vocab: HashMap<String, u32> = tokens.iter().enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        // merges "a b" → (a, b)
        let merges_pairs: Vec<(String, String)> = merges.iter()
            .filter_map(|m| {
                let mut it = m.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                    _ => None,
                }
            })
            .collect();

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges_pairs)
            .build()
            .map_err(|e| err(format!("BPE build: {}", e)))?;

        let mut inner = Tokenizer::new(bpe);
        // GPT-2/Llama-3/Qwen/SmolLM2 usam BPE byte-level: o ByteLevel é tanto
        // pré-tokenizador (encode) quanto decodificador (espaços → 'Ġ' e volta).
        let bl = ByteLevel::new(false, true, true);
        inner.with_pre_tokenizer(bl.clone());
        inner.with_decoder(bl);

        // Tokens especiais (bos/eos/unk) → reconhecidos no encode e puláveis no decode.
        let mut specials: Vec<AddedToken> = Vec::new();
        for id in [bos, eos, unk].into_iter().flatten() {
            if let Some(content) = tokens.get(id as usize) {
                specials.push(AddedToken::from(content.clone(), true));
            }
        }
        if !specials.is_empty() {
            inner.add_special_tokens(&specials);
        }

        Ok(Self { inner, raw_vocab: tokens.to_vec() })
    }

    /// Codifica uma string raw para Token IDs.
    pub fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, NodeStorError> {
        let enc = self.inner.encode(text, false)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
            
        Ok(enc.get_ids().to_vec())
    }

    /// Decodifica Token IDs para String (ideal para stream iterativo de volta pro usuário).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> std::result::Result<String, NodeStorError> {
        // Suppress BPE errors — fall through to raw_vocab when BPE fails or returns empty.
        // SmolLM2 and models whose BPE merges don't round-trip through the tokenizers lib
        // (byte-level tokens, unknown merge refs) hit this path.
        let text = self.inner.decode(ids, skip_special).unwrap_or_default();

        // Raw vocab fallback: direct array lookup + byte-level decoding.
        if text.is_empty() && !self.raw_vocab.is_empty() {
            let mut out = String::new();
            for &id in ids {
                if let Some(tok) = self.raw_vocab.get(id as usize) {
                    if skip_special && (tok.starts_with('<') && tok.ends_with('>')) {
                        continue;
                    }
                    // ByteLevel: Ġ → space, ▁ → space
                    let piece = tok.replace('Ġ', " ").replace('▁', " ");
                    // Hex byte escapes <0xNN> → actual byte
                    let piece = if piece.starts_with("<0x") && piece.ends_with('>') {
                        if let Ok(b) = u8::from_str_radix(&piece[3..piece.len()-1], 16) {
                            std::str::from_utf8(&[b]).unwrap_or("").to_string()
                        } else { piece }
                    } else { piece };
                    out.push_str(&piece);
                }
            }
            return Ok(out);
        }

        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_tokenizer_flow() {
        // JSON completo compatível com a API atual da lib tokenizers
        let dummy_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<unk>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "Hello": 1,
                    "World": 2
                },
                "unk_token": "<unk>"
            }
        }"#;

        let manager = TokenizerManager::from_string(dummy_json).unwrap();
        
        // Encode: WordLevel + Whitespace tokenizer divide por espaço
        let ids = manager.encode("Hello World").unwrap();
        assert_eq!(ids, vec![1, 2], "Hello=1, World=2");
        
        // Token fora do vocab → <unk> = 0
        let ids_unk = manager.encode("Hello Unknown").unwrap();
        assert_eq!(ids_unk[0], 1, "Hello deve ser token 1");
        assert_eq!(ids_unk[1], 0, "Unknown deve ser <unk> = 0");
        
        // Decode: skip_special=false mantém <unk> no output
        let text = manager.decode(&ids, false).unwrap();
        assert!(text.contains("Hello"), "Decode deve conter 'Hello'");
        assert!(text.contains("World"), "Decode deve conter 'World'");
    }

    #[test]
    fn test_invalid_json_errors() {
        let bad_json = r#"{ "broken": json "#;
        let result = TokenizerManager::from_string(bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_string_encode() {
        let dummy_json = r#"{
            "version": "1.0",
            "model": { "type": "WordLevel", "vocab": {"<unk>": 0}, "unk_token": "<unk>" }
        }"#;
        let manager = TokenizerManager::from_string(dummy_json).unwrap();
        let ids = manager.encode("").unwrap();
        assert_eq!(ids.len(), 0);
    }
}
