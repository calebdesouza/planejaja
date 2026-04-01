//! Wrapper para o sistema de tokenização nativo (HuggingFace tokenizers).
//!
//! NodeStor delega BPE e decodificação universal para a biblioteca oficial,
//! mantendo o pipeline rápido e sem reinventar a roda complexa de acentos e ByteLevel.

use nodestor_core::NodeStorError;
use tokenizers::Tokenizer;
use std::str::FromStr;

pub struct TokenizerManager {
    inner: Tokenizer,
}

impl TokenizerManager {
    /// Carrega o tokenizer de uma string JSON (ex: tokenizer.json de um repo HuggingFace).
    pub fn from_string(json_content: &str) -> std::result::Result<Self, NodeStorError> {
        let inner = Tokenizer::from_str(json_content)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
        
        Ok(Self { inner })
    }

    /// Codifica uma string raw para Token IDs.
    pub fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, NodeStorError> {
        let enc = self.inner.encode(text, false)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
            
        Ok(enc.get_ids().to_vec())
    }

    /// Decodifica Token IDs para String (ideal para stream iterativo de volta pro usuário).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> std::result::Result<String, NodeStorError> {
        let text = self.inner.decode(ids, skip_special)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| NodeStorError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())))?;
            
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_tokenizer_flow() {
        // Criamos um vocabulário JSON dummy mínimo para validar que o binding está compilando e operando.
        // Simulando BPE minúsculo
        let dummy_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
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
        
        // Encode
        let ids = manager.encode("Hello World Hello!").unwrap();
        assert_eq!(ids, vec![1, 2, 1, 0]); // Hello = 1, World = 2, Hello = 1, ! = 0 (unk)
        
        // Decode
        let text = manager.decode(&ids, true).unwrap();
        assert_eq!(text.trim(), "Hello World Hello <unk>");
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
