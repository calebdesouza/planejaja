use std::collections::HashMap;

/// NodeStor COBER v2 - Subsistema 3: Esqueleto Sintático
///
/// "Cérebro Reserva" focado apenas na estrutura da linguagem.
/// Não sugere palavras exatas, mas sim [Moldes Gamatiacais].
/// Exemplo: em vez de sugerir "gato", sugere [Substantivo].
/// Se a IA está em modo preenchimento (criatividade alta), os rascunhos são
/// gerados em cima desses moldes, mantendo a coerência estrutural e 
/// colidindo perfeitamente com a verificação da GPU.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrammarCategory {
    Article,
    Noun,
    Verb,
    Adjective,
    Adverb,
    Preposition,
    Conjunction,
    Punctuation,
    Unknown,
}

/// Um "Molde" é uma sequência de categorias. Ex: [Article, Noun, Verb]
pub type SyntacticTemplate = Vec<GrammarCategory>;

pub struct SyntacticSkeleton {
    /// Probabilidades de transição de Markov (Categoria atual -> Próximas categorias possíveis)
    transitions: HashMap<GrammarCategory, Vec<(GrammarCategory, u32)>>,
    /// Dicionário minimalista que mapeia alguns tokens comuns para sua categoria (Mock/Exemplo)
    /// Em um sistema real, isso usaria hashes ou embeds simplificados
    lexicon: HashMap<u32, GrammarCategory>,
}

impl SyntacticSkeleton {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            lexicon: HashMap::new(),
        }
    }

    /// Treina as transições (HMM n-gram) lendo templates de exemplo
    pub fn train_from_templates(&mut self, templates: &[SyntacticTemplate]) {
        for template in templates {
            for window in template.windows(2) {
                let current = window[0];
                let next = window[1];
                let entry = self.transitions.entry(current).or_insert_with(Vec::new);
                
                let mut found = false;
                for (cat, count) in entry.iter_mut() {
                    if *cat == next {
                        *count += 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    entry.push((next, 1));
                }
            }
        }

        // Ordenar por probabilidade (contagem)
        for entry in self.transitions.values_mut() {
            entry.sort_by(|a, b| b.1.cmp(&a.1));
        }
    }

    pub fn add_lexicon(&mut self, token_id: u32, category: GrammarCategory) {
        self.lexicon.insert(token_id, category);
    }

    pub fn get_category(&self, token_id: u32) -> GrammarCategory {
        *self.lexicon.get(&token_id).unwrap_or(&GrammarCategory::Unknown)
    }

    /// Dado o último token, prevê as próximas N categorias prováveis (o "Molde")
    pub fn predict_template(&self, last_token: u32, steps: usize) -> Vec<GrammarCategory> {
        let mut template = Vec::new();
        let mut current_cat = self.get_category(last_token);

        for _ in 0..steps {
            if let Some(next_cats) = self.transitions.get(&current_cat) {
                if !next_cats.is_empty() {
                    // Pega a transição mais provável (Ganancioso)
                    current_cat = next_cats[0].0;
                    template.push(current_cat);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        template
    }

    /// Valida se uma sequência de tokens gerada "caixinha de areia" obedece ao molde
    pub fn conforms_to_template(&self, tokens: &[u32], template: &[GrammarCategory]) -> bool {
        if tokens.len() > template.len() {
            return false;
        }
        for (i, &token) in tokens.iter().enumerate() {
            let cat = self.get_category(token);
            if cat != GrammarCategory::Unknown && cat != template[i] {
                // Se sabemos a classe e não bate, rejeitamos o draft
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntactic_skeleton() {
        let mut skeleton = SyntacticSkeleton::new();
        
        // Exemplo: O gato subiu -> Article, Noun, Verb
        skeleton.train_from_templates(&[
            vec![GrammarCategory::Article, GrammarCategory::Noun, GrammarCategory::Verb]
        ]);

        // Registrar léxico
        skeleton.add_lexicon(10, GrammarCategory::Article); // O
        skeleton.add_lexicon(20, GrammarCategory::Noun);    // gato
        skeleton.add_lexicon(30, GrammarCategory::Verb);    // subiu
        skeleton.add_lexicon(40, GrammarCategory::Adjective); // azul

        // Se a ultima palavra foi "O" (10 -> Article), deve prever Noun, Verb
        let template = skeleton.predict_template(10, 2);
        assert_eq!(template, vec![GrammarCategory::Noun, GrammarCategory::Verb]);

        // "gato subiu" (20, 30) -> [Noun, Verb] -> conforms!
        assert!(skeleton.conforms_to_template(&[20, 30], &template));

        // "azul subiu" (40, 30) -> [Adjective, Verb], expected [Noun, Verb] -> fail!
        assert!(!skeleton.conforms_to_template(&[40, 30], &template));
    }
}
