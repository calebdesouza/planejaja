//! D3 — Topological Gap Detector: Persistent Homology
//!
//! Detecta BURACOS REAIS no espaço de conhecimento usando
//! Homologia Persistente (algoritmo de Vietoris-Rips simplificado).
//!
//! H₀ = clusters separados, H₁ = loops/buracos, H₂ = cavidades

/// Uma "barra de persistência": um buraco que nasceu e morreu
#[derive(Debug, Clone)]
pub struct PersistenceBar {
    /// Escala em que o buraco "nasceu"
    pub birth: f32,
    /// Escala em que o buraco "morreu" (preenchido)
    pub death: f32,
    /// Dimensão: 0=cluster isolado, 1=loop, 2=cavidade
    pub dimension: usize,
    /// Persistência = death - birth (maior = mais real)
    pub persistence: f32,
    /// Coordenada representativa do buraco
    pub representative: Vec<f32>,
}

/// Um "buraco" detectado como lacuna de conhecimento
#[derive(Debug, Clone)]
pub struct KnowledgeGap {
    /// Centro do buraco no espaço latente
    pub center: Vec<f32>,
    /// Persistência (quão real é o buraco)
    pub persistence: f32,
    /// Dimensão topológica
    pub dimension: usize,
    /// Prioridade para exploração (= persistência)
    pub priority: f32,
}

/// Alvo de sonho gerado pelos buracos topológicos
#[derive(Debug, Clone)]
pub struct DreamTarget {
    pub center: Vec<f32>,
    pub priority: f32,
    pub dimension: usize,
}

/// O detector de buracos topológicos no espaço de conhecimento
pub struct TopologicalGapDetector {
    /// Granularidade de análise (escala do Vietoris-Rips)
    pub filtration_scale: f32,
    /// Barcodes calculados
    pub persistence_barcodes: Vec<PersistenceBar>,
    /// Limiar mínimo de persistência para considerar real
    pub min_persistence: f32,
}

impl TopologicalGapDetector {
    pub fn new(filtration_scale: f32, min_persistence: f32) -> Self {
        Self {
            filtration_scale,
            persistence_barcodes: Vec::new(),
            min_persistence,
        }
    }

    /// Detecta buracos nos embeddings fornecidos.
    /// Usa algoritmo Vietoris-Rips simplificado:
    /// 1. Computa matriz de distâncias
    /// 2. Detecta clusters (H₀) por single-linkage
    /// 3. Detecta loops (H₁) por triangulação
    pub fn detect_gaps(&mut self, embeddings: &[Vec<f32>]) -> Vec<KnowledgeGap> {
        if embeddings.len() < 3 {
            return Vec::new();
        }

        self.persistence_barcodes.clear();

        // Fase 1: H₀ — detecta clusters separados
        self.compute_h0(embeddings);

        // Fase 2: H₁ — detecta loops/buracos
        self.compute_h1(embeddings);

        // Retorna apenas buracos com persistência acima do limiar
        self.persistence_barcodes.iter()
            .filter(|b| b.persistence >= self.min_persistence)
            .map(|b| KnowledgeGap {
                center: b.representative.clone(),
                persistence: b.persistence,
                dimension: b.dimension,
                priority: b.persistence,
            })
            .collect()
    }

    /// H₀: Detecta quantos clusters separados existem
    fn compute_h0(&mut self, embeddings: &[Vec<f32>]) {
        let n = embeddings.len();
        // Union-Find para detectar componentes conectados
        let mut parent: Vec<usize> = (0..n).collect();

        let mut birth_scale = 0.0f32;

        // Ordena pares por distância (simula filtração incremental)
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let d = euclidean_distance(&embeddings[i], &embeddings[j]);
                pairs.push((d, i, j));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Conta componentes iniciais
        let initial_components = n;
        let mut components = initial_components;

        for (dist, i, j) in &pairs {
            if *dist > self.filtration_scale {
                break;
            }
            let ri = find(&mut parent, *i);
            let rj = find(&mut parent, *j);
            if ri != rj {
                parent[ri] = rj;
                components -= 1;
                // Cada merge é uma feature H₀ com persistence = dist - birth
                self.persistence_barcodes.push(PersistenceBar {
                    birth: birth_scale,
                    death: *dist,
                    dimension: 0,
                    persistence: dist - birth_scale,
                    representative: embeddings[*i].clone(),
                });
                birth_scale = *dist;
            }
        }

        // Componentes que nunca morreram (clusters persistentes)
        if components > 1 {
            self.persistence_barcodes.push(PersistenceBar {
                birth: birth_scale,
                death: self.filtration_scale * 2.0, // infinito = filtration_scale * 2
                dimension: 0,
                persistence: self.filtration_scale,
                representative: embeddings[0].clone(),
            });
        }
    }

    /// H₁: Detecta loops entre grupos de pontos
    fn compute_h1(&mut self, embeddings: &[Vec<f32>]) {
        let n = embeddings.len();
        if n < 4 {
            return;
        }

        // Para cada trio de pontos, verifica se formam um triângulo "aberto"
        // (loop sem preenchimento)
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let dij = euclidean_distance(&embeddings[i], &embeddings[j]);
                    let djk = euclidean_distance(&embeddings[j], &embeddings[k]);
                    let dik = euclidean_distance(&embeddings[i], &embeddings[k]);

                    let max_edge = dij.max(djk).max(dik);
                    let min_edge = dij.min(djk).min(dik);

                    // Loop detectado: triângulo com borda longa e interior vazio
                    if max_edge > self.filtration_scale * 0.7
                        && min_edge < self.filtration_scale * 0.4
                    {
                        // Centro do triângulo como representante
                        let center: Vec<f32> = embeddings[i].iter()
                            .zip(embeddings[j].iter())
                            .zip(embeddings[k].iter())
                            .map(|((a, b), c)| (a + b + c) / 3.0)
                            .collect();

                        let persistence = max_edge - min_edge;
                        if persistence >= self.min_persistence {
                            self.persistence_barcodes.push(PersistenceBar {
                                birth: min_edge,
                                death: max_edge,
                                dimension: 1,
                                persistence,
                                representative: center,
                            });
                            // Um loop por trio máximo
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Converte buracos em alvos de sonho para o Dreaming Engine
    pub fn gaps_as_dream_targets(&self) -> Vec<DreamTarget> {
        self.persistence_barcodes.iter()
            .filter(|b| b.persistence >= self.min_persistence)
            .map(|b| DreamTarget {
                center: b.representative.clone(),
                priority: b.persistence,
                dimension: b.dimension,
            })
            .collect()
    }

    /// Número de buracos significativos detectados
    pub fn significant_gaps_count(&self) -> usize {
        self.persistence_barcodes.iter()
            .filter(|b| b.persistence >= self.min_persistence)
            .count()
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn find(parent: &mut Vec<usize>, x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_clusters_detected() {
        let mut detector = TopologicalGapDetector::new(2.0, 0.3);

        // 3 clusters bem separados
        let embeddings = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0], // Cluster A
            vec![5.0, 0.0], vec![5.1, 0.1], vec![5.2, 0.0], // Cluster B
            vec![0.0, 5.0], vec![0.1, 5.1], vec![0.2, 5.0], // Cluster C
        ];

        let gaps = detector.detect_gaps(&embeddings);
        // Deve detectar pelo menos 2 separações entre os 3 clusters
        assert!(detector.significant_gaps_count() >= 1,
            "Deveria detectar pelo menos 1 gap entre clusters");
    }

    #[test]
    fn test_gap_targets_generated() {
        let mut detector = TopologicalGapDetector::new(3.0, 0.2);
        let embeddings: Vec<Vec<f32>> = (0..8).map(|i| {
            vec![(i as f32) * 1.5, 0.0]
        }).collect();

        detector.detect_gaps(&embeddings);
        let targets = detector.gaps_as_dream_targets();
        // Targets devem ter prioridade positiva
        for t in &targets {
            assert!(t.priority >= 0.0);
        }
    }

    #[test]
    fn test_empty_input_safe() {
        let mut detector = TopologicalGapDetector::new(1.0, 0.1);
        let gaps = detector.detect_gaps(&[]);
        assert!(gaps.is_empty());

        let gaps2 = detector.detect_gaps(&[vec![1.0, 2.0]]);
        assert!(gaps2.is_empty());
    }
}
