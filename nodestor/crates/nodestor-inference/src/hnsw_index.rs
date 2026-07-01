/// HNSW Index — Hierarchical Navigable Small World com SIMD AVX2
///
/// Substituição do FlatVectorIndex O(n) por busca O(log n) com recall >98%.
/// Para contexto infinito com milhões de blocos KV arquivados, é obrigatório.
///
/// Referência: Malkov & Yashunin, 2018.
///
/// Parâmetros padrão:
///   M  = 16  — conexões por nó (níveis > 0)
///   M0 = 32  — conexões na camada 0 (mais densa)
///   ef_construction = 200  — candidatos durante inserção
///   ef_search = 50         — candidatos durante busca
///
/// SIMD: AVX2 (8 × f32 por ciclo) com fallback escalar automático.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
};

// ─── SIMD dot product ─────────────────────────────────────────────────────────

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n      = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let hi   = _mm256_extractf128_ps(acc, 1);
    let lo   = _mm256_castps256_ps128(acc);
    let s128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(s128);
    let s64  = _mm_add_ps(s128, shuf);
    let s32  = _mm_add_ss(s64, _mm_movehl_ps(shuf, s64));
    let mut result = 0.0f32;
    _mm_store_ss(&mut result, s32);
    for i in (chunks * 8)..n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let denom = l2_norm(a) * l2_norm(b);
    if denom < 1e-8 { return 0.0; }
    (dot_product(a, b) / denom).clamp(-1.0, 1.0)
}

#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_sim(a, b)
}

// ─── Ordered float for heaps ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── HNSW node ────────────────────────────────────────────────────────────────

struct Node {
    id:        u64,
    vector:    Vec<f32>,
    level:     usize,
    neighbors: Vec<Vec<usize>>,  // neighbors[l] = neighbor node indices at level l
}

impl Node {
    fn new(id: u64, vector: Vec<f32>, level: usize, m0: usize, m: usize) -> Self {
        let neighbors = (0..=level).map(|l| {
            Vec::with_capacity(if l == 0 { m0 } else { m })
        }).collect();
        Self { id, vector, level, neighbors }
    }
}

// ─── HNSW Index ───────────────────────────────────────────────────────────────

pub struct HnswIndex {
    nodes:           Vec<Node>,
    entry_point:     Option<usize>,
    max_level:       usize,
    dim:             usize,
    m:               usize,
    m0:              usize,
    ef_construction: usize,
    ef_search:       usize,
    ml:              f64,
    id_to_idx:       HashMap<u64, usize>,
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        Self::with_params(dim, 16, 200, 50)
    }

    pub fn with_params(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            dim,
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            id_to_idx: HashMap::new(),
        }
    }

    pub fn len(&self)      -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool  { self.nodes.is_empty() }

    fn random_level(&self) -> usize {
        // Deterministic LCG from node count — replace with rand for production
        let mut s = (self.nodes.len() as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s ^= s >> 33;
        s = s.wrapping_mul(0xff51afd7ed558ccd);
        s ^= s >> 33;
        let u = (s & 0x0000_FFFF_FFFF_FFFF) as f64 / ((1u64 << 48) as f64);
        ((-u.ln()) * self.ml).floor() as usize
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        debug_assert_eq!(vector.len(), self.dim);
        if self.id_to_idx.contains_key(&id) { return; }

        let level     = self.random_level();
        let node_idx  = self.nodes.len();
        self.nodes.push(Node::new(id, vector, level, self.m0, self.m));
        self.id_to_idx.insert(id, node_idx);

        let Some(ep) = self.entry_point else {
            self.entry_point = Some(node_idx);
            self.max_level   = level;
            return;
        };

        let mut ep_set = vec![ep];

        // Greedy descent above the new node's level
        for l in (level + 1..=self.max_level).rev() {
            let nearest = self.greedy_single(node_idx, &ep_set, l);
            ep_set = vec![nearest];
        }

        // Insert and connect at each level ≤ level
        for l in (0..=level.min(self.max_level)).rev() {
            let m_l = if l == 0 { self.m0 } else { self.m };
            let cands = self.search_layer(node_idx, &ep_set, self.ef_construction, l);

            let selected: Vec<usize> = cands.iter().take(m_l).map(|&(_, i)| i).collect();

            self.nodes[node_idx].neighbors[l].extend_from_slice(&selected);

            for &nb in &selected {
                self.nodes[nb].neighbors[l].push(node_idx);
                if self.nodes[nb].neighbors[l].len() > m_l {
                    let nb_vec = self.nodes[nb].vector.clone();
                    let nbs    = self.nodes[nb].neighbors[l].clone();
                    self.nodes[nb].neighbors[l] = self.prune(&nb_vec, &nbs, m_l);
                }
            }

            ep_set = cands.into_iter().map(|(_, i)| i).collect();
        }

        if level > self.max_level {
            self.max_level   = level;
            self.entry_point = Some(node_idx);
        }
    }

    /// Compat alias usado por `latent_drafter`: returns `(usize_id, score)`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.query(query, k)
            .into_iter()
            .map(|(id, sim)| (id as usize, sim))
            .collect()
    }

    /// Returns top-K nearest neighbors as `(external_id, cosine_similarity)`.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        debug_assert_eq!(query.len(), self.dim);
        let Some(ep) = self.entry_point else { return Vec::new(); };

        let mut ep_set = vec![ep];

        for l in (1..=self.max_level).rev() {
            let nearest = self.greedy_single_vec(query, &ep_set, l);
            ep_set = vec![nearest];
        }

        let cands = self.search_layer_vec(query, &ep_set, self.ef_search, 0);

        cands.into_iter()
            .take(k)
            .map(|(dist, idx)| (self.nodes[idx].id, 1.0 - dist.0))
            .collect()
    }

    // ── Internals ────────────────────────────────────────────────────────────

    fn search_layer(&self, node_idx: usize, ep_set: &[usize], ef: usize, layer: usize)
        -> Vec<(OrdF32, usize)>
    {
        self.search_layer_vec(&self.nodes[node_idx].vector.clone(), ep_set, ef, layer)
    }

    fn search_layer_vec(&self, query: &[f32], ep_set: &[usize], ef: usize, layer: usize)
        -> Vec<(OrdF32, usize)>
    {
        let mut visited:    HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new();
        let mut result:     BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();

        for &ep in ep_set {
            let d = OrdF32(cosine_dist(query, &self.nodes[ep].vector));
            candidates.push(Reverse((d, ep)));
            result.push((d, ep));
            visited.insert(ep);
        }

        while let Some(Reverse((d_c, c))) = candidates.pop() {
            let d_f = result.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if d_c.0 > d_f && result.len() >= ef { break; }

            let nbs: Vec<usize> = self.nodes[c].neighbors
                .get(layer).cloned().unwrap_or_default();

            for e in nbs {
                if visited.insert(e) {
                    let d_e = OrdF32(cosine_dist(query, &self.nodes[e].vector));
                    let d_f2 = result.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                    if d_e.0 < d_f2 || result.len() < ef {
                        candidates.push(Reverse((d_e, e)));
                        result.push((d_e, e));
                        if result.len() > ef { result.pop(); }
                    }
                }
            }
        }

        let mut out: Vec<(OrdF32, usize)> = result.into_iter().collect();
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn greedy_single(&self, node_idx: usize, ep_set: &[usize], layer: usize) -> usize {
        self.greedy_single_vec(&self.nodes[node_idx].vector.clone(), ep_set, layer)
    }

    fn greedy_single_vec(&self, query: &[f32], ep_set: &[usize], layer: usize) -> usize {
        let mut best_idx  = ep_set[0];
        let mut best_dist = cosine_dist(query, &self.nodes[best_idx].vector);
        let mut changed   = true;
        while changed {
            changed = false;
            let nbs: Vec<usize> = self.nodes[best_idx].neighbors
                .get(layer).cloned().unwrap_or_default();
            for nb in nbs {
                let d = cosine_dist(query, &self.nodes[nb].vector);
                if d < best_dist { best_dist = d; best_idx = nb; changed = true; }
            }
        }
        best_idx
    }

    fn prune(&self, query: &[f32], cands: &[usize], m: usize) -> Vec<usize> {
        let mut scored: Vec<(OrdF32, usize)> = cands.iter()
            .map(|&i| (OrdF32(cosine_dist(query, &self.nodes[i].vector)), i))
            .collect();
        scored.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, i)| i).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..dim).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect()
    }

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim]; v[hot] = 1.0; v
    }

    #[test]
    fn dot_avx2_matches_scalar() {
        let a = rand_vec(512, 42);
        let b = rand_vec(512, 99);
        let s = dot_scalar(&a, &b);
        let d = dot_product(&a, &b);
        assert!((s - d).abs() < 1e-3, "scalar={} simd={}", s, d);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = unit(8, 0);
        let b = unit(8, 1);
        assert!(cosine_sim(&a, &a) > 0.999);
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn semantic_retrieval() {
        let dim = 8;
        let mut idx = HnswIndex::new(dim);
        idx.insert(10, unit(dim, 0));
        idx.insert(20, unit(dim, 1));
        idx.insert(30, unit(dim, 2));

        let res = idx.query(&unit(dim, 0), 1);
        assert_eq!(res[0].0, 10);
        assert!(res[0].1 > 0.99);

        let res = idx.query(&unit(dim, 1), 1);
        assert_eq!(res[0].0, 20);
    }

    #[test]
    fn high_recall_random() {
        let dim = 64;
        let mut idx = HnswIndex::new(dim);
        let vecs: Vec<Vec<f32>> = (0u64..50).map(|i| rand_vec(dim, i * 1111 + 3)).collect();
        for (i, v) in vecs.iter().enumerate() { idx.insert(i as u64, v.clone()); }

        let mut correct = 0;
        for (i, v) in vecs.iter().enumerate() {
            let res = idx.query(v, 1);
            if !res.is_empty() && res[0].0 == i as u64 { correct += 1; }
        }
        assert!(correct >= 45, "Recall {}/50", correct);
    }

    #[test]
    fn insert_1000_and_query_perf() {
        let dim = 128;
        let mut idx = HnswIndex::new(dim);
        for i in 0u64..1000 { idx.insert(i, rand_vec(dim, i)); }
        assert_eq!(idx.len(), 1000);

        let q   = rand_vec(dim, 9999);
        let t0  = std::time::Instant::now();
        let res = idx.query(&q, 10);
        let ms  = t0.elapsed().as_millis();

        assert!(!res.is_empty());
        println!("HNSW query 128d/1000 nodes: {}ms", ms);
        assert!(ms < 50, "Query too slow: {}ms", ms);
    }

    #[test]
    fn duplicate_id_idempotent() {
        let mut idx = HnswIndex::new(4);
        idx.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert(1, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
    }
}
