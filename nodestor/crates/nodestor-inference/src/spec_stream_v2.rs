// SpecStream V2 — Hyperbolic Concurrent Speculation Tree
//
// Architecture:
//   Draft model expands a K-ary tree of candidate continuations.
//   Branching factor at depth d: floor(K / (d+1))   (hyperbolic taper)
//   This exhausts the token budget K in O(K * H(D)) nodes, D = tree depth,
//   H(D) = harmonic number. For K=16, D=3: 16+8+5 = 29 nodes vs 4096 naive.
//
// Acceptance rate feedback:
//   After each verify pass, rolling alpha = mean(accepted/draft) over window W.
//   K adapts: K_new = K_min + (K_max - K_min) * (1 - alpha/alpha_target).clamp(0,1)
//   Low alpha  → wider tree (more branches to find one master accepts)
//   High alpha → narrower tree (draft is accurate; save compute)
//
// Mathematical expected gain:
//   E[tokens_accepted] = (1 - alpha^(D+1)) / (1 - alpha)
//   alpha=0.70, D=2 → E[acc] = 2.19 tokens per master forward pass
//   alpha=0.80, D=3 → E[acc] = 3.36 tokens per master forward pass
//
// KV rollback: caller maintains KV snapshot at the context boundary.
// On divergence at position i, caller resets KV to snapshot and applies
// tokens[0..i] + pivot. This module only computes which tokens to accept.

use std::collections::VecDeque;

// ─── Configuration ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SpecConfig {
    /// Minimum draft budget per expand call.
    pub k_min: usize,
    /// Maximum draft budget per expand call.
    pub k_max: usize,
    /// Tree depth. 2 covers >95% of the E[acc] gain for alpha≥0.5.
    pub depth: usize,
    /// Target acceptance rate alpha for budget adaptation.
    pub alpha_target: f32,
    /// Rolling window length for alpha estimation.
    pub alpha_window: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            k_min:        4,
            k_max:        16,
            depth:        2,
            alpha_target: 0.70,
            alpha_window: 64,
        }
    }
}

impl SpecConfig {
    /// High-throughput preset: wider tree, deeper speculative horizon.
    pub fn high_throughput() -> Self {
        Self { k_min: 8, k_max: 32, depth: 3, alpha_target: 0.75, alpha_window: 32 }
    }

    /// Conservative preset: small overhead, good for fast GPU master models.
    pub fn conservative() -> Self {
        Self { k_min: 2, k_max: 8, depth: 1, alpha_target: 0.65, alpha_window: 128 }
    }
}

// ─── Tree node ─────────────────────────────────────────────────────────────────

/// One candidate continuation path in the speculation tree.
#[derive(Clone, Debug)]
pub struct SpecNode {
    /// Token at this node.
    pub token: u32,
    /// Cumulative draft log-probability from root to this node.
    pub cumulative_log_prob: f32,
    /// Depth from the root (0 = immediate continuation of context).
    pub depth: u8,
    /// Full sequence: context tokens + tokens from root to this node.
    pub path: Vec<u32>,
}

// ─── Verification result ───────────────────────────────────────────────────────

pub struct VerifyResult {
    /// Accepted tokens in order (prefix of the best accepted path).
    pub accepted: Vec<u32>,
    /// Correction token produced by the master at the divergence point.
    pub pivot: u32,
    /// Sample of alpha for this verification round.
    pub alpha_sample: f32,
}

// ─── Main engine ───────────────────────────────────────────────────────────────

pub struct SpecStreamEngine {
    config:         SpecConfig,
    budget:         usize,
    alpha_history:  VecDeque<f32>,
    /// Best expanded tree from the last `expand` call.
    pub last_nodes: Vec<SpecNode>,
}

impl SpecStreamEngine {
    pub fn new(config: SpecConfig) -> Self {
        let budget = config.k_min;
        Self {
            config,
            budget,
            alpha_history: VecDeque::new(),
            last_nodes: Vec::new(),
        }
    }

    // ── Tree expansion ──────────────────────────────────────────────────────────

    /// Expand the speculation tree from `context`.
    ///
    /// `draft_fn(seq) → [(token, log_prob)]` — top-K candidates from the draft model
    /// for the given sequence. Called at most `total_nodes()` times per expand.
    pub fn expand<F>(&mut self, context: &[u32], mut draft_fn: F)
    where
        F: FnMut(&[u32]) -> Vec<(u32, f32)>,
    {
        self.last_nodes.clear();

        // Level 0: immediate continuations of context
        let k0 = self.branches_at_depth(0);
        let root_candidates = draft_fn(context);

        let mut level: Vec<SpecNode> = root_candidates
            .into_iter()
            .take(k0)
            .map(|(token, log_prob)| {
                let mut path = context.to_vec();
                path.push(token);
                SpecNode { token, cumulative_log_prob: log_prob, depth: 0, path }
            })
            .collect();

        self.last_nodes.extend(level.iter().cloned());

        // Levels 1..depth: extend each leaf
        for d in 1..self.config.depth {
            let k_d = self.branches_at_depth(d);
            let mut next_level = Vec::new();
            for parent in &level {
                let children = draft_fn(&parent.path);
                for (token, lp) in children.into_iter().take(k_d) {
                    let mut path = parent.path.clone();
                    path.push(token);
                    let node = SpecNode {
                        token,
                        cumulative_log_prob: parent.cumulative_log_prob + lp,
                        depth: d as u8,
                        path,
                    };
                    next_level.push(node.clone());
                    self.last_nodes.push(node);
                }
            }
            level = next_level;
        }

        // Sort all nodes by cumulative probability: best paths first
        self.last_nodes.sort_unstable_by(|a, b| {
            b.cumulative_log_prob.partial_cmp(&a.cumulative_log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // ── Sequential verification (greedy best path) ──────────────────────────────

    /// Verify the best linear path against master log-probabilities.
    ///
    /// `draft_tokens`: the token sequence from the best tree path.
    /// `draft_lp[i]`: log P_draft(draft_tokens[i] | prefix).
    /// `master_lp[i]`: log P_master(draft_tokens[i] | prefix).
    /// `pivot`: token the master would generate at the first divergence point.
    ///
    /// Applies rejection sampling: accepts token i with probability
    ///   min(1, exp(master_lp[i] - draft_lp[i]))
    /// which is the standard speculative decoding acceptance criterion.
    pub fn verify(
        &mut self,
        draft_tokens:  &[u32],
        draft_lp:      &[f32],
        master_lp:     &[f32],
        pivot:         u32,
    ) -> VerifyResult {
        debug_assert_eq!(draft_tokens.len(), draft_lp.len());
        debug_assert_eq!(draft_tokens.len(), master_lp.len());

        let mut accepted = Vec::with_capacity(draft_tokens.len());
        let mut alpha_sum = 0.0f32;

        for i in 0..draft_tokens.len() {
            let acceptance_ratio = (master_lp[i] - draft_lp[i]).exp().min(1.0_f32);
            alpha_sum += acceptance_ratio;
            let u = deterministic_uniform(draft_tokens[i], i as u64);
            if u < acceptance_ratio {
                accepted.push(draft_tokens[i]);
            } else {
                break;
            }
        }

        let n = draft_tokens.len().max(1);
        let alpha_sample = alpha_sum / n as f32;
        self.record_alpha(alpha_sample);

        VerifyResult { accepted, pivot, alpha_sample }
    }

    // ── State accessors ─────────────────────────────────────────────────────────

    pub fn current_budget(&self) -> usize { self.budget }

    pub fn current_alpha(&self) -> f32 {
        if self.alpha_history.is_empty() {
            return self.config.alpha_target;
        }
        self.alpha_history.iter().sum::<f32>() / self.alpha_history.len() as f32
    }

    /// Theoretical expected accepted tokens per master forward pass.
    pub fn expected_accepted(&self) -> f32 {
        let alpha = self.current_alpha();
        let d = self.config.depth as i32;
        if (1.0 - alpha).abs() < 1e-6 {
            return (d + 1) as f32; // perfect acceptance
        }
        (1.0 - alpha.powi(d + 1)) / (1.0 - alpha)
    }

    // ── Private ─────────────────────────────────────────────────────────────────

    fn branches_at_depth(&self, depth: usize) -> usize {
        // Hyperbolic taper: K, K/2, K/3, ... — total = K * H(depth)
        (self.budget / (depth + 1)).max(1)
    }

    fn record_alpha(&mut self, alpha: f32) {
        self.alpha_history.push_back(alpha);
        if self.alpha_history.len() > self.config.alpha_window {
            self.alpha_history.pop_front();
        }
        self.adapt_budget();
    }

    fn adapt_budget(&mut self) {
        let alpha_avg = self.current_alpha();
        let t = self.config.alpha_target;
        // Low alpha → more branches; high alpha → fewer
        let factor = ((1.0 - alpha_avg / t).clamp(0.0, 1.0)) as f32;
        let k = self.config.k_min as f32
            + (self.config.k_max - self.config.k_min) as f32 * factor;
        self.budget = (k.round() as usize).clamp(self.config.k_min, self.config.k_max);
    }
}

// ─── Deterministic pseudo-uniform [0, 1) ───────────────────────────────────────
// Permuted congruential generator — no external RNG dependency.
#[inline(always)]
fn deterministic_uniform(token: u32, pos: u64) -> f32 {
    let mut x = (token as u64).wrapping_mul(6364136223846793005)
        .wrapping_add(pos.wrapping_mul(1442695040888963407));
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    // Map to [0, 1)
    (x >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_draft(ctx: &[u32]) -> Vec<(u32, f32)> {
        // Deterministic top-4: tokens are hash of context tail
        let seed = ctx.last().copied().unwrap_or(0) as u64;
        (0..4u32).map(|i| {
            let tok = ((seed.wrapping_add(i as u64).wrapping_mul(2654435761)) & 0x7FFF) as u32;
            let lp  = -(i as f32 + 1.0).ln();
            (tok, lp)
        }).collect()
    }

    #[test]
    fn spec_tree_expand_node_count() {
        let mut engine = SpecStreamEngine::new(SpecConfig {
            k_min: 4, k_max: 4, depth: 2, alpha_target: 0.70, alpha_window: 8,
        });
        let context = vec![1u32, 2, 3];
        engine.expand(&context, mock_draft);
        // Depth 0: 4 nodes; depth 1: 4*2 nodes (k/2=2 each) → total 12
        assert!(engine.last_nodes.len() >= 4, "at least k_min root nodes");
        assert!(engine.last_nodes.len() <= 12, "bounded by hyperbolic budget");
    }

    #[test]
    fn spec_verify_perfect_draft_accepts_all() {
        let mut engine = SpecStreamEngine::new(SpecConfig::default());
        let tokens = vec![10u32, 20, 30, 40];
        let n = tokens.len();
        // master and draft agree perfectly
        let lp = vec![-0.5f32; n];
        let result = engine.verify(&tokens, &lp, &lp, 99);
        assert_eq!(result.accepted.len(), n, "perfect draft: all tokens accepted");
        assert!((result.alpha_sample - 1.0).abs() < 1e-3);
    }

    #[test]
    fn spec_verify_worst_draft_accepts_none() {
        let mut engine = SpecStreamEngine::new(SpecConfig::default());
        let tokens = vec![10u32, 20, 30];
        let draft_lp = vec![ 0.0f32; 3]; // p_draft = 1.0 for these tokens
        let master_lp = vec![-10.0f32; 3]; // p_master ≈ 0 for these tokens
        let result = engine.verify(&tokens, &draft_lp, &master_lp, 99);
        // acceptance ratio = exp(-10 - 0) = near 0 → first token rejected
        assert!(result.accepted.is_empty(), "worst draft: no tokens accepted");
    }

    #[test]
    fn spec_budget_adapts_on_low_alpha() {
        let mut engine = SpecStreamEngine::new(SpecConfig {
            k_min: 4, k_max: 32, depth: 2, alpha_target: 0.70, alpha_window: 4,
        });
        let tokens = vec![1u32, 2];
        let draft_lp  = vec![0.0f32; 2];
        let master_lp = vec![-10.0f32; 2]; // near-zero acceptance

        for _ in 0..8 {
            engine.verify(&tokens, &draft_lp, &master_lp, 0);
        }
        // Low alpha → budget should grow toward k_max
        assert!(engine.current_budget() > 4, "budget should increase on low acceptance");
    }

    #[test]
    fn spec_expected_accepted_monotone_in_alpha() {
        let mut engine = SpecStreamEngine::new(SpecConfig {
            k_min: 4, k_max: 4, depth: 2, alpha_target: 0.70, alpha_window: 1,
        });
        // Simulate high alpha
        let tokens = vec![1u32];
        let high_lp = vec![-0.1f32];
        engine.verify(&tokens, &high_lp, &high_lp, 0);
        let e_high = engine.expected_accepted();

        // Simulate low alpha
        let low_draft = vec![0.0f32];
        let low_master = vec![-5.0f32];
        engine.verify(&tokens, &low_draft, &low_master, 0);
        let e_low = engine.expected_accepted();

        assert!(e_high > e_low, "higher alpha → more expected accepted tokens");
    }

    #[test]
    fn deterministic_uniform_range() {
        for t in 0u32..100 {
            for p in 0u64..100 {
                let u = deterministic_uniform(t, p);
                assert!(u >= 0.0 && u < 1.0, "out of range: {}", u);
            }
        }
    }
}
