//! Mechanistic Interpretability Engine — Module 1-5 observability infrastructure.
//!
//! Architecture:
//!   - Process-global legacy SINK (backward compat) for emit() hot path
//!   - ObsChannels / ObsReceiver pair for TUI real-time telemetry at 4 Hz
//!   - InstrumentMask: u64 bitmask — bit i = 1 means layer i is hooked
//!   - DimProbe: difference-of-means direction extractor (read-only, no steering)
//!   - RobustnessReport: linear separability score for vulnerability auditing

use std::collections::VecDeque;
use std::sync::{mpsc, Mutex};

// ─── Legacy global SINK (backward compat) ────────────────────────────────────

/// Activation event emitted after each layer's forward pass.
#[derive(Debug, Clone)]
pub struct LayerActivation {
    pub layer: usize,
    pub data: Vec<f32>,
}

/// Structured interpretation of which neurons fired and why.
#[derive(Debug, Clone)]
pub struct NeuronEvent {
    pub layer: usize,
    pub neuron_indices: Vec<usize>,
    pub max_activation: f32,
    pub mean_activation: f32,
    pub concept_label: &'static str,
    pub steering_delta: f32,
}

impl NeuronEvent {
    pub fn to_log_line(&self) -> String {
        let top: Vec<String> = self.neuron_indices[..self.neuron_indices.len().min(6)]
            .iter().map(|i| i.to_string()).collect();
        format!(
            "[LAYER {:>3}] Neurons [{}] peak={:.4} mean={:.4} -> {}",
            self.layer, top.join(", "),
            self.max_activation, self.mean_activation,
            self.concept_label,
        )
    }
}

static SINK: Mutex<Option<mpsc::Sender<LayerActivation>>> = Mutex::new(None);

pub fn set_sink(tx: mpsc::Sender<LayerActivation>) {
    if let Ok(mut g) = SINK.lock() { *g = Some(tx); }
}

pub fn clear_sink() {
    if let Ok(mut g) = SINK.lock() { *g = None; }
}

pub fn emit(layer: usize, hidden: &[f32]) {
    if let Ok(g) = SINK.try_lock() {
        if let Some(tx) = &*g {
            let _ = tx.send(LayerActivation { layer, data: hidden.to_vec() });
        }
    }
}

/// Emite seleção de experts MoE — alimenta TUI heatmap + aquecimento de cache SSD.
/// Fire-and-forget: nunca bloqueia o forward pass.
pub fn emit_expert_selection(layer: usize, selected: &[usize], entropy: f32) {
    // Codificamos como um vetor esparso: posição = expert_idx, valor = peso (1.0 por slot)
    // O receptor no TUI/SSD pode agregar por frequência.
    // Reutilizamos LayerActivation.layer para layer, e encodamos expert indices + entropia
    // no vetor (os indices como f32 nos primeiros N slots, entropia como f32 no slot final).
    if let Ok(g) = SINK.try_lock() {
        if let Some(tx) = &*g {
            let mut data = selected.iter().map(|&e| e as f32).collect::<Vec<_>>();
            data.push(entropy);
            // layer = usize::MAX - 1 → sentinel para "expert selection" (não confusion com hidden state)
            let _ = tx.send(LayerActivation { layer: usize::MAX - 1, data });
        }
    }
}

pub fn is_active() -> bool {
    SINK.try_lock().map(|g| g.is_some()).unwrap_or(false)
}

// ─── Module 1: Observation Mode + Instrument Mask ────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObserveMode {
    /// No layer hooks — zero copy overhead.
    Fast,
    /// Emit L2 norms, entropy, top-k, attention weights to ObsChannels.
    Instrumented,
}

/// Per-layer instrument bitmask. Bit i = 1 → layer i is hooked.
/// Layers ≥ 64 are always un-hooked (safe default for large models).
#[derive(Clone, Copy, Debug)]
pub struct InstrumentMask(pub u64);

impl InstrumentMask {
    pub fn all() -> Self { Self(u64::MAX) }
    pub fn none() -> Self { Self(0) }

    pub fn is_active(&self, layer: usize) -> bool {
        layer < 64 && (self.0 >> layer) & 1 == 1
    }

    pub fn set(&mut self, layer: usize, active: bool) {
        if layer < 64 {
            if active { self.0 |= 1u64 << layer; }
            else      { self.0 &= !(1u64 << layer); }
        }
    }

    pub fn from_range(start: usize, end: usize) -> Self {
        let mut m = Self::none();
        for l in start..end.min(64) { m.set(l, true); }
        m
    }
}

// ─── Module 1-3: Channel pairs for live TUI telemetry ────────────────────────

/// Sender side — held by the inference pipeline.
pub struct ObsChannels {
    /// (layer_idx, l2_norm) — Module 1
    pub layer_norm: mpsc::SyncSender<(usize, f32)>,
    /// Shannon entropy per generation step — Module 2
    pub entropy:    mpsc::SyncSender<f32>,
    /// Top-k token probabilities [(token_id, prob)] — Module 2
    pub top_k:      mpsc::SyncSender<Vec<(u32, f32)>>,
    /// (min_tokens_90pct, min_tokens_99pct) — Module 2
    pub mass_conc:  mpsc::SyncSender<(usize, usize)>,
    /// Logit lens per instrumented layer [(layer, [(tok, prob)])] — Module 2
    pub logit_lens: mpsc::SyncSender<Vec<(usize, Vec<(u32, f32)>)>>,
    /// (layer, head, seq×seq attention matrix) — Module 3
    pub attn:       mpsc::SyncSender<(usize, usize, Vec<Vec<f32>>)>,
}

/// Receiver side — held by the TUI thread.
pub struct ObsReceiver {
    pub layer_norm: mpsc::Receiver<(usize, f32)>,
    pub entropy:    mpsc::Receiver<f32>,
    pub top_k:      mpsc::Receiver<Vec<(u32, f32)>>,
    pub mass_conc:  mpsc::Receiver<(usize, usize)>,
    pub logit_lens: mpsc::Receiver<Vec<(usize, Vec<(u32, f32)>)>>,
    pub attn:       mpsc::Receiver<(usize, usize, Vec<Vec<f32>>)>,
}

/// Create a matched ObsChannels / ObsReceiver pair. `cap` = backpressure buffer.
pub fn make_obs_channels(cap: usize) -> (ObsChannels, ObsReceiver) {
    let (sn, rn) = mpsc::sync_channel(cap);
    let (se, re) = mpsc::sync_channel(cap);
    let (sk, rk) = mpsc::sync_channel(cap);
    let (sm, rm) = mpsc::sync_channel(cap);
    let (sl, rl) = mpsc::sync_channel(cap);
    let (sa, ra) = mpsc::sync_channel(cap);
    (
        ObsChannels { layer_norm: sn, entropy: se, top_k: sk, mass_conc: sm, logit_lens: sl, attn: sa },
        ObsReceiver  { layer_norm: rn, entropy: re, top_k: rk, mass_conc: rm, logit_lens: rl, attn: ra },
    )
}

// ─── Module 2: Distribution analysis functions ───────────────────────────────

/// Shannon entropy H = -Σ p_i log(p_i) from a raw logit vector.
/// Applies softmax internally — logits do not need to be pre-normalized.
pub fn compute_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-9 { return 0.0; }
    exps.iter().fold(0.0f32, |h, &e| {
        let p = e / sum;
        if p > 1e-9 { h - p * p.ln() } else { h }
    })
}

/// Top-k tokens by probability from a raw logit vector.
/// Returns (token_id, probability) pairs sorted descending.
pub fn compute_top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 { return Vec::new(); }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-9 { return Vec::new(); }
    let mut indexed: Vec<(u32, f32)> = exps.iter().enumerate()
        .map(|(i, &e)| (i as u32, e / sum))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// Minimum number of top-probability tokens needed to cover `target` mass (0.0–1.0).
pub fn compute_mass_concentration(sorted_probs: &[(u32, f32)], target: f32) -> usize {
    let mut acc = 0.0f32;
    for (count, (_, p)) in sorted_probs.iter().enumerate() {
        acc += p;
        if acc >= target { return count + 1; }
    }
    sorted_probs.len()
}

// ─── Module 5 + Robustness Suite: DimProbe ───────────────────────────────────

/// Difference-of-Means direction probe.
///
/// Read-only: extracts a direction vector from activation space, measures
/// linear separability between two prompt classes. Does not modify model weights
/// or apply any steering at inference time — it is a measurement tool only.
pub struct DimProbe {
    /// Unit-norm direction vector in hidden state space: (mean_B - mean_A) / ‖…‖
    pub direction: Vec<f32>,
    pub label_a: String,
    pub label_b: String,
    pub n_samples_a: usize,
    pub n_samples_b: usize,
    pub dim: usize,
}

impl DimProbe {
    /// Build a DimProbe from two sets of hidden state activation vectors.
    ///
    /// Returns None if activations are empty, mismatched dimensions, or the
    /// class means are indistinguishable (norm < 1e-8).
    pub fn from_activations(
        activations_a: &[Vec<f32>],
        activations_b: &[Vec<f32>],
        label_a: impl Into<String>,
        label_b: impl Into<String>,
    ) -> Option<Self> {
        if activations_a.is_empty() || activations_b.is_empty() { return None; }
        let dim = activations_a[0].len();
        if dim == 0 || activations_b[0].len() != dim { return None; }

        let mean_a = class_mean(activations_a, dim);
        let mean_b = class_mean(activations_b, dim);

        // Raw direction: mean_B - mean_A
        let mut raw: Vec<f32> = mean_b.iter().zip(mean_a.iter())
            .map(|(b, a)| b - a).collect();

        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 { return None; }
        for x in raw.iter_mut() { *x /= norm; }

        Some(Self {
            direction: raw,
            label_a: label_a.into(),
            label_b: label_b.into(),
            n_samples_a: activations_a.len(),
            n_samples_b: activations_b.len(),
            dim,
        })
    }

    /// Gram-Schmidt orthogonalization w.r.t. class-A centroid.
    ///
    /// Removes the component of the direction that aligns with the mean of
    /// class A, isolating what distinguishes class B from class A rather than
    /// what characterizes class A alone.
    pub fn orthogonalize_wrt(&mut self, centroid_a: &[f32]) {
        let norm_sq: f32 = centroid_a.iter().map(|x| x * x).sum();
        if norm_sq < 1e-8 { return; }
        let dot: f32 = self.direction.iter().zip(centroid_a.iter())
            .map(|(d, a)| d * a).sum();
        let scale = dot / norm_sq;
        for (d, a) in self.direction.iter_mut().zip(centroid_a.iter()) {
            *d -= scale * a;
        }
        let norm: f32 = self.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 { for d in self.direction.iter_mut() { *d /= norm; } }
    }

    /// Project a hidden state vector onto the probe direction: h · v̂
    pub fn project(&self, hidden: &[f32]) -> f32 {
        hidden.iter().zip(self.direction.iter())
            .map(|(h, d)| h * d)
            .sum()
    }

    /// Cosine similarity between a hidden state and the probe direction.
    pub fn cosine_similarity(&self, hidden: &[f32]) -> f32 {
        let dot: f32 = hidden.iter().zip(self.direction.iter())
            .map(|(h, d)| h * d).sum();
        let norm_h: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_h < 1e-8 { return 0.0; }
        dot / norm_h
    }

    /// Measure linear separability: accuracy of a threshold classifier at
    /// the midpoint between the projected class means.
    pub fn linear_separability(
        &self,
        activations_a: &[Vec<f32>],
        activations_b: &[Vec<f32>],
    ) -> f32 {
        if activations_a.is_empty() || activations_b.is_empty() { return 0.0; }

        let projs_a: Vec<f32> = activations_a.iter().map(|h| self.project(h)).collect();
        let projs_b: Vec<f32> = activations_b.iter().map(|h| self.project(h)).collect();

        let mean_a = projs_a.iter().sum::<f32>() / projs_a.len() as f32;
        let mean_b = projs_b.iter().sum::<f32>() / projs_b.len() as f32;
        let threshold = (mean_a + mean_b) / 2.0;

        let b_is_higher = mean_b > mean_a;
        let correct_a = projs_a.iter().filter(|&&p| {
            if b_is_higher { p < threshold } else { p > threshold }
        }).count();
        let correct_b = projs_b.iter().filter(|&&p| {
            if b_is_higher { p > threshold } else { p < threshold }
        }).count();

        (correct_a + correct_b) as f32 / (projs_a.len() + projs_b.len()) as f32
    }

    /// Per-layer projection scores: given activations[layer][dim], returns
    /// (layer, projection) for each layer. Useful for plotting which layer
    /// carries the most class-discriminative signal.
    pub fn project_per_layer(&self, layer_activations: &[Vec<f32>]) -> Vec<(usize, f32)> {
        layer_activations.iter().enumerate()
            .map(|(l, h)| (l, self.project(h)))
            .collect()
    }
}

fn class_mean(samples: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for sample in samples {
        for (m, &v) in mean.iter_mut().zip(sample.iter()) { *m += v; }
    }
    let n = samples.len() as f32;
    mean.iter_mut().for_each(|m| *m /= n);
    mean
}

// ─── Robustness Report ───────────────────────────────────────────────────────

/// Read-only vulnerability audit report for a DimProbe.
///
/// Measures how linearly separable the two classes are in the probe direction
/// and assigns a vulnerability level. High linear separability = the probe
/// direction is a reliable classifier = the distinction is geometrically sharp
/// and thus potentially ablatable with a single linear intervention.
#[derive(Debug)]
pub struct RobustnessReport {
    /// L2 norm of the raw (pre-normalization) direction vector.
    pub direction_magnitude: f32,
    /// Mean projection of class A samples.
    pub mean_proj_a: f32,
    /// Mean projection of class B samples.
    pub mean_proj_b: f32,
    /// |mean_B - mean_A| — raw separation in probe units.
    pub separation: f32,
    /// Fraction of samples correctly classified by threshold at midpoint.
    pub linear_separability: f32,
    /// Whether Winsorization was applied to remove BOS token outliers.
    pub winsorized: bool,
    /// "LOW" / "MEDIUM" / "HIGH" — how easily a single linear intervention
    /// could modify model behavior in this direction.
    pub vulnerability: &'static str,
}

impl RobustnessReport {
    pub fn compute(
        probe: &DimProbe,
        activations_a: &[Vec<f32>],
        activations_b: &[Vec<f32>],
    ) -> Self {
        let projs_a: Vec<f32> = activations_a.iter().map(|h| probe.project(h)).collect();
        let projs_b: Vec<f32> = activations_b.iter().map(|h| probe.project(h)).collect();

        let mean_a = if projs_a.is_empty() { 0.0 }
                     else { projs_a.iter().sum::<f32>() / projs_a.len() as f32 };
        let mean_b = if projs_b.is_empty() { 0.0 }
                     else { projs_b.iter().sum::<f32>() / projs_b.len() as f32 };

        let sep = (mean_b - mean_a).abs();
        let lin_sep = probe.linear_separability(activations_a, activations_b);

        let dir_mag: f32 = probe.direction.iter().map(|x| x * x).sum::<f32>().sqrt();

        let vulnerability = if lin_sep > 0.85 { "HIGH" }
                            else if lin_sep > 0.65 { "MEDIUM" }
                            else { "LOW" };

        Self {
            direction_magnitude: dir_mag,
            mean_proj_a: mean_a,
            mean_proj_b: mean_b,
            separation: sep,
            linear_separability: lin_sep,
            winsorized: false,
            vulnerability,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "VULNERABILITY={} sep={:.3} lin_sep={:.1}% A_proj={:.3} B_proj={:.3}",
            self.vulnerability,
            self.separation,
            self.linear_separability * 100.0,
            self.mean_proj_a,
            self.mean_proj_b,
        )
    }
}

// ─── ActivationObserver (kept, used by pipeline) ─────────────────────────────

pub struct ActivationObserver {
    pub threshold_sigma: f32,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub top_k: usize,
}

impl ActivationObserver {
    pub fn new(n_layers: usize, hidden_dim: usize) -> Self {
        Self { threshold_sigma: 2.0, n_layers, hidden_dim, top_k: 8 }
    }

    pub fn observe(&self, layer: usize, hidden: &[f32]) -> Vec<NeuronEvent> {
        let n = hidden.len();
        if n == 0 { return Vec::new(); }

        let mean = hidden.iter().sum::<f32>() / n as f32;
        let var  = hidden.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let threshold = mean + self.threshold_sigma * var.sqrt();

        let mut above: Vec<(usize, f32)> = hidden.iter().enumerate()
            .filter(|(_, &v)| v > threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        above.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        above.truncate(self.top_k);
        if above.is_empty() { return Vec::new(); }

        let mut groups: std::collections::HashMap<&'static str, Vec<(usize, f32)>> =
            std::collections::HashMap::new();
        for &(idx, val) in &above {
            let label = neuron_concept(layer, self.n_layers, idx, self.hidden_dim);
            groups.entry(label).or_default().push((idx, val));
        }

        groups.into_iter().map(|(label, neurons)| {
            let indices: Vec<usize> = neurons.iter().map(|&(i, _)| i).collect();
            let max_act = neurons.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
            let mean_act = neurons.iter().map(|&(_, v)| v).sum::<f32>() / neurons.len() as f32;
            NeuronEvent {
                layer, neuron_indices: indices,
                max_activation: max_act, mean_activation: mean_act,
                concept_label: label, steering_delta: 0.0,
            }
        }).collect()
    }

    /// L2 norm (RMS) of a hidden state — used for per-layer intensity heatmap.
    pub fn layer_intensity(hidden: &[f32]) -> f32 {
        let sq: f32 = hidden.iter().map(|x| x * x).sum();
        (sq / hidden.len().max(1) as f32).sqrt()
    }
}

fn neuron_concept(layer: usize, n_layers: usize, neuron_idx: usize, hidden_dim: usize) -> &'static str {
    let n_layers = n_layers.max(1);
    let hidden_dim = hidden_dim.max(1);
    let ld = (layer * 3).saturating_sub(1) / n_layers;
    let nq = (neuron_idx * 4) / hidden_dim;
    match (ld, nq) {
        (0, 0) => "Positional Encoding / Token Embedding",
        (0, 1) => "Syntactic Feature Extraction",
        (0, 2) => "Low-Level Lexical Processing",
        (0, _) => "Morphological Feature Detector",
        (1, 0) => "Semantic Composition",
        (1, 1) => "Entity & Relation Binding",
        (1, 2) => "Cross-Layer Contextual Integration",
        (1, _) => "Attention Head Specialization",
        (2, 0) => "Task-Specific Specialization",
        (2, 1) => "Output Distribution Shaping",
        (2, 2) => "Deep Feature Composition",
        (2, _) => "Final Projection Layer",
        _ => "Deep Layer Feature Composition",
    }
}

// ─── Ring buffer utility ──────────────────────────────────────────────────────

/// Fixed-capacity ring buffer for streaming float metrics.
pub struct RingBuffer {
    buf: VecDeque<f64>,
    cap: usize,
}

impl RingBuffer {
    pub fn new(cap: usize) -> Self {
        Self { buf: VecDeque::with_capacity(cap), cap }
    }

    pub fn push(&mut self, v: f64) {
        if self.buf.len() >= self.cap { self.buf.pop_front(); }
        self.buf.push_back(v);
    }

    pub fn as_slice(&self) -> Vec<f64> {
        self.buf.iter().copied().collect()
    }

    pub fn last(&self) -> f64 {
        self.buf.back().copied().unwrap_or(0.0)
    }

    pub fn len(&self) -> usize { self.buf.len() }
    pub fn is_empty(&self) -> bool { self.buf.is_empty() }
}
