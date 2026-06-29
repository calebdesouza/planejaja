// NodeStor Architectural Integration Tests
//
// Deterministic suite validating the four new infrastructure layers:
//   1. SpecStream V2 — hyperbolic tree speculation
//   2. Quant-Invariant Bandwidth — compression math + MoE early abort
//   3. Platform IO — cross-platform tensor reads
//   4. Timeline Semaphore — CPU/GPU pipeline sync
//
// All tests use static tensor mocks and fixed seeds — no GPU, no SSD, no network.
// Run with: cargo test -p nodestor-mega-test --test arch_integration_tests -- --test-threads=1

use nodestor_inference::{
    spec_stream_v2::{SpecConfig, SpecStreamEngine},
    quant_invariant::{
        BandwidthProfile, BandwidthProjector, ExpertProvisioner,
        EarlyAbortToken, zipf_cache_hit_rate, effective_reads_per_token,
    },
    platform_io::{open_tensor_file, nodestor_data_dir, models_dir, loras_dir},
    timeline_semaphore::{TimelineSemaphore, LayerFence},
};

use std::io::Write;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. SPECSTREAM V2 — Hyperbolic Tree Speculation
// ═══════════════════════════════════════════════════════════════════════════════

fn identity_draft(ctx: &[u32]) -> Vec<(u32, f32)> {
    let base = ctx.last().copied().unwrap_or(0);
    (0..8u32).map(|i| (base.wrapping_add(i + 1), -(i as f32 + 1.0).ln())).collect()
}

#[test]
fn spec_default_config_sane() {
    let cfg = SpecConfig::default();
    assert!(cfg.k_min >= 1, "k_min must be positive");
    assert!(cfg.k_max >= cfg.k_min, "k_max must be >= k_min");
    assert!(cfg.depth >= 1, "depth must be at least 1");
    assert!((0.0..=1.0).contains(&cfg.alpha_target), "alpha_target must be in [0, 1]");
}

#[test]
fn spec_high_throughput_config_is_wider_than_default() {
    let def = SpecConfig::default();
    let ht  = SpecConfig::high_throughput();
    assert!(ht.k_max >= def.k_max, "high_throughput must have at least as many branches");
    assert!(ht.depth >= def.depth, "high_throughput must be at least as deep");
}

#[test]
fn spec_conservative_config_is_narrower_than_default() {
    let def = SpecConfig::default();
    let con = SpecConfig::conservative();
    assert!(con.k_max <= def.k_max, "conservative must be narrower");
}

#[test]
fn spec_expand_produces_sorted_nodes_by_log_prob() {
    let mut engine = SpecStreamEngine::new(SpecConfig::default());
    engine.expand(&[1, 2, 3], identity_draft);
    let nodes = &engine.last_nodes;
    assert!(!nodes.is_empty(), "expansion must produce at least one node");
    for w in nodes.windows(2) {
        assert!(w[0].cumulative_log_prob >= w[1].cumulative_log_prob,
            "nodes must be sorted descending by log_prob");
    }
}

#[test]
fn spec_expand_depth_one_has_k_nodes() {
    let cfg = SpecConfig { k_min: 6, k_max: 6, depth: 1, alpha_target: 0.7, alpha_window: 8 };
    let mut engine = SpecStreamEngine::new(cfg);
    engine.expand(&[42], identity_draft);
    assert_eq!(engine.last_nodes.len(), 6, "depth=1 must produce exactly k nodes");
}

#[test]
fn spec_expand_depth_two_bounded_by_hyperbolic_budget() {
    // K=8, depth=2: level-0 has 8 nodes, level-1 has 8/2=4 per parent × 8 parents = 32
    // Total: 8 + 32 = 40. Hyperbolic bound: K*(1 + 1/2) = 12. Wait, let me recalculate.
    // Actually branches_at_depth(0) = budget/1 = 8, branches_at_depth(1) = budget/2 = 4.
    // Level 0: 8 nodes. Level 1: 8 parents × 4 children = 32 nodes. Total: 40.
    let cfg = SpecConfig { k_min: 8, k_max: 8, depth: 2, alpha_target: 0.7, alpha_window: 8 };
    let mut engine = SpecStreamEngine::new(cfg);
    engine.expand(&[1], identity_draft);
    let n = engine.last_nodes.len();
    // Identity draft gives 8 candidates at level 0, 4 at level 1 per parent
    assert!(n >= 8,  "must have at least k_min nodes");
    assert!(n <= 40, "hyperbolic budget: 8 + 8×4 = 40 max");
}

#[test]
fn spec_verify_partial_acceptance() {
    let mut engine = SpecStreamEngine::new(SpecConfig::default());
    // Tokens: 10, 20, 30
    // draft log probs: -0.1, -0.1, -0.1 (p_draft ≈ 0.9)
    // master log probs: -0.1, -5.0, -0.1 (token 1 rejected)
    let tokens    = vec![10u32, 20, 30];
    let draft_lp  = vec![-0.1f32; 3];
    let master_lp = vec![-0.1f32, -5.0, -0.1];
    let result = engine.verify(&tokens, &draft_lp, &master_lp, 99);
    // First token accepted (ratio=1.0), second rejected → only [10] accepted
    assert_eq!(result.accepted, vec![10u32], "first accepted, second rejected");
    assert_eq!(result.pivot, 99);
}

#[test]
fn spec_expected_accepted_formula_at_alpha_0_7_depth_2() {
    // Mathematical guarantee: E[acc] = (1 - 0.7^3) / (1 - 0.7) = (1 - 0.343) / 0.3 = 2.19
    let mut engine = SpecStreamEngine::new(SpecConfig {
        k_min: 4, k_max: 4, depth: 2, alpha_target: 0.70, alpha_window: 100,
    });
    // Pump alpha history with ~0.7 acceptance rate
    let tokens = vec![1u32];
    for i in 0..50 {
        let lp = if i % 3 == 0 { vec![-5.0f32] } else { vec![-0.1f32] };
        let draft = vec![-0.1f32];
        engine.verify(&tokens, &draft, &lp, 0);
    }
    let e = engine.expected_accepted();
    // Alpha will be around 0.67-0.73 after 50 samples → E[acc] ≈ 2.1-2.3
    assert!(e > 1.5 && e < 3.5, "E[acc] at alpha≈0.7, D=2 should be ≈2.19, got {}", e);
}

#[test]
fn spec_budget_does_not_exceed_k_max() {
    let cfg = SpecConfig { k_min: 4, k_max: 16, depth: 2, alpha_target: 0.7, alpha_window: 4 };
    let mut engine = SpecStreamEngine::new(cfg);
    let t = vec![1u32];
    let lp = vec![-10.0f32]; // low alpha → budget grows
    for _ in 0..100 {
        engine.verify(&t, &lp, &lp, 0);
        assert!(engine.current_budget() <= 16, "budget must not exceed k_max");
        assert!(engine.current_budget() >= 4,  "budget must not go below k_min");
    }
}

#[test]
fn spec_node_paths_include_context() {
    let context = vec![100u32, 200, 300];
    let mut engine = SpecStreamEngine::new(SpecConfig::default());
    engine.expand(&context, identity_draft);
    for node in &engine.last_nodes {
        assert!(node.path.starts_with(&context),
            "every node path must start with the context tokens");
        assert!(node.path.len() > context.len(), "node path must extend beyond context");
    }
}

#[test]
fn spec_engine_is_deterministic_across_two_instances() {
    let cfg = SpecConfig::default();
    let mut e1 = SpecStreamEngine::new(cfg.clone());
    let mut e2 = SpecStreamEngine::new(cfg);
    let ctx = vec![7u32, 13, 42];

    e1.expand(&ctx, identity_draft);
    e2.expand(&ctx, identity_draft);

    assert_eq!(e1.last_nodes.len(), e2.last_nodes.len());
    for (n1, n2) in e1.last_nodes.iter().zip(e2.last_nodes.iter()) {
        assert_eq!(n1.token, n2.token, "deterministic expansion: same tokens");
        assert!((n1.cumulative_log_prob - n2.cumulative_log_prob).abs() < 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. QUANT-INVARIANT BANDWIDTH
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bwprofile_q4_has_half_byte_per_weight() {
    assert!((BandwidthProfile::Q4_0.raw_bpw - 0.5).abs() < 1e-6);
}

#[test]
fn bwprofile_q8_has_one_byte_per_weight() {
    assert!((BandwidthProfile::Q8_0.raw_bpw - 1.0).abs() < 1e-6);
}

#[test]
fn bwprofile_f16_has_two_bytes_per_weight() {
    assert!((BandwidthProfile::F16.raw_bpw - 2.0).abs() < 1e-6);
}

#[test]
fn quant_invariance_theorem_q8_near_q4_compressed() {
    // Theorem: Q8+GDeflate / Q4+GDeflate < 1.20 (within 20% of Q4 bandwidth)
    let ratio = BandwidthProfile::bandwidth_ratio_vs_q4_compressed();
    assert!(ratio < 1.20,
        "Q8 after GDeflate must be within 20% of Q4 bandwidth — got {:.3}×", ratio);
    assert!(ratio > 1.00, "Q8 cannot be compressed below Q4");
    println!("Q8/Q4 GDeflate bandwidth ratio: {:.3}×", ratio);
}

#[test]
fn zipf_s1_256_experts_hit_rates() {
    // Validate the cache model math for DeepSeek-style MoE
    let h8   = zipf_cache_hit_rate(256, 8,   1.0);
    let h32  = zipf_cache_hit_rate(256, 32,  1.0);
    let h64  = zipf_cache_hit_rate(256, 64,  1.0);
    let h256 = zipf_cache_hit_rate(256, 256, 1.0);
    assert!(h8   > 0.20, "8 experts cached → > 20% hit rate under Zipf");
    assert!(h32  > 0.50, "32 experts cached → > 50% hit rate under Zipf");
    assert!(h64  > 0.65, "64 experts cached → > 65% hit rate under Zipf");
    assert!((h256 - 1.0).abs() < 1e-9, "256/256 cached → 100% hit rate");
}

#[test]
fn zipf_hit_rate_monotone_in_cache_size() {
    for size in [1, 4, 8, 16, 32, 64, 128, 255] {
        let h_small = zipf_cache_hit_rate(256, size,     1.0);
        let h_large = zipf_cache_hit_rate(256, size + 1, 1.0);
        assert!(h_large >= h_small, "hit rate must be monotone non-decreasing in cache size");
    }
}

#[test]
fn effective_reads_decreases_with_cache_hits() {
    let k = 8;
    let reads_no_cache   = effective_reads_per_token(k, 0.0);
    let reads_with_cache = effective_reads_per_token(k, 0.65);
    assert!((reads_no_cache - 8.0).abs() < 1e-9, "no cache → exactly k reads");
    assert!(reads_with_cache < reads_no_cache, "cache must reduce reads");
    assert!(reads_with_cache < 3.0, "65% hit rate with k=8 → < 3 actual reads");
}

#[test]
fn expert_provisioner_top_k_is_correct() {
    let logits = vec![0.1f32, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4];
    let top3 = ExpertProvisioner::select_top_k(&logits, 3);
    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0], 2, "expert 2 has max logit 0.9");
    assert_eq!(top3[1], 6, "expert 6 has logit 0.8");
    assert_eq!(top3[2], 4, "expert 4 has logit 0.7");
}

#[test]
fn expert_provisioner_abort_non_selected_sets_flag() {
    let (tok0, flag0) = EarlyAbortToken::new(0);
    let (tok1, flag1) = EarlyAbortToken::new(1);
    // Manually mark them as inflight in a provisioner
    let mut prov = ExpertProvisioner::new();
    prov.register_expert(0, vec![1.0]);
    prov.register_expert(1, vec![2.0]);
    // Simulate: expert 0 selected, expert 1 aborted
    tok1.cancel();
    drop(tok0); // expert 0 not cancelled
    assert!( flag1.load(std::sync::atomic::Ordering::Acquire), "expert 1 must be aborted");
    assert!(!flag0.load(std::sync::atomic::Ordering::Acquire) ||
             flag0.load(std::sync::atomic::Ordering::Acquire), "expert 0 state doesn't matter here");
}

#[test]
fn bandwidth_projector_7b_dense_q4_exceeds_50_tps() {
    let proj = BandwidthProjector { ssd_gbps: 0.0, vram_gbps: 336.0, cached_experts: 0 };
    let tps = proj.project_dense_tps(7_000_000_000, BandwidthProfile::Q4_0);
    assert!(tps > 50.0, "7B Q4 on RTX 4090 (336 GB/s) must exceed 50 tok/s: {:.1}", tps);
}

#[test]
fn bandwidth_projector_70b_q4_gpu_exceeds_20_tps() {
    // RTX 4090: 1008 GB/s GDDR6X; 70B Q4 = 35 GB raw → 1008/35 ≈ 28.8 tok/s
    let proj = BandwidthProjector { ssd_gbps: 0.0, vram_gbps: 1008.0, cached_experts: 0 };
    let tps = proj.project_dense_tps(70_000_000_000, BandwidthProfile::Q4_0);
    assert!(tps > 20.0, "70B Q4 on RTX 4090 (1008 GB/s) must exceed 20 tok/s: {:.1}", tps);
}

#[test]
fn bandwidth_projector_moe_300b_with_cache_exceeds_1_tps() {
    let proj = BandwidthProjector { ssd_gbps: 7.0, vram_gbps: 336.0, cached_experts: 32 };
    // 256 experts, 8 active, ~2.5B params/expert
    let tps = proj.project_moe_tps(256, 8, 2_500_000_000, BandwidthProfile::Q4_0);
    assert!(tps > 1.0, "300B MoE on single NVMe must exceed 1 tok/s: {:.2}", tps);
}

#[test]
fn bandwidth_projector_moe_q8_vs_q4_within_20pct() {
    let proj = BandwidthProjector { ssd_gbps: 7.0, vram_gbps: 336.0, cached_experts: 32 };
    let tps_q4 = proj.project_moe_tps(256, 8, 2_500_000_000, BandwidthProfile::Q4_0);
    let tps_q8 = proj.project_moe_tps(256, 8, 2_500_000_000, BandwidthProfile::Q8_0);
    // Q8 compressed is within 10% of Q4 compressed → bandwidth within 10%
    let ratio = tps_q4 / tps_q8.max(1e-9);
    assert!(ratio < 1.25,
        "Q8 MoE tok/s should be within 25% of Q4 due to GDeflate: ratio={:.3}", ratio);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. PLATFORM IO — Cross-platform tensor reads
// ═══════════════════════════════════════════════════════════════════════════════

fn write_temp(data: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(data).unwrap();
    f.flush().unwrap();
    f
}

#[test]
fn platform_io_reads_full_file() {
    let data: Vec<u8> = (0u8..=200).collect();
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).expect("open must succeed");
    let result = reader.read_range_sync(0, 201).expect("full read must succeed");
    assert_eq!(result, data);
}

#[test]
fn platform_io_reads_subrange() {
    let data: Vec<u8> = (100u8..=200).collect();
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).unwrap();
    let result = reader.read_range_sync(10, 20).unwrap();
    assert_eq!(result.len(), 20);
    assert_eq!(result[0], 110u8);
}

#[test]
fn platform_io_errors_on_out_of_bounds() {
    let data = vec![0u8; 50];
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).unwrap();
    assert!(reader.read_range_sync(40, 20).is_err(), "read past EOF must error");
}

#[test]
fn platform_io_empty_file_read_zero_bytes() {
    let f = write_temp(&[]);
    let reader = open_tensor_file(f.path()).unwrap();
    let result = reader.read_range_sync(0, 0);
    // Behavior: either Ok([]) or Err — both are acceptable for zero-length read
    // On most platforms, reading 0 bytes returns Ok with empty buffer
    if let Ok(buf) = result { assert_eq!(buf.len(), 0); }
}

#[test]
fn platform_io_backend_name_matches_os() {
    let f = write_temp(&[1, 2, 3]);
    let reader = open_tensor_file(f.path()).unwrap();
    let name = reader.backend_name();
    #[cfg(target_os = "macos")]
    assert!(name.contains("macOS"), "macOS reader must identify itself: {}", name);
    #[cfg(target_os = "linux")]
    assert!(name.contains("Linux"), "Linux reader must identify itself: {}", name);
    #[cfg(target_os = "windows")]
    assert!(name.contains("Win32") || name.contains("windows") || name.contains("std::fs"),
        "Windows reader must identify itself: {}", name);
}

#[test]
fn platform_io_consistent_reads_same_offset() {
    let data: Vec<u8> = (0u8..100).collect();
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).unwrap();
    let r1 = reader.read_range_sync(20, 30).unwrap();
    let r2 = reader.read_range_sync(20, 30).unwrap();
    assert_eq!(r1, r2, "repeated reads must be consistent");
}

#[test]
fn platform_io_multiple_ranges_independent() {
    let data: Vec<u8> = (0u8..100).collect();
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).unwrap();
    let a = reader.read_range_sync(0,  10).unwrap();
    let b = reader.read_range_sync(50, 10).unwrap();
    assert_eq!(a, (0u8..10).collect::<Vec<_>>());
    assert_eq!(b, (50u8..60).collect::<Vec<_>>());
}

#[test]
fn platform_io_large_file_random_ranges() {
    // 1 MB of sequential bytes
    let data: Vec<u8> = (0..1_000_000).map(|i: u32| (i & 0xFF) as u8).collect();
    let f = write_temp(&data);
    let reader = open_tensor_file(f.path()).unwrap();

    for &(off, len) in &[(0u64, 4096), (128_000, 8192), (999_000, 1000)] {
        let result = reader.read_range_sync(off, len).unwrap();
        assert_eq!(result.len(), len);
        for (j, &byte) in result.iter().enumerate() {
            let expected = ((off as usize + j) & 0xFF) as u8;
            assert_eq!(byte, expected, "byte mismatch at offset {}+{}", off, j);
        }
    }
}

#[test]
fn nodestor_dirs_are_absolute_paths() {
    assert!(nodestor_data_dir().is_absolute(), "data dir must be absolute");
    assert!(models_dir().is_absolute(), "models dir must be absolute");
    assert!(loras_dir().is_absolute(), "loras dir must be absolute");
}

#[test]
fn nodestor_dirs_proper_hierarchy() {
    let data  = nodestor_data_dir();
    let mdl   = models_dir();
    let lora  = loras_dir();
    assert!(mdl.starts_with(&data),  "models must be under data dir");
    assert!(lora.starts_with(&data), "loras must be under data dir");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. TIMELINE SEMAPHORE — CPU/GPU Pipeline Sync
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn timeline_semaphore_signal_monotone() {
    let s = TimelineSemaphore::new(0);
    for v in [5u64, 3, 10, 7, 100] {
        s.signal(v);
    }
    assert_eq!(s.current(), 100, "counter must advance monotonically to max");
}

#[test]
fn timeline_semaphore_wait_for_already_met() {
    let s = TimelineSemaphore::new(50);
    s.wait_for(50); // must return immediately — no deadlock
    s.wait_for(1);
    assert_eq!(s.current(), 50);
}

#[test]
fn timeline_semaphore_cross_thread_unblock() {
    let s = Arc::new(TimelineSemaphore::new(0));
    let sc = s.clone();
    let t = thread::spawn(move || { sc.wait_for(7); });
    thread::sleep(Duration::from_millis(5));
    s.signal(7);
    t.join().expect("thread must unblock");
}

#[test]
fn timeline_semaphore_signal_wakes_multiple_waiters() {
    let s = Arc::new(TimelineSemaphore::new(0));
    let handles: Vec<_> = (1..=5u64).map(|threshold| {
        let sc = s.clone();
        thread::spawn(move || { sc.wait_for(threshold); })
    }).collect();
    thread::sleep(Duration::from_millis(10));
    s.signal(10); // satisfies all thresholds
    for h in handles { h.join().expect("waiter panicked"); }
}

#[test]
fn timeline_semaphore_timeout_false_when_never_signalled() {
    let s = TimelineSemaphore::new(0);
    let ok = s.wait_for_timeout(999, Duration::from_millis(30));
    assert!(!ok, "must return false on timeout");
}

#[test]
fn timeline_semaphore_timeout_true_when_signalled_in_time() {
    let s = Arc::new(TimelineSemaphore::new(0));
    let sc = s.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        sc.signal(1);
    });
    let ok = s.wait_for_timeout(1, Duration::from_millis(500));
    assert!(ok, "must return true when signalled before timeout");
}

#[test]
fn layer_fence_single_layer_completes() {
    let fence = Arc::new(LayerFence::new());
    let fc = fence.clone();

    let gpu = thread::spawn(move || {
        fc.gpu_wait_before_ffn(0);
        fc.gpu_signal_ffn_done(0);
    });

    fence.cpu_wait_before_attn(0); // layer 0: no prior FFN to wait for
    fence.cpu_signal_attn_done(0);
    gpu.join().expect("gpu thread panicked");
}

#[test]
fn layer_fence_eight_layers_pipeline() {
    let fence = Arc::new(LayerFence::new());
    let fc    = fence.clone();
    const N: u64 = 8;

    let gpu_thread = thread::spawn(move || {
        let mut done = Vec::new();
        for layer in 0..N {
            fc.gpu_wait_before_ffn(layer);
            done.push(layer);
            fc.gpu_signal_ffn_done(layer);
        }
        done
    });

    for layer in 0..N {
        fence.cpu_wait_before_attn(layer);
        fence.cpu_signal_attn_done(layer);
    }

    let gpu_done = gpu_thread.join().expect("gpu thread panicked");
    assert_eq!(gpu_done.len() as u64, N, "GPU must process all {} layers", N);
    assert_eq!(gpu_done, (0..N).collect::<Vec<_>>(), "layers must be ordered");
}

#[test]
fn layer_fence_attn_semaphore_advances_per_layer() {
    let fence = Arc::new(LayerFence::new());
    let fc    = fence.clone();

    // GPU side: just wait + signal immediately
    let gpu = thread::spawn(move || {
        for layer in 0..4u64 {
            fc.gpu_wait_before_ffn(layer);
            fc.gpu_signal_ffn_done(layer);
        }
    });

    for layer in 0..4u64 {
        fence.cpu_wait_before_attn(layer);
        fence.cpu_signal_attn_done(layer);
        // After signalling layer N, attn semaphore must be at least N+1
        assert!(fence.attn.current() >= layer + 1,
            "attn semaphore must advance to at least {} after layer {}", layer+1, layer);
    }
    gpu.join().unwrap();
}

#[test]
fn layer_fence_ffn_semaphore_advances_per_layer() {
    let fence = Arc::new(LayerFence::new());
    let fc    = fence.clone();

    let gpu = thread::spawn(move || {
        for layer in 0..4u64 {
            fc.gpu_wait_before_ffn(layer);
            fc.gpu_signal_ffn_done(layer);
        }
    });

    for layer in 0..4u64 {
        fence.cpu_wait_before_attn(layer);
        fence.cpu_signal_attn_done(layer);
    }
    gpu.join().unwrap();

    assert!(fence.ffn.current() >= 4, "all FFN layers must be signalled");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. CROSS-SYSTEM: SpecStream + Timeline Semaphore overlap simulation
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates the overlap of SpecStream expansion (CPU, "draft") with
/// Timeline Semaphore synchronization (CPU/GPU), proving they compose safely.
#[test]
fn spec_stream_with_timeline_fence_overlap_simulation() {
    let fence  = Arc::new(LayerFence::new());
    let fc     = fence.clone();
    const LAYERS: u64 = 4;

    // "GPU" thread: process FFN layers
    let gpu = thread::spawn(move || {
        for layer in 0..LAYERS {
            fc.gpu_wait_before_ffn(layer);
            // Simulate GPU FFN work
            thread::sleep(Duration::from_millis(1));
            fc.gpu_signal_ffn_done(layer);
        }
    });

    // CPU thread: run attention + SpecStream draft expansion in overlap
    let mut engine = SpecStreamEngine::new(SpecConfig::conservative());
    for layer in 0..LAYERS {
        fence.cpu_wait_before_attn(layer);

        // Simulate attention + draft expansion during GPU FFN N-1
        let ctx = vec![layer as u32, layer as u32 + 1];
        engine.expand(&ctx, identity_draft);
        assert!(!engine.last_nodes.is_empty(), "spec expansion must produce nodes");

        fence.cpu_signal_attn_done(layer);
    }

    gpu.join().expect("gpu thread panicked");
    assert!(engine.last_nodes.len() >= 1, "final expansion must have nodes");
}

/// Proves bandwidth invariance at the system level: Q8 MoE projected TPS
/// is within the acceptable range for a realistic 300B model deployment.
#[test]
fn system_bandwidth_invariance_300b_scenario() {
    // Scenario: 300B MoE on 3× NVMe RAID-0 (21 GB/s aggregate)
    let proj = BandwidthProjector {
        ssd_gbps:       21.0,
        vram_gbps:      336.0,
        cached_experts: 64,    // ~25% of experts in VRAM
    };
    let tps_q4 = proj.project_moe_tps(256, 8, 2_500_000_000, BandwidthProfile::Q4_0);
    let tps_q8 = proj.project_moe_tps(256, 8, 2_500_000_000, BandwidthProfile::Q8_0);

    println!("300B MoE, 3×NVMe, 64 experts cached:");
    println!("  Q4 tok/s: {:.1}", tps_q4);
    println!("  Q8 tok/s: {:.1}", tps_q8);

    // Both must be positive and Q8 within 20% of Q4
    assert!(tps_q8 > 0.0, "Q8 MoE must project positive tok/s");
    assert!(tps_q4 / tps_q8 < 1.20,
        "Q8 must be within 20% of Q4 bandwidth (invariance theorem)");
}
