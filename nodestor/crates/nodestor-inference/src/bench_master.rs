//! # MASTER BENCHMARK — Prova Empírica Total do Framework Davi / COBER v2
//!
//! Testa CADA subsistema sob carga extrema e produz relatório forense.
//! Cada cenário mapeia uma "Muralha Fundamental da IA" do documento de auditoria.

#[cfg(test)]
mod tests {
    use std::time::Instant;

    // ═══════════════════════════════════════════════════════════════════
    //  UTILIDADES
    // ═══════════════════════════════════════════════════════════════════

    struct BenchResult {
        name: &'static str,
        wall: &'static str,
        passed: bool,
        duration_us: u128,
        metric_name: &'static str,
        metric_value: f64,
        detail: String,
    }

    fn print_header() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║     MASTER BENCHMARK — Framework Davi / COBER v2               ║");
        println!("║     Prova Empírica Total: 8 Cenários × 15 Muralhas da IA       ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
    }

    fn print_result(r: &BenchResult) {
        let s = if r.passed { "PASS" } else { "FAIL" };
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ {} | {}", s, r.name);
        println!("│ Muralha: {}", r.wall);
        println!("│ Tempo: {}us | {}: {:.4}", r.duration_us, r.metric_name, r.metric_value);
        println!("│ {}", r.detail);
        println!("└─────────────────────────────────────────────────────────────────┘");
    }

    fn print_final(results: &[BenchResult]) {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  RESULTADO FINAL                                               ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        for r in results {
            let icon = if r.passed { "OK" } else { "XX" };
            println!("║ {} {:<54} {:>6}us ║", icon, r.name, r.duration_us);
        }
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  TOTAL: {} testes | {} passed | {} failed                       ║",
            total, passed, total - passed);
        println!("╚══════════════════════════════════════════════════════════════════╝");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  O MASTER TEST
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn master_benchmark_all_subsystems() {
        print_header();
        let mut results = Vec::new();

        results.push(scenario_1_error_accumulation());
        results.push(scenario_2_vram_collapse());
        results.push(scenario_3_multimodal_symphony());
        results.push(scenario_4_vulkan_fuzzing());
        results.push(scenario_5_crystal_pipeline());
        results.push(scenario_6_fractal_resilience());
        results.push(scenario_7_immune_system());
        results.push(scenario_8_topological_discovery());

        print_final(&results);

        let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
        assert!(
            failed.is_empty(),
            "MASTER BENCHMARK FALHOU em {} cenarios: {:?}",
            failed.len(),
            failed.iter().map(|r| r.name).collect::<Vec<_>>()
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 1: Anti-Acumulação de Erros (Muralha 4)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_1_error_accumulation() -> BenchResult {
        use crate::vram_budget::{VramBudget, InferenceMode};
        use crate::cober::CoberEngine;

        let start = Instant::now();
        let budget = VramBudget::new(
            8 * 1024 * 1024 * 1024,
            (8.0 * 0.18) as u64 * 1024 * 1024 * 1024,
            InferenceMode::Dense,
        );
        let mut engine = CoberEngine::new_dense(budget);
        let mut context = vec![1u32, 2, 3, 4, 5];
        let hidden = vec![0.5f32; 128];

        let total_tokens = 2000;
        let mut draft_accepted = 0u64;
        let mut draft_rejected = 0u64;
        let mut total_drafted = 0u64;

        for i in 0..total_tokens {
            let draft = engine.draft_with_crystal_skeleton(&context, &hidden);
            total_drafted += draft.len() as u64;
            // Simulated master verification: reject tokens > 40000 (unlikely real tokens)
            let verified: Vec<u32> = draft.iter().filter(|t| **t < 40000).copied().collect();
            draft_accepted += verified.len() as u64;
            draft_rejected += (draft.len() - verified.len()) as u64;
            if let Some(&first) = verified.first() {
                context.push(first);
            } else {
                context.push((i % 50000 + 100) as u32);
            }
            if let Some(golden) = &mut engine.golden_ngrams {
                if context.len() >= 3 && !verified.is_empty() {
                    golden.insert_verified(&context, &verified);
                }
            }
        }

        let elapsed = start.elapsed();
        let acceptance_rate = if total_drafted > 0 {
            draft_accepted as f64 / total_drafted as f64
        } else { 0.0 };
        // The COBER must have drafted tokens and finished fast
        let passed = total_drafted > 0 && elapsed.as_millis() < 500;

        let r = BenchResult {
            name: "Cenario 1: Anti-Acumulacao de Erros",
            wall: "Muralha 4 - Autoregressao Exponencial",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "acceptance_rate",
            metric_value: acceptance_rate,
            detail: format!(
                "Drafted: {} | Accepted: {} | Rejected: {} | Ctx: {} tokens",
                total_drafted, draft_accepted, draft_rejected, context.len()
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 2: Esmagamento VRAM 70B em 8GB (Muralha 7)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_2_vram_collapse() -> BenchResult {
        use crate::vulkan_adaptive::{VirtualVram, MemoryTier};

        let start = Instant::now();
        let num_layers = 140u64;
        let layer_size = 250 * 1024 * 1024u64; // 250MB
        let vram_budget = 8u64 * 1024 * 1024 * 1024;
        let ram_budget = 32u64 * 1024 * 1024 * 1024;

        let mut vram = VirtualVram::new(vram_budget, ram_budget);
        for i in 0..num_layers {
            vram.register_page(i, layer_size, MemoryTier::Ssd);
        }

        let mut tdr_count = 0u64;
        let mut total_requests = 0u64;

        for pass in 0..5u64 {
            for layer in 0..num_layers {
                total_requests += 1;
                let epoch = pass * num_layers + layer;
                match vram.request_page(layer, MemoryTier::Vram, epoch) {
                    Ok(_) => {}
                    Err(_) => { tdr_count += 1; }
                }
            }
        }

        let elapsed = start.elapsed();
        let hit_rate = vram.hit_rate();
        let vram_safe = vram.vram_used_bytes <= vram_budget;
        let passed = vram_safe && tdr_count == 0;

        let r = BenchResult {
            name: "Cenario 2: Esmagamento VRAM 70B->8GB",
            wall: "Muralha 7 - Abismo Energetico",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "hit_rate",
            metric_value: hit_rate,
            detail: format!(
                "Camadas: {} | Requests: {} | TDR: {} | VRAM: {}MB/{}MB | Prom: {} | Dem: {}",
                num_layers, total_requests, tdr_count,
                vram.vram_used_bytes / (1024 * 1024), vram_budget / (1024 * 1024),
                vram.stats.promotions, vram.stats.demotions,
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 3: Sinfonia Multimodal 3 vias (Muralha 3)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_3_multimodal_symphony() -> BenchResult {
        use crate::jitter_buffer::{JitterBuffer, SymphonyConfig};
        use crate::cross_modal::{ModalityType, ModalDraft};

        let start = Instant::now();
        let config = SymphonyConfig {
            expected_modalities: vec![
                ModalityType::Text,
                ModalityType::Image,
                ModalityType::Audio,
            ],
            pulse_interval_ns: 5_000_000,
            tensor_alignment: 256,
            ..Default::default()
        };
        let mut buffer = JitterBuffer::new(config);

        let total_concepts = 1000u64;

        for i in 0..total_concepts {
            let base_ns = i * 10_000; // 10us between concepts

            // Text arrives first
            buffer.ingest_draft(i, &ModalDraft {
                modality: ModalityType::Text,
                draft_tokens: vec![i as u32; 30],
                confidence: 0.95,
                projected_embedding: vec![0.5; 4],
            }, base_ns);

            // Image arrives 1us later
            buffer.ingest_draft(i, &ModalDraft {
                modality: ModalityType::Image,
                draft_tokens: vec![i as u32; 4],
                confidence: 0.85,
                projected_embedding: vec![0.5; 4],
            }, base_ns + 1_000);

            // Audio arrives 2us later
            buffer.ingest_draft(i, &ModalDraft {
                modality: ModalityType::Audio,
                draft_tokens: vec![i as u32; 16],
                confidence: 0.80,
                projected_embedding: vec![0.5; 4],
            }, base_ns + 2_000);
        }

        let elapsed = start.elapsed();
        let per_concept_us = elapsed.as_micros() as f64 / total_concepts as f64;
        let complete = buffer.stats.concepts_dispatched;
        let incomplete = buffer.stats.incomplete_dispatches;

        let passed = complete == total_concepts
            && incomplete == 0
            && per_concept_us < 200.0;

        let r = BenchResult {
            name: "Cenario 3: Sinfonia Multimodal 3 Vias",
            wall: "Muralha 3 - Ausencia de Modelo de Mundo",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "us/conceito",
            metric_value: per_concept_us,
            detail: format!(
                "Conceitos: {} | Completos: {} | Incompletos: {} | Jitter: {:.0}ns",
                total_concepts, complete, incomplete, buffer.stats.avg_jitter_ns,
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 4: Fuzzing Vulkan 1000 mutações (Muralha 11)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_4_vulkan_fuzzing() -> BenchResult {
        use crate::vulkan_adaptive::{AdaptivePipeline, GpuProfile, GpuVendor};

        let start = Instant::now();

        // Teste com 4 perfis de GPU diferentes para provar adaptação universal
        let gpu_profiles = vec![
            ("RTX 4090", GpuVendor::Nvidia, 24u64, 128u32, 32u32, 1024u32),
            ("Intel UHD 770", GpuVendor::Intel, 2u64, 32u32, 16u32, 512u32),
            ("AMD RX 7900", GpuVendor::Amd, 20u64, 96u32, 64u32, 1024u32),
            ("Qualcomm Adreno", GpuVendor::Qualcomm, 4u64, 8u32, 32u32, 256u32),
        ];

        let mut total_mutations = 0u64;
        let mut total_valid = 0u64;

        for (name, vendor, vram_gb, cus, subgrp, max_wg) in &gpu_profiles {
            let gpu = GpuProfile {
                name: name.to_string(),
                vendor: vendor.clone(),
                vram_total_bytes: vram_gb * 1024 * 1024 * 1024,
                vram_free_bytes: (vram_gb * 3 / 4) * 1024 * 1024 * 1024,
                memory_bandwidth_gbps: 500.0,
                compute_units: *cus,
                max_workgroup_size_x: *max_wg,
                max_workgroup_size_y: *max_wg,
                max_shared_memory_bytes: 49152,
                supports_subgroup_ops: *subgrp >= 16,
                subgroup_size: *subgrp,
            };

            let mut pipeline = AdaptivePipeline::new(gpu);
            let mutations = pipeline.adapt_to_hardware().unwrap();
            total_mutations += mutations as u64;

            // Verifica integridade pós-adaptação
            let integrity = pipeline.verify_all_templates();
            if integrity.is_ok() {
                total_valid += 1;
            }
        }

        let elapsed = start.elapsed();
        let all_valid = total_valid == gpu_profiles.len() as u64;

        // Gera audit report do último pipeline
        let gpu_final = GpuProfile {
            name: "Audit GPU".to_string(),
            vendor: GpuVendor::Nvidia,
            vram_total_bytes: 8 * 1024 * 1024 * 1024,
            vram_free_bytes: 6 * 1024 * 1024 * 1024,
            memory_bandwidth_gbps: 500.0,
            compute_units: 64,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_shared_memory_bytes: 49152,
            supports_subgroup_ops: true,
            subgroup_size: 32,
        };
        let mut final_pipe = AdaptivePipeline::new(gpu_final);
        let _ = final_pipe.adapt_to_hardware();
        let integrity = final_pipe.verify_all_templates();
        let passed = all_valid && integrity.is_ok();
        let report_lines = final_pipe.audit_report().lines().count();

        let r = BenchResult {
            name: "Cenario 4: Vulkan Multi-GPU Adapt (4 GPUs)",
            wall: "Muralha 11 - Opacidade da Caixa Preta",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "mutations_applied",
            metric_value: total_mutations as f64,
            detail: format!(
                "GPUs: {} | Mutations: {} | Valid: {} | Integrity: {} | Audit: {} lines | Hash: {:#018x}",
                gpu_profiles.len(), total_mutations, total_valid,
                if integrity.is_ok() { "OK" } else { "BROKEN" },
                report_lines, final_pipe.last_state_hash,
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 5: Crystal Pipeline 10K drafts (Muralha 4)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_5_crystal_pipeline() -> BenchResult {
        use crate::vram_budget::{VramBudget, InferenceMode};
        use crate::cober::CoberEngine;

        let start = Instant::now();
        let budget = VramBudget::new(
            16 * 1024 * 1024 * 1024,
            (16.0 * 0.18) as u64 * 1024 * 1024 * 1024,
            InferenceMode::Dense,
        );
        let mut engine = CoberEngine::new_dense(budget);
        let mut context = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let hidden = vec![0.5f32; 128];

        let iterations = 10_000u64;
        let mut total_drafted = 0u64;

        for i in 0..iterations {
            let draft = engine.draft_with_crystal_skeleton(&context, &hidden);
            total_drafted += draft.len() as u64;
            if let Some(&tok) = draft.first() {
                context.push(tok);
            } else {
                context.push((i % 50000 + 100) as u32);
            }
            if let Some(golden) = &mut engine.golden_ngrams {
                if context.len() >= 3 {
                    let tail = &context[context.len()-3..];
                    golden.insert_verified(tail, &[42, 43, 44]);
                }
            }
        }

        let elapsed = start.elapsed();
        let drafts_per_sec = iterations as f64 / elapsed.as_secs_f64();
        let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
        let passed = elapsed.as_millis() < 300 && total_drafted > 0;

        let r = BenchResult {
            name: "Cenario 5: Crystal Pipeline 10K Drafts",
            wall: "Muralha 4 - Throughput O(1)",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "drafts/sec",
            metric_value: drafts_per_sec,
            detail: format!(
                "{} iters | {} drafted | {:.1}ns/draft | Ctx: {}",
                iterations, total_drafted, ns_per, context.len(),
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 6: Resiliência Fractal L0→L3 (Muralha 8)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_6_fractal_resilience() -> BenchResult {
        use crate::fractal_memory::FractalMemory;

        let start = Instant::now();
        // l0_max=50, l1_max=50, l2_max=50
        let mut memory = FractalMemory::new(50, 50, 50);

        let tokens_to_ingest = 500;
        for i in 0..tokens_to_ingest {
            // Push blocks of 10 tokens each
            let block_tokens: Vec<u32> = (i*10..i*10+10).map(|x| x as u32).collect();
            memory.push_l0_block(block_tokens);
        }

        let elapsed = start.elapsed();

        // Count blocks per level
        let l0 = memory.blocks.values()
            .filter(|b| b.level == crate::fractal_memory::MemoryLevel::L0Tokens).count();
        let l1 = memory.blocks.values()
            .filter(|b| b.level == crate::fractal_memory::MemoryLevel::L1Vectors).count();
        let l2 = memory.blocks.values()
            .filter(|b| b.level == crate::fractal_memory::MemoryLevel::L2Summaries).count();
        let l3 = memory.blocks.values()
            .filter(|b| b.level == crate::fractal_memory::MemoryLevel::L3Index).count();
        let total = memory.blocks.len();

        // Must have retention and cascading
        let passed = total > 0 && (l1 > 0 || l2 > 0 || l3 > 0) && elapsed.as_millis() < 200;

        let r = BenchResult {
            name: "Cenario 6: Resiliencia Fractal L0-L3",
            wall: "Muralha 8 - Muro de Dados",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "total_blocks",
            metric_value: total as f64,
            detail: format!(
                "Ingested: {} | L0: {} | L1: {} | L2: {} | L3: {} | Total: {}",
                tokens_to_ingest, l0, l1, l2, l3, total,
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 7: Sistema Imunológico O(1) (Muralha 1)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_7_immune_system() -> BenchResult {
        use crate::bloom_filter::TokenBloomFilter;
        use crate::golden_ngrams::GoldenNgramCache;

        let start = Instant::now();

        // BLOOM FILTER: 10K valid, test 50K
        // BLOOM FILTER: opere com n-gramas (slices de u32)
        let mut bloom = TokenBloomFilter::new(100_000, 0.01);
        // Insere 1000 n-gramas como "válidos"
        for i in 0..1000u32 {
            bloom.insert(&[i, i + 1, i + 2]);
        }

        let mut tp = 0u64;
        let mut tn = 0u64;
        let mut fp = 0u64;

        // Testa 2000 n-gramas: 1000 válidos + 1000 inválidos
        for i in 0..2000u32 {
            let ngram = if i < 1000 {
                vec![i, i + 1, i + 2]  // Válido
            } else {
                vec![i + 50000, i + 50001, i + 50002] // Inválido
            };
            let result = bloom.maybe_valid(&ngram);
            if i < 1000 {
                if result { tp += 1; }
            } else {
                if result { fp += 1; } else { tn += 1; }
            }
        }

        // GOLDEN N-GRAMS: 1000 sequences
        let mut ngrams = GoldenNgramCache::new(5000, 3);
        for i in 0..1000u32 {
            let ctx = vec![i, i + 1, i + 2];
            let cont = vec![i + 3, i + 4, i + 5];
            ngrams.insert_verified(&ctx, &cont);
        }

        let mut hits = 0u64;
        let mut misses = 0u64;
        for i in 0..2000u32 {
            let ctx = vec![i, i + 1, i + 2];
            match ngrams.try_get(&ctx) {
                Some(_) => hits += 1,
                None => misses += 1,
            }
        }

        let elapsed = start.elapsed();
        let fp_rate = fp as f64 / 1_000.0;
        let bloom_ok = tp == 1000 && fp_rate < 0.05;
        let ngram_ok = hits >= 900;
        let passed = bloom_ok && ngram_ok && elapsed.as_millis() < 200;

        let r = BenchResult {
            name: "Cenario 7: Sistema Imunologico O(1)",
            wall: "Muralha 1 - Logica Quebradica",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "bloom_fp_rate",
            metric_value: fp_rate,
            detail: format!(
                "Bloom: TP={} FP={} TN={} FP%={:.4}% | NGrams: H={} M={}",
                tp, fp, tn, fp_rate * 100.0, hits, misses,
            ),
        };
        print_result(&r);
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    //  CENÁRIO 8: Descoberta Topológica (Muralha 2)
    // ═══════════════════════════════════════════════════════════════════

    fn scenario_8_topological_discovery() -> BenchResult {
        use crate::insight_indexer::{InsightIndexer, Modality, SynapseType};

        let start = Instant::now();
        let mut indexer = InsightIndexer::new(0.1);

        // Cluster A: Física (embeddings near [1,0,0,...])
        for i in 0..100u64 {
            let mut embed = vec![0.0f32; 16];
            embed[0] = 1.0 + (i as f32) * 0.01;
            embed[1] = (i as f32) * 0.005;
            indexer.record_insight(
                &[i as u32, (i+1) as u32],
                embed,
                vec![(i+100) as u32],
                Modality::Text,
                vec!["Fisica".to_string()],
            );
        }

        // Cluster B: Biologia
        for i in 100..200u64 {
            let mut embed = vec![0.0f32; 16];
            embed[0] = (i as f32 - 100.0) * 0.005;
            embed[1] = 1.0 + (i as f32 - 100.0) * 0.01;
            indexer.record_insight(
                &[i as u32, (i+1) as u32],
                embed,
                vec![(i+100) as u32],
                Modality::Text,
                vec!["Biologia".to_string()],
            );
        }

        // Cluster C: Economia
        for i in 200..300u64 {
            let mut embed = vec![0.0f32; 16];
            embed[2] = 1.0 + (i as f32 - 200.0) * 0.01;
            indexer.record_insight(
                &[i as u32, (i+1) as u32],
                embed,
                vec![(i+100) as u32],
                Modality::Text,
                vec!["Economia".to_string()],
            );
        }

        // Search physics region
        let mut query = vec![0.0f32; 16];
        query[0] = 1.05;
        let similar = indexer.search_nearby(&query, 10);
        let similar_found = similar.len();

        // Record rejections to create knowledge gaps
        let mut gap_embed = vec![0.0f32; 16];
        gap_embed[0] = 0.5;
        gap_embed[1] = 0.5; // Between physics and biology
        indexer.record_rejection(gap_embed.clone(), vec!["Desconhecido".to_string()]);
        indexer.record_rejection(gap_embed.clone(), vec!["Desconhecido".to_string()]);
        indexer.record_rejection(gap_embed.clone(), vec!["Desconhecido".to_string()]);

        // Connect insights and test Hebbian
        indexer.connect_insights(0, 100, SynapseType::CrossDomain);
        indexer.hebbian_strengthen(&[0, 100]);
        let synapse_w = indexer.synapses.first().map(|s| s.weight).unwrap_or(0.0);

        // Cross-domain search
        let cross = indexer.find_cross_domain_candidates(&query, "Fisica", 5);
        let cross_found = cross.len();

        // Gaps: computed after mutable borrows are done
        let gaps_found = indexer.critical_gaps(5).len();

        let elapsed = start.elapsed();

        let passed = gaps_found > 0
            && similar_found > 0
            && cross_found > 0
            && synapse_w > 1.0  // Hebbian strengthened
            && elapsed.as_millis() < 200;

        let r = BenchResult {
            name: "Cenario 8: Descoberta Topologica",
            wall: "Muralha 2 - Escada de Pearl",
            passed,
            duration_us: elapsed.as_micros(),
            metric_name: "gaps_detected",
            metric_value: gaps_found as f64,
            detail: format!(
                "Clusters: 3 | Insights: 300 | Gaps: {} | Similar: {} | Cross: {} | Synapse: {:.3}",
                gaps_found, similar_found, cross_found, synapse_w,
            ),
        };
        print_result(&r);
        r
    }
}
