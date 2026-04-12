/// NodeStor COBER v2 — Fase 12: Sincronizador de Tensores (Sinfonia)
///
/// O "Maestro" da Orquestra Multimodal.
///
/// Problema: Texto, Imagem e Áudio chegam à GPU em tempos diferentes.
///           Se o texto chega em 1ms e a imagem em 3ms, a GPU valida o texto
///           sozinho e perde a coerência cross-modal.
///
/// Solução:  Jitter Buffer Semântico + Systolic Pulse.
///           O Maestro segura todos os rascunhos até o mais lento chegar,
///           depois despacha o Conceito Unificado de uma vez para a VRAM.
///
/// Inspiração: TPUs do Google (systolic array), jitter buffers de VoIP (Opus),
///            e pipeline de GPU estilo wavefront.
///
/// Funciona com QUALQUER modelo. Não depende de hardware específico.

use crate::cross_modal::{ModalityType, ModalDraft};
use crate::semantic_attention;
use std::collections::HashMap;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────
// 1. TIPOS FUNDAMENTAIS
// ─────────────────────────────────────────────────────────────────────

/// Um slot no Jitter Buffer: contém o rascunho de uma modalidade
#[derive(Debug, Clone)]
pub struct ModalSlot {
    /// Qual modalidade este slot representa
    pub modality: ModalityType,
    /// Tokens do rascunho (prontos para VRAM)
    pub draft_tokens: Vec<u32>,
    /// Confiança do rascunho
    pub confidence: f32,
    /// Embedding projetado no espaço unificado
    pub projected_embedding: Vec<f32>,
    /// Timestamp de chegada (nanosegundos desde o início do pulse)
    pub arrival_ns: u64,
    /// Se true, este slot já foi preenchido neste pulse
    pub ready: bool,
}

/// Pacote Unificado: o que a GPU recebe — todos os modais sincronizados
#[derive(Debug, Clone)]
pub struct UnifiedConcept {
    /// ID do conceito (correlato ao CrossModalBus)
    pub concept_id: u64,
    /// Slots sincronizados (um por modalidade)
    pub slots: Vec<ModalSlot>,
    /// Jitter máximo entre o mais rápido e o mais lento (ns)
    pub sync_jitter_ns: u64,
    /// Timestamp do dispatch (quando o Maestro liberou para a VRAM)
    pub dispatch_ns: u64,
    /// Se todos os slots esperados chegaram
    pub is_complete: bool,
    /// Modalidade dominante calculada via Cross-Modal Attention
    pub dominant_modality: ModalityType,
}

/// Layout de um tensor alinhado para VRAM zero-copy
#[derive(Debug, Clone)]
pub struct AlignedTensorLayout {
    /// Offset no buffer de VRAM compartilhado (em bytes)
    pub vram_offset: usize,
    /// Tamanho total em bytes
    pub size_bytes: usize,
    /// Alinhamento requerido (128 bytes para Vulkan, 256 para CUDA)
    pub alignment: usize,
    /// Modalidade que ocupa esta região
    pub modality: ModalityType,
}

/// Configuração do Jitter Buffer
#[derive(Debug, Clone)]
pub struct SymphonyConfig {
    /// Intervalo do Systolic Pulse em nanosegundos
    /// default: 5_000_000 ns (5ms) — um "batimento" a cada 5ms
    pub pulse_interval_ns: u64,
    /// Timeout máximo: se alguma modalidade não chegar, despacha sem ela
    pub max_wait_ns: u64,
    /// Modalidades esperadas (quais esteiras estão ativas)
    pub expected_modalities: Vec<ModalityType>,
    /// Alinhamento de tensor para VRAM (128 = Vulkan, 256 = CUDA)
    pub tensor_alignment: usize,
    /// Tamanho máximo do buffer de conceitos em fila
    pub max_queued_concepts: usize,
}

impl Default for SymphonyConfig {
    fn default() -> Self {
        Self {
            pulse_interval_ns: 5_000_000,  // 5ms
            max_wait_ns: 10_000_000,       // 10ms timeout absoluto
            expected_modalities: vec![ModalityType::Text],
            tensor_alignment: 128,         // Vulkan default
            max_queued_concepts: 16,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. SYSTOLIC PULSE CLOCK
// ─────────────────────────────────────────────────────────────────────

/// O "batimento cardíaco" sistólico do motor.
/// A cada tick, empurra um bloco de dados para a validação.
/// Imita o systolic array das TPUs: fluxo constante, GPU nunca ociosa.
#[derive(Debug)]
pub struct SystolicPulse {
    /// Intervalo entre pulsos em nanosegundos
    pub interval_ns: u64,
    /// Contador de pulsos desde o boot
    pub pulse_count: u64,
    /// Timestamp do último pulso (ns)
    pub last_pulse_ns: u64,
    /// Timestamp atual simulado (ns)
    pub current_ns: u64,
}

impl SystolicPulse {
    pub fn new(interval_ns: u64) -> Self {
        Self {
            interval_ns,
            pulse_count: 0,
            last_pulse_ns: 0,
            current_ns: 0,
        }
    }

    /// Avança o clock em `delta_ns` nanosegundos.
    /// Retorna o número de pulsos que dispararam nesse intervalo.
    pub fn advance(&mut self, delta_ns: u64) -> u64 {
        self.current_ns += delta_ns;
        let mut fired = 0u64;
        while self.current_ns >= self.last_pulse_ns + self.interval_ns {
            self.last_pulse_ns += self.interval_ns;
            self.pulse_count += 1;
            fired += 1;
        }
        fired
    }

    /// Retorna os nanosegundos até o próximo pulso
    pub fn ns_until_next_pulse(&self) -> u64 {
        let next = self.last_pulse_ns + self.interval_ns;
        if next > self.current_ns {
            next - self.current_ns
        } else {
            0
        }
    }

    /// Timestamp do início do pulso atual
    pub fn current_pulse_start(&self) -> u64 {
        self.last_pulse_ns
    }
}

// ─────────────────────────────────────────────────────────────────────
// 3. JITTER BUFFER SEMÂNTICO (O MAESTRO)
// ─────────────────────────────────────────────────────────────────────

/// O Maestro: sincroniza rascunhos multimodais antes de enviar à VRAM.
///
/// Diferente de um jitter buffer de VoIP (que equaliza latência de rede),
/// este é SEMÂNTICO: ele garante que texto + imagem + áudio representem
/// o MESMO CONCEITO antes de liberar para a GPU.
pub struct JitterBuffer {
    /// Configuração da sinfonia
    pub config: SymphonyConfig,
    /// Clock sistólico
    pub pulse: SystolicPulse,
    /// Buffer atual: slots por modalidade aguardando despacho
    pub current_slots: HashMap<ModalityType, ModalSlot>,
    /// ID do conceito sendo montado
    pub current_concept_id: u64,
    /// Fila de conceitos prontos para VRAM
    pub dispatch_queue: Vec<UnifiedConcept>,
    /// Layout de tensores alinhados (simulação zero-copy)
    pub tensor_layouts: Vec<AlignedTensorLayout>,
    /// Estatísticas
    pub stats: SymphonyStats,
}

#[derive(Debug, Default)]
pub struct SymphonyStats {
    /// Total de conceitos unificados despachados
    pub concepts_dispatched: u64,
    /// Total de pulsos sistólicos disparados
    pub pulses_fired: u64,
    /// Jitter médio entre modalidades (ns)
    pub avg_jitter_ns: f64,
    /// Conceitos despachados incompletos (nem todas modalidades chegaram)
    pub incomplete_dispatches: u64,
    /// Total de bytes alinhados para VRAM
    pub total_aligned_bytes: u64,
    /// Pior caso de jitter (ns)
    pub worst_jitter_ns: u64,
}

impl JitterBuffer {
    pub fn new(config: SymphonyConfig) -> Self {
        let pulse = SystolicPulse::new(config.pulse_interval_ns);
        Self {
            config,
            pulse,
            current_slots: HashMap::new(),
            current_concept_id: 0,
            dispatch_queue: Vec::new(),
            tensor_layouts: Vec::new(),
            stats: SymphonyStats::default(),
        }
    }

    /// Insere um rascunho de uma modalidade no buffer.
    /// O Maestro segura até o próximo pulse ou até todas as modalidades chegarem.
    pub fn ingest_draft(
        &mut self,
        concept_id: u64,
        modal_draft: &ModalDraft,
        arrival_ns: u64,
    ) {
        // Se é um conceito novo, limpa o buffer
        if concept_id != self.current_concept_id {
            // Despacha o conceito anterior se tinha algo
            if !self.current_slots.is_empty() {
                self.dispatch_current(arrival_ns);
            }
            self.current_concept_id = concept_id;
            self.current_slots.clear();
        }

        let slot = ModalSlot {
            modality: modal_draft.modality,
            draft_tokens: modal_draft.draft_tokens.clone(),
            confidence: modal_draft.confidence,
            projected_embedding: modal_draft.projected_embedding.clone(),
            arrival_ns,
            ready: true,
        };

        self.current_slots.insert(modal_draft.modality, slot);

        // Se todas as modalidades esperadas chegaram → despacha imediatamente
        if self.all_expected_arrived() {
            self.dispatch_current(arrival_ns);
        }
    }

    /// Avança o clock sistólico. Se um pulse disparou, força o despacho.
    pub fn tick(&mut self, delta_ns: u64) -> Vec<UnifiedConcept> {
        let pulses = self.pulse.advance(delta_ns);
        self.stats.pulses_fired += pulses;

        // Se um pulse disparou e temos slots pendentes → despacha
        if pulses > 0 && !self.current_slots.is_empty() {
            self.dispatch_current(self.pulse.current_ns);
        }

        // Retorna os conceitos prontos
        std::mem::take(&mut self.dispatch_queue)
    }

    /// Despacha o conceito atual para a fila de VRAM
    fn dispatch_current(&mut self, dispatch_ns: u64) {
        let mut slots: Vec<ModalSlot> = self.current_slots.drain().map(|(_, v)| v).collect();
        if slots.is_empty() {
            return;
        }

        // Calcula jitter: diferença entre o mais rápido e o mais lento
        let min_arrival = slots.iter().map(|s| s.arrival_ns).min().unwrap_or(0);
        let max_arrival = slots.iter().map(|s| s.arrival_ns).max().unwrap_or(0);
        let jitter = max_arrival - min_arrival;

        // Atualiza estatísticas
        let is_complete = self.is_complete_set(&slots);
        if !is_complete {
            self.stats.incomplete_dispatches += 1;
        }
        if jitter > self.stats.worst_jitter_ns {
            self.stats.worst_jitter_ns = jitter;
        }
        let prev_total = self.stats.avg_jitter_ns * self.stats.concepts_dispatched as f64;
        self.stats.concepts_dispatched += 1;
        self.stats.avg_jitter_ns =
            (prev_total + jitter as f64) / self.stats.concepts_dispatched as f64;

        // --- Cross-Modal Attention ---
        // Calcula pesos de atenção via softmax sobre as confianças.
        // Simulamos a temperatura = 1.0
        let mut confidences: Vec<f32> = slots.iter().map(|s| s.confidence).collect();
        semantic_attention::inplace_softmax(&mut confidences);

        // Soft-clip cognitivo: Garante que nenhuma modalidade fique cega (< 8%)
        let min_baseline_attention = 0.08;
        let mut sum_alpha = 0.0;
        for alpha in confidences.iter_mut() {
            if *alpha < min_baseline_attention {
                *alpha = min_baseline_attention;
            }
            sum_alpha += *alpha;
        }
        for alpha in confidences.iter_mut() {
            *alpha /= sum_alpha;
        }

        // Identifica dominante
        let mut dominant_modality = slots[0].modality;
        let mut max_alpha = -1.0;
        for (i, slot) in slots.iter_mut().enumerate() {
            let alpha = confidences[i];
            // Opcional: A integridade do slot poderia guardar o peso para o Vulkan (gpu_buffer_weight)
            if alpha > max_alpha {
                max_alpha = alpha;
                dominant_modality = slot.modality;
            }
        }

        // Calcula layouts de tensor alinhados (simulação zero-copy)
        let layouts = self.compute_aligned_layouts(&slots);
        let total_bytes: usize = layouts.iter().map(|l| l.size_bytes).sum();
        self.stats.total_aligned_bytes += total_bytes as u64;

        let concept = UnifiedConcept {
            concept_id: self.current_concept_id,
            slots,
            sync_jitter_ns: jitter,
            dispatch_ns,
            is_complete,
            dominant_modality,
        };

        self.tensor_layouts = layouts;
        self.dispatch_queue.push(concept);
    }

    /// Verifica se todas as modalidades esperadas chegaram
    fn all_expected_arrived(&self) -> bool {
        self.config
            .expected_modalities
            .iter()
            .all(|m| self.current_slots.contains_key(m))
    }

    /// Verifica se o set de slots cobre todas as modalidades esperadas
    fn is_complete_set(&self, slots: &[ModalSlot]) -> bool {
        self.config.expected_modalities.iter().all(|expected| {
            slots.iter().any(|s| s.modality == *expected)
        })
    }

    /// Calcula layouts de tensor alinhados para zero-copy VRAM mapping.
    ///
    /// Cada modalidade recebe uma região contígua e alinhada no buffer VRAM.
    /// Alinhamento = poder de 2 (128 para Vulkan, 256 para CUDA).
    fn compute_aligned_layouts(&self, slots: &[ModalSlot]) -> Vec<AlignedTensorLayout> {
        let alignment = self.config.tensor_alignment;
        let mut layouts = Vec::new();
        let mut current_offset = 0usize;

        for slot in slots {
            // Tamanho raw: tokens × 4 bytes (u32)
            let raw_size = slot.draft_tokens.len() * std::mem::size_of::<u32>();
            // Alinha ao próximo múltiplo de alignment
            let aligned_size = (raw_size + alignment - 1) & !(alignment - 1);

            layouts.push(AlignedTensorLayout {
                vram_offset: current_offset,
                size_bytes: aligned_size,
                alignment,
                modality: slot.modality,
            });

            current_offset += aligned_size;
        }

        layouts
    }

    /// Relatório de performance da sinfonia
    pub fn stats_report(&self) -> String {
        format!(
            "Symphony Stats:\n\
             Concepts Dispatched: {}\n\
             Pulses Fired: {}\n\
             Avg Jitter: {:.0} ns\n\
             Worst Jitter: {} ns\n\
             Incomplete: {}\n\
             Total VRAM Aligned: {} bytes",
            self.stats.concepts_dispatched,
            self.stats.pulses_fired,
            self.stats.avg_jitter_ns,
            self.stats.worst_jitter_ns,
            self.stats.incomplete_dispatches,
            self.stats.total_aligned_bytes,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// 4. TESTES
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_modal::{ModalDraft, ModalityType};

    fn make_draft(modality: ModalityType, tokens: Vec<u32>, conf: f32) -> ModalDraft {
        ModalDraft {
            modality,
            draft_tokens: tokens,
            confidence: conf,
            projected_embedding: vec![0.5; 4],
        }
    }

    // ── Test 1: Jitter Buffer sincroniza texto + imagem ──
    #[test]
    fn test_jitter_buffer_sync_two_modalities() {
        let config = SymphonyConfig {
            expected_modalities: vec![ModalityType::Text, ModalityType::Image],
            pulse_interval_ns: 5_000_000,
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        // Texto chega em t=1ms
        let text_draft = make_draft(ModalityType::Text, vec![100, 101, 102], 0.95);
        buf.ingest_draft(1, &text_draft, 1_000_000);

        // Nada despachado ainda (falta imagem)
        assert!(buf.dispatch_queue.is_empty());

        // Imagem chega em t=3ms (2ms de jitter)
        let image_draft = make_draft(ModalityType::Image, vec![500, 501, 502, 503], 0.85);
        buf.ingest_draft(1, &image_draft, 3_000_000);

        // Agora SIM: ambas modalidades chegaram → despacho automático
        assert_eq!(buf.dispatch_queue.len(), 1);
        let concept = &buf.dispatch_queue[0];
        assert!(concept.is_complete);
        assert_eq!(concept.sync_jitter_ns, 2_000_000); // 2ms de jitter
        assert_eq!(concept.slots.len(), 2);
    }

    // ── Test 2: Systolic Pulse força despacho se timeout ──
    #[test]
    fn test_systolic_pulse_forces_dispatch() {
        let config = SymphonyConfig {
            expected_modalities: vec![ModalityType::Text, ModalityType::Image],
            pulse_interval_ns: 5_000_000, // 5ms
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        // Só texto chega
        let text_draft = make_draft(ModalityType::Text, vec![1, 2, 3], 0.9);
        buf.ingest_draft(1, &text_draft, 1_000_000);

        // Nada despachado (falta imagem)
        assert!(buf.dispatch_queue.is_empty());

        // Pulse dispara em t=5ms → força despacho mesmo incompleto
        let dispatched = buf.tick(5_000_000);
        assert_eq!(dispatched.len(), 1);
        assert!(!dispatched[0].is_complete); // Incompleto: só texto
        assert_eq!(buf.stats.incomplete_dispatches, 1);
    }

    // ── Test 3: Zero-Copy VRAM Alignment ──
    #[test]
    fn test_vram_tensor_alignment() {
        let config = SymphonyConfig {
            expected_modalities: vec![ModalityType::Text],
            tensor_alignment: 128,
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        // 30 tokens × 4 bytes = 120 bytes → alinhado para 128
        let draft = make_draft(ModalityType::Text, vec![0; 30], 0.9);
        buf.ingest_draft(1, &draft, 0);

        assert_eq!(buf.tensor_layouts.len(), 1);
        let layout = &buf.tensor_layouts[0];
        assert_eq!(layout.vram_offset, 0);
        assert_eq!(layout.size_bytes, 128); // 120 → alinhado para 128
        assert_eq!(layout.alignment, 128);
    }

    // ── Test 4: Multi-modal VRAM layout (texto + imagem + áudio) ──
    #[test]
    fn test_multimodal_vram_layout() {
        let config = SymphonyConfig {
            expected_modalities: vec![
                ModalityType::Text,
                ModalityType::Image,
                ModalityType::Audio,
            ],
            tensor_alignment: 256, // CUDA alignment
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        // Texto: 30 tokens (120 bytes → 256)
        buf.ingest_draft(
            1,
            &make_draft(ModalityType::Text, vec![0; 30], 0.95),
            100_000,
        );
        // Imagem: 4 patches (16 bytes → 256)
        buf.ingest_draft(
            1,
            &make_draft(ModalityType::Image, vec![0; 4], 0.85),
            200_000,
        );
        // Áudio: 16 mel tokens (64 bytes → 256)
        buf.ingest_draft(
            1,
            &make_draft(ModalityType::Audio, vec![0; 16], 0.80),
            300_000,
        );

        assert_eq!(buf.dispatch_queue.len(), 1);
        let concept = &buf.dispatch_queue[0];
        assert!(concept.is_complete);
        assert_eq!(concept.sync_jitter_ns, 200_000); // 300k - 100k

        // Layouts: 3 tensores, todos alinhados a 256
        assert_eq!(buf.tensor_layouts.len(), 3);
        for layout in &buf.tensor_layouts {
            assert_eq!(layout.size_bytes % 256, 0, "VRAM não alinhado!");
        }
        // Contíguos e não-overlapping
        let l0 = &buf.tensor_layouts[0];
        let l1 = &buf.tensor_layouts[1];
        let l2 = &buf.tensor_layouts[2];
        assert_eq!(l1.vram_offset, l0.vram_offset + l0.size_bytes);
        assert_eq!(l2.vram_offset, l1.vram_offset + l1.size_bytes);
    }

    // ── Test 5: Systolic Pulse counting ──
    #[test]
    fn test_systolic_pulse_counting() {
        let mut pulse = SystolicPulse::new(1_000_000); // 1ms per tick

        // Avança 5ms → devem ter 5 pulsos
        let fired = pulse.advance(5_000_000);
        assert_eq!(fired, 5);
        assert_eq!(pulse.pulse_count, 5);

        // Avança 500µs → nenhum pulse (metade de um intervalo)
        let fired = pulse.advance(500_000);
        assert_eq!(fired, 0);
        assert_eq!(pulse.pulse_count, 5);

        // Avança mais 500µs → 1 pulse (completou o 6º)
        let fired = pulse.advance(500_000);
        assert_eq!(fired, 1);
        assert_eq!(pulse.pulse_count, 6);
    }

    // ── Test 6: Stress test — muitos conceitos rápidos ──
    #[test]
    fn test_rapid_concept_throughput() {
        let config = SymphonyConfig {
            expected_modalities: vec![ModalityType::Text],
            pulse_interval_ns: 1_000_000, // 1ms
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        // Despacha 1000 conceitos em 100ms
        for i in 0..1000u64 {
            let draft = make_draft(ModalityType::Text, vec![i as u32; 10], 0.9);
            buf.ingest_draft(i, &draft, i * 100_000); // 100µs entre cada
        }

        // Todos devem estar despachados (single modality → despacho imediato)
        assert_eq!(buf.stats.concepts_dispatched, 1000);
        assert_eq!(buf.stats.incomplete_dispatches, 0);
    }

    // ── Test 7: Symphony bench — simula inferência completa multimodal ──
    #[test]
    fn bench_symphony_multimodal() {
        use std::time::Instant;

        let config = SymphonyConfig {
            expected_modalities: vec![ModalityType::Text, ModalityType::Image],
            pulse_interval_ns: 5_000_000,
            tensor_alignment: 128,
            ..Default::default()
        };
        let mut buf = JitterBuffer::new(config);

        let start = Instant::now();
        let iterations = 500;

        for i in 0..iterations {
            let concept_id = i as u64;
            // Texto: 30 tokens
            buf.ingest_draft(
                concept_id,
                &make_draft(ModalityType::Text, vec![0; 30], 0.95),
                (i as u64) * 10_000, // cada 10µs
            );
            // Imagem: 4 patches (chega 2µs depois)
            buf.ingest_draft(
                concept_id,
                &make_draft(ModalityType::Image, vec![0; 4], 0.85),
                (i as u64) * 10_000 + 2_000,
            );
        }

        let elapsed = start.elapsed();
        let per_concept = elapsed / iterations;

        println!("═══════════════════════════════════════════");
        println!("  BENCHMARK: Sinfonia Multimodal");
        println!("═══════════════════════════════════════════");
        println!("  Conceitos:       {}", iterations);
        println!("  Total:           {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Por conceito:    {:.0}ns", per_concept.as_nanos());
        println!("  Throughput:      {:.0} concepts/sec",
            iterations as f64 / elapsed.as_secs_f64());
        println!("  Avg Jitter:      {:.0}ns", buf.stats.avg_jitter_ns);
        println!("  Worst Jitter:    {}ns", buf.stats.worst_jitter_ns);
        println!("  VRAM Aligned:    {} bytes", buf.stats.total_aligned_bytes);
        println!("  Incompletos:     {}", buf.stats.incomplete_dispatches);
        println!("═══════════════════════════════════════════");

        // Critérios de aceite
        assert!(per_concept.as_micros() < 100,
            "Cada conceito DEVE levar < 100µs (levou {}µs)", per_concept.as_micros());
        assert_eq!(buf.stats.concepts_dispatched, iterations as u64);
        assert_eq!(buf.stats.incomplete_dispatches, 0);
        assert!(buf.stats.avg_jitter_ns < 10_000.0, "Jitter médio deve ser < 10µs");
    }
}
