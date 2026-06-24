# NodeStor — Visão Geral

NodeStor é um motor de inferência de IA local. Sua missão: **rodar qualquer modelo open-weight com qualidade total, em qualquer hardware, sem nuvem e sem dependências proprietárias.**

## Filosofia

A maioria das ferramentas de IA força uma escolha: qualidade *ou* acessibilidade. NodeStor recusa os dois extremos.

- **Lossless primeiro**: O caminho base é inferência em precisão total (F16/F32/BF16) com decodificação especulativa (COBER). Quantização (Q4/Q5/Q8) é uma conveniência opcional — nunca obrigatória.
- **Agnóstico ao hardware**: Shaders de compute em Vulkan rodam identicamente em NVIDIA, AMD, Intel e GPUs mobile. DirectStorage (Windows) e io_uring (Linux) eliminam a CPU como intermediário no transporte de tensores.
- **Soberano**: Sem telemetria, sem contas, sem servidores. O modelo roda na sua máquina, seus dados ficam locais.
- **Contexto infinito**: Evição do KV-cache com recall vetorial — o modelo nunca esquece. Tokens evictados são indexados semanticamente e recuperados via HNSW+BM25.

## Os Quatro Sistemas

### COBER — Motor de Especulação Lossless
Geração de tokens acelerada por decodificação especulativa. Um modelo rascunhador propõe sequências; o modelo principal verifica em paralelo (estilo EAGLE-2). Garantia matemática: a distribuição de saída é idêntica à do modelo completo rodando sozinho.

Também inclui: busca MCTS em árvore, Medusa multi-head, conjuntos de predição conformal.

### PROBES — Camada de Interpretabilidade
Inspeciona as representações internas do modelo em tempo real durante a geração:
- **ELK Probe**: Detecta inconsistência factual entre hidden states e tokens de saída (o "detector de mentiras").
- **CoT Monitor**: Sinaliza quando a cadeia de raciocínio diverge da resposta final.
- **RAISE Detector**: Mede níveis de consciência situacional SA1→SA5 (o modelo sabe que está sendo avaliado?).

PROBES roda como trait-object injetado no pipeline — zero dependências circulares.

### DAVI — Inteligência Sonhante
12 módulos cognitivos que dão ao NodeStor capacidade de descoberta autônoma:

| Módulo | Papel |
|--------|-------|
| `topology` | Encontra lacunas no espaço de conhecimento via Homologia Persistente (TDA) |
| `free_energy` | Prioriza exploração usando o Princípio de Energia Livre de Friston |
| `annealing` | Annealing Semântico — temperatura controla exploração criativa |
| `nash_tribunal` | 3 agentes adversariais validam cada hipótese |
| `functors` | Category Theory mapeia descobertas entre domínios |
| `stigmergy` | Ferômônios digitais guiam exploração tipo swarm |
| `autopoiesis` | O sistema melhora sua própria estratégia de busca |
| `dreaming_engine` | Orquestra os 7 subsistemas em um ciclo de sonho |
| `latent_jump` | Saltos entre regiões distantes do espaço latente |
| `intent_compiler` | Compila intenção do usuário em plano de execução |
| `elk_probe` | Detecção de mentira no nível de ativações |
| `raise_detector` | Monitoramento de consciência situacional |

O ciclo de sonho roda quando a GPU está ociosa. Detecta lacunas topológicas no grafo de conhecimento, gera hipóteses cross-domain e as valida por equilíbrio de Nash entre agentes adversariais.

### APEX — Camada de Transporte
A abstração de hardware que torna "contexto infinito em qualquer SSD" real:
- NTFS junction / io_uring para streaming de pesos do disco diretamente
- Shaders Vulkan para operações de tensor
- Paginação de KV-cache (estilo PagedAttention) para cenários multi-tenant
- Roteamento adaptativo: detecta geração do SSD, fabricante da GPU, VRAM disponível

## Visão de Longo Prazo

NodeStor é o "VLC da memória de IA": um executável estático único em Rust que detecta o hardware e abre a rota de dados correta (NVIDIA/cuFile, AMD/ROCm, Windows/DirectStorage, Linux/io_uring) de forma invisível. O objetivo é fazer com que o contexto infinito seja a norma — não o luxo — e devolver às pessoas e empresas o controle sobre sua própria inteligência artificial.

## Métricas Reais (SmolLM2-135M na CPU)

| Métrica | Valor |
|---------|-------|
| TTFT | ~930ms |
| Throughput | ~5 tok/s (CPU) |
| Contexto | Infinito (evição + recall vetorial) |
| Testes passando | 296 |

O caminho GPU via Vulkan é o alvo principal de performance — CPU é o caminho de simulação/compatibilidade.
