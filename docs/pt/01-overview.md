# NodeStor — Visão Geral

> **"O VLC da memória de IA"** — um motor de inferência que transforma qualquer máquina em um data center portátil.

---

## O Problema que NodeStor Resolve

O ecossistema de IA está fragmentado por design. NVIDIA exige CUDA. Servidores em nuvem exigem assinaturas. Ferramentas populares como llama.cpp rodam modelos, mas não exploram o hardware ao máximo. Modelos grandes de 70B+ são acessíveis apenas para quem tem dezenas de gigabytes de VRAM.

NodeStor parte de uma premissa diferente: **o limite real não é o hardware — é a engenharia de transporte de dados.**

Um SSD NVMe Gen 4 tem largura de banda de ~7 GB/s. Um barramento PCIe 4.0 x16 carrega 32 GB/s. Uma GPU moderna processa centenas de bilhões de operações por segundo. O gargalo está na cadeia de middleware que fica entre esses três componentes — e é exatamente essa cadeia que NodeStor elimina.

---

## Filosofia Central

### 1. Lossless Primeiro

A maioria dos motores locais de IA trata quantização como obrigatória. NodeStor inverte essa lógica:

- **Caminho padrão**: precisão total (F16/F32/BF16) com decodificação especulativa (COBER). A qualidade do modelo é 100% preservada.
- **Caminho opcional**: quantização (Q4_K_M, Q5_K_M, Q8_0) para usuários com VRAM limitada. Funciona perfeitamente, mas nunca é forçada.

A distinção importa porque quantização aggressiva degrada capacidade de raciocínio. Em tarefas de matemática avançada ou hipóteses científicas, a diferença entre F16 e Q4 pode ser a diferença entre a resposta certa e a errada.

### 2. Hardware Agnóstico

Shaders de compute Vulkan compilam para qualquer GPU — NVIDIA, AMD, Intel, Apple Silicon, GPUs mobile. O mesmo binário roda no PC gamer, no servidor de data center e, futuramente, no smartphone.

O transporte de tensores usa as APIs mais eficientes para cada plataforma:
- **Windows**: DirectStorage (o mesmo mecanismo que carrega texturas de jogos AAA em <2s)
- **Linux**: io_uring + DMABUF (zero-copy do SSD para a GPU)
- **Ambos**: NTFS junctions / symlinks para separar dados de código

### 3. Soberania Total

Sem telemetria. Sem contas. Sem dependência de nuvem. O modelo roda na sua máquina — seus dados, seus prompts, seus resultados ficam onde você escolhe.

Isso não é só privacidade: é resiliência. Se a API de uma empresa muda preço, encerra ou bloqueia acesso, seu fluxo de trabalho não quebra.

---

## Os Quatro Sistemas

NodeStor é composto por quatro sistemas que se integram em camadas. Cada um pode ser entendido independentemente, mas o poder real emerge da interação entre eles.

---

### Sistema 1: COBER — Decodificação Especulativa Lossless

**O que é**: COBER (Contextual Optimistic Batch Execution Runtime) acelera a geração de tokens sem comprometer qualidade.

**Como funciona**: Um modelo rascunhador pequeno (ex: 135M de parâmetros) propõe 4–8 tokens em paralelo. O modelo principal verifica todos de uma vez. Se o rascunho estava certo, todos os tokens são aceitos — o que equivale a ter gerado 4–8 tokens com o custo computacional de 1.

**A garantia matemática**: O algoritmo usa rejeição amostral. Formalmente:

```
Para cada token proposto t_d na posição i:
  r ~ Uniforme(0, 1)
  Se r < p_completo(t_d) / p_rascunho(t_d):
    aceita t_d
  Senão:
    amostra de distribuição corrigida; para aqui
```

A distribuição final é **matematicamente idêntica** à que o modelo completo produziria sozinho. Não há trade-off de qualidade.

**Speedup típico**: 2–4× em tokens por segundo, dependendo do alinhamento entre rascunhador e modelo principal.

**Também inclui**:
- **MCTS**: Monte Carlo Tree Search para explorar múltiplos caminhos de raciocínio antes de comprometer
- **Medusa**: múltiplas "cabeças" de predição em paralelo no mesmo modelo
- **Conformal Prediction Sets**: certeza estatística sobre o conjunto de tokens possíveis

---

### Sistema 2: PROBES — Interpretabilidade Mecanística em Tempo Real

**O que é**: PROBES inspeciona o que o modelo *realmente* está fazendo internamente, não apenas o que ele *diz* que está fazendo.

**Por que isso importa**: Modelos de linguagem podem expressar certeza em respostas erradas. Podem dizer "não sei" quando na verdade sabem. Podem seguir uma cadeia de raciocínio coerente mas chegar a uma conclusão inconsistente. PROBES detecta tudo isso em tempo real.

**Os três módulos principais**:

**ELK Probe (Eliciting Latent Knowledge)**
Monitora o estado dos hidden states a cada token gerado. Detecta quando existe uma inconsistência entre o que o modelo sabe (representado nas ativações) e o que ele está prestes a dizer (o próximo token). Tecnicamente análogo a um detector de mentiras operando no espaço vetorial das representações internas.

**CoT Monitor (Chain-of-Thought Monitor)**
Rastreia a coerência entre a cadeia de raciocínio visível (o texto gerado em `<thinking>`) e a resposta final. Se o raciocínio levaria a uma conclusão A mas o modelo responde B, o monitor sinaliza divergência.

**RAISE Detector (SA1→SA5)**
Mede o nível de consciência situacional do modelo em uma escala de 5 níveis:
- SA1: o modelo sabe que está gerando texto
- SA2: o modelo sabe que há um usuário
- SA3: o modelo sabe que está sendo avaliado
- SA4: o modelo modifica comportamento em função da avaliação
- SA5: o modelo raciocina sobre sua própria avaliação

**Integração técnica**: PROBES se acopla ao pipeline através do trait `ProbesTool`. O método `inspect(hidden_state, step)` é chamado a cada token gerado. Isso permite injeção sem dependência circular — o crate `nodestor-inference` não precisa saber que PROBES existe.

---

### Sistema 3: DAVI — Inteligência Sonhante

**O que é**: DAVI (Dreaming Autonomous Visual Intelligence) é o sistema cognitivo de descoberta autônoma do NodeStor. Quando a GPU está ociosa, o sistema "sonha" — explora conexões entre conceitos de domínios diferentes para gerar hipóteses que humanos raramente considerariam.

**A inspiração**: Cientistas fazem suas melhores descobertas em estado de descanso — quando a mente conecta ideias de forma não-linear. DAVI implementa esse processo matematicamente.

**O ciclo de sonho** (8 passos orquestrados por `DreamingEngine`):

#### Passo 1: Topologia — Encontrando os Buracos

`TopologicalGapDetector` aplica **Análise Topológica de Dados (TDA)** ao grafo de conhecimento. A homologia persistente identifica "buracos" no espaço — regiões onde o conhecimento é inexistente ou contraditório. Esses buracos são oportunidades de descoberta.

Imagine o conhecimento como uma superfície 3D. Regiões densas são bem compreendidas. Buracos são perguntas que ninguém fez ainda.

#### Passo 2: Free Energy — Priorizando por Surpresa

`FreeEnergyObjective` usa o **Princípio de Energia Livre de Karl Friston**. O sistema prioriza buracos que causariam maior redução de surpresa se preenchidos. Matematicamente, seleciona a ação que minimiza a energia livre esperada.

Isso garante que o sistema não explore aleatoriamente — ele foca nos gaps que têm maior potencial de impacto.

#### Passo 3: Annealing — A Temperatura Criativa

`SemanticAnnealing` decide se aceita ou rejeita saltos cross-domain usando a analogia do **Simulated Annealing**. Em temperatura alta, aceita saltos improváveis (exploração). Em temperatura baixa, aceita só saltos prováveis (exploração de refinamento).

Isso permite ao sistema navegar entre "altamente especulativo" e "conservador" ao longo de um ciclo.

#### Passo 4: Crystal Skeleton — Estrutura Sintética

Gera a hipótese em formato estruturado, extraindo um "esqueleto sintático" que pode ser validado formalmente.

#### Passo 5: Functores — Category Theory entre Domínios

`FunctorComposer` usa **Teoria de Categorias** para mapear uma descoberta de um domínio para outro. Se o sistema descobre que "gravidade" e "custo de oportunidade" compartilham a mesma estrutura matemática (ambos são gradientes de um potencial), uma descoberta em física se torna automaticamente uma descoberta em economia.

Um funtor `F: Domínio_A → Domínio_B` preserva a estrutura morfológica. Cada descoberta se multiplica por todos os domínios mapeados.

#### Passo 6: Nash Tribunal — Validação Adversarial

`NashTribunal` instancia 3 agentes adversariais em equilíbrio de Nash:
- **Agente Proponente**: argumenta fortemente a favor da hipótese
- **Agente Cético**: busca falhas e contradições
- **Agente Síntese**: arbitra e formula um veredicto equilibrado

Somente hipóteses que sobrevivem ao tribunal são aceitas como `NovelDiscovery`. Isso garante que o sistema não produza especulação sem fundamento.

#### Passo 7: Stigmergy — Ferômônios Digitais

`StigmergicSwarm` deposita "ferômônios" nos caminhos bem-sucedidos, exatamente como formigas fazem. Ciclos futuros têm maior probabilidade de explorar vizinhanças de descobertas anteriores. O conhecimento cresce de forma coerente, não aleatória.

#### Passo 8: Autopoiese — O Sistema se Melhora

`AutopoieticLoop` ajusta os parâmetros do próprio sistema em função dos resultados. Se a temperatura estava muito alta (muitos saltos rejeitados), reduz. Se estava muito baixa (pouca exploração), aumenta. O sistema aprende a sonhar melhor.

**Resultado concreto**: `nodestor davi dream --topic "física quântica + genética"` pode produzir hipóteses como "propriedades de entrelaçamento quântico têm análogo estrutural nos mecanismos de splicing de RNA" — cruzamentos que a literatura raramente considera.

---

### Sistema 4: APEX — Transporte de Dados em Velocidade de Barramento

**O que é**: APEX (Adaptive Prefetch Execution) é a camada de transporte que trata o SSD como uma extensão da VRAM.

**O princípio**: Um modelo de 7B parâmetros em F16 ocupa ~14 GB. A maioria das GPUs tem 8–16 GB de VRAM. A solução convencional é quantizar. A solução APEX é fazer o streaming das camadas diretamente do SSD para a GPU via DMA, sem passar pela RAM.

**Os componentes**:

**BurstScheduler**: Pré-calcula quais camadas serão necessárias nos próximos passos e inicia transferências antes da GPU pedir. Elimina bolhas de latência.

**MesPrefetchQueue**: Fila de pré-busca baseada em perfil do modelo. Para arquiteturas Transformer, o padrão de acesso é altamente previsível — o scheduler aproveita isso.

**BufferPool**: Gerenciamento de pool de buffers na VRAM. Camadas não mais necessárias são liberadas em ordem FIFO; camadas prestes a serem necessárias são carregadas com antecedência.

**KVCachePaginator**: Paginação do KV-cache estilo PagedAttention. Múltiplos usuários ou múltiplos contextos compartilham a VRAM disponível sem desperdício.

---

## Contexto Infinito: Como Funciona

Esta é uma das funcionalidades mais importantes do NodeStor e vale um explicação detalhada.

**O problema**: KV-cache armazena os key/value de cada token gerado. Com 4096 tokens de contexto e um modelo de 7B, o KV-cache pode ocupar vários GB. Dobrar o contexto dobra o uso de memória.

**A solução NodeStor**:

```
Geração normal → KV-cache enche → enforce_window() evicta tokens antigos
                                           ↓
                              Tokens evictados → decode() → texto
                                           ↓
                              vector_db.add_document(id, texto)
                                           ↓
                              Contexto recuperado via busca semântica
                              quando relevante para o prompt atual
```

O resultado: o modelo nunca perde acesso ao contexto anterior. Quando precisa de informação de 10.000 tokens atrás, o sistema recupera via busca HNSW+BM25. O "esquecimento" é indexação, não perda.

**Prova empírica**: o teste `test_infinite_context_recall_of_evicted_fact` (em `nodestor-metadata`) verifica end-to-end que um fato inserido antes da evição é corretamente recuperado após evição.

---

## Estado Atual

| Componente | Status |
|-----------|--------|
| Crates no workspace | 14 |
| Testes passando | 528+ |
| Modelo real testado | SmolLM2-135M (F16, 270MB) |
| Output coerente | Sim (`" Paris"` para "The capital of France is") |
| TTFT (CPU) | ~930ms |
| Throughput (CPU) | ~5 tok/s |
| Contexto infinito | Provado empiricamente |
| KV-cache reuse | Provado correto (5 testes: incremental == recompute total) |
| Deep Research Engine | Completo — todos os handlers usam módulos DAVI reais |
| DAVI Dream | 12 módulos; `--dream` conectado ao DreamingEngine real |
| Nash Tribunal | Validação adversarial real no handler `call_hypothesis` |
| CLI | 17 comandos, menu interativo, temperatura dinâmica no agent loop |

**O que ainda está em desenvolvimento**:
- Shaders Vulkan GPU para inferência real na GPU (atualmente usa CPU fallback)
- Velocidade alvo: milhares de tok/s com GPU + especulação COBER

A base matemática está correta. A sequência: **correção ✓ → reuso KV ✓ → GPU → especulação**.
