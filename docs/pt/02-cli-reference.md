# NodeStor CLI — Referência Completa

> Todos os comandos produzem métricas reais (TTFT, tok/s, latência) — nada é estimado ou hardcoded.

Execute `nodestor` sem argumentos para abrir o menu interativo colorido.

---

## Índice de Comandos

| Comando | O que faz |
|---------|-----------|
| [`run`](#run) | Inferência direta (sem servidor) com todas as funcionalidades |
| [`models`](#models) | Lista modelos instalados com tamanho e tipo de quantização |
| [`davi`](#davi) | Motor de sonho DAVI — descoberta cross-domain autônoma |
| [`pull`](#pull) | Download de modelos do HuggingFace |
| [`train`](#train) | Treinamento local de adaptadores LoRA |
| [`calibrate`](#calibrate) | Calibração de hardware e extração de vetores de steering |
| [`inspect`](#inspect) | Análise detalhada de arquivos de modelo |
| [`scan`](#scan) | Detecção de hardware e capacidades do sistema |
| [`start`](#start) | Inicia o daemon NodeStor em background |
| [`chat`](#chat) | Chat interativo via servidor |
| [`latency`](#latency) | Benchmark de TTFT e tok/s |
| [`compress`](#compress) | Compressão de modelos e arquivos |
| [`search`](#search) | Busca semântica no banco vetorial |

---

## `nodestor run` {#run}

**Inferência local direta** — carrega o modelo, roda o prompt, imprime tokens em stream e mede métricas reais.

```bash
nodestor run "<prompt>" --model <caminho_ou_nome> [opções]
```

### Opções Completas

| Flag | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `--model, -m` | `String` | **obrigatório** | Caminho para arquivo `.gguf` ou `.safetensors` |
| `--max-tokens` | `usize` | `128` | Número máximo de tokens a gerar |
| `--system` | `String` | — | Prompt de sistema (texto livre) |
| `--profile` | `String` | — | Perfil pronto (ver lista abaixo) |
| `--steer-vector` | `String` | — | Nome ou caminho do vetor de direção |
| `--intensity` | `f32` | `1.0` | Força do steering (0.0–2.0) |
| `--auto-steer` | `bool` | `false` | DSCP: auto-calibração de steering em memória |
| `--auto-calibrate` | `bool` | `false` | DSCP + micro-adaptação LoRA antes da geração |
| `--loras` | `Vec<String>` | `[]` | Adaptadores LoRA (repetível) |
| `--deep-research` | `bool` | `false` | Ativa loop de agente com ferramentas |
| `--max-loops` | `usize` | `10` | Ciclos máximos no Deep Research |
| `--tools-kit` | `String` | — | Kit de ferramentas JSON da comunidade |
| `--dream` | `bool` | `false` | Ativa handler DAVI no Deep Research |

### Perfis Prontos (`--profile`)

| Perfil | Prompt de sistema aplicado |
|--------|---------------------------|
| `cientista` | Raciocínio rigoroso, separação fato/hipótese, evidências |
| `programador` | Código limpo, idiomático, trade-offs explícitos |
| `advogado` | Precisão jurídica, citação de princípios, incertezas flagradas |
| `professor` | Explicação didática com exemplos, verificação de compreensão |
| `conciso` | Resposta direta e mínima, sem preâmbulo |
| `security` | Análise de vulnerabilidades para pesquisa autorizada |

### Resolução de Caminho de Modelo

Se `--model` for um caminho absoluto ou relativo com `/` ou `\`, é usado diretamente. Caso contrário, o sistema busca em `~/.nodestor/models/`.

### Modo Deep Research (`--deep-research`)

Ativa o `AgentExecutionLoop`. O modelo pode emitir tags de ferramenta no texto gerado:

```
<call_vector_db>sua query de busca</call_vector_db>
<call_dream>domínio1 + domínio2</call_dream>
<call_think>raciocínio interno</call_think>
<call_hypothesis>hipótese formal</call_hypothesis>
```

O pipeline intercepta essas tags, executa a ferramenta, injeta a resposta no contexto e retoma a geração. O ciclo continua até que o modelo produza uma resposta sem tool call, ou `--max-loops` seja atingido.

**Temperatura dinâmica**: O sistema detecta estagnação (repetição de tokens) e aumenta automaticamente a temperatura para forçar exploração criativa. Quando a estagnação se resolve, a temperatura resfria gradualmente.

### Exemplos Detalhados

```bash
# Inferência simples
nodestor run "O que é entropia em termodinâmica?" \
  --model ~/.nodestor/models/modelo.gguf \
  --max-tokens 512

# Com perfil de cientista e limite de tokens generoso
nodestor run "Explique o paradoxo EPR de Einstein-Podolsky-Rosen" \
  --model modelo.gguf \
  --profile cientista \
  --max-tokens 1024

# Com vetor de steering (remove direção de "recusa" das ativações)
nodestor run "Analise vulnerabilidades neste código" \
  --model modelo.gguf \
  --steer-vector security \
  --intensity 0.8 \
  --profile security

# Auto-calibração dinâmica (sem arquivo externo)
nodestor run "Explique a teoria dos jogos" \
  --model modelo.gguf \
  --auto-steer \
  --intensity 1.2

# Com adaptadores LoRA
nodestor run "Escreva em estilo jurídico" \
  --model modelo.gguf \
  --loras juridico --loras formal

# Deep Research completo com DAVI
nodestor run "Proponha um mecanismo molecular para cura de Alzheimer" \
  --model modelo.gguf \
  --deep-research \
  --dream \
  --max-loops 8 \
  --max-tokens 512

# Kit de ferramentas da comunidade
nodestor run "Derive os postulados da mecânica quântica desde primeiros princípios" \
  --model modelo.gguf \
  --deep-research \
  --tools-kit ~/kits/fisica_quantica.json \
  --max-loops 5
```

### Saída do Deep Research

```
╔══ DEEP RESEARCH MODE ═══════════════════════════════╗
║  Raciocínio em ciclos | Ferramentas ativas | DAVI ON  ║
╚══════════════════════════════════════════════════════╝

[Loop 1] T=0.70 — Gerando...
[TTFT: 930ms] Preciso verificar o estado atual da pesquisa sobre tau...
<call_vector_db>proteína tau Alzheimer mecanismo</call_vector_db>

[call_vector_db] Query: "proteína tau Alzheimer mecanismo"
Resultado: [1] Tau forms neurofibrillary tangles via phosphorylation...

[Loop 2] T=0.70 — Gerando...
Baseado nos dados, vou cruzar com descobertas em biofísica de proteínas...
<call_dream>Alzheimer tau + física de polímeros + teoria de percolação</call_dream>

[DAVI Dream] Cruzando domínios: Alzheimer tau + física de polímeros...
Hipótese: A propagação de tau segue dinâmicas de percolação em rede...

[Loop concluído — resposta final acima]

╔══ DEEP RESEARCH — MÉTRICAS ══════════════════════════╗
║  Loops executados : 3     Max configurado : 8          ║
║  Tokens totais    : 847   Velocidade      : 4.8 t/s    ║
║  Tempo total      : 176s                               ║
╚══════════════════════════════════════════════════════╝
```

---

## `nodestor models` {#models}

Lista modelos instalados com tamanho, formato e tipo de quantização detectado.

```bash
nodestor models          # apenas ~/.nodestor/models/
nodestor models --all    # + Downloads, Documents, cache HuggingFace
```

### Saída de Exemplo

```
╔══ MODELOS INSTALADOS ══════════════════════════════════╗
  📁 C:\Users\Adm\.nodestor\models
  ✔  SmolLM2-135M-Q4_K_M.gguf                     0.09 GB  [GGUF/Q4_K_M]
  ✔  Llama-3.1-8B-Instruct-Q5_K_M.gguf            5.73 GB  [GGUF/Q5_K_M]
  ✔  Mistral-7B-v0.3.F16.gguf                     13.98 GB  [GGUF/F16]
╚════════════════════════════════════════════════════════╝
```

**Tipos de quantização detectados automaticamente**: Q4_K_M, Q4_K_S, Q5_K_M, Q8_0, Q4_0, F16, F32, BF16, SafeTensors.

---

## `nodestor davi` {#davi}

Interface direta com o sistema DAVI. Não requer modelo carregado.

### `nodestor davi status`

Mostra o status de todos os 12 módulos DAVI e como ativá-los.

```bash
nodestor davi status
```

### `nodestor davi dream`

Executa ciclos de sonho autônomos. O sistema gera hipóteses cross-domain sem precisar de um modelo de linguagem — usa apenas os módulos matemáticos do DAVI.

```bash
nodestor davi dream --topic "<domínio1> + <domínio2> + ..."
```

| Flag | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `--topic` | `String` | `física + matemática + biologia` | Domínios a cruzar (separados por `+`) |
| `--cycles` | `usize` | `3` | Número de ciclos de sonho |
| `--temperature` | `f32` | `5.0` | Temperatura inicial do annealing (maior = mais exploratório) |

#### O que acontece internamente

1. O tópico é dividido em domínios por `+`
2. Cada domínio gera embeddings via FNV-1a hashing (64 dimensões)
3. Os embeddings são passados para `DreamingEngine::dream_cycle()`
4. O ciclo completo TDA→FEP→Annealing→Nash→Functors→Stigmergy→Autopoiese roda
5. Descobertas validadas pelo Nash Tribunal são exibidas com confiança

#### Exemplos

```bash
# Crossover física + biologia (clássico)
nodestor davi dream --topic "física quântica + biologia molecular"

# Mais exploratório (temperatura alta)
nodestor davi dream --topic "topologia + economia + neurociência" \
  --cycles 7 --temperature 9.0

# Mais conservador (refinamento)
nodestor davi dream --topic "termodinâmica + teoria da informação" \
  --cycles 2 --temperature 2.0
```

#### Saída de Exemplo

```
╔══ DAVI DREAM ENGINE ══════════════════════════════════╗
  Tópico  : física quântica + genética
  Ciclos  : 3 | Temperatura inicial: 5.0
╠═══════════════════════════════════════════════════════╣

  [Ciclo 1/3] Annealing semântico... 2 descoberta(s)
  [Ciclo 2/3] Annealing semântico... 1 descoberta(s)
  [Ciclo 3/3] Annealing semântico... 3 descoberta(s)

  ══ DESCOBERTAS VALIDADAS PELO NASH TRIBUNAL ══

  [1] DOMÍNIO: física quântica + genética
      Hipótese  : Entrelaçamento quântico exibe análogo estrutural em mecanismos de splicing de RNA.
      Confiança : 71.3%

  [2] DOMÍNIO: genética
      Hipótese  : Propriedades de superposição têm correspondência em estados epigenéticos pluripotentes.
      Confiança : 63.8%

  Ciclos: 3 | Embeddings: 8 | Total DAVI discoveries: 6
╚═══════════════════════════════════════════════════════╝
```

---

## `nodestor pull` {#pull}

Download de modelos do HuggingFace Hub com barra de progresso.

```bash
nodestor pull <model-id> --filename <arquivo>
```

```bash
# SmolLM2 quantizado (recomendado para testar)
nodestor pull bartowski/SmolLM2-135M-GGUF \
  --filename SmolLM2-135M-Q4_K_M.gguf

# Llama 3.1 8B
nodestor pull bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --filename Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

# Mistral em precisão total
nodestor pull TheBloke/Mistral-7B-v0.1-GGUF \
  --filename mistral-7b-v0.1.Q8_0.gguf
```

Modelos são salvos em `~/.nodestor/models/<filename>`.

---

## `nodestor train` {#train}

Treina adaptadores LoRA localmente. Os pesos base ficam **100% congelados** — apenas matrizes low-rank são aprendidas.

```bash
nodestor train \
  --model <modelo.gguf> \
  --dataset <dados.jsonl> \
  --output <adaptador.lora>
```

### Opções

| Flag | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `--model, -m` | `String` | obrigatório | Modelo base GGUF/SafeTensors |
| `--dataset, -d` | `String` | obrigatório | Dataset em formato JSONL |
| `--output, -o` | `String` | obrigatório | Nome do adaptador de saída |
| `--rank` | `usize` | `8` | Rank do LoRA (4–16 recomendado) |
| `--alpha` | `f32` | `8` | Alpha (escala do LoRA) |
| `--lr` | `f32` | `0.0001` | Learning rate do otimizador AdamW |
| `--max-steps` | `usize` | `0` | Passos máximos (0 = dataset completo) |
| `--grad-accum` | `usize` | `4` | Acumulação de gradientes (para controle de VRAM) |

### Formato do Dataset JSONL

Qualquer dos três formatos é aceito:

```jsonl
{"input": "Pergunta aqui", "output": "Resposta aqui"}
{"text": "Texto de treinamento livre"}
{"prompt": "Instrução", "completion": "Resposta esperada"}
```

### Exemplo de Uso

```bash
# Especializar em linguagem jurídica brasileira
nodestor train \
  --model modelo.gguf \
  --dataset contratos_juridicos.jsonl \
  --output juridico_br.lora \
  --rank 16 \
  --lr 0.00005 \
  --max-steps 500

# Depois de treinar, usar o adaptador
nodestor run "Redija uma cláusula de confidencialidade" \
  --model modelo.gguf \
  --loras juridico_br
```

Adaptadores são salvos em `~/.nodestor/loras/<nome>.lora`.

---

## `nodestor calibrate` {#calibrate}

### Modo 1: Calibração de Hardware

Sem argumentos, mede capacidades do sistema e salva configuração ótima.

```bash
nodestor calibrate
```

Detecta: velocidade de I/O do SSD, VRAM disponível, suporte Vulkan, largura de banda de memória.

### Modo 2: Extração de Vetor de Steering

Com `--positive` e `--negative`, executa o **DSCP (Dynamic Self-Calibration Pipeline)**:

1. Carrega o modelo
2. Processa exemplos positivos (comportamento desejado) → extrai hidden states
3. Processa exemplos negativos (comportamento indesejado) → extrai hidden states
4. Calcula a diferença média normalizada → vetor de direção `d̂`
5. Salva em `~/.nodestor/vectors/<output>.bin`

```bash
nodestor calibrate \
  --positive exemplos_tecnicos.txt \
  --negative exemplos_genericos.txt \
  --output direcao_tecnica \
  --model modelo.gguf
```

Depois: `nodestor run "..." --model modelo.gguf --steer-vector direcao_tecnica`

### Steering no Modo Auto (`--auto-steer`)

Sem arquivos externos, usa 6 templates internos positivos e 6 negativos para gerar o vetor em memória. Útil quando não há dataset disponível.

```bash
nodestor run "..." --model modelo.gguf --auto-steer --intensity 1.0
```

---

## `nodestor inspect` {#inspect}

Analisa metadados de um arquivo de modelo sem carregá-lo completamente.

```bash
nodestor inspect /caminho/modelo.gguf
nodestor inspect /caminho/modelo.gguf --tensors   # lista todos os tensores
```

**Informações mostradas**: versão GGUF, número de parâmetros, arquitetura, vocabulário, contexto máximo, tipo de quantização, lista de tensores com nome/shape/tipo.

---

## `nodestor scan` {#scan}

Detecta hardware e mostra informações do sistema.

```bash
nodestor scan
```

**Detecta**: GPU (nome, fabricante, VRAM), CPU (núcleos, frequência), RAM disponível, SSD NVMe vs SATA, suporte Vulkan, DirectStorage (Windows), geração PCIe.

---

## `nodestor start` {#start}

Inicia o daemon NodeStor (processo em background) e acopla a interface de chat.

```bash
nodestor start                            # auto-detecta modelo em ~/.nodestor/models/
nodestor start --model /caminho/modelo.gguf
nodestor start --model modelo.gguf --quiet  # silencioso (para scripts)
```

O daemon expõe: `GET /health`, `GET /status`, `POST /chat`, `GET /search`.

---

## `nodestor chat` {#chat}

Abre interface de chat interativo conectando ao servidor.

```bash
nodestor chat                                   # padrão http://localhost:8080
nodestor chat --server http://192.168.1.10:8080 # servidor remoto
```

---

## `nodestor latency` {#latency}

Benchmark preciso de TTFT (Time to First Token) e velocidade de geração.

```bash
nodestor latency                          # usa modelo auto-detectado
nodestor latency --model modelo.gguf
```

Usa prompt padrão de 32 tokens ("The quick brown fox") para medição reprodutível.

---

## `nodestor compress` {#compress}

Comprime modelos e arquivos com GDeflate (aceleração Vulkan) ou Zstd.

```bash
nodestor compress modelo.gguf modelo.gdf --format gdeflate
nodestor compress modelo.gguf modelo.zst --format zstd
```

GDeflate é descomprimido na GPU em microssegundos — permite que um SSD de 7 GB/s pareça entregar 14–20 GB/s de dados de modelo.

---

## `nodestor search` {#search}

Busca semântica no banco vetorial (requer daemon ativo com documentos indexados).

```bash
nodestor search "mecanismo de atenção multi-head" --k 5
```

O endpoint `/search` usa HNSW + BM25 com fusão RRF (Reciprocal Rank Fusion). Coloque arquivos `.txt` ou `.md` na pasta `./knowledge/` e o daemon os indexa automaticamente.

---

## Kit de Ferramentas da Comunidade

O formato `.json` permite criar e compartilhar kits de ferramentas especializados.

### Estrutura do Arquivo

```json
{
  "name": "nome_do_kit",
  "version": "1.0",
  "description": "Descrição do propósito do kit",
  "author": "Seu Nome <seu@email.com>",
  "tools": [
    {
      "tag": "call_vector_db",
      "description": "O que esta ferramenta faz (injetado no system prompt)",
      "system_hint": "Quando o modelo deve usar esta ferramenta"
    },
    {
      "tag": "call_dream",
      "description": "Gera hipóteses cross-domain via DAVI",
      "system_hint": "Use quando precisar combinar conceitos de domínios diferentes"
    }
  ],
  "system_prompt": "Prompt de sistema especializado que define o comportamento do agente neste kit"
}
```

### Tags de Ferramenta Disponíveis

| Tag | O que faz | Quando usar |
|-----|-----------|-------------|
| `call_vector_db` | Busca no banco vetorial HNSW+BM25 | Recuperar fatos, papers, memórias |
| `call_dream` | Invoca DAVI para hipóteses cross-domain | Combinar conceitos de áreas diferentes |
| `call_think` | Reflexão interna estruturada | Revisar raciocínio antes de concluir |
| `call_hypothesis` | Valida hipótese no Nash Tribunal | Testar uma afirmação formalmente |

### Kits Embutidos

NodeStor inclui dois kits sem necessidade de arquivo externo:

- **`science`** (padrão): 4 ferramentas, prompt de cientista autônomo
- **`coding`**: 2 ferramentas, prompt de engenheiro de software

Para usar o kit embutido, basta `--deep-research` sem `--tools-kit`.

---

## Menu Interativo

Executar `nodestor` sem argumentos abre o menu:

```
NODE-PANEL
❯ RUN      - Rodar prompt direto (inferência local, sem servidor)
  START    - Iniciar Motor + Interface (FULL ENGINE)
  ATTACH   - Conectar a Motor Residente (RESIDENT)
  CHATTING - Iniciar Conversa Local (Modo Streaming)
  DREAM    - DAVI: Motor de Sonho Cross-Domain
  MODELS   - Listar modelos instalados
  PULL     - Baixar modelo do HuggingFace
  LATENCY  - Teste de Resposta 7-Camadas (TTFT)
  SCANNER  - Inspeção de Hardware Industrial
  DETACH   - Sair e manter motor em Background
  EXIT     - Encerrar tudo
```

Use `↑↓` para navegar, `Enter` para selecionar, `Esc` para voltar.
