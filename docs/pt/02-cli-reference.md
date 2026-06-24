# NodeStor — Referência da CLI

Todos os comandos: `nodestor <comando> [opções]`  
Rode `nodestor` sozinho para abrir o menu interativo.

---

## `nodestor run`

Roda um prompt direto contra um modelo local. Mede TTFT e tok/s em tempo real.

```bash
nodestor run "Seu prompt aqui" --model /caminho/modelo.gguf
```

**Opções:**

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--model, -m` | obrigatório | Caminho do modelo GGUF ou SafeTensors |
| `--max-tokens` | 128 | Máximo de tokens a gerar |
| `--system` | — | Prompt de sistema (a "constituição" do modelo) |
| `--profile` | — | Perfil pronto: `cientista`, `programador`, `advogado`, `professor`, `conciso`, `security` |
| `--steer-vector` | — | Vetor de direção de ativação (gerado por `calibrate`) |
| `--intensity` | 1.0 | Intensidade do steering (0.0 = desligado, 1.0 = total) |
| `--auto-steer` | false | Pipeline de auto-calibração dinâmica (DSCP) |
| `--auto-calibrate` | false | DSCP + micro-adaptação LoRA de um passo |
| `--loras` | — | Adaptadores LoRA a aplicar (repetível: `--loras a --loras b`) |
| `--deep-research` | false | Ativa loop de agente com chamada de ferramentas |
| `--max-loops` | 10 | Máximo de ciclos de raciocínio no modo Deep Research |
| `--tools-kit` | — | Caminho para arquivo JSON de kit de ferramentas da comunidade |
| `--dream` | false | Ativa o motor de sonho DAVI (hipóteses cross-domain) |

**Exemplos:**

```bash
# Uso simples
nodestor run "Explique o embedding posicional RoPE" --model modelo.gguf

# Com steering (remove a direção de "recusa" das ativações)
nodestor run "Analise este exploit" --model modelo.gguf --steer-vector security --profile security

# Deep Research com DAVI
nodestor run "Proponha um tratamento inovador para Alzheimer" \
  --model modelo.gguf --deep-research --dream --max-loops 5

# Com kit de ferramentas da comunidade
nodestor run "Derive a equação de Schrödinger desde os primeiros princípios" \
  --model modelo.gguf --deep-research --tools-kit kit_fisica.json
```

---

## `nodestor models`

Lista modelos instalados.

```bash
nodestor models         # mostra modelos em ~/.nodestor/models/
nodestor models --all   # também busca em Downloads, Documents, cache HuggingFace
```

---

## `nodestor davi`

Interface direta com o sistema de sonho DAVI.

### `nodestor davi dream`

Executa ciclos de sonho autônomos (sem necessidade de modelo).

```bash
nodestor davi dream --topic "física quântica + genética"
nodestor davi dream --topic "economia + topologia" --cycles 5 --temperature 8.0
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--topic` | `física + matemática + biologia` | Domínios a cruzar |
| `--cycles` | 3 | Número de ciclos de sonho |
| `--temperature` | 5.0 | Temperatura do annealing semântico (maior = mais exploratório) |

### `nodestor davi status`

Mostra o status de todos os 12 módulos DAVI.

```bash
nodestor davi status
```

---

## `nodestor pull`

Baixa um modelo do HuggingFace Hub.

```bash
nodestor pull bartowski/SmolLM2-135M-GGUF --filename SmolLM2-135M-Q4_K_M.gguf
```

Modelos são salvos em `~/.nodestor/models/`.

---

## `nodestor train`

Treina um adaptador LoRA em um dataset JSONL local.

```bash
nodestor train --model modelo.gguf --dataset dados.jsonl --output meu_adaptador.lora
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--model, -m` | obrigatório | Caminho do modelo base |
| `--dataset, -d` | obrigatório | Dataset JSONL (`{"input":...,"output":...}` ou `{"text":...}`) |
| `--output, -o` | obrigatório | Nome/caminho do adaptador de saída |
| `--rank` | 8 | Rank do LoRA (4, 8 ou 16 recomendados) |
| `--alpha` | 8 | Alpha do LoRA |
| `--lr` | 0.0001 | Learning rate do AdamW |
| `--max-steps` | 0 | Máximo de passos (0 = dataset completo) |
| `--grad-accum` | 4 | Acumulação de gradientes (controle de VRAM) |

---

## `nodestor calibrate`

### Calibração de hardware (sem argumentos)
Mede I/O e capacidades da GPU, salva configuração ótima.

```bash
nodestor calibrate
```

### Extração de vetor de steering
Extrai um vetor de direção a partir de hidden states contrastivos.

```bash
nodestor calibrate \
  --positive exemplos_positivos.txt \
  --negative exemplos_negativos.txt \
  --output minha_direcao \
  --model modelo.gguf
```

Vetores são salvos em `~/.nodestor/vectors/<nome>.bin`.

---

## `nodestor inspect`

Analisa um arquivo de modelo (GGUF ou SafeTensors).

```bash
nodestor inspect /caminho/modelo.gguf
nodestor inspect /caminho/modelo.gguf --tensors   # mostra todos os tensores
```

---

## `nodestor scan`

Detecta hardware: GPU, VRAM, CPU, velocidade do SSD, suporte Vulkan.

```bash
nodestor scan
```

---

## `nodestor start`

Inicia o daemon NodeStor (servidor em background) e acopla a interface de chat.

```bash
nodestor start --model modelo.gguf
```

---

## `nodestor chat`

Conecta a um servidor NodeStor em execução.

```bash
nodestor chat --server http://localhost:8080
```

---

## `nodestor latency`

Mede TTFT (Time to First Token) e tok/s com um benchmark de 32 tokens.

```bash
nodestor latency --model modelo.gguf
```

---

## `nodestor compress`

Comprime um modelo ou arquivo.

```bash
nodestor compress modelo.gguf modelo.gz --format gdeflate
```

---

## `nodestor search`

Busca semântica na base de conhecimento indexada (requer servidor rodando).

```bash
nodestor search "mecanismo de atenção transformer" --k 5
```

---

## Formato de Kit de Ferramentas da Comunidade

Crie um arquivo `.json` para definir ferramentas personalizadas para o motor de pesquisa profunda:

```json
{
  "name": "kit_fisica",
  "version": "1.0",
  "description": "Ferramentas de pesquisa em física quântica",
  "author": "Seu Nome",
  "tools": [
    {
      "tag": "call_vector_db",
      "description": "Busca artigos e fatos de física",
      "system_hint": "Use para recuperar informações factuais antes de responder."
    },
    {
      "tag": "call_dream",
      "description": "Gera hipóteses cross-domain",
      "system_hint": "Use para combinar física com outros campos."
    }
  ],
  "system_prompt": "Você é um físico teórico de elite. Use ferramentas para raciocinar passo a passo."
}
```

Use com: `nodestor run --model modelo.gguf --deep-research --tools-kit kit_fisica.json`
