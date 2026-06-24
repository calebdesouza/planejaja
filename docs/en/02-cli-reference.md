# NodeStor CLI Reference

All commands: `nodestor <command> [options]`  
Run `nodestor` alone to open the interactive menu.

---

## `nodestor run`

Run a prompt directly against a local model. Measures TTFT and tok/s in real time.

```bash
nodestor run "Your prompt here" --model /path/to/model.gguf
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | required | Path to GGUF or SafeTensors model |
| `--max-tokens` | 128 | Maximum tokens to generate |
| `--system` | — | System prompt (the model's "constitution") |
| `--profile` | — | Preset: `cientista`, `programador`, `advogado`, `professor`, `conciso`, `security` |
| `--steer-vector` | — | Activation steering vector (from `calibrate`) |
| `--intensity` | 1.0 | Steering strength (0.0 = off, 1.0 = full) |
| `--auto-steer` | false | Dynamic self-calibration pipeline (DSCP) |
| `--auto-calibrate` | false | DSCP + one-step LoRA micro-adaptation |
| `--loras` | — | LoRA adapters to apply (repeatable: `--loras a --loras b`) |
| `--deep-research` | false | Enable agent loop with tool calling |
| `--max-loops` | 10 | Maximum reasoning cycles in deep research mode |
| `--tools-kit` | — | Path to community toolkit JSON |
| `--dream` | false | Activate DAVI dream engine (cross-domain hypotheses) |

**Examples:**

```bash
# Simple run
nodestor run "Explain RoPE positional encoding" --model model.gguf

# With steering (removes "refusal" direction from activations)
nodestor run "Analyze this exploit" --model model.gguf --steer-vector security --profile security

# Deep research with DAVI
nodestor run "Propose a novel treatment for Alzheimer's" \
  --model model.gguf --deep-research --dream --max-loops 5

# Use a community toolkit
nodestor run "Derive the Schrödinger equation from first principles" \
  --model model.gguf --deep-research --tools-kit physics_kit.json
```

---

## `nodestor models`

List installed models.

```bash
nodestor models         # shows models in ~/.nodestor/models/
nodestor models --all   # also searches Downloads, Documents, HuggingFace cache
```

---

## `nodestor davi`

Direct interface to the DAVI dreaming system.

### `nodestor davi dream`

Run standalone dream cycles (no model required).

```bash
nodestor davi dream --topic "quantum physics + genetics"
nodestor davi dream --topic "economics + topology" --cycles 5 --temperature 8.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | `física + matemática + biologia` | Domains to cross-pollinate |
| `--cycles` | 3 | Number of dream cycles |
| `--temperature` | 5.0 | Semantic annealing temperature (higher = more exploratory) |

### `nodestor davi status`

Show the status of all 12 DAVI modules.

```bash
nodestor davi status
```

---

## `nodestor pull`

Download a model from HuggingFace Hub.

```bash
nodestor pull bartowski/SmolLM2-135M-GGUF --filename SmolLM2-135M-Q4_K_M.gguf
```

Models are saved to `~/.nodestor/models/`.

---

## `nodestor train`

Train a LoRA adapter on a local JSONL dataset.

```bash
nodestor train --model model.gguf --dataset data.jsonl --output my_adapter.lora
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | required | Base model path |
| `--dataset, -d` | required | JSONL dataset (`{"input":...,"output":...}` or `{"text":...}`) |
| `--output, -o` | required | Output adapter name/path |
| `--rank` | 8 | LoRA rank (4, 8, or 16 recommended) |
| `--alpha` | 8 | LoRA alpha |
| `--lr` | 0.0001 | AdamW learning rate |
| `--max-steps` | 0 | Max training steps (0 = full dataset) |
| `--grad-accum` | 4 | Gradient accumulation steps (VRAM control) |

---

## `nodestor calibrate`

### Hardware calibration (no arguments)
Measures I/O and GPU capabilities, saves optimal config.

```bash
nodestor calibrate
```

### Activation steering vector extraction
Extracts a direction vector from contrastive hidden states.

```bash
nodestor calibrate \
  --positive positive_examples.txt \
  --negative negative_examples.txt \
  --output my_direction \
  --model model.gguf
```

Vectors are saved to `~/.nodestor/vectors/<name>.bin`.

---

## `nodestor inspect`

Analyze a model file (GGUF or SafeTensors).

```bash
nodestor inspect /path/to/model.gguf
nodestor inspect /path/to/model.gguf --tensors   # show all tensors
```

---

## `nodestor scan`

Detect hardware: GPU, VRAM, CPU, SSD speed, Vulkan support.

```bash
nodestor scan
```

---

## `nodestor start`

Start the NodeStor daemon (background server) and attach the chat interface.

```bash
nodestor start --model model.gguf
```

---

## `nodestor chat`

Connect to a running NodeStor server.

```bash
nodestor chat --server http://localhost:8080
```

---

## `nodestor latency`

Measure TTFT (Time to First Token) and tok/s with a standard 32-token benchmark.

```bash
nodestor latency --model model.gguf
```

---

## `nodestor compress`

Compress a model or file.

```bash
nodestor compress model.gguf model.gz --format gdeflate
```

---

## `nodestor search`

Semantic search on the indexed knowledge base (requires running server).

```bash
nodestor search "transformer attention mechanism" --k 5
```

---

## Community Toolkit Format

Create a `.json` file to define custom tools for the deep research engine:

```json
{
  "name": "physics_kit",
  "version": "1.0",
  "description": "Quantum physics research tools",
  "author": "Your Name",
  "tools": [
    {
      "tag": "call_vector_db",
      "description": "Search physics papers and facts",
      "system_hint": "Use to retrieve factual information before answering."
    },
    {
      "tag": "call_dream",
      "description": "Generate cross-domain hypotheses",
      "system_hint": "Use to combine physics with other fields."
    }
  ],
  "system_prompt": "You are a theoretical physicist. Use tools to reason step by step."
}
```

Use with: `nodestor run --model model.gguf --deep-research --tools-kit physics_kit.json`
