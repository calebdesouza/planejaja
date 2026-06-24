# NodeStor CLI — Complete Reference

> All commands produce real metrics (TTFT, tok/s, latency) — nothing is estimated or hardcoded.

Run `nodestor` without arguments to open the interactive menu.

---

## Command Index

| Command | What it does |
|---------|-------------|
| [`run`](#run) | Direct inference (no server) with full feature set |
| [`models`](#models) | List installed models with size and quantization type |
| [`davi`](#davi) | DAVI dream engine — autonomous cross-domain discovery |
| [`pull`](#pull) | Download models from HuggingFace |
| [`train`](#train) | Local LoRA adapter training |
| [`calibrate`](#calibrate) | Hardware calibration and activation steering vector extraction |
| [`inspect`](#inspect) | Detailed analysis of model files |
| [`scan`](#scan) | Hardware detection and system capabilities |
| [`start`](#start) | Start the NodeStor daemon in background |
| [`chat`](#chat) | Interactive chat via server |
| [`latency`](#latency) | TTFT and tok/s benchmark |
| [`compress`](#compress) | Model and file compression |
| [`search`](#search) | Semantic search on the vector database |

---

## `nodestor run` {#run}

**Direct local inference** — loads the model, runs the prompt, streams tokens, and measures real metrics.

```bash
nodestor run "<prompt>" --model <path_or_name> [options]
```

### Full Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model, -m` | `String` | **required** | Path to `.gguf` or `.safetensors` file |
| `--max-tokens` | `usize` | `128` | Maximum number of tokens to generate |
| `--system` | `String` | — | System prompt (free text) |
| `--profile` | `String` | — | Ready-made profile (see list below) |
| `--steer-vector` | `String` | — | Direction vector name or path |
| `--intensity` | `f32` | `1.0` | Steering strength (0.0–2.0) |
| `--auto-steer` | `bool` | `false` | DSCP: in-memory auto-calibration steering |
| `--auto-calibrate` | `bool` | `false` | DSCP + LoRA micro-adaptation before generation |
| `--loras` | `Vec<String>` | `[]` | LoRA adapters (repeatable) |
| `--deep-research` | `bool` | `false` | Enable agent loop with tools |
| `--max-loops` | `usize` | `10` | Maximum cycles in Deep Research |
| `--tools-kit` | `String` | — | Community toolkit JSON file |
| `--dream` | `bool` | `false` | Enable DAVI handler in Deep Research |

### Ready-Made Profiles (`--profile`)

| Profile | Applied system prompt |
|---------|----------------------|
| `scientist` / `cientista` | Rigorous reasoning, fact/hypothesis separation, evidence citation |
| `programmer` / `programador` | Clean, idiomatic code, explicit trade-offs |
| `lawyer` / `advogado` | Legal precision, principle citation, uncertainty flagging |
| `teacher` / `professor` | Didactic explanation with examples, comprehension check |
| `concise` / `conciso` | Direct, minimal answer with no preamble |
| `security` | Vulnerability analysis for authorized security research |

### Model Path Resolution

If `--model` is an absolute or relative path with `/` or `\`, it's used directly. Otherwise, the system looks in `~/.nodestor/models/`.

### Deep Research Mode (`--deep-research`)

Activates the `AgentExecutionLoop`. The model can emit tool tags in the generated text:

```
<call_vector_db>your search query</call_vector_db>
<call_dream>domain1 + domain2</call_dream>
<call_think>internal reasoning</call_think>
<call_hypothesis>formal hypothesis</call_hypothesis>
```

The pipeline intercepts these tags, executes the tool, injects the response into context, and resumes generation. The cycle continues until the model produces a response without a tool call, or `--max-loops` is reached.

**Dynamic temperature**: The system detects stagnation (token repetition) and automatically raises temperature to force creative exploration. When stagnation resolves, temperature gradually cools.

### Detailed Examples

```bash
# Simple inference
nodestor run "What is entropy in thermodynamics?" \
  --model ~/.nodestor/models/model.gguf \
  --max-tokens 512

# With scientist profile and generous token budget
nodestor run "Explain the EPR paradox (Einstein-Podolsky-Rosen)" \
  --model model.gguf \
  --profile scientist \
  --max-tokens 1024

# With activation steering (removes "refusal" direction from activations)
nodestor run "Analyze vulnerabilities in this code" \
  --model model.gguf \
  --steer-vector security \
  --intensity 0.8 \
  --profile security

# Dynamic self-calibration (no external files)
nodestor run "Explain game theory" \
  --model model.gguf \
  --auto-steer \
  --intensity 1.2

# With LoRA adapters
nodestor run "Write in formal legal style" \
  --model model.gguf \
  --loras legal --loras formal

# Full Deep Research with DAVI
nodestor run "Propose a molecular mechanism for Alzheimer's cure" \
  --model model.gguf \
  --deep-research \
  --dream \
  --max-loops 8 \
  --max-tokens 512

# Community toolkit
nodestor run "Derive quantum mechanics postulates from first principles" \
  --model model.gguf \
  --deep-research \
  --tools-kit ~/kits/quantum_physics.json \
  --max-loops 5
```

### Deep Research Output

```
╔══ DEEP RESEARCH MODE ═══════════════════════════════╗
║  Cyclic reasoning | Tools active | DAVI ON           ║
╚══════════════════════════════════════════════════════╝

[Loop 1] T=0.70 — Generating...
[TTFT: 930ms] I need to check the current state of tau research...
<call_vector_db>tau protein Alzheimer mechanism</call_vector_db>

[call_vector_db] Query: "tau protein Alzheimer mechanism"
Result: [1] Tau forms neurofibrillary tangles via phosphorylation...

[Loop 2] T=0.70 — Generating...
Based on the data, I'll cross with protein biophysics findings...
<call_dream>Alzheimer tau + polymer physics + percolation theory</call_dream>

[DAVI Dream] Crossing domains: Alzheimer tau + polymer physics...
Hypothesis: Tau propagation follows network percolation dynamics...

[Loop complete — final answer above]

╔══ DEEP RESEARCH — METRICS ═══════════════════════════╗
║  Loops executed: 3      Max configured: 8              ║
║  Total tokens:  847     Speed: 4.8 t/s                 ║
║  Total time:    176s                                   ║
╚══════════════════════════════════════════════════════╝
```

---

## `nodestor models` {#models}

Lists installed models with size, format, and auto-detected quantization type.

```bash
nodestor models          # only ~/.nodestor/models/
nodestor models --all    # + Downloads, Documents, HuggingFace cache
```

### Example Output

```
╔══ INSTALLED MODELS ════════════════════════════════════╗
  📁 C:\Users\User\.nodestor\models
  ✔  SmolLM2-135M-Q4_K_M.gguf                     0.09 GB  [GGUF/Q4_K_M]
  ✔  Llama-3.1-8B-Instruct-Q5_K_M.gguf            5.73 GB  [GGUF/Q5_K_M]
  ✔  Mistral-7B-v0.3.F16.gguf                     13.98 GB  [GGUF/F16]
╚════════════════════════════════════════════════════════╝
```

**Auto-detected quantization types**: Q4_K_M, Q4_K_S, Q5_K_M, Q8_0, Q4_0, F16, F32, BF16, SafeTensors.

---

## `nodestor davi` {#davi}

Direct interface to the DAVI system. No model required.

### `nodestor davi status`

Shows the status of all 12 DAVI modules and how to activate them.

```bash
nodestor davi status
```

### `nodestor davi dream`

Runs autonomous dream cycles. The system generates cross-domain hypotheses without needing a language model — it uses only DAVI's mathematical modules.

```bash
nodestor davi dream --topic "<domain1> + <domain2> + ..."
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--topic` | `String` | `physics + mathematics + biology` | Domains to cross-pollinate (separated by `+`) |
| `--cycles` | `usize` | `3` | Number of dream cycles |
| `--temperature` | `f32` | `5.0` | Initial annealing temperature (higher = more exploratory) |

#### What happens internally

1. Topic is split into domains by `+`
2. Each domain generates embeddings via FNV-1a hashing (64 dimensions)
3. Embeddings are passed to `DreamingEngine::dream_cycle()`
4. Full TDA→FEP→Annealing→Nash→Functors→Stigmergy→Autopoiesis cycle runs
5. Discoveries validated by the Nash Tribunal are displayed with confidence scores

#### Examples

```bash
# Classic physics + biology crossover
nodestor davi dream --topic "quantum physics + molecular biology"

# More exploratory (high temperature)
nodestor davi dream --topic "topology + economics + neuroscience" \
  --cycles 7 --temperature 9.0

# More conservative (refinement)
nodestor davi dream --topic "thermodynamics + information theory" \
  --cycles 2 --temperature 2.0
```

---

## `nodestor pull` {#pull}

Download models from HuggingFace Hub with a progress bar.

```bash
nodestor pull <model-id> --filename <file>
```

```bash
# SmolLM2 quantized (recommended for testing)
nodestor pull bartowski/SmolLM2-135M-GGUF \
  --filename SmolLM2-135M-Q4_K_M.gguf

# Llama 3.1 8B
nodestor pull bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --filename Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

# Mistral full precision
nodestor pull TheBloke/Mistral-7B-v0.1-GGUF \
  --filename mistral-7b-v0.1.Q8_0.gguf
```

Models are saved to `~/.nodestor/models/<filename>`.

---

## `nodestor train` {#train}

Train LoRA adapters locally. Base weights are **100% frozen** — only low-rank matrices are learned.

```bash
nodestor train \
  --model <model.gguf> \
  --dataset <data.jsonl> \
  --output <adapter.lora>
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model, -m` | `String` | required | Base model GGUF/SafeTensors |
| `--dataset, -d` | `String` | required | JSONL dataset |
| `--output, -o` | `String` | required | Output adapter name/path |
| `--rank` | `usize` | `8` | LoRA rank (4–16 recommended) |
| `--alpha` | `f32` | `8` | LoRA alpha (scale) |
| `--lr` | `f32` | `0.0001` | AdamW optimizer learning rate |
| `--max-steps` | `usize` | `0` | Max training steps (0 = full dataset) |
| `--grad-accum` | `usize` | `4` | Gradient accumulation steps (VRAM control) |

### JSONL Dataset Format

Any of three formats is accepted:

```jsonl
{"input": "Question here", "output": "Answer here"}
{"text": "Free training text"}
{"prompt": "Instruction", "completion": "Expected response"}
```

### Example

```bash
# Specialize in medical terminology
nodestor train \
  --model model.gguf \
  --dataset medical_notes.jsonl \
  --output medical_en.lora \
  --rank 16 \
  --lr 0.00005 \
  --max-steps 500

# Then use the adapter
nodestor run "Summarize this patient history" \
  --model model.gguf \
  --loras medical_en
```

Adapters are saved to `~/.nodestor/loras/<name>.lora`.

---

## `nodestor calibrate` {#calibrate}

### Mode 1: Hardware Calibration

Without arguments, measures system capabilities and saves optimal configuration.

```bash
nodestor calibrate
```

Detects: SSD I/O speed, available VRAM, Vulkan support, memory bandwidth.

### Mode 2: Activation Steering Vector Extraction

With `--positive` and `--negative`, runs the **DSCP (Dynamic Self-Calibration Pipeline)**:

1. Loads the model
2. Processes positive examples (desired behavior) → extracts hidden states
3. Processes negative examples (undesired behavior) → extracts hidden states
4. Computes normalized mean difference → direction vector `d̂`
5. Saves to `~/.nodestor/vectors/<output>.bin`

```bash
nodestor calibrate \
  --positive technical_examples.txt \
  --negative generic_examples.txt \
  --output technical_direction \
  --model model.gguf
```

Then: `nodestor run "..." --model model.gguf --steer-vector technical_direction`

### Auto Mode (`--auto-steer`)

Without external files, uses 6 internal positive and 6 negative templates to generate the vector in memory.

```bash
nodestor run "..." --model model.gguf --auto-steer --intensity 1.0
```

---

## `nodestor inspect` {#inspect}

Analyzes model file metadata without fully loading it.

```bash
nodestor inspect /path/to/model.gguf
nodestor inspect /path/to/model.gguf --tensors   # list all tensors
```

**Information shown**: GGUF version, parameter count, architecture, vocabulary, max context, quantization type, tensor list with name/shape/type.

---

## `nodestor scan` {#scan}

Detects hardware and shows system information.

```bash
nodestor scan
```

**Detects**: GPU (name, vendor, VRAM), CPU (cores, frequency), available RAM, NVMe vs SATA SSD, Vulkan support, DirectStorage (Windows), PCIe generation.

---

## `nodestor start` {#start}

Starts the NodeStor daemon (background process) and attaches the chat interface.

```bash
nodestor start                              # auto-detects model in ~/.nodestor/models/
nodestor start --model /path/to/model.gguf
nodestor start --model model.gguf --quiet   # silent mode (for scripts)
```

The daemon exposes: `GET /health`, `GET /status`, `POST /chat`, `GET /search`.

---

## `nodestor chat` {#chat}

Opens an interactive chat interface connecting to the server.

```bash
nodestor chat                                      # default http://localhost:8080
nodestor chat --server http://192.168.1.10:8080    # remote server
```

---

## `nodestor latency` {#latency}

Precise benchmark of TTFT (Time to First Token) and generation speed.

```bash
nodestor latency                     # uses auto-detected model
nodestor latency --model model.gguf
```

Uses standard 32-token prompt ("The quick brown fox") for reproducible measurement.

---

## `nodestor compress` {#compress}

Compresses models and files with GDeflate (Vulkan-accelerated) or Zstd.

```bash
nodestor compress model.gguf model.gdf --format gdeflate
nodestor compress model.gguf model.zst --format zstd
```

GDeflate is decompressed on the GPU in microseconds — allows a 7 GB/s SSD to appear to deliver 14–20 GB/s of model data.

---

## `nodestor search` {#search}

Semantic search on the vector database (requires active daemon with indexed documents).

```bash
nodestor search "multi-head attention mechanism" --k 5
```

The `/search` endpoint uses HNSW + BM25 with RRF (Reciprocal Rank Fusion). Place `.txt` or `.md` files in the `./knowledge/` folder and the daemon indexes them automatically.

---

## Community Toolkit Format

The `.json` format allows creating and sharing specialized toolkits.

### File Structure

```json
{
  "name": "toolkit_name",
  "version": "1.0",
  "description": "Purpose description of this toolkit",
  "author": "Your Name <your@email.com>",
  "tools": [
    {
      "tag": "call_vector_db",
      "description": "What this tool does (injected into system prompt)",
      "system_hint": "When the model should use this tool"
    },
    {
      "tag": "call_dream",
      "description": "Generates cross-domain hypotheses via DAVI",
      "system_hint": "Use when you need to combine concepts from different fields"
    }
  ],
  "system_prompt": "Specialized system prompt that defines the agent behavior for this toolkit"
}
```

### Available Tool Tags

| Tag | What it does | When to use |
|-----|-------------|-------------|
| `call_vector_db` | Searches the HNSW+BM25 vector database | Retrieving facts, papers, memories |
| `call_dream` | Invokes DAVI for cross-domain hypotheses | Combining concepts from different areas |
| `call_think` | Structured internal reflection | Reviewing reasoning before concluding |
| `call_hypothesis` | Validates hypothesis through Nash Tribunal | Formally testing a claim |

### Built-in Kits

NodeStor includes two kits requiring no external file:

- **`science`** (default): 4 tools, autonomous scientist system prompt
- **`coding`**: 2 tools, senior software engineer system prompt

To use a built-in kit, just use `--deep-research` without `--tools-kit`.

---

## Interactive Menu

Running `nodestor` without arguments opens the menu:

```
NODE-PANEL
❯ RUN      - Run prompt directly (local inference, no server)
  START    - Start Engine + Interface (FULL ENGINE)
  ATTACH   - Connect to Resident Engine (RESIDENT)
  CHATTING - Start Local Chat (Streaming Mode)
  DREAM    - DAVI: Cross-Domain Dream Engine
  MODELS   - List installed models
  PULL     - Download model from HuggingFace
  LATENCY  - 7-Layer Response Test (TTFT)
  SCANNER  - Industrial Hardware Inspection
  DETACH   - Exit and keep engine in Background
  EXIT     - Shut down everything
```

Use `↑↓` to navigate, `Enter` to select, `Esc` to go back.
