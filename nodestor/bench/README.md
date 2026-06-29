# NodeStor Bench — vast.ai Setup Guide

## Quick Start (copy-paste into vast.ai SSH session)

```bash
# 1. Setup environment (Rust + Vulkan + build NodeStor + llama.cpp)
curl -fsSL https://raw.githubusercontent.com/calebdesouza/planejaja/main/nodestor/bench/setup_vast.sh | bash

# 2. Download models (options: small | medium | large | all)
#    small  = SmolLM2-135M  (~258 MB)   — any GPU, <30s download
#    medium = + Llama-3.2-1B (~770 MB)  — tests OOM fix
#    large  = + Llama-3.1-8B (~4 GB)    — tests 8GB VRAM GPU path
bash ~/planejaja/nodestor/bench/download_models.sh medium

# 3. Run benchmarks (NodeStor vs llama.cpp, saves JSON to ~/bench_results/)
bash ~/planejaja/nodestor/bench/run_bench.sh
```

## Instance Recommendations

| GPU          | Price/hr | VRAM | Best for               |
|--------------|----------|------|------------------------|
| GTX 1080     | $0.063   | 8GB  | medium tier, 1B model  |
| RTX 2060     | $0.076   | 6GB  | small+medium           |
| RTX 2070     | $0.103   | 8GB  | large tier, 7B model   |

**Pick GTX 1080 ($0.063/hr)** for the best cost/value ratio.
The full benchmark run (small+medium tier) takes ~15 minutes.

## What Gets Measured

- **TTFT** (Time to First Token) — latency
- **TPS** (Tokens per Second) — throughput
- Warmup runs discarded; 3 measurement runs per model
- Same prompt, same token budget for NodeStor and llama.cpp

## Output

JSON files in `~/bench_results/`:
```
nodestor_SmolLM2-135M-Instruct-F16_20260629T143000.json
llama_cpp_SmolLM2-135M-Instruct-F16_20260629T143000.json
```

Each file has `.summary.tps_mean` and per-run breakdown.

## Private Repo

If the GitHub repo is private, set your token before running setup:
```bash
export GITHUB_TOKEN=ghp_your_token_here
bash setup_vast.sh
```
