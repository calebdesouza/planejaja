#!/usr/bin/env bash
# Download benchmark models from HuggingFace
# Usage: bash download_models.sh [small|medium|large|all]
# Default: small (SmolLM2-135M only — fast, fits any GPU)
set -euo pipefail

TIER="${1:-small}"
MODELS_DIR="$HOME/.nodestor/models"
mkdir -p "$MODELS_DIR"

pip3 install -q huggingface_hub 2>/dev/null || true

hf_download() {
    local repo="$1" file="$2" dest="$3"
    if [ -f "$dest" ]; then
        echo "  [SKIP] $(basename "$dest") already downloaded ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "  Downloading $file from $repo ..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$repo', filename='$file', local_dir='/tmp/hf_dl')
shutil.move(path, '$dest')
print('  OK:', '$dest', '(' + str(round(os.path.getsize('$dest')/1e6)) + ' MB)')
"
}

echo "═══ NodeStor model download (tier: $TIER) ═══"
echo "  Models → $MODELS_DIR"

# SmolLM2-135M-Instruct (258 MB) — baseline, tests GPU path on ANY GPU
if [[ "$TIER" =~ ^(small|medium|large|all)$ ]]; then
    hf_download \
        "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF" \
        "SmolLM2-135M-Instruct-F16.gguf" \
        "$MODELS_DIR/SmolLM2-135M-Instruct-F16.gguf"
fi

# Llama-3.2-1B-Instruct Q4_K_M (770 MB) — tests OOM fix, CPU fallback
if [[ "$TIER" =~ ^(medium|large|all)$ ]]; then
    hf_download \
        "bartowski/Llama-3.2-1B-Instruct-GGUF" \
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
fi

# Llama-3.1-7B-Instruct Q4_K_M (~4 GB) — tests 8GB VRAM full-GPU path
if [[ "$TIER" =~ ^(large|all)$ ]]; then
    hf_download \
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
fi

# Mistral-7B Q4_K_M (~4 GB) — second 7B for cross-model comparison
if [[ "$TIER" =~ ^(all)$ ]]; then
    hf_download \
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" \
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
        "$MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
fi

echo ""
echo "═══ Models ready ═══"
ls -lh "$MODELS_DIR"
