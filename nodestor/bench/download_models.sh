#!/usr/bin/env bash
# Download benchmark models from HuggingFace (public, no auth required)
# Usage: bash download_models.sh [small|medium|large|all]
#   small  = Llama-3.2-1B Q4_K_M  (~875 MB)
#   medium = + Llama-3.2-3B Q4_K_M (~2.0 GB)
#   large  = + Llama-3.1-8B Q4_K_M (~4.9 GB)
#   all    = + Mistral-7B Q4_K_M    (~4.1 GB)
#
# Gemma requires accepting Google license at huggingface.co first.
# To use Gemma: export HF_TOKEN=hf_xxx before running this script.
set -euo pipefail

TIER="${1:-small}"
MODELS_DIR="$HOME/.nodestor/models"
mkdir -p "$MODELS_DIR"

hf_wget() {
    local repo="$1" file="$2" dest="$3"
    if [ -f "$dest" ]; then
        echo "  [SKIP] $(basename "$dest") ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    local auth_header=""
    [ -n "${HF_TOKEN:-}" ] && auth_header="--header=Authorization: Bearer ${HF_TOKEN}"
    echo "  Downloading $(basename "$dest") ..."
    wget -q --show-progress $auth_header -O "$dest.tmp" "$url" \
        && mv "$dest.tmp" "$dest" \
        && echo "  [OK] $(du -h "$dest" | cut -f1)" \
        || { rm -f "$dest.tmp"; echo "  [FAIL] $url"; return 1; }
}

echo "═══ NodeStor model download (tier: $TIER) ═══"
echo "  Models → $MODELS_DIR"
echo ""

# Llama-3.2-1B Q4_K_M (~875 MB) — Apache 2.0, fully public
if [[ "$TIER" =~ ^(small|medium|large|all)$ ]]; then
    hf_wget \
        "bartowski/Llama-3.2-1B-Instruct-GGUF" \
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
fi

# Llama-3.2-3B Q4_K_M (~2.0 GB) — shows memory bandwidth advantage
if [[ "$TIER" =~ ^(medium|large|all)$ ]]; then
    hf_wget \
        "bartowski/Llama-3.2-3B-Instruct-GGUF" \
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
fi

# Llama-3.1-8B Q4_K_M (~4.9 GB) — industry standard benchmark
if [[ "$TIER" =~ ^(large|all)$ ]]; then
    hf_wget \
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
fi

# Mistral-7B-v0.3 Q4_K_M (~4.1 GB) — Apache 2.0, cross-model comparison
if [[ "$TIER" =~ ^(all)$ ]]; then
    hf_wget \
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF" \
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
        "$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
fi

echo ""
echo "═══ Models ready ═══"
ls -lh "$MODELS_DIR"
