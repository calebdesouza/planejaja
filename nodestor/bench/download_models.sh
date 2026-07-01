#!/usr/bin/env bash
# Download benchmark models from HuggingFace (public, no auth required)
# Usage: bash download_models.sh [small|medium|large|all]
#   small  = Gemma-3-1B Q4_K_M  (~700 MB)
#   medium = + Gemma-3-4B Q4_K_M (~2.5 GB)
#   large  = + Llama-3.1-8B Q4_K_M (~4.7 GB)
#   all    = + Gemma-3-12B Q4_K_M (~7.5 GB)
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
    echo "  Downloading $(basename "$dest") ..."
    wget -q --show-progress -O "$dest.tmp" "$url" && mv "$dest.tmp" "$dest" \
        && echo "  [OK] $(du -h "$dest" | cut -f1)  $dest" \
        || { rm -f "$dest.tmp"; echo "  [FAIL] $url"; return 1; }
}

echo "═══ NodeStor model download (tier: $TIER) ═══"
echo "  Models → $MODELS_DIR"
echo ""

# Gemma-3-1B Q4_K_M (~700 MB) — roda em qualquer GPU, boa qualidade
if [[ "$TIER" =~ ^(small|medium|large|all)$ ]]; then
    hf_wget \
        "bartowski/gemma-3-1b-it-GGUF" \
        "gemma-3-1b-it-Q4_K_M.gguf" \
        "$MODELS_DIR/gemma-3-1b-it-Q4_K_M.gguf"
fi

# Gemma-3-4B Q4_K_M (~2.5 GB) — mostra vantagem SSD streaming
if [[ "$TIER" =~ ^(medium|large|all)$ ]]; then
    hf_wget \
        "bartowski/gemma-3-4b-it-GGUF" \
        "gemma-3-4b-it-Q4_K_M.gguf" \
        "$MODELS_DIR/gemma-3-4b-it-Q4_K_M.gguf"
fi

# Llama-3.1-8B Q4_K_M (~4.7 GB) — referência padrão da industria
if [[ "$TIER" =~ ^(large|all)$ ]]; then
    hf_wget \
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
        "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
fi

# Gemma-3-12B Q4_K_M (~7.5 GB) — cabe em 11GB VRAM (RTX 2080 Ti / 3080)
if [[ "$TIER" =~ ^(all)$ ]]; then
    hf_wget \
        "bartowski/gemma-3-12b-it-GGUF" \
        "gemma-3-12b-it-Q4_K_M.gguf" \
        "$MODELS_DIR/gemma-3-12b-it-Q4_K_M.gguf"
fi

echo ""
echo "═══ Models ready ═══"
ls -lh "$MODELS_DIR"
