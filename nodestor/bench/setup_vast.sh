#!/usr/bin/env bash
# NodeStor — vast.ai instance setup
# Run once after SSH into the instance:
#   bash <(curl -fsSL https://raw.githubusercontent.com/calebdesouza/planejaja/main/nodestor/bench/setup_vast.sh)
# Or: upload this file and run: bash setup_vast.sh
set -euo pipefail

echo "═══ NodeStor Setup for vast.ai ═══"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "CPU: $(nproc) cores  RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential pkg-config curl git wget \
    libvulkan-dev vulkan-tools \
    libssl-dev \
    python3-pip python3-venv \
    jq bc

# Vulkan ICD for NVIDIA (driver already installed on vast.ai)
# Check Vulkan works
if vulkaninfo --summary 2>/dev/null | grep -q "apiVersion"; then
    echo "  [OK] Vulkan available: $(vulkaninfo --summary 2>/dev/null | grep apiVersion | head -1)"
else
    echo "  [WARN] vulkaninfo failed — inference will fall back to CPU path"
fi

# ── 2. Rust toolchain ─────────────────────────────────────────────────────────
echo "[2/5] Installing Rust..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
else
    echo "  [OK] Rust already installed: $(rustc --version)"
fi
source "$HOME/.cargo/env"
rustup update stable

# ── 3. Clone NodeStor ─────────────────────────────────────────────────────────
echo "[3/5] Cloning NodeStor..."
REPO_DIR="$HOME/nodestor"
if [ -d "$REPO_DIR/.git" ]; then
    echo "  [OK] Repo exists, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    # If repo is private, set GITHUB_TOKEN env var before running this script
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git clone "https://$GITHUB_TOKEN@github.com/calebdesouza/planejaja.git" "$HOME/planejaja"
    else
        git clone "https://github.com/calebdesouza/planejaja.git" "$HOME/planejaja"
    fi
    REPO_DIR="$HOME/planejaja"
fi

NODESTOR_DIR="$REPO_DIR/nodestor"
echo "  [OK] NodeStor at: $NODESTOR_DIR"

# ── 4. Build release ──────────────────────────────────────────────────────────
echo "[4/5] Building NodeStor (release)..."
cd "$NODESTOR_DIR"
export CARGO_TARGET_DIR="/tmp/nodestor-target"
cargo build --release -p nodestor-cli 2>&1 | tail -5
NODESTOR_BIN="/tmp/nodestor-target/release/nodestor"
echo "  [OK] Binary: $NODESTOR_BIN"

# Symlink for convenience
ln -sf "$NODESTOR_BIN" /usr/local/bin/nodestor
echo "  [OK] Installed to /usr/local/bin/nodestor"

# ── 5. Install llama.cpp for comparison ───────────────────────────────────────
echo "[5/5] Building llama.cpp for comparison..."
LLAMA_DIR="$HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth=1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON 2>&1 | tail -3
cmake --build build --config Release -j$(nproc) --target llama-cli 2>&1 | tail -5
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-cli"
ln -sf "$LLAMA_BIN" /usr/local/bin/llama-cli
echo "  [OK] llama.cpp: $LLAMA_BIN"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══ Setup complete ═══"
echo "  nodestor : $(nodestor --version 2>/dev/null || echo 'ready')"
echo "  Next     : bash $NODESTOR_DIR/bench/download_models.sh"
echo "  Then     : bash $NODESTOR_DIR/bench/run_bench.sh"
