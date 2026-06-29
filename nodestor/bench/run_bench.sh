#!/usr/bin/env bash
# NodeStor benchmark — runs all available models and outputs JSON results
# Usage: bash run_bench.sh [output_dir]
set -euo pipefail

OUTPUT_DIR="${1:-$HOME/bench_results}"
MODELS_DIR="$HOME/.nodestor/models"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
mkdir -p "$OUTPUT_DIR"

PROMPT="Explain in detail the architecture of a transformer language model, including attention mechanisms, layer normalization, feed-forward networks, and the role of positional encodings. Be thorough and precise."

echo "═══ NodeStor Benchmark Run — $TIMESTAMP ═══"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'CPU')"
echo "  CPU: $(nproc) cores  RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "  Output: $OUTPUT_DIR"
echo ""

run_nodestor() {
    local model_file="$1"
    local model_name
    model_name=$(basename "$model_file" .gguf)
    local out_file="$OUTPUT_DIR/nodestor_${model_name}_${TIMESTAMP}.json"

    echo "── NodeStor: $model_name ──"
    nodestor bench-infer \
        --model "$model_file" \
        --tokens 200 \
        --warmup 1 \
        --runs 3 \
        --prompt "$PROMPT" \
        --output "$out_file" \
        && echo "  Saved: $out_file" \
        || echo "  [FAILED] $model_name"
    echo ""
}

run_llama_cpp() {
    local model_file="$1"
    local model_name
    model_name=$(basename "$model_file" .gguf)
    local out_file="$OUTPUT_DIR/llama_cpp_${model_name}_${TIMESTAMP}.json"

    if ! command -v llama-cli &>/dev/null; then
        echo "  [SKIP] llama-cli not found — run setup_vast.sh first"
        return
    fi

    echo "── llama.cpp: $model_name ──"

    local total_tps=0
    local runs=3
    local tps_list=()

    for i in $(seq 1 $runs); do
        local t0=$SECONDS
        local output
        output=$(llama-cli \
            -m "$model_file" \
            -p "$PROMPT" \
            -n 200 \
            --n-gpu-layers 9999 \
            --threads "$(nproc)" \
            --log-disable \
            2>&1 | tail -3)
        local elapsed=$((SECONDS - t0))
        local tps
        tps=$(echo "$output" | grep -oP '\d+\.\d+ tokens per second' | grep -oP '[\d.]+' | head -1 || echo "0")
        tps_list+=("$tps")
        echo "  run $i/$runs: ${tps} tok/s"
    done

    # Compute stats
    local tps_json
    tps_json=$(printf '"%s",' "${tps_list[@]}")
    tps_json="[${tps_json%,}]"

    local tps_mean
    tps_mean=$(printf '%s\n' "${tps_list[@]}" | awk '{s+=$1;n++} END {printf "%.2f", s/n}')

    cat > "$out_file" << EOF
{
  "model": "$model_name",
  "backend": "llama.cpp",
  "hardware": { "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'CPU')" },
  "config": { "prompt_chars": ${#PROMPT}, "tokens_per_run": 200, "runs": $runs },
  "tps_runs": $tps_json,
  "summary": { "tps_mean": $tps_mean },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "  Mean: ${tps_mean} tok/s  Saved: $out_file"
    echo ""
}

# Find available models
declare -a MODELS=()
while IFS= read -r -d '' f; do
    MODELS+=("$f")
done < <(find "$MODELS_DIR" -name "*.gguf" -print0 2>/dev/null | sort -z)

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "[ERROR] No .gguf models found in $MODELS_DIR"
    echo "  Run: bash download_models.sh"
    exit 1
fi

echo "Found ${#MODELS[@]} model(s): ${MODELS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    run_nodestor "$model"
    run_llama_cpp "$model"
done

# ── Summary comparison ─────────────────────────────────────────────────────────
echo "═══ SUMMARY ═══"
echo "NodeStor results:"
for f in "$OUTPUT_DIR"/nodestor_*_"$TIMESTAMP".json; do
    [ -f "$f" ] || continue
    local_model=$(jq -r '.model' "$f" 2>/dev/null || echo "?")
    local_tps=$(jq -r '.summary.tps_mean' "$f" 2>/dev/null || echo "?")
    echo "  $local_model: $local_tps tok/s"
done

echo ""
echo "llama.cpp results:"
for f in "$OUTPUT_DIR"/llama_cpp_*_"$TIMESTAMP".json; do
    [ -f "$f" ] || continue
    local_model=$(jq -r '.model' "$f" 2>/dev/null || echo "?")
    local_tps=$(jq -r '.summary.tps_mean' "$f" 2>/dev/null || echo "?")
    echo "  $local_model: $local_tps tok/s"
done

echo ""
echo "All results in: $OUTPUT_DIR"
