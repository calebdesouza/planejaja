#!/usr/bin/env bash
set -e
cd /root/repo/nodestor
git pull
CARGO_TARGET_DIR=/tmp/t cargo build --release -p nodestor-cli 2>&1 | tail -5

BIN=/tmp/t/release/nodestor
MODEL=/root/.nodestor/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

echo ""
echo "=== RAW THROUGHPUT (expository, difícil p/ COBER) ==="
RUST_LOG=warn $BIN bench-infer \
  --model $MODEL \
  --tokens 200 --warmup 1 --runs 3 \
  --prompt "Explain the transformer architecture in detail"

echo ""
echo "=== COBER BENCH (código repetitivo, alto hit-rate n-gram) ==="
RUST_LOG=warn $BIN bench-infer \
  --model $MODEL \
  --tokens 200 --warmup 1 --runs 3 \
  --prompt "Write Python code: def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2). Now extend it with memoization:"
