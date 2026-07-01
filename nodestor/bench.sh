#!/usr/bin/env bash
set -e
cd /root/repo/nodestor
git pull
CARGO_TARGET_DIR=/tmp/t cargo build --release -p nodestor-cli 2>&1 | tail -5
RUST_LOG=info /tmp/t/release/nodestor bench-infer \
  --model /root/.nodestor/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --tokens 200 --warmup 1 --runs 3 \
  --prompt "Explain the transformer architecture in detail"
