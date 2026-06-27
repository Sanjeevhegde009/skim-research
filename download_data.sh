#!/usr/bin/env bash
#
# Download the benchmark datasets fresh from their original public sources.
# Both files are .gitignore'd (large) — run this once in the project root.
#
#   LoCoMo         locomo10.json       (~2.7M)   snap-research/locomo       (GitHub)
#   LongMemEval_S  longmemeval_s.json  (~265M)   xiaowu0162/longmemeval     (HuggingFace, public)
#
# (longmemeval_session_summaries.json is NOT a source dataset — run_longmemeval.py
#  regenerates it as a cache.)
#
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/2] locomo10.json  <- snap-research/locomo (GitHub) ..."
curl -L --fail -o locomo10.json \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

echo "[2/2] longmemeval_s.json  <- HuggingFace xiaowu0162/longmemeval (~265M) ..."
curl -L --fail -o longmemeval_s.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"

echo
echo "Done:"
ls -lh locomo10.json longmemeval_s.json
