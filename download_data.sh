#!/usr/bin/env bash
#
# Download the benchmark datasets fresh from their original public sources.
# Both files are .gitignore'd (large) — run this once; they land in data/.
#
#   LoCoMo         locomo10.json       (~2.7M)   snap-research/locomo       (GitHub)
#   LongMemEval_S  longmemeval_s.json  (~265M)   xiaowu0162/longmemeval-cleaned (HuggingFace, official)
#                  (the original xiaowu0162/longmemeval is deprecated; cleaned removes noisy sessions)
#
# (longmemeval_session_summaries.json is NOT a source dataset — run_longmemeval.py
#  regenerates it as a cache.)
#
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

echo "[1/2] data/locomo10.json  <- snap-research/locomo (GitHub) ..."
curl -L --fail -o data/locomo10.json \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

echo "[2/2] data/longmemeval_s.json  <- HuggingFace xiaowu0162/longmemeval-cleaned (~265M) ..."
curl -L --fail -o data/longmemeval_s.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

echo
echo "Done:"
ls -lh data/locomo10.json data/longmemeval_s.json
