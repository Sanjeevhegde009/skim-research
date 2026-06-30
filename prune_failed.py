"""Remove network-casualty records from a LongMemEval results file, so a resumed run
re-attempts exactly those. A casualty = the reader never successfully ran (reader_calls==0),
i.e. the recorded answer is a network artifact, not a model result. Safe: a real question
always makes >=1 reader call.

Usage:  python prune_failed.py [results/longmemeval/results.json]
Then re-run:  python run_longmemeval.py   (it resumes and redoes the removed question_ids)
"""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "results/longmemeval/results.json"
recs = json.load(open(p, encoding="utf-8"))
casualty = lambda r: (r.get("reader_calls", 0) or 0) == 0
keep = [r for r in recs if not casualty(r)]
removed = [r["question_id"] for r in recs if casualty(r)]
json.dump(keep, open(p, "w", encoding="utf-8"))
print(f"{p}: kept {len(keep)}, removed {len(removed)} network-casualty record(s): {removed}")
