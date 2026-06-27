"""
LoCoMo data loader — parses locomo10.json into sessions and QA pairs.
"""

import json
import config


def load_conversations(path=None):
    """Load all LoCoMo conversations with sessions and QA."""
    path = path or config.LOCOMO_PATH
    with open(path) as f:
        data = json.load(f)

    conversations = []
    for conv_data in data:
        c = conv_data["conversation"]

        # Extract sessions in chronological order (natural sort: session_1, session_2, ..., session_19)
        session_keys = sorted(
            [k for k in c.keys() if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda x: int(x.split("_")[1])
        )
        sessions = []
        all_turns = []
        for sk in session_keys:
            dt = c.get(f"{sk}_date_time", "")
            turns = c[sk]
            if isinstance(turns, list):
                for t in turns:
                    if isinstance(t, dict) and "text" in t:
                        t["session"] = sk
                all_turns.extend([t for t in turns if isinstance(t, dict) and "text" in t])
                sessions.append({
                    "key": sk,
                    "date_time": dt,
                    "turns": [t for t in turns if isinstance(t, dict) and "text" in t],
                })

        # Extract QA
        qa_pairs = []
        for q in conv_data.get("qa", []):
            cat = q.get("category", 0)
            cat_name = config.QA_CATEGORIES.get(cat, f"cat_{cat}")

            # Adversarial questions have no 'answer', only 'adversarial_answer'
            if cat == 5:
                answer = "NOT_ANSWERABLE"
            else:
                answer = str(q.get("answer", ""))

            qa_pairs.append({
                "question": q["question"],
                "answer": answer,
                "category": cat,
                "category_name": cat_name,
                "evidence": q.get("evidence", []),
            })

        conversations.append({
            "speakers": f"{c.get('speaker_a', '?')} & {c.get('speaker_b', '?')}",
            "sessions": sessions,
            "all_turns": all_turns,
            "qa": qa_pairs,
            "sample_id": conv_data.get("sample_id", ""),
        })

    return conversations


def format_turns_as_text(turns):
    """Format turns as readable text for baselines."""
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


if __name__ == "__main__":
    convs = load_conversations()
    print(f"Loaded {len(convs)} conversations")
    for i, c in enumerate(convs):
        print(f"  {i}: {c['speakers']}, {len(c['sessions'])} sessions, "
              f"{len(c['all_turns'])} turns, {len(c['qa'])} QA")
        cats = {}
        for q in c["qa"]:
            cats[q["category_name"]] = cats.get(q["category_name"], 0) + 1
        print(f"     QA: {cats}")
