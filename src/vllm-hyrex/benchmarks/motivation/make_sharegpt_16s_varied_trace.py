#!/usr/bin/env python3
"""Build a fixed 16-session, 5--10-turn ShareGPT recovery trace."""

import json
from pathlib import Path

import ijson
from tokenizers import Tokenizer


SOURCE = Path("/root/dataset/ShareGPT_V3_unfiltered_cleaned_split.json")
TOKENIZER = Path("/root/models/Qwen3.5-9B/tokenizer.json")
OUTPUT = Path("/root/hyrex_results/motivation_sharegpt_16s_varied_trace.jsonl")
TURN_TARGETS = (5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9)


def prompts(conversation, limit):
    history = ""
    result = []
    for user, assistant in zip(conversation[::2], conversation[1::2]):
        if user.get("from") != "human" or assistant.get("from") != "gpt":
            break
        question = user.get("value") or ""
        answer = assistant.get("value") or ""
        if not question or not answer:
            break
        result.append(history + f"User: {question}\n\nAssistant:")
        history += f"User: {question}\n\nAssistant: {answer}\n\n"
        if len(result) == limit:
            break
    return result


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    selected = []
    with SOURCE.open("rb") as source:
        for item in ijson.items(source, "item"):
            target = TURN_TARGETS[len(selected)]
            turns = prompts(item.get("conversations", []), target)
            if len(turns) != target or len(turns[0]) < 1200:
                continue
            lengths = [len(tokenizer.encode(prompt).ids) for prompt in turns]
            if 528 <= lengths[0] <= 1000 and lengths[-1] <= 3000:
                selected.append((item["id"], turns, lengths))
                if len(selected) == len(TURN_TARGETS):
                    break

    if len(selected) != len(TURN_TARGETS):
        raise RuntimeError(f"found only {len(selected)} suitable sessions")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    arrival = 0
    with OUTPUT.open("w") as out:
        for session_id, turns, _ in selected:
            for turn_index, prompt in enumerate(turns):
                out.write(json.dumps({
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "arrival_index": arrival,
                    "cache_prompt": prompt,
                    "resume_prompt": prompt,
                }) + "\n")
                arrival += 1

    print(json.dumps({
        "output": str(OUTPUT),
        "sessions": [
            {"id": session_id, "turns": len(turns), "token_lengths": lengths}
            for session_id, turns, lengths in selected
        ],
        "requests": arrival,
    }, indent=2))


if __name__ == "__main__":
    assert prompts([
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"},
    ], 1) == ["User: hi\n\nAssistant:"]
    main()
