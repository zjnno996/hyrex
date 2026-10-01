#!/usr/bin/env python3
"""Pick four real 10-turn ShareGPT sessions for CPU-restore replay."""

import json
from pathlib import Path

import ijson


SOURCE = Path("/root/dataset/ShareGPT_V3_unfiltered_cleaned_split.json")
OUTPUT = Path("/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl")
SESSION_IDS = ("Ud3L2sd_124", "Pr8nMeM_0", "WRAImOg_0", "WnjND3T_0")
TURNS = 10


def prompts(conversation):
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
        if len(result) == TURNS:
            break
    return result


def main():
    sessions = {}
    with SOURCE.open("rb") as source:
        for item in ijson.items(source, "item"):
            if item["id"] not in SESSION_IDS:
                continue
            turns = prompts(item.get("conversations", []))
            assert len(turns) == TURNS, item["id"]
            sessions[item["id"]] = turns
            if len(sessions) == len(SESSION_IDS):
                break
    assert len(sessions) == len(SESSION_IDS), "missing selected sessions"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as out:
        for session_id in SESSION_IDS:
            turns = sessions[session_id]
            for turn in range(TURNS):
                prompt = turns[turn]
                out.write(json.dumps({"session_id": session_id, "turn_index": turn,
                                      "arrival_index": SESSION_IDS.index(session_id) * TURNS + turn,
                                      "cache_prompt": prompt, "resume_prompt": prompt}) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sessions": SESSION_IDS,
                      "requests": len(SESSION_IDS) * TURNS}))


if __name__ == "__main__":
    assert prompts([{"from": "human", "value": "hi"},
                    {"from": "gpt", "value": "hello"}]) == ["User: hi\n\nAssistant:"]
    main()
