#!/usr/bin/env python3
"""Materialize one small BFCL session as cumulative Agent requests."""

import json
from pathlib import Path


SOURCE = Path("/root/dataset/hyrex_traces/bfcl_v4_agent_single_session.json")
TOOL_DOCS = Path(
    "/root/dataset/BFCL/berkeley-function-call-leaderboard/"
    "bfcl_eval/data/multi_turn_func_doc/gorilla_file_system.json"
)
OUTPUT = Path("/root/hyrex_results/motivation_bfcl_agent_1s10r_trace.jsonl")
USED_TOOLS = {"ls", "cd", "mv", "grep", "tail"}


def tool_catalog():
    docs = [json.loads(line) for line in TOOL_DOCS.read_text().splitlines()]
    return [doc for doc in docs if doc["name"] in USED_TOOLS]


def materialize(trace):
    system = (
        "System: You are a tool-using assistant. Available tools:\n"
        + json.dumps(tool_catalog(), separators=(",", ":"))
        + "\nUse a tool when needed.\n\n"
    )
    history = system
    rows = []
    request_index = 0
    pending_call = None
    for event in trace["events"]:
        role = event["role"]
        if role == "user":
            history += f"User: {event['content']}\n\nAssistant:"
            rows.append({
                "session_id": trace["trace_id"],
                "turn_index": request_index,
                "arrival_index": request_index,
                "conversation_turn": event["turn"],
                "request_kind": "user",
                "source_event_index": event["event_index"],
                "cache_prompt": history,
                "resume_prompt": history,
            })
            request_index += 1
        elif role == "assistant" and event["kind"] == "tool_call":
            pending_call = event
            call = {"name": event["name"], "arguments": event["arguments"]}
            history += " <tool_call>" + json.dumps(call, separators=(",", ":")) + "</tool_call>\n\n"
        elif role == "tool":
            assert pending_call and pending_call["name"] == event["name"]
            result = {"name": event["name"], "result": event["content"]}
            history += "Tool: " + json.dumps(result, separators=(",", ":")) + "\n\nAssistant:"
            rows.append({
                "session_id": trace["trace_id"],
                "turn_index": request_index,
                "arrival_index": request_index,
                "conversation_turn": event["turn"],
                "request_kind": "post_tool",
                "source_event_index": event["event_index"],
                "cache_prompt": history,
                "resume_prompt": history,
            })
            request_index += 1
            if event.get("assistant_followup"):
                history += " " + event["assistant_followup"] + "\n\n"
            pending_call = None
    assert pending_call is None
    return rows


def main():
    rows = materialize(json.loads(SOURCE.read_text()))
    assert len(rows) == 10
    assert [row["turn_index"] for row in rows] == list(range(10))
    assert all(rows[i]["resume_prompt"].startswith(rows[i - 1]["resume_prompt"])
               for i in range(1, len(rows)))
    OUTPUT.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"output": str(OUTPUT), "sessions": 1, "requests": len(rows)}))


if __name__ == "__main__":
    main()
