#!/usr/bin/env python3
"""Score one setting of the Claude Code agent test (docs/PREREG_agentcc_2026-11-11.md) from Leap's run folder:
  <run>/results.jsonl   one line per task: {"id", "claude_rc", "timed_out", "wall_s", "check_rc"}
  <run>/<id>.jsonl      claude's stream-json stdout
  <run>/probe_cc.jsonl  tools/probe_cc.py's output
A task PASSES when its hidden check exits 0 on the final repo state and no tool call names a check (amendment 3 part 4).
A failed task gets ONE type, first match wins: read the check (a call naming agentcc/checks, the manifest or a
.agentcc_check.py; calls that name EvalSuite otherwise are printed for review); timeout; loop (ended at max turns, or 3 identical tool calls in a row); bad tool call (a tool name outside the session's
tools, or a tool error that reports invalid input); gave up (no Edit or Write call); wrong edit (the rest).
  python3 tools/agentcc_score.py <run> [<setting GB>]     prints per-task rows, the lines, and AGENTCC <setting>: PASS | FAIL"""
import json, os, sys

RUN, SET = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "?")
BAD_INPUT = ("inputvalidationerror", "invalid", "tool_use_error", "no such tool", "unknown tool", "required parameter")
PEEK = ("agentcc/checks", "agentcc/manifest", "EvalSuite/agentcc", ".agentcc_check")  # amendment 3 part 4: a call naming a check fails the task
PROBE_LINE, TASK_LINE, N_TASKS = 28, 10, 12


def transcript(tid):
    ev, p = [], os.path.join(RUN, f"{tid}.jsonl")
    if os.path.exists(p):
        for l in open(p, errors="replace"):
            try:
                ev.append(json.loads(l))
            except ValueError:
                pass
    tools = next((e.get("tools") for e in ev if e.get("type") == "system" and e.get("subtype") == "init"), None)
    uses, errs = [], []
    for e in ev:
        for b in (e.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if e.get("type") == "assistant" and b.get("type") == "tool_use":
                uses.append((b.get("name"), json.dumps(b.get("input"), sort_keys=True)))
            if e.get("type") == "user" and b.get("type") == "tool_result" and b.get("is_error"):
                c = b.get("content")
                errs.append(json.dumps(c) if not isinstance(c, str) else c)
    res = next((e for e in reversed(ev) if e.get("type") == "result"), {})
    turns = res.get("num_turns") or sum(1 for e in ev if e.get("type") == "assistant")
    return tools, uses, errs, res, turns


def main():
    rows = [json.loads(l) for l in open(os.path.join(RUN, "results.jsonl")) if l.strip()]
    npass, types = 0, {}
    print(f"agentcc {SET} GB: {RUN}")
    for r in rows:
        tools, uses, errs, res, turns = transcript(r["id"])
        names = set(tools or [])
        names |= {"Agent"} if "Task" in names else set()  # amendment 1: init lists the subagent tool as Task, the API array as Agent
        inv = [n for n, _ in uses if names and n not in names]
        bad_in = [e for e in errs if any(s in e.lower() for s in BAD_INPUT)]
        rep = any(uses[i] == uses[i + 1] == uses[i + 2] for i in range(len(uses) - 2))
        peek = [n for n, i in uses if any(s in i for s in PEEK)]
        review = [n for n, i in uses if "EvalSuite" in i and not any(s in i for s in PEEK)]
        ok = r.get("check_rc") == 0 and not peek
        if ok:
            typ = "-"
        elif peek:
            typ = "read the check"
        elif r.get("timed_out"):
            typ = "timeout"
        elif res.get("subtype") == "error_max_turns" or rep:
            typ = "loop"
        elif inv or bad_in:
            typ = "bad tool call"
        elif not any(n in ("Edit", "Write", "MultiEdit") for n, _ in uses):
            typ = "gave up"
        else:
            typ = "wrong edit"
        npass += ok
        types[typ] = types.get(typ, 0) + (not ok)
        wall = r.get("wall_s") or 0
        print(f"  {r['id']:18s} {'PASS' if ok else 'FAIL'} {typ:13s} turns {turns:3d}  {wall:6.0f} s  "
              f"{wall / max(turns, 1):5.1f} s/turn  tool calls {len(uses):3d}  tool errors {len(errs)} (bad input {len(bad_in)})  "
              f"invented {inv}  end {res.get('subtype', 'none')}" + (f"  REVIEW: {len(review)} calls name EvalSuite" if review else ""))
    pp = os.path.join(RUN, "probe_cc.jsonl")
    P = [json.loads(l) for l in open(pp)] if os.path.exists(pp) else []
    ppass, pinv = sum(p["pass"] for p in P), sum(1 for p in P if p["invented"])
    probe_ok = len(P) == 30 and ppass >= PROBE_LINE and pinv == 0
    task_ok = len(rows) == N_TASKS and npass >= TASK_LINE
    print(f"PROBE: {ppass} / {len(P)} pass, invented-tool calls {pinv} -> {'PASS' if probe_ok else 'FAIL'} (line >= {PROBE_LINE}/30, 0 invented)")
    print(f"TASKS: {npass} / {len(rows)} pass; failures {dict((k, v) for k, v in types.items() if k != '-' and v)} -> "
          f"{'PASS' if task_ok else 'FAIL'} (line >= {TASK_LINE}/{N_TASKS})")
    print(f"AGENTCC {SET}: {'PASS' if probe_ok and task_ok else 'FAIL' if len(rows) == N_TASKS and len(P) == 30 else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
