#!/usr/bin/env python3
"""T1 / T2 replay for the conversation cache: post Leap's recorded requests (agent/sessions_real_1110.jsonl, kind=request:
Claude Code /v1/messages, Codex /v1/responses, the tool tests on all three APIs) to a running serve.py, in their order, each
answer capped at --cap tokens. The prompts do not depend on the answers: each body carries its history as recorded. Then
the cache's own lines (CONV_CACHE_LOG) are read, one per request, and reported: hit or miss, restore point, tokens
prefilled, and with CONV_CACHE_CHECK=1 the comparison with a cold prefill. T1 = the server started with CONV_CACHE_CHECK=1.
usage: python3 conv/t1_replay.py --log <CONV_CACHE_LOG> [--port 8080] [--cap 64] [--out data/conv/t1_replay.json]"""
import argparse, json, os, re, time, urllib.request

REC = ("/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/"
       "sessions_real_1110.jsonl")
KEEP = ("anthropic-version", "anthropic-beta", "x-app", "user-agent", "openai-beta", "originator")
LINE = re.compile(r"convcache: (hit|miss) N (\d+) c (\d+) \(([^)]*)\) prefilled (\d+), restore ([\d.]+) ms, prefill "
                  r"([\d.]+) ms; entries (\d+), ([\d.]+) GB; edges \[([^\]]*)\](?:; check (.*))?$")


def capped(path, body, cap):
    b = dict(body)
    if path.endswith("/responses"):
        b["max_output_tokens"] = min(int(b.get("max_output_tokens") or cap), cap)
    else:
        for k in ("max_tokens", "max_completion_tokens"):
            if k in b or k == "max_tokens":
                b[k] = min(int(b.get(k) or cap), cap)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=REC)
    ap.add_argument("--log", required=True, help="the server's CONV_CACHE_LOG file")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--only", default="", help="replay only requests whose user-agent contains this")
    ap.add_argument("--out", default=os.path.expanduser("~/BigMoE/data/conv/t1_replay.json"))
    a = ap.parse_args()
    reqs = [r for r in (json.loads(l) for l in open(a.rec) if l.strip()) if r["kind"] == "request"]
    if a.only:
        reqs = [r for r in reqs if a.only in r["headers"].get("user-agent", "")]
    off = os.path.getsize(a.log) if os.path.exists(a.log) else 0
    rows = []
    for i, r in enumerate(reqs):
        body = capped(r["path"], r["body"], a.cap)
        hd = {k: v for k, v in r["headers"].items() if k.lower() in KEEP}
        hd["content-type"] = "application/json"
        q = urllib.request.Request(f"http://127.0.0.1:{a.port}{r['path']}", data=json.dumps(body).encode(), headers=hd,
                                   method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(q, timeout=3600) as resp:
                n = len(resp.read())
                st = resp.status
        except Exception as e:                             # an HTTP error still ran (or refused) the request: keep going
            n, st = 0, getattr(e, "code", str(e))
        rows.append(dict(i=i, rid=r.get("rid"), path=r["path"], ua=r["headers"].get("user-agent", "")[:24], status=st,
                         bytes=n, wall_s=round(time.time() - t0, 2)))
        print(json.dumps(rows[-1]), flush=True)
    time.sleep(1)
    with open(a.log) as f:
        f.seek(off)
        lines = [m.groups() for m in (LINE.search(l.rstrip("\n")) for l in f) if m]
    for row, g in zip(rows, lines):
        row.update(hit=g[0] == "hit", N=int(g[1]), c=int(g[2]), src=g[3], prefilled=int(g[4]), restore_ms=float(g[5]),
                   prefill_ms=float(g[6]), entries=int(g[7]), gb=float(g[8]), check=g[10] or "")
    checks = [r.get("check", "") for r in rows if r.get("hit")]
    R = dict(what="conversation-cache replay of Leap's recorded requests", rec=a.rec, cap=a.cap, requests=len(rows),
             cache_lines=len(lines), hits=sum(bool(r.get("hit")) for r in rows), checks=len([c for c in checks if c]),
             check_pass=sum(c == "PASS" for c in checks), check_fail=[r["i"] for r in rows if r.get("check", "").startswith("FAIL")],
             prefilled=sum(r.get("prefilled", 0) for r in rows), prompt_tokens=sum(r.get("N", 0) for r in rows), rows=rows)
    R["PASS"] = bool(R["checks"]) and not R["check_fail"] and len(lines) == len(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"{'PASS' if R['PASS'] else 'FAIL'}: {R['requests']} requests, {R['cache_lines']} cache lines, {R['hits']} hits, "
          f"checks {R['check_pass']}/{R['checks']} PASS (fail {R['check_fail']}), prefilled {R['prefilled']} of "
          f"{R['prompt_tokens']} prompt tokens -> {a.out}")


if __name__ == "__main__":
    main()
