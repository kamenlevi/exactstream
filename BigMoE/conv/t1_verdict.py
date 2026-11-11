#!/usr/bin/env python3
"""T1 verdict from Leap's run (agent/t1_runs/<run>/: convcache.log, requests.jsonl): one convcache line per recorded
request, in order; every hit must say "check PASS" (CONV_CACHE_CHECK=1: a cold prefill before it, same process); the hits
are compared with data/conv/t1_predicted.json (the module's rules on the recorded ids). First-token times (stats.pfs)
are reported, but with the check on a warm request's includes its cold check (check_ms) and runs with that check's experts
resident, so they are info only; the warm-only prefill is convcache's prefill_ms. Per-request stats come from serve.log's
"done:" lines (Leap's runner; requests.jsonl holds only the bodies there); only numbers are written, no request content.
A miss's reason may hold parentheses ("none, branch (CONV_CACHE_BRANCH=0)"). The prediction file: the third argument or
T1_PRED (T1d, CONV_CACHE_BRANCH=0: data/conv/t1_predicted_branch0.json).
usage: python3 -I conv/t1_verdict.py <run dir> [out.json] [predicted.json]"""
import json, os, re, sys

LINE = re.compile(r"convcache: (hit|miss) N (\d+) c (\d+) \((.*?)\) prefilled (\d+), restore ([\d.]+) ms, prefill "
                  r"([\d.]+) ms; entries (\d+), ([\d.]+) GB; edges \[([^\]]*)\](?:; check (.*))?$")
DONE = re.compile(r"done: (\d+) prompt tokens -> (\d+) tokens, whole decode ([\d.]+) tok/s, first token ([\d.]+) s, peak "
                  r"([\d.]+) GB")


def main():
    d = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/BigMoE/data/conv/t1/verdict.json")
    pp = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("T1_PRED", "~/BigMoE/data/conv/t1_predicted.json")
    pred = json.load(open(os.path.expanduser(pp)))["rows"]
    lines = []
    for l in open(os.path.join(d, "convcache.log")):
        m = LINE.search(l.rstrip("\n"))
        if m:
            g = m.groups()
            lines.append(dict(hit=g[0] == "hit", N=int(g[1]), c=int(g[2]), src=g[3].split(",")[0], prefilled=int(g[4]),
                              restore_ms=float(g[5]), prefill_ms=float(g[6]), entries=int(g[7]), gb=float(g[8]),
                              edges=g[9], check=(g[10] or "").strip()))
    res = []
    p = os.path.join(d, "requests.jsonl")
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            st = r.get("stats") or (r.get("result") or {}).get("stats")
            if isinstance(st, dict):
                res.append(st)
    dn = []                                                # serve.log: the server's own per-request lines
    p = os.path.join(d, "serve.log")
    if os.path.exists(p):
        for l in open(p):
            m = DONE.search(l)
            if m:
                g = m.groups()
                dn.append(dict(N=int(g[0]), n=int(g[1]), tps=float(g[2]), pfs=float(g[3]), peak_gb=float(g[4])))
    rows, bad, off = [], [], []
    for i, (x, q) in enumerate(zip(lines, pred)):
        st = res[i] if i < len(res) else {}
        cc = st.get("cache") if isinstance(st.get("cache"), dict) else {}
        dd = dn[i] if i < len(dn) and dn[i]["N"] == x["N"] else {}
        same = (x["hit"], x["c"], x["N"]) == (q["hit"], q["c"], q["N"]) and (not x["hit"] or x["src"] == q["src"])
        if x["hit"] and x["check"] != "PASS":
            bad.append(i)
        if not same:
            off.append(i)
        rows.append(dict(i=i, N=x["N"], hit=x["hit"], c=x["c"], src=x["src"], prefilled=x["prefilled"], check=x["check"],
                         predicted=dict(hit=q["hit"], c=q["c"], src=q["src"], N=q["N"]), as_predicted=same,
                         prefill_ms=x["prefill_ms"], restore_ms=x["restore_ms"], check_ms=cc.get("check_ms"),
                         pfs=st.get("pfs", dd.get("pfs")), n=st.get("n", dd.get("n")), tps=st.get("tps", dd.get("tps")),
                         peak_gb=dd.get("peak_gb"), gb=x["gb"],
                         entries=x["entries"]))
    hits = [r for r in rows if r["hit"]]
    R = dict(run=os.path.basename(os.path.normpath(d)), predicted_file=os.path.basename(pp), lines=len(lines), predicted_requests=len(pred), hits=len(hits),
             predicted_hits=sum(q["hit"] for q in pred), check_pass=sum(r["check"] == "PASS" for r in hits),
             check_fail=bad, not_as_predicted=off, rows=rows,
             prefilled=sum(r["prefilled"] for r in rows), prompt_tokens=sum(r["N"] for r in rows),
             max_gb=max((r["gb"] for r in rows), default=0))
    R["PASS"] = len(lines) == len(pred) and not bad and bool(hits)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(R, open(dst, "w"), indent=1)
    for r in rows:
        if r["N"] > 1024 or not r["as_predicted"]:
            print(f"{r['i']:>2} N {r['N']:>6} {'hit ' + r['src'] if r['hit'] else 'miss':9s} c {r['c']:>6} prefilled "
                  f"{r['prefilled']:>6} check {r['check'] or '-':5s} prefill {r['prefill_ms']:>8.1f} ms  pfs {r['pfs']}  "
                  f"{'' if r['as_predicted'] else 'NOT AS PREDICTED ' + str(r['predicted'])}")
    print(f"{'PASS' if R['PASS'] else 'FAIL'}: {R['lines']} cache lines for {R['predicted_requests']} requests; hits "
          f"{R['hits']} (predicted {R['predicted_hits']}); checks {R['check_pass']}/{R['hits']} PASS, fail {bad}; not as "
          f"predicted {off}; prefilled {R['prefilled']} of {R['prompt_tokens']} tokens; max cache {R['max_gb']} GB -> {dst}")


if __name__ == "__main__":
    main()
