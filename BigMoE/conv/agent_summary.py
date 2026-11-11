#!/usr/bin/env python3
"""Cache numbers of one cached agent run (Leap's agentcc_*c_* runs), recomputed from its convcache.log and serve.log:
hits (live / copy), misses, prompt sizes, tokens prefilled, warm prefill and restore times, first-token times and the
lifetime peak (serve.log "done:" lines), the cache's size. Only numbers are written; no request content.
extra = a JSON object merged in (what, setup, the runner's reported agent results); "what" goes first.
usage: python3 -I conv/agent_summary.py <convcache.log> <serve.log> <out.json> '<extra JSON>'"""
import json, re, statistics as St, sys

LINE = re.compile(r"convcache: (hit|miss) N (\d+) c (\d+) \((.*?)\) prefilled (\d+), restore ([\d.]+) ms, prefill ([\d.]+) ms; "
                  r"entries (\d+), ([\d.]+) GB; edges \[([^\]]*)\](?:; check (.*))?$")
DONE = re.compile(r"done: (\d+) prompt tokens -> (\d+) tokens, whole decode ([\d.]+) tok/s, first token ([\d.]+) s, peak "
                  r"([\d.]+) GB")


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(round(p * (len(a) - 1))))]


def main():
    cl, sl, dst, extra = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    R, inst = [], None
    for l in open(cl):
        if "convcache: installed" in l:
            inst = l.split("convcache: ", 1)[1].strip()
        m = LINE.search(l.rstrip("\n"))
        if m:
            g = m.groups()
            R.append(dict(t=l[:8], hit=g[0] == "hit", N=int(g[1]), c=int(g[2]), src=g[3].split(",")[0], pf=int(g[4]),
                          rms=float(g[5]), pms=float(g[6]), ent=int(g[7]), gb=float(g[8])))
    D = [tuple(float(x) for x in m.groups()) for m in map(DONE.search, open(sl)) if m]
    h = [r for r in R if r["hit"]]
    cp = [r for r in h if r["src"] == "copy"]
    ft = [x[3] for x in D]
    S = dict(what=extra.pop("what", ""), convcache_installed=inst, **extra.pop("head", {}),
             requests=len(R), serve_done_lines=len(D), hits=len(h), live=len(h) - len(cp), copy=len(cp),
             misses=[dict(t=r["t"], N=r["N"], prefill_s=round(r["pms"] / 1e3, 2)) for r in R if not r["hit"]],
             prompt_tokens=dict(median=St.median(r["N"] for r in R), max=max(r["N"] for r in R), total=sum(r["N"] for r in R)),
             prefilled=dict(median=St.median(r["pf"] for r in R), p90=q([r["pf"] for r in R], .9), max=max(r["pf"] for r in R),
                            total=sum(r["pf"] for r in R),
                            share_pct=round(100 * sum(r["pf"] for r in R) / sum(r["N"] for r in R), 1)),
             copies=dict(c=sorted({r["c"] for r in cp}), prefilled_min=min((r["pf"] for r in cp), default=None),
                         prefilled_max=max((r["pf"] for r in cp), default=None)),
             hit_prefill_s=dict(median=round(St.median(r["pms"] for r in h) / 1e3, 2), max=round(max(r["pms"] for r in h) / 1e3, 2)),
             restore_ms=dict(median=St.median(r["rms"] for r in h), max=max(r["rms"] for r in h)),
             first_token_s=dict(median=round(St.median(ft), 2), p90=round(q(ft, .9), 2), max=round(max(ft), 2)),
             peak_gb=round(max(x[4] for x in D), 2), cache_entries_max=max(r["ent"] for r in R),
             cache_gb_max=max(r["gb"] for r in R), cache_at_end=dict(entries=R[-1]["ent"], gb=R[-1]["gb"]), **extra)
    json.dump(S, open(dst, "w"), indent=1)
    print(json.dumps({k: S[k] for k in ("requests", "serve_done_lines", "hits", "live", "copy", "misses", "prompt_tokens",
                                        "prefilled", "copies", "hit_prefill_s", "restore_ms", "first_token_s", "peak_gb",
                                        "cache_gb_max", "cache_at_end")}))


if __name__ == "__main__":
    main()
