#!/usr/bin/env python3
"""What convcache should do on Leap's recorded requests (agent/sessions_real_1110.jsonl, kind=render: the token ids and
the hints serve.py sends), by the module's own rules (conv/convcache.py: edges, lookup, take-over or copy), on the CPU:
per request, hit or miss, the restore point c, live or copy, tokens prefilled. T1's convcache.log is compared with this.
The byte budget is not simulated (the 32 GB default 2.5 GB holds these few entries).
usage: python3 -I conv/predict_hits.py [rec.jsonl] [out.json]"""
import json, os, sys

import numpy as np

REC = ("/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/"
       "sessions_real_1110.jsonl")
CHUNK, GRID = 1024, 8192


def im_start_id():
    tk = json.load(open(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json")))
    for t in tk.get("added_tokens", []):
        if t.get("content") == "<|im_start|>":
            return int(t["id"])
    return int(tk["model"]["vocab"]["<|im_start|>"])


def lcp(a, b):
    n = min(len(a), len(b))
    d = np.flatnonzero(a[:n] != b[:n])
    return int(d[0]) if len(d) else n


def edges(ids, h, c, IM):
    N = len(ids)
    end = N - 1
    pos = np.flatnonzero(ids == IM)
    if len(pos):
        end = min(end, int(pos[-1]))
    e_end = (end // CHUNK) * CHUNK
    out = {e_end}
    se = int(h.get("sys_end") or 0)
    if se >= CHUNK:
        out.add(min((se // CHUNK) * CHUNK, e_end))
    out.update(range(GRID, e_end, GRID))
    return sorted(e for e in out if c < e <= N - 1 and e >= CHUNK)


def main():
    rec = sys.argv[1] if len(sys.argv) > 1 else REC
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/BigMoE/data/conv/t1_predicted.json")
    IM = im_start_id()
    ents, rows = [], []
    for r in (json.loads(l) for l in open(rec) if l.strip()):
        if r["kind"] != "render":
            continue
        ids, m = np.asarray(r["ids"], np.int32), r["meta"]
        N, side = len(ids), bool(m.get("side"))
        lane, conv = m.get("lane") or ("side" if side else "main"), m.get("conv")
        best, c = None, 0
        if N > CHUNK:
            for e in ents:
                lim = min(lcp(e["ids"], ids) - 1, N - 1)
                ok = [k for k in e["ck"] if k <= lim]
                if ok and max(ok) > c:
                    best, c = e, max(ok)
        src = "none"
        src_conv0 = None if best is None else best.get("conv")
        src_idx0 = None if best is None else next(i for i, x in enumerate(ents) if x is best)
        if best is not None:
            same = (not side) and best["lane"] == lane and (conv is None or best["conv"] in (None, conv))
            copy = side or not (same or c >= max(best["ck"]))
            if copy and os.environ.get("CONV_CACHE_BRANCH", "1") == "0":   # the fallback: a branch is a miss
                best, c = None, 0
        if best is not None:
            kept = {k for k in best["ck"] if k <= c}
            if copy:
                cur = dict(ck=kept)
                ents.append(cur)
            else:
                cur = best
                cur["ck"] = kept
            src = "copy" if copy else "live"
        else:
            cur = dict(ck=set())
            ents.append(cur)
        cur["ck"] |= set(edges(ids, m, c, IM))
        cur.update(ids=ids, lane=lane, conv=conv)
        if not cur["ck"]:
            ents.remove(cur)
        rows.append(dict(rid=r["rid"], N=N, sys_end=m.get("sys_end"), lane=lane, conv=conv, hit=best is not None, c=c,
                         src_entry=src_idx0, src_conv=src_conv0,
                         src=src, prefilled=N - c, edges=sorted(cur["ck"])))
    R = dict(what="convcache's expected behaviour on the recorded renders (CPU, the module's rules)", im_start=IM,
             requests=len(rows), hits=sum(r["hit"] for r in rows), prompt_tokens=sum(r["N"] for r in rows),
             prefilled=sum(r["prefilled"] for r in rows), rows=rows)
    json.dump(R, open(dst, "w"), indent=1)
    for r in rows:
        if r["N"] > CHUNK:
            print(f"{r['rid'][-6:]} N {r['N']:>6} sys_end {r['sys_end']:>6} conv {str(r['conv'])[:6]:>6}: "
                  f"{'hit ' + r['src'] if r['hit'] else 'miss':9s} c {r['c']:>6} prefilled {r['prefilled']:>6} edges {r['edges']}")
    print(f"{R['requests']} requests, {R['hits']} hits; prefilled {R['prefilled']} of {R['prompt_tokens']} prompt tokens "
          f"({100 * R['prefilled'] / R['prompt_tokens']:.1f} %) -> {dst}")


if __name__ == "__main__":
    main()
