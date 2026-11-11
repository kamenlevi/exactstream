"""Prefix check on a serve.py request log (EXACTSTREAM_LOG_REQUESTS): for each session (conv key), each request's prompt
plus the model's answer against the next request's prompt. 'reuse' = tokens a cache holding the earlier prompt + answer
could keep; 'whole' = the earlier prompt + answer is a prefix of the next prompt (only the new turn needs a prefill).
Also prints each request's sys_end / lane / side and whether the system turn's tokens stay the same within a session.
Usage: prefix_check.py <log.jsonl> [tokenizer dir, to print the first differing text]"""
import json, sys
from collections import OrderedDict

recs = [json.loads(l) for l in open(sys.argv[1])]
tok = None
if len(sys.argv) > 2:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[2])
by = {}
for r in recs:
    by.setdefault(r.get("rid"), {})[r.get("kind")] = r
sess = OrderedDict()
for rid, d in by.items():
    if "render" in d:
        m = d["render"]["meta"]
        sess.setdefault((d["request"]["path"], m.get("conv")), []).append(d)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


tot = {"pairs": 0, "whole": 0}
for (path, conv), ds in sess.items():
    if conv is None:                                   # no session id: independent requests (e.g. a probe), not follow-ups
        print(f"no session id {path}: {len(ds)} independent requests, lanes {sorted(set(d['render']['meta']['lane'] for d in ds))}")
        continue
    print(f"session {path} conv {conv}: {len(ds)} requests")
    keys = set()
    for k, d in enumerate(ds):
        m, ids = d["render"]["meta"], d["render"]["ids"]
        out = (d.get("result") or {}).get("out_ids") or []
        keys.add(m["prefix_key"])
        line = f"  #{k} prompt {len(ids)} sys_end {m['sys_end']} lane {m['lane']} side {m['side']} out {len(out)}"
        if k + 1 < len(ds):
            nxt = ds[k + 1]["render"]["ids"]
            held = ids + out
            n = lcp(held, nxt)
            whole = n == len(held)
            tot["pairs"] += 1; tot["whole"] += whole
            line += f" | next prompt {len(nxt)}: reuse {n} of {len(held)} ({'whole' if whole else f'first difference at {n}: prompt {len(ids)} + answer {n - len(ids)}'})"
            if not whole and tok is not None:
                line += f"\n      held: {tok.decode(held[max(0, n - 8):n + 12])!r}\n      next: {tok.decode(nxt[max(0, n - 8):n + 12])!r}"
        print(line)
    print(f"  system turn tokens the same in all requests: {len(keys) == 1}")
print(f"PREFIX: {tot['whole']} of {tot['pairs']} follow-up requests start with the earlier prompt + answer")
