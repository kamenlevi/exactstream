"""Strata head-to-head: the exact prompt token ids our engine used, as Strata --tokens-file inputs + a manifest.

Read-only sources (nothing is written outside ~/StrataEval/h2h):
  - keys:     S5  (BigMoE scripts/s5p.sh K / KM: 12 HumanEval + 12 MBPP; the order of data/r57/s5q1 / s5mq1 runs)
              N45 (BigMoE data/n45/keys.txt: the first 12 IFEval + 12 math of NAX6-N's AR file)
  - ids:      EvalSuite data/items/dec_<set>.jsonl "prompt_ids" (built once with the chat template; S5 and N45 feed them
              unchanged: neither config sets R57_NOCOM or any other prompt switch)
  - AR ref:   EvalSuite results/ref2tnax/decode.jsonl "completion_ids" (NAX6-N's one-token AR; no MBPP rows exist)
  - checks:   S5's data/r57/s5{q1,mq1}_256_runs.json "P" and N45's data/n45/ar_ref_tokens.json "n_prompt" against len(ids)
  - template: BigMoE models/bare/tokenizer.json, only to decode the first / last tokens for the manifest (information)
Usage: python3 -I make_prompts.py   (stdlib only)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

HOME = Path.home()
OUT = HOME / "StrataEval" / "h2h" / "prompts"
ITEMS = HOME / "EvalSuite" / "data" / "items"
ARFILE = HOME / "EvalSuite" / "results" / "ref2tnax" / "decode.jsonl"
S5SH = HOME / "BigMoE" / "scripts" / "s5p.sh"
N45KEYS = HOME / "BigMoE" / "data" / "n45" / "keys.txt"
N45AR = HOME / "BigMoE" / "data" / "n45" / "ar_ref_tokens.json"
S5RUNS = {"humaneval": HOME / "BigMoE" / "data" / "r57" / "s5q1_256_runs.json",
          "mbpp": HOME / "BigMoE" / "data" / "r57" / "s5mq1_256_runs.json"}
TOKJSON = HOME / "BigMoE" / "models" / "bare" / "tokenizer.json"
WARMUP_KEY = "gsm8k/708"          # S5 / N45's discarded warm-up prompt (R57_WARM), from dec_math


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def s5_keys() -> dict[str, list[str]]:
    """K / KM from scripts/s5p.sh (the K="..." and K="$K,..." lines, then KM="...")."""
    text = S5SH.read_text()
    k: list[str] = []
    for m in re.finditer(r'^K="(?:\$K,)?([^"]+)"', text, re.M):
        k += m.group(1).split(",")
    km = re.search(r'^KM="([^"]+)"', text, re.M).group(1).split(",")
    return {"humaneval": k, "mbpp": km}


def n45_keys() -> dict[str, list[str]]:
    lines = [l for l in N45KEYS.read_text().splitlines() if l.strip()]
    return {"ifeval": lines[0].split(","), "math": lines[1].split(",")}


def byte_decoder():
    """GPT-2 byte-level BPE: the unicode stand-in of each byte -> the byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def make_decoder():
    try:
        tj = json.loads(TOKJSON.read_text())
    except OSError:
        return None
    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    for a in tj.get("added_tokens", []):
        inv[a["id"]] = a["content"]
    bd = byte_decoder()

    def dec(ids):
        out = bytearray()
        for i in ids:
            t = inv.get(int(i), f"<?{i}>")
            if t.startswith("<|") or t.startswith("<think") or t.startswith("</think"):
                out += t.encode()
            else:
                out += bytes(bd.get(c, ord("?")) for c in t)
        return out.decode("utf-8", "replace")
    return dec


def main() -> int:
    sets = {**s5_keys(), **n45_keys()}
    for s, ks in sets.items():
        assert len(ks) == 12 and len(set(ks)) == 12, (s, len(ks))
    items = {}
    for s in ("humaneval", "mbpp", "ifeval", "math"):
        items[s] = {d["key"]: d for d in map(json.loads, open(ITEMS / f"dec_{s}.jsonl"))}
    ar = {d["key"]: d for d in map(json.loads, open(ARFILE))}
    s5p = {s: {r["key"]: r.get("P") for r in json.load(open(p))} for s, p in S5RUNS.items()}
    n45ar = json.load(open(N45AR))
    dec = make_decoder()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, arref, problems = [], {}, []

    def add(s, key, max_new, role):
        it = items[s][key]
        ids = [int(x) for x in it["prompt_ids"]]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
        f = OUT / s / f"{safe}.ids"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(",".join(map(str, ids)) + "\n")
        n = len(ids)
        checks = {}
        if s in s5p and key in s5p[s]:
            checks["s5_P"] = s5p[s][key]
            if s5p[s][key] != n:
                problems.append(f"{key}: S5 P {s5p[s][key]} != {n}")
        if key in n45ar:
            checks["n45_n_prompt"] = n45ar[key].get("n_prompt")
            if n45ar[key].get("n_prompt") not in (None, n):
                problems.append(f"{key}: N45 n_prompt {n45ar[key].get('n_prompt')} != {n}")
        if key in ar:
            checks["ar_n_prompt"] = ar[key].get("n_prompt")
            if ar[key].get("n_prompt") not in (None, n):
                problems.append(f"{key}: AR n_prompt {ar[key].get('n_prompt')} != {n}")
            if role == "measure":
                arref[key] = {"set": s, "completion_ids": ar[key]["completion_ids"], "finish": ar[key].get("finish")}
        row = {"set": s, "key": key, "role": role, "file": str(f.relative_to(OUT.parent)), "n_prompt": n,
               "max_new": max_new, "ids_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
               "ar_ref": key in arref, "checks": checks,
               "special_ids": {"im_start(248045)": ids.count(248045), "im_end(248046)": ids.count(248046)}}
        if dec is not None:
            row["head"] = dec(ids[:12])
            row["tail"] = dec(ids[-12:])
        manifest.append(row)

    for s in ("humaneval", "mbpp", "ifeval", "math"):
        for key in sets[s]:
            add(s, key, 256, "measure")
    add("math", WARMUP_KEY, 32, "warmup")
    src = {str(p): sha_file(p) for p in [S5SH, N45KEYS, N45AR, ARFILE, *S5RUNS.values()] +
           [ITEMS / f"dec_{s}.jsonl" for s in ("humaneval", "mbpp", "ifeval", "math")]}
    (OUT / "manifest.json").write_text(json.dumps({"prompts": manifest, "sources_sha256": src,
                                                   "sets": {s: sets[s] for s in ("humaneval", "mbpp", "ifeval", "math")},
                                                   "problems": problems}, indent=1))
    (OUT / "ar_ref.json").write_text(json.dumps({"source": str(ARFILE), "source_sha256": src[str(ARFILE)],
                                                 "note": "NAX6-N one-token AR (EvalSuite results/ref2tnax); no MBPP rows exist",
                                                 "refs": arref}, indent=1))
    by = {}
    for r in manifest:
        by.setdefault((r["set"], r["role"]), []).append(r["n_prompt"])
    for (s, role), ns in sorted(by.items()):
        print(f"{s:9s} {role:7s} {len(ns):2d} prompts, n_prompt {min(ns)}..{max(ns)} (sum {sum(ns)}), AR refs "
              f"{sum(1 for r in manifest if r['set'] == s and r['role'] == role and r['ar_ref'])}")
    print("problems:", problems or "none")
    if dec is not None:
        for s in ("humaneval", "mbpp", "ifeval", "math"):
            r = next(x for x in manifest if x["set"] == s)
            print(f"{s} {r['key']}: head {r['head']!r} ... tail {r['tail']!r}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
