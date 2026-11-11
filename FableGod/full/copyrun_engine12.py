"""copyrun_engine12.py (Fable FULL, LOSSY350): the answer shape on the benchmark's own window.
(1) The walk's 24 HumanEval+ answers: the verbatim prompt-copy share within the first 256 generated tokens (the engine's
    CAP 256) and where the run starts.
(2) EvalSuite's stored REF2-T answers (results/ref2t/decode.jsonl, 60 HumanEval+ rows): the share of docstring / comment /
    code / prose / string tokens by relax_core.CodeScan, for all rows and for the engine's 12 prompts, full and first 256.
The npz is the project's own recorded walk (allow_pickle only for its names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I copyrun_engine12.py > copyrun_engine12.out"""
import importlib.util, json
import numpy as np
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
ENG = ["68", "16", "91", "6", "79", "19", "161", "17", "56", "141", "101", "63"]   # BigMoE scripts/allin_wed.sh
W = np.load("/Users/kamen/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
tot = cop = 0; starts = []
for wi in range(len(W["names"])):
    ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]
    g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:][:256]
    pos = {}
    for i, t in enumerate(P): pos.setdefault(t, []).append(i)
    q = 0; first = None
    while q < len(A):
        best = 0
        for i in pos.get(A[q], ()):
            L = 0
            while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
            best = max(best, L)
        if best >= 8:
            first = q if first is None else first; cop += best; q += best
        else: q += 1
    tot += len(A); starts.append(first)
print(f"(1) walk, 24 answers, first 256 generated tokens: {tot} tokens, verbatim prompt copy {100 * cop / tot:.1f} %; the run starts at token {sorted(set(starts))}")
rows = [json.loads(l) for l in open("/Users/kamen/EvalSuite/results/ref2t/decode.jsonl")]
he = [r for r in rows if "umaneval" in r["key"].lower()]
def show(label, rs, cap):
    cnt = {"docstring": 0, "comment": 0, "code": 0, "prose": 0, "string": 0}; n = 0
    for r in rs:
        ids = r["completion_ids"][:cap] if cap else r["completion_ids"]; sc = rc.CodeScan(); n += len(ids)
        for i in ids:
            s = TOK.decode([int(i)]); bc = bool(sc.code and sc.com)
            k0 = "docstring" if (sc.code and sc.doc) else ("string" if (sc.code and (sc.str_ or sc.qn)) else ("code" if sc.code else "prose")); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            cnt["comment" if (anyc or bc) else k0] += 1
    print(f"(2) {label}: {len(rs)} answers, {n} tokens: " + ", ".join(f"{k} {100 * v / max(1, n):.1f} %" for k, v in cnt.items()))
show("all HumanEval+ rows, full answers", he, None); show("all HumanEval+ rows, first 256 tokens", he, 256)
eng = [r for r in he if r["key"].split("/")[-1] in ENG]
print("    the engine's 12 found:", len(eng))
show("the engine's 12, full answers", eng, None); show("the engine's 12, first 256 tokens", eng, 256)
print("    sample, HumanEval/68 starts:", repr(TOK.decode([r for r in eng if r["key"].endswith("/68")][0]["completion_ids"][:40])))
