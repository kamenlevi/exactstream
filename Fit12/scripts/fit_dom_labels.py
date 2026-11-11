"""Fit12 (Wed 4 Nov 22:4x): per-token domain labels for B10-r's parts R0..R9 (round-robin over math code instr multi wiki know
agent; chat items end at the answer's im_end + newline, instr items after "assistant\n", wiki 300 / agent 400 tokens), checked
== tokens_by_domain in every part; writes /Users/kamen/Fit12/data/b10rcap/domain_labels.npz. [BigMoE venv] python -B fit_dom_labels.py"""
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/BigMoE/models/bare"))
enc = lambda s: tok.encode(s, add_special_tokens=False)
IMS, IME = enc("<|im_start|>"), enc("<|im_end|>")
print("im_start", IMS, "im_end", IME, "end-of-answer", enc("x<|im_end|>\n")[1:], "instr tail", enc("<|im_end|>\n<|im_start|>assistant\n"))
TX = "/Users/kamen/Fit12/data/b10r/texts"
DOM = ("math", "code", "instr", "multi", "wiki", "know", "agent")
ANS_END = enc("x<|im_end|>\n")[1:]                    # im_end + "\n"
INS_END = enc("<|im_end|>\n<|im_start|>assistant\n")

def find(ids, pat, start):
    n = len(pat)
    for i in range(start, len(ids) - n + 1):
        if ids[i:i + n] == pat:
            return i
    return -1

def parse(ids):
    lab = np.full(len(ids), -1, np.int8); pos = 0; T = len(ids)
    while pos < T:
        for d, name in enumerate(DOM):
            if pos >= T:
                break
            if name == "wiki":
                end = pos + 300
            elif name == "agent":
                end = pos + 400
            else:
                assert ids[pos:pos + 1] == IMS, (name, pos, ids[pos:pos + 3])
                if name == "instr":
                    i = find(ids, INS_END, pos); end = T if i < 0 else i + len(INS_END)
                else:
                    i = find(ids, ANS_END, find(ids, IME, pos) + 1) if find(ids, IME, pos) >= 0 else -1
                    end = T if i < 0 else i + len(ANS_END)
            end = min(end, T); lab[pos:end] = d; pos = end
    return lab

out = {}
for p in [f"R{i}" for i in range(10)]:
    j = json.load(open(f"{TX}/{p}.json")); lab = parse(j["ids"])
    got = {n: int((lab == d).sum()) for d, n in enumerate(DOM)}
    ok = got == j["tokens_by_domain"]
    print(p, j["role"], "parse == tokens_by_domain:", ok, "" if ok else (got, j["tokens_by_domain"]))
    out[p] = lab
np.savez("/Users/kamen/Fit12/data/b10rcap/domain_labels.npz", **out)
print("saved labels")
