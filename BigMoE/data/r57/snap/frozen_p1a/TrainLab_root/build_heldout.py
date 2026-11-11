"""TrainLab extra held-out segments (1365 + 1 tokens each, frozen ids): multilingual (bg, fr, de from the Eval suite's
raw documents, middle windows) and chat (a account-1 Claude Code transcript; training uses only the main account's).
Writes data/heldout_extra_ids.npz. The exact reference for them is computed on the GPU by train35.py (ref_extra.npz)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/TrainLab_root")); sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts"))
import build_data as B, defer35 as D
from mlx_lm.utils import load_tokenizer
from pathlib import Path
N = 1366
tok = load_tokenizer(Path(D.MODEL))
out = {}
raw = os.path.expanduser("~/EvalSuite/data/raw")
for lang, f in [("multi_bg", "doc_bg_wikipedia.txt"), ("multi_fr", "doc_fr_monte_cristo.txt"), ("multi_de", "doc_de_der_prozess.txt")]:
    ids = tok.encode(open(os.path.join(raw, f)).read())
    mid = len(ids) // 2
    out[lang] = np.array(ids[mid:mid + N], np.int32)
B.H = os.path.expanduser("~")
fs = sorted(glob.glob(os.path.expanduser("~/.claude-account-1/projects/*/*.jsonl")), key=os.path.getsize)
orig = B.glob.glob
B.glob.glob = lambda p, **k: [fs[len(fs) // 2]] if p.endswith("projects/*/*.jsonl") else orig(p, **k)
docs = [d for d in B.chat_docs()]
ids = tok.encode("\n\n".join(docs))
out["chat"] = np.array(ids[len(ids) // 3: len(ids) // 3 + N], np.int32)
for k, v in out.items():
    assert v.size == N, (k, v.size)
    print(k, v.size, repr(tok.decode(v[:40].tolist()))[:120])
np.savez(os.path.expanduser("~/TrainLab/data/heldout_extra_ids.npz"), **out)
