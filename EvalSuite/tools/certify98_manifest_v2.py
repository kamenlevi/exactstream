"""CERTIFY98's fresh sets after the lineage amendment (docs/PREREG_certify98_2026-11-15.md amendment 9, Wed 2026-11-18,
before the capture; the orchestrator, received 11:1x). CPU only. Reads the frozen v1 manifest (amendments 2 / 3) and writes
results/certify98/fresh_manifest_v2_2026-11-18.json, the manifest the capture's files_sha256.json names.
  .venv/bin/python tools/certify98_manifest_v2.py [--check]     # --check: rebuild and compare with the committed file
- The lineage: t30b starts from MTP Head's D-22 b5 (BigMoE scripts/dd_ttt.py loads ~/TrainLab/ckpt/d22_b5.safetensors).
  b5 trained with --data l120,d22c (~/TrainLab/ckpt/d22_b5.json): every train-split window of ~/Leap/data/l120 and
  ~/TrainLab/data/d22c that has a d16w file (TrainLab sticky/d22_ttt.py cmd_train), 246 windows, one pass. 72 v1 HE+
  problems (L1-20's he_<id>) and 57 v1 IFEval prompts (d22c_ifeval_<key>) are among them.
- HE+: all 140 are captured. The gated set is the 68 outside b5's training, and the strict-80 line is dropped (72 of its
  80 are b5's). The engine count check reads the first 24 of the 68, in manifest order.
- MBPP: the 300, unchanged (b5 has no MBPP; t30b's own TTT data is n1112 Magicoder only).
- Everyday (reported, never gated): agent 29 and IFBench 61 unchanged. IFEval keeps its 3 prompts outside b5's training,
  and 57 replacements are drawn by v1's method (seed [SEED, 2]) from IFEval's other 441 (data/items/dec_ifevalx.jsonl)
  minus the 30 tf_ifeval prompts (data/items/tf_prompts.jsonl; d16w, d02a and D-22's read-outs hold them). The v1 IFEval
  57 are still captured, as information."""
import hashlib, json, os, re, sys
import numpy as np

SEED, N_IFX = 20261115, 57
H = os.path.expanduser("~")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V1 = os.path.join(ROOT, "results", "certify98", "fresh_manifest_2026-11-15.json")
V1_SHA = "acdf3e62d4fd8f74aed520c1855a61415c4632269cfcad2e77f69f498a82720c"
OUT = os.path.join(ROOT, "results", "certify98", "fresh_manifest_v2_2026-11-18.json")
B5 = f"{H}/TrainLab/ckpt/d22_b5.json"
FOLD = {"l120": f"{H}/Leap/data/l120", "d22c": f"{H}/TrainLab/data/d22c"}
D16W = f"{H}/TrainLab/data/d16w"
# folders that hold IFEval windows by name (a scan of every drafter data tree, Wed 11:2x: only dec_ifeval's 100 and
# tf_ifeval's 30 appear); no replacement may have a file there
IFEVAL_DIRS = [f"{H}/TrainLab/data/{d}" for d in ("d16w", "d22c", "d02a", "tree3_src/ifeval", "tree3_map/ifeval",
                                                    "d22r_b3", "d22r_b4")] + [f"{H}/Chat1000/data/drafter/r25_tfgen4"]


def b5_windows():
    """b5's training windows, listed as d22_ttt.py cmd_train lists them for its --data"""
    S = json.load(open(B5))
    assert S["args"]["data"] == "l120,d22c" and S["args"]["epochs"] == 1.0 and S["status"] == "done", S["args"]
    keys = []
    for nm in S["args"]["data"].split(","):
        for f in sorted(os.listdir(FOLD[nm])):
            if f.endswith(".npz") and str(np.load(os.path.join(FOLD[nm], f))["split"]) == "train" \
                    and os.path.exists(os.path.join(D16W, f)):
                keys.append(f"{nm}/{f[:-4]}")
    assert len(keys) == int(S["step"]) == 246, (len(keys), S["step"])   # one epoch: one step per window
    return keys


def build():
    v1txt = open(V1, "rb").read()
    assert hashlib.sha256(v1txt).hexdigest() == V1_SHA, "the v1 manifest moved"
    v1 = json.loads(v1txt)
    win = b5_windows()
    he_b5 = sorted(int(m.group(1)) for k in win for m in [re.fullmatch(r"l120/he_(\d+)", k)] if m)
    ife_b5 = sorted(f"ifeval/{m.group(1)}" for k in win for m in [re.fullmatch(r"d22c/d22c_ifeval_(\d+)", k)] if m)
    he = v1["heplus_fresh"]
    he72 = [i for i in he if i in set(he_b5)]
    he68 = [i for i in he if i not in set(he_b5)]
    ev1 = v1["everyday_fresh"]
    ife57 = [k for k in ev1["ifeval"] if k in set(ife_b5)]
    ife3 = [k for k in ev1["ifeval"] if k not in set(ife_b5)]
    keys = lambda f: [json.loads(l)["key"] for l in open(os.path.join(ROOT, "data", "items", f"{f}.jsonl"))]
    x = sorted(keys("dec_ifevalx"))
    tf = {k.split("/")[1] for k in keys("tf_prompts") if k.startswith("tf_ifeval/")}
    pool = [k for k in x if k.split("/")[1] not in tf]
    assert len(x) == 441 and len(tf) == 30 and len(pool) == 411, (len(x), len(tf), len(pool))
    ifx = sorted(np.array(pool)[np.random.default_rng([SEED, 2]).permutation(len(pool))[:N_IFX]].tolist())
    held = [k for k in ifx for d in IFEVAL_DIRS if os.path.isdir(d)
            for f in os.listdir(d) if re.fullmatch(rf".*ifeval_{k.split('/')[1]}\.npz", f)]
    assert not held, f"a replacement has a drafter-data file: {held[:3]}"
    assert len(he72) == 72 and len(he68) == 68 and len(ife57) == 57 and len(ife3) == 3, \
        (len(he72), len(he68), len(ife57), len(ife3))
    return {
        "rule": "docs/PREREG_certify98_2026-11-15.md amendment 9 (the lineage amendment, Wed 2026-11-18, before the capture)",
        "seed": SEED,
        "v1": {"path": "results/certify98/fresh_manifest_2026-11-15.json", "sha256": V1_SHA},
        "lineage": {"t30b_start": "b5 (BigMoE scripts/dd_ttt.py loads ~/TrainLab/ckpt/d22_b5.safetensors)",
                    "b5": "~/TrainLab/ckpt/d22_b5.json: --data l120,d22c, the train split, one epoch",
                    "b5_windows": win,
                    "contamination_reference": "n1112 (all 3,072 captures) + b5's 246 windows"},
        "heplus_capture": he,
        "heplus_lineage_b5": he72,
        "heplus_gated": he68,
        "heplus_count_check": he68[:24],
        "mbpp_fresh": v1["mbpp_fresh"],
        "everyday_capture": {"agent": ev1["agent"], "ifbench": ev1["ifbench"], "ifeval": ev1["ifeval"], "ifevalx": ifx},
        "everyday_reported": {"agent": ev1["agent"], "ifbench": ev1["ifbench"], "ifeval": ife3, "ifevalx": ifx},
        "everyday_lineage_b5": {"ifeval": ife57},
        "everyday_sources": {"ifevalx": f"data/items/dec_ifevalx.jsonl (IFEval's other 441) minus tf_prompts' 30 "
                                        f"tf_ifeval = 411; seed [SEED, 2], the first {N_IFX}, sorted"},
        "dropped": {"heplus_fresh_strict": "the strict-80 line (72 of its 80 are in b5's training)"},
    }


def main(argv):
    m = build()
    txt = json.dumps(m, indent=1, sort_keys=True) + "\n"
    if "--check" in argv:
        same = os.path.exists(OUT) and open(OUT).read() == txt
        print(f"manifest v2 rebuild {'matches' if same else 'DIFFERS FROM'} {OUT}")
        return 0 if same else 1
    open(OUT, "w").write(txt)
    print(f"b5's windows {len(m['lineage']['b5_windows'])}; HE+ capture {len(m['heplus_capture'])}, lineage "
          f"{len(m['heplus_lineage_b5'])}, gated {len(m['heplus_gated'])}, count check {m['heplus_count_check']}; MBPP "
          f"{len(m['mbpp_fresh'])}; everyday capture {sum(len(v) for v in m['everyday_capture'].values())}, reported "
          f"{sum(len(v) for v in m['everyday_reported'].values())}; written {os.path.normpath(OUT)}, sha256 "
          f"{hashlib.sha256(txt.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
