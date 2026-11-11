"""PMIN12 A/B (Breakthrough Sun 1 Nov 11:58; lines in SCOREBOARD 'PMIN12 A/B LINES', written before the run; shared GPU,
rough): spec125's p_min (stop the MTP chain where the head's running probability < p_min) on the 12 GB line as it ships
(bt_b12's own defaults = fold-fast, stock MTP head; spec125 3:3:0 with the suite's prefill, as r60_screen.py runs it), one
process. Per prompt: an untimed warm-up (p_min 0, 48 tokens), then arms p_min 0 / 0.3 / 0.5 / 0. Per run: tokens (sha, first
difference vs the first p_min 0 run), passes, rows verified (passes + drafted), decode s, lifetime peak.
Options (env): PM_MODE=exact (B12_MODE=exact: the token check), PM_KEYS=k1,k2 (a subset of the prompts), PM_ARMS=0,0.5,...
(p_min per run, in order; the first must be 0), PM_TAG (output suffix), PM_CAP (new tokens; default 192, agent 128).
Usage: [BigMoE venv] BT_DIR=<pinned bt_b12 dir> pmin12_ab.py -> data/pmin12/ab<TAG>.jsonl + stdout"""
import ctypes, hashlib, json, os, sys, time
import numpy as np

for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_CM", "B12_PFR_MIN", "B12_MODE", "B12_M01",
          "BIGMOE_M01", "B12_FRS_NOCYCLE", "MTPLX_NGRAM_HOT_MB", "B12_MTP_HEAD", "B12_MTP_HEAD_BITS", "EVALSUITE_FRUGAL",
          "EVALSUITE_FAST24", "B12_W", "B12_CM", "B12_CL", "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR"):
    os.environ.pop(k, None)                          # the live file's own defaults (fold-fast), as the suite's live process
os.environ.update({"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0",   # r60_screen's
                   "BIGMOE_PF_GROUP": "64", "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1",
                   "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
                   "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1", "MTPLX_DEFER_REPAIR_EVAL": "1",
                   "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1", "B12_TRIMS": "1",
                   "B12_PFR": "1", "B12_ALLWAIT": "0", "EVALSUITE_SHIP": "1", "MTPLX_NGRAM_PREWARM": "off",
                   "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0", "MTPLX_COMPILE_AR_FORWARD": "0",
                   "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"})
if os.environ.get("PM_MODE") == "exact":
    os.environ["B12_MODE"] = "exact"
ARMS = tuple(float(x) for x in os.environ.get("PM_ARMS", "0,0.3,0.5,0").split(","))
assert ARMS[0] == 0.0, ARMS
TAG = os.environ.get("PM_TAG", "")
OUTD = os.path.expanduser("~/Breakthrough/data/pmin12")
os.makedirs(OUTD, exist_ok=True)


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def ru():                                            # (footprint GB, lifetime peak GB)
    r = _RU()
    assert _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) == 0
    return r.u7 / 1e9, r.u28 / 1e9


import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__))
BTD = os.environ.get("BT_DIR") or HERE
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, BTD)
BTSHA = hashlib.sha256(open(os.path.join(BTD, "bt_b12.py"), "rb").read()).hexdigest()[:10]
import bt_b12 as B
assert os.path.dirname(os.path.abspath(B.__file__)) == os.path.abspath(BTD), (B.__file__, BTD)
t0 = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
t_load = time.perf_counter() - t0
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model = rt.model
assert "mtp_head" not in B.CFG, B.CFG.get("mtp_head")
print(f"[pmin12{TAG}] arms {ARMS} | bt_b12 {BTSHA} engine {B.V3F} mode {B.MODE} | stock MTP head | load {t_load:.1f} s | footprint "
      f"{ru()[0]:.3f} GB, lifetime {ru()[1]:.3f}", flush=True)
EOS = {248044, 248046}
ES = os.path.expanduser("~/EvalSuite/data/items/")
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
PICK = [("chat", "ifeval", "ifeval/3091"), ("chat", "ifeval", "ifeval/1051"), ("chat", "multi", "multi/zh15"),
        ("chat", "multi", "multi/ja18"), ("chat", "ifeval", "ifeval/3223"), ("chat", "ifeval", "ifeval/2467"),
        ("chat", "multi", "multi/ja20"), ("chat", "multi", "multi/zh16"), ("code", "humaneval", "humaneval/HumanEval/45"),
        ("code", "humaneval", "humaneval/HumanEval/27"), ("math", "math", "math500/test/prealgebra/1973.json"),
        ("math", "math", "gsm8k/708"), ("agent", "agent", "agent/edit10")]
if os.environ.get("PM_KEYS"):
    PICK = [p for p in PICK if p[2] in os.environ["PM_KEYS"].split(",")]
ITEMS = {}
for s in {p[1] for p in PICK}:
    ITEMS.update({it["key"]: it for it in map(json.loads, open(ES + f"dec_{s}.jsonl"))})


def run(ids, cap, p_min):
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    state = B.b12_prefill_state(model, ids, 512, 4096)
    mx.eval(state["logits"])
    td = time.perf_counter()
    out, s = spec125.generate(model, ids, cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                              stats={}, stop_ids=EOS, p_min=p_min)
    td = time.perf_counter() - td
    out = [int(x) for x in out]
    for j, x in enumerate(out):
        if x in EOS:
            out = out[:j + 1]
            break
    del state
    mx.clear_cache()
    return out, s, td


with open(os.path.join(OUTD, f"ab{TAG}.jsonl"), "w") as f:
    pass
for typ, s_, key in PICK:
    ids = list(ITEMS[key]["prompt_ids"])
    cap = int(os.environ.get("PM_CAP", "0")) or (128 if typ == "agent" else 192)
    run(ids, 48, 0.0)                                # warm-up: this prompt's experts into the pools, untimed
    base = None
    for i, p in enumerate(ARMS):
        out, s, td = run(ids, cap, p)
        base = out if base is None else base
        diff = next((j for j, (a, b) in enumerate(zip(out, base)) if a != b), None if len(out) == len(base) else min(len(out), len(base)))
        row = dict(type=typ, key=key, run=i, p_min=p, n_prompt=len(ids), ntok=len(out), sha=sha(out), passes=int(s.get("passes", 0)),
                   rows=int(s.get("passes", 0)) + int(s.get("drafted", 0)), accepted=int(s.get("accepted", 0)),
                   decode_s=round(td, 3), tok_s=round(len(out) / td, 3), first_diff=diff, lifetime=round(ru()[1], 3), bt_b12=BTSHA)
        with open(os.path.join(OUTD, f"ab{TAG}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[pmin12] {key} p_min {p} RESULT {len(out)} tok, {row['passes']} passes, {row['rows'] / max(1, row['passes']):.2f} "
              f"rows / pass, {len(out) / max(1, row['passes']):.3f} tok / pass | decode {td:.2f} s = {row['tok_s']:.2f} tok/s | "
              f"first diff vs run 0: {diff} | lifetime {row['lifetime']:.3f} GB", flush=True)
# summary: per type, tok/s = sum tokens / sum decode s; baseline = both p_min 0 runs
R = [json.loads(l) for l in open(os.path.join(OUTD, f"ab{TAG}.jsonl"))]
for typ in ("chat", "code", "math", "agent", "all"):
    rr = [r for r in R if typ in ("all", r["type"])]
    v = {}
    for p in sorted(set(ARMS)) if rr else ():
        x = [r for r in rr if r["p_min"] == p]
        v[p] = sum(r["ntok"] for r in x) / sum(r["decode_s"] for r in x)
    eq = {p: sum(r["first_diff"] is None for r in rr if r["p_min"] == p and r["run"] > 0) for p in sorted(set(ARMS))}
    if not rr:
        continue
    print(f"[pmin12] SUMMARY {typ}: " + "; ".join(f"p_min {p} {v[p]:.2f} tok/s ({100 * (v[p] / v[0.0] - 1):+.1f} %)" for p in v)
          + f" | runs equal to run 0: {eq}", flush=True)
# robust to a stalled run: per prompt the median tok/s of each arm's runs, then per type the geometric mean of the ratios
for typ in ("chat", "code", "math", "agent", "all"):
    ks = sorted({r["key"] for r in R if typ in ("all", r["type"])})
    if not ks:
        continue
    med = {k: {p: float(np.median([r["tok_s"] for r in R if r["key"] == k and r["p_min"] == p])) for p in sorted(set(ARMS))} for k in ks}
    gm = {p: float(np.exp(np.mean([np.log(med[k][p] / med[k][0.0]) for k in ks]))) for p in sorted(set(ARMS))}
    print(f"[pmin12] MEDIANS {typ} ({len(ks)} prompts): " + "; ".join(f"p_min {p} {100 * (gm[p] - 1):+.1f} %" for p in gm), flush=True)
print(f"[pmin12] done, lifetime {ru()[1]:.3f} GB", flush=True)
