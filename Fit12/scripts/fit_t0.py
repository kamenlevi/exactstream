"""Fit12 T0 gates (Mon 2 Nov 2026; PLAN.md 'T0', written before any run): the two-pass decode of t0_twopass.py on the live
12 GB default (pinned bt_b12 8db71cc537, the suite's process env as fit_creep.py's F12_ARM=live, load_b12(0.10) without the
MTP head: T0 has no drafter). Per prompt, in one process, arms in this order, each from its own exact prefill:
  ex   the exact reference: t0_twopass.generate_exact (B12's one-row AR decode under ALLWAIT, no pass 1)
  g1   theta 1.01: every row runs pass 1, is rolled back and re-run exactly (gate G1: = ex bit for bit)
  g0   theta 0: every row from pass 1 (ranks 0-1 only)
  t80  theta 0.8 (code's setting), t95 theta 0.95 (chat's), t80b = t80 again (determinism)
Per arm: tokens (sha1 of the id list, first divergence from ex), lsha (sha1 of every step's chosen logits, fp32), pass-2
rate, ALLWAIT layer calls (B.AW_N), the loader's defer counters (rows, waited) per arm. Usage: [BigMoE venv] fit_t0.py TAG"""
import hashlib, json, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",  # SHIP_AR
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",              # B12_ENV b12ll
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",                               # FOLD_SHIP
       "BIGMOE_LOOP_GUARD": "0", "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"}
for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_CM", "B12_CL", "B12_FOLD_CM", "B12_FOLD_AW", "B12_W", "B12_PMIN"):
    os.environ.pop(k, None)                          # the live file's own mode defaults (fast mode), as fit_creep's F12_ARM=live
os.environ.update(ENV)
TAG = sys.argv[1] if len(sys.argv) > 1 else "t0"
OUTD = os.environ.get("F12_OUT", os.path.expanduser("~/Fit12/data/t0"))
CAP = int(os.environ.get("F12_CAP", "96"))
ES = os.path.expanduser("~/EvalSuite")
BT = os.environ.get("F12_BT", os.path.expanduser("~/Fit12/pinned/8db7/bt_b12.py"))
EOS = (248044, 248046)
sha = hashlib.sha256(open(BT, "rb").read()).hexdigest()[:10]
assert sha == "8db71cc537", f"bt_b12.py is {sha}, T0 is written against 8db71cc537"
REF = [json.loads(l) for l in open(os.path.join(ES, "results", "b12llff4", "decode.jsonl"))]
WANT = os.environ.get("F12_KEYS", "humaneval/HumanEval/156,ifeval/,math500/test/number_theory/459.json").split(",")
SEL = [next(r for r in REF if r["key"].startswith(w) if w.endswith("/") or r["key"] == w) for w in WANT]
ITEMS = {}
for st in {r["set"] for r in SEL}:
    ITEMS.update({it["key"]: it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{st}.jsonl")))})
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
import bt_b12 as B
import t0_twopass as T
assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
T.apply(B)                                           # before load_b12: the block is built there
rt, _ = B.load_b12(0.10, mtp=False)
model = rt.model
from bigmoe_v3f.gated import ext
os.makedirs(OUTD, exist_ok=True)
print(f"[{TAG}] bt_b12 {sha}, engine {getattr(B, 'V3F', '?')}, C_m {B.CFG.get('cm')} C_l {B.CFG.get('cl')}, cap {CAP}; prompts "
      + ", ".join(f"{r['key']} (n {r['n_prompt']})" for r in SEL), flush=True)
ARMS = [("ex", None), ("g1", 1.01), ("g0", 0.0), ("t80", 0.8), ("t95", 0.95), ("t80b", 0.8)]
for r in SEL:
    ids = ITEMS[r["key"]]["prompt_ids"]
    res = {}
    for arm, th in ARMS:
        aw0, d0 = B.AW_N[0], ext.defer_stats()
        tr = []
        if th is None:
            toks, fin, s = T.generate_exact(B, model, ids, CAP, stop_ids=EOS)
        else:
            toks, fin, s = T.generate(B, model, ids, CAP, th, stop_ids=EOS, trace=tr)
        d1 = ext.defer_stats()
        res[arm] = (toks, s)
        ex = res["ex"][0]
        div = next((i for i, (a, b) in enumerate(zip(toks, ex)) if a != b), None if len(toks) == len(ex) else min(len(toks), len(ex)))
        rows, waited = d1[0] - d0[0], d1[1] - d0[1]
        print(f"[{TAG}] {r['key']} {arm} theta {th} | tok {len(toks)} {fin} sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} "
              f"| lsha {s['lsha']} | = ex tokens {toks == ex} (first divergence {div}) | pass2 {s['pass2']} of {s['steps']} "
              f"({100 * s['rate']:.1f} %) | AW calls {B.AW_N[0] - aw0} | loader rows {rows} waited {waited} "
              f"({waited / max(1, rows):.2f} per row-layer) | prefill {s['pf_s']:.2f} s, decode {s['dec_s']:.2f} s "
              f"({len(toks) / max(1e-9, s['dec_s']):.2f} tok/s)", flush=True)
        if tr:
            json.dump(tr, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}.json"), "w"))
    g1 = res["g1"][0] == res["ex"][0] and res["g1"][1]["lsha"] == res["ex"][1]["lsha"]
    det = res["t80"][0] == res["t80b"][0] and res["t80"][1]["lsha"] == res["t80b"][1]["lsha"]
    print(f"[{TAG}] {r['key']} GATES | G1 (theta 1.01 = exact, tokens + logits): {'PASS' if g1 else 'FAIL'} | determinism "
          f"(t80 = t80b): {'PASS' if det else 'FAIL'} | g0 pass-1 logits differ from ex: {res['g0'][1]['lsha'] != res['ex'][1]['lsha']}",
          flush=True)
print(f"[{TAG}] done", flush=True)
