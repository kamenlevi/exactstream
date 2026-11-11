"""Fit12 T2 gates (Mon 2 Nov 2026; PLAN.md 'T2', written before any run): the two-pass MTP check of t2_twopass.py on the
live 12 GB default (pinned bt_b12 8db71cc537, the suite's MTP process env as fit_creep.py's F12_ARM=live, load_b12(0.10,
mtp=True)). Per prompt, in one process, each arm from its own exact prefill (b12_prefill_state, 512 / 4096):
  ref  spec125.generate (the fold-wrapped live decode: k 3, max_draft 3, lookup 0) = the shipped exact decode
  ex   t2_twopass.generate(theta None): my loop without pass 1 (must equal ref's tokens: the loop mirrors spec125)
  g1   theta 1.01: every check runs pass 1, rolls back at row 0 and runs the exact check (gate G1: = ex bit for bit)
  g0   theta 0: no pass 2;  t80 / t95: theta 0.8 / 0.95;  t80b: t80 again (determinism)
Per arm: tokens (sha1, = ref?, first divergence), lsha (sha1 of the logits that decided every committed token, fp32),
checks, checks with pass 2, rows in pass 1 / 2, tok / check, refused commits, tok/s (shared GPU, rough).
Usage: [BigMoE venv] fit_t2.py TAG"""
import hashlib, json, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",  # SHIP_AR
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",              # B12_ENV b12ll
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",                               # FOLD_SHIP
       "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1",                # SHIP_MTP
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0",
       "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",                # env.py
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"}
for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_CM", "B12_CL", "B12_FOLD_CM", "B12_FOLD_AW", "B12_W", "B12_PMIN"):
    os.environ.pop(k, None)                          # the live file's own mode defaults (fast mode), as fit_creep's F12_ARM=live
os.environ.update(ENV)
TAG = sys.argv[1] if len(sys.argv) > 1 else "t2"
OUTD = os.environ.get("F12_OUT", os.path.expanduser("~/Fit12/data/t2"))
CAP = int(os.environ.get("F12_CAP", "128"))
ES = os.path.expanduser("~/EvalSuite")
BT = os.environ.get("F12_BT", os.path.expanduser("~/Fit12/pinned/8db7/bt_b12.py"))
EOS = (248044, 248046)
sha = hashlib.sha256(open(BT, "rb").read()).hexdigest()[:10]
assert sha == "8db71cc537", f"bt_b12.py is {sha}, T2 is written against 8db71cc537"
REF = [json.loads(l) for l in open(os.path.join(ES, "results", "b12llff4", "decode.jsonl"))]
WANT = os.environ.get("F12_KEYS", "humaneval/HumanEval/156,ifeval/,math500/test/number_theory/459.json").split(",")
SEL = [next(r for r in REF if r["key"].startswith(w) if w.endswith("/") or r["key"] == w) for w in WANT]
ITEMS = {}
for st_ in {r["set"] for r in SEL}:
    ITEMS.update({it["key"]: it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{st_}.jsonl")))})
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
import bt_b12 as B
import t2_twopass as T
assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
T.apply(B)                                           # before load_b12: the decode block is built there
rt, _ = B.load_b12(0.10, mtp=True)
model = rt.model
import spec125
from bigmoe_v3f.fuse import reset_generation_state
os.makedirs(OUTD, exist_ok=True)
print(f"[{TAG}] bt_b12 {sha}, engine {getattr(B, 'V3F', '?')}, C_m {B.CFG.get('cm')} C_l {B.CFG.get('cl')}, cap {CAP}; prompts "
      + ", ".join(f"{r['key']} (n {r['n_prompt']})" for r in SEL), flush=True)
ARMS = [("ref", "spec"), ("ex", None), ("g1", 1.01), ("g0", 0.0), ("t80", 0.8), ("t95", 0.95), ("t80b", 0.8)]
if os.environ.get("F12_T2ARMS"):                     # (Mon 22:5x) e.g. ref:spec,t80:0.8,t80s:0.8s; an 's' = skip (2, 8)
    ARMS = [(a.split(":")[0], (lambda v: "spec" if v == "spec" else None if v == "none" else v)(a.split(":")[1]))
            for a in os.environ["F12_T2ARMS"].split(",")]
SKIP = tuple(int(x) for x in os.environ.get("F12_SKIP", "2,8").split(","))
for r in SEL:
    ids = ITEMS[r["key"]]["prompt_ids"]
    res = {}
    for arm, th in ARMS:
        reset_generation_state(model)                # = decode._gen_mtp
        B.STASH.clear(); B.PENDH.clear()
        state = B.b12_prefill_state(model, list(ids), 512, 4096)
        t0 = time.perf_counter()
        if th == "spec":
            toks, sst = spec125.generate(model, list(ids), CAP, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512,
                                         prefilled=state, stop_ids=EOS)
            s = dict(passes=sst.get("passes"), pass2=0, rows1=0, rows2=0, refused=sst.get("commit_refused"), lsha="-")
        else:
            lg = []
            sk = SKIP if isinstance(th, str) and th.endswith("s") else None
            thf = None if th is None else float(th.rstrip("s")) if isinstance(th, str) else th
            toks, s = T.generate(B, model, list(ids), CAP, thf, state, stop_ids=EOS, log=lg, skip=sk)
            json.dump(lg, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}.json"), "w"))
        dt = time.perf_counter() - t0
        toks = [int(x) for x in toks]
        for j, x in enumerate(toks):
            if x in EOS:
                toks = toks[:j + 1]
                break
        res[arm] = (toks, s)
        rf = res["ref"][0]
        div = next((i for i, (a, b) in enumerate(zip(toks, rf)) if a != b), None if len(toks) == len(rf) else min(len(toks), len(rf)))
        print(f"[{TAG}] {r['key']} {arm} theta {th} | tok {len(toks)} sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} "
              f"| lsha {s['lsha']} | = ref tokens {toks == rf} (first divergence {div}) | checks {s['passes']}, with pass 2 "
              f"{s['pass2']}, rows pass 1 {s['rows1']} / pass 2 {s['rows2']}, refused {s['refused']} | {len(toks) / max(1, s['passes'] or 1):.2f} "
              f"tok / check | skipped {s.get('skipped', 0)} | {dt:.2f} s, {len(toks) / max(1e-9, dt):.2f} tok/s", flush=True)
    if not all(a in res for a in ("ex", "g1", "t80", "t80b", "g0")):
        continue
    ok_loop = res["ex"][0] == res["ref"][0]
    g1 = res["g1"][0] == res["ex"][0] and res["g1"][1]["lsha"] == res["ex"][1]["lsha"]
    det = res["t80"][0] == res["t80b"][0] and res["t80"][1]["lsha"] == res["t80b"][1]["lsha"]
    print(f"[{TAG}] {r['key']} GATES | loop = spec125 (tokens): {'PASS' if ok_loop else 'FAIL'} | G1 (theta 1.01 = exact loop, "
          f"tokens + logits): {'PASS' if g1 else 'FAIL'} | determinism (t80 = t80b): {'PASS' if det else 'FAIL'} | g0 pass 2 "
          f"checks {res['g0'][1]['pass2']} (must be 0)", flush=True)
print(f"[{TAG}] done", flush=True)
