"""Breakthrough R-41 (NewIdeas Wed 10-28): bisect the fold on the Eval suite's agent/write16, the one row (of the first
45) where B12_FOLD on and off give different tokens (from token 32; on 321 tokens, off 466). One arm per process, exactly
as decode._gen_mtp decodes a row: write16's prompt_ids (~/EvalSuite/data/items/dec_agent.jsonl, read-only), then
B.b12_prefill_state + spec125.generate 3:3:0, greedy, stop at {248044, 248046}; BI_N tokens (default 64, past the split).
The arm is set by env (B12_FOLD and B12_FOLD_<PIECE>=0); the RESULT line says whether its tokens equal the suite's
fold-off row (results/b12llfoff) or fold-on row (results/b12llfon), and where each first differs.
Usage: [BigMoE venv] B12_ENGINE=... B12_FOLD=... [B12_FOLD_AW=0 ...] [BT_DIR=...] w16_bisect.py TAG"""
import hashlib, json, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1",
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0"}          # the suite's b12llfoff / b12llfon env (cli.py)
if os.environ.get("BI_AR") == "1":                   # the AR path's shipped env (pfr_ab.py): no MTP head, no MTP stream
    ENV = {"B12_CACHE_MB": "64", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_GROUP": "64", "BIGMOE_MTP_C": "32",
           "BIGMOE_ASYNC_EVERY": "1", "B12_LATE3": "2,10", "B12_NORM": "mass"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "w16"
N = int(os.environ.get("BI_N", "64"))
KEY, EOS = "agent/write16", {248044, 248046}
ES = os.path.expanduser("~/EvalSuite/")
item = next(json.loads(l) for l in open(ES + "data/items/dec_agent.jsonl") if json.loads(l)["key"] == KEY)
ref = {a: next(json.loads(l)["completion_ids"] for l in open(f"{ES}results/{a}/decode.jsonl") if json.loads(l)["key"] == KEY)
       for a in ("b12llfoff", "b12llfon")}

import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
if os.environ.get("BT_DIR"):                         # a bt_b12 under test, before it replaces the live one
    sys.path.insert(0, os.environ["BT_DIR"])
import bt_b12 as B
AR = os.environ.get("BI_AR") == "1"                  # the AR path instead: b12_prefill + one-row greedy decode (no refs)
rt, st = B.load_b12(0.10, mtp=not AR)
import spec125
from bigmoe_v3f.fuse import reset_generation_state
model = rt.model
if AR:
    ids = list(item["prompt_ids"])
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    cache, t0 = model.make_cache(), time.perf_counter()
    B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, 4096)
    mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
    tp, y, out = time.perf_counter() - t0, ids[-1], []
    for _ in range(N):
        lg = model(mx.array([[y]]), cache=cache)
        y = int(mx.argmax((lg[0] if isinstance(lg, tuple) else lg)[0, -1]).item())
        out.append(y)
        if y in EOS:
            break
    st.gate.check()
    print(f"[{TAG}] AR RESULT sha {hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10]} | {len(out)} tokens | B12_PFR "
          f"{os.environ.get('B12_PFR')} grid {os.environ.get('B12_PFR_GRID', 'balanced')} | PFR {B.PFR_LOG[-1] if B.PFR_LOG else '-'} "
          f"| prefill {tp:.1f} s | tokens 0-40 {out[:41]}", flush=True)
    sys.exit(0)
arm = {k: os.environ.get(k) for k in ("B12_ENGINE", "B12_FOLD", "B12_FOLD_PFR", "B12_FOLD_CM", "B12_FOLD_ROWS", "B12_FOLD_MOV",
                                      "B12_FOLD_AW", "B12_ALLWAIT", "B12_CM", "BT_DIR") if os.environ.get(k)}
print(f"[{TAG}] bt_b12 {B.__file__} | {arm} | write16: {len(item['prompt_ids'])} prompt tokens, refs off {len(ref['b12llfoff'])} "
      f"/ on {len(ref['b12llfon'])} tokens, first differ at {next(i for i, (a, b) in enumerate(zip(*ref.values())) if a != b)}",
      flush=True)
ids = list(item["prompt_ids"])
reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
aw0, t0 = getattr(B, "AW_N", [0])[0], time.perf_counter()
state = B.b12_prefill_state(model, ids, 512, 4096)
tp = time.perf_counter() - t0
toks, s = spec125.generate(model, ids, N, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stop_ids=EOS,
                           stats={})
st.gate.check()
toks = [int(x) for x in toks]


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


n = len(toks)
fo, fn = first_diff(toks, ref["b12llfoff"]), first_diff(toks, ref["b12llfon"])
verdict = "= OFF" if fo is None else "= ON" if fn is None else "NEITHER"
print(f"[{TAG}] RESULT {verdict} | {n} tokens: vs off first diff {fo}, vs on first diff {fn} | passes {s.get('passes')} "
      f"accepted {s.get('accepted')} drafted {s.get('drafted')} | C_m {B.CFG['cm']} | ALLWAIT calls "
      f"{getattr(B, 'AW_N', [0])[0] - aw0} | PFR {B.PFR_LOG[-1] if B.PFR_LOG else '-'} | prefill {tp:.1f} s | sha "
      f"{hashlib.sha1(json.dumps(toks).encode()).hexdigest()[:10]} | tokens 28-36 {toks[28:37]} (off {ref['b12llfoff'][28:37]}, "
      f"on {ref['b12llfon'][28:37]})", flush=True)
