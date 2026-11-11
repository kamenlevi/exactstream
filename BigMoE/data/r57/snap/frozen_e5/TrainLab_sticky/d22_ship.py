"""D-22 on the SHIPPED 45 GB line (MTP Head, Fri 10-30 22:2x; the orchestrator's suggestion; settings from BigMoE 21:56):
H50 = the 24 GB v3 stack at C 256 (0.50 x 512), MTP drafting with n-gram lookup and the adaptive width (k_mtp 1,
max_draft 7, AdaptiveWidth V0, step 2.2 ms, alpha 2). Arms, interleaved per prompt: 'stockinst' = the head's own weights
installed through mtp125.install (bf16 Linear copies of fc_hidden / fc_embedding / q,k,v,o / router / shared expert +
MTPLX's plain MoE block: the same code path as the trained arm) vs a D-22 head (ckpt/d22_<run>.safetensors). Run with
BIGMOE_MTP_STREAM unset (the install would bypass its streamed MTP pool). Greedy verification is exact: both arms'
outputs must be token-identical (a difference = an install clash, not the head).
Usage (env from turns/turn53.sh): d22_ship.py RUN N [prompts: code,gsm,ifeval,multi] -> data/d22_ship_<RUN>.txt"""
import hashlib, json, os, sys, time
os.environ["TL_INFER"] = "1"                                 # mtp125: keep MTPLX's fused HC read
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
assert os.environ.get("BIGMOE_MTP_STREAM") != "1", "run with BIGMOE_MTP_STREAM unset"
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/TrainLab_root"))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
sys.path.insert(2, os.path.expanduser("~/SpecDecode"))
sys.path.insert(3, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
import numpy as np
import mlx.core as mx
import mtp125 as M
from bigmoe.loader import load_streamed
import pausewait                                             # passes wait at their boundary while PAUSE exists
import spec125

H = os.path.expanduser
RUN, N = sys.argv[1], int(sys.argv[2])
DOMS = (sys.argv[3] if len(sys.argv) > 3 else "code,gsm,ifeval,multi").split(",")
MD = H("~/BigMoE/models/bare")
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

t0 = time.time()
CE = int(os.environ.get("D22_SHIP_C", "256"))                 # 256 = H50; lower when memory is short
rt, st = load_streamed(MD, CE, mtp=True)
head = rt.model.language_model.mtp
T0 = M.Trainable(head)                                       # the head's own weights (fp32 copies)
T1 = M.Trainable(head); T1.load_weights(H(f"~/TrainLab/ckpt/d22_{RUN}.safetensors"))
mx.eval(T0.parameters(), T1.parameters())
tok = rt.tokenizer
say(f"D-22 shipped-line A/B: H50 stack (C {CE}), head 'stockinst' vs '{RUN}', {N} tokens per decode; loaded in "
    f"{time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB active")

def ids_of(t):
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])

P = [("code/prompt_code", ids_of(open(H("~/BigMoE/data/prompt_code.txt")).read()))]
per = {"code": 3, "gsm": 2, "ifeval": 2, "multi": 2}
cnt = {}
for l in open(H("~/TrainLab/data/d05_prompts.jsonl")):
    r = json.loads(l)
    if r["domain"] in DOMS and cnt.get(r["domain"], 0) < per.get(r["domain"], 0):
        cnt[r["domain"]] = cnt.get(r["domain"], 0) + 1
        P.append((r["domain"] + "/" + r["key"], [int(t) for t in r["ids"]]))

def arm(T):
    M.install(head, T, dtype=mx.bfloat16)

MODE = os.environ.get("D22_SHIP_MODE", "adapt")             # adapt = the shipped width rule; k7 = a fixed 7-draft MTP chain

def decode(ids):
    s = {}
    t1 = time.perf_counter()
    if MODE == "k7":
        out, s = spec125.generate(rt.model, ids, N, k_mtp=7, max_draft=7, lookup=0, stats=s)
    else:
        out, s = spec125.generate(rt.model, ids, N, k_mtp=1, max_draft=7, lookup=1,
                                  adapt=spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0), stats=s)
    st.gate.check()
    wall = time.perf_counter() - t1
    return out, s, wall

def rec(out, s, wall):
    ps = max(1, int(s.get("passes", 1)))
    return dict(tok_s=float(s.get("tok_s", len(out) / wall)), n=len(out), passes=ps, per_pass=len(out) / ps,
                drafted=int(s.get("drafted", 0)) / ps, accepted=int(s.get("accepted", 0)) / ps,
                sha=hashlib.sha256(np.array(out, np.int64).tobytes()).hexdigest()[:12],
                paused=os.path.exists(H("~/.local/share/quiet/PAUSE")))

arm(T0)
spec125.generate(rt.model, ids_of("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
st.gate.check()
say(f"  mode {MODE}; per prompt: one discarded warm-up decode, then A-B-A (A = stockinst, B = {RUN}); A's tok/s = the mean "
    f"of its two runs, so cache warmth is balanced")
res = []
for name, ids in P:
    row = {"prompt": name}
    arm(T0); decode(ids)                                     # discarded: the prompt's experts warm for every measured run
    a1 = rec(*decode(ids))
    arm(T1); b = rec(*decode(ids))
    arm(T0); a2 = rec(*decode(ids))
    a = dict(a1); a["tok_s"] = (a1["tok_s"] + a2["tok_s"]) / 2
    row["stockinst"], row[RUN], row["a2"] = a, b, a2
    same = a1["sha"] == b["sha"] == a2["sha"]
    res.append(row)
    say(f"  {name}: tok/s A {a1['tok_s']:.1f} / {a2['tok_s']:.1f}, B {b['tok_s']:.1f} ({100 * (b['tok_s'] / a['tok_s'] - 1):+.1f} %); "
        f"per pass: tokens {a['per_pass']:.2f} -> {b['per_pass']:.2f}, drafted {a['drafted']:.2f} -> {b['drafted']:.2f}, "
        f"accepted {a['accepted']:.2f} -> {b['accepted']:.2f}; outputs identical {same}")
ok = all(r["stockinst"]["sha"] == r[RUN]["sha"] == r["a2"]["sha"] for r in res)
for dom in sorted({r["prompt"].split("/")[0] for r in res}):
    rs = [r for r in res if r["prompt"].startswith(dom + "/")]
    a = np.mean([r["stockinst"]["tok_s"] for r in rs]); b = np.mean([r[RUN]["tok_s"] for r in rs])
    pa = np.mean([r["stockinst"]["per_pass"] for r in rs]); pb = np.mean([r[RUN]["per_pass"] for r in rs])
    da = np.mean([r["stockinst"]["drafted"] for r in rs]); db = np.mean([r[RUN]["drafted"] for r in rs])
    say(f"  {dom} ({len(rs)}): mean tok/s {a:.1f} -> {b:.1f} ({100 * (b / a - 1):+.1f} %), tokens per pass {pa:.2f} -> {pb:.2f}, "
        f"drafted per pass {da:.2f} -> {db:.2f}")
say(f"  every output identical across arms: {ok}; MLX peak {mx.get_peak_memory() / 1e9:.1f} GB; {time.time() - t0:.0f} s")
json.dump(res, open(H(f"~/TrainLab/data/d22_ship_{RUN}_{MODE}_aba.json"), "w"))
open(H(f"~/TrainLab/data/d22_ship_{RUN}.txt"), "a").write("\n".join(lines) + "\n")
