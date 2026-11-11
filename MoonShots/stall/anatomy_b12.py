"""Stall anatomy part 2: Breakthrough's B12 (frugal 10 % on BigMoE's v3 tier engine, pinned copy ~/Breakthrough/v3f, imported
read-only through bt_b12.load_b12) + MTP 3:3:0 via SpecDecode's spec125, the smoke-test settings (BIGMOE_PF_GROUP=64,
BIGMOE_MTP_C=32), with the same per-pass ext hook as anatomy.py (B12's own ext: bigmoe_v3f) plus its defer_stats
(rows, waited, late, over). Writes data/stall/anatomy_PROMPT_b12_f0.1.json for report.py (tag b12).
usage: anatomy_b12.py PROMPT [N=160]   (GPU, ~12 GB; quiet lock)"""
import json, os, sys, time
for k, v in (("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "32"), ("BIGMOE_MAXK", "80"), ("BIGMOE_FRSPEC", "1"),
             ("BIGMOE_PF_GROUP", "64"), ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"),
             ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"), ("MTPLX_DEFER_REPAIR_EVAL", "1"), ("MTPLX_NGRAM_PREWARM", "off")):
    os.environ.setdefault(k, v)
PROMPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 160
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import bt_b12 as B
t0 = time.time()
rt, st = B.load_b12(0.10, mtp=True)
import mlx.core as mx
import spec125
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state

model = rt.model
PROMPTS = {"code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": "Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
                    "Use one everyday analogy and keep it to about 300 words."}
ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[PROMPT]}], add_generation_prompt=True,
                                       enable_thinking=False)
ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
KEYS = ("uses", "misses", "stalled", "gated", "io_s", "cpu_wait_s", "proc_s", "slow_calls", "slow_s", "pf_issued", "pf_used",
        "pf_ontime", "pf_late", "pf_waited_s", "miss_unpred", "miss_pred_unissued")
last = {}


def hook():
    s = ext.stats()
    d = {k: s[k] - last.get(k, 0) for k in KEYS}
    last.update({k: s[k] for k in KEYS})
    rw = list(ext.defer_stats())
    dd = [a - b for a, b in zip(rw, last.get("_defer", [0] * len(rw)))]
    last["_defer"] = rw
    out = {"x_" + k: v for k, v in d.items()}
    out.update(x_rows=dd[0], x_waited=dd[1], x_late=dd[2], x_over=dd[3])
    return out


def prep():
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()


prep()
spec125.generate(model, ids, 16, k_mtp=3, max_draft=3, lookup=0)                       # warm-up
st.gate.check()
prep()
hook()
stats = {"trace": [], "pass_hook": hook}
t1 = time.perf_counter()
out, s = spec125.generate(model, ids, N, k_mtp=3, max_draft=3, lookup=0, stats=stats, prefill_chunk=2048)
wall = time.perf_counter() - t1
st.gate.check()
recs = [{k: (v if isinstance(v, (int, float)) else None) for k, v in r.items() if k in ("S", "n", "t_verify") or k.startswith("x_")}
        for r in stats["trace"]]
summ = {k: s[k] for k in ("passes", "tok_s", "tok_per_pass", "t_verify", "t_draft", "t_build", "t_wait", "t_commit", "t_head",
                          "t_dwait", "lookup_used") if k in s}
res = dict(prompt=PROMPT, N=N, frac=0.1, C=51, tier=True, b12=True, wall_s=wall, summary=summ, passes=recs, prompt_tokens=len(ids))
os.makedirs(os.path.expanduser("~/MoonShots/data/stall"), exist_ok=True)
json.dump(res, open(os.path.expanduser(f"~/MoonShots/data/stall/anatomy_{PROMPT}_b12_f0.1.json"), "w"))
print(f"[{PROMPT}_b12] {s['tok_s']:.1f} tok/s, {s['tok_per_pass']:.2f} tok/pass, {s['passes']} passes; verify {s['t_verify']:.2f} s, "
      f"draft {s['t_draft']:.2f} s; misses {sum(r['x_misses'] for r in recs)}, waited {sum(r['x_waited'] for r in recs)}, "
      f"io {sum(r['x_io_s'] for r in recs):.2f} s (loaded {time.time() - t0:.0f} s)", flush=True)
