"""Leap L4 generation (Sat 2026-10-24, DFL2D_PREREG.md part 2): the 125B's own outputs on dfl2d_samples.jsonl. Exact
greedy: SpecDecode's spec125.generate (MTP + lookup drafts, verified; defaults), C = 51 experts resident as L3's
extraction, up to 768 tokens per turn, stopping at the end of the turn (as specdec.py: eos ids + <|im_end|>).
Stops at ~36K generated tokens or 45 min. Output ~/Leap/data/dfl2d_windows.jsonl in dfl2c_extract.py's input format:
key, project, kind, ids = prompt + output, asst = 1 on the output, gen = prompt length, split. Resumable (appends,
skips done keys). Usage: [BigMoE venv] dfl2d_gen.py [TOKENS] [MINUTES]"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import mlx.core as mx
from bigmoe.loader import load_streamed
import pausewait   # quiet windows: passes wait at their boundary while PAUSE exists (RULES.md, 10-24)
import spec125

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 36_000
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 45
# L5 (block-16, code / math): DFL2D_IN / DFL2D_OUT / DFL2D_C / DFL2D_MAXT override the L4 settings
MAXT = int(os.environ.get("DFL2D_MAXT", "768"))
C = int(os.environ.get("DFL2D_C", "51"))
MD = os.path.expanduser("~/BigMoE/models/bare")
IN, OUT = (os.path.expanduser(os.environ.get(e, f"~/Leap/data/{n}"))
           for e, n in (("DFL2D_IN", "dfl2d_samples.jsonl"), ("DFL2D_OUT", "dfl2d_windows.jsonl")))
rows = [json.loads(l) for l in open(IN)]
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
total = sum(len(json.loads(l)["ids"]) - json.loads(l)["gen"] for l in open(OUT)) if os.path.exists(OUT) else 0

t0 = time.time()
rt, st = load_streamed(MD, C, mtp=True)
stop = set(getattr(rt.tokenizer, "eos_token_ids", None) or []) | {rt.tokenizer.convert_tokens_to_ids("<|im_end|>")}
spec125.generate(rt.model, rt.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                 enable_thinking=False), 8, stop_ids=stop)   # warm-up
print(f"[l4gen] loaded {time.time() - t0:.0f}s (C={C}); {len(rows)} turns, {len(done)} done ({total} tokens); "
      f"stop ids {sorted(stop)}; {IN} -> {OUT}", flush=True)
t0, work = time.time(), 0.0         # work time: a turn counts at most 300 s, so a frozen (SIGSTOP) job keeps its budget
with open(OUT, "a") as fo:
    for r in rows:
        if r["key"] in done:
            continue
        if total >= TOKENS or work > MINUTES * 60:
            print(f"[l4gen] budget reached: {total} tokens, {work / 60:.1f} min of work ({(time.time() - t0) / 60:.1f} wall)",
                  flush=True)
            break
        t1, s = time.time(), {}
        out, s = spec125.generate(rt.model, r["prompt"], MAXT, stop_ids=stop, stats=s)
        st.gate.check()
        out = [int(t) for t in out]
        ids = [int(t) for t in r["prompt"]] + out
        fo.write(json.dumps({"key": r["key"], "project": r.get("project", ""), "kind": r["kind"], "ids": ids,
                             "asst": [0] * len(r["prompt"]) + [1] * len(out), "gen": len(r["prompt"]),
                             "split": r["split"]}) + "\n")
        fo.flush()
        total += len(out)
        dt = time.time() - t1; work += min(dt, 300.0)
        print(f"[l4gen] {r['key']} ({r['kind']}, {r['split']}): prompt {len(r['prompt'])}, {len(out)} tokens in {dt:.0f}s "
              f"(prefill {s.get('t_prefill', 0):.0f}s); total {total} in {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"[l4gen] done: {total} generated tokens; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
