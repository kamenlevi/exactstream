"""MTP Head copy of ~/BigMoE/scripts/mtp_depth.py (a1af5e4) for D-05: a prompt line may carry "ids" (EvalSuite's
tf_prompts prompt_ids, already chat-templated) instead of "text"; its "key" is kept in the record. Otherwise unchanged.
Per-pass MTP-chain log for TrainLab's D-05 MTPHYB (MTP Head, Wed 2026-10-28): the shipped 24 GB v3 engine's greedy
decode with the native MTP head drafting a k-deep chain EVERY pass (no n-gram lookup, no adaptive width: each pass
verifies exactly the k MTP drafts). One record per pass: n = drafts accepted, mtp = the chain's k drafts, g = the 125B's
own greedy tokens at the verified rows. The output tokens are exact (greedy acceptance), so sha must match a plain decode.
Per-depth agreement: depth 1 = mtp[0] == g[0] over all passes; depth j given depths 1..j-1 accepted = mtp[j-1] == g[j-1]
over the passes with n >= j - 1 (g at deeper rows is conditioned on the drafted prefix, so only that conditional rate
is defined from a decode; it equals the teacher-forced rate on those positions).
usage (env: scripts/mtp_depth.sh sets the shipped stack): mtp_depth.py PROMPTS.jsonl OUT.json [k=3] [N=300]
  PROMPTS.jsonl: one {"domain": ..., "text": ...} per line, chat-templated here with thinking off"""
import hashlib, json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for _k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(_k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
from bigmoe.loader import load_streamed
import spec125

PROMPTS, OUT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
rt, st = load_streamed(MD, max(16, round(float(os.environ.get("TL_CFRAC", "0.25")) * cfg["num_experts"])), mtp=True)
tok = rt.tokenizer


def ids_of(t):
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])


spec125.generate(rt.model, ids_of("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
st.gate.check()
res = {"k": K, "N": N, "env": {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}, "prompts": []}
agree = {}                                                  # domain -> [[hits, trials] per depth]
seen = {}
for line in open(PROMPTS):
    if not line.strip():
        continue
    p = json.loads(line)
    if os.environ.get("TL_DOMAINS") and p["domain"] not in os.environ["TL_DOMAINS"].split(","):   # 10-28 19:4x: subsets
        continue
    seen[p["domain"]] = seen.get(p["domain"], 0) + 1
    if seen[p["domain"]] > int(os.environ.get("TL_PER_DOMAIN", "999")):
        continue
    stats = {"trace": []}
    t0 = time.perf_counter()
    out, s = spec125.generate(rt.model, list(p["ids"]) if "ids" in p else ids_of(p["text"]), N, k_mtp=K, max_draft=K, lookup=0, stats=stats)
    st.gate.check()
    passes = [[r["n"], [int(t) for t in r["mtp"][:K]], [int(t) for t in r["g"][:K]]] for r in stats["trace"]]
    a = agree.setdefault(p["domain"], [[0, 0] for _ in range(K)])
    for n, m, g in passes:
        for j in range(min(K, len(m), len(g))):
            if j == 0 or n >= j:                             # depth j + 1 is defined when depths 1..j were accepted
                a[j][0] += int(m[j] == g[j]); a[j][1] += 1
    res["prompts"].append(dict(domain=p["domain"], key=p.get("key"), n_tok=len(out), tok_s=s["tok_s"], wall=time.perf_counter() - t0,
                               sha=hashlib.sha1(str(list(out)).encode()).hexdigest()[:10], passes=passes,
                               out=[int(t) for t in out], timers={k: float(v) for k, v in s.items()   # 10-28: ids + timers
                                                                  if isinstance(v, (int, float)) and not isinstance(v, bool)}))
    print(f"[{p['domain']}] {len(out)} tok, {s['tok_s']:.1f} tok/s, {len(passes)} passes", flush=True)
    res["agreement"] = {d: [round(h / max(t, 1), 4) for h, t in v] for d, v in agree.items()}
    json.dump(res, open(OUT, "w"))
for d, v in res.get("agreement", {}).items():
    print(f"{d}: per-depth agreement " + " / ".join(f"{x:.3f}" for x in v), flush=True)
print("saved", OUT, flush=True)
