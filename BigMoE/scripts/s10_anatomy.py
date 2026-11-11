"""S10 / B5s step 1 (NewIdeas QUEUE.md): where a 24 GB streamed verify check loses time, from BigMoE's ext trace.
Runs the shipped v3 decode (spec125 adaptive width, MTP) with BIGMOE_TRACE=1 on code / prose / edit and records:
- gate events (ext.trace(), kind 1: loader saw the GPU's routing row -> released the layer; kind 4: that call had misses),
- every expert read (ext.trace_reads(): demand or prefetch, issue -> bytes in place),
- per gated call, how its experts were found (ext.trace_cov(): resident, prefetched on time / late, queued read cancelled
  into demand, demand miss predicted-but-unissued / unpredicted).
Per verify pass (passes with >= 40 trunk layers; MTP draft calls are layer 48): span, gate-wait (stall) vs GPU time,
SSD busy, reads in flight during stalls and during GPU time, and the share of misses known >= 1 layer ahead.
usage: s10_anatomy.py FRAC N OUT.json [prompts=code,prose,edit]   (the stack's env from the runner)"""
import json, os, sys, time
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
from bigmoe.loader import load_streamed
from bigmoe.gated import ext, PAUSED
import spec125

FRAC, N, OUT = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PROMPTS = (sys.argv[4] if len(sys.argv) > 4 else "code,prose,edit").split(",")
MD = os.path.expanduser("~/BigMoE/models/bare")
cfg = json.load(open(os.path.join(MD, "config.json"))); cfg = cfg.get("text_config", cfg)
NL = cfg["num_hidden_layers"]
rt, st = load_streamed(MD, max(16, round(FRAC * cfg["num_experts"])), mtp=True)
tok = rt.tokenizer
TEXT = {
    "code": open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "edit": "Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
            + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block.",
    "prose": "Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
             "Use one everyday analogy and keep it to about 300 words.",
    # R-43 (b43): a math-text copy task, so lookup drafts run the full width on GSM-style reasoning text
    "copy_math": "Copy the following worked solutions exactly, with no changes and no commentary:\n\n" + "\n\n".join(
        json.loads(l)["answer"] for l in open(os.path.expanduser("~/EvalSuite/data/raw/gsm8k_test.jsonl")).readlines()[:6]),
}
V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}


def ids_of(t):
    i = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, enable_thinking=False)
    return list(i if isinstance(i[0], int) else i["input_ids"])


def drain():
    return ext.trace(), ext.trace_reads(), ext.trace_cov()


def union_len(iv):
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def inflight_profile(reads, a, b, gaps):
    """time-weighted reads in flight over [a, b], split by whether the instant lies in a gate wait (stall) or not"""
    ev = []
    for r in reads:
        t0, t1 = max(r[3], a), min(r[4], b)
        if t1 > t0:
            ev += [(t0, 1), (t1, -1)]
    ev.sort()
    stall_iv = sorted(gaps)
    out = {"stall": np.zeros(40), "gpu": np.zeros(40)}               # seconds spent at each in-flight count
    cur, last = 0, a

    def add(t_from, t_to, n):
        if t_to <= t_from:
            return
        # split [t_from, t_to] by stall intervals
        pos = t_from
        for s0, s1 in stall_iv:
            if s1 <= pos or s0 >= t_to:
                continue
            if s0 > pos:
                out["gpu"][min(n, 39)] += s0 - pos
            out["stall"][min(n, 39)] += min(s1, t_to) - max(s0, pos)
            pos = min(s1, t_to)
        if pos < t_to:
            out["gpu"][min(n, 39)] += t_to - pos

    for t, d in ev:
        add(last, t, cur)
        cur += d
        last = t
    add(last, b, cur)
    return out


res = {"frac": FRAC, "N": N, "env": {k: v for k, v in os.environ.items() if k.startswith(("BIGMOE_", "MTPLX_"))}, "prompts": {}}
spec125.generate(rt.model, ids_of("Hi"), 8, k_mtp=1, max_draft=7, lookup=1)   # warm-up
st.gate.check()
for P in PROMPTS:
    drain()
    # S10_FIXED=1: fixed widths (lookup drafts, else MTP k=1). Adaptive width feeds measured verify times back into its
    # cost model, so its passes differ run to run; an oracle replay (BIGMOE_ORACLE_PLAY, keyed by (pass, layer)) needs
    # the recording run's passes exactly
    aw = None if os.environ.get("S10_FIXED") == "1" else spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0)
    stats = {"trace": []}
    p0 = dict(PAUSED); PAUSED["t_last"] = 0.0          # no gap across the prompt boundary
    fc0 = ext.stats()
    t0 = time.perf_counter()
    # R-43 (b43): S10_MAX_DRAFT / S10_LOOKUP set the verify width on copy-heavy prompts (with S10_FIXED=1: the widest
    # lookup draft is verified whole), to price wide (DFlash2-like) checks from the widths the gate row can carry today
    out, s = spec125.generate(rt.model, ids_of(TEXT[P]), N, k_mtp=1, max_draft=int(os.environ.get("S10_MAX_DRAFT", "7")),
                              lookup=int(os.environ.get("S10_LOOKUP", "1")), stats=stats, adapt=aw)
    wall = time.perf_counter() - t0
    paused = PAUSED["pause_s"] - p0["pause_s"]; gap_s = PAUSED["gap_s"] - p0["gap_s"]; ngap = PAUSED["n_gap"] - p0["n_gap"]
    clean = paused < 0.5 and ngap == 0
    st.gate.check()
    gev, rds, cov = drain()
    gev, rds, cov = np.array(gev), np.array(rds), np.array(cov)
    # verify passes: seq values whose gated calls cover >= 40 trunk layers
    passes = []
    if len(gev):
        k1 = gev[gev[:, 2] == 1]
        for sq in np.unique(k1[:, 1]):
            e = k1[k1[:, 1] == sq]
            ml = e[:, 0].astype(int) % (NL + 1)                       # base L and sidecar (NL+1+L) -> model layer
            trunk = e[ml < NL]
            if len(np.unique(ml[ml < NL])) < 40:
                continue
            a, b = trunk[:, 3].min(), trunk[:, 4].max()
            gaps = [(x[3], x[4]) for x in trunk]
            stall = union_len(gaps)
            rr = rds[(rds[:, 4] > a) & (rds[:, 3] < b)] if len(rds) else np.zeros((0, 5))
            prof = inflight_profile(rr, a, b, gaps)
            cv = cov[cov[:, 1] == sq] if len(cov) else np.zeros((0, 10))
            cv = cv[(cv[:, 0].astype(int) % (NL + 1)) < NL]
            need, hit = cv[:, 2].sum(), cv[:, 3].sum()
            passes.append(dict(seq=int(sq), span=(b - a) * 1e3, stall=stall * 1e3, gpu=(b - a - stall) * 1e3,
                               ssd_busy=union_len([(max(x[3], a), min(x[4], b)) for x in rr]) * 1e3,
                               reads_demand=int((rr[:, 2] == 0).sum()), reads_pf=int((rr[:, 2] == 1).sum()),
                               reads_joint=int((rr[:, 2] == 2).sum()),
                               inflight_stall=(prof["stall"] * np.arange(40)).sum() / max(prof["stall"].sum(), 1e-12),
                               inflight_gpu=(prof["gpu"] * np.arange(40)).sum() / max(prof["gpu"].sum(), 1e-12),
                               ge8_stall=prof["stall"][8:].sum() / max(prof["stall"].sum(), 1e-12),
                               ge8_gpu=prof["gpu"][8:].sum() / max(prof["gpu"].sum(), 1e-12),
                               idle_gpu=prof["gpu"][0] / max(prof["gpu"].sum(), 1e-12),
                               need=int(need), hit=int(hit), miss=int(need - hit), pf_ready=int(cv[:, 4].sum()),
                               pf_late=int(cv[:, 5].sum()), q_demand=int(cv[:, 6].sum()),
                               pred_unissued=int(cv[:, 7].sum()), unpred=int(cv[:, 8].sum())))
    S = np.mean([r["S"] for r in stats["trace"]]) if stats["trace"] else 0
    import hashlib
    _d = ext.stats()
    if _d.get("joint_tried"):
        print(f"   joint: tried {_d['joint_tried']} paired {_d['joint_paired']} reads {_d['joint_reads']} "
              f"wait {_d['joint_wait_s'] * 1e3:.1f} ms (cumulative)", flush=True)
    FK = ["fc_miss", "fc_lead2"] + [f"{a}{k}" for a in ("fc_d", "iss_d", "iss_used_d", "fc_u", "fc_ev") for k in (1, 2, 3)]
    fc = {k: int(_d.get(k, 0) - fc0.get(k, 0)) for k in FK}
    sha = hashlib.sha1(str(list(out)).encode()).hexdigest()[:10]
    vms = 1e3 * s["t_verify"] / max(s["passes"], 1)
    print(f"[{P}] {s['tok_s']:.1f} tok/s, {s['tok_per_pass']:.2f} tok/pass, S {S:.2f}, verify {vms:.1f} ms/pass x "
          f"{s['passes']} passes, sha {sha}" + ("" if clean else f"  NOT CLEAN: paused {paused:.1f} s, {ngap} gaps "
                                                 f"{gap_s:.1f} s"), flush=True)
    if fc["fc_miss"]:                                 # S10 per-distance forecast log (ext, JIT)
        M = fc["fc_miss"]; pc = lambda a, b: 100.0 * a / b if b else float("nan")
        r = "/".join(f"{pc(fc[f'fc_d{k}'], M):.0f}" for k in (1, 2, 3))
        u = "/".join(f"{fc[f'fc_u{k}'] / fc[f'fc_ev{k}']:.0f}" if fc[f"fc_ev{k}"] else "-" for k in (1, 2, 3))
        i = "/".join(str(fc[f"iss_d{k}"]) for k in (1, 2, 3))
        pr = "/".join(f"{pc(fc[f'iss_used_d{k}'], fc[f'iss_d{k}']):.0f}" for k in (1, 2, 3))
        ru = "/".join(f"{pc(fc[f'iss_used_d{k}'], M):.0f}" for k in (1, 2, 3))
        print(f"   forecasts: {M} misses at forecast layers; raw recall at lead 1/2/3 {r} % (unique list {u}), known >= 2 "
              f"ahead {pc(fc['fc_lead2'], M):.0f} %; reads issued {i}, precision {pr} %, recall of issued {ru} %", flush=True)
    res["prompts"][P] = dict(tok_s=s["tok_s"], tok_per_pass=s["tok_per_pass"], S=float(S), wall=wall, verify_ms=vms,
                             n_passes=s["passes"], sha=sha, passes=passes, paused=paused, gaps=gap_s, n_gap=ngap, clean=clean,
                             fc=fc, stats={k: v for k, v in s.items() if isinstance(v, (int, float))})   # spec125's timers
    if passes:
        A = {k: float(np.mean([p[k] for p in passes])) for k in passes[0] if k != "seq"}
        M = A["miss"]
        print(f"   {len(passes)} traced verify passes: "
              f"span {A['span']:.1f} ms = gate wait {A['stall']:.1f} + GPU {A['gpu']:.1f}; SSD busy {A['ssd_busy']:.1f} ms; "
              f"reads/pass demand {A['reads_demand']:.1f} + joint {A['reads_joint']:.1f} + prefetch {A['reads_pf']:.1f}", flush=True)
        print(f"   reads in flight: during gate waits {A['inflight_stall']:.1f} (>= 8: {100 * A['ge8_stall']:.0f} % of the time), "
              f"during GPU time {A['inflight_gpu']:.1f} (>= 8: {100 * A['ge8_gpu']:.0f} %, SSD idle {100 * A['idle_gpu']:.0f} %)", flush=True)
        print(f"   experts/pass {A['need']:.1f}: resident {A['hit']:.1f}, misses {M:.1f} = prefetched on time {A['pf_ready']:.1f} "
              f"+ late {A['pf_late']:.1f} + predicted-unissued {A['pred_unissued']:.1f} + unpredicted {A['unpred']:.1f} "
              f"(queued-then-cancelled {A['q_demand']:.1f}); known >= 1 layer ahead "
              f"{100 * (A['pf_ready'] + A['pf_late'] + A['pred_unissued']) / max(M, 1e-9):.0f} %, read in time "
              f"{100 * A['pf_ready'] / max(M, 1e-9):.0f} %", flush=True)
json.dump(res, open(OUT, "w"))
print("saved", OUT)
