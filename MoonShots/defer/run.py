"""dh_wW on BigMoE's v3 engine (opt-in NON-EXACT fast mode; NewIdeas' authorized build 2026-10-22): one harness, one arm
per process, on the pinned copy defer/v3d (= Breakthrough's v3f = BigMoE 7b8c7b4 + flag-mode loader; + MoonShots'
full-slot phase B and the multi-row dh path in tier.py). Env = BigMoE b13's deployed v3 at 24 GB (stall/anatomy.py with
STALL_TIER=1: tiers 4433333333, 3c, fused combine, OoO, JIT prefetch, C = round(FRAC x 512)).
ARM (env DH_ARM): v3 (control) | dh (BIGMOE_DH=DH_W, default 4; BIGMOE_DH_D=1: d1, one full layer late) | dh10 (W=10: flag mode, nothing late; must equal v3
bit for bit) | force (every rank >= W late, resident or not) | ref (force's split from v3's synchronous per-rank outputs;
must equal force bit for bit).
MODE:
  gen PROMPT [N=160]   SpecDecode spec125, the deployed adaptive config (k_mtp=1, max_draft=7, lookup=1, AdaptiveWidth),
                       per-pass ext deltas + defer stats -> data/defer/gen_PROMPT_ARM.json; prints tok/s, sha
  tf [NT=32] [S=4]     teacher-forced S-row forwards over the code prompt's last NT tokens -> data/defer/tf_ARM_SS.npz
  fixed [S=4] [NT=48]  fixed-state all-hits verify passes (engine/test_m1b_speed.py's FIXED repeat) -> median ms
usage: run.py MODE ...   (GPU, ~22 GB; quiet lock). DH_TRACE=1: ext per-layer trace (kind 5 = phase A release).
Pipelined late reads: BIGMOE_EXT_BUILD=build_pipe BIGMOE_DEFER_PIPE_US=d DH_TAG=_piped (ext built with -DDEFER_PIPE=1)."""
import hashlib, json, os, sys, time
ENV = dict(BIGMOE_DUPROW="1", BIGMOE_COMPILE_HC="1", BIGMOE_HC_VREAD="1", BIGMOE_ASYNC_EVERY="1", BIGMOE_PF_JIT="1",
           BIGMOE_PF_QD="4", BIGMOE_PREFETCH="12", BIGMOE_DUPROW_CHUNK="5", BIGMOE_MAXK="80", BIGMOE_FRSPEC="1",
           BIGMOE_SCOUT_BUNDLE="1", MTPLX_QWEN4_VERIFY_GLUE="1", BIGMOE_MTP_STREAM="1", MTPLX_NGRAM_PREWARM="off",
           MTPLX_FAMILY_CAPTURE_COMMIT="1", MTPLX_DROP_EVENTS="1", MTPLX_LAZY_MTP_HISTORY_APPEND="1",
           MTPLX_DEFER_REPAIR_EVAL="1", MTPLX_COMPILED_VERIFY="0", MTPLX_QWEN4EXP_COMPILE="0", MTPLX_COMPILE_AR_FORWARD="0",
           MTPLX_COMPILED_TARGET_PREFIX="0", BIGMOE_GDN_EXACT="1",
           BIGMOE_TIER="4433333333", BIGMOE_TIER_GLU="3c", BIGMOE_TIER_COMBINE="1", BIGMOE_OOO="1")
ARM = os.environ.get("DH_ARM", "v3")
W = int(os.environ.get("DH_W", "4"))
assert ARM in ("v3", "dh", "dh10", "force", "ref"), ARM
if ARM != "v3":
    ENV.update(BIGMOE_DH=str(10 if ARM == "dh10" else W), BIGMOE_DEFER="1", BIGMOE_DEFER_FULL="1")
    if ARM in ("force", "ref"):
        ENV["BIGMOE_DH_FORCE" if ARM == "force" else "BIGMOE_DH_REF"] = "1"
if sys.argv[1] != "gen":                                      # no MTP loaded: MTP_STREAM would shift the sidecar ids
    ENV.pop("BIGMOE_MTP_STREAM")                              # (stream.install_sidecars asserts side0 = 49)
if os.environ.get("DH_TRACE") == "1":
    ENV["BIGMOE_TRACE"] = "1"
if os.environ.get("DH_NOJIT") == "1":                          # the loader turns JIT on if the variable exists at all
    ENV.pop("BIGMOE_PF_JIT"); os.environ.pop("BIGMOE_PF_JIT", None)
for k, v in ENV.items():
    os.environ.setdefault(k, v)
MODE = sys.argv[1]
FRAC = float(os.environ.get("DH_FRAC", "0.25"))
sys.path.insert(0, os.path.expanduser("~/MoonShots/defer/" + os.environ.get("DH_ENGINE", "v3d")))   # v3d2 = + seq stamps
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import numpy as np
import mlx.core as mx
from bigmoe_v3d.loader import load_streamed
from bigmoe_v3d.gated import ext
from bigmoe_v3d import tier as T
import importlib, bigmoe_v3d                                 # Scout's bundle (sharedfirst, vreadchunk) and spec125 import
sys.modules.setdefault("bigmoe", bigmoe_v3d)                  # bigmoe.gated / .fuse / .repguard: alias them to this copy, as
for _m in ("gated", "stream", "tier", "fuse", "decode", "hcpack", "repguard", "loader"):   # BigMoE's package is in the
    sys.modules.setdefault(f"bigmoe.{_m}", importlib.import_module(f"bigmoe_v3d.{_m}"))   # deployed setup

OUT = os.path.expanduser("~/MoonShots/data/defer")
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/BigMoE/models/bare")
C = max(16, round(FRAC * 512))
tag = (ARM + (f"_w{W}" if ARM in ("dh", "force", "ref") and W != 4 else "") + (f"_f{FRAC}" if FRAC != 0.25 else "")
       + ("_d1" if os.environ.get("BIGMOE_DH_D") == "1" else "") + ("_nojit" if os.environ.get("DH_NOJIT") == "1" else "")
       + os.environ.get("DH_TAG", ""))                        # e.g. DH_TAG=_pipe100 (BIGMOE_EXT_BUILD=build_pipe)
t0 = time.time()
rt, st = load_streamed(MD, C, mtp=(MODE == "gen"))
model = rt.model
if ARM != "v3":
    T.install_dh(model)
print(f"[{tag}] loaded in {time.time() - t0:.0f} s; C={C}; DH={os.environ.get('BIGMOE_DH', 'off')} "
      f"force={os.environ.get('BIGMOE_DH_FORCE', '0')} ref={os.environ.get('BIGMOE_DH_REF', '0')}", flush=True)
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
PROMPTS = {                                                    # verbatim from SpecDecode's test_spec125.py (anatomy.py)
    "code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
    "edit": lambda: ("Here is a file:\n\n```python\n" + open(os.path.expanduser("~/SpecDecode/scripts/inventory_sample.py")).read()
                     + "```\n\nRewrite the whole file with total_value renamed to total everywhere. Output only the full file in one code block."),
    "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs faster. "
                      "Use one everyday analogy and keep it to about 300 words.")}


def chat_ids(text):
    ids = rt.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


if MODE == "gen":
    import spec125
    PROMPT = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    ids = chat_ids(PROMPTS[PROMPT]())
    KEYS = ("uses", "misses", "stalled", "gated", "io_s", "cpu_wait_s", "proc_s", "slow_calls", "slow_s", "pf_issued",
            "pf_used", "pf_ontime", "pf_late", "pf_waited_s", "miss_unpred", "miss_pred_unissued")
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

    spec125.generate(model, ids, 16, k_mtp=3, max_draft=5, lookup=1)                    # warm-up, as anatomy.py
    st.gate.check()
    V0 = {1: 42, 2: 59, 3: 74, 4: 87, 5: 94, 6: 107, 7: 117, 8: 123}
    aw = spec125.AdaptiveWidth(V0, step_ms=2.2, alpha=2.0)
    hook()
    if os.environ.get("DH_TRACE") == "1":
        ext.trace()
    stats = {"trace": [], "pass_hook": hook}
    REPLAY = os.environ.get("DH_REPLAY")     # A7 (MoonShots 10-23): replay a recorded gen JSON's drafts, MTP head off
    if REPLAY:
        RD = [r["d"] for r in json.load(open(REPLAY))["passes"]]

        class _Replay:                                                # stands in for spec125's PromptLookup: pass i's drafts
            last_n = 0

            def __init__(self, *a):
                self.i = 0

            def start(self, ids):
                pass

            def extend(self, toks):
                pass

            def draft(self):
                self.i += 1
                return list(RD[self.i - 1]) if self.i <= len(RD) else []
        spec125.PromptLookup = _Replay
    t1 = time.perf_counter()
    if REPLAY:
        out, s = spec125.generate(model, ids, N, k_mtp=0, max_draft=7, lookup=1, stats=stats)
    else:
        out, s = spec125.generate(model, ids, N, k_mtp=1, max_draft=7, lookup=1, stats=stats, adapt=aw)
    wall = time.perf_counter() - t1
    st.gate.check()
    recs = [{k: (v if isinstance(v, (int, float)) or k == "d" else None) for k, v in r.items()
             if k in ("S", "n", "t_verify", "d") or k.startswith("x_")} for r in stats["trace"]]
    summ = {k: s[k] for k in ("passes", "time", "tok_s", "tok_per_pass", "t_verify", "t_draft", "t_build", "t_wait", "t_commit",
                              "t_head", "t_dwait", "lookup_used") if k in s}
    toks = [int(v) for v in out]
    res = dict(prompt=PROMPT, arm=ARM, W=W, N=N, frac=FRAC, C=C, wall_s=wall, summary=summ, passes=recs, sha=sha(toks),
               tokens=toks, prompt_tokens=len(ids), dh_calls=T.DH["calls"],
               ext_trace=ext.trace() if os.environ.get("DH_TRACE") == "1" else None)
    json.dump(res, open(f"{OUT}/gen_{PROMPT}_{tag}{'_trace' if os.environ.get('DH_TRACE') == '1' else ''}.json", "w"))
    late = sum(r["x_late"] for r in recs); mis = sum(r["x_misses"] for r in recs)
    print(f"[{PROMPT}_{tag}] {s['tok_s']:.1f} tok/s, {s['tok_per_pass']:.2f} tok/pass, {s['passes']} passes, verify "
          f"{1e3 * s['t_verify'] / max(1, s['passes']):.1f} ms/pass, draft {s['t_draft']:.2f} s; reads {mis} ({late} late, "
          f"{sum(r['x_waited'] for r in recs)} waited in flag rows), stalled {sum(r['x_stalled'] for r in recs)}, io "
          f"{sum(r['x_io_s'] for r in recs):.2f} s; sha {sha(toks)} | {rt.tokenizer.decode(toks)[:90]!r}", flush=True)
    xs = ext.stats()
    if "stamp_calls" in xs:
        print(f"[stamps] calls {xs['stamp_calls']}, re-reads {xs['stamp_rereads']}, wait {1e3 * xs['stamp_wait_s']:.2f} ms, "
              f"spin stale pairs {int(np.frombuffer(memoryview(st.gate.status), np.int32)[17])}", flush=True)

elif MODE == "tf":
    from mtplx.attention_context import attention_phase
    NT = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    S = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ids = chat_ids(PROMPTS["code"]())
    P0 = len(ids) - NT
    cache = model.make_cache()
    with attention_phase("prefill"):                                    # as spec125._prefill
        for i in range(0, P0, 2048):
            lg = model(mx.array([ids[i:min(P0, i + 2048)]]), cache=cache)
            mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    with attention_phase("decode_verify"):
        logits = []
        for i in range(P0, len(ids), S):
            lg = model(mx.array([ids[i:i + S]]), cache=cache)
            lg = (lg[0] if isinstance(lg, tuple) else lg)[0].astype(mx.float32)
            mx.eval(lg)
            logits.append(np.array(lg))
    st.gate.check()
    L_ = np.concatenate(logits)
    np.savez(f"{OUT}/tf_{tag}_S{S}.npz", logits=L_, ids=np.array(ids[P0:]))
    print(f"[tf_{tag}_S{S}] {L_.shape[0]} positions (S={S}), top-1 sha {sha(L_.argmax(-1).tolist())}, dh calls {T.DH['calls']}, "
          f"defer stats {list(ext.defer_stats())}", flush=True)

elif MODE == "fixed":
    from mtplx.attention_context import attention_phase
    from mtplx.models.qwen4_exp import QSACache
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    NT = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    WARM = 32
    ids = rt.tokenizer.encode(open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:12000])[:WARM + NT + 8]
    cache = model.make_cache()
    snap = lambda: [("qsa", c.offset) if isinstance(c, QSACache) else ("arr", list(c.state)) for c in cache]

    def restore(sv):
        for c, (kind, v) in zip(cache, sv):
            if kind == "qsa":
                c.trim(int(c.offset) - v)
            else:
                c.state = list(v)
    times = []
    with attention_phase("decode_verify"):
        SV = None
        for s in range(0, WARM + NT, S):
            if s == WARM - S:
                SV = snap()
            if s >= WARM:
                restore(SV)
            c = mx.array([ids[s:s + S]]) if s < WARM else mx.array([ids[WARM - S:WARM]])
            a = time.perf_counter()
            lg = model(c, cache=cache)
            top = mx.argmax((lg[0] if isinstance(lg, tuple) else lg), axis=-1)
            mx.eval(top)
            if s >= WARM:
                times.append((time.perf_counter() - a) * 1e3)
    st.gate.check()
    t = np.array(times)
    print(f"[fixed_{tag}] S={S}: verify pass median {np.median(t):.2f} ms, mean {t.mean():.2f} (p10 {np.percentile(t, 10):.2f}, "
          f"p90 {np.percentile(t, 90):.2f}) over {len(t)} all-hit passes; defer stats {list(ext.defer_stats())}; "
          f"misses {ext.stats()['misses']}", flush=True)
