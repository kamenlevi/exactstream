"""B8 + MTP with the multi-row phase B (b8mr/mrlate.py) vs today's wait-all verify, on Breakthrough's real v3f path
(bt_b12.load_b12(0.05) + B12_LATE3=2,4 + B12_NORM=mass, imported read-only; HC 8-bit mixers as shipped).
ARM (env MR_ARM): wait (bt_b12 as is: verify rows wait for every forced rank) | mr (verify rows of 2..4 defer cold ranks 2-3)
| mr1 (mr also takes 1-row calls: bit-exact test vs wait at S=1) | force (every rank 2-3 late) | ref (force's numerics with
synchronous late slots; force == ref bit for bit). Phase-B pair capacity BIGMOE_DEFER = 8 in every arm (bt_b12 sets 2).
MODE:
  tf1 [N=32]          teacher-forced 1-row steps over the code prompt's last N tokens -> data/b8mr/tf1_ARM.npz (logits)
  tfs [N=32] [S=4]    teacher-forced S-row chunks -> data/b8mr/tfS_ARM.npz
  kld [EVERY=8] [S=4] the Eval suite's own-output tf sequences [::EVERY] (per set), prompt prefilled, completion in S-row
                      teacher-forced chunks, KLD vs the suite's exact 4-bit reference (engine.var_head) -> data/b8mr/kld_ARM_eE.npz
  gen PROMPT [N=192]  spec125 MTP 3:3:0 (B8's verified 17:47 setting: MTP_C 32, PF_GROUP 64, d1i prefilled state), per-pass
                      S, verify ms, reads, waited, late -> data/b8mr/gen_PROMPT_ARM.json
usage: run.py MODE ...   (GPU, ~9 GB; quiet lock)"""
import hashlib, json, os, sys, time
ARM = os.environ.get("MR_ARM", "wait")
assert ARM in ("wait", "mr", "mr1", "force", "ref"), ARM
MODE = sys.argv[1]
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
os.environ.setdefault("B12_LATE3", "2,4")                         # B8 / B12L; B12LL = 2,10
os.environ.setdefault("B12_NORM", "mass")
if MODE == "kld":
    os.environ["EVALSUITE_FRUGAL"] = "1"                         # the suite's scorer: none of the speed switches it refuses
else:
    for k, v in (("BIGMOE_ASYNC_EVERY", "1"), ("BIGMOE_MAXK", "80"), ("BIGMOE_PF_GROUP", "64"), ("D1I_ATTN", "512")):
        os.environ.setdefault(k, v)
    if MODE == "gen":
        for k, v in (("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "32"), ("BIGMOE_FRSPEC", "1"),
                     ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
                     ("MTPLX_DEFER_REPAIR_EVAL", "1")):
            os.environ.setdefault(k, v)
os.environ.setdefault("BIGMOE_MAXK", "80")                       # SMAX = 80 // 20 = 4 verify rows on the frugal path
if ARM == "mr1":
    os.environ["MR_MIN"] = "1"
if ARM in ("force", "ref"):
    os.environ["MR_TEST"] = ARM
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/MoonShots/b8mr"))
import bt_b12 as B
_env0 = B._env


def _env():                                                      # bt_b12 sets BIGMOE_DEFER = 2 (1-row pairs); multi-row
    _env0()                                                      # verify needs Mp x SMAX = 8 pairs (same in every arm)
    os.environ["BIGMOE_DEFER"] = "8"


B._env = _env
t0 = time.time()
FRAC = float(os.environ.get("MR_FRAC", "0.05"))                # 0.05 = B8; 0.10 = B12 + late ranks 2-3
rt, st = B.load_b12(FRAC, mtp=(MODE == "gen"))
import numpy as np
import mlx.core as mx
from bigmoe_v3f import tier as T
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state
from mtplx.attention_context import attention_phase
model, tok = rt.model, rt.tokenizer
import mtplx.models.qwen4_exp as _Q
_lc = _Q.DecoderLayer.__call__                                   # bt_b12's layer_call (late writes)


def _lc_fixed(self, hidden, *, input_ids, ssm_mask, cache):
    """FIX (both arms; reported to Breakthrough 20:3x): MTPLX's capture-commit (commit_verified_window) replays the PLE
    layer (trunk layer 1) ALONE for the kept rows after a partial accept; through bt_b12's layer_call its MoE stashes a late
    entry for layer 2 that no layer consumes, and the next verify applies it at layer 2 (crash when the row counts differ,
    a stale late part when they match). The replay only rebuilds layer 1's cache, so its MLP output (and late part) is
    unused: drop leftover late writes at the start of every trunk forward (layer 0)."""
    if getattr(self, "_b12_L", None) == 0:
        B.PENDH.clear(); B.STASH.clear()
    return _lc(self, hidden, input_ids=input_ids, ssm_mask=ssm_mask, cache=cache)


_Q.DecoderLayer.__call__ = _lc_fixed
MR = None
if ARM != "wait":
    import mrlate
    MR = mrlate.install(mx, B, T, model)
OUT = os.path.expanduser("~/MoonShots/data/b8mr")
os.makedirs(OUT, exist_ok=True)
TAG = os.environ.get("MR_TAG", "")                               # output-name suffix (stress conditions, A7 arms)
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
print(f"[b8mr {ARM}] loaded in {time.time() - t0:.0f}s; DEFER={os.environ.get('BIGMOE_DEFER')} MAXK={os.environ.get('BIGMOE_MAXK')} "
      f"MR_MIN={os.environ.get('MR_MIN', '2')} MR_TEST={os.environ.get('MR_TEST', '')}", flush=True)
PROMPTS = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes programs "
                             "faster. Use one everyday analogy and keep it to about 300 words.")}


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


def fresh():
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()


def prefill(ids):
    """plain chunked prefill of ids into a fresh cache (S > SMAX: v3's exact tier path in every arm)"""
    cache = model.make_cache()
    with attention_phase("prefill"):
        for i in range(0, len(ids), 2048):
            lg = model(mx.array([ids[i:i + 2048]]), cache=cache)
            mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    return cache


if MODE == "denseinv":                                            # NewIdeas line 1 (350 on code, 8-16-row passes): MLX's
    import mlx.nn as nn                                              # dense kernels on the model's real dense layers, M = 1..16:
    groups = {}                                                      # is every row bitwise equal to its M=1 / dup-row M=2 result
    for name, mod in model.named_modules():                          # (batch invariance), and what do plain vs chunked cost
        if isinstance(mod, nn.QuantizedLinear):                      # (each shape's instances round-robin: DRAM, not cache)
            key = ("q%d" % mod.bits, mod.weight.shape[0], mod.weight.shape[1] * 32 // mod.bits, mod.group_size)
        elif isinstance(mod, nn.Linear):
            key = ("bf16", mod.weight.shape[0], mod.weight.shape[1], 0)
        else:
            continue
        groups.setdefault(key, []).append((name, mod))

    def op(mod, x):
        if isinstance(mod, nn.QuantizedLinear):
            y = mx.quantized_matmul(x, mod.weight, mod.scales, mod.biases, transpose=True, group_size=mod.group_size,
                                    bits=mod.bits)
        else:
            y = x @ mod.weight.T
        return y + mod.bias if "bias" in mod else y
    res = []
    for key, mods in sorted(groups.items(), key=lambda kv: -kv[0][1] * kv[0][2] * len(kv[1])):
        kind, N, K, gs = key
        m0 = mods[0][1]
        X = (mx.random.normal((16, K), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
        R1 = [op(m0, X[r:r + 1]) for r in range(16)]
        R2 = [op(m0, mx.concatenate([X[r:r + 1], X[r:r + 1]]))[:1] for r in range(16)]
        mx.eval(R1, R2)
        R1 = [np.array(a.view(mx.uint16)) for a in R1]; R2 = [np.array(a.view(mx.uint16)) for a in R2]
        inv1, inv2 = {}, {}
        for M in range(1, 17):
            Y = op(m0, X[:M]); mx.eval(Y); Y = np.array(Y.view(mx.uint16))
            inv1[M] = all(np.array_equal(Y[r:r + 1], R1[r]) for r in range(M))
            inv2[M] = all(np.array_equal(Y[r:r + 1], R2[r]) for r in range(M))
        band = max(c for c in range(2, 17) if all(inv2[m] for m in range(2, c + 1))) if inv2[2] else 1
        inst = [m for _, m in mods][:48]
        tm = {}
        for M in (1, 2, 4, 5, 8, 12, 16):
            for chunk in ((None, band) if M > band else (None,)):
                best = 1e9
                for rep in range(6):
                    outs = []
                    for i in range(48):
                        x = X[:M]
                        parts = [x[a:a + chunk] for a in range(0, M, chunk)] if chunk else [x]
                        outs += [op(inst[i % len(inst)], p if p.shape[0] > 1 else mx.concatenate([p, p])) for p in parts]
                    t0 = time.perf_counter(); mx.eval(outs)
                    if rep:
                        best = min(best, time.perf_counter() - t0)
                tm[f"{M}{'c%d' % chunk if chunk else ''}"] = best / 48 * 1e6
        r = dict(kind=kind, N=N, K=K, gs=gs, count=len(mods), names=[n for n, _ in mods[:3]], band=band,
                 inv1=[M for M in range(1, 17) if inv1[M]], inv2=[M for M in range(1, 17) if inv2[M]], us=tm)
        res.append(r)
        print(f"[denseinv] {kind} {N}x{K} x{len(mods)} ({mods[0][0]}): dup-row band 2..{band}; ==M1 rows at M "
              f"{r['inv1']}; ==M2 at M {r['inv2']}; us/call " + " ".join(f"{k}:{v:.0f}" for k, v in tm.items()), flush=True)
    json.dump(res, open(f"{OUT}/denseinv{TAG}.json", "w"), indent=1)

elif MODE == "tgtest":                                            # B8 load divergence: is tier_glu itself bit-stable?
    ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000          # fixed inputs per trunk layer (4 rows, frugal-style
    ids = chat_ids(PROMPTS["code"]())                                # slots: 10 resident ranks, the last 2 padding at weight
    fresh()                                                          # 0 on rank 0's slot), fused combine (scores=, as the
    cache = prefill(ids[:len(ids) - 64])                             # frugal block) and unfused + MLX sum (v3-like);
    with attention_phase("decode_verify"):                           # 16 verify-shaped chunks fill the decode pools (v3g's
        for i in range(len(ids) - 64, len(ids), 4):                  # prefill may leave them empty)
            lg = model(mx.array([ids[i:i + 4]]), cache=cache)
            mx.eval(lg[0] if isinstance(lg, tuple) else lg)
    TG = T._tier_glu()                                               # every call vs its first output, bit for bit
    PLAN = "".join(T.PLAN)
    TMm = model.language_model.model if hasattr(model, "language_model") else model.model
    rng = np.random.default_rng(0)
    cases = []
    for L, lay in enumerate(TMm.layers):
        bm = getattr(lay.mlp.switch_mlp, "_bm", None) if hasattr(lay, "mlp") and hasattr(lay.mlp, "switch_mlp") else None
        if bm is None or "tier" not in bm:
            continue
        t = bm["tier"]
        rb = [s for s in ext.lookup(L, list(range(512))) if s >= 0]
        rl = [s for s in ext.lookup(t["side"], list(range(512))) if s >= 0]
        if len(rb) < 10 or len(rl) < 2:
            continue
        s10 = np.array([rng.permutation(rb)[:10] for _ in range(4)], np.int32)
        s10[:, 8:] = s10[:, :1]                                      # padding: rank 0's slot at weight 0
        l2 = np.array([rng.permutation(rl)[:2] for _ in range(4)], np.int32)
        w = rng.random((4, 10)).astype(np.float32); w[:, 8:] = 0; w /= w.sum(-1, keepdims=True)
        g = lay.mlp.gate                                             # true input width (packed if quantized)
        KX = g.weight.shape[-1] * 32 // g.bits if hasattr(g, "bits") else g.weight.shape[-1]
        x2 = (mx.random.normal((4, KX), key=mx.random.key(L)) * 0.5).astype(mx.bfloat16)
        cases.append((L, t, x2, mx.array(s10), mx.array(l2), mx.array(w)))

    def calls(fused):
        if fused:
            return [TG(x2, t["base"], t["sideP"], s10, l2, plan=PLAN, scores=w) for _, t, x2, s10, l2, w in cases]
        return [(TG(x2, t["base"], t["sideP"], s10, l2, plan=PLAN) * w.astype(mx.bfloat16)[..., None]).sum(-2)
                for _, t, x2, s10, l2, w in cases]
    for fused in (True, False):
        ref = calls(fused); mx.eval(ref)
        ref = [np.array(r.view(mx.uint16)) for r in ref]
        bad, first, t1 = 0, None, time.time()
        for it in range(ITERS):
            ys = calls(fused); mx.eval(ys)
            for (L, *_), y, r in zip(cases, ys, ref):
                if not np.array_equal(np.array(y.view(mx.uint16)), r):
                    bad += 1
                    first = first or (it, L)
        print(f"[tgtest{TAG}] {'fused combine' if fused else 'unfused + MLX sum'}: {ITERS} x {len(cases)} calls, "
              f"{bad} differ from the first output (first at iteration / layer {first}), {time.time() - t1:.0f}s", flush=True)

elif MODE in ("tf1", "tfs"):
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    S = 1 if MODE == "tf1" else (int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    ids = chat_ids(PROMPTS["code"]())
    P0 = len(ids) - N
    OBS = None
    if os.environ.get("MR_OBS"):                                     # per-call record (b8mr/obs.py), verify chunks only
        import obs
        OBS = obs.Obs(st.gate, _Q, mx)
    fresh()
    cache = prefill(ids[:P0])
    logits = []
    if OBS:
        OBS.on = True
    with attention_phase("decode_verify"):
        for i in range(P0, len(ids), S):
            lg = model(mx.array([ids[i:i + S]]), cache=cache)
            lg = (lg[0] if isinstance(lg, tuple) else lg)[0].astype(mx.float32)
            mx.eval(lg)
            logits.append(np.array(lg))
            if OBS:
                OBS.flush((i - P0) // S)
    st.gate.check()
    if OBS:
        OBS.save(f"{OUT}/obs_{MODE}{S}_{ARM}{TAG}.pkl")
    L_ = np.concatenate(logits)
    tag = f"{MODE}{S if MODE == 'tfs' else ''}_{ARM}" + TAG
    np.savez(f"{OUT}/{tag}.npz", logits=L_)
    print(f"[{tag}] {L_.shape[0]} positions, logits sha {hashlib.sha1(L_.tobytes()).hexdigest()[:10]}, top-1 sha "
          f"{sha(L_.argmax(-1).tolist())}, mr calls "
          f"{MR['calls'] if MR else 0}, defer stats {list(ext.defer_stats())}", flush=True)

elif MODE == "kld":
    EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    S = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
    from evalsuite import stages as SG, engine as EN
    items = SG.tf_items()
    head = EN.var_head(SG.load_ref_tf(items), SG._targets(items))
    SETS = ["tf_gsm", "tf_code", "tf_ifeval", "tf_multi"]
    seqs = [i for s in SETS for i in [j for j, it in enumerate(items) if it["set"] == s][::EVERY]]
    res = {s: {"kld": [], "agree": [], "seq": []} for s in SETS}
    t1 = time.time()
    for i in seqs:
        it = items[i]
        p0, pos = int(it["pos"][0]), list(it["pos"])
        ids = it["ids"]
        fresh()
        cache = prefill(ids[:p0])
        with attention_phase("decode_verify"):
            for a in range(0, len(pos), S):
                chunk = pos[a:a + S]                                  # positions chunk[0]..; feed ids at those positions
                lg = model(mx.array([ids[chunk[0]:chunk[-1] + 1]]), cache=cache)
                lg = (lg[0] if isinstance(lg, tuple) else lg)[0]
                r = head(i, np.array(chunk), lg)
                mx.eval(r["kld"], r["agree"])
                res[it["set"]]["kld"].append(np.array(r["kld"])); res[it["set"]]["agree"].append(np.array(r["agree"]))
                res[it["set"]]["seq"].append(np.full(len(chunk), i))
        st.gate.check()
    out = {}
    line = []
    for s, d in res.items():
        if d["kld"]:
            k = np.concatenate(d["kld"]); a = np.concatenate(d["agree"]); q = np.concatenate(d["seq"])
            out[f"kld_{s}"], out[f"agree_{s}"], out[f"seq_{s}"] = k, a, q
            line.append(f"{s} {k.mean():.5f} (top-1 {a.mean():.3f}, {len(k)} pos)")
    np.savez(f"{OUT}/kld_{ARM}_e{EVERY}_S{S}.npz", **out)
    print(f"[kld_{ARM}_e{EVERY}_S{S}] " + " | ".join(line) + f" | {len(seqs)} seqs, {time.time() - t1:.0f}s; mr calls "
          f"{MR['calls'] if MR else 0}", flush=True)

elif MODE == "gen":
    sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
    import spec125
    P = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 192
    ids = chat_ids(PROMPTS[P]())
    RESID = []                               # MR_RESID=1: every pool's residency (ext.lookup) after prefill, warm-up, each pass
    NGL = 0
    if os.environ.get("MR_RESID"):
        while True:
            try:
                ext.lookup(NGL, [0]); NGL += 1
            except Exception:
                break

    def resid(tag):
        if NGL:
            RESID.append((tag, np.array([ext.lookup(g, list(range(512))) for g in range(NGL)], np.int16)))
    state = B.b12_prefill_state(model, ids, int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096")))
    resid("prefill")
    KEYS = ("misses", "stalled", "io_s", "uses")
    last = {}

    def hook():
        s = ext.stats(); d = list(ext.defer_stats())
        out = {"x_" + k: s[k] - last.get(k, 0) for k in KEYS}
        dd = [a - b for a, b in zip(d, last.get("_d", [0] * len(d)))]
        last.update({k: s[k] for k in KEYS}); last["_d"] = d
        out.update(x_rows=dd[0], x_waited=dd[1], x_late=dd[2], x_over=dd[3])
        resid("pass")
        if CS:
            CS.flush()
        return out

    REPLAY = os.environ.get("MR_REPLAY")     # A7: replay a recorded gen JSON's drafts with the MTP head off (same passes)

    def run(n, stats=None, replay=False):
        fresh()
        out = []
        t_first = {}

        def cb(tk):
            if not t_first:
                t_first["t"] = time.perf_counter()
            out.append(int(tk))
        s = stats if stats is not None else {}
        spec125.generate(model, ids, n, k_mtp=0 if replay else 3, max_draft=3, lookup=1 if replay else 0, stats=s,
                         on_token=cb, prefilled=state)
        st.gate.check()
        return out, s, time.perf_counter() - t_first["t"]

    CS = None
    if os.environ.get("MR_CSUM"):                                  # per-op checksums (b8mr/csum.py)
        import csum
        CS = csum.CSum(mx, _Q, st.gate, ext=ext if os.environ.get("MR_HANDOFF") else None)
    run(16)                                                           # warm-up (MTP on in every arm: same pools after it)
    if CS:
        CS.reset()
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
    hook()
    stats = {"trace": [], "pass_hook": hook}
    out, s, tsec = run(N, stats, replay=bool(REPLAY))
    recs = [{k: (v if isinstance(v, (int, float)) or k == "d" else None) for k, v in r.items()
             if k in ("S", "n", "t_verify", "d") or k.startswith("x_")} for r in stats["trace"]]
    nt = max(1, len(out) - 1)
    summ = {k: s[k] for k in ("passes", "time", "tok_per_pass", "t_verify", "t_draft", "t_head", "t_dwait", "t_build", "t_wait",
                              "t_commit") if k in s}
    res = dict(prompt=P, arm=ARM, N=N, tok_s=nt / tsec, tok_per_pass=s.get("tok_per_pass"), passes=recs, sha=sha(out),
               tokens=out, mr_calls=MR["calls"] if MR else 0, mr_rows=MR["rows"] if MR else 0, summary=summ, replay=REPLAY)
    FT = "" if FRAC == 0.05 else f"_f{FRAC}"
    res["frac"] = FRAC
    json.dump(res, open(f"{OUT}/gen_{P}_{ARM}{FT}{TAG}.json", "w"))
    if CS:
        json.dump(CS.passes, open(f"{OUT}/csum_{P}_{ARM}{FT}{TAG}.json", "w"))
        if CS.ext is not None:
            json.dump(CS.bad, open(f"{OUT}/handoff_{P}_{ARM}{FT}{TAG}.json", "w"))
            print(f"[handoff{TAG}] {P}: {sum(b[0] == 'ids' for b in CS.bad)} calls where the loader read other ids than "
                  f"posted, {sum(b[0] == 'slots' for b in CS.bad)} where the spin released other slots than published; "
                  f"first {CS.bad[:3]}", flush=True)
    xs = ext.stats()
    if "stamp_calls" in xs:                      # engine v3h: seq-stamp re-reads (torn routing rows caught)
        print(f"[stamps{TAG}] calls {xs['stamp_calls']}, re-reads {xs['stamp_rereads']}, wait {1e3 * xs['stamp_wait_s']:.2f} ms, "
              f"joint declines {xs.get('stamp_joint', 0)}, spin stale pairs {int(np.frombuffer(memoryview(st.gate.status), np.int32)[17])}", flush=True)
    if RESID:                                # [prefill, after warm-up, after pass 0, 1, ...] x gate layers x 512 slots
        np.savez(f"{OUT}/resid_{P}_{ARM}{FT}{TAG}.npz", R=np.stack([r for _, r in RESID]))
    P1 = recs[1:] or recs
    mean = lambda k: float(np.mean([r[k] for r in P1]))
    np_ = max(1, s["passes"])
    print(f"[{P}_{ARM}{FT}{TAG}] {nt / tsec:.1f} tok/s, {s.get('tok_per_pass', float('nan')):.2f} tok/pass, {len(recs)} passes; per pass: "
          f"S {mean('S'):.2f}, verify {1e3 * mean('t_verify'):.1f} ms, reads {mean('x_misses'):.1f}, waited {mean('x_waited'):.1f}, "
          f"late {mean('x_late'):.1f}, io {1e3 * mean('x_io_s'):.1f} ms; loop {1e3 * s['time'] / np_:.1f} ms/pass (draft "
          f"{1e3 * s['t_draft'] / np_:.1f}, head {1e3 * s['t_head'] / np_:.1f}); mr calls {res['mr_calls']} ({res['mr_rows']} rows); "
          f"sha {res['sha']} | {tok.decode(out)[:70]!r}", flush=True)
