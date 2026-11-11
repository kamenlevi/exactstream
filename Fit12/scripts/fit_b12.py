"""Fit12 (2026-10-23): speed levers for B12LL (B12 + B12_LATE3=2,10 + mass + MTP), measured as the Eval suite runs it.
Imports Breakthrough's bt_b12 (read-only); every lever is a patch inside this process:
  F12_JIT=W         JIT look-ahead prefetch: BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=W; the frugal block's base-pool
                    post carries layer L+1's top-W experts per row (stream.py's d=1 look-ahead). Residency / timing only.
  F12_BUDGET=b16    per-pass expert budget (copy of BigMoE budget.py): 2..8-row calls take each row's top-10 within the
                    pass set (NON-EXACT, FM1 semantics). F12_BUDGET_JIT=1: look-ahead routers budget-masked too.
  F12_ASYNC=1       BIGMOE_ASYNC_EVERY=1 (hand each layer's graph to the GPU while Python builds the rest).
  F12_V3F=path      another v3f tree (e.g. ~/Fit12/v3g: BIGMOE_EVICT_LFU).
  F12_OOO=1         out-of-order experts on the verify rows (fit_ooo.py; exact, shas must equal the plain run).
  F12_GDNX=1        MoonShots' exact fused GDN conv+norm and gated norm (BigMoE's BIGMOE_GDN_EXACT; bitwise).
Arms: F12_ARMS="code:3,prose:3,..." = prompt:k (spec125 k:k:0, lookup off, guard off), BT_N tokens after a 16-token warm-up.
Usage: [BigMoE venv] fit_b12.py TAG"""
import ctypes, hashlib, os, sys, time

ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0",       # cli.SHIP_AR
       "B12_LATE3": "2,10", "B12_NORM": "mass",                                                      # cli.B12_ENV b12ll
       "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1",     # cli.SHIP_MTP
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64"}
for k, v in ENV.items():
    os.environ.setdefault(k, v)
if os.environ.get("F12_ASYNC") == "1":
    os.environ["BIGMOE_ASYNC_EVERY"] = "1"
JITW = int(os.environ.get("F12_JIT", "0") or 0)
BUD, BJIT = os.environ.get("F12_BUDGET", ""), os.environ.get("F12_BUDGET_JIT") == "1"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(2, os.path.expanduser("~/Fit12/scripts"))
import bt_b12 as B
B.KC = int(os.environ.get("F12_KC", "20"))   # 10 = B12LL exactly (ranks 10-19 are never used: K - NF = 0); lets S reach 8
if os.environ.get("F12_V3F"):
    B.V3F = os.path.expanduser(os.environ["F12_V3F"])
_env0 = B._env


def _env():                                    # bt_b12 pops PF_JIT and zeroes spares; JIT here uses victim slots
    _env0()
    if os.environ.get("F12_WIRED_GB"):          # RES3: wire the GPU pools (MLX wired limit) before load_b12 allocates them
        import mlx.core as _mx
        _prev = _mx.set_wired_limit(int(float(os.environ["F12_WIRED_GB"]) * 1e9))
        print(f"[fit12] mx.set_wired_limit({os.environ['F12_WIRED_GB']} GB), previous {_prev / 1e9:.1f} GB", flush=True)
    if JITW:
        os.environ.update(BIGMOE_PF_JIT="1", BIGMOE_PREFETCH=str(JITW))
        os.environ.setdefault("BIGMOE_PF_QD", "4")
    if os.environ.get("F12_SPARE"):            # spare slots out of C (stream.py: C_main = C - spare): prefetch without victims
        os.environ["BIGMOE_SPARE"] = os.environ["F12_SPARE"]
B._env = _env
if os.environ.get("F12_HC8LW") == "1":         # layer-wise hc8 install (bitwise the same weights, smaller load transient)
    import fit_hc8lw
    fit_hc8lw.apply(B.V3F)
import fit_trim                                # L2-02/03: F12_EMBED=1 (streamed embed), F12_HCPRE=1 (hc8 before the eval)
fit_trim.from_env(B)
import fit_ooo                                 # O1: F12_OOO=1, out-of-order experts on the verify rows (exact)
fit_ooo.apply(B)
import fit_dup                                 # GP3: F12_DUP=tg,rt,sh,sel,k10 (duplicate work on the verify rows, exact)
fit_dup.apply(B)
import fit_topk                                # TK1: F12_TOPK=k / F12_TOPP=p, fewer experts per verify row (NON-EXACT)
fit_topk.apply(B)

rt, st = B.load_b12(float(os.environ.get("F12_FRAC", "0.10")), mtp=True)   # F12_FRAC: residency (0.375 = C_m 192)
if os.environ.get("F12_GDNX") == "1":            # MoonShots' exact GDN conv+silu+l2norm + gated norm (BigMoE 40ed49f,
    sys.path.insert(3, os.path.expanduser("~/MoonShots/patches/planes_engine"))   # bitwise; read-only import, after
    import gdn_exact_patch                     # load_b12 so no MTPLX module is imported before its _env())
    gdn_exact_patch.install()
    os.environ["MTPLX_FUSED_GDN_CONVNORM"] = "1"; os.environ["MTPLX_FUSED_CONVNORM_VERIFY"] = "1"
    print("fit12: exact GDN (MoonShots gdn_exact_patch) installed", flush=True)
import fit_gdnr                                # GR1: F12_GDNR=1, Leap's exact recurrence_r4 for mlx_lm's gated_delta_kernel
fit_gdnr.apply()
import fit_pr1                                 # PR1: F12_PR1=1, keep-1 commits replay the PLE layer on 2 rows (multi-row state)
import mlx.core as mx
import spec125
from mtplx.models import qwen4_exp as Q
from bigmoe_v3f import gated as Gm
from bigmoe_v3f.gated import ext
from bigmoe_v3f.fuse import reset_generation_state
model, tok = rt.model, rt.tokenizer
fit_pr1.apply(Q)
import fit_lr1                                 # LR1: F12_LR1=1, async_eval the commit replay right after the commit
fit_lr1.apply(Q, mx)
import fit_prof                                 # GP1: F12_PROF=1, per-pass mixer / PLE times (profile arms only)
fit_prof.apply(Q, model)
_nd = fit_dup.apply_model(model)               # GP5: F12_DUPP=gdnp,attp (projections twice on the verify rows, exact)
if _nd:
    print(f"[{TAG}] F12_DUPP: {_nd} projection modules duplicated on S > 1 calls", flush=True)
SMAX = Gm.MAXK // B.KC

if BUD:
    import fit_budget
    print(f"fit12: budget {BUD}: {fit_budget.install(model, BUD)} routers", flush=True)
    if BJIT:
        print(f"fit12: budget-aware JIT: {fit_budget.install_jit(model, BUD)} look-ahead routers", flush=True)
UGB = float(os.environ.get("F12_UGB", "0") or 0)   # uncertainty-gated budget: on for a pass iff min draft prob >= UGB
if UGB:
    import fit_budget
    _hinit0, _hfin0 = spec125._MTPHead.__init__, spec125._MTPHead.draft_finish

    def _hinit(self, m):
        _hinit0(self, m)
        self.want_probs = True                 # draft_start then keeps each draft token's head probability

    def _hfin(self, *a, **kw):
        out = _hfin0(self, *a, **kw)
        on = bool(self.last_probs) and min(self.last_probs) >= UGB
        fit_budget.GATE.update(on=on, n=fit_budget.GATE["n"] + 1, on_n=fit_budget.GATE["on_n"] + int(on))
        return out
    spec125._MTPHead.__init__, spec125._MTPHead.draft_finish = _hinit, _hfin
    print(f"fit12: uncertainty gate: budget only on passes whose min MTP draft prob >= {UGB}", flush=True)

PRED = {}
if JITW:                                       # look-ahead predictions on the frugal block's base-pool post
    _post0, _blk0 = Gm.Gate.post, Q.SparseMoeBlock.__call__
    left0 = 30 + 31 * Gm.PRED_LINES

    def post(self, L, ids, seq=None, preds=None, counts=None, wk=0):
        if preds is None and L in PRED:
            preds, counts = PRED.pop(L)
        return _post0(self, L, ids, seq=seq, preds=preds, counts=counts, wk=wk)

    def blk(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if bm is not None and "tier" in bm and bm.get("next") and x.ndim == 3 and x.shape[0] == 1 and 1 <= x.shape[-2] <= SMAX:
            S = x.shape[-2]
            rows, parts, cnt, left = x.reshape(S, x.shape[-1]), [], [], left0
            for router, ratio, Bw in bm["next"]:
                b = min(Bw, left // S)
                if b <= 0:
                    break
                lg = router(rows * ratio if ratio is not None else rows)
                parts.append(mx.argpartition(-lg, kth=b - 1, axis=-1)[:, :b].reshape(-1).astype(mx.int32))
                cnt.append(b * S); left -= cnt[-1]
            if parts:
                PRED[bm["L"]] = (mx.concatenate(parts) if len(parts) > 1 else parts[0], tuple(cnt + [0] * (3 - len(cnt))) + (0,))
        return _blk0(self, x)
    Gm.Gate.post, Q.SparseMoeBlock.__call__ = post, blk
    nnext = sum(1 for l in model.language_model.model.layers if l.mlp.switch_mlp._bm.get("next"))
    print(f"fit12: JIT width {JITW}, look-ahead on {nnext} layers, QD {os.environ.get('BIGMOE_PF_QD')}", flush=True)


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def _ru():
    ru = _RU()
    return ru if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else None


_r = _ru()
print(f"[{TAG}] after load: footprint now {_r.u7 / 1e9:.2f} GB, lifetime peak {_r.u28 / 1e9:.2f} GB, MLX peak "
      f"{mx.get_peak_memory() / 1e9:.2f} GB, active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)


def chat_ids(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids[0], int) else ids["input_ids"])


PROMPTS = {"code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
           "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                             "programs faster. Use one everyday analogy and keep it to about 300 words."),
           "math": lambda: ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
                            "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 "
                            "per fresh duck egg. How much in dollars does she make every day at the farmers' market? Solve it "
                            "step by step."),                                  # GSM8K test 0 (TrainLab bench_budget24)
           "write": lambda: ("Write a product description for a reusable steel water bottle for hikers. Use exactly four "
                             "bullet points, keep every word lowercase, and end with a one-sentence call to action."),
           "lru": lambda: ("Implement a thread-safe LRU cache class in Python with get and put methods and a fixed capacity, "
                           "using only the standard library, then show a short usage example."),
           "csv": lambda: ("Write a Python script that reads a CSV file of transactions (customer_id, amount, date), computes "
                           "each customer's total spend per month, and prints the five biggest spenders of the latest month.")}
TFK = {"tfc0": "tf_code/HumanEval/103", "tfc1": "tf_code/HumanEval/10", "tfc2": "tf_code/HumanEval/153",
       "tfg0": "tf_gsm/600", "tfg1": "tf_gsm/601", "tfg2": "tf_gsm/602"}   # Eval suite tf_gen items (templated prompt ids)
TF = {}


def prompt_ids(P):
    if P in TFK:
        if not TF:
            import json
            for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
                r = json.loads(l)
                TF[r["key"]] = [int(t) for t in r["prompt_ids"]]
        return TF[TFK[P]]
    return chat_ids(PROMPTS[P]())


AW = os.environ.get("F12_AW") == "1"          # spec125.AdaptiveWidth (AW1): per-pass width from head probs and V[S]
LK = int(os.environ.get("F12_LOOKUP", "0") or 0)   # LK1: spec125 prompt lookup (1 standard, 2 >= 8-gram only); xcheck on
WTAU = float(os.environ.get("F12_WTAU", "0") or 0)   # WT1: verify MTP draft tokens while their cumulative head prob >= WTAU,
                                                    # at least 1 (a 1-row pass takes B12's late path: other arithmetic)


def _width(d, src, probs, match_n, max_draft):
    if src != "mtp" or not probs:
        return len(d)                                # lookup drafts whole (LK1)
    run, n = 1.0, 0
    for p in probs[:len(d)]:
        run *= p
        if run < WTAU:
            break
        n += 1
    return max(1, n)
V12 = {1: 70.0, 2: 92.0, 3: 100.0, 4: 107.0, 5: 114.0, 6: 120.0, 7: 130.0, 8: 140.0}   # 12 GB b16 verify ms (F2)
N = int(os.environ.get("BT_N", "128"))
STATE = {}


def run(P, k, n):
    ids = prompt_ids(P)
    if P not in STATE:
        tp = time.perf_counter()
        STATE[P] = B.b12_prefill_state(model, ids, 512, 4096)
        print(f"[{TAG}] {P}: prefill {len(ids) - 1} tokens in {time.perf_counter() - tp:.2f} s", flush=True)
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear(); PRED.clear()
    e0, d0, r0, s = ext.stats(), ext.defer_stats(), _ru(), {}
    t0 = time.perf_counter()
    ad = spec125.AdaptiveWidth(V12, step_ms=2.7) if AW else None
    out, _ = spec125.generate(model, ids, n, k_mtp=k, max_draft=k, lookup=LK, prefill_chunk=512, prefilled=STATE[P],
                              stats=s, adapt=ad, width=_width if WTAU else None)
    t = time.perf_counter() - t0
    st.gate.check()
    e1, d1, r1 = ext.stats(), ext.defer_stats(), _ru()
    npass = max(1, s["passes"])
    disk = (r1.u16 - r0.u16) if r0 and r1 else 0
    return out, dict(tps=len(out) / s["time"], tpp=s["tok_per_pass"], passes=npass, t=t, lku=s.get("lookup_used", 0),
                     dpp=s.get("drafted", 0) / npass,
                     ver=1e3 * s["t_verify"] / npass, dra=1e3 * s["t_draft"] / npass, head=1e3 * s["t_head"] / npass,
                     wait=(d1[1] - d0[1]) / npass, reads=(e1["misses"] - e0["misses"]) / npass,
                     pfi=(e1["pf_issued"] - e0["pf_issued"]) / npass, pfu=(e1["pf_used"] - e0["pf_used"]) / npass,
                     gbs=disk / 1e9 / s["time"], mbp=disk / 1e6 / npass, fp=(r1.u28 / 1e9) if r1 else float("nan"),
                     slow=1e3 * (e1["slow_s"] - e0["slow_s"]) / npass, io=1e3 * (e1["io_s"] - e0["io_s"]) / npass,
                     slowc=(e1["slow_calls"] - e0["slow_calls"]) / npass, gated=(e1["gated"] - e0["gated"]) / npass,
                     pfd={k_: round((e1[k_] - e0[k_]) / npass, 1) for k_ in ("pf_ontime", "pf_late", "pf_victims", "pf_cancel",
                          "pf_qdemand", "pf_nofit", "miss_pred_unissued", "miss_unpred")}
                         | {"pf_wait_ms": round(1e3 * (e1["pf_waited_s"] - e0["pf_waited_s"]) / npass, 1)})


import fit_wv                                  # WV1: F12_WV=code,prose,... width-invariance probe instead of the arms
if fit_wv.CAP:                                  # Q2: capture MoE inputs during one ~4K prefill, then exit
    fit_wv.capture(Q, model, B, tok, mx)
    print(f"[{TAG}] done", flush=True)
    sys.exit(0)
if fit_wv.PROMPTS3:                             # WV3: partial-accept replay vs full accepts, then exit
    for P in fit_wv.PROMPTS3:
        ids = prompt_ids(P)
        fit_wv.probe3(P, ids, model, B.b12_prefill_state(model, ids, 512, 4096), B, spec125, reset_generation_state, mx)
    print(f"[{TAG}] done", flush=True)
    sys.exit(0)
if fit_wv.PROMPTS2:                             # WV2: chunking probe (4 vs 2+2 rows), then exit
    for P in fit_wv.PROMPTS2:
        ids = prompt_ids(P)
        fit_wv.probe2(P, ids, model, B.b12_prefill_state(model, ids, 512, 4096), B, spec125, reset_generation_state, mx)
    print(f"[{TAG}] done", flush=True)
    sys.exit(0)
if fit_wv.apply(Q, model):
    for P in fit_wv.PROMPTS:
        ids = prompt_ids(P)
        fit_wv.probe(P, ids, model, B.b12_prefill_state(model, ids, 512, 4096), B, spec125, reset_generation_state, mx)
    print(f"[{TAG}] done", flush=True)
    sys.exit(0)
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
for arm in os.environ.get("F12_ARMS", "code:3,prose:3").split(","):
    P, k = arm.split(":"); k = int(k)
    run(P, k, 16)                                                    # warm-up (pools, compiled shapes)
    fit_prof.reset()
    if UGB:
        fit_budget.GATE.update(n=0, on_n=0)
    out, r = run(P, k, N)
    if UGB:
        print(f"[{TAG}] {P} k={k} gate: budget on {fit_budget.GATE['on_n']} of {fit_budget.GATE['n']} drafted passes "
              f"({100 * fit_budget.GATE['on_n'] / max(1, fit_budget.GATE['n']):.0f} %)", flush=True)
    print(f"[{TAG}] {P} k={k}{' AW' if AW else ''}: {r['tps']:.1f} tok/s, {r['tpp']:.2f} tok/pass, {r['passes']} passes | per pass: verify "
          f"{r['ver']:.1f} ms, draft {r['dra']:.1f} ms, head {r['head']:.1f} ms, waited {r['wait']:.1f}, reads {r['reads']:.1f}, "
          f"pf issued {r['pfi']:.1f} used {r['pfu']:.1f}, SSD {r['gbs']:.1f} GB/s {r['mbp']:.0f} MB | peak footprint "
          f"{r['fp']:.2f} GB | loader per pass: stalled calls {r['slowc']:.1f} of {r['gated']:.1f}, stall {r['slow']:.1f} ms, "
          f"io {r['io']:.1f} ms | sha {sha(out)} | tok0 {list(out).count(0)} | {tok.decode(out[:40])!r}", flush=True)
    fit_prof.report(TAG, P)
    if LK:
        print(f"[{TAG}] {P} k={k} lookup={LK}: lookup drafts used in {r['lku']} of {r['passes']} passes", flush=True)
    if WTAU:
        print(f"[{TAG}] {P} k={k} tau={WTAU}: {r['dpp']:.2f} draft tokens verified per pass", flush=True)
    if JITW:
        print(f"[{TAG}] {P} k={k} prefetch detail per pass: {r['pfd']}", flush=True)
    if os.environ.get("F12_SAVE"):
        import json
        json.dump([int(t) for t in out], open(os.path.join(os.path.expanduser(os.environ["F12_SAVE"]), f"{TAG}_{P}.json"), "w"))
if fit_ooo.ON:
    print(f"[{TAG}] OoO block calls (graph builds): {fit_ooo.CALLS[0]}", flush=True)
    if fit_ooo.CNT is not None:
        h1, ht = fit_ooo.hit_share()
        print(f"[{TAG}] OoO phase-1 (resident at post) pairs: {h1} of {ht} ({100 * h1 / max(1, ht):.1f} %)", flush=True)
if fit_lr1.ON:
    print(f"[{TAG}] LR1: {fit_lr1.CALLS[0]} commits' replays kicked with async_eval", flush=True)
if fit_pr1.ON:
    print(f"[{TAG}] PR1: {fit_pr1.CALLS[0]} keep-1 commits replayed on 2 rows", flush=True)
if fit_gdnr.ON:
    print(f"[{TAG}] GDNR: recurrence_r4 calls {fit_gdnr.CALLS[0]}, mlx_lm fallback calls {fit_gdnr.CALLS[1]}", flush=True)
print(f"[{TAG}] done", flush=True)
