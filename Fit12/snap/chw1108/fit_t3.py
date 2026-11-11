"""Fit12 T3 gates (Mon 2 Nov 2026 23:4x; PLAN.md 'T3', written before any run): the two-pass 16-row check of t3_twopass.py
with DFlash2 L1-20 (D7b) on the live 12 GB default (pinned bt_b12 8db71cc537, the suite's AR process env as fit_t0.py,
load_b12(0.10) WITHOUT the MTP head: DFlash2 drafts). Per prompt, in one process, each arm from its own exact prefill:
  ex    theta None: the exact check (16 rows in <= 4-row calls, stop at the first miss)
  g1    theta 1.01, p2_all: every check runs pass 1 (16 rows, wide), rolls back, then the exact check (G1: = ex bit for bit)
  g0    theta 0: pass 1 only
  t80   theta 0.8 (pass 2 on rows n..j);  t80c: + cf gate 0.9;  t80cs: + cf gate + skip (2, 8);  t80b: t80 again
Per arm: tokens (sha1, first divergence from ex), lsha, checks, kept per check, checks with pass 2 / exact / cf-gated /
skipped, rows in pass 1 / 2, pass-2 calls, refused commits, tok/s (shared GPU, rough); the process's lifetime peak.
Usage: [BigMoE venv] fit_t3.py TAG
(Sat 7 Nov) EvalSuite's entry point (t80ck1u code same-scores): F12_IDS_OUT=<file.json> with exactly ONE arm (F12_T3ARMS=
t80ck1u, t90ck1u, t95ck1u or ex) writes {key: {arm: [full-output ids]}}, the input of EvalSuite's tools/r57rows.py (EOS kept as the
last id, like EvalSuite's completion_ids), and
<file>_meta.json ({key: arm, theta, n, cap, finish stop / length, pf_s, dec_s, tok_s, checks, paused_s}) after every prompt,
in run order; F12_CAP=item caps each prompt at its item's max_tokens; F12_KEYS = the keys in run order (the history)."""
import ctypes, hashlib, json, os, sys, time

os.environ["HF_HUB_OFFLINE"] = "1"                   # no downloads: DFlash2's config must already be local
ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",
       "BIGMOE_LOOP_GUARD": "0", "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1",
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_DEFER_REPAIR_EVAL": "1"}   # verify capture / commit
for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_CM", "B12_CL", "B12_FOLD_CM", "B12_FOLD_AW", "B12_W", "B12_PMIN"):
    os.environ.pop(k, None)
os.environ.update(ENV)
if os.environ.get("F12_PFRGRID"):                    # (Sun 8 Nov) PFR's MoE grid: "keep" = the caller's s_moe (F12_SMOE);
    os.environ["B12_PFR_GRID"] = os.environ["F12_PFRGRID"]   # bt_b12's fast-mode default is "balanced" (one chunk <= 16K)
if os.environ.get("F12_CMCL"):                       # (23:2x) pass 1 on 16 rows posts up to 32 rank-0-1 LSB parts: C_l 20 -> 32
    os.environ["B12_CM"], os.environ["B12_CL"] = os.environ["F12_CMCL"].split(",")   # (C_m 51 -> 47 keeps the bytes)
TAG = sys.argv[1] if len(sys.argv) > 1 else "t3"
t_start = time.time()
OUTD = os.environ.get("F12_OUT", os.path.expanduser("~/Fit12/data/t3"))
CAPS = os.environ.get("F12_CAP", "128")             # (Sat 7 Nov) "item" = each item's own max_tokens (EvalSuite's rows)
CAP = 0 if CAPS == "item" else int(CAPS)
ES = os.path.expanduser("~/EvalSuite")
BT = os.environ.get("F12_BT", os.path.expanduser("~/Fit12/pinned/8db7/bt_b12.py"))
EOS = (248044, 248046)
sha = hashlib.sha256(open(BT, "rb").read()).hexdigest()[:10]
assert sha == "8db71cc537", f"bt_b12.py is {sha}, T3 is written against 8db71cc537"
REF = [json.loads(l) for l in open(os.path.join(ES, "results", "b12llff4", "decode.jsonl"))]
WANT = os.environ.get("F12_KEYS", "humaneval/HumanEval/156,humaneval/HumanEval/42,ifeval/,math500/test/number_theory/459.json").split(",")
def _sel(w):                                         # (Sat 7 Nov) b12llff4's row, else (e.g. MBPP for the 45 / 24 / 12 GB table)
    r = next((r for r in REF if r["key"].startswith(w) if w.endswith("/") or r["key"] == w), None)   # the item's own row from
    if r is None:                                    # EvalSuite's data/items/dec_<set>.jsonl (set = the key's prefix)
        it = next(it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{w.split('/')[0]}.jsonl")))
                  if it["key"] == w)
        r = dict(key=w, set=it["set"], n_prompt=len(it["prompt_ids"]))
    return r


SEL = [_sel(w) for w in WANT]
ITEMS = {}
for st_ in {r["set"] for r in SEL}:
    ITEMS.update({it["key"]: it for it in map(json.loads, open(os.path.join(ES, "data", "items", f"dec_{st_}.jsonl")))})


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


def life_gb():
    ru = _RU()
    return ru.u28 / 1e9 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # (snapshot chw1108) its own frozen modules
FAKE = os.environ.get("F12_FAKE") == "1"             # (T3R2, Tue 19:3x) harness dry test: no model, no GPU (fit_t3_fake.py)
KP = None                                              # (Fri 6 Nov) f12_kports when F12_KPORTS=1
if FAKE:
    import fit_t3_fake as FK
    B, T3, model, draft, DF, _EXT = FK.B, FK.T3, None, None, None, FK.Ext
    os.makedirs(OUTD, exist_ok=True)
    print(f"[{TAG}] FAKE harness dry test (no model); prompts " + ", ".join(r["key"] for r in SEL), flush=True)
else:
    import bt_b12 as B
    import t3_twopass as T3
    assert os.path.realpath(B.__file__) == os.path.realpath(BT), B.__file__
    T3.apply(B)
    rt, _ = B.load_b12(0.10, mtp=False)
    model = rt.model
    draft, DF = T3.load_draft(B, model)
    if os.environ.get("F12_PR1") == "1":                # (Sun 8 Nov, T2 stage 1b) PR1: keep-1 commits replay the PLE layer on 2 rows
        import fit_pr1                                   # (MTPLX's commit_verified_window patched in this process; asserts the
        from mtplx.models import qwen4_exp as _Q         # source still matches), so verify widths >= 2 give one token stream
        fit_pr1.install(_Q)
        print(f"[{TAG}] PR1 installed: keep-1 commits replay the PLE layer on 2 rows", flush=True)
    if os.environ.get("F12_WIRED") == "1":             # (Sat 7 Nov, Leap's FIRSTCHK fix, BigMoE's R57_WIRED form) MLX keeps its
        _wl = int(mx.device_info()["max_recommended_working_set_size"])   # buffers wired for the run (IOGPU stops unwiring the
        print(f"[{TAG}] WIRED: mx.set_wired_limit({_wl / 1e9:.2f} GB), was {mx.set_wired_limit(_wl) / 1e9:.2f} GB", flush=True)   # pools)
    from bigmoe_v3f.gated import ext as _EXT          # (T3R2) was imported per arm; the same module object
    if os.environ.get("F12_K12SYNC") == "1":           # (Tue 3 Nov 23:0x, K12; orchestrator 22:5x) Leap's [K] A/B
        import k12sync                                 # (50de28c): mx.eval at every MoE block's entry and exit
        k12sync.wrap(mx)
        print(f"[{TAG}] K12: mx.eval at every MoE block's entry and exit (mtplx qwen4_exp.SparseMoeBlock)", flush=True)
    if os.environ.get("F12_KPORTS") == "1":            # (Fri 6 Nov 09:4x, 12 GB plan lever 3) Leap's Fri switches, ported
        import f12_kports as KP                        # (inert until an arm sets them; pinned kernels, a GPU self-test)
        KP.install(model, draft)
        print(f"[{TAG}] KPORTS installed (f12_kports.py; kernels from {KP.PIN})", flush=True)
        KP.selftest()
    dbytes = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(draft.parameters()))
    os.makedirs(OUTD, exist_ok=True)
    print(f"[{TAG}] bt_b12 {sha}, engine {getattr(B, 'V3F', '?')}, C_m {B.CFG.get('cm')} C_l {B.CFG.get('cl')}, cap {CAPS}; drafter "
          f"{dbytes / 1e9:.3f} GB (own params), SW {draft.config.sliding_window}; after load: lifetime {life_gb():.3f} GB, MLX active "
          f"{mx.get_active_memory() / 1e9:.3f}; prompts " + ", ".join(f"{r['key']} (n {r['n_prompt']})" for r in SEL), flush=True)
ARMS = [("ex", None, {}), ("g1", 1.01, dict(p2_all=True)), ("g0", 0.0, {}), ("t80", 0.8, {}),
        ("t80c", 0.8, dict(cfmin=0.9)), ("t80cs", 0.8, dict(cfmin=0.9, skip=(2, 8))), ("t80b", 0.8, {}),
        ("g0k2", 0.0, dict(_k2=True)), ("t80ck2", 0.8, dict(cfmin=0.9, _k2=True)),
        ("t80csk2", 0.8, dict(cfmin=0.9, skip=(2, 8), _k2=True)),
        ("t80ck2m3", 0.8, dict(cfmin=0.9, _k2=True, _m3=True)),                                 # (Tue 10:4x) INFO: M-18
        ("t80ck1um3", 0.8, dict(cfmin=0.9, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")), _m3=True)),   # INFO
        ("t80ck1u", 0.8, dict(cfmin=0.9, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")))),   # (Tue 10:3x) P1-K1
        ("t90ck1u", 0.9, dict(cfmin=0.9, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")))),   # (Sat 7 Nov) theta 0.9
        ("t95ck1u", 0.95, dict(cfmin=0.9, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")))),  # (Sat 17:1x) theta 0.95
        ("t80ck2pf", 0.8, dict(cfmin=0.9, _k2=True, _pf2=int(os.environ.get("F12_PF2_B", "6")))),   # (Tue 15:0x) P2-PREFETCH
        ("t80ck1upf", 0.8, dict(cfmin=0.9, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")), _pf2=int(os.environ.get("F12_PF2_B", "6")))),
        # (Thu 5 Nov 20:0x, 12 GB plan lever 1) certificate calibration: t80ck1u's pass 1 on EVERY check (no cf gate), then
        # the exact check of all rows (G1's form, so the outputs = ex); with F12_CAL=1 each check logs pass 1's per-row
        # p1 / margin / argmax beside the exact tokens, for an offline replay of any (theta, cfmin, certificate)
        ("g1k1u", 1.01, dict(p2_all=True, _k2=True, _k1u=int(os.environ.get("F12_K1U_KU", "4")))),
        # (Thu 5 Nov 20:3x, Breakthrough 19:4x: price pass-1 width with theta) the same with t80ck2's top-2 pass 1, the form
        # the map's certificate sweep has quality for (top-2 theta 0.95 = 1.22 % code flips, PASS; theta 0.8 = 1.63 %, KILL)
        ("g1k2", 1.01, dict(p2_all=True, _k2=True)),
        # (Thu 5 Nov 20:2x, lever 2: the wider pass 1, Breakthrough 19:5x) calibration with a top-k pass 1 (k 3 / 4 / 5)
        ("g1k3", 1.01, dict(p2_all=True, _k2=True, _topk=3)), ("g1k4", 1.01, dict(p2_all=True, _k2=True, _topk=4)),
        ("g1k5", 1.01, dict(p2_all=True, _k2=True, _topk=5)),
        # (Fri 6 Nov 09:4x, 12 GB plan lever 3, the orchestrator 09:3x) Leap's Fri kernel switches, ported (f12_kports.py, needs
        # F12_KPORTS=1): dnax = the drafter on qmm16c + per-shape ks (exact for the target); att1 = pass 1's attention as one
        # SDPA; dn1 = pass 1's 4-bit dense on qmm16c + per-shape ks (att1 / dn1: FAST, pass 1 only)
        ("exd", None, dict(_dnax=True)),
        ("t80ck2a", 0.8, dict(cfmin=0.9, _k2=True, _att1=True)),
        ("t80ck2n", 0.8, dict(cfmin=0.9, _k2=True, _dn1=True)),
        ("t80ck2f", 0.8, dict(cfmin=0.9, _k2=True, _dnax=True, _att1=True, _dn1=True))]
if os.environ.get("F12_T3ARMS"):
    want = os.environ["F12_T3ARMS"].split(",")
    ARMS = [a for a in ARMS if a[0] in want]
IDS = os.environ.get("F12_IDS_OUT")                      # (Sat 7 Nov, EvalSuite's t80ck1u same-scores) full-output ids out
assert not IDS or len(ARMS) == 1, "F12_IDS_OUT takes exactly one arm (F12_T3ARMS=<arm>)"
import threading                                         # (T3R2, NewIdeas' QUEUE 4dbe21a; PLAN.md 'T3R2') PAUSE watch, SSD per pair
PAUSEF = os.environ.get("F12_PAUSEF", os.path.expanduser("~/.local/share/quiet/PAUSE"))
HCSV = os.path.expanduser("~/Library/Logs/health.csv")
RERUN = os.environ.get("F12_RERUN") == "1"               # a PAUSE inside a prompt's arms voids them: re-run once at the end
COOL_C, COOL_S = int(os.environ.get("F12_COOL_C", "55")), int(os.environ.get("F12_COOL_S", "120"))
PZ = []                                                  # PAUSE intervals [start, end or None], wall clock


def _watch():                                            # the engine waits on PAUSE inside a pass (v3p gated.py new_pass)
    while True:
        e, t = os.path.exists(PAUSEF), time.time()
        if e and (not PZ or PZ[-1][1] is not None):
            PZ.append([t, None])
        elif not e and PZ and PZ[-1][1] is None:
            PZ[-1][1] = t
        time.sleep(0.25)


threading.Thread(target=_watch, daemon=True).start()


def paused_s(t0, t1):                                    # seconds of PAUSE inside [t0, t1]; a PAUSE present now counts
    now = time.time()
    v = sum(max(0.0, min(t1, e if e is not None else now) - max(t0, s_)) for s_, e in list(PZ))
    return max(v, 0.25) if os.path.exists(PAUSEF) else v


def ssd_c():                                             # the health monitor's SSD temperature (screens.sh ssd_t), None if stale
    try:
        with open(HCSV, "rb") as f:
            f.seek(-512, 2)
            l_ = f.read().decode(errors="ignore").strip().splitlines()[-1].split(",")
        return int(float(l_[5])) if time.time() - float(l_[0]) < 180 else None
    except Exception:
        return None


def wait_cool(why):                                      # SSD <= COOL_C and no PAUSE, at most COOL_S
    t0 = time.time()
    while True:
        c = ssd_c()
        if c is not None and c <= COOL_C and not os.path.exists(PAUSEF):
            print(f"[{TAG}] {why}: SSD {c} C, no PAUSE (waited {time.time() - t0:.0f} s)", flush=True)
            return True
        if time.time() - t0 >= COOL_S:
            print(f"[{TAG}] {why}: SSD {c} C, PAUSE {os.path.exists(PAUSEF)} after {COOL_S} s: going on", flush=True)
            return False
        time.sleep(5)


def files(r):                                            # one prompt's output files
    kf = r["key"].replace("/", "_")
    return [f"{TAG}_{kf}_{a}{x}.json" for a, _, _ in ARMS for x in ("", "_toks", "_ph")]


def capk(key):                                           # (Sat 7 Nov) one prompt's token cap: F12_CAP, or the item's max_tokens
    return int(ITEMS[key]["max_tokens"]) if CAPS == "item" else CAP


PZK = {}                                                 # (Sat 7 Nov) key -> PAUSE seconds inside the run that counts


def dump_ids():                                          # (Sat 7 Nov, EvalSuite) after every prompt, in run order, atomically:
    if not IDS:                                          # IDS = {key: {arm: [ids]}} (EOS kept as the last id, like EvalSuite's
        return                                           # completion_ids); <IDS>_meta.json = {key: arm, theta, n, cap, finish, ...}
    arm, th = ARMS[0][0], ARMS[0][1]
    ids_, meta = {}, {}
    for k, v in PER.items():
        if arm in v:
            toks, s = v[arm]
            ids_[k] = {arm: [[int(t_) for t_ in toks]]}  # EvalSuite's tools/r57rows.py input: {key: {arm: [run ids]}}
            meta[k] = dict(arm=arm, theta=th, n=len(toks), cap=capk(k), finish="stop" if toks and toks[-1] in EOS else "length",
                           pf_s=s["pf_s"], dec_s=s["dec_s"], tok_s=round(s["tok_s"], 3), checks=s["checks"],
                           paused_s=round(PZK.get(k, 0.0), 1))
    for p_, o_ in ((IDS, ids_), ((IDS[:-5] if IDS.endswith(".json") else IDS) + "_meta.json", meta)):
        json.dump(o_, open(p_ + ".tmp", "w"))
        os.replace(p_ + ".tmp", p_)


def run_prompt(ri, r):                                   # all arms of one prompt; T3P's order rule (F12_ALT)
    ids = ITEMS[r["key"]]["prompt_ids"]
    cap_ = capk(r["key"])
    res = {}
    temps = []
    t0_ = time.time()
    arms = ARMS
    if os.environ.get("F12_ALT") == "1" and ri % 2 == 1:   # (T3P) odd prompts run the non-ex arms in reverse order
        arms = [a for a in ARMS if a[0] == "ex"] + [a for a in ARMS if a[0] != "ex"][::-1]
    for arm, th, kw in arms:
        lg = []
        kw = dict(kw)
        T3._K2[0] = kw.pop("_k2", False)                 # pass 1's tier_glu on 2 ranks for this arm
        T3._K1U[0] = kw.pop("_k1u", 0)                   # P1-K1: wide pass 1 = top-1 + union members (KU columns)
        T3._M3[0] = kw.pop("_m3", False)                 # M3: wide pass 1 at 3 bits (MSB-3 parts only)
        T3._PF2[0] = kw.pop("_pf2", 0)                   # P2-PREFETCH: late-flagged pass-2 ids in pass 1's base posts
        T3._TOPK[0] = kw.pop("_topk", 0)                 # (Thu 5 Nov) lever 2: wide pass 1 on ranks 0..k-1 (REF2-T tiers)
        kp = dict(dnax=kw.pop("_dnax", False), att1=kw.pop("_att1", False), dn1=kw.pop("_dn1", False))   # (Fri 6 Nov) lever 3
        assert KP is not None or not any(kp.values()), f"arm {arm} needs F12_KPORTS=1"
        kpn = KP.set_arm(**kp) if KP is not None else 0
        kc0 = KP.counts() if KP is not None else {}
        T3.PHLOG[0] = [] if os.environ.get("F12_PHLOG") == "1" else None   # (T3P) per-phase marks
        T3.CALLS[0] = [] if os.environ.get("F12_PHLOG") == "1" else None   # (Sun 8 Nov, T2) per exact call
        if os.environ.get("F12_GQAUDIT") == "1" and hasattr(T3, "gq_install"):   # (Sun 8 Nov) MLX sorted-gather audit / pad
            T3.gq_install(os.environ.get("F12_GQPAD") == "1")
        T3.CAL[0] = [] if os.environ.get("F12_CAL") == "1" else None       # (Thu 5 Nov) certificate calibration records
        ca = ssd_c()
        dfs0 = _EXT.defer_stats()                        # [rows, waited, late reads, late ids forced to wait]
        try:
            toks, s = T3.generate(B, model, draft, DF, list(ids), cap_, th, stop_ids=EOS, log=lg, **kw)
        finally:
            T3._K2[0] = False
            T3._K1U[0] = 0
            T3._M3[0] = False
            T3._PF2[0] = 0
            T3._TOPK[0] = 0
            if KP is not None:
                KP.set_arm()                             # every port off between arms
            ph, T3.PHLOG[0] = T3.PHLOG[0], None
            cl_, T3.CALLS[0] = T3.CALLS[0], None
            cal, T3.CAL[0] = T3.CAL[0], None
        if cal is not None:                              # one dict per check: path, d, cf, p1 / m1 / g1 (pass 1), new
            json.dump(cal, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}_cal.json"), "w"))
        dfs = [b_ - a_ for a_, b_ in zip(dfs0, _EXT.defer_stats())]
        kc = {k_: v_ - kc0.get(k_, 0) for k_, v_ in KP.counts().items()} if KP is not None else {}
        kps = ("" if not any(kp.values()) else " | KP " + "+".join(k_ for k_, v_ in kp.items() if v_)
               + f" ({kpn} drafter projections on NaxQL): " + ", ".join(f"{k_} {v_}" for k_, v_ in kc.items() if v_))
        json.dump(lg, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}.json"), "w"))
        json.dump([int(t_) for t_ in toks], open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}_toks.json"), "w"))
        if cl_ is not None:                              # (T2) check, call index, width, kept in the call, marks at start, at end
            json.dump(cl_, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}_calls.json"), "w"))
        if ph is not None:                               # name, check, (uptime s, gate seq, disk bytes, misses) at start, at end
            json.dump(ph, open(os.path.join(OUTD, f"{TAG}_{r['key'].replace('/', '_')}_{arm}_ph.json"), "w"))
            agg = {}
            for p in ph:
                a_ = agg.setdefault(p[0], [0, 0.0, 0, 0, 0])
                a_[0] += 1; a_[1] += p[6] - p[2]; a_[2] += p[8] - p[4]; a_[3] += p[9] - p[5]; a_[4] += p[7] - p[3]
            print(f"[{TAG}] {r['key']} {arm} PHASES " + " | ".join(
                f"{k}: {v[0]} runs, {1e3 * v[1] / v[0]:.1f} ms, {v[2] / v[0] / 1e6:.0f} MB, {v[3] / v[0]:.0f} misses, "
                f"{v[4] / v[0]:.2f} passes per run, {v[2] / max(1e-9, v[1]) / 1e9:.2f} GB/s" for k, v in agg.items()), flush=True)
        res[arm] = (toks, s)
        ex = res["ex"][0] if "ex" in res else toks
        div = next((i for i, (a, b) in enumerate(zip(toks, ex)) if a != b), None if len(toks) == len(ex) else min(len(toks), len(ex)))
        print(f"[{TAG}] {r['key']} {arm} theta {th} {kw or ''} | tok {len(toks)} sha {hashlib.sha1(str(toks).encode()).hexdigest()[:10]} "
              f"| lsha {s['lsha']} | = ex tokens {toks == ex} (first divergence {div}) | checks {s['checks']}, kept {s['kept']:.2f} "
              f"| pass2 {s['pass2']} exact {s['exact']} cfgate {s['cfgate']} skipped {s['skipped']} | rows pass 1 {s['rows1']} / pass 2 "
              f"{s['rows2']} in {s['calls2']} calls, refused {s['refused']} | prefill {s['pf_s']:.2f} s, decode {s['dec_s']:.2f} s, "
              f"{s['tok_s']:.2f} tok/s | lifetime {life_gb():.3f} GB | per check ms: draft {1e3 * s['t_draft'] / max(1, s['checks']):.1f} "
              f"pass 1 {1e3 * s['t_p1'] / max(1, s['checks']):.1f} pass 2 / exact {1e3 * s['t_p2'] / max(1, s['checks']):.1f} commit "
              f"{1e3 * s['t_commit'] / max(1, s['checks']):.1f}; misses per check pass 1 {s['miss1'] / max(1, s['checks']):.0f} "
              f"pass 2 {s['miss2'] / max(1, s['checks']):.0f}"
              + (f" | K1U: {s['k1u_kept'] / max(1, s['k1u_rl']):.3f} experts per row-layer, {s['k1u_over'] / max(1, s['k1u_rl']):.4f} "
                 f"of row-layers over KU ({s['k1u_rl']} row-layers)" if s.get("k1u_rl") else "")
              + (f" | PF2: {s['pf2_ids']} late ids posted in {s['pf2_calls']} wide calls, loader late reads {dfs[2]}, "
                 f"forced to wait {dfs[3]}" if s.get("pf2_calls") else "") + kps, flush=True)
        if "pf_mb" in s:                                 # (Sun 8 Nov, 12 GB prefill) the prefill of ids[:-1] alone, per arm
            gqs = f" | pfr {s.get('pfr')} | gq {s['gq']}" if "gq" in s else ""   # PFR's (P, s_moe); the gather audit
            print(f"[{TAG}] {r['key']} {arm} PREFILL n {len(ids)} smoe {getattr(T3, 'S_MOE', [0])[0]} | {s['pf_s']:.3f} s, "
                  f"{(len(ids) - 1) / max(1e-9, s['pf_s']):.1f} tok/s, SSD {s['pf_mb']:.0f} MB | lifetime {life_gb():.3f} GB{gqs}",
                  flush=True)
        temps.append(f"{arm} {ca}->{ssd_c()}")
    for b_, a_ in (("t80ck2", "t80ck2pf"), ("t80ck1u", "t80ck1upf")):   # P2-PREFETCH is exact: = its base arm
        if a_ in res and b_ in res:
            ok_ = res[a_][0] == res[b_][0] and res[a_][1]["lsha"] == res[b_][1]["lsha"]
            print(f"[{TAG}] {r['key']} GATE PF2 ({a_} = {b_}, tokens + logits): {'PASS' if ok_ else 'FAIL'}", flush=True)
    if all(a in res for a in ("g0", "g0k2")):
        k2 = res["g0"][0] == res["g0k2"][0] and res["g0"][1]["lsha"] == res["g0k2"][1]["lsha"]
        print(f"[{TAG}] {r['key']} GATE K2 (pass 1 on 2 ranks = 10 ranks with 8 weight-0 pairs, tokens + logits): "
              f"{'PASS' if k2 else 'FAIL'}", flush=True)
    if all(a in res for a in ("ex", "g1", "t80", "t80b")):
        g1 = res["g1"][0] == res["ex"][0] and res["g1"][1]["lsha"] == res["ex"][1]["lsha"]
        det = res["t80"][0] == res["t80b"][0] and res["t80"][1]["lsha"] == res["t80b"][1]["lsha"]
        print(f"[{TAG}] {r['key']} GATES | G1 (theta 1.01 + p2_all = exact, tokens + logits): {'PASS' if g1 else 'FAIL'} | "
              f"determinism (t80 = t80b): {'PASS' if det else 'FAIL'}", flush=True)
    return res, t0_, time.time(), temps


PER, VOID = {}, []                                       # key -> {arm: (tokens, stats)} of the run that counts; re-run list
if RERUN:
    wait_cool("start (SSD <= %d C, no PAUSE)" % COOL_C)
for ri, r in enumerate(SEL):
    res, t0, t1, temps = run_prompt(ri, r)
    pz = paused_s(t0, t1)
    PER[r["key"]] = res
    if pz > 0 and RERUN:
        VOID.append((ri, r))
        PER.pop(r["key"], None)                          # (Tue 3 Nov 22:3x) counts again only if its re-run is clean; the void
        for f_ in files(r):                              # run's files are moved to void1_* NOW, not at the re-run: a process
            if os.path.exists(os.path.join(OUTD, f_)):   # cut before its re-runs (cap, the 13:40 hard stop) leaves no live
                os.replace(os.path.join(OUTD, f_), os.path.join(OUTD, "void1_" + f_))   # files of a paused prompt
    print(f"[{TAG}] PAIR {r['key']}: SSD per arm (C) {', '.join(temps)} | PAUSE {pz:.1f} s inside -> "
          f"{'ok' if pz == 0 else 'VOID: re-run at the end' if RERUN else 'PAUSE-FLAGGED (timings void)'}", flush=True)
    PZK[r["key"]] = pz
    dump_ids()
for ri, r in VOID:                                       # once, in order; the void run's files are kept as void1_* (moved above)
    wait_cool(f"re-run {r['key']}")
    res, t0, t1, temps = run_prompt(ri, r)
    pz = paused_s(t0, t1)
    if pz > 0:                                           # paused again: the prompt leaves the verdict (files kept as void2_*)
        for f_ in files(r):
            if os.path.exists(os.path.join(OUTD, f_)):
                os.replace(os.path.join(OUTD, f_), os.path.join(OUTD, "void2_" + f_))
        PER.pop(r["key"], None)
    else:
        PER[r["key"]] = res
    print(f"[{TAG}] PAIR {r['key']} (re-run): SSD per arm (C) {', '.join(temps)} | PAUSE {pz:.1f} s inside -> "
          f"{'ok' if pz == 0 else 'VOID again: excluded'}", flush=True)
    PZK[r["key"]] = pz
    dump_ids()
print(f"[{TAG}] PAUSE summary: intervals {[(round(a_ - t_start, 1), None if b_ is None else round(b_ - t_start, 1)) for a_, b_ in PZ]} s "
      f"from the start; re-run {[r['key'].split('/')[-1] for _, r in VOID]}; excluded "
      f"{[r['key'].split('/')[-1] for _, r in VOID if r['key'] not in PER]}", flush=True)
TOT = {}                                                 # per arm, over the runs that count: tokens, decode s, checks, phase s
for key_, res_ in PER.items():
    for arm, (toks, s) in res_.items():
        a_ = TOT.setdefault(arm, dict(tok=0, dec=0.0, checks=0, kept_rows=0, t_draft=0.0, t_p1=0.0, t_p2=0.0, t_commit=0.0,
                                      miss1=0, miss2=0, pass2=0, exact=0, n=0))
        a_["n"] += 1; a_["tok"] += len(toks); a_["dec"] += s["dec_s"]; a_["checks"] += s["checks"]
        for k_ in ("t_draft", "t_p1", "t_p2", "t_commit", "miss1", "miss2", "pass2", "exact"):
            a_[k_] += s[k_]
for arm, a_ in TOT.items():                              # summed tokens / summed decode s (T3's totals)
    c_ = max(1, a_["checks"])
    print(f"[{TAG}] TOTAL {arm}: {a_['n']} prompts, {a_['tok']} tokens in {a_['dec']:.2f} s = {a_['tok'] / max(1e-9, a_['dec']):.2f} "
          f"tok/s | {a_['checks']} checks, kept {a_['tok'] / c_:.2f} | pass2 {a_['pass2']} exact {a_['exact']} | per check ms: "
          f"draft {1e3 * a_['t_draft'] / c_:.1f} pass 1 {1e3 * a_['t_p1'] / c_:.1f} pass 2 / exact {1e3 * a_['t_p2'] / c_:.1f} "
          f"commit {1e3 * a_['t_commit'] / c_:.1f}; misses per check pass 1 {a_['miss1'] / c_:.0f} pass 2 {a_['miss2'] / c_:.0f}",
          flush=True)
if os.environ.get("F12_K12SYNC") == "1" and not FAKE:
    print(f"[{TAG}] K12: {k12sync.N[0]} synced MoE block calls; wrapper still installed: {k12sync.installed()}", flush=True)
if IDS:
    dump_ids()
    print(f"[{TAG}] IDS: {len([k for k, v in PER.items() if ARMS[0][0] in v])} keys -> {IDS}", flush=True)
if os.environ.get("F12_PR1") == "1" and not FAKE:      # (stage 1b) how many keep-1 commits PR1 replayed on 2 rows
    import fit_pr1
    print(f"[{TAG}] PR1: {fit_pr1.CALLS[0]} keep-1 commits replayed on 2 rows", flush=True)
print(f"[{TAG}] done; lifetime {life_gb():.3f} GB", flush=True)
