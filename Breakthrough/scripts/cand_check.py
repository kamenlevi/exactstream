"""Breakthrough: check a bt_b12 candidate in BT_DIR before it goes live (NewIdeas Thu 18:3x: the post-R-41D candidate =
fold-exact defaults + B12_FRS_NOCYCLE + engine v3n / M-01). The candidate runs at ITS defaults (no B12_ENGINE etc. set),
one process per mode:
  CC_MODE=hash  p2k and p8k (fold_check's method: b12_prefill_state + spec125.generate 3:3:0, greedy, 256 tokens) and the
                Eval suite's agent/write16 (256 tokens, stops at EOS): hashes as fold_check (first-step logits, sha32 /
                sha256 of the tokens, decode-end trunk kv and MTP head cache) vs R-48's (= the fold-exact regression's),
                write16's tokens vs the suite's b12llfx row
  CC_MODE=rows  R-41C's first CC_NROWS (16) math rows in one process, each exactly as the suite's decode._gen_mtp runs
                it (reset_generation_state, STASH / PENDH cleared, b12_prefill_state, spec125.generate 3:3:0 to EOS or the
                item's cap; no gc, no settle): tokens vs the suite's b12llfx rows; per-row footprint start / max / end;
                lifetime peak. CC_DEADLINE (epoch s): a row that would not end before it (1.15 x the suite's seconds for that
                row + 20 s) is not started; the process writes OUT/TAG.wait and idles (no GPU work) until OUT/TAG.go holds
                the next deadline (the runner's next quiet lock), so one process spans quiet locks of <= 15 min.
CC_M01_EXPECT=1|0: the engine must (not) read records3m. Writes OUT/TAG.jsonl (one row per prompt / item).
Usage: [BigMoE venv] BT_DIR=<candidate dir> CC_MODE=hash|rows [CC_NROWS=16] [CC_DEADLINE=epoch] cand_check.py TAG"""
import ctypes, hashlib, json, os, sys, threading, time

MODE = os.environ.get("CC_MODE", "hash")
TAG = sys.argv[1] if len(sys.argv) > 1 else "cand_" + MODE
OUTD = os.path.expanduser(os.environ.get("CC_OUT", "~/Breakthrough/data/cand"))
os.makedirs(OUTD, exist_ok=True)
EXP_M01 = os.environ.get("CC_M01_EXPECT", "1") == "1"
for k in ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_CM", "B12_M01", "BIGMOE_M01", "B12_FRS_NOCYCLE"):
    assert k not in os.environ, f"{k} is set: the candidate must run at its own defaults"
ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "BIGMOE_PF_GROUP": "64",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80",
       "BIGMOE_FRSPEC": "1", "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
       "MTPLX_DEFER_REPAIR_EVAL": "1", "BIGMOE_LOOP_GUARD": "0", "B12_CACHE_MB": "64", "BIGMOE_ASYNC_EVERY": "1",
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0"}   # = fold_check's / w16_bisect's env (the suite's b12ll env)
if MODE == "rows":                                   # the suite's b12llfx process env = creep_fix's POP / ENV (Fit12), minus
    for k in ("EVALSUITE_FRUGAL", "EVALSUITE_SHIP", "EVALSUITE_FAST24", "B12_LATE3", "B12_W", "B12_NORM",   # B12_ENGINE /
              "BIGMOE_LOOP_GUARD", "B12_CM", "B12_CL", "B12_TRIMS", "B12_PFR", "B12_ALLWAIT", "B12_CACHE_MB",  # B12_FOLD /
              "B12_FOLD_ROWS", "B12_FOLD_PFR", "B12_FOLD_MOV", "BIGMOE_PF_GROUP", "BIGMOE_ASYNC_EVERY",      # B12_PFR_GRID:
              "BIGMOE_PF_PREFILL", "BIGMOE_PF_FULLSCR", "MTPLX_NGRAM_HOT_MB"):                             # the candidate's
        os.environ.pop(k, None)                                                                            # defaults
    ENV.update({"EVALSUITE_SHIP": "1", "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0",
                "MTPLX_QWEN4EXP_COMPILE": "0", "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0",
                "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1"})
    os.environ.update(ENV)
else:
    for k, v in ENV.items():
        os.environ.setdefault(k, v)


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def ru():                                            # (footprint GB, lifetime peak GB, disk bytes read) of this process
    r = _RU()
    assert _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) == 0
    return r.u7 / 1e9, r.u28 / 1e9, r.u16


import numpy as np
import mlx.core as mx
PEAK = [0.0]


def _sampler():
    while True:
        PEAK[0] = max(PEAK[0], ru()[0])
        time.sleep(0.02)


threading.Thread(target=_sampler, daemon=True).start()
HERE = os.path.dirname(os.path.abspath(__file__))
BTD = os.environ.get("BT_DIR") or HERE
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, BTD)
BTSHA = hashlib.sha256(open(os.path.join(BTD, "bt_b12.py"), "rb").read()).hexdigest()[:10]
import bt_b12 as B
assert os.path.dirname(os.path.abspath(B.__file__)) == os.path.abspath(BTD), (B.__file__, BTD)
t_load = time.perf_counter()
rt, st = B.load_b12(0.10, mtp=True)
t_load = time.perf_counter() - t_load
import spec125
import bigmoe_ext as EXT
from bigmoe_v3f.fuse import reset_generation_state
m01_on = getattr(EXT, "m01_on", lambda: False)()
model, tok = rt.model, rt.tokenizer
_TM = model.language_model.model if hasattr(model, "language_model") else model.model
tier_dirs = sorted({os.path.basename(os.path.dirname(l.mlp.switch_mlp._bm["tier"]["path"])) for l in _TM.layers
                    if "tier" in getattr(getattr(getattr(l, "mlp", None), "switch_mlp", None), "_bm", {})})
assert m01_on == EXP_M01 and tier_dirs == (["records3m"] if EXP_M01 else ["records3"]), (m01_on, tier_dirs)
print(f"[{TAG}] bt_b12 {B.__file__} sha {BTSHA} | engine {B.V3F} ext {EXT.__file__} m01_on {m01_on} records {tier_dirs} | "
      f"B12_FOLD {int(B.FOLD[0])} FRS_NOCYCLE {os.environ.get('B12_FRS_NOCYCLE', 'default')} | C_m {B.CFG['cm']} C_l "
      f"{B.CFG['cl']} | load {t_load:.1f} s | footprint {ru()[0]:.3f} GB", flush=True)
EOS = {248044, 248046}
ES = os.path.expanduser("~/EvalSuite/")
sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
out_rows = []


def emit(row):
    out_rows.append(row)
    with open(os.path.join(OUTD, f"{TAG}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if MODE == "hash":
    EXPECT = {"p2k": ("98c6bc1c0d", "82995cdb4e", "a0774ec635", "37028a6545", "8563204e11"),   # R-48 (all 4 runs) =
              "p8k": ("61b2520e58", "0b0d91cd43", "ed67050746", "5a760388d6", "f59755cf27"),   # the fold-exact regression
              "write16": ("418633931b", "a1789d46bf", "df432791d6", "368d4f4e01", "5c58dda19b")}
    W16 = next(json.loads(l) for l in open(ES + "data/items/dec_agent.jsonl") if json.loads(l)["key"] == "agent/write16")
    W16REF = next(json.loads(l)["completion_ids"] for l in open(ES + "results/b12llfx/decode.jsonl")
                  if json.loads(l)["key"] == "agent/write16")

    def chat_ids(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        return list(ids if isinstance(ids[0], int) else ids["input_ids"])

    HEAD = "Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
    R = os.path.expanduser("~/Breakthrough/")
    PROMPTS = {"p2k": lambda: chat_ids(HEAD + open(R + "scripts/tf_frugal_var.py").read()[:7600] + "\n```"),
               "p8k": lambda: chat_ids(HEAD + open(R + "data/prompts/ctx32k.txt").read()[:26000] + "\n```"),
               "write16": lambda: list(W16["prompt_ids"])}

    def kv_hash(cache):
        full = hashlib.sha1()
        for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
            h = hashlib.sha1()
            for a in (c.state if hasattr(c, "state") else []):
                if isinstance(a, mx.array):
                    h.update(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes())
            full.update(h.digest())
        return full.hexdigest()[:10]

    lgh = lambda a: hashlib.sha1(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:10]
    CAPT = []
    allok = True
    for P in ("p2k", "p8k", "write16"):
        ids = PROMPTS[P]()
        reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
        npf, nrf, tp = len(B.PFR_LOG), len(getattr(B, "REFILL_LOG", [])), time.perf_counter()
        state = B.b12_prefill_state(model, ids, 512, 4096)
        mx.eval(state["logits"])
        tp, pfr = time.perf_counter() - tp, [list(x) for x in B.PFR_LOG[npf:]]
        rfl = [list(x) for x in getattr(B, "REFILL_LOG", [])[nrf:]]
        lg = lgh(state["logits"])
        if not CAPT:                                 # the decode's own cache + head (after the fold's wrapper)
            _rs = spec125.restore_state

            def _cap(model_, state_, with_head=True):
                r = _rs(model_, state_, with_head=with_head)
                CAPT.append(r)
                return r
            spec125.restore_state = _cap
            CAPT.append(None)
        kw = {"stop_ids": EOS} if P == "write16" else {}
        tg = time.perf_counter()
        out, s = spec125.generate(model, ids, 256, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state,
                                  stats={}, **kw)
        st.gate.check()
        td = time.perf_counter() - tg
        out = [int(x) for x in out]
        dc, hd = CAPT[-1]
        got = (lg, sha(out[:32]), sha(out), kv_hash(dc), kv_hash(hd.cache) if hd is not None else "none")
        ok = got == EXPECT[P]
        ref = next((i for i, (a, b) in enumerate(zip(out, W16REF)) if a != b), None) if P == "write16" else None
        allok &= ok and ref is None
        emit(dict(tag=TAG, mode=MODE, prompt=P, hashes=got, expect=EXPECT[P], equal=ok, ntok=len(out), decode_s=round(td, 2), n_prompt=len(ids), prefill_s=round(tp, 2), pfr=pfr or None, refill=rfl or None,
                  b12llfx_first_diff=ref, lifetime_peak=round(ru()[1], 3), bt_b12=BTSHA, m01=m01_on))
        print(f"[{TAG}] {P} RESULT hashes {' '.join(got)} | = R-48's {ok}" + (f" | vs b12llfx first diff {ref}" if P == "write16"
              else "") + f" | {len(ids)}-token prefill {tp:.2f} s, PFR {'yes' if pfr else 'no'}, refill {rfl[0] if rfl else 'no'} | {len(out)} tok in {td:.1f} s | lifetime peak "
              f"{ru()[1]:.3f} GB", flush=True)
        CAPT[-1:] = [None]
        del state, dc, hd
        mx.clear_cache()
    print(f"[{TAG}] HASH VERDICT {'PASS: every hash = R-48 / b12llfx' if allok else 'FAIL: a hash differs'}", flush=True)
    if not allok:                                    # the runner stops: no rows on a candidate whose bits moved
        print(f"[{TAG}] done (exit 3)", flush=True)
        sys.exit(3)
else:
    NR = int(os.environ.get("CC_NROWS", "16"))
    REF = [r for r in map(json.loads, open(ES + "results/b12llfx/decode.jsonl")) if r["set"] == "math"][:NR]
    ITEMS = {it["key"]: it for it in map(json.loads, open(ES + "data/items/dec_math.jsonl"))}
    DL = [float(os.environ.get("CC_DEADLINE", "0")) or float("inf")]
    waited = 0.0
    same_all = True
    for k, r in enumerate(REF, 1):
        need = 1.15 * r["seconds"] + 20
        if time.time() + need > DL[0]:               # would not end inside this quiet lock: idle until the next one
            wf, gf = os.path.join(OUTD, f"{TAG}.wait"), os.path.join(OUTD, f"{TAG}.go")
            open(wf, "w").write(f"{k} {need:.0f}\n")
            print(f"[{TAG}] WAIT before row {k} (needs ~{need:.0f} s, {DL[0] - time.time():.0f} s left): idle until {gf}", flush=True)
            t_w = time.time()
            while not os.path.exists(gf):
                time.sleep(2)
            time.sleep(0.5)
            DL[0] = float(open(gf).read().split()[0])
            os.remove(gf)
            waited += time.time() - t_w
            print(f"[{TAG}] GO after {time.time() - t_w:.0f} s idle; new deadline in {DL[0] - time.time():.0f} s", flush=True)
        it = ITEMS[r["key"]]
        ids, cap = it["prompt_ids"], it["max_tokens"]
        assert len(ids) == r["n_prompt"], r["key"]
        f0 = ru()[0]; PEAK[0] = f0
        t0 = time.perf_counter()
        reset_generation_state(model)                # = decode._gen_mtp
        B.STASH.clear(); B.PENDH.clear()
        state = B.b12_prefill_state(model, list(ids), 512, 4096)
        toks, sst = spec125.generate(model, list(ids), cap, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512,
                                     prefilled=state, stop_ids=EOS)
        toks = [int(x) for x in toks]
        fin = "length"
        for j, x in enumerate(toks):
            if x in EOS:
                toks, fin = toks[:j + 1], "eos"
                break
        else:
            toks = toks[:cap]
        secs = time.perf_counter() - t0
        del state, sst
        same = toks == r["completion_ids"]
        same_all &= same
        f1, life = ru()[0], ru()[1]
        emit(dict(tag=TAG, mode=MODE, row=k, key=r["key"], ntok=len(toks), fin=fin, equal=same, seconds=round(secs, 1),
                  suite_seconds=r["seconds"], start=round(f0, 3), max=round(PEAK[0], 3), end=round(f1, 3),
                  life=round(life, 3), suite_life=r["mem"]["life"], bt_b12=BTSHA, m01=m01_on, cm=B.CFG["cm"]))
        print(f"[{TAG}] {r['key']} RESULT row {k} | tok {len(toks)} {fin} | b12llfx tokens equal {same} | {secs:.0f} s (suite "
              f"{r['seconds']:.0f}) | footprint start {f0:.3f} max {PEAK[0]:.3f} end {f1:.3f} GB | lifetime peak {life:.3f} "
              f"(suite {r['mem']['life']:.3f}) | C_m {B.CFG['cm']}", flush=True)
    life = ru()[1]
    print(f"[{TAG}] ROWS VERDICT {len(out_rows)} rows, tokens = b12llfx on all: {same_all}; lifetime peak {life:.3f} GB "
          f"(line <= 11.85; suite b12llfx 16 rows {REF[-1]['mem']['life']:.3f} unfixed); idle between locks {waited:.0f} s -> "
          f"{'PASS' if same_all and life <= 11.85 and len(out_rows) == NR else 'FAIL'}", flush=True)
print(f"[{TAG}] done", flush=True)
