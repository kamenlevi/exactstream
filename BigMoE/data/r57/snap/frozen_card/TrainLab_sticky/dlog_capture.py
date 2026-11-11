> (a quote from the project owner, kept private)
125B's runs, MTP Head leads). Fable's E0 / E1 (~/FableGod/FINDINGS.md, MISSION 4 sec 4): the 125B's OWN answers in de / ru /
ja, logged from the 45 GB default's own live loop, every verified row's training data kept for free.
Harness = D-23's (sticky/d23c_mtptrunc.py with D23_NEWDEF=0, i.e. D-23's pre-OOO preset: REF2-T tiers 4433333333 / 3c, C_l 112,
MAXK 160, build_next, DFlash2 L1-20 with the 0.3 running-confidence cut), dfl arm only, no timing claims (a data run). Per
prompt, one file ~/TrainLab/data/dlog/<night>/<key>.npz, the union of the two trainers' formats:
  h   (N-1, 2560) uint16  DFlash2's own context input hidden_norm(fc(taps)) as bf16 bits (taps = the mean of the 4 HC streams
                          after layers 5/15/25/35/45, exactly the arrays the live drafter is fed); rows < gen - 2048 zero
  wd  (N-1, 10240) fp16   the trunk's pre-mixer widened stream (the native MTP head's input, as data/d16w's wd)
  tk / lp (N-1, 32)       the 125B's top-32 next-token ids (int32, best first) / log-probs (fp16), from the verify rows
                          (REF2 = batch-invariant, so = the AR distribution); rows < gen - 1 zero (the prompt)
  ids (N) int32, asst (N) uint8 (1 on the 125B's answer), split 'train', gen, key, lang, tmpl, src, preset
= Leap's dfl2c format (sticky/d13_train.py / dfl2c_train.py read h / tk / lp / ids / asst / split) plus wd for d22_ttt.py.
Only rows with a correct prefix are logged: the prompt (prefill) and, per check, the kept rows (the accepted drafts' inputs +
the anchor), i.e. every position exactly once. Self-check per prompt: the logged top-1 must equal the next token on every
answer row (greedy); a prompt below 99.5 % stops the run (DLOG-ONPATH-FAIL).
Prompts: MGSM (data/raw/mgsm, Kamen-approved download 10-22) de / ru / ja, minus the 80 problem indices dec_mgsm uses in any
language (eval hygiene: D-24 / EvalSuite measure on those), 4 instruction templates in the target language (solve / story /
explain-to-a-child / dialogue), one template per problem, languages interleaved; 384 tokens max per answer, thinking off
(dec_multi's template). Stops: all prompts done, every language >= DLOG_TARGET answer tokens, DLOG_CAP_GB written tonight,
free disk < DLOG_MIN_FREE_GB, or the time budget (DLOG_BUDGET / DLOG_HARD s of running time, SIGSTOP freezes excluded).
Resumable (a prompt with a file is skipped). Usage: [BigMoE venv] dlog_capture.py [C=224]
DLOG_SRC=codemath (11-02, for HPD-2 / Fable's E2): English code (sticky/code_tasks.py stubs) + GSM8K math rendered as
L1-20's own data, "languages" code / math (DLOG_NIGHT names the folder)."""
import json, os, random, re, shutil, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.pop("BIGMOE_MTP_STREAM", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"   # k69_live R57_PRESET=45 (REF2-T)
os.environ.setdefault("BIGMOE_TIER_CL", "112")
os.environ.pop("BIGMOE_TIER_COMBINE", None); os.environ.pop("BIGMOE_OOO", None)  # D-23's preset (D23_NEWDEF=0)
os.environ.setdefault("BIGMOE_MAXK", str(10 * int(os.environ.get("DLOG_ROWS", "16"))))   # E2: 10 x rows (160 as shipped)
os.environ.setdefault("BIGMOE_EXT_BUILD", "build_next")
assert os.environ.get("BIGMOE_MAXK") == str(10 * int(os.environ.get("DLOG_ROWS", "16")))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

H = os.path.expanduser
NIGHT = os.environ.get("DLOG_NIGHT", "n1101")
OUTD = H(f"~/TrainLab/data/dlog/{NIGHT}"); os.makedirs(OUTD, exist_ok=True)
IDX = os.path.join(OUTD, "index.jsonl")
PAUSE = H("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CAP = int(os.environ.get("DLOG_MAXTOK", "384"))
TARGET = int(os.environ.get("DLOG_TARGET", "30000"))     # answer tokens per language (E1 needs 20K; margin for a held-out cut)
CAP_GB, MIN_FREE = float(os.environ.get("DLOG_CAP_GB", "8")), float(os.environ.get("DLOG_MIN_FREE_GB", "114"))
BUDGET, HARD = int(os.environ.get("DLOG_BUDGET", "1500")), int(os.environ.get("DLOG_HARD", "1590"))
LANGS = os.environ.get("DLOG_LANGS", "de,ru,ja").split(",")
SRC = os.environ.get("DLOG_SRC", "mgsm")                 # 11-02: "codemath" = English code + math (HPD-2 / E2), see below
if SRC == "codemath":
    LANGS = ["code", "math"]
elif SRC == "mbpp":                                       # 11-02: MBPP code only (E2 / D2, Kamen's download OK)
    LANGS = ["code"]
TRUNC, R, K, TOPK = 0.3, 16, 16, 32
ROWS = int(os.environ.get("DLOG_ROWS", "16"))           # E2: rows per check (16 = as shipped; fewer at small C, exact)
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PRESET = "45 (REF2-T 4433333333/3c, C_l 112, MAXK 160, build_next, L1-20 + cut 0.3; D-23's)"

ACT = [0.0]                                              # running time without SIGSTOP freezes (Leap's ticker)
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
def say(s):
    print(f"[dlog {time.strftime('%H:%M:%S')}] {s}", flush=True)

# ---------------------------------------------------------------- prompts (built before the model: no GPU, fails fast)
TMPL = {
    "de": {"S": "Löse die folgende Aufgabe Schritt für Schritt und antworte auf Deutsch. Gib das Endergebnis in \\boxed{} an.\n\n{q}",
           "G": "Schreibe auf Deutsch eine kurze Geschichte (etwa 150 Wörter) über die Situation in dieser Aufgabe. Löse die Aufgabe nicht.\n\n{q}",
           "E": "Erkläre auf Deutsch einem zehnjährigen Kind, wie man an diese Aufgabe herangeht, ohne die Lösung zu verraten.\n\n{q}",
           "D": "Schreibe auf Deutsch einen kurzen Dialog zwischen zwei Freunden, die über die Situation in dieser Aufgabe sprechen.\n\n{q}"},
    "ru": {"S": "Реши следующую задачу шаг за шагом и отвечай на русском языке. Запиши окончательный ответ в \\boxed{}.\n\n{q}",
           "G": "Напиши на русском языке короткий рассказ (около 150 слов) о ситуации из этой задачи. Не решай задачу.\n\n{q}",
           "E": "Объясни на русском языке десятилетнему ребёнку, как подступиться к этой задаче, не раскрывая ответа.\n\n{q}",
           "D": "Напиши на русском языке короткий диалог двух друзей, которые обсуждают ситуацию из этой задачи.\n\n{q}"},
    "ja": {"S": "次の問題を段階的に解き、日本語で答えてください。最終的な答えは \\boxed{} に入れてください。\n\n{q}",
           "G": "この問題の状況について、日本語で短い物語（約300字）を書いてください。問題は解かないでください。\n\n{q}",
           "E": "10歳の子どもに、この問題にどう取り組めばよいかを日本語で説明してください。答えは教えないでください。\n\n{q}",
           "D": "この問題の状況について話し合う二人の友人の短い会話を日本語で書いてください。\n\n{q}"},
}
PLAN = []                                                # (key, lang, tmpl, user text), languages interleaved
per = {}
if SRC == "codemath":
    # Code: sticky/code_tasks.py (new HumanEval-style stubs: every HumanEval+ problem is already train / held-out / live),
    # rendered as L1-20's data (Leap's dfl2e_data.py he_user = EvalSuite's tf_code text). Math: the GSM8K test problems no
    # set uses (0-119 L1-20's data and 0-249 MGSM = DLOG / dec_mgsm; 600-659 tf_gsm = the dfl2d_math held-out; 700-724 the
    # decode set), with L1-20's GSM instruction, in a fixed random order.
    sys.path.insert(0, H("/Users/kamen/BigMoE/data/r57/snap/frozen_card/TrainLab_sticky"))
    from code_tasks import TASKS
    HE_USER = ("Complete the following Python function. Reply with the whole function in one ```python code block.\n\n"
               "```python\n{q}```")
    GSM_INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
    gsm = [json.loads(l) for l in open(H("~/EvalSuite/data/raw/gsm8k_test.jsonl"))]
    free = [i for i in range(len(gsm)) if not (i < 250 or 600 <= i < 660 or 700 <= i < 725)]
    per["code"] = [(f"dlog/{k}", "code", "he", HE_USER.replace("{q}", s)) for k, s in TASKS]
    per["math"] = [(f"dlog/gsm{i}", "math", "gsm", GSM_INSTR + gsm[i]["question"]) for i in random.Random(1102).sample(free, len(free))]
    excl = f"{len(gsm) - len(free)} GSM8K problems excluded"
elif SRC == "mbpp":
    # Code: MBPP (Austin et al. 2021; ~/TrainLab/data/mbpp/mbpp.jsonl, 974 problems, SHA256.txt), none in any set we score on
    # (held-out / live code = HumanEval+). The task text + its tests (they fix the function name), L1-20-like instruction,
    # in a fixed random order; the 125B's own answer is the data.
    MB_USER = ("Write the following Python function. Reply with the whole code in one ```python code block.\n\n"
               "{q}\nYour code should pass these tests:\n\n{t}")
    mb = [json.loads(l) for l in open(H("~/TrainLab/data/mbpp/mbpp.jsonl"))]
    assert len(mb) == 974, len(mb)
    per["code"] = [(f"dlog/mbpp{x['task_id']}", "code", "mbpp", MB_USER.replace("{q}", x["text"].strip()).replace("{t}", "\n".join(x["test_list"])))
                   for x in random.Random(1103).sample(mb, len(mb))]
    excl = "MBPP all 974, none excluded"
else:
    EXCL = {int(re.match(r"mgsm/[a-z]+(\d+)$", json.loads(l)["key"]).group(1))
            for l in open(H("~/EvalSuite/data/items/dec_mgsm.jsonl"))}
    assert len(EXCL) == 80, len(EXCL)
    for lang in LANGS:
        probs = [line.rstrip("\n").split("\t") for line in open(H(f"~/EvalSuite/data/raw/mgsm/mgsm_{lang}.tsv"), encoding="utf-8")]
        assert len(probs) == 250, (lang, len(probs))
        idx = [i for i in random.Random(1101 + LANGS.index(lang)).sample(range(250), 250) if i not in EXCL]   # own order per language
        per[lang] = [(f"dlog/{lang}{i}{'SGED'[j % 4]}", lang, "SGED"[j % 4], TMPL[lang]["SGED"[j % 4]].replace("{q}", probs[i][0]))
                     for j, i in enumerate(idx)]
    excl = f"{len(EXCL)} MGSM indices excluded"
for j in range(max(len(v) for v in per.values())):
    PLAN += [per[lang][j] for lang in LANGS if j < len(per[lang])]
say(f"plan: {len(PLAN)} prompts ({', '.join(f'{l} {len(per[l])}' for l in LANGS)}), {excl}; "
    f"target {TARGET} answer tokens per language, cap {CAP} per answer, {CAP_GB} GB / free >= {MIN_FREE} GB; -> {OUTD}")
if os.environ.get("DLOG_PLAN_ONLY"):                    # CPU check of the plan, no model
    for x in PLAN[:6]:
        say(f"  {x[0]} [{x[2]}]: {x[3][:160]!r}")
    sys.exit(0)

# ---------------------------------------------------------------- the default's model (D-23's preset; no MTP head needed)
t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
try:                                                     # E2 small-C mode: rows x top-10 must fit every layer's base pool
    _cm = min(sw._bm["C"] for sw in st._tier_sws)
    if ROWS * 10 > _cm:
        say(f"E2: smallest base pool C_m {_cm} < {ROWS} rows x 10: rows -> {_cm // 10}")
        ROWS = _cm // 10
    say(f"E2: C {C}, base pool C_m >= {_cm}, rows per check {ROWS}, BIGMOE_MAXK {os.environ.get('BIGMOE_MAXK')}")
except Exception as e:
    say(f"E2: C_m check skipped ({e!r})")
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
import bigmoe.tier as _BT
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == int(os.environ["BIGMOE_MAXK"]) and BG.XLINES == (0 if BG.MAXK <= 61 else min(4, (BG.MAXK - 61 + 30) // 31)), \
    (BG.MAXK, BG.XLINES)                                # 160 / 4 as shipped; E2 small-C mode: 10 x rows
assert not _BT._COMBINE and not _BT._OOO, "D-23's pre-OOO preset expected"
TXT = model.language_model.model
lay = TXT.layers
SW_ = [l.mlp.switch_mlp for l in lay]
TOKZ = rt.tokenizer
say(f"loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB active; C {C}; preset {PRESET}")

def set_arm(arm):                                        # k69_live's: the prefill as shipped ('slot'), the loop on the gate
    BS.GATE_N[0] = min(40, BG.MAXK) if arm == "slot" else BG.MAXK   # 40 / 160 as shipped (E2 small-C: <= the pool)
    BS._PF_RANKED = False
    for sw in SW_:
        sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

# ---------------------------------------------------------------- taps for the drafter (Leap's S45 harness, as D-23)
TAP = {"on": False, "x": [[] for _ in TAPS]}
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L, W = o.shape
        TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["x"] = [[] for _ in TAPS]
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(H("~/Leap/data/l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SWIN = int(draft.config.mask_token_id), int(draft.config.sliding_window)

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

_midloop = Q._prefill_midloop_eval_layers
def ref_on(on):
    if on:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

# ---------------------------------------------------------------- the logger: reads arrays the loop already evaluated
def top32(lg):
    """Leap's dfl2c_extract recipe: the 125B's top-32 ids (best first) and log-probs (fp16) per row"""
    lg = lg.astype(mx.float32)
    top = mx.argpartition(-lg, TOPK - 1, axis=-1)[:, :TOPK]
    v = mx.take_along_axis(lg, top, axis=-1)
    o = mx.argsort(-v, axis=-1)
    top, v = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
    return top.astype(mx.int32), (v - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)

def hbits(tp):
    """the drafter's own context input for these tap rows, hidden_norm(fc(taps)) in bf16 (stored as uint16 bits in save)"""
    return draft.hidden_norm(draft.fc(tp.astype(mx.bfloat16)))

class Log:
    """per-check arrays, submitted with async_eval (no extra sync in the loop: they queue behind the commit replay and free
    their inputs once computed); converted to numpy once per prompt in save()"""
    def __init__(self):
        self.h, self.wd, self.tk, self.lp = [], [], [], []
    def add(self, h, wd, tk, lp):
        mx.async_eval(h, wd, tk, lp)
        self.h.append(h); self.wd.append(wd); self.tk.append(tk); self.lp.append(lp)
    def np(self):
        hb = [(np.array(x.astype(mx.float32)).view(np.uint32) >> 16).astype(np.uint16) for x in self.h]   # bf16 bits, exact
        return hb, [np.array(x) for x in self.wd], [np.array(x) for x in self.tk], [np.array(x) for x in self.lp]

def prefill(ids, chunk=1024):
    """k69_live's prefill (chunks of 1024 on the PF1 path, taps) + each chunk's widened stream and logits' top-32"""
    pf1(True)
    cache, lg, ws, tks = model.make_cache(), None, [], []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        w = TXT._last_widened
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]], w)
        assert w.shape[1] == min(chunk, len(ids) - a), w.shape
        ws.append(w[0].astype(mx.float16))
    TAP["on"] = False
    pf1(False)
    return cache, lg[0, -1], taps(), mx.concatenate(ws) if len(ws) > 1 else ws[0]

def run(ids, cap):
    while os.path.exists(PAUSE):
        time.sleep(5)
    set_arm("slot")                                      # the prefill as shipped
    ref_on(True)
    t1 = time.perf_counter()
    cache, last, T, W = prefill(ids)
    pf = time.perf_counter() - t1
    P = len(ids)
    tok = int(mx.argmax(last).item())
    L = Log()
    lo = max(0, P - 2048)                                # rows the trainer reads for the first anchor (window 2048)
    ref2.remove()                                        # the logging ops on the stock path, as the drafter's own fc runs
    tk0, lp0 = top32(last[None])                         # the label of row P - 1 (the first answer token)
    L.add(hbits(T[lo:P]), W[:P], tk0, lp0)
    dcache = draft.make_cache()
    start = max(0, P - (SWIN - 1))
    for c in dcache:
        c.offset = start
    ctx = T[start:P]
    ref2.apply()
    a = P
    set_arm("gate")
    out, kept, cut = [tok], [], False
    td = time.perf_counter()
    while len(out) < cap and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        ref2.remove()
        pth, cfa = propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
        dr, cf = np.array(pth[0]).tolist(), np.array(cfa[0]).tolist()
        dr = dr[:int((np.cumprod(np.array(cf)) >= TRUNC).sum())]   # Leap's S45c: L1-14 running-confidence truncation
        dr = dr[:ROWS - 1]                              # E2 small-C mode: rows x top-10 must fit the base pool (C_m)
        if (trim := dcache[0].offset - a) > 0:
            DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        TAP["on"] = False
        am = mx.argmax(lg[0], axis=-1)
        tp = taps()
        wv = TXT._last_widened
        mx.eval(am, tp, wv)                              # the loop's own eval (D-23: am, tp) + the widened stream it computed
        am = am.tolist()
        k = 0
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        kept.append(k + 1)
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        m = len(new)                                     # rows 0..m-1 hold the inputs tok, new[:-1]: correct prefixes
        ref2.remove()                                    # the logging ops on the stock path (outside the verify scope)
        tk_, lp_ = top32(lg[0, :m])
        L.add(hbits(tp[:m]), wv[0, :m].astype(mx.float16), tk_, lp_)
        ref2.apply()
        ctx = tp[:k + 1]
        out += new
        tok, a = out[-1], a + k + 1
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    bad = ""
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f"INVALID ({e})"
    set_arm("slot")
    ref_on(False)
    del cache, dcache
    mx.clear_cache()
    return out, L, dict(pf=pf, dec=dec, checks=len(kept), kept=float(np.mean(kept)) if kept else 0.0, cut=cut, bad=bad)

def save(key, lang, tmpl, ids, out, L):
    full = np.array(list(ids) + list(out), np.int32)
    N, P = len(full), len(ids)
    lo = max(0, P - 2048)
    hb, wds, tks, lps = L.np()
    h = np.zeros((N - 1, 2560), np.uint16)
    hcat = np.concatenate(hb)
    assert hcat.shape[0] == N - 1 - lo, (key, hcat.shape, N, lo)
    h[lo:] = hcat
    wd = np.concatenate(wds); tk = np.zeros((N - 1, TOPK), np.int32); lp = np.zeros((N - 1, TOPK), np.float16)
    assert wd.shape == (N - 1, 10240), (key, wd.shape, N)
    tkc, lpc = np.concatenate(tks), np.concatenate(lps)
    assert tkc.shape[0] == N - P, (key, tkc.shape, N, P)
    tk[P - 1:], lp[P - 1:] = tkc, lpc
    asst = np.zeros(N, np.uint8); asst[P:] = 1
    onp = float(np.mean(tk[P - 1:, 0] == full[P:])) if N > P else 1.0
    path = os.path.join(OUTD, key.replace("/", "__") + ".npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, h=h, wd=wd, tk=tk, lp=lp, ids=full, asst=asst, split="train", gen=P, key=key, lang=lang, tmpl=tmpl,
                 src=f"dlog-{SRC}", preset=PRESET)
    os.replace(tmp, path)
    return path, onp, os.path.getsize(path)

# ---------------------------------------------------------------- driver (resumable)
done_tok = {l: 0 for l in LANGS}
written = 0
if os.path.exists(IDX):
    for l in open(IDX):
        d = json.loads(l)
        if os.path.exists(os.path.join(OUTD, d["key"].replace("/", "__") + ".npz")):
            done_tok[d["lang"]] = done_tok.get(d["lang"], 0) + d["n"]; written += d["bytes"]
have = {f[:-4].replace("__", "/") for f in os.listdir(OUTD) if f.endswith(".npz")}
say(f"resume: {len(have)} prompts on disk, answer tokens {done_tok}, {written / 1e9:.2f} GB")
it0 = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
out0, L0, r0 = run(list(it0["prompt_ids"]), 64)          # warm-up (discarded), as D-23; its log goes through save() once so a
p0, onp0, nb0 = save("warmup/gsm8k708", "en", "-", list(it0["prompt_ids"]), out0, L0)   # format error stops here, in the first minute
z0 = np.load(p0)
assert z0["h"].shape[0] == z0["tk"].shape[0] == z0["wd"].shape[0] == len(z0["ids"]) - 1 and int(z0["gen"]) == len(it0["prompt_ids"])
assert np.any(z0["h"][-1] != 0) and np.all(np.isfinite(z0["wd"][-1].astype(np.float32))), "empty / non-finite rows"
os.remove(p0)
say(f"warm-up (discarded): gsm8k/708 {len(out0)} tokens, {len(out0) / r0['dec']:.1f} tok/s, logged file OK ({nb0 / 1e6:.1f} MB, "
    f"on-path {onp0:.4f}) {r0['bad']}")
if onp0 < 0.995:
    say(f"DLOG-ONPATH-FAIL: the warm-up's logged top-1 == next token on only {onp0:.4f} of answer rows: stopping"); sys.exit(3)
n_new, t_dec, n_dec = 0, 0.0, 0
for key, lang, tmpl, user in PLAN:
    if key in have:
        continue
    if all(done_tok[l] >= TARGET for l in LANGS):
        say(f"every language has >= {TARGET} answer tokens: done"); break
    if done_tok[lang] >= TARGET:
        continue
    if el() > BUDGET:
        say("time budget: stopping (resumable)"); break
    if written / 1e9 >= CAP_GB:
        say(f"cap {CAP_GB} GB reached: stopping"); break
    free = shutil.disk_usage(OUTD).free / 1e9
    if free < MIN_FREE:
        say(f"free disk {free:.1f} GB < {MIN_FREE}: stopping"); break
    ids = ED.chat(TOKZ, user)
    if n_new == 0:
        say(f"first prompt as tokenized: {TOKZ.decode(ids)!r}")
    out, L, r = run(ids, CAP)
    if r["bad"]:
        say(f"{key}: {r['bad']}: not saved, stopping (the gate check failed)"); break
    path, onp, nb = save(key, lang, tmpl, ids, out, L)
    n = len(out)
    rec = dict(key=key, lang=lang, tmpl=tmpl, P=len(ids), n=n, onpath=round(onp, 4), bytes=nb, checks=r["checks"],
               kept=round(r["kept"], 3), dec_s=round(r["dec"], 2), pf_s=round(r["pf"], 2), cut=r["cut"], C=C, at=time.strftime("%m-%d %H:%M:%S"))
    with open(IDX, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done_tok[lang] += n; written += nb; n_new += 1; t_dec += r["dec"]; n_dec += n
    say(f"{key} [{tmpl}] P {len(ids)}, {n} tokens, {n / max(r['dec'], 1e-9):.1f} tok/s, kept {r['kept']:.2f}/check, on-path "
        f"{onp:.4f}, {nb / 1e6:.1f} MB; totals {done_tok}, {written / 1e9:.2f} GB")
    if onp < 0.995:
        say(f"DLOG-ONPATH-FAIL: {key} logged top-1 == next token on only {onp:.4f} of answer rows: stopping"); break
    if r["cut"]:
        break
say(f"session: {n_new} prompts, {n_dec} answer tokens in {t_dec:.0f} s of decode ({n_dec / max(t_dec, 1e-9):.1f} tok/s); totals "
    f"{done_tok}, {written / 1e9:.2f} GB on disk; footprint now {footprint_now_gb():.1f} GB, peak {footprint_peak_gb():.1f} GB; "
    f"{time.time() - t0:.0f} s")
