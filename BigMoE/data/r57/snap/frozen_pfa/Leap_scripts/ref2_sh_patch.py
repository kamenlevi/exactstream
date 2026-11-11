# SCREENED VERIFIER HEAD (Leap, Wed 2026-11-04 16:5x; TABLE_350 'SVH'). FAST, default OFF and inert: CFG["screened_head"] /
# REF2_SCREENED_HEAD=1. A registered 4-bit lm_head (screened_head_install(lm_head)) takes its <= 16-row bf16 calls
# (REF2 on: the verify pass) on qmm16c over a token subset: FR-Spec's top-N (default 98,304) + up to 4,096 session ids
# (screened_head_add(ids): the prompt at prefill, each check's committed ids). The subset's logits are scattered into a
# full-vocabulary -inf array (the engine's interface is unchanged). Inside the subset the logits equal REF2's full head
# bitwise (qmm16c / qmm16n columns are independent of N); a row whose true argmax is outside changes (FAST). With REF2
# removed (the drafter's propose, the MTP chain) the head is untouched. Patches ref2.py (backup
# data/ref2_before_sh.py), atomic write.
# Usage: python ref2_sh_patch.py [--dry]
import os, sys, shutil, ast
DRY = "--dry" in sys.argv
p = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts/ref2.py")
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
rep("""  - D5 (Wed 2026-11-04, EXACT, off by default; CFG["hc_pend_mlp"] or REF2_HC_PEND=1): a layer's MLP-side residual write
""", """  - SVH (Wed 2026-11-04, FAST, off by default; CFG["screened_head"] or REF2_SCREENED_HEAD=1): the screened verifier head.
    screened_head_install(lm_head) registers the trunk's 4-bit head; its <= 16-row calls (REF2 on) run qmm16c over
    FR-Spec's top-N (CFG screened_head_n, 98,304) + up to 4,096 session ids (screened_head_add(ids) / _reset()), and
    the logits are scattered into a full-vocabulary -inf array. Inside the subset == REF2's full head bitwise; rows
    whose true argmax is outside change. COUNTS screened_head.
  - D5 (Wed 2026-11-04, EXACT, off by default; CFG["hc_pend_mlp"] or REF2_HC_PEND=1): a layer's MLP-side residual write
""")
rep('''       "hc_pend_mlp": os.environ.get("REF2_HC_PEND") == "1"}''',
    '''       "hc_pend_mlp": os.environ.get("REF2_HC_PEND") == "1",
       "screened_head": os.environ.get("REF2_SCREENED_HEAD") == "1",
       "screened_head_n": int(os.environ.get("REF2_SCREENED_HEAD_N", "98304"))}''')
rep("""small_native=0, hc_pend=0)""", """small_native=0, hc_pend=0, screened_head=0)""")
rep('''def _qlin(self, x):
    """DENSE-N / SMALL-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless one of
    the switches is on)"""
''', '''_SH = {}                                             # SVH: id(lm_head weight) -> its subset state

def screened_head_install(lm_head, base_n=None, cap=4096, rank_path=None):
    """SVH: register the trunk's 4-bit g64 lm_head (bf16 scales) for the screened verifier head: the base subset
    (FR-Spec's top base_n ids, sorted) gathered once, plus room for cap session ids. Returns the subset's base rows."""
    w, sc, bi = lm_head["weight"], lm_head["scales"], lm_head["biases"]
    assert lm_head.bits == 4 and lm_head.group_size == 64 and sc.dtype == mx.bfloat16 and w.ndim == 2, "4-bit g64 head"
    base_n = base_n or CFG["screened_head_n"]
    rank = np.load(rank_path or os.path.expanduser("~/Leap/data/frspec_rank.npy"))
    ids = np.unique(np.asarray(rank[:base_n], np.int64)).astype(np.int32)
    base = mx.array(ids)
    e = dict(w=w, s=sc, b=bi, V=int(w.shape[0]), ids=base, set=set(ids.tolist()), extra=[], cap=int(cap),
             wb=mx.contiguous(mx.take(w, base, axis=0)), sb=mx.contiguous(mx.take(sc, base, axis=0)),
             bb=mx.contiguous(mx.take(bi, base, axis=0)),
             xids=mx.zeros((cap,), mx.int32), xw=mx.zeros((cap, w.shape[1]), w.dtype), xs=mx.zeros((cap, sc.shape[1]), sc.dtype),
             xb=mx.zeros((cap, bi.shape[1]), bi.dtype))
    mx.eval(e["ids"], e["wb"], e["sb"], e["bb"], e["xids"], e["xw"], e["xs"], e["xb"])
    _SH.clear()
    _SH[id(w)] = e
    return int(ids.size)

def screened_head_add(token_ids):
    """SVH: the session's ids (prompt, committed tokens) outside the base join the subset (once each, up to cap)"""
    for e in _SH.values():
        new = [t for t in dict.fromkeys(int(t) for t in np.asarray(token_ids).reshape(-1)) if t not in e["set"]]
        new = new[:max(0, e["cap"] - len(e["extra"]))]
        if not new:
            continue
        a, n = len(e["extra"]), len(new)
        idx = mx.array(np.asarray(new, np.int32))
        e["xids"][a:a + n] = idx
        e["xw"][a:a + n] = mx.take(e["w"], idx, axis=0)
        e["xs"][a:a + n] = mx.take(e["s"], idx, axis=0)
        e["xb"][a:a + n] = mx.take(e["b"], idx, axis=0)
        mx.eval(e["xids"], e["xw"], e["xs"], e["xb"])
        e["extra"] += new
        e["set"].update(new)
    return sum(len(e["extra"]) for e in _SH.values())

def screened_head_reset():
    """SVH: drop the session ids (a new prompt)"""
    for e in _SH.values():
        e["set"] -= set(e["extra"])
        e["extra"] = []

def _screened(e, x, scales, biases):
    lead, rows = _rows(x)
    if rows > R or x.dtype != mx.bfloat16:
        return None
    x16 = _pad16(x.reshape(rows, x.shape[-1]))
    y = qmm16c(x16, e["wb"], e["sb"], e["bb"])[:rows]
    ids = e["ids"]
    n = len(e["extra"])
    if n:
        yx = qmm16c(x16, e["xw"], e["xs"], e["xb"])[:rows, :n]
        y = mx.concatenate([y, yx], axis=1)
        ids = mx.concatenate([ids, e["xids"][:n]])
    out = mx.put_along_axis(mx.full((rows, e["V"]), -float("inf"), y.dtype),
                            mx.broadcast_to(ids[None].astype(mx.int32), y.shape), y, axis=1)
    COUNTS["screened_head"] += 1
    return out.reshape(*lead, e["V"])

def _qlin(self, x):
    """DENSE-N / SMALL-N: nn.QuantizedLinear's call with its weight marked as resident for _qmm (inert unless one of
    the switches is on); SVH: the registered lm_head's screened call (CFG screened_head)"""
    if _SH and _ON[0] and CFG["screened_head"]:
        e = _SH.get(id(self["weight"]))
        if e is not None and e["w"] is self["weight"]:
            y = _screened(e, x, self["scales"], self["biases"])
            if y is not None:
                return y
''')
ast.parse(s)
if DRY:
    d = os.environ.get("TMPDIR", "/tmp")
    open(os.path.join(d, "ref2_sh_dry.py"), "w").write(s)
    print("dry: applies and parses ->", d)
else:
    shutil.copy(p, os.path.expanduser("~/Leap/data/ref2_before_sh.py"))
    tmp = p + ".tmp_sh"
    open(tmp, "w").write(s)
    os.replace(tmp, p)
    print("applied (backup data/ref2_before_sh.py); now self_test + sh_wire_check.py")
