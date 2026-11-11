"""SESSION-VOCAB check (Leap, Wed 2026-11-04 17:3x; TABLE_350 'SESSION-VOCAB'): the r57 hook harness for SVH + BP-9
(/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts/session_vocab.py; the proposed r57 diff ~/Leap/data/r57_session_vocab.diff), on the 125B's real 4-bit
embed / lm_head and the D8-NAX drafter (as BP-9).
 (a) subsets: new_prompt / committed / the cap / a new prompt -> the SVH (ref2) and DSUB (DraftSubset) extras hold the
     same ids in the same order = the session's unique ids outside the base; a new prompt drops the last one's;
 (b) head2s (the drafter's hidden rows = the block's positions 1.., P = 15 at a 16-position block): SV.head2s ==
     d7b_nax.screened_head2s bitwise; R57_D17's P + 1 = 16 rows: rows :P == the P-row call's, row P (a copy of row 3)
     == row 3's, bitwise; 18 rows (the > 16-row chunk path) == a 16-row call + a 2-row call, bitwise; every candidate
     inside the subset, finite; check-mode counters count the rows;
 (c) SVH only with REF2 on: ref2 applied -> the trunk head at 16 rows screened (counted, finite exactly on the subset),
     17 rows -> the full head (uncounted); REF2 removed (r57 removes it around each propose) -> the plain head;
 (d) the walk (BP-9's, ~/Leap/data/dfl2d_{code,math}, cut t 0.3): A = BP-9's own DraftSubset + its feeding; B0 =
     SessionVocab with A's feeding (the plumbing: == A in every window); B = SessionVocab in r57's hook order
     (new_prompt(prompt), committed([first token]), committed(kept + the target's token) per check), check mode on;
     kept per check, committed() cost per check;
 (e) ref2.self_test PASS with SVH on; switches restored (CFG screened_head False at exit).
PASS (pre-registered) = (a), (b), (c), (e) true; B0 == A in every window; B / A within x0.995-x1.005 in both folders.
usage: [BigMoE venv] session_vocab_check.py -> ~/Leap/data/session_vocab_check.txt (GPU, MEM ~3 GB; share lock)"""
import os, sys, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
import dflash_model_mlx as DF
import draft_fused as DFU
from chain_fused import chain_fused
import d7b_nax
import ref2
import session_vocab as SVM

mx.set_cache_limit(256 << 20)
T_CUT, R = 0.3, 16
OUT = os.path.expanduser("~/Leap/data/session_vocab_check.txt")
def say(s=""):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
t00 = time.time()
assert "screened_head" in ref2.CFG and not ref2.CFG["screened_head"], "ref2 SVH patch missing or switch not default off"


class Target(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


d8 = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
d8.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
N8 = d7b_nax.apply8(d8)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
V = d8.config.vocab_size
tgt = Target(V)
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
mx.eval(tgt.parameters())
d8.bind(tgt)
mx.eval(d8.parameters())
DFU.install(forms=False, draft=d8)
MASK, SW = int(d8.config.mask_token_id), d8.config.sliding_window
say(f"\nSESSION-VOCAB check {time.strftime('%a %H:%M:%S')}: D8-NAX ({N8} NaxQL8) + the real 4-bit head; loaded in "
    f"{time.time() - t00:.0f} s")

# ---- (a) both subsets track the session the same way ----
SV = SVM.SessionVocab(tgt.lm_head, svh=True, dsub=True, check=True)
e = next(iter(ref2._SH.values()))
base = np.array(SV.sub.ids)
bset = set(base.tolist())
ok_a = ref2.CFG["screened_head"] and np.array_equal(np.array(e["ids"]), base) and e["set"] == SV.sub.set == bset
outside = np.setdiff1d(np.arange(V, dtype=np.int64), base)
rng = np.random.default_rng(0)
def uniq_out(seq, have):
    out, h = [], set(have)
    for t in (int(t) for t in seq):
        if t not in bset and t not in h:
            out.append(t); h.add(t)
    return out
def same(exp):
    return SV.sub.extra == exp and e["extra"] == exp and SV.sub.set == bset | set(exp) == e["set"] \
        and int(np.array(SV.sub.xok).sum()) == len(exp)
p1 = np.concatenate([rng.choice(outside, 300), rng.choice(base, 200), rng.choice(outside, 50)])
p1 = np.concatenate([p1, p1[:40]]); rng.shuffle(p1)
SV.new_prompt(p1); exp = uniq_out(p1, [])
st = [same(exp)]
c1 = np.concatenate([rng.choice(outside, 20), p1[:10], rng.choice(base, 5)])
SV.committed(c1); exp = exp + uniq_out(c1, exp)
st.append(same(exp))
SV.committed([]); st.append(same(exp))
big = rng.permutation(outside)[:4200]
SV.new_prompt(big); exp = [int(t) for t in big[:4096]]
st.append(same(exp) and len(exp) == 4096)
SV.committed(rng.choice(outside, 10)); st.append(same(exp))
p2 = rng.choice(outside, 30)
SV.new_prompt(p2); exp = uniq_out(p2, [])
st.append(same(exp))
ok_a = bool(ok_a and all(st))
say(f"(a) base {base.size} ids, the same in both; extras equal and as expected after prompt / commit / empty commit / cap "
    f"4,096 / past the cap / a new prompt: {st} -> {'OK' if ok_a else 'FAIL'}")

# ---- (b) head2s: bitwise == screened_head2s; 17 rows == 16 + 1; candidates valid ----
paths = sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz")))
z = np.load(paths[0])
f0, ids0, off0, gen0 = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
dc = d8.make_cache()
start = max(off0, gen0 - (SW - 1))
for c in dc:
    c.offset = start
hidP = d8.hidden_states(mx.array([[int(ids0[gen0 - off0])] + [MASK] * (R - 1)]),
                         mx.array(f0[start - off0:gen0 - off0][None]).astype(mx.bfloat16), dc, 1)
P = hidP.shape[1]
hid16 = mx.concatenate([hidP, hidP[:, 3:4]], axis=1)                       # R57_D17's drafter rows (P + 1)
hid18 = mx.concatenate([hidP, hidP[:, 3:4], hidP[:, 5:6], hidP[:, 7:8]], axis=1)
hid17 = mx.concatenate([hid16, hidP[:, :1]], axis=1)                       # (c): a 17-row trunk call
mx.eval(hidP, hid16, hid18, hid17)
assert hid16.shape[1] == 16 and hid18.shape[1] == 18, (hidP.shape, hid16.shape, hid18.shape)
SV.new_prompt(ids0[:gen0 - off0])
bits = lambda a: np.array(a.astype(mx.float32))
n0 = dict(SV.n)
ca, ua = SV.head2s(d8, hidP)
cb, ub = d7b_nax.screened_head2s(d8, hidP, SV.sub)
c16, u16 = SV.head2s(d8, hid16)
c18, u18 = SV.head2s(d8, hid18)
c18a, u18a = SV.head2s(d8, hid18[:, :16])
c18b, u18b = SV.head2s(d8, hid18[:, 16:])
mx.eval(ca, ua, cb, ub, c16, u16, c18, u18, c18a, u18a, c18b, u18b)
A_ = lambda c, u: (np.array(c), bits(u))
(ca, ua), (cb, ub), (c16, u16), (c18, u18), (c18a, u18a), (c18b, u18b) = (
    A_(ca, ua), A_(cb, ub), A_(c16, u16), A_(c18, u18), A_(c18a, u18a), A_(c18b, u18b))
eq = lambda x, y: np.array_equal(x, y)
eqP = eq(ca, cb) and eq(ua, ub)
eqD17 = eq(c16[:, :P], ca) and eq(u16[:, :P], ua) and eq(c16[:, P], ca[:, 3]) and eq(u16[:, P], ua[:, 3])
eq18 = (eq(c18[:, :16], c18a) and eq(u18[:, :16], u18a) and eq(c18[:, 16:], c18b) and eq(u18[:, 16:], u18b)
        and eq(c18[:, 16], ca[:, 5]) and eq(c18[:, 17], ca[:, 7]) and eq(u18[:, 16], ua[:, 5]) and eq(u18[:, 17], ua[:, 7]))
cand_all = np.concatenate([ca.reshape(-1), c16.reshape(-1), c18.reshape(-1)])
valid = bool(np.all((cand_all >= 0) & (cand_all < V)) and all(int(t) in SV.sub.set for t in cand_all)
             and all(np.all(np.isfinite(u)) for u in (ua, u16, u18)))
dn = {k: SV.n[k] - n0[k] for k in ("head2s", "chk_rows", "chk_out")}
cnt = dn["head2s"] == 5 and dn["chk_rows"] == P + 16 + 18 + 16 + 2 and 0 <= dn["chk_out"] <= dn["chk_rows"]
ok_b = bool(eqP and eqD17 and eq18 and valid and cnt)
say(f"(b) P = {P} rows: SV.head2s == screened_head2s bitwise {eqP}; D17's {P + 1} rows (row {P} == row 3) {eqD17}; 18 rows "
    f"== 16 + 2-row calls {eq18}; candidates inside the subset and finite {valid}; check counters {dn} -> "
    f"{'OK' if ok_b else 'FAIL'}")

# ---- (c) SVH only with REF2 on (the trunk head) ----
ref2.apply()
k0 = ref2.COUNTS["screened_head"]
y16, y17 = tgt.lm_head(hid16), tgt.lm_head(hid17)
mx.eval(y16, y17)
k1 = ref2.COUNTS["screened_head"]
fin16 = (~np.isneginf(bits(y16)[0])).sum(axis=1)
nsub = base.size + len(e["extra"])
ok_c = k1 == k0 + 1 and bool(np.all(fin16 == nsub)) and not bool(np.any(np.isneginf(bits(y17))))
ref2.remove()
y_off = tgt.lm_head(hid16); mx.eval(y_off)
ok_c = bool(ok_c and ref2.COUNTS["screened_head"] == k1 and not np.any(np.isneginf(bits(y_off))))
say(f"(c) REF2 on: 16 rows screened (finite on exactly the {nsub:,} subset ids: {bool(np.all(fin16 == nsub))}, counted "
    f"{k1 - k0}), 17 rows the full head; REF2 removed: the plain head, uncounted -> {'OK' if ok_c else 'FAIL'}")

# ---- (d) the walk: A (BP-9) vs B0 (SessionVocab, A's feeding) vs B (SessionVocab, r57's hook order, check mode) ----
SUBA = d7b_nax.DraftSubset(tgt.lm_head)
SVW = {"B0": SVM.SessionVocab(tgt.lm_head, dsub=True), "B": SVM.SessionVocab(tgt.lm_head, dsub=True, check=True)}
TC = {"n": 0, "t": 0.0, "grow": 0}


def walk(mode, folder):
    ks, per = [], []
    for p in sorted(glob.glob(os.path.expanduser(f"~/Leap/data/{folder}/*.npz"))):
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        dcache = d8.make_cache()
        prompt = ids[:max(0, gen - off)]
        if mode == "A":
            SUBA.reset(); SUBA.add(prompt)
        else:
            SVW[mode].new_prompt(prompt)
            if mode == "B":
                SVW[mode].committed([int(ids[gen - off])])     # r57: SV.committed([tok]) after the prefill's argmax
        start = max(off, gen - (SW - 1))
        for c in dcache:
            c.offset = start
        fed, a, kw = start, gen, []
        while a < off + len(ids) - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            blk = mx.array([[int(ids[a - off])] + [MASK] * (R - 1)])
            hid = d8.hidden_states(blk, ctx, dcache, 1)
            cand, unary = (d7b_nax.screened_head2s(d8, hid, SUBA) if mode == "A" else SVW[mode].head2s(d8, hid))
            pth, cf = chain_fused(blk, unary, cand, d8.candidate_selector.hidden_projection(hid), d8.candidate_selector)
            mx.eval(pth, cf)
            path, cfl = np.array(pth[0]).tolist(), np.array(cf[0].astype(mx.float32)).tolist()
            c_ = int((np.cumprod(np.array(cfl)) >= T_CUT).sum())
            fed = a
            true = ids[a - off + 1:a - off + 1 + c_]
            k = 0
            while k < len(true) and path[k] == true[k]:
                k += 1
            kw.append(k + 1)
            new = ids[a - off + 1:a - off + k + 2]
            if mode == "A":
                SUBA.add(new)
            else:
                n_ = len(SVW[mode].sub.extra)
                t0 = time.perf_counter()
                SVW[mode].committed(new)
                if mode == "B":
                    TC["t"] += time.perf_counter() - t0; TC["n"] += 1; TC["grow"] += len(SVW[mode].sub.extra) > n_
            if a + k + 1 - off >= len(ids):
                break
            a = a + k + 1
        ks += kw; per.append((sum(kw), len(kw)))
    return float(np.mean(ks)), np.array(per, float)


ok_d, lines = True, []
for folder in ("dfl2d_code", "dfl2d_math"):
    o = {m: walk(m, folder) for m in ("A", "B0", "B")}
    eq0 = bool(np.array_equal(o["B0"][1], o["A"][1]))
    rB = o["B"][0] / o["A"][0]
    pa, pb = o["A"][1], o["B"][1]
    lo = int(np.sum(pb[:, 0] / pb[:, 1] < pa[:, 0] / pa[:, 1])); hi = int(np.sum(pb[:, 0] / pb[:, 1] > pa[:, 0] / pa[:, 1]))
    ok_d &= eq0 and 0.995 <= rB <= 1.005
    say(f"(d) {folder}: kept per check A (BP-9) {o['A'][0]:.3f}, B0 {o['B0'][0]:.3f} (== A in every window: {eq0}), "
        f"B (r57 order) {o['B'][0]:.3f} = x{rB:.4f} (lower in {lo}, higher in {hi} of {len(pa)})")
nB = SVW["B"].n
say(f"    B's check mode: {nB['chk_rows']:,} draft rows, full top-1 outside the subset in {nB['chk_out']:,} "
    f"({100 * nB['chk_out'] / max(1, nB['chk_rows']):.2f} %); committed(): {1e3 * TC['t'] / max(1, TC['n']):.3f} ms per "
    f"check (mean; {TC['grow']} of {TC['n']} checks gathered new rows); B extras at the end {len(SVW['B'].sub.extra)}")

# ---- (e) ref2.self_test with SVH on; restore ----
ref2.apply()
stt = ref2.self_test()
ok_e = all(bool(v) for v in stt.values())
ref2.remove()
ref2.CFG["screened_head"] = False
ok_e = bool(ok_e and not ref2.CFG["screened_head"])
say(f"(e) ref2.self_test with SVH on: {'PASS' if ok_e else 'FAIL ' + str(stt)}; CFG screened_head restored off")
ok = ok_a and ok_b and ok_c and bool(ok_d) and ok_e
say(f"SESSION-VOCAB check: {'PASS' if ok else 'FAIL'} (a {ok_a}, b {ok_b}, c {ok_c}, d {bool(ok_d)}, e {ok_e}); mlx peak "
    f"{mx.get_peak_memory() / 2**30:.2f} GB; {time.time() - t00:.0f} s")
