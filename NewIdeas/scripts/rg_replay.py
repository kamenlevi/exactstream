"""Replay the repetition guard's rule over stored guard-off decodes (CPU only, no model).
For each row of an Eval suite decode.jsonl, drive MTPLX's real LoopGuard (BigMoE's repguard) over the committed tokens one
step at a time, as the decode loop does while armed, and record the first position where the guard would subtract a
penalty from the token the model actually chose ("hit": the guard-on output can diverge there; rows without a hit are
provably unchanged under greedy decoding, since penalties on other tokens cannot move the argmax) and the largest such
penalty in the row. Variants:
  R0      the shipped rule (12-gram x4 arms, DRY steering from 12-token matches)
  R1      R0 + LaTeX math spans masked like tool-call spans ($..$, $$..$$, \\(..\\), \\[..\\], \\begin{align|equation|..})
  Rp      R0 but steering only on matches that end a periodic tail (the last 3 matches of the suffix are equally spaced)
  R1p     R1 + Rp
  R2      R1, and arming also needs >= 50 % of the last 512 unmasked tokens covered by 12-grams recurring >= 4x there
          (Welleck et al. 2020's seq-rep-n degeneration measure used as the arming test). MISSES LOOSE LOOPS (--loose).
  R3      R1, and arming needs the R2 coverage OR one unmasked 12-gram recurring >= 8x in the window
--loose: synthetic loops from the ref's IFEval prose, a 16-token connective sentence every 16 + K tokens (K = 0..320), as in
MTPLX's chess marathons (fresh text between cycles): where each variant arms and first hits.
A loop row = capped row whose last 1,024 tokens are >= 60 % covered by 12-grams recurring >= 4x in that tail.
usage: rg_replay.py RUN [RUN ...]   (runs under ~/EvalSuite/results)"""
import json, os, re, sys
import numpy as np
from collections import Counter
from mtplx.loop_guard import LoopGuard, loop_guard_config_from_env
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/BigMoE/models/bare"))
CFG = loop_guard_config_from_env(True, tokenizer=tok)
OPEN, CLOSE = CFG.mask_open_token, CFG.mask_close_token
ENVS = re.compile(r"\\(begin|end)\{(align|equation|gather|multline|eqnarray|array|cases|pmatrix|bmatrix)\*?\}")


def tool_mask(ids):
    m, s = [], False
    for t in ids:
        if t == OPEN:
            s = True; m.append(True)
        elif t == CLOSE:
            m.append(True); s = False
        else:
            m.append(s)
    return m


def math_mask(ids):
    """per-token: True if any character of the token lies in a LaTeX math span (delimiters included)"""
    pieces = [tok.decode([t]) for t in ids]
    text = "".join(pieces)
    flag = bytearray(len(text))
    st, i, n = None, 0, len(text)
    while i < n:
        c2 = text[i:i + 2]
        m = ENVS.match(text, i) if text[i] == "\\" else None
        if m:
            st = "env" if m.group(1) == "begin" else (None if st == "env" else st)
            flag[i:m.end()] = b"\x01" * (m.end() - i); i = m.end(); continue
        if st in (None, "$") and c2 == "\n\n":
            st = None
        if c2 == "$$" and st in (None, "$$"):
            st = None if st == "$$" else "$$"; flag[i:i + 2] = b"\x01\x01"; i += 2; continue
        if text[i] == "$" and st in (None, "$"):
            st = None if st == "$" else "$"; flag[i] = 1; i += 1; continue
        if c2 in ("\\(", "\\[") and st is None:
            st = c2; flag[i:i + 2] = b"\x01\x01"; i += 2; continue
        if (c2 == "\\)" and st == "\\(") or (c2 == "\\]" and st == "\\["):
            st = None; flag[i:i + 2] = b"\x01\x01"; i += 2; continue
        if st is not None:
            flag[i] = 1
        i += 1
    out, p = [], 0
    for s in pieces:
        out.append(any(flag[p:p + len(s)]) if s else (out[-1] if out else False)); p += len(s)
    return out


class Replay(LoopGuard):
    """LoopGuard with a precomputed (causal) span mask and an optional periodic-tail condition on steering."""

    def __init__(self, cfg, mask, periodic, cov_min=0.0, count_min=0):
        super().__init__(cfg)
        self._masking = True
        self._full = mask
        self._periodic = periodic
        self._cov_min = cov_min
        self._count_min = count_min

    def observe(self, tokens):
        ev = super().observe(tokens)
        if ev == "armed" and self._cov_min > 0:
            keep = [t for t, m in zip(tokens, self._full) if not m]
            ok = loop_tail(keep[-512:], tail=512) >= self._cov_min
            if not ok and self._count_min:
                win = keep[-self.config.window:]
                c = Counter(tuple(win[i:i + 12]) for i in range(len(win) - 11))
                ok = bool(c) and max(c.values()) >= self._count_min
            if not ok:
                self.armed = False
                self.arm_events -= 1
                return None
        return ev

    def _advance_mask(self, tokens):
        self._mask = self._full[:len(tokens)]

    def _working_mask(self, working, n):
        return np.asarray(self._full[:len(working)][-n:], dtype=bool)

    def penalties_for(self, working):
        pen = super().penalties_for(working)
        if not pen or not self._periodic:
            return pen
        # keep steering only when the current suffix ends a run of >= 3 equally spaced recent occurrences (a cycle)
        arr = np.asarray(working[-self.config.window:], dtype=np.int64)
        L = self.config.allowed_length
        suf = arr[-L:]
        occ = [j for j in range(len(arr) - L, -1, -1) if np.array_equal(arr[j:j + L], suf)][:4]
        if len(occ) < 3:
            return None
        gaps = [occ[k] - occ[k + 1] for k in range(len(occ) - 1)]
        if max(gaps) - min(gaps) > max(2, min(gaps) // 8):
            return None
        return pen


def loop_tail(ids, tail=1024, N=12, K=4):
    a = ids[-tail:]
    if len(a) < N * K:
        return 0.0
    c = Counter(tuple(a[i:i + N]) for i in range(len(a) - N + 1))
    cov = np.zeros(len(a), bool)
    for i in range(len(a) - N + 1):
        if c[tuple(a[i:i + N])] >= K:
            cov[i:i + N] = True
    return float(cov.mean())


def replay(ids, mask, periodic, cov_min=0.0, count_min=0):
    g = Replay(CFG, mask, periodic, cov_min, count_min)
    """first arming position, first hit (penalty on the chosen token), and the largest penalty any chosen token in the row
    would take if no earlier hit had flipped the argmax (>= ~8 logits: a flip is likely)"""
    first_arm = first_hit = None
    max_val = 0.0
    for i in range(len(ids)):
        g.observe(ids[:i])
        if g.armed:
            if first_arm is None:
                first_arm = i
            pen = g.penalties_for(ids[:i])
            if pen and ids[i] in pen:
                if first_hit is None:
                    first_hit = i
                max_val = max(max_val, pen[ids[i]])
    return first_arm, first_hit, max_val


def VARIANTS(tm, both):
    return (("R0", tm, False, 0, 0), ("R1", both, False, 0, 0), ("Rp", tm, True, 0, 0), ("R1p", both, True, 0, 0),
            ("R2", both, False, 0.5, 0), ("R3", both, False, 0.5, 8))


def loose():
    rows = [json.loads(l) for l in open(os.path.expanduser("~/EvalSuite/results/ref/decode.jsonl"))]
    fresh = [t for r in rows if r["set"] == "ifeval" for t in r["completion_ids"]]
    sent = tok.encode(" OK, let me start building the files now, one at a time.\n\n")
    for K in (0, 20, 40, 80, 160, 320):
        seq, p = [], 0
        while len(seq) < 3000:
            seq += sent + fresh[p:p + K]
            p += K
        seq = seq[:3000]
        tm = tool_mask(seq)
        both = [a or b for a, b in zip(tm, math_mask(seq))]
        out = {name: replay(seq, m, per, cov, cnt)[:2] for name, m, per, cov, cnt in VARIANTS(tm, both)}
        print(f"K={K:3d} (cycle {len(sent) + K}): armed / first hit", out)


def main():
    if sys.argv[1] == "--loose":
        return loose()
    res = {}
    for run in sys.argv[1:]:
        rows = [json.loads(l) for l in open(os.path.expanduser(f"~/EvalSuite/results/{run}/decode.jsonl"))]
        out = []
        for r in rows:
            ids = r["completion_ids"]
            tm = tool_mask(ids)
            mm = math_mask(ids)
            both = [a or b for a, b in zip(tm, mm)]
            rec = {"key": r["key"], "set": r["set"], "n": len(ids), "finish": r["finish"],
                   "loop": loop_tail(ids) if r["finish"] == "length" else 0.0, "math_frac": round(sum(mm) / max(1, len(mm)), 3)}
            for name, mask, per, cov, cnt in VARIANTS(tm, both):
                a, h, v = replay(ids, mask, per, cov, cnt)
                rec[name] = [a, h, round(v, 2)]
                if name == "R0" and h is not None:
                    rec["R0_ctx"] = tok.decode(ids[max(0, h - 30):h]) + " [[" + tok.decode([ids[h]]) + "]]"
            out.append(rec)
        res[run] = out
        loops = [o for o in out if o["loop"] >= 0.6]
        normal = [o for o in out if o["loop"] < 0.6]
        print(f"\n== {run}: {len(out)} rows, {len(loops)} loop rows (capped, tail >= 60 % recurring 12-grams)")
        for name in ("R0", "R1", "Rp", "R1p", "R2", "R3"):
            arm_n = sum(o[name][0] is not None for o in normal)
            hit_n = sum(o[name][1] is not None for o in normal)
            big_n = sum(o[name][2] >= 8 for o in normal)
            by = Counter(o["set"] for o in normal if o[name][1] is not None)
            rec_arm = sum(o[name][0] is not None for o in loops)
            rec_hit = sum(o[name][1] is not None for o in loops)
            print(f"  {name}: normal rows armed {arm_n}, hit {hit_n} {dict(by)}, penalty >= 8 on a chosen token {big_n} | "
                  f"loop rows armed {rec_arm}/{len(loops)}, "
                  f"hit {rec_hit}/{len(loops)}")
    od = os.path.expanduser("~/NewIdeas/data/rg")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "replay_" + "_".join(sys.argv[1:]) + ".json"), "w") as f:
        json.dump(res, f, indent=0)


if __name__ == "__main__":
    main()
