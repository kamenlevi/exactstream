"""Repetition guard for BigMoE's own decode loops (decode.greedy, SpecDecode's spec125.generate): MTPLX's LoopGuard
(loop-armed DRY steering, mtplx/loop_guard.py) on the committed tokens. Release must-have (Eval suite: the 3-bit plan
loops to the cap on ifeval/1219).

Exactness: while disarmed nothing changes (same graph, same argmax). It arms only when a 12-gram recurs 4x in the last
2,048 tokens past 256 tokens; while armed the loops decode 1-token steps and subtract the guard's penalties from that
step's logits before the argmax; it disarms after 256 tokens without a penalty. On by default; BIGMOE_LOOP_GUARD=0 off.
MTPLX knobs (MTPLX_LOOP_GUARD_*) apply.

R1 (NewIdeas GAPS.md G, 10-22; default on, BIGMOE_LOOP_GUARD_MATH=0 = the shipped rule R0): LaTeX math spans ($..$,
$$..$$, \\(..\\), \\[..\\], align/equation/... environments) are masked exactly like MTPLX's tool-call spans: their
shingles never arm the guard and their positions are never steered. The shipped rule armed on repeated formula lines and
steered correct math ('= 32' -> '= 16' on math500 prealgebra/1640; 46 of 250 replayed math rows touched); R1 touches 0
of 250 and still arms on every loop tested. The span mask is computed on demand (scan and steer only), from the decoded
token pieces, with NewIdeas' rg_replay.py state machine."""
import os
import re
import numpy as np
import mlx.core as mx

STATS = {"armed": 0, "steered": 0, "penalized": 0}
_TOK = {}
_ENVS = re.compile(r"\\(begin|end)\{(align|equation|gather|multline|eqnarray|array|cases|pmatrix|bmatrix)\*?\}")


def _tokenizer():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(os.path.expanduser(os.environ.get("BIGMOE_MD", "~/BigMoE/models/bare")))
    return _TOK["t"]


def _math_flags(pieces):
    """per piece: True if any character lies in a LaTeX math span (delimiters included); NewIdeas rg_replay.math_mask"""
    text = "".join(pieces)
    flag = bytearray(len(text))
    st, i, n = None, 0, len(text)
    while i < n:
        c2 = text[i:i + 2]
        m = _ENVS.match(text, i) if text[i] == "\\" else None
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


def _guard_class():
    from mtplx.loop_guard import LoopGuard

    class R1Guard(LoopGuard):
        """LoopGuard whose span mask = tool-call spans | LaTeX math spans, computed on demand."""

        def __init__(self, cfg, tok):
            super().__init__(cfg)
            self._masking = True
            self._tok, self._ids, self._pieces = tok, [], []

        def _full_mask(self, tokens):
            k = 0                                              # reuse decoded pieces for the common prefix
            lim = min(len(tokens), len(self._ids))
            while k < lim and self._ids[k] == tokens[k]:
                k += 1
            self._ids = list(tokens)
            self._pieces = self._pieces[:k] + [self._tok.decode([int(t)]) for t in tokens[k:]]
            o, c = self.config.mask_open_token, self.config.mask_close_token
            tool, s = [], False
            for t in tokens:
                if o is not None and t == o:
                    s = True; tool.append(True)
                elif c is not None and t == c:
                    tool.append(True); s = False
                else:
                    tool.append(s)
            return [a or b for a, b in zip(tool, _math_flags(self._pieces))]

        def _advance_mask(self, tokens):                       # nothing per token: the mask is built when read
            pass

        def _shingle_recurrence(self, tokens):
            self._mask = self._full_mask(tokens)
            return super()._shingle_recurrence(tokens)

        def _working_mask(self, working, n):
            return np.asarray(self._full_mask(working)[-n:], dtype=bool)

    return R1Guard


def make(tokenizer=None):
    if os.environ.get("BIGMOE_LOOP_GUARD", "1") != "1":
        return None
    from mtplx.loop_guard import LoopGuard, loop_guard_config_from_env
    tok = tokenizer or _tokenizer()
    cfg = loop_guard_config_from_env(True, tokenizer=tok)
    if not cfg.enabled:
        return None
    if os.environ.get("BIGMOE_LOOP_GUARD_MATH", "1") == "1":
        return _guard_class()(cfg, tok)
    return LoopGuard(cfg)


def observe(guard, out):
    if guard is not None and guard.observe(out) == "armed":
        STATS["armed"] += 1


def steer(guard, working, logits):
    """logits [V] (any dtype) -> float32 logits with the guard's penalties for the position after `working`."""
    lg = logits.astype(mx.float32)
    pen = guard.penalties_for(working) if guard is not None and guard.armed else None
    STATS["steered"] += 1
    if pen:
        STATS["penalized"] += 1
        idx = mx.array(list(pen.keys()), dtype=mx.int32)
        lg = lg.at[idx].add(-mx.array(list(pen.values()), dtype=mx.float32))
    return lg
