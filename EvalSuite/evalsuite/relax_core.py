"""RELAX's pure helpers (docs/PREREG_relax_2026-11-02.md): the row gate, the acceptance rules, the expert-union bytes,
the top-m KLD and the token classes. No model, no GPU, no package imports, so relax.py, replay.py,
tools/relax_report.py and its selftest share one definition of each rule."""
from __future__ import annotations
import math
import numpy as np

R = 16                       # rows per check: the last kept token + 15 drafts
COPY_N = 8                   # the copy gate's span: the row's last 8 input tokens
LN_BLUNDER = math.log(0.01)  # a blunder: the exact model gives the committed token p < 0.01


def prompt_grams(ids, n=COPY_N):
    """the n-grams of the prompt: the copy gate's sources (the context before the answer, COPYGATE's rule 1)"""
    return {tuple(ids[i - n:i]) for i in range(n, len(ids) + 1)}


def gate_rows(cf, tail, drafts, thr, grams):
    """per row of a 16-row check: 0 exact, 1 cf gate, 2 copy gate.
    - cf gate: the drafter's cf for the draft the row verifies is >= thr (row j verifies draft j, cf[j]);
    - copy gate: a row the cf gate leaves exact whose last 8 input tokens occur in the prompt;
    - row 15 (the bonus after 15 kept drafts) is always exact.
    tail: the committed tokens so far (at least the last 8; the last one is row 0's input); drafts: the 15 drafts."""
    g = [0] * R
    if thr is not None:
        for j in range(R - 1):
            if cf[j] >= thr:
                g[j] = 1
    if grams is not None:
        w = list(tail[-COPY_N:]) + list(drafts)
        assert len(w) == COPY_N + R - 1, len(w)
        for j in range(R - 1):
            if not g[j] and tuple(w[j:j + COPY_N]) in grams:
                g[j] = 2
    return g


def keep_mask(g, k_top=10, k_gate=2):
    """[R, k_top] bool, or None when no row is gated: a gated row keeps its ranks < k_gate"""
    if not any(g):
        return None
    m = np.ones((R, k_top), dtype=bool)
    m[[j for j in range(R) if g[j]], k_gate:] = False
    return m


R1 = ("R1", 0.5)             # accept the draft while log p(d) >= log p1 - 0.5 nat
R2 = ("R2", 2, 0.25)         # accept the draft while it is the row's top-2 and p(d) >= 0.25 p1
TOOL_PREFIXES = ("#type:", "#noqa", "#pragma", "#fmt:", "#pylint", "#mypy", "#-*-coding", "#!")


def rule_ok(rule, dlm, drk):
    """does one relaxed rule accept a draft: dlm = logit(d) - max logit on its row (<= 0), drk = the draft's rank
    there (0 = the argmax: tokens with a strictly larger logit)"""
    if rule[0] == "R1":
        return dlm >= -rule[1]
    if rule[0] == "R2":
        return drk < rule[1] and dlm >= math.log(rule[2])
    raise ValueError(rule)


def kept_rows(drafts, am, dlm, drk, rules):
    """(k exact, k under the rules). rules[j]: the relaxed rules row j may also use (a tuple, empty = exact). Exact
    acceptance is live16b's (the draft equals the row's argmax); a relaxed rule keeps every draft it keeps."""
    kx = 0
    while kx < R - 1 and drafts[kx] == am[kx]:
        kx += 1
    k = 0
    while k < R - 1 and (drafts[k] == am[k] or any(rule_ok(r_, dlm[k], drk[k]) for r_ in rules[k])):
        k += 1
    assert k >= kx, (k, kx)
    return kx, k


class CodeScan:
    """the comment loophole's region state, fed the answer's text token by token (Fable mission 25's line scan,
    COMMENT_LOOPHOLE.md, run incrementally): outside a ``` fence = prose; inside = python, tracked as code / '#'
    comment / string / docstring. in_comment(): a '#' comment of the fence's third line or later; comment_rows() adds
    the tool-prefix and token rules."""

    def __init__(self):
        self.code, self.line, self.cl = False, "", 0          # in a fence; the current line's text; line index there
        self.doc, self.dq, self.dn = False, "", 0              # in a docstring; its quote char; closing quotes in a row
        self.str_, self.esc = None, False                      # in a string: its quote char; a backslash pending
        self.com, self.ctext = False, ""                       # in a '#' comment; its text so far
        self.qn, self.qc = 0, ""                               # quotes in a row outside strings (open: 1 / 2 / 3)

    def copy(self):
        c = CodeScan.__new__(CodeScan)
        c.__dict__.update(self.__dict__)
        return c

    def _quotes_done(self):                                    # a run of qn quotes outside strings has ended
        if self.qn >= 3:
            self.doc, self.dq, self.dn = True, self.qc, 0
        elif self.qn == 1:
            self.str_ = self.qc
        self.qn, self.qc = 0, ""                               # qn == 2: an empty string

    def feed(self, text):
        for ch in text:
            self._ch(ch)

    def _ch(self, ch):
        if ch == "\n":
            if self.qn:
                self._quotes_done()
            fence = self.line.strip().startswith("```")
            if not self.code:
                if fence:
                    self.__init__()
                    self.code = True
            elif fence and not self.doc:
                self.__init__()
            else:
                self.str_, self.esc, self.com, self.ctext, self.dn = None, False, False, "", 0
                self.cl += 1
            self.line = ""
            return
        self.line += ch
        if not self.code:
            return
        if self.doc:
            self.dn = self.dn + 1 if ch == self.dq else 0
            if self.dn == 3:
                self.doc, self.dq, self.dn = False, "", 0
            return
        if self.com:
            self.ctext += ch
            return
        if self.str_:
            if self.esc:
                self.esc = False
            elif ch == "\\":
                self.esc = True
            elif ch == self.str_:
                self.str_ = None
            return
        if self.qn and ch == self.qc and self.qn < 3:
            self.qn += 1
            if self.qn == 3:
                self._quotes_done()
            return
        if self.qn:
            self._quotes_done()
            if self.str_ or self.doc:                          # the char after the opening quote(s) is inside
                return self._ch_inside(ch)
        if ch == "#":
            self.com, self.ctext = True, "#"
        elif ch in "\"'":
            self.qn, self.qc = 1, ch

    def _ch_inside(self, ch):
        if self.doc:
            self.dn = 1 if ch == self.dq else 0
            return
        if ch == "\\":
            self.esc = True
        elif ch == self.str_:
            self.str_ = None

    def in_comment(self):
        """inside a '#' comment on the fence's third line or later (no string, docstring or quote run around)"""
        return self.code and self.com and self.cl >= 2 and not self.qn


def nontool(text):
    """a comment text that has left every tool prefix (spaces dropped, any case): it neither is one nor could become one"""
    t = text.replace(" ", "").lower()
    return not any(p.startswith(t) or t.startswith(p) for p in TOOL_PREFIXES)


def comment_rows(pre, dtexts, atexts):
    """the rows the comment rule may relax: before its draft the row is inside a comment (pre[j] = (in_comment,
    comment text so far)), the draft is a comment-safe token, and the comment stays a non-tool comment with the draft and
    with the row's own argmax"""
    return [j for j, (inc, ct) in enumerate(pre)
            if inc and comment_draft_ok(dtexts[j]) and nontool(ct + dtexts[j]) and nontool(ct + atexts[j])]


def comment_draft_ok(text):
    """a draft the comment rule may relax: it does not start a new line, and anything after a newline in it is blank"""
    if text.startswith("\n"):
        return False
    return "\n" not in text or not text.split("\n", 1)[1].strip()


def q1u_keep(ids, sure, k_gate=2):
    """Q1-UNION's keep mask for one layer of one check (Breakthrough's dm_q1u_masks.py, OEA's union step on Q1's rows):
    U = every rank of the unsure rows + ranks < k_gate of the sure rows; a sure row keeps a rank >= k_gate iff its expert
    is in U. ids [rows, K] ints (rank-sorted), sure [rows] bool -> keep [rows, K] bool. Numpy reference for the wrapper."""
    ids, sure = np.asarray(ids), np.asarray(sure, bool)
    rows, K = ids.shape
    contrib = (~sure)[:, None] | (np.arange(K) < k_gate)[None, :]
    u = set(ids[contrib].tolist())
    keep = contrib.copy()
    for j in np.nonzero(sure)[0]:
        keep[j, k_gate:] = [e in u for e in ids[j, k_gate:].tolist()]
    return keep


def gate_keep_mx(indices, sure, union, k_gate=2):
    """the gate's keep mask in the graph (the streamed switch's wrapper uses this): indices [.., R, K] rank-sorted ids of
    one layer, sure [R] bool (gated rows). Q1: a sure row keeps ranks < k_gate. Q1-UNION: + its ranks >= k_gate whose
    expert is in the check's union (q1u_keep's rule). -> keep [R, K] bool (mlx)."""
    import mlx.core as mx
    K = indices.shape[-1]
    keep = mx.logical_or(mx.logical_not(sure)[:, None], (mx.arange(K) < k_gate)[None, :])
    if union:
        f = indices.reshape(-1)
        member = mx.any(mx.logical_and(f[:, None] == f[None, :], keep.reshape(-1)[None, :]), axis=1)
        keep = mx.logical_or(keep, member.reshape(keep.shape))
    return keep


def kept(drafts, am, dlm, delta):
    """(k exact, k under the rule). Exact: leading drafts equal to their row's argmax. R1(delta): leading drafts whose
    logit is within delta of their row's max logit; dlm[j] = logit(d_j) - max logit on row j (<= 0). R1 keeps every
    draft the exact rule keeps (an argmax draft has dlm 0), so k >= k exact."""
    kx = 0
    while kx < R - 1 and drafts[kx] == am[kx]:
        kx += 1
    if delta is None:
        return kx, kx
    k = 0
    while k < R - 1 and dlm[k] >= -delta:
        k += 1
    assert k >= kx, (k, kx)
    return kx, k


def committed(drafts, am, k, eos, room):
    """the check's new tokens: the k kept drafts, then row k's argmax (the bonus), cut after an EOS and to the room left"""
    new = list(drafts[:k]) + [am[k]]
    for j, x in enumerate(new):
        if x in eos:
            new = new[:j + 1]
            break
    return new[:room]


def union_counts(ids, r4):
    """ids [layers, rows, K] (ints): per layer the unique ids over all ranks (the base pool's experts) and over the
    ranks < r4 (the 4-bit sidecar's)"""
    a = np.sort(ids.reshape(ids.shape[0], -1), axis=1)
    ub = 1 + (np.diff(a, axis=1) != 0).sum(axis=1)
    if not r4:
        return ub, np.zeros_like(ub)
    s = np.sort(ids[:, :, :r4].reshape(ids.shape[0], -1), axis=1)
    return ub, 1 + (np.diff(s, axis=1) != 0).sum(axis=1)


def union_bytes(ids, r4, b_base, b_side):
    """the expert bytes one check reads from RAM: per layer, unique base experts x the base pool's bytes per expert,
    plus unique sidecar experts x the sidecar's bytes per expert. -> (bytes, base experts, sidecar experts)"""
    ub, us = union_counts(ids, r4)
    return int((ub * np.asarray(b_base)).sum() + (us * np.asarray(b_side)).sum()), int(ub.sum()), int(us.sum())


def kld_top(lp_e, lp_g):
    """KLD(E || G) with G known on its top-m ids only: exact on those ids (E's log-probs lp_e there, G's lp_g), plus
    E's remaining mass against G's remaining mass as one lumped term"""
    lp_e, lp_g = np.asarray(lp_e, np.float64), np.asarray(lp_g, np.float64)
    pe, pg = np.exp(lp_e), np.exp(lp_g)
    re_, rg = max(1.0 - pe.sum(), 1e-12), max(1.0 - pg.sum(), 1e-12)
    return float((pe * (lp_e - lp_g)).sum() + re_ * math.log(re_ / rg))


def token_class(text, code):
    """the must-fix classes (information): number, code (any non-space token in a code or agent answer), word, space"""
    t = text.strip()
    if not t:
        return "space"
    if any(ch.isdigit() for ch in t):
        return "number"
    if code:
        return "code"
    return "word" if any(ch.isalpha() for ch in t) else "punct"


def quarters(n):
    """the quarter (0-3) of each of n positions"""
    return [min(3, 4 * i // n) for i in range(n)]
