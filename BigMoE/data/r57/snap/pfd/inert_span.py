> (a quote from the project owner, kept private)
acceptance on rows whose draft sits in text the tests never execute, for r57_screen_rlx.py's verify step.
Copy this file to ~/BigMoE/scripts/; patch: apply_inert.py beside it. Design and numbers: ~/FableGod/goal/LOSSY350_GOAL.md.

The tagger is the engine's own (relax_core.CodeScan, the one rule C uses), wrapped by InertScan, which adds what CodeScan
does not track: the fence's language (only a fence that says python / py / python3 is ever open: CodeScan reads every fence
as python, and in a C fence '#include' would be a comment), bracket depth, whether an open triple-quoted string is a
statement of its own (a docstring, inert) or data, the doctest lines inside it, and the index of a comment line inside its
block of consecutive comment lines.
Checked on CPU (test_inert.py): on the 24 HumanEval+ and 100 MBPP answers of the walks, every token this tagger opens to
wholesale acceptance (3,231 + 5,647) lies inside a COMMENT token or a string statement of Python's own tokenizer / ast.
Known limit: a program that reads its own docstrings (__doc__, argparse description=__doc__, a doctest runner with the
guard lifted) makes docstring text functional; the tagger cannot see that.

A drafted row j may be kept although its draft differs from the 125B's argmax when, with the scan fed the answer's text up
to and excluding row j's draft (the committed text, then the block's drafts 0..j-1):
  comment rows (CodeScan.in_comment(): in a ``` fence, in a '#' comment, the fence's third line or later, no quote run):
    1. the comment neither is nor could become a tool comment, with the draft and with the argmax (relax_core.nontool);
    2. the line-end guard: no '\\n' in the argmax (when the 125B ends the comment line, the row stays exact);
    3. mode "whole": no '\\n' in the draft either (the span's end is never relaxed)  -> kept, no likelihood condition;
       mode "ruleC": relax_core.comment_draft_ok(draft) and R2 top-3 at 0.1 p1 (rule C as shipped);
       mode "guard": lines 1-2 of a block of consecutive comment lines (and trailing comments) as "whole", later lines on
       the 125B's top-8 (R2 top-8, any p): the long blocks are where the 125B thinks aloud.
  docstring rows (InertScan.in_doc(): inside a triple-quoted string that opened at the start of a statement, bracket depth
  0, no backslash continuation, no f prefix; not while its closing quotes are running):
    1. the docstring's quote character is neither in the draft nor in the argmax (the span's end is never relaxed);
    2. the row is not on a doctest line ('>>>' / '...'), before or after its draft (R57_INERT_DOCTEST=0 lifts this);
    -> kept, no likelihood condition (mode "whole").
Every other row is exact. The block is kept up to the first row that neither equals its argmax, nor passes the arm's own
R57_FAST rules (fast_ok), nor is kept by the above. Code tokens never deviate.

Before the check (pre_block): rows that are "whole" on the draft side count as sure in the running-confidence cut (inside
a span the cut does not apply), and a block longer than 16 rows keeps its drafts past the 15th only while they sit in such
rows (or are used copies).

Arms (R57_INERT): CD16 / CD32  comments + docstrings whole;  D16 / D32  docstrings whole, comments on rule C;
G16 / G32  docstrings whole, comments guarded;  C16  comments whole only. The number = rows per check. Unset = exact: r57
never calls this module.
Fri 6 Nov, after CD16 failed MBPP12 on quality (the drafter's free text ran away: BigMoE data/fday): the GUARDED arms.
T16  comments + docstrings, every row kept only when the 125B ranks the draft in its top 3 at p >= 0.1 p1 (rule C's form);
TC16 the same on comments only;  U16 top-8 at 0.01 p1 (looser, not the safe pick). No row is kept wholesale in these arms.
Every arm that keeps text wholesale (CD16, C16, D16's docstrings, G16) is NOT quality-safe on text the 125B writes fresh. R57_INERT_LOG=<path>: one JSON line per relaxed accept (region, mode, texts, rank, dlm).
COPYRUN (R57_COPYRUN=64, exact, needs R57_COPY and R57_ROWS=64): copyrun_extend() below.
Replay twin and tests: precheck_inert.py, test_inert.py beside this file."""
import json
import math

RULE_C = ("R2", 3, 0.1)          # rule C as shipped: the draft's rank < 3 and p(d) >= 0.1 p1
RULE_TOP8 = ("R2", 8, 1e-12)     # the 125B's top-8, any p
# Fri 6 Nov (after CD16's MBPP12 quality KILL: wholesale text ran away). The guarded forms: a row inside a span is kept
# only if the 125B itself ranks the draft near its top. The 4th field is the row's confidence FLOOR in the running-
# confidence cut before the check (the check still decides the row): a block is cut later inside a span, but not never.
# 0.8 = at most ~5 low-confidence guarded rows per block; the replay's tok/s is flat between 0.8 and 1.0 and the floor
# bounds the rows wasted if the engine refuses more than the replay expects (kwall_replay.py lift, guard_lift.out).
RULE_T3 = ("R2", 3, 0.1, 0.8)      # T16: rule C's form on every row of a span
RULE_T8 = ("R2", 8, 0.01, 0.8)     # U16: the 125B's top-8 at p >= 0.01 p1
TYP = {"typ3": RULE_T3, "typ8": RULE_T8}
ARMS = {
    "CD16": dict(doc="whole", com="whole", rows=16),
    "CD32": dict(doc="whole", com="whole", rows=32),
    "D16": dict(doc="whole", com="ruleC", rows=16),
    "D32": dict(doc="whole", com="ruleC", rows=32),
    "G16": dict(doc="whole", com="guard", rows=16),
    "G32": dict(doc="whole", com="guard", rows=32),
    "C16": dict(doc=None, com="whole", rows=16),
    "T16": dict(doc="typ3", com="typ3", rows=16),      # comments + docstrings, top-3 at 0.1 p1 on every row
    "TC16": dict(doc=None, com="typ3", rows=16),       # comments only
    "U16": dict(doc="typ8", com="typ8", rows=16),      # comments + docstrings, top-8 at 0.01 p1 (looser; not the safe pick)
}


def parse_arm(spec, doctest="1"):
    """R57_INERT -> the arm's dict (a copy, + name, + the doctest guard), or None when unset / 0"""
    if not spec or spec == "0":
        return None
    if spec not in ARMS:
        raise ValueError(f"R57_INERT: {spec!r} (use one of {', '.join(ARMS)})")
    return dict(ARMS[spec], name=spec, doctest=doctest not in ("0", ""))


def rule_ok(rule, dlm, drk):
    """relax_core.rule_ok, restated: dlm = logit(d) - max logit (<= 0), drk = the draft's rank (0 = the argmax)"""
    if rule[0] == "R1":
        return dlm >= -rule[1]
    if rule[0] == "R2":
        return drk < rule[1] and dlm >= math.log(rule[2])
    raise ValueError(rule)


class InertScan:
    """relax_core.CodeScan fed the answer's text, plus: depth (open brackets in code), inert (the open docstring is a
    statement of its own), run (full-line comment lines directly above the current line), bs (the previous code line
    ended with a backslash)."""

    def __init__(self, rxc):
        self.rxc, self.s = rxc, rxc.CodeScan()
        self.depth, self.inert, self.run, self.bs, self.py = 0, False, 0, False, False

    def copy(self):
        c = InertScan.__new__(InertScan)
        c.rxc, c.s = self.rxc, self.s.copy()
        c.depth, c.inert, c.run, c.bs, c.py = self.depth, self.inert, self.run, self.bs, self.py
        return c

    def feed(self, text):
        for ch in text:
            self._ch(ch)

    def _ch(self, ch):
        s = self.s
        code0, doc0 = s.code, s.doc
        plain = s.code and not s.doc and not s.str_ and not s.com and not s.qn   # a code character (not text of any kind)
        if ch == "\n":
            line0 = s.line
            full = s.code and s.com and not s.doc and line0.lstrip().startswith("#")   # the line that ends is a comment line
            bs = plain and line0.rstrip().endswith("\\")
            s._ch(ch)
            if s.code != code0 or not s.code:            # a fence opened or closed: nothing carries over
                self.depth, self.inert, self.run, self.bs = 0, False, 0, False
                # only a fence that says it is python: CodeScan reads every fence as python, and in a C fence
                # '#include' would be a comment
                self.py = bool(s.code) and line0.strip()[3:].strip().lower() in ("python", "py", "python3")
                return
            self.run = self.run + 1 if full else 0
            self.bs = bs
            return
        if plain:
            if ch in "([{":
                self.depth += 1
            elif ch in ")]}":
                self.depth = max(0, self.depth - 1)
        s._ch(ch)
        if s.doc and not doc0:                           # the third opening quote: s.line ends with the three quotes
            pre = s.line[:-3].strip().lower()
            self.inert = self.depth == 0 and not self.bs and pre in ("", "r", "u")
        elif doc0 and not s.doc:
            self.inert = False

    # ---- the states INERT reads
    def in_comment(self):
        """CodeScan.in_comment() inside a python fence"""
        return bool(self.py and self.s.in_comment())

    def in_doc(self):
        """inside a docstring that is a statement of its own, in a python fence; False while its closing quotes run"""
        s = self.s
        return bool(self.py and s.code and s.doc and self.inert and s.dn == 0)

    def doctest(self):
        """the current line of the docstring is a doctest line"""
        t = self.s.line.lstrip()
        return t.startswith(">>>") or t.startswith("...")

    def cblock(self):
        """the current comment line's index in its block of consecutive comment lines (1 = first); 0 = a trailing comment"""
        return self.run + 1 if self.s.line.lstrip().startswith("#") else 0


def classify(rxc, scan, dtexts, atexts, arm):
    """per row j: (kind, mode, ctext) with the scan state BEFORE row j's draft. kind: 'comment' / 'docstring' / None;
    mode: 'whole', a rule tuple, or None (exact). atexts None = the draft side only (before the check: the argmax guards
    are not applied). scan is not changed: a copy is fed the drafts, as r57's comment loop does."""
    s_, out = scan.copy(), []
    for j, d in enumerate(dtexts):
        a = None if atexts is None else atexts[j]
        kind = mode = None
        ct = ""
        if s_.in_comment():
            kind, ct = "comment", s_.s.ctext
            cm = arm["com"]
            if cm == "guard":
                cm = "whole" if s_.cblock() <= 2 else "top8"
            if cm is not None and rxc.nontool(ct + d) and (a is None or (rxc.nontool(ct + a) and "\n" not in a)):
                if cm == "whole":
                    if "\n" not in d:
                        mode = "whole"
                elif cm in TYP:
                    if "\n" not in d:
                        mode = TYP[cm]
                elif rxc.comment_draft_ok(d):
                    mode = RULE_C if cm == "ruleC" else RULE_TOP8
        elif s_.in_doc():
            kind = "docstring"
            q = s_.s.dq
            if arm["doc"] in ("whole", "typ3", "typ8") and q not in d and (a is None or q not in a):
                dt = False
                if arm.get("doctest", True):
                    dt = s_.doctest()
                    if not dt:
                        c_ = s_.copy(); c_.feed(d)
                        dt = c_.doctest()
                if not dt:
                    mode = "whole" if arm["doc"] == "whole" else TYP[arm["doc"]]
        out.append((kind, mode, ct))
        s_.feed(d)
    return out


def pre_block(rxc, scan, dr, cfl, dtexts, arm, used=None):
    """before the check. -> (n, cfl2, whole): whole[j] = row j is 'whole' on the draft side; cfl2 = the confidences with
    those rows sure (1.0), so the running-confidence cut passes through a span; n = the drafts a block keeps: all of them
    at 16 rows; past the 15th only while the rows are whole (or used copies, used[j]) when the arm runs longer blocks."""
    cls = classify(rxc, scan, dtexts, None, arm)
    whole = [c[1] == "whole" for c in cls]
    floor = [1.0 if w else (c[1][3] if isinstance(c[1], tuple) and len(c[1]) > 3 else 0.0) for w, c in zip(whole, cls)]
    cfl2 = [max(float(cfl[j]), floor[j]) for j in range(len(cfl))]      # whole rows sure; guarded rows floored
    n = len(dr)
    if arm["rows"] > 16 and n > 15:                      # a 16-row arm leaves the block as the engine built it
        n = 15
        while n < len(dr) and (whole[n] or (used is not None and n < len(used) and used[n])):
            n += 1
    return n, cfl2, whole


def inert_k(dr, am, dlm, drk, cls, fast_ok=None):
    """(k, relaxed): the leading kept drafts and [(row, kind, 'whole' | 'rule')] for the rows kept only by INERT. Row j is
    kept when dr[j] == am[j], or fast_ok(j) (the arm's own R57_FAST rules), or its mode is 'whole', or its mode is a rule
    that rule_ok accepts. With every mode None and no fast_ok this is the exact rule."""
    k, relaxed = 0, []
    while k < len(dr):
        kind, mode, _ = cls[k]
        if dr[k] == am[k] or (fast_ok is not None and fast_ok(k)):
            pass
        elif mode == "whole":
            relaxed.append((k, kind, "whole"))
        elif mode is not None and rule_ok(mode, dlm[k], drk[k]):
            relaxed.append((k, kind, "rule"))
        else:
            break
        k += 1
    return k, relaxed


def copyrun_extend(cg, d0, dr, cu, ci, cfl, rows):
    """COPYRUN (exact): a block whose drafts 15..30 are all used copies sits in a long verbatim run; let the copy gate
    go on past the drafter's 31 picks, up to rows - 1 drafts, for as long as it keeps proposing allowed copies. The check
    still decides every token. -> (d0, dr, cu, ci, cfl) extended; the positions past the drafter's picks carry the copy
    token as 'the drafter's pick' too, so CopyGate.learn labels no duel there and no change is counted."""
    n0 = len(dr)
    if rows <= 32 or n0 < 31 or not all(cu[15:31]):
        return d0, dr, cu, ci, cfl
    ext = rows - 1 - n0
    out, used, info = cg.propose(list(d0) + [int(dr[-1])] * ext, len(cg.ids) - 1)   # deterministic: the first n0 repeat
    assert out[:n0] == list(dr), "copyrun_extend: the gate's second pass changed the block"
    m = n0
    while m < len(out) and used[m]:
        m += 1
    if m == n0:
        return d0, dr, cu, ci, cfl
    return (list(d0) + out[n0:m], out[:m], list(cu) + [True] * (m - n0), list(ci) + info[n0:m],
            list(cfl) + [1.0] * (m - n0))


class Log:
    """R57_INERT_LOG: one JSON line per relaxed accept"""

    def __init__(self, path):
        self.f = open(path, "a") if path else None

    def write(self, **kw):
        if self.f:
            self.f.write(json.dumps(kw) + "\n")
            self.f.flush()
