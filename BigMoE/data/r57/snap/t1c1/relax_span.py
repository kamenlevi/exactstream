"""R57_RELAX_SPAN (MoonShots, Wed 2026-11-04; the orchestrator 22:5x: build FULL's rule C for BigMoE's Fri 00:30 ticket):
relaxed acceptance on code-comment rows only, decided by the engine's causal tagger, for r57_screen_rlx.py's verify step.
Copy this file to ~/BigMoE/scripts/; patch: apply_span.py beside it. Rule C (~/FableGod/full/FULL.md, Round 2, R2.1), in
EvalSuite relax_core's terms (RXC = ~/BigMoE/scripts/relax_core.py, the vendored copy):
  a drafted row j may relax iff
  1. RXC.CodeScan, fed the answer's text up to and excluding row j's draft (the committed text, then the block's drafts
     0..j-1), says in_comment(): inside a ``` fence, in a '#' comment, the fence's third line or later, no quote run open;
  2. RXC.comment_draft_ok(draft_j): the draft does not start a new line, and anything after a newline in it is blank;
  3. RXC.nontool(ctext + draft_j) and RXC.nontool(ctext + argmax_j): the comment neither is nor could become a tool comment;
  4. the line-end guard: '\\n' not in argmax_j (when the 125B ends the comment line, the row stays exact);
  and then it is kept when rule_ok(("R2", 3, 0.1)): the draft's rank < 3 and logit(d) - max logit >= log 0.1.
Every other row is exact. The block is kept up to the first row that neither equals its argmax, nor passes the arm's own
R57_FAST rules (fast_ok), nor passes rule C. R57_RELAX_SPAN unset = exact: r57 never calls this module.
R57_RELAX_SPAN_RULE=R2:<k>:<x> | R1:<delta> changes step 5 (default R2:3:0.1). R57_RELAX_LOG=<path>: one JSON line per
relaxed accept, with the comment text so far, the draft and argmax texts, rank and dlm (every accept auditable by region).
Replay twin: ~/MoonShots/scripts/span_relax_c.py (M-41); tests: test_relax_span.py."""
import json
import math

DEFAULT_RULE = ("R2", 3, 0.1)


def parse_rule(spec):
    """R57_RELAX_SPAN_RULE -> relax_core rule tuple; '' = R2 top-3 at 0.1 p1 (rule C)"""
    if not spec:
        return DEFAULT_RULE
    p = spec.split(":")
    if p[0] == "R2" and len(p) == 3:
        return ("R2", int(p[1]), float(p[2]))
    if p[0] == "R1" and len(p) == 2:
        return ("R1", float(p[1]))
    raise ValueError(f"R57_RELAX_SPAN_RULE: {spec!r} (use R2:<k>:<x> or R1:<delta>)")


def rule_ok(rule, dlm, drk):
    """relax_core.rule_ok, restated so the module needs no import: dlm = logit(d) - max logit (<= 0), drk = the draft's rank"""
    if rule[0] == "R1":
        return dlm >= -rule[1]
    if rule[0] == "R2":
        return drk < rule[1] and dlm >= math.log(rule[2])
    raise ValueError(rule)


def span_rows(rxc, scan, dtexts, atexts):
    """(rows, pre): the rows of this block that may relax under conditions 1-4, and pre[j] = (in_comment, comment text so
    far) before row j's draft. scan = the run's CodeScan after the committed text (not changed here: a copy is fed the
    drafts, as r57's comment loop does). dtexts / atexts: each row's draft and argmax, decoded alone."""
    s_, pre = scan.copy(), []
    for x in dtexts:
        pre.append((s_.in_comment(), s_.ctext))
        s_.feed(x)
    rows = set(rxc.comment_rows(pre, dtexts, atexts))                 # conditions 1-3, as relax_core writes them
    rows = {j for j in rows if "\n" not in atexts[j]}                 # 4: the 125B's line end stays its own
    return rows, pre


def may_touch(scan, dtexts):
    """v2 (Fri 2026-11-06): can any row of this block be inside a comment? CodeScan opens a comment only on a '#' it is fed,
    so if the tagger is not in one at the block start and no draft holds a '#', span_rows would return no rows: skip it.
    Exactly equivalent (test_relax_v2.py checks it on every recorded check)."""
    return bool(scan.com) or any("#" in t for t in dtexts)


def span_k(dr, am, dlm, drk, rows, rule=DEFAULT_RULE, fast_ok=None):
    """(k, relaxed rows): the leading kept drafts and the rows kept only by rule C. Row j is kept when dr[j] == am[j], or
    fast_ok(j) (the arm's own R57_FAST rules), or j in rows and rule_ok(rule, dlm[j], drk[j]). With rows empty and no
    fast_ok this is the exact rule."""
    k, relaxed = 0, []
    while k < len(dr):
        if dr[k] == am[k] or (fast_ok is not None and fast_ok(k)):
            pass
        elif k in rows and rule_ok(rule, dlm[k], drk[k]):
            relaxed.append(k)
        else:
            break
        k += 1
    return k, relaxed


class Log:
    """R57_RELAX_LOG for span accepts: one JSON line each (prompt, check, row, region, comment text so far, draft / argmax
    ids and texts, rank, dlm, cf)"""

    def __init__(self, path):
        self.f = open(path, "a") if path else None

    def write(self, **kw):
        if self.f:
            self.f.write(json.dumps(kw) + "\n")
            self.f.flush()
