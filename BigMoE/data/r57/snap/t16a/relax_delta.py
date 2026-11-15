"""R57_RELAX_DELTA (MoonShots, Wed 2026-11-04; the orchestrator's FAST build 17:5x): relaxed acceptance at any delta for
r57_screen.py's verify step, for BigMoE to integrate (copy this file to ~/BigMoE/scripts/; patch: apply.py beside it).
A draft d on row j is kept while it equals the row's argmax, or passes the arm's own relaxed rules (R57_FAST, EvalSuite's
relax_core), or (when row j may relax) log p(d) >= log p1 - delta, i.e. dlm[j] = logit(d) - max logit >= -delta: EvalSuite's
R1 at any delta (RELAX_350 on code: 0.5 nat = kept 10.5 -> 11.7 per check at 1.19 % deviated tokens; 1.39 = 12.9 at 2.4 %).
delta 0 / unset = exact: r57 never calls this module. Which rows may relax (R57_RELAX_GATE): every row (unset), the rows
whose drafter cf >= thr ('cf:<thr>'), the rows whose last 8 input tokens occur in the prompt ('copy'), or either
('cf:<thr>+copy'), by relax_core.gate_rows on the drafted block before the cut, as the R57_FAST expert masks gate it.
R57_RELAX_LOG=<path>: one JSON line per relaxed accept."""
import json
import math

R = 16                                                            # relax_core's check rows (15 drafts + the bonus row)


def parse_gate(spec):
    """R57_RELAX_GATE -> (cf threshold or None, copy gate on?); '' = every row may relax"""
    thr, copy = None, False
    for part in filter(None, (spec or "").split("+")):
        if part.startswith("cf:"):
            thr = float(part[3:])
        elif part == "copy":
            copy = True
        else:
            raise ValueError(f"R57_RELAX_GATE: {spec!r} (use cf:<thr>, copy or cf:<thr>+copy)")
    return thr, copy


def gate_flags(rxc, thr, copy, cf, tail, drafts, grams):
    """per drafted row: may it relax? None = no gate set (every row may). relax_core.gate_rows needs a full 15-draft
    block; a longer block (32-row checks, DRAFT17) gates its first 15 rows and never relaxes the rest"""
    if thr is None and not copy:
        return None
    n = len(drafts)
    d15 = (list(drafts) + [-1] * (R - 1))[:R - 1]
    c15 = (list(cf) + [0.0] * (R - 1))[:R - 1]
    g = rxc.gate_rows(c15, tail, d15, thr, grams if copy else None)
    return [j < R - 1 and g[j] != 0 for j in range(n)]


def relaxed_k(dr, am, dlm, delta, may=None, fast_ok=None):
    """(k, relaxed rows): the leading kept drafts, and the rows kept only by R1(delta). dr / am: the drafts and their
    rows' argmaxes; dlm[j] = logit(dr[j]) - max logit on row j (<= 0; -inf if unknown); may[j]: row j may relax (None =
    every row); fast_ok(j): the arm's own relaxed rules keep row j (R57_FAST), or None. delta 0 keeps exactly what the
    exact rule keeps (an argmax draft has dlm 0 and is kept as exact anyway)."""
    k, relaxed = 0, []
    while k < len(dr):
        if dr[k] == am[k] or (fast_ok is not None and fast_ok(k)):
            pass
        elif (may is None or may[k]) and dlm[k] >= -delta:
            relaxed.append(k)
        else:
            break
        k += 1
    return k, relaxed


def row_dlm(logp_d, logp_1):
    """dlm from stored log-probabilities (replays): log p(d) - log p1; -inf when d is not in the stored top-k"""
    return -math.inf if logp_d is None else logp_d - logp_1


class Log:
    """R57_RELAX_LOG: one JSON line per relaxed accept (prompt, check, row, draft, argmax, dlm, rank, cf, gate)"""

    def __init__(self, path):
        self.f = open(path, "a") if path else None

    def write(self, **kw):
        if self.f:
            self.f.write(json.dumps(kw) + "\n")
            self.f.flush()
