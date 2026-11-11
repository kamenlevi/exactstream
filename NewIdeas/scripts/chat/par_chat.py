"""PAR-CHAT kill test (NewIdeas Sun 8 Nov; pre-registered in IDEAS.md 15:38, commit 1cac993, before this ran).
How much of the 125B's own chat answers could run as parallel streams (Skeleton-of-Thought style), priced on CHS5's
measured chat check? Data: NAX6-N AR reference answers (EvalSuite ref2tnax IFEval 100, ref2tnaxn.r6 IFBench 150).
Segmentation: STRICT = runs of >= 3 sibling list items (same indent) or >= 3 sibling sections (same header level, or
bold-title lines); LOOSE = STRICT + runs of >= 3 plain paragraphs in what STRICT leaves sequential. A branch = one
item / section / paragraph with everything nested in it. Characters convert to tokens with each answer's own ratio
(len(completion_ids) / len(text)).
Time model (tok/s per stream with k streams active, scaled so k = 1 is CHS5 zsgcb's measured 53.8 whole decode):
cycle(k) = 25.5 + (1.96 x 2.96 + 0.161 x 13) k + 8.0 + 1.2 (k - 1) ms, kept 2.35 per stream per cycle.
A group runs on <= 4 streams with dynamic assignment (a free stream takes the next branch; active streams progress at
the k-stream rate). Overheads: a skeleton line of 8 tokens per branch at the 1-stream rate, and one check (40 ms) per
branch-start event (simultaneous starts share one).
Lines: KILL if the pooled speedup < 1.25 under LOOSE. PASS-to-quality-test if >= 1.5 under STRICT. Else BETWEEN.
usage: /usr/bin/python3 -I par_chat.py"""
import json, os, re

R = os.path.expanduser("~/EvalSuite/results")
SRC = [("ifeval", f"{R}/ref2tnax/decode.jsonl"), ("ifbench", f"{R}/ref2tnaxn.r6/decode.jsonl")]
B, SKEL_TOK, START_S = 4, 8, 0.040


def cycle_ms(k):
    return 25.5 + (1.96 * 2.96 + 0.161 * 13) * k + 8.0 + 1.2 * (k - 1)


SCALE = 53.8 / (2.35 / cycle_ms(1) * 1e3)
RATE = {k: 2.35 / cycle_ms(k) * 1e3 * SCALE for k in range(1, B + 1)}     # tok/s per stream with k streams

ITEM = re.compile(r"^(\s*)([-*•+]|\d+[.)])\s+\S")
HEAD = re.compile(r"^(#{1,6})\s+\S")
BOLD = re.compile(r"^\s*\*\*[^*]+\*\*\s*:?\s*$")


def line_kind(s):
    if not s.strip():
        return ("blank", 0)
    m = HEAD.match(s)
    if m:
        return ("head", len(m.group(1)))
    if BOLD.match(s):
        return ("head", 7)                                  # a bold title line: one level below any '#'
    m = ITEM.match(s)
    if m:
        return ("item", len(m.group(1).expandtabs(4)))
    return ("text", len(s) - len(s.lstrip()))


def groups_strict(lines):
    """[(start, end, [branch (start, end) line ranges])] for >= 3 sibling sections, else >= 3 sibling list items."""
    kinds = [line_kind(s) for s in lines]
    out = []
    heads = [(i, lv) for i, (k, lv) in enumerate(kinds) if k == "head"]
    if heads:
        lv0 = min(lv for _, lv in heads)
        top = [i for i, lv in heads if lv == lv0]
        if len(top) >= 3:
            br = [(a, b) for a, b in zip(top, top[1:] + [len(lines)])]
            return [(top[0], len(lines), br)]
    i, n = 0, len(lines)
    while i < n:
        k, ind = kinds[i]
        if k != "item":
            i += 1; continue
        starts, j = [i], i + 1
        while j < n:                                        # extend the run: same-indent items; deeper lines nest
            kj, ij = kinds[j]
            if kj == "item" and ij == ind:
                starts.append(j)
            elif kj == "item" and ij > ind:
                pass
            elif kj == "text" and ij > ind:
                pass
            elif kj == "text" and kinds[j - 1][0] != "blank":
                pass                                        # a wrapped continuation line
            elif kj == "blank":
                nxt = next((kinds[t] for t in range(j + 1, n) if kinds[t][0] != "blank"), None)
                if not (nxt and ((nxt[0] == "item" and nxt[1] >= ind) or (nxt[0] == "text" and nxt[1] > ind))):
                    break
            else:
                break
            j += 1
        if len(starts) >= 3:
            out.append((starts[0], j, [(a, b) for a, b in zip(starts, starts[1:] + [j])]))
        i = j
    return out


def groups_loose(lines, strict):
    """STRICT + runs of >= 3 blank-line-separated plain paragraphs outside the STRICT groups."""
    used = set()
    for a, b, _ in strict:
        used.update(range(a, b))
    paras, cur = [], None
    for i, s in enumerate(lines):
        if i in used or not s.strip():
            if cur is not None:
                paras.append(cur); cur = None
            if i in used and paras and paras[-1] != "BREAK":
                paras.append("BREAK")
            continue
        cur = (cur[0], i + 1) if cur else (i, i + 1)
    if cur is not None:
        paras.append(cur)
    out, run = list(strict), []
    for p in paras + ["BREAK"]:
        if p == "BREAK":
            if len(run) >= 3:
                out.append((run[0][0], run[-1][1], run))
            run = []
        else:
            run.append(p)
    return out


def par_time(branch_tok):
    """Dynamic assignment of branches to <= B streams; returns (seconds, start events)."""
    q, act, t, starts = list(branch_tok), [], 0.0, 0
    while q or act:
        if q and len(act) < B:
            while q and len(act) < B:
                act.append(q.pop(0))
            starts += 1
        r = RATE[len(act)]
        m = min(act)
        t += m / r
        act = [x - m for x in act if x - m > 1e-9]
    return t, starts


def price(text, ntok, mode):
    lines = text.split("\n")
    lens = [len(s) + 1 for s in lines]
    tpc = ntok / max(1, sum(lens))                          # this answer's tokens per character
    g = groups_strict(lines)
    if mode == "loose":
        g = groups_loose(lines, g)
    par_chars, t = 0, 0.0
    for a, b, br in g:
        bt = [sum(lens[x:y]) * tpc for x, y in br]
        par_chars += sum(lens[a:b])
        tg, st = par_time(bt)
        t += tg + st * START_S + SKEL_TOK * len(br) / RATE[1]
    t += (sum(lens) - par_chars) * tpc / RATE[1]
    return t, par_chars / max(1, sum(lens)), len(g)


print("PAR-CHAT: per-stream tok/s by active streams " + ", ".join(f"{k}: {RATE[k]:.1f}" for k in RATE) +
      f"; aggregate at {B}: {B * RATE[B]:.1f} ({B * RATE[B] / RATE[1]:.2f}x)")
tot = {}
for st, path in SRC:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("set") == st and r.get("text")]
    for mode in ("strict", "loose"):
        seq = par = 0.0; fr = []; ng = 0; anyg = 0
        for r in rows:
            n = len(r["completion_ids"])
            tp, f, k = price(r["text"], n, mode)
            seq += n / RATE[1]; par += tp; fr.append(f); ng += k; anyg += k > 0
        tot.setdefault(mode, [0.0, 0.0])
        tot[mode][0] += seq; tot[mode][1] += par
        print(f"{st:8s} {mode:6s}: n {len(rows):3d}, answers with a group {anyg:3d}, groups {ng:3d}, mean parallel "
              f"char share {100 * sum(fr) / len(fr):5.1f} %, decode {seq:7.1f} s -> {par:7.1f} s = x{seq / par:.3f}")
for mode, (s, p) in tot.items():
    print(f"POOLED {mode:6s}: x{s / p:.3f}")
sl, ss = tot["loose"][0] / tot["loose"][1], tot["strict"][0] / tot["strict"][1]
v = "KILL" if sl < 1.25 else ("PASS-to-quality-test" if ss >= 1.5 else "BETWEEN")
print(f"VERDICT: {v} (LOOSE x{sl:.3f} vs the 1.25 kill line; STRICT x{ss:.3f} vs the 1.5 pass line)")
