"""SPANRELAX (MoonShots, Wed 2026-11-04; the orchestrator 21:0x: Fable FULL's bet, ~/FableGod/full/FULL.md section 5,
pre-registered there): relax only inside '#' comments. CPU replay, one core, data on disk, no GPU.
Positions = RELAX_350's (FableGod relax_curves.py; its loader and kept_from_profile executed read-only): every drafted
position DFlash2 L1-20 reached in the HPD-0 code walk (~/TrainLab/data/d02b_walk_hpd0_code.npz, 24 HumanEval+ windows),
joined with the 125B's top-8 at the verifying row (decision_map/out/baseline.json). That top-8 exists for the 12 map
windows only, so the relaxed decisions (and RELAX_350's code table) are on those 12; the region scan runs on all 24.
Regions: Mission 25's scan (comment_loophole.py's regions(), executed read-only): the label of the token the row
predicts. A span position = a '#'-comment label, not a tool comment (relax_core.TOOL_PREFIXES: type:, noqa, pragma, fmt:,
pylint, mypy, coding, shebang), not on the stream's first two lines, and a draft that stays inside the comment
(relax_core.comment_draft_ok: no newline unless only blanks follow it).
Rules (FULL 2.1 / 5): A = inside a span accept the draft if it is in the 125B's top-3 and p(d) >= 0.1 p1, elsewhere exact;
B = A + R1 0.25 nat outside the spans. kept(16) = 10.54 x profile kept(rule) / profile kept(EXACT) (RELAX_350's engine
ratio method). Deviation = an accepted draft that is not the 125B's argmax (first order, teacher forced), by region as %
of that region's positions; code-token deviation = deviations outside the spans / positions outside the spans (also
shown: code-region only, and as % of all positions). Disagreement = a relaxed accept where Mission 25's character scan
and the engine's token-level view (EvalSuite relax_core.CodeScan fed the walk's tokens + comment_rows' rule, read-only)
differ on 'relaxable comment'. QOM-gated rows: at cf 0 every draft row is gated; shares at cf 0.5 / 0.9 from the drafter's cf.
Lines (FULL section 5): PASS = rule A kept(16) >= 11.6 AND 0 deviations outside comments; rule B kept(16) >= 11.8 AND
code-token deviation <= 0.4 %. KILL = either rule's kept < 11.0, or any code-token deviation > 0.6 %, or > 2 % of
relaxed accepts where the region scan and the tokenizer disagree. FULL's prediction: A 12.1, B 12.3, code-token deviation
0.2 %, comment deviation 1.6-1.8 %.
Checks: RELAX_350's code table (EXACT profile kept 7.78, R1(0.5) 8.61 at 16 rows); Mission 25's prose-only R2 top-3 at
0.25 p1 (its walk-acceptance convention) = x1.157 / 1.68 % / 55 comment deviations.
usage: [BigMoE venv] python span_relax_check.py -> stdout + data/span_relax_check.txt"""
import importlib.util, math, os, time
from collections import defaultdict

H = os.path.expanduser("~")
TD = f"{H}/TrainLab/data"
RC, CL = f"{H}/FableGod/relax_curves.py", f"{H}/FableGod/comment_loophole.py"
BASE16 = 10.54
REGS = ("code", "comment", "docstring", "doctest", "string")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


t0 = time.time()
src = open(RC).read()
env = {"__name__": "relax_curves_prefix"}
exec(compile(src[:src.index("\nfor dom in DOMS:\n    P = [q for q in POS")], RC, "exec"), env)   # loaders + POS + kept_from_profile
npz, dist_tables, kfp = env["npz"], env["dist_tables"], env["kept_from_profile"]
RAW_CODE = [q for q in env["POS"] if q["dom"] == "code"]
src = open(CL).read()
envc = {"__name__": "comment_loophole_prefix"}
exec(compile(src[:src.index("\nSETS = [")], CL, "exec"), envc)                                   # regions(), dec()
regions, dec = envc["regions"], envc["dec"]
spec = importlib.util.spec_from_file_location("relax_core_ro", f"{H}/EvalSuite/evalsuite/relax_core.py")
RCORE = importlib.util.module_from_spec(spec); spec.loader.exec_module(RCORE)

W = npz(f"{TD}/d02b_walk_hpd0_code.npz")
D = dist_tables("code")
ALL, POS = [], []                                                    # ALL: every tested position (24 windows); POS: with the top-8
fences = 0
for i, nm in enumerate(W["names"]):
    ids = list(W[f"{i}_ids"][0])
    checks, (nc, _) = W[f"{i}_checks"]
    dcf, dtok = W[f"{i}_dcf"][0], W[f"{i}_dtok"][0]
    per = len(dtok) // max(nc, 1)
    reg = regions(ids)
    pieces = [dec(t) for t in ids]
    text = "".join(pieces)
    starts, o = [], 0
    for s_ in pieces:
        starts.append(o); o += len(s_)
    lineno = [text.count("\n", 0, st) for st in starts]
    tl = text.split("\n")
    fl_ = [i for i, L in enumerate(tl) if L.strip().startswith("```")]
    flin = [ln_ - max([f for f in fl_ if f < ln_], default=-1) - 1 for ln_ in lineno]   # line index inside the code block
    fenced = "```" in text
    fences += fenced
    sc = RCORE.CodeScan()
    if not fenced:
        sc.code = True                                               # a bare code stream: the scan starts in code
    eng = []                                                         # engine view before token t+1: (in_comment, comment text)
    for t in range(len(ids)):
        sc.feed(pieces[t])
        eng.append((sc.in_comment(), sc.ctext, sc.cl, sc.code))
    key = nm if nm in D else (nm + ".npz" if nm + ".npz" in D else None)
    row0, rows = D[key] if key else (0, [])
    for c in range(nc):
        k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
        for j in range(min(k + 1, m)):
            p = A + j
            if p + 1 >= len(ids):
                continue
            d = dtok[c * per + j]
            dtext = dec(d)
            ln = text[text.rfind("\n", 0, starts[p + 1]) + 1:]
            ln = ln[:ln.find("\n")] if "\n" in ln else ln
            ctext = ln[ln.index("#"):] if "#" in ln else ""
            m25s = (reg[p + 1] == "comment" and RCORE.nontool(ctext) and lineno[p + 1] >= 2 and RCORE.comment_draft_ok(dtext))
            m25 = (reg[p + 1] == "comment" and RCORE.nontool(ctext) and flin[p + 1] >= 2 and RCORE.comment_draft_ok(dtext))
            q = dict(win=nm, p=p, depth=j + 1, reg=reg[p + 1], wright=j < k, cf=dcf[c * per + j], m25=m25, m25s=m25s,
                     line=ln, dtext=dtext, ttext=pieces[p + 1], flin=flin[p + 1],
                     straddle=reg[p + 1] == "comment" and not RCORE.comment_draft_ok(dtext))
            ALL.append(q)
            r = p - row0
            if not key or not (0 <= r < len(rows)):
                continue
            tk, lp = rows[r]
            q.update(right=d == tk[0], lp0=lp[0], lpd=lp[tk.index(d)] if d in tk else None, lps=lp,
                     rank=sum(1 for v in lp if v > lp[tk.index(d)]) if d in tk else 99)
            inc, ct, ecl, ecode = eng[p]
            q.update(eng_in=bool(inc), eng_ct=ct, eng_cl=ecl, atext=dec(tk[0]))
            q["eng"] = bool(inc and RCORE.comment_draft_ok(dtext) and RCORE.nontool(ct + dtext) and RCORE.nontool(ct + dec(tk[0])))
            POS.append(q)
assert len(POS) == len(RAW_CODE), (len(POS), len(RAW_CODE))
assert all(a["right"] == b["right"] and a["depth"] == b["depth"] for a, b in zip(POS, RAW_CODE))
say(f"== SPANRELAX (MoonShots, {time.strftime('%a %Y-%m-%d %H:%M')}): HPD-0 code walk, {len(W['names'])} windows ({fences} with "
    f"``` fences), {len(ALL)} tested positions; the 125B's top-8 on {len(set(q['win'] for q in POS))} windows, {len(POS)} positions "
    f"(= RELAX_350's code positions)")
say("  regions (Mission 25's scan), all 24 windows | the 12 with the top-8: share of positions, walk-exact right %")
for rg in REGS:
    a = [q for q in ALL if q["reg"] == rg]; b = [q for q in POS if q["reg"] == rg]
    say(f"    {rg:9s}: {100 * len(a) / len(ALL):5.1f} % ({100 * sum(q['wright'] for q in a) / max(len(a), 1):5.1f}) | "
        f"{100 * len(b) / len(POS):5.1f} % ({100 * sum(q['right'] for q in b) / max(len(b), 1):5.1f})")
say(f"  span positions (comment label, non-tool, line >= 2, draft stays in the comment): {sum(q['m25'] for q in POS)} of "
    f"{sum(q['reg'] == 'comment' for q in POS)} comment positions; comment positions whose draft would leave the comment: "
    f"{sum(q['straddle'] for q in POS)}")


def acc_rule(q, rule):
    if q["right"]:
        return True
    if q["lpd"] is None:
        return False
    if rule == "exact":
        return False
    if rule == "r1_05":
        return q["lpd"] >= q["lp0"] - 0.5
    span = q["eng"] if rule.endswith("_eng") else (q["m25s"] if rule.endswith("_stream") else q["m25"])
    if span and q["rank"] < 3 and q["lpd"] >= q["lp0"] + math.log(0.1):
        return True
    if rule.startswith("B") and not span:
        return q["lpd"] >= q["lp0"] - 0.25
    return False


def run(rule, P=POS):
    acc = [acc_rule(q, rule) for q in P]
    prof = {}
    for dd in range(1, 16):
        S = [a for a, q in zip(acc, P) if q["depth"] == dd]
        prof[dd] = sum(S) / len(S) if S else float("nan")
    return acc, kfp(prof, 16), kfp(prof, 32)


acc0, e16, e32 = run("exact")
_, r16, r32 = run("r1_05")
say(f"  check RELAX_350: EXACT profile kept {e16:.2f} / {e32:.2f} at 16 / 32 rows [7.78 / 8.54]; R1(0.5) {r16:.2f} / {r32:.2f} [8.61 / 9.49]")
# Mission 25's own convention: right = the walk's acceptance (j < k), prose = comment + docstring, R2 top-3 at 0.25 p1
C = POS
m_acc = [q["wright"] or (q["reg"] in ("comment", "docstring") and q["lpd"] is not None and q["rank"] < 3
                         and q["lpd"] >= q["lp0"] + math.log(0.25)) for q in C]
m_dev = [a and not q["wright"] for a, q in zip(m_acc, C)]
mp = lambda acc: {dd: sum(a for a, q in zip(acc, C) if q["depth"] == dd) / max(1, sum(1 for q in C if q["depth"] == dd)) for dd in range(1, 16)}
say(f"  check Mission 25: prose-only R2 top-3 0.25: kept ratio {kfp(mp(m_acc), 16) / kfp(mp([q['wright'] for q in C]), 16):.3f} [1.157], "
    f"deviation {100 * sum(m_dev) / len(C):.2f} % [1.68], comment / docstring {sum(dv for dv, q in zip(m_dev, C) if q['reg'] == 'comment')} / "
    f"{sum(dv for dv, q in zip(m_dev, C) if q['reg'] == 'docstring')} [55 / 0]")
res = {}
for q in POS:
    q["m25_alt"] = q["m25s"]
for rule in ("A", "B", "A_eng", "B_eng", "A_stream"):
    acc, k16, k32 = run(rule)
    dev = [a and not q["right"] for a, q in zip(acc, POS)]
    rel = [a and not a0 for a, a0 in zip(acc, acc0)]                 # relaxed accepts (beyond exact)
    span = [(q["eng"] if rule.endswith("_eng") else (q["m25s"] if rule.endswith("_stream") else q["m25"])) for q in POS]
    out = [not s for s in span]
    dis = sum(1 for rl, q, sp in zip(rel, POS, span) if rl and (q["m25s"] if rule.endswith("_stream") else q["m25"]) != q["eng"])
    dev_out = sum(dv for dv, o in zip(dev, out) if o)
    res[rule] = dict(k16=BASE16 * k16 / e16, k32=BASE16 * k32 / e16, ratio=k16 / e16, dev=dev, rel=sum(rel),
                     dev_out_pct=100 * dev_out / max(1, sum(out)), dev_code_pct=100 * sum(dv for dv, q in zip(dev, POS) if q["reg"] == "code")
                     / max(1, sum(q["reg"] == "code" for q in POS)), dev_out_all=100 * dev_out / len(POS), dev_out=dev_out,
                     dis=dis, dis_pct=100 * dis / max(1, sum(rel)))
    r = res[rule]
    say(f"  rule {rule:5s}: kept(16) {r['k16']:.2f} (x{r['ratio']:.3f}; 32 rows {r['k32']:.2f}), relaxed accepts {r['rel']}, deviations "
        f"{sum(dev)} = {100 * sum(dev) / len(POS):.2f} % of positions; by region (% of the region's positions): " +
        ", ".join(f"{rg} {sum(dv for dv, q in zip(dev, POS) if q['reg'] == rg)} ({100 * sum(dv for dv, q in zip(dev, POS) if q['reg'] == rg) / max(1, sum(q['reg'] == rg for q in POS)):.2f})" for rg in REGS)
        + f"; outside the spans {dev_out} = {r['dev_out_pct']:.2f} % of outside positions (code region {r['dev_code_pct']:.2f} %, "
        f"{r['dev_out_all']:.2f} % of all positions); scan-vs-engine disagreement on {dis} relaxed accepts ({r['dis_pct']:.1f} %)")
for rule in ("A", "B"):
    rel_q = [q for q, a, a0 in zip(POS, run(rule)[0], acc0) if a and not a0]
    say(f"  rule {rule}: relaxed accepts on QOM-gated rows: cf 0 = all draft rows ({len(rel_q)}), cf >= 0.5: "
        f"{100 * sum(q['cf'] >= 0.5 for q in rel_q) / max(1, len(rel_q)):.0f} %, cf >= 0.9: {100 * sum(q['cf'] >= 0.9 for q in rel_q) / max(1, len(rel_q)):.0f} %")
say("  relaxed accepts where the scan and the engine's token view disagree (rules A / B / A_stream / A_eng / B_eng; each position "
    "once), with the cause:")
seen = set()
for rule in ("A", "B", "A_stream", "A_eng", "B_eng"):
    for q, a, a0 in zip(POS, run(rule)[0], acc0):
        m = q["m25s"] if rule == "A_stream" else q["m25"]
        if a and not a0 and m != q["eng"] and (q["win"], q["p"]) not in seen:
            seen.add((q["win"], q["p"]))
            cause = ("the target token opens the comment (scan: comment; engine: not yet)" if m and not q["eng_in"] and "#" in q["ttext"]
                     else "engine: comment on the code block's line < 2" if m and q["eng_in"] and q["eng_cl"] < 2
                     else "engine: tool-prefix rule (comment so far + draft / argmax)" if m and q["eng_in"]
                     else "scan: target starts a new line (engine: still in the comment)" if not m and q["eng_in"]
                     else "other")
            say(f"    [{rule}] {q['win'][:28]} p {q['p']} depth {q['depth']}: {cause}; line {q['line'][:70]!r}; target {q['ttext']!r}, "
                f"draft {q['dtext']!r}, argmax {q['atext']!r}; block line {q['flin']} (engine {q['eng_cl']}), engine comment so far {q['eng_ct'][:30]!r}")
A, B = res["A"], res["B"]
kill = min(A["k16"], B["k16"]) < 11.0 or max(A["dev_out_pct"], B["dev_out_pct"]) > 0.6 or max(A["dis_pct"], B["dis_pct"]) > 2.0
pas = A["k16"] >= 11.6 and A["dev_out"] == 0 and B["k16"] >= 11.8 and B["dev_out_pct"] <= 0.4
call = "KILL" if kill else ("PASS" if pas else "BETWEEN")
say(f"Lines (FULL section 5): rule A kept(16) {A['k16']:.2f} (>= 11.6) with {A['dev_out']} deviations outside comments (0); rule B "
    f"{B['k16']:.2f} (>= 11.8) at code-token deviation {B['dev_out_pct']:.2f} % (<= 0.4; KILL > 0.6); disagreement {max(A['dis_pct'], B['dis_pct']):.1f} % "
    f"of relaxed accepts (KILL > 2) -> {call} [prediction: A 12.1, B 12.3, 0.2 %]")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/span_relax_check.txt"), "a").write("\n".join(lines) + "\n")
