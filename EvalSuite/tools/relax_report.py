"""RELAX's tables (docs/PREREG_relax_2026-11-02.md). CPU only.
  .venv/bin/python tools/relax_report.py [--arms control,A,D,C,B] [--tag T] [--modes] [--score] [--selftest]
Per arm and set, from results/relax_<arm>[_T]/:
  - speed and bytes (G's logs): kept per check under the arm's rule vs the exact rule on the same checks, gated rows (cf /
    copy), expert-union GB per check (as run vs the same checks' exact union), SSD GB per check, tok/s and GB/s; paired
    against the control on the keys both ran;
  - quality (the REF2-T replay): deviated tokens (committed != E's argmax) by cause (rule / gate cf / gate copy /
    context), blunders (E's p of the committed token < 0.01) per 10K, drift (rows >= 512 tokens: quarter 4 vs quarter 1
    of KLD(E||G) and of the deviation rate, pooled per set), and every must-fix deviation (numbers; any code token).
--modes writes each mode's 210 decode rows (results/relax_mode_<m>/decode.jsonl) from its arms; --score runs
`evalsuite report --decode --ref ref2t` and tools/d4.py on them and prints each mode's line (PASS / BETWEEN / KILL).
--selftest builds synthetic logs in a temp dir, checks the tables against hand counts and exits 0 / 1."""
from __future__ import annotations
import argparse, glob, gzip, json, math, os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in (os.path.join(ROOT, "evalsuite"), os.path.join(ROOT, "pending")):   # relax_core: evalsuite/ once moved
    if os.path.exists(os.path.join(d, "relax_core.py")):
        sys.path.insert(0, d)
        break
from relax_core import LN_BLUNDER, quarters, token_class  # noqa: E402

RES = os.path.join(ROOT, "results")
D4_SETS = ("math", "ifeval", "humaneval")
MODES = {                                                # mode -> set -> the arm whose rows it takes (PLAN_relax sec 9)
    "ship": dict(math="B", humaneval="B", ifeval="E"),   # cloud 14's ship check: the code mode B + chat R1 (E)
    "A_E": dict(math="A", humaneval="A", ifeval="E"),    # the gate alone on code (B's control)
    "C_E": dict(math="C", humaneval="C", ifeval="E"),    # + R1 on code
    "B_F": dict(math="B", humaneval="B", ifeval="F"),    # chat + R2
    "B_G": dict(math="B", humaneval="B", ifeval="G"),    # chat + the gate
    "T3ALL": dict(math="T3ALL", humaneval="T3ALL", ifeval="T3ALL"),   # every rank at 3 bits (NewIdeas 02:42); its
                                                         # code-only fallback reads the math and HumanEval+ lines
    "ship8": dict(math="B", humaneval="B", ifeval="E8"), # Fri: ship with HC8 on chat too (E8 = E + HC8; HC8 is global)
}
BARS = {"blunder_per_10k": 1.0, "drift": 1.5}            # math + code; none in agent tool calls; q4 / q1 <= 1.5
RELAXED = ("B", "C", "E", "F", "G", "E8")                 # arms with a relaxed acceptance rule (R1 / R2 / the comment rule)
TASK_SLOWER = 1.05     # the per-task line (the orchestrator Wed 21:5x; PREREG_relax amendment): faster per token AND
#                        > 5 % slower per task (prefill + decode s) vs the control on shared keys = KILL for FAST
LONG = 512


def arm_dir(arm, tag):
    return os.path.join(RES, f"relax_{arm}" + (f"_{tag}" if tag else ""))


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_arm(arm, tag):
    d = arm_dir(arm, tag)
    p = os.path.join(d, "decode.jsonl")
    if not os.path.exists(p):
        return None
    rows = {r["key"]: r for r in jl(p)}
    out = {}
    for k, r in rows.items():
        f = k.replace("/", "__") + ".jsonl.gz"
        cp, rp = os.path.join(d, "checks", f), os.path.join(d, "replay", f)
        rep = jl(rp) if os.path.exists(rp) else None
        cs = jl(cp) if os.path.exists(cp) else []
        if rep is None and exact_rows(r):                # no gate, no HC8: G's rows are REF2-T's on the same prefix
            rep = [None] + self_replay(cs)
        out[k] = dict(row=r, checks=cs, head=rep[0] if rep else None, replay=rep[1:] if rep else None)
    chunks = jl(os.path.join(d, "chunks.jsonl")) if os.path.exists(os.path.join(d, "chunks.jsonl")) else []
    return dict(arm=arm, prompts=out, chunks=chunks)


def exact_rows(row):
    """a row whose every check ran REF2-T's own numerics (no gate, no HC8): only the acceptance rule differs"""
    c = row.get("cfg")
    return (isinstance(c, dict) and "thr" in c and c["thr"] is None and not c.get("copy") and not row.get("hc8")
            and row.get("tier", "4433333333") == "4433333333")


def self_replay(checks):
    """E from G's own logs where G = E: argmax rows, the committed token's log-prob (row max + logit gap; the bonus is
    the argmax), KLD 0"""
    out = []
    for c in checks:
        n, k = c["n"], c["k"]
        lp1 = [max(c["tl"][j]) for j in range(n)]
        out.append(dict(c=c["c"], pos=c["pos"], eam=c["am"][:n], kld=[0.0] * n,
                        lpc=[lp1[j] + (c["dlm"][j] if j < k else 0.0) for j in range(n)], lp1=lp1))
    return out


def tokens(p):
    """per committed token of one prompt: (pos, row j, gate kind, G argmax, committed, E argmax, E lp, KLD) from the
    checks and their replay (None fields without a replay)"""
    out, rep = p["row"]["completion_ids"], {r["c"]: r for r in (p["replay"] or [])}
    t = []
    for c in p["checks"]:
        e = rep.get(c["c"])
        for j in range(c["n"]):
            pos = c["pos"] + j
            t.append(dict(pos=pos, g=c["g"][j], gam=c["am"][j], tok=out[pos], eam=e["eam"][j] if e else None,
                          lp=e["lpc"][j] if e else None, kld=e["kld"][j] if e else None, cm=j in c.get("cm", ())))
    return t


def cause(t):
    if t["tok"] != t["gam"]:
        return "rule_comment" if t["cm"] else "rule"
    return {1: "gate_cf", 2: "gate_copy"}.get(t["g"], "context")


def quality(arm, sets=None):
    """per set: tokens, deviations by cause, blunders, drift, must-fix list"""
    res = {}
    for k, p in arm["prompts"].items():
        s = p["row"]["set"]
        if sets and s not in sets or p["replay"] is None:
            continue
        q = res.setdefault(s, dict(prompts=0, tokens=0, dev=0, causes={}, blunders=[], long=0,
                                   qk=[[0.0, 0] for _ in range(4)], qd=[[0, 0] for _ in range(4)], mustfix=[]))
        q["prompts"] += 1
        ts = tokens(p)
        q["tokens"] += len(ts)
        long_ = len(p["row"]["completion_ids"]) >= LONG
        qs = quarters(len(p["row"]["completion_ids"]))
        q["long"] += long_
        for t in ts:
            dev = t["tok"] != t["eam"]
            if dev:
                q["dev"] += 1
                c = cause(t)
                q["causes"][c] = q["causes"].get(c, 0) + 1
                q["mustfix"].append((k, t["pos"], t["tok"], t["eam"], c))
            if t["lp"] < LN_BLUNDER:
                q["blunders"].append((k, t["pos"], t["tok"], round(math.exp(t["lp"]), 5)))
            if long_:
                qq = qs[t["pos"]]
                q["qk"][qq][0] += t["kld"]; q["qk"][qq][1] += 1
                q["qd"][qq][0] += dev; q["qd"][qq][1] += 1
    for q in res.values():
        mk = [a / b if b else 0.0 for a, b in q["qk"]]
        md = [a / b if b else 0.0 for a, b in q["qd"]]
        q["kld_q"], q["dev_q"] = mk, md
        q["drift"] = (mk[3] / mk[0]) if mk[0] > 0 else (None if mk[3] == 0 else math.inf)
    return res


def speed(arm, control=None):
    """per set: kept, gated rows, bytes, tok/s; paired with the control on shared keys"""
    dense = arm["chunks"][-1]["dense"] if arm["chunks"] else 0
    res = {}
    for k, p in arm["prompts"].items():
        r, cs = p["row"], p["checks"]
        if not cs:
            continue
        s = r["set"]
        q = res.setdefault(s, dict(prompts=0, checks=0, kept=0, kept_x=0, rows=0, g_cf=0, g_copy=0, ub=0, ubx=0, ssd=0,
                                   tok=0, dec_s=0.0, verify=[], keys=[]))
        q["prompts"] += 1
        q["checks"] += len(cs)
        q["kept"] += sum(c["k"] + 1 for c in cs)
        q["kept_x"] += sum(c["kx"] + 1 for c in cs)
        g = [x for c in cs for x in c["g"]]
        q["rows"] += len(g); q["g_cf"] += g.count(1); q["g_copy"] += g.count(2)
        q["ub"] += sum(c["ub"] for c in cs); q["ubx"] += sum(c["ubx"] for c in cs); q["ssd"] += sum(c["ssd"] for c in cs)
        if not r.get("paused") and r.get("tok_s"):
            q["tok"] += len(r["completion_ids"]); q["dec_s"] += len(r["completion_ids"]) / r["tok_s"]
            q["keys"].append(k)
        q["verify"] += [c["t"][1] for c in cs]
    for s, q in res.items():
        n = max(q["checks"], 1)
        q.update(kept_c=q["kept"] / n, kept_xc=q["kept_x"] / n, gain=q["kept"] / max(q["kept_x"], 1),
                 cf=q["g_cf"] / max(q["rows"], 1), copy=q["g_copy"] / max(q["rows"], 1), ub_gb=q["ub"] / n / 1e9,
                 ubx_gb=q["ubx"] / n / 1e9, ssd_gb=q["ssd"] / n / 1e9, tok_s=q["tok"] / q["dec_s"] if q["dec_s"] else None,
                 verify_ms=float(np.median(q["verify"])) if q["verify"] else None)
        q["gbs"] = (q["ub"] / n + dense) / (q["verify_ms"] / 1e3) / 1e9 if q["verify_ms"] else None
        if control is not None and s in control["sets"]:
            ck = set(q["keys"]) & set(control["keys"].get(s, []))
            a_t = sum(len(arm["prompts"][k]["row"]["completion_ids"]) for k in ck)
            a_s = sum(len(arm["prompts"][k]["row"]["completion_ids"]) / arm["prompts"][k]["row"]["tok_s"] for k in ck)
            c_t = sum(len(control["prompts"][k]["row"]["completion_ids"]) for k in ck)
            c_s = sum(len(control["prompts"][k]["row"]["completion_ids"]) / control["prompts"][k]["row"]["tok_s"] for k in ck)
            q["paired"] = dict(keys=len(ck), ratio=(a_t / a_s) / (c_t / c_s) if ck and a_s and c_s else None)
    return res


def per_task(arm, control):
    """tokens and seconds per task vs the control on the keys both ran unpaused: per set and pooled. s/token = decode s
    per committed token; s/task = prefill s + decode s. -> {set or "pooled": dict(n, tok_r, task_r, tokpt_r)}"""
    out, pool = {}, dict(a_tok=0, a_dec=0.0, a_task=0.0, c_tok=0, c_dec=0.0, c_task=0.0, n=0)
    if control is None:
        return out
    for k, p in arm["prompts"].items():
        r = p["row"]
        c = control["prompts"].get(k, {}).get("row")
        if c is None or r.get("paused") or c.get("paused") or not r.get("dec_s") or not c.get("dec_s"):
            continue
        q = out.setdefault(r["set"], dict(a_tok=0, a_dec=0.0, a_task=0.0, c_tok=0, c_dec=0.0, c_task=0.0, n=0))
        for d in (q, pool):
            d["n"] += 1
            d["a_tok"] += len(r["completion_ids"]); d["c_tok"] += len(c["completion_ids"])
            d["a_dec"] += r["dec_s"]; d["c_dec"] += c["dec_s"]
            d["a_task"] += (r.get("prefill_s") or 0) + r["dec_s"]; d["c_task"] += (c.get("prefill_s") or 0) + c["dec_s"]
    if pool["n"]:
        out["pooled"] = pool
    for q in out.values():
        q["tok_r"] = (q["a_dec"] / max(q["a_tok"], 1)) / (q["c_dec"] / max(q["c_tok"], 1))   # s per token, arm / control
        q["task_r"] = q["a_task"] / q["c_task"]                                             # s per task, arm / control
        q["tokpt_r"] = q["a_tok"] / max(q["c_tok"], 1)                                      # tokens per task, arm / control
    return out


def task_kill(pt):
    """the per-task line on the pooled shared keys: faster per token AND more than 5 % slower per task"""
    q = pt.get("pooled")
    return bool(q) and q["tok_r"] < 1.0 and q["task_r"] > TASK_SLOWER


def ref_lengths(arm):
    """tokens per task, arm / REF2-T's stored rows (results/ref2t), per set on the keys both hold (no timing noise)"""
    p = os.path.join(ROOT, "results", "ref2t", "decode.jsonl")
    ref = {r["key"]: len(r["completion_ids"]) for r in jl(p)} if os.path.exists(p) else {}
    out = {}
    for k, pr in arm["prompts"].items():
        if k in ref:
            q = out.setdefault(pr["row"]["set"], [0, 0, 0])
            q[0] += 1; q[1] += len(pr["row"]["completion_ids"]); q[2] += ref[k]
    return {s: (n, a / max(b, 1)) for s, (n, a, b) in out.items()}


def control_index(ctl):
    if ctl is None:
        return None
    keys = {}
    for k, p in ctl["prompts"].items():
        if not p["row"].get("paused") and p["row"].get("tok_s"):
            keys.setdefault(p["row"]["set"], []).append(k)
    return dict(prompts=ctl["prompts"], keys=keys, sets=set(keys))


def build_modes(arms, tag):
    made = {}
    for m, spec in MODES.items():
        rows = []
        for s, a in spec.items():
            arm = arms.get(a)
            if arm is None:
                break
            rows += [p["row"] for p in arm["prompts"].values() if p["row"]["set"] == s]
        else:
            d = os.path.join(RES, f"relax_mode_{m}" + (f"_{tag}" if tag else ""))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "decode.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            made[m] = (d, len(rows))
    return made


def score(mode_dir, ref="ref2t"):
    """decision 4 on a mode's rows: `evalsuite report` prints the table; it goes to results/report_<name>_vs_<ref>.txt"""
    name = os.path.basename(mode_dir)
    r = subprocess.run([os.path.join(ROOT, "bin", "evalsuite"), "report", "--variant", name, "--decode", "--ref", ref],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    rep = os.path.join(RES, f"report_{name}_vs_{ref}.txt")
    open(rep, "w").write(r.stdout)
    r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, "tools", "d4.py"), rep],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()


def bars(arms, spec):
    """mission 7's bars for a mode from its arms' replays: blunders per 10K on math + code, agent blunders, drift"""
    qa = {}
    for s, a in list(spec.items()) + [("agent", spec.get("humaneval")), ("multi", spec.get("humaneval"))]:
        arm = arms.get(a)
        if arm is not None:
            qa[s] = quality(arm, {s}).get(s)
    mc = [q for s, q in qa.items() if s in ("math", "humaneval") and q]
    tok = sum(q["tokens"] for q in mc)
    bl = sum(len(q["blunders"]) for q in mc)
    per10k = 1e4 * bl / tok if tok else None
    agent_bl = len(qa["agent"]["blunders"]) if qa.get("agent") else None
    drifts = {s: q["drift"] for s, q in qa.items() if q and s in D4_SETS}
    ok_b = per10k is not None and per10k <= BARS["blunder_per_10k"] and not agent_bl
    ok_d = all(d is None or d <= BARS["drift"] for d in drifts.values()) and bool(drifts)
    return dict(per10k=per10k, agent_blunders=agent_bl, drift=drifts, blunders_ok=ok_b, drift_ok=ok_d)


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.2f} %"


def report(arm_names, tag, modes=False, do_score=False, tok=None, outfile=None):
    arms = {a: load_arm(a, tag) for a in arm_names}
    arms = {a: x for a, x in arms.items() if x is not None}
    ctl = control_index(arms.get("control"))
    L = []
    say = L.append
    say(f"RELAX report ({', '.join(arms)}; tag {tag or '-'})")
    say("\n== speed and bytes (G's logs; tok/s on unpaused prompts; paired = tok/s ratio vs the control on shared keys)")
    say(f"{'arm':8}{'set':10}{'n':>4}{'checks':>8}{'kept':>7}{'exact':>7}{'gain':>7}{'cf':>8}{'copy':>8}"
        f"{'union':>8}{'exact':>8}{'SSD':>7}{'tok/s':>8}{'GB/s':>7}{'paired':>8}")
    for a, arm in arms.items():
        for s, q in sorted(speed(arm, ctl if a != "control" else None).items()):
            pr = q.get("paired", {}).get("ratio")
            say(f"{a:8}{s:10}{q['prompts']:>4}{q['checks']:>8}{q['kept_c']:>7.2f}{q['kept_xc']:>7.2f}{q['gain']:>7.3f}"
                f"{100 * q['cf']:>7.1f}%{100 * q['copy']:>7.1f}%{q['ub_gb']:>8.2f}{q['ubx_gb']:>8.2f}{q['ssd_gb']:>7.2f}"
                f"{q['tok_s'] or 0:>8.1f}{q['gbs'] or 0:>7.0f}{(f'x{pr:.3f}' if pr else '-'):>8}")
    at = os.path.join(ROOT, "results", "relax", "keys_at14.txt")
    if os.path.exists(at):
        at14 = open(at).read().split()
        say("\n== AT14, the agent-turn read-out (no line; answers cap at the items' 128 tokens, real agent answers run longer)")
        say(f"{'arm':8}{'turn':16}{'prompt':>7}{'prefill s':>10}{'1st chk ms':>11}{'decode s':>9}{'answer':>7}{'turn tok/s':>11}"
            f"{'prefill %':>10}")
        for a in ("control", "B"):
            arm = arms.get(a)
            if arm is None:
                continue
            got = [arm["prompts"][k]["row"] for k in at14 if k in arm["prompts"]]
            for r in got:
                pf, dc, n = r.get("prefill_s") or 0.0, r.get("dec_s") or 0.0, len(r["completion_ids"])
                say(f"{a:8}{r['key'][6:]:16}{r['n_prompt']:>7}{pf:>10.1f}{(r.get('first_check_ms') or 0):>11.0f}{dc:>9.1f}"
                    f"{n:>7}{n / (pf + dc) if pf + dc else 0:>11.2f}{100 * pf / (pf + dc) if pf + dc else 0:>9.0f}%"
                    + ("  (paused)" if r.get("paused") else ""))
            ok = [r for r in got if not r.get("paused")]
            if ok:
                pf = sum(r.get("prefill_s") or 0 for r in ok); dc = sum(r.get("dec_s") or 0 for r in ok)
                n = sum(len(r["completion_ids"]) for r in ok)
                say(f"{a:8}{'pooled':16}{'':>7}{pf:>10.1f}{'':>11}{dc:>9.1f}{n:>7}{n / (pf + dc):>11.2f}{100 * pf / (pf + dc):>9.0f}%"
                    f"  ({len(ok)} unpaused turns; TTFT = prefill s)")
    say("\n== per task (the orchestrator Wed 21:5x; PREREG amendment): tokens / s per task vs the control on shared unpaused"
        " keys; KILL for FAST = faster per token and > 5 % slower per task (pooled); tokens per task vs REF2-T's rows")
    say(f"{'arm':8}{'set':10}{'n':>4}{'tok/task':>10}{'s/task':>9}{'s/token':>9}{'vs ref2t':>10}")
    pts = {}
    for a, arm in arms.items():
        if a == "control":
            continue
        pts[a] = per_task(arm, ctl)
        rl = ref_lengths(arm)
        for s_ in sorted(set(pts[a]) | set(rl)):
            q = pts[a].get(s_)
            r_ = rl.get(s_)
            say(f"{a:8}{s_:10}{(q['n'] if q else 0):>4}" + (f"{q['tokpt_r']:>9.3f}x{q['task_r']:>8.3f}x{q['tok_r']:>8.3f}x"
                if q else f"{'-':>10}{'-':>9}{'-':>9}") + (f"{r_[1]:>9.3f}x" if r_ else f"{'-':>10}"))
        if a in RELAXED:
            say(f"{a:8}per-task line: " + ("KILL (faster per token, slower per task)" if task_kill(pts[a])
                else "no kill" if pts[a].get("pooled") else "no shared keys"))
    say("\n== quality (REF2-T replay along G's prefix)")
    say(f"{'arm':8}{'set':10}{'n':>4}{'tokens':>8}{'dev':>9}{'rule':>6}{'cmnt':>6}{'cf':>6}{'copy':>6}{'other':>6}{'blund':>6}"
        f"{'/10K':>7}{'long':>5}{'KLD q1':>9}{'q4':>9}{'drift':>7}{'dev q1':>8}{'q4':>8}")
    allq = {}
    for a, arm in arms.items():
        allq[a] = quality(arm)
        for s, q in sorted(allq[a].items()):
            c = q["causes"]
            dr = q["drift"]
            say(f"{a:8}{s:10}{q['prompts']:>4}{q['tokens']:>8}{fmt_pct(q['dev'] / max(q['tokens'], 1)):>9}"
                f"{c.get('rule', 0):>6}{c.get('rule_comment', 0):>6}{c.get('gate_cf', 0):>6}{c.get('gate_copy', 0):>6}"
                f"{c.get('context', 0):>6}"
                f"{len(q['blunders']):>6}{1e4 * len(q['blunders']) / max(q['tokens'], 1):>7.2f}{q['long']:>5}"
                f"{q['kld_q'][0]:>9.5f}{q['kld_q'][3]:>9.5f}{('-' if dr is None else f'{dr:.2f}'):>7}"
                f"{fmt_pct(q['dev_q'][0]):>8}{fmt_pct(q['dev_q'][3]):>8}")
    top_sets = ("humaneval", "humanevalx", "humanevalt")   # PREREG_relax_drift amendment 2: the code-only top-up
    for a in arms:
        if "humanevalt" not in allq[a]:
            continue
        rows_, lq = [], [[0.0, 0] for _ in range(4)]
        for k, pr in sorted(arms[a]["prompts"].items()):
            n_ = len(pr["row"]["completion_ids"])
            if pr["row"]["set"] in top_sets and n_ >= LONG and pr["replay"] is not None:
                qs = quarters(n_)
                acc = [[0.0, 0] for _ in range(4)]
                for t in tokens(pr):
                    acc[qs[t["pos"]]][0] += t["kld"]; acc[qs[t["pos"]]][1] += 1
                for i in range(4):
                    lq[i][0] += acc[i][0]; lq[i][1] += acc[i][1]
                rows_.append((k, n_, [x / m if m else 0.0 for x, m in acc]))
        keys_ = {k for k, _, _ in rows_}
        tok_ = sum(n_ for _, n_, _ in rows_)
        bl_ = [b for s in top_sets if s in allq[a] for b in allq[a][s]["blunders"] if b[0] in keys_]
        per = 1e4 * len(bl_) / tok_ if tok_ else None
        pq = [x / m if m else 0.0 for x, m in lq]
        mx_ = max((q_[3] for _, _, q_ in rows_), default=0.0)
        ok = per is not None and per <= 1.0 and pq[3] <= 0.02 and mx_ <= 0.05
        say(f"\n== code-only top-up (PREREG_relax_drift amendment 2): arm {a}, {len(rows_)} code rows >= {LONG} tokens over "
            f"{', '.join(s for s in top_sets if s in allq[a])} ({tok_} tokens): blunders {len(bl_)} = "
            + ("-" if per is None else f"{per:.2f}") + f"/10K (line <= 1), pooled q4 KLD {pq[3]:.5f} (line <= 0.02), largest "
            f"row q4 {mx_:.5f} (line <= 0.05): " + ("PASS" if ok else "FAIL")
            + ("" if len(rows_) >= 20 else f" (only {len(rows_)} long rows: the orchestrator decides if enough)"))
        say(f"  information: the code-only q4 / q1 ratio " + (f"{pq[3] / pq[0]:.2f}" if pq[0] > 0 else "-")
            + f" (q1 {pq[0]:.5f})")
        for k, n_, q_ in rows_:
            say(f"  {k}: {n_} tokens, KLD q1 {q_[0]:.5f}, q4 {q_[3]:.5f}")
    code_sets = ("humaneval", "humanevalx", "agentlong")   # PREREG_relax_drift: every code row of >= 512 tokens, pooled
    for a in arms:
        qs_ = [allq[a][s] for s in code_sets if s in allq[a]]
        if not any(s in allq[a] for s in ("humanevalx", "agentlong")):
            continue
        k_ = [[sum(q["qk"][i][0] for q in qs_), sum(q["qk"][i][1] for q in qs_)] for i in range(4)]
        mk = [x / n if n else 0.0 for x, n in k_]
        r_ = (mk[3] / mk[0]) if mk[0] > 0 else None
        say(f"\n== long-code drift (PREREG_relax_drift): arm {a}, {sum(q['long'] for q in qs_)} rows >= {LONG} tokens "
            f"over {', '.join(s for s in code_sets if s in allq[a])}: KLD q1 {mk[0]:.5f}, q4 {mk[3]:.5f}, ratio "
            + ("-" if r_ is None else f"{r_:.2f}") + f" (line <= {BARS['drift']}): "
            + ("no long rows" if r_ is None else "PASS" if r_ <= BARS["drift"] else "FAIL"))
        for s in code_sets:
            if s in allq[a]:
                q = allq[a][s]
                dr_ = "-" if q["drift"] is None else f"{q['drift']:.2f}"
                say(f"  {s}: {q['long']} long rows, drift {dr_}")
        for k, pr in sorted(arms[a]["prompts"].items()):
            if pr["row"]["set"] in code_sets and len(pr["row"]["completion_ids"]) >= LONG and pr["replay"] is not None:
                qs = quarters(len(pr["row"]["completion_ids"]))
                acc = [[0.0, 0] for _ in range(4)]
                for t in tokens(pr):
                    acc[qs[t["pos"]]][0] += t["kld"]; acc[qs[t["pos"]]][1] += 1
                m_ = [x / n if n else 0.0 for x, n in acc]
                say(f"  {k}: {len(pr['row']['completion_ids'])} tokens, KLD q1 {m_[0]:.5f}, q4 {m_[3]:.5f}")
    say("\n== blunders (E's p of the committed token < 0.01)")
    for a in arms:
        for s, q in sorted(allq[a].items()):
            for k, pos, t, p in q["blunders"]:
                where = ""
                if tok and s in ("agent", "agentlong"):        # the tool call's first parameter (command / code) or later
                    ids_ = arms[a]["prompts"][k]["row"]["completion_ids"][:pos]
                    where = " (after the first parameter)" if "</parameter>" in "".join(tok(i) for i in ids_) \
                        else " (in the first parameter: the command / code)"
                say(f"{a} {s} {k} @{pos}: token {t}{(' ' + repr(tok(t))) if tok else ''}, p {p}{where}")
    say("\n== must-fix deviations (numbers; any token in code / agent answers), information")
    for a in arms:
        for s, q in sorted(allq[a].items()):
            code = s in ("humaneval", "agent")
            mf = [(k, pos, t, e, c) for k, pos, t, e, c in q["mustfix"]
                  if tok is None or token_class(tok(t), code) in ("number", "code") or token_class(tok(e), code) in ("number", "code")]
            say(f"{a} {s}: {len(mf)} of {len(q['mustfix'])} deviations")
            for k, pos, t, e, c in mf[:40]:
                say(f"  {k} @{pos} ({c}): {repr(tok(t)) if tok else t} for E's {repr(tok(e)) if tok else e}")
    if modes:
        made = build_modes(arms, tag)
        say("\n== modes (decision 4 vs REF2-T + mission 7's bars)")
        for m, (d, n) in made.items():
            b = bars(arms, MODES[m])
            line = f"{m}: {n} rows ({d}); blunders {b['per10k'] if b['per10k'] is None else round(b['per10k'], 2)}/10K on " \
                   f"math + code, agent {b['agent_blunders']}, drift {b['drift']}"
            if do_score and n == 210:
                ok4, txt = score(d)
                verdict = "KILL" if not ok4 else ("PASS" if b["blunders_ok"] and b["drift_ok"] else "BETWEEN")
                gaps = {x: sum(1 for pr in arms[x]["prompts"].values() if pr["replay"] is None)
                        for x in set(MODES[m].values()) if x in arms}
                gaps = {x: n_ for x, n_ in gaps.items() if n_}
                if ok4 and gaps:                  # the bars need every gated arm's REF2-T replay (Thu 07:3x: C, G, T3ALL short)
                    verdict = "INCOMPLETE (replays missing: " + ", ".join(f"{x} {n_}" for x, n_ in sorted(gaps.items())) + ")"
                slow = sorted({x for x in MODES[m].values() if x in RELAXED and task_kill(pts.get(x, {}))})
                if slow:                          # the per-task line (amendment Wed 21:5x) overrides decision 4
                    verdict = f"KILL (per task: {', '.join(slow)} faster per token, slower per task)"
                line += f"\n{txt}\n  -> {m}: {verdict}"
            say(line)
    text = "\n".join(L)
    if outfile:
        open(outfile, "w").write(text + "\n")
    return text, arms, allq


# ---------------------------------------------------------------------------------------------- selftest
def selftest():
    """synthetic arms: a control and an arm A with known deviations, blunders, gates and bytes"""
    global RES
    tmp = tempfile.mkdtemp(prefix="relax_selftest_")
    RES_SAVE, RES = RES, tmp
    try:
        rng = np.random.default_rng(0)

        def write_arm(arm, rows, checks, reps):
            d = arm_dir(arm, "st")
            os.makedirs(os.path.join(d, "checks"), exist_ok=True)
            os.makedirs(os.path.join(d, "replay"), exist_ok=True)
            with open(os.path.join(d, "decode.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for k, cs in checks.items():
                with gzip.open(os.path.join(d, "checks", k.replace("/", "__") + ".jsonl.gz"), "wt") as f:
                    for c in cs:
                        f.write(json.dumps(c) + "\n")
            for k, rs in reps.items():
                with gzip.open(os.path.join(d, "replay", k.replace("/", "__") + ".jsonl.gz"), "wt") as f:
                    f.write(json.dumps(dict(key=k, first_equal=True)) + "\n")
                    for r in rs:
                        f.write(json.dumps(r) + "\n")
            with open(os.path.join(d, "chunks.jsonl"), "w") as f:
                f.write(json.dumps(dict(dense=3_000_000_000)) + "\n")

        def prompt(key, s, n_tok, devs, blunders, gate_rows_, rule_rows, tok_s, ub, ubx):
            """n_tok tokens in checks of 8; devs / blunders / rule_rows: sets of positions"""
            out = [int(x) for x in rng.integers(10, 1000, n_tok)]
            cs, rs, pos, c = [], [], 1, 0
            while pos < n_tok:
                n = min(8, n_tok - pos)
                g = [1 if pos + j in gate_rows_ else 0 for j in range(n)] + [0] * (16 - n)
                am = [out[pos + j] if pos + j not in rule_rows else out[pos + j] + 1 for j in range(n)] + [0] * (16 - n)
                cs.append(dict(c=c, pos=pos, inp=[out[pos - 1]] + [0] * 15, g=g, am=am, k=n - 1, kx=max(n - 2, 0), n=n,
                               ti=[[0] * 4] * n, tl=[[-0.1] * 4] * n, ub=ub, ubx=ubx, ssd=10**8, t=[10.0, 100.0, 1.0]))
                eam = [out[pos + j] + (7 if pos + j in devs else 0) for j in range(n)]
                lpc = [math.log(0.005) if pos + j in blunders else -0.01 for j in range(n)]
                kld = [0.01 * (1 + 3 * (pos + j >= 3 * n_tok // 4)) if s != "ifeval" else 0.0 for j in range(n)]
                rs.append(dict(c=c, pos=pos, eam=eam, lpc=lpc, lp1=[-0.01] * n, kld=kld))
                pos += n; c += 1
            row = dict(key=key, set=s, completion_ids=out, tok_s=tok_s, paused=False)
            return row, cs, rs

        rows, chk, rep = [], {}, {}
        for i in range(3):
            r, c, x = prompt(f"humaneval/H/{i}", "humaneval", 600, set(), set(), set(), set(), 50.0, 7e9, 7e9)
            rows.append(r); chk[r["key"]] = c; rep[r["key"]] = x
        write_arm("control", rows, chk, rep)
        rows, chk, rep = [], {}, {}
        for i in range(3):     # 6 deviations per prompt: 2 rule, 2 gate (cf), 2 context; 1 blunder
            r, c, x = prompt(f"humaneval/H/{i}", "humaneval", 600, {10, 11, 20, 21, 30, 31}, {40}, {20, 21, 50},
                             {10, 11}, 60.0, 5e9, 7e9)
            rows.append(r); chk[r["key"]] = c; rep[r["key"]] = x
        write_arm("A", rows, chk, rep)
        # an exact-numerics arm (R1 only, no replay files): E comes from G's own logs
        rows, chk = [], {}
        for i in range(2):
            out = [int(x) for x in rng.integers(10, 1000, 41)]
            cs, pos, c = [], 1, 0
            while pos < 41:
                n = min(8, 41 - pos)
                am = [out[pos + j] + (1 if (pos + j) in (5, 30) else 0) for j in range(n)] + [0] * (16 - n)
                dlm = [(-0.3 if (pos + j) in (5, 30) else 0.0) for j in range(n)] + [-2.0] * (15 - n)
                tl = [[-3.0 if (pos + j) == 30 else -0.05] + [-9.0] * 3 for j in range(n)]
                cs.append(dict(c=c, pos=pos, inp=[0] * 16, g=[0] * 16, am=am, k=n - 1, kx=0, n=n, dlm=dlm[:15], ti=[[0] * 4] * n,
                               tl=tl, ub=1, ubx=1, ssd=0, t=[1.0, 1.0, 1.0], cm=[]))
                pos += n; c += 1
            r = dict(key=f"ifeval/{i}", set="ifeval", completion_ids=out, tok_s=20.0, paused=False, hc8=False,
                     cfg=dict(thr=None, copy=False, rules=[["R1", 0.5]], loop=None))
            rows.append(r); chk[r["key"]] = cs
        write_arm("E", rows, chk, {})
        text, arms, allq = report(["control", "A", "E"], "st")
        qe = allq["E"]["ifeval"]
        # per prompt: rows 5 and 30 committed a draft (R1) != G's argmax: 2 rule deviations; row 30's lp = -3.0 - 0.3
        # (p 0.037, no blunder); KLD 0 everywhere
        okE = (qe["tokens"] == 80 and qe["dev"] == 4 and qe["causes"] == {"rule": 4} and not qe["blunders"]
               and qe["kld_q"] == [0.0] * 4)
        print(f"selftest E (no replay): tokens {qe['tokens']}, dev {qe['dev']}, causes {qe['causes']}, "
              f"blunders {len(qe['blunders'])}: {'OK' if okE else 'FAIL'}")
        q = allq["A"]["humaneval"]
        ok = okE
        exp = dict(tokens=3 * 599, dev=18, rule=6, gate_cf=6, context=6, blunders=3)
        got = dict(tokens=q["tokens"], dev=q["dev"], rule=q["causes"].get("rule", 0), gate_cf=q["causes"].get("gate_cf", 0),
                   context=q["causes"].get("context", 0), blunders=len(q["blunders"]))
        ok &= got == exp
        ok &= abs(q["drift"] - 4.0) < 1e-9                    # KLD 0.04 in the last quarter vs 0.01 in the first
        sp = speed(arms["A"], control_index(arms["control"]))["humaneval"]
        ok &= abs(sp["paired"]["ratio"] - 1.2) < 1e-9         # 60 vs 50 tok/s
        ok &= abs(sp["ub_gb"] - 5.0) < 1e-9 and abs(sp["ubx_gb"] - 7.0) < 1e-9
        ok &= abs(sp["gain"] - sum(min(8, 599 - 8 * i) for i in range(75)) / sum(max(min(8, 599 - 8 * i) - 1, 1)
                                                                                for i in range(75))) < 1e-9
        print(text)
        print(f"\nselftest: expected {exp}, got {got}; drift {q['drift']}; paired x{sp['paired']['ratio']:.3f}; "
              f"union {sp['ub_gb']:.2f} / {sp['ubx_gb']:.2f} GB; gain {sp['gain']:.4f}: {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        RES = RES_SAVE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="control,A,B,E,T3ALL,F,C,G")
    ap.add_argument("--tag", default="")
    ap.add_argument("--modes", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    tok = None
    try:
        from tokenizers import Tokenizer
        t_ = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
        tok = lambda i: t_.decode([int(i)])
    except Exception:
        pass
    text, _, _ = report(a.arms.split(","), a.tag, modes=a.modes or a.score, do_score=a.score, tok=tok,
                        outfile=a.out or os.path.join(RES, "relax_report" + (f"_{a.tag}" if a.tag else "") + ".txt"))
    print(text)


if __name__ == "__main__":
    main()
