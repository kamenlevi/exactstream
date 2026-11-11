"""nocomment_replay.py (Fable FULL, LOSSY350, Thu 5 Nov 2026): what the recorded HPD-0 walks say about a code mode
in which the 125B never opens a '#' comment (the path is changed, not chased).
Data [K]: TrainLab's HPD-0 walks (DFlash2 L1-20, block 16: per check kept / nrows / anchor, the 15 drafted tokens, the
15 true tokens), the model's tokenizer, EvalSuite's causal tagger (relax_core.CodeScan, the engine's own).
CPU only, numpy, seconds. Run: nice -n 15 ~/BigMoE/.venv/bin/python -I nocomment_replay.py > nocomment_replay.out
Estimates (all on the recorded walk, no model):
  base      = the walk's kept drafted rows per check (+ 1 bonus).
  pure      = the same on checks with no comment token among the 15 true or the 15 drafted tokens (code-only stretches).
  strip     = draft minus its comment tokens against the true text minus its comment tokens, from the same anchor
              (anchors inside a comment dropped); the draft is NOT refilled, so this undercounts.
  hazard    = per-row first-miss hazard on comment / non-comment rows, and on code rows that follow a comment line.
"""
import importlib.util, sys
import numpy as np
from tokenizers import Tokenizer

TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
WALK = "/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz"
_dec = {}


def dec(i):
    i = int(i)
    if i not in _dec:
        _dec[i] = TOK.decode([i])
    return _dec[i]


def scan_tokens(sc, toks):
    """feed token texts through the causal scanner; per token: 1 = a comment token (any of its chars is in a comment),
    2 = the newline token that ends a comment, 0 = other. Returns the flags; sc is advanced."""
    out = []
    for s in toks:
        before = bool(sc.code and sc.com)
        anyc = False
        for ch in s:
            sc._ch(ch)
            if sc.code and sc.com:
                anyc = True
        if anyc:
            out.append(1)
        elif before:
            out.append(2 if s.startswith("\n") else 1)
        else:
            out.append(0)
    return out


def clone(sc):
    c = rc.CodeScan(); c.__dict__.update(sc.__dict__); return c


def run(dom):
    W = np.load(WALK % dom, allow_pickle=True)
    names = list(W["names"])
    tot = dict(checks=0, kept=0, rows=0, miss=0, com_rows=0, com_miss=0, end_rows=0, end_miss=0,
               pure_checks=0, pure_kept=0, pure_full=0, strip_checks=0, strip_kept=0, strip_len=0,
               gen=0, gen_com=0, gen_end=0, gen_comline=0,
               after_rows=0, after_miss=0, other_rows=0, other_miss=0, fenced=0)
    fm_pure = np.zeros(16, int)
    for wi in range(len(names)):
        ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dtok = W[f"{wi}_dtok"]; ttok = W[f"{wi}_ttok"]
        texts = [dec(i) for i in ids]
        # causal states: st[p] = scanner state BEFORE token p; flag[p] of token p
        sc = rc.CodeScan(); st = []; flag = []
        for p, s in enumerate(texts):
            st.append(clone(sc)); flag.extend(scan_tokens(sc, [s]))
        tot["fenced"] += int(any(s.code for s in st))
        gen0 = int(ck[0, 2]) + 1 if ck.shape[0] else len(ids)
        g = flag[gen0:]
        tot["gen"] += len(g); tot["gen_com"] += sum(1 for f in g if f == 1); tot["gen_end"] += sum(1 for f in g if f == 2)
        # comment-only lines in the generated part: a comment whose line has only blanks before the '#'
        # after_com[p]: token p is a code token on the line right after a comment-ending newline
        after = [False] * len(ids); live = False
        for p in range(len(ids)):
            if flag[p] == 2:
                live = True; continue
            if live:
                if flag[p] == 0 and "\n" in texts[p]:
                    live = False
                elif flag[p] == 0:
                    after[p] = True
        # comment-only lines: count comment runs that start right after a newline / indent-only tokens
        p = gen0
        while p < len(ids):
            if flag[p] == 1 and (p == 0 or flag[p - 1] != 1):
                q = p - 1
                while q >= 0 and texts[q].strip() == "" and "\n" not in texts[q]:
                    q -= 1
                lead = texts[p].split("#", 1)[0]
                if (q < 0 or "\n" in texts[q]) and lead.strip() == "":
                    tot["gen_comline"] += 1
            p += 1
        for ci in range(ck.shape[0]):
            kept, nrows, a = int(ck[ci, 0]), int(ck[ci, 1]), int(ck[ci, 2])
            d = [int(x) for x in dtok[ci][:nrows]]; t = [int(x) for x in ttok[ci][:nrows]]
            tot["checks"] += 1; tot["kept"] += kept
            fm = next((j for j in range(nrows) if d[j] != t[j]), nrows)
            # tested rows: up to and including the first miss
            for j in range(min(fm + 1, nrows)):
                p = a + 1 + j
                if p >= len(ids):
                    break
                f = flag[p]; m = int(j == fm)
                tot["rows"] += 1; tot["miss"] += m
                if f == 1:
                    tot["com_rows"] += 1; tot["com_miss"] += m
                elif f == 2:
                    tot["end_rows"] += 1; tot["end_miss"] += m
                elif after[p]:
                    tot["after_rows"] += 1; tot["after_miss"] += m
                else:
                    tot["other_rows"] += 1; tot["other_miss"] += m
            # draft-side flags from the anchor's state
            scd = clone(st[a + 1]) if a + 1 < len(st) else clone(sc)
            dflag = scan_tokens(scd, [dec(x) for x in d])
            tflag = [flag[a + 1 + j] if a + 1 + j < len(ids) else 0 for j in range(nrows)]
            anchor_in_com = bool(st[a + 1].code and st[a + 1].com) if a + 1 < len(st) else False
            if not any(tflag) and not any(dflag) and not anchor_in_com:
                tot["pure_checks"] += 1; tot["pure_kept"] += kept; tot["pure_full"] += int(fm == nrows); fm_pure[fm] += 1
            if not anchor_in_com:
                dp = [x for x, f in zip(d, dflag) if f == 0]
                # the true text minus comments from the anchor on (beyond the check's 15 rows if needed)
                tp = []; p = a + 1
                while p < len(ids) and len(tp) < len(dp):
                    if flag[p] == 0:
                        tp.append(ids[p])
                    elif flag[p] == 2:
                        # the newline that ends a comment: kept for an inline comment, gone with a comment-only line.
                        # Keep it only if the last non-comment token before the comment was not a newline / indent.
                        q = p - 1
                        while q >= 0 and flag[q] == 1:
                            q -= 1
                        while q >= 0 and texts[q].strip() == "" and "\n" not in texts[q]:
                            q -= 1
                        if q >= 0 and "\n" not in texts[q]:
                            tp.append(ids[p])
                    p += 1
                k = 0
                while k < len(dp) and k < len(tp) and dp[k] == tp[k]:
                    k += 1
                tot["strip_checks"] += 1; tot["strip_kept"] += k; tot["strip_len"] += len(dp)
    return names, tot, fm_pure


def chain(h, R):
    """expected kept drafted rows for R-1 drafted rows at a per-row hazard list h (0-based depth), flat past its end"""
    s = 0.0; surv = 1.0
    for j in range(R - 1):
        surv *= 1.0 - (h[j] if j < len(h) else h[-1])
        s += surv
    return s


for dom in sys.argv[1:] or ["code", "mbpp"]:
    names, t, fm_pure = run(dom)
    print(f"== {dom}: {len(names)} windows ({t['fenced']} with a code fence seen by the tagger), {t['checks']} checks, {t['rows']} tested rows")
    base = t["kept"] / t["checks"]
    print(f"base: kept drafted rows per check {base:.2f} (+1 bonus = {base + 1:.2f}); first-miss hazard per tested row {t['miss'] / t['rows']:.4f}")
    print(f"generated tokens {t['gen']}: comment tokens {t['gen_com']} ({100 * t['gen_com'] / t['gen']:.1f} %), comment-ending newlines {t['gen_end']} "
          f"({100 * t['gen_end'] / t['gen']:.1f} %), comment-only lines {t['gen_comline']}")
    gone = t["gen_com"] + t["gen_comline"]
    print(f"tokens a comment-free answer drops: {gone} = {100 * gone / t['gen']:.1f} % of generated tokens (comment tokens + one newline per comment-only line)")
    cm = t["com_miss"] + t["end_miss"]
    print(f"tested rows: comment {t['com_rows']} ({100 * t['com_rows'] / t['rows']:.1f} %) hazard {t['com_miss'] / max(1, t['com_rows']):.4f}; "
          f"comment-end newline {t['end_rows']} hazard {t['end_miss'] / max(1, t['end_rows']):.4f}; "
          f"code after a comment line {t['after_rows']} hazard {t['after_miss'] / max(1, t['after_rows']):.4f}; "
          f"other code {t['other_rows']} hazard {t['other_miss'] / max(1, t['other_rows']):.4f}")
    print(f"first misses: {t['miss']}; on comment tokens or the comment's end {cm} = {100 * cm / max(1, t['miss']):.1f} %")
    if t["pure_checks"]:
        pk = t["pure_kept"] / t["pure_checks"]
        print(f"pure-code checks (no comment in the truth or the draft): {t['pure_checks']} ({100 * t['pure_checks'] / t['checks']:.0f} % of checks), "
              f"kept {pk:.2f} (+1 = {pk + 1:.2f}) = x{(pk + 1) / (base + 1):.3f} of base; all 15 right {100 * t['pure_full'] / t['pure_checks']:.0f} %")
        n = fm_pure.sum(); surv = n; h = []
        for j in range(15):
            h.append(fm_pure[j] / surv if surv else 0.0); surv -= fm_pure[j]
        print("pure-code hazard by depth 1-15: " + " ".join(f"{x:.3f}" for x in h))
        tail = float(np.mean(h[10:15]))
        for mult in (1.0, 1.5, 2.0):
            h32 = h + [tail * mult] * 16
            print(f"  chain on the pure-code hazards: 16 rows {chain(h, 16) + 1:.2f}; 32 rows {chain(h32, 32) + 1:.2f} "
                  f"(rows 16-31 at x{mult} the depth 11-15 hazard {tail:.3f})")
    if t["strip_checks"]:
        sk = t["strip_kept"] / t["strip_checks"]
        print(f"strip (draft minus its comments vs truth minus comments, no refill; anchors in comments dropped): {t['strip_checks']} checks, "
              f"kept {sk:.2f} of {t['strip_len'] / t['strip_checks']:.2f} drafted code rows (+1 = {sk + 1:.2f}) = x{(sk + 1) / (base + 1):.3f} of base")
    noncode_h = (t["after_miss"] + t["other_miss"]) / max(1, t["after_rows"] + t["other_rows"])
    print(f"code-row hazard {noncode_h:.4f} vs all-row {t['miss'] / t['rows']:.4f}; if code after a comment line took the 'other code' hazard x1 .. "
          f"its own: range {t['other_miss'] / max(1, t['other_rows']):.4f} .. {noncode_h:.4f}")
    print()
