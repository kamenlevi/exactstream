#!/bin/bash
# Fri 6 Nov ~01:40 (queued 00:32 behind MTP Head's P-OFF; the orchestrator Thu 10:3x approved the layout): one run on the 12
# HumanEval, memory-adaptive, every arm logging the ext's uses / stalled per check (C3-R8):
#   1. frx0: the relaxed copy with its flags off (2 prompts, CAP 64) == REF2-T (ref14 gate+t prefix); if it differs, the
#      relaxed-copy arms (qallr / span) are skipped (exactness first)
#   2. a palindrome, each arm twice and mirrored, so linear drift and warm-up cancel in the pair means (Thu 07:38's C 208
#      trio ran in a fixed order under a rising die and got no verdict):
#        u3c / qall / s99 / s90 / qallr / span | span / qallr / s90 / s99 / qall / u3c
#      u3c = U3C len; qall = the QOM all-in (as allin_wed.sh); s99 / s90 = qall + Breakthrough's split (QOM's resident keep
#      off on rows with cf >= T 0.99 / 0.9: R57_QOM_SPLIT, keepset 4963da6; FRI_SPLIT99 / 90 empty = skip);
#      qallr = qall + relaxed acceptance 0.5 nat (MoonShots 32edb12); span = qall + rule C (R57_RELAX_SPAN, MoonShots
#      9113d8d). u3c / qall / s99 / s90 run on the frozen screen, qallr / span on the relaxed copy (both snapshots; the
#      copy imports relax_delta / relax_span from its own folder, so both modules are copied beside it)
#   3. fexr: the exact default (R57_VTIER= to spare swap) for C3-R8's group bracket (FAST >= 0.85x exact's); timing unused
# C = the largest of 224 / 208 / 192 whose peak + 4 fits NO-SWAP at the start (qall 37.6 GB at C 208, measured Thu, + 0.134
# per row; the exact default ~40.3 at C 224): 224 >= 44.3, 208 >= 42.2, 192 >= 40.0, else nothing runs.
# Lines (pre-registered Thu 10:40, before arming):
#   split (each T): PASS = pair-mean geo steady tok/s >= +8 % vs qall's AND kept per check within 1 % (then EvalSuite's
#     same-scores); else NOT PASS.
#   rule C: PASS = median s per task <= 0.92x AND tokens per task <= 1.05x vs qall (AND comment share <= 1.15x: MoonShots
#     tags the tokens.json); KILL = s >= 0.97x OR tokens >= 1.15x; else BETWEEN.
#   relaxed 0.5: reported (no line). C3-R8: groups per check, FAST arms / exact, reported (the audit's PASS needs >= 0.85x).
# usage (a harness background task): NOTBEFORE=00:32 NOTAFTER=02:30 WAITMEM=40 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh \
#   37 60 "BigMoE: Fri palindrome ..." /bin/bash scripts/fri_night.sh   -> data/fri/fri.log, data/fri/fri_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FRI_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fri data/r57/snap
LOG="$PWD/data/fri/fri.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FRI_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:44.3 208:42.2 192:40.0 176:37.9 160:35.7 144:33.6 128:31.4; do   # Thu 23:4x: lower rungs (NO-SWAP 32.4 GB with apps closed; qall 31.2 / exact 31.7 GB at C 160, 0.134 per row)
 fits ${c#*:} && { CC=${c%%:*}; break; }; done
[ -z "$CC" ] && { say "NO-SWAP $A GB < 31.4: nothing fits, nothing runs"; exit 0; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_fri.py && cp scripts/r57_screen_rlx.py data/r57/snap/r57_screen_frir.py \
  && cp scripts/relax_delta.py scripts/relax_span.py data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fri.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_fri.sh"
FS=data/r57/snap/r57_screen_fri.py; RS=data/r57/snap/r57_screen_frir.py
export C45_OUT="$PWD/data/fri" C45_SWAP0=$(swap_gb) C45_ALARM=900 MG=${FRI_MG:-4}
export C45_DEADLINE=${FRI_DEADLINE:-$(date -v+45M +%H%M)}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
U="BIGMOE_EXT_BUILD=build_qom,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
S99=${FRI_SPLIT99-R57_QOM_SPLIT=0.99}; S90=${FRI_SPLIT90-R57_QOM_SPLIT=0.9}   # Breakthrough keepset 4963da6 (empty = skip)
say "Fri run start (pid $$, NO-SWAP $A GB -> C $CC, deadline $C45_DEADLINE, swap $C45_SWAP0 GB, split switches '${S99}' / '${S90}')"
R57_KEYS=humaneval/HumanEval/91,gsm8k/708 C45_CAP=64 C45_SCREEN=$RS C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" \
  C45_ARMS="frx0_$CC:$CC" $RUN
RX=$("${FRI_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$HOME/BigMoE/data/d45/ref14_tokens.json" 2>&1 <<'EOF'
import json, sys
C = sys.argv[1]
try:
    rs = json.load(open(f"data/r57/frx0_{C}_runs.json")); tk = json.load(open(f"data/r57/frx0_{C}_tokens.json"))
except OSError as e:
    print(f"FAIL: rx0 files missing ({e})"); raise SystemExit
ref = json.load(open(sys.argv[2]))
tok = lambda v: (v.get("gate+t") or [[]])[0]
d = [k for k, v in tk.items() if not tok(v) or tok(v) != ref[k]["gate+t"][0][:len(tok(v))]]
n = sum(len(tok(v)) for v in tk.values())
print(("OK" if len(rs) == 2 and n > 0 and not d else "FAIL") + f": runs {len(rs)}, {n} tokens, differ {d}")
EOF
)
say "frx0 (relaxed copy, flags off) vs REF2-T: $RX"
RLX=; [ "${RX:0:2}" = "OK" ] && RLX=1
arm() {   # arm TAG SCREEN ENV: one runner call, skipped (logged) when its switch is missing
  [ -z "$3" ] && { say "$1 skipped: no switch"; return; }
  [ "$2" = "$RS" ] && [ -z "$RLX" ] && { say "$1 skipped: the relaxed copy's flag-off check did not pass"; return; }
  R57_KEYS=$K C45_SCREEN=$2 C45_REFTOK= C45_ARMS="$1_$CC:$CC:$3" $RUN
  say "$1 rc=$?"
}
RA="$QA,R57_RELAX_DELTA=0.5,R57_RELAX_LOG=$PWD/data/fri/relax_accepts.jsonl"
SA="$QA,R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/fri/span_accepts.jsonl"
arm fu3c $FS "$U";   arm fqall $FS "$QA";   arm fs99 $FS "${S99:+$QA,$S99}";   arm fs90 $FS "${S90:+$QA,$S90}"
arm fqallr $RS "$RA"; arm fspan $RS "$SA"
arm fspanb $RS "$SA"; arm fqallrb $RS "$RA"
arm fs90b $FS "${S90:+$QA,$S90}";   arm fs99b $FS "${S99:+$QA,$S99}";   arm fqallb $FS "$QA";   arm fu3cb $FS "$U"
C45_DEADLINE=${FRI_DEADLINE2:-$(date -v+8M +%H%M)} arm fexr $FS "R57_VTIER="
"${FRI_PY:-$HOME/BigMoE/.venv/bin/python}" -I - "$CC" "$RX" >> "$LOG" 2>&1 <<'EOF'
import json, math, statistics as S, sys
C, RX = sys.argv[1], sys.argv[2]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
B = ("u3c", "qall", "s99", "s90", "qallr", "span")
runs, chk = {}, {}
for t in [f"f{b}" for b in B] + [f"f{b}b" for b in B] + ["fexr"]:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
    except OSError:
        runs[t] = {}
    try:
        chk[t] = json.load(open(f"data/r57/{t}_{C}_checks.json"))
    except OSError:
        chk[t] = []
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
def pair(b, f):   # per prompt: the mean of f over the arm's two passes (one if only one ran)
    out = {}
    for k in set(runs[f"f{b}"]) | set(runs[f"f{b}b"]):
        v = [f(runs[t][k]) for t in (f"f{b}", f"f{b}b") if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
tps = {b: pair(b, lambda r: r["tpsw"]) for b in B}
kept = {b: pair(b, lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else S.mean(r["kept"])) for b in B}
ntok = {b: pair(b, lambda r: r["n"]) for b in B}
secs = {b: pair(b, lambda r: r["n"] / r["tps"]) for b in B}
def per_check(t, f):   # mean over the run's checks of f(check row)
    xs = [x for e in chk[t] if not e.get("paused") for x in f(e)]
    return S.mean(xs) if xs else float("nan")
def grp(t):   # groups computed per check: QOM arms count the distinct base slots after QOM-KEEP's drop (qc[2], Thu 19:0x);
    q = per_check(t, lambda e: [c[2] for c in e.get("qc", []) if c[2] > 0])   # 'uses' is the loader's need.size(), counted
    return q if q == q else per_check(t, lambda e: e.get("uses", []))       # before that drop (the pre-flight's 3,739 vs 3,730)
def ratio(b, d, a="qall", agg=g):
    ks = [k for k in d[b] if k in d[a]]
    return (agg([d[b][k] / d[a][k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
L = [f"FRI palindrome, C {C}, 12 HumanEval (data/fri): u3c / qall / s99 / s90 / qallr / span, then mirrored; pair means. "
     f"frx0 (relaxed copy, flags off) vs REF2-T: {RX}"]
for b in B:
    n1, n2 = len(runs[f"f{b}"]), len(runs[f"f{b}b"])
    if not tps[b]:
        L.append(f"{b:6s} MISSING (passes {n1} / {n2})"); continue
    ug = [grp(t) for t in (f"f{b}", f"f{b}b") if chk[t]]
    stc = [per_check(t, lambda e: [c[3] for c in e.get("qc", [])]) for t in (f"f{b}", f"f{b}b") if chk[t]]
    sl = [per_check(t, lambda e: e.get("stl", [])) for t in (f"f{b}", f"f{b}b") if chk[t]]
    rx = [r.get("rlx") for t in (f"f{b}", f"f{b}b") for r in runs[t].values() if r.get("rlx")]
    sp = [r.get("spn") for t in (f"f{b}", f"f{b}b") for r in runs[t].values() if r.get("spn")]
    L.append(f"{b:6s} geo steady {g(list(tps[b].values())):6.1f} tok/s over {len(tps[b])} (passes {n1} / {n2}, first / second "
             f"{g([r['tpsw'] for k, r in runs[f'f{b}'].items() if ok(f'f{b}', k)]):.1f} / "
             f"{g([r['tpsw'] for k, r in runs[f'f{b}b'].items() if ok(f'f{b}b', k)]):.1f}); kept per check "
             f"{g(list(kept[b].values())):.2f}; tokens per task {S.median(ntok[b].values()):.0f}; s per task "
             f"{S.median(secs[b].values()):.2f}; " +
             ("groups computed" if b == "u3c" or any(c[2] > 0 for t in (f"f{b}", f"f{b}b") for e in chk[t] for c in e.get("qc", []))
              else "uses posted (pre-drop; the relaxed copy has no QOM counters)") +
             f" per check {S.mean(ug) if ug else float('nan'):.0f}" +
             (f" (strict rows per check {S.mean(stc):.1f})" if stc and S.mean(stc) > 0 else "") + "; stalled "
             f"layer calls per check {S.mean(sl) if sl else float('nan'):.1f}" +
             (f"; relaxed accepts {sum(x['relaxed'] for x in rx)}" if rx else "") +
             (f"; span: rows {sum(x['rows'] for x in sp)}, span rows {sum(x['span_rows'] for x in sp)}, relaxed "
              f"{sum(x['relaxed'] for x in sp)}" if sp else ""))
for b in ("u3c", "s99", "s90", "qallr", "span"):
    r_, n_ = ratio(b, tps); k_, _ = ratio(b, kept)
    L.append(f"{b} vs qall: tok/s {100 * (r_ - 1):+.1f} % over {n_}, kept {100 * (k_ - 1):+.1f} %")
IND = f" INDICATIVE ONLY at C {C} < 160 (the design C of the FAST path and of the replay lines is >= 160)" if int(C) < 160 else ""
for b in ("s99", "s90"):
    r_, n_ = ratio(b, tps); k_, _ = ratio(b, kept)
    v = "MISSING" if n_ == 0 else "PASS" if r_ >= 1.08 and abs(k_ - 1) <= 0.01 else "NOT PASS"
    L.append(f"Verdict split {b} (pre-registered Thu 10:40){IND}: {v} (PASS = tok/s >= +8 % vs qall AND kept within 1 %)")
s_, n_ = ratio("span", secs, agg=S.median); t_, _ = ratio("span", ntok, agg=S.median)
v = ("MISSING" if n_ == 0 else "KILL" if s_ >= 0.97 or t_ >= 1.15 else
     "PASS pending comment share (MoonShots tags tokens.json)" if s_ <= 0.92 and t_ <= 1.05 else "BETWEEN")
L.append(f"Verdict rule C (pre-registered){IND}: {v}; s per task {s_:.3f}x, tokens per task {t_:.3f}x vs qall (medians of per-prompt "
         f"ratios over {n_})")
ex = [per_check("fexr", lambda e: e.get("uses", []))] if chk["fexr"] else []
gq = {b: [grp(t) for t in (f"f{b}", f"f{b}b") if chk[t]] for b in ("qall", "s99", "s90")}
for b in ("s99", "s90"):
    st = [per_check(t, lambda e: [c[3] for c in e.get("qc", [])]) for t in (f"f{b}", f"f{b}b") if chk[t]]
    if gq[b] and gq["qall"]:
        L.append(f"split {b} engaged: strict rows per check {S.mean(st) if st else float('nan'):.1f}; groups computed "
                 f"{S.mean(gq[b]):.0f} vs qall {S.mean(gq['qall']):.0f} = {S.mean(gq[b]) / S.mean(gq['qall']):.3f}x "
                 f"(Breakthrough's replay: x0.72 / x0.63 at C 160 on full blocks, x0.67 / x0.56 live-cut)")
for b in ("qall", "s99", "s90"):
    ug = gq[b]
    if ug and ex:
        L.append(f"C3-R8 groups per check: {b} {S.mean(ug):.0f} vs exact {ex[0]:.0f} = {S.mean(ug) / ex[0]:.3f}x (audit: >= 0.85x)")
if not ex:
    L.append("C3-R8: the exact arm fexr did not run")
open("data/fri/fri_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/fri/fri_report.txt)"
say "Fri run end"
