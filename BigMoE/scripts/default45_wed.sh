#!/bin/bash
# WED_TESTS item 5, THE COMBINED EXACT DEFAULT (BigMoE = the integrator, the only writer of R57_PRESET=45; goal Thu 15:00):
# the night's exact levers that sped up their own stack, all together, A-B-A-B against today's default on W1B's six + the
# combination's 11 prompts (14 unique: code 4 / math 4 / chat 4 / agent 2), CAP 256, one C for every arm: C 224 if NO-SWAP
# >= 52 GB (44 + the volatile tier's 7.8 GB) at the start, else C 160. Runs after item 0 (vtier needs the rebuilt ext/build).
#   dfa / dfa2  R57_PRESET=45 as it is (the control; twice, because a later arm reads a warmer page cache)
#   dfb / dfb2  + the chat switch (R57_CHATSW=1: chat +21.6 % exact on DAY9's 20 prompts, +10.8 % in the combination)
#               + the SSD keepalive (BIGMOE_SSD_KEEPALIVE_US=50: S2, -3 ms per check on clean prompts)
#               + the volatile tier 64/32 (R57_VTIER=64/32: SSD MB per kept token -30 %, tok/s +3-7 %; 7.8 GB purgeable,
#                 outside phys_footprint)
#               + K-83 (K83_EXACT=1: draft ms -0.7..-1.4, +0.5..+2.4 % on 3 of 4 domains)
#               + D45_W1="<VAR=val,...>": lever 1's winner from item 2 (empty: none)
# Left out by the night's numbers (C 160): chat at 8 rows (chat -3.3 % on the switch), chat t 0.4 (KILL), A6 keep (chat
# -8 %), COPYGATE g2 / g2t (agent -21 / -9 %), D7b (math / chat down), JOINT (the traced pair: no gain), 32 rows (tokens).
# Line (pre-registered Wed 4 Nov 06:5x): ADOPT dfb's stack as R57_PRESET=45 if every token of all four arms == the stored
# REF2-T (data/d45/ref14_tokens.json = s45t_B + cmb0_160) AND the geo-mean over the 14 prompts of steady tok/s (tpsw; the
# mean of dfb / dfb2 over the mean of dfa / dfa2) >= +3 % AND no domain's geo-mean below -3 %; KEEP the default if any
# token differs or the geo-mean < +1 %; else BETWEEN (per-lever arms the next night). The preset is edited by hand after a
# look, never by this script.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), frozen copies data/r57/snap/*_d45.* -> data/d45/d45.log, d45_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/d45
LOG="$HOME/BigMoE/data/d45/d45.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
RUN="/bin/bash data/r57/snap/c45_runner_d45.sh"
export C45_SCREEN=data/r57/snap/r57_screen_d45.py C45_DEADLINE=${D45_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/d45"
[ -s data/d45/ref14_tokens.json ] || { say "no data/d45/ref14_tokens.json: STOP"; exit 2; }
.venv/bin/python -I -c "import sys; sys.path.insert(0, 'ext/build'); import bigmoe_ext as e; assert hasattr(e, 'set_vtier')" 2>> "$LOG" \
  || { say "ext/build has no set_vtier (item 0's rebuild first): STOP"; exit 2; }
a=$(noswap); CC=160; [ "$a" -ge 52 ] && CC=224
K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
K="$K,agent/bash18,agent/text23"
B="R57_CHATSW=1,BIGMOE_SSD_KEEPALIVE_US=50,R57_VTIER=64/32,K83_EXACT=1${D45_W1:+,$D45_W1}"
# Wed 07:5x: item 2 (lever 1) came out BETWEEN by its own line (exact, steady +9.0 % w1uf vs w1u at C 224, but the fixed part
# stays 19.5 ms per check, line <= 10), so D45_W1 stays empty and lever 1's full stack rides as its own arms on top of dfb:
# dfc / dfc2 = dfb + W, order A-B-C-C-B-A (every arm's mean position 3.5). Secondary line (pre-registered Wed 07:5x, before
# the run): ADD lever 1 if every dfb and dfc token == REF2-T AND dfc / dfb geo-mean >= +3 % AND no domain < -3 %; LEAVE OUT
# if a token differs or < +1 %; else BETWEEN. dfb's own verdict (dfb vs dfa) is unchanged.
W="BIGMOE_JOINT=1,BIGMOE_COPY_PAR=1,BIGMOE_SPIN_COHERENT=1,BIGMOE_POST_SYS=1"   # R57_D45=0 on every arm: since 08:4x R57_PRESET=45 IS dfc
say "combined default start (pid $$, C $CC at NO-SWAP $a GB; dfb = $B; dfc = dfb + $W)"
R57_KEYS=$K MG=${D45_MG:-4} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" \
  C45_ARMS="dfa_$CC:$CC:R57_D45=0 dfb_$CC:$CC:R57_D45=0,$B dfc_$CC:$CC:R57_D45=0,$B,$W dfc2_$CC:$CC:R57_D45=0,$B,$W dfb2_$CC:$CC:R57_D45=0,$B dfa2_$CC:$CC:R57_D45=0" $RUN
say "arms rc=$?"
B="$B" W="$W" .venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, os, sys
C = sys.argv[1]; R = "data/r57"
ref = json.load(open("data/d45/ref14_tokens.json"))
dom = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math500", "gsm8k")) else \
    "agent" if k.startswith("agent") else "chat"
runs, toks = {}, {}
for t in ("dfa", "dfb", "dfc", "dfc2", "dfb2", "dfa2"):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"{R}/{t}_{C}_runs.json"))}
        toks[t] = json.load(open(f"{R}/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], toks[t] = {}, {}
L = [f"THE COMBINED EXACT DEFAULT at C {C} (data/d45; dfb = {os.environ.get('B', 'see d45.log')}; dfc = dfb + "
     f"{os.environ.get('W', 'lever 1')})"]
bad = {t: [k for k in ref if toks[t].get(k, {}).get("gate+t") != ref[k]["gate+t"]] for t in toks}
for t, b in bad.items():
    L.append(f"tokens {t}: {len(ref) - len(b)} / {len(ref)} == REF2-T" + (f"  DIFFER/MISSING {b}" if b else ""))
# a heat pause inside a prompt (fanwatch at the fan cap; the run's "bad" says PAUSED) is not the stack's effect: such a run
# is not timed (its tokens still count); the prompt is timed on the other run of that side, or left out (Wed 07:4x: a
# 282 s pause put one u3n prompt at 0.8 tok/s, which alone would move a 14-prompt geo-mean by ~-25 %)
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
rat, per, nop = [], {}, []
for k in ref:
    a = [runs[t][k]["tpsw"] for t in ("dfa", "dfa2") if ok(t, k)]
    b = [runs[t][k]["tpsw"] for t in ("dfb", "dfb2") if ok(t, k)]
    if a and b:
        x = (sum(b) / len(b)) / (sum(a) / len(a)); rat.append(x); per.setdefault(dom(k), []).append(x)
        L.append(f"  {k}: control {sum(a) / len(a):.1f} -> stack {sum(b) / len(b):.1f} tok/s steady ({100 * (x - 1):+.1f} %)"
                 + ("" if len(a) == len(b) == 2 else f"  [{len(a)} + {len(b)} unpaused runs]"))
    else:
        nop.append(k)
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
G = g(rat); D = {d: g(v) for d, v in per.items()}
L.append("geo-mean steady tok/s: %+.1f %% over %d prompts; by domain: %s" % (100 * (G - 1), len(rat),
         ", ".join(f"{d} {100 * (v - 1):+.1f} %" for d, v in sorted(D.items()))))
if nop:
    L.append(f"not timed (every run of one side paused or missing): {nop}")
exact = not any(bad[t] for t in ("dfa", "dfb", "dfa2", "dfb2"))
v = ("KEEP the default" if not exact else f"BETWEEN (only {len(rat)} of {len(ref)} prompts timed: re-run)" if len(rat) < len(ref) - 2
     else "ADOPT" if G >= 1.03 and min(D.values()) >= 0.97 else "KEEP the default" if G < 1.01 else "BETWEEN")
L.append(f"Verdict (pre-registered): {v} (ADOPT: all tokens == REF2-T AND geo-mean >= +3 % AND no domain < -3 %; KEEP: a "
         f"token differs or < +1 %; heat-paused runs are not timed, >= {len(ref) - 2} prompts must be)")
# the secondary line (pre-registered Wed 07:5x, before the run): lever 1's full stack on top of dfb (dfc / dfc2 vs dfb / dfb2)
rat2, per2 = [], {}
for k in ref:
    b = [runs[t][k]["tpsw"] for t in ("dfb", "dfb2") if ok(t, k)]
    c = [runs[t][k]["tpsw"] for t in ("dfc", "dfc2") if ok(t, k)]
    if b and c:
        x = (sum(c) / len(c)) / (sum(b) / len(b)); rat2.append(x); per2.setdefault(dom(k), []).append(x)
G2 = g(rat2); D2 = {d: g(v_) for d, v_ in per2.items()}
L.append("lever 1 on top (dfc vs dfb): geo-mean steady tok/s %+.1f %% over %d prompts; by domain: %s" % (100 * (G2 - 1),
         len(rat2), ", ".join(f"{d} {100 * (v_ - 1):+.1f} %" for d, v_ in sorted(D2.items()))))
ex2 = not any(bad.get(t) for t in ("dfb", "dfb2", "dfc", "dfc2"))
v2 = ("LEAVE OUT" if not ex2 else f"BETWEEN (only {len(rat2)} prompts timed: re-run)" if len(rat2) < len(ref) - 2
      else "ADD lever 1" if G2 >= 1.03 and min(D2.values()) >= 0.97 else "LEAVE OUT" if G2 < 1.01 else "BETWEEN")
L.append(f"Lever 1 verdict (pre-registered 07:5x): {v2} (ADD: every dfb and dfc token == REF2-T AND dfc / dfb geo-mean >= +3 % "
         f"AND no domain < -3 %; LEAVE OUT: a token differs or < +1 %)")
open("data/d45/d45_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/d45/d45_report.txt)"
