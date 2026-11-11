#!/bin/bash
# Fable's mission 20 engine test (~/FableGod/CACHE.md 5; the orchestrator 11-02 19:4x: "one exact engine test for you, after
# A6"): PREFILL_WARM (r57_screen R57_A6=warm: each layer's prompt experts into its pools by frequency at prefill end, the
# most used = the most recent, up to the pool size) vs the control (R57_A6=ctl: nothing changed, the first checks logged)
# at C 160 and C 256, two separate passes per arm (c-w-c-w), 3 code + 3 agent prompts, CAP 64. Line (on C 256): misses in
# the first three checks <= -8 % with every token equal to REF2-T = PASS, > -3 % = KILL. Tokens: chat23's REF2-T reference
# for 5 of the 6 keys; HumanEval/16 has none stored, so its runs are checked against each other (scripts/pw_report.py).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies of the runner and the harness (data/r57/snap/*_pw.*)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/pw
LOG="$HOME/BigMoE/data/pw/pw.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_pw.sh"
export C45_SCREEN=data/r57/snap/r57_screen_pw.py C45_DEADLINE=${PW_DEADLINE:-1430} C45_OUT="$HOME/BigMoE/data/pw"
# PW_CS (11-03, the orchestrator: idle NO-SWAP is only 45-47 GB, so turns must peak <= ~41 GB): the Cs to run, e.g. "160 224";
# the report's line moves to the largest C that ran (pre-registered C 256 -> C 224, set before the run)
KEYS="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,agent/bash18,agent/text23,agent/bash17"
say "mission 20 start (pid $$, HAVE_LOCK=$HAVE_LOCK, NO-SWAP $(vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}') GB)"
A=""
for C in ${PW_CS:-160 256}; do
  A="$A pwc_$C:$C:R57_A6=ctl,R57_REFTOK= pww_$C:$C:R57_A6=warm,R57_REFTOK= pwc_${C}r:$C:R57_A6=ctl,R57_REFTOK= pww_${C}r:$C:R57_A6=warm,R57_REFTOK="
done
R57_KEYS=$KEYS MG=${PW_MG:-7} C45_CAP=64 C45_ALARM=480 C45_ARMS="${A# }" $RUN
say "arms done rc=$?"
PW_CS="${PW_CS:-160 256}" "$HOME/BigMoE/.venv/bin/python" scripts/pw_report.py >> "$LOG" 2>&1; say "report rc=$? (data/pw/pw_report.txt)"
say "mission 20 end"
