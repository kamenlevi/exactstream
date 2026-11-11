#!/bin/bash
# PR2 (Wed 4 Nov 05:5x): the pre-route re-run with the FIXED hyper-connection read. MTP Head (ojox-4a, Wed 05:3x) found that
# the pr_160 night's full / p3 / p6 scored L+1's router on L+1's ATTENTION-side read: qwen4_exp has no layernorms, each
# block has its own hc_norm + read mix per side, and the router sees the MLP-side read. r57_screen now reads the MLP side
# ("full" at L+1's start, PLE included; p3 / p6 on L's partial outputs) and keeps the old read as "fullA", the control.
# Same 18 prompts, CAP 128 and C 160 as pr_160; every token vs pr_160's tokens (exact: chat == chat23 REF2-T, HumanEval ==
# u3c_160). + R57_PR_DUMP=1: MTP Head's SAMELAYER-T rows ("full"'s input X and L+1's true MoE input Y per row and layer,
# float32 x 2560, ~2.5 GB -> data/r57/pr_dump/pr2_160_NN.npz).
# Lines (pre-registered 11-02, unchanged): PASS p3 >= 0.85 AND p6 >= 0.90 recall@10 per row (all prompts pooled); KILL
# p3 < 0.75; else BETWEEN. Control: fullA within 0.01 of pr_160's full (all 0.565; code 0.475, chat 0.602), else the two
# runs are not comparable: look first. MTP Head's prediction: full code 0.70-0.76, chat 0.78-0.84, p3 within 0.01 of full,
# all 0.76-0.81 (= BETWEEN). No timing is read: a shared slot does (pr_160: 139 s, lifetime peak 29.1 GB at C 160).
# usage: bash scripts/pr2_run.sh   (CLAUDE_PID set; inside `peer lock share 15m --cpu 200 --gpu 60 "... MEM 30 GB"`)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/pr2
LOG="$HOME/BigMoE/data/pr2/pr2.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
[ -e data/r57/pr2_160_preroute.json ] && { say "data/r57/pr2_160_preroute.json exists: look first"; exit 2; }
DISK=$(df -g "$HOME/BigMoE" | awk 'NR==2 {print $4}')
[ "$DISK" -ge 115 ] || { say "free disk $DISK GB < 115: STOP (the dump writes ~2.5 GB)"; exit 2; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_pr2.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_pr2.sh || exit 1
say "pr2 start (pid $$; frozen copies r57_screen_pr2.py / c45_runner_pr2.sh; disk $DISK GB free)"
export C45_SCREEN=data/r57/snap/r57_screen_pr2.py C45_DEADLINE=${PR2_DEADLINE:-2330} C45_OUT="$HOME/BigMoE/data/pr2"
PRK="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
PRK="$PRK,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
PRK="$PRK,humaneval/HumanEval/101,humaneval/HumanEval/63,ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/de3,multi/es10"
R57_KEYS=$PRK MG=${PR2_MG:-3} C45_CAP=128 C45_ALARM=600 C45_REFTOK="$HOME/BigMoE/data/r57/pr_160_tokens.json:gate+t" \
  C45_ARMS="pr2_160:160:R57_PREROUTE=1,R57_PR_DUMP=1" /bin/bash data/r57/snap/c45_runner_pr2.sh
say "runner rc=$? ($(grep -c . data/pr2/pr2_160.txt 2>/dev/null) screen lines; dump $(du -sh data/r57/pr_dump 2>/dev/null | cut -f1))"
grep -h "every token equal" data/pr2/pr2_160.txt 2>/dev/null | tail -1 | tee -a "$LOG"
.venv/bin/python scripts/pr_report.py pr2_160 >> "$LOG" 2>&1; say "report rc=$? (data/r57/pr2_160_preroute.txt)"
tail -4 data/r57/pr2_160_preroute.txt 2>/dev/null | tee -a "$LOG"
