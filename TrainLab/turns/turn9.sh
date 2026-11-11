#!/bin/sh
# TrainLab turn 9 (10-22): score the natively trained 125B MTP head (mtp125n_d4, 131K tokens).
# (1) Chat1000's 12 held-out agentic turns, his protocol (agent125_tl.py = his agent125.py with the tuned head, fp32
#     projections as trained) and his analyzer; (2) NewIdeas 02:30: general text too: simulated passes on the 125B's own
#     greedy continuations (EvalSuite tf_gen, round-robin over GSM8K / IFEval / HumanEval / multilingual), one trunk
#     prefill shared by stock / tuned fp32 / tuned stored as the stock head stores it (4-bit) / tuned in bf16.
# Keep line: >= +15 % tau(16) on the 12 agentic turns with no class worse, and no tf_gen set worse (within noise).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
G="grep --line-buffered -v moe-sorted-gather"
echo "== $(date '+%m-%d %H:%M:%S') turn 9 start (limit $TL_LIMIT s)"
START=$(date +%s)
TL_MTP_RUN=mtp125n_d4 $PY agent125_tl.py 0.25 160 2>&1 | $F | $G
AGENT_DIR=$T/data/agent125_mtp125n_d4 $PY ~/Chat1000/scripts/agent_analyze.py 2>&1 | tee logs/agent_analyze_mtp125n_d4.out | tail -40
USED=$(( $(date +%s) - START ))
B=$(( TL_LIMIT - USED - 330 ))
echo "== sim125 budget $B s"
[ $B -gt 240 ] && $PY sim125.py --runs stock,mtp125n_d4,mtp125n_d4@q,mtp125n_d4@bf16 --K 15 --per-set 30 --budget $B 2>&1 | $F | $G
echo "== $(date '+%m-%d %H:%M:%S') turn 9 end"
