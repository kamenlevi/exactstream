#!/bin/bash
# R-40 step 0 (docs/PREREG_ref2_2026-10-28.md): today's stock harness (ref.r1 = ref's settings) decodes the first 2 rows
# of each core set; all 6 must be token-identical to results/ref for results/ref to stay the baseline. One queued quiet
# lock (cli.gpu). Detached; log results/ident_ref.log. Stop it by pid only.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/ident_ref.log
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
while pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null || [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 30; done
say "identity check start (pid $$)"
./bin/evalsuite gpu decode ref.r1 --sets math,ifeval,humaneval --limit 2 --minutes 16 >> "$LOG" 2>&1
say "decode rc=$? ($(wc -l < results/ref.r1/decode.jsonl 2>/dev/null | tr -d ' ') rows)"
say "same tokens: $(.venv/bin/python tools/same_tokens.py ref.r1 ref 2>&1 | tail -3 | tr '\n' ' ')"
say "identity check end"
