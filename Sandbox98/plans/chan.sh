# CHAN: Lunatic's CODE || COMMENTARY quality pre-screen on the 2B (results/LEVERS_A.md section CHAN; orchestrator 11:5x:
# after FAR's PREFIX EYES). Smoke + identity check (J(CC)'s exact loop must reproduce MoonShots' arm0 ids on 3 keys; else
# our own coupled arm runs too), the two-process decode (HE+ / MBPP halves), the scorer + verdict (EvalSuite's scorers,
# read-only). Real decoding of the 2B.
NAME=chan
# Lock sizes = measured LIFETIME footprints: one J(CC)-style decode process 2.1-2.9 GB (MoonShots' smoke); two in parallel
# ~6 GB. mem N sets the share lock (MEMGB) and the NO-SWAP start gate (N + 8).
mem() { MEMGB=$1; GUARD_NOSWAP=$(( $1 + 8 )); }
plan() {
  mem 4
  step smoke logs/chan_smoke.log 10 $T -l bash -c "$PY -I -B sb/chan.py --mode plain --keys 0:3 --tag smoke && $PY -I -B sb/chan.py --mode chan --keys 0:3 --tag smoke"
  if [ "$DRY" != 1 ] && [ ! -f logs/.done/$RUN/idcheck ]; then
    if $PY -I -B -c 'import json, sys
a = {r["key"]: r for r in map(json.loads, open("/Users/kamen/MoonShots/week98/jcc/arm0.jsonl"))}
p = [json.loads(l) for l in open("results/chan/plain_smoke.jsonl")]
c = [json.loads(l) for l in open("results/chan/chan_smoke.jsonl")]
same = [r["ids"] == a[r["key"]]["ids"] and r["checks"] == a[r["key"]]["checks"] for r in p]
print("IDCHECK", "SAME" if p and all(same) else "DIFFER", "| plain vs arm0 ids + checks per key:", same,
      "| chan smoke: slots", [len(r["slots"]) for r in c], "code-only tokens", [r["n"] for r in c],
      "| spliced parses:", [r["spliced"].count("```") for r in c])
sys.exit(0 if p and all(same) else 3)' > logs/chan_idcheck.log 2>&1; then echo arm0 > logs/chan_coupled; else echo plain > logs/chan_coupled; fi
    say "$(tail -1 logs/chan_idcheck.log | cut -c1-250)"
    touch logs/.done/$RUN/idcheck
  fi
  COUPLED=$(cat logs/chan_coupled 2>/dev/null || echo arm0)
  mem 7
  step decode logs/chan_decode.log 45 $T -l bash scripts/chan_par.sh $COUPLED
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $HOME/EvalSuite/.venv/bin/python -I -B sb/chan_score.py --coupled $COUPLED > logs/chan_score.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/chan_score.log | cut -c1-120 | tr '\n' ' ' | cut -c1-250)
}
