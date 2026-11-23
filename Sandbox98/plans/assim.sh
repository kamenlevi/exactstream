# Drift screen 1 (NewIdeas 18:5x): ASSIM, carried-state correction from the checks' accepted rows (results/LEVERS_A.md,
# section ASSIM). Smoke (3 windows per set, the no-op arm and the SIB cells' counts must match), then the full cells.
NAME=assim
plan() {
  step smoke logs/assim_smoke.log 10 $T -l $PY sb/assim.py --rows sib --n 3 --smoke --out results/assim_smoke.json
  if [ "$DRY" != 1 ]; then
    $PY -I -c 'import json, sys
v = json.load(open("results/assim_smoke.json"))
bad = [(s, k) for s, r in v["sets"].items() for k, ok in r["voids"].items() if not ok]
miss = [s for s, r in v["sets"].items() if len(r["voids"]) < 2]
print("SMOKECHECK", "FAIL" if bad or miss else "OK", {s: r["voids"] for s, r in v["sets"].items()})
sys.exit(1 if bad or miss else 0)' > logs/assim_smokecheck.log 2>&1 || { STATUS="smoke check failed: $(tail -1 logs/assim_smokecheck.log | cut -c1-250)"; exit 5; }
    say "$(tail -1 logs/assim_smokecheck.log)"
  fi
  step main logs/assim_main.log 30 $T -l $PY sb/assim.py --rows sib --out results/assim_sib.json
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  VERDICT=$(grep -E "^VERDICT" logs/assim_main.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
