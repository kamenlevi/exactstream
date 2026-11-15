"""FLAT-2 (zero session, Thu 5 Nov 2026; ~/FableGod/zero/LOSSY350_ZERO.md section 4; the orchestrator Thu 18:0x "build
everything so we get actual numbers"): the measurement that decides whether 350 tok/s lossy is possible at all on the
125B at 45 GB. With the split form and never-wait in place, the 350 inequality flat + e x K <= 2.857 x K has one
unknown left: FLAT, what a cycle costs before any expert row is paid (dense pass + draft + commit, no SSD wait).
What runs: BigMoE's live loop (scripts/r57_screen.py at its HEAD, every lever of the all-in FAST arm on; the env comes
from run_flat2.sh) on one code prompt, each check held to 2 rows (run()'s own draft cap, maxd 1; cut t ~0, so a check
is always [the last kept token, one draft]) for FLAT2_CAP tokens, repeated:
  pass 1          cold, discarded: kernels, pools and the drafter warm up
  FLAT2_PASSES    measured: the engine's loop untouched (the three per-check hooks are off). Before each pass the pools
                  are put back as the pass before left them (ext.ensure: FAR 4b's oracle-prewarmed form), so the checks
                  miss nothing (the prefill adopts the prompt's experts over them, so a repeated pass would start
                  cold); a cycle counts only if its check had zero misses and no stalled layer call
  stage pass      the same loop, but each commit replay is waited for: draft, check and commit one after the other
  probe pass      Leap's C0 probe at checks FLAT2_AT: warm repeats, the synced split (GDN, HC reads, attention, router +
                  shared, experts, tail, HC write, PLE, head), the row sweep 2 / 4 / 8 / 12 / 16 of the uncut draft block
  real-block      the engine's own blocks (its t cut, up to 16 rows), all-resident: one pass to load their rows, FLAT2_REAL
                  measured, one with the commit waited for. Not in the verdict: a real block's commit, and the no-wait tok/s
A cycle runs from one cycle start to the next (run() stamps them), so it holds whatever the loop does after a commit.
FLAT = median steady cycle - 2 x the row sweep's warm slope (the 2 rows' expert work; the slope is all experts: every
dense piece is flat from 2 to 16 rows, Leap's C0). Lines: PASS FLAT <= 30 ms, BETWEEN 30-36, KILL > 36 (350 needs <= 25).
Tokens are not judged here: 2-row checks change what the FAST gate keeps, and FLAT-2 measures time only.
Usage: run_flat2.sh (lock, memory ladder, swap guard, both lever arms). By hand, inside a quiet lock:
  [BigMoE venv] <the arm's env> python flat2.py C -> out/<FLAT2_TAG>_screen.txt, _flat2.json, _generated.py
Env: FLAT2_CAP 100 tokens per pass, FLAT2_PASSES 5, FLAT2_AT 8,16,24,32, FLAT2_REAL 2 passes of FLAT2_REAL_CAP 160
tokens (0 = none), FLAT2_SWEEP 2,4,8,12,16, FLAT2_KEY humaneval/HumanEval/68, FLAT2_SMOKE=1
(one short pass of each kind: does this lever set run?), FLAT2_SCREEN (the harness source), FLAT2_OUT, FLAT2_TAG."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                                    # after the engine's own paths: nothing here shadows them
import flat2_driver
import flat2_patch

assert os.environ.get("R57_PRESET") == "45", "FLAT-2 runs on the 45 GB preset: use run_flat2.sh (or set the arm's env)"
F2_CFG = flat2_driver.config()
os.makedirs(F2_CFG["out"], exist_ok=True)
os.environ.setdefault("R57_TAG", F2_CFG["tag"])
os.environ.setdefault("R57_KEYS", F2_CFG["key"])
F2_GEN, F2_INFO = flat2_patch.build(open(F2_CFG["screen"]).read())   # refuses a HEAD whose loop moved (names the line)
F2_GENPATH = os.path.join(F2_CFG["out"], f"{F2_CFG['tag']}_generated.py")
open(F2_GENPATH, "w").write(F2_GEN)
F2ON, F2H, F2C = [False], None, []                       # the hooks' switch and object, the cycle starts: run() looks them up
sys.argv = [F2_GENPATH, sys.argv[1] if len(sys.argv) > 1 else "208", str(F2_CFG["cap"])]
exec(compile(F2_GEN, F2_GENPATH, "exec"), globals())     # the harness: env preset, load, levers, run() -- as BigMoE's HEAD
sys.exit(flat2_driver.main(globals(), F2_CFG, F2_INFO))
