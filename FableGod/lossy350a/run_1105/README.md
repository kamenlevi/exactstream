# run_1105: T1 on the map from a patched copy (Fable A, Thu 5 Nov 2026)

**Not started. The GPU part waits for Kamen's own yes** (this session's standing rule is no GPU; the orchestrator asked for
the run at 18:5x, and Fable C holds its own run the same way).

- What is here: `run_map.py` (the map's file), `dm_common.py`, `dm_ablations.py`, `dm_xs.py` (the map's files with the T1
  patch of `../build_1105`), `run_t1.sh` (the guarded runner), `wiring_check.py`, `decide.py`, `out/`, `out_chat/`.
- The shared `~/FableGod/decision_map` is not touched; the map process reads its other modules, masks and baseline there.
- To start, after the yes:
  ```bash
  nohup bash ~/FableGod/lossy350a/run_1105/run_t1.sh > /dev/null 2>&1 &
  ```
  It waits until: no map process, no other session's shared run over 5 GB or quiet window, Fable C's batch gone (or
  21:44), NO-SWAP >= 24 GB, lid open, no PAUSE, no STOP. Last start of the code arms 21:45; the chat arm ends by 22:15.
  Swap growing by more than 1 GB stops the run. Share lock "MEM 18 GB". Log: `run.log`; it writes `DONE` at the end.
- Order: the wiring check (CPU), `xs64+code xs48+code xs40+code xs32+code` in one process, `decide.py` + `read_xs.py`,
  then `xs48` on IFEval + multi if the identity arm is clean, no window carries a warning, and xs48's non-tie code flips
  are within 0.66 %.
- Dry check of the conditions, nothing started: `bash run_t1.sh dry`.
