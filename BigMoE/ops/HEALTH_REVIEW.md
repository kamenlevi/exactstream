# 3-hourly laptop + research health review (BigMoE's standing duty since 2026-10-25, from the orchestrator)

Run `taskpolicy -b python3 ~/BigMoE/ops/health_review.py` (read-only facts), then judge:

1. **HEALTH.md verdict** OK / WATCH / ACT, health_alerts.log since the last review, fanwatch events (HOT / SAFETY / WARM)
   and fan swings > 500 rpm within a minute: signs alternating up/down = oscillation (Kamen can't stand switching);
   a one-way ramp to a new curve, or a controller restart's 0 rpm, is not.
   **Fan controller alive** (facts line "fan controller:"): fanwatch_ctl.py running, fanwatch.log written in the last
   ~2 min, Baro mode fixed, no "did not follow fixed mode" fallback since its last start. If it is DEAD (10-25 19:27-22:13
   it was: fans pinned at the last fixed RPM under full GPU load) and no quiet window is running (no quiet_window.sh /
   quiet_auto.sh / night_silent.sh process: those stop fanwatch on purpose and restart it at their end), restart it
   yourself (orchestrator's OK, 10-25 22:15), detached and by FULL path so the quiet windows' `pkill -f quiet/fanwatch.sh`
   still matches it: `( nohup /bin/zsh $HOME/.local/share/quiet/fanwatch.sh > /dev/null 2>&1 & )`; after ~60 s check that
   "/Library/Application Support/Baro/helper-status.json" shows mode fixed for every fan; log it and tell the orchestrator
   with `peer send`. A live controller in fallback or started another way: tell the orchestrator, don't restart it.
2. **Swap:** an in-RAM run must stop if swap grew > 2 GB (tell its owner). **Disk:** stop big writes below 110 GB free,
   floor 100.
3. **Runaway / leftover processes:** a research Python at high CPU whose session is gone; stopped (T) leftovers of
   killed runners (kill only your own; tell the owner otherwise). **CPU-only jobs above 4 cores (400 %).**
4. **GPU queue (`peer ls`):** a frozen waiter at the head, a quiet lock held with nothing on the GPU while others wait
   (RULES.md "CPU jobs vs the GPU queue"), a lock held by a session that is gone.
5. Fix what is clearly inside ~/.local/share/pacer/RULES.md; tell the owner with `peer send <name> "..."`.
6. Append ONE line to ~/.local/share/quiet/health_reviews.log:
   `YYYY-MM-DD HH:MM EET <OK|WATCH|ACT>: temps, fans, swap, disk, monitor alive; research/GPU queue; actions.`
   (run `date` first; the facts script prints the time)
7. Message the orchestrator ONLY for an ACT verdict, a hardware risk, or a decision only Kamen can make. Everything else
   stays in the log. Its address changes when the Claude app restarts (20139.sock died 10-25 10:25): SendMessage to the
   address `~/.local/share/pacer/orch_addr.sh` prints (looked up by session id; 42262 died at the 10-30 00:40
   restart, then 7333). Two main sessions are named "AI/LLM project recommendations" (7366 is NOT it), so `peer show` /
   `peer send` by that name can reach the wrong one; owner notes that aren't urgent go to that same uds address.

Pacing: `pacer check --session bigmoe --account sub`.
