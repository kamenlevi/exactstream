"""Leap, Sat 10-24 (RULES.md, orchestrator 04:00): a quiet window creates ~/.local/share/quiet/PAUSE and sends SIGSTOP
1 s later; a stop that lands mid-pass trips BigMoE's spin gates ("spin gate(s) timed out"; L4's generation died so at
03:32). Importing this module makes every BigMoE pass wait at its boundary while PAUSE exists, as the RULES.md snippet
does inside gated.py's new_pass(). Patched from outside, so BigMoE's engine stays read-only. Costs one stat per pass and
changes no output."""
import os, time
import bigmoe.gated as _gated

_PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
_new_pass = _gated.Gate.new_pass

def new_pass(self):
    while os.path.exists(_PAUSE):
        time.sleep(0.2)
    return _new_pass(self)

_gated.Gate.new_pass = new_pass
