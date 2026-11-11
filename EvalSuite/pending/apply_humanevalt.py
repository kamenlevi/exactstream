"""Apply after runner 47 (the agent check) ends, while no pinned runner lives (its pin covers evalsuite/):
  python3 pending/apply_humanevalt.py
PREREG_relax_drift amendment 2 (the code-only top-up): relax.py's set list gains humanevalt (41 HumanEval+ problems asking
for the function + a pytest module, max_tokens 1536; tools/drift_build.py buildt), and arm B takes it with its HumanEval+
settings (Q1-UNION cf 0.9 + copy + the comment rule). replay.py imports SETS and ARMS from relax.py, so it follows."""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evalsuite")
p = os.path.join(D, "relax.py")
s = open(p).read()
for old, new in (
    ('SETS = ("math", "ifeval", "humaneval", "multi", "agent", "humanevalx", "agentlong")   # + the drift re-check\'s (Fri)',
     'SETS = ("math", "ifeval", "humaneval", "multi", "agent", "humanevalx", "agentlong",   # + the drift re-check\'s (Fri)\n'
     '        "humanevalt")                                                               # + its code-only top-up (amend. 2)'),
    ('''                                  humanevalx=S_(**G99, loop=R2), agentlong=S_(**G99))),   # PREREG_relax_drift''',
     '''                                  humanevalx=S_(**G99, loop=R2), agentlong=S_(**G99),     # PREREG_relax_drift
                                  humanevalt=S_(**G99, loop=R2))),                        # its amendment 2'''),
):
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w").write(s)
print("evalsuite/relax.py: + humanevalt (B, HumanEval+ settings)")
