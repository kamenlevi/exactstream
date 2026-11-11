"""Apply after runner 41 (T0) stops, while no pinned runner lives (its pin covers evalsuite/):
  python3 pending/apply_drift_e8.py
For RELAX's Friday checks (docs/PREREG_relax_drift_2026-11-05.md: the long-code drift re-check, then E8), after
Thursday's records3 conversion:
1. evalsuite/relax.py gains the drift sets: humanevalx (HumanEval+'s other 104) and agentlong (dec_agent's 17 write /
   edit turns at 1024 tokens).
   - Arm B gets them with its HumanEval+ / agent settings; the control arm takes every set (OFF) by construction.
   - Arm E8 = E + HC8 on chat (HC8 is process-global as built: NewIdeas Thu 08:5x; the orchestrator confirmed E8).
2. relax.py and replay.py take the REF2-T variant from EVALSUITE_RELAX_VARIANT (default ref2t, the night's: snap/leap-
   faff8e5, a row-major reader).
   - On the converted store: ref2tf0 (REF2-T on snap/leap-52cf2c9 with every switch off; 3c-N on native layers, bitwise
     == 3c by the records3 check 2) with EVALSUITE_BIGMOE_CODE=snap/bigmoe-da8e7a6.
   - Every RELAX process's step 0 (HumanEval/27 == results/ref2t or ref2th8) bridges the new code to the night's
     numbers.
3. The gate's ext: build_next (BigMoE's Oct 30 gate build), or ext/build on a snapshot after Nov 4 (MAXK is runtime;
   BigMoE 11-04 07:55), as live16b.
replay.py imports SETS and ARMS from relax.py, so it follows (1)."""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evalsuite")
GATE_OLD = '''    assert os.environ.get("BIGMOE_MAXK") == "160" and os.environ.get("BIGMOE_EXT_BUILD") == "build_next"'''
GATE_NEW = '''    ext_ok = (os.environ.get("BIGMOE_EXT_BUILD") == "build_next"     # the Oct 30 gate build; since BigMoE's Nov 4 build
              or (os.path.basename(env.BIGMOE_CODE.rstrip("/")) != "bigmoe-6c64148"   # ext/build serves MAXK 160 too
                  and os.environ.get("BIGMOE_EXT_BUILD") in (None, "build")))           # (MAXK is runtime)
    assert os.environ.get("BIGMOE_MAXK") == "160" and ext_ok'''
VAR_OLD = '    rv = EM.variant("ref2t")\n'
VAR_NEW = '    rv = EM.variant(os.environ.get("EVALSUITE_RELAX_VARIANT", "ref2t"))   # ref2tf0 on the converted store (Fri)\n'


def patch(name, pairs):
    p = os.path.join(D, name)
    s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:70])
        s = s.replace(old, new)
    open(p, "w").write(s)


patch("relax.py", [
    ('SETS = ("math", "ifeval", "humaneval", "multi", "agent")',
     'SETS = ("math", "ifeval", "humaneval", "multi", "agent", "humanevalx", "agentlong")   # + the drift re-check\'s (Fri)'),
    ('''    "B": dict(hc8=True, sets=dict(humaneval=S_(**G99, loop=R2), math=S_(**G99), agent=S_(**G99))),    # the mode''',
     '''    "B": dict(hc8=True, sets=dict(humaneval=S_(**G99, loop=R2), math=S_(**G99), agent=S_(**G99),       # the mode
                                  humanevalx=S_(**G99, loop=R2), agentlong=S_(**G99))),   # PREREG_relax_drift'''),
    ('''    "E": dict(hc8=False, sets=dict(ifeval=S_(rules=[R1]), multi=S_(rules=[R1]))),                     # chat R1''',
     '''    "E": dict(hc8=False, sets=dict(ifeval=S_(rules=[R1]), multi=S_(rules=[R1]))),                     # chat R1
    "E8": dict(hc8=True, sets=dict(ifeval=S_(rules=[R1]), multi=S_(rules=[R1]))),                     # E + HC8 (Fri)'''),
    ('''    assert os.environ.get("BIGMOE_MAXK") == "160" and os.environ.get("BIGMOE_EXT_BUILD") == "build_next", \\
        "RELAX runs on the gate: BIGMOE_MAXK=160 and BIGMOE_EXT_BUILD=build_next"''',
     GATE_NEW + ''', \\
        "RELAX runs on the gate: BIGMOE_MAXK=160 and the gate's ext (build_next, or ext/build after Nov 4)"'''),
    (VAR_OLD, VAR_NEW),
])
patch("replay.py", [(GATE_OLD, GATE_NEW), (VAR_OLD, VAR_NEW)])
print("evalsuite/relax.py: + humanevalx / agentlong (B), arm E8, EVALSUITE_RELAX_VARIANT, the gate-ext rule; "
      "evalsuite/replay.py: EVALSUITE_RELAX_VARIANT, the gate-ext rule")
