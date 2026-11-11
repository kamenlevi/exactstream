import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep('''    a = pair(NW1, T208); p = pair(NWP, TP); r = pair(R1, T208); s = pair(STK, TS)
    if s is not None: pair(STK, TP, quiet=True)
''', '''    a = pair(NW1, T208); p = pair(NWP, TP); r = pair(R1, T208); s = pair(STK, TS)
    if s is not None: pair(STK, TP, quiet=True)
    c0 = pair(NW0, T208, quiet=True)                             # the control: the split write alone (lateness 0)
    if c0 is not None and a is not None:
        print("  control: nw0 writes each late term straight after its own layer's write, so its D is the split write's bf16 "
              "re-rounding alone; nw1 against nw0 is lateness itself:")
        pair(NW1, NW0, quiet=True)
''')
rep('''    if s is not None: V.append(f"the stack (split 0.99 + nw1, published pool): {'PASS' if s[0] <= 1.34 else 'FAIL'}: {s[0]:.2f} % (PASS <= 1.34), D {s[2]:+.2f} vs split 0.99")
''', '''    if s is not None: V.append(f"the stack (split 0.99 + nw1, published pool): {'PASS' if s[0] <= 1.34 else 'FAIL'}: {s[0]:.2f} % (PASS <= 1.34), D {s[2]:+.2f} vs split 0.99")
    if c0 is not None and a is not None:
        V.append(f"control (not a line): of nw1's D {a[2]:+.2f}, the split write alone (nw0) gives {c0[2]:+.2f}; lateness itself {a[2] - c0[2]:+.2f}")
''')
rep('''Arms: for each arm against its own twin,''', '''Control: nw0 (if it ran on the 12 windows) against the twin is the split write's bf16 re-rounding alone, and nw1 against
nw0 is lateness itself; the engine form has the split write too, so the lines stay on nw1 against the twin.
Arms: for each arm against its own twin,''')
open(p, "w").write(s); print("ok")
