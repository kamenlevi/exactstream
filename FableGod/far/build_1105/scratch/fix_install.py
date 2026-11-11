import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep("to 3e-16 on a 48-layer model made of the real layer classes, and leaves every old arm bit-equal. What only the GPU can\nshow (bf16 kernels, the streamed 125B) is covered by the self-check in step 2, about 2 GPU minutes.",
    "to 3e-16 on a 48-layer model made of the real layer classes, and leaves every old arm bit-equal. What only the GPU can\nshow (the bf16 Metal kernels, the streamed 125B) is covered by the self-check in step 2, about 2 GPU minutes.")
rep("(`nw_patch.py`: anchored edits, each anchor must occur exactly once), runs the CPU self-test on them (`test_nw.py`,\n  39 checks),",
    "(`nw_patch.py`: anchored edits, each anchor must occur exactly once), runs the CPU self-test on them (`test_nw.py`,\n  44 checks),")
rep("""No separate slot? Then keep the two self-check lines in the arm list (step 3): `nwz` and `nw0` run on the 12 code
windows (about 4 more GPU minutes) and `nw_read.py` prints the same three checks before the verdicts. A failed check
makes every arm line INVALID.""",
    """No separate slot? Then keep the `nwz` line in the arm list (step 3): `nwz` and `nw0` run on the 12 code windows and
`nw_read.py` prints the same three checks before the verdicts. A failed check makes every arm line INVALID.

Keep `nw0` on the 12 windows in either case. It is also the control: it splits each late term off and writes it
straight after its own layer's write, so its D is the split write's bf16 re-rounding alone, and nw1 against nw0 is
lateness itself. In the CPU toy at bf16 that re-rounding is as large as the late terms' whole effect (2.7 % of the
stream; a random 48-layer toy is far more sensitive than the 125B) [K CPU], so the size on the real model is worth
its 2 minutes. The engine form has the same split write, so the lines stay on nw1 against the twin.""")
rep("| `qom_0_live_ball_c208_nwz+code` | `qom_0_live_ball_c208+code` | self-check (i); drop if step 2 passed |\n| `qom_0_live_ball_c208_nw0+code` | `qom_0_live_ball_c208+code` | self-check (ii); drop if step 2 passed |",
    "| `qom_0_live_ball_c208_nwz+code` | `qom_0_live_ball_c208+code` | self-check (i); drop if step 2 passed |\n| `qom_0_live_ball_c208_nw0+code` | `qom_0_live_ball_c208+code` | the control (and self-check ii): lateness 0, the split write alone; keep |")
rep("Five arms, about 10 GPU minutes; 14 with the two self-check arms. Tier 2 is built too and commented out in the file:",
    "Five arms and the control, about 12 GPU minutes; 14 with `nwz`. Tier 2 is built too and commented out in the file:")
rep("CPU, seconds, reads the map's JSONs, writes nothing. It prints the self-checks (if their arms ran), each arm against\nits own twin (flips, D in points, McNemar counts, D by the draft's cf band) and `VERDICT` lines.",
    "CPU, seconds, reads the map's JSONs, writes nothing. It prints the self-checks (if their arms ran), each arm against\nits own twin (flips, D in points, McNemar counts, D by the draft's cf band), the control (nw0 against the twin, nw1\nagainst nw0) and `VERDICT` lines.")
rep("`test_nw.py` (`test_nw.log`, 39 checks, 4 s, 230 MB, MLX's CPU device):", "`test_nw.py` (`test_nw.log`, 44 checks, 4 s, 230 MB, MLX's CPU device):")
rep("| the real `measure()` and `run_window()` end to end (fake prefill), two windows | `nwz` JSON equals the twin's; nw1's top-1 equals the reference run's; a prefill that skips rows trips both WARNINGs |",
    "| the real `measure()` and `run_window()` end to end (fake prefill), two windows | `nwz` JSON equals the twin's; nw1's top-1 equals the reference run's; a prefill that skips rows trips both WARNINGs |\n| the same test in bfloat16, the 125B's dtype (`test_nw_bf16.log`) | every exact check passes: the bit-equalities, the counts, the guards, `measure()`; the reference comparisons are notes there (8-bit mantissas over 48 layers allow no tolerance) |")
rep("Not checkable on CPU: the bf16 combine kernel and the compiled write on the late path (the same calls the map's reuse\narms already make), and the streamed model's chunking. Step 2 covers them.",
    "Not checkable on CPU: the bf16 Metal combine kernel on the late path (the same call the map's reuse arms already make)\nand the streamed model's chunking. The streamer evaluates the expert outputs before it returns them\n(`bigmoe/stream.py`: \"before the next group can evict these experts\"), so a stashed term depends only on concrete\narrays. Step 2 covers the rest.")
rep("| `test_nw.py`, `test_nw.log` | the CPU self-test and its last run |", "| `test_nw.py`, `test_nw.log`, `test_nw_bf16.log` | the CPU self-test, its last run in float64 and in bfloat16 |")
open(p, "w").write(s); print("INSTALL.md updated")
