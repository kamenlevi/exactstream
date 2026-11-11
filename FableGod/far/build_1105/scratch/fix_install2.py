p = "/Users/kamen/FableGod/far/build_1105/INSTALL.md"; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep("and `patched/dm_ablations.py` (`c3be81b8dbe0bd7e3d8e7cab4df9b303`).", "and `patched/dm_ablations.py` (`c94b21e6c410bcc15c3d121499efbe54`).")
rep("**The hook** (`neverwait.patch`, 197 lines;", "**The hook** (`neverwait.patch`, 199 lines;")
rep("**The masks** (`masks/`, 11 folders x 36 windows;", "**The masks** (`masks/`, 13 folders x 36 windows;")
rep("Every file's md5 is in `MD5SUMS` (441 files:", "Every file's md5 is in `MD5SUMS` (516 files:")
rep("""Five arms and the control, about 12 GPU minutes; 14 with `nwz`. Tier 2 is built too and commented out in the file:""",
"""Five arms and the control, about 12 GPU minutes; 14 with `nwz`.

**Tier 1b, recommended if 4 more GPU minutes exist** (commented out in the file): `qom_0_live_ball_c266+code` and
`qom_0_live_ball_c266_nw1+code`, the same pair on 266 uniform slots. Fable ZERO's reading of the engine's log
(`~/FableGod/zero/LOSSY350_ZERO.md` 5d) puts the engine's by-day base pool at the replay's 266-290 slots: its
"C 208" run holds 226 base slots plus a 64-slot victim tier, and about 3.0 of its 6.6 reads per kept token are base
records [P]. So 208 slots is about twice the engine's late-term exposure and the published pool about 70 % of it; the
266-slot pair sits on it (3.54 misses per kept token, 5.36 late terms per row) and settles a "pass only at the low
pool" outcome the same night.

Tier 2 is built too and commented out in the file:""")
rep("""| nw1 at the published pool | clean = D <= +0.15 against 0.51 % | | |
""", """| nw1 at the published pool | clean = D <= +0.15 against 0.51 % | | |
| nw1 at 266 slots (tier 1b) | nw1's lines, against its own twin | as nw1 | as nw1 |
""")
rep("""| `qom_0_live_ball_nw1` | published (`laysz45_256`) | 2.17 | 3.25 (1.97 + 1.28) | 1.14 % | 77 % | `61578850...` |
""", """| `qom_0_live_ball_nw1` | published (`laysz45_256`) | 2.17 | 3.25 (1.97 + 1.28) | 1.14 % | 77 % | `61578850...` |
| `qom_0_live_ball_c266` (tier 1b twin) | 266 | 3.54 | | | | `f5e3a0e4...` |
| `qom_0_live_ball_c266_nw1` (tier 1b) | 266 | 3.54 | 5.36 (3.22 + 2.14) | 1.88 % | 87 % | `b0f47773...` |
""")
rep("| `masks/`, `build_masks.log`, `build_masks_tier2.log` | the masks, their stats (`*.stats.json`), the builder's output |",
    "| `masks/`, `build_masks.log`, `build_masks_tier2.log`, `build_masks_c266.log` | the masks, their stats (`*.stats.json`), the builder's output |")
open(p, "w").write(s); print("INSTALL.md updated")
