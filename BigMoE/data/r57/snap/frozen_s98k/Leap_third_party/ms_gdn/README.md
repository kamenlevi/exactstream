Frozen copy (Leap, Sun 2026-11-01) of MoonShots' exact fused GDN pieces for REF2's CFG gdn_exact (R-61a):
gdn_exact.py, gdn_exact_patch.py and tierglu.py (gdn_exact imports _HDR from it), copied from
~/MoonShots/patches/planes_engine at MoonShots commit 83109c8. Read-only reuse; do not edit here. REF2 imports
these copies, so a later change in MoonShots' tree cannot change REF2's bits. Bitwise == MTPLX's eager GDN chain:
MoonShots S 1-8 (+ model-level 4-row logits), Leap S 1-16 (scripts/ms_copy tests) and the live loop (R-61a, tokens
equal across runs at C 224).
