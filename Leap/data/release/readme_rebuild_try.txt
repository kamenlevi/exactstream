ENGINE PACKAGE: trial README-only rebuild, Mon 9 Nov 2026 12:2x (CPU; the tested package 242553d and release/out/pkg untouched)

WHY: the README's results / limits / labels are filled after the clean-clone PASS (final_1210.txt), so the shipped package
will be a rebuild. This checks now, with the drafted README (Leap de0b1c9), that a rebuild changes nothing but the README.
COMMAND: python3.12 release/make_release.py rel1 --commit 0b4df8c --refs release/refs_rel1.json --out <scratchpad>/pkg_try
  -> "PACKAGE rel1: 225 files, 6.3 MB; BigMoE 0b4df8c3, EvalSuite 15e1073a; 43 modules in the manifest union, 0 not in this
  snapshot []; 43 files path-rewritten, 1 scrubbed, 4 quotes redacted"; leaks 0; unscrubbed quote lines 0; home_paths_left
  43; live research paths 0; syntax errors 0; COMMITTED 56bf1bc (rc 0).
DIFF vs the tested package (git fetch release/cleanclone/final/clone HEAD = 242553d; git diff --stat 242553d 56bf1bc):
  MANIFEST.json | 78 +++---, README.md | 31 +++--; 2 files changed.
  MANIFEST (parsed): top-level keys changed = leap_commit, evalsuite_commit only; files whose sha256 changed = README.md
  only; files added / removed: none. The EvalSuite files' "source" labels name 15e1073a instead of 98c6072d with the same
  sha256 (git diff --stat 98c6072d 15e1073a -- evalsuite: empty).
VERDICT: PASS. A README-only rebuild keeps every tested file byte-identical. Re-run the same check on the real rebuild after
the 16:00 labels (EvalSuite's evalsuite/ and the untracked item files must still be unchanged then).
