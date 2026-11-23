# Staged release edits (not applied)

Each file here is a change to release/src that waits for a decision. Apply it, rebuild docs-only with make_release.py,
re-run the diff and leak checks, and record it.

- `limits_pfd_erratum.patch` (LIMITS.md lines 12-13). Prepared Tue 10 Nov (orchestrator 10:4x). Apply only if Kamen
  approves the papers' erratum.
  - Brings in BigMoE's long-prompt PF-D check (BM:data/pfd_long, 926b727). On 10K and 30K-token prompts PF-D is exact:
    tokens 64/64, logits hashes 25/25 and 27/27, 705,420 rows checked with 0 differing.
  - Its first-token gain there is only 3-4 % (78.9 -> 76.6 s, 249.4 -> 238.4 s). The shipped `chat` (with PF-D) peaks at
    47.5 GB on 10K tokens and 48.6 GB on 30K (45.2 / 46.7 GB with PF-D off).
  - Apply with `cd release/src && patch -p1 -F0 < ../staged/limits_pfd_erratum.patch`.

- `v110_notes.md`: the FINAL v1.1.0 release notes (body; release title "ExactStream 1.1.0"), as the orchestrator gave
  them at 05:3x after review (its four wording changes over the Leap draft). Earlier: draft v1.1.0 release notes (orchestrator 05:2x: what is new = the 24 GB setting, chat.py and
  serve.py, the paper; what is corrected = the PF-A short-chunk qualifier; the plain style of v101_notes.md). Every number
  from pkg_110j's README / INSTALL.md / LIMITS.md / RESULTS.md and data/pfa_tail_1111.md, data/idle_ms_1111.md. For the
  orchestrator's review; the release text is the orchestrator's to publish.
