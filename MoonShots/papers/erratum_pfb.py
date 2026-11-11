"""PF-B erratum (Leap 10 Nov 00:3x): the published papers call PF-B / "the batched prefill" exact with no length limit;
its exactness evidence is three agent prompts of 1.6K-7.9K tokens, and on a 30K-token prompt (32 GB chat configuration,
LP:data/prefill_1109.md) its decoded tokens differed from the default prefill's from the 43rd token on.

Branches (the PF-B part):
  none  : no PF-B change (Leap's 30K hash check passes at the published setting); only the PF-D / memory part below.
  scope : only limit "exact" to the tested prompts (true whatever Leap's logits-hash check shows; no new source key).
  fail  : scope + the 30K result (needs Leap's check: pass --hash / --hashrow; needs Leap/data/prefill_1109.md and
          Leap/data/prefill_1109/ in the 350-tps snapshot, or keycheck fails).
Every branch also applies the PF-D / memory part (BigMoE 10 Nov 00:5x, BM:data/pfd_long/summary.txt at 926b727):
RELEASE section 5's PF-D sentence was wrong (PF-D also runs on long prompts, once per 1,024-token chunk, and stays
exact), and the shipped configurations go over their memory lines on 10K / 30K-token prompts (45 GB chat 47.46 / 48.56,
45.17 / 46.65 without PF-D; 32 GB chat 34.01 / 35.48, LP:data/prefill_1109.md), which MAIN's abstract and 6.1 did not
say. Needs BigMoE/data/pfd_long/ in the 350-tps snapshot.
Usage: python3 -I erratum_pfb.py none|scope|fail [--hash "<clause ending in ', and '>"] [--hashrow "<cell text>"]
       [--notes] [--out DIR] [--write]   (--notes: a dated '## Corrections' section in each changed paper)
"""
import sys, pathlib, time

P = pathlib.Path.home() / "Papers"
args = sys.argv[1:]
branch = args[0] if args and args[0] in ("none", "scope", "fail") else sys.exit(__doc__)
def opt(name, default=""):
    return args[args.index(name) + 1] if name in args else default
# fail texts carry Leap's exact-32 check (10 Nov 00:57-01:01; LP:data/prefill_1109/exact30k/summary.txt, re-derived from
# the run records Leap/release/cleanclone/final/clone/out/cli_1110_0057{16},0058{38},0059{48}: hashes 0 / 25 at TOK 8192
# and 4096, first difference at check 1; tokens equal through the 39th). Not 'one call per layer': the published TOK 2048
# arm (BM pfd_report a6d5347 part B, pdb32s: 3-4 calls per layer at 4.7K / 7.9K) kept hashes 8/8, 28/28, 22/22.
FAIL = branch == "fail"

def scoped(base, extra):
    return base + (extra if FAIL else "")

R = {"MAIN.md": [], "KERNELS.md": [], "RELEASE.md": []}

# MAIN 6.2, last bullet
R["MAIN.md"].append((
    "is exact since its gate fix: tokens and logits hashes equal the layer-major prefill's (8 / 8,\n"
    "  28 / 28, 22 / 22) at x1.12-1.21 its speed (500 tok/s at 7.9K), but it peaks at 45.8-46.2 GB, over the line, so it\n"
    "  does not ship [K, BM:data/pfd/pfd.log part B; [KERNELS] section 5.2].",
    scoped(
    "is exact on the three agent prompts (1.6K-7.9K tokens) since its gate fix: tokens and logits hashes\n"
    "  equal the layer-major prefill's (8 / 8, 28 / 28, 22 / 22) at x1.12-1.21 its speed (500 tok/s at 7.9K), but it peaks\n"
    "  at 45.8-46.2 GB, over the line, so it does not ship [K, BM:data/pfd/pfd.log part B; [KERNELS] section 5.2].",
    "\n  It is not exact on a 30K-token prompt: on the 32 GB exact configuration its logits hashes equal the layer-major\n"
    "  prefill's on 0 of 25 checks, both at 8,192 rows per call (the published setting) and at 4,096, and its tokens differ\n"
    "  from the 40th on, while the layer-major prefill's hashes equal the default prefill's on 26 of 26 [K,\n"
    "  LP:data/prefill_1109/exact30k/summary.txt]. PF-B's exactness is shown only up to 7.9K tokens, and the cause of the\n"
    "  difference is not yet found.")))

# MAIN 10, PF-B bullet
R["MAIN.md"].append((
    "- **One MoE call per layer in prefill (PF-B) was not exact at first** (0 of 7-28 logits hashes equal, +15-22 %) [K,\n"
    "  BM:data/pfb/pfb_report.txt]: the shared expert's one-output gate splits its sums by row count ([KERNELS] section 5.2).\n"
    "  Kept per chunk, that gate makes PF-B exact (8 / 8, 28 / 28, 22 / 22 logits hashes, x1.12-1.21), but it peaks at\n"
    "  45.8-46.2 GB, over the line, and does not ship [K, BM:data/pfd/pfd.log part B].",
    ("- **One MoE call per layer in prefill (PF-B) was not exact at first, and is not exact on a 30K-token prompt** (0 of\n"
     "  7-28 logits hashes equal at first, +15-22 %) [K, BM:data/pfb/pfb_report.txt]: the shared expert's one-output gate\n"
     "  splits its sums by row count ([KERNELS] section 5.2). Kept per chunk, that gate makes PF-B exact on the three agent\n"
     "  prompts of 1.6K-7.9K tokens (8 / 8, 28 / 28, 22 / 22 logits hashes, x1.12-1.21), but it peaks at 45.8-46.2 GB, over\n"
     "  the line, and does not ship [K, BM:data/pfd/pfd.log part B]. A 30K-token prompt breaks it again, for a cause not yet\n"
     "  found (section 6.2)."
     if FAIL else
     "- **One MoE call per layer in prefill (PF-B) was not exact at first** (0 of 7-28 logits hashes equal, +15-22 %) [K,\n"
     "  BM:data/pfb/pfb_report.txt]: the shared expert's one-output gate splits its sums by row count ([KERNELS] section 5.2).\n"
     "  Kept per chunk, that gate makes PF-B exact on the three agent prompts of 1.6K-7.9K tokens (8 / 8, 28 / 28, 22 / 22\n"
     "  logits hashes, x1.12-1.21), but it peaks at 45.8-46.2 GB, over the line, and does not ship [K, BM:data/pfd/pfd.log\n"
     "  part B].")))

# MAIN 11.1 row
R["MAIN.md"].append((
    "| one MoE call per layer in prefill (PF-B) | same | a further 12-21 %, exact; peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B] | over the line |",
    "| one MoE call per layer in prefill (PF-B) | same | a further 12-21 %, exact on these prompts"
    + (" but not on a 30K-token prompt (section 6.2); peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B; LP:data/prefill_1109/exact30k/summary.txt] | over the line |"
       if FAIL else "; peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B] | over the line |")))

# KERNELS abstract
R["KERNELS.md"].append((
    "layers over all chunks is exact once two kernels are pinned, and prefills 1.6K-7.9K-token prompts x1.14-1.21 faster\n"
    "[K]. We map which MLX kernels are row-invariant at decode and prefill widths, and report ten MLX findings with",
    "layers over all chunks is exact on 1.6K-7.9K-token prompts once two kernels are pinned, and prefills them x1.14-1.21\n"
    + ("faster [K]; a 30K-token prompt still breaks it, for a reason not yet found [K]. We map which MLX kernels are\n"
       "row-invariant at decode and prefill widths, and report ten MLX findings with"
       if FAIL else
       "faster [K]. We map which MLX kernels are row-invariant at decode and prefill widths, and report ten MLX findings with")))

# KERNELS "Where we lose"
R["KERNELS.md"].append((
    "The exact batched prefill peaks at 46.2 GB, over\nthe 45 GB line [K]. And on the same machine",
    ("The batched prefill peaks at 46.2 GB, over\nthe 45 GB line [K], and on a 30K-token prompt it is not exact (section 4.6) [K].\n"
     "And on the same machine"
     if FAIL else
     "The batched prefill (exact on 1.6K-7.9K-token prompts) peaks\nat 46.2 GB, over the 45 GB line [K]. And on the same machine")))

# KERNELS 4.6: scope the row-invariance sentence (both branches); fail adds the 30K bullet after the engine bullet
R["KERNELS.md"].append((
    "  shared expert's one-output gate kept per chunk (section 5.2). Every other op of the block is row-invariant: router,\n"
    "  softmax, top-10 selection and normalisation, the padded expert GLU even with 6,216 rows on one expert, the weighted\n"
    "  sum, the shared expert and the residual write [K, R15, R16].",
    "  shared expert's one-output gate kept per chunk (section 5.2). At the shapes tested (up to 7,932 rows per call), every\n"
    "  other op of the block is row-invariant: router, softmax, top-10 selection and normalisation, the padded expert GLU even\n"
    "  with 6,216 rows on one expert, the weighted sum, the shared expert and the residual write [K, R15, R16]."))
if FAIL:
    R["KERNELS.md"].append((
        "  is 45.81 GB at x1.12-1.17 [K, R20]: both stay over the 45 GB line.\n",
        "  is 45.81 GB at x1.12-1.17 [K, R20]: both stay over the 45 GB line.\n"
        "- On a 30K-token prompt it is not exact. On 29,988 tokens (the 32 GB exact configuration, 64 tokens) the logits hashes\n"
        "  equal the layer-major prefill's on 0 of 25 checks, from the first check on, at 8,192 rows per call (four calls per\n"
        "  layer) and at 4,096 (eight), and the tokens differ from the 40th [K, R33]. Its exactness is shown only up to 7.9K\n"
        "  tokens; the cause is not yet found.\n"))

# KERNELS 7.2
R["KERNELS.md"].append((
    "logits hash; the gate's split-K was the cause (section 5.2), and keeping it per chunk made the prefill exact [K, R15,\n"
    "  R16; LT:'PF-B EXACTNESS', CORRECTION].",
    scoped(
    "logits hash; the gate's split-K was the cause (section 5.2), and keeping it per chunk made the prefill exact on the\n"
    "  tested prompts of 1.6K-7.9K tokens [K, R15, R16, R20; LT:'PF-B EXACTNESS', CORRECTION].",
    "\n- **\"Keeping the gate per chunk makes the batched prefill exact.\"** It did on the 1.6K-7.9K-token prompts, not on a\n"
    "  30K-token prompt (section 4.6); the cause is not yet found [K, R33].")))

# KERNELS 9, prefill peak
R["KERNELS.md"].append((
    "- **The prefill peak.** The exact batched prefill gains x1.14-1.21 but peaks at 45.8-46.2 GB, over the 45 GB line; a\n"
    "  variant under the line was not found [K, R20].",
    scoped(
    "- **The prefill peak.** The batched prefill gains x1.14-1.21 and is exact on the tested 1.6K-7.9K-token prompts, but\n"
    "  peaks at 45.8-46.2 GB, over the 45 GB line; a variant under the line was not found [K, R20].",
    "\n- **Long prompts.** The batched prefill is not exact on a 30K-token prompt, and why is open (section 4.6) [K, R33].")))

# KERNELS result table: R33 after R32
if FAIL:
    R["KERNELS.md"].append(("\n| R32 | ", None))  # placeholder: handled below
# RELEASE long prompts
R["RELEASE.md"].append((
    "+ read-ahead; exact since the seg-per-chunk fix) prefill a 7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime\n"
    "  peak 45.09 / 46.23 GB [K], above the 45.0 line on that prompt).",
    ("+ read-ahead; exact on the 1.6K-7.9K-token agent prompts since the seg-per-chunk fix, not on a 30K-token prompt\n"
     "  ([MAIN] section 6.2)) prefill a 7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 /\n"
     "  46.23 GB [K], above the 45.0 line on that prompt)."
     if FAIL else
     "+ read-ahead; exact on the 1.6K-7.9K-token agent prompts since the seg-per-chunk fix) prefill a 7.9K-token agent\n"
     "  prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 / 46.23 GB [K], above the 45.0 line on that prompt).")))

R33 = ("| R33 | the engine's batched prefill against the layer-major prefill on a 29,988-token prompt (32 GB exact "
       "configuration, 64 tokens, one run each) | **not** exact | verify-logits hashes equal on 0 / 25 checks at 8,192 rows "
       "per call (4 calls per layer) and at 4,096 (8 calls), first difference at check 1; tokens equal through the 39th, "
       "differ from the 40th; the layer-major prefill == the default at 30K (hashes 26 / 26, tokens 64 / 64). On the 32 GB chat configuration: tokens differ from the 43rd at 30K, equal at 10K (no hashes) "
       "| [K] | D:prefill_1109/exact30k/summary.txt; D:prefill_1109.md |")

if branch == "none":
    R = {"MAIN.md": [], "KERNELS.md": [], "RELEASE.md": []}

# ---- PF-D / memory part (every branch; BigMoE 00:5x, re-derived: lh 25 / 25 and 27 / 27, 705,420 rows, peaks) ----
R["MAIN.md"].append((   # abstract
    "bits per weight and the process's lifetime peak at most 45 GB (plus a 7.45 GB expert tier that macOS may reclaim). The",
    "bits per weight and the process's lifetime peak at most 45 GB on the benchmark prompts (plus a 7.45 GB expert tier that\n"
    "macOS may reclaim; longer prompts go over it, section 6.1). The"))
R["MAIN.md"].append((   # 4.5, PF-D paragraph
    "reference rebuild and its tokens equal to one-token decoding on 12 / 12 prompts [K, BM:data/pfd/pfd_report_0009.txt].\n",
    "reference rebuild and its tokens equal to one-token decoding on 12 / 12 prompts [K, BM:data/pfd/pfd_report_0009.txt].\n"
    "It stays exact on 10K- and 30K-token prompts, where it runs once per layer and 1,024-token chunk: tokens and logits\n"
    "hashes (25 / 25, 27 / 27) equal the same configuration without it, and a check arm found all 705,420 staged rows equal to the reference\n"
    "rebuild. There it saves only 3-4 % of the first token and adds about 2 GB of lifetime peak [K,\n"
    "BM:data/pfd_long/summary.txt].\n"))
R["MAIN.md"].append((   # 6.1, memory paragraph: header
    "**The memory line has two known exceptions [K].** On three long agent prompts",
    "**Long prompts go over the memory line [K].** On three long agent prompts"))
R["MAIN.md"].append((   # 6.1, memory paragraph: 10K / 30K peaks
    "0.1 GB [K]. Every row of Tables 6.1a-c is at or under 43.7 GB (the PF-D row; every other row at or under 43.2 GB).",
    "0.1 GB [K]. Prompts of 10K and 30K tokens go further over it: the chat configuration as shipped peaks at 47.5 and\n"
    "48.6 GB on them (45.2 and 46.7 GB with PF-D off) [K, BM:data/pfd_long/summary.txt], and the 32 GB chat configuration at\n"
    "34.0 and 35.5 GB [K, LP:data/prefill_1109.md]. Every row of Tables 6.1a-c is at or under 43.7 GB (the PF-D row; every\n"
    "other row at or under 43.2 GB)."))
R["RELEASE.md"].append((   # section 5, known limits
    "- PF-D was measured on chat-length prompts (37-101 tokens) only; long prompts take the grouped prefill path without it.",
    "- PF-D's speed was measured on chat-length prompts (37-101 tokens). On long prompts it also runs, once per layer and\n"
    "  1,024-token chunk, and stays exact: on 10K- and 30K-token prompts its tokens and logits hashes equal the same configuration without\n"
    "  it, and a check arm found all 705,420 staged rows equal to the reference rebuild [K, BM:data/pfd_long/summary.txt].\n"
    "  There it shortens the first token by only 3-4 % (78.9 -> 76.6 s at 10K, 249.4 -> 238.4 s at 30K) and raises the\n"
    "  lifetime peak by about 2 GB [K]."))
R["RELEASE.md"].append((
    "- Lifetime peaks above 45.0 GB on 7.9K-token prompts (45.1-46.4 GB) [K].",
    "- Lifetime peaks above the memory line on long prompts [K]: 45.1-46.4 GB on a 7.9K-token prompt; on 10K / 30K-token\n"
    "  prompts 45.2 / 46.7 GB without PF-D and 47.5 / 48.6 GB with it (the shipped 45 GB chat configuration) [K,\n"
    "  BM:data/pfd_long/summary.txt], and 34.0 / 35.5 GB for the 32 GB chat configuration [K, LP:data/prefill_1109.md]."))

# ---- optional dated corrections note (--notes), placed before the generated "## Use of AI tools" tail ----
NOTES = {
 "MAIN.md": "- **10 November 2026 (v1.0.1).** The abstract's memory line now applies to the benchmark prompts: longer prompts go\n"
            "  over it (section 6.1). PF-D's exactness on 10K- and 30K-token prompts was added (section 4.5)."
            + (" PF-B's exactness is now limited to\n  the prompts it was checked on, up to 7.9K tokens; on a 30K-token prompt it is not exact (section 6.2)." if branch != "none" else ""),
 "KERNELS.md": "- **10 November 2026 (v1.0.1).** The batched prefill's exactness is now limited to prompts of up to 7.9K tokens; on a\n"
               "  30K-token prompt it is not exact (section 4.6, row R33).",
 "RELEASE.md": "- **10 November 2026 (v1.0.1).** Section 5 said long prompts run without PF-D; they run with it, and it stays exact.\n"
               "  Long prompts go over the memory lines (section 5)."
               + (" PF-B's exactness is limited to the tested prompts, up to 7.9K tokens." if branch != "none" else ""),
}
if "--notes" in args:
    for f, note in NOTES.items():
        if branch == "none" and f == "KERNELS.md":
            continue
        R[f].append(("\n## Use of AI tools\n", "\n## Corrections\n\n" + note + "\n\n## Use of AI tools\n"))

bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        if w is None:
            lines = t.split("\n")
            idx = [i for i, l in enumerate(lines) if l.startswith("| R32 | ")]
            if len(idx) != 1 or any(l.startswith("| R33 | ") for l in lines):
                print(f"BAD R33 insert in {f}"); bad += 1; continue
            lines.insert(idx[0] + 1, R33); t = "\n".join(lines); print(f"OK  {f}: R33 row after R32"); continue
        c = t.count(o)
        if c != 1:
            print(f"BAD x{c} {f}: {o[:80]!r}"); bad += 1; continue
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f, t in out.items():
    for i, l in enumerate(t.split("\n"), 1):
        if len(l) > 125 and not l.startswith("|") and l not in (P / f).read_text():
            print(f"   long line {f}:{i} ({len(l)})")
if "--out" in args and not bad:
    d = pathlib.Path(opt("--out")); d.mkdir(parents=True, exist_ok=True)
    for f, t in out.items(): (d / f).write_text(t)
    print("rendered to", d)
if "--write" in args and not bad:
    stamp = time.strftime("%H%M")
    bak = P / f".erratum_bak_{stamp}"  # private, not git: paper copies carry the author block
    bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
elif "--write" in args:
    print("NOT written:", bad, "bad")
