"""v1.1.0: PF-B's cause found and a research fix (orchestrator 10 Nov 14:3x; Leap bf9d1ad + f432586,
LP:data/tue_afternoon_1110.md sections 3, 5 and 6). Applies on top of v1.0.1 (+ the two formatting fixes); the SMALL_RAM
release (release_smallram.py) applies after it. Numbers re-derived by the editor from Leap/data/tue_afternoon_1110/:
bis_{10k,15k,20k,25k}_{pfa,pfb8k}_runs.json (lh 25/25, 28/28, 25/25, 24/24; tails 772 / 666 / 547 / 427) and
fix_{tail,30k}_{pfa,pfb,pfbfix}_runs.json (tail 5,412: 0/20 -> 20/20, x1.12 -> x1.17; 30K: 0/25 -> 26/26, x1.18 -> x1.15,
29,988 / 54.01 s = 555 tok/s, peak 40.42 vs PF-A 38.11 GB). Threshold: a grouped 292-token tail fails; grouped tails of
541 (published 1.6K), 547, 634, 666, 764, 772 pass; 427 passed alone in its group (so not a grouped-tail data point).
usage: python3 -I v110_pfb.py [--write]   (backups: ~/Papers/.v110_pfb_bak_<time>/)"""
import os, sys, time, pathlib
P = pathlib.Path.home() / "Papers"
R = {"MAIN.md": [], "KERNELS.md": [], "RELEASE.md": []}

R["MAIN.md"] += [
("PF-B's exactness is shown only up to 7.9K tokens, and the cause of the\n  difference is not yet found.",
 "The cause is the prompt's last chunk, not its length:\n"
 "  in PF-B's grouped call the router and the shared expert compute the rows of a short last chunk (292 tokens at 30K)\n"
 "  with other bits than a per-chunk call does. Last chunks of 541 tokens or more are unaffected, and so is a 427-token\n"
 "  last chunk alone in its call; prompts of 10K, 15K, 20K and 25K tokens are exact (hashes 25 / 25, 28 / 28, 25 / 25,\n"
 "  24 / 24) [K, LP:data/tue_afternoon_1110.md, sections 3 and 5]. Running those two parts per chunk slice fixes both\n"
 "  failing prompts: 20 / 20 hashes on a 5,412-token prompt with a 292-token last chunk (0 / 20 without the fix) and\n"
 "  26 / 26 at 30K, at x1.15-1.17 the layer-major prefill's speed (555 tok/s at 30K) and 2.3 GB more peak there [K, same\n"
 "  file, section 6]. The fix is in a research copy, not in the package. Its exactness rests on those two prompts and on\n"
 "  the lengths that passed without it, which were not re-run with it."),
("- **One MoE call per layer in prefill (PF-B) was not exact at first, and is not exact on a 30K-token prompt** (0 of\n",
 "- **One MoE call per layer in prefill (PF-B) was not exact at first, and is not exact when a prompt's last chunk is\n  short** (0 of\n"),
("A 30K-token prompt breaks it again, for a cause not yet\n  found (section 6.2).",
 "A short last chunk breaks it again, as in a 30K-token\n  prompt; a fix in a research copy makes those prompts exact (section 6.2)."),
("a further 12-21 %, exact on these prompts but not on a 30K-token prompt (section 6.2); peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B; LP:data/prefill_1109/exact30k/summary.txt] | over the line |",
 "a further 12-21 %, exact on these prompts, not when a prompt's last chunk is short (section 6.2); peaks 45.8-46.2 GB | [K, BM:data/pfd/pfd.log part B; LP:data/prefill_1109/exact30k/summary.txt; LP:data/tue_afternoon_1110.md] | over the line; the short-chunk fix is in a research copy only |"),
("  the prompts it was checked on, up to 7.9K tokens; on a 30K-token prompt it is not exact (section 6.2).\n",
 "  the prompts it was checked on, up to 7.9K tokens; on a 30K-token prompt it is not exact (section 6.2).\n"
 "- **11 November 2026 (v1.1.0).** [SMALL-RAM] is published. PF-B's difference on a 30K-token prompt has a found cause, a\n"
 "  short last chunk, and a fix in a research copy, not in the package (section 6.2).\n"),
]
R["KERNELS.md"] += [
("faster [K]; a 30K-token prompt still breaks it, for a reason not yet found [K]. We map which MLX kernels are\n"
 "row-invariant at decode and prefill widths, and report ten MLX findings with",
 "faster [K]. A short last chunk sharing the batched call still breaks it, as in a 30K-token prompt; computing the router\n"
 "and the shared expert per chunk fixes that in a research copy [K]. We map which MLX kernels are row-invariant at decode\n"
 "and prefill widths, and report ten MLX findings with"),
("the 45 GB line [K], and on a 30K-token prompt it is not exact (section 4.6) [K].",
 "the 45 GB line [K], and it is not exact when a prompt's last chunk is short, as at 30K tokens (section 4.6) [K]."),
("Its exactness is shown only up to 7.9K\n  tokens; the cause is not yet found.\n",
 "The cause is the last chunk (292\n"
 "  tokens here): inside the batched call, the router and the shared expert compute its rows with other bits than a\n"
 "  per-chunk call does. Last chunks of 541 tokens or more are unaffected, and so is a 427-token last chunk alone in its\n"
 "  call; prompts of 10K, 15K, 20K and 25K tokens are exact [K, R34].\n"
 "- Computing the router and the shared expert per chunk slice inside the batched call, as the gate already is, makes both\n"
 "  failing prompts exact (a 5,412-token prompt with a 292-token last chunk, and the 30K prompt) at x1.15-1.17 the\n"
 "  layer-major prefill's speed [K, R35]. The fix is in a research copy, not in the package; its exactness rests on those\n"
 "  two prompts and on the lengths that passed without it.\n"),
("It did on the 1.6K-7.9K-token prompts, not on a\n  30K-token prompt (section 4.6); the cause is not yet found [K, R33].",
 "It did on the 1.6K-7.9K-token prompts, not on a\n"
 "  30K-token prompt (section 4.6). The cause is a short last chunk, whose router and shared expert rows the batched call\n"
 "  computes with other bits; a per-chunk fix exists in a research copy [K, R33-R35]."),
("- **Long prompts.** The batched prefill is not exact on a 30K-token prompt, and why is open (section 4.6) [K, R33].",
 "- **Short last chunks.** The batched prefill is not exact when a prompt's last chunk is short, as at 30K tokens. The fix\n"
 "  (router and shared expert per chunk slice) is in a research copy, not in the package (section 4.6) [K, R33-R35]."),
("  30K-token prompt it is not exact (section 4.6, row R33).\n",
 "  30K-token prompt it is not exact (section 4.6, row R33).\n"
 "- **11 November 2026 (v1.1.0).** The batched prefill's 30K difference has a found cause, a short last chunk, and a fix in\n"
 "  a research copy, not in the package (section 4.6, rows R34 and R35).\n"),
]
R34 = ("| R34 | the batched prefill by prompt length, and the cause (32 GB exact configuration; a bit check of layers 1 and 30's "
       "router, shared expert and gate on random inputs) | exact unless the last chunk is short | 10K / 15K / 20K / 25K tokens "
       "(last chunks 772 / 666 / 547 / 427): hashes 25 / 25, 28 / 28, 25 / 25, 24 / 24. Bit check: a grouped call of 5 x 1,024 "
       "+ 292 rows computes the 292 tail rows of the router (N 512) and the shared expert with other bits than per-chunk calls; "
       "tails of 547, 666 and 772 rows and full chunks are equal | [K] | D:tue_afternoon_1110.md (sections 3 and 5); "
       "D:tue_afternoon_1110/kcheck.py |")
R35 = ("| R35 | the fix: router and shared expert per chunk slice inside the batched call (research copy, off by default, not in "
       "the package) | exact on both failing prompts | 5,412 tokens (5 x 1,024 + 292): hashes 20 / 20 (0 / 20 without), "
       "prefill x1.17 over the layer-major prefill (x1.12 without); 29,988 tokens: 26 / 26 (0 / 25 without), x1.15 (x1.18 "
       "without), 555 tok/s, peak 40.42 vs 38.11 GB | [K] | D:tue_afternoon_1110.md (section 6); "
       "D:tue_afternoon_1110/pflm_segall_research.diff |")
R["RELEASE.md"] += [
("since the seg-per-chunk fix, not on a 30K-token prompt\n  ([MAIN] section 6.2))",
 "since the seg-per-chunk fix, not on prompts whose last\n"
 "  chunk is short, such as a 30K-token prompt; its fix is not in this package ([MAIN] section 6.2))"),
("  Long prompts go over the memory lines (section 5). PF-B's exactness is limited to the tested prompts, up to 7.9K tokens.\n",
 "  Long prompts go over the memory lines (section 5). PF-B's exactness is limited to the tested prompts, up to 7.9K tokens.\n"
 "- **11 November 2026 (v1.1.0).** PF-B's difference on a 30K-token prompt comes from a short last chunk; a fix exists in a\n"
 "  research copy, not in this package ([MAIN] section 6.2).\n"),
]
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o[:80]!r}"); continue
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:80]}")
    if f == "KERNELS.md":
        lines = t.split("\n"); idx = [i for i, l in enumerate(lines) if l.startswith("| R33 | ")]
        if len(idx) == 1 and not any(l.startswith("| R34 | ") for l in lines):
            lines[idx[0] + 1:idx[0] + 1] = [R34, R35]; t = "\n".join(lines); print("OK  KERNELS.md: R34, R35 after R33")
        else:
            bad += 1; print("BAD R34/R35 insert")
    out[f] = t
if "--write" in sys.argv and not bad:
    bak = P / f".v110_pfb_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
