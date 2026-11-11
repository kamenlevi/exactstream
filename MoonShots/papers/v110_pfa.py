"""v1.1.0: the layer-major prefill (PF-A) is exact only when a prompt's last 1,024-token chunk is long enough
(orchestrator Wed 11 Nov 01:5x; Leap 7e520f7, LP:data/pfa_tail_1111.md). Applies on top of v110_pfb.py; the SMALL_RAM
release (release_smallram.py) still applies after it (MAIN's "- **11 November 2026 (v1.1.0).** PF-B's difference" kept).
Numbers re-derived by the editor from the raw runs:
- Leap/data/pfa_tail_1111/t{116,300}_{default,pfa}_{runs,tokens}.json (exact-32, 1.1.0 clean clone, 64 tokens):
  P 5,235 = 5 x 1,024 + 115: lh equal 0 / 20 (default 20 checks, PF-A 21; first difference at check 1), tokens differ from
  the 36th; P 5,420 = 5 x 1,024 + 300: lh 20 / 20, tokens 64 / 64. First token 42.03 / 13.42 s and 41.98 / 13.56 s.
- The switch: Leap/data/agent_cc_1111.md T1 table (chat-32, cache hits whose tail ran chunk-major against the cold PF-A
  sweep): k42 N 8,389 = 8 x 1,024 + 197 FAIL A, k31 N 12,496 = 12 x 1,024 + 208 PASS; T1b (tail through PF-A) passes 197.
- The 45 GB audit prompts (BM:data/pfx/pfx_report.txt): P 1,565 / 4,730 / 7,932 -> last chunks 541 / 634 / 764.
  Other PF-A comparisons in MAIN / KERNELS: 30K 292, 5,412 292, 10K-25K 772 / 666 / 547 / 427; SMALL_RAM's 24 GB
  PF-A answer check: 6,980 tokens (836), 256 / 256 tokens (LP:data/tue_afternoon_1110.md section 2).
- Cause, marked [G]: pflm.py's R57_PFLM=1 note (full-E scratch, original ids, E = 512 vs len(unique) changes MLX's gather
  kernel choice on short calls); no kernel-level bit check was run for PF-A.
Applied Wed 11 Nov 02:00 (SMALL_RAM's sentence reworded 02:01 as below: the jsonl keeps the answer text, not ids).
usage: python3 -I v110_pfa.py [--write]   (backups: ~/Papers/.v110_pfa_bak_<time>/)"""
import pathlib
import re
import sys
import time

P = pathlib.Path.home() / "Papers"
R = {"MAIN.md": [], "RELEASE.md": [], "SMALL_RAM.md": [], "PATH_DRAFT.md": []}

PATH_ROW = ("| exact layer-major prefill | 45 GB, 1.6K-7.9K-token prompts | prefill 1.9-3.4x | [K, F:2026-11-08_0800] | built; "
            "needs a 0.1 GB memory trim |")
PATH_ROW_NEW = ("| layer-major prefill (exact unless the last chunk is under about 200 tokens) | 45 GB, 1.6K-7.9K-token prompts | "
                "prefill 1.9-3.4x | [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md] | built; needs a 0.1 GB memory trim |")

R["MAIN.md"] += [
# outline, section 4
("  keep, wired limit, the exact-safe bundle, coherent reads, the 1-row check); prefill (layer-major, exact); the memory line\n"
 "  and where it is exceeded (agent prompts, 45.1 GB).\n",
 "  keep, wired limit, the exact-safe bundle, coherent reads, the 1-row check); prefill (layer-major, exact unless the last\n"
 "  chunk is short); the memory line and where it is exceeded (agent prompts, 45.1 GB).\n"),
# 3.2 audits table: last chunks of the three prompts + the short-chunk row
("| layer-major prefill | 3 agent prompts, 1.6K-7.9K tokens | decode tokens equal; logits hash equal on 8 / 8, 28 / 28, "
 "22 / 22 checks | F:2026-11-08_0800 |\n",
 "| layer-major prefill | 3 agent prompts, 1.6K-7.9K tokens (last chunks of 541, 634 and 764 tokens) | decode tokens equal; "
 "logits hash equal on 8 / 8, 28 / 28, 22 / 22 checks | F:2026-11-08_0800 |\n"
 "| layer-major prefill, short last chunk (32 GB exact configuration, 64 tokens) | 5,235 and 5,420 tokens (last chunks of "
 "115 and 300 tokens) | logits hash equal on 0 / 20 and 20 / 20 checks; with the 115-token chunk the tokens differ from "
 "the 36th | LP:data/pfa_tail_1111.md |\n"),
# 4.5
("prompt and reads each expert about once (68 GB), exactly: 1.9-3.4x faster at 1.6K-7.9K tokens (section 6.2) [K,\n"
 "F:2026-11-08_0800]. On short prompts the first token is still 2.7-4.6 s (section 6.3), except on chat with PF-D.\n",
 "prompt and reads each expert about once (68 GB): 1.9-3.4x faster at 1.6K-7.9K tokens, and exact except when the\n"
 "prompt's last 1,024-token chunk is under about 200 tokens (section 6.2) [K, F:2026-11-08_0800; LP:data/pfa_tail_1111.md].\n"
 "On short prompts the first token is still 2.7-4.6 s (section 6.3), except on chat with PF-D.\n"),
# 6.1 memory paragraph
("; the layer-major exact prefill of\nsection 6.2 peaks at 45.10 GB",
 "; the layer-major prefill of\nsection 6.2 peaks at 45.10 GB"),
# 6.2 heading and table head
("### 6.2 Prefill on long prompts (exact)\n", "### 6.2 Prefill on long prompts\n"),
("| layer-major prefill (exact): first token (tok/s) |", "| layer-major prefill: first token (tok/s) |"),
# 6.2 first bullet + the short-chunk bullet
("  hashes equal today's on every check (8 / 8, 28 / 28, 22 / 22) [K, F:2026-11-08_0800; BM:data/pfx/pfx_report.txt]. Peak:\n"
 "  45.10 GB (see above).\n",
 "  hashes equal today's on every check (8 / 8, 28 / 28, 22 / 22) [K, F:2026-11-08_0800; BM:data/pfx/pfx_report.txt]. Peak:\n"
 "  45.10 GB (see above). These prompts' last 1,024-token chunks are 541, 634 and 764 tokens long [K, same report].\n"
 "- It is exact only when the prompt's last 1,024-token chunk is long enough. On the 32 GB exact configuration, a\n"
 "  prompt of 5,235 tokens (5 x 1,024 + 115) gives logits hashes equal to the default prefill's on 0 of 20 checks, and\n"
 "  its tokens differ from the 36th; a prompt of 5,420 tokens (a 300-token last chunk) gives 20 of 20 [K,\n"
 "  LP:data/pfa_tail_1111.md]. Runs of the same two paths on the 32 GB chat configuration put the switch between a\n"
 "  last chunk of 197 tokens (differs) and one of 208 (equal) [K, LP:data/agent_cc_1111.md]. The likely cause [G]: on a\n"
 "  short call MLX picks another gather kernel for the layer-major prefill's full expert scratch (all 512 experts' rows)\n"
 "  than for the default prefill's compact one (the call's own experts); both paths are deterministic but round\n"
 "  differently. The other prompts on which this paper and [KERNELS] compare with the layer-major prefill end in chunks\n"
 "  of 292 tokens or more, so those results stand. The 24 and 45 GB lines were not checked on a short last chunk; the\n"
 "  mechanism is the same there [G].\n"),
# 6.2 PF-B bullet: the 30K reference's last chunk
("  from the 40th on, while the layer-major prefill's hashes equal the default prefill's on 26 of 26 [K,\n"
 "  LP:data/prefill_1109/exact30k/summary.txt].",
 "  from the 40th on, while the layer-major prefill's hashes equal the default prefill's on 26 of 26 (its last chunk, 292\n"
 "  tokens, is long enough for it) [K, LP:data/prefill_1109/exact30k/summary.txt]."),
# section 7 table, row 7
("layer-major 1.9-3.4x at 1.6K-7.9K tokens, exact (built, not shipped)",
 "layer-major 1.9-3.4x at 1.6K-7.9K tokens, exact unless the last chunk is under about 200 tokens (built, not shipped)"),
# 11.1 table
(PATH_ROW, PATH_ROW_NEW),
# Corrections
("- **11 November 2026 (v1.1.0).** PF-B's difference on a 30K-token prompt has a found cause, a short last chunk, and a\n"
 "  fix in a research copy, not in the package (section 6.2).\n",
 "- **11 November 2026 (v1.1.0).** PF-B's difference on a 30K-token prompt has a found cause, a short last chunk, and a\n"
 "  fix in a research copy, not in the package (section 6.2). The layer-major prefill is exact only when a prompt's\n"
 "  last 1,024-token chunk is about 200 tokens or longer; a shorter one changed the tokens (sections 3.2, 4.5, 6.2, 7\n"
 "  and 11.1).\n"),
]

R["RELEASE.md"] += [
("- Long prompts: the default prefill re-reads each layer's experts per 1,024-token chunk. PF-A (layer-major) and PF-B (layer split\n"
 "  and read-ahead; exact on the 1.6K-7.9K-token agent prompts since the seg-per-chunk fix, not on prompts whose last\n"
 "  chunk is short, such as a 30K-token prompt; its fix is not in this package ([MAIN] section 6.2)) prefill a 7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 /\n"
 "  46.23 GB [K], above the 45.0 line on that prompt).\n",
 "- Long prompts: the default prefill re-reads each layer's experts per 1,024-token chunk. PF-A (layer-major; exact except\n"
 "  when the prompt's last 1,024-token chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]) and PF-B (layer\n"
 "  split and read-ahead; exact on the 1.6K-7.9K-token agent prompts since the seg-per-chunk fix, not on prompts whose\n"
 "  last chunk is short, such as a 30K-token prompt; its fix is not in this package ([MAIN] section 6.2)) prefill a\n"
 "  7.9K-token agent prompt at 433 and 500 tok/s [K] (exact E; lifetime peak 45.09 / 46.23 GB [K], above the 45.0 line\n"
 "  on that prompt).\n"),
("- Prefill: BM:data/pfd/pfd_report.txt at a6d5347 (part B; PF-A / PF-B), BM:data/pfb/pfb_report.txt (CT1). PF-D:",
 "- Prefill: BM:data/pfd/pfd_report.txt at a6d5347 (part B; PF-A / PF-B), BM:data/pfb/pfb_report.txt (CT1),\n"
 "  LP:data/pfa_tail_1111.md (PF-A on a short last chunk). PF-D:"),
("  30K-token prompt comes from a short last chunk; a fix exists in a research copy, not in this package ([MAIN]\n"
 "  section 6.2).\n",
 "  30K-token prompt comes from a short last chunk; a fix exists in a research copy, not in this package ([MAIN]\n"
 "  section 6.2). PF-A is exact only when a prompt's last 1,024-token chunk is about 200 tokens or longer (section 2).\n"),
]

R["SMALL_RAM.md"] += [
("  9,988 and 195.0 s at 29,988 with the default prefill. The exact layer-major prefill ([MAIN] section 6.2) cuts that\n"
 "  2.4-2.7x (17.7 s at 6,980 tokens, 23.7 s at 9,988, 71.7 s at 29,988), for 0.3-2.4 GB more peak (26.54 GB at 10K,\n"
 "  30.60 GB at 30K) [K, LP:data/tue_afternoon_1110.md].\n",
 "  9,988 and 195.0 s at 29,988 with the default prefill. The layer-major prefill ([MAIN] section 6.2) cuts that\n"
 "  2.4-2.7x (17.7 s at 6,980 tokens, 23.7 s at 9,988, 71.7 s at 29,988), for 0.3-2.4 GB more peak (26.54 GB at 10K,\n"
 "  30.60 GB at 30K) [K, LP:data/tue_afternoon_1110.md]. Its 256-token answer to the 6,980-token prompt is the default\n"
 "  prefill's, character for character [K, same file]. At 32 GB it is exact except when the prompt's last 1,024-token\n"
 "  chunk is under about 200 tokens [K, LP:data/pfa_tail_1111.md]; the 24 GB line was not checked on such a prompt.\n"),
]

R["PATH_DRAFT.md"] += [(PATH_ROW, PATH_ROW_NEW)]

MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s")      # a wrapped line must not start like a list item
bad, out = 0, {}
for f, reps in R.items():
    t = (P / f).read_text()
    for o, w in reps:
        c = t.count(o)
        if c != 1:
            bad += 1; print(f"BAD x{c} {f}: {o.splitlines()[0][:90]!r}"); continue
        for ln in w.split("\n")[1:]:                      # continuation lines of the new text
            if ln.strip() and MARKER.match(ln) and not ln.startswith("- ") and not ln.startswith("| "):
                bad += 1; print(f"BAD marker {f}: {ln[:90]!r}")
        t = t.replace(o, w); print(f"OK  {f}: {o.splitlines()[0][:90]}")
    out[f] = t
for f in ("MAIN.md", "RELEASE.md", "SMALL_RAM.md"):           # what still calls the layer-major prefill exact, unqualified
    for i, ln in enumerate(out[f].split("\n"), 1):
        if re.search(r"(?i)layer-major[^.|]{0,40}\bexact\b|\bexact\b[^.|]{0,20}layer-major", ln) and not re.search(
                r"(?i)unless|except|only when|short", ln):
            print(f"CHECK {f}:{i}: {ln.strip()[:120]}")
if "--write" in sys.argv and not bad:
    bak = P / f".v110_pfa_bak_{time.strftime('%H%M')}"; bak.mkdir(exist_ok=True)
    for f, t in out.items():
        (bak / f).write_text((P / f).read_text()); (P / f).write_text(t)
    print("written; backups in", bak)
print("bad", bad)
