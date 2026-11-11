#!/bin/zsh
# SMALL_RAM section 5.4 label check (Breakthrough, Mon 2026-11-09 10:3x). Read-only on BigMoE, EvalSuite, findings and ~/Papers.
# Question: which configuration is CHS2's "L" (and N45's), and which label may SMALL_RAM give the CHS2 rows?
# Writes ~/Breakthrough/data/papers/chat45_labels_check.txt. Usage: zsh chat45_labels_check.sh
out=~/Breakthrough/data/papers/chat45_labels_check.txt
{
echo "SMALL_RAM 5.4 label check, $(date '+%a %d %b %H:%M'), read-only"
echo "1. CHS2's L (BigMoE scripts/chsb.sh): the last line of data/stack/s5p_env.txt"
grep -n '^L=' ~/BigMoE/scripts/chsb.sh
echo "   s5p_env.txt header:"; sed -n 1p ~/BigMoE/data/stack/s5p_env.txt | cut -c1-220
echo "   the bundle's marker REF2_COMMIT_SUB=9 in that line: $(grep -v '^#' ~/BigMoE/data/stack/s5p_env.txt | tr ',' '\n' | grep -c '^REF2_COMMIT_SUB=9$')"
echo "2. N45's L (scripts/n45.sh) is the same line, and refuses to run without the bundle:"
grep -n '^L=\|REF2_COMMIT_SUB=9' ~/BigMoE/scripts/n45.sh
echo "3. CHS2 per-arm token match to one-token decoding (data/chs/chs2_report.txt):"
grep -o '^  x2[a-z0-9]* *: WHOLE DECODE *[0-9.]*\|== AR [0-9]*/12' ~/BigMoE/data/chs/chs2_report.txt | paste - - | sed 's/  */ /g'
echo "4. E without coherent reads, later run (findings/2026-11-08_0750_45gb-chat-chs4.md):"
sed -n 32,33p ~/FableGod/findings/2026-11-08_0750_45gb-chat-chs4.md
echo "5. L + COH's card gate (EvalSuite docs/RESULT_card_lcoh_2026-11-09.md; heading, config, verdict):"
sed -n 1p ~/EvalSuite/docs/RESULT_card_lcoh_2026-11-09.md
grep -n '^- L + COH = \|^| \*\*Card gate' ~/EvalSuite/docs/RESULT_card_lcoh_2026-11-09.md
echo "6. How the companions label these configurations:"
grep -n "Table 6.1b. LOSSY\|N45 L (n\|^| 12 HumanEval | X = " ~/Papers/MAIN.md | cut -c1-260
grep -n "L adds to it a bundle\|L adds the exact-safe bundle" ~/Papers/EVALUATION.md | cut -c1-220
echo "Conclusion: CHS2's L and N45's L are S5' with the exact-safe bundle ([MAIN] 6.1c's X: 'not gated'), not the certified"
echo "flags alone; CHS2 also predates coherent reads. SMALL_RAM 5.4 now labels L and M1-B 'lossy, ungated as run' and E"
echo "'not exact on chat without coherent reads' (12 / 12 here, 3 / 12 later). MAIN 6.1b lists N45 L's IFEval and math"
echo "rows under the certified flags: sent to the editor as a question."
} > $out
cat $out
