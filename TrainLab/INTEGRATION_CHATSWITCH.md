# Chat switch: the native MTP head on chat prompts, DFlash2 elsewhere

Status (Mon 11-02 08:3x): D-23d = PASS (REPORT "D-23d CHATSWITCH RESULT"). On 19 chat prompts outside every training set,
at C 192 on the live 45 GB default, the switch is +24.5 % time-weighted (geo-mean 1.223, MTP faster on 19 of 19). It is
exact: every token was equal across arms. Each arm ran its own pass, so there is no warm-expert order bias (w 0.992). Agent
stays on DFlash2 (G -0.5 %). Ready to wire; the owners of the live default (Leap / BigMoE) decide when, with Kamen's word for
the default. (The earlier D-23c' table, arms back to back, carried an order bias and is superseded.)

## What it is
- Per request, NewIdeas' type rule (scripts/type_rule.py; the copy in sticky/d23c_mtptrunc.py `rule()`) labels the
  prompt code / math / agent / chat from its text: tool schemas -> agent; code markers -> code; math markers -> math;
  else chat. It labelled all 19 D-23d chat prompts chat and all 4 agent prompts agent.
- code, math, agent: the default as it ships (DFlash2 L1-20, 15 drafts, the running-confidence cut 0.3).
- chat: the 125B's native MTP chain instead of DFlash2:
  - head weights: the MTP layer with ~/TrainLab/ckpt/d22_b5.safetensors loaded over it (D-22 b5: attention, router,
    shared expert trained; experts, norms, lm_head as shipped);
  - K15: up to 15 chained steps; each step's top-1 p goes into a running product; drafting stops at the first step
    whose running product falls below 0.3, and only the drafts before it are verified (the default's own cut rule);
  - the head's history: built from the prompt after prefill (0.03-0.34 s on our prompts), then each check's kept
    rows are appended; the chain's own entries are trimmed after each check (not committed history).
- Exactness: verification is unchanged, so every output token equals the default's; D-23c' and D-23d checked every
  token of all 41 prompts across arms.

## Measured (D-23d: C 192, the 45 GB default with OOO, 23 prompts outside every training set, each arm its own pass;
## data/d23d_read.txt)
| type | prompts | DFlash2 tok/s | MTP tok/s | time-weighted change | kept per check |
|---|---|---|---|---|---|
| chat, non-English (bg de fr es ru zh ja ar) | 16 | 22.9 | 29.1 | +27.4 % (geo 1.265, faster on 16 of 16) | 2.09 -> 2.65 |
| chat, English (IFEval) | 3 | 25.5 | 25.9 | +1.4 % (geo 1.022) | 4.55 -> 5.36 |
| agent (stays on DFlash2) | 4 | 21.5 | 21.4 | -0.5 % (geo 0.981) | 4.80 -> 8.20 |

Code and math stay on DFlash2 (D-23c', 4 + 4 prompts, order-corrected geo: code 0.998, math 0.909).
Whole set: on D-23c''s mix (chat = 45 % of DFlash2's time, half of it non-English) +7.0 %. If all chat is English, +0.6 %.
If all chat is non-English, ~+9.8 % (the pooled chat rate). The gain scales with the non-English share of chat.

## Why chat, and why not agent
DFlash2 L1-20 was trained on code + math (English) only and keeps ~2.1 tokens per check on non-English chat. The gain is
largest for the languages furthest from English: bg +50 %, ar +38 %, zh/ja/ru +25-35 %, fr +5-10 %. The 125B's own MTP head
knows every language the 125B does (2.65 kept). On agent text, the MTP chain keeps +71 % more per check. But its longer
chains verify more rows, which touch more cold experts (misses +24-76 %), so tok/s is flat at C 192. It may pay at
C 224 or with more memory. On code, the 15 sequential MTP steps cost 37 ms per cycle vs DFlash2's 12 ms, which eats the
extra kept tokens.

## Wiring notes
- Both drafters resident: the D-23c' process peaked at 38.2 GB at C 192 with both (BIGMOE_MTP_STREAM unset).
- The harness that measured it: sticky/d23c_mtptrunc.py (a copy of Leap's k69_live.py / r57_screen.py, the default
  R57_PRESET=45 + OOO + TIER_COMBINE through BigMoE's routing-scores hook; D23_NEWDEF=1), arm `mtp15e`; D23_ORDER=sep for
  the separate passes.
- Open: the default runs at C 224; D-20b's switch gained more at higher C (+10 % at C 160, +27 % at C 224), so C 192
  is likely the conservative reading. The C 224 confirmation needs a night with ~46 GB free.
