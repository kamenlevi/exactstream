"""long_context.py (Fable B, mission B6): the check's cost against context length from measured rows [K] and arithmetic
[P]. Attention per check: C0's 5.82 ms GPU at ~200 keys (12 QSA layers) and R-81's REF2-T curve (12 layers, 16-row
calls, shared GPU, rough): 2K 12.6, 4K 16.1, 8K 22.7, 10K 24.3, 16K 36.9, 20K 37.2 ms. KV: 12 layers x 2 kv heads x
256 x (K + V) bf16 = 24 KB per token; the indexer's keys 1 head x 128 bf16 = 3 KB per token. The rest of the check as
C0 (16 rows, warm, GPU eval): dense 34.0 of which attention 5.82; experts 50.9 (3c) / 43.2 (NAX6) / 27 (floor);
draft 12.0 (ANE 12.4), commit 4.7, rest 5; SSD exposed 65 on code today (RELAX_350), 0 at the floor engine's night.
Chat: 4-8-row checks pad to REF2's 16-row attention kernels, so attention costs the same per check; experts 6 x 3.0 + 7.
Kept: code 10.54, chat 2.53 (the MTP chain) [K]. Run: ~/BigMoE/.venv/bin/python long_context.py > long_context.out"""
import numpy as np

ATT = {200: 5.82, 1000: 9.0, 2000: 12.6, 4000: 16.1, 8000: 22.7, 10000: 24.3, 16000: 36.9, 20000: 37.2}   # ms per check [K]; 1K interpolated [G]
KV_B = 24e3 + 3e3                                                                                           # bytes per token of K + V + indexer keys
BW = 280e9
print("== attention per 16-row check vs keys [K C0 at 200, R-81 above]; KV bytes read per check and their time at 280 GB/s [P]; FLOPs of the dense QK^T + PV at 16 rows x 24 heads x 256 [P] ==")
print("%7s | %7s | %9s | %7s | %7s | %8s" % ("keys", "att ms", "KV MB", "KV ms", "GFLOP", "TFLOPS"))
for n, a in ATT.items():
    kv = KV_B * n; fl = 2 * 2 * 16 * 24 * 256 * n * 12 / 1e9
    print("%7d | %7.1f | %9.1f | %7.2f | %7.1f | %8.1f" % (n, a, kv / 1e6, 1e3 * kv / BW, fl, fl / a))
print("slope 2K -> 20K: %.2f ms per 1K keys per check [P on K]" % ((ATT[20000] - ATT[2000]) / 18))

print("\n== the check and tok/s vs keys (today's engine, warm numbers + the SSD term; code 16 rows kept 10.54, chat 6 rows kept 2.53) [P on K] ==")
print("%7s | %-38s | %-38s" % ("keys", "code: check GPU ms, cycle, tok/s, att share", "chat: check GPU, cycle, tok/s, att share"))
for n, a in ATT.items():
    extra = a - ATT[200]
    code_gpu = 34.0 + extra + 50.9; code_cycle = code_gpu + 12.0 + 4.7 + 5 + 65.0
    chat_gpu = 34.0 + extra + 6 * 3.0 + 7; chat_cycle = chat_gpu + 10.0 + 4.7 + 5 + 30.0
    print("%7d | GPU %5.0f, cycle %5.0f, %5.1f tok/s, att %3.0f %% | GPU %5.0f, cycle %5.0f, %5.1f tok/s, att %3.0f %%"
          % (n, code_gpu, code_cycle, 1e3 * 10.54 / code_cycle, 100 * a / code_gpu, chat_gpu, chat_cycle, 1e3 * 2.53 / chat_cycle, 100 * a / chat_gpu))

print("\n== where attention overtakes the experts (per check) [P on K]: att(N) = 5.82 + 1.37 x (N - 0.2K) / 1K ==")
slope = (ATT[20000] - ATT[2000]) / 18.0
for name, ex in (("today 3c, 16 rows", 50.9), ("NAX6, 16 rows", 43.2), ("floor-engine experts, 16 rows", 27.0), ("chat 6 rows today", 25.0), ("chat 6 rows at the floor", 13.0)):
    n = 200 + 1000 * (ex - ATT[200]) / slope
    print("  %-32s experts %4.0f ms: attention equals it at ~%4.1fK keys" % (name, ex, n / 1e3))

print("\n== the floor engine at long context (dense 23 EXACT / 17 FAST at 200 keys, experts 27, draft 5, SSD hidden; code kept 10.54 -> 12.5 with the override / pick; chat kept 2.53) [P] ==")
for n, a in ATT.items():
    extra = a - ATT[200]
    for track, d in (("EXACT", 23.0), ("FAST", 17.0)):
        c16 = d + extra + 27 + 5 + 4.7 + 5; c6 = d + extra + 6 * 0.9 + 5 + 5 + 4.7 + 5
        print("  %6d keys %5s: code cycle %5.0f ms -> %5.0f tok/s (kept 10.54) / %5.0f (12.5); chat cycle %4.0f -> %5.1f tok/s" % (n, track, c16, 1e3 * 10.54 / c16, 1e3 * 12.5 / c16, c6, 1e3 * 2.53 / c6))
