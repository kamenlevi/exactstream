| stage (cumulative) | code | math | chat | agent | note |
|---|---|---|---|---|---|
| S0 today, D-23 base (measured 51/44/23/21) | code   227 ms    46 | math   153 ms    43 | chat    83 ms    24 | agent   232 ms    21 | model vs measured within 5-8 % |
| S1 + OOO (adopted; exposed SSD x0.7) | code   208 ms    51 | math   144 ms    46 | chat    80 ms    25 | agent   226 ms    22 | R-69b: cycle -11 % |
| S2 + GPU constant: dense 27.4->16, rest 1.5, draft chat/agent 8 | code   195 ms    54 | math   131 ms    50 | chat    63 ms    32 | agent   209 ms    24 | REF2c -6 (meas), bmm16 -2.5 (meas), GDN one-kernel -3 (est), MTP early exit (est) |
| S3 + cache C 320 + M-17 (~50.5 GB): exposed SSD x0.35 | code   172 ms    61 | math   121 ms    55 | chat    60 ms    34 | agent   201 ms    24 | cachesize45: 128 -> 37.9 MB per kept token |
| S4 + per-row 7.82 -> 4.0 (NAX at 16 rows, fused combine) | code   125 ms    84 | math    86 ms    77 | chat    48 ms    43 | agent   173 ms    28 | K-62j 16-row 1.46-1.71x of 3c; only pays once GPU-bound |
| S5 + kept: MTP K15+cut (code x1.145, math x1.04, chat/agent x1.3) | code   132 ms    91 | math    88 ms    78 | chat    51 ms    51 | agent   182 ms    35 | D-24 on paper, D-23b est.; D-23c tonight |
| S6 + context 2.27->1.0 per K, agent start-up 86->20 | code   114 ms   106 | math    80 ms    86 | chat    47 ms    56 | agent   102 ms    63 | block-sparse att (R-81 est.), K-80 fix (est.) |
| FLOOR physics, today's kept: draft 12/8, dense 11.7, rows 1.7, SSD 0 | code    45 ms   233 | math    40 ms   167 | chat    26 ms    80 | agent    35 ms   139 | bytes at 300 GB/s; SSD fully hidden |
| FLOOR + S5 kept                    | code    48 ms   250 | math    40 ms   171 | chat    27 ms    97 | agent    39 ms   163 |  |
| FLOOR + kept 14 code / 4 chat (what 350 code needs) | code    52 ms   269 | math    44 ms   195 | chat    31 ms   131 | agent    43 ms   183 | needs a drafter we do not have |
