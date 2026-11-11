| arm | distinct experts / pass / layer (natural) | code KLD (target 0.0025, frugal-10 0.0147) | gsm KLD (target 0.0045, frugal-10 0.0307) | ifeval KLD (target 0.0204, frugal-10 0.1153) | multi KLD (target 0.0205, frugal-10 0.0924) |
|---|---|---|---|---|---|
| exact | natural | 0.0002 top-1 0.999 n 1416 (<= target) | 0.0007 top-1 0.995 n 3230 (<= target) | 0.0040 top-1 0.976 n 1657 (<= target) | 0.0023 top-1 0.985 n 1663 (<= target) |
| b16 | 16.0 (27.1, 0.59x) | 0.0021 top-1 0.994 n 1416 (<= target) | 0.0070 top-1 0.985 n 3230 (<= frugal-10) | 0.0364 top-1 0.924 n 1657 (<= frugal-10) | 0.0177 top-1 0.962 n 1663 (<= target) |
| o4 | 14.4 (27.1, 0.53x) | 0.0042 top-1 0.991 n 1416 (<= frugal-10) | 0.0126 top-1 0.983 n 3230 (<= frugal-10) | 0.0730 top-1 0.904 n 1657 (<= frugal-10) | 0.0374 top-1 0.945 n 1663 (<= frugal-10) |
| b14 | 14.0 (27.1, 0.52x) | 0.0031 top-1 0.992 n 1416 (<= frugal-10) | 0.0096 top-1 0.983 n 3230 (<= frugal-10) | 0.0557 top-1 0.909 n 1657 (<= frugal-10) | 0.0268 top-1 0.951 n 1663 (<= frugal-10) |
| b18 | 17.9 (27.1, 0.66x) | 0.0019 top-1 0.994 n 1416 (<= target) | 0.0050 top-1 0.986 n 3230 (<= frugal-10) | 0.0245 top-1 0.943 n 1657 (<= frugal-10) | 0.0126 top-1 0.969 n 1663 (<= target) |
| o5 | 14.9 (27.1, 0.55x) | 0.0036 top-1 0.993 n 1416 (<= frugal-10) | 0.0076 top-1 0.984 n 3230 (<= frugal-10) | 0.0539 top-1 0.914 n 1657 (<= frugal-10) | 0.0351 top-1 0.947 n 1663 (<= frugal-10) |
