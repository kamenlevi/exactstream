"""price.py (Fable B, DENSE_REBUILD): bytes and time of the 45 GB FAST check's dense part, today's kernel set vs the
rebuilt one. CPU only, no MLX, no GPU. Model per dependent dispatch: t = t_sync + bytes / B.
  t_sync = the fixed cost of one global sync point in a dependent chain (ramp + tail + launch), by kernel class
  B      = the in-kernel streaming rate while the kernel is in full flight
Both are [P] until Leap's step 0 part B bench: run `price.py --fit N_GDN N_HC N_ATT N_MLP` with its kernel counts, or
edit TODAY_N. Shapes from ~/BigMoE/cfg/config.json; piece times [K] from ~/Leap/TABLE_350.md 'STEP 0, PART A'."""
import sys
H, HC, LR, NE = 2560, 4, 320, 512
q4 = lambda n, k: n * k * (0.5 + 4 / 64)            # 4-bit g64 codes + bf16 scales / biases, bytes
q8 = lambda n, k: n * k * (1.0 + 4 / 64)
bf = lambda n, k: n * k * 2.0
MB = 1e6
P = {  # bytes per call
    "gdn in_proj (qkv|z|a|b)": q4(10240 + 6144 + 96, H),
    "gdn state (fp32 in + out)": 2 * 48 * 128 * 128 * 4,
    "gdn out_proj": q4(H, 6144),
    "att q|k|v proj": q4(12288 + 1024, H),
    "att keys + values (2,048 x 512 bf16 x 2)": 2 * 2048 * 512 * 2,
    "att o_proj": q4(H, 6144),
    "hc down + inject (HC8)": q8(LR, HC * H) + bf(HC, HC * H),
    "hc up (HC8)": q8(HC * H, LR),
    "router (8-bit) + shared gate|up + sgate": q8(NE, H) + q4(1280 + 1, H),
    "shared down": q4(H, 640),
}
NG, NA, NL = 36, 12, 48
PIECE = {  # piece -> (calls per check, bytes per call, GPU ms today [K])
    "GDN": (NG, P["gdn in_proj (qkv|z|a|b)"] + P["gdn state (fp32 in + out)"] + P["gdn out_proj"], 13.2),
    "HC reads": (2 * NL, P["hc down + inject (HC8)"] + P["hc up (HC8)"], 8.6),
    "attention": (NA, P["att q|k|v proj"] + P["att keys + values (2,048 x 512 bf16 x 2)"] + P["att o_proj"], 6.4),
    "router + shared + tail + HC write": (NL, P["router (8-bit) + shared gate|up + sgate"] + P["shared down"], 5.7 + 1.1 + 0.8),
}
TODAY_N = {"GDN": 10, "HC reads": 3, "attention": 60, "router + shared + tail + HC write": 14}   # dispatches per call [P]
# the rebuilt set: dependent dispatches per call (see DENSE_REBUILD.md section 3)
NEW_N = {"GDN": 1, "HC reads": 3, "attention": 4, "router + shared + tail + HC write": 2}
NEW_N_MERGED = {"GDN": 1, "HC reads": 2, "attention": 3, "router + shared + tail + HC write": 1}

def table():
    tot_b = 0.0
    print("piece                               calls   MB/call   GB/check   ms today   floor@250   today GB/s")
    for k, (n, b, ms) in PIECE.items():
        tot_b += n * b
        print("%-34s %5d %9.2f %10.3f %10.1f %11.2f %12.0f" % (k, n, b / MB, n * b / 1e9, ms, n * b / 250e9 * 1e3, n * b / 1e9 / ms * 1e3))
    ms = sum(v[2] for v in PIECE.values())
    print("%-34s %5s %9s %10.3f %10.1f %11.2f %12.0f" % ("sum (PLE, head, embed not in)", "", "", tot_b / 1e9, ms, tot_b / 250e9 * 1e3, tot_b / 1e9 / ms * 1e3))
    return tot_b, ms

def fit(today_n):
    """t_sync per piece that explains today's ms at each B; one number per piece because kernel classes differ"""
    print("\nwhat one dependent dispatch costs today (us), by the streaming rate B assumed in flight:")
    print("  B GB/s  " + "".join("%-36s" % k[:34] for k in PIECE))
    out = {}
    for B in (110, 150, 200, 235, 250):
        row = {}
        for k, (n, b, ms) in PIECE.items():
            row[k] = (ms * 1e3 / n - b / (B * 1e3)) / today_n[k]
        out[B] = row
        print("  %5d   " % B + "".join("%-36s" % ("%.1f us x %d" % (row[k], today_n[k])) for k in PIECE))
    return out

def predict(fitd, today_n):
    print("\nrebuilt dense part, ms per check (same B, t_sync per dispatch = today's fitted value for the piece, capped at 20 us;")
    print("the 'B up' column also lifts B to min(235, B + 60): the fused kernels stream longer per sync)")
    print("  B GB/s   today   rebuilt (9-10 per layer)   merged (6-7 per layer)   merged, B up")
    for B, row in fitd.items():
        def tot(nn, Bx):
            s = 0.0
            for k, (n, b, ms) in PIECE.items():
                ts = min(max(row[k], 4.0), 20.0)
                s += n * (nn[k] * ts + b / (Bx * 1e3)) / 1e3
            return s
        today = sum(v[2] for v in PIECE.values())
        print("  %5d   %5.1f   %12.1f   %20.1f   %18.1f" % (B, today, tot(NEW_N, B), tot(NEW_N_MERGED, B), tot(NEW_N_MERGED, min(235, B + 60))))

if __name__ == "__main__":
    tn = dict(TODAY_N)
    if "--fit" in sys.argv:
        v = [int(a) for a in sys.argv[sys.argv.index("--fit") + 1:][:4]]
        tn = dict(zip(PIECE, v))
    for k, b in P.items():
        print("%-46s %7.2f MB" % (k, b / MB))
    print()
    table()
    predict(fit(tn), tn)
