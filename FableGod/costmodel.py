# End-to-end check cost model from measured numbers (sources in FINDINGS.md Step 6). ms per check.
# check = draft + dense + rest + ssd_exposed + ctx + rows*r + startup ; tok/s = 1000*kept/check
types = {  # kept, rows, ssd_exposed(0.559*io), ctx_ms, startup_ms   (D-23 logs, C 224, t 0.3, pre-OOO)
 "code":  dict(kept=10.54, rows=12.3, ssd=65.3, ctx=0.5,  su=23.1),
 "math":  dict(kept=6.61,  rows=9.0,  ssd=31.0, ctx=0.5,  su=9.4),
 "chat":  dict(kept=2.03,  rows=3.1,  ssd=10.9, ctx=0.4,  su=5.6),
 "agent": dict(kept=4.92,  rows=7.4,  ssd=22.6, ctx=23.5, su=86.2),
}
measured = {"code":51.1,"math":43.9,"chat":22.9,"agent":20.7}   # s45d_replay.txt, t 0.3
def run(name, draft, dense, rest, ssd_k, ctx_k, r, kept_k, su_k, note=""):
    out=[]
    for t,v in types.items():
        d = draft[t] if isinstance(draft,dict) else draft
        kept = v["kept"]*kept_k.get(t,1.0) if isinstance(kept_k,dict) else v["kept"]*kept_k
        rows = v["rows"]*(kept/v["kept"])
        chk = d + dense + rest + v["ssd"]*ssd_k + v["ctx"]*ctx_k + rows*r + v["su"]*su_k
        out.append((t, chk, 1000*kept/chk))
    s = f"| {name:<34} | " + " | ".join(f"{t} {chk:5.0f} ms {tps:5.0f}" for t,chk,tps in out) + f" | {note} |"
    return s
rows_ = []
rows_.append("| stage (cumulative) | code | math | chat | agent | note |")
rows_.append("|---|---|---|---|---|---|")
rows_.append(run("S0 today, D-23 base (measured 51/44/23/21)", 12, 27.4, 2.8, 1.0, 1.0, 7.82, 1.0, 1.0, "model vs measured within 5-8 %"))
rows_.append(run("S1 + OOO (adopted; exposed SSD x0.7)", 12, 27.4, 2.8, 0.7, 1.0, 7.82, 1.0, 1.0, "R-69b: cycle -11 %"))
rows_.append(run("S2 + GPU constant: dense 27.4->16, rest 1.5, draft chat/agent 8", {"code":12,"math":12,"chat":8,"agent":8}, 16, 1.5, 0.7, 1.0, 7.82, 1.0, 1.0, "REF2c -6 (meas), bmm16 -2.5 (meas), GDN one-kernel -3 (est), MTP early exit (est)"))
rows_.append(run("S3 + cache C 320 + M-17 (~50.5 GB): exposed SSD x0.35", {"code":12,"math":12,"chat":8,"agent":8}, 16, 1.5, 0.35, 1.0, 7.82, 1.0, 1.0, "cachesize45: 128 -> 37.9 MB per kept token"))
rows_.append(run("S4 + per-row 7.82 -> 4.0 (NAX at 16 rows, fused combine)", {"code":12,"math":12,"chat":8,"agent":8}, 16, 1.5, 0.35, 1.0, 4.0, 1.0, 1.0, "K-62j 16-row 1.46-1.71x of 3c; only pays once GPU-bound"))
rows_.append(run("S5 + kept: MTP K15+cut (code x1.145, math x1.04, chat/agent x1.3)", {"code":12,"math":12,"chat":8,"agent":8}, 16, 1.5, 0.35, 1.0, 4.0, {"code":1.145,"math":1.042,"chat":1.3,"agent":1.3}, 1.0, "D-24 on paper, D-23b est.; D-23c tonight"))
rows_.append(run("S6 + context 2.27->1.0 per K, agent start-up 86->20", {"code":12,"math":12,"chat":8,"agent":8}, 16, 1.5, 0.35, 0.44, 4.0, {"code":1.145,"math":1.042,"chat":1.3,"agent":1.3}, 0.23, "block-sparse att (R-81 est.), K-80 fix (est.)"))
rows_.append(run("FLOOR physics, today's kept: draft 12/8, dense 11.7, rows 1.7, SSD 0", {"code":12,"math":12,"chat":8,"agent":8}, 11.7, 0.5, 0.0, 0.11, 1.7, 1.0, 0.0, "bytes at 300 GB/s; SSD fully hidden"))
rows_.append(run("FLOOR + S5 kept", {"code":12,"math":12,"chat":8,"agent":8}, 11.7, 0.5, 0.0, 0.11, 1.7, {"code":1.145,"math":1.042,"chat":1.3,"agent":1.3}, 0.0, ""))
rows_.append(run("FLOOR + kept 14 code / 4 chat (what 350 code needs)", {"code":12,"math":12,"chat":8,"agent":8}, 11.7, 0.5, 0.0, 0.11, 1.7, {"code":14/10.54,"math":1.3,"chat":4/2.03,"agent":1.6}, 0.0, "needs a drafter we do not have"))
txt = "\n".join(rows_)
print(txt)
open("costmodel_table.md","w").write(txt+"\n")
