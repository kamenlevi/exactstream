# D1 kill test on D-23's logs (CPU). Per check: pc=[rows, kept, check_ms, draft_ms], cf=per-position drafter confidence.
# Rows 0..kept-1 accepted; row `kept` rejected if kept < rows; later rows unverified. Positions with cf >= T are "sure".
import json, os, collections, statistics as st
R=[json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
R=[r for r in R if r.get("arm")=="dfl" and not r.get("invalid")]
def analyse(T):
    out={}
    for typ in ("code","math","chat","agent"):
        tested=sure=sure_rej=unc=unc_rej=0; runs=[]; rows_tot=0; checks=0; kept_tot=0
        for r in R:
            if r["type"]!=typ: continue
            for (rows,kept,ck,dr),cf in zip(r["pc"],r["cf"]):
                checks+=1; rows_tot+=rows; kept_tot+=kept
                n_tested=min(kept,rows)+ (1 if kept<rows else 0)
                run=0
                for j in range(n_tested):
                    c=cf[j] if j<len(cf) else 0.0
                    rejected=(j==kept and kept<rows)
                    tested+=1
                    if c>=T:
                        sure+=1; sure_rej+=rejected; run+=1
                    else:
                        unc+=1; unc_rej+=rejected; runs.append(run); run=0
                if run: runs.append(run)
        out[typ]=dict(checks=checks,rows=rows_tot,kept=kept_tot,tested=tested,sure=sure,sure_rej=sure_rej,unc=unc,unc_rej=unc_rej,
                      sure_share=sure/max(tested,1),sure_dis=sure_rej/max(sure,1),unc_dis=unc_rej/max(unc,1),
                      run_mean=(st.mean(runs) if runs else 0),run_med=(st.median(runs) if runs else 0),runs_n=len(runs))
    return out
for T in (0.99,0.995,0.999):
    print(f"\n=== threshold T={T} ===")
    for typ,o in analyse(T).items():
        print(f"{typ:6s} checks {o['checks']:4d} rows {o['rows']:5d} kept {o['kept']:6.0f} tested {o['tested']:5d} | sure share {o['sure_share']:.3f} "
              f"sure-dis {o['sure_dis']:.4f} ({o['sure_rej']}/{o['sure']}) | unc-dis {o['unc_dis']:.3f} ({o['unc_rej']}/{o['unc']}) | runs n {o['runs_n']} mean {o['run_mean']:.2f} med {o['run_med']}")
print("\nkept per check, rows per check (dfl arm):")
for typ in ("code","math","chat","agent"):
    rs=[r for r in R if r["type"]==typ]; pc=[p for r in rs for p in r["pc"]]
    print(typ, "checks",len(pc),"rows/check %.2f"%(sum(p[0] for p in pc)/len(pc)),"kept/check %.2f"%(sum(p[1] for p in pc)/len(pc)))
