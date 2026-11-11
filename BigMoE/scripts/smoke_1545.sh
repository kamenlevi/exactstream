#!/bin/bash
# Wed 15:5x smoke 2 on today's screen: (1) the exact default with lever 2 on, as tonight's l2 arms run it (build_l2, arm
# gate+t+pf, BIGMOE_PRED_LINES=4, FC 4 / 2): every token == REF2-T on 2 prompts (CAP 32) and forecast reads issued + used;
# (2) the QOM all-in + lever 2, as tonight's qallf arm (build_qom): it must run, issue forecast reads, post optional uses.
# C 160 uniform pools (the FAST path), share lock via scripts/guarded.sh. -> data/smk/ (smoke 1, 15:48: QOM + DRAFT17 OK;
# the exact arm hit the PF_B <= 30 + 31 * PRED_LINES assert, hence PRED_LINES 4)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/smk
cp scripts/r57_screen.py data/r57/snap/r57_screen_smk.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_smk.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_smk.sh"
export C45_SCREEN=data/r57/snap/r57_screen_smk.py C45_DEADLINE=1800 C45_OUT="$HOME/BigMoE/data/smk" KEEP_LOCK=1
P="R57_LAYSZ=0,BIGMOE_TIER_CL=64,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4,BIGMOE_FC_F=4,BIGMOE_FC_FS=2"
R57_KEYS="humaneval/HumanEval/91,gsm8k/708" MG=2 C45_ALARM=600 C45_CAP=32 \
  C45_REFTOK="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t" C45_ARMS="exf_160:160:BIGMOE_EXT_BUILD=build_l2,$P" $RUN
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/91" MG=2 C45_ALARM=600 C45_CAP=64 \
  C45_ARMS="qaf_160:160:BIGMOE_EXT_BUILD=build_qom,$P,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1,REF2_SMALL_NATIVE=1" $RUN
.venv/bin/python -I - <<'EOF'
import json
ref = json.load(open("data/d45/ref14_tokens.json"))
for t in ("exf_160", "qaf_160"):
    try:
        rs = json.load(open(f"data/r57/{t}_runs.json")); tk = json.load(open(f"data/r57/{t}_tokens.json"))
    except OSError as e:
        print(t, "NO RUNS", e); continue
    pf = lambda k: sum(r.get("pf", {}).get(k, 0) for r in rs)
    tok = lambda v: v.get("gate+t+pf", v.get("gate+t"))
    eq = [k for k, v in tk.items() if tok(v) == ref[k]["gate+t"][:len(tok(v))]]
    print(f"{t}: runs {len(rs)}, tok/s {[round(r['tps'], 1) for r in rs]}, tokens == REF2-T prefix {len(eq)} / {len(tk)}, "
          f"forecast reads {pf('fc_issued')} used {pf('fc_hits')} ({pf('fc_late')} landing) dropped {pf('fc_drops')}, "
          f"QOM optional {pf('qom_opt')} not read {pf('qom_skip')}")
EOF
