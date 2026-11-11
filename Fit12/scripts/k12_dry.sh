#!/bin/sh
# Fit12 (Tue 3 Nov 23:1x): dry test of the K12 read-out (/Users/kamen/Fit12/scripts/k12_verdict.py) on fit_t3.py's FAKE
# harness (no model, no GPU, CPU only, seconds). A = TAG r47, B = TAG k12, same 4 prompts / arms / ALT order, in one
# scratch dir. FAKE mode skips the real wrapper, so case (a) must be VOID; the other cases append the K12 lines the real run
# prints. (b) PASS 12/12; (c) one token changed in B -> FAIL; (d) A's prompt 42 paused and re-run with other bits (main
# run in void1_*) -> PASS (the main run is compared); (e) B's prompt 91 missing -> PASS on 9, 91 not comparable.
D=${1:-/private/tmp/claude-501/-Users-kamen-OjoX/7dd61b7e-2c45-43ec-b373-7768d2c96a18/scratchpad/k12dry}
PY=/Users/kamen/BigMoE/.venv/bin/python; V="$PY -B /Users/kamen/Fit12/scripts/k12_verdict.py $D r47 k12"
rm -rf "$D"; mkdir -p "$D"
K=humaneval/HumanEval
for t in r47 k12; do
  env F12_FAKE=1 F12_OUT="$D" F12_CAP=64 F12_FAKE_DT=0.002 F12_PHLOG=1 F12_ALT=1 F12_K1U_KU=4 F12_PAUSEF="$D/none.PAUSE" \
    F12_KEYS=$K/156,$K/42,$K/68,$K/91 F12_T3ARMS=t80ck2,t80ck1um3,t80ck1u F12_CMCL=47,32 \
    $( [ $t = k12 ] && echo F12_K12SYNC=1 ) $PY -B /Users/kamen/Fit12/scripts/fit_t3.py $t > "$D/$t.txt" 2>&1 || echo "$t rc $?"
done
echo "== (a) fake run, no K12 lines"; $V | tail -1
printf "[k12] K12: mx.eval at every MoE block's entry and exit (mtplx qwen4_exp.SparseMoeBlock)\n[k12] K12: 2304 synced MoE block calls; wrapper still installed: True\n" >> "$D/k12.txt"
echo "== (b) K12 lines added"; $V | grep -E 'equal|VERDICT'
f="$D/k12_humaneval_HumanEval_42_t80ck1u_toks.json"; cp "$f" "$f.keep"
$PY -c "import json,sys; t=json.load(open(sys.argv[1])); t[20]+=1; json.dump(t,open(sys.argv[1],'w'))" "$f"
echo "== (c) one token changed in B"; $V | grep -E 'DIFFERENT|42/t80ck1u|VERDICT' | head -4
mv "$f.keep" "$f"
for x in "" _toks _ph; do for a in t80ck2 t80ck1um3 t80ck1u; do
  mv "$D/r47_humaneval_HumanEval_42_${a}${x}.json" "$D/void1_r47_humaneval_HumanEval_42_${a}${x}.json"
  cp "$D/r47_humaneval_HumanEval_68_${a}${x}.json" "$D/r47_humaneval_HumanEval_42_${a}${x}.json"; done; done
echo "== (d) A's 42 paused and re-run (other bits live, main run in void1_)"; $V | grep -E 'equal|DIFFERENT|VERDICT'
rm -f "$D"/k12_humaneval_HumanEval_91_*
echo "== (e) B's 91 missing"; $V | grep -E 'equal|not comparable|VERDICT'
