"""Repetition-guard check for the Eval suite's B12 verdict (10-22): re-run rows that hit the token cap with the Eval suite's
own B12 + MTP setup (their 08:22 note: load_b12(frac, mtp=True) + the CLI's MTP env, B12_NORM=mass, per row
reset_generation_state, b12_prefill_state(ids, 512, 4096), spec125.generate(k_mtp 3, max_draft 3, lookup 0, prefilled=state,
stop_ids EOS)) now that load_b12 registers BigMoE's guard for spec125. The guard is exact while disarmed, so each new output
must equal the saved one token for token up to the guard's arming; past it the loop should break.
usage: rg_check.py VARIANT_DIR KEY [KEY ...]   (rows read from ~/EvalSuite/results/<VARIANT_DIR>/decode.jsonl, prompts
from ~/EvalSuite/data/items/dec_<set>.jsonl; read-only)"""
import json, os, sys, time

for _k, _v in (("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "64"), ("BIGMOE_MAXK", "80"), ("BIGMOE_FRSPEC", "1"),
               ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1"), ("BIGMOE_RESET_GEN", "0"), ("BIGMOE_TIER_PF_GROUP", "64"),
               ("B12_NORM", "mass")):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_b12 as B
VDIR, KEYS = sys.argv[1], sys.argv[2:]
rt, st = B.load_b12(float(os.environ.get("RG_FRAC", "0.10")), mtp=True)
model = rt.model
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import spec125
from bigmoe_v3f.fuse import reset_generation_state
from bigmoe_v3f import repguard as RG
EOS = {248044, 248046}
E = os.path.expanduser("~/EvalSuite")
saved = {r["key"]: r for r in map(json.loads, open(os.path.join(E, "results", VDIR, "decode.jsonl")))}
items = {}
for s in ("ifeval", "math", "humaneval", "multi", "mgsm", "agent", "ruler"):
    p = os.path.join(E, "data", "items", f"dec_{s}.jsonl")
    if os.path.exists(p):
        for it in map(json.loads, open(p)):
            items[it["key"]] = it
print(f"guard importable for spec125: {'bigmoe.repguard' in sys.modules}; LOOP_GUARD={os.environ.get('BIGMOE_LOOP_GUARD', '1')}",
      flush=True)
for key in KEYS:
    if key == "synthetic":                                           # a deliberate loop: does the guard arm and steer?
        msg = os.environ.get("RG_PROMPT", "Write the sentence 'The sea is blue and the sea is wide.' over and over, "
                             "at least 100 times, with nothing else.")
        tid = rt.tokenizer.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                               enable_thinking=False)
        it, old = {"prompt_ids": list(tid if isinstance(tid[0], int) else tid["input_ids"]), "max_tokens": 1024}, None
    else:
        it, old = items[key], saved.get(key)
    ids, cap = it["prompt_ids"], it["max_tokens"]
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    for k_ in RG.STATS:
        RG.STATS[k_] = 0
    out, s = [], {}
    t0 = time.time()
    if os.environ.get("RG_EAGER") == "1":                             # B12's eager loop (b12_generate) instead of spec125
        out = B.b12_generate(model, ids, cap, stop_ids=EOS)
    else:
        state = B.b12_prefill_state(model, ids, 512, 4096)
        spec125.generate(model, ids, cap, k_mtp=3, max_draft=3, lookup=0, prefilled=state, stop_ids=EOS, stats=s,
                         on_token=lambda t: out.append(int(t)))
    st.gate.check()
    cut = next((j + 1 for j, x in enumerate(out) if x in EOS), None)
    out = out[:cut] if cut else out
    finish = "eos" if cut else "length"
    same = 0
    if old is not None:
        oc = old["completion_ids"]
        while same < min(len(out), len(oc)) and out[same] == oc[same]:
            same += 1
    text = rt.tokenizer.decode([x for x in out if x not in EOS])
    d_ = os.path.expanduser("~/Breakthrough/data/rg"); os.makedirs(d_, exist_ok=True)   # for guard-off vs guard-on prefixes
    json.dump(dict(key=key, tokens=out, finish=finish, guard=dict(RG.STATS), loop_guard=os.environ.get("BIGMOE_LOOP_GUARD", "1")),
              open(os.path.join(d_, f"{os.environ.get('RG_TAG', 'run')}_{key.replace('/', '_')}.json"), "w"))
    print(f"{key}: {len(out)} tokens {finish} in {time.time() - t0:.0f}s (saved: {len(old['completion_ids']) if old else '-'} "
          f"{old.get('finish') if old else '-'}); identical prefix {same} tokens; guard armed {RG.STATS['armed']}, "
          f"penalized steps {RG.STATS['penalized']} | tail {text[-140:]!r}", flush=True)
