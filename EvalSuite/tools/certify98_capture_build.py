"""CERTIFY98's capture list for BigMoE's dd_capture.py (docs/PREREG_certify98_2026-11-15.md amendments 2, 3, 5, 7 and 9):
one jsonl line per window, {key, set, src, prompt_ids, max_tokens}, in the v2 manifest's capture order. CPU only.
  ~/BigMoE/.venv/bin/python tools/certify98_capture_build.py      # the chat template needs BigMoE's venv
- key '<set>/<name>' is amendment 5's layout (<root>/<set>/<name>.npz): set heplus | mbpp | everyday; name HumanEval_<id>,
  mbpp_<task_id>, or the everyday key with / -> _. src: heplus | mbpp | agent | ifbench | ifeval | ifevalx.
- Order: HE+ 140, MBPP 300, everyday agent / IFBench / IFEval (v1's 60), then the 57 replacement IFEval (ifevalx) last.
- prompt_ids, in amendment 2's d16w / d1102mbpp prompt formats (thinking off), checked here token for token:
  - HE+: EvalSuite's dec_humaneval / dec_humanevalx ids (D.chat of _he_user). They must equal tf_prompts' tf_code ids and
    the prompt rows of d16w's 24 tf_code windows (the dev bank's heplus24); L1-20's 80 he_<id> windows are compared too.
  - MBPP: D.chat of MTP Head's MB_USER (tools/mbpp_build.py, = TrainLab sticky/dlog_capture.py's). It must equal the
    prompt rows (ids[:gen]) of all 375 d1102mbpp captures, and dec_mbpp's stored ids.
  - everyday: the item files' ids as stored (dec_agent, dec_ifbench, dec_ifeval, dec_ifevalx); the chat items must equal
    D.chat of their user text.
- max_tokens 512 on every line (amendment 7: each answer to EOS, capped at the dev captures' 512).
Writes data/certify98/capture_list_2026-11-18.jsonl (data/ stays out of git: the agent prompts are agent turns) and prints
its sha256 for the amendment."""
import hashlib, json, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from evalsuite import data as D
from mbpp_build import MB_USER
import numpy as np

H = os.path.expanduser("~")
V2 = os.path.join(ROOT, "results", "certify98", "fresh_manifest_v2_2026-11-18.json")
OUT = os.path.join(ROOT, "data", "certify98", "capture_list_2026-11-18.jsonl")
D16W, D1102 = f"{H}/TrainLab/data/d16w", f"{H}/TrainLab/data/dlog/d1102mbpp"
CAP = 512


def main():
    v2 = json.load(open(V2))
    tok = D.tokenizer()
    fails = []
    # HE+: the suite's 164 problems, against the dev bank's d16w windows (and L1-20's)
    he = {}
    for f in ("dec_humaneval", "dec_humanevalx"):
        for it in D.load(f):
            i = int(it["task_id"].split("/")[1])
            assert i not in he and it["prompt_ids"] == D.chat(tok, it["user"]), it["key"]
            he[i] = it["prompt_ids"]
    assert sorted(he) == list(range(164))
    tf = {int(it["key"].split("/")[-1]): it["prompt_ids"] for it in D.load("tf_prompts") if it["key"].startswith("tf_code/")}
    assert len(tf) == 24
    ok_tf = [i for i, p in tf.items() if p == he[i] and np.load(f"{D16W}/tf_code_HumanEval_{i}.npz")["ids"][:len(p)].tolist() == p
             and int(np.load(f"{D16W}/tf_code_HumanEval_{i}.npz")["off"]) == 0]
    l120 = sorted(int(m.group(1)) for f in os.listdir(D16W) for m in [re.fullmatch(r"he_(\d+)\.npz", f)] if m)
    ok_l = [i for i in l120 if np.load(f"{D16W}/he_{i}.npz")["ids"][:len(he[i])].tolist() == he[i]]
    print(f"HE+ format: d16w tf_code (the dev bank's heplus24) {len(ok_tf)} / {len(tf)} identical to the suite's ids and "
          f"tf_prompts'; L1-20's he_<id> windows {len(ok_l)} / {len(l120)} identical (so he_<id> = HumanEval/<id>)")
    if len(ok_tf) != len(tf):
        fails.append("HE+ format")
    # MBPP: MB_USER through the chat template, against all 375 d1102mbpp captures and dec_mbpp
    mb = {x["task_id"]: x for x in map(json.loads, open(f"{H}/TrainLab/data/mbpp/mbpp.jsonl"))}
    assert len(mb) == 974
    mb_ids = lambda i: D.chat(tok, MB_USER.replace("{q}", mb[i]["text"].strip()).replace("{t}", "\n".join(mb[i]["test_list"])))
    caps = sorted(f for f in os.listdir(D1102) if re.fullmatch(r"dlog__mbpp\d+\.npz", f))
    bad = [f for f in caps for z in [np.load(os.path.join(D1102, f))]
           if z["ids"][:int(z["gen"])].tolist() != mb_ids(int(re.search(r"mbpp(\d+)", f).group(1)))]
    es = [it for it in D.load("dec_mbpp") if it["prompt_ids"] != mb_ids(it["task_id"])]
    print(f"MBPP format: d1102mbpp {len(caps) - len(bad)} / {len(caps)} prompt rows identical; dec_mbpp "
          f"{100 - len(es)} / 100" + (f"; differ: {bad[:3]} {[it['key'] for it in es[:3]]}" if bad or es else ""))
    if len(caps) != 375 or bad or es:
        fails.append("MBPP format")
    # everyday: the item files as stored; the chat ones rebuilt from their user text
    items = {it["key"]: it for f in ("dec_agent", "dec_ifbench", "dec_ifeval", "dec_ifevalx") for it in D.load(f)}
    rows = [dict(key=f"heplus/HumanEval_{i}", set="heplus", src="heplus", prompt_ids=he[i], max_tokens=CAP)
            for i in v2["heplus_capture"]]
    rows += [dict(key=f"mbpp/mbpp_{i}", set="mbpp", src="mbpp", prompt_ids=mb_ids(i), max_tokens=CAP) for i in v2["mbpp_fresh"]]
    nchat = 0
    for src in ("agent", "ifbench", "ifeval", "ifevalx"):
        for k in v2["everyday_capture"][src]:
            it = items[k]
            if src != "agent":
                nchat += 1
                if it["prompt_ids"] != D.chat(tok, it["user"]):
                    fails.append(f"everyday {k}")
            rows.append(dict(key=f"everyday/{k.replace('/', '_')}", set="everyday", src=src, prompt_ids=it["prompt_ids"],
                             max_tokens=CAP))
    print(f"everyday: {nchat} chat items rebuilt from their user text, "
          f"{nchat - sum(f.startswith('everyday') for f in fails)} identical; agent {len(v2['everyday_capture']['agent'])} as stored")
    assert len(rows) == 647 and len({r["key"] for r in rows}) == 647, len(rows)
    cc = [f"heplus/HumanEval_{i}" for i in v2["heplus_count_check"]]
    assert set(cc) <= {r["key"] for r in rows[:140]}
    if fails:
        print(f"CAPTURE LIST NOT WRITTEN: {fails[:5]}")
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, OUT)
    n = {s: sum(r["set"] == s for r in rows) for s in ("heplus", "mbpp", "everyday")}
    L = [len(r["prompt_ids"]) for r in rows]
    print(f"{os.path.relpath(OUT, ROOT)}: {len(rows)} windows ({n}), prompt tokens {sum(L)} (max {max(L)}), max_tokens {CAP}; "
          f"sha256 {hashlib.sha256(open(OUT, 'rb').read()).hexdigest()}; manifest v2 sha256 "
          f"{hashlib.sha256(open(V2, 'rb').read()).hexdigest()}; count check: {', '.join(cc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
