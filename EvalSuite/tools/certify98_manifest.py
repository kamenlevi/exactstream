"""CERTIFY98's fresh test sets, drawn before any candidate exists (docs/PREREG_certify98_2026-11-15.md amendment 2).
CPU only. Writes results/certify98/fresh_manifest_2026-11-15.json (the ids, the exclusions and their sources); BigMoE's
Wed 18 capture adds each file's sha256 to a copy, which the proof-run amendment then freezes.
  .venv/bin/python tools/certify98_manifest.py [--check]     # --check: rebuild and compare with the committed file
- HumanEval+ fresh: HumanEval 0-163 minus the 24 d16w windows (the 125B's dev set) = 140, the gated set.
  The strict subset also drops Sandbox98's 60 HumanEval windows (the 2B's dev set) = 80, reported beside it.
- MBPP fresh: MBPP's test split (task_ids 11-510) minus every id TrainLab captured in data/dlog/d1102mbpp (375; they
  hold d1102mbpp's 100 eval windows = EvalSuite's dec_mbpp 100, and Sandbox98's 100 MBPP windows are among them), then
  a seeded draw of 300 (numpy default_rng(SEED).permutation, first 300, sorted).
- Everyday text (amendment 3, the orchestrator 02:0x: local sources only): all 29 agent windows (data/items/dec_agent:
  Claude Code-style turns) + a seeded 60 of IFEval's 100 and 61 of IFBench's 150 (EvalSuite's instruction-following chat
  prompts) = 150. No drafter trained on them: t30b's data is n1112 magicoder, Sandbox98's is magicoder, and no drafter
  screen used them. Exact 45 GB answers exist for all of them (ref2tnax, ref2tnaxn.r6, ref2tnaxn + .r4) as a cross-check
  of the capture's own runtime-path answers."""
import json, os, re, sys
import numpy as np

SEED, N_MBPP = 20261115, 300
H = os.path.expanduser("~")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certify98", "fresh_manifest_2026-11-15.json")


def ids(path, pat):
    return sorted({int(m.group(1)) for f in os.listdir(path) for m in [re.search(pat, f)] if m})


def build():
    split = json.load(open(f"{H}/BigMoE/data/deepdraft/bank/split.json"))
    d16w = sorted(int(re.search(r"HumanEval_(\d+)", f).group(1)) for f in split["heplus"])
    sb_he = ids(f"{H}/Sandbox98/data/he", r"HumanEval_(\d+)\.npz$")
    d1102 = ids(f"{H}/TrainLab/data/dlog/d1102mbpp", r"mbpp(\d+)\.npz$")
    es = sorted(json.loads(l)["task_id"] for l in open(f"{H}/EvalSuite/data/items/dec_mbpp.jsonl"))
    sb_mb = ids(f"{H}/Sandbox98/data/mbpp", r"mbpp_(\d+)\.npz$")
    mbpp = [json.loads(l)["task_id"] for l in open(f"{H}/TrainLab/data/mbpp/mbpp.jsonl")]
    assert len(d16w) == 24 and len(sb_he) == 60 and set(es) <= set(d1102) and set(sb_mb) <= set(d1102), "dev sets moved"
    he_fresh = [i for i in range(164) if i not in set(d16w)]
    he_strict = [i for i in he_fresh if i not in set(sb_he)]
    pool = sorted(i for i in mbpp if 11 <= i <= 510 and i not in set(d1102))
    mb_fresh = sorted(int(x) for x in np.array(pool)[np.random.default_rng(SEED).permutation(len(pool))[:N_MBPP]])
    keys = lambda f: [json.loads(l)["key"] for l in open(f"{H}/EvalSuite/data/items/{f}.jsonl")]
    ev = {}
    for i, (st, n) in enumerate((("ifeval", 60), ("ifbench", 61))):
        k = sorted(keys(f"dec_{st}"))
        assert len(k) == {"ifeval": 100, "ifbench": 300}[st], f"dec_{st} moved"
        rows = {json.loads(l)["key"] for r in ("ref2tnax", "ref2tnaxn.r6") for l in open(f"{H}/EvalSuite/results/{r}/decode.jsonl")}
        k = [x for x in k if x in rows]                            # the ones with an exact answer to cross-check
        ev[st] = sorted(np.array(k)[np.random.default_rng([SEED, i]).permutation(len(k))[:n]].tolist())
    ev["agent"] = sorted(set(keys("dec_agent")))
    assert len(ev["agent"]) == 29 and len(ev["ifeval"]) == 60 and len(ev["ifbench"]) == 61, {k: len(v) for k, v in ev.items()}
    return {
        "rule": "docs/PREREG_certify98_2026-11-15.md amendment 2 (drawn Sun 2026-11-15 before any candidate)",
        "seed": SEED,
        "heplus_fresh": he_fresh, "heplus_fresh_strict": he_strict,
        "mbpp_fresh": mb_fresh, "mbpp_pool_size": len(pool),
        "excluded": {"heplus_d16w_125b_dev": d16w, "heplus_sandbox98_2b_dev": sb_he,
                     "mbpp_trainlab_d1102mbpp_captured": d1102, "mbpp_evalsuite_dec_mbpp": es, "mbpp_sandbox98": sb_mb},
        "sources": {"mbpp": "~/TrainLab/data/mbpp/mbpp.jsonl (974; test split = task_ids 11-510)",
                    "heplus": "HumanEval/0-163 (HumanEval+ shares the prompts)"},
        "everyday_fresh": ev,
        "everyday_sources": {"agent": "data/items/dec_agent.jsonl (all 29)", "ifeval": "dec_ifeval.jsonl (60 of 100; seed [SEED, 0])",
                             "ifbench": "dec_ifbench.jsonl (61 of the 150 with exact rows in ref2tnaxn.r6; seed [SEED, 1])"},
    }


def main(argv):
    m = build()
    txt = json.dumps(m, indent=1, sort_keys=True) + "\n"
    if "--check" in argv:
        same = os.path.exists(OUT) and open(OUT).read() == txt
        print(f"manifest rebuild {'matches' if same else 'DIFFERS FROM'} {OUT}")
        return 0 if same else 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(txt)
    print(f"HE+ fresh {len(m['heplus_fresh'])} (strict {len(m['heplus_fresh_strict'])}); MBPP pool {m['mbpp_pool_size']} -> "
          f"fresh {len(m['mbpp_fresh'])} (seed {SEED}); everyday {sum(len(v) for v in m['everyday_fresh'].values())} "
          f"({', '.join(f'{k} {len(v)}' for k, v in m['everyday_fresh'].items())}); written {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
