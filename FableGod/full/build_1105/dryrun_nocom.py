"""dryrun_nocom.py (Fable FULL, LOSSY350 T-B1 + the Friday script): CPU dry-run. No mlx, no model, nothing outside out/.
  1. the prompt splice of the patched screen (its three lines, verbatim) on the engine's 12 prompts;
  2. nocom_report.py on SYNTHETIC arm files: A = the stored REF2-T answers cut at 256 tokens, P = the same answers with
     their comments removed at text level; checks laid out by a simple rule (16 tokens per check in the copy run, 2-3 on
     comment tokens, 7 elsewhere; 95 ms each). It exercises the counters and the verdict, it is not a measurement;
  3. nocom_tests.py under EvalSuite's venv on two finished tasks (A and the comment-stripped stand-in);
  4. the patched fri_night.sh in a scratch root with a STUB runner (no engine): the arm order, the skip rules, the reports.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I dryrun_nocom.py"""
import importlib.util, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.expanduser("~")
OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
ENG = ["68", "16", "91", "6", "79", "19", "161", "17", "56", "141", "101", "63"]
KEYS = [f"humaneval/HumanEval/{n}" for n in ENG]
ITEMS = {r["key"]: r for r in map(json.loads, open(f"{H}/EvalSuite/data/items/dec_humaneval.jsonl"))}
TRUTH = {r["key"]: r["completion_ids"] for r in map(json.loads, open(f"{H}/EvalSuite/results/ref2t/decode.jsonl"))}
EOS = {248044, 248046}


def load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


NR = load("nocom_report"); RC = NR._load_core()
dec = lambda t: TOK.decode([int(t)])


def splice_test():
    NOCOM_END = TOK.token_to_id("<|im_end|>")
    NOCOM_IDS = TOK.encode("\n\nDo not write comments in the code.", add_special_tokens=False).ids
    for key in KEYS:
        ids = list(ITEMS[key]["prompt_ids"]); ids0 = list(ids)
        _j = len(ids) - 1 - ids[::-1].index(NOCOM_END)            # the patched screen's lines
        ids = ids[:_j] + NOCOM_IDS + ids[_j:]
        assert ids[:_j] == ids0[:_j] and ids[_j + len(NOCOM_IDS):] == ids0[_j:] and ids0.count(NOCOM_END) == 1
        assert TOK.decode(ids[_j:_j + len(NOCOM_IDS)]) == "\n\nDo not write comments in the code."
    tail = TOK.decode(ids[-(len(NOCOM_IDS) + 14):], skip_special_tokens=False)
    print(f"1. splice: {len(NOCOM_IDS)} tokens before the user turn's <|im_end|> on all 12 prompts, the rest of each prompt "
          f"unchanged; the last prompt now ends {tail!r}")


def strip_comments(ids):
    """a comment-free stand-in for an answer: comment-only lines dropped, inline comments cut (text level), re-encoded"""
    text = TOK.decode(ids, skip_special_tokens=True); sc = RC.CodeScan(); out = []; line = []; had_code = False; in_com = False
    for ch in text:
        was = bool(sc.code and sc.com); sc._ch(ch); now = bool(sc.code and sc.com)
        if ch == "\n":
            if in_com and not had_code:
                pass                                             # a comment-only line: gone with its newline
            else:
                out.append("".join(line).rstrip(" ") if in_com else "".join(line)); out.append("\n")
            line, had_code, in_com = [], False, False
            continue
        if now or was:
            in_com = True
            continue
        line.append(ch); had_code = had_code or not ch.isspace()
    out.append("".join(line))
    return TOK.encode("".join(out), add_special_tokens=False).ids + ([ids[-1]] if ids and ids[-1] in EOS else [])


def lay_checks(prompt, ans):
    """kept per check by a simple rule: 16 in the usable copy run, 2 / 3 on comment tokens, 7 elsewhere"""
    fl = NR.flags(RC, dec, list(prompt) + list(ans))[len(prompt):]; run = NR.copy_run(list(prompt), list(ans))
    kept, p, alt = [], 1, 0
    while p < len(ans):
        if run and run[0] + 8 <= p < run[1]:
            k = min(16, run[1] - p)
        elif fl[p] == 1:
            k = 2 + alt % 2; alt += 1
            k = next((i for i in range(k) if p + i >= len(ans) or fl[p + i] != 1), k) or 1
        else:
            k = next((i for i in range(1, 7) if p + i >= len(ans) or fl[p + i] == 1), 7)
        k = max(1, min(k, len(ans) - p)); kept.append(k); p += k
    return kept


def write_arm(d, tag, answers):
    runs, chks, toks = [], [], {}
    for key, ans in answers.items():
        kept = lay_checks(ITEMS[key]["prompt_ids"], ans); secs = 0.095 * len(kept)
        runs.append(dict(key=key, arm="gate+t", bad="", n=len(ans), tps=len(ans) / secs, tpsw=sum(kept[2:]) / max(0.095 * len(kept[2:]), 1e-9)))
        chks.append(dict(key=key, arm="gate+t", kept=kept)); toks[key] = {"gate+t": [ans]}
    for nm, obj in (("runs", runs), ("checks", chks), ("tokens", toks)):
        json.dump(obj, open(os.path.join(d, f"{tag}_224_{nm}.json"), "w"))


def report_test():
    d = os.path.join(OUTD, "dryfri", "data", "r57"); os.makedirs(d, exist_ok=True)
    A = {k: TRUTH[k][:256] for k in KEYS}; P = {k: strip_comments(TRUTH[k])[:256] for k in KEYS}
    for tag, ans in (("fqall", A), ("fqallb", A), ("fqallp", P), ("fqallpb", P)):
        write_arm(d, tag, ans)
    rep = NR.report("224", d, tests=os.path.join(OUTD, "dryfri", "no_tests.json"), items=ITEMS, dec=dec)
    print("2. nocom_report on SYNTHETIC arm files (A = stored REF2-T answers, P = the same with comments removed; checks by rule):")
    print(rep)
    nA = sum(len(a) for a in A.values()); nP = sum(len(p) for p in P.values())
    print(f"   (stand-in P: {nP} tokens against A's {nA} inside the cap, x{nP / nA:.3f})")
    return d


def tests_test():
    done = [k for k in KEYS if len(TRUTH[k]) < 256 and TRUTH[k][-1] in EOS][:2]
    d = os.path.join(OUTD, "dryfri", "t2"); os.makedirs(d, exist_ok=True)
    json.dump({k: {"gate+t": [TRUTH[k]]} for k in done}, open(os.path.join(d, "fqall_224_tokens.json"), "w"))
    json.dump({k: {"gate+t": [strip_comments(TRUTH[k])]} for k in done}, open(os.path.join(d, "fqallp_224_tokens.json"), "w"))
    r = subprocess.run([f"{H}/EvalSuite/.venv/bin/python", os.path.join(HERE, "nocom_tests.py"), "224", "--dir", d, "--out",
                        os.path.join(d, "nocom_tests.json"), "--tags", "fqall,fqallp"], capture_output=True, text=True, timeout=300)
    print(f"3. nocom_tests.py (EvalSuite's venv, {len(done)} finished tasks: {[k.split('/')[-1] for k in done]}): rc {r.returncode}; "
          + " | ".join(l for l in r.stdout.strip().splitlines()[-3:]) + (f" | stderr: {r.stderr.strip()[-300:]}" if r.returncode else ""))


def fri_test():
    root = os.path.join(OUTD, "fri_root"); shutil.rmtree(root, ignore_errors=True); os.makedirs(os.path.join(root, "scripts"))
    shutil.copy(f"{H}/BigMoE/scripts/r57_screen.py", os.path.join(root, "scripts", "r57_screen.py"))
    shutil.copy(f"{H}/BigMoE/scripts/r57_screen.py", os.path.join(root, "scripts", "r57_screen_rlx.py"))   # a stand-in copy
    for f in ("relax_delta.py", "relax_span.py"):
        open(os.path.join(root, "scripts", f), "w").write("# stub\n")
    open(os.path.join(root, "scripts", "c45_runner.sh"), "w").write(
        '#!/bin/bash\n# STUB runner for the dry-run: no engine, it only records what it was asked to run\n'
        'echo "$C45_ARMS | screen=$C45_SCREEN | keys=$(echo "$R57_KEYS" | tr "," "\\n" | wc -l | tr -d " ") | deadline=$C45_DEADLINE" >> "$C45_OUT/stub_arms.log"\n')
    patched = os.path.join(OUTD, "fri_night_full.sh")
    assert os.path.exists(patched), "run apply_fri.py first"
    res = []
    for label, env in (("default", {}), ("FRI_FULL=0", {"FRI_FULL": "0"}), ("applier missing", {"FRI_B1105": "/nonexistent"}),
                       ("FRI_CPROBE_T=1, no split", {"FRI_CPROBE_T": "1", "FRI_SPLIT90": "", "FRI_SPLIT99": ""})):
        shutil.rmtree(os.path.join(root, "data"), ignore_errors=True)
        e = dict(os.environ, FRI_ROOT=root, FRI_AVAIL="50", FRI_PY=f"{H}/BigMoE/.venv/bin/python")
        e.update({"FRI_B1105": HERE}); e.update(env)
        r = subprocess.run(["/bin/bash", patched], env=e, capture_output=True, text=True, timeout=120)
        arms = [l.split(" | ")[0].split(":")[0] + ("" if "frix" not in l else "*") for l in open(os.path.join(root, "data", "fri", "stub_arms.log"))]
        log = open(os.path.join(root, "data", "fri", "fri.log")).read()
        xs = os.path.exists(os.path.join(root, "data", "r57", "snap", "r57_screen_frix.py"))
        skipped = [l.split()[2] for l in log.splitlines() if "skipped: no switch" in l]
        envs = [l.split(" | ")[0] for l in open(os.path.join(root, "data", "fri", "stub_arms.log")) if "fcprobe" in l or "fqallp_" in l]
        res.append((label, r.returncode, arms, xs, skipped, envs, [l[15:] for l in log.splitlines() if "report rc" in l]))
    print("4. the patched fri_night.sh with a STUB runner in a scratch root (arms marked * run on XS, the patched screen):")
    for label, rc, arms, xs, skipped, envs, reps in res:
        print(f"   [{label}] rc {rc}; XS written {xs}; arms: {' '.join(arms)}; skipped by 'no switch': {skipped}; reports: {reps}")
        for e in envs:
            print(f"      {e[:230]}")


if __name__ == "__main__":
    splice_test()
    report_test()
    tests_test()
    fri_test()
