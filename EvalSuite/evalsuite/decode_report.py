"""Scoring of the decode sets (CPU; EvalSuite venv: IFEval checkers, evalplus). Rows for report.py.

Per set: accuracy of each model, the variant's paired delta vs the reference with an exact McNemar test, and answer
agreement with the reference (math/RULER: same extracted answer; IFEval/HumanEval+: same pass/fail) next to the floor
and the accepted build. Behaviour over all generations: response length ratio, hesitation-word rate, truncations,
loops."""
from __future__ import annotations
import json, os, re
import numpy as np
from .report import MARGINS, mcnemar_p, paired_ci, RES, ANCHOR, FLOOR

HESITATION = re.compile(r"\b(Wait|Hmm+|Actually|Alternatively|But wait|Let me re-?check|On second thought)\b")
CARD = ("ifbench", "gpqad", "lcb6")   # the model card's benchmarks (tools/card_items.py; PREREG_card_gates)
AGREE = ("multi", "agent")     # behaviour rows without gold: agreement with the reference, read against the floor's
SCRIPT = {"ar": r"[؀-ۿ]", "bg": r"[Ѐ-ӿ]", "ru": r"[Ѐ-ӿ]", "ja": r"[぀-ヿ一-鿿]",
          "zh": r"[一-鿿]"}   # non-Latin prompt languages: share of the answer's letters in the language's script


def load(name, thinking=False):
    p = os.path.join(RES, name, "decode_think.jsonl" if thinking else "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else None


def items():
    from . import data as D
    out = {}
    for s in ("math", "ifeval", "humaneval", "ruler", "mgsm") + CARD + AGREE:
        try:
            for it in D.load("dec_" + s):
                out[it["key"]] = it
        except FileNotFoundError:
            pass
    return out


def strip_think(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text


# ---------------------------------------------------------------------------------------------- per-set scoring
def boxed(text):
    """last \\boxed{...} content (balanced braces)"""
    i = text.rfind("\\boxed{")
    if i < 0:
        m = re.findall(r"Answer:\s*\$?([^\n$]+)", text)
        return m[-1].strip() if m else None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1]


def norm_math(a):
    if a is None:
        return None
    s = a.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace(" ", "").rstrip(".")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    try:
        return str(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s
    except ValueError:
        return s


def score_math(rows, its, set_="math"):
    out = {}
    for k, r in rows.items():
        if its.get(k, {}).get("set") != set_:
            continue
        a = norm_math(boxed(strip_think(r["text"])))
        out[k] = (a, float(a is not None and a == norm_math(its[k]["gold"])))
    return out


def score_ruler(rows, its):
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if not it or it["set"] != "ruler":
            continue
        t = strip_think(r["text"])
        ok = all(g in t for g in it["gold"])
        found = tuple(sorted(set(re.findall(r"\b\d{7}\b|\b[A-Z]{5}\b", t))))
        out[k] = (found, float(ok))
    return out


def score_ifeval(rows, its):
    from .ifeval import evaluation_lib as ev
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if not it or it["set"] != "ifeval":
            continue
        inp = ev.InputExample(key=k, instruction_id_list=it["instruction_id_list"], prompt=it["prompt"], kwargs=it["kwargs"])
        resp = strip_think(r["text"])
        s = ev.test_instruction_following_strict(inp, {it["prompt"]: resp})
        out[k] = (tuple(s.follow_instruction_list), float(s.follow_all_instructions))
    return out


_HE = {}


def score_humaneval(rows, its):
    # macOS refuses evalplus's RLIMIT_AS/RLIMIT_DATA cap ("current limit exceeds maximum limit"), which made every check
    # fail (0 % for every model on 2026-10-20); skip the memory cap, the per-test time limits stay
    os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import check_correctness, get_groundtruth
    if not _HE:
        probs = get_human_eval_plus()
        _HE["p"] = probs
        _HE["exp"] = get_groundtruth(probs, get_human_eval_plus_hash(), [])
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if not it or it["set"] != "humaneval":
            continue
        t = strip_think(r["text"])
        blocks = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
        code = max(blocks, key=len) if blocks else t
        imports = "\n".join(l for l in it["prompt"].splitlines() if l.startswith(("import ", "from ")))
        res = check_correctness("humaneval", 0, _HE["p"][it["task_id"]], imports + "\n" + code, _HE["exp"][it["task_id"]],
                                base_only=False, fast_check=True)
        ok = res["base"][0] == "pass" and res["plus"][0] == "pass"
        out[k] = (ok, float(ok))
    return out


def _ifb_kw(it):
    """a fresh copy without the None arguments: IFBench's run_eval.py grades strict first, which drops them in place, so
    its loose pass (which would choke on them) sees them dropped"""
    return [{a: v for a, v in kw.items() if v is not None} for kw in it["kwargs"]]


def score_ifbench(rows, its):
    """IFBench, the vendored official grader: prompt-level loose (the paper's metric; strict = score_ifbench_strict)"""
    from .ifbench import evaluation_lib as ev
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if not it or it["set"] != "ifbench":
            continue
        inp = ev.InputExample(key=k, instruction_id_list=it["instruction_id_list"], prompt=it["prompt"], kwargs=_ifb_kw(it))
        s = ev.test_instruction_following_loose(inp, {it["prompt"]: strip_think(r["text"])})
        out[k] = (tuple(s.follow_instruction_list), float(s.follow_all_instructions))
    return out


def score_ifbench_strict(rows, its):
    from .ifbench import evaluation_lib as ev
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if it and it["set"] == "ifbench":
            inp = ev.InputExample(key=k, instruction_id_list=it["instruction_id_list"], prompt=it["prompt"], kwargs=_ifb_kw(it))
            s = ev.test_instruction_following_strict(inp, {it["prompt"]: strip_think(r["text"])})
            out[k] = (tuple(s.follow_instruction_list), float(s.follow_all_instructions))
    return out


GPQA_ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")     # simple-evals ANSWER_PATTERN_MULTICHOICE


def score_gpqad(rows, its):
    """GPQA Diamond, simple-evals' scoring: the first ANSWER_PATTERN_MULTICHOICE match == the gold letter"""
    out = {}
    for k, r in rows.items():
        it = its.get(k)
        if not it or it["set"] != "gpqad":
            continue
        m = GPQA_ANSWER.search(strip_think(r["text"]))
        a = m.group(1) if m else None
        out[k] = (a, float(a == it["gold"]))
    return out


def score_lcb6(rows, its):
    """LiveCodeBench v6: LCB's run_test on the public + private tests (evalsuite/lcb/grade.py), pass@1 of the greedy row"""
    from .lcb import grade
    return grade.score({k: r["text"] for k, r in rows.items() if its.get(k, {}).get("set") == "lcb6"}, its)


SCORERS = {"math": score_math, "ruler": score_ruler, "ifeval": score_ifeval, "humaneval": score_humaneval,
           "mgsm": lambda rows, its: score_math(rows, its, "mgsm"),    # MGSM: dec_math's scoring (informational)
           "ifbench": score_ifbench, "gpqad": score_gpqad, "lcb6": score_lcb6}


def looped(ids, n=20, reps=3):
    """a 20-token n-gram repeated >= 3 times in the last 400 tokens"""
    t = ids[-400:]
    seen = {}
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return True
    return False


def rows(variant, thinking=False, ref="ref"):
    """ref: the decode run the variant is compared with ("ref" = the 4-bit reference; e.g. f24nat = the v3 natural arm)"""
    its = items()
    data = {n: load(n, thinking) for n in ("ref", FLOOR, ANCHOR, variant)}
    if ref != "ref":
        data["ref"] = load(ref, thinking)
    if data["ref"] is None or data[variant] is None:
        return [], {"missing": [n for n, d in data.items() if d is None]}
    dm = MARGINS["decode"]
    out, info = [], {}
    both = set(data["ref"]) & set(data[variant])
    keys_all = sorted(k for k in both if its.get(k, {}).get("set") in SCORERS)   # the scored sets (rules below)
    agree_rows = agreement_rows(variant, data, its, both)
    if not keys_all:
        return agree_rows, info
    agree_pool = {n: [] for n in data}
    for s, fn in SCORERS.items():
        keys = [k for k in keys_all if its.get(k, {}).get("set") == s]
        if not keys:
            continue
        sc = {n: fn({k: d[k] for k in keys if k in d}, its) for n, d in data.items() if d is not None}
        cr = np.array([sc["ref"][k][1] for k in keys])
        cv = np.array([sc[variant][k][1] for k in keys])
        b, c = int(((cr == 1) & (cv == 0)).sum()), int(((cr == 0) & (cv == 1)).sum())
        e, lo, hi = paired_ci(100 * (cv - cr))
        p = mcnemar_p(b, c)
        r = {"test": f"{s} (n={len(keys)})", "variant": f"{100 * cv.mean():.1f}% vs ref {100 * cr.mean():.1f}%",
             "vs_ref": f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}] pt, McNemar p={p:.2f} ({b}/{c})"}
        for n, col in ((FLOOR, "floor"), (ANCHOR, "accepted")):
            if n in sc and all(k in sc[n] for k in keys):
                r[col] = f"{100 * np.mean([sc[n][k][1] for k in keys]):.1f}%"
        if s in ("ifeval", "humaneval"):
            r["ok"] = e > dm["capability_drop_min_points"] and p >= dm["mcnemar_alpha"]
        elif s == "ruler":
            r["ok"] = (b - c) <= dm["ruler_net_items_lost_max"]
        elif s == "mgsm":                         # per language: variant / ref (/ floor) correct out of n
            info["mgsm_by_lang"] = {}
            for lang in sorted({its[k]["lang"] for k in keys}):
                kl = [k for k in keys if its[k]["lang"] == lang]
                cnt = lambda n: int(sum(sc[n][k][1] for k in kl)) if n in sc and all(k in sc[n] for k in kl) else None
                info["mgsm_by_lang"][lang] = f"{cnt(variant)} vs ref {cnt('ref')}" + (
                    f" (floor {cnt(FLOOR)})" if cnt(FLOOR) is not None else "") + f" of {len(kl)}"
        out.append(r)
        ra = {"test": f"{s} final-answer agreement with ref"}
        for n, col in ((variant, "variant"), (FLOOR, "floor"), (ANCHOR, "accepted")):
            if n in sc and all(k in sc[n] for k in keys):
                ag = [float(sc[n][k][0] == sc["ref"][k][0]) for k in keys]
                agree_pool[n] += ag
                ra[col] = f"{100 * np.mean(ag):.1f}% ({int(sum(ag))}/{len(ag)})"
        out.append(ra)
    av = np.array(agree_pool[variant])
    r = {"test": f"decode answer agreement with ref (n={len(av)})", "variant": f"{100 * av.mean():.1f}%"}
    for n, col in ((FLOOR, "floor"), (ANCHOR, "accepted")):
        if len(agree_pool[n]) == len(av):
            r[col] = f"{100 * np.mean(agree_pool[n]):.1f}%"
    if len(agree_pool[ANCHOR]) == len(av):
        e, lo, hi = paired_ci(100 * (av - np.array(agree_pool[ANCHOR])))
        r["vs_accepted"] = f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}] pt"
        r["ok"] = e >= -dm["answer_agreement_drop_max_points"]
    out.append(r)
    ident = {"test": "token-identical outputs (context only)"}
    for n, col in ((variant, "variant"), (FLOOR, "floor"), (ANCHOR, "accepted")):
        if data.get(n) is not None and all(k in data[n] for k in keys_all):
            ident[col] = f"{sum(data[n][k]['completion_ids'] == data['ref'][k]['completion_ids'] for k in keys_all)}/{len(keys_all)}"
    out.append(ident)
    # behaviour fingerprint over all generations
    lr = np.array([len(data[variant][k]["completion_ids"]) for k in keys_all], float)
    lref = np.array([len(data["ref"][k]["completion_ids"]) for k in keys_all], float)
    hv = sum(len(HESITATION.findall(data[variant][k]["text"])) for k in keys_all) / max(lr.sum(), 1) * 1000
    hr = sum(len(HESITATION.findall(data["ref"][k]["text"])) for k in keys_all) / max(lref.sum(), 1) * 1000
    ratio = lr.sum() / max(lref.sum(), 1)
    lo_, hi_ = dm["thinking_length_ratio"]
    out.append({"test": "decode length ratio vs ref", "variant": f"x{ratio:.3f}", "ok": lo_ <= ratio <= hi_})
    hrat = hv / hr if hr else float("nan")
    hlo, hhi = dm["hesitation_rate_ratio"]
    out.append({"test": "hesitation words / 1K tokens", "variant": f"{hv:.2f} vs ref {hr:.2f}",
                "ok": (max(hv, hr) < 1.0 and abs(hv - hr) < 0.5) or hlo <= hrat <= hhi})
    tv = sum(data[variant][k]["finish"] == "length" for k in keys_all)
    tr = sum(data["ref"][k]["finish"] == "length" for k in keys_all)
    loops_v = sum(looped(data[variant][k]["completion_ids"]) for k in keys_all)
    loops_r = sum(looped(data["ref"][k]["completion_ids"]) for k in keys_all)
    out.append({"test": "truncations / loops", "variant": f"{tv} / {loops_v} vs ref {tr} / {loops_r}",
                "ok": tv <= tr + 1 and loops_v <= loops_r + 1})
    info["generated_tokens"] = {"variant": int(lr.sum()), "ref": int(lref.sum())}
    for n in (variant, "ref", FLOOR):
        d = data.get(n)
        if d is not None:
            ks = [k for k in keys_all if k in d]
            info[f"loops_{n}"] = [k for k in ks if looped(d[k]["completion_ids"])]
            info[f"truncated_{n}"] = sorted(k.split("/")[0] for k in ks if d[k]["finish"] == "length")
            info[f"rows_{n}"] = len(ks)
    return out + agree_rows, info


def agreement_rows(variant, data, its, both):
    """multi / agent (no gold): token-identical continuations and token-sequence similarity to the reference, next to
    the floor's (same weights, other summation order: what 'unchanged' means here); multi also: the share of the
    answer's letters in the prompt's script for the non-Latin languages. Informational: no pre-registered rule."""
    from difflib import SequenceMatcher
    out = []
    for s in AGREE:
        ks = sorted(k for k in both if its.get(k, {}).get("set") == s)
        if not ks:
            continue
        ref = {k: data["ref"][k]["completion_ids"] for k in ks}

        def agree(n):
            d = data.get(n)
            if d is None or not all(k in d for k in ks):
                return None
            same = np.array([float(d[k]["completion_ids"] == ref[k]) for k in ks])
            sim = np.array([SequenceMatcher(None, ref[k], d[k]["completion_ids"], autojunk=False).ratio() for k in ks])
            return same, sim
        got = {n: agree(n) for n in (variant, FLOOR, ANCHOR)}
        cols = ((variant, "variant"), (FLOOR, "floor"), (ANCHOR, "accepted"))
        r = {"test": f"{s} (n={len(ks)}): identical to ref"}
        r.update({c: f"{int(got[n][0].sum())}/{len(ks)}" for n, c in cols if got[n] is not None})
        out.append(r)
        r = {"test": f"{s}: similarity to ref (token sequences)"}
        r.update({c: f"{got[n][1].mean():.3f}" for n, c in cols if got[n] is not None})
        if got[FLOOR] is not None:
            e, lo, hi = paired_ci(got[variant][1] - got[FLOOR][1])
            r["note"] = f"variant - floor {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired, n={len(ks)})"
        out.append(r)
        if s == "multi":
            def inlang(n):
                vals = []
                for k in ks:
                    pat = SCRIPT.get(its[k].get("lang"))
                    letters = [ch for ch in data[n][k]["text"] if ch.isalpha()] if pat else []
                    if letters:
                        vals.append(sum(bool(re.match(pat, ch)) for ch in letters) / len(letters))
                return 100 * float(np.mean(vals)) if vals else float("nan")
            r = {"test": "multi: answer letters in the prompt's script (ar/bg/ru/ja/zh)",
                 "variant": f"{inlang(variant):.1f}% vs ref {inlang('ref'):.1f}%"}
            if got[FLOOR] is not None:
                r["floor"] = f"{inlang(FLOOR):.1f}%"
            out.append(r)
    return out
