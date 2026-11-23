"""CHAN scorer + verdict (results/LEVERS_A.md section CHAN; Lunatic's lines, pre-registered before data, with Lunatic's
12:0x note: when the code channel's and the spliced answer's scores differ, the verdict reads from the spliced answer).
Pass@1 with EvalSuite's own scorers, imported read-only (run with ~/EvalSuite/.venv/bin/python -I -B; nothing is written
under ~/EvalSuite): HE+ = evalsuite/decode_report.score_humaneval (evalplus base + plus; humanevalx as HumanEval+),
MBPP = tools/mbpp_score.score (the 3 asserts in an isolated child, 10 s), as MoonShots' J(CC) scorer.
Coupled arm = MoonShots' J(CC) exact arm (week98/jcc/arm0.jsonl + its scores), or results/chan/plain_he / plain_mbpp.jsonl when
the identity check failed (--coupled plain). Arm = results/chan/chan_he.jsonl + chan_mbpp.jsonl ('text' = the code channel, 'spliced').
Per-token classes of the coupled arm (check-ending commits): results/chan/arm0_cls.json (sb/chan_cls.py).
LINES (per answer form, pooled over the keys both arms hold):
  HOLDS  pooled drop <= 1.0 pt AND exact McNemar p > 0.2 AND neither set loses with p < 0.05
  FAILS  lower with p < 0.05 pooled OR pooled drop > 3 pt
  BETWEEN otherwise
  speed claim KILL if the code channel's check-ending misses per code (+ string) token > 1.5x the coupled decode's
usage: ~/EvalSuite/.venv/bin/python -I -B sb/chan_score.py [--coupled arm0|plain]"""
import argparse, difflib, json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/EvalSuite")); sys.path.insert(0, H("~/EvalSuite/tools")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402  (read-only)

RD = os.environ.get("CHAN_DIR", H("~/Sandbox98/results/chan"))
J = H("~/MoonShots/week98/jcc")
CODEC = ("code", "string")


def load(p):
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def grp(r):
    return "mbpp" if r["set"] == "mbpp" else "humaneval"


def score_all(rows):
    from evalsuite import decode_report as DR
    import mbpp_score as MS
    its = {}
    for f in ("dec_humaneval", "dec_humanevalx"):
        for l in open(H(f"~/EvalSuite/data/items/{f}.jsonl")):
            it = json.loads(l); it = dict(it, set="humaneval"); its[it["key"]] = it
    he = DR.score_humaneval({k: r for k, r in rows.items() if grp(r) == "humaneval"}, its)
    mb = MS.score({k: r for k, r in rows.items() if grp(r) == "mbpp"}, MS.items())
    return {k: bool(v[0]) for k, v in list(he.items()) + list(mb.items())}


def mcnemar(b, c):
    from evalsuite.report import mcnemar_p
    return mcnemar_p(b, c)


def qline(ref, arm, keys):
    n = len(keys)
    b = sum(1 for k in keys if arm[k] and not ref[k]); c = sum(1 for k in keys if ref[k] and not arm[k])
    pr, pa = 100 * sum(ref[k] for k in keys) / max(n, 1), 100 * sum(arm[k] for k in keys) / max(n, 1)
    return dict(n=n, coupled=round(pr, 2), arm=round(pa, 2), delta_pt=round(pa - pr, 2), arm_only=b, coupled_only=c,
                p=round(mcnemar(b, c), 4))


def verdict(ref, arm, keys, ex):
    out = {"pooled": qline(ref, arm, keys)}
    for g in ("humaneval", "mbpp"):
        out[g] = qline(ref, arm, [k for k in keys if grp(ex[k]) == g])
    po = out["pooled"]
    set_loss = any(out[g]["delta_pt"] < 0 and out[g]["p"] < 0.05 for g in ("humaneval", "mbpp"))
    if (po["delta_pt"] < 0 and po["p"] < 0.05) or po["delta_pt"] < -3:
        out["verdict"] = "FAILS"
    elif po["delta_pt"] >= -1.0 and po["p"] > 0.2 and not set_loss:
        out["verdict"] = "HOLDS"
    else:
        out["verdict"] = "BETWEEN"
    return out


def code_lines(text):
    """the answer's code with '#' comments and docstrings removed (MoonShots' tracker, character by character), prose
    outside the fences dropped, trailing blanks stripped, blank lines dropped"""
    p = SpanTracker({}); keep = []
    for ch in text:
        before = p.region == "C" and (p.mode == "K" or (p.mode == "T" and p.doc))
        reg0 = p.region
        p.step(ch)
        now = p.region == "C" and (p.mode == "K" or (p.mode == "T" and p.doc))
        if reg0 == "C" and p.region == "C" and not before and not now:
            keep.append(ch)
        elif reg0 == "C" and p.region == "C" and ch == "\n":
            keep.append(ch)
    s = "".join(keep)
    # a docstring's opening quotes are consumed before the tracker enters T: drop lines that are only quotes now
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() and ln.strip() not in ('"""', "'''", '""', "''")]


def sweeps_of(r):
    return len(r["checks"]) + (1 if r.get("cut_check") else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupled", default="arm0", choices=("arm0", "plain"))
    ap.add_argument("--out", default=os.path.join(RD, "chan_verdict.json"))
    a = ap.parse_args()
    ex = load(os.path.join(J, "arm0.jsonl")) if a.coupled == "arm0" else {**load(os.path.join(RD, "plain_he.jsonl")),
                                                                           **load(os.path.join(RD, "plain_mbpp.jsonl"))}
    arm = {**load(os.path.join(RD, "chan_he.jsonl")), **load(os.path.join(RD, "chan_mbpp.jsonl"))}
    cache_p = os.path.join(RD, "scores.json")
    sc = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    if a.coupled == "arm0":
        sc["coupled"] = json.load(open(os.path.join(J, "scores.json")))["arm0"]
    else:
        todo = {k: r for k, r in ex.items() if k not in sc.get("coupled_plain", {})}
        if todo:
            sc.setdefault("coupled_plain", {}).update(score_all(todo))
        sc["coupled"] = sc["coupled_plain"]
    for form, field in (("code", "text"), ("spliced", "spliced")):
        todo = {k: dict(r, text=r[field]) for k, r in arm.items() if k not in sc.get(form, {})}
        if todo:
            sc.setdefault(form, {}).update(score_all(todo))
    json.dump(sc, open(cache_p, "w"), indent=0)
    ref = sc["coupled"]
    keys = sorted(set(arm) & set(ex) & set(sc["code"]) & set(sc["spliced"]) & set(ref))
    res = {"coupled": a.coupled, "keys": len(keys)}
    res["code_channel"] = verdict(ref, sc["code"], keys, ex)
    res["spliced"] = verdict(ref, sc["spliced"], keys, ex)
    differ = [k for k in keys if sc["code"][k] != sc["spliced"][k]]
    res["code_vs_spliced_differ"] = differ
    res["quality_verdict"] = res["spliced"]["verdict"]          # Lunatic 12:0x: the spliced answer is what a user gets
    # speed claim: check-ending misses per code (+ string) token
    cls0 = json.load(open(os.path.join(RD, "arm0_cls.json"))) if a.coupled == "arm0" else None
    sp = {}
    for g in ("humaneval", "mbpp", "pooled"):
        ks = [k for k in keys if g == "pooled" or grp(ex[k]) == g]
        if cls0 is not None:
            cm = sum(sum(1 for c in cls0[k]["end_cls"] if c in CODEC) for k in ks)
            ct = sum(sum(cls0[k]["cls"].get(c, 0) for c in CODEC) for k in ks)
            split = {c: sum(sum(1 for x in cls0[k]["end_cls"] if x == c) for k in ks) for c in ("comment", "docstring", "code", "string", "prose")}
        else:
            cm = sum(sum(1 for c in ex[k]["end_cls"] if c in CODEC) for k in ks)
            ct = sum(sum(ex[k]["cls"].get(c, 0) for c in CODEC) for k in ks)
            split = {c: sum(sum(1 for x in ex[k]["end_cls"] if x == c) for k in ks) for c in ("comment", "docstring", "code", "string", "prose")}
        am = sum(sum(1 for c in arm[k]["end_cls"] if c in CODEC) for k in ks)
        at = sum(sum(arm[k]["cls"].get(c, 0) for c in CODEC) for k in ks)
        rc, ra = cm / max(ct, 1), am / max(at, 1)
        nm = max(sum(split.values()), 1)
        # sweeps: coupled checks vs the channel critical path; live channels and rows per sweep
        cpl, chn, live_all = [], [], []
        for k in ks:
            r = arm[k]; ts = sweeps_of(r)
            ends = [ts] + [s["check"] + 1 + len(s["branch_checks"]) + (1 if s.get("branch_cut") else 0)
                           for s in r["slots"] if "branch_checks" in s]
            cp = max(ends)
            cpl.append(sweeps_of(ex[k])); chn.append(cp)
            live = np.zeros(cp, int); live[:ts] += 1
            for s in r["slots"]:
                if "branch_checks" in s:
                    b0 = s["check"] + 1; b1 = b0 + len(s["branch_checks"]) + (1 if s.get("branch_cut") else 0)
                    live[b0:b1] += 1
            live_all += live.tolist()
        la = np.array(live_all) if live_all else np.zeros(1)
        sp[g] = dict(coupled_misses_per_code_token=round(rc, 4), chan_misses_per_code_token=round(ra, 4),
                     ratio=round(ra / max(rc, 1e-9), 3), coupled_code_tokens=ct, chan_code_tokens=at,
                     coupled_miss_split={c: round(v / nm, 3) for c, v in split.items()},
                     coupled_sweeps=int(sum(cpl)), channel_sweeps=int(sum(chn)),
                     fewer_sweeps=round(sum(cpl) / max(sum(chn), 1), 3),
                     live_channels_mean=round(float(la.mean()), 2), live_channels_p90=float(np.percentile(la, 90)),
                     rows_per_sweep_mean=round(32 * float(la.mean()), 1), rows_per_sweep_p90=float(32 * np.percentile(la, 90)))
    res["speed"] = sp
    res["speed_claim"] = "KILL" if sp["pooled"]["ratio"] > 1.5 else "stands"
    # token agreement (code only), comment consistency, indentation carry, slots
    ag = {}
    for g in ("humaneval", "mbpp", "pooled"):
        ks = [k for k in keys if g == "pooled" or grp(ex[k]) == g]
        same, ratios, firsts = 0, [], []
        for k in ks:
            A, B = code_lines(ex[k]["text"]), code_lines(arm[k]["text"])
            same += int(A == B)
            ratios.append(difflib.SequenceMatcher(a=A, b=B, autojunk=False).ratio())
            fd = next((i for i, (x, y) in enumerate(zip(A, B)) if x != y), None if len(A) == len(B) else min(len(A), len(B)))
            if fd is not None:
                firsts.append(fd / max(len(A), 1))
        sl = [s for k in ks for s in arm[k]["slots"] if "kind" in s]
        fl = [s for s in sl if s["full_line"] and s["closed"]]
        ag[g] = dict(answers=len(ks), identical_code=same, identical_share=round(same / max(len(ks), 1), 3),
                     line_match_ratio_mean=round(float(np.mean(ratios)), 3) if ratios else None,
                     first_diff_at_code_frac_median=round(float(np.median(firsts)), 3) if firsts else None,
                     slots=len(sl), slots_per_answer=round(len(sl) / max(len(ks), 1), 2),
                     slots_by_kind={kd: sum(1 for s in sl if s["kind"] == kd) for kd in ("#", "doc")},
                     branch_caps=sum(1 for s in sl if not s["closed"]),
                     comment_consistency=round(sum(s["agree"] for s in sl if s["closed"]) / max(sum(1 for s in sl if s["closed"]), 1), 3),
                     comment_consistency_by_kind={kd: round(sum(s["agree"] for s in sl if s["closed"] and s["kind"] == kd)
                                                            / max(sum(1 for s in sl if s["closed"] and s["kind"] == kd), 1), 3) for kd in ("#", "doc")},
                     indent_carry=sum(1 for s in fl if s["indent_trunk"] != s["indent_probe"] and s["probe"]),
                     indent_carry_share=round(sum(1 for s in fl if s["indent_trunk"] != s["indent_probe"] and s["probe"]) / max(len(fl), 1), 3),
                     cap_hits=sum(1 for k in ks if arm[k]["stop"] == "cap"), coupled_cap_hits=sum(1 for k in ks if ex[k]["stop"] == "cap"),
                     trunk_tokens=sum(arm[k]["n"] for k in ks), branch_tokens=sum(s["branch_n"] for s in sl),
                     coupled_tokens=sum(ex[k]["n"] for k in ks))
    res["agreement"] = ag
    res["discordant_spliced"] = [dict(key=k, coupled=ref[k], spliced=sc["spliced"][k], code=sc["code"][k],
                                      slots=len(arm[k]["slots"])) for k in keys if ref[k] != sc["spliced"][k]]
    json.dump(res, open(a.out, "w"), indent=1)
    q = res["spliced"]; c_ = res["code_channel"]; s_ = sp["pooled"]; g_ = ag["pooled"]
    print(f"coupled = {a.coupled}; {len(keys)} keys; code-channel vs spliced scores differ on {len(differ)} keys")
    for nm_, v in (("spliced", q), ("code channel", c_)):
        print(f"  {nm_}: pooled {v['pooled']['coupled']:.2f} -> {v['pooled']['arm']:.2f} ({v['pooled']['delta_pt']:+.2f} pt,"
              f" {v['pooled']['arm_only']} / {v['pooled']['coupled_only']}, p {v['pooled']['p']}); HE+ {v['humaneval']['delta_pt']:+.2f}"
              f" (p {v['humaneval']['p']}), MBPP {v['mbpp']['delta_pt']:+.2f} (p {v['mbpp']['p']}): {v['verdict']}")
    for g in ("humaneval", "mbpp", "pooled"):
        x = sp[g]; y = ag[g]
        print(f"  {g}: misses / code token {x['coupled_misses_per_code_token']} -> {x['chan_misses_per_code_token']} (x{x['ratio']});"
              f" sweeps {x['coupled_sweeps']} -> {x['channel_sweeps']} (x{x['fewer_sweeps']} fewer); live channels {x['live_channels_mean']}"
              f" (p90 {x['live_channels_p90']:.0f}), rows / sweep {x['rows_per_sweep_mean']} (p90 {x['rows_per_sweep_p90']:.0f});"
              f" coupled miss split {x['coupled_miss_split']}")
        print(f"     code identical {y['identical_code']} / {y['answers']} (line match {y['line_match_ratio_mean']}); slots {y['slots']}"
              f" ({y['slots_per_answer']} / answer, {y['slots_by_kind']}), caps {y['branch_caps']}; comment consistency {y['comment_consistency']}"
              f" {y['comment_consistency_by_kind']}; indent carry {y['indent_carry']} ({y['indent_carry_share']} of full-line slots)")
    print(f"VERDICT QUALITY: {res['quality_verdict']} (spliced; code channel {c_['verdict']}) -- pooled {q['pooled']['delta_pt']:+.2f} pt"
          f" (p {q['pooled']['p']}), HE+ {q['humaneval']['delta_pt']:+.2f}, MBPP {q['mbpp']['delta_pt']:+.2f}", flush=True)
    print(f"VERDICT SPEED-CLAIM: {res['speed_claim']} -- misses per code token x{s_['ratio']} (line 1.5); sweeps x{s_['fewer_sweeps']} fewer;"
          f" code identical {g_['identical_share']}", flush=True)


if __name__ == "__main__":
    main()
