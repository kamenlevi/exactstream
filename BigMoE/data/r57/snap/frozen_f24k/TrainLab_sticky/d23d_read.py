"""D-23d CHATSWITCH read-out (MTP Head, Mon 11-02; REPORT "D-23d CHATSWITCH: pre-registration" + amendment 1). CPU only:
reads the clean records of ~/TrainLab/data/d23c_runs.jsonl (head b5, C = D23_C, default 192; preset 45+ooo) for the D-23d keys
(D23_KEYS; D-23e: the same keys at D23_C=224),
run with D23_ORDER=sep (each arm its own pass over the set, the first arm alternating within each set).
PRIMARY: time-weighted chat gain G = sum(t dfl) / sum(t mtp15e) - 1 on the chat prompts (set multi / ifeval), every token equal
across arms: PASS G >= +10 % and geo-mean ratio >= 1.03; KILL G < +3 %; BETWEEN otherwise (PARTIAL below 19 prompts).
SECONDARY: agent goes on MTP only if its G >= +8 % and geo >= 1.00 (4 prompts). Checks: the order effect left after the
separation, w = sqrt(geo ratio where dfl ran first / geo ratio where mtp15e ran first) (D-23c', arms back to back: ~1.12-1.20),
and the type rule's labels. Information: the switch projected on D-23c''s mix (its dfl times per type as weights).
Output: data/d23d_read.txt (appended)."""
import json, os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
KEYS = os.environ["D23_KEYS"].split(",")
CC = int(os.environ.get("D23_C", "192"))
OLD = ("humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18,multi/bg1,humaneval/HumanEval/6,"
       "math500/test/intermediate_algebra/1422.json,ifeval/374,agent/edit01,multi/fr6,humaneval/HumanEval/79,"
       "math500/test/algebra/694.json,ifeval/3407,agent/bash19,multi/es9,humaneval/HumanEval/19,"
       "math500/test/number_theory/427.json,agent/edit02").split(",")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

R, AT, R192 = {}, {}, {}
for i, l in enumerate(open(os.path.join(TD, "d23c_runs.jsonl"))):
    d = json.loads(l)
    if d["head"] == "b5" and d.get("pre") == "45+ooo" and not (d["cut"] or d["paused"] or d["invalid"]):
        if d["C"] == 192:
            R192.setdefault(d["key"], {})[d["arm"]] = d             # D-23c's mix, the projection's weights (C 192)
        if d["C"] == CC:
            R.setdefault(d["key"], {})[d["arm"]] = d               # the last clean record per (key, arm) and its line
            AT.setdefault(d["key"], {})[d["arm"]] = i
both = lambda keys: [k for k in keys if {"dfl", "mtp15e"} <= set(R.get(k, {}))]
t = lambda k, a: R[k][a]["n"] / R[k][a]["tok_s"]
ratio = lambda k: R[k]["mtp15e"]["tok_s"] / R[k]["dfl"]["tok_s"]
def gain(keys, pol=lambda k: "mtp15e"):
    return sum(t(k, "dfl") for k in keys) / max(sum(t(k, pol(k)) for k in keys), 1e-9) - 1 if keys else float("nan")
geo = lambda keys: float(np.exp(np.mean([np.log(ratio(k)) for k in keys]))) if keys else float("nan")
first = lambda k: min(AT[k], key=AT[k].get)                       # the arm that ran first on this prompt
chat = lambda keys: [k for k in keys if k.split("/")[0] in ("ifeval", "multi")]
agent = lambda keys: [k for k in keys if k.startswith("agent/")]

F = both(KEYS)
eq = all(R[k]["mtp15e"]["out"] == R[k]["dfl"]["out"] for k in F)
say(f"D-23{'d' if CC == 192 else 'e'} CHATSWITCH read-out (C {CC}, head b5, cut t 0.3, preset 45+ooo, arms in separate passes): {len(F)} of {len(KEYS)} "
    f"prompts clean in both arms; every token equal across arms: {eq}")
for k in F:
    say(f"  {k:14s} rule {R[k]['dfl']['type']:5s} {R[k]['dfl']['n']:3d} tok, first {first(k):6s}: dfl {R[k]['dfl']['tok_s']:5.1f} -> "
        f"mtp15e {R[k]['mtp15e']['tok_s']:5.1f} tok/s ({100 * (ratio(k) - 1):+5.1f} %); kept {R[k]['dfl']['kept']:.2f} of "
        f"{R[k]['dfl']['rows']:.1f} -> {R[k]['mtp15e']['kept']:.2f} of {R[k]['mtp15e']['rows']:.1f}; misses {R[k]['dfl']['miss']:.0f} / "
        f"{R[k]['mtp15e']['miss']:.0f}")
FC, FA = chat(F), agent(F)
G, g = gain(FC), geo(FC)
v = ("FAIL (a token differs across arms)" if not eq else "PASS" if G >= 0.10 and g >= 1.03 else "KILL" if G < 0.03 else "BETWEEN")
part = "" if len(FC) >= 19 else f" (PARTIAL: {len(FC)} of 19 chat prompts)"
say(f"  PRIMARY chat ({len(FC)}): time-weighted G {100 * G:+.1f} %, geo-mean ratio {g:.3f}, mtp15e faster on "
    f"{sum(ratio(k) > 1 for k in FC)} -> D-23d verdict (PASS G >= +10 % and geo >= 1.03, KILL G < +3 %): {v}{part}")
pol = lambda k: "mtp15e" if R[k]["dfl"]["type"] == "chat" else "dfl"
lab = ", ".join(f"{x} {sum(R[k]['dfl']['type'] == x for k in FC)}" for x in ("chat", "code", "math", "agent"))
say(f"    routed by the type rule (a chat prompt the rule calls code / math stays on dfl): G {100 * gain(FC, pol):+.1f} %; "
    f"rule labels {lab}")
for nm, ks in (("multi", [k for k in FC if k.startswith("multi/")]), ("ifeval", [k for k in FC if k.startswith("ifeval/")])):
    say(f"    {nm} ({len(ks)}): G {100 * gain(ks):+.1f} %, geo-mean {geo(ks):.3f}")
Ga, ga = gain(FA), geo(FA)
on = Ga >= 0.08 and ga >= 1.0
say(f"  SECONDARY agent ({len(FA)}): G {100 * Ga:+.1f} %, geo-mean {ga:.3f} -> agent "
    f"{'joins the switch (MTP)' if on else 'stays on DFlash2'} (line: G >= +8 % and geo >= 1.00)")
for nm, ks in (("chat", FC), ("agent", FA), ("all", F)):
    a_, b_ = [k for k in ks if first(k) == "dfl"], [k for k in ks if first(k) == "mtp15e"]
    if a_ and b_:
        say(f"  order check {nm}: geo ratio {geo(a_):.3f} where dfl ran first ({len(a_)}), {geo(b_):.3f} where mtp15e ran first "
            f"({len(b_)}) -> order effect w {np.sqrt(geo(a_) / geo(b_)):.3f} (1 = none; D-23c' back to back ~1.12-1.20)")
O = [k for k in OLD if {"dfl", "mtp15e"} <= set(R192.get(k, {}))]
if len(O) == len(OLD):
    T = {ty: sum(R192[k]["dfl"]["n"] / R192[k]["dfl"]["tok_s"] for k in O if R192[k]["dfl"]["type"] == ty) for ty in ("code", "math", "chat", "agent")}
    rc, ra = 1 + G, 1 + Ga
    sw = T["code"] + T["math"] + T["chat"] / rc + (T["agent"] / ra if on else T["agent"])
    say(f"  projection on D-23c''s mix (its C 192 dfl seconds per type as weights, {sum(T.values()):.1f} s): chat x{rc:.3f}"
        f"{f', agent x{ra:.3f}' if on else ', agent on dfl'} -> the switch {100 * (sum(T.values()) / sw - 1):+.1f} % whole set")
    Tm = sum(R192[k]["dfl"]["n"] / R192[k]["dfl"]["tok_s"] for k in O if k.startswith("multi/"))
    Ti = sum(R192[k]["dfl"]["n"] / R192[k]["dfl"]["tok_s"] for k in O if k.startswith("ifeval/"))
    FM, FI = [k for k in FC if k.startswith("multi/")], [k for k in FC if k.startswith("ifeval/")]
    if FM and FI and abs(Tm + Ti - T["chat"]) < 1e-6:
        sw2 = sw - T["chat"] / rc + Tm / (1 + gain(FM)) + Ti / (1 + gain(FI))
        say(f"    with multi and IFEval at their own rates (x{1 + gain(FM):.3f} / x{1 + gain(FI):.3f}): the switch "
            f"{100 * (sum(T.values()) / sw2 - 1):+.1f} % whole set; all chat English: "
            f"{100 * (sum(T.values()) / (sw - T['chat'] / rc + T['chat'] / (1 + gain(FI))) - 1):+.1f} %")
open(os.environ.get("D23_OUT", os.path.join(TD, "d23d_read.txt")), "a").write("\n".join(lines) + "\n\n")
