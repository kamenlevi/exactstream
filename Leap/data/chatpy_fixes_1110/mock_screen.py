"""a stand-in for the decode loop: same globals and call pattern (ED.load per key, SV.new_prompt, SV.committed per check)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))
CAP = int(sys.argv[2]); HARD = 600; PAUSE = "/nonexistent"
from evalsuite import data as ED
import session_vocab as _SVM
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(os.environ["MOCK_STORE"])
EOS = {tk.convert_tokens_to_ids("<|im_end|>")}
SV = _SVM.SessionVocab()
SETOF = {"multi": "multi", "humaneval": "humaneval", "math500": "math"}
PICK = tuple((SETOF[k.split("/")[0]], k) for k in os.environ["R57_KEYS"].split(","))
RUNS, TOK = [], {}
def rule(item):
    return "chat"                                  # like the engine: a plain request reads as chat
print("mock loop loaded", flush=True)
it = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")      # warm-up (its commits must not show)
SV.new_prompt(it["prompt_ids"]); SV.committed([11, 12, 13])
for n_p, (s, key) in enumerate(PICK):
    it = next(x for x in ED.load("dec_" + s) if x["key"] == key)
    ids = it["prompt_ids"]
    typ = rule(it); drafter = "mtp" if typ == "chat" else "dfl"
    long_ = "long" in it["user"]
    reply = tk.encode(("word " * 400 if long_ else f"Mock answer to: {it['user']} (prompt {len(ids)} tokens). Ünïcödé ✓ done."), add_special_tokens=False)
    if "You have no tools" in tk.decode(ids):            # chat.py's built-in system line reached the prompt
        reply = tk.encode("SYS-OK ", add_special_tokens=False) + reply
    reply = reply + [next(iter(EOS))]
    t0 = time.perf_counter(); SV.new_prompt(ids); time.sleep(0.2)
    out = [reply[0]]; SV.committed([reply[0]]); t1 = time.perf_counter(); j = 1
    while len(out) < CAP and out[-1] not in EOS:
        if time.perf_counter() - t0 > HARD: break
        time.sleep(0.03 if long_ else 0.01)
        new = reply[j:j + 3]; j += 3
        if not new: break
        for q, x in enumerate(new):
            if x in EOS: new = new[:q + 1]; break
        out += new; SV.committed(new)
    dec = time.perf_counter() - t1
    RUNS.append(dict(key=key, n=min(len(out), CAP), tps=min(len(out), CAP) / dec, pfs=0.2, mem=[0, 0, 0, 0, 0, 0, 31.25], bad="", drafter=drafter, kept=2.5 if drafter == "mtp" else 5.5))
    TOK[key] = out[:CAP]
    print(f"mock run {key}: {len(out)} tokens", flush=True)
