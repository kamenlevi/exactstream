"""A stand-in for the decode loop that answers like a tool-using model (protocol tests, no GPU): decodes each prompt; if the
prompt lists tools and its last user turn is a request, the reply is a sentence plus ONE tool call in the model's own
format (Write for Claude Code, exec_command for Codex, else the first listed tool with no arguments); if the last turn holds
a <tool_response>, the reply is "Done." Same globals and call pattern as the engine loop (ED.load per key, SV.new_prompt,
SV.committed per check). MOCK_WORKDIR = the folder Write targets."""
import os, re, sys, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/chatmock/mods")
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
WD = os.environ.get("MOCK_WORKDIR", "/tmp")


def rule(item):
    return "chat"


def call(name, **kw):
    return (f"<tool_call>\n<function={name}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in kw.items())
            + "</function>\n</tool_call>")


def reply_for(text):
    last = text.rsplit("<|im_start|>user", 1)[-1]
    if "tasked with summarizing conversations" in text or "Respond with TEXT ONLY" in last:   # compaction: a summary
        return ("<analysis>The user asked to read two files; both were read.</analysis>\n<summary>\n1. Primary Request and "
                "Intent: read a.txt and b.txt (mock summary).\n2. Key Technical Concepts: none.\n</summary>")
    if "STOPSEQ" in last:                                  # scenario: a stop sequence in the answer
        return "alpha beta STOP gamma delta"
    if "SLOW" in last:                                     # scenario: a long, slow answer (pings, cancel, the queue)
        return "word " * 1500
    if "<tools>" not in text:
        return "Mock answer (no tools)."
    if "<tool_response>" in last:
        n = last.count("<tool_response>")
        img = "this model reads text only" in last
        return f"Done: {n} tool result(s)" + (", an image noted." if img else ".")
    conv = text.split("<|im_end|>", 1)[-1]                 # the conversation after the system turn (Claude Code sends
    if "TWO_READS" in conv:                                # its environment block as a later turn): two calls in one turn
        return "I'll read both files.\n\n" + call("Read", file_path=WD + "/a.txt") + "\n" + call("Read", file_path=WD + "/b.txt")
    if "READ_IMAGE" in conv:                               # scenario: a tool result that holds an image
        return call("Read", file_path=WD + "/pic.png")
    if '"name": "Write"' in text:
        return ("I'll create the file.\n\n<tool_call>\n<function=Write>\n<parameter=file_path>\n" + WD + "/hello.txt\n"
                "</parameter>\n<parameter=content>\nhello from the mock\n</parameter>\n</function>\n</tool_call>")
    if '"name": "exec_command"' in text:
        return ("I'll run a command.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\necho hello from the mock > hello.txt\n"
                "</parameter>\n</function>\n</tool_call>")
    m = re.search(r'"name": "([^"]+)"', text.split("<tools>", 1)[1])
    return f"<tool_call>\n<function={m.group(1)}>\n</function>\n</tool_call>" if m else "No tool."


print("mock tool loop loaded", flush=True)
it = next(x for x in ED.load("dec_math") if x["key"] == "gsm8k/708")
SV.new_prompt(it["prompt_ids"]); SV.committed([11, 12, 13])
for n_p, (s, key) in enumerate(PICK):
    it = next(x for x in ED.load("dec_" + s) if x["key"] == key)
    ids = it["prompt_ids"]
    reply = tk.encode(reply_for(tk.decode(ids)), add_special_tokens=False) + [next(iter(EOS))]
    t0 = time.perf_counter(); SV.new_prompt(ids); time.sleep(0.2)
    out = [reply[0]]; SV.committed([reply[0]]); t1 = time.perf_counter(); j = 1
    slow = "SLOW" in tk.decode(ids).rsplit("<|im_start|>user", 1)[-1]
    while len(out) < CAP and out[-1] not in EOS:
        time.sleep(0.05 if slow else 0.01)
        new = reply[j:j + 3]; j += 3
        if not new:
            break
        out += new; SV.committed(new)
    dec = time.perf_counter() - t1
    RUNS.append(dict(key=key, n=min(len(out), CAP), tps=min(len(out), CAP) / max(dec, 1e-6), pfs=0.2,
                     mem=[0, 0, 0, 0, 0, 0, 0.5], bad="", drafter="mtp", kept=2.5))
    TOK[key] = out[:CAP]
    print(f"mock run {key}: {len(out)} tokens", flush=True)
