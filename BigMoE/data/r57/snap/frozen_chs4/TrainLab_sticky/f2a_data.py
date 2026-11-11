"""F2a step 1 (MTP Head, Sun 11-08; CPU, no GPU): the chat corpus -> token sequences for the night's capture.
Inputs (untrusted downloads, parsed as data only, read with python -I): ~/TrainLab/ext/chatcorpus/
  ultrachat_200k_train_sft_00000.parquet  HuggingFaceH4/ultrachat_200k @8049631, MIT (English multi-turn)
  aya_train.parquet                        CohereForAI/aya_dataset @f9ea045, Apache-2.0 (65 languages, single turn)
Rendering = the 125B's chat template (models/bare/chat_template.jinja) with thinking off: every assistant turn starts
'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' and ends '<|im_end|>\\n'; no system message (as EvalSuite's IFEval
prompts, checked by decoding one). Targets (asst = 1): every assistant token incl. its <|im_end|>; the 125B's labels come
from the capture (its own top-32 at each position), so the corpus text is context only.
Mix: UltraChat : Aya ~ 2 : 1 in tokens; Aya restricted to the multi set's languages (bg de fr es ru zh ja ar) + it pt nl pl
tr uk ko vi id. Sequences <= 2048 tokens (multi-turn dialogues cut at a turn boundary). Shuffled, seed 11.
-> data/f2a/seqs.npz (ids int32 concatenated, asst uint8, off int64 [n + 1], src uint8 0 = UltraChat 1 = Aya)"""
import os, sys, json
import numpy as np

SRC = os.path.expanduser("~/TrainLab/ext/chatcorpus")
OUT = os.path.expanduser("~/TrainLab/data/f2a"); os.makedirs(OUT, exist_ok=True)
MAXLEN = 2048
TARGET_TOK = int(float(sys.argv[1])) if len(sys.argv) > 1 else 8_000_000
LANGS = {"bul", "deu", "fra", "spa", "rus", "zho", "jpn", "arb", "ita", "por", "nld", "pol", "tur", "ukr", "kor", "vie", "ind",
         "bg", "de", "fr", "es", "ru", "zh", "ja", "ar", "it", "pt", "nl", "pl", "tr", "uk", "ko", "vi", "id"}

import pyarrow.parquet as pq
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
assert IMS is not None and IME is not None, "chat special tokens missing"


def enc(s):
    return tok.encode(s, add_special_tokens=False).ids


H_USER = [IMS] + enc("user\n")
H_ASST = [IMS] + enc("assistant\n<think>\n\n</think>\n\n")
NL = enc("\n")


def render(turns):
    """turns: [(role, text)] alternating user / assistant -> ids, asst mask; cut at a turn boundary <= MAXLEN"""
    ids, m = [], []
    for role, text in turns:
        if role == "user":
            seg = H_USER + enc(text) + [IME] + NL; ms = [0] * len(seg)
        elif role == "assistant":
            body = enc(text) + [IME]
            seg = H_ASST + body + NL; ms = [0] * len(H_ASST) + [1] * len(body) + [0] * len(NL)
        else:
            continue
        if len(ids) + len(seg) > MAXLEN:
            break
        ids += seg; m += ms
    return ids, m


rng = np.random.default_rng(11)
seqs = []
t = pq.read_table(os.path.join(SRC, "ultrachat_200k_train_sft_00000.parquet"), columns=["messages"])
uc = t.column("messages").to_pylist()
del t
for i in rng.permutation(len(uc)):
    turns = [(str(x.get("role")), str(x.get("content") or "")) for x in uc[i] if isinstance(x, dict)]
    if not turns or turns[0][0] != "user":
        continue
    ids, m = render(turns)
    if sum(m) >= 16:
        seqs.append((ids, m, 0))
    if sum(len(s[0]) for s in seqs) >= TARGET_TOK * 2 // 3:
        break
n_uc = len(seqs)
t = pq.read_table(os.path.join(SRC, "aya_train.parquet"), columns=["inputs", "targets", "language_code"])
ay = list(zip(t.column("inputs").to_pylist(), t.column("targets").to_pylist(), t.column("language_code").to_pylist()))
del t
tot = sum(len(s[0]) for s in seqs)
for i in rng.permutation(len(ay)):
    q, a, lc = ay[i]
    if str(lc) not in LANGS or not q or not a:
        continue
    ids, m = render([("user", str(q)), ("assistant", str(a))])
    if sum(m) >= 16:
        seqs.append((ids, m, 1)); tot += len(ids)
    if tot >= TARGET_TOK:
        break
order = rng.permutation(len(seqs))
seqs = [seqs[i] for i in order]
off = np.cumsum([0] + [len(s[0]) for s in seqs]).astype(np.int64)
np.savez(os.path.join(OUT, "seqs.npz"), ids=np.concatenate([np.asarray(s[0], np.int32) for s in seqs]),
         asst=np.concatenate([np.asarray(s[1], np.uint8) for s in seqs]), off=off, src=np.array([s[2] for s in seqs], np.uint8))
src = np.array([s[2] for s in seqs])
msg = (f"F2a corpus: {len(seqs)} sequences ({n_uc} UltraChat, {len(seqs) - n_uc} Aya), {off[-1]:,} tokens, "
       f"{int(sum(sum(s[1]) for s in seqs)):,} assistant targets; UltraChat tokens {int(sum(len(s[0]) for s in seqs if s[2] == 0)):,}, "
       f"Aya {int(sum(len(s[0]) for s in seqs if s[2] == 1)):,}; median length {int(np.median(np.diff(off)))}")
print(msg)
open(os.path.join(OUT, "seqs_read.txt"), "w").write(msg + "\n")
