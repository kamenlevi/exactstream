"""Fit12 R-01 CONF prompts (Mon 26 Oct 2026; PLAN.md 'Rapid screens, batch 2'): c<L>_<i> = seeded shuffle i of the frozen
corpus /Users/kamen/Fit12/data/conf_corpus.txt (185 .py files of Breakthrough/scripts, SpecDecode, MoonShots/scripts, joined
by '\\n\\x1e\\n'; sha1 4c0f31fb1a11), cut to the length's character count (~3.0-3.1 characters per token on this code)."""
import os, random

CORPUS = os.path.expanduser("~/Fit12/data/conf_corpus.txt")
CHARS = {"2k": 7600, "8k": 26000, "16k": 49000, "32k": 98000}
HEAD = "Here are several Python scripts. Summarise what each does, section by section.\n\n"


def is_conf(name):
    return name[:1] == "c" and "_" in name and name[1:].split("_")[0] in CHARS


def conf_text(name):
    L, i = name[1:].split("_")
    segs = open(CORPUS).read().split("\n\x1e\n")
    random.Random(int(i)).shuffle(segs)
    return HEAD + "\n\n".join(segs)[:CHARS[L]] + "\n"
