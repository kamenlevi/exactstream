> (a quote from the project owner, kept private)
Applies the corrections to a clone of the public main (argument: its path) and keeps MANIFEST.json's hashes true:
  - LIMITS.md lines 12-13: PF-D checked exact on 10K / 30K-token prompts; the long-prompt peaks (= staged/limits_pfd_erratum.patch,
    re-anchored on the public file, which has no 24 GB lines);
  - run.py --env help: PF-B is not exact past ~8K tokens (it said "all exact"); help text only, no code change;
  - CITATION.cff: version 1.0.1.
The papers' own erratum reaches docs/ separately (push_pkg.sh docs, the Editor's and Breakthrough's rebuilt site)."""
import hashlib, json, os, sys

T = sys.argv[1]
EDITS = {
    "LIMITS.md": [
        ("- The chat prefill path (PF-D) is on for `chat` at 45 GB only and was measured on chat-length prompts (37-101 tokens).\n",
         "- The chat prefill path (PF-D) is on for `chat` at 45 GB only. It was measured on chat-length prompts (37-101 tokens)\n"
         "  and checked exact on 10K and 30K-token prompts, where it shortens the first token by only 3-4 % and adds about 2 GB\n"
         "  to the lifetime peak.\n"),
        ("- Lifetime peaks go above 45.0 GB on long prompts (45.1-46.4 GB on 7.9K tokens).\n",
         "- Lifetime peaks go above the memory line on long prompts. At 45 GB: 45.1-46.4 GB on 7.9K tokens, and `chat` as shipped\n"
         "  (with PF-D) reaches 47.5 GB on 10K tokens and 48.6 GB on 30K. At 32 GB, `chat` reaches 34.0 GB on 10K tokens and\n"
         "  35.5 GB on 30K.\n")],
    "run.py": [
        ('"prefill paths: R57_PFLM=1 PF-A, R57_PFLM=3 PF-B, R57_PFD=1 PF-D; all exact)")',
         '"prefill paths: R57_PFLM=1 PF-A and R57_PFD=1 PF-D, both exact; R57_PFLM=3 PF-B, exact only up to about "\n'
         '                    "8K tokens)")')],
    "CITATION.cff": [("version: 1.0.0\n", "version: 1.0.1\n")],
}
man_p = os.path.join(T, "MANIFEST.json")
man = json.load(open(man_p))
for rel, subs in EDITS.items():
    p = os.path.join(T, rel)
    t = open(p, encoding="utf-8").read()
    for old, new in subs:
        assert t.count(old) == 1, f"{rel}: the old text occurs {t.count(old)} times"
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8").write(t)
    if rel.endswith(".py"):
        compile(t, rel, "exec")
    f = man["files"][rel]
    f["sha256"] = hashlib.sha256(t.encode("utf-8")).hexdigest()
    f["source"] = f["source"] + " + v1.0.1 correction (Leap release/staged/make_v101.py)"
    print(f"{rel}: {len(subs)} edit(s); MANIFEST sha256 updated")
json.dump(man, open(man_p, "w"), indent=1)
bad = [r for r, f in man["files"].items() if os.path.isfile(os.path.join(T, r)) and
       hashlib.sha256(open(os.path.join(T, r), "rb").read()).hexdigest() != f["sha256"]]
print("MANIFEST files whose sha256 != the tree's:", bad or "none")
