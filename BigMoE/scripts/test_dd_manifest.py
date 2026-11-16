"""Desk test of scripts/dd_manifest.py (the CERTIFY98 dump manifest; no model, fake files):
verify_inputs (OK / a missing entry / a changed file), the "_run" fields after TOPK + T8 parts + merge + a T8(127) pass,
EvalSuite's rule (each file's args appear in _run.args), the refusals that leave the manifest byte-identical (another
checkpoint, another capture manifest, other dd_topk args, a changed module), and bigmoe_commits_seen.
  .venv/bin/python scripts/test_dd_manifest.py [workdir]   -> data/deepdraft/dd_manifest_test.json"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_manifest as DM

D = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(os.path.join(D, "cap", "heplus"), exist_ok=True)
os.makedirs(os.path.join(D, "out"), exist_ok=True)
res, fails = {}, []


def put(p, b):
    with open(p, "wb") as f:
        f.write(b)
    return p


def check(name, ok):
    res[name] = bool(ok)
    if not ok:
        fails.append(name)


def refused(name, fn):
    before = open(MP, "rb").read()
    try:
        fn()
        check(name, False)
    except RuntimeError as e:
        check(name, open(MP, "rb").read() == before)
        res[name + "_why"] = str(e)[:160]


caps = [put(os.path.join(D, "cap", "heplus", f"HumanEval_{i}.npz"), os.urandom(4096 + i)) for i in range(3)]
FM = os.path.join(D, "cap", "files_sha256.json")
json.dump({"manifest_sha256": "acdf3e62-test", "files": {f"heplus/{os.path.basename(p)}": DM.sha256_file(p) for p in caps}},
          open(FM, "w"))
fsha = DM.verify_inputs(FM, caps)
check("verify_ok_returns_manifest_sha", fsha == DM.sha256_file(FM))
extra = put(os.path.join(D, "cap", "heplus", "HumanEval_9.npz"), b"x")
try:
    DM.verify_inputs(FM, caps + [extra]); check("verify_refuses_missing_entry", False)
except RuntimeError as e:
    check("verify_refuses_missing_entry", "no entry" in str(e))
b0 = open(caps[1], "rb").read()
put(caps[1], b0 + b"!")
try:
    DM.verify_inputs(FM, caps); check("verify_refuses_changed_file", False)
except RuntimeError as e:
    check("verify_refuses_changed_file", "!=" in str(e))
put(caps[1], b0)

ck, ck2 = put(os.path.join(D, "dd_test.safetensors"), os.urandom(8192)), put(os.path.join(D, "dd_other.safetensors"), b"o")
MP = os.path.join(D, "out", "dumps_sha256.json")
o = {n: put(os.path.join(D, "out", n), os.urandom(2048)) for n in
     ("topk_test_heplus.npz", "tree_test_heplus.part0of2.npz", "tree_test_heplus.part1of2.npz", "tree_test_heplus.npz",
      "tree_test_heplus_n127k8.part0of2.npz")}
a_tk, a_t8, a_t127 = {"k": 8, "parts": 1}, {"k": 8, "nodes": 32, "parts": 2}, {"k": 8, "nodes": 127, "parts": 2}
DM.update(MP, [o["topk_test_heplus.npz"]], "dd_topk", a_tk, checkpoint=ck, files_sha=fsha)
DM.update(MP, [o["tree_test_heplus.part0of2.npz"]], "dd_tree", a_t8, part="0/2", checkpoint=ck, files_sha=fsha)
DM.update(MP, [o["tree_test_heplus.part1of2.npz"]], "dd_tree", a_t8, part="1/2", checkpoint=ck, files_sha=fsha)
DM.update(MP, [o["tree_test_heplus.npz"]], "dd_tree", a_t8)                       # the merge: no model, no inputs
DM.update(MP, [o["tree_test_heplus_n127k8.part0of2.npz"]], "dd_tree", a_t127, part="0/2", checkpoint=ck, files_sha=fsha)
M = json.load(open(MP))
R = M["_run"]
check("run_fields_present", all(R.get(k) not in (None, "", {}, []) for k in
                                ("files_sha256_json", "checkpoint_path", "checkpoint_sha256", "bigmoe_commit", "dd_topk_sha256",
                                 "dd_tree_sha256", "args", "modules")) and "bigmoe_dirty" in R)
check("files_sha256_json", R["files_sha256_json"] == fsha)
check("checkpoint", R["checkpoint_sha256"] == DM.sha256_file(ck) and R["checkpoint_path"] == DM.tilde(ck))
check("args_nested", R["args"] == {"dd_topk": a_tk, "dd_tree": [a_t8, a_t127]})
check("modules_pin_this_code", "~/BigMoE/scripts/dd_manifest.py" in R["modules"] and
      all(not p.startswith("~/BigMoE/.venv") for p in R["modules"]))
E = {k: v for k, v in M.items() if k != "_run"}
check("each_file_args_in_run_args", all((v["args"] == R["args"]["dd_topk"]) if v["script"] == "dd_topk" else
                                        (v["args"] in R["args"]["dd_tree"]) for v in E.values()))
check("file_sha_and_part", all(v["sha256"] == DM.sha256_file(o[k]) for k, v in E.items()) and
      E["tree_test_heplus.part1of2.npz"].get("part") == "1/2" and "part" not in E["tree_test_heplus.npz"])
refused("refuses_other_checkpoint", lambda: DM.update(MP, [o["tree_test_heplus.npz"]], "dd_tree", a_t8, checkpoint=ck2))
refused("refuses_other_capture_manifest", lambda: DM.update(MP, [o["tree_test_heplus.npz"]], "dd_tree", a_t8, files_sha="0" * 64))
refused("refuses_other_topk_args", lambda: DM.update(MP, [o["topk_test_heplus.npz"]], "dd_topk", {"k": 16, "parts": 1}))
mod0 = next(iter(R["modules"]))
refused("refuses_changed_module", lambda: DM.update(MP, [o["tree_test_heplus.npz"]], "dd_tree", a_t8,
                                                     modules={mod0: "f" * 64}))
DM.update(MP, [o["tree_test_heplus.npz"]], "dd_tree", a_t8, commit="deadbeef")
R2 = json.load(open(MP))["_run"]
check("commit_change_listed", R2["bigmoe_commit"] == R["bigmoe_commit"] and R2.get("bigmoe_commits_seen") ==
      [R["bigmoe_commit"], "deadbeef"])
out = {"workdir": D, "checks": res, "fails": fails, "verdict": "PASS" if not fails else "FAIL", "sample_run": R2}
print(json.dumps({k: out[k] for k in ("checks", "fails", "verdict")}, indent=1))
json.dump(out, open(os.path.join(DM.BM, "data/deepdraft/dd_manifest_test.json"), "w"), indent=1)
sys.exit(0 if not fails else 1)
