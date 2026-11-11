"""run_arm.py - ONE warm Strata engine process for one arm and pass of the head-to-head (stdlib only).

`strata --serve` (the engine's own line protocol, as serve/server.py drives it) under `/usr/bin/time -l`:
  READY <ctx> stop                    <- the engine, once loaded
  GEN <max_new> ckpt=0 <id,id,...>    -> one request: greedy (no temperature key = the engine's argmax), EOS stop (248044 /
                                         248046, always on in serve mode), ckpt=0 (a one-shot: no checkpoint, nothing parked)
  T <id> ... DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_acc> <drafts_off> <reused> ...  <- the engine
  QUIT                                -> ends it
Order: a discarded warm-up (the manifest's role "warmup": gsm8k/708, 32 tokens, as S5 / N45 warm up), then every "measure" prompt
in manifest order. Between requests: PAUSE / closed lid / soft_end.sh stop the process (exit 4, the driver re-runs it later).
A watchdog kills the engine when swap grows by more than --swap-limit-mb (exit 3).
Writes into --out: <tag>.requests.jsonl (one line per request), <tag>.engine.log (engine stderr + /usr/bin/time),
<tag>.summary.json, and <tag>.done when every request was answered.
Exit: 0 ok, 2 engine did not start / died, 3 swap guard, 4 paused (PAUSE / lid / soft_end), 5 a request failed.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PAUSE = Path.home() / ".local" / "share" / "quiet" / "PAUSE"


def swap_mb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        return float(m.group(1)) if m else 0.0
    except (OSError, subprocess.SubprocessError):
        return 0.0


def lid_shut() -> bool:
    """EvalSuite runner 55's rule: the clamshell is closed and no external display is attached."""
    try:
        o = subprocess.run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"], capture_output=True, text=True, timeout=10).stdout
        if not re.search(r'"AppleClamshellState" = Yes', o):
            return False
        d = subprocess.run(["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"], capture_output=True, text=True,
                           timeout=60).stdout
        return d.count("Resolution:") <= 1
    except (OSError, subprocess.SubprocessError):
        return False


def soft_end_running() -> bool:
    try:
        o = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True, timeout=10).stdout
        return any(re.match(r"^/bin/zsh \S*soft_end\.sh", l) for l in o.splitlines())
    except (OSError, subprocess.SubprocessError):
        return False


def calm_reason() -> str | None:
    if PAUSE.exists():
        return "PAUSE file"
    if soft_end_running():
        return "soft_end.sh"
    if lid_shut():
        return "lid closed, no external display"
    return None


def parse_done(line: str) -> dict:
    f = line.split()
    d = {"generated": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]), "finish": f[5],
         "drafts_accepted": int(f[6]), "drafts_offered": int(f[7]), "reused": int(f[8])}
    names = ["hits", "lookups", "ram_blobs", "file_blobs", "file_mb", "prompt_read", "offloaded"]
    pos = [x for x in f[9:] if "=" not in x]
    for n, v in zip(names, pos):
        try:
            d[n] = float(v) if n == "file_mb" else int(v)
        except ValueError:
            pass
    for x in f[9:]:
        if "=" in x:
            k, _, v = x.partition("=")
            try:
                d[k] = int(v)
            except ValueError:
                d[k] = v
    return d


ENGINE_PATTERNS = {
    "planned_footprint_gb": r"planned resident footprint ([\d.]+) GB = arena ([\d.]+)",
    "arena": r"expert arena ([\d.]+) GB, (\d+) slots in (\d+) size classes, (\d+) experts read from the profile's top \((\d+) ranked\) in ([\d.]+) s",
    "arena_fill_stopped_gb": r"the arena's fill stopped after ([\d.]+) GB",
    "wired_gb": r"([\d.]+) GB of the arena wired \(mlock\)",
    "after_fill": r"resident footprint after the fill: the process holds ([\d.]+) GB \(([\d.]+) GB resident\), planned ([\d.]+) GB, cap ([\d.]+) GB",
    "hc_q8": r"strata metal: HC read in Q8_0",
    "decode_window": r"strata metal: decode window on (.*?), ready in ([\d.]+) ms",
    "cap_text": r"cap ([\d.]+) GB = the lesser of the GPU working set ([\d.]+) GB and the RAM ([\d.]+) GB less ([\d.]+) GB",
    "peak_footprint_b": r"^\s*(\d+)\s+peak memory footprint",
    "max_rss_b": r"^\s*(\d+)\s+maximum resident set size",
    "real_s": r"^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys",
}


def parse_engine_log(path: Path) -> dict:
    out: dict = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    for k, pat in ENGINE_PATTERNS.items():
        ms = list(re.finditer(pat, text, re.M))
        if not ms:
            continue
        m = ms[-1]
        if k == "hc_q8":
            out[k] = True
        elif k == "arena":
            out.update(arena_gb=float(m.group(1)), arena_slots=int(m.group(2)), arena_classes=int(m.group(3)),
                       arena_profile_experts=int(m.group(4)), arena_fill_s=float(m.group(6)))
        elif k == "planned_footprint_gb":
            out.update(planned_footprint_gb=float(m.group(1)), planned_arena_gb=float(m.group(2)))
        elif k == "after_fill":
            out.update(after_fill_footprint_gb=float(m.group(1)), after_fill_resident_gb=float(m.group(2)),
                       footprint_cap_gb=float(m.group(4)))
        elif k == "decode_window":
            out.update(decode_window=m.group(1)[:200], window_ready_ms=float(m.group(2)))
        elif k == "cap_text":
            out.update(gpu_working_set_gb=float(m.group(2)), ram_gb=float(m.group(3)), left_to_macos_gb=float(m.group(4)))
        elif k in ("peak_footprint_b", "max_rss_b"):
            out[k] = int(m.group(1))
        elif k == "real_s":
            out.update(time_real_s=float(m.group(1)), time_user_s=float(m.group(2)), time_sys_s=float(m.group(3)))
        else:
            out[k] = float(m.group(1))
    out.setdefault("hc_q8", False)
    out["strata_lines"] = [l for l in text.splitlines() if l.startswith("strata metal")][:40]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pass", dest="pas", type=int, required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-args", required=True, help="JSON list: the engine's arguments after --serve")
    ap.add_argument("--limit", type=int, default=0, help="dry runs: at most N measure prompts per set")
    ap.add_argument("--swap-limit-mb", type=float, default=2048.0)
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--req-timeout", type=float, default=900.0)
    a = ap.parse_args()
    tag = f"{a.arm}{a.pas}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    base = Path(a.manifest).resolve().parent.parent
    prompts = [p for p in man["prompts"] if p["role"] == "warmup"]
    per_set: dict = {}
    for p in man["prompts"]:
        if p["role"] != "measure":
            continue
        per_set[p["set"]] = per_set.get(p["set"], 0) + 1
        if a.limit and per_set[p["set"]] > a.limit:
            continue
        prompts.append(p)
    engine_args = json.loads(a.engine_args)
    cmd = ["/usr/bin/time", "-l", a.exe, "--serve", *engine_args]
    reqf = out / f"{tag}.requests.jsonl"
    logf = out / f"{tag}.engine.log"
    for f in (reqf, logf, out / f"{tag}.summary.json", out / f"{tag}.done"):
        f.unlink(missing_ok=True)
    t_start = time.time()
    swap0 = swap_mb()
    summary = {"arm": a.arm, "pass": a.pas, "tag": tag, "cmd": cmd, "env": {k: v for k, v in os.environ.items()
               if k.startswith("STRATA_")}, "t_start": t_start, "swap0_mb": swap0, "prompts": len(prompts)}
    logh = open(logf, "w")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logh, text=True, bufsize=1,
                         start_new_session=True)
    lines: queue.Queue = queue.Queue()
    abort = {"why": None, "code": 0}

    def reader():
        for ln in p.stdout:
            lines.put(ln.rstrip("\n"))
        lines.put(None)

    def watchdog():
        while p.poll() is None and abort["why"] is None:
            s = swap_mb()
            if s - swap0 > a.swap_limit_mb:
                abort.update(why=f"swap grew {s - swap0:.0f} MB (> {a.swap_limit_mb:.0f})", code=3)
                kill()
                return
            time.sleep(5)

    def kill():
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass

    def stop(sig, _frm):
        abort.update(why=f"signal {sig}", code=2)
        kill()
        sys.exit(2)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    def next_line(deadline):
        while True:
            left = deadline - time.time()
            if left <= 0:
                return "TIMEOUT"
            try:
                return lines.get(timeout=min(left, 5.0))
            except queue.Empty:
                if abort["why"]:
                    return None

    # ---- READY
    ready = None
    dl = time.time() + a.ready_timeout
    while True:
        ln = next_line(dl)
        if ln is None or ln == "TIMEOUT":
            break
        if ln.startswith("READY"):
            ready = ln
            break
    summary["ready_s"] = round(time.time() - t_start, 2)
    summary["ready_line"] = ready
    rc = 0
    n_ok = n_err = 0
    if ready is None:
        abort["why"] = abort["why"] or ("no READY within %.0f s or the engine ended" % a.ready_timeout)
        rc = abort["code"] or 2
        kill()
    else:
        with open(reqf, "a") as rq:
            for i, pr in enumerate(prompts):
                why = calm_reason() if pr["role"] == "measure" else None
                if why:
                    abort.update(why=f"paused: {why}", code=4)
                    break
                if abort["why"]:
                    break
                ids = Path(base / pr["file"]).read_text().strip()
                line = f"GEN {int(pr['max_new'])} ckpt=0 {ids}\n"
                rec = {"arm": a.arm, "pass": a.pas, "tag": tag, "i": i, "set": pr["set"], "key": pr["key"], "role": pr["role"],
                       "n_prompt": pr["n_prompt"], "max_new": pr["max_new"], "swap_mb_before": swap_mb()}
                t0 = time.time()
                try:
                    p.stdin.write(line)
                    p.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    abort.update(why=f"engine pipe: {e}", code=2)
                    break
                toks, t_first, other = [], None, []
                dl = t0 + a.req_timeout
                while True:
                    ln = next_line(dl)
                    if ln is None:
                        rec["error"] = abort["why"] or "engine ended"
                        break
                    if ln == "TIMEOUT":
                        rec["error"] = f"no DONE within {a.req_timeout:.0f} s"
                        abort.update(why=rec["error"], code=5)
                        kill()
                        break
                    if ln.startswith("T "):
                        if t_first is None:
                            t_first = time.time()
                        toks.append(int(ln.split()[1]))
                    elif ln.startswith("DONE "):
                        rec.update(parse_done(ln))
                        break
                    elif ln.startswith("ERR"):
                        rec["error"] = ln[:300]
                        break
                    elif len(other) < 5:
                        other.append(ln[:200])
                t1 = time.time()
                rec.update(t_send=t0, t_first=t_first, t_done=t1, wall_ttft_s=(t_first - t0) if t_first else None,
                           wall_total_s=t1 - t0, n_ids=len(toks), ids=toks, other_lines=other, swap_mb_after=swap_mb())
                if "error" in rec:
                    n_err += 1
                elif pr["role"] == "measure":
                    n_ok += 1
                rq.write(json.dumps(rec) + "\n")
                rq.flush()
                if abort["why"]:
                    break
        if abort["why"] is None:
            try:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                p.wait(120)
            except subprocess.TimeoutExpired:
                kill()
        else:
            kill()
        rc = abort["code"] if abort["why"] else (5 if n_err else 0)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        kill()
    logh.close()
    summary.update(t_end=time.time(), wall_s=round(time.time() - t_start, 1), abort=abort["why"], rc=rc,
                   measured_ok=n_ok, errors=n_err, swap_end_mb=swap_mb(), engine_returncode=p.returncode)
    summary.update(parse_engine_log(logf))
    (out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=1))
    want = sum(1 for x in prompts if x["role"] == "measure")
    if rc == 0 and n_ok == want:
        (out / f"{tag}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"{tag}: rc {rc}; {n_ok}/{want} measured, {n_err} errors; ready {summary['ready_s']} s; wall {summary['wall_s']} s; "
          f"peak footprint {summary.get('peak_footprint_b', 0) / 1e9:.2f} GB; arena {summary.get('arena_gb')} GB; "
          f"HC Q8 {summary.get('hc_q8')}; abort {abort['why']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
