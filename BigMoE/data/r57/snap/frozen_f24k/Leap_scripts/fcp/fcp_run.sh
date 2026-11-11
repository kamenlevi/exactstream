#!/bin/bash
# FCP (Leap, Sat 2026-11-07 08:3x; the orchestrator 08:1x: scope FIRSTCHK). BigMoE's FIRSTCHK control (scripts/fc_kill.sh's
# fcctl: v1' = data/stack/final_env.txt with QOM sets humaneval, R57_A6=ctl R57_TCPU=1, 3 HumanEval prompts, CAP 64, C 256)
# on a COPY of BigMoE's scripts/r57_screen_fast.py (read only; outputs under ~/Leap/data/fcp), with fcprobe.py patched in
# and R57_LOGHASH=1 (each check's verify logits hashed, for the bitwise comparison across arms). Arms, one process each:
#   fcpa  the probe only (nothing changed)
#   fcpb  the probe + FCP_NOCLEAR=1 (tier.prefill_done's mx.clear_cache() skipped; it runs inside check 0's graph build)
# Inside Leap's own quiet lock (HAVE_LOCK=1; the lock taken in a separate command). Guards: PAUSE / HOLD, NO-SWAP >= 47.2 GB
# (fc_kill's C 256 gate), swap +2 GB during an arm, alarm 420 s per arm.  usage: HAVE_LOCK=1 bash fcp_run.sh [arms]
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$HOME/Leap/data/fcp"; SD="$D/snap"; mkdir -p "$SD" "$D/r57"; LOG="$D/fcp.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
PY="$HOME/BigMoE/.venv/bin/python"
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
[ -z "$HAVE_LOCK" ] && [ -z "$FCP_DRY" ] && { say "fcp_run.sh runs inside Leap's quiet lock: HAVE_LOCK=1"; exit 1; }
B="$HOME/BigMoE/scripts"
cp "$B/r57_screen_fast.py" "$SD/r57_screen_fast_src.py" && cp "$B/relax_delta.py" "$B/relax_span.py" "$B/inert_span.py" "$SD/" || exit 2
"$PY" -I -B - "$SD/r57_screen_fast_src.py" "$SD/r57_screen_fcp.py" "$HERE" <<'EOF' || exit 2
import sys
src, dst, here = sys.argv[1:4]
s = open(src).read()
def ins_after_line(s, anchor, text):
    assert s.count(anchor) == 1, (anchor, s.count(anchor))
    e = s.index("\n", s.index(anchor)) + 1
    return s[:e] + text + s[e:]
s = ins_after_line(s, "import mlx.core as mx\n",
                   f"import sys as _fsys; _fsys.path.insert(0, {here!r}); import fcprobe as _FCP; _FCP.install()   # Leap FCP\n")
a = "        c1 = time.perf_counter()\n        drf.append(c1 - c0)"   # (BigMoE's S0CARRY, Sat 15:3x, rewrote the s0 line above it)
assert s.count(a) == 1, s.count(a)
s = s.replace(a, "        _FCP.mark('c1', len(drf))                     # Leap FCP: check len(drf) starts\n" + a)
s = ins_after_line(s, "        cb = time.perf_counter()", "        _FCP.mark('cb')                                # Leap FCP\n")
s = ins_after_line(s, "        ce = time.perf_counter()", "        _FCP.mark('ce')                                # Leap FCP\n")
s = ins_after_line(s, "        c0 = time.perf_counter()", "        _FCP.mark('c0')                                # Leap FCP seams\n")
s = ins_after_line(s, "        c2 = time.perf_counter()", "        _FCP.mark('c2')                                # Leap FCP seams\n")
a_ = "            dr = np.array(pth[0]).tolist()"
assert s.count(a_) == 1
s = s.replace(a_, "            _FCP.mark('rb0')                          # Leap FCP seams: the draft's readbacks start\n" + a_)
s = ins_after_line(s, "            cfl = np.array(cf[0]).tolist()", "            _FCP.mark('rb1')                          # Leap FCP seams: the GPU drained\n")
o = 'OUT = os.path.expanduser("~/BigMoE/data/r57")'
assert s.count(o) == 1
s = s.replace(o, 'OUT = os.path.expanduser("~/Leap/data/fcp/r57")')
open(dst, "w").write(s)
print("patched", dst, s.count("_FCP."), "probe lines")
EOF
V1P=$(grep -v '^#' "$HOME/BigMoE/data/stack/final_env.txt" | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$/,R57_QOM_SETS=humaneval/')
echo "$V1P" | grep -q NEVERWAIT && { say "final_env carries NW: not v1'"; exit 2; }
ENV0="R57_PRESET=45 BIGMOE_EXT_BUILD=build R57_PROMPTS=6 R57_REPS=1 R57_WARM=1 $(echo "$V1P" | tr ',' ' ') R57_A6=ctl R57_TCPU=1 R57_LOGHASH=1"
export R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91"
ARMS=${1:-"fcpa fcpb"}
A=$(avail_gb)
say "FCP start (pid $$, NO-SWAP $A GB, arms $ARMS; screen md5 $(md5 -q "$B/r57_screen_fast.py" | cut -c1-8), probe $(md5 -q "$HERE/fcprobe.py" | cut -c1-8))"
[ -n "$FCP_DRY" ] && { say "DRY: env $ENV0"; exit 0; }
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
cd "$SD" || exit 1
rc_all=0
EXE="R57_PRESET=45 BIGMOE_EXT_BUILD=build R57_PROMPTS=6 R57_REPS=1 R57_WARM=1 R57_NAX=1 R57_REFTOK= R57_A6=ctl R57_TCPU=1 R57_LOGHASH=1"
KX="humaneval/HumanEval/68,humaneval/HumanEval/16,ifeval/1220"
KC6="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91"   # (HumanEval/0 is not in the screen's set)
KC="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/0,humaneval/HumanEval/23,humaneval/HumanEval/52"   # PLAN 220 commit split   # BigMoE's KEEP exact proof (s2c_ab.sh 0.): NAX6-N, C 224
for arm in $ARMS; do
  E="$ENV0" CC=256 KK="$R57_KEYS" CP=64
  case $arm in fcpa) X="";; fcpb) X="FCP_NOCLEAR=1";; fcpw) X="FCP_WIRED=1";;
    fcpc) X="FCP_COMMIT=1 R57_WIRED=1" KK="$KC" CP=128;; fcpd) X="FCP_COMMIT=1 FCP_CSYNC=1 R57_WIRED=1" KK="$KC" CP=128;;
    fcpx0) X="" E="$EXE" CC=224 KK="$KX";;
    fcpx2) X="REF2_EPREP=1 REF2_GDN_TG=1" E="$EXE" CC=224 KK="$KX";;   # PLAN 220 1a bundle in the exact env
    fcpe) X="REF2_EPREP=1 REF2_GDN_TG=1 R57_WIRED=1";;   # the bundle on v1' + wired (FAST): engagement + tokens
    fcpx3) X="REF2_SB_FIRST=1 REF2_PLE_LATE=1 REF2_COMMIT_SUB=9" E="$EXE" CC=224 KK="$KX";;   # seams (b, d, e) in the exact env
    fcpf) X="REF2_SB_FIRST=1 REF2_PLE_LATE=1 REF2_COMMIT_SUB=9 R57_WIRED=1";;   # seams (b, d, e) on v1' + wired: engagement
    fcpk) X="REF2_TIER_CHUNK=1 R57_WIRED=1" CC=112 KK="$KC6";;   # F24: v1' + wired at C 112 with row-chunking (the ungated check)
    fcpxk) X="REF2_TIER_CHUNK=1" E="$EXE" CC=112 KK="$KX";;   # F24 proof: the exact env at C 112, chunked sync path vs fcpx0 (C 224)
    fcps) X="FCP_SEAMS=1 R57_WIRED=1" KK="$KC6" CP=128;;   # PLAN 220 (4): the cycle's GPU-idle seams on v1' + wired
    fcpx1) X="FCP_WIRED=1" E="$EXE" CC=224 KK="$KX";; *) say "unknown arm $arm"; continue;; esac
  if [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; then say "PAUSE / HOLD: $arm skipped"; rc_all=1; continue; fi
  SW0=$(swap_mb); say "run $arm (swap $SW0 MB)"
  env $E $X R57_KEYS=$KK R57_TAG=$arm FCP_OUT="$D" VECLIB_MAXIMUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    perl -e 'alarm shift; exec @ARGV' 420 "$PY" -B r57_screen_fcp.py $CC $CP > "$D/$arm.txt" 2>&1 &
  jp=$!
  while kill -0 $jp 2>/dev/null; do
    sleep 5
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then say "SWAP GUARD: stopping $arm"; kill $jp; sleep 10; kill -9 $jp 2>/dev/null; break; fi
  done
  wait $jp; rc=$?
  say "done $arm rc=$rc: $(grep -h -E '\[fcp\] [0-9]+ check|footprint now' "$D/$arm.txt" | tail -2 | cut -c1-150 | tr '\n' ' ')"
  [ $rc != 0 ] && rc_all=1
done
say "FCP end (rc $rc_all)"
exit $rc_all
