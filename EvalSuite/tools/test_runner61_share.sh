#!/bin/bash
# CPU test of runner 61's share_lock / rebase_swap / calm (11-09 fixes): the swap base is re-read after a yield, and a
# PAUSE right at the grant releases the lock and waits. A fake `peer` and a temp PAUSE file (CARD_PAUSE_FILE), never
# the real one; the functions are taken from tools/day_runner61.sh as they are. Run: tools/test_runner61_share.sh
R=$(cd "$(dirname "$0")" && pwd)/day_runner61.sh
T=$(mktemp -d); mkdir -p "$T/bin"
for f in QPAUSE= 'rebase_swap()' 'share_lock()'; do
  awk -v s="$f" 'index($0, s) == 1 {p = 1} p {print} /^}/ {if (p) exit}' "$R"
done > "$T/fns.sh"
cat > "$T/bin/peer" <<'EOF'
#!/bin/bash
S=$FAKE_STATE; echo "$*" >> "$S/calls"
if [ "$1 $2" = "lock share" ]; then
  n=$(($(cat "$S/n" 2>/dev/null || echo 0) + 1)); echo $n > "$S/n"
  act=$(sed -n "${n}p" "$S/plan")
  case ${act:-grant} in
    refuse) echo "another session needs the Mac quiet"; exit 3 ;;
    grant_pause) touch "$CARD_PAUSE_FILE"; echo granted ;;
    *) echo granted ;;
  esac
fi
exit 0
EOF
chmod +x "$T/bin/peer"
run() {         # plan (comma list: refuse | grant | grant_pause), swap GB per swapgb call (comma list)
  ( export PATH="$T/bin:$PATH" FAKE_STATE=$(mktemp -d "$T/s.XXXX"); export CARD_PAUSE_FILE="$FAKE_STATE/PAUSE"
    echo "$1" | tr ',' '\n' > "$FAKE_STATE/plan"; echo "$2" | tr ',' '\n' > "$FAKE_STATE/swaps"
    say() { echo "SAY: $*"; }; waitlog() { :; }; lid_shut() { return 1; }; pgrep() { return 1; }
    SL=0; sleep() { SL=$((SL + 1)); [ "$SL" -ge 2 ] && rm -f "$CARD_PAUSE_FILE"; }
    swapgb() { local n; n=$(($(cat "$FAKE_STATE/sw" 2>/dev/null || echo 0) + 1)); echo $n > "$FAKE_STATE/sw"; sed -n "${n}p" "$FAKE_STATE/swaps"; }
    source "$T/fns.sh"; SWAP0=$(swapgb)
    share_lock "evalsuite: test" 30; echo "rc=$? SWAP0=$SWAP0 calls=$(tr '\n' ';' < "$FAKE_STATE/calls")" )
}
fail=0
check() { echo "$2" | grep -q -- "$3" && echo "ok: $1" || { echo "FAIL: $1 (wanted '$3')"; echo "$2"; fail=1; }; }
o=$(run refuse,grant 5,8);      check "yield re-reads the swap base" "$o" "swap base 5 -> 8 GB (re-read after a yield)"
                                check "yield: base is now 8" "$o" "rc=0 SWAP0=8"
o=$(run refuse,grant 5,5);      check "yield, swap unchanged: no message" "$o" "rc=0 SWAP0=5"
o=$(run grant_pause,grant 5,5); check "PAUSE at the grant: released" "$o" "PAUSE right after the shared lock: released it"
                                check "PAUSE at the grant: unlock, then asked again" "$o" "calls=lock share.*;unlock;lock share"
o=$(run grant 5,9);             check "no wait: no re-read (growth is ours)" "$o" "rc=0 SWAP0=5"
rm -rf "$T"
[ "$fail" = 0 ] && echo "ALL OK"
exit "$fail"
