#!/bin/bash
# release/push_pkg.sh pkg|docs   (DRY=1: everything but the push)
# pkg : put the freshly built release/out/pkg (make_release.py makes a one-commit repo) on top of GitHub's main as a
#       fast-forward: fetch main, move HEAD there keeping the new tree (git reset --soft), keep docs/ as the remote has it,
#       commit with the build's message, show the diff, push. Never a force push.
# docs: copy Breakthrough's final public site (~/Papers/public/site/) to docs/ in one commit on top of main, after checks:
#       no small-ram file while the papers' publish set withholds SMALL_RAM (~/Papers/public/src/publish_set.txt;
#       1.1.0 publishes it), every docs/pdf/*.pdf the package README links present in the site, no /Users/ path in
#       the HTML, no home path in the PDFs' text.
# The repository stays private, the release a draft and Pages off: only the orchestrator flips them, on the owner's word.
set -uo pipefail
MODE=${1:?pkg|docs}
P=${PKG_DIR:-/Users/kamen/Leap/release/out/pkg}
URL=https://github.com/kamenlevi/exactstream.git
SITE=${SITE_DIR:-/Users/kamen/Papers/public/site}
g() { git -C "$P" -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
        -c user.name=kamenlevi -c user.email=kamenlevi@users.noreply.github.com "$@"; }
VIS=$(gh repo view kamenlevi/exactstream --json visibility --jq .visibility)
# Since 18:35 (the owner's go, via the orchestrator) the repo is public: a push needs PUBLIC_OK=1, set only for a change
# the owner asked for, so nothing reaches the public repo by accident.
# DRY=1 never pushes (it exits before the only `git push` below), so a dry run may proceed on the public repo (Tue 10 Nov).
[ "$VIS" = PRIVATE ] || [ "${PUBLIC_OK:-0}" = 1 ] || [ "${DRY:-0}" = 1 ] || { echo "STOP: the repo is $VIS (set PUBLIC_OK=1 only for an owner-requested change)"; exit 2; }
[ "$VIS" = PRIVATE ] || [ "${PUBLIC_OK:-0}" = 1 ] || echo "DRY run on the $VIS repo: nothing will be pushed"
g fetch -q "$URL" main || { echo "STOP: fetch failed"; exit 2; }
REMOTE=$(g rev-parse FETCH_HEAD)
if [ "$MODE" = pkg ]; then
  MSG=$(g log -1 --format=%B HEAD)
  if g merge-base --is-ancestor FETCH_HEAD HEAD 2>/dev/null; then echo "HEAD already contains the remote main"
  else
    g reset -q --soft FETCH_HEAD
    if g cat-file -e FETCH_HEAD:docs 2>/dev/null; then g checkout -q FETCH_HEAD -- docs; echo "kept docs/ from the remote"; fi
    if g diff --cached --quiet FETCH_HEAD; then echo "NO CHANGE vs the remote main ($(g rev-parse --short FETCH_HEAD))"; exit 0; fi
    g commit -q -m "$MSG" || { echo "STOP: commit failed"; exit 2; }
  fi
elif [ "$MODE" = docs ]; then
  if [ "${AMEND:-0}" = 1 ]; then   # fold docs/ into a package commit made by `DRY=1 push_pkg.sh pkg` (one commit for both)
    [ "$(g rev-parse HEAD^)" = "$REMOTE" ] || { echo "STOP: AMEND needs HEAD's parent = the remote main"; exit 2; }
  else
    [ "$(g rev-parse HEAD)" = "$REMOTE" ] || { echo "STOP: local HEAD $(g rev-parse --short HEAD) is not the remote main $(g rev-parse --short "$REMOTE"): push pkg first"; exit 2; }
  fi
  [ -f "$SITE/index.html" ] || { echo "STOP: no $SITE/index.html"; exit 2; }
  PUBSET=${PUBSET_FILE:-/Users/kamen/Papers/public/src/publish_set.txt}
  if ! grep -v '^#' "$PUBSET" 2>/dev/null | grep -qw SMALL_RAM; then      # SMALL_RAM withheld: no small-ram file may go out
    bad=$(find "$SITE" -iname '*small-ram*' -o -iname '*small_ram*' | head -3)
    [ -z "$bad" ] || { echo "STOP: small-ram files in the site while SMALL_RAM is withheld ($PUBSET): $bad"; exit 2; }
  fi
  [ -f "$P/README.md" ] || { echo "STOP: no $P/README.md to check the PDF links against (set PKG_DIR)"; exit 2; }
  for f in $(grep -o 'docs/pdf/[A-Za-z0-9_.-]*\.pdf' "$P/README.md" | sort -u); do   # every PDF the README links is in the site
    [ -f "$SITE/${f#docs/}" ] || { echo "STOP: README links $f, but the site has no ${f#docs/}"; exit 2; }
  done
  LOCAL='/Users/|~/(BigMoE|Leap|EvalSuite|FableGod|NewIdeas|MoonShots|TrainLab|Breakthrough|Fit12|Papers)\b'
  # the private names (account, address, personal phrases) come from the gitignored file, never written here: written
  # out they would block the public snapshot, whose own leak check refuses them (as make_release.py, 11-09)
  PRIVRE=$(/opt/homebrew/opt/python@3.12/bin/python3.12 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["leak"])' \
           /Users/kamen/Leap/release/private_patterns.json) || { echo "STOP: cannot read release/private_patterns.json"; exit 2; }
  hits=$( { grep -rIl -E "$LOCAL" "$SITE"; grep -rIl -i -E "$PRIVRE" "$SITE"; } | sort -u | head -3)
  [ -z "$hits" ] || { echo "STOP: local paths or private names in: $hits"; exit 2; }
  for f in "$SITE"/pdf/*.pdf; do
    if command -v pdftotext >/dev/null; then t=$(pdftotext -q "$f" - 2>/dev/null); else t=$(strings "$f"); fi
    if echo "$t" | grep -q -E "$LOCAL" || echo "$t" | grep -q -i -E "$PRIVRE"; then echo "STOP: a local path or private name in $f"; exit 2; fi
  done
  rm -rf "$P/docs"; mkdir -p "$P/docs"; cp -R "$SITE"/. "$P/docs/"
  g add -A docs
  DMSG="docs: the papers (PDF) and their index page, from the final public build ($(date -r "$SITE/index.html" '+%d %b %H:%M'))"
  if [ "${AMEND:-0}" = 1 ]; then
    g commit -q --amend -m "$(g log -1 --format=%s HEAD); $DMSG" || { echo "STOP: amend failed"; exit 2; }
  else
    g diff --cached --quiet && { echo "NO CHANGE in docs/"; exit 0; }
    g commit -q -m "$DMSG" || { echo "STOP: commit failed"; exit 2; }
  fi
else echo "usage: push_pkg.sh pkg|docs"; exit 2; fi
echo "diff vs the remote main $(g rev-parse --short "$REMOTE"):"; g diff --stat "$REMOTE" HEAD | tail -12
if [ "${DRY:-0}" = 1 ]; then echo "DRY: not pushed ($(g rev-parse --short HEAD))"; exit 0; fi
g push -q "$URL" HEAD:main && echo "PUSHED $(g rev-parse --short HEAD) to main (fast-forward)" || { echo "STOP: push refused"; exit 2; }
[ "$(g ls-remote "$URL" refs/heads/main | cut -f1)" = "$(g rev-parse HEAD)" ] && echo "remote main == local HEAD"
