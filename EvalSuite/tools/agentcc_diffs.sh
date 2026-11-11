#!/bin/bash
# agentcc_diffs.sh <run> <out dir> [tasks|checks]: each task's final files against its template, as a git diff, for the
# agentcc RESULTs. From runner v5 (tasks5) on, checks/<id> is the runner's copy that the check graded; tasks/ is the default.
# Never runs git on the agent's own .git (its config or hooks could run code): a fresh repo of ours gets the manifest's
# template as its first commit, then the agent's files without .git and the hidden check. Refuses a symlink in <run>
# outside the agent's own folders and the runner's copies of them (tasks/, checks/, cc_home/, cctmp/ <id>); a link inside is copied as a link
# (rsync -a), never followed, and shows in the diff as its target path.
set -eu
RUN=${1:?run}; OUT=${2:?out}; SUB=${3:-tasks}; MAN=$HOME/EvalSuite/agentcc/manifest.json
case "$SUB" in tasks|checks) ;; *) echo "the third argument is tasks or checks" >&2; exit 2;; esac
links=$(find "$RUN" -type l | grep -vE "^$RUN/(tasks|checks|cc_home|cctmp)/[^/]+/" | head -5 || true)
[ -z "$links" ] || { echo "REFUSED: symlinks in the run folder outside the agent's own folders: $links" >&2; exit 3; }
inside=$(find "$RUN/$SUB" -type l | wc -l | tr -d ' '); [ "$inside" = 0 ] || echo "note: $inside symlinks inside task folders (copied as links)" >&2
W=$(mktemp -d "${TMPDIR:-/tmp}/agentcc_diffs.XXXX"); trap 'rm -rf "$W"' EXIT
g() { GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git -c core.hooksPath=/dev/null -c core.fsmonitor=false \
      -c user.name=agentcc -c user.email=agentcc@localhost -c commit.gpgsign=false -C "$1" "${@:2}"; }
mkdir -p "$OUT"
for T in "$RUN/$SUB"/t[0-9][0-9]_*; do
  id=${T##*/}; R="$W/$id"
  tpl=$(python3 -I -c "import json,sys; print(next(t['template'] for t in json.load(open(sys.argv[1]))['tasks'] if t['id'] == sys.argv[2]))" "$MAN" "$id")
  mkdir -p "$R"; rsync -a "$tpl/" "$R/"; g "$R" init -q; g "$R" add -A; g "$R" commit -q -m template
  rsync -a --delete --exclude .git --exclude .agentcc_check.py "$T/" "$R/"
  g "$R" add -A; g "$R" diff --cached HEAD > "$OUT/$id.diff"
  echo "$id: $(wc -l < "$OUT/$id.diff" | tr -d ' ') lines; $(g "$R" diff --cached --name-only HEAD | grep -v __pycache__ | tr '\n' ' ')"
done
