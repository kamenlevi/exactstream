#!/bin/bash
# (bashfix_git.sh: R = 32_1111b's setup (real HOME), F = the cached runs' (neutral HOME), both in a git repo: git status /
# diff / log / commit, since Kamen's gitconfig signs commits with an SSH key and the profile denies ~/.ssh.)
# (bashfix_final.sh: variant F = the cached runner's exact home (agent_home_tpl) in a git repo task folder, vs bashfix_home's B.)
# (bashfix_home.sh: variant H = HOME=<run>/home, since ZDOTDIR had no effect: Claude Code sources $HOME/.zshrc itself.)
# A neutral zsh startup for the sandboxed agent runs (orchestrator 04:0x): Claude Code builds its shell snapshot from the
# user's zsh startup files, so the local model can read the user's aliases (32_1111 t01 did). From the next runs on, the
# claude processes get ZDOTDIR=<run>/zdot, whose .zshrc only exports the exact PATH today's runs use (and any env the
# tasks need), no aliases or functions. This check, on the stand-in engine (mock_bash.py; no GPU, < 1 GB):
#   B: today's setup (regex read-deny profile, CLAUDE_CODE_TMPDIR), no ZDOTDIR: the Bash tool's PATH, tool binaries,
#      versions, encodings, env var NAMES (values stay in <run>/tasks/t00/envfull.txt, local only, never printed);
#   Z: the same + ZDOTDIR=<run>/zdot with .zshrc = `export PATH='<B's PATH>'`: the same report, compared with B.
# PASS = PATH, every tool's binary, python3 / git versions and the encodings equal; aliases 0; functions listed.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8131; B=$A/bashfix_git; mkdir -p $B; H=/Users/kamen
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
export MOCK_BASH_CMD='env > envfull.txt; { echo "PATH=$PATH"; echo "ZDOTDIR=${ZDOTDIR:-unset}"; echo "HOMEDIR $HOME"; git status --porcelain >/dev/null 2>&1 && echo "GIT status ok" || echo "GIT status FAIL"; git config --global user.email >/dev/null 2>&1 && echo "GITID set" || echo "GITID unset"; echo y > model.txt; git add model.txt && git commit -q -m model && echo "GITCOMMIT ok" || echo "GITCOMMIT FAIL"; git commit -q -m model2 --allow-empty 2>&1 | head -2 | sed "s/^/GITCOMMITERR /" | cut -c1-160; git diff HEAD~1 --stat >/dev/null 2>&1 && echo "GITDIFF ok" || echo "GITDIFF FAIL"; git log --oneline -3 >/dev/null 2>&1 && echo "GITLOG ok" || echo "GITLOG FAIL"; echo "ALIASNAMES $(alias | cut -d= -f1 | sed "s/^alias //" | tr "\n" " ")"; for t in python3 python pip3 pytest git sh bash zsh ls cat grep sed awk find wc sort head tail diff patch make env xargs tr cut node npm; do echo "$t -> $(command -v $t 2>/dev/null || echo none)"; done; echo "python3 $(python3 --version 2>&1)"; echo "git $(git --version)"; python3 -c "import sys, locale; print(\"py\", sys.executable, locale.getpreferredencoding(), sys.getfilesystemencoding(), sys.stdout.encoding)"; echo "LANG=${LANG:-unset} LC_ALL=${LC_ALL:-unset} LC_CTYPE=${LC_CTYPE:-unset}"; echo "ALIASES $(alias | wc -l | tr -d " ")"; echo "FUNCS $( (typeset +f 2>/dev/null || declare -F) | wc -l | tr -d " ")"; echo "ENVNAMES $(env | cut -d= -f1 | grep -E "^[A-Za-z_][A-Za-z0-9_]*$" | sort | tr "\n" " ")"; } > envreport.txt; cat envreport.txt'
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_bash.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP)"
for V in R F; do
  RUN=$B/$V; rm -rf $RUN; mkdir -p $RUN/tasks/t00 $RUN/cctmp
  cat > $RUN/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
(allow file-write* (subpath "$RUN") (subpath "/private/var/folders") (regex #"^/dev/"))
(deny file-read* (subpath "$H/.ssh") (subpath "$H/.aws") (subpath "$H/.gnupg") (regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")
  (subpath "$H/.codex") (subpath "$H/.config/gh") (literal "$H/.netrc") (literal "$H/.git-credentials")
  (subpath "$H/Library/Keychains") (subpath "$H/Library/Mail") (subpath "$H/Library/Messages") (subpath "$H/Library/Cookies")
  (subpath "$H/Library/Safari") (subpath "$H/Library/Application Support/Google") (subpath "$H/Library/Application Support/Claude")
  (subpath "$H/Documents") (subpath "$H/Desktop") (subpath "$H/Downloads") (subpath "$H/Papers") (subpath "$H/.local/share/exactstream-sessions"))
SB
  rsync -a --exclude projects --exclude sessions --exclude backups --exclude shell-snapshots --exclude todos $A/cc_home/ $RUN/cc_home/
  EXTRA=""
  ( cd $RUN/tasks/t00 && git init -q && echo x > seed.txt && git add -A && git -c user.email=agentcc@localhost -c user.name=agentcc -c commit.gpgsign=false commit -q -m template )
  if [ $V = F ]; then
    mkdir -p $RUN/home && cp -p $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
    EXTRA="HOME=$RUN/home"
  fi
  ( cd $RUN/tasks/t00 && exec env CLAUDE_CONFIG_DIR=$RUN/cc_home EXACTSTREAM_PORT=$PORT CLAUDE_BIN=/Users/kamen/.local/bin/claude \
      CLAUDE_CODE_TMPDIR=$RUN/cctmp $EXTRA \
      sandbox-exec -f $RUN/sandbox.sb $CL/tools/claude_local.sh -p "BASHFIX: run the shell check." --output-format stream-json \
      --verbose --max-turns 3 --allowedTools Bash Edit Glob Grep Read Write ) < /dev/null > $RUN/out.jsonl 2> $RUN/stderr.txt &
  CP=$!; c0=$(date +%s)
  while kill -0 $CP 2>/dev/null; do [ $(( $(date +%s) - c0 )) -gt 180 ] && { echo "$V: claude over 180 s: stopping"; kill -TERM $CP; }; sleep 1; done
  wait $CP; echo "$V: claude rc $?; report lines $(grep -c . $RUN/tasks/t00/envreport.txt 2>/dev/null)"
done
stopsrv; echo "mock serve.py stopped"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B/R/tasks/t00/envreport.txt $B/F/tasks/t00/envreport.txt <<'EOF'
import sys
def rd(p):
    d = {}
    for l in open(p):
        l = l.rstrip("\n")
        if l.startswith("ENVNAMES "): d["ENVNAMES"] = set(l.split()[1:])
        elif " -> " in l: k, v = l.split(" -> ", 1); d["tool " + k] = v
        elif "=" in l and l.split("=", 1)[0] in ("PATH", "ZDOTDIR"): k, v = l.split("=", 1); d[k] = v
        else: d[l.split(" ", 1)[0]] = l
    return d
b, z = rd(sys.argv[1]), rd(sys.argv[2])
same = [k for k in b if k not in ("ENVNAMES", "ZDOTDIR", "ALIASES", "FUNCS", "HOMEDIR", "ALIASNAMES", "GITID") and b.get(k) == z.get(k)]
diff = [k for k in b if k not in ("ENVNAMES", "ZDOTDIR", "ALIASES", "FUNCS", "HOMEDIR", "ALIASNAMES", "GITID") and b.get(k) != z.get(k)]
print("equal:", len(same), "| differ:", diff)
for k in diff: print("  ", k, "| B:", b.get(k), "| Z:", z.get(k))
print("PATH equal:", b.get("PATH") == z.get("PATH"), "| PATH:", b.get("PATH"))
print("B", b.get("ALIASES"), b.get("FUNCS"), b.get("GITID"), b.get("GIT"), "| H", z.get("ALIASES"), z.get("FUNCS"), z.get("GITID"), z.get("GIT"))
print("env names only in B:", sorted(b["ENVNAMES"] - z["ENVNAMES"]))
print("env names only in H:", sorted(z["ENVNAMES"] - b["ENVNAMES"]))
print("python:", b.get("py"), "|", z.get("py")); print(b.get("LANG=" + "") or "")
EOF
grep -h "^LANG=" $B/R/tasks/t00/envreport.txt $B/F/tasks/t00/envreport.txt
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B <<'PYEOF'
import json, re, sys
pat = re.compile(json.load(open('/Users/kamen/Leap/release/private_patterns.json'))['leak'], re.I)
for V, d in (('R', sys.argv[1]), ('F', sys.argv[1])):
    for l in open(d + '/' + V + '/tasks/t00/envreport.txt'):
        if l.startswith(('ALIASNAMES', 'GIT', 'HOMEDIR', 'ALIASES', 'FUNCS')):
            print(V, pat.sub('<private>', l.strip()))
PYEOF
