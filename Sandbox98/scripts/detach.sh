#!/bin/bash
# Start a command fully detached: a new session and process group (setsid), stdin from /dev/null, so no tool or shell
# clean-up that signals a process group can reach it. usage: scripts/detach.sh LOG CMD ARGS...
LOG=$1; shift
nohup /usr/bin/python3 -I -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" < /dev/null > "$LOG" 2>&1 &
echo $!
