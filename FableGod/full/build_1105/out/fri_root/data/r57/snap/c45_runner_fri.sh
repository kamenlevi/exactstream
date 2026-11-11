#!/bin/bash
# STUB runner for the dry-run: no engine, it only records what it was asked to run
echo "$C45_ARMS | screen=$C45_SCREEN | keys=$(echo "$R57_KEYS" | tr "," "\n" | wc -l | tr -d " ") | deadline=$C45_DEADLINE" >> "$C45_OUT/stub_arms.log"
