#!/bin/sh
# L1-19 (Leap, Sat 2026-10-24): 2-step refining DFlash2 drafter replay on the 24 public HumanEval samples. ~3 min.
cd "$HOME/Leap" || exit 1
$HOME/BigMoE/.venv/bin/python scripts/refine19.py
