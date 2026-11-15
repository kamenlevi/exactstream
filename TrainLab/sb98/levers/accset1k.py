"""MTP Head accset arm 'cs1k' (WEEK98 S7 ACCSET): delta 1, LAM 1, exact survival (isolates the loss term).
Implementation: accset_base.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accset_base import make

standin, setup = make(1.0, 1.0, False)
