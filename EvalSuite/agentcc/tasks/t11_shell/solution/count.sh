#!/bin/sh
find data -name '*.txt' -type f -exec cat {} + | wc -l | tr -d ' '
