#!/bin/sh
cat data/*.txt | wc -l | tr -d ' '
