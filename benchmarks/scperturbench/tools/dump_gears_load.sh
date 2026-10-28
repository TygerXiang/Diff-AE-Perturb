#!/usr/bin/env bash
set -u
GEARS=/usr/local/anaconda3/envs/gears/lib/python3.9/site-packages/gears
sed -n '131,265p' "$GEARS/pertdata.py" > /root/perturbench/logs/gears_load_body.txt 2>&1
echo DONE
