#!/bin/bash
# 刀1-3 刀a _commit_frozen_f_k 抽取。before=commit_frozen_f_k.before.py, after=absorb3.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/commit_frozen_f_k.before.py refactor/work/absorb3.before.py --moved _commit_frozen_f_k --src run_all_windows --block for --index 0 --slice 120:121
