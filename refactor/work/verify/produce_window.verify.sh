#!/bin/bash
# 刀1-2 A段 _produce_window 抽取。before=produce_window.before.py, after=commit_frozen_f_k.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/produce_window.before.py refactor/work/commit_frozen_f_k.before.py --moved _produce_window --src run_all_windows --block for --index 0 --slice 129:211
