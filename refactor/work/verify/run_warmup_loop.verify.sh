#!/bin/bash
# 刀1-5 刀b _run_warmup_loop 抽取。before=run_warmup_loop.before.py, after=summarize_warmup.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/run_warmup_loop.before.py refactor/work/summarize_warmup.before.py --moved _run_warmup_loop --src run_all_windows --block for --index 0 --slice 99:100
