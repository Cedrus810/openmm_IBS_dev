#!/bin/bash
# 刀1-6 刀c _summarize_warmup 抽取。before=summarize_warmup.before.py, after=production_state.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/summarize_warmup.before.py refactor/work/production_state.before.py --moved _summarize_warmup --src run_all_windows --block for --index 0 --slice 100:120
