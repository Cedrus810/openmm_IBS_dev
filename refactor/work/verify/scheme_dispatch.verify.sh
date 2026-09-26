#!/bin/bash
# 刀2 三路分派抽双方法(三道门)。before=scheme_dispatch.before.py, after=pipeline_state.before.py
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_extract.py refactor/work/scheme_dispatch.before.py refactor/work/pipeline_state.before.py --moved _run_dual_lambda_scheme --src run_full_pipeline --block if --index 21
python3 refactor/tools/verify_extract.py refactor/work/scheme_dispatch.before.py refactor/work/pipeline_state.before.py --moved _run_single_lambda_scheme --src run_full_pipeline --block if --index 21 --orelse
python3 refactor/tools/verify_dispatch.py refactor/work/scheme_dispatch.before.py refactor/work/pipeline_state.before.py --src run_full_pipeline --pos 55
