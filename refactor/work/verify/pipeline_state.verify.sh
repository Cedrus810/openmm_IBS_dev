#!/bin/bash
# 刀2-B PipelineState 收编。before=pipeline_state.before.py, after=pipeline_state2.before.py(下一刀留底)
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/pipeline_state.before.py refactor/work/pipeline_state2.before.py --fn _run_dual_lambda_scheme --bind dual --names-file refactor/work/pipeline_state_names.txt --dataclass PipelineState --in-place --construction 1
