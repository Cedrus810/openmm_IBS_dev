#!/bin/bash
# 刀2-B 第二波收编(stage1_states 形参传初值 + opt_res)。before=pipeline_state2.before.py, after=partition_windows.before.py(下一刀留底)
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/pipeline_state2.before.py refactor/work/partition_windows.before.py --fn _run_dual_lambda_scheme --bind dual --names stage1_states,opt_res --keep-file refactor/work/pipeline_state_names.txt --dataclass PipelineState --in-place --construction 1
