#!/bin/bash
# 刀1-1 WarmupState 收拢(51 init → 数据类)。before=warmup_state.before.py, after=produce_window.before.py(下一刀留底)
cd "$(dirname "$0")/../../.."
python3 refactor/tools/verify_rename.py refactor/work/warmup_state.before.py refactor/work/produce_window.before.py --fn run_all_windows --bind warm --names-file refactor/work/warmup_state_names.txt --dataclass WarmupState --block for --index 0 --construction 1
